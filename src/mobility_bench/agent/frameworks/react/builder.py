"""
ReAct framework graph builder.

This module provides the framework class and graph building functions
for the Reasoning-Action-Observation loop architecture.
"""

import json
from typing import Any

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from mobility_bench.agent.frameworks._cli_common import (
    build_worker_output,
    extract_steps_from_trajectory,
)
from mobility_bench.agent.frameworks.base import BaseFramework
from mobility_bench.agent.frameworks.react.nodes import (
    action_node,
    reasoning_node,
    should_continue,
)
from mobility_bench.agent.graph.state import State
from mobility_bench.config.settings import ModelConfig, Settings


def _build_react_graph() -> StateGraph:
    """Build and return the ReAct state graph."""
    builder = StateGraph(State)

    # Nodes
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("action", action_node)

    # Entry point
    builder.set_entry_point("reasoning")

    # Edges
    builder.add_edge("reasoning", "action")
    builder.add_conditional_edges(
        "action",
        should_continue,
        {
            "continue": "reasoning",
            "end": "__end__",
        }
    )

    return builder


def build_graph() -> CompiledStateGraph:
    """Build and return the compiled ReAct agent workflow graph."""
    builder = _build_react_graph()
    return builder.compile()


class ReactFramework(BaseFramework):
    """
    ReAct framework implementation.

    This framework uses the Reasoning-Action-Observation loop:
    - Reasoning: Analyze situation and decide next action
    - Action: Execute tool or finish
    - Observation: Process tool results
    - Repeat until task is complete
    """

    def __init__(self, settings: Settings | None = None, model_config: ModelConfig | None = None):
        super().__init__(settings, model_config)
        self._graph: CompiledStateGraph | None = None

    @property
    def name(self) -> str:
        return "react"

    def build_graph(self) -> CompiledStateGraph:
        """Build and return the compiled graph."""
        if self._graph is None:
            # Set up LLM manager with model config before building
            if self.model_config:
                from mobility_bench.agent.roles.llm_manager import set_model_config
                set_model_config(self.model_config)

            self._graph = build_graph()
        return self._graph

    def prepare_initial_state(self, query: str, context: str | None = None, **kwargs) -> dict[str, Any]:
        """
        Prepare initial state for ReAct execution.

        Args:
            query: User query string
            context: Optional context information
            **kwargs: Additional state fields

        Returns:
            Initial state dictionary
        """
        state = {
            "query": query,
            "context": context,
            "messages": [],
            "poi_map": kwargs.get("poi_map", {}),
            # ReAct-specific fields
            "react_iterations": 0,
            "react_thoughts": [],
            "react_actions": [],
            "react_finish": False,
            "plan_result": None,
        }

        # Merge any additional kwargs
        for key, value in kwargs.items():
            if key not in state:
                state[key] = value

        return state

    def extract_result(self, final_state: dict[str, Any]) -> dict[str, Any]:
        """
        Extract result from final state.

        Args:
            final_state: Final state after graph execution

        Returns:
            Result dictionary with ReAct-specific fields
        """
        plan_result = final_state.get("plan_result", "") or ""
        actions = final_state.get("react_actions", []) or []

        # The tool trajectory is mapped like a CLI harness's: one entry per real
        # tool call, so the tool metric (TS_P/TS_R/SC) has something to score.
        tool_calls = [
            {"id": f"react_{i}", "name": a.get("tool_name", ""), "args": a.get("tool_args") or {}}
            for i, a in enumerate(actions)
            if a.get("action") == "call_tool" and a.get("tool_name")
        ]
        results = {
            f"react_{i}": json.dumps(a.get("result"), ensure_ascii=False)
            for i, a in enumerate(actions)
            if a.get("action") == "call_tool" and a.get("tool_name")
        }
        worker = build_worker_output(tool_calls)
        for entry in worker:
            entry["tool_results"] = {entry["task_id"]: results.get(entry["task_id"], "")}

        # The answer metric reads route_index / content out of a ```json block in
        # the reporter answer (extract_route_index_from_answer /
        # extract_content_from_answer), so append one; the prose answer stays the
        # body that DR's delivery check and the human reader see.
        if plan_result:
            answer_json = {
                "intent": final_state.get("react_intent", "") or "",
                "route_index": final_state.get("react_route_index", "") or "",
                "content": plan_result,
                "route_info": final_state.get("react_route_info", "") or "",
            }
            plan_result = (
                f"{plan_result}\n\n```json\n"
                f"{json.dumps(answer_json, ensure_ascii=False)}\n```"
            )

        # react has no planner/reporter split; like the CLI harnesses, report the
        # whole accumulated trajectory under `reporter` so the efficiency metric
        # (which sums the three nested slots) counts it once.
        return {
            "plan_result": plan_result,
            "react_iterations": final_state.get("react_iterations", 0),
            "react_thoughts": final_state.get("react_thoughts", []),
            "react_actions": actions,
            "planner_training_data": {
                "steps": extract_steps_from_trajectory(tool_calls, plan_result),
                "thinking": "\n".join(final_state.get("react_thoughts", []) or []),
                "intent": final_state.get("react_intent", "") or "",
            },
            "worker_training_data": worker,
            "token_usage": {
                "planner": {},
                "worker": {},
                "reporter": final_state.get("token_usage", {}),
            },
        }
