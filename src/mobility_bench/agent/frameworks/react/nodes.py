"""
ReAct framework nodes.

This module implements the core node functions for the Reasoning-Action-Observation loop:
- reasoning_node: Generates thoughts and decides next action
- action_node: Executes tools or finishes
- should_continue: Routes to continue or end
"""

import json
import logging
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from mobility_bench.agent.graph.decorators import (
    get_execution_recorder,
    record_execution,
)
from mobility_bench.agent.graph.state import State
from mobility_bench.agent.prompts import get_system_prompt
from mobility_bench.agent.roles import AgentType, get_llm_for_agent
from mobility_bench.agent.utils import log_event, state_context
from mobility_bench.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Maximum iterations for ReAct loop
try:
    from mobility_bench.config.settings import Settings
    MAX_REACT_ITERATIONS = Settings.get_instance().agent.react.max_iterations
except Exception:
    MAX_REACT_ITERATIONS = 15


_REPAIR_PROMPT = (
    "上一条回复不是合法的 ReAct JSON。请只输出一个 JSON 对象，不要附带任何解释文字、"
    "不要使用 Markdown：调用工具时为 "
    '{"intent": "...", "thought": "...", "action": "call_tool", "tool_name": "...", "tool_args": {...}}，'
    "结束时为 "
    '{"intent": "...", "thought": "...", "action": "finish", "final_answer": "...", '
    '"route_index": "...", "route_info": "..."}。'
    "intent 从「地点查询、位置查询、附近周边搜索、天气查询、拥堵、距离、出发时间、路线规划」中选一个；"
    'route_index 为选定路线在 paths 中从 0 开始的索引（字符串），无可用路线填 "-1"，非路线任务填空字符串。'
)


def _parse_react_response(content: str) -> dict[str, Any] | None:
    """Parse one ReAct turn into ``{thought, action, tool_name, tool_args}``.

    Accepts the JSON object the prompt asks for and, as a fallback, the plain
    ``Thought:/Action:`` transcript form models drift into once that form shows
    up in the message history. Returns None when neither can be read, so the
    caller can ask the model to repair its reply instead of ending the case.
    """
    text = (content or "").strip()

    candidate = text
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        candidate = json_match.group(1)
    else:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            candidate = json_match.group(0)

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict) and parsed.get("action"):
            return parsed
    except json.JSONDecodeError as e:
        logger.warning(f"[REACT] JSON parse failed ({e}), trying text form")

    action_match = re.search(r'Action\s*[:：]\s*["\']?(\w+)', text)
    if not action_match or action_match.group(1) not in {"call_tool", "finish"}:
        logger.warning(f"[REACT] Unparseable response: {text[:500]!r}")
        return None

    action = action_match.group(1)
    thought = re.sub(r'^Thought\s*[:：]\s*', '', text[:action_match.start()].strip())

    tool_name = ""
    tool_args: dict[str, Any] = {}
    if action == "call_tool":
        name_match = re.search(r'tool_name\s*[:：]\s*["\']?([\w.\-]+)', text)
        tool_name = name_match.group(1) if name_match else ""
        args_match = re.search(r'tool_args\s*[:：]\s*(\{.*\})', text, re.DOTALL)
        if args_match:
            try:
                parsed_args = json.loads(args_match.group(1))
                if isinstance(parsed_args, dict):
                    tool_args = parsed_args
            except json.JSONDecodeError:
                pass
        if not tool_name:
            logger.warning(f"[REACT] Text action without tool_name: {text[:500]!r}")
            return None

    logger.warning(f"[REACT] Recovered text-form response: action={action} tool={tool_name}")
    return {
        "thought": thought,
        "action": action,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "final_answer": text if action == "finish" else "",
    }


def _accumulate_usage(previous: dict | None, response: Any, llm: Any) -> dict:
    """Add one call's usage onto the running total and emit an ``llm_call`` event.

    react is a loop (reason -> action -> reason -> ...) and the efficiency
    metric sums per-iteration usage across the whole trajectory.
    """
    step_prompt = step_completion = step_total = 0
    if getattr(response, "usage_metadata", None):
        usage = response.usage_metadata
        step_prompt = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        step_completion = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        step_total = usage.get("total_tokens", step_prompt + step_completion) or (
            step_prompt + step_completion
        )
        log_event(
            event_type="llm_call",
            payload={
                "model": getattr(llm, "model_name", str(llm)),
                "role": "react_reasoning",
                "prompt_tokens": step_prompt,
                "completion_tokens": step_completion,
                "total_tokens": step_total,
            },
        )

    total = dict(previous or {})
    total["prompt_tokens"] = total.get("prompt_tokens", 0) + step_prompt
    total["completion_tokens"] = total.get("completion_tokens", 0) + step_completion
    total["total_tokens"] = total.get("total_tokens", 0) + step_total
    return total


def _build_tool_descriptions() -> str:
    """Build tool descriptions for the prompt."""
    tools = ToolRegistry.get_all()
    if not tools:
        return "No tools available."

    descriptions = []
    for tool in tools:
        desc = f"- {tool.name}"
        if hasattr(tool, 'description') and tool.description:
            desc += f": {tool.description[:100]}"
        descriptions.append(desc)

    return "\n".join(descriptions)


@record_execution("react_reasoning", "reasoning")
async def reasoning_node(state: State, config: RunnableConfig) -> Command:
    """
    Reasoning node: Analyzes situation and decides next action.

    This node:
    1. Gets current context from messages
    2. Uses LLM to generate thought and decide action
    3. Updates state with thought and action decision
    """
    recorder = get_execution_recorder(state, "react_reasoning")
    iterations = state.get("react_iterations", 0)

    # Check iteration limit
    if iterations >= MAX_REACT_ITERATIONS:
        logger.warning(f"[REACT] Reached max iterations ({MAX_REACT_ITERATIONS}), forcing finish")
        return Command(
            update={
                "react_finish": True,
                "plan_result": "Reached maximum iterations. Please try with a simpler query.",
            },
            goto="action"
        )

    logger.info(f"[REACT] Reasoning iteration {iterations + 1}/{MAX_REACT_ITERATIONS}")

    try:
        llm = get_llm_for_agent(AgentType.PLANNER)  # Use reasoning model
        system_prompt = get_system_prompt("react", "reasoning")

        # Build messages
        messages = [SystemMessage(content=system_prompt)]

        # Add context and query on first iteration
        if iterations == 0:
            context = state.get("context")
            if context:
                messages.append(HumanMessage(content=f"Context: {context}"))

            query = state.get("query")
            if query:
                messages.append(HumanMessage(content=f"User query: {query}"))

            # Add available tools info
            tools_info = _build_tool_descriptions()
            messages.append(HumanMessage(content=f"Available tools:\n{tools_info}"))

        # Add conversation history
        messages.extend(state.get("messages", []))

        # Call LLM
        response = await llm.ainvoke(messages)
        token_usage = _accumulate_usage(state.get("token_usage"), response, llm)

        parsed = _parse_react_response(response.content)
        if parsed is None:
            # One repair round: show the model its own reply, restate the schema.
            repair_messages = list(messages)
            if response.content:
                repair_messages.append(AIMessage(content=response.content))
            repair_messages.append(HumanMessage(content=_REPAIR_PROMPT))
            response = await llm.ainvoke(repair_messages)
            token_usage = _accumulate_usage(token_usage, response, llm)
            parsed = _parse_react_response(response.content)

        if parsed is None:
            logger.warning("[REACT] Response unreadable after repair, finishing case")
            parsed = {
                "thought": "Unable to parse response, ending with available information.",
                "action": "finish",
                "final_answer": response.content or "",
            }

        thought = parsed.get("thought", "")
        action = parsed.get("action", "finish")
        tool_name = parsed.get("tool_name", "")
        tool_args = parsed.get("tool_args", {})
        final_answer = parsed.get("final_answer", "")
        intent = parsed.get("intent", "")
        route_index = parsed.get("route_index", "")
        route_info = parsed.get("route_info", "")

        logger.info(f"[REACT] Thought: {thought[:100]}...")
        logger.info(f"[REACT] Action: {action}")

        # Update state
        thoughts = list(state.get("react_thoughts", []))
        thoughts.append(thought)

        actions = list(state.get("react_actions", []))
        actions.append({
            "action": action,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": None,
        })

        # Replay the model's own reply verbatim: re-rendering it as
        # "Thought: .../Action: ..." teaches the model to drop the JSON format.
        new_messages = list(state.get("messages", []))
        if response.content:
            new_messages.append(AIMessage(content=response.content))

        update_dict = {
            "messages": new_messages,
            "react_iterations": iterations + 1,
            "react_thoughts": thoughts,
            "react_actions": actions,
            "react_current_action": action,
            "react_current_tool_name": tool_name,
            "react_current_tool_args": tool_args,
            "react_current_action_index": len(actions) - 1,
            "token_usage": token_usage,
        }

        # Keep the last intent the model declared; it is what the instruction
        # metric (ID) scores, so an empty later turn must not erase it.
        if isinstance(intent, str) and intent.strip():
            update_dict["react_intent"] = intent.strip()

        # route_index / route_info are what the answer metric reads for route
        # tasks; carry the latest declared values through to the final state.
        if isinstance(route_index, (str, int)) and str(route_index).strip() != "":
            update_dict["react_route_index"] = str(route_index).strip()
        if isinstance(route_info, str) and route_info.strip():
            update_dict["react_route_info"] = route_info.strip()

        if action == "finish":
            update_dict["react_finish"] = True
            update_dict["plan_result"] = final_answer

        recorder.tool_call("reasoning_complete", {"action": action, "iteration": iterations + 1})

        return Command(update=update_dict, goto="action")

    except Exception as e:
        logger.error(f"[REACT] Reasoning error: {e}", exc_info=True)
        return Command(
            update={
                "react_finish": True,
                "plan_result": f"Error during reasoning: {str(e)}",
            },
            goto="action"
        )


@record_execution("react_action", "acting")
async def action_node(state: State, config: RunnableConfig) -> Command:
    """
    Action node: Executes tool or prepares to finish.

    This node:
    1. Checks if action is 'finish' - if so, just pass through
    2. If action is 'call_tool', executes the tool
    3. Adds observation (tool result) to messages
    """
    recorder = get_execution_recorder(state, "react_action")

    action = state.get("react_current_action", "finish")
    tool_name = state.get("react_current_tool_name", "")
    tool_args = state.get("react_current_tool_args", {})

    logger.info(f"[REACT] Action node - action: {action}, tool: {tool_name}")

    # If finish action, nothing to do
    if action == "finish" or state.get("react_finish"):
        recorder.tool_call("action_finish", {})
        return Command(update={}, goto="__end__")

    # Execute tool
    if action == "call_tool" and tool_name:
        try:
            # Get tool from registry
            tools = ToolRegistry.get_all()
            if not tools:
                ToolRegistry.load_default_tools(mode="sandbox")
                tools = ToolRegistry.get_all()

            tool_map = {t.name: t for t in tools}

            if tool_name not in tool_map:
                error_msg = f"Unknown tool: {tool_name}"
                logger.warning(f"[REACT] {error_msg}")
                observation = f"Error: {error_msg}"
            else:
                # Execute tool
                tool = tool_map[tool_name]
                logger.info(f"[REACT] Calling tool: {tool_name} with args: {tool_args}")

                with state_context(dict(state)):
                    result = tool.invoke(tool_args, config)

                observation = str(result)
                logger.info(f"[REACT] Tool result: {observation[:200]}...")

            # Add observation to messages, and attach the result to the action
            # entry so the tool trajectory can be exported after the run.
            new_messages = list(state.get("messages", []))
            new_messages.append(HumanMessage(content=f"Observation: {observation}"))

            actions = list(state.get("react_actions", []))
            idx = state.get("react_current_action_index", -1)
            if 0 <= idx < len(actions):
                actions[idx] = {**actions[idx], "result": observation[:4000]}

            recorder.tool_call("tool_executed", {"tool_name": tool_name})

            return Command(
                update={"messages": new_messages, "react_actions": actions},
                goto="reasoning"
            )

        except Exception as e:
            logger.error(f"[REACT] Tool execution error: {e}", exc_info=True)
            error_observation = f"Tool execution error: {str(e)}"

            new_messages = list(state.get("messages", []))
            new_messages.append(HumanMessage(content=f"Observation: {error_observation}"))

            return Command(
                update={"messages": new_messages},
                goto="reasoning"
            )

    # Default: continue to reasoning
    return Command(update={}, goto="reasoning")


def should_continue(state: State) -> Literal["continue", "end"]:
    """
    Routing function: decides whether to continue loop or end.

    Returns:
        "continue" to go back to reasoning
        "end" to finish execution
    """
    if state.get("react_finish"):
        return "end"

    iterations = state.get("react_iterations", 0)
    if iterations >= MAX_REACT_ITERATIONS:
        return "end"

    action = state.get("react_current_action", "")
    if action == "finish":
        return "end"

    return "continue"
