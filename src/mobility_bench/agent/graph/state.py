"""
State types for the mobility agent system.

This module defines the core data structures used in the planning system,
including Plan, Step, Task, and the main State class.
"""

from enum import Enum
from typing import Annotated, Any
from uuid import uuid4

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, field_validator


class Status(str, Enum):
    """Unified status for Plan, Step, and Task."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class RawPlan(BaseModel):
    """
    Raw plan structure for LLM output.

    A two-level list where:
    - Outer list represents steps (executed sequentially)
    - Inner list represents tasks (executed in parallel)
    """

    steps: list[list[str]] = Field(default_factory=list)
    thinking: str = Field(description="Reasoning process")
    intent: str = Field(description="User intent")

    @field_validator("steps", mode="before")
    @classmethod
    def _wrap_bare_step(cls, v: Any) -> Any:
        """Accept a bare string where a step's task list is expected.

        Weaker models sometimes emit ``["定位", "路线规划"]`` style plans where a
        step is a plain string instead of ``[task, ...]``. A bare string would be
        iterated character by character downstream, so wrap it in a one-task step.
        """
        if not isinstance(v, list):
            return v
        return [[step] if isinstance(step, str) else step for step in v]


class Tool(BaseModel):
    """Tool definition with name and arguments."""

    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """Single task definition."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid4().hex[:8]}")
    task_description: str
    execution_result: str | None = None
    tool_list: list[Tool] = Field(default_factory=list)
    status: Status = Status.PENDING


class Step(BaseModel):
    """
    Plan step containing multiple tasks.

    Tasks within the same step are executed in parallel.
    """

    tasks: list[Task] = Field(default_factory=list)
    status: Status = Status.PENDING


class Plan(BaseModel):
    """
    Execution plan with sequential steps.

    Steps are executed sequentially, while tasks within
    each step are executed in parallel.
    """

    steps: list[Step] = Field(default_factory=list)
    current_step_index: int = 0
    status: Status = Status.PENDING


def raw_plan_to_plan(raw_plan: RawPlan) -> Plan:
    """
    Convert LLM-generated RawPlan to runtime Plan.

    Cleaning rules:
    - Skip empty steps
    - Trim task descriptions; discard if empty
    """
    steps: list[Step] = []

    for raw_step in raw_plan.steps or []:
        tasks: list[Task] = []
        for raw_task in raw_step or []:
            desc = str(raw_task).strip()
            if not desc:
                continue
            tasks.append(Task(task_description=desc))

        if not tasks:
            continue

        steps.append(Step(tasks=tasks, status=Status.PENDING))

    return Plan(steps=steps, current_step_index=0, status=Status.EXECUTING)


# ============================================================================
# Reducer Functions
# ============================================================================


def add_worker_training_data(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reducer for worker_training_data: merge worker training data lists."""
    return left + right


def merge_token_usage(existing: dict | None, new: dict | None) -> dict | None:
    """Merge two token usage dictionaries, accumulating numeric fields."""
    if new is None:
        return existing

    if existing is None:
        return new.copy() if new else None

    merged = existing.copy()
    for key, value in new.items():
        if isinstance(value, (int, float)) and key in merged:
            merged[key] += value
        else:
            merged[key] = value
    return merged


# ============================================================================
# State Definition
# ============================================================================


class State(MessagesState):
    """
    State for the mobility agent system.

    Extends MessagesState with fields for:
    - User input (query, context)
    - POI caching
    - Plan execution
    - Training data collection
    - Token usage tracking
    """

    # User input
    query: str  # User query
    context: str | None = None  # Context information

    # POI cache mapping: key is POI name, value is {"coord": "xxx,xxx", "poiid": "xxx"}
    poi_map: dict[str, dict[str, Any]] = {}

    # Core planning system
    current_plan: Plan | None = None  # Current execution plan
    plan_iterations: int = 0  # Plan iteration count

    # Planner output
    planner_thinking: str | None = None  # Planning reasoning process
    planner_intent: str | None = None  # Identified user intent

    # Final result - Markdown report generated by Reporter
    plan_result: str | None = None

    # ReAct loop state (react framework). Every key the nodes write must be
    # declared here: LangGraph silently drops updates for undeclared keys, which
    # freezes the iteration counter at 0 and leaves the loop unbounded.
    react_iterations: int = 0
    react_finish: bool = False
    react_thoughts: list = []
    react_actions: list = []
    react_current_action: str = ""
    react_current_tool_name: str = ""
    react_current_tool_args: dict = {}
    react_current_action_index: int = -1
    react_intent: str = ""
    react_route_index: str = ""
    react_route_info: str = ""
    token_usage: dict | None = None

    # Parallel task execution
    current_task: Task | None = None  # Current task for concurrent branches

    # Training data collection
    planner_training_data: dict[str, Any] | None = None
    worker_training_data: Annotated[
        list[dict[str, Any]], add_worker_training_data
    ] = Field(default_factory=list)
    reporter_training_data: dict[str, Any] | None = None

    # Token usage tracking
    planner_token_usage: dict | None = None
    worker_token_usage: Annotated[dict | None, merge_token_usage] = None
    reporter_token_usage: dict | None = None
