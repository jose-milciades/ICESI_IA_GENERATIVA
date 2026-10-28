"""Estado persistente del grafo."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from ecomarket_agent.models import ApprovalRequest, PlanStep, StepResult


class PlanExecuteState(TypedDict, total=False):
    input: str
    objective: str
    plan: list[PlanStep]
    past_steps: Annotated[list[StepResult], operator.add]
    response: str

    thread_id: str
    execution_count: int
    replan_count: int
    pending_question: str | None
    pending_approval: ApprovalRequest | None
    user_decision: bool | None

    order_code: str | None
    order: dict[str, Any] | None
    retrieved_documents: list[dict[str, Any]]
    eligibility: dict[str, Any] | None
    return_record: dict[str, Any] | None
    label: dict[str, Any] | None
    last_error: dict[str, Any] | None
