"""Estado persistente del grafo."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class PlanExecuteState(TypedDict, total=False):
    input: str
    objective: str
    plan: list[dict[str, Any]]
    past_steps: Annotated[list[dict[str, Any]], operator.add]
    response: str

    thread_id: str
    execution_count: int
    replan_count: int
    pending_question: str | None
    pending_approval: dict[str, Any] | None
    user_decision: bool | None
    decision: dict[str, Any] | None

    order_code: str | None
    order: dict[str, Any] | None
    orders: list[dict[str, Any]]
    retrieved_documents: list[dict[str, Any]]
    eligibility: dict[str, Any] | None
    eligibilities: list[dict[str, Any]]
    return_record: dict[str, Any] | None
    label: dict[str, Any] | None
    last_error: dict[str, Any] | None
