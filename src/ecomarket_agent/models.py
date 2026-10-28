"""Contratos compartidos entre planner, ejecutor, herramientas y replanner."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolName(StrEnum):
    CONSULTAR_PEDIDO = "consultar_pedido"
    BUSCAR_POLITICA = "buscar_politica_devolucion"
    VERIFICAR_ELEGIBILIDAD = "verificar_elegibilidad"
    REGISTRAR_DEVOLUCION = "registrar_devolucion"
    GENERAR_ETIQUETA = "generar_etiqueta"


class Intent(StrEnum):
    GREETING = "greeting"
    PERSONALIZED_ATTENTION = "personalized_attention"
    ORDER_STATUS = "order_status"
    PRODUCT_INFORMATION = "product_information"
    ORDER_RETURN = "order_return"


class Order(StrictModel):
    order_code: str
    product_type: str
    product_name: str
    product_description: str
    order_date: str
    last_event: str
    last_event_date: str
    days_delayed: int
    initial_delivery_date: str
    estimated_delivery_date: str
    tracking_number: str
    tracking_url: str


class RetrievedDocument(StrictModel):
    content: str
    source: str
    document_type: str
    page: int = 0
    row: int = 0
    score: float | None = None


class EligibilityResult(StrictModel):
    order_code: str
    eligible: bool
    decision: Literal["preeligible", "not_eligible", "needs_information"]
    reason: str
    return_window: str | None = None
    conditions: list[str] = Field(default_factory=list)
    policy_reference: str


class ReturnRecord(StrictModel):
    return_id: str
    order_code: str
    reason: str
    status: Literal["registered"] = "registered"
    created_at: str
    idempotency_key: str


class ReturnLabel(StrictModel):
    label_id: str
    return_id: str
    file_path: str
    created_at: str
    idempotency_key: str


class PlanStep(StrictModel):
    step_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    allowed_tools: list[ToolName] = Field(default_factory=list, max_length=1)
    requires_approval: bool = False


class ExecutionPlan(StrictModel):
    objective: str = Field(min_length=1)
    steps: list[PlanStep] = Field(min_length=1)


class StepResult(StrictModel):
    step_id: str
    task: str
    status: Literal["completed", "failed", "needs_input", "needs_approval"]
    tool_name: ToolName | None = None
    observation: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None


class ApprovalRequest(StrictModel):
    action: ToolName
    summary: str
    idempotency_key: str


class ReplanDecision(StrictModel):
    action: Literal["continue", "respond", "ask_user", "request_approval", "escalate"]
    remaining_plan: list[PlanStep] = Field(default_factory=list)
    response: str | None = None
    question: str | None = None
    approval_request: ApprovalRequest | None = None
    reason_summary: str = Field(min_length=1)
