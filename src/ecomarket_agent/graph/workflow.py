"""Construcción del StateGraph Plan-and-Execute."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy, interrupt

from ecomarket_agent.agents import PlanExecuteAgents
from ecomarket_agent.graph.state import PlanExecuteState
from ecomarket_agent.models import ReplanDecision, StepResult, ToolName
from ecomarket_agent.repositories import extract_order_code


class NodeName(StrEnum):
    PREPARE_INPUT = "prepare_input"
    PLANNER = "planner"
    EXECUTOR = "executor"
    REPLANNER = "replanner"
    HUMAN_INPUT = "human_input"
    APPROVAL = "approval"
    FINAL_RESPONSE = "final_response"
    HUMAN_ESCALATION = "human_escalation"


def _tool_state_updates(result: StepResult) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if not result.observation:
        return updates
    payloads = result.observation.get("tool_results", [])
    for payload in payloads:
        if payload.get("ok") is not True:
            updates["last_error"] = payload.get("error")
            continue
        data = payload.get("data")
        if result.tool_name == ToolName.CONSULTAR_PEDIDO:
            if isinstance(data, dict) and "orders" in data:
                updates["orders"] = data["orders"]
            else:
                updates["order"] = data
        elif result.tool_name == ToolName.BUSCAR_POLITICA:
            updates["retrieved_documents"] = data
        elif result.tool_name == ToolName.VERIFICAR_ELEGIBILIDAD:
            if isinstance(data, dict) and "eligibilities" in data:
                updates["eligibilities"] = data["eligibilities"]
            else:
                updates["eligibility"] = data
        elif result.tool_name == ToolName.REGISTRAR_DEVOLUCION:
            updates["return_record"] = data
        elif result.tool_name == ToolName.GENERAR_ETIQUETA:
            updates["label"] = data
    return updates


def build_workflow(agents: PlanExecuteAgents, checkpointer=None):
    def prepare_input(state: PlanExecuteState) -> dict[str, Any]:
        user_input = (state.get("input") or "").strip()
        if not user_input:
            raise ValueError("La consulta no puede estar vacía.")
        return {
            "input": user_input,
            "objective": user_input,
            "order_code": extract_order_code(user_input),
            "execution_count": 0,
            "replan_count": 0,
            "retrieved_documents": [],
            "orders": [],
            "eligibilities": [],
            "pending_question": None,
            "pending_approval": None,
            "user_decision": None,
            "decision": None,
            "last_error": None,
        }

    def planner(state: PlanExecuteState) -> dict[str, Any]:
        plan = agents.plan(state["objective"], state.get("order_code"))
        return {
            "plan": [step.model_dump(mode="json") for step in plan.steps],
            "decision": None,
        }

    def executor(state: PlanExecuteState) -> dict[str, Any]:
        if not state.get("plan"):
            return {"response": "No quedan pasos pendientes."}
        result = agents.execute(dict(state))
        keep_current = result.status in {"needs_input", "needs_approval"}
        remaining_plan = list(state["plan"] if keep_current else state["plan"][1:])
        updates: dict[str, Any] = {
            "past_steps": [result.model_dump(mode="json")],
            "plan": remaining_plan,
            "execution_count": state.get("execution_count", 0) + 1,
            "decision": None,
        }
        if result.tool_name == ToolName.REGISTRAR_DEVOLUCION and result.status == "completed":
            updates["user_decision"] = None
        updates.update(_tool_state_updates(result))
        return updates

    def replanner(state: PlanExecuteState) -> dict[str, Any]:
        decision = agents.replan(dict(state))
        updates: dict[str, Any] = {
            "decision": decision.model_dump(mode="json"),
            "plan": [step.model_dump(mode="json") for step in decision.remaining_plan],
            "replan_count": state.get("replan_count", 0) + 1,
            "pending_question": decision.question,
            "pending_approval": (
                decision.approval_request.model_dump(mode="json")
                if decision.approval_request
                else None
            ),
        }
        if decision.response:
            updates["response"] = decision.response
        return updates

    def human_input(state: PlanExecuteState) -> dict[str, Any]:
        answer = interrupt(
            {
                "kind": "input",
                "question": state.get("pending_question")
                or "Necesito información adicional para continuar.",
            }
        )
        text = str(answer.get("text", "") if isinstance(answer, dict) else answer).strip()
        if not text:
            return {
                "response": "No recibí la información necesaria; el proceso fue detenido.",
                "decision": {"action": "respond"},
            }
        complete_input = f"{state['input']}\nInformación adicional del usuario: {text}"
        return {
            "input": complete_input,
            "objective": complete_input,
            "order_code": extract_order_code(text) or state.get("order_code"),
            "pending_question": None,
            "decision": None,
        }

    def approval(state: PlanExecuteState) -> dict[str, Any]:
        request = state.get("pending_approval")
        answer = interrupt({"kind": "approval", "request": request})
        approved = bool(answer.get("approved")) if isinstance(answer, dict) else bool(answer)
        if not approved:
            return {
                "user_decision": False,
                "pending_approval": None,
                "response": "La operación fue cancelada. No se registró la devolución.",
            }
        return {
            "user_decision": True,
            "pending_approval": None,
            "decision": None,
        }

    def final_response(state: PlanExecuteState) -> dict[str, Any]:
        if state.get("response"):
            return {}
        latest = state.get("past_steps", [])[-1] if state.get("past_steps") else None
        answer = ""
        if latest and latest.get("observation"):
            answer = str(latest["observation"].get("answer", ""))
        return {"response": answer or "La solicitud fue procesada."}

    def human_escalation(state: PlanExecuteState) -> dict[str, Any]:
        decision = ReplanDecision.model_validate(state.get("decision") or {})
        return {
            "response": decision.response
            or "Esta solicitud requiere atención de un asesor de EcoMarket."
        }

    def route_after_replan(state: PlanExecuteState) -> str:
        decision = state.get("decision") or {}
        return str(decision.get("action", "escalate"))

    def route_after_human_input(state: PlanExecuteState) -> str:
        return "final" if state.get("response") else "planner"

    def route_after_approval(state: PlanExecuteState) -> str:
        return "executor" if state.get("user_decision") is True else "final"

    workflow = StateGraph(PlanExecuteState)
    workflow.add_node(NodeName.PREPARE_INPUT, prepare_input)
    workflow.add_node(
        NodeName.PLANNER,
        planner,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    workflow.add_node(NodeName.EXECUTOR, executor)
    workflow.add_node(
        NodeName.REPLANNER,
        replanner,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    workflow.add_node(NodeName.HUMAN_INPUT, human_input)
    workflow.add_node(NodeName.APPROVAL, approval)
    workflow.add_node(NodeName.FINAL_RESPONSE, final_response)
    workflow.add_node(NodeName.HUMAN_ESCALATION, human_escalation)

    workflow.add_edge(START, NodeName.PREPARE_INPUT)
    workflow.add_edge(NodeName.PREPARE_INPUT, NodeName.PLANNER)
    workflow.add_edge(NodeName.PLANNER, NodeName.EXECUTOR)
    workflow.add_edge(NodeName.EXECUTOR, NodeName.REPLANNER)
    workflow.add_conditional_edges(
        NodeName.REPLANNER,
        route_after_replan,
        {
            "continue": NodeName.EXECUTOR,
            "respond": NodeName.FINAL_RESPONSE,
            "ask_user": NodeName.HUMAN_INPUT,
            "request_approval": NodeName.APPROVAL,
            "escalate": NodeName.HUMAN_ESCALATION,
        },
    )
    workflow.add_conditional_edges(
        NodeName.HUMAN_INPUT,
        route_after_human_input,
        {"planner": NodeName.PLANNER, "final": NodeName.FINAL_RESPONSE},
    )
    workflow.add_conditional_edges(
        NodeName.APPROVAL,
        route_after_approval,
        {"executor": NodeName.EXECUTOR, "final": NodeName.FINAL_RESPONSE},
    )
    workflow.add_edge(NodeName.FINAL_RESPONSE, END)
    workflow.add_edge(NodeName.HUMAN_ESCALATION, END)
    return workflow.compile(checkpointer=checkpointer)
