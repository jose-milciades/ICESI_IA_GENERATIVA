from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ecomarket_agent.graph.workflow import build_workflow
from ecomarket_agent.models import (
    ApprovalRequest,
    ExecutionPlan,
    PlanStep,
    ReplanDecision,
    StepResult,
    ToolName,
)


class FakeAgents:
    def plan(self, objective: str, order_code: str | None) -> ExecutionPlan:
        return ExecutionPlan(
            objective=objective,
            steps=[PlanStep(step_id="step-1", task="Responder")],
        )

    def execute(self, state) -> StepResult:
        return StepResult(
            step_id="step-1",
            task="Responder",
            status="completed",
            observation={"tool_results": [], "answer": "¡Hola!"},
        )

    def replan(self, state) -> ReplanDecision:
        return ReplanDecision(
            action="respond",
            response="¡Hola!",
            reason_summary="El saludo quedó respondido.",
        )


def test_plan_execute_graph_reaches_final_response() -> None:
    graph = build_workflow(FakeAgents(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test-thread"}}

    result = graph.invoke(
        {"input": "Hola", "thread_id": "test-thread", "past_steps": []},
        config=config,
    )

    assert result["response"] == "¡Hola!"
    assert result["execution_count"] == 1
    assert len(result["past_steps"]) == 1


class FakeApprovalAgents:
    def plan(self, objective: str, order_code: str | None) -> ExecutionPlan:
        return ExecutionPlan(
            objective=objective,
            steps=[
                PlanStep(
                    step_id="register",
                    task="Registrar devolución",
                    allowed_tools=[ToolName.REGISTRAR_DEVOLUCION],
                    requires_approval=True,
                ),
                PlanStep(step_id="respond", task="Responder"),
            ],
        )

    def execute(self, state) -> StepResult:
        step = state["plan"][0]
        if step["step_id"] == "register" and state.get("user_decision") is not True:
            return StepResult(
                step_id="register",
                task="Registrar devolución",
                status="needs_approval",
                tool_name=ToolName.REGISTRAR_DEVOLUCION,
            )
        return StepResult(
            step_id=step["step_id"],
            task=step["task"],
            status="completed",
            observation={"tool_results": [], "answer": "Proceso completado"},
        )

    def replan(self, state) -> ReplanDecision:
        latest = state["past_steps"][-1]
        remaining = [PlanStep.model_validate(step) for step in state["plan"]]
        if latest["status"] == "needs_approval":
            return ReplanDecision(
                action="request_approval",
                remaining_plan=remaining,
                approval_request=ApprovalRequest(
                    action=ToolName.REGISTRAR_DEVOLUCION,
                    summary="Confirmar devolución",
                    idempotency_key="test-key",
                ),
                reason_summary="Acción con efecto.",
            )
        if remaining:
            return ReplanDecision(
                action="continue",
                remaining_plan=remaining,
                reason_summary="Aún quedan pasos.",
            )
        return ReplanDecision(
            action="respond",
            response="Proceso completado",
            reason_summary="Objetivo alcanzado.",
        )


def test_graph_requires_and_resumes_approval() -> None:
    graph = build_workflow(FakeApprovalAgents(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "approval-thread"}}

    graph.invoke(
        {
            "input": "Devuelve PED-2026-0003",
            "thread_id": "approval-thread",
            "past_steps": [],
        },
        config=config,
    )
    snapshot = graph.get_state(config)

    assert snapshot.interrupts[0].value["kind"] == "approval"

    result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert result["response"] == "Proceso completado"
    assert len(result["past_steps"]) == 3
