from ecomarket_agent.models import PlanStep, ToolName
from ecomarket_agent.tools import TOOL_CATALOG


def test_tool_catalog_contains_five_tools() -> None:
    assert len(TOOL_CATALOG) == 5


def test_plan_step_accepts_at_most_one_tool() -> None:
    step = PlanStep(
        step_id="step-1",
        task="Consultar el pedido",
        allowed_tools=[ToolName.CONSULTAR_PEDIDO],
    )

    assert step.allowed_tools == [ToolName.CONSULTAR_PEDIDO]
