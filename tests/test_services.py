import json
from pathlib import Path

from ecomarket_agent.agents import is_order_list_request, is_return_process_question
from ecomarket_agent.eligibility import EligibilityService
from ecomarket_agent.models import Order, ToolName
from ecomarket_agent.repositories import (
    LabelRepository,
    OrderRepository,
    ReturnRepository,
    extract_order_code,
)
from ecomarket_agent.retrieval.service import load_knowledge_documents
from ecomarket_agent.tools import ToolDependencies, build_tool_registry


def sample_order(**updates) -> Order:
    payload = {
        "order_code": "PED-2026-0003",
        "product_type": "hogar",
        "product_name": "Paños reutilizables",
        "product_description": "Paños de algodón reciclado.",
        "order_date": "2026-09-16",
        "last_event": "entregado",
        "last_event_date": "2026-09-18",
        "days_delayed": 0,
        "initial_delivery_date": "2026-09-18",
        "estimated_delivery_date": "2026-09-18",
        "tracking_number": "SIM-1",
        "tracking_url": "https://example.test/SIM-1",
    }
    payload.update(updates)
    return Order.model_validate(payload)


def test_extract_order_code_is_deterministic() -> None:
    assert extract_order_code("Devuelve ped-2026-0003, por favor") == "PED-2026-0003"
    assert extract_order_code("No tengo el código") is None


def test_detects_order_list_request() -> None:
    assert is_order_list_request("Dime qué productos he pedido") is True
    assert is_order_list_request("¿Qué pedidos he realizado?") is True
    assert is_order_list_request("¿Cuáles son mis pedidos?") is True
    assert is_order_list_request("De mis pedidos cuáles son elegibles para devolución") is True
    assert is_order_list_request("¿Dónde está mi pedido?") is False


def test_distinguishes_return_information_from_return_action() -> None:
    assert is_return_process_question("¿Cómo puedo hacer para regresar un pedido?") is True
    assert is_return_process_question("¿Cómo puedo realizar una devolución?") is True
    assert is_return_process_question("¿Cuáles son los pasos para devolver un pedido?") is True
    assert is_return_process_question("Explícame el proceso de devolución") is True
    assert is_return_process_question("Quiero devolver mi pedido") is False
    assert is_return_process_question("Necesito realizar una devolución") is False
    assert is_return_process_question("Devuelve PED-2026-0003") is False


def test_order_repository_uses_exact_match(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    path.write_text(json.dumps([sample_order().model_dump(mode="json")]), encoding="utf-8")
    repository = OrderRepository(path)

    assert repository.get("ped-2026-0003") == sample_order()
    assert repository.get("PED-2026-9999") is None


def test_consultar_pedido_lists_products_without_code(tmp_path: Path) -> None:
    orders_path = tmp_path / "orders.json"
    orders_path.write_text(json.dumps([sample_order().model_dump(mode="json")]), encoding="utf-8")
    orders = OrderRepository(orders_path)
    returns = ReturnRepository(tmp_path / "returns.json")
    registry = build_tool_registry(
        ToolDependencies(
            orders=orders,
            returns=returns,
            labels=LabelRepository(tmp_path / "labels", returns),
            knowledge=object(),
            eligibility=EligibilityService(),
        )
    )

    result = json.loads(
        registry[ToolName.CONSULTAR_PEDIDO].invoke({"order_code": "", "list_all": True})
    )

    assert result["ok"] is True
    assert result["data"]["total"] == 1
    assert result["data"]["orders"][0]["product_name"] == "Paños reutilizables"

    eligibility_result = json.loads(
        registry[ToolName.VERIFICAR_ELEGIBILIDAD].invoke({"order_code": "", "list_all": True})
    )
    assert eligibility_result["ok"] is True
    assert eligibility_result["data"]["eligibilities"][0]["eligible"] is True


def test_eligibility_rejects_perishables_and_accepts_home() -> None:
    service = EligibilityService()

    assert service.evaluate(sample_order()).eligible is True
    assert (
        service.evaluate(sample_order(product_type="perecederos", product_name="Huevos")).eligible
        is False
    )


def test_return_and_label_are_idempotent(tmp_path: Path) -> None:
    returns = ReturnRepository(tmp_path / "returns.json")
    labels = LabelRepository(tmp_path / "labels", returns)

    first = returns.register("PED-2026-0003", "Cambio de opinión", "same-key")
    second = returns.register("PED-2026-0003", "Otro texto", "same-key")
    first_label = labels.generate(first.return_id, "label-key")
    second_label = labels.generate(first.return_id, "label-key")

    assert first.return_id == second.return_id
    assert first_label.label_id == second_label.label_id
    assert Path(first_label.file_path).exists()


def test_knowledge_base_loads_pdf_and_csv() -> None:
    documents = load_knowledge_documents(Path("dataset/knowledge_base"))

    assert len(documents) >= 30
    assert any(item.metadata["document_type"] == "return_policy" for item in documents)
    assert any(item.metadata["document_type"] == "return_policy_addendum" for item in documents)
