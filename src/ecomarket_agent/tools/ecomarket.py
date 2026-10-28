"""Construcción de las cinco herramientas permitidas al ejecutor ReAct."""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.tools import BaseTool, tool

from ecomarket_agent.eligibility import EligibilityService
from ecomarket_agent.models import ToolName
from ecomarket_agent.repositories import LabelRepository, OrderRepository, ReturnRepository
from ecomarket_agent.retrieval import KnowledgeService


def _success(data: object) -> str:
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False)


def _failure(code: str, message: str) -> str:
    return json.dumps(
        {"ok": False, "error": {"code": code, "message": message}},
        ensure_ascii=False,
    )


@dataclass(slots=True)
class ToolDependencies:
    orders: OrderRepository
    returns: ReturnRepository
    labels: LabelRepository
    knowledge: KnowledgeService
    eligibility: EligibilityService


def build_tool_registry(dependencies: ToolDependencies) -> dict[ToolName, BaseTool]:
    @tool(ToolName.CONSULTAR_PEDIDO.value)
    def consultar_pedido(order_code: str = "", list_all: bool = False) -> str:
        """Consulta un pedido exacto o lista productos cuando list_all es verdadero."""
        try:
            if list_all:
                orders = dependencies.orders.list()
                summaries = [
                    {
                        "order_code": order.order_code,
                        "product_name": order.product_name,
                        "product_type": order.product_type,
                        "order_date": order.order_date,
                        "last_event": order.last_event,
                    }
                    for order in orders
                ]
                return _success({"orders": summaries, "total": len(summaries)})
            if not order_code.strip():
                return _failure(
                    "missing_order_code",
                    "Indica un código de pedido o usa list_all=true para listar productos.",
                )
            order = dependencies.orders.get(order_code)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return _failure("order_repository_error", str(exc))
        if order is None:
            return _failure("order_not_found", f"No existe el pedido {order_code}.")
        return _success(order)

    @tool(ToolName.BUSCAR_POLITICA.value)
    def buscar_politica_devolucion(query: str, product_type: str = "") -> str:
        """Busca políticas, catálogo y preguntas frecuentes en la base RAG."""
        complete_query = f"{query} {product_type}".strip()
        try:
            results = dependencies.knowledge.search(complete_query, k=4)
        except (OSError, ValueError) as exc:
            return _failure("retrieval_error", str(exc))
        return _success([result.model_dump(mode="json") for result in results])

    @tool(ToolName.VERIFICAR_ELEGIBILIDAD.value)
    def verificar_elegibilidad(order_code: str = "", list_all: bool = False) -> str:
        """Preevalúa un pedido o todos aplicando reglas deterministas."""
        try:
            if list_all:
                evaluations = []
                for order in dependencies.orders.list():
                    result = dependencies.eligibility.evaluate(order)
                    evaluations.append(
                        {
                            "order_code": order.order_code,
                            "product_name": order.product_name,
                            **result.model_dump(mode="json"),
                        }
                    )
                return _success({"eligibilities": evaluations, "total": len(evaluations)})
            if not order_code.strip():
                return _failure(
                    "missing_order_code",
                    "Indica un código o usa list_all=true para comparar pedidos.",
                )
            order = dependencies.orders.get(order_code)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return _failure("order_repository_error", str(exc))
        if order is None:
            return _failure("order_not_found", f"No existe el pedido {order_code}.")
        return _success(dependencies.eligibility.evaluate(order))

    @tool(ToolName.REGISTRAR_DEVOLUCION.value)
    def registrar_devolucion(
        order_code: str,
        reason: str,
        idempotency_key: str,
        confirmed: bool,
    ) -> str:
        """Registra una devolución preelegible después de aprobación explícita."""
        if not confirmed:
            return _failure(
                "approval_required",
                "La devolución no puede registrarse sin confirmación explícita.",
            )
        order = dependencies.orders.get(order_code)
        if order is None:
            return _failure("order_not_found", f"No existe el pedido {order_code}.")
        eligibility = dependencies.eligibility.evaluate(order)
        if not eligibility.eligible:
            return _failure("not_eligible", eligibility.reason)
        try:
            return _success(dependencies.returns.register(order_code, reason, idempotency_key))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return _failure("return_repository_error", str(exc))

    @tool(ToolName.GENERAR_ETIQUETA.value)
    def generar_etiqueta(return_id: str, idempotency_key: str) -> str:
        """Genera la Guía Verde de una devolución previamente registrada."""
        try:
            return _success(dependencies.labels.generate(return_id, idempotency_key))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return _failure("label_error", str(exc))

    return {
        ToolName.CONSULTAR_PEDIDO: consultar_pedido,
        ToolName.BUSCAR_POLITICA: buscar_politica_devolucion,
        ToolName.VERIFICAR_ELEGIBILIDAD: verificar_elegibilidad,
        ToolName.REGISTRAR_DEVOLUCION: registrar_devolucion,
        ToolName.GENERAR_ETIQUETA: generar_etiqueta,
    }
