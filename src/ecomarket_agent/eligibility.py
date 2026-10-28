"""Reglas deterministas para la preevaluación de devoluciones."""

from ecomarket_agent.models import EligibilityResult, Order


class EligibilityService:
    POLICY_REFERENCE = "POL-DEV-2026-V4 y POL-DEV-2026-V4-AD1"

    def evaluate(self, order: Order) -> EligibilityResult:
        category = order.product_type.casefold()
        name = order.product_name.casefold()

        if category == "perecederos":
            return EligibilityResult(
                order_code=order.order_code,
                eligible=False,
                decision="not_eligible",
                reason=(
                    "Los alimentos y productos perecederos están excluidos de la "
                    "devolución libre por razones sanitarias."
                ),
                policy_reference=self.POLICY_REFERENCE,
            )

        if category == "aseo general" or "recarga" in name or "a granel" in name:
            return EligibilityResult(
                order_code=order.order_code,
                eligible=False,
                decision="not_eligible",
                reason=(
                    "Los productos de aseo recargables, a granel o de higiene no se "
                    "preevalúan como retornables sin comprobar sello y ausencia de uso."
                ),
                conditions=["Un asesor debe validar que el producto esté cerrado y sin uso."],
                policy_reference=self.POLICY_REFERENCE,
            )

        return EligibilityResult(
            order_code=order.order_code,
            eligible=True,
            decision="preeligible",
            reason=(
                "El producto pertenece a una categoría durable admitida para "
                "preevaluación de devolución."
            ),
            return_window="30 días calendario desde la entrega efectiva",
            conditions=[
                "Producto sin uso, lavado, manchas ni modificaciones.",
                "Empaque ecológico original en el mejor estado posible.",
                "Accesorios, manuales y etiquetas completos.",
                "Tres fotografías para validación previa.",
                "Inspección final en bodega antes de cualquier reembolso.",
            ],
            policy_reference=self.POLICY_REFERENCE,
        )
