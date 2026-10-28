"""Repositorios locales reemplazables por servicios transaccionales reales."""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from ecomarket_agent.models import Order, ReturnLabel, ReturnRecord

ORDER_CODE_PATTERN = re.compile(r"\bPED-\d{4}-\d{4}\b", re.IGNORECASE)


def extract_order_code(text: str) -> str | None:
    """Extrae un código sin delegar una coincidencia exacta al modelo."""
    match = ORDER_CODE_PATTERN.search(text or "")
    return match.group(0).upper() if match else None


class OrderRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list(self) -> list[Order]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("orders.json debe contener una lista.")
        return [Order.model_validate(item) for item in payload]

    def get(self, order_code: str) -> Order | None:
        normalized = order_code.strip().upper()
        matches = [order for order in self.list() if order.order_code == normalized]
        if len(matches) > 1:
            raise ValueError(f"Existen varias coincidencias para {normalized}.")
        return matches[0] if matches else None


class ReturnRepository:
    """Registro JSON atómico e idempotente para la simulación académica."""

    _lock = threading.Lock()

    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> list[ReturnRecord]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [ReturnRecord.model_validate(item) for item in payload]

    def register(self, order_code: str, reason: str, idempotency_key: str) -> ReturnRecord:
        with self._lock:
            records = self._read()
            existing = next(
                (item for item in records if item.idempotency_key == idempotency_key),
                None,
            )
            if existing:
                return existing

            digest = sha256(idempotency_key.encode("utf-8")).hexdigest()[:10].upper()
            record = ReturnRecord(
                return_id=f"DEV-{digest}",
                order_code=order_code,
                reason=reason,
                created_at=datetime.now(UTC).isoformat(),
                idempotency_key=idempotency_key,
            )
            records.append(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(
                    [item.model_dump(mode="json") for item in records],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
            return record

    def get(self, return_id: str) -> ReturnRecord | None:
        return next((item for item in self._read() if item.return_id == return_id), None)


class LabelRepository:
    """Genera una etiqueta de texto descargable y reproducible por idempotencia."""

    def __init__(self, directory: Path, returns: ReturnRepository) -> None:
        self.directory = directory
        self.returns = returns

    def generate(self, return_id: str, idempotency_key: str) -> ReturnLabel:
        record = self.returns.get(return_id)
        if record is None:
            raise ValueError(f"No existe la devolución {return_id}.")

        digest = sha256(idempotency_key.encode("utf-8")).hexdigest()[:10].upper()
        label_id = f"ETQ-{digest}"
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{label_id}.txt"
        created_at = datetime.now(UTC).isoformat()
        if not path.exists():
            path.write_text(
                "\n".join(
                    [
                        "ECOMARKET — GUÍA VERDE DE DEVOLUCIÓN",
                        f"Etiqueta: {label_id}",
                        f"Devolución: {record.return_id}",
                        f"Pedido: {record.order_code}",
                        "Destino: Centro de Logística Inversa EcoMarket",
                        "Estado: solicitud registrada; pendiente de inspección final",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        return ReturnLabel(
            label_id=label_id,
            return_id=return_id,
            file_path=str(path),
            created_at=created_at,
            idempotency_key=idempotency_key,
        )
