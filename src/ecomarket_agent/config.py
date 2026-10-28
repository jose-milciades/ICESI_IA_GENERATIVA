"""Configuración central, sin acceso directo a secretos desde otros módulos."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    environment: str
    model_name: str
    data_dir: Path
    milvus_uri: Path
    langsmith_project: str
    max_executions: int
    max_replans: int

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Settings":
        load_dotenv(env_file, override=False)
        return cls(
            environment=os.getenv("APP_ENV", "dev"),
            model_name=os.getenv("ECOMARKET_MODEL", "gemini-2.5-flash-lite"),
            data_dir=Path(os.getenv("ECOMARKET_DATA_DIR", "dataset")),
            milvus_uri=Path(
                os.getenv("ECOMARKET_MILVUS_URI", "dataset/milvus_lite.db")
            ),
            langsmith_project=os.getenv(
                "LANGSMITH_PROJECT", "ecomarket-final-dev"
            ),
            max_executions=int(os.getenv("ECOMARKET_MAX_EXECUTIONS", "10")),
            max_replans=int(os.getenv("ECOMARKET_MAX_REPLANS", "3")),
        )
