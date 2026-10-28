"""Composición de dependencias y creación de la aplicación ejecutable."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI

from ecomarket_agent.agents import PlanExecuteAgents
from ecomarket_agent.config import Settings
from ecomarket_agent.eligibility import EligibilityService
from ecomarket_agent.graph.workflow import build_workflow
from ecomarket_agent.repositories import LabelRepository, OrderRepository, ReturnRepository
from ecomarket_agent.retrieval import KnowledgeService
from ecomarket_agent.tools import ToolDependencies, build_tool_registry


@dataclass(slots=True)
class EcoMarketApplication:
    graph: object
    settings: Settings
    checkpoint_connection: sqlite3.Connection | None = None


def _create_checkpointer(settings: Settings):
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver(), None

    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
    return SqliteSaver(connection), connection


def create_application(settings: Settings | None = None) -> EcoMarketApplication:
    settings = settings or Settings.from_env()
    model = ChatGoogleGenerativeAI(
        model=settings.model_name,
        temperature=0,
        timeout=60,
        max_retries=2,
    )
    orders = OrderRepository(settings.data_dir / "orders.json")
    returns = ReturnRepository(settings.returns_file)
    labels = LabelRepository(settings.labels_dir, returns)
    dependencies = ToolDependencies(
        orders=orders,
        returns=returns,
        labels=labels,
        knowledge=KnowledgeService(settings),
        eligibility=EligibilityService(),
    )
    registry = build_tool_registry(dependencies)
    agents = PlanExecuteAgents(model, registry, settings)
    checkpointer, connection = _create_checkpointer(settings)
    graph = build_workflow(agents, checkpointer=checkpointer)
    return EcoMarketApplication(
        graph=graph,
        settings=settings,
        checkpoint_connection=connection,
    )
