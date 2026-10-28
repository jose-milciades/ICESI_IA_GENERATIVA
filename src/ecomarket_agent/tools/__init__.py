"""Herramientas autorizadas para el ejecutor ReAct."""

from ecomarket_agent.models import ToolName
from ecomarket_agent.tools.ecomarket import ToolDependencies, build_tool_registry

TOOL_CATALOG: tuple[ToolName, ...] = tuple(ToolName)

__all__ = ["TOOL_CATALOG", "ToolDependencies", "build_tool_registry"]
