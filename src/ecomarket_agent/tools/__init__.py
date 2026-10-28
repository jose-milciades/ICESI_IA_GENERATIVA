"""Herramientas autorizadas para el ejecutor ReAct."""

from ecomarket_agent.models import ToolName

TOOL_CATALOG: tuple[ToolName, ...] = tuple(ToolName)

__all__ = ["TOOL_CATALOG"]
