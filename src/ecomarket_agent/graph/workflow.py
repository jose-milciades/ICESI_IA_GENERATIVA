"""Nombres estables de los nodos que compondrán el StateGraph."""

from enum import StrEnum


class NodeName(StrEnum):
    PREPARE_INPUT = "prepare_input"
    PLANNER = "planner"
    EXECUTOR = "executor"
    REPLANNER = "replanner"
    HUMAN_INPUT = "human_input"
    APPROVAL = "approval"
    FINAL_RESPONSE = "final_response"
    HUMAN_ESCALATION = "human_escalation"
