"""Agent 角色库。"""
from __future__ import annotations

from .base import Agent
from .roles import (
    CatfishAgent,
    EthicsCommitteeAgent,
    FamilyAgent,
    HospitalAdminAgent,
    PatientAgent,
    PhysicianAgent,
)

AGENT_CLASSES = {
    "patient": PatientAgent,
    "family": FamilyAgent,
    "physician": PhysicianAgent,
    "ethics_committee": EthicsCommitteeAgent,
    "hospital_admin": HospitalAdminAgent,
    "catfish": CatfishAgent,
}


def build_agent(spec, backend=None) -> Agent:
    cls = AGENT_CLASSES.get(spec.id)
    if cls is None:
        raise KeyError(f"未知 Agent 角色: {spec.id}")
    return cls(spec, backend)


__all__ = [
    "Agent",
    "CatfishAgent",
    "EthicsCommitteeAgent",
    "FamilyAgent",
    "HospitalAdminAgent",
    "PatientAgent",
    "PhysicianAgent",
    "build_agent",
    "AGENT_CLASSES",
]
