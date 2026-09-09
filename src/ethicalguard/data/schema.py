"""统一场景适配层。

把 5 个公开伦理数据集适配为《项目架构规划》§4.5 的标准场景 schema。
标准场景类型定义在 ``ethicalguard.types`` 中（Scenario/FourBoxState/Party/Constraint/Reference）。
"""
from __future__ import annotations

from ..types import (
    Scenario,
    FourBoxState,
    Party,
    Constraint,
    ConstraintKind,
    Reference,
    TemporalEvent,
)

__all__ = [
    "Scenario",
    "FourBoxState",
    "Party",
    "Constraint",
    "ConstraintKind",
    "Reference",
    "TemporalEvent",
]
