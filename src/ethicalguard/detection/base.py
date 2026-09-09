"""冲突识别：抽象基类与常量（数据模型在 ``ethicalguard.types`` 中定义，避免循环导入）。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import ConflictReport, ConflictType, PrincipleVector, Scenario  # noqa: F401

# 建议行动标签（与 ConflictReport.action 对应）
ACTION_LABELS = {
    "observe": "常规观察，无需干预",
    "review_24h": "24小时内复查伦理风险",
    "family_communication": "启动家属沟通",
    "ethics_consult": "启动伦理会诊",
    "intervene": "立即伦理介入/仲裁",
}


class ConflictDetector(ABC):
    """冲突检测器抽象：scenario -> ConflictReport。"""

    channel: str = "base"

    @abstractmethod
    def detect(self, scenario: Scenario) -> ConflictReport:
        raise NotImplementedError
