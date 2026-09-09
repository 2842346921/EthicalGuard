"""冲突识别模块：ERS 伦理风险评分 + 冲突类型/强度 + 原则向量。

对应《项目架构规划》§1.2 工作① 与《组件设计/1冲突检测器.md》：
- 输出连续风险评分 ERS（预警式门控，而非二元判决）→ 高风险触发 MANE 协商；
- 三轨实现：监督检测器（四维注意力，第三方标注训练，主）/ LLM 检测器（语义）/ 规则基线。
"""
from __future__ import annotations

from ..types import ConflictReport, ConflictType
from .base import ACTION_LABELS, ConflictDetector
from .rule import RuleConflictDetector
from .llm import LLMConflictDetector
from .registry import make_detector

try:
    from .supervised import SupervisedConflictDetector
    _HAS_SUPERVISED = True
except ImportError:  # pragma: no cover
    _HAS_SUPERVISED = False

__all__ = [
    "ConflictReport",
    "ConflictType",
    "ACTION_LABELS",
    "ConflictDetector",
    "RuleConflictDetector",
    "LLMConflictDetector",
    "SupervisedConflictDetector",
    "make_detector",
    "_HAS_SUPERVISED",
]
