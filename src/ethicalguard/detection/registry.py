"""检测器工厂：监督模型（checkpoint 配置）> LLM 检测器（api/local）> 规则基线（rule）。"""
from __future__ import annotations

from ..llm.base import LLMBackend
from .base import ConflictDetector
from .llm import LLMConflictDetector
from .rule import RuleConflictDetector


def make_detector(backend: LLMBackend | None = None, checkpoint: str = "",
                  threshold: float = 0.5) -> ConflictDetector:
    """按可用性选择检测器。

    优先级：监督模型权重（checkpoint 非空且可加载）→ LLM 检测器（api/local）→ 规则基线。
    threshold: ERS 门控阈值（O4：传入 LLM 检测器，替换 prompt 内硬编码的 0.3，见 config.detection.threshold）。
    """
    if checkpoint:
        try:
            from .supervised import SupervisedConflictDetector
            return SupervisedConflictDetector(checkpoint, threshold=threshold)
        except Exception:
            pass  # 权重加载失败 → 降级
    if backend is not None and backend.mode != "rule":
        return LLMConflictDetector(backend, threshold=threshold)
    return RuleConflictDetector()
