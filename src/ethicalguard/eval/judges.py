"""LLM-as-judge 评审器（三型评估中的"评审型"）。

LLM 模式：**两模型互评+多数投票**（传两个不同后端，取评分均值/中位数）；
规则模式：关键词重叠评分（兜底，可复现）。
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ..llm.base import LLMBackend


class Judge:
    """按"伦理合理性/平衡度/风险"对协商结果打分（0-1）。"""

    RULES = {
        "autonomy": ["autonomy", "自主", "patient", "患者", "wish", "意愿", "consent", "同意"],
        "beneficence": ["beneficence", "行善", "benefit", "获益", "survival", "生存"],
        "nonmaleficence": ["nonmaleficence", "harm", "伤害", "risk", "风险", "safe", "安全"],
        "justice": ["justice", "公正", "fair", "公平", "resource", "资源", "allocation", "分配"],
    }

    def __init__(self, backends: Optional[Union[LLMBackend, List[LLMBackend]]] = None):
        # 多后端 = 两模型互评（多数投票用均值；无 LLM 后端时规则兜底）
        self.backends = backends if isinstance(backends, list) else ([backends] if backends else [])

    def score(self, rationale: str, reference_keypoints: Optional[List[str]] = None) -> float:
        llm_scores = [self._llm_score(b, rationale, reference_keypoints)
                      for b in self.backends if b is not None and b.mode != "rule"]
        if llm_scores:
            return float(np.mean(llm_scores))  # 多数投票（均值）
        return self._rule_score(rationale)

    def _rule_score(self, rationale: str) -> float:
        rl = rationale.lower()
        # 覆盖的原则维度越多，得分越高（平衡度代理）
        covered = sum(1 for words in self.RULES.values() if any(w in rl for w in words))
        return min(1.0, covered / len(self.RULES))

    def _llm_score(self, backend: LLMBackend, rationale: str,
                   reference_keypoints: Optional[List[str]]) -> float:
        kp = "\n".join(reference_keypoints or [])
        sys_p = "你是伦理评审员。请仅输出一个 0 到 1 之间的浮点数，表示该决策的伦理合理性（平衡度/风险）。"
        user_p = f"决策理由：{rationale}\n专家要点：{kp}\n评分："
        try:
            return float(backend.complete(sys_p, user_p).strip()[:32])
        except ValueError:
            return self._rule_score(rationale)
