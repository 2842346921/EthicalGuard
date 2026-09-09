"""贝叶斯编排（EmoMAS 风格）：按协商反馈动态更新各 Agent 可靠性权重。"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..types import PrincipleVector, Proposal


class BayesianOrchestrator:
    """每个 Agent 可靠性服从 Beta(a, b)，权重 = a/(a+b)；被采纳 → a+=1，否则 b+=1。"""

    def __init__(self, agent_ids: List[str]):
        self.agent_ids = list(agent_ids)
        self.alpha: Dict[str, float] = {i: 1.0 for i in self.agent_ids}
        self.beta: Dict[str, float] = {i: 1.0 for i in self.agent_ids}

    def reset(self) -> None:
        """重置贝叶斯先验——每次协商会话（一次 run）独立。

        修复：orchestrator 此前是 engine 成员且跨 run 累积 alpha/beta，
        规则模式重跑被前一次会话污染（确定性协商出现 recovery_std≠0），
        且跨场景评估时先验串扰（场景 A 的可靠性历史影响场景 B 的集体向量）。
        """
        self.alpha = {i: 1.0 for i in self.agent_ids}
        self.beta = {i: 1.0 for i in self.agent_ids}

    def weights(self) -> np.ndarray:
        ids = list(self.alpha.keys())
        w = np.array([self.alpha[i] / (self.alpha[i] + self.beta[i]) for i in ids], dtype=float)
        s = w.sum()
        return w / s if s > 0 else np.ones(len(ids)) / len(ids)

    def update(self, agent_id: str, accepted: bool) -> None:
        if accepted:
            self.alpha[agent_id] += 1.0
        else:
            self.beta[agent_id] += 1.0

    def collective_vector(self, proposals: List[Proposal]) -> PrincipleVector:
        """集体原则向量 v = Σ w_i α_i（编排权重加权）。"""
        ids = [p.agent for p in proposals]
        w = self.weights()
        order = list(self.alpha.keys())
        wmap = {i: w[order.index(i)] for i in ids}
        v = np.zeros(4)
        tot = 0.0
        for p in proposals:
            wi = wmap.get(p.agent, 1.0 / max(1, len(proposals)))
            v += wi * p.principle_weights.as_array()
            tot += wi
        v = v / tot if tot > 0 else np.full(4, 0.25)
        return PrincipleVector.from_array(v)

    def satisfaction_from_vector(self, agent_id: str, v: np.ndarray, agent_weights: np.ndarray) -> float:
        """该 Agent 对当前集体向量的满意度（自身权重与集体向量的余弦相似）。"""
        a = np.asarray(agent_weights, dtype=float)
        s = a.sum()
        a = a / s if s > 0 else np.full(4, 0.25)
        denom = np.linalg.norm(a) * np.linalg.norm(v)
        if denom == 0:
            return 0.0
        return float(np.dot(a, v) / denom)
