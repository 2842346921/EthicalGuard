"""协商质量指标（闭环环 9："个体让步 vs 集体稳定"）。

审稿人视角：组件独立性会招致"你只是把 15 个模块放一个框里"的批评。
本模块把"协商过程"本身量化——韧性压力测试不再只报集体层面指标，
而是反过来验证协商质量：压力下个体是否合理让步（让步幅度/对称性）、
集体是否保持稳定（跨轮波动小、收敛不崩）。个体让步 vs 集体稳定
是"系统而非组合"的最后一环证据：每个机制的输出进入下一个。

- concessions(history)：各方从首轮到末轮提案权重向量的偏移（谁让了步、让了多少）
- collective_volatility(history)：集体向量跨轮最大波动（稳定度）
- concession_inequality(concessions)：让步不对称性（离散系数；0=均衡让步）
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..types import NegotiationRound


def concessions(history: List[NegotiationRound]) -> Dict[str, float]:
    """各方让步幅度 = ‖首轮提案权重 − 末轮提案权重‖_1。

    返回 {agent: 让步幅度}；单轮或无双提案的场景该 Agent 让步为 0。
    规则模式确定性（compromise 向集体靠拢）下让步应随轮次递增；
    压力下让步增大 = 个体妥协换共识；让步为 0 或微小 = 立场僵持。
    """
    if not history:
        return {}
    first: Dict[str, np.ndarray] = {}
    last: Dict[str, np.ndarray] = {}
    for r in history:
        for p in r.proposals:
            w = p.principle_weights.as_array()
            if p.agent not in first:
                first[p.agent] = w
            last[p.agent] = w  # 最后出现的即为该轮末态
    out: Dict[str, float] = {}
    for agent, w0 in first.items():
        w1 = last.get(agent)
        if w1 is None:
            out[agent] = 0.0
        else:
            out[agent] = float(np.abs(w0 - w1).sum())
    return out


def collective_volatility(history: List[NegotiationRound]) -> float:
    """集体向量跨轮最大波动 = max_t ‖v_{t+1} − v_t‖_1。

    压力下集体仍稳定（波动小）= 协商过程有韧性；波动大说明集体被扰动带偏。
    结合 concessions 一起看：个体让得多而集体波动小 = 健康让步；
    个体没让而集体波动大 = 集体被单方带偏（不健康）。
    """
    vecs = [r.collective_vector.as_array() for r in history if r.collective_vector is not None]
    if len(vecs) < 2:
        return 0.0
    return float(max(np.abs(vecs[t + 1] - vecs[t]).sum() for t in range(len(vecs) - 1)))


def concession_inequality(concessions_map: Dict[str, float]) -> float:
    """让步不对称性 = 让步幅度的变异系数 CV = σ/μ（0=各方让步均衡）。

    失衡（高 CV）：少数方独自让步、其余僵持——非多方共识的迹象，
    与"对各方合理的共识"（核心目标）相矛盾，审计时应关注。
    """
    vals = np.array(list(concessions_map.values()), dtype=float)
    mu = float(vals.mean())
    if mu <= 1e-9:
        return 0.0
    return float(vals.std() / mu)


def summary(history: List[NegotiationRound]) -> Dict[str, float]:
    """一次协商的协商质量摘要（写进韧性报告 per_perturbation）。"""
    c = concessions(history)
    vol = collective_volatility(history)
    return {
        "concession_mean": float(np.mean(list(c.values()))) if c else 0.0,
        "concession_max": float(np.max(list(c.values()))) if c else 0.0,
        "concession_inequality": concession_inequality(c),
        "collective_volatility": vol,
        "n_agents_conceded": float(sum(1 for v in c.values() if v > 1e-6)),
    }
