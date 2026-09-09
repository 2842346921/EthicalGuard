"""基线方法：单 LLM 直接决策 与 HARMONY 式原则对抗（规则模式实现）。"""
from __future__ import annotations

from typing import List

import numpy as np

from ...types import PrincipleVector, Proposal, Scenario, Syllogism


def single_llm_baseline(scenario: Scenario) -> Proposal:
    """单 LLM 直接决策（规则近似）：医师视角的单一方案，无协商。"""
    sev = scenario.state.medical.get("severity", 0.5)
    t = int(round(np.clip(sev, 0.0, 1.0) * 3))
    w = np.array([0.40, 0.35, 0.15, 0.10])
    w = w / w.sum()
    return Proposal(
        agent="single_llm", round=1, treatment_level=t, principle_weights=PrincipleVector.from_array(w),
        confidence=0.5, rationale="单一模型直接决策（无多方协商）。",
        syllogism=Syllogism(major_premise="单模型给出临床决策", minor_premise="", conclusion=f"治疗强度 {t}"),
    )


def harmony_style_baseline(scenario: Scenario, rounds: int = 3) -> List[Proposal]:
    """HARMONY 式原则对抗：行善原则 vs 自主原则 相互挑战，胜者输出（规则近似）。"""
    pro_beneficence = [0.45, 0.35, 0.10, 0.10]
    pro_autonomy = [0.10, 0.10, 0.55, 0.25]
    preference_autonomy = scenario.state.preference.get("clarity", 0.5) * scenario.state.preference.get("capacity", 0.5)
    medical_severity = scenario.state.medical.get("severity", 0.5)
    # 对抗结果：以"行善 vs 自主"张力强度决定胜者
    if medical_severity > preference_autonomy:
        winner = pro_beneficence
        rationale = "对抗结论：医疗紧迫压倒自主（行善胜出）"
    else:
        winner = pro_autonomy
        rationale = "对抗结论：患者自主清晰压倒医疗建议（自主胜出）"
    w = np.array(winner, dtype=float)
    w = w / w.sum()
    return [Proposal(
        agent="harmony_adversarial", round=r + 1, treatment_level=2, principle_weights=PrincipleVector.from_array(w),
        confidence=0.5, rationale=f"第{r+1}轮对抗：{rationale}",
    ) for r in range(rounds)]


def rule_collective_weights(scenario: Scenario) -> np.ndarray:
    """无 GNE 的朴素集体：规则五方提案权重的简单平均（替代"贝叶斯加权+求解器"）。

    与 06_baselines 共用同一实现——分布型相对比较（07）须与基线表（06）同一口径。
    """
    from ...config import default_agent_specs
    from ...mane.agents import build_agent
    specs = default_agent_specs()
    w = np.zeros(4)
    n = 0
    for aid in ("physician", "patient", "family", "ethics_committee", "hospital_admin"):
        agent = build_agent(specs[aid], None)  # 规则模式
        p = agent._rule_act(scenario, 1, [])
        w += p.principle_weights.as_array()
        n += 1
    return w / n
