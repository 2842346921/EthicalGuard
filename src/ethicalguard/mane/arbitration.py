"""伦理委员会仲裁（L1 硬否决 + 保护最脆弱方 + CAMP 证据仲裁 + 资源强制修正）。"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import ConstraintKind, Proposal, Scenario, PrincipleVector, Syllogism
from .agents import EthicsCommitteeAgent
from .gne_solver import DEFAULT_TREATMENT_RESOURCE, scenario_resource_cap


def check_veto(scenario: Scenario, proposal: Proposal) -> List[str]:
    """检查提案是否违反 L1 硬约束/原则底线（用约束的 value/bound 代理）。

    **排除资源类约束**（resource_hard 等）：其 value 是场景静态压力值、与提案无关，
    检查它等于"场景 rp>bound 就否决"——会绕过精细的资源修正（enforce_resource_cap
    按 Σρ vs cap 逐级降级），把本可 t=2/3 的方案一刀切到 t=1
    （第五轮 PQ-4-9/PQ-4-10/MEE/MEQA 等 6 场景误触发）。资源问题统一由资源修正处理。
    """
    violated = []
    for c in scenario.constraints:
        if "resource" in c.name:
            continue
        if c.kind.value in ("L1", "L3") and c.violation() > 0:
            violated.append(c.name)
    return violated


def resource_usage(proposals) -> float:
    """Σρ(t_i)：治疗等级 → 资源用度（ρ: 0/0.3/0.6/1.0）。"""
    return float(sum(DEFAULT_TREATMENT_RESOURCE[p.treatment_level] for p in proposals))


def enforce_resource_cap(scenario: Scenario, proposal_map: Dict[str, Proposal]) -> Tuple[Dict[str, Proposal], List[str], str]:
    """资源 L1 强制修正（仲裁层职责，对应 gne_solver 注释"违规由仲裁层强制修正"）。

    当 Σρ(t_i) > cap 时，按治疗等级从高到低逐级降级（每次降 1 级）直到可行。
    cap 由 scenario_resource_cap 从场景资源压力推导（压力越大人均预算越低）。

    :param proposal_map: {agent: Proposal}（最后一个完整协商轮的提案，不含鲶鱼）
    :returns: (修正后的 map, 被降级的 agent 列表, 说明文本；已可行时原样返回)
    """
    cap = scenario_resource_cap(scenario, len(proposal_map))
    fixed = {a: p for a, p in proposal_map.items()}
    lowered: List[str] = []
    while resource_usage(list(fixed.values())) > cap + 1e-9:
        a, p = max(fixed.items(), key=lambda kv: kv[1].treatment_level)
        if p.treatment_level <= 0:
            break  # 已全部舒适护理仍超限（cap 理论 ≥0，不应发生）
        fixed[a] = p.model_copy(update={"treatment_level": p.treatment_level - 1})
        lowered.append(a)
    if not lowered:
        return fixed, lowered, ""
    used = resource_usage(list(fixed.values()))
    note = (f"资源 L1 强制修正：Σρ={used:.2f}（cap={cap:.2f}），"
            f"对 {sorted(set(lowered))} 逐级降级至可行")
    return fixed, lowered, note


def most_vulnerable_party(scenario: Scenario) -> str:
    """识别最脆弱方：决策能力最低的患者 或 经济压力最大的家庭。"""
    capacity = scenario.state.preference.get("capacity", 0.5)
    econ = scenario.state.context.get("insurance_stress", 0.0)
    if capacity < 0.4:
        return "patient"
    if econ > 0.5:
        return "family"
    return "patient"


def camp_vote(scenario: Scenario, proposal: Proposal, satisfaction: Optional[float] = None,
              satisfaction_floor: float = 0.3) -> str:
    """CAMP 三值投票（KEEP / REFUSE / NEUTRAL 弃权）——F3 玩家完备性机制。

    - KEEP    ：不违反 L1/L3、资源可承受、满意度达线 → 可采纳
    - REFUSE  ：违反 L1 硬约束（含资源超限）→ 否决
    - NEUTRAL ：满意度未知/信息不足 → 弃权（消除"被迫对专业外投票"的噪声）
    """
    if check_veto(scenario, proposal):
        return "REFUSE"
    cap = scenario_resource_cap(scenario, 1)  # 人均资源份额近似
    if resource_usage([proposal]) > cap + 1e-9:
        return "REFUSE"
    if satisfaction is not None and satisfaction < satisfaction_floor:
        return "REFUSE"
    if satisfaction is None:
        return "NEUTRAL"
    return "KEEP"


def camp_summary(votes: Dict[str, str]) -> str:
    """CAMP 投票摘要文本（写入裁决 rationale，供审计/论文）。"""
    from collections import Counter
    c = Counter(votes.values())
    return f"CAMP: KEEP×{c.get('KEEP', 0)} REFUSE×{c.get('REFUSE', 0)} NEUTRAL×{c.get('NEUTRAL', 0)}"


class Arbitrator:
    """伦理委员会：完全信息、L1 硬否决、多均衡/僵局时向最脆弱方倾斜。"""

    def __init__(self, committee: EthicsCommitteeAgent, satisfaction_floor: float = 0.3):
        self.committee = committee
        self.satisfaction_floor = satisfaction_floor

    def decide(
        self,
        scenario: Scenario,
        proposals: List[Proposal],
        collective: Optional[PrincipleVector],
        satisfactions: Dict[str, float],
    ) -> Proposal:
        # 鲶鱼异议者不占资源、不进集体/GNE —— 资源口径必须与 GNE 一致（5 方），
        # 否则 cap 按 6 方放大、Σρ 虚高、降级名单混入 catfish（B2 修复）
        proposals = [p for p in proposals if p.agent != "catfish"]
        # 0) CAMP 三值投票（KEEP/REFUSE/NEUTRAL）——F3 玩家完备性：记录每方立场，
        #    弃权消除"被迫对专业外投票"的噪声；投票摘要进入裁决 rationale（审计/论文）
        votes = {p.agent: camp_vote(scenario, p, satisfactions.get(p.agent), self.satisfaction_floor)
                 for p in proposals}
        vote_txt = camp_summary(votes)
        n_refuse = sum(1 for v in votes.values() if v == "REFUSE")

        # 1) L1 硬否决：任何提案违反硬约束 → 委员会强制降级
        for p in proposals:
            if check_veto(scenario, p):
                return self._constrained_ruling(scenario,
                                                f"{vote_txt} | 提案 {p.agent} 违反硬约束，强制修正", level=1)

        # 1.5) 资源 L1 硬否决：Σρ(t_i) > cap → 强制降级裁决（GNE 求解器无法自愈治疗方案）
        usg = resource_usage(proposals)
        cap = scenario_resource_cap(scenario, len(proposals))
        if usg > cap + 1e-9:
            fixed, lowered, _ = enforce_resource_cap(scenario, {p.agent: p for p in proposals})
            level = max(p.treatment_level for p in fixed.values())
            return self._constrained_ruling(
                scenario,
                f"{vote_txt} | 资源 L1 否决：Σρ(t)={usg:.2f} > cap={cap:.2f}，"
                f"对 {sorted(set(lowered))} 强制降级至可行",
                level=level,
            )

        # 2) 低满意度 → 向最脆弱方倾斜（CAMP 投出多数 REFUSE 时同样触发）
        if collective is not None and (n_refuse >= max(1, len(proposals) // 2)
                                       or min(satisfactions.values()) < self.satisfaction_floor):
            vul = most_vulnerable_party(scenario)
            return self._constrained_ruling(scenario,
                                            f"{vote_txt} | 有方满意度过低/多数否决，向最脆弱方({vul})倾斜", level=1)

        # 3) 默认：委员会给出平衡裁决（CAMP 无否决）
        ruling = self.committee._rule_act(scenario, 0, [])
        return ruling.model_copy(update={"rationale": f"{vote_txt} | {ruling.rationale}"})

    def _constrained_ruling(self, scenario: Scenario, rationale: str, level: int) -> Proposal:
        # 向最脆弱方倾斜，且裁决权重必须满足 L3 底线（[B,N,A,J]≥[0.15,0.20,0.15,0.20]）：
        # 旧权重 [0.20,0.25,0.40,0.15] 的 J=0.15 低于底线 0.20——裁决本身违反底线，
        # 与仲裁"底线守护者"定位矛盾（P-D 修复）。
        vul = most_vulnerable_party(scenario)
        if vul == "patient":
            w = np.array([0.21, 0.24, 0.35, 0.20])  # 抬自主 A，公正 J 恰好压底线 0.20
        else:
            w = np.array([0.22, 0.24, 0.22, 0.32])  # 抬公正 J（家庭/经济脆弱）
        w = w / w.sum()
        return Proposal(
            agent="ethics_committee", round=0, treatment_level=level,
            principle_weights=PrincipleVector.from_array(w), confidence=0.9,
            rationale=rationale,
            syllogism=Syllogism(major_premise="伦理委员会保护最脆弱方的利益，且不突破四原则底线",
                                minor_premise=rationale,
                                conclusion=f"约束性裁决：治疗强度 {level}"),
        )
