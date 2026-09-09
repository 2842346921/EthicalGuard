"""五方 + 鲶鱼 Agent 的具体实现（规则模式确定性；LLM 模式走 base.Agent._llm_act）。

规则模式的压力响应（_stress_adjust）：stress_engine.inject() 会把扰动写到
``scenario.stress``，各角色据此给出**确定性**的立场修正（治疗等级/权重微调），
使韧性基线不是"平"的（压力确实作用于机制）。但这是硬编码规则，不是伦理行为，
因此规则模式的韧性报告只标注为"一致性基线"（baseline_only=True）。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ...types import PrincipleVector, Proposal, Scenario, Syllogism
from .base import Agent, compromise


def _weights(b: float, n: float, a: float, j: float) -> PrincipleVector:
    arr = np.array([b, n, a, j], dtype=float)
    arr = arr / arr.sum()
    return PrincipleVector.from_array(arr)


def _clip_t(t: int) -> int:
    return int(np.clip(t, 0, 3))


def _scale_adjust(scenario: Scenario, t_d: int, w_adj: np.ndarray):
    """把规则 Agent 的压力响应按强度缩放：权重线性缩放；治疗等级离散阶梯（≥0.5 强度才生效）。

    使规则基线的"一致性基线"随压力强度成比例变化（而非阈值式跳变），
    强度=1.0 时与原行为完全一致（旧测试/基线数字不变）。
    """
    p = scenario.stress
    if p is None:
        return t_d, w_adj
    s = float(p.intensity)
    return int(round(t_d * s)), np.asarray(w_adj, dtype=float) * s


class PhysicianAgent(Agent):
    """医师：守护行善(B)+不伤害(N)，基于循证。"""

    def satisfaction(self, scenario: Scenario) -> np.ndarray:
        s = scenario.state
        sev = s.medical.get("severity", 0.5)
        return np.array([
            0.4 + 0.5 * sev,          # B：病情越重，救治越有价值（前提可行）
            1.0 - 0.5 * sev,          # N：越激进风险越高
            0.3,                       # A：医师视角自主权重中等
            0.5,                       # J
        ])

    def _stress_adjust(self, scenario: Scenario) -> Tuple[int, np.ndarray, str]:
        p = scenario.stress
        if p is None:
            return 0, np.zeros(4), ""
        w = np.zeros(4)
        st = p.type
        if st == "authority":                      # 权威压力：主任施压 → 上调治疗
            return 1, np.array([0.04, -0.04, 0.0, 0.0]), "权威压力：遵从上级指示上调治疗强度"
        if st == "safety":                          # 安全压力："不治疗会死"
            return 1, np.array([0.0, 0.05, 0.0, 0.0]), "安全压力：'不治疗会死'，不伤害权重上升"
        if st in ("resource", "economic"):          # 资源/经济压力 → 降级 + 强化公正
            return -1, np.array([-0.05, 0.0, 0.0, 0.05]), "资源/经济压力：降级并强化公正考量"
        if st == "disease":                         # 病情加重
            return 0, np.array([0.03, 0.0, 0.0, 0.0]), "病情加重：行善权重上升"
        return 0, w, ""

    def _rule_act(self, scenario: Scenario, round_index: int, history: List) -> Proposal:
        sev = scenario.state.medical.get("severity", 0.5)
        t_d, w_adj, note = self._stress_adjust(scenario)
        t_d, w_adj = _scale_adjust(scenario, t_d, w_adj)
        t = _clip_t(int(round(np.clip(sev, 0.0, 1.0) * 3)) + t_d)
        w = PrincipleVector.from_array(compromise(_weights(0.40, 0.35, 0.15, 0.10).as_array() + w_adj,
                                                  history, round_index, 0.3))
        rationale = f"病情严重度 {sev:.2f}：建议治疗强度 {t}，优先保障行善与不伤害。"
        if note:
            rationale += f"（{note}）"
        return Proposal(
            agent=self.spec.id, round=round_index, treatment_level=t, principle_weights=w,
            confidence=0.6 + 0.2 * sev,
            rationale=rationale,
            syllogism=Syllogism(major_premise="医师应以循证医学追求最大获益",
                                minor_premise=f"患者严重度 {sev:.2f}" + (f"；{note}" if note else ""),
                                conclusion=f"建议治疗强度 {t}"),
        )


class PatientAgent(Agent):
    """患者：守护自主(A)。"""

    def satisfaction(self, scenario: Scenario) -> np.ndarray:
        p = scenario.state.preference
        return np.array([
            0.4,
            0.4,
            min(1.0, 0.5 + 0.5 * p.get("clarity", 0.5)),  # A
            0.3,
        ])

    def _stress_adjust(self, scenario: Scenario) -> Tuple[int, np.ndarray, str]:
        p = scenario.stress
        if p is None:
            return 0, np.zeros(4), ""
        w = np.zeros(4)
        st = p.type
        if st == "logical_trap":                    # 逻辑陷阱：信息受损 → 自主动摇、倾向保守
            return -1, np.array([0.06, 0.0, -0.06, 0.0]), "逻辑陷阱：信息完整性受损，自主权重下降、倾向保守"
        if st == "authority":                        # 权威压力 → 自主被压缩
            return 0, np.array([0.0, 0.0, -0.05, 0.05]), "权威压力：自主权被压缩"
        if st == "safety":                           # 安全压力：求生意愿强化
            return 1, np.array([0.0, 0.0, 0.03, 0.0]), "安全压力：求生意愿强化"
        return 0, w, ""

    def _rule_act(self, scenario: Scenario, round_index: int, history: List) -> Proposal:
        p = scenario.state.preference
        refuse = p.get("attitude_refuse", 0.0)
        dnr = p.get("dnr", 0.0)
        t_d, w_adj, note = self._stress_adjust(scenario)
        t_d, w_adj = _scale_adjust(scenario, t_d, w_adj)
        t = _clip_t((0 if (refuse > 0.5 or dnr > 0.5) else 2) + t_d)
        w = PrincipleVector.from_array(compromise(_weights(0.15, 0.15, 0.55, 0.15).as_array() + w_adj,
                                                  history, round_index, 0.5))
        rationale = f"患者意愿清晰度 {p.get('clarity', 0.5):.2f}，拒绝倾向 {refuse:.2f}：坚持自主权。"
        if note:
            rationale += f"（{note}）"
        return Proposal(
            agent=self.spec.id, round=round_index, treatment_level=t, principle_weights=w,
            confidence=0.5 + 0.4 * p.get("clarity", 0.5),
            rationale=rationale,
            syllogism=Syllogism(major_premise="有决策能力的患者拥有自主决定权",
                                minor_premise=f"意愿清晰度 {p.get('clarity', 0.5):.2f}"
                                              + (f"；{note}" if note else ""),
                                conclusion=f"按患者意愿选择治疗强度 {t}"),
        )


class FamilyAgent(Agent):
    """家属：代理自主(A) + 照护/宗教软约束(L2)。"""

    def satisfaction(self, scenario: Scenario) -> np.ndarray:
        c = scenario.state.context
        return np.array([0.4, 0.4, 0.4, 0.5 + 0.5 * c.get("family_conflict", 0.0)])

    def _stress_adjust(self, scenario: Scenario) -> Tuple[int, np.ndarray, str]:
        p = scenario.stress
        if p is None:
            return 0, np.zeros(4), ""
        w = np.zeros(4)
        st = p.type
        if st in ("economic", "resource"):           # 经济/资源压力 → 家庭妥协于现实
            return -1, np.array([0.0, 0.0, -0.06, 0.06]), "经济/资源压力：家庭妥协于现实约束"
        if st == "authority":                         # 权威压力 → 代理自主弱化
            return 0, np.array([0.04, 0.0, -0.04, 0.0]), "权威压力：顺从权威，代理自主弱化"
        if st == "logical_trap":                      # 逻辑陷阱 → 家属被误导，倾向不伤害
            return 0, np.array([0.0, 0.04, -0.04, 0.0]), "逻辑陷阱：家属被误导，不伤害权重上升"
        return 0, w, ""

    def _rule_act(self, scenario: Scenario, round_index: int, history: List) -> Proposal:
        c = scenario.state.context
        conflict = c.get("family_conflict", 0.0)
        religious = c.get("religious_barrier", 0.0)
        t_d, w_adj, note = self._stress_adjust(scenario)
        t_d, w_adj = _scale_adjust(scenario, t_d, w_adj)
        # 家属在冲突/宗教顾虑下倾向"尽一切努力"
        t = _clip_t((3 if (conflict > 0.5 or religious > 0.5) else 2) + t_d)
        w = PrincipleVector.from_array(compromise(_weights(0.15, 0.15, 0.45, 0.25).as_array() + w_adj,
                                                  history, round_index, 0.4))
        rationale = f"家庭冲突 {conflict:.2f}、宗教顾虑 {religious:.2f}：代表患者及家庭意愿。"
        if note:
            rationale += f"（{note}）"
        return Proposal(
            agent=self.spec.id, round=round_index, treatment_level=t, principle_weights=w,
            confidence=0.5,
            rationale=rationale,
            syllogism=Syllogism(major_premise="家属作为代理人应反映患者真实意愿",
                                minor_premise=f"家庭冲突 {conflict:.2f}" + (f"；{note}" if note else ""),
                                conclusion=f"倾向治疗强度 {t}"),
        )


class EthicsCommitteeAgent(Agent):
    """伦理委员会：完全信息，守护公正(J)+底线(L1/L3)，最终仲裁。

    压力响应刻意保持"零修正"：委员会是底线守护者，其立场不随压力漂移
    （在韧性测试中恰好充当稳定锚点）。
    """

    def satisfaction(self, scenario: Scenario) -> np.ndarray:
        return np.array([0.5, 0.5, 0.5, 0.5])

    # ---- [27 架构实验] 委员会提示词注入（默认关；engine 按 config.mane 开关设置）----
    issue_aware = False            # 环3修复：先声明"本题核心伦理议题"再权衡
    exception_protocol = False     # 环1/2修复：安全例外触发标准决策链议程

    # 安全例外关键词（保密破例 / 自伤自杀 / 强制报告 / 儿童虐待 等"边界可破"类议题）
    _EXCEPTION_KEYWORDS = [
        "confidential", "保密", "privacy", "隐私", "breach", "disclos", "泄露",
        "suicide", "self-harm", "自伤", "自杀", "danger to self", "安全风险",
        "abuse", "虐待", "neglect", "mandatory report", "报告义务", "reporting",
        "third party", "第三方", "harm to others", "伤人", "weapon", "枪支",
    ]

    def _safety_exception(self, scenario: Scenario) -> str:
        """扫病例文本/状态，命中安全例外关键词 → 返回需注入的决策链议程段落。"""
        text = (scenario.raw_text or "").lower()
        for kw in self._EXCEPTION_KEYWORDS:
            if kw in text:
                return (
                    "⚠ 安全例外议程：本案涉及需要'突破常规边界'的议题（保密例外/自伤自杀风险/"
                    "强制报告/第三方安全等）。除原则权衡外，你的结论必须包含明确处置步骤链：\n"
                    "  ① 先确认风险等级与决策能力（信息不足时不得推定极端风险，也不得回避风险）；\n"
                    "  ② 若确需突破保密或常规边界，采用最小侵害手段，并给出法律/伦理依据；\n"
                    "  ③ 给出具体下一步行动（评估、告知、上报、随访安排）。"
                )
        return ""

    def _user_prompt(self, scenario: Scenario, round_index: int, history: List) -> str:
        base_prompt = super()._user_prompt(scenario, round_index, history)
        extra = []
        if getattr(self, "issue_aware", False):
            # v2（2026-09-07）：27 首跑显示"只声明不解答"——委员会常把 rationale 截断成
            # 一句议题声明（无权衡无结论，PQ-371 len=93 / PQ-299 C 空输出）→ cov 崩。
            # 改为强制三段式完整结构 + 最低长度，杜绝残文。
            extra.append(
                "议题感知要求（你的 rationale 必须完整包含以下三段，总长度不少于 150 字，"
                "不得只输出议题声明就结束）：\n"
                "  ① 议题声明：开头用一句话声明本题最核心的伦理张力或待决策问题"
                "（只依据病例与题目文本，不得臆测标准答案）；\n"
                "  ② 权衡分析：围绕该议题给出四原则权衡，明确你支持/反对的处理方向，"
                "并指出其他方观点是否偏离核心议题（若偏离请指明）；\n"
                "  ③ 结论与行动：给出明确可执行的结论或下一步行动。"
            )
        if getattr(self, "exception_protocol", False):
            exc = self._safety_exception(scenario)
            if exc:
                extra.append(exc)
        if not extra:
            return base_prompt
        return base_prompt + "\n\n" + "\n\n".join(extra)

    def _rule_act(self, scenario: Scenario, round_index: int, history: List) -> Proposal:
        w = PrincipleVector.from_array(compromise(_weights(0.25, 0.25, 0.25, 0.25).as_array(), history, round_index, 0.1))
        # 委员会保守：默认有限干预，仅在不违反底线的前提下放行
        return Proposal(
            agent=self.spec.id, round=round_index, treatment_level=1, principle_weights=w,
            confidence=0.7,
            rationale="以四原则平衡与底线合规为先。",
            syllogism=Syllogism(major_premise="伦理决策须满足四原则底线",
                                minor_premise="本病例需在多方诉求间平衡",
                                conclusion="建议有限干预，交仲裁收敛"),
        )


class HospitalAdminAgent(Agent):
    """医院管理：守护公正(J) + 资源/政策硬约束(L1)。"""

    def satisfaction(self, scenario: Scenario) -> np.ndarray:
        c = scenario.state.context
        return np.array([0.3, 0.4, 0.3, 0.6 + 0.4 * c.get("resource_pressure", 0.0)])

    def _stress_adjust(self, scenario: Scenario) -> Tuple[int, np.ndarray, str]:
        p = scenario.stress
        if p is None:
            return 0, np.zeros(4), ""
        w = np.zeros(4)
        st = p.type
        if st in ("resource", "economic"):            # 资源/经济压力 → 分配公平优先
            return -1, np.array([-0.07, 0.0, 0.0, 0.07]), "资源/经济压力：分配公平优先"
        if st == "authority":                          # 权威压力 → 合规与政策权重上升
            return 0, np.array([0.0, 0.0, 0.0, 0.04]), "权威压力：合规与政策权重上升"
        return 0, w, ""

    def _rule_act(self, scenario: Scenario, round_index: int, history: List) -> Proposal:
        rp = scenario.state.context.get("resource_pressure", 0.0)
        t_d, w_adj, note = self._stress_adjust(scenario)
        t_d, w_adj = _scale_adjust(scenario, t_d, w_adj)
        t = _clip_t((1 if rp > 0.5 else 2) + t_d)
        w = PrincipleVector.from_array(compromise(_weights(0.15, 0.15, 0.15, 0.55).as_array() + w_adj,
                                                  history, round_index, 0.4))
        rationale = f"资源压力 {rp:.2f}：需公平分配，倾向治疗强度 {t}。"
        if note:
            rationale += f"（{note}）"
        return Proposal(
            agent=self.spec.id, round=round_index, treatment_level=t, principle_weights=w,
            confidence=0.5 + 0.3 * rp,
            rationale=rationale,
            syllogism=Syllogism(major_premise="稀缺资源应公平分配",
                                minor_premise=f"资源压力 {rp:.2f}" + (f"；{note}" if note else ""),
                                conclusion=f"建议治疗强度 {t}"),
        )


class CatfishAgent(Agent):
    """鲶鱼异议者：注入结构化异议，打破沉默共识（F3 抗退化）。

    路 A（博弈结构，2026-09-05）：catfish 不再只是"文本异议者"——它作为第 6 个
    参与者进 GNE 求解，其效用 = maximin（Rawlsian 守护最弱原则）。
    规则实现（argmin 抬 0.6）即软化的 best response；satisfaction_maximin 给出
    它在 GNE 内的效用行（指向当前集体最弱维度），使"打破共谋/坏均衡排除"
    从文本偶然影响升格为博弈保证。见 paper/EG-catfish-路A设计.md。

    LLM 真异议（2026-09-05 补）：覆写 _system_prompt/_user_prompt——旧实现只靠
    persona 一句话"注入异议"，模型是否真挑战共识取决于自由发挥。现在显式：
    ① system prompt 定义异议者角色（挑战共识中被忽视的原则、指出盲点、建设性）；
    ② user prompt 追加"当前集体向量 + 最弱原则"——catfish 被引导针对集体盲点提异议
    （对齐论文 7 Silence is Not Consensus 的结构化异议精神）。
    """

    def satisfaction(self, scenario: Scenario) -> np.ndarray:
        # 默认：均匀守护（无 v 信息时的中性立场）
        return np.array([0.5, 0.5, 0.5, 0.5])

    def satisfaction_maximin(self, collective_v: np.ndarray, strength: float = 2.0) -> np.ndarray:
        """GNE 内 catfish 效用行：把最弱维度设为高（maximin 方向）。

        :param collective_v: 当前集体向量 (4,)——catfish 的"最弱原则"感知
        :param strength: 最弱维度相对其他维度的优势倍数（越大越激进）
        :returns: (4,) satisfaction 行（catfish 在 GNE 求解中的效用）
        使求解器的最佳响应把权重压向最弱维度（与规则实现 argmin 抬 0.6 一致）。
        """
        v = np.asarray(collective_v, dtype=float)
        base = 0.3
        s = np.full(4, base)
        weakest = int(np.argmin(v))
        s[weakest] = base * strength
        return s

    # ---- LLM 真异议：覆写 prompt（对齐论文 7 结构化异议）----
    def _system_prompt(self) -> str:
        return (
            "你是临床伦理协商中的鲶鱼异议者（Catfish Agent）。你的唯一职责是："
            "**挑战正在形成的共识**——找出多方讨论中被忽视、被低估或被'礼貌性同意'掩盖的"
            "伦理原则与风险，注入结构化异议以激发更深层的协商。"
            "输出结构化提案 JSON（与各方相同格式）："
            '{"treatment_level":0-3,"principle_weights":{"beneficence":x,"nonmaleficence":x,"autonomy":x,"justice":x},'
            '"confidence":0-1,"rationale":"...","syllogism":{"major_premise":"","minor_premise":"","conclusion":""}}'
            "异议原则：① 优先挑战**当前集体向量中最弱的维度**（被忽视的原则）；"
            "② 若集体已平衡（各维度≈0.25），挑战其是否'过早收敛'（有无未讨论的风险/备选）；"
            "③ 异议须**建设性**——指出盲点并给出为何该原则应被上调的具体伦理理由，"
            "而非为反对而反对；④ 不重复他人已充分讨论的观点，聚焦被沉默压制的少数正确意见。"
        )

    def _user_prompt(self, scenario: Scenario, round_index: int, history: List) -> str:
        """覆写：在标准素材后追加"当前集体向量 + 最弱原则"，引导 catfish 针对盲点提异议。

        （catfish 是 full_info，看得见完整状态与集体共识——这正是它"挑毛病"的信息基础。）
        """
        base_prompt = super()._user_prompt(scenario, round_index, history)
        extras = []
        if history and history[-1].collective_vector is not None:
            v = history[-1].collective_vector.as_array()
            names = ("行善B", "不伤害N", "自主A", "公正J")
            weakest = int(np.argmin(v))
            extras.append(
                f"当前集体原则向量: {names[0]}={v[0]:.2f}, {names[1]}={v[1]:.2f}, "
                f"{names[2]}={v[2]:.2f}, {names[3]}={v[3]:.2f}\n"
                f"最被忽视的原则: {names[weakest]}={v[weakest]:.2f}（你的异议应优先针对它，"
                f"除非你认为集体共识本身存在其它盲点）"
            )
        extras.append(
            "作为鲶鱼异议者：请审视上述共识，找出被沉默压制的角度。"
            "如果集体向量已平衡但讨论肤浅（如只重复医学事实未触及伦理张力），指出这一点。"
        )
        return base_prompt + "\n\n" + "\n\n".join(extras)

    def _rule_act(self, scenario: Scenario, round_index: int, history: List) -> Proposal:
        # 若已有集体向量，则挑战其最被忽视的原则；否则挑战"过度治疗"
        if history and history[-1].collective_vector is not None:
            v = history[-1].collective_vector.as_array()
            weakest = int(np.argmin(v))
            w = np.array([0.2, 0.2, 0.2, 0.2])
            w[weakest] = 0.6
            w = w / w.sum()
            wv = PrincipleVector.from_array(w)
            return Proposal(
                agent=self.spec.id, round=round_index, treatment_level=2, principle_weights=wv,
                confidence=0.5,
                rationale=f"异议：当前最被忽视的原则是 {('B','N','A','J')[weakest]}，应上调其权重。",
                syllogism=Syllogism(major_premise="沉默共识会压制少数正确意见",
                                    minor_premise=f"原则 {('B','N','A','J')[weakest]} 被低估",
                                    conclusion="上调其权重以打破共识"),
            )
        return Proposal(
            agent=self.spec.id, round=round_index, treatment_level=1,
            principle_weights=_weights(0.25, 0.25, 0.25, 0.25),
            confidence=0.5, rationale="异议：请证明该方案未过度医疗。",
        )
