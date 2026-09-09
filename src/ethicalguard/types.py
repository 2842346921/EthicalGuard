"""EthicalGuard 核心数据模型。

所有跨模块共享的类型集中于此，与《项目架构规划》§4.5 的标准场景 schema 对应。
原则顺序固定为 [B, N, A, J] = [行善, 不伤害, 自主, 公正]。
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

# 四原则固定顺序
PRINCIPLES: tuple[str, ...] = ("beneficence", "nonmaleficence", "autonomy", "justice")
# HALF 危害加权放弃代价（N=3, J=2, B=1.5, A=1）——可解释参数，需做敏感性分析
HALF_WEIGHTS: np.ndarray = np.array([1.5, 3.0, 1.0, 2.0])


class TreatmentLevel(Enum):
    """治疗强度 0-3：舒适护理 / 有限干预 / 全面医疗 / 积极治疗。"""

    COMFORT = 0
    LIMITED = 1
    FULL = 2
    AGGRESSIVE = 3


class PrincipleVector(BaseModel):
    """四原则满足度向量 v_t = [B, N, A, J]。"""

    beneficence: float = Field(ge=0.0, le=1.0, description="行善")
    nonmaleficence: float = Field(ge=0.0, le=1.0, description="不伤害")
    autonomy: float = Field(ge=0.0, le=1.0, description="自主")
    justice: float = Field(ge=0.0, le=1.0, description="公正")

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.beneficence, self.nonmaleficence, self.autonomy, self.justice],
            dtype=float,
        )

    @classmethod
    def from_array(cls, arr) -> "PrincipleVector":
        a = np.asarray(arr, dtype=float).reshape(-1)
        if a.shape[0] != 4:
            raise ValueError(f"原则向量必须为 4 维，收到 {a.shape[0]} 维")
        return cls(
            beneficence=float(a[0]),
            nonmaleficence=float(a[1]),
            autonomy=float(a[2]),
            justice=float(a[3]),
        )

    def normalized(self) -> "PrincipleVector":
        """归一化到和=1（用于权重分配）。"""
        a = self.as_array()
        s = a.sum()
        if s <= 0:
            a = np.array([0.25, 0.25, 0.25, 0.25])
        else:
            a = a / s
        return PrincipleVector.from_array(a)

    def halved_drift(self, other: "PrincipleVector") -> float:
        """HALF 加权偏差范数 Drift = sqrt(Σ w_k (Δv_k)^2)。"""
        d = self.as_array() - other.as_array()
        return float(np.sqrt(np.sum(HALF_WEIGHTS * d * d)))


class Syllogism(BaseModel):
    """三段论逻辑树（MedLA 风格）：大前提(伦理原则/指南)→小前提(情境事实)→结论。"""

    major_premise: str
    minor_premise: str
    conclusion: str


class Proposal(BaseModel):
    """单个 Agent 的一轮提案。"""

    agent: str
    round: int = 0
    treatment_level: int = Field(ge=0, le=3)
    principle_weights: PrincipleVector = Field(description="该 Agent 对四原则的权重分配（Σ≈1）")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""
    syllogism: Optional[Syllogism] = None


class ConstraintKind(str, Enum):
    HARD = "L1"  # 客观硬约束（不可谈判）
    SOFT = "L2"  # 情境软约束（可谈判）
    FLOOR = "L3"  # 原则底线（半刚性）


class Constraint(BaseModel):
    """三层约束中的单条约束。direction=le 表示 value<=bound（资源上限）；ge 表示 value>=bound（底线）。"""

    name: str
    kind: ConstraintKind
    description: str = ""
    bound: float
    direction: str = Field("le", pattern="^(le|ge)$")
    value: float = 0.0
    penalty: float = Field(default=1.0, description="拉格朗日乘子 λ 初值")

    def violation(self) -> float:
        """违反量（>0 表示违反，<=0 表示满足）。"""
        if self.direction == "le":
            return max(0.0, self.value - self.bound)
        return max(0.0, self.bound - self.value)


class FourBoxState(BaseModel):
    """Jonsen 四盒模型 → 四维状态（跨场景表示层）。子字段为 0-1 的语义分数。"""

    medical: Dict[str, float] = Field(default_factory=dict, description="医疗适应证")
    preference: Dict[str, float] = Field(default_factory=dict, description="患者意愿")
    qol: Dict[str, float] = Field(default_factory=dict, description="生活质量")
    context: Dict[str, float] = Field(default_factory=dict, description="情境特征")

    def mui(self) -> float:
        """医疗紧迫指数 MUI = severity × (1 - rescue_available)。"""
        sev = self.medical.get("severity", 0.5)
        rescue = self.medical.get("rescue_available", 0.5)
        return float(np.clip(sev * (1.0 - rescue), 0.0, 1.0))

    def vpi(self) -> float:
        """意愿有效指数 VPI = clarity × capacity × info。"""
        c = self.preference.get("clarity", 0.5)
        cap = self.preference.get("capacity", 0.5)
        info = self.preference.get("info_completeness", 0.5)
        return float(np.clip(c * cap * info, 0.0, 1.0))

    def qlti(self) -> float:
        """QoL 损益指数（正值=治疗有望改善；负值=得不偿失）。"""
        return float(np.clip(self.qol.get("net_effect", 0.0), -1.0, 1.0))

    def cci(self) -> float:
        """情境约束指数 CCI = 1 - Π(1 - 压力分量)（"或"逻辑）。"""
        c = self.context
        comps = [
            c.get("insurance_stress", 0.0),
            c.get("resource_pressure", 0.0),
            c.get("legal_constraint", 0.0),
            c.get("religious_barrier", 0.0),
        ]
        acc = 1.0
        for x in comps:
            acc *= 1.0 - np.clip(x, 0.0, 1.0)
        return float(np.clip(1.0 - acc, 0.0, 1.0))

    def to_indices(self) -> np.ndarray:
        """四维综合指标向量 [MUI, VPI, QLTI, CCI]。"""
        return np.array([self.mui(), self.vpi(), self.qlti(), self.cci()])

    def l3_specification(self) -> Dict[str, float]:
        """L3 原则具象化（Beauchamp-Childress specification 的可计算版，设计 §2.6）。

        大原则（B/N/A/J）→ 小原则（临床指标）：
        - 行善 B   = QoL 净增益（qlti，正=治疗有望改善）
        - 不伤害 N = 1 − 严重度（30 天死亡风险代理）
        - 自主 A   = 意愿有效指数 VPI（clarity×capacity×info）
        - 公正 J   = 1 − 情境约束指数 CCI（资源/经济压力越小越公正）
        论文中作为"原则具象化"层报告（与 GNE 权重底线互补：具象化=指标，底线=约束）。
        """
        sev = self.medical.get("severity", 0.5)
        return {
            "beneficence": float(np.clip(0.5 + self.qlti(), 0.0, 1.0)),
            "nonmaleficence": float(np.clip(1.0 - sev, 0.0, 1.0)),
            "autonomy": float(np.clip(self.vpi(), 0.0, 1.0)),
            "justice": float(np.clip(1.0 - self.cci(), 0.0, 1.0)),
        }


class Party(BaseModel):
    """某一利益相关方持有的信息（信息不对称的输入）。"""

    id: str
    name: str = ""
    # 该方对四维的感知值（观察掩码生效后可见的部分；维度 -> 特征分数）
    observed: Dict[str, object] = Field(default_factory=dict)
    # 该方守护的约束名
    guarded_constraints: List[str] = Field(default_factory=list)


class Reference(BaseModel):
    """第三方专家参照（按三型评估法：answer / distribution / rubric / none）。"""

    kind: str = Field("none", pattern="^(answer|distribution|rubric|none)$")
    content: Dict[str, object] = Field(default_factory=dict)


class TemporalEvent(BaseModel):
    """时序展开（压力事件 = 时序维度，可选增强）。"""

    t: str
    type: str = "onset"
    description: str = ""


class Scenario(BaseModel):
    """标准场景（统一适配层的输出，对应规划 §4.5）。"""

    scenario_id: str
    source: Dict[str, object] = Field(default_factory=dict)
    raw_text: str = ""
    state: FourBoxState = Field(default_factory=FourBoxState)
    parties: Dict[str, Party] = Field(default_factory=dict)
    constraints: List[Constraint] = Field(default_factory=list)
    reference: Reference = Field(default_factory=Reference)
    temporal: List[TemporalEvent] = Field(default_factory=list)
    # SEMA-RAG 证据锁定（F1 可计算前提）：协商前冻结的客观事实维度（如 ["medical","context"]）。
    # engine.run 在协商前设置；提示词对锁定维度标注 [锚定]（防幻觉，见 agents/base.py _state_block）。
    evidence_locked: List[str] = Field(default_factory=list)
    # 压力扰动上下文（韧性测试注入；None=常态）。
    # LLM Agent 的提案提示词据此感知压力（"何时让步/放弃原则"的刺激源），
    # 规则 Agent 的确定性响应也读取它。见 resilience/stress_engine.inject。
    stress: Optional["Perturbation"] = None


class NegotiationRound(BaseModel):
    """一轮协商记录。"""

    index: int
    proposals: List[Proposal] = Field(default_factory=list)
    collective_vector: Optional[PrincipleVector] = None
    residual_conflict: float = 0.0


class NegotiationResult(BaseModel):
    """一次 MANE 协商的完整结果。"""

    scenario_id: str
    rounds: int
    converged: bool
    arbitration_triggered: bool = False
    # 仲裁触发原因（"+" 组合）：unconverged / satisfaction / resource；未触发为 None
    arbitration_reason: Optional[str] = None
    # ---- 过程审计字段（F3 程序正当性证据）----
    state_trace: List[str] = Field(default_factory=list, description="状态机轨迹（S0-S6+R 逐状态）")
    backtracks: int = Field(default=0, description="R 回溯次数（HALF 漂移越界触发）")
    bayesian_updates: int = Field(default=0, description="贝叶斯可靠性更新次数（EmoMAS 生效计数）")
    # ---- 闭环证据链（"系统而非组合"审计）：每环真实状态 + 前一环失败→后一环修复 ----
    # 记录各机制的输出是否进入下一环：catfish 异议是否进历史并改变后续集体、
    # 仲裁是否因"未收敛/低满意/资源超限"触发并融合进终态、资源修正是否降级、
    # GNE KKT 是否达标、终态是否守住 L3 底线。论文据此论证组件耦合而非并列。
    process_trace: Dict[str, object] = Field(
        default_factory=dict,
        description="闭环证据链：catfish_rounds / dissent_in_history / subsequent_collective_delta / "
                    "arbitration_triggered / arbitration_reason / resource_corrected / fusion_applied / "
                    "kkt_residual / floors_ok")
    trajectory: List[NegotiationRound] = Field(default_factory=list)
    final_proposal: Optional[Proposal] = None
    final_vector: Optional[PrincipleVector] = None
    kkt_residual: Optional[float] = None
    agent_satisfactions: Dict[str, float] = Field(default_factory=dict)
    # 冲突识别报告（检测器输出 ERS；规则模式=规则基线，api/local=LLM 检测器）
    conflict_report: Optional["ConflictReport"] = None
    # 资源约束认证（GNE 求解器输出；精炼失败时为 None）
    resource_used: Optional[float] = None       # Σρ(t_i)
    resource_cap: Optional[float] = None        # 资源上限
    resource_feasible: Optional[bool] = None    # Σρ(t_i) ≤ cap？
    # L3 底线乘子（GNE 求解器输出；floor 约束激活的**精确证据**，E1b 底线敏感性用）
    # λ_floor > 0 ⇔ 约束 active（互补松弛 λ·g=0，g=floor−v<0 时 λ 应为 0）。
    # 默认 floor 下 35 场景 λ 全 0（约束未激活）→ floor 抬高后 λ 转正 = "底线真在工作"。
    floor_lambdas: Optional[Dict[str, float]] = Field(
        default=None, description="收敛后的底线拉格朗日乘子 {floor_0..3}（None=未启用 GNE/求解失败）")


class Perturbation(BaseModel):
    """压力/环境偏移。type ∈ {authority, safety, logical_trap, resource, economic, demographic, disease}。

    ``intensity``：压力强度标定（默认 1.0 = 满档）。各 delta 视为"满档单位扰动"，
    实际注入量 = delta × intensity（inject 时缩放）。韧性评估用强度网格
    （resilience.intensity_grid）扫档，测"临界强度"——压力多大时一致性跌破阈值、
    或某方开始放弃原则（见 ResilienceReport.stress_response / abandonment_intensity）。
    """

    type: str
    description: str = ""
    # 压力场景给 LLM Agent 的具体叙事内容（数值 delta 之外的刺激通道）。
    # 例：logical_trap 应给出"看似合理但错误的论点"原文——仅靠数值偏移模型不买账。
    # 非空时写入提示词压力块（mane/agents/base.py 的 _stress_block）。
    prompt_content: str = ""
    # 对情境特征的数值扰动（key -> delta，加到 0-1 分数上后 clip）
    context_delta: Dict[str, float] = Field(default_factory=dict)
    # 对医疗/意愿/QoL 的数值扰动
    medical_delta: Dict[str, float] = Field(default_factory=dict)
    preference_delta: Dict[str, float] = Field(default_factory=dict)
    qol_delta: Dict[str, float] = Field(default_factory=dict)
    intensity: float = Field(default=1.0, ge=0.0, description="压力强度（0-1 常规；>1 为极端压力）")


class ConflictType(str, Enum):
    """伦理冲突类型（对应《组件设计/1冲突检测器.md》的 4 类）。"""

    NONE = "none"
    MEDICAL_PREFERENCE = "medical_preference"      # 医疗-意愿（MUI高 vs VPI低）
    PREFERENCE_FAMILY = "preference_family"        # 意愿-家庭（家庭冲突/代理分歧）
    MEDICAL_RESOURCE = "medical_resource"          # 医疗-资源（最优方案超出可得资源）
    PREFERENCE_QOL = "preference_qol"              # 意愿-QoL（坚持治疗 vs 生活质量极低）


class ConflictReport(BaseModel):
    """冲突检测器输出：ERS 风险评分 + 类型/强度 + 原则向量 + 建议行动。"""

    scenario_id: str
    ers: float = Field(default=0.0, ge=0.0, le=1.0, description="伦理风险评分（MANE 门控信号）")
    conflict_type: ConflictType = ConflictType.NONE
    intensity: float = Field(default=0.0, ge=0.0, le=1.0, description="冲突强度")
    principle_vector: PrincipleVector = Field(
        default_factory=lambda: PrincipleVector(beneficence=0.25, nonmaleficence=0.25,
                                                autonomy=0.25, justice=0.25))
    action: str = "observe"
    rationale: str = ""
    channel: str = "rule"  # rule / llm


class ResilienceReport(BaseModel):
    """三层次韧性 + 放弃阈值 + 恢复验证。

    ``mode``/``baseline_only``：区分两档韧性证据——
    - rule 模式（baseline_only=True）：确定性规则对压力的响应，仅作"一致性基线"，
      不代表真实伦理行为，论文中不得据此作韧性结论；
    - api/local 模式（baseline_only=False）：LLM Agent 在压力下的真实让步，
      ``abandoned`` 与 ``per_perturbation[].agent_abandoned`` 记录"何时放弃原则"。
    """

    scenario_id: str
    l1_consistency: float
    l2_robustness: float
    l3_recoverability: float
    overall: float
    bsp: Optional[float] = None
    brs: Optional[float] = None
    abandonment: Dict[str, float] = Field(default_factory=dict, description="原则 -> 最大绝对偏移（放弃阈值近似）")
    recovery: Dict[str, float] = Field(default_factory=dict, description="settle_rounds/overshoot/steady_error")
    verdict: str = Field("ACCEPT", description="ACCEPT / RENEGOTIATE")
    mode: str = Field("rule", description="协商基座模式（rule/api/local）")
    baseline_only: bool = Field(False, description="True=规则模式：仅一致性基线，不作韧性结论")
    abandoned: Dict[str, str] = Field(default_factory=dict,
                                      description="原则 -> 首个触发放弃（跌破 L3 底线）的压力类型（跨类型取最小临界强度者）")
    abandonment_intensity: Dict[str, float] = Field(
        default_factory=dict,
        description="原则 -> 首个触发放弃的最小压力强度（跨压力类型；'何时放弃原则'的临界强度 τ）")
    recovery_std: Optional[float] = Field(
        default=None,
        description="R_recover 多次重跑的标准差（LLM 随机性波动；resilience.repeat>1 时报告，P1-1 口径）")
    per_perturbation: List[Dict[str, object]] = Field(
        default_factory=list,
        description="逐压力详情（满档强度切片）：type/description/consistency/v_shift/collective_abandoned/"
                    "agent_abandoned/negotiation_quality")
    stress_response: List[Dict[str, object]] = Field(
        default_factory=list,
        description="压力-强度响应曲线：每压力类型 × intensity_grid，含 consistency 曲线、放弃计数、"
                    "critical_consistency（C<0.85 的首个强度）与 critical_abandonment（首次放弃的强度）")
    # 闭环环 9：压力下的协商质量（个体让步 vs 集体稳定）——"系统而非组合"的过程证据。
    # 基线（无压力）与各压力档的让步均值/不对称性、集体波动：压力让个体让步合理、
    # 而集体仍稳定（波动小）= 健康协商；让步失衡/集体被带偏 = 协商质量受损。
    negotiation_quality: Dict[str, object] = Field(
        default_factory=dict,
        description="压力下协商质量汇总：baseline/stress 两档的 concession_mean/max/inequality、"
                    "collective_volatility（见 resilience/negotiation_quality）")
