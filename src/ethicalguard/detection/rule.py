"""规则冲突检测器（基线）：由四盒综合指标（MUI/VPI/QLTI/CCI）的伦理张力启发式计算。

定位：零成本、可复现的基线。论文中标注为 Rule Baseline；主识别交给 LLM 检测器。

**文本原则识别（A 轨 MCQ 用）**：原则向量除状态推导外，叠加题目文本的关键词信号
（_PRINCIPLE_KEYWORDS，中英双语），使规则通道能回答"该题涉及哪些伦理原则"——
规则模式在 MCQ 上不再输出恒定值（旧版只由 severity 推导，MCQ 无临床信号 → 恒定 0.355）。
词典可随 A 轨评估结果反向优化（错误分析 → 补词/调权 → 重测）。
"""
from __future__ import annotations

import numpy as np

from ..types import ConflictReport, ConflictType, PrincipleVector, Scenario
from .base import ACTION_LABELS, ConflictDetector

# 四原则关键词（中英双语）——A 轨 MCQ 文本原则识别 + 反向优化对象。
# 结构：{原则: [(关键词, 权重), ...]}；命中累加权重，权重可调（评估后反向优化）。
_PRINCIPLE_KEYWORDS = {
    "beneficence": [
        ("beneficence", 2.0), ("benefit", 1.5), ("beneficial", 1.5), ("help", 1.0),
        ("treatment", 1.0), ("treat", 1.0), ("intervention", 1.0), ("vaccinat", 1.5),
        ("cure", 1.2), ("save", 1.5), ("life-saving", 2.0), ("survival", 1.5),
        ("行善", 2.0), ("获益", 1.5), ("治疗", 1.0), ("救治", 1.5), ("救命", 2.0),
        ("疫苗", 1.5), ("治愈", 1.2), ("挽救", 1.5), ("compulsory", 1.0),
    ],
    "nonmaleficence": [
        ("nonmaleficence", 2.0), ("non-maleficence", 2.0), ("harm", 2.0), ("harmful", 2.0),
        ("risk", 1.5), ("danger", 1.5), ("side effect", 1.5), ("injur", 1.5),
        ("damage", 1.2), ("toxic", 1.5), ("safety", 1.0), ("avoid", 1.0),
        ("不伤害", 2.0), ("伤害", 2.0), ("风险", 1.5), ("危险", 1.5), ("副作用", 1.5),
        ("损伤", 1.5), ("危害", 1.5), ("致死", 1.5), ("毒", 1.2),
    ],
    "autonomy": [
        ("autonomy", 2.0), ("autonomous", 2.0), ("patient consent", 2.0), ("consent", 1.5),
        ("informed consent", 2.0), ("self-determination", 2.0), ("voluntary", 1.5),
        ("refus", 1.5), ("choice", 1.0), ("freedom", 1.2), ("right to", 1.2),
        ("privacy", 1.5), ("confidential", 1.5), ("respect for persons", 2.0),
        ("patient's wishes", 1.5), ("autonomy", 2.0), ("religio", 1.0),
        ("自主", 2.0), ("自愿", 1.5), ("同意", 1.5), ("知情同意", 2.0), ("自决", 2.0),
        ("拒绝", 1.5), ("自由", 1.2), ("隐私", 1.5), ("保密", 1.5), ("意愿", 1.2),
        ("权利", 1.0), ("宗教", 1.0),
    ],
    "justice": [
        ("justice", 2.0), ("fair", 2.0), ("fairness", 2.0), ("equit", 1.8),
        ("equality", 1.5), ("equal", 1.5), ("allocation", 2.0), ("distribut", 1.8),
        ("resource", 1.8), ("scarce", 1.8), ("ration", 1.8), ("access", 1.2),
        ("discriminat", 2.0), ("bias", 1.5), ("public health", 1.2), ("utilitarian", 1.5),
        ("公正", 2.0), ("公平", 2.0), ("平等", 1.5), ("分配", 1.8), ("资源", 1.8),
        ("稀缺", 1.8), ("歧视", 2.0), ("公共卫生", 1.2), ("功利", 1.5), ("强制", 1.0),
    ],
}


def _text_principle_signal(text: str) -> np.ndarray:
    """从题目文本关键词累加四原则信号（A 轨 MCQ 原则识别）。

    返回未归一化 [B,N,A,J] 权重。词典命中即累加——可随评估反向优化（补词/调权）。
    """
    t = text.lower()
    sig = np.zeros(4)
    for i, name in enumerate(("beneficence", "nonmaleficence", "autonomy", "justice")):
        total = 0.0
        for kw, w in _PRINCIPLE_KEYWORDS[name]:
            if kw in t:
                total += w
        sig[i] = total
    return sig


class RuleConflictDetector(ConflictDetector):
    channel = "rule"

    def detect(self, scenario: Scenario) -> ConflictReport:
        st = scenario.state
        mui = st.mui()
        vpi = st.vpi()
        qlti = st.qlti()
        cci = st.cci()
        c = st.context

        # 四类冲突信号（0-1）
        signals = {
            ConflictType.MEDICAL_PREFERENCE: float(np.clip(mui * (1.0 - vpi) * 2.0, 0.0, 1.0)),
            ConflictType.PREFERENCE_FAMILY: float(np.clip(
                (c.get("family_conflict", 0.0) + 0.5 * (1.0 - vpi)) / 1.5, 0.0, 1.0)),
            ConflictType.MEDICAL_RESOURCE: float(np.clip(
                mui * c.get("resource_pressure", 0.0) * 3.0, 0.0, 1.0)),
            ConflictType.PREFERENCE_QOL: float(np.clip(
                vpi * max(0.0, -qlti) * 3.0, 0.0, 1.0)),
        }
        best_type = max(signals, key=signals.get)
        intensity = signals[best_type]
        ers = float(np.clip(max(signals.values()) + 0.2 * cci, 0.0, 1.0))

        # 建议行动按强度分级
        if ers < 0.3:
            action = "observe"
        elif ers < 0.5:
            action = "review_24h"
        elif ers < 0.7:
            action = "family_communication" if best_type == ConflictType.PREFERENCE_FAMILY else "ethics_consult"
        else:
            action = "intervene"

        # 原则向量：状态推导 + 文本关键词信号融合（A 轨 MCQ 原则识别）
        sev = st.medical.get("severity", 0.5)
        state_pv = np.array([
            float(np.clip(0.4 + 0.5 * sev, 0.0, 1.0)),
            float(np.clip(1.0 - 0.5 * sev, 0.0, 1.0)),
            float(np.clip(0.3 + 0.7 * vpi, 0.0, 1.0)),
            float(np.clip(0.5 + 0.5 * (1.0 - cci), 0.0, 1.0)),
        ])
        text_sig = _text_principle_signal(scenario.raw_text or "")
        # 融合：文本信号为主（原则识别任务），状态信号为底（临床场景保底）
        # 文本信号无命中（纯状态场景）→ 退回状态推导
        if text_sig.sum() > 0:
            # 文本信号归一化后与状态信号 0.7/0.3 加权（文本为主）
            pv_arr = 0.7 * (text_sig / text_sig.sum()) + 0.3 * (state_pv / state_pv.sum())
        else:
            pv_arr = state_pv
        pv = PrincipleVector.from_array(pv_arr).normalized()

        return ConflictReport(
            scenario_id=scenario.scenario_id,
            ers=ers,
            conflict_type=best_type if ers >= 0.3 else ConflictType.NONE,
            intensity=intensity,
            principle_vector=pv,
            action=action,
            rationale=f"规则基线：MUI={mui:.2f} VPI={vpi:.2f} QLTI={qlti:.2f} CCI={cci:.2f}，"
                      f"最强信号 {best_type.value}={intensity:.2f}",
            channel=self.channel,
        )
