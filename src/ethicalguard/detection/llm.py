"""LLM 冲突检测器（主识别）：语义理解四盒状态与文本，输出 ERS/类型/强度/原则向量。

对应 Aegle（LLM 结构化抽取）与 ConflictScope（LLM 价值评估）的范式；
与规则检测器构成双轨，LLM 失败时回退规则。
"""
from __future__ import annotations

import json
import re

from ..llm.base import LLMBackend
from ..types import ConflictReport, ConflictType, PrincipleVector, Scenario
from .base import ACTION_LABELS, ConflictDetector
from .rule import RuleConflictDetector

_PROMPT_TMPL = """你是临床伦理冲突识别专家。给定病例的四盒状态（医疗适应证/患者意愿/生活质量/情境特征）
与病例原文，判断是否存在需要关注的伦理冲突。只输出 JSON：
{
  "ers": 0-1,                     // 伦理风险评分（连续，预警式）
  "conflict_type": "medical_preference|preference_family|medical_resource|preference_qol|none",
  "intensity": 0-1,               // 冲突强度
  "principle_vector": {"beneficence":0-1,"nonmaleficence":0-1,"autonomy":0-1,"justice":0-1},
  "action": "observe|review_24h|family_communication|ethics_consult|intervene",
  "rationale": "一句话理由"
}
规则：ers<{threshold} 时 conflict_type=none、action=observe；有真实张力（如拒绝治疗但医疗紧迫）应给高分。仅输出 JSON。"""


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"LLM 输出未包含 JSON: {text[:200]}")
    return json.loads(m.group(0))


def _clamp(v, lo=0.0, hi=1.0) -> float:
    try:
        return float(max(lo, min(hi, float(v))))
    except (TypeError, ValueError):
        return 0.5


class LLMConflictDetector(ConflictDetector):
    channel = "llm"

    def __init__(self, backend: LLMBackend, fallback: ConflictDetector | None = None,
                 threshold: float = 0.5):
        self.backend = backend
        self.fallback = fallback or RuleConflictDetector()
        self.threshold = float(threshold)  # ERS 门控阈值（config.detection.threshold，O4）

    def detect(self, scenario: Scenario) -> ConflictReport:
        try:
            st = scenario.state
            user = (
                f"【四盒状态】\n医疗:{st.medical}\n意愿:{st.preference}\n"
                f"QoL:{st.qol}\n情境:{st.context}\n"
                f"【病例原文】{scenario.raw_text[:3000]}"
            )
            # 注意：必须用 replace 而非 str.format —— prompt 内含 JSON 字面量花括号，
            # format() 会把 {"beneficence":...} 当字段名抛 KeyError（B1 回归，曾致 10/10 回退规则）
            data = _extract_json(self.backend.complete(
                _PROMPT_TMPL.replace("{threshold}", f"{self.threshold:g}"), user))
            pv = data.get("principle_vector", {})
            ctype = str(data.get("conflict_type", "none"))
            if ctype not in {t.value for t in ConflictType}:
                ctype = "none"
            action = str(data.get("action", "observe"))
            if action not in ACTION_LABELS:
                action = "observe"
            return ConflictReport(
                scenario_id=scenario.scenario_id,
                ers=_clamp(data.get("ers", 0.0)),
                conflict_type=ConflictType(ctype),
                intensity=_clamp(data.get("intensity", 0.0)),
                principle_vector=PrincipleVector(
                    beneficence=_clamp(pv.get("beneficence", 0.25)),
                    nonmaleficence=_clamp(pv.get("nonmaleficence", 0.25)),
                    autonomy=_clamp(pv.get("autonomy", 0.25)),
                    justice=_clamp(pv.get("justice", 0.25)),
                ).normalized(),
                action=action,
                rationale=str(data.get("rationale", "")),
                channel=self.channel,
            )
        except Exception:
            # LLM 失败 → 回退规则基线（保持链路可用）
            report = self.fallback.detect(scenario)
            report.rationale = f"[LLM解析失败回退规则] {report.rationale}"
            report.channel = "rule"
            return report
