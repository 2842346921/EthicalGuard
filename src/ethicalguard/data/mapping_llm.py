"""文本 → 四盒状态映射（LLM 通道）。

用 LLM 抽取 Jonsen 四盒字段为结构化打分卡（JSON），覆盖规则通道的盲区。
- 与规则通道（mapping_rules.map_text_to_state）构成"双通道交叉验证"（规划 §4.6）。
- 输出：FourBoxState + 置信度；解析失败/无 LLM 时回退到规则通道。
"""
from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from ..llm.base import LLMBackend
from ..types import FourBoxState
from .mapping_rules import map_text_to_state

# 四盒字段 schema：维度 -> 字段 -> 说明（LLM 提示词用）
FOUR_BOX_SCHEMA = {
    "medical": {
        "severity": "病情严重度（0-1）",
        "rescue_available": "抢救资源可得性（0-1）",
        "acuity": "疾病急迫性（0-1）",
    },
    "preference": {
        "dnr": "是否有 DNR/DNI（0 或 1）",
        "attitude_refuse": "患者拒绝治疗倾向（0-1）",
        "clarity": "患者意愿清晰度（0-1）",
        "capacity": "患者决策能力（0-1）",
        "info_completeness": "信息完整度（0-1）",
    },
    "qol": {
        "burden": "痛苦/残疾负担（0-1）",
        "net_effect": "QoL 净损益（-1 到 1，正=改善）",
    },
    "context": {
        "resource_pressure": "资源压力（0-1）",
        "insurance_stress": "经济/医保压力（0-1）",
        "family_conflict": "家庭冲突（0-1）",
        "religious_barrier": "宗教文化障碍（0-1）",
        "legal_constraint": "法律约束（0-1）",
    },
}

_PROMPT = """你是临床伦理结构化抽取器。请从给定病例文本中抽取 Jonsen 四盒模型字段，只输出 JSON。

输出格式：
{
  "medical": {"severity": 0-1, "rescue_available": 0-1, "acuity": 0-1},
  "preference": {"dnr": 0/1, "attitude_refuse": 0-1, "clarity": 0-1, "capacity": 0-1, "info_completeness": 0-1},
  "qol": {"burden": 0-1, "net_effect": -1 到 1},
  "context": {"resource_pressure": 0-1, "insurance_stress": 0-1, "family_conflict": 0-1, "religious_barrier": 0-1, "legal_constraint": 0-1},
  "confidence": 0-1
}
规则：优先基于文本证据给出非中性估计（如"部分/有倾向"用 0.3/0.6/0.8 等）；文本明确否定给 0；
明确肯定给 1；确实完全无法推断的字段才给 0.5（中性）。所有字段必须出现在输出 JSON 中。仅输出 JSON。"""


def _extract_json(text: str) -> dict:
    # Qwen3 默认开 thinking：剥离 <think> 块并清理控制字符，否则严格模式 json.loads 会拒绝
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"LLM 输出未包含 JSON: {text[:200]}")
    payload = re.sub(r"[\x00-\x1f\x7f]", " ", m.group(0))
    return json.loads(payload)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return float(max(lo, min(hi, float(v))))
    except (TypeError, ValueError):
        return 0.5


class LLMFourBoxMapper:
    """LLM 通道：文本 → 四盒状态 + 置信度。"""

    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def map(self, text: str) -> Tuple[FourBoxState, float]:
        """返回 (四盒状态, 置信度)。解析失败时回退规则通道并置低置信度。"""
        try:
            data = _extract_json(self.backend.complete(_PROMPT, f"病例：{text[:3000]}"))
            state = FourBoxState(
                medical={k: _clamp(data.get("medical", {}).get(k, 0.5)) for k in FOUR_BOX_SCHEMA["medical"]},
                preference={k: _clamp(data.get("preference", {}).get(k, 0.5)) for k in FOUR_BOX_SCHEMA["preference"]},
                qol={k: _clamp(data.get("qol", {}).get(k, 0.5), -1.0, 1.0) for k in FOUR_BOX_SCHEMA["qol"]},
                context={k: _clamp(data.get("context", {}).get(k, 0.5)) for k in FOUR_BOX_SCHEMA["context"]},
            )
            conf = _clamp(data.get("confidence", 0.5))
            return state, conf
        except Exception:
            return map_text_to_state(text), 0.2  # 回退：规则通道 + 低置信度


def llm_mapper_factory(backend: Optional[LLMBackend]):
    """工厂：无后端或 rule 模式时返回 None（此时双通道只走规则）。"""
    if backend is None or backend.mode == "rule":
        return None
    return LLMFourBoxMapper(backend)
