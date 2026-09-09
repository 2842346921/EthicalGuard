"""文本 → 四盒状态映射：规则通道 + LLM 通道 + 双通道交叉验证。

- 规则通道：``mapping_rules.map_text_to_state``（确定性关键词，高置信可复现）。
- LLM 通道：``mapping_llm.LLMFourBoxMapper``（LLM 结构化打分卡，覆盖规则盲区）。
- 双通道：``DualChannelMapper`` 逐字段比对 → 分歧率 → 仲裁（规划 §4.6）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..llm.base import LLMBackend
from ..types import Constraint, ConstraintKind, FourBoxState, Party
from .mapping_llm import FOUR_BOX_SCHEMA, LLMFourBoxMapper, llm_mapper_factory
from .mapping_rules import default_constraints, default_parties, map_text_to_state

__all__ = [
    "map_text_to_state",
    "default_parties",
    "default_constraints",
    "MappingResult",
    "DualChannelMapper",
    "MappingQualityReport",
    "report_mapping",
    "Constraint",
    "ConstraintKind",
    "FourBoxState",
    "Party",
]

# 分歧判定阈值：|rule − llm| > DISAGREE_THRESHOLD 记为"分歧字段"
DISAGREE_THRESHOLD = 0.25


@dataclass
class MappingResult:
    state: FourBoxState
    channel: str                       # "rule" | "llm" | "dual"
    confidence: float = 0.5
    disagreement_rate: float = 0.0     # 整体分歧率（0-1）
    per_dim_disagreement: Dict[str, float] = field(default_factory=dict)
    flagged_fields: List[str] = field(default_factory=list)


def _field_values(state: FourBoxState) -> Dict[str, float]:
    """把四盒状态展开为 {dim.field: value}。"""
    out: Dict[str, float] = {}
    for dim, fields in FOUR_BOX_SCHEMA.items():
        sub = getattr(state, dim, {})
        for k in fields:
            out[f"{dim}.{k}"] = float(sub.get(k, 0.0))
    return out


def _merge_field(v_rule: float, v_llm: float) -> float:
    """仲裁：分歧 ≤ 阈值 → 取均值；分歧 > 阈值 → 取 LLM 值（语义通道优先）。

    规则通道的 0 多为"关键词未命中"而非"确信为 0"，不能当可靠锚点；
    分歧大说明两通道语义理解不同，LLM 语义打分更可信。可复现性由
    rule 单通道保证，dual 模式的定位是语义优先（A1/O1 修复：旧策略分歧取规则值，
    实测平均分歧率 0.741 时会把 LLM 语义信息全部丢弃、退化为规则通道）。

    B3 无信息过滤：LLM 通道对"文本无法判断"的字段按 prompt 规则给 0.5 中性默认，
    若此时规则通道也无命中（≈0），说明**双方都无证据**——0.5 只是占位而非语义判断，
    必须归零，否则中性默认值会被下游（ERS 门控/资源 cap 收紧）当成假信号。
    """
    if abs(v_rule - v_llm) <= DISAGREE_THRESHOLD:
        return 0.5 * (v_rule + v_llm)
    if abs(v_llm - 0.5) < 0.05 and abs(v_rule) < 0.15:
        return v_rule  # 无证据不假设：取规则（≈0），不采信中性的 0.5
    return v_llm


class DualChannelMapper:
    """规则 + LLM 双通道映射，逐字段比对并仲裁。

    - 无 LLM 后端（rule 模式）→ 只走规则通道，分歧率为 0。
    - 有 LLM 后端 → 双通道，报告分歧率（适配层质量指标）。
    """

    def __init__(self, backend: Optional[LLMBackend] = None):
        self.rule = map_text_to_state
        self.llm = llm_mapper_factory(backend)  # rule 模式 → None

    def __call__(self, text: str) -> MappingResult:
        return self.map(text)

    def map(self, text: str) -> MappingResult:
        state_rule = self.rule(text)
        if self.llm is None:
            return MappingResult(state=state_rule, channel="rule", confidence=0.5,
                                 disagreement_rate=0.0, per_dim_disagreement={}, flagged_fields=[])

        state_llm, conf = self.llm.map(text)
        rule_flat = _field_values(state_rule)
        llm_flat = _field_values(state_llm)

        per_dim: Dict[str, float] = {}
        merged_flat: Dict[str, float] = {}
        flagged: List[str] = []
        for key in rule_flat:
            dim = key.split(".")[0]
            d = abs(rule_flat[key] - llm_flat[key])
            per_dim[dim] = max(per_dim.get(dim, 0.0), d)
            merged_flat[key] = _merge_field(rule_flat[key], llm_flat[key])
            if d > DISAGREE_THRESHOLD:
                flagged.append(key)

        overall = len(flagged) / max(1, len(rule_flat))
        state = FourBoxState(
            medical={k.replace("medical.", ""): merged_flat[k] for k in rule_flat if k.startswith("medical.")},
            preference={k.replace("preference.", ""): merged_flat[k] for k in rule_flat if k.startswith("preference.")},
            qol={k.replace("qol.", ""): merged_flat[k] for k in rule_flat if k.startswith("qol.")},
            context={k.replace("context.", ""): merged_flat[k] for k in rule_flat if k.startswith("context.")},
        )
        return MappingResult(state=state, channel="dual", confidence=conf,
                             disagreement_rate=overall, per_dim_disagreement=per_dim, flagged_fields=flagged)


@dataclass
class MappingQualityReport:
    n_texts: int
    channel: str
    mean_disagreement: float
    max_disagreement: float
    per_dim_disagreement: Dict[str, float] = field(default_factory=dict)
    flagged_examples: List[Dict[str, object]] = field(default_factory=list)


def report_mapping(mapper: DualChannelMapper, texts: List[str], max_examples: int = 5) -> MappingQualityReport:
    """对整个语料做双通道质量报告（分歧率聚合）。"""
    results = [mapper.map(t) for t in texts]
    dims: Dict[str, float] = {}
    flagged: List[Dict[str, object]] = []
    for i, r in enumerate(results):
        for dim, d in r.per_dim_disagreement.items():
            dims[dim] = max(dims.get(dim, 0.0), d)
        for f in r.flagged_fields[:3]:
            if len(flagged) < max_examples:
                flagged.append({"idx": i, "field": f, "disagreement": r.per_dim_disagreement.get(f.split('.')[0], 0.0)})
    rates = [r.disagreement_rate for r in results]
    channel = results[0].channel if results else "rule"
    return MappingQualityReport(
        n_texts=len(results), channel=channel,
        mean_disagreement=float(np.mean(rates)) if rates else 0.0,
        max_disagreement=float(np.max(rates)) if rates else 0.0,
        per_dim_disagreement=dims, flagged_examples=flagged,
    )
