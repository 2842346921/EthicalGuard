"""LLMEval-Med 适配器（EMNLP 2025 Findings）。

dataset.json 顶层为 5 大类字典（共 667 条）。伦理相关为「医疗安全伦理」类（109 条：
医学伦理/药品安全/医疗违规/设施安全），带 sanswer（参考回答）与 checklist（评分要点）。
"""
from __future__ import annotations

from typing import Iterator, List

from ..types import Reference, Scenario
from .base import DatasetAdapter, register


@register("llmevalmed")
class LLMEvalMedAdapter(DatasetAdapter):
    name = "llmevalmed"
    folder = "LLMEval-Med"

    def _ethics_items(self) -> List[dict]:
        data = self._load_json("dataset", "dataset.json")
        if not isinstance(data, dict):
            return []
        for key, value in data.items():
            if isinstance(value, list) and ("伦理" in str(key) or "安全" in str(key)):
                return value
        return []

    def load_scenarios(self) -> Iterator[Scenario]:
        for it in self._ethics_items():
            ref = Reference(kind="answer", content={
                "sanswer": it.get("sanswer"),
                "checklist": it.get("checklist"),
                "category2": it.get("category2"),
            })
            yield self._build_scenario(
                scenario_id=f"LLMEvalMed-{it.get('groupCode')}-{it.get('round')}",
                raw_text=f"[{it.get('category1','')}/{it.get('category2','')}/{it.get('scene','')}]\n{it.get('problem','')}",
                source={"dataset": "llmevalmed", "kind": "medical_safety_ethics", "groupCode": it.get("groupCode")},
                reference=ref,
            )

    def load_knowledge(self) -> List[Scenario]:
        return []
