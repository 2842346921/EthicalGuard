"""VITAL 适配器（ACL 2025，多元化对齐）。

- vital_distributional_moralchoice.json（临床道德困境二选一，gold_distribution）→ 协商场景
- vital_distributional_globalopinionqa.json（社会舆论调查题）→ **不产出协商场景**
  （"things that may be problems in our country" 这类题无临床决策情境，四盒映射零状态、
  协商空转——实测 ERS=0.10/resource=0/5 轮；仅适合分布型评估靶子）
- vital_overton_valuekaleidoscope.json（开放评论，vrd 价值标注，含电车难题等情境）→ 协商场景
- vital_steerable_valuekaleidoscope.json（label A/B/C）→ 知识检查
"""
from __future__ import annotations

from typing import Iterator, List

from ..types import Reference, Scenario
from .base import DatasetAdapter, register


@register("vital")
class VITALAdapter(DatasetAdapter):
    name = "vital"
    folder = "VITAL"

    def load_scenarios(self) -> Iterator[Scenario]:
        # 临床道德困境（有真实医疗情境，可协商；gold_distribution 供分布型评估）
        items = self._load_json("dataset", "vital_distributional_moralchoice.json")
        for it in items:
            ref = Reference(kind="distribution", content={
                "gold_distribution": it.get("gold_distribution"),
                "options": it.get("options"),
                "attribute": it.get("attribute"),
            })
            yield self._build_scenario(
                scenario_id=f"VITAL-mc-{it.get('id')}",
                raw_text=f"{it.get('question','')}\n{it.get('options',[])}",
                source={"dataset": "vital", "kind": "moralchoice", "id": it.get("id")},
                reference=ref,
            )
        # 开放评论型（vrd 价值标注，无标准答案；有具体情境文本）
        items = self._load_json("dataset", "vital_overton_valuekaleidoscope.json")
        for it in items:
            ref = Reference(kind="none", content={"vrd": it.get("vrd", [])})
            yield self._build_scenario(
                scenario_id=f"VITAL-overton-{it.get('id')}",
                raw_text=f"{it.get('situation','')}\n{it.get('input','')}",
                source={"dataset": "vital", "kind": "overton_valuekaleidoscope", "id": it.get("id")},
                reference=ref,
            )

    def load_knowledge(self) -> List[Scenario]:
        items = self._load_json("dataset", "vital_steerable_valuekaleidoscope.json")
        out = []
        for it in items:
            ref = Reference(kind="answer", content={"correct": it.get("label"), "label_text": it.get("label_text")})
            out.append(self._build_scenario(
                scenario_id=f"VITAL-vk-{it.get('id')}",
                raw_text=f"{it.get('situation','')}\n{it.get('input','')}",
                source={"dataset": "vital", "kind": "steerable_valuekaleidoscope", "id": it.get("id")},
                reference=ref,
            ))
        return out
