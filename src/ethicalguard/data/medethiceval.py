"""MedEthicEval 适配器（NAACL 2025 Industry，中文）。

- medical_ethics_knowledge.csv（629，有 answer）→ 知识检查
- medical_ethics_priority_dilemma.csv（100，无答案）→ 临床优先级困境（评审/协商）
- medical_ethics_equilibrium_dilemma.csv（100，无答案）→ 临床平衡困境（天然协商场景）
- medical_ethics_detecting_violation.csv（236，无答案）→ 违规检测；**仅保留临床主题**，
  剔除 theme_tag1=="医学科研"（动物实验/人体试验 50 题——非临床医疗决策，MANE 五方
  临床角色无法适用；用户决策：只做医疗决策）。产出顺序：equilibrium（协商主战场）→
  priority → violation（临床），保证 limit 均分取样先落到临床 dilemma。
"""
from __future__ import annotations

import csv
from typing import Iterator, List

from ..types import Reference, Scenario
from .base import DatasetAdapter, register


@register("medethiceval")
class MedEthicEvalAdapter(DatasetAdapter):
    name = "medethiceval"
    folder = "MedEthicEval"

    def _read_csv(self, fname: str) -> List[dict]:
        with open(self._path("dataset", fname), "r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def load_scenarios(self) -> Iterator[Scenario]:
        # ① 平衡困境（协商质量主战场，全临床）
        for i, row in enumerate(self._read_csv("medical_ethics_equilibrium_dilemma.csv")):
            yield self._build_scenario(
                scenario_id=f"MEE-equilibrium-{i}",
                raw_text=f"{row.get('case','')}\n{row.get('query','')}",
                source={"dataset": "medethiceval", "kind": "equilibrium_dilemma", "idx": i},
                reference=Reference(kind="none"),
            )
        # ② 优先级困境（临床，全保留）
        for i, row in enumerate(self._read_csv("medical_ethics_priority_dilemma.csv")):
            yield self._build_scenario(
                scenario_id=f"MEE-priority-{i}",
                raw_text=f"{row.get('case','')}\n{row.get('query','')}",
                source={"dataset": "medethiceval", "kind": "priority_dilemma", "idx": i},
                reference=Reference(kind="none"),
            )
        # ③ 违规检测：只保留临床主题，剔除"医学科研"（动物实验/人体试验——非临床医疗决策）
        for row in self._read_csv("medical_ethics_detecting_violation.csv"):
            if row.get("theme_tag1", "").strip() == "医学科研":
                continue
            yield self._build_scenario(
                scenario_id=f"MEE-violation-{row.get('uuid','')[:8]}",
                raw_text=f"{row.get('scenario','')}\n{row.get('query','')}",
                source={"dataset": "medethiceval", "kind": "detecting_violation",
                        "uuid": row.get("uuid"), "theme": row.get("theme_tag1")},
                reference=Reference(kind="none"),
            )

    def load_knowledge(self) -> List[Scenario]:
        out = []
        for row in self._read_csv("medical_ethics_knowledge.csv"):
            ref = Reference(kind="answer", content={"correct": row.get("answer"), "options": row.get("options")})
            out.append(self._build_scenario(
                scenario_id=f"MEE-knowledge-{row.get('uuid','')[:8]}",
                raw_text=f"{row.get('question','')}\n{row.get('options','')}",
                source={"dataset": "medethiceval", "kind": "knowledge", "uuid": row.get("uuid")},
                reference=ref,
            ))
        return out
