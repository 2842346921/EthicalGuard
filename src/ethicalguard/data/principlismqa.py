"""PrinciplismQA 适配器（ACL 2026 Findings）。

- knowledge-mcqa.json: 2182 道 MCQ（correct_answer + principlism 四原则标注）→ 知识检查
- open-ended-qa.json + open-ended-rubric-principles.json: 677 病例 / 1466 开放题（专家 rubric keypoints）→ 主测试床
"""
from __future__ import annotations

from typing import Iterator, List

from ..types import Reference, Scenario
from .base import DatasetAdapter, register


@register("principlismqa")
class PrinciplismQAAdapter(DatasetAdapter):
    name = "principlismqa"
    folder = "PrinciplismQA"

    def load_scenarios(self) -> Iterator[Scenario]:
        cases = self._load_json("data", "open-ended-qa.json")
        rubrics = self._load_json("data", "open-ended-rubric-principles.json")
        rubric_by_qid = {int(r["qid"]): r for r in rubrics} if isinstance(rubrics, list) else {}

        for case in cases:
            case_id = case.get("id")
            title = case.get("title", "")
            case_text = case.get("case_rewrite") or case.get("case") or ""
            tags = case.get("tags", [])
            for issue in case.get("ethical_issues", []):
                qid = issue.get("qid")
                question = issue.get("question", "")
                keypoints = issue.get("keypoints", [])
                rubric = rubric_by_qid.get(int(qid), {})
                principles = rubric.get("principles", []) if isinstance(rubric, dict) else []
                ref = Reference(
                    kind="rubric",
                    content={"keypoints": keypoints, "principles": principles, "qid": qid},
                )
                # 问题前置、case 截断：让映射对"伦理议题"（而非整段病例）敏感，避免同 case 多题状态坍缩
                yield self._build_scenario(
                    scenario_id=f"PQ-{case_id}-{qid}",
                    raw_text=f"{title}\n【伦理议题】{question}\n【病例摘要】{case_text[:800]}",
                    source={"dataset": "principlismqa", "case_id": case_id, "qid": qid, "tags": tags},
                    reference=ref,
                )

    def load_knowledge(self) -> List[Scenario]:
        items = self._load_json("data", "knowledge-mcqa.json")
        out = []
        for it in items:
            qid = it.get("question_id") or it.get("id")
            options = it.get("options", {})
            principlism = it.get("principlism", {})
            ref = Reference(
                kind="answer",
                content={"correct": it.get("correct_answer"), "options": options, "principlism": principlism},
            )
            out.append(self._build_scenario(
                scenario_id=f"PQ-MCQ-{qid}",
                raw_text=f"{it.get('question','')}\n{options}",
                source={"dataset": "principlismqa", "kind": "knowledge_mcqa", "id": qid},
                reference=ref,
            ))
        return out
