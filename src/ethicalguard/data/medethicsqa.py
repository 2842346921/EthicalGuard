"""MedEthicsQA 适配器（arXiv 2506.22808）。

- MedEthicsQA_MCQ.json（5623，correct）→ 知识检查
- MedEthicsQA_open.zip → MedEthicsQA_open.json（5351，answer+reference 专家参考）→ 评审型场景（LLM-as-judge 对比参考）

健壮性（B 修复）：优先读**已解压的 MedEthicsQA_open.json**（服务器上 zip 传输损坏时，
解压一次放同目录即可绕开）；zip 损坏时给出明确诊断（目录内容），不再裸抛 BadZipFile。
"""
from __future__ import annotations

import json
import os
import zipfile
from typing import Iterator, List

from ..types import Reference, Scenario
from ..utils import setup_logging
from .base import DatasetAdapter, register

logger = setup_logging()


@register("medethicsqa")
class MedEthicsQAAdapter(DatasetAdapter):
    name = "medethicsqa"
    folder = "MedEthicsQA"

    def _open_json(self) -> List[dict]:
        """读取开放题数据：优先已解压 json，其次 zip；任一损坏时回退/给出可操作诊断。

        - json 存在但解析失败（截断/损坏，如服务器上 unzip 中断的产物）→ **回退 zip**，
          不让坏 json 屏蔽好 zip（本次 JSONDecodeError 根因修复）；
        - zip 也损坏 → 明确诊断（目录内容 + 修复建议），不裸抛。
        """
        jpath = self._path("MedEthicsQA_open.json")
        if os.path.exists(jpath):
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:  # noqa: BLE001
                logger.warning("MedEthicsQA_open.json 解析失败（%s: %s），回退 zip：%s",
                               type(e).__name__, str(e)[:80], jpath)
        zpath = self._path("MedEthicsQA_open.zip")
        d = os.path.dirname(zpath)
        if not os.path.exists(zpath):
            raise FileNotFoundError(
                f"未找到 MedEthicsQA 开放题数据：{jpath} 或 {zpath}\n"
                f"目录 {d} 实际内容：{os.listdir(d) if os.path.isdir(d) else '（目录不存在）'}")
        try:
            with zipfile.ZipFile(zpath) as z:
                name = z.namelist()[0]
                with z.open(name) as f:
                    return json.loads(f.read().decode("utf-8"))
        except zipfile.BadZipFile:
            raise RuntimeError(
                f"MedEthicsQA_open.zip 损坏或不是有效 zip：{zpath}（{os.path.getsize(zpath)} 字节）\n"
                f"目录 {d} 实际内容：{os.listdir(d)}\n"
                f"修复：重新上传该文件，或解压后把 MedEthicsQA_open.json 放到同目录（适配器优先读 json）。")

    def load_scenarios(self) -> Iterator[Scenario]:
        items = self._open_json()
        for it in items:
            ref = Reference(kind="answer", content={
                "reference": it.get("reference"),
                "answer": it.get("answer"),
                "meta_data": it.get("meta_data"),
            })
            yield self._build_scenario(
                scenario_id=f"MEQA-open-{it.get('id')}",
                raw_text=f"{it.get('context','')}\n{it.get('question','')}",
                source={"dataset": "medethicsqa", "kind": "open_ended", "id": it.get("id")},
                reference=ref,
            )

    def load_knowledge(self) -> List[Scenario]:
        items = self._load_json("MedEthicsQA_MCQ.json")
        out = []
        for it in items:
            ref = Reference(kind="answer", content={"correct": it.get("correct"), "options": it.get("options")})
            out.append(self._build_scenario(
                scenario_id=f"MEQA-MCQ-{it.get('id')}",
                raw_text=f"{it.get('question','')}\n{it.get('options',{})}",
                source={"dataset": "medethicsqa", "kind": "mcq", "id": it.get("id")},
                reference=ref,
            ))
        return out
