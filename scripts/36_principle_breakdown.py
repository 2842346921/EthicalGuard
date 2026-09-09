"""36_principle_breakdown：By-Principles 分解 + Knowledge-Practice 对照（离线，对齐 PrinciplismQA Table 5）。

PrinciplismQA 论文按四原则报告 Know./Prac. 分（Table 5）。本脚本：
  ① Practice：读 34（content_vs_baseline.jsonl，30 题同批 6 方法）与 37（practice_scale.jsonl，全量基线）
     产物（均含 qid + pqa），按 rubric 的 principles 标签分组聚合 → 方法 × 四原则的官方分表；
  ② Knowledge：读 21（mcqa_official.jsonl，2182 全量，含 principlism+correct）→ 逐原则答对率。
输出对齐论文 Table 5 的"行=模型/方法，列=四原则（Know 一列 + Practice 每方法多列）"。
用法：
  python scripts/36_principle_breakdown.py \
      --mcq runs/mcqa_official.jsonl \
      --content runs/content_vs_baseline.jsonl \
      --practice runs/practice_scale.jsonl \
      --rubric data/PrinciplismQA/data/open-ended-rubric-principles.json \
      [--out runs/principle_breakdown.txt]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

PRIN = {"autonomy": "自主A", "beneficence": "行善B", "beneficience": "行善B",
        "non_maleficence": "不伤害N", "nonmaleficence": "不伤害N",
        "nonmaleficience": "不伤害N", "justice": "公正J"}
ORDER = ["行善B", "不伤害N", "自主A", "公正J"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcq", default="runs/mcqa_official.jsonl")
    ap.add_argument("--content", default="runs/content_vs_baseline.jsonl")
    ap.add_argument("--practice", default="runs/practice_scale.jsonl")
    ap.add_argument("--rubric", default=None, help="open-ended-rubric-principles.json（qid→principles）")
    ap.add_argument("--out", default="runs/principle_breakdown.txt")
    args = ap.parse_args()

    rubric_prin = {}
    if args.rubric and os.path.exists(args.rubric):
        for it in json.load(open(args.rubric, encoding="utf-8")):
            rubric_prin[it.get("qid")] = [PRIN.get(p, p) for p in it.get("principles", [])]
    lines = []
    def emit(s=""):
        lines.append(s)
        print(s)

    # ---- ① Knowledge 逐原则（21 产物：Qwen3-8B 官方 MCQ）----
    know = defaultdict(lambda: {"n": 0, "ok": 0})
    if os.path.exists(args.mcq):
        for l in open(args.mcq, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            o = json.loads(l)
            pl = o.get("principlism") or {}
            ok = 1 if o.get("correct") else 0
            for k in ("autonomy", "beneficience", "nonmaleficience", "justice"):
                if pl.get(k):
                    key = PRIN.get(k, k)
                    know[key]["n"] += 1
                    know[key]["ok"] += ok
    emit("===== Knowledge（MCQ 2182，Qwen3-8B，21 产物）逐原则答对率 =====")
    krow = {}
    for p in ORDER:
        d = know.get(p)
        acc = d["ok"] / d["n"] if d and d["n"] else float("nan")
        krow[p] = acc
        emit(f"  {p:<6} n={d['n'] if d else 0:>4}  acc={acc:.3f}" if d and d["n"] else f"  {p} 无")
    emit(f"  总 acc = {np.mean([krow[p] for p in ORDER if not np.isnan(krow[p])]):.3f}\n")

    # ---- ② Practice 按原则聚合（34 产物（方法）与 37 产物（基线全量），同口径 Gained/Sum）----
    def load_practice(path, tag):
        out = defaultdict(dict)  # method -> {principle: [pqa]}
        if not os.path.exists(path):
            return out
        for l in open(path, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            o = json.loads(l)
            pqa = o.get("pqa")
            if not isinstance(pqa, float):
                continue
            qid = o.get("qid")
            if qid is None and o.get("scenario_id"):
                # 回退：PQ-368-756 → qid=756（基础版 34 产物无 qid 字段）
                _p = str(o["scenario_id"]).split("-")
                if len(_p) >= 2 and _p[-1].isdigit():
                    qid = int(_p[-1])
            prins = rubric_prin.get(qid) or []
            for p in prins:
                out[o.get("method", tag)].setdefault(p, []).append(pqa)
        return out

    prac = load_practice(args.content, "34")
    prac2 = load_practice(args.practice, "37")
    for m, d in prac2.items():
        prac.setdefault(m, {})
        for p, xs in d.items():
            prac[m].setdefault(p, []).extend(xs)

    emit("===== Practice 按原则聚合（官方 Gained/Sum；34 同批方法 + 37 全量基线）=====")
    emit(f"{'方法':<22}" + "".join(f"{p:>10}" for p in ORDER) + f"{'整体':>8}")
    for m in sorted(prac):
        row = f"{m:<22}"
        row += "".join(f"{np.mean(prac[m][p]):>10.3f}" if prac[m].get(p) else f"{'-':>10}"
                       for p in ORDER)
        allx = [x for p in ORDER for x in prac[m].get(p, [])]
        row += f"{np.mean(allx):>8.3f}" if allx else f"{'-':>8}"
        emit(row)

    emit("\n口径：与 PrinciplismQA Table 5 同构（Practice 分按题含该原则分组）。")
    emit("34 的 30 题与 37 的全量子集不同——分开看，勿合并均数做跨批比较；")
    emit("Knowledge 与 Practice 数值不可直接比大小（不同度量），论文同此口径只并排看趋势。")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[已保存] {args.out}")


if __name__ == "__main__":
    main()
