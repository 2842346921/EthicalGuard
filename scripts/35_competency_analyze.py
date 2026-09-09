"""35_competency_analyze：按 ACGME 临床能力维度聚合官方 keypoint 覆盖（离线，读 34 产物）。

PrinciplismQA 的每条专家 keypoint 带 ACGME competency 标签（PATIENT CARE / MEDICAL KNOWLEDGE /
PROFESSIONALISM / SYSTEMS-BASED PRACTICE / PRACTICE-BASED LEARNING / INTERPERSONAL & COMMUNICATION）。
34 已落盘逐条 scores + competencies 顺序 → 本脚本把"覆盖分"按能力维度聚合，回答：
  "系统在哪些临床能力维度上最接近专家标准？议题感知的增益集中在哪几维？"
（比单一总分细，对接 deployment readiness 叙事；审稿人偏好这种可解释分解。）

聚合口径：每题内先按 competency 对 scores 取均值 → 跨题按 (方法, competency) 取均值。
用法：
  python scripts/35_competency_analyze.py \
      --in runs/content_vs_baseline.jsonl [--out runs/competency_analysis.txt]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/content_vs_baseline.jsonl")
    ap.add_argument("--out", default="runs/competency_analysis.txt")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    # 只保留有逐条 scores + competency 的行
    valid = [r for r in rows if r.get("pqa_scores") and r.get("competencies")]
    print(f"有效行 {len(valid)}/{len(rows)}（34 需带 --rubric 跑才含 competency）")
    if not valid:
        print("无 competency 数据——用 --rubric 指向 open-ended-rubric-principles.json 重跑 34。")
        return

    # 题内先按 comp 平均，再跨题平均
    per = defaultdict(lambda: defaultdict(list))  # method -> competency -> [题内均值]
    for r in valid:
        scores = r["pqa_scores"]
        comps = r["competencies"]
        byc = defaultdict(list)
        for s, c in zip(scores, comps):
            byc[c].append(s)
        for c, ss in byc.items():
            per[r["method"]][c].append(float(np.mean(ss)))

    methods = sorted({r["method"] for r in valid})
    comps_all = sorted({c for m in per.values() for c in m})
    lines = []
    def emit(s=""):
        lines.append(s)
        print(s)

    emit("===== 按 ACGME competency 聚合的官方 keypoint 覆盖（题内均值 → 跨题均值）=====")
    emit(f"{'方法':<20}" + "".join(f"{c[:14]:>16}" for c in comps_all) + f"{'整体':>8}")
    for m in methods:
        row = f"{m:<20}"
        for c in comps_all:
            xs = per[m].get(c, [])
            row += f"{np.mean(xs):>16.3f}" if xs else f"{'-':>16}"
        allx = [x for c in comps_all for x in per[m].get(c, [])]
        row += f"{np.mean(allx):>8.3f}" if allx else f"{'-':>8}"
        emit(row)

    emit("\n判读：看 B（议题感知）在哪些 competency 上显著高于单 LLM/medagents——")
    emit("若集中在 PATIENT CARE / PROFESSIONALISM 等伦理核心域 → 增益有临床能力语义；")
    emit("整体列 ≈ 官方 pqa 均值（交叉核对 34 汇总）。")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[已保存] {args.out}")


if __name__ == "__main__":
    main()
