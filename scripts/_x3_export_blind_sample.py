# -*- coding: utf-8 -*-
"""_x3_export_blind_sample：导出人工盲审抽样（四方法尽量均衡 + 密钥表）。

背景：文本级安全审计（_x2）与 floor 指标都来自机器，人工盲审是最后的证据。为免方法名
影响标注者，导出样本 CSV 不含方法列，另存一张 idx→(sid,method) 密钥表；标注者只填
violation_label(0/1) 与可选的 comment 列。

用法：
  python scripts/_x3_export_blind_sample.py --texts runs/cross_method_texts.jsonl \
      [--n 60] [--seed 7] [--out runs/audit_blind_sample.csv] \
      [--key runs/audit_blind_key.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default="runs/cross_method_texts.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/audit_blind_sample.csv")
    ap.add_argument("--key", default="runs/audit_blind_key.csv")
    args = ap.parse_args()

    random.seed(args.seed)
    rows = []
    with open(args.texts, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                if (o.get("rationale") or "").strip():
                    rows.append(o)

    # 按方法尽量均衡：先各取 ceil(n/4)，不足则由其余方法补齐
    methods = ["single_llm_llm", "neutral_single_llm", "medagents_style", "MANE"]
    per = {m: [r for r in rows if r["method"] == m] for m in methods}
    picked = []
    quota = max(1, args.n // len(methods))
    for m in methods:
        random.shuffle(per[m])
        picked.extend(per[m][:quota])
    # 补足到 n
    rest = [r for r in rows if r not in picked]
    random.shuffle(rest)
    picked.extend(rest[:max(0, args.n - len(picked))])
    random.shuffle(picked)
    picked = picked[:args.n]

    key = []
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "scenario_id", "rationale", "violation_label", "comment"])
        for i, r in enumerate(picked, 1):
            w.writerow([i, r["scenario_id"], r["rationale"], "", ""])
            key.append({"idx": i, "scenario_id": r["scenario_id"], "method": r["method"]})
    with open(args.key, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "scenario_id", "method"])
        w.writeheader()
        w.writerows(key)

    from collections import Counter
    print(f"导出 {len(picked)} 条 -> {args.out}（不含方法列，盲审用）")
    print(f"方法分布: {dict(Counter(r['method'] for r in picked))}")
    print(f"密钥表  -> {args.key}（标注完再打开核对）")


if __name__ == "__main__":
    main()
