"""33_expert_analyze：专家回填分析（R1 人类锚收口）——读 expert_survey/survey_rater*.csv。

输出三块：
  ① Q2 底线标定：每原则 专家"不可接受线"分布（全部标注者×全部病例）
     min/p10/median/p90/max vs 系统默认 [0.15,0.20,0.15,0.20]
     ——若专家中位数落在 0.10-0.25 且接近默认值 → 底线是专家标定的，非自定义；
  ② Q1 权重分配：每病例 专家平均分配 vs MANE 终态向量 → L1/JS 距离
     （"系统共识 vs 专家共识"的结构一致性）；
  ③ Q3 决策可接受性：多标注者一致性 —— 配对 Cohen κ（两两平均）或 Fleiss κ（≥2）
     + 可接受率（Q3≥2 占比）。人类 Kappa 收口。
用法：
  python scripts/33_expert_analyze.py \
      --survey-dir runs/expert_survey \
      --runs runs/mane_pqa30.jsonl \
      [--out runs/expert_analysis.txt]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ethicalguard.data.loaders import iter_negotiation_records

SYSTEM_FLOORS = np.array([0.15, 0.20, 0.15, 0.20])
PV = ["beneficence", "nonmaleficence", "autonomy", "justice"]


def _parse4(s: str):
    """'a,b,c,d' → [float]*4；失败返回 None。"""
    if s is None:
        return None
    try:
        xs = [float(x.strip()) for x in str(s).split(",")]
        return xs if len(xs) == 4 else None
    except Exception:  # noqa: BLE001
        return None


def _js(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / a.sum(), b / b.sum()
    m = 0.5 * (a + b)
    eps = 1e-12
    return float(0.5 * ((a * np.log((a + eps) / (m + eps))).sum()
                        + (b * np.log((b + eps) / (m + eps))).sum()))


def _cohen_kappa(a, b, k=4):
    """两标注者序数 0..k-1 的 Cohen κ（加权=线性? 用标准 Cohen 未加权按类别）。"""
    n = len(a)
    if n == 0:
        return float("nan")
    cats = range(k)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[int(x), int(y)] += 1
    po = np.trace(obs) / n
    pe = sum(obs[:, i].sum() * obs[i, :].sum() for i in cats) / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey-dir", default="runs/expert_survey")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="runs/expert_analysis.txt")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.survey_dir)
                   if f.startswith("survey_rater") and f.endswith(".csv"))
    if not files:
        print(f"[错误] {args.survey_dir} 下无 survey_rater*.csv（先跑 32_expert_package 并让专家回填）")
        sys.exit(1)
    data = {}   # scenario_id -> {rater -> row}
    for fn in files:
        rater = int(fn.replace("survey_rater", "").replace(".csv", ""))
        with open(os.path.join(args.survey_dir, fn), encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                data.setdefault(row["scenario_id"], {})[rater] = row

    mane = {r["scenario_id"]: r for r in iter_negotiation_records(args.runs)}
    lines = []
    def emit(s=""):
        lines.append(s)
        print(s)

    emit(f"标注者 {len(files)} 位 × 病例 {len(data)} 个\n")
    n_rat = len(files)

    # ---- ① Q2 底线标定 ----
    emit("===== ① 专家'不可接受线'分布（Q2，全部标注者×病例）vs 系统默认 =====")
    emit(f"{'原则':<14}{'样本':>5}{'min':>6}{'p10':>6}{'中位':>6}{'p90':>6}{'max':>6}  系统默认")
    for i, pname in enumerate(["行善B", "不伤害N", "自主A", "公正J"]):
        vals = []
        for sid, raters in data.items():
            for rater, row in raters.items():
                q = _parse4(row.get("Q2不可接受线(B,N,A,J)"))
                if q is not None:
                    vals.append(q[i])
        if not vals:
            continue
        v = np.array(vals)
        emit(f"{pname:<14}{len(v):>5}{v.min():>6.2f}{np.percentile(v, 10):>6.2f}"
             f"{np.median(v):>6.2f}{np.percentile(v, 90):>6.2f}{v.max():>6.2f}"
             f"{SYSTEM_FLOORS[i]:>8.2f}")
    emit("  判读：专家中位数落 0.10-0.25 且接近默认 → 底线被专家标定；偏差大 → 报告差异并讨论。\n")

    # ---- ② Q1 专家分配 vs MANE 终态 ----
    emit("===== ② 专家 Q1 平均分配 vs MANE 终态（同病例）=====")
    l1s, jss, n_pair = [], [], 0
    for sid, raters in data.items():
        qs = [_parse4(row.get("Q1分配100点(B,N,A,J)")) for row in raters.values()]
        qs = [np.array(q) for q in qs if q is not None]
        if not qs or sid not in mane:
            continue
        expert = np.mean(qs, axis=0)
        expert = expert / expert.sum()
        v = mane[sid].get("final_vector") or {}
        mvec = np.array([v.get(k, 0.25) for k in PV])
        mvec = mvec / mvec.sum()
        l1s.append(float(np.abs(expert - mvec).sum()))
        jss.append(_js(expert, mvec))
        n_pair += 1
    if l1s:
        emit(f"配对 {n_pair} 例：专家平均分配 vs MANE 终态  L1 均值={np.mean(l1s):.3f}"
             f"  JS 均值={np.mean(jss):.3f}")
        emit("  判读：L1 越小 → 系统共识越接近专家权重分配（结构一致性）。\n")
    else:
        emit("  无有效 Q1 数据（专家未回填？）\n")

    # ---- ③ Q3 Kappa ----
    emit("===== ③ 决策可接受性：Kappa + 可接受率 =====")
    if n_rat >= 2:
        ks = []
        rat_ids = sorted({r for rr in data.values() for r in rr})
        for i in range(n_rat):
            for j in range(i + 1, n_rat):
                a, b = [], []
                for sid, raters in data.items():
                    if rat_ids[i] in raters and rat_ids[j] in raters:
                        x = raters[rat_ids[i]].get("Q3决策可接受性(0-3)")
                        y = raters[rat_ids[j]].get("Q3决策可接受性(0-3)")
                        if x not in (None, "") and y not in (None, ""):
                            a.append(min(3, max(0, int(float(x)))))
                            b.append(min(3, max(0, int(float(y)))))
                if len(a) >= 5:
                    ks.append(_cohen_kappa(a, b, k=4))
        if ks:
            emit(f"配对 Cohen κ 均值 = {np.mean(ks):.3f}（{len(ks)} 对，κ≥0.6 为实质一致）")
        else:
            emit("  配对样本不足（需 ≥5 例 × ≥2 标注者）")
    acc = []
    for raters in data.values():
        for row in raters.values():
            x = row.get("Q3决策可接受性(0-3)")
            if x not in (None, ""):
                acc.append(int(float(x)))
    if acc:
        emit(f"可接受性分布：n={len(acc)}  均值={np.mean(acc):.2f}  "
             f"可接受率(Q3≥2)={np.mean([1 if x >= 2 else 0 for x in acc]):.1%}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[已保存] {args.out}")


if __name__ == "__main__":
    main()
