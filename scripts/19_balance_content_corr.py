"""19_balance_content_corr：平衡系数 ↔ 内容质量锚的收敛效度分析（离线，无 LLM）。

回答审稿人核心质疑："FDBI（平衡系数）凭什么衡量'好的医疗伦理决策'？"
论证第③层（构念效度）：若平衡的结构质量与专家锚定的内容质量共变 → FDBI 有内容意义；
若正交 → 如实报告"平衡是互补的结构维度"。两种结果都可写，关键是把数字拿到。

数据源（全部离线，已由 07/20 生成或可从 runs 重算）：
  --runs   runs/mane_pqa30.jsonl        （30 场景 MANE 协商 → 逐场景 FDBI）
  --align  runs/decision_alignment.json （scripts/20 产物：逐场景 rubric_alignment +
           judge{decision_clear,keypoint_cov,principle_align} 三轴）
指标对（Spearman ρ + 近似 p，同场景配对）：
  FDBI × rubric_alignment          （结构平衡 vs 词面对齐）
  FDBI × keypoint_cov              （结构平衡 vs 专家要点实质覆盖 —— 最关键）
  FDBI × decision_clear / principle_align / 综合
  rubric × keypoint_cov            （词面 vs 实质 的表面效度检查：若高相关 → rubric 可信；
                                     若低相关 → 印证 20 的发现：词面像≠实质达标）
n≈29 提示性；正文按"探索性构念效度"口径，需更大样本确认。

用法：
  python scripts/19_balance_content_corr.py \
      --runs runs/mane_pqa30.jsonl --align runs/decision_alignment.json \
      [--out runs/balance_content_corr.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ethicalguard.data.loaders import iter_negotiation_records


def fdbi_of(v: dict) -> float:
    arr = np.array([v.get("beneficence", 0), v.get("nonmaleficence", 0),
                    v.get("autonomy", 0), v.get("justice", 0)], dtype=float)
    mu = arr.mean()
    return float(1.0 - arr.std() / mu) if mu > 0 else 0.0


def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均秩（ties 取平均），与 scipy.stats.rankdata 一致。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman ρ + 双侧近似 p（z = ρ·sqrt(n−1) 正态近似；n<30 时标注提示性）。
    若环境有 scipy 则用精确 t 分布 p。"""
    try:
        from scipy.stats import spearmanr  # type: ignore
        rho, p = spearmanr(a, b)
        return float(rho), float(p), "scipy"
    except Exception:  # noqa: BLE001
        ra, rb = _rankdata(np.asarray(a, float)), _rankdata(np.asarray(b, float))
        ra_m, rb_m = ra - ra.mean(), rb - rb.mean()
        denom = math.sqrt((ra_m ** 2).sum() * (rb_m ** 2).sum())
        rho = float((ra_m * rb_m).sum() / denom) if denom > 0 else 0.0
        n = len(a)
        z = rho * math.sqrt(max(0, n - 1))
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        return rho, min(1.0, p), "normal-approx"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/mane_pqa30.jsonl")
    ap.add_argument("--align", default="runs/decision_alignment.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 1) FDBI 逐场景
    fdbi_by = {}
    for r in iter_negotiation_records(args.runs):
        v = r.get("final_vector")
        if v:
            fdbi_by[r["scenario_id"]] = fdbi_of(v)
    # 2) rubric + judge 三轴逐场景
    align = json.load(open(args.align, encoding="utf-8"))
    align_by = {}
    for row in align:
        j = row.get("judge")
        if j and "keypoint_cov" in j:
            align_by[row["scenario_id"]] = {
                "rubric": float(row.get("rubric_alignment", float("nan"))),
                "clear": float(j["decision_clear"]),
                "cov": float(j["keypoint_cov"]),
                "align": float(j["principle_align"]),
                "comp": (float(j["decision_clear"]) + float(j["keypoint_cov"])
                         + float(j["principle_align"])) / 3.0,
            }

    scs = sorted(set(fdbi_by) & set(align_by))
    print(f"同场景配对: {len(scs)} 个（runs {len(fdbi_by)} × align {len(align_by)} 交集）")
    if len(scs) < 10:
        print("样本过少，相关性无意义——需补 align 覆盖后再跑。")
        return

    fx = np.array([fdbi_by[s] for s in scs])
    pairs = [
        ("FDBI × rubric_alignment", "rubric", "词面对齐"),
        ("FDBI × keypoint_cov", "cov", "专家要点实质覆盖"),
        ("FDBI × decision_clear", "clear", "决策明确性"),
        ("FDBI × principle_align", "align", "原则立场一致"),
        ("FDBI × 综合", "comp", "三轴均值"),
        ("rubric × keypoint_cov", "cov", "词面 vs 实质（表面效度）"),
    ]
    print(f"\n{'指标对':<28}{'n':>4}{'ρ':>8}{'p':>8}  口径")
    print("-" * 78)
    results = []
    for label, key, note in pairs:
        y0 = np.array([align_by[s][key] for s in scs])
        x0 = fx if label.startswith("FDBI") else np.array([align_by[s]["rubric"] for s in scs])
        mask = ~(np.isnan(x0) | np.isnan(y0))
        if mask.sum() < 10:
            continue
        rho, p, method = spearman(x0[mask], y0[mask])
        print(f"{label:<28}{int(mask.sum()):>4}{rho:>8.3f}{p:>8.4f}  {note}（p:{method}）")
        results.append({"pair": label, "n": int(mask.sum()), "rho": rho, "p": p,
                        "p_method": method, "note": note})

    print("\n口径：ρ>0 且 p 小 → 平衡的结构质量与专家锚定的内容质量共变（收敛效度证据）；")
    print("ρ≈0 → 平衡是正交结构维度（互补而非替代）。n≈29 提示性，正文按探索性表述。")
    print("rubric×cov 若低相关 → 印证决策对齐发现：词面重叠 ≠ 实质要点覆盖（表面效度局限）。")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"n": len(scs), "scenarios": scs,
                       "correlations": results,
                       "per_scenario": [{"scenario_id": s, "fdbi": fdbi_by[s], **align_by[s]}
                                        for s in scs]},
                      f, ensure_ascii=False, indent=2)
        print(f"[已保存] {args.out}")


if __name__ == "__main__":
    main()
