"""18_e2_significance：E2 真实 LLM 基线的配对显著性检验（离线，无 LLM 调用）。

读 runs/real_llm_baselines.jsonl（scripts/11_real_llm_baselines.py 产物；
行字段：scenario/method/vector([B,N,A,J])/t/llm_calls[/sat_min/arb]）。
对每个基线方法（single_llm_llm / neutral_single_llm / medagents_style）vs MANE：
  1. FDBI 配对差异 bootstrap 95% CI + 双侧 p（场景重采样 20000 次）；
  2. 底线违反率（v < [0.15,0.20,0.15,0.20] 任一维度）McNemar 精确二项检验
     （b=MANE 守住而基线违反 的 discordant 对 = MANE 拯救数；c=反向）。
审稿人 MC1 补强：效应量 + 不确定性（不能只报均值差）。

用法：
  python scripts/18_e2_significance.py --runs runs/real_llm_baselines.jsonl \
      [--n-boot 20000] [--seed 42] [--out runs/e2_significance.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

FLOORS = np.array([0.15, 0.20, 0.15, 0.20])
TARGETS = ["MANE"]
BASELINES = ["single_llm_llm", "neutral_single_llm", "medagents_style"]


def fdbi(v) -> float:
    v = np.asarray(v, dtype=float)
    mu = v.mean()
    if mu <= 0:
        return 0.0
    return float(1.0 - v.std() / mu)


def violates(v) -> bool:
    return bool((np.asarray(v, dtype=float) < FLOORS).any())


def _binom_tail_two_sided(k: int, t: int) -> float:
    """X ~ Binomial(t, 0.5) 双侧精确 p（无 scipy 依赖；k=0 或 k=t 时取 2·(1/2)^t）。"""
    if t <= 0:
        return 1.0
    probs = [math.comb(t, i) / (2.0 ** t) for i in range(t + 1)]
    left = sum(probs[:k + 1])
    right = sum(probs[k:])
    return min(1.0, 2.0 * min(left, right))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/real_llm_baselines.jsonl")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    by_sc = {}   # scenario_id -> {method: vector}
    with open(args.runs, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v = r.get("vector")
            if v is None:
                continue
            by_sc.setdefault(r["scenario"], {})[r["method"]] = np.asarray(v, dtype=float)

    scs = sorted(by_sc)
    print(f"加载 {len(scs)} 场景（方法: {sorted({m for s in by_sc.values() for m in s})}）")
    rng = np.random.default_rng(args.seed)
    report = {"n_scenarios": len(scs), "pairs": []}

    print(f"\n{'基线':<20}{'n':>4}{'MANE FDBI':>10}{'基FDBI':>9}{'Δ':>8}"
          f"{'95%CI':>22}{'p_boot':>8}{'胜场':>7}")
    print("-" * 96)
    for base in BASELINES:
        pairs = [(s, by_sc[s]["MANE"], by_sc[s][base]) for s in scs
                 if "MANE" in by_sc[s] and base in by_sc[s]]
        if not pairs:
            continue
        n = len(pairs)
        mv = np.array([fdbi(m) for _, m, _ in pairs])
        bv = np.array([fdbi(b) for _, _, b in pairs])
        d = mv - bv
        wins = int(np.mean(d > 1e-9) * n)
        boots = np.empty(args.n_boot)
        for i in range(args.n_boot):
            idx = rng.integers(0, n, n)
            boots[i] = d[idx].mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p_boot = 2.0 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0)))
        # McNemar（底线违反）
        disc = [(violates(m), violates(b)) for _, m, b in pairs]
        b_cnt = sum(1 for m, b in disc if (not m) and b)   # MANE 守、基线违
        c_cnt = sum(1 for m, b in disc if m and (not b))    # MANE 违、基线守
        p_mcn = _binom_tail_two_sided(b_cnt, b_cnt + c_cnt)
        mane_viol = np.mean([1.0 if m else 0.0 for m, _ in disc])
        base_viol = np.mean([1.0 if b else 0.0 for _, b in disc])
        print(f"{base:<20}{n:>4}{mv.mean():>10.3f}{bv.mean():>9.3f}{d.mean():>+8.3f}"
              f"[{lo:+.3f},{hi:+.3f}]{p_boot:>8.3f}{wins:>4}/{n:<3}")
        print(f"{'':<20}{'':>4}{'底线违反率':>18}{'':>2}"
              f"{mane_viol:>9.1%}{base_viol:>9.1%}{'':>11}"
              f"McNemar b/c={b_cnt}/{c_cnt} p={p_mcn:.4f}")
        report["pairs"].append({
            "baseline": base, "n": n,
            "mane_fdbi": float(mv.mean()), "base_fdbi": float(bv.mean()),
            "delta_fdbi": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_boot": float(p_boot), "wins": wins,
            "mane_viol_rate": float(mane_viol),
            "base_viol_rate": float(base_viol),
            "mcnemar_b": b_cnt, "mcnemar_c": c_cnt, "mcnemar_p": p_mcn,
        })

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[已保存] {args.out}")

    print("\n口径：Δ>0 = MANE FDBI 更高；p_boot<0.05 = 配对差异显著；"
          "McNemar b 显著大 = MANE 在底线上'拯救'了更多场景。")


if __name__ == "__main__":
    main()
