"""E4：指标可信度校准（审稿人 A3/F 问题）——纯分析，读 runs 数据。

1. **FDBI 退化校准**：对"恒定 0.25×4""纯单极权重""随机游走""规则基线"跑同一 FDBI 管线，
   证明 MANE 的 FDBI 0.892 不是低信息量输出也能拿到的机械分数。
2. **L3 盒机械抬升检查**：FDBI = 1−σ/μ 只测离散度；L3 底线约束把权重钉在盒内会压缩 σ。
   这里对已有结果做"盒敏感性"近似：把终态向量 clip 到不同盒宽，看 FDBI 变化——
   若 clip 到 [0.15,0.20,0.15,0.20] 与 clip 到 [0,0,0,0] 的 FDBI 差异大 → 机械抬升嫌疑大。
3. **双轴主表**：保障轴（底线违例率/资源可行/KKT）与满意度轴分开报告，FDBI 降为诊断指标。

用法：
  python scripts/12_metric_calibration.py --runs runs/mane_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ethicalguard.eval import metrics as M

FLOORS = np.array([0.15, 0.20, 0.15, 0.20])


def _fdbi(v) -> float:
    return M.fdbi(np.asarray(v, dtype=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/mane_results.jsonl")
    args = ap.parse_args()

    runs = []
    from ethicalguard.data.loaders import iter_negotiation_records
    runs = iter_negotiation_records(args.runs)
    vectors = []
    for r in runs:
        v = r.get("final_vector") or {}
        vectors.append(np.array([v.get("beneficence", 0), v.get("nonmaleficence", 0),
                                 v.get("autonomy", 0), v.get("justice", 0)], dtype=float))
    vs = np.array(vectors)

    print("===== E4a: FDBI 退化校准（证明 0.892 非机械分数）=====")
    n = len(vs)
    # 退化系统
    rng = np.random.default_rng(42)
    constant = np.full((n, 4), 0.25)
    uniform_jitter = rng.dirichlet(np.ones(4), size=n)          # 均匀随机
    single_pole = np.tile(np.array([0.97, 0.01, 0.01, 0.01]), (n, 1))  # 单极
    bimodal = np.array([np.array([0.5, 0.5, 0, 0]) if i % 2 == 0 else np.array([0, 0, 0.5, 0.5])
                        for i in range(n)])                     # 双极（冲突）
    random_walk = rng.uniform(0, 1, size=(n, 4))
    random_walk = random_walk / random_walk.sum(axis=1, keepdims=True)

    print(f"{'系统':<16}{'FDBI 均值':>10}{'说明':<30}")
    print(f"{'恒定 0.25':<16}{np.mean([_fdbi(x) for x in constant]):>10.4f}  理想平衡（无信息）")
    print(f"{'均匀随机':<16}{np.mean([_fdbi(x) for x in uniform_jitter]):>10.4f}  随机也能拿的分")
    print(f"{'单极 0.97':<16}{np.mean([_fdbi(x) for x in single_pole]):>10.4f}  极度偏斜")
    print(f"{'双极冲突':<16}{np.mean([_fdbi(x) for x in bimodal]):>10.4f}  两派对立")
    print(f"{'随机游走':<16}{np.mean([_fdbi(x) for x in random_walk]):>10.4f}  无协商")
    print(f"{'MANE 实际':<16}{np.mean([_fdbi(x) for x in vs]):>10.4f}  本系统 35 场景")
    print("解读：MANE 应显著高于随机游走/单极/双极；若接近均匀随机说明 FDBI 区分度低。")

    print()
    print("===== E4b: L3 盒机械抬升检查（clip 敏感性）=====")
    print("把终态 clip 到不同'盒'内（模拟约束盒作用），看 FDBI 变化：")
    for label, lo, hi in (("无盒[0,1]", 0.0, 1.0),
                          ("默认盒[0.15/0.2]", None, None),
                          ("宽盒[0.1,0.4]", 0.10, 0.40),
                          ("窄盒[0.2,0.3]", 0.20, 0.30)):
        if label == "默认盒[0.15/0.2]":
            cl = np.array([0.15, 0.20, 0.15, 0.20])
            cu = np.ones(4)
            clipped = np.clip(vs, cl, cu)
        else:
            clipped = np.clip(vs, lo, hi)
        # 归一化（clip 后和可能≠1）
        clipped = clipped / clipped.sum(axis=1, keepdims=True)
        f = np.mean([_fdbi(x) for x in clipped])
        print(f"  clip 到 {label:<18} FDBI={f:.4f}")
    print("解读：若 clip 到默认盒与无盒的 FDBI 差异大 → 约束盒机械抬升嫌疑；"
          "若差异小 → MANE 的平衡是协商产物而非盒的作用。")

    print()
    print("===== E4c: 双轴主表（保障轴 vs 满意度轴）=====")
    print(f"{'场景':<22}{'保障轴':<28}{'满意度':<8}")
    print(f"{'':<22}{'底线违':>7}{'可行':>7}{'KKT':>7}{'仲裁':>6}{'min':>8}")
    viol_n = 0
    for r, v in zip(runs, vs):
        below = int((v < FLOORS).any())
        viol_n += below
        feas = r.get("resource_feasible")
        kkt = r.get("kkt_residual")
        sat = min(r.get("agent_satisfactions", {}).values()) if r.get("agent_satisfactions") else float("nan")
        print(f"{r['scenario_id']:<22}{below:>7}{str(feas):>7}"
              f"{('%.1e' % kkt) if kkt is not None else 'None':>7}"
              f"{str(r.get('arbitration_triggered')):>6}{sat:>8.3f}")
    print(f"\n底线违反场景数: {viol_n}/{len(runs)}  "
          f"资源可行率: {np.mean([1.0 if r.get('resource_feasible') else 0.0 for r in runs if r.get('resource_feasible') is not None]):.0%}")
    print("解读：双轴报告——主结果看'底线违 0 + 可行 100% + KKT 达标'（保障），"
          "满意度 min 看'对最脆弱方是否可接受'（软质量），FDBI 仅作诊断。")


if __name__ == "__main__":
    main()
