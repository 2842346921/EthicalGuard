"""E1b 底线敏感性：扫 floor 0.15→0.30，证明"底线=约束"何时真激活（非平凡）。

背景（审稿人 T1/G10b）：默认 floor=[0.15,0.20,0.15,0.20] 远低于 LLM 协商终态（~0.2-0.3），
35 场景无任何维度触底——约束从未被乘子激活，T1（底线保证）是 KKT 平凡推论。
本实验扫 floor 高度，报告：
- 每个 floor 档下，终态最小维度（越接近 floor = 约束越被"顶住"）
- KKT 乘子是否非零（激活证据；乘子>0 ⇔ 约束 active）
- 底线违反率（floor 抬高后若违反 → 约束不够强；全守住 → 约束在工作）

用法：
  python scripts/15_floor_sensitivity.py --config configs/config.yaml \
      --mode local --input data_cache/scenarios.jsonl --limit 20
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ethicalguard.config import Config
from ethicalguard.data import load_scenarios_from_jsonl
from ethicalguard.mane import MANEEngine
from ethicalguard.utils import set_seed, setup_logging

logger = setup_logging()

# floor 有效范围：四原则权重在单纯形 Σv=1 上，floor 上限 = 0.25（4×0.25=1）。
# floor>0.25 数学不可满足（E1b 诊断确认）——0.28/0.30 的"违反 100%"是测试设计错误。
FLOOR_GRID = [0.15, 0.18, 0.20, 0.22, 0.24, 0.25]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--mode", default=None, choices=["rule", "api", "local"])
    ap.add_argument("--input", required=True, help="场景 jsonl（01 产物）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grid", default=None, help="floor 网格（逗号分隔，默认 0.15..0.30）")
    ap.add_argument("--seed", type=int, default=42)
    # 论文级精度：收紧求解器容差（默认 1e-3）。实测 floor≤0.24 时 tol=1e-4 可收敛
    # （~2500 迭代）；floor=0.25 角落振荡极限 ~2.4e-4（达不到 1e-4，如实报告）。
    # 求解器耗时相对 LLM 协商可忽略 → 收紧免费。max_iter 自动抬高（1e-4 需 ~3000 迭代）。
    ap.add_argument("--tol", type=float, default=None, help="求解器 KKT 容差（默认取 config，1e-3）")
    ap.add_argument("--max-iter", type=int, default=None, help="求解器最大外层迭代（默认取 config）")
    args = ap.parse_args()

    cfg0 = Config.load(args.config)
    cfg0.llm.mode = args.mode or cfg0.llm.mode
    if args.tol is not None:
        cfg0.gne.tol = float(args.tol)
    if args.max_iter is not None:
        cfg0.gne.max_iter = int(args.max_iter)
    elif args.tol is not None and args.tol < 1e-3:
        cfg0.gne.max_iter = max(int(getattr(cfg0.gne, "max_iter", 2000) or 2000), 4000)
    if cfg0.llm.mode == "rule":
        print("[警告] mode=rule——只作机制验证，不作论文证据。实验请用 local。")
    set_seed(args.seed)
    grid = [float(x) for x in args.grid.split(",")] if args.grid else FLOOR_GRID

    scenarios = list(load_scenarios_from_jsonl(args.input))
    if args.limit > 0:
        scenarios = scenarios[:args.limit]
    # 违约判据容差与求解器 tol 对齐：KKT primal ≤ tol 的出口 slack 是数值容差（量级 ≤1e-3），
    # 不是机制失败——严格 1e-9 判据会把 tol 级出口误报为"违约"（旧版 F=0.22 的 20% 即此类）。
    solver_tol = max(1e-9, float(getattr(cfg0.gne, "tol", 1e-3) or 1e-3))
    print(f"===== 底线敏感性（floor 扫描，{len(scenarios)} 场景，mode={cfg0.llm.mode}，"
          f"违约判据容差={solver_tol:.0e}）=====")
    print(f"{'floor':>6}{'终态min均值':>10}{'乘子激活场景':>10}{'终态<floor':>10}"
          f"{'最大违约':>9}{'终态min最低':>10}{'仲裁率':>8}{'KKT均':>9}")
    print("-" * 76)
    for f in grid:
        floors = [f] * 4
        mins, lamb_active, below, viols, kkts, arbs = [], [], [], [], [], []
        for sc in scenarios:
            vcfg = Config.load(args.config)
            vcfg.llm.mode = args.mode or cfg0.llm.mode
            # tol/max_iter 覆盖透传（cfg0 已含 --tol/--max-iter 覆盖）
            vcfg.gne.tol = cfg0.gne.tol
            vcfg.gne.max_iter = cfg0.gne.max_iter
            vcfg.gne.floors = floors
            r = MANEEngine(vcfg).run(sc)
            v = r.final_vector.as_array() if r.final_vector is not None else np.full(4, 0.25)
            mins.append(float(v.min()))
            # 乘子激活（精确证据）：floor_lambdas 任一 >0 ⇔ 互补松弛下约束 active
            lam = r.floor_lambdas or {}
            active = any(abs(val) > 1e-6 for val in lam.values())
            lamb_active.append(1.0 if active else 0.0)
            below.append(1.0 if (v < np.array(floors) - solver_tol).any() else 0.0)
            viols.append(float(np.maximum(np.array(floors) - v, 0.0).max()))
            if r.kkt_residual is not None:
                kkts.append(r.kkt_residual)
            arbs.append(r.arbitration_triggered)
        print(f"{f:>6.2f}{np.mean(mins):>10.4f}{np.mean(lamb_active):>10.0%}"
              f"{np.mean(below):>10.0%}{max(viols):>9.4f}{np.min(mins):>10.4f}"
              f"{np.mean(arbs):>8.0%}{(np.mean(kkts) if kkts else float('nan')):>9.1e}")

    print("\n解读：")
    print("  - 乘子激活场景率随 floor 升高（λ_floor>0 = 互补松弛下约束真 active）→ 底线实证激活点")
    print("  - 终态min均值 ≈ floor → 约束把向量顶在底线上（真拉回）")
    print("  - 违约判据容差 = 求解器 tol：仅当违约量级 > tol 才算底线未守住——那是合成层问题")
    print("    （仲裁 0.5/0.5 融合稀释：裁决向量对 floor 无感知，弱原则恒 0.20~0.22，高 floor 被拉穿），")
    print("    而非求解器失败（KKT primal ≤ tol 已守）。修复见 negotiation.py _project_onto_floor。")


if __name__ == "__main__":
    main()
