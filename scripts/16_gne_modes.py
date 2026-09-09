"""16_gne_modes：GNE 结构组合实验——"公共 v 让步 vs 独立策略" × balance × floor（标签错位实证）。

背景：审稿人质疑"所有 agent 共享公共 v 的最佳响应 = 非教科书 GNE"。
数学重估修正：α_i 各自独立（Δ⁴ softmax），v=Σwα 是集体量，floor 约束 g(v)≤0 共享耦合
——符合 GNE 定义。真正可辨的是 balance（软目标惩罚 γ 进 J_i，非硬约束）与耦合模式。

本实验做**三开关组合矩阵**（每场景每组合一次完整协商）：
  1. coupling   : collective（公共 v 让步，γ·balance_force(v) 进目标）
                 / independent（教科书对照：去掉 balance 让步，agent 纯追自己 s_i）
  2. use_balance: True（γ>0）/ False（γ=0，w/o 平衡正则）
  3. floors     : 默认 0.15/0.20/0.15/0.20 或 --floors 指定（如 0.28 激活档）

⚠️ 退化说明：independent 定义上忽略 balance（_best_response 强制 balance=0），
故 independent+balance_on ≡ independent+balance_off。用 --matrix 显式给组合可跳过退化格。

用法（推荐真 4 格：coupling × floor）：
  python scripts/16_gne_modes.py --config configs/config.yaml \
      --input data_cache/scenarios.jsonl --mode local --limit 50 \
      --matrix "collective@0.15,collective@0.28,independent@0.15,independent@0.28"

或（coupling × balance，3 有效格，independent 只有 1 格）：
  python scripts/16_gne_modes.py ... --matrix "collective@bal,collective@nobal,independent@nobal"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ethicalguard.config import Config
from ethicalguard.data import load_scenarios_from_jsonl
from ethicalguard.eval import metrics as M
from ethicalguard.mane import MANEEngine
from ethicalguard.utils import set_seed, setup_logging

logger = setup_logging()
FLOORS_DEFAULT = np.array([0.15, 0.20, 0.15, 0.20])


def _parse_matrix(spec: str):
    """解析组合矩阵："coupling@floor|bal" 列表。
    floor 形式：'0.15'=统一 / '0.15,0.2,0.15,0.2'=逐维；'bal'/'nobal' 控制 balance。
    例：collective@bal,collective@nobal,independent@nobal
        collective@0.15,collective@0.28,independent@0.15,independent@0.28
    """
    combos = []
    for item in spec.split(","):
        item = item.strip()
        if not item or "@" not in item:
            continue
        coupling, val = item.split("@", 1)
        val = val.strip()
        if val in ("bal", "nobal"):
            floors = FLOORS_DEFAULT
            use_balance = val == "bal"
        elif val == "def":
            floors = FLOORS_DEFAULT  # 默认 [0.15,0.20,0.15,0.20]
            use_balance = True
        else:
            parts = [float(x) for x in val.split(",") if x.strip()]
            # 单标量=统一档（敏感性用）；四值=逐维显式
            floors = np.array(parts if len(parts) == 4 else [parts[0]] * 4)
            use_balance = True  # floor 组合默认 balance on（退化格跳过即可）
        label_val = "bal" if val in ("bal", "nobal") else ("def" if val == "def" else str(round(float(floors[0]), 2)))
        combos.append({"coupling": coupling, "floors": floors, "use_balance": use_balance,
                       "label": f"{coupling}@{label_val}"})
    return combos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", required=True, help="场景 jsonl（01 产物）")
    ap.add_argument("--mode", default=None, choices=["rule", "api", "local"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--matrix", required=True, help="组合矩阵（见 docstring）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg0 = Config.load(args.config)
    cfg0.llm.mode = args.mode or cfg0.llm.mode
    if cfg0.llm.mode == "rule":
        print("[警告] mode=rule——只作机制验证。论文证据用 local。")
    set_seed(args.seed)
    combos = _parse_matrix(args.matrix)

    scenarios = list(load_scenarios_from_jsonl(args.input))
    if args.limit > 0:
        scenarios = scenarios[:args.limit]

    print(f"===== GNE 结构组合实验（{len(scenarios)} 场景，{len(combos)} 组合，mode={cfg0.llm.mode}）=====")
    for c in combos:
        print(f"  组合: coupling={c['coupling']} floors={c['floors'].tolist()} use_balance={c['use_balance']}")
    print(f"\n{'场景':<22}{'组合':<22}{'FDBI':>7}{'PCI':>7}{'底线违':>7}{'满意min':>8}{'轮次':>5}")
    rows = []
    for sc in scenarios:
        for c in combos:
            vcfg = Config.load(args.config)
            vcfg.llm.mode = args.mode or cfg0.llm.mode
            vcfg.gne.floors = c["floors"].tolist()
            vcfg.gne.coupling = c["coupling"]
            vcfg.mane.use_balance = c["use_balance"]
            r = MANEEngine(vcfg).run(sc)
            v = r.final_vector.as_array() if r.final_vector is not None else np.full(4, 0.25)
            below = float((v < c["floors"]).mean())
            sat_min = min(r.agent_satisfactions.values()) if r.agent_satisfactions else float("nan")
            print(f"{sc.scenario_id:<22}{c['label']:<22}{M.fdbi(v):>7.3f}{M.pci(v):>7.3f}"
                  f"{below:>7.3f}{sat_min:>8.3f}{r.rounds:>5}")
            rows.append({"scenario": sc.scenario_id, "combo": c["label"],
                         "fdbi": M.fdbi(v), "pci": M.pci(v), "below": below,
                         "sat_min": sat_min, "rounds": r.rounds,
                         "coupling": c["coupling"], "use_balance": c["use_balance"],
                         "floor": float(c["floors"][0])})

    # 汇总
    print()
    print("===== 汇总（同场景均值）=====")
    from collections import defaultdict
    agg = defaultdict(list)
    for row in rows:
        agg[row["combo"]].append(row)
    print(f"{'组合':<22}{'FDBI':>8}{'PCI':>8}{'底线违':>8}{'满意min':>9}{'轮次':>6}")
    for label in [c["label"] for c in combos]:
        rs = agg[label]
        if not rs:
            continue
        print(f"{label:<22}{np.mean([x['fdbi'] for x in rs]):>8.3f}"
              f"{np.mean([x['pci'] for x in rs]):>8.3f}"
              f"{np.mean([x['below'] for x in rs]):>8.3f}"
              f"{np.nanmean([x['sat_min'] for x in rs]):>9.3f}"
              f"{np.mean([x['rounds'] for x in rs]):>6.2f}")

    # 两两配对差异
    if len(combos) >= 2:
        print()
        print("===== 配对差异（同场景）=====")
        by_sc = defaultdict(dict)
        for row in rows:
            by_sc[row["scenario"]][row["combo"]] = row
        labels = [c["label"] for c in combos]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                scs = [s for s in by_sc if a in by_sc[s] and b in by_sc[s]]
                if not scs:
                    continue
                d_fdbi = [by_sc[s][a]["fdbi"] - by_sc[s][b]["fdbi"] for s in scs]
                d_sat = [by_sc[s][a]["sat_min"] - by_sc[s][b]["sat_min"] for s in scs
                         if not np.isnan(by_sc[s][a]["sat_min"]) and not np.isnan(by_sc[s][b]["sat_min"])]
                print(f"  {a} vs {b} (n={len(scs)}): FDBI 差 {np.mean(d_fdbi):+.3f}"
                      + (f" | 满意min 差 {np.mean(d_sat):+.3f}" if d_sat else ""))
    print("\n解读：")
    print("  - coupling 维度：collective vs independent 的 FDBI/满意min 差 = '公共 v 让步'贡献")
    print("  - floor 维度：0.15 vs 0.28 的底线违 = 约束激活点（配合乘子看）")
    print("  - balance 维度：bal vs nobal = 软正则边际（仅 collective 有意义）")


if __name__ == "__main__":
    main()
