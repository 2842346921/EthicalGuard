"""06_baselines：基线对比（按数据集分组）——顶会审稿人第一张表的证据。

对比方法（对同一批场景）：
- single_llm : 单模型直接决策（无协商，规则近似，eval/baselines.single_llm_baseline）
- harmony    : HARMONY 式原则对抗（行善 vs 自主 胜者输出，规则近似）
- rule_no_gne: 规则五方提案的朴素集体（贝叶斯加权替代——无 GNE 求解/仲裁的精炼）
- MANE       : 完整框架（从 runs/mane_results.jsonl 读取 final_vector）

指标：FDBI（平衡）/ PCI（冲突强度）/ dist_to_ideal（‖v−0.25‖）/ 放弃率（权重跌破 L3 底线维度占比）。
按数据集（source.dataset）分组输出——5 个数据集分别给性能。

用法：
  python scripts/06_baselines.py --config configs/config.yaml --input data_cache/scenarios.jsonl
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
from ethicalguard.eval.baselines import harmony_style_baseline, rule_collective_weights, single_llm_baseline
from ethicalguard.utils import setup_logging

logger = setup_logging()

IDEAL = np.array([0.25, 0.25, 0.25, 0.25])


def _summarize(weights_list, floors) -> dict:
    ws = np.array(weights_list)
    fdbis = [M.fdbi(w) for w in ws]
    pcis = [M.pci(w) for w in ws]
    dists = [float(np.linalg.norm(w - IDEAL)) for w in ws]
    ab_rate = float(np.mean([(w < floors).mean() for w in ws]))
    # ECS：底线一票否决 + FDBI（基线无 sat/kkt/feasible，只判底线）
    ecs = M.ethical_composite(ws, [None] * len(ws), floors, kkt_gate=False)
    return {"FDBI": float(np.mean(fdbis)), "PCI": float(np.mean(pcis)),
            "dist": float(np.mean(dists)), "放弃率": ab_rate,
            "ECS": ecs["mean"], "通过率": ecs["pass_rate"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", required=True, help="场景 jsonl（01 产物）")
    ap.add_argument("--runs", default=None, help="MANE 结果 jsonl（默认 config.run.out_dir/mane_results.jsonl）")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    runs_path = args.runs or os.path.join(cfg.run.out_dir, "mane_results.jsonl")
    floors = np.asarray(cfg.resilience.floors, dtype=float)

    mane_by_id = {}
    if os.path.exists(runs_path):
        from ethicalguard.data.loaders import iter_negotiation_records
        for r in iter_negotiation_records(runs_path):
            v = r.get("final_vector")
            if v:
                mane_by_id[r["scenario_id"]] = np.array(
                    [v["beneficence"], v["nonmaleficence"], v["autonomy"], v["justice"]], dtype=float)
        logger.info("MANE 结果 %d 条 <- %s", len(mane_by_id), runs_path)
    else:
        logger.warning("未找到 MANE 结果 %s —— MANE 行将显示 N/A", runs_path)

    from collections import defaultdict
    methods = defaultdict(lambda: defaultdict(list))  # dataset -> method -> [weights]
    for sc in load_scenarios_from_jsonl(args.input):
        ds = sc.source.get("dataset", "?")
        methods[ds]["single_llm"].append(single_llm_baseline(sc).principle_weights.as_array())
        methods[ds]["harmony"].append(harmony_style_baseline(sc)[-1].principle_weights.as_array())
        methods[ds]["rule_no_gne"].append(rule_collective_weights(sc))
        if sc.scenario_id in mane_by_id:
            methods[ds]["MANE"].append(mane_by_id[sc.scenario_id])

    print(f"{'数据集':<14}{'方法':<12}{'FDBI':>7}{'PCI':>7}{'dist':>7}{'ECS':>7}{'通过率':>7}{'放弃率':>8}")
    print("-" * 70)
    for ds in sorted(methods):
        for method in ("single_llm", "harmony", "rule_no_gne", "MANE"):
            ws = methods[ds].get(method)
            if not ws:
                continue
            s = _summarize(ws, floors)
            print(f"{ds:<14}{method:<12}{s['FDBI']:>7.3f}{s['PCI']:>7.3f}{s['dist']:>7.3f}"
                  f"{s['ECS']:>7.3f}{s['通过率']:>7.0%}{s['放弃率']:>8.3f}")
    # 汇总行（跨数据集）
    print("-" * 70)
    for method in ("single_llm", "harmony", "rule_no_gne", "MANE"):
        ws = [w for ds in methods.values() for w in ds.get(method, [])]
        if not ws:
            continue
        s = _summarize(ws, floors)
        print(f"{'ALL':<14}{method:<12}{s['FDBI']:>7.3f}{s['PCI']:>7.3f}{s['dist']:>7.3f}"
              f"{s['ECS']:>7.3f}{s['通过率']:>7.0%}{s['放弃率']:>8.3f}")


if __name__ == "__main__":
    main()
