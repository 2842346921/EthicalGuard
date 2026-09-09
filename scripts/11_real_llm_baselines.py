"""E2：真实 LLM 基线（解决"基线稻草人"问题，审稿人 A1/T4）。

问题：single_llm/harmony 是硬编码规则（固定权重，从不调 LLM），"MANE 优于基线"
可被平凡解释为"整台机器 vs 一块砖头"。本脚本实现两个**真实 LLM** 对照：

- single_llm_llm：同一 Qwen3-8B 直接决策（**physician 角色**——显式"守护行善"，
  用于回答"角色化 prompt 是否偏置"，见 E2 违反模式深挖）；
- neutral_single_llm：同一 Qwen3-8B **中性**直接决策（无角色/原则偏置）——主基线；
- medagents_style：MedAgents 式多角色 LLM 协作（有协商但**剥离 GNE/仲裁/底线/状态机**）——
  五方角色各自用 LLM 提案（1 轮讨论后各给立场），贝叶斯等权/简单平均出集体向量；
- MANE：完整框架。

对照组逻辑（E2 违反模式深挖）：若 neutral 也违反 J（公正底线）→ 模型内在偏置（真发现）；
若 neutral 守住而 physician 角色违反 → 违反来自角色 prompt（基线偏置），主基线用 neutral。

对比指标（同 29-35 场景、同 Qwen3-8B、同 token 预算）：
  FDBI / 底线违例率（[B,N,A,J]≥[0.15,0.20,0.15,0.20]）/ 轮次 / 满意度 min / 资源可行率 / LLM 调用数

用法：
  python scripts/11_real_llm_baselines.py --config configs/config.yaml \
      --input data_cache/scenarios.jsonl --mode local --limit 35 [--out runs/real_llm_baselines.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ethicalguard.config import Config, default_agent_specs
from ethicalguard.data import load_scenarios_from_jsonl
from ethicalguard.eval import metrics as M
from ethicalguard.llm import make_backend
from ethicalguard.mane.agents import build_agent
from ethicalguard.mane.engine import MANEEngine
from ethicalguard.types import PrincipleVector, Proposal, Scenario, Syllogism
from ethicalguard.utils import set_seed, setup_logging

logger = setup_logging()

# 基线函数已迁移至 src/ethicalguard/eval/baselines/llm_baselines.py（11/22 共用单源）：
from ethicalguard.eval.baselines.llm_baselines import (  # noqa: E402
    FLOORS,
    llm_single,
    medagents_style,
    neutral_single_llm,
    violation_rate as _violation_rate,
)


def _violation_rate(v: np.ndarray) -> float:
    return float((v < FLOORS).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", required=True, help="场景 jsonl（01 产物）")
    ap.add_argument("--mode", default=None, choices=["rule", "api", "local"])
    ap.add_argument("--limit", type=int, default=0, help="场景上限（0=全部）")
    ap.add_argument("--out", default=None, help="结果 jsonl 保存路径")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.mode:
        cfg.llm.mode = args.mode
    if cfg.llm.mode == "rule":
        print("[警告] 当前 mode=rule——真实 LLM 基线必须 local（rule 下基线仍是规则，无意义）。")
    set_seed(args.seed)
    backend = make_backend(cfg.llm)
    specs = cfg.agents or default_agent_specs()
    engine = MANEEngine(cfg)
    # 医师角色（单 LLM 基线的"决策者"）
    physician = build_agent(specs["physician"], backend)

    scenarios = list(load_scenarios_from_jsonl(args.input))
    if args.limit > 0:
        scenarios = scenarios[:args.limit]

    print(f"===== E2 真实 LLM 基线对比（mode={cfg.llm.mode}，场景 {len(scenarios)}）=====")
    print(f"四方法：single_llm_llm(医师角色) / neutral_single_llm(中性) / medagents_style / MANE")
    print(f"{'场景':<22}{'方法':<20}{'FDBI':>7}{'底线违':>7}{'t':>3}{'满意min':>8}{'LLM调用':>8}")
    rows = []
    for sc in scenarios:
        # ① 真·单 LLM（医师角色，行善偏置对照）
        p1 = llm_single(sc, backend, physician)
        v1 = p1.principle_weights.as_array()
        print(f"{sc.scenario_id:<22}{'single_llm_llm':<20}{M.fdbi(v1):>7.3f}{_violation_rate(v1):>7.3f}"
              f"{p1.treatment_level:>3}{'-':>8}{1:>8}")
        rows.append({"scenario": sc.scenario_id, "method": "single_llm_llm",
                     "vector": v1.tolist(), "t": p1.treatment_level, "llm_calls": 1})
        # ①½ 中性单 LLM（无角色/原则偏置）
        p1n = neutral_single_llm(sc, backend)
        v1n = p1n.principle_weights.as_array()
        print(f"{sc.scenario_id:<22}{'neutral_single_llm':<20}{M.fdbi(v1n):>7.3f}{_violation_rate(v1n):>7.3f}"
              f"{p1n.treatment_level:>3}{'-':>8}{1:>8}")
        rows.append({"scenario": sc.scenario_id, "method": "neutral_single_llm",
                     "vector": v1n.tolist(), "t": p1n.treatment_level, "llm_calls": 1})
        # ② MedAgents 式协作（5 次 LLM 调用）
        p2 = medagents_style(sc, backend, specs)
        v2 = p2.principle_weights.as_array()
        sat2 = "-"
        print(f"{sc.scenario_id:<22}{'medagents_style':<20}{M.fdbi(v2):>7.3f}{_violation_rate(v2):>7.3f}"
              f"{p2.treatment_level:>3}{sat2:>8}{5:>8}")
        rows.append({"scenario": sc.scenario_id, "method": "medagents_style",
                     "vector": v2.tolist(), "t": p2.treatment_level, "llm_calls": 5})
        # ③ MANE（完整）
        r = engine.run(sc)
        v3 = r.final_vector.as_array() if r.final_vector is not None else np.full(4, 0.25)
        sat_min = min(r.agent_satisfactions.values()) if r.agent_satisfactions else float("nan")
        print(f"{sc.scenario_id:<22}{'MANE':<20}{M.fdbi(v3):>7.3f}{_violation_rate(v3):>7.3f}"
              f"{r.final_proposal.treatment_level if r.final_proposal else '-':>3}"
              f"{sat_min:>8.3f}{'-':>8}")
        rows.append({"scenario": sc.scenario_id, "method": "MANE",
                     "vector": v3.tolist(), "t": r.final_proposal.treatment_level if r.final_proposal else None,
                     "sat_min": sat_min, "arb": r.arbitration_triggered})

    # 汇总
    print()
    print("===== 汇总（同场景均值）=====")
    from collections import defaultdict
    agg = defaultdict(list)
    for row in rows:
        agg[row["method"]].append(row)
    print(f"{'方法':<16}{'n':>4}{'FDBI':>8}{'底线违':>8}{'违反场景率':>10}")
    for method, rs in agg.items():
        vs = np.array([x["vector"] for x in rs])
        fdbis = [M.fdbi(v) for v in vs]
        viol = [1.0 if (v < FLOORS).any() else 0.0 for v in vs]
        print(f"{method:<16}{len(rs):>4}{np.mean(fdbis):>8.3f}"
              f"{np.mean([_violation_rate(v) for v in vs]):>8.3f}{np.mean(viol):>10.1%}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("saved %d rows -> %s", len(rows), args.out)


if __name__ == "__main__":
    main()
