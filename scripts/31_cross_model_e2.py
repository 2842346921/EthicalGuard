"""31_cross_model_e2：跨模型复现 E2（回答 R2："0% vs 50-90% 是不是 Qwen3-8B 特有？"）。

对指定 OpenAI 兼容后端（vLLM 起的 Llama-3.1-8B / Mistral-7B 等）在**同 35 场景**上跑
四方法（single_llm_llm / neutral_single_llm / medagents_style / MANE），逐行存 jsonl，
并（可选）与 Qwen3-8B 的 real_llm_baselines.jsonl 对照打印跨模型矩阵：
  模型 × 方法 × {FDBI, 违反场景率, 均维违反}
判定：若 Llama/Mistral 上 MANE 仍 0% 违反、基线仍显著 >0 → 协议增益跨模型稳健；
若某模型基线违反率低（如 Mistral 更稳）→ 如实报告 Δ 变化，但机制化保证仍应成立。

用法（由 run_cross_model.sh 调用，也可单独）：
  python scripts/31_cross_model_e2.py --config configs/config.yaml \
      --input data_cache/scenarios.jsonl \
      --base-url http://127.0.0.1:8011/v1 --model Meta-Llama-3.1-8B-Instruct \
      --model-name llama31 --out runs/cross_model_llama31.jsonl \
      [--qwen-runs runs/real_llm_baselines.jsonl] [--limit 0]
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
from ethicalguard.eval.baselines.llm_baselines import (
    FLOORS, llm_single, medagents_style, neutral_single_llm,
)
from ethicalguard.llm import make_backend
from ethicalguard.mane.agents import build_agent
from ethicalguard.mane.engine import MANEEngine
from ethicalguard.utils import set_seed

METHODS = ["single_llm_llm", "neutral_single_llm", "medagents_style", "MANE"]


def _agg(rows):
    """rows: list of {method, vector, violation?} → 方法级 FDBI/违规场景率/均维违反。"""
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    out = {}
    for m, rs in by.items():
        vs = np.array([r["vector"] for r in rs])
        out[m] = {
            "n": len(rs),
            "fdbi": float(np.mean([M.fdbi(v) for v in vs])),
            "viol_scen": float(np.mean([1.0 if (v < FLOORS).any() else 0.0 for v in vs])),
            "viol_dim": float(np.mean([float((v < FLOORS).mean()) for v in vs])),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", required=True, help="场景 jsonl（与 E2 同 35 场景）")
    ap.add_argument("--base-url", required=True, help="vLLM OpenAI 兼容地址（到 /v1）")
    ap.add_argument("--model", required=True, help="服务上注册的模型名（--served-model-name）")
    ap.add_argument("--model-name", default=None, help="报告标签（默认=model）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--qwen-runs", default=None, help="可选：Qwen3-8B 的 E2 runs（对照）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="读已有 out，跳过 (场景,方法) 已存在的行——只补缺失（如 3 个 MANE 场景）")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.llm.mode = "local"
    cfg.llm.local.base_url = args.base_url
    cfg.llm.local.model = args.model
    set_seed(args.seed)
    backend = make_backend(cfg.llm)
    specs = cfg.agents or default_agent_specs()
    physician = build_agent(specs["physician"], backend)
    engine = MANEEngine(cfg)
    label = args.model_name or args.model
    print(f"[模型] {label} @ {args.base_url}（mode=local）")

    scenarios = list(load_scenarios_from_jsonl(args.input))
    if args.limit > 0:
        scenarios = scenarios[:args.limit]
    print(f"场景 {len(scenarios)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    rows.append(o)
                    done.add((o["scenario"], o["method"]))
        print(f"[resume] 已有 {len(rows)} 行 → 只补缺失的 (场景,方法)")
    n_skip = 0
    for i, sc in enumerate(scenarios):
        sid = sc.scenario_id
        # per-method 容错（2026-09-07）：单个基线解析失败（Mistral syllogism 缺字段等）
        # 只跳过该方法，不中断整个模型跑（此前 31 在 Mistral 5/35 场景处整体崩掉）
        def _try(name: str, fn, method: str):
            nonlocal n_skip
            if (sid, method) in done:      # resume：已有行直接跳过
                return None
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                n_skip += 1
                print(f"[warn] {sid} {name} 失败跳过: {str(e)[:120]}")
                return None
        p1 = _try("single_llm_llm", lambda: llm_single(sc, backend, physician), "single_llm_llm")
        if p1 is not None:
            v1 = p1.principle_weights.as_array()
            rows.append({"scenario": sid, "model": label, "method": "single_llm_llm",
                         "vector": v1.tolist(), "t": p1.treatment_level, "llm_calls": 1})
        p1n = _try("neutral_single_llm", lambda: neutral_single_llm(sc, backend), "neutral_single_llm")
        if p1n is not None:
            v1n = p1n.principle_weights.as_array()
            rows.append({"scenario": sid, "model": label, "method": "neutral_single_llm",
                         "vector": v1n.tolist(), "t": p1n.treatment_level, "llm_calls": 1})
        p2 = _try("medagents_style", lambda: medagents_style(sc, backend, specs), "medagents_style")
        if p2 is not None:
            v2 = p2.principle_weights.as_array()
            rows.append({"scenario": sid, "model": label, "method": "medagents_style",
                         "vector": v2.tolist(), "t": p2.treatment_level, "llm_calls": 5})
        r = _try("MANE", lambda: engine.run(sc), "MANE")
        if r is not None:
            v3 = r.final_vector.as_array() if r.final_vector is not None else np.full(4, 0.25)
            rows.append({"scenario": sid, "model": label, "method": "MANE",
                         "vector": v3.tolist(),
                         "t": (r.final_proposal.treatment_level if r.final_proposal else None),
                         "llm_calls": -1,
                         "sat_min": (min(r.agent_satisfactions.values())
                                     if r.agent_satisfactions else None),
                         "arb": r.arbitration_triggered,
                         "kkt": r.kkt_residual})
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(scenarios)} 场景完成（累计跳过 {n_skip} 条）")
        with open(args.out, "w", encoding="utf-8") as f:  # 每 5 场景落盘一次
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if n_skip:
        print(f"[完成] 共跳过 {n_skip} 条（解析失败，不影响其余数据）")

    # ---- 汇总 ----
    print("\n===== 跨模型汇总：%s =====" % label)
    agg = _agg(rows)
    print(f"{'方法':<18}{'n':>4}{'FDBI':>7}{'违反场景率':>9}{'均维违反':>8}")
    for m in METHODS:
        a = agg.get(m)
        if not a:
            continue
        print(f"{m:<18}{a['n']:>4}{a['fdbi']:>7.3f}{a['viol_scen']:>9.1%}{a['viol_dim']:>8.1%}")

    if args.qwen_runs and os.path.exists(args.qwen_runs):
        qw = _agg([json.loads(l) for l in open(args.qwen_runs, encoding="utf-8") if l.strip()])
        print("\n===== 跨模型对照（Qwen3-8B vs %s）：违反场景率 =====" % label)
        print(f"{'方法':<18}{'Qwen3-8B':>10}{label[:12]:>12}")
        for m in METHODS:
            q, x = qw.get(m), agg.get(m)
            if not q or not x:
                continue
            print(f"{m:<18}{q['viol_scen']:>10.1%}{x['viol_scen']:>12.1%}")
        print(f"判定：若 {label} 上 MANE 违反场景率仍 0% 而基线明显 >0 → '协商消除底线违反'跨模型稳健；")
        print("若某基线在此模型上违反率也低 → 如实报告该模型更稳，但机制化保证（E1b 激活曲线）不变。")

    print(f"[已保存] {args.out}")


if __name__ == "__main__":
    main()
