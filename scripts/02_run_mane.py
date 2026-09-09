"""02_run_mane：对标准场景运行 MANE 协商。

用法：
  python scripts/02_run_mane.py --config configs/config.yaml \
      --input data_cache/scenarios.jsonl --limit 10
说明：--mode / --limit / --seed / --out 未给时取自 config.yaml。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethicalguard.config import Config
from ethicalguard.data import load_scenarios_from_jsonl
from ethicalguard.mane import MANEEngine
from ethicalguard.utils import set_seed, setup_logging

logger = setup_logging()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml", help="统一配置文件（YAML/JSON）")
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default=None, choices=["rule", "api", "local"],
                    help="基座模式（默认取配置 llm.mode）")
    ap.add_argument("--limit", type=int, default=None, help="场景数上限（默认 config.run.limit）")
    ap.add_argument("--out", default=None, help="输出 jsonl 路径（默认 config.run.out_dir/mane_results.jsonl）")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.llm.mode = args.mode or cfg.llm.mode
    seed = args.seed if args.seed is not None else cfg.run.seed
    limit = args.limit if args.limit is not None else cfg.run.limit
    out = args.out or os.path.join(cfg.run.out_dir, "mane_results.jsonl")

    set_seed(seed)
    engine = MANEEngine(cfg)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        n = 0
        for scenario in load_scenarios_from_jsonl(args.input):
            result = engine.run(scenario)
            f.write(result.model_dump_json() + "\n")
            n += 1
            # 仲裁审计（O3）：触发原因 + 裁决摘要
            ruling_note = ""
            if result.arbitration_triggered:
                ruling = result.trajectory[-1].proposals[-1]
                ruling_note = f" | 仲裁[{result.arbitration_reason}] 裁决t={ruling.treatment_level}: {ruling.rationale[:60]}"
            # 闭环证据链（process_trace）：catfish 异议→历史→集体→仲裁→融合→资源→KKT→底线
            pt = result.process_trace or {}
            chain = (f"catfish={len(pt.get('catfish_rounds', []))}轮"
                     f"Δ集体={pt.get('subsequent_collective_delta', 0.0):.3f}"
                     f"仲裁={'Y' if pt.get('arbitration_triggered') else 'N'}"
                     f"融合={'Y' if pt.get('fusion_applied') else 'N'}"
                     f"资源修正={'Y' if pt.get('resource_corrected') else 'N'}"
                     f"底线={'OK' if pt.get('floors_ok') else 'V!'}")
            logger.info("[%s] ERS=%.2f(%s) rounds=%d converged=%s kkt=%s resource=%s/%s feas=%s%s | 闭环: %s",
                        scenario.scenario_id,
                        result.conflict_report.ers if result.conflict_report else -1.0,
                        result.conflict_report.conflict_type.value if result.conflict_report else "none",
                        result.rounds, result.converged,
                        f"{result.kkt_residual:.4f}" if result.kkt_residual is not None else "None",
                        f"{result.resource_used:.2f}" if result.resource_used is not None else "None",
                        f"{result.resource_cap:.2f}" if result.resource_cap is not None else "None",
                        result.resource_feasible, ruling_note, chain)
            if n >= limit:
                break
    logger.info("wrote %d results -> %s", n, out)


if __name__ == "__main__":
    main()
