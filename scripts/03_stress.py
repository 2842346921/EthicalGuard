"""03_stress：Med-Ethical-Stress 韧性压力测试。

用法：
  python scripts/03_stress.py --config configs/config.yaml \
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
from ethicalguard.resilience import ResilienceEvaluator
from ethicalguard.utils import set_seed, setup_logging

logger = setup_logging()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml", help="统一配置文件（YAML/JSON）")
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default=None, choices=["rule", "api", "local"],
                    help="基座模式（默认取配置 llm.mode）")
    ap.add_argument("--limit", type=int, default=None, help="场景数上限（默认 config.run.limit）")
    ap.add_argument("--out", default=None, help="输出 jsonl 路径（默认 config.run.out_dir/resilience.jsonl）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--intensity-grid", default=None,
                    help="压力强度网格（逗号分隔，如 0.25,0.5,0.75,1.0；默认 config.resilience.intensity_grid；"
                         "设 1.0 即旧版单档）")
    ap.add_argument("--repeat", type=int, default=None,
                    help="基线/恢复重跑次数（默认 config.resilience.repeat；>1 时 R_recover 报 mean±std，"
                         "调用量×repeat，P1-1 口径）")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.llm.mode = args.mode or cfg.llm.mode
    if args.intensity_grid:
        cfg.resilience.intensity_grid = [float(x.strip()) for x in args.intensity_grid.split(",") if x.strip()]
    if args.repeat is not None:
        cfg.resilience.repeat = args.repeat
    seed = args.seed if args.seed is not None else cfg.run.seed
    limit = args.limit if args.limit is not None else cfg.run.limit
    out = args.out or os.path.join(cfg.run.out_dir, "resilience.jsonl")

    set_seed(seed)
    engine = MANEEngine(cfg)
    mode = engine.backend.mode
    evaluator = ResilienceEvaluator(engine.run, cfg.resilience, mode=mode)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        n = 0
        for scenario in load_scenarios_from_jsonl(args.input):
            report = evaluator.evaluate(scenario)
            f.write(report.model_dump_json() + "\n")
            n += 1
            # 两档证据：rule=基线一致性（不作韧性结论）；api/local=完整韧性（含原则放弃记录）
            tag = "基线一致性" if report.baseline_only else "完整韧性"
            # 全局最小放弃临界强度（跨压力类型；None=全档未放弃）
            crits = [s["critical_abandonment"] for s in report.stress_response
                     if s.get("critical_abandonment") is not None]
            tau = min(crits) if crits else None
            rec_str = (f"{report.l3_recoverability:.3f}±{report.recovery_std:.3f}"
                       if report.recovery_std is not None else f"{report.l3_recoverability:.3f}")
            # 闭环环 9：压力下协商质量（个体让步 vs 集体稳定）
            nq = report.negotiation_quality
            q_str = "nq=?"
            if nq:
                b = nq.get("baseline", {})
                s = nq.get("stress", {})
                q_str = (f"nq[{nq.get('verdict','?')}] convB={b.get('concession_mean',0):.3f}→"
                         f"{s.get('concession_mean_mean',0):.3f} "
                         f"volB={b.get('collective_volatility',0):.3f}→{s.get('collective_volatility_mean',0):.3f}")
            logger.info("[%s][%s] R_ethical=%.3f C=%.3f robust=%.3f recover=%s "
                        "BSP=%.2f abandoned=%s tau_abandon=%s verdict=%s %s",
                        scenario.scenario_id, tag, report.overall, report.l1_consistency,
                        report.l2_robustness, rec_str,
                        report.bsp or 0.0, report.abandoned,
                        f"{tau:.2f}" if tau is not None else "-", report.verdict, q_str)
            if n >= limit:
                break
    logger.info("wrote %d reports (mode=%s, intensity_grid=%s) -> %s",
                n, mode, cfg.resilience.intensity_grid, out)


if __name__ == "__main__":
    main()
