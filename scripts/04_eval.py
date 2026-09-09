"""04_eval：对 MANE 结果计算 MedEval 指标（FDBI/PCI/KKT/资源/轮次/仲裁率）。

用法：
  python scripts/04_eval.py --config configs/config.yaml [--runs runs/mane_results.jsonl]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ethicalguard.config import Config
from ethicalguard.eval import metrics as M
from ethicalguard.types import NegotiationResult
from ethicalguard.utils import setup_logging

logger = setup_logging()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml", help="统一配置文件（YAML/JSON）")
    ap.add_argument("--runs", default=None, help="MANE 结果 jsonl（默认 config.run.out_dir/mane_results.jsonl）")
    ap.add_argument("--scenarios", default=None, help="场景 jsonl（可选；统计四盒状态去重率）")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    runs = args.runs or os.path.join(cfg.run.out_dir, "mane_results.jsonl")

    results = []
    from ethicalguard.data.loaders import iter_negotiation_records
    for r in iter_negotiation_records(runs):
        results.append(NegotiationResult.model_validate(r))

    fdbis, pcis, kkts, rounds = [], [], [], []
    res_used, res_cap, res_feas = [], [], []
    for r in results:
        v = r.final_vector.as_array() if r.final_vector is not None else np.full(4, 0.25)
        fdbis.append(M.fdbi(v))
        pcis.append(M.pci(v))
        if r.kkt_residual is not None:
            kkts.append(r.kkt_residual)
        rounds.append(r.rounds)
        if r.resource_used is not None:
            res_used.append(r.resource_used)
        if r.resource_cap is not None:
            res_cap.append(r.resource_cap)
        if r.resource_feasible is not None:
            res_feas.append(r.resource_feasible)

    print(f"场景数            : {len(results)}")
    print(f"FDBI  平衡指数均值 : {np.mean(fdbis):.4f}  (越高越平衡)")
    print(f"PCI   冲突强度均值 : {np.mean(pcis):.4f}  (越低越协调)")
    if kkts:
        print(f"KKT   残差均值     : {np.mean(kkts):.4f}  (越低越接近 GNE)")
    print(f"平均协商轮次       : {np.mean(rounds):.2f}")
    print(f"仲裁触发率         : {np.mean([r.arbitration_triggered for r in results]):.2%}")
    # 仲裁原因分布（O3 审计：resource / satisfaction / unconverged / 组合）
    reasons = [r.arbitration_reason for r in results if r.arbitration_triggered and r.arbitration_reason]
    if reasons:
        from collections import Counter
        print(f"仲裁原因分布       : {dict(Counter(reasons))}")
    if res_used and res_cap:
        ratio = np.mean(np.array(res_used) / np.maximum(np.array(res_cap), 1e-9))
        print(f"资源可行率         : {np.mean(res_feas):.2%}" if res_feas else "资源可行率: N/A")
        print(f"平均资源用度/上限  : {np.mean(res_used):.3f} / {np.mean(res_cap):.3f}")
        print(f"平均资源占用率     : {ratio:.1%}  (用度/上限)")

    # 满意度区分度（各方满意度均值/最低值；全 >0.9 说明规则模式"太顺"，LLM 模式会更分化）
    sats = [v for r in results for v in r.agent_satisfactions.values()] if results else []
    if sats:
        print(f"平均满意度         : {np.mean(sats):.3f}  (最低 {np.min(sats):.3f}; 全>0.9 说明冲突被规则基线平滑)")

    # ---- 闭环证据链（"系统而非组合"审计）：每环的输出是否进入下一环 ----
    print("\n===== 闭环证据链（各机制输出→下一环输入的接线率）=====")
    n = max(1, len(results))
    pt = [r.process_trace for r in results if r.process_trace]
    if pt:
        catfish_hist = np.mean([1.0 if p.get("dissent_in_history") else 0.0 for p in pt])
        # 异议被消费判定：位移需显著高于规则本底（贝叶斯可靠性权重演化也会微动集体，
        # 量级≈0.01；规则 compromise 不读提案文本故异议内容不产生显著位移）。
        # 显著位移（>0.05，让步量级 0.1+ 的一半）才计为"异议被听见"——local 模式可见。
        DISSENT_EFFECT_TH = 0.05
        dissent_consumed = np.mean([1.0 if p.get("subsequent_collective_delta", 0.0) > DISSENT_EFFECT_TH else 0.0
                                    for p in pt])
        arb = np.mean([1.0 if p.get("arbitration_triggered") else 0.0 for p in pt])
        fused = np.mean([1.0 if p.get("fusion_applied") else 0.0 for p in pt])
        res_corr = np.mean([1.0 if p.get("resource_corrected") else 0.0 for p in pt])
        kkt_ok = np.mean([1.0 if p.get("kkt_residual") is not None and p["kkt_residual"] < 0.5 else 0.0 for p in pt])
        floors = np.mean([1.0 if p.get("floors_ok") else 0.0 for p in pt])
        print(f"异议进历史率        : {catfish_hist:.1%}  (Catfish 提案写入历史，成为下一轮输入)")
        print(f"异议被消费率        : {dissent_consumed:.1%}  (异议轮后集体位移>{DISSENT_EFFECT_TH}=异议内容被听见；"
              f"规则模式≈0=未消费)")
        print(f"仲裁触发率          : {arb:.1%}  (未收敛/低满意/资源超限 → 仲裁接管)")
        print(f"仲裁融合率          : {fused:.1%}  (裁决与 GNE 均衡融合进终态)")
        print(f"资源修正率          : {res_corr:.1%}  (L1 护栏降级发生；w/o 仲裁消融应为 0)")
        print(f"GNE KKT 达标率      : {kkt_ok:.1%}  (KKT 残差<0.5)")
        print(f"L3 底线守住率       : {floors:.1%}  (终态四原则≥底线)")
        # 异议→集体→收敛的耦合：异议入历史 + 被消费（显著位移）且未触发仲裁
        # = 异议已并入共识（完整机制链，未被仲裁兜底遮蔽）
        coupled = np.mean([1.0 if (p.get("dissent_in_history") and p.get("subsequent_collective_delta", 0.0) > DISSENT_EFFECT_TH
                                   and not p.get("arbitration_triggered")) else 0.0 for p in pt])
        print(f"异议→集体→收敛路径  : {coupled:.1%}  (异议入历史→被消费→并入共识，未触发仲裁)")
        # 失败→修复：触发仲裁的场景中，终态是否守住底线（仲裁修复了"低满意/超限/不收敛"）
        arb_pts = [p for p in pt if p.get("arbitration_triggered")]
        if arb_pts:
            arb_floors = np.mean([1.0 if p.get("floors_ok") else 0.0 for p in arb_pts])
            arb_kkt = np.mean([1.0 if p.get("kkt_residual") is not None and p["kkt_residual"] < 0.5 else 0.0
                               for p in arb_pts])
            print(f"仲裁场景终态底线率  : {arb_floors:.1%}  (仲裁修复后的底线守成——仲裁是'修复环'证据)")
            print(f"仲裁场景 KKT 达标率 : {arb_kkt:.1%}")
    else:
        print("  （结果中无 process_trace 字段——请重跑 02 生成闭环证据）")

    # 冲突识别统计（ERS / 冲突类型 / 建议行动分布）
    ers_list = [r.conflict_report.ers for r in results if r.conflict_report is not None]
    if ers_list:
        from collections import Counter
        types = Counter(r.conflict_report.conflict_type.value for r in results if r.conflict_report is not None)
        actions = Counter(r.conflict_report.action for r in results if r.conflict_report is not None)
        print(f"ERS   风险评分均值 : {np.mean(ers_list):.3f}  (≥0.3 视为需关注，≥0.7 需介入)")
        print(f"冲突类型分布       : {dict(types)}")
        print(f"建议行动分布       : {dict(actions)}")

    # 四盒状态去重率（场景映射质量：同 case 多题应产生不同状态）
    if args.scenarios and os.path.exists(args.scenarios):
        from ethicalguard.types import Scenario
        seen = set()
        total = 0
        with open(args.scenarios, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = Scenario.model_validate_json(line)
                key = (tuple(sorted(s.state.medical.items())),
                       tuple(sorted(s.state.preference.items())),
                       tuple(sorted(s.state.context.items())))
                seen.add(key)
                total += 1
        print(f"场景唯一状态率     : {len(seen)}/{total} = {len(seen) / max(1, total):.1%}  (越低越说明映射坍缩)")


if __name__ == "__main__":
    main()
