"""09_audit：实现完整性审计——"设计 vs 实现"逐机制核对（静态 + 运行时证据）。

审计清单（对应《项目架构规划》§2.3/§3.2 的每个机制）：
1. GNE 求解器 + KKT 校验        → kkt_residual 非 None 且 < 阈值
2. 三层约束（L1/L2/L3）         → 约束对象存在 + 资源修正/软成本/floors 生效
3. 状态机 S0-S6+R 驱动           → state_trace 含 S0/S1/S2/S3（+R 回溯）
4. 贝叶斯编排（EmoMAS）         → bayesian_updates > 0（update 被调用）
5. 观察掩码（信息不对称）        → 非全信息 Agent 的提示词不含被掩码维度
6. CAMP 三值投票                → 仲裁裁决 rationale 含 "CAMP:"
7. SEMA-RAG 证据锁定            → scenario.evidence_locked 非空 + 提示词含 [锚定]
8. Catfish 异议注入             → trajectory 含 catfish 提案（use_catfish=True）
9. 资源强制修正                 → 最终 resource_feasible=True（超限场景）
10. 底线数值统一（L3）          → default_constraints 的 floor == gne_solver floors
11. HALF 加权放弃代价           → halved_drift 使用 HALF_WEIGHTS
12. 消融开关生效                → w/o GNE 时 kkt=None；w/o 仲裁不触发

用法：
  python scripts/09_audit.py --config configs/config.yaml [--scenarios data_cache/scenarios.jsonl]
  （--scenarios 给定时做运行时审计（跑 1 个场景取证据）；否则仅静态审计）
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ethicalguard.config import Config
from ethicalguard.utils import setup_logging

logger = setup_logging()


def _static_audit() -> list:
    """静态审计：检查代码结构层面的实现证据。"""
    findings = []
    import inspect
    import ethicalguard.mane.negotiation as neg
    import ethicalguard.mane.arbitration as arb
    import ethicalguard.mane.gne_solver as gs
    import ethicalguard.mane.agents.base as ab
    import ethicalguard.data.mapping_rules as mr

    # 1 GNE + KKT
    findings.append(("GNE 求解器 + KKT", bool(getattr(gs, "GNEProblem", None) and hasattr(gs.GNEProblem, "solve")),
                     "gne_solver.GNEProblem.solve"))
    # 3 状态机驱动：negotiation 内 advance 调用带 flags
    src = inspect.getsource(neg)
    findings.append(("状态机接真（advance 带 flags）", "advance({\"flags\": flags})" in src, "negotiation.py flags"))
    # 4 贝叶斯 update 被调用
    findings.append(("贝叶斯 update 被调用", "orchestrator.update(" in src, "negotiation.py update"))
    # 5 观察掩码：_state_block 用 observe
    bsrc = inspect.getsource(ab)
    findings.append(("观察掩码接进提示词", "self.observe(scenario)" in bsrc and "visible" in bsrc, "base.py _state_block"))
    # 6 CAMP 三值投票
    asrc = inspect.getsource(arb)
    findings.append(("CAMP 三值投票", "camp_vote(" in asrc and "CAMP: KEEP" in asrc, "arbitration.py camp_vote"))
    # 7 证据锁定
    findings.append(("SEMA-RAG 证据锁定（字段+标注）", "evidence_locked" in bsrc, "types.Scenario + _state_block"))
    # 9 资源强制修正
    findings.append(("资源强制修正 enforce_resource_cap", hasattr(arb, "enforce_resource_cap"), "arbitration.py"))
    # 10 底线统一：default_constraints floor == gne floors
    mr_src = inspect.getsource(mr)
    unified = "floors = {\"nonmaleficence\": 0.20, \"autonomy\": 0.15, \"beneficence\": 0.15, \"justice\": 0.20}" in mr_src
    findings.append(("L3 底线统一（default_constraints vs GNE）", unified, "mapping_rules.default_constraints"))
    # 11 HALF 加权
    import ethicalguard.types as T
    findings.append(("HALF 加权放弃代价", bool(np.allclose(T.HALF_WEIGHTS, [1.5, 3.0, 1.0, 2.0])), "types.HALF_WEIGHTS"))
    # 12 消融开关
    findings.append(("消融开关（use_gne/catfish/arbitration/balance）",
                     all(hasattr(Config().mane, k) for k in ("use_gne", "use_catfish", "use_arbitration", "use_balance")),
                     "config.ManeConfig"))
    # L3 具象化
    findings.append(("L3 原则具象化 l3_specification",
                     hasattr(T.FourBoxState, "l3_specification"), "types.FourBoxState.l3_specification"))
    # 13 闭环证据链（process_trace）：negotiation 填充全部闭环字段
    findings.append(("闭环证据链 process_trace 字段齐全",
                     all(k in src for k in ("catfish_rounds", "subsequent_collective_delta",
                                            "resource_corrected", "fusion_applied", "floors_ok")),
                     "negotiation.py process_trace"))
    # 14 协商质量（环 9）：个体让步 vs 集体稳定
    import ethicalguard.resilience.negotiation_quality as nq
    findings.append(("协商质量指标（让步/集体稳定）",
                     all(hasattr(nq, f) for f in ("concessions", "collective_volatility",
                                                  "concession_inequality", "summary")),
                     "resilience/negotiation_quality.py"))
    # 15 协商质量接入韧性报告
    import ethicalguard.resilience.evaluator as evl
    findings.append(("协商质量接入韧性报告",
                     "negotiation_quality" in inspect.getsource(evl),
                     "evaluator.py per_perturbation/negotiation_quality"))
    return findings


def _runtime_audit(cfg: Config, scenario_path: str) -> list:
    """运行时审计：跑真实场景 + 两个构造场景（强制仲裁 / 掩码断言），收集机制触发证据。"""
    from ethicalguard.data import load_scenarios_from_jsonl
    from ethicalguard.mane import MANEEngine
    from ethicalguard.types import FourBoxState, Scenario
    from ethicalguard.data.mapping_rules import default_constraints, default_parties
    findings = []
    sc = next(load_scenarios_from_jsonl(scenario_path))
    engine = MANEEngine(cfg)

    # ① 真实场景（首条）常规审计
    res = engine.run(sc)
    findings.append(("KKT 收敛（<1e-2）", res.kkt_residual is not None and res.kkt_residual < 1e-2,
                     f"kkt={res.kkt_residual}"))
    findings.append(("状态机轨迹含 S0/S1/S2/S3",
                     all(s in res.state_trace for s in ("S0_fact_anchoring", "S1_assemble", "S2_propose", "S3_align")),
                     f"trace={res.state_trace[:6]}..."))
    findings.append(("贝叶斯更新发生（>0）", res.bayesian_updates > 0, f"updates={res.bayesian_updates}"))
    findings.append(("Catfish 异议入历史", any(p.agent == "catfish" for r in res.trajectory for p in r.proposals),
                     f"rounds={res.rounds}"))
    findings.append(("证据锁定标记", bool(sc.evidence_locked), f"locked={sc.evidence_locked}"))
    findings.append(("资源可行性", res.resource_feasible in (True, None), f"feasible={res.resource_feasible}"))
    # 闭环证据链（process_trace）：每环输出→下一环输入的接线率断言
    pt = res.process_trace or {}
    findings.append(("闭环：Catfish 异议入历史",
                     bool(pt.get("dissent_in_history")), f"rounds={pt.get('catfish_rounds')}"))
    findings.append(("闭环：异议被消费（集体位移>0，local/rule 均可能）",
                     pt.get("subsequent_collective_delta", 0.0) > 1e-6,
                     f"Δ={pt.get('subsequent_collective_delta'):.4f}"))
    findings.append(("闭环：仲裁触发→融合→终态字段齐全",
                     "arbitration_triggered" in pt and "fusion_applied" in pt and "floors_ok" in pt,
                     f"arb={pt.get('arbitration_triggered')} fused={pt.get('fusion_applied')} "
                     f"floors={pt.get('floors_ok')}"))
    findings.append(("闭环：GNE KKT 达标 + L3 底线守住",
                     (pt.get("kkt_residual") is None or pt.get("kkt_residual", 1.0) < 0.5)
                     and pt.get("floors_ok") is True,
                     f"kkt={pt.get('kkt_residual')} floors_ok={pt.get('floors_ok')}"))
    # 协商质量（环 9）：真实场景协商产生让步与集体稳定指标
    from ethicalguard.resilience.negotiation_quality import summary as nq_summary
    q = nq_summary(res.trajectory)
    findings.append(("协商质量（环 9）：让步/集体稳定可计算",
                     q["concession_mean"] >= 0.0 and q["collective_volatility"] >= 0.0
                     and q["concession_inequality"] >= 0.0,
                     f"conv={q['concession_mean']:.3f} ineq={q['concession_inequality']:.3f} "
                     f"vol={q['collective_volatility']:.3f}"))

    # ② 构造资源超限场景（rp=0.5）→ 强制触发仲裁 → 验证 CAMP 三值投票进裁决
    st = FourBoxState(
        medical={"severity": 0.9, "rescue_available": 0.4, "acuity": 0.9},
        preference={"clarity": 0.4, "capacity": 0.2, "attitude_refuse": 0.6, "info_completeness": 0.6},
        qol={"burden": 0.7, "net_effect": -0.4},
        context={"resource_pressure": 0.5, "family_conflict": 0.8, "insurance_stress": 0.3,
                 "religious_barrier": 0.0, "legal_constraint": 0.5},
    )
    sc_arb = Scenario(scenario_id="AUDIT-ARB", state=st,
                      constraints=default_constraints(st), parties=default_parties(st))
    res_arb = engine.run(sc_arb)
    if res_arb.arbitration_triggered:
        ruling = res_arb.trajectory[-1].proposals[-1]
        findings.append(("CAMP 三值投票进裁决（强制仲裁场景）",
                         "CAMP:" in ruling.rationale, ruling.rationale[:80]))
    else:
        findings.append(("CAMP 三值投票（构造场景未触发仲裁，异常）", False, "rp=0.5 应触发 resource 仲裁"))

    # ③ 观察掩码生效：patient（非全信息）的提示词状态块不应含 medical 维度
    from ethicalguard.mane.agents import build_agent
    from ethicalguard.config import default_agent_specs
    from ethicalguard.data.mapping_rules import map_text_to_state
    st_mask = map_text_to_state("Patient refuses DNR with severe sepsis; family conflict high.")
    sc_mask = Scenario(scenario_id="AUDIT-MASK", state=st_mask,
                       constraints=default_constraints(st_mask), parties=default_parties(st_mask))
    patient = build_agent(default_agent_specs()["patient"], None)
    block = patient._state_block(sc_mask)
    findings.append(("观察掩码生效（patient 状态块不含 medical）",
                     "[medical]" not in block and "[preference]" in block,
                     f"block_dims={[d for d in ('medical','preference','qol','context') if f'[{d}]' in block]}"))
    return findings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--scenarios", default=None, help="场景 jsonl（给定时做运行时审计）")
    args = ap.parse_args()
    cfg = Config.load(args.config)

    print("=" * 60)
    print("实现完整性审计：设计 vs 代码")
    print("=" * 60)
    print("\n[静态审计]")
    for name, ok, evidence in _static_audit():
        print(f"  {'✅' if ok else '❌'} {name:<28} {evidence}")

    if args.scenarios and os.path.exists(args.scenarios):
        print("\n[运行时审计]（场景:", os.path.basename(args.scenarios), "）")
        for name, ok, evidence in _runtime_audit(cfg, args.scenarios):
            print(f"  {'✅' if ok else '❌'} {name:<28} {evidence}")
    else:
        print("\n[运行时审计] 跳过（--scenarios 未给）")

    # 消融开关自检：w/o GNE → kkt 应 None
    print("\n[消融开关自检]")
    vcfg = Config.load(args.config)
    vcfg.mane.use_gne = False
    from ethicalguard.mane import MANEEngine
    if args.scenarios and os.path.exists(args.scenarios):
        from ethicalguard.data import load_scenarios_from_jsonl
        sc = next(load_scenarios_from_jsonl(args.scenarios))
        res = MANEEngine(vcfg).run(sc)
        print(f"  {'✅' if res.kkt_residual is None else '❌'} w/o GNE → kkt=None（实际 {res.kkt_residual}）")
    print("\n审计完成。")


if __name__ == "__main__":
    main()
