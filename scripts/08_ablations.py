"""08_ablations：消融实验（§5.3 机制必要性）——按 5 个数据集分组输出。

变体（config.mane 开关）：
- MANE         : 全开（use_gne/use_catfish/use_arbitration/use_balance = True）
- w/o GNE      : use_gne=False → 朴素贝叶斯集体作终态（KKT=None）
- w/o Catfish  : use_catfish=False → 无异议注入（静默共识退化）
- w/o 仲裁      : use_arbitration=False → 无伦理委员会裁决
- w/o 平衡      : use_balance=False → GNE 平衡正则 γ·L_balance 关闭

指标（按数据集）：FDBI / PCI / KKT 满足率 / 平均轮次 / 仲裁率 / 满意度最低。
预期退化：w/o GNE → KKT 无/伪均衡；w/o Catfish → 过早收敛（轮次↓、伪共识）；
         w/o 仲裁 → 资源可行率↓；w/o 平衡 → FDBI 回落。

用法：
  python scripts/08_ablations.py --config configs/config.yaml --input data_cache/scenarios.jsonl [--mode rule]
  （--mode rule 本地秒级；--mode local 需 vLLM，服务器跑全量）
"""
from __future__ import annotations

import argparse
import json
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
IDEAL = np.array([0.25, 0.25, 0.25, 0.25])

VARIANTS = {
    "MANE":        {"use_gne": True,  "use_catfish": True,  "use_arbitration": True,  "use_balance": True},
    "w/o GNE":     {"use_gne": False, "use_catfish": True,  "use_arbitration": True,  "use_balance": True},
    "w/o Catfish": {"use_gne": True,  "use_catfish": False, "use_arbitration": True,  "use_balance": True},
    "w/o 仲裁":      {"use_gne": True,  "use_catfish": True,  "use_arbitration": False, "use_balance": True},
    "w/o 平衡":      {"use_gne": True,  "use_catfish": True,  "use_arbitration": True,  "use_balance": False},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", required=True, help="场景 jsonl（01 产物）")
    ap.add_argument("--mode", default="rule", choices=["rule", "api", "local"])
    ap.add_argument("--variants", default=None, help="逗号分隔变体（默认全部）")
    ap.add_argument("--limit", type=int, default=0, help="每变体场景上限（0=全部）")
    ap.add_argument("--gamma-sweep", action="store_true",
                    help="γ 敏感性分析：扫描 balance_penalty（L_balance 权重）0→1.0，"
                         "补 w/o 平衡消融的弱差异论证（可解释参数敏感性，设计 §2.9）")
    ap.add_argument("--gamma-grid", default="0.0,0.2,0.4,0.6,0.8,1.0", help="γ 网格（逗号分隔）")
    ap.add_argument("--out", default=None, help="结果保存路径（txt，便于论文表格/追溯）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.llm.mode = args.mode
    set_seed(args.seed)
    floors = np.asarray(cfg.resilience.floors, dtype=float)
    variants = [v.strip() for v in args.variants.split(",")] if args.variants else list(VARIANTS)

    # 结果收集（--out 时写文件，便于论文表）
    _lines: list = []

    def _emit(line: str) -> None:
        _lines.append(line)
        print(line)

    scenarios = list(load_scenarios_from_jsonl(args.input))
    if args.limit > 0:
        scenarios = scenarios[:args.limit]

    # γ 敏感性分析（§2.9 可解释参数）：扫描 balance_penalty，展示 L_balance 的边际作用
    if args.gamma_sweep:
        gammas = [float(x.strip()) for x in args.gamma_grid.split(",") if x.strip()]
        _emit(f"===== γ 敏感性（balance_penalty 扫描，mode={args.mode}，场景 {len(scenarios)}）=====")
        _emit(f"{'γ':>6}{'FDBI':>8}{'PCI':>8}{'底线违':>8}{'满意min':>9}")
        for g in gammas:
            vcfg = Config.load(args.config)
            vcfg.llm.mode = args.mode
            vcfg.gne.balance_penalty = g
            engine = MANEEngine(vcfg)
            fdbis, pcis, viol, satmin = [], [], [], []
            for sc in scenarios:
                res = engine.run(sc)
                v = res.final_vector.as_array() if res.final_vector is not None else np.full(4, 0.25)
                fdbis.append(M.fdbi(v)); pcis.append(M.pci(v))
                viol.append(float((v < floors).mean()))
                if res.agent_satisfactions:
                    satmin.append(min(res.agent_satisfactions.values()))
            _emit(f"{g:>6.2f}{np.mean(fdbis):>8.3f}{np.mean(pcis):>8.3f}"
                  f"{np.mean(viol):>8.3f}{np.mean(satmin) if satmin else float('nan'):>9.3f}")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write("\n".join(_lines) + "\n")
            print(f"[已保存] {args.out}")
        return

    # 按数据集分组
    from collections import defaultdict
    by_ds = defaultdict(list)
    for sc in scenarios:
        by_ds[sc.source.get("dataset", "?")].append(sc)
    logger.info("场景 %d 个，数据集 %s", len(scenarios), dict((k, len(v)) for k, v in by_ds.items()))
    if args.mode == "rule":
        _emit("[提示] rule 模式可证：w/o GNE（求解器跳过）/ w/o 仲裁（资源修正路径，KKT/可行率退化）/ "
              "w/o 平衡（目标函数参数）。"
              "w/o Catfish 在 rule 模式零差异——规则 Agent 的 compromise 只读上一轮集体向量、"
              "不读提案文本，鲶鱼异议仅经'历史文本 → LLM 提案'路径生效，必须 --mode local 复验。")

    _emit(f"{'数据集':<14}{'变体':<12}{'FDBI':>7}{'PCI':>7}{'ECS':>7}{'通过率':>7}{'KKT率':>7}{'轮次':>6}{'仲裁率':>7}{'可行率':>7}{'底线违':>7}{'满意min':>8}")
    _emit("-" * 92)
    for ds in sorted(by_ds):
        scs = by_ds[ds]
        for name in variants:
            # 构造变体引擎
            vcfg = Config.load(args.config)
            vcfg.llm.mode = args.mode
            for k, val in VARIANTS[name].items():
                setattr(vcfg.mane, k, val)
            engine = MANEEngine(vcfg)
            fdbis, pcis, kkts, rounds, arb, feas = [], [], [], [], [], []
            floor_viol, sat_min = [], []
            vecs, satmins, feasibles, kktvals = [], [], [], []
            for sc in scs:
                res = engine.run(sc)
                v = res.final_vector.as_array() if res.final_vector is not None else np.full(4, 0.25)
                fdbis.append(M.fdbi(v))
                pcis.append(M.pci(v))
                if res.kkt_residual is not None:
                    kkts.append(res.kkt_residual < 1e-2)
                rounds.append(res.rounds)
                arb.append(res.arbitration_triggered)
                if res.resource_feasible is not None:
                    feas.append(res.resource_feasible)
                # 底线违反率（final 权重跌破 L3 floors 的维度占比）+ 满意度最低
                floor_viol.append(float((v < floors).mean()))
                if res.agent_satisfactions:
                    sat_min.append(min(res.agent_satisfactions.values()))
                # ECS 场景级输入（一票否决需要逐场景 kkt/feasible/sat）
                vecs.append(v)
                satmins.append(min(res.agent_satisfactions.values()) if res.agent_satisfactions else None)
                feasibles.append(res.resource_feasible)
                kktvals.append(res.kkt_residual)
            ecs = M.ethical_composite(vecs, satmins, floors, feasibles, kktvals,
                                      kkt_gate=(name != "w/o GNE"))
            print(f"{ds:<14}{name:<12}{np.mean(fdbis):>7.3f}{np.mean(pcis):>7.3f}"
                  f"{ecs['mean']:>7.3f}{ecs['pass_rate']:>7.0%}"
                  f"{np.mean(kkts) if kkts else float('nan'):>7.2%}"
                  f"{np.mean(rounds):>6.2f}{np.mean(arb):>7.2%}"
                  f"{np.mean(feas) if feas else float('nan'):>7.2%}"
                  f"{np.mean(floor_viol):>7.3f}"
                  f"{np.mean(sat_min) if sat_min else float('nan'):>8.3f}")
    _emit("-" * 92)
    # 汇总行
    for name in variants:
        vcfg = Config.load(args.config)
        vcfg.llm.mode = args.mode
        for k, val in VARIANTS[name].items():
            setattr(vcfg.mane, k, val)
        engine = MANEEngine(vcfg)
        fdbis, pcis, kkts, rounds, arb, feas = [], [], [], [], [], []
        floor_viol, sat_min = [], []
        vecs, satmins, feasibles, kktvals = [], [], [], []
        for sc in scenarios:
            res = engine.run(sc)
            v = res.final_vector.as_array() if res.final_vector is not None else np.full(4, 0.25)
            fdbis.append(M.fdbi(v)); pcis.append(M.pci(v))
            if res.kkt_residual is not None:
                kkts.append(res.kkt_residual < 1e-2)
            rounds.append(res.rounds); arb.append(res.arbitration_triggered)
            if res.resource_feasible is not None:
                feas.append(res.resource_feasible)
            floor_viol.append(float((v < floors).mean()))
            if res.agent_satisfactions:
                sat_min.append(min(res.agent_satisfactions.values()))
            vecs.append(v)
            satmins.append(min(res.agent_satisfactions.values()) if res.agent_satisfactions else None)
            feasibles.append(res.resource_feasible)
            kktvals.append(res.kkt_residual)
        ecs = M.ethical_composite(vecs, satmins, floors, feasibles, kktvals,
                                  kkt_gate=(name != "w/o GNE"))
        _emit(f"{'ALL':<14}{name:<12}{np.mean(fdbis):>7.3f}{np.mean(pcis):>7.3f}"
              f"{ecs['mean']:>7.3f}{ecs['pass_rate']:>7.0%}"
              f"{np.mean(kkts) if kkts else float('nan'):>7.2%}"
              f"{np.mean(rounds):>6.2f}{np.mean(arb):>7.2%}"
              f"{np.mean(feas) if feas else float('nan'):>7.2%}"
              f"{np.mean(floor_viol):>7.3f}"
              f"{np.mean(sat_min) if sat_min else float('nan'):>8.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(_lines) + "\n")
        print(f"[已保存] {args.out}")


if __name__ == "__main__":
    main()
