# -*- coding: utf-8 -*-
"""最终全量分析：协商层 + 韧性 + 消融 + γ。"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, r"E:\信息\论文\医疗诊断\多目标压力\论文\EthicalGuard\src")
from ethicalguard.eval import metrics as M

base = r"E:\信息\论文\医疗诊断\多目标压力\论文\EthicalGuard"
ORD = ["beneficence", "nonmaleficence", "autonomy", "justice"]


def vec(d):
    return np.array([d[k] for k in ORD], dtype=float)


mrs = [json.loads(l) for l in open(base + r"\runs\mane_results.jsonl", encoding="utf-8")]
print("===== 协商层（%d 场景）=====" % len(mrs))
print("FDBI %.4f | PCI %.4f | KKT %d/%d | 轮次 %.2f | 仲裁 %.1f%% | 可行 %.0f%% | 占用 %.1f%%" % (
    np.mean([M.fdbi(vec(r["final_vector"])) for r in mrs]),
    np.mean([M.pci(vec(r["final_vector"])) for r in mrs]),
    sum(1 for r in mrs if (r.get("kkt_residual") or 0) < 1e-2), len(mrs),
    np.mean([r["rounds"] for r in mrs]),
    100 * np.mean([r["arbitration_triggered"] for r in mrs]),
    100 * np.mean([r["resource_feasible"] for r in mrs]),
    100 * np.mean([r["resource_used"] / r["resource_cap"] for r in mrs])))
sats = [v for r in mrs for v in r["agent_satisfactions"].values()]
print("满意度 %.3f (min %.3f)" % (np.mean(sats), min(sats)))
ers = [r["conflict_report"]["ers"] for r in mrs]
print("ERS %.3f (%.2f-%.2f) | 类型 %s | action %s" % (
    np.mean(ers), min(ers), max(ers),
    dict(collections.Counter(r["conflict_report"]["conflict_type"] for r in mrs)),
    dict(collections.Counter(r["conflict_report"]["action"] for r in mrs))))
print("仲裁原因 %s | CAMP %d/%d | S3 %d/%d | 贝叶斯 %d" % (
    dict(collections.Counter(r.get("arbitration_reason") for r in mrs if r.get("arbitration_triggered"))),
    sum(1 for r in mrs if r.get("arbitration_triggered") and "CAMP" in r["trajectory"][-1]["proposals"][-1]["rationale"]),
    sum(1 for r in mrs if r.get("arbitration_triggered")),
    sum(1 for r in mrs if "S3_align" in (r.get("state_trace") or [])), len(mrs),
    sum(r.get("bayesian_updates", 0) for r in mrs)))

print()
reps = [json.loads(l) for l in open(base + r"\runs\resilience.jsonl", encoding="utf-8")]
print("===== 韧性（%d 场景）=====" % len(reps))
print("C %.3f | R_robust %.3f | R_rec %.3f | BSP %.2f | 放弃场景 %d/%d" % (
    np.mean([r["l1_consistency"] for r in reps]),
    np.mean([r["l2_robustness"] for r in reps]),
    np.mean([r["l3_recoverability"] for r in reps]),
    np.mean([r["bsp"] or 0 for r in reps]),
    sum(1 for r in reps if r.get("abandoned")), len(reps)))
for r in reps:
    rec = r.get("l3_recoverability", 0)
    std = r.get("recovery_std")
    rec_s = "%.2f±%.2f" % (rec, std) if std is not None else "%.2f" % rec
    print("  %-12s C=%.3f R=%.3f rec=%s BSP=%s ab=%s" % (
        r["scenario_id"], r["l1_consistency"], r["l2_robustness"], rec_s, r.get("bsp"), r.get("abandoned")))

print()
print("===== 消融（local）=====")
print(open(base + r"\runs\ablations_local.txt", encoding="utf-8").read())
print("===== γ =====")
print(open(base + r"\runs\gamma_sweep.txt", encoding="utf-8").read())
