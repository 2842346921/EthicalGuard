"""23_pqa_official_contrast：与 PrinciplismQA 的"同尺子对比 + 定位声明"（离线，读 22 产物）。

输出（runs/pqa_official_contrast.txt）：
  ① 四方法在 **PrinciplismQA 官方 keypoint 0/0.5/1.0 协议**下的内容分均值（pqa_official）
     + rubric/judge/cov 对照列——这是"用他们的尺子量我们"，尺度与论文同源；
  ② MANE vs 各基线的官方协议差（配对，同题）——"协议升级带来内容增益"的直接证据；
  ③ 定位声明（可直接进论文 related work/讨论）：
     - PrinciplismQA 的论点是**诊断性**的：直接问答式 LLM 知识好、开放伦理推理差
       （"knowing ≠ doing"，部署前须测伦理）；
     - 我们不与论文中 GPT-4 级模型比大小（不同基座、不同题子集），不构成反驳；
     - 我们做的是**处方性延伸**：把他们的评测协议原样用于"协议消融"
       （同一 Qwen3-8B：直答 → 多角色协作 → 协商+GNE+底线），问"缺口有多少来自交互协议"。
  ④ 同 22 口径的底线违反率对照（texts 产物）——保证层证据并排。

用法：
  python scripts/23_pqa_official_contrast.py \
      --texts runs/cross_method_texts.jsonl --scores runs/cross_method_scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default="runs/cross_method_texts.jsonl")
    ap.add_argument("--scores", default="runs/cross_method_scores.jsonl")
    ap.add_argument("--out", default="runs/pqa_official_contrast.txt")
    args = ap.parse_args()

    texts, scores = {}, {}
    for path, box in ((args.texts, texts), (args.scores, scores)):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    box[(o["scenario_id"], o["method"])] = o

    methods = ["single_llm_llm", "neutral_single_llm", "medagents_style", "MANE"]
    methods = [m for m in methods if any(k[1] == m for k in scores)]
    floors = np.array([0.15, 0.20, 0.15, 0.20])
    viol = {}          # 方法 -> 每场景"跌破维度占比"均值（22 texts 的 violation 字段同口径）
    viol_scen = {}     # 方法 -> 违反场景率（(v<floors).any()，与 E2/18 McNemar 口径一致）
    for (sid, m), o in texts.items():
        viol.setdefault(m, []).append(o.get("violation", 0.0))
        v = np.asarray(o.get("vector") or [0.25] * 4, dtype=float)
        viol_scen.setdefault(m, []).append(1.0 if (v < floors).any() else 0.0)
    metrics = ["pqa_official", "rubric", "judge", "cov"]
    out_lines = []
    def emit(s=""):
        out_lines.append(s)
        print(s)

    emit("===== ① 四方法内容分（PrinciplismQA 官方同尺子；同题同 Qwen3-8B）=====")
    emit(f"{'方法':<18}{'n':>4}" + "".join(f"{name:>10}" for name in metrics)
         + f"{'违反场景率':>10}{'均维度违反':>10}")
    means = {}
    for m in methods:
        rows = [scores[(sid, mm)] for (sid, mm), v in scores.items() if mm == m]
        n = len(rows)
        if n == 0:
            continue
        vals = {}
        for name in metrics:
            xs = [r.get(name) for r in rows if isinstance(r.get(name), (int, float))]
            vals[name] = float(np.mean(xs)) if xs else float("nan")
        means[m] = vals
        emit(f"{m:<18}{n:>4}" + "".join(f"{vals[name]:>10.3f}" for name in metrics)
             + f"{np.mean(viol_scen.get(m, [0.0])):>10.1%}"
             + f"{np.mean(viol.get(m, [0.0])):>10.1%}")

    emit("")
    emit("===== ② MANE vs 基线（官方协议 pqa_official，配对同题）=====")
    emit(f"{'基线':<18}{'n':>4}{'MANE':>8}{'基线':>8}{'Δ':>8}")
    for base in ["single_llm_llm", "neutral_single_llm", "medagents_style"]:
        if base not in means:
            continue
        pairs = []
        for (sid, m), o in scores.items():
            if m == base and (sid, "MANE") in scores:
                a = scores[(sid, "MANE")].get("pqa_official")
                b = o.get("pqa_official")
                if isinstance(a, float) and isinstance(b, float):
                    pairs.append((a, b))
        if len(pairs) < 5:
            continue
        av = np.mean([p[0] for p in pairs])
        bv = np.mean([p[1] for p in pairs])
        emit(f"{base:<18}{len(pairs):>4}{av:>8.3f}{bv:>8.3f}{av - bv:>+8.3f}")

    emit("")
    emit("===== ③ 定位声明（可进 related work / 讨论）=====")
    emit("- PrinciplismQA（Findings of ACL 2026）的论点是诊断性的：直接问答式 LLM 在知识 MCQ 上")
    emit("  得分高、但在开放伦理推理上与专家对齐不足（'knowing ≠ doing'）→ 部署前必须测伦理。")
    emit("- 本文不与论文中的 GPT-4 级模型比数字：不同基座、不同题子集，任何大小比较都无意义且不构成反驳。")
    emit("- 本文做的是处方性延伸：把 PrinciplismQA 的官方 keypoint 评分协议（0/0.5/1.0）原样用于")
    emit("  '协议消融'——同一 Qwen3-8B 基座上对比 直答 → 多角色协作 → 协商+GNE+底线。")
    # 口径统一：保证层数字从数据算（单 LLM 场景违反率 / 均维度违反），不再写死历史数字
    _m_scen = np.mean(viol_scen.get("MANE", [0.0]))
    _s_scen = np.mean(viol_scen.get("single_llm_llm", [0.0]))
    _s_dim = np.mean(viol.get("single_llm_llm", [0.0]))
    emit(f"- 若②中 MANE 官方协议分显著高于单 LLM → 他们诊断的'缺口'部分来自交互协议而非纯模型缺陷；")
    emit(f"  若持平 → 协商增益集中在保证层（本 runs 口径：跌破底线 MANE {_m_scen:.0%} vs "
         f"单 LLM 场景违反率 {_s_scen:.0%} / 均维度违反 {_s_dim:.0%}），内容层由外部锚继续检验。")
    emit("  两种结果都与其论点兼容：我们不是证明他们错，而是把他们的评测工具用于系统层改进。")

    emit("")
    emit("===== ④ 保证层并排（texts 产物：场景违反率 与 均维度违反 双口径）=====")
    for m in methods:
        emit(f"  {m:<18} 违反场景率 = {np.mean(viol_scen.get(m, [0.0])):.1%}"
             f"  |  均维度违反 = {np.mean(viol.get(m, [0.0])):.1%}"
             f"  (n={len(viol_scen.get(m, []))})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"\n[已保存] {args.out}")


if __name__ == "__main__":
    main()
