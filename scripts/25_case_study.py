"""25_case_study：协商"发力/失力"案例挖潜（离线，零 LLM）——论文 case study 素材。

自动扫描 30 题 × 4 方法的官方内容分，挑出：
  - MANE 优势案例：pqa_official 高于**全部**基线且差 ≥0.15 的场景（协商发力）
  - MANE 劣势案例：pqa_official 低于最佳基线 ≥0.15 的场景（协商失力/测量错配）
另输出 --cases 指定的场景（默认 PQ-474-931 / PQ-369-758）。
每个案例给：题目（截断）/专家 keypoints/四方法官方分与 cov/各方法 rationale（截断）/
MANE 终态向量/底线违反——供你写"为什么发力/失力"的分析段落。

用法：
  python scripts/25_case_study.py \
      --input data_cache/scenarios_pqa30.jsonl \
      --texts runs/cross_method_texts.jsonl \
      --scores runs/cross_method_scores.jsonl \
      [--cases PQ-474-931,PQ-369-758] [--out runs/case_study.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

METHODS = ["single_llm_llm", "neutral_single_llm", "medagents_style", "MANE"]
BASE_NAMES = {"single_llm_llm": "单LLM(医师)", "neutral_single_llm": "单LLM(中性)",
              "medagents_style": "MedAgents", "MANE": "MANE"}
FLOORS = np.array([0.15, 0.20, 0.15, 0.20])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--texts", default="runs/cross_method_texts.jsonl")
    ap.add_argument("--scores", default="runs/cross_method_scores.jsonl")
    ap.add_argument("--cases", default="PQ-474-931,PQ-369-758")
    ap.add_argument("--out", default="runs/case_study.md")
    ap.add_argument("--rationale-len", type=int, default=420)
    args = ap.parse_args()

    from ethicalguard.data import load_scenarios_from_jsonl
    scenes = {sc.scenario_id: sc for sc in load_scenarios_from_jsonl(args.input)}
    texts, scores = {}, {}
    for path, box in ((args.texts, texts), (args.scores, scores)):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    box.setdefault(o["scenario_id"], {})[o["method"]] = o

    def method_pqa(sid):
        out = {}
        for m in METHODS:
            o = (scores.get(sid) or {}).get(m) or {}
            if isinstance(o.get("pqa_official"), float):
                out[m] = o["pqa_official"]
        return out

    # 自动扫描优势/劣势案例
    adv, dis = [], []
    for sid, sm in scores.items():
        mp = method_pqa(sid)
        if len(mp) < 4:
            continue
        mane = mp.get("MANE")
        bl = {m: v for m, v in mp.items() if m != "MANE"}
        if mane is None or not bl:
            continue
        if mane >= max(bl.values()) + 0.15:
            adv.append((sid, mane - max(bl.values())))
        if mane <= min(bl.values()) - 0.15:
            dis.append((sid, mane - min(bl.values())))
    adv.sort(key=lambda x: -x[1])
    dis.sort(key=lambda x: x[1])
    cases = [s for s in args.cases.split(",") if s.strip()]
    if adv:
        cases.insert(0, f"[自动-优势]{adv[0][0]}")
    if dis:
        cases.insert(0, f"[自动-劣势]{dis[0][0]}")
    cases = list(dict.fromkeys(cases))

    out = ["# EthicalGuard 案例研究（协商发力/失力）\n"]
    out.append(f"- 扫描 {len(scenes)} 题 × {len(METHODS)} 方法（官方协议 pqa_official）")
    out.append(f"- MANE 优势案例（pqa 高于全部基线 ≥0.15）：{len(adv)} 个，最佳 {adv[:2]}")
    out.append(f"- MANE 劣势案例（pqa 低于最佳基线 ≥0.15）：{len(dis)} 个，最差 {dis[:2]}\n")

    for tag in cases:
        sid = tag.split("]", 1)[-1]
        if sid not in scenes or sid not in scores:
            print(f"[跳过] 无数据 {sid}")
            continue
        sc = scenes[sid]
        content = (sc.reference.content or {}) if sc.reference else {}
        kps = content.get("keypoints") or []
        question = (content.get("question") or sc.raw_text)[:400]
        out.append(f"## {tag}\n")
        out.append(f"**题目**：{question}\n")
        if kps:
            out.append("**专家要点**：" + " | ".join(k[:120] for k in kps) + "\n")
        out.append("| 方法 | 官方分 | cov | judge | 违反 | 文本长度 |")
        out.append("|---|---|---|---|---|---|")
        for m in METHODS:
            so = (scores.get(sid) or {}).get(m) or {}
            to = (texts.get(sid) or {}).get(m) or {}
            rat = to.get("rationale", "")
            out.append(f"| {BASE_NAMES.get(m, m)} | {so.get('pqa_official', '-'):.2f} | "
                       f"{so.get('cov', '-'):.2f} | {so.get('judge', '-'):.2f} | "
                       f"{so.get('violation', to.get('violation', '-'))} | {len(rat)} |")
        out.append("")
        for m in METHODS:
            to = (texts.get(sid) or {}).get(m) or {}
            rat = to.get("rationale", "")
            if not rat:
                continue
            clip = rat if len(rat) <= args.rationale_len else rat[:args.rationale_len] + "…"
            out.append(f"**{BASE_NAMES.get(m, m)} rationale**：{clip}\n")
        v = np.asarray((texts.get(sid) or {}).get("MANE", {}).get("vector") or [0.25] * 4)
        out.append(f"**MANE 终态向量** [B,N,A,J]={np.round(v, 3).tolist()} "
                   f"(违反={1 if (v < FLOORS).any() else 0})\n")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"[已保存] {args.out}（案例 {len(cases)} 个）")


if __name__ == "__main__":
    main()
