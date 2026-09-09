"""24_mane_rationale_expand：MANE 最终文本"输出充分性"实验（回答 cov/官方覆盖缺口是否
只是"只取了委员会一句话"的界面问题）。

背景（2026-09-07 实验四）：MANE final rationale = 伦理委员会简短提案（~400 字符），
cov/官方内容分系统性略低；假说 = 协商其实产出了完整多方论证，只是**输出界面**只取
委员会文本（任务错配：决策文本 vs 建议清单）。本实验构造 MANE 的"完整共识论证"变体，
只改输出界面、不改协商/GNE/底线：
  short  : 委员会提案（现状，cross_method_texts 已有 MANE 行）
  full   : 末轮全部在场方（非 catfish）提案 rationale 拼接（委员会在前），
           代表"协商共识的完整论证"
对每个场景在官方尺子（pqa_official）与决策对齐 cov 上对比 short vs full（配对），
并与基线（single/neutral/medagents 的已有 pqa_official）对照——
  若 full 显著 > short 且 ≥ 基线 → 缺口是输出界面问题，协商论证本身充分
  （论文："以完整共识论证输出时，要点覆盖追平甚至超过直接问答基线"）；
  若 full ≈ short → 缺口在协商产出本身，写入 limitation。

用法：
  python scripts/24_mane_rationale_expand.py --config configs/config.yaml \
      --input data_cache/scenarios_pqa30.jsonl \
      --mane-runs runs/mane_pqa30.jsonl \
      --texts runs/cross_method_texts.jsonl \
      --scores runs/cross_method_scores.jsonl \
      [--skip-existing] [--out runs/mane_rationale_expand.jsonl]
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
from ethicalguard.data.loaders import iter_negotiation_records
from ethicalguard.eval.content_judges import dec_axes, pqa_keypoint_score
from ethicalguard.llm import make_backend


def _consensus_round(r: dict) -> dict:
    """末个 ≥2 非 catfish 提案的协商轮（与 07/20 回溯口径一致）。"""
    for tr in reversed(r.get("trajectory", [])):
        props = [p for p in tr.get("proposals", []) if p.get("agent") != "catfish"]
        if len(props) >= 2:
            return {p["agent"]: p.get("rationale", "") for p in props}
    return {}


def _full_rationale(r: dict, max_chars: int = 2500) -> str:
    """完整共识论证：末轮在场各方 rationale 拼接（委员会在前，catfish 异议在后可选）。"""
    order = ["ethics_committee", "physician", "patient", "family", "hospital_admin"]
    props = _consensus_round(r)
    parts = []
    for aid in order:
        if aid in props and props[aid]:
            parts.append(f"[{aid}] {props[aid]}")
    # 鲶鱼异议（若在末轮）作为"异议视角"附后
    for tr in reversed(r.get("trajectory", [])):
        dissent = [p for p in tr.get("proposals", []) if p.get("agent") == "catfish" and p.get("rationale")]
        if dissent:
            parts.append(f"[catfish 异议] {dissent[-1]['rationale']}")
            break
    text = "\n\n".join(parts)
    return text[:max_chars]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", required=True, help="开放题场景 jsonl")
    ap.add_argument("--mane-runs", required=True, help="MANE 协商结果（含 trajectory）")
    ap.add_argument("--texts", default="runs/cross_method_texts.jsonl")
    ap.add_argument("--scores", default="runs/cross_method_scores.jsonl")
    ap.add_argument("--out", default="runs/mane_rationale_expand.jsonl")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    backend = make_backend(cfg.llm)
    if backend.mode == "rule":
        print("[错误] 需要 LLM judge（local/api）")
        sys.exit(2)

    runs = {r["scenario_id"]: r for r in iter_negotiation_records(args.mane_runs)}
    texts = {}
    with open(args.texts, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                texts.setdefault((o["scenario_id"], o["method"]), o)
    base_scores = {}
    if os.path.exists(args.scores):
        with open(args.scores, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    base_scores.setdefault((o["scenario_id"], o["method"]), o)

    done = set()
    if args.skip_existing and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["scenario_id"])

    rows = []
    n_short = n_full = 0
    for sc in load_scenarios_from_jsonl(args.input):
        sid = sc.scenario_id
        if sid in done or sid not in runs:
            continue
        ref = sc.reference
        content = (ref.content or {}) if ref else {}
        keypoints = content.get("keypoints") or []
        if not keypoints:
            continue
        question = content.get("question") or sc.raw_text[:500]
        short = (texts.get((sid, "MANE")) or {}).get("rationale", "")
        full = _full_rationale(runs[sid])
        if not short or not full:
            continue
        print(f"--- {sid}: short={len(short)} 字符, full={len(full)} 字符")
        s_pqa = pqa_keypoint_score(backend, question, keypoints, short)
        f_pqa = pqa_keypoint_score(backend, question, keypoints, full)
        s_dec = dec_axes(backend, question, keypoints, short) or {}
        f_dec = dec_axes(backend, question, keypoints, full) or {}
        row = {"scenario_id": sid, "short_len": len(short), "full_len": len(full),
               "short_pqa": s_pqa, "full_pqa": f_pqa,
               "short_cov": s_dec.get("keypoint_cov"), "full_cov": f_dec.get("keypoint_cov"),
               "short_clear": s_dec.get("decision_clear"), "full_clear": f_dec.get("decision_clear")}
        rows.append(row)
        n_short += 1 if s_pqa is not None else 0
        n_full += 1 if f_pqa is not None else 0
        print(f"    pqa short={s_pqa} full={f_pqa} | cov short={row['short_cov']} full={row['full_cov']}")
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- 配对汇总 ----
    print("\n===== 配对：full(完整共识论证) − short(委员会) =====")
    if rows:
        for k, name in (("pqa", "官方协议"), ("cov", "要点覆盖"), ("clear", "决策明确")):
            ds = [(r[f"full_{k}"], r[f"short_{k}"]) for r in rows
                  if r[f"full_{k}"] is not None and r[f"short_{k}"] is not None]
            if ds:
                d = np.mean([b - a for a, b in ds])  # full - short
                print(f"  {name:<8} n={len(ds):>3}  full均值={np.mean([b for _, b in ds]):.3f}"
                      f" short均值={np.mean([a for a, _ in ds]):.3f}  Δ(full−short)={d:+.3f}")

    # 对照：基线 pqa_official（已有 scores）
    print("\n===== 对照：基线 pqa_official（cross_method_scores）=====")
    for m in ["single_llm_llm", "neutral_single_llm", "medagents_style", "MANE(short)"]:
        key_m = "MANE" if m == "MANE(short)" else m
        xs = [o.get("pqa_official") for (sid, mm), o in base_scores.items()
              if mm == key_m and isinstance(o.get("pqa_official"), float)]
        if xs:
            print(f"  {m:<18} n={len(xs):>3}  pqa均值={np.mean(xs):.3f}")
    if rows:
        fxs = [r["full_pqa"] for r in rows if r["full_pqa"] is not None]
        if fxs:
            print(f"  {'MANE(full)':<18} n={len(fxs):>3}  pqa均值={np.mean(fxs):.3f}")

    print("\n口径：Δ>0 = 完整共识论证提升官方内容分。full ≥ 基线 → 缺口来自'输出界面只取委员会'；")
    print("full ≈ short → 缺口在协商产出本身（limitation 如实写）。产物:", args.out)


if __name__ == "__main__":
    main()
