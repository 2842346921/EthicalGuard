"""34_content_vs_baseline_same_batch：议题感知 MANE(B) vs 单 LLM/基线 的**同批**内容对照。

背景（审稿攻击面 A）：27 v2 证明 B（议题感知）显著优于 A（MANE 现状）——同批 judge；
但"B vs 单 LLM/medagents"没有同批证据（跨批次 judge 有 ±0.06 运行间噪声，不能直接比）。
本脚本把 6 个方法的 rationale 放进**同一批 judge**（pqa_official + cov），直接回答：
  "协商+议题感知是否显著优于直接问答"（内容层完整证据）。
方法（文本源）：
  single_llm_llm / neutral_single_llm / medagents_style / MANE(A) ← runs/cross_method_texts.jsonl
  MANE+议题(B) / MANE+C(议题+例外)                          ← runs/arch_texts_v2.jsonl
判定：B 的 pqa/cov 同批显著 > single_llm_llm/medagents → 内容层故事完整
（保证层 0% vs 23-100% 已证，不在此脚本）。

用法：
  python scripts/34_content_vs_baseline_same_batch.py --config configs/config.yaml \
      --input data_cache/scenarios_pqa30.jsonl \
      --texts runs/cross_method_texts.jsonl \
      --arch-texts runs/arch_texts_v2.jsonl \
      [--limit 30] [--out runs/content_vs_baseline.jsonl] [--skip-existing]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ethicalguard.config import Config
from ethicalguard.data import load_scenarios_from_jsonl
from ethicalguard.eval.content_judges import dec_axes, pqa_keypoint_scores
from ethicalguard.llm import make_backend

# 方法标签 → (文本源文件, 匹配键)
METHODS = [
    ("single_llm_llm", "cross", "single_llm_llm"),
    ("neutral_single_llm", "cross", "neutral_single_llm"),
    ("medagents_style", "cross", "medagents_style"),
    ("MANE(现状A)", "cross", "MANE"),
    ("MANE+议题(B)", "arch", "B_议题感知"),
    ("MANE+C(议题+例外)", "arch", "C_议题+例外"),
]


def _load_texts(path: str, key_field: str):
    """cross 文件：行 {scenario_id, method, rationale}；arch 文件：行 {scenario_id, tag, rationale}。"""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            k = o.get(key_field)
            if o.get("rationale"):
                out[(o["scenario_id"], k)] = o["rationale"]
    return out


def _wilcoxon_or_sign(d: np.ndarray):
    try:
        from scipy.stats import wilcoxon  # type: ignore
        return float(wilcoxon(d[np.abs(d) > 1e-12]).pvalue), "w"
    except Exception:  # noqa: BLE001
        pos = int((d > 1e-12).sum())
        nz = int((np.abs(d) > 1e-12).sum())
        if nz == 0:
            return 1.0, "s0"
        L = sum(math.comb(nz, i) / (2.0 ** nz) for i in range(pos + 1))
        R = sum(math.comb(nz, i) / (2.0 ** nz) for i in range(pos, nz + 1))
        return min(1.0, 2.0 * min(L, R)), "s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--texts", default="runs/cross_method_texts.jsonl")
    ap.add_argument("--arch-texts", default="runs/arch_texts_v2.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/content_vs_baseline.jsonl")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--rubric", default=None,
                    help="open-ended-rubric-principles.json（按 qid 取 keypoint+ACGME competency 顺序）")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    backend = make_backend(cfg.llm)
    if backend.mode == "rule":
        print("[错误] 需要 LLM judge（local/api）")
        sys.exit(2)

    cross = _load_texts(args.texts, "method")
    arch = _load_texts(args.arch_texts, "tag")
    scenes = [sc for sc in load_scenarios_from_jsonl(args.input)
              if sc.reference is not None and sc.reference.kind == "rubric"
              and (sc.reference.content or {}).get("keypoints")]
    if args.limit > 0:
        scenes = scenes[:args.limit]

    # rubric 按 qid 索引：(keypoint, ACGME competency) 顺序表（keypoints 顺序 = 官方评分顺序）
    rubric_by_qid = {}
    if args.rubric and os.path.exists(args.rubric):
        import json as _json
        for it in _json.load(open(args.rubric, encoding="utf-8")):
            kc = [(k.get("keypoint", ""), k.get("competency", ""))
                  for k in it.get("keypoint_competencies", [])]
            rubric_by_qid[it.get("qid")] = kc
        print(f"[rubric] 按 qid 索引 {len(rubric_by_qid)} 条（含 ACGME competency）")

    done = set()
    if args.skip_existing and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    done.add((o["scenario_id"], o["method"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"场景 {len(scenes)} × 方法 {len(METHODS)}（同批 judge：pqa + cov）")
    rows = []
    with open(args.out, "a", encoding="utf-8") as f:
        for sc in scenes:
            sid = sc.scenario_id
            content = (sc.reference.content or {}) if sc.reference else {}
            qid = content.get("qid")
            comps = None
            if qid in rubric_by_qid:
                kc = rubric_by_qid[qid]
                keypoints = [kp for kp, _ in kc]
                comps = [c for _, c in kc]
            else:
                keypoints = content.get("keypoints") or []
            if not keypoints:
                continue
            question = content.get("question") or sc.raw_text[:500]
            for mname, src, key in METHODS:
                if (sid, mname) in done:
                    continue
                pool = cross if src == "cross" else arch
                rat = pool.get((sid, key))
                if not rat:
                    continue
                pqa_scores = pqa_keypoint_scores(backend, question, keypoints, rat)
                dec = dec_axes(backend, question, keypoints, rat) or {}
                row = {"scenario_id": sid, "qid": qid, "method": mname,
                       "pqa": (float(np.mean(pqa_scores)) if pqa_scores else None),
                       "pqa_scores": pqa_scores, "competencies": comps,
                       "cov": dec.get("keypoint_cov")}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  {sid:<18} {mname:<16} pqa={row['pqa']} cov={row['cov']}")

    # ---- 同批汇总 + 配对 ----
    all_rows = []
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))
    print("\n===== 同批内容对照（pqa / cov 均值）=====")
    by = {}
    for r in all_rows:
        by.setdefault(r["method"], []).append(r)
    for mname, _, _ in METHODS:
        rs = by.get(mname) or []
        if not rs:
            continue
        pqa = np.mean([r["pqa"] for r in rs if isinstance(r.get("pqa"), float)])
        cov = np.mean([r["cov"] for r in rs if isinstance(r.get("cov"), float)])
        print(f"  {mname:<18} n={len(rs):>3}  pqa={pqa:.3f}  cov={cov:.3f}")

    print("\n===== 配对（同题）：MANE+议题(B) vs 基线 =====")
    bmap = {r["scenario_id"]: r for r in by.get("MANE+议题(B)", [])}
    for base in ("single_llm_llm", "neutral_single_llm", "medagents_style", "MANE(现状A)"):
        bmap2 = {r["scenario_id"]: r for r in by.get(base, [])}
        for k, name in (("pqa", "官方协议"), ("cov", "要点覆盖")):
            ds = [(bmap[s][k], bmap2[s][k]) for s in bmap if s in bmap2
                  and isinstance(bmap[s].get(k), float) and isinstance(bmap2[s].get(k), float)]
            if len(ds) < 10:
                continue
            av = np.array([a for a, _ in ds])
            bv = np.array([b for _, b in ds])
            diff = av - bv  # B − base
            p, pm = _wilcoxon_or_sign(diff)
            print(f"  B vs {base:<18} {name:<6} n={len(ds):>3} B={av.mean():.3f} "
                  f"{base}={bv.mean():.3f} Δ={diff.mean():+.3f} p={p:.4f}({pm})")

    print("\n判定：B 的 pqa/cov 同批显著 > single_llm/medagents → '协商+议题感知'内容层完整优于直答；")
    print("若仅 > A（MANE 内部）而 ≈ 单 LLM → 内容增益属'议题感知'贡献，表述为增强而非方法层优势。")
    print("[产物]", args.out)


if __name__ == "__main__":
    main()
