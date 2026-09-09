"""26_hetero_judge：LLM-as-judge 自评偏置检验——异构/更强模型交叉复评。

现状：内容尺子（官方 pqa_official / judge / cov）全部由 Qwen3-8B 自评（它也是回答生成方）→
审稿人问"自评偏置"。本脚本用**另一个 OpenAI 兼容后端**（如 Meta-Llama-3.1-8B-Instruct /
Qwen3-14B / Mistral-7B）对同一批 (场景,方法) 回答重跑**官方 keypoint 协议**（与 PrinciplismQA
论文同款尺子，最不易受风格偏置影响），然后对比：
  ① 各方法在"Qwen 自评"与"异构 judge"下的 pqa_official 均值 —— 方法排序是否一致；
  ② 逐场景 Spearman（两位 judge 的评分一致性）；
  ③ 配对差（异构 − 自评）是否系统偏向某方法（若某方法被自评高估 → 偏置证据）。

用法（先起 judge 服务，再跑本脚本）：
  # 起异构 judge vLLM（guard env；显存不够就把 gpu-memory-utilization 调小）：
  nohup vllm serve /mnt/model/Meta-Llama-3.1-8B-Instruct \
      --served-model-name hetero-judge --port 8010 \
      --gpu-memory-utilization 0.6 --max-model-len 16384 \
      > logs/vllm_hetero_judge.log 2>&1 &
  python scripts/26_hetero_judge.py --config configs/config.yaml \
      --input data_cache/scenarios_pqa30.jsonl \
      --texts runs/cross_method_texts.jsonl \
      --judge-url http://127.0.0.1:8010/v1 --judge-model hetero-judge \
      --judge-name Meta-Llama-3.1-8B \
      [--ruler pqa] [--out runs/hetero_judge_Meta-Llama-3.1-8B.jsonl]
  # Qwen 系 judge（Qwen3-14B 等）需关 thinking：加 --qwen3
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
from ethicalguard.eval.content_judges import dec_axes, pqa_keypoint_score
from ethicalguard.llm.api import APIBackend

METHODS = ["single_llm_llm", "neutral_single_llm", "medagents_style", "MANE"]


def _rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _spearman(a, b):
    ra, rb = _rankdata(np.asarray(a, float)), _rankdata(np.asarray(b, float))
    d = ra - rb
    n = len(a)
    rho = 1.0 - 6.0 * float((d ** 2).sum()) / (n * (n * n - 1)) if n > 1 else 0.0
    z = rho * math.sqrt(max(0, n - 1))
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return rho, min(1.0, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--texts", default="runs/cross_method_texts.jsonl")
    ap.add_argument("--self-scores", default="runs/cross_method_scores.jsonl",
                    help="Qwen3-8B 自评的 scores（含 pqa_official 基线）")
    ap.add_argument("--judge-url", required=True, help="异构 judge 的 OpenAI 兼容 base_url（到 /v1）")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-name", default="hetero", help="报告用名（写文件名）")
    ap.add_argument("--qwen3", action="store_true", help="Qwen 系 judge：关 thinking")
    ap.add_argument("--ruler", default="pqa", choices=["pqa", "dec"],
                    help="pqa=官方 keypoint 协议（主）；dec=决策对齐三轴")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    kwargs = {}
    if args.qwen3:
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    judge = APIBackend(base_url=args.judge_url, api_key=None, model=args.judge_model,
                       temperature=0.0, max_tokens=512, **kwargs)
    print(f"[judge] {args.judge_model} @ {args.judge_url}（qwen3 关 thinking={args.qwen3}）")

    scenes = {sc.scenario_id: sc for sc in load_scenarios_from_jsonl(args.input)}
    texts = {}
    with open(args.texts, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                texts.setdefault(o["scenario_id"], {})[o["method"]] = o
    self_scores = {}
    with open(args.self_scores, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                self_scores.setdefault(o["scenario_id"], {})[o["method"]] = o

    out_path = args.out or f"runs/hetero_judge_{args.judge_name}.jsonl"
    done = set()
    if args.skip_existing and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add((json.loads(line)["scenario_id"], json.loads(line)["method"]))

    rows = []
    for sid, sm in texts.items():
        if sid not in scenes:
            continue
        ref = scenes[sid].reference
        content = (ref.content or {}) if ref else {}
        keypoints = content.get("keypoints") or []
        if not keypoints:
            continue
        question = content.get("question") or scenes[sid].raw_text[:500]
        for m in METHODS:
            t = sm.get(m)
            if t is None or not t.get("rationale"):
                continue
            if (sid, m) in done:
                continue
            if args.ruler == "pqa":
                score = pqa_keypoint_score(judge, question, keypoints, t["rationale"])
            else:
                score = dec_axes(judge, question, keypoints, t["rationale"])
            row = {"scenario_id": sid, "method": m,
                   "hetero_" + args.ruler: score,
                   "self_" + args.ruler: (self_scores.get(sid, {}).get(m) or {}).get(
                       "pqa_official" if args.ruler == "pqa" else "cov")}
            rows.append(row)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {sid:<18} {m:<18} hetero_{args.ruler}={score}")

    # ---- 一致性分析（读全文件：含本次新增 + skip-existing 保留的历史行）----
    print(f"\n===== 异构 judge（{args.judge_name}）vs Qwen3-8B 自评（{args.ruler}）=====")
    hk, sk = f"hetero_{args.ruler}", f"self_{args.ruler}"
    print(f"{'方法':<20}{'n':>4}{'异构分':>8}{'自评分':>8}{'Δ(异−自)':>10}")
    all_rows = []
    with open(out_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))
    valid = [r for r in all_rows if isinstance(r.get(hk), float) and isinstance(r.get(sk), float)]
    by_m = {}
    for m in METHODS:
        xs = [r for r in valid if r["method"] == m]
        if not xs:
            continue
        hv = np.mean([r[hk] for r in xs])
        sv = np.mean([r[sk] for r in xs])
        by_m[m] = (hv, sv)
        print(f"{m:<20}{len(xs):>4}{hv:>8.3f}{sv:>8.3f}{hv - sv:>+10.3f}")
    if len(valid) >= 8:
        rho, p = _spearman([r[hk] for r in valid], [r[sk] for r in valid])
        print(f"\n逐场景一致性 Spearman ρ={rho:.3f} (p={p:.4f}, n={len(valid)})")
    print("\n口径：方法排序在两位 judge 下一致 + ρ 高 + Δ 无系统偏向某方法 → 自评偏置不显著；")
    print("若某方法被自评明显高估（Δ 大）→ 报告该偏置并优先采信异构 judge/人类。")
    print("[产物]", out_path)


if __name__ == "__main__":
    main()
