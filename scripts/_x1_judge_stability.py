# -*- coding: utf-8 -*-
"""_x1_judge_stability：内容分（PQA官方 / judge）评分器稳定性。

问题：scripts/22/23 的评审 = 单基座（本地 Qwen3-8B）且 temperature=0.0 —— 输出确定性好，
但无法回答“数字对评审随机性稳不稳”，也无法回答“换个评审基座方向是否一致”。

本脚本对 cross_method_texts 的每条最终文本，把评审后端温度临时设为 --temperature（默认 0.7）
重复 --rounds 次 PQA 官方评分（可 --metric judge 改测 judge 合理性 0-1 分），输出：
  runs/judge_stability.txt —— 每方法：
    * 场景均分 = mean(每场景多次评分均值)
    * 场景间 std（方法可区分度参考）
    * 评审噪声 = mean(每场景多次评分的 std)（judge 自身不稳度）
  * --judge2 时另用 API 基座（configs 的 llm.api + 环境 OPENAI_API_KEY 等）做同款评分，
    报告“本地 vs API 评审”的场景均值差异（方向一致性检查）。

用法：
  python scripts/_x1_judge_stability.py --config configs/config.yaml \
      --texts runs/cross_method_texts.jsonl [--rounds 3] [--temperature 0.7] \
      [--metric pqa_official|judge] [--judge2] [--out runs/judge_stability.txt]
注意：--rounds × 行数 × 方法数 = LLM 调用量；默认 120 行 × 3 ≈ 360 次单轮生成，请按预算调整。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _judge_one(c22, backend, metric, question, kps, rat, judge):
    if metric == "pqa_official":
        return c22._pqa_keypoint_score(backend, question, kps, rat)
    # judge 合理性：复用 Judges 的 LLM 浮点分（多轮均值，与 22 的 judge 列口径一致）
    return judge.score(rat, kps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--texts", default="runs/cross_method_texts.jsonl")
    ap.add_argument("--input", default="data_cache/scenarios_pqa30.jsonl")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--metric", default="pqa_official", choices=["pqa_official", "judge"])
    ap.add_argument("--judge2", action="store_true", help="追加 API 基座评审（config llm.api + OPENAI_* 环境变量）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/judge_stability.txt")
    args = ap.parse_args()

    from ethicalguard.config import Config
    from ethicalguard.llm import make_backend
    from ethicalguard.data import load_scenarios_from_jsonl
    from ethicalguard.eval.judges import Judge

    c22 = _load_module(os.path.join(REPO, "scripts", "22_content_cross_method.py"), "c22")

    # 主评审（本地，热温度）
    cfg = Config.load(args.config)
    cfg.llm.temperature = args.temperature
    backend = make_backend(cfg.llm)
    judge = Judge([backend])

    # 第二评审（API 基座，同温度）
    judge2 = None
    if args.judge2:
        cfg2 = Config.load(args.config)
        cfg2.llm.mode = "api"
        cfg2.llm.temperature = args.temperature
        backend2 = make_backend(cfg2.llm)
        judge2 = backend2
        print(f"[judge2] API 评审基座：{cfg2.llm.api.model} @ {cfg2.llm.api.base_url}")

    scenes = {sc.scenario_id: sc for sc in load_scenarios_from_jsonl(args.input)
              if sc.reference is not None and (sc.reference.content or {}).get("keypoints")}

    rows = []
    with open(args.texts, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit > 0:
        rows = rows[:args.limit]

    out_lines = []
    def emit(s=""):
        out_lines.append(s); print(s)

    emit(f"===== judge 稳定性（metric={args.metric}, temp={args.temperature}, rounds={args.rounds}）=====")

    methods = ["single_llm_llm", "neutral_single_llm", "medagents_style", "MANE"]
    per_method = {m: {"scene_means": [], "scene_stds": [], "n_ok": 0, "n_fail": 0}
                  for m in methods}
    per_method2 = {m: {"scene_means": [], "n_ok": 0, "n_fail": 0} for m in methods} if judge2 else None

    for o in rows:
        sid, method = o["scenario_id"], o["method"]
        sc = scenes.get(sid)
        rat = (o.get("rationale") or "").strip()
        if sc is None or not rat:
            continue
        content = (sc.reference.content or {}) if sc.reference else {}
        kps = content.get("keypoints") or []
        if not kps:
            continue
        question = content.get("question") or sc.raw_text[:500]

        vals = [_judge_one(c22, backend, args.metric, question, kps, rat, judge)
                for _ in range(args.rounds)]
        vals = [v for v in vals if v is not None]
        if not vals:
            per_method[method]["n_fail"] += 1
            continue
        per_method[method]["scene_means"].append(float(np.mean(vals)))
        per_method[method]["scene_stds"].append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        per_method[method]["n_ok"] += 1

        if judge2 is not None:
            v2 = _judge_one(c22, judge2, args.metric, question, kps, rat, Judge([judge2]))
            if v2 is not None:
                per_method2[method]["scene_means"].append(float(v2))
                per_method2[method]["n_ok"] += 1
            else:
                per_method2[method]["n_fail"] += 1

    emit(f"{'方法':<18}{'n':>4}{'场景均分':>10}{'场景间std':>12}{'评审噪声(轮内std)':>16}{'解析失败':>8}")
    for m in methods:
        d = per_method[m]
        if not d["scene_means"]:
            emit(f"{m:<18}{0:>4}  无有效评分")
            continue
        emit(f"{m:<18}{d['n_ok']:>4}"
             f"{np.mean(d['scene_means']):>10.3f}{np.std(d['scene_means']):>12.3f}"
             f"{np.mean(d['scene_stds']):>16.3f}{d['n_fail']:>8}")

    if judge2 is not None:
        emit("")
        emit("===== 本地 vs API 评审 方向一致性（同 metric；逐方法场景均分）=====")
        emit(f"{'方法':<18}{'n':>4}{'本地':>10}{'API':>10}{'Δ(API-本地)':>12}")
        for m in methods:
            a = np.mean(per_method[m]["scene_means"]) if per_method[m]["scene_means"] else float("nan")
            b = np.mean(per_method2[m]["scene_means"]) if per_method2[m]["scene_means"] else float("nan")
            emit(f"{m:<18}{per_method2[m]['n_ok']:>4}{a:>10.3f}{b:>10.3f}{b - a:>+12.3f}")

    emit("")
    emit("判读：评审噪声(轮内std)≈0 且 --judge2 方向一致 → Δ 无显著结论可信；"
         "噪声大或方向翻转 → 需加大 rounds/换评审基座后再下结论。")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"[已保存] {args.out}")


if __name__ == "__main__":
    main()
