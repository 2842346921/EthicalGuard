# -*- coding: utf-8 -*-
"""_x0_patch_missing_rows：补 PQA30 跨方法对比中 rationale 为空的缺行，并刷新配对统计。

问题背景：scripts/22 的 Stage A 在 30 题上跑 4 协议时，个别协议对个别题给出空 rationale
（MANE 缺 PQ-530-1024、medagents_style 缺 PQ-681-1202），导致内容分 scores 只有
30/30/29/29 行、MANE 与 medagents 的配对 n=28。本脚本：
  ① 只对这两个 (scenario, method) 重跑对应协议，重出最终文本（MANE 走 MANEEngine，
     medagents_style 走 llm_baselines），更新 cross_method_texts.jsonl 中该行；
  ② 只给这两个格补 4 把尺子（rubric/judge/cov/PQA官方），upsert 进 scores；
  ③ 重算 MANE vs 各基线在 rubric/judge/cov/pqa_official 上的配对 Δ + bootstrap95%CI + p
     （优先 wilcoxon，无 scipy 退化为符号检验），打印并写 runs/pqa_stats_n30.txt。

用法（仓库根目录）：
  python scripts/_x0_patch_missing_rows.py \
      --config configs/config.yaml \
      --input data_cache/scenarios_pqa30.jsonl \
      --texts runs/cross_method_texts.jsonl \
      --scores runs/cross_method_scores.jsonl
可选：
  --mane-runs runs/mane_pqa30.jsonl   已有 MANE 记录（仅用于对照，若含目标题则打印其旧 rationale 便于排错）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

# 默认补跑目标：MANE→PQ-530-1024（空 rationale），medagents_style→PQ-681-1202（空 rationale）
DEFAULT_TARGETS = [("PQ-530-1024", "MANE"), ("PQ-681-1202", "medagents_style")]

METRICS = [("pqa_official", "PQA官方"), ("rubric", "词面"), ("judge", "合理性"), ("cov", "要点覆盖")]


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _sign_or_wilcoxon_p(d: np.ndarray):
    n = int((np.abs(d) > 1e-12).sum())
    pos = int((d > 1e-12).sum())
    if n == 0:
        return 1.0, "ties=0"
    try:
        from scipy.stats import wilcoxon  # type: ignore
        return float(wilcoxon(d[np.abs(d) > 1e-12]).pvalue), "wilcoxon"
    except Exception:  # noqa: BLE001
        left = sum(math.comb(n, i) / (2.0 ** n) for i in range(pos + 1))
        right = sum(math.comb(n, i) / (2.0 ** n) for i in range(pos, n + 1))
        return min(1.0, 2.0 * min(left, right)), "sign"


def _load_jsonl_lines(path: str):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", default="data_cache/scenarios_pqa30.jsonl")
    ap.add_argument("--texts", default="runs/cross_method_texts.jsonl")
    ap.add_argument("--scores", default="runs/cross_method_scores.jsonl")
    ap.add_argument("--mane-runs", default=None)
    ap.add_argument("--out", default="runs/pqa_stats_n30.txt")
    ap.add_argument("--targets", default=None,
                    help="JSON: [[scenario_id, method], ...]；默认补两个已知空行")
    args = ap.parse_args()

    from ethicalguard.config import Config, default_agent_specs
    from ethicalguard.llm import make_backend
    from ethicalguard.data import load_scenarios_from_jsonl
    from ethicalguard.mane.engine import MANEEngine
    from ethicalguard.eval.baselines.llm_baselines import medagents_style
    from ethicalguard.eval.judges import Judge
    from ethicalguard.eval import metrics as M
    from ethicalguard.utils import set_seed

    c22 = _load_module(os.path.join(REPO, "scripts", "22_content_cross_method.py"), "c22")

    cfg = Config.load(args.config)
    set_seed(42)
    backend = make_backend(cfg.llm)
    specs = cfg.agents or default_agent_specs()

    scenes = {sc.scenario_id: sc for sc in load_scenarios_from_jsonl(args.input)
              if sc.reference is not None
              and (sc.reference.content or {}).get("keypoints")}
    print(f"rubric+keypoints 场景可用：{len(scenes)}")

    targets = DEFAULT_TARGETS if not args.targets else [tuple(t) for t in json.loads(args.targets)]

    # ---- 1) 重出两条空 rationale 文本 ----
    texts_by = {}
    for o in _load_jsonl_lines(args.texts):
        texts_by[(o["scenario_id"], o["method"])] = o

    for sid, method in targets:
        sc = scenes.get(sid)
        if sc is None:
            print(f"[skip] 场景不存在：{sid}"); continue
        old = texts_by.get((sid, method), {})
        old_len = len(old.get("rationale") or "")
        if method == "MANE":
            engine = MANEEngine(cfg)
            rec = engine.run(sc)
            rationale = c22._mane_rationale(rec)
            if not rationale.strip():
                # 兜底：直接取 final_proposal.rationale / 委员会提案，便于排错
                rationale = ((rec.get("final_proposal") or {}).get("rationale") or "")
            vec = np.array([rec.get("final_vector", {}).get(k, 0.0)
                            for k in ("beneficence", "nonmaleficence", "autonomy", "justice")],
                           dtype=float)
            treat = None
            print(f"MANE 重跑 {sid}：rationale_len={len(rationale)}（旧 {old_len}）")
        else:
            prop = medagents_style(sc, backend, specs)
            rationale = prop.rationale or ""
            vec = prop.principle_weights.as_array()
            treat = prop.treatment_level
            print(f"medagents 重跑 {sid}：rationale_len={len(rationale)}（旧 {old_len}）")
        if not rationale.strip():
            print(f"[warn] {sid} {method} 重跑后 rationale 仍为空——请人工排查该题协商过程")
        texts_by[(sid, method)] = {"scenario_id": sid, "method": method,
                                   "rationale": rationale,
                                   "vector": [float(x) for x in vec],
                                   "t": treat,
                                   "violation": float((vec < np.array([0.15, 0.20, 0.15, 0.20])).mean())}

    with open(args.texts, "w", encoding="utf-8") as f:
        for o in texts_by.values():
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[texts] 已更新 {args.texts}（共 {len(texts_by)} 行）")

    # ---- 2) 给这两个格补打分（只补缺失/官方分为空的格）----
    scores_rows = _load_jsonl_lines(args.scores)
    score_idx = {(o["scenario_id"], o["method"]): i for i, o in enumerate(scores_rows)}
    judge = Judge([backend])

    for sid, method in targets:
        sc = scenes.get(sid)
        t = texts_by.get((sid, method))
        if sc is None or t is None or not (t.get("rationale") or "").strip():
            print(f"[skip 打分] {sid} {method} 无文本"); continue
        content = (sc.reference.content or {}) if sc.reference else {}
        kps = content.get("keypoints") or []
        if not kps:
            print(f"[skip 打分] {sid} 无 keypoints"); continue
        question = content.get("question") or sc.raw_text[:500]
        rat = t["rationale"]
        row = {"scenario_id": sid, "method": method,
               "rubric": float(M.rubric_alignment(rat, kps)),
               "judge": judge.score(rat, kps)}
        dec = c22._dec_axes(backend, question, kps, rat) or {}
        row.update({"clear": dec.get("decision_clear"), "cov": dec.get("keypoint_cov"),
                    "align": dec.get("principle_align")})
        row["pqa_official"] = c22._pqa_keypoint_score(backend, question, kps, rat)
        if (sid, method) in score_idx:
            scores_rows[score_idx[(sid, method)]].update({k: v for k, v in row.items()
                                                          if v is not None})
            print(f"[scores 更新] {sid} {method}")
        else:
            scores_rows.append(row)
            score_idx[(sid, method)] = len(scores_rows) - 1
            print(f"[scores 新增] {sid} {method}")

    with open(args.scores, "w", encoding="utf-8") as f:
        for o in scores_rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[scores] 已写回 {args.scores}（共 {len(scores_rows)} 行）")

    # ---- 3) 配对统计（n=30 后）----
    sc_by_id = {(o["scenario_id"], o["method"]): o for o in scores_rows}
    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    rng = np.random.default_rng(42)
    emit("===== 配对统计（补行后；MANE vs 基线，同题同尺子）=====")
    emit(f"{'基线':<18}{'指标':<12}{'n':>4}{'MANE':>8}{'基线':>8}{'Δ':>8}{'95%CI':>20}{'p':>8}")
    emit("-" * 92)
    for base in ["single_llm_llm", "neutral_single_llm", "medagents_style"]:
        for key, name in METRICS:
            pairs = []
            for sid in scenes:
                a = sc_by_id.get((sid, "MANE"), {}).get(key)
                b = sc_by_id.get((sid, base), {}).get(key)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    pairs.append((float(a), float(b)))
            if len(pairs) < 10:
                continue
            d = np.array([a - b for a, _ in pairs])
            boots = np.empty(5000)
            for i in range(5000):
                idx = rng.integers(0, len(d), len(d))
                boots[i] = d[idx].mean()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p, pm = _sign_or_wilcoxon_p(d)
            emit(f"{base:<18}{name:<12}{len(pairs):>4}"
                 f"{np.mean([a for a, _ in pairs]):>8.3f}{np.mean([b for _, b in pairs]):>8.3f}"
                 f"{d.mean():>+8.3f}[{lo:+.3f},{hi:+.3f}]{p:>8.4f}({pm})")
    # 说明仍缺的格
    missing = [(sid, m) for sid in scenes
               for m in ["single_llm_llm", "neutral_single_llm", "medagents_style", "MANE"]
               if (sid, m) not in sc_by_id or not isinstance(sc_by_id.get((sid, m), {}).get("pqa_official"), float)]
    if missing:
        emit(f"[warn] 仍有缺行/官方分缺失：{missing}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[已保存] {args.out}")


if __name__ == "__main__":
    main()
