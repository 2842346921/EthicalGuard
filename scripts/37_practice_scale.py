"""37_practice_scale：PrinciplismQA Practice 全量/大子集官方评估（基线直答）——对齐论文 Table 4。

论文 Practice 分 = 1466 开放题 × 逐 keypoint 0/0.5/1.0 Gained/Sum（他们 Evaluator 同款）。
我们的 22/27/34 只在 30 题子集（MANE 协商成本限制）→ 无法与论文全量绝对分对照。
本脚本在 open-ended-qa（1466 问，含 case+question）上对**基线直答**（single_llm_llm /
neutral_single_llm，成本低）跑官方评分 → 得到可与论文 Table 4 同级定位的"基座 Practice 全量分"
（他们 Llama-3.1-8B=48.5 / Qwen2.5-7B=49.4 为 8B 参考），同时验证我们的 judge 管线与官方口径一致。
MANE/议题感知仍报告在 30 题子集的相对增益（34）。

用法：
  python scripts/37_practice_scale.py --config configs/config.yaml \
      --qa data/PrinciplismQA/data/open-ended-qa.json \
      --rubric data/PrinciplismQA/data/open-ended-rubric-principles.json \
      [--methods single_llm_llm,neutral_single_llm] [--limit 300] \
      [--out runs/practice_scale.jsonl] [--skip-existing]
  # --limit 0 = 全量 1466（≈ 3-5h）；建议先 --limit 300 验证，再 nohup 全量。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ethicalguard.config import Config, default_agent_specs
from ethicalguard.data.mapping_rules import default_constraints, default_parties, map_text_to_state
from ethicalguard.eval.baselines.llm_baselines import llm_single, neutral_single_llm
from ethicalguard.eval.content_judges import pqa_keypoint_scores
from ethicalguard.llm import make_backend
from ethicalguard.mane.agents import build_agent
from ethicalguard.types import Scenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--qa", required=True, help="open-ended-qa.json")
    ap.add_argument("--rubric", required=True, help="open-ended-rubric-principles.json")
    ap.add_argument("--methods", default="single_llm_llm,neutral_single_llm")
    ap.add_argument("--limit", type=int, default=300, help="问数上限（0=全量 1466）")
    ap.add_argument("--out", default="runs/practice_scale.jsonl")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    backend = make_backend(cfg.llm)
    if backend.mode == "rule":
        print("[错误] 需要 local/api")
        sys.exit(2)
    specs = cfg.agents or default_agent_specs()
    physician = build_agent(specs["physician"], backend)
    methods = [m for m in args.methods.split(",") if m.strip()]

    # 索引：qid -> {case 文本, question, [(keypoint, competency)], principles}
    qa = json.load(open(args.qa, encoding="utf-8"))
    by_qid = {}
    for case in qa:
        for issue in case.get("ethical_issues", []):
            by_qid[issue.get("qid")] = {
                "case": (case.get("case") or case.get("case_rewrite") or "")[:2500],
                "question": issue.get("question", ""),
            }
    rubric = {}
    for it in json.load(open(args.rubric, encoding="utf-8")):
        rubric[it.get("qid")] = {
            "kc": [(k.get("keypoint", ""), k.get("competency", ""))
                   for k in it.get("keypoint_competencies", [])],
            "principles": it.get("principles", []),
        }
    qids = sorted(by_qid)
    if args.limit > 0:
        qids = qids[:args.limit]
    print(f"问数 {len(qids)}（全量 1466 中取前 {len(qids)}）方法 {methods}")

    done = set()
    if args.skip_existing and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    done.add((o["qid"], o["method"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    with open(args.out, "a", encoding="utf-8") as f:
        for qid in qids:
            info = by_qid[qid]
            rub = rubric.get(qid)
            if rub is None or not rub["kc"]:
                continue
            keypoints = [kp for kp, _ in rub["kc"]]
            comps = [c for _, c in rub["kc"]]
            text = f"{info['case']}\n\nEthical question: {info['question']}"
            st = map_text_to_state(text)
            sc = Scenario(scenario_id=f"PQ-{qid}", raw_text=text, state=st,
                          constraints=default_constraints(st), parties=default_parties(st))
            for m in methods:
                if (qid, m) in done:
                    continue
                try:
                    if m == "single_llm_llm":
                        p = llm_single(sc, backend, physician)
                    elif m == "neutral_single_llm":
                        p = neutral_single_llm(sc, backend)
                    else:
                        continue
                    rat = p.rationale or ""
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] qid={qid} {m} 生成失败: {str(e)[:100]}")
                    continue
                scores = pqa_keypoint_scores(backend, info["question"], keypoints, rat)
                row = {"qid": qid, "method": m,
                       "pqa": (float(np.mean(scores)) if scores else None),
                       "pqa_scores": scores,
                       "competencies": comps, "principles": rub["principles"]}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  qid={qid:<6} {m:<18} pqa={row['pqa']}")

    # 汇总（含历史行）
    all_rows = []
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))
    print("\n===== Practice 基线汇总（官方 Gained/Sum）=====")
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_rows:
        if isinstance(r.get("pqa"), float):
            agg[r["method"]].append(r["pqa"])
    for m, xs in agg.items():
        print(f"  {m:<20} n={len(xs):>4}  Practice={np.mean(xs):.3f}")
    print("\n对照（论文 Table 4，8B 参考）：Llama-3.1-8B=48.5 / Qwen2.5-7B=49.4（同 1466 全量口径）。")
    print("[产物]", args.out)


if __name__ == "__main__":
    main()
