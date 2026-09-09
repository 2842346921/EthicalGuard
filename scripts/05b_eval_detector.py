"""05b_eval_detector：监督冲突检测器 held-out 泛化对比（非训练 eval）。

背景：detector_v2.pt 训练日志的 principle_acc=0.715 / ERS MAE=0.043 是**训练内 eval 集**
自报；审稿人会问"独立测试集呢？"。本脚本用 PrinciplismQA MCQ（knowledge-mcqa，2182 题）
做 principle 头的 **held-out 泛化测试**（MCQ 从未进监督训练——14 docstring 口径）：
对比 detector.pt(v1) / detector_v2.pt(v2) 的 top-1/多标签/逐原则，规则通道作参照。

用法：
  python scripts/05b_eval_detector.py --config configs/config.yaml \
      [--checkpoints runs/detector.pt,runs/detector_v2.pt] \
      [--out runs/detector_heldout.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ethicalguard.config import Config
from ethicalguard.data.mapping_rules import default_constraints, default_parties, map_text_to_state
from ethicalguard.types import Scenario

P_NAMES = ["B(行善)", "N(不伤害)", "A(自主)", "J(公正)"]


def _load_mcq(data_dir: str, limit: int):
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        data_dir,
    ]
    for base in candidates:
        cand = os.path.join(base, "PrinciplismQA", "data", "knowledge-mcqa.json")
        if os.path.exists(cand):
            items = json.load(open(cand, encoding="utf-8"))
            return items[:limit] if limit > 0 else items
    raise FileNotFoundError("未找到 knowledge-mcqa.json")


def _gold_vector(principlism: dict) -> np.ndarray:
    return np.array([
        1.0 if principlism.get("beneficience") or principlism.get("beneficence") else 0.0,
        1.0 if principlism.get("nonmaleficience") or principlism.get("nonmaleficence") else 0.0,
        1.0 if principlism.get("autonomy") else 0.0,
        1.0 if principlism.get("justice") else 0.0,
    ])


def _scenario_for(it: dict) -> Scenario:
    text = f"{it.get('question','')}\n{str(it.get('options',{}))}"
    st = map_text_to_state(text)
    return Scenario(scenario_id=f"PQ-MCQ-{it.get('question_id') or it.get('id')}",
                    raw_text=text, state=st,
                    constraints=default_constraints(st), parties=default_parties(st))


def _detect_pv(detector, sc) -> np.ndarray:
    report = detector.detect(sc)
    pv = report.principle_vector
    return np.array([pv.beneficence, pv.nonmaleficence, pv.autonomy, pv.justice])


def _score_channel(name: str, pv_fn, items) -> dict:
    top1_hit = exact = 0
    jacc, n_valid = [], 0
    per = np.zeros(4)
    per_n = np.zeros(4)
    for it in items:
        gold = _gold_vector(it.get("principlism", {}))
        if gold.sum() == 0:
            continue
        n_valid += 1
        pv_arr = pv_fn(it)
        top1 = int(np.argmax(pv_arr))
        pred = (pv_arr > 0.5).astype(float)
        if gold[top1] == 1.0:
            top1_hit += 1
        for gi in range(4):
            if gold[gi] == 1.0:
                per_n[gi] += 1
                if top1 == gi:
                    per[gi] += 1
        if np.array_equal(pred, gold):
            exact += 1
        inter = float(np.logical_and(pred, gold).sum())
        union = float(np.logical_or(pred, gold).sum())
        jacc.append(inter / union if union > 0 else 1.0)
    return {
        "channel": name, "n": n_valid,
        "top1": top1_hit / n_valid if n_valid else 0.0,
        "exact": exact / n_valid if n_valid else 0.0,
        "jaccard": float(np.mean(jacc)) if jacc else 0.0,
        "per_principle_top1": {P_NAMES[i]: (per[i] / per_n[i] if per_n[i] else None)
                               for i in range(4)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoints", default="runs/detector.pt,runs/detector_v2.pt",
                    help="逗号分隔的监督检测器权重（与规则通道对比）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data_dir = cfg.datasets.data_dir or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data")
    items = _load_mcq(data_dir, args.limit)
    print(f"===== 监督检测器 held-out 泛化（MCQ {len(items)} 题，从未进监督训练）=====")

    from ethicalguard.detection.rule import RuleConflictDetector
    from ethicalguard.detection.supervised import SupervisedConflictDetector

    channels = []
    # 规则通道（参照）
    rule = RuleConflictDetector()
    channels.append(_score_channel("rule", lambda it: _detect_pv(rule, _scenario_for(it)), items))
    # 各监督 checkpoint
    for ck in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        if not os.path.exists(ck):
            print(f"[跳过] {ck} 不存在")
            continue
        try:
            det = SupervisedConflictDetector(ck, threshold=cfg.detection.threshold)
            channels.append(_score_channel(
                os.path.basename(ck), lambda it, d=det: _detect_pv(d, _scenario_for(it)), items))
            print(f"[加载] {ck}")
        except Exception as e:  # noqa: BLE001
            print(f"[警告] {ck} 加载失败: {e}")

    print(f"\n{'通道':<22}{'top1':>7}{'exact':>7}{'jacc':>7}  逐原则 top1")
    for ch in channels:
        pp = " ".join(f"{k}={v:.2f}" if v else f"{k}=n/a" for k, v in ch["per_principle_top1"].items())
        print(f"{ch['channel']:<22}{ch['top1']:>7.3f}{ch['exact']:>7.3f}{ch['jaccard']:>7.3f}  {pp}")

    print("\n口径：MCQ 为 held-out（监督训练只用 open-ended rubric 原则标签）→ 这是 principle 头的")
    print("独立泛化数字（非训练 eval）。ERS 轴暂无独立标注集：训练 eval MAE 见 detector_v2_train.log")
    print("（0.043，属开发集自报，正文须标注）。")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"n": len(items), "channels": channels}, f, ensure_ascii=False, indent=2)
        print(f"[已保存] {args.out}")


if __name__ == "__main__":
    main()
