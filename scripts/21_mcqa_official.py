"""21_mcqa_official：PrinciplismQA 官方口径 MCQ 全量答对率（2,182 题）。

背景：14_mcq_eval 只测了"原则识别"（模型输出四原则之一），从未让模型按官方协议
答 A/B/C/D 与 correct_answer 比对——知识轨（knowledge level）的表现是空的。
本脚本补上官方口径（对齐数据集官方 mcqa_eval.py 的协议）：
  给四选项 → 模型只输出选项字母 → 与 correct_answer 比对；
  报告：总答对率（含 parsed/全量两种分母）+ 拒答/不可解析计数 +
       按 principlism 标签分组的答对率（autonomy/beneficence/nonmaleficence/justice 子集）。
结果逐题追加写 jsonl（可 --resume 断点续跑）。

用法：
  # 先小样本验证（10 题）
  python scripts/21_mcqa_official.py --config configs/config.yaml --limit 10
  # 全量 2,182（local ≈ 1.5-3h，建议 nohup）
  python scripts/21_mcqa_official.py --config configs/config.yaml --limit 0 \
      2>&1 | tee runs/mcqa_official.txt
  # 中断后续跑（跳过已有 qid）
  python scripts/21_mcqa_official.py --config configs/config.yaml --limit 0 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethicalguard.config import Config
from ethicalguard.llm import make_backend

PRINCIPLES = ["autonomy", "beneficience", "nonmaleficience", "justice"]
P_SHOW = {"autonomy": "A(自主)", "beneficience": "B(行善)", "nonmaleficience": "N(不伤害)", "justice": "J(公正)"}

SYS_PROMPT = (
    "You are answering clinical medical-ethics multiple-choice questions. "
    "Each question has four options labeled A, B, C, D, with exactly one best answer. "
    "Output ONLY the option letter (A, B, C, or D). Do not explain."
)


def _load_items(data_dir: str, limit: int):
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


def _extract_letter(text: str):
    """从模型输出提取选项字母（官方协议：只应输出 A-D）。"""
    m = re.search(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', text.strip())
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--limit", type=int, default=0, help="题数上限（0=全量 2182）")
    ap.add_argument("--out", default="runs/mcqa_official.jsonl")
    ap.add_argument("--resume", action="store_true", help="跳过 out 中已有的 qid（断点续跑）")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    backend = make_backend(cfg.llm)
    if backend.mode == "rule":
        print("[错误] llm.mode=rule 无法答 MCQ——请用 local/api。")
        sys.exit(2)
    items = _load_items(cfg.datasets.data_dir, args.limit)
    done_qids = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_qids.add(json.loads(line).get("qid"))
        print(f"[resume] 已有 {len(done_qids)} 题，跳过")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_parsed = n_correct = n_refusal = n_unparse = 0
    per_principle = {p: {"n": 0, "correct": 0, "parsed": 0} for p in PRINCIPLES}
    print(f"===== PrinciplismQA 官方口径 MCQ（{len(items)} 题，全量 2182）=====")
    with open(args.out, "a", encoding="utf-8") as f:
        for i, it in enumerate(items):
            qid = it.get("question_id") or it.get("id")
            if qid in done_qids:
                continue
            opts = it.get("options") or {}
            user = (f"Question: {it.get('question', '')}\n\n"
                    + "\n".join(f"{k}) {opts.get(k, '')}" for k in ("A", "B", "C", "D"))
                    + "\n\nAnswer (letter only):")
            try:
                resp = backend.complete(SYS_PROMPT, user).strip()
            except Exception as e:  # noqa: BLE001
                resp = ""
                print(f"[warn] qid={qid} 调用失败: {e}")
            letter = _extract_letter(resp)
            gold = str(it.get("correct_answer", "")).strip().upper()
            status = "parsed" if letter else ("refusal" if re.search(r'(?i)refus|cannot|can\'t|unable', resp) else "unparse")
            is_correct = (letter == gold) if letter else None
            if status == "parsed":
                n_parsed += 1
                if is_correct:
                    n_correct += 1
            elif status == "refusal":
                n_refusal += 1
            else:
                n_unparse += 1
            pl = it.get("principlism") or {}
            for p in PRINCIPLES:
                if pl.get(p):
                    per_principle[p]["n"] += 1
                    if status == "parsed":
                        per_principle[p]["parsed"] += 1
                        if is_correct:
                            per_principle[p]["correct"] += 1
            f.write(json.dumps({"qid": qid, "gold": gold, "pred": letter,
                                "status": status, "correct": is_correct,
                                "principlism": pl, "response": resp[:300]},
                               ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                acc = n_correct / n_parsed if n_parsed else float("nan")
                print(f"  {i + 1}/{len(items)}  parsed={n_parsed} correct={n_correct}"
                      f" acc(parsed)={acc:.3f} refusal={n_refusal} unparse={n_unparse}")

    print("\n===== 官方口径结果 =====")
    print(f"题数: {len(items)}  parsed={n_parsed}  refusal={n_refusal}  unparse={n_unparse}")
    if n_parsed:
        print(f"答对率(parsed 分母): {n_correct / n_parsed:.4f}")
    print(f"答对率(全量分母)  : {n_correct / len(items):.4f}")
    print(f"\n按 principlism 标签分组（parsed 分母）:")
    for p in PRINCIPLES:
        d = per_principle[p]
        acc = d["correct"] / d["parsed"] if d["parsed"] else float("nan")
        print(f"  {P_SHOW[p]:<10} n={d['n']:>5}  parsed={d['parsed']:>5}  acc={acc:.3f}")

    print("\n口径：与官方 mcqa_eval.py 同协议（四选一比对 correct_answer）；parsed/全量双分母；")
    print("拒答与不可解析单列。此数=知识轨表现；与 14 的原则识别轨（0.613）是不同任务，勿混报。")


if __name__ == "__main__":
    main()
