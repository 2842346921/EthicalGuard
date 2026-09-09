"""A 轨：PrinciplismQA MCQ（2182 题）原则识别评估——规则 vs 监督 双通道。

MCQ 是"哪个原则最受挑战/最相关"的选择题（选项=四原则名，gold=principlism 多标签）。
用识别器（detector）通道评估：MCQ 文本 → 规则四盒映射 → ConflictReport.principle_vector
→ argmax 原则 vs gold principlism（多标签 → 用预测 top-1 是否在 gold 集 / 多标签匹配两种口径）。

注意：监督模型只在 open-ended-rubric 的 principles 上训练过（dataset.py），**MCQ 从未进训练**
= held-out 泛化测试。规则通道零成本基线。

用法：
  python scripts/14_mcq_eval.py --config configs/config.yaml \
      [--checkpoint runs/detector.pt] [--limit 500] [--out runs/mcq_eval.txt]
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

# 原则名 → 索引（与 PrincipleVector 的 [B,N,A,J] 顺序一致）；显示用中文短名
P_NAMES = ["B(行善)", "N(不伤害)", "A(自主)", "J(公正)"]
# MCQ principlism 标注的键名（数据集拼写：beneficience/nonmaleficience）
MCQ_KEYS = ["beneficience", "nonmaleficience", "autonomy", "justice"]  # 注意拼写


def _load_mcq(data_dir: str, limit: int):
    # 本地/服务器路径探测：优先本地项目 data/，其次配置 data_dir
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        data_dir,
    ]
    p = None
    for base in candidates:
        cand = os.path.join(base, "PrinciplismQA", "data", "knowledge-mcqa.json")
        if os.path.exists(cand):
            p = cand
            break
    if p is None:
        raise FileNotFoundError(f"未找到 knowledge-mcqa.json（尝试: {candidates}）")
    with open(p, encoding="utf-8") as f:
        items = json.load(f)
    if limit > 0:
        items = items[:limit]
    return items


def _gold_vector(principlism: dict) -> np.ndarray:
    """MCQ principlism 标注 → [B,N,A,J] 多标签。"""
    return np.array([
        1.0 if principlism.get("beneficience") or principlism.get("beneficence") else 0.0,
        1.0 if principlism.get("nonmaleficience") or principlism.get("nonmaleficence") else 0.0,
        1.0 if principlism.get("autonomy") else 0.0,
        1.0 if principlism.get("justice") else 0.0,
    ])


# ---- MCQ 专用 LLM 答题器（修复：旧版用冲突检测器答 MCQ = 任务错配）----
# MCQ 问"哪个原则最被挑战/最相关"，是四选一原则选择题；冲突检测器问"病例有什么冲突"
# 是不同任务。这里给 LLM 正确的 few-shot 原则题 prompt，直接输出原则名。
MCQ_LLM_SYS = (
    "You are answering clinical ethics principle-selection questions. "
    "Each question asks which ethical principle (Autonomy / Beneficence / Non-maleficence / Justice) "
    "is most challenged, most relevant, most violated, or should guide the action. "
    "Read the question carefully: 'most challenged' asks which principle is under threat "
    "(not which one should win). Output ONLY the principle name from the four options.\n\n"
    "Example 1: Q: Which ethical principle is most challenged in balancing patient confidentiality "
    "with public health during COVID-19? Options: A: Autonomy B: Non-maleficence C: Justice D: Beneficence\n"
    "A: Autonomy\n\n"
    "Example 2: Q: Which ethical principle should guide a physician when a patient insists on unproven "
    "alternative medicine that may delay effective treatment? Options: A: Autonomy B: Beneficence C: Justice D: Nonmaleficence\n"
    "B: Beneficence\n\n"
    "Now answer the question below. Output ONLY the principle name."
)

_MCQ_NAME2IDX = {
    "autonomy": 2, "beneficence": 0, "beneficience": 0, "non-maleficence": 1,
    "nonmaleficence": 1, "non_maleficence": 1, "justice": 3, "respect for persons": 2,
}


def _llm_mcq_answer(backend, question: str, options_text: str) -> np.ndarray:
    """专用 MCQ 答题：LLM 直接选原则名 → one-hot [B,N,A,J]。
    修复旧 bug：冲突检测器输出 principle_vector 但那是冲突分析不是答题。
    """
    from ethicalguard.mane.agents.base import _extract_json  # noqa: F401 (warm)
    user = f"Question: {question}\nOptions: {options_text}\nAnswer (principle name):"
    try:
        text = backend.complete(MCQ_LLM_SYS, user).strip()
        # 提取原则名（容错：可能带选项字母/标点）
        low = text.lower()
        name = None
        for cand in ("autonomy", "beneficence", "beneficience", "non-maleficence",
                     "nonmaleficence", "non_maleficence", "justice", "respect for persons"):
            if cand in low:
                name = cand
                break
        if name is None:
            return np.full(4, 0.25)  # 解析失败 → 中性
        idx = _MCQ_NAME2IDX[name]
        v = np.zeros(4)
        v[idx] = 1.0
        return v
    except Exception:
        return np.full(4, 0.25)


def _mcq_scenario(it: dict, text_mapper) -> Scenario:
    """MCQ 题 → Scenario（规则映射四盒状态，供识别器 detect）。"""
    text = f"{it.get('question','')}\n{str(it.get('options',{}))}"
    st = map_text_to_state(text)
    return Scenario(
        scenario_id=f"PQ-MCQ-{it.get('question_id') or it.get('id')}",
        raw_text=text, state=st,
        constraints=default_constraints(st), parties=default_parties(st),
    )


def _pred_top1(pv) -> int:
    """principle_vector → argmax 原则索引（单选口径）。pv 为 PrincipleVector 或 dict。"""
    if hasattr(pv, "beneficence"):  # PrincipleVector
        arr = np.array([pv.beneficence, pv.nonmaleficence, pv.autonomy, pv.justice])
    else:
        arr = np.array([pv.get("beneficence", 0), pv.get("nonmaleficence", 0),
                        pv.get("autonomy", 0), pv.get("justice", 0)])
    return int(np.argmax(arr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default=None, help="监督模型权重（None=只跑规则通道）")
    ap.add_argument("--channels", default="rule,supervised,llm",
                    help="逗号分隔通道（rule/supervised/llm；默认三通道全跑；无权重/无后端自动跳过）")
    ap.add_argument("--limit", type=int, default=0, help="MCQ 上限（0=全部 2182）")
    ap.add_argument("--out", default=None, help="结果保存路径（json）")
    ap.add_argument("--n-errors", type=int, default=5, help="每通道打印错误样例数")
    ap.add_argument("--llm-subset", type=int, default=200,
                    help="LLM 通道每题一次调用较贵，默认只跑前 N 题（0=全部）")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data_dir = cfg.datasets.data_dir or r"E:\信息\论文\医疗诊断\多目标压力\论文\数据集"
    items = _load_mcq(data_dir, args.limit)
    want = [c.strip() for c in args.channels.split(",") if c.strip()]
    print(f"===== A 轨 MCQ 原则识别（{len(items)} 题，通道: {want}）=====")

    # 通道构造（按可用性自动跳过）
    detectors = {}
    from ethicalguard.detection.rule import RuleConflictDetector
    if "rule" in want:
        detectors["rule"] = RuleConflictDetector()
    if "supervised" in want and args.checkpoint and os.path.exists(args.checkpoint):
        from ethicalguard.detection.supervised import SupervisedConflictDetector
        try:
            detectors["supervised"] = SupervisedConflictDetector(args.checkpoint, threshold=cfg.detection.threshold)
            print(f"[监督通道] {os.path.basename(args.checkpoint)} 加载成功")
        except Exception as e:  # noqa: BLE001
            print(f"[警告] 监督模型加载失败（{e}）→ 跳过")
    if "llm" in want:
        from ethicalguard.llm import make_backend
        backend = make_backend(cfg.llm)
        if backend.mode != "rule":
            # 专用 MCQ 答题器（修复任务错配：冲突检测器不适用于原则选择题）
            detectors["llm_mcq"] = backend
            print(f"[LLM-MCQ 通道] backend={backend.mode}，few-shot 原则题直接答题"
                  f"（subset={args.llm_subset}，每题 1 次调用）")
        else:
            print("[警告] llm.mode=rule → LLM 通道跳过")

    # 评估
    results = {}
    for ch, det in detectors.items():
        top1_hit = 0          # 单选口径：预测 top-1 ∈ gold 集
        exact = 0             # 多标签精确匹配
        jacc = []             # 多标签 Jaccard
        per_principle = np.zeros(4)   # 逐原则 top1 命中
        per_principle_n = np.zeros(4)
        n_gold_valid = 0
        err_samples = []
        is_llm_mcq = ch == "llm_mcq"
        pool = items if not is_llm_mcq else items[:args.llm_subset] if args.llm_subset > 0 else items
        for it in pool:
            gold = _gold_vector(it.get("principlism", {}))
            if gold.sum() == 0:
                continue
            n_gold_valid += 1
            if is_llm_mcq:
                # LLM-MCQ：直接答题 → one-hot 向量
                pv_arr = _llm_mcq_answer(det, it.get("question", ""), str(it.get("options", {})))
            else:
                sc = _mcq_scenario(it, None)
                report = det.detect(sc)
                pv = report.principle_vector
                pv_arr = np.array([pv.beneficence, pv.nonmaleficence, pv.autonomy, pv.justice])
            top1 = int(np.argmax(pv_arr))
            pred = (pv_arr > 0.5).astype(float)
            if gold[top1] == 1.0:
                top1_hit += 1
            # 逐原则：gold 含该原则且 top1 命中
            for gi in range(4):
                if gold[gi] == 1.0:
                    per_principle_n[gi] += 1
                    if top1 == gi:
                        per_principle[gi] += 1
            if np.array_equal(pred, gold):
                exact += 1
            # Jaccard
            inter = float(np.logical_and(pred, gold).sum())
            union = float(np.logical_or(pred, gold).sum())
            jacc.append(inter / union if union > 0 else 1.0)
            if top1 != int(np.argmax(gold)) and len(err_samples) < args.n_errors:
                err_samples.append({"q": it.get("question", "")[:120],
                                    "gold": P_NAMES[int(np.argmax(gold))],
                                    "pred": P_NAMES[top1],
                                    "pv": np.round(pv_arr, 2).tolist()})
        results[ch] = {
            "top1": top1_hit / n_gold_valid if n_gold_valid else 0,
            "exact": exact / n_gold_valid if n_gold_valid else 0,
            "jaccard": float(np.mean(jacc)) if jacc else 0,
            "n": n_gold_valid,
            "per_principle_top1": {P_NAMES[i]: (per_principle[i] / per_principle_n[i]
                                                if per_principle_n[i] else None)
                                   for i in range(4)},
        }
        print(f"\n[{ch} 通道] 有效题 {n_gold_valid}")
        print(f"  top-1 命中率（预测主原则 ∈ gold 原则集）: {results[ch]['top1']:.3f}")
        print(f"  多标签精确匹配率: {results[ch]['exact']:.3f}")
        print(f"  多标签 Jaccard: {results[ch]['jaccard']:.3f}")
        pp = results[ch]["per_principle_top1"]
        print(f"  逐原则 top1: " + " ".join(f"{k}={v:.2f}" if v else f"{k}=n/a" for k, v in pp.items()))
        if err_samples:
            print(f"  错误样例（前 {len(err_samples)}）:")
            for e in err_samples:
                print(f"    gold={e['gold']} pred={e['pred']} pv={e['pv']}")
                print(f"      Q: {e['q']}")

    # 规则 vs 监督对比（若有）
    if "supervised" in results and "rule" in results:
        print("\n===== 通道对比 =====")
        for m in ("top1", "jaccard"):
            print(f"  {m}: rule={results['rule'][m]:.3f} vs supervised={results['supervised'][m]:.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[已保存] {args.out}")


if __name__ == "__main__":
    main()
