"""07_eval_types：三型评估汇总（按数据集分组）——"三型评估法"方法论 claim 的证据。

- 答案型：MANE final_proposal.rationale vs 场景 reference.keypoints → rubric_alignment
  （PrinciplismQA 开放题 / MedEthicEval / MedEthicsQA / LLMEvalMed 有 rubric/answer 参照）
- 分布型：VITAL distribution 场景（reference.kind=distribution，gold_distribution 为选项分布）
  ——相对比较：MANE 与三种基线（single_llm / harmony / rule_no_gne）对同一批分布型场景，
  用同一"选项→原则"映射得到人类原则分布，各自与人类分布算 JS/L1，横向比较谁更接近人类分布。
  （修复：旧版仅 MANE vs human 的绝对距离无参照系——human 分布本身有映射偏差，须做相对比较）
- 评审型：Judge（LLM-as-judge，有 LLM 后端用 LLM；rule 模式关键词兜底）对 rationale 打分

用法：
  python scripts/07_eval_types.py --config configs/config.yaml \
      --input data_cache/scenarios.jsonl --runs runs/mane_results.jsonl
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
from ethicalguard.eval import metrics as M
from ethicalguard.eval.baselines import harmony_style_baseline, rule_collective_weights, single_llm_baseline
from ethicalguard.eval.judges import Judge
from ethicalguard.llm import make_backend
from ethicalguard.utils import setup_logging

logger = setup_logging()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", required=True, help="场景 jsonl（01 产物）")
    ap.add_argument("--runs", default=None, help="MANE 结果 jsonl（默认 config.run.out_dir/mane_results.jsonl）")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    runs_path = args.runs or os.path.join(cfg.run.out_dir, "mane_results.jsonl")
    backend = make_backend(cfg.llm)
    judge = Judge(backend)

    results = {}
    from ethicalguard.data.loaders import iter_negotiation_records
    for r in iter_negotiation_records(runs_path):
        results[r["scenario_id"]] = r

    from collections import defaultdict
    answer_align = defaultdict(list)      # dataset -> rubric_alignment
    judge_scores = defaultdict(list)      # dataset -> judge score
    vital_dist = []                       # VITAL 分布型场景（各方法原则向量 + gold 摘要）
    vital_js = defaultdict(list)          # 分布型：方法 -> JS 距离列表（相对比较）
    vital_l1 = defaultdict(list)          # 分布型：方法 -> L1 距离列表

    # 相对比较的方法集合：MANE + 三种基线（06_baselines 同源实现）
    DIST_METHODS = ("MANE", "single_llm", "harmony", "rule_no_gne")

    n_rubric = n_judged = 0
    for sc in load_scenarios_from_jsonl(args.input):
        r = results.get(sc.scenario_id)
        if r is None or r.get("final_proposal") is None:
            continue
        ds = sc.source.get("dataset", "?")
        rationale = _answer_rationale(r)
        ref = sc.reference
        kind = ref.kind if ref else "none"
        keypoints = (ref.content or {}).get("keypoints", []) if ref else []

        # 答案型：有 rubric keypoints 的场景
        if kind == "rubric" and keypoints:
            answer_align[ds].append(M.rubric_alignment(rationale, keypoints))
            n_rubric += 1
        # 评审型：对所有有 rationale 的场景打分
        if rationale:
            judge_scores[ds].append(judge.score(rationale, keypoints or None))
            n_judged += 1
        # 分布型：VITAL gold_distribution → 选项文本 → 原则向量 → gold 加权 → JS/L1
        # 相对比较：同一批场景、同一"选项→人类原则分布"映射，各方法分别算 JS/L1，
        # 横向比较"谁更接近人类分布"（旧版仅 MANE 绝对距离无参照系——修复）。
        if ds == "vital" and kind == "distribution":
            gold = (ref.content or {}).get("gold_distribution") or []
            opts = (ref.content or {}).get("options") or []
            human_pv = _gold_to_principle_distribution(opts, gold)
            methods_pv = _method_principle_vectors(r, sc)
            vital_dist.append({
                "scenario_id": sc.scenario_id,
                **{f"{k}_principle": [round(float(x), 3) for x in v] for k, v in methods_pv.items()},
                "human_principle": [round(float(x), 3) for x in human_pv] if human_pv is not None else None,
                "gold_distribution": gold,
            })
            if human_pv is not None:
                for name, pv in methods_pv.items():
                    vital_js[name].append(M.js_distance(pv, human_pv))
                    vital_l1[name].append(M.l1_distance(pv, human_pv))

    print("===== 答案型：rubric 对齐（rationale vs 专家 keypoints）=====")
    print(f"{'数据集':<14}{'样本':>6}{'rubric对齐':>10}")
    for ds in sorted(answer_align):
        v = answer_align[ds]
        print(f"{ds:<14}{len(v):>6}{np.mean(v):>10.3f}")

    print("\n===== 评审型：Judge 打分（LLM-as-judge 或规则兜底）=====")
    print(f"{'数据集':<14}{'样本':>6}{'评审分':>8}")
    for ds in sorted(judge_scores):
        v = judge_scores[ds]
        print(f"{ds:<14}{len(v):>6}{np.mean(v):>8.3f}")

    print("\n===== 分布型：VITAL 相对比较（各方法 vs 人类原则分布，同一映射）=====")
    print(f"VITAL 分布型场景 {len(vital_dist)} 个。")
    print("映射：选项文本 → 规则四盒 → 原则向量，按 gold_distribution 加权合成'人类原则分布'；")
    print("MANE 与三种基线用同一映射，各自与人类分布算 JS/L1——比较相对接近度（§4.3 分布型落地）。")
    has = {k: len(v) for k, v in vital_js.items()}
    # 分布型降级检查（P0-3 落地：最快方案 = 降级探索 + 坍缩分析）：
    # ① human 参照坍缩：各场景 human 原则分布若全相同 → JS 比较退化为"离固定点距离"，
    #    结论只作探索性（旧版 6 场景 human 恒 [0.115,0.385,0.115,0.385] 即此类）；
    # ② 场景过少（<10）同样只作探索性报告。
    _harr = [np.asarray(d.get("human_principle"), dtype=float) for d in vital_dist
             if d.get("human_principle") is not None]
    collapse = len(_harr) > 1 and float(np.max(np.std(np.stack(_harr), axis=0))) < 1e-9
    _golds = [tuple(d.get("gold_distribution") or []) for d in vital_dist]
    gold_collapse = len(_golds) > 1 and len(set(_golds)) == 1
    exploratory = len(vital_dist) < 10 or collapse
    if any(has.values()):
        print(f"{'方法':<12}{'JS距离':>9}{'L1距离':>9}{'样本':>5}")
        print("-" * 38)
        for name in DIST_METHODS:
            if not vital_js[name]:
                continue
            print(f"{name:<12}{np.mean(vital_js[name]):>9.4f}{np.mean(vital_l1[name]):>9.4f}{len(vital_js[name]):>5}")
        # 相对比较结论：MANE 是否优于最佳基线（同映射下 JS 更小）
        if vital_js["MANE"]:
            baselines_js = {k: np.mean(v) for k, v in vital_js.items() if k != "MANE" and v}
            if baselines_js:
                best_bl, best_val = min(baselines_js.items(), key=lambda kv: kv[1])
                mane_val = np.mean(vital_js["MANE"])
                delta = mane_val - best_val
                if delta < 0:
                    print(f"结论：MANE JS {mane_val:.4f} < 最佳基线 {best_bl} {best_val:.4f}"
                          f"（降 {abs(delta):.4f}）——协商在分布型上也优于基线。")
                else:
                    print(f"结论：MANE JS {mane_val:.4f} ≥ 最佳基线 {best_bl} {best_val:.4f}"
                          f"（差 {delta:.4f}）——注意分布型场景少，属提示性结论。")
        if exploratory:
            print(f"  ⚠ 分布型证据**降级为探索性**："
                  + ("人类参照坍缩（各场景 human 原则分布相同）→ JS 比较退化为'离固定点距离'"
                     if collapse else f"场景数 {len(vital_dist)} < 10")
                  + "——仅供趋势参考，不作主结论（正文须按此口径表述）。")
            if gold_collapse:
                print("     （gold_distribution 亦全相同——二选一题标签退化，参照信息量有限）")
    else:
        print("  （无场景可映射——选项文本过短/无信号，报告限制）")
    if vital_dist:
        print("  样本对比（各方法 vs 人类原则分布）:")
        for d in vital_dist[:5]:
            print("  ", d["scenario_id"], "MANE=", d["MANE_principle"],
                  "single_llm=", d["single_llm_principle"],
                  "harmony=", d["harmony_principle"],
                  "rule_no_gne=", d["rule_no_gne_principle"],
                  "human=", d["human_principle"], "gold=", d["gold_distribution"])

    print(f"\n覆盖统计：rubric 对齐 {n_rubric} 场景 / 评审 {n_judged} 场景 / 分布型 {len(vital_dist)} 场景")


def _answer_rationale(r: dict) -> str:
    """答案型/评审型评估取"最终共识的伦理论证"，而非程序文本或异议者。

    修复两处：
    ① 仲裁场景旧版 final_proposal.rationale 是裁决轮 "CAMP: ... 资源 L1 否决" 程序摘要，
       对程序文本做 rubric/Judge 评估恒 ≈0；
    ② 未仲裁场景旧版 final_proposal 是 **catfish 异议**（trajectory append 顺序
       proposals+[dissent]，鲶鱼恒在最后 → proposals[-1]=catfish）——共识提案取错对象。
    统一：回溯最后一个多方协商轮（≥2 个非 catfish 提案），优先取伦理委员会提案的
    rationale 作为最终论证。兼容旧 runs 与新版（02 修复后 final_proposal 已正确）。
    """
    fp = r.get("final_proposal") or {}
    rat = fp.get("rationale", "")
    fp_agent = fp.get("agent")
    # 新版 02 已修复（final_proposal=协商轮 ethics_committee）→ 直接用；
    # 旧 runs：仲裁程序文本 或 catfish 文本 → 回溯协商轮
    if not r.get("arbitration_triggered") and "CAMP:" not in rat and fp_agent != "catfish":
        return rat
    for tr in reversed(r.get("trajectory", [])):
        props = [p for p in tr.get("proposals", []) if p.get("agent") != "catfish"]
        if len(props) >= 2:
            return next((p.get("rationale", "") for p in props if p.get("agent") == "ethics_committee"), "")
    return rat


def _method_principle_vectors(r: dict, sc) -> dict:
    """同一批场景上各方法的四原则向量（归一化）。MANE 取 final_vector，基线为规则近似。"""
    v = r.get("final_vector") or {}
    mane = np.array([v.get("beneficence", 0), v.get("nonmaleficence", 0),
                     v.get("autonomy", 0), v.get("justice", 0)], dtype=float)
    single = single_llm_baseline(sc).principle_weights.as_array()
    harmony = harmony_style_baseline(sc)[-1].principle_weights.as_array()
    rule = rule_collective_weights(sc)
    out = {}
    for name, pv in (("MANE", mane), ("single_llm", single),
                     ("harmony", harmony), ("rule_no_gne", rule)):
        p = np.asarray(pv, dtype=float)
        s = p.sum()
        out[name] = p / s if s > 1e-12 else np.zeros(4)
    return out


def _gold_to_principle_distribution(options, gold_distribution):
    """VITAL 分布型落地：选项文本 → 规则四盒 → 原则向量，按 gold 分布加权合成'人类原则分布'。

    :param options: 选项文本列表（如 ["I follow surgical procedures...", "I make a mistake..."]）
    :param gold_distribution: 人类选择分布（与 options 等长，如 [1, 0]）
    :returns: (4,) 归一化原则向量；选项无法映射时返回 None
    """
    from ethicalguard.data.mapping_rules import map_text_to_state
    if not options or not gold_distribution or len(options) != len(gold_distribution):
        return None
    vecs = []
    weights = []
    for opt, g in zip(options, gold_distribution):
        w = float(g)
        if w <= 0:
            continue
        st = map_text_to_state(str(opt))
        pv = np.array([st.medical.get("severity", 0.0) * 0.5 + 0.3,
                       1.0 - st.medical.get("severity", 0.5),
                       st.preference.get("clarity", 0.5),
                       0.5 + 0.5 * (1.0 - st.cci())], dtype=float)
        pv = np.clip(pv, 0.0, 1.0)
        if pv.sum() <= 1e-6:
            continue
        vecs.append(pv)
        weights.append(w)
    if not vecs:
        return None
    total = sum(weights)
    human = sum(w * v for w, v in zip(weights, vecs)) / total
    s = human.sum()
    return human / s if s > 0 else None


if __name__ == "__main__":
    main()
