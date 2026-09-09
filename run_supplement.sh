#!/usr/bin/env bash
# ============================================================
# run_supplement.sh —— 补充实验统一运行（2026-09-07 架构修正批）
# 覆盖：P0(编码修复重出 / E2 显著性 / 分布型降级落地) + P1(17c 守护座位 / MCQ v2 /
#       detector held-out / 20 决策对齐)
# 前置：conda guard 环境已激活（或本脚本内 source）、在项目根目录执行、
#       代码已同步（negotiation/gne_solver/config/loaders + 新脚本 05b/17c/18/20）。
# 耗时：步骤 1-4 ≈ 10-20 分钟；步骤 5（17c）≈ 3h（22 场景 × 4 座位 local LLM）。
# ============================================================
set -u
cd "$(dirname "$0")"
echo "项目根: $(pwd)  开始: $(date '+%F %T')"

# 若未在 conda 环境内，自动 source（按需放开）
# source /home/qluai/miniconda3/etc/profile.d/conda.sh && conda activate guard

run_step() {
  echo ""
  echo "================================================================"
  echo "[$1] $2   $(date '+%T')"
  echo "================================================================"
}

# ---------- [0] E2 显著性（离线，无 LLM）----------
run_step "0" "E2 配对显著性（bootstrap CI + McNemar）"
python scripts/18_e2_significance.py --runs runs/real_llm_baselines.jsonl \
    --n-boot 20000 --seed 42 --out runs/e2_significance.json 2>&1 | tee runs/e2_significance.txt
echo "[0 done] $?"

# ---------- [1] eval 编码修复后重出正式产物 ----------
run_step "1" "重出评估正式产物（编码修复验证）"
python scripts/04_eval.py --config configs/config.yaml --runs runs/mane_pqa30.jsonl \
    > runs/eval4_pqa30.txt 2>&1
echo "  04(PQA30) exit=$? → runs/eval4_pqa30.txt"
python scripts/07_eval_types.py --config configs/config.yaml \
    --input data_cache/scenarios_pqa30.jsonl --runs runs/mane_pqa30.jsonl \
    > runs/eval_pqa30.txt 2>&1
echo "  07(PQA30 rubric/Judge) exit=$? → runs/eval_pqa30.txt"
python scripts/07_eval_types.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --runs runs/mane_results.jsonl \
    > runs/eval_types_full29.txt 2>&1
echo "  07(29 场景全三型含分布型降级/坍缩提示) exit=$? → runs/eval_types_full29.txt"
echo "[1 done]"

# ---------- [2] MCQ 三通道收尾（detector_v2 held-out）----------
run_step "2" "MCQ 三通道：rule + supervised(detector_v2.pt) + llm_mcq"
python scripts/14_mcq_eval.py --config configs/config.yaml \
    --checkpoint runs/detector_v2.pt \
    --channels rule,supervised,llm --llm-subset 300 \
    --out runs/mcq_eval_v2.json 2>&1 | tee runs/mcq_eval_v2.txt
echo "[2 done] $?"

# ---------- [3] 监督检测器 held-out 泛化对比（v1 vs v2 + 规则）----------
run_step "3" "detector held-out（MCQ 2182 独立泛化）"
python scripts/05b_eval_detector.py --config configs/config.yaml \
    --checkpoints runs/detector.pt,runs/detector_v2.pt \
    --out runs/detector_heldout.json 2>&1 | tee runs/detector_heldout.txt
echo "[3 done] $?"

# ---------- [4] 决策对齐：MANE 开放回答 vs PrinciplismQA 标准答案 ----------
run_step "4" "决策对齐（PQA30，30 次 judge 调用）"
python scripts/20_decision_alignment.py --config configs/config.yaml \
    --input data_cache/scenarios_pqa30.jsonl --runs runs/mane_pqa30.jsonl \
    --out runs/decision_alignment.json 2>&1 | tee runs/decision_alignment.txt
echo "[4 done] $?"

# ---------- [5] 17c 守护座位对比（长任务，LLM 大头）----------
run_step "5" "17c 守护座位对比（22 场景 × 4 座位 ≈ 3h）"
python scripts/17c_catfish_seat.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --mode local --limit 20 --include-collusion \
    2>&1 | tee runs/catfish_seat.txt
echo "[5 done] $?"

echo ""
echo "全部完成 $(date '+%F %T')"
echo "产物："
echo "  runs/e2_significance.txt/.json      ([0] E2 显著性)"
echo "  runs/eval4_pqa30.txt                ([1] PQA30 协商质量正式版)"
echo "  runs/eval_pqa30.txt                 ([1] PQA30 rubric/Judge 正式版)"
echo "  runs/eval_types_full29.txt          ([1] 29 场景三型 + 分布型降级/坍缩提示)"
echo "  runs/mcq_eval_v2.txt/.json          ([2] MCQ 三通道 v2)"
echo "  runs/detector_heldout.txt/.json     ([3] 监督 held-out 泛化)"
echo "  runs/decision_alignment.txt/.json   ([4] 决策 vs 标准答案)"
echo "  runs/catfish_seat.txt               ([5] 守护座位对比)"
