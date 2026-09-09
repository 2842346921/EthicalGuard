#!/usr/bin/env bash
# ============================================================
# run_extra.sh —— 补充批：24 rationale 扩展 / 25 案例挖潜 / 26 异构 judge
# 前置：conda guard 激活；代码已同步（content_judges.py + 24/25/26 + sh）；
#       runs/cross_method_texts.jsonl & scores.jsonl 已存在（实验四产物）。
# 耗时：24 ≈ 20-40min（60 次 judge ×2 尺子）；25 秒级；26 ≈ 20-40min（118 次 judge）。
# 异构 judge 服务需先起（见下方 [26] 提示；默认端口 8010，模型名 hetero-judge）。
# ============================================================
set -u
cd "$(dirname "$0")"
echo "项目根: $(pwd)  开始: $(date '+%F %T')"

JUDGE_PORT="${JUDGE_PORT:-8010}"
JUDGE_MODEL="${JUDGE_MODEL:-hetero-judge}"

echo ""
echo "================================================================"
echo "[24] MANE rationale 输出充分性（short=委员会 vs full=完整共识论证） $(date '+%T')"
echo "================================================================"
python scripts/24_mane_rationale_expand.py --config configs/config.yaml \
    --input data_cache/scenarios_pqa30.jsonl \
    --mane-runs runs/mane_pqa30.jsonl \
    --texts runs/cross_method_texts.jsonl \
    --scores runs/cross_method_scores.jsonl \
    --out runs/mane_rationale_expand.jsonl 2>&1 | tee runs/mane_rationale_expand.txt
echo "[24 done] $?"

echo ""
echo "================================================================"
echo "[25] 案例挖潜（自动优势/劣势 + 指定案例，离线） $(date '+%T')"
echo "================================================================"
python scripts/25_case_study.py \
    --input data_cache/scenarios_pqa30.jsonl \
    --texts runs/cross_method_texts.jsonl \
    --scores runs/cross_method_scores.jsonl \
    --cases PQ-474-931,PQ-369-758 \
    --out runs/case_study.md 2>&1 | tee runs/case_study.log
echo "[25 done] $?"

echo ""
echo "================================================================"
echo "[26] 异构 judge 自评偏置检验（需 judge 服务在 :$JUDGE_PORT） $(date '+%T')"
echo "================================================================"
if curl -s -m 3 "http://127.0.0.1:${JUDGE_PORT}/v1/models" > /dev/null 2>&1; then
    python scripts/26_hetero_judge.py --config configs/config.yaml \
        --input data_cache/scenarios_pqa30.jsonl \
        --texts runs/cross_method_texts.jsonl \
        --self-scores runs/cross_method_scores.jsonl \
        --judge-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
        --judge-model "${JUDGE_MODEL}" \
        --judge-name "${JUDGE_MODEL}" \
        --out "runs/hetero_judge_${JUDGE_MODEL}.jsonl" \
        2>&1 | tee "runs/hetero_judge_${JUDGE_MODEL}.txt"
    echo "[26 done] $?"
else
    echo "[26 跳过] judge 服务未在 :${JUDGE_PORT} —— 先执行（guard env；8003 已占 GPU0 时用 GPU1："
    echo "  CUDA_VISIBLE_DEVICES=1 nohup vllm serve /mnt/model/Meta-Llama-3.1-8B-Instruct \\
      --served-model-name ${JUDGE_MODEL} --port ${JUDGE_PORT} \\
      --gpu-memory-utilization 0.9 --max-model-len 16384 \\
      > logs/vllm_hetero_judge.log 2>&1 &"
    echo "  起好后重跑：JUDGE_PORT=${JUDGE_PORT} bash run_extra.sh）"
fi

echo ""
echo "全部完成 $(date '+%F %T')"
echo "产物：runs/mane_rationale_expand.txt/.jsonl（[24]） runs/case_study.md（[25]）"
echo "      runs/hetero_judge_*.txt/.jsonl（[26]）"
