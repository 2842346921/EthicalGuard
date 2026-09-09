#!/usr/bin/env bash
# ============================================================
# run_mcq_corr.sh —— ① 平衡×内容 Spearman（离线秒级）② MCQ 官方答对率全量 2182（≈1.5-3h）
# 前置：conda guard 激活、代码已同步（19_balance_content_corr.py / 21_mcqa_official.py）
# ============================================================
set -u
cd "$(dirname "$0")"
echo "项目根: $(pwd)  开始: $(date '+%F %T')"

echo ""
echo "================================================================"
echo "[1] 平衡系数 ↔ 内容质量锚 Spearman（离线，无需 LLM） $(date '+%T')"
echo "================================================================"
python scripts/19_balance_content_corr.py \
    --runs runs/mane_pqa30.jsonl \
    --align runs/decision_alignment.json \
    --out runs/balance_content_corr.json 2>&1 | tee runs/balance_content_corr.txt
echo "[1 done] $?"

echo ""
echo "================================================================"
echo "[2] PrinciplismQA 官方口径 MCQ 全量 2182（≈1.5-3h，先 10 题验证可 Ctrl-C） $(date '+%T')"
echo "================================================================"
# 建议先单独小跑验证： python scripts/21_mcqa_official.py --config configs/config.yaml --limit 10
python scripts/21_mcqa_official.py --config configs/config.yaml --limit 0 \
    2>&1 | tee runs/mcqa_official.txt
echo "[2 done] $?（中断后续跑：python scripts/21_mcqa_official.py --limit 0 --resume）"

echo ""
echo "全部完成 $(date '+%F %T')"
echo "产物：runs/balance_content_corr.txt/.json（[1]） runs/mcqa_official.txt/.jsonl（[2]）"
