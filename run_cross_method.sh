#!/usr/bin/env bash
# ============================================================
# run_cross_method.sh —— 实验四：同题·同尺子·跨方法内容对比 + PQA 官方同尺子定位
# 前置：conda guard 激活、代码已同步（eval/baselines/llm_baselines.py + 11/22/23 + sh）
# 耗时：Step1 ≈ 2-3h（3 基线×30 场景协商 + 4 方法×30 场景×3 次 judge）；
#       Step2 离线秒级。断点续跑：22 --skip-existing 即可。
# ============================================================
set -u
cd "$(dirname "$0")"
echo "项目根: $(pwd)  开始: $(date '+%F %T')"

echo ""
echo "================================================================"
echo "[1] 22 内容跨方法对比（同题同尺子；MANE 复用 mane_pqa30.jsonl 省 ~50min） $(date '+%T')"
echo "================================================================"
python scripts/22_content_cross_method.py --config configs/config.yaml \
    --input data_cache/scenarios_pqa30.jsonl \
    --mane-runs runs/mane_pqa30.jsonl \
    --texts-out runs/cross_method_texts.jsonl \
    --scores-out runs/cross_method_scores.jsonl \
    2>&1 | tee runs/cross_method_run.txt
echo "[1 done] $?（中断后续跑：上面命令加 --skip-existing）"

echo ""
echo "================================================================"
echo "[2] 23 PQA 官方同尺子对比 + 定位声明（离线） $(date '+%T')"
echo "================================================================"
python scripts/23_pqa_official_contrast.py \
    --texts runs/cross_method_texts.jsonl \
    --scores runs/cross_method_scores.jsonl \
    --out runs/pqa_official_contrast.txt 2>&1 | tee runs/pqa_official_contrast.log
echo "[2 done] $?"

echo ""
echo "全部完成 $(date '+%F %T')"
echo "产物：runs/cross_method_texts.jsonl / cross_method_scores.jsonl / cross_method_run.txt"
echo "      runs/pqa_official_contrast.txt（① 官方尺子内容分 ② 配对差 ③ 定位声明 ④ 保证层）"
