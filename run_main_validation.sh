#!/bin/bash
# run_main_validation.sh — EthicalGuard 统一主验证（2026-09-05 整合版）
#
# 范围（不含消融 08 / 韧性 03 / GNE 组合对比 16）：
#   [1] 15 底线敏感性  —— 乘子激活曲线（审稿 T1："底线真工作过吗"）
#   [2] 14 MCQ 三通道  —— 识别器能力 + llm_mcq 修正（任务错配修复验证）
#   [3] 17 catfish 验证 —— 真异议 prompt + 路 A maximin（审稿 T2："catfish 博弈结构"）
#   [4] PrinciplismQA 抽 300（B 轨主战场）
#   [5] 02 协商 300（local）
#   [6] 07 三型评估（rubric 对齐 + Judge + 分布）
#   [7] 04 协商质量汇总（闭环证据链）
#
# 用法: nohup bash run_main_validation.sh > logs/main_validation.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source /home/qluai/miniconda3/etc/profile.d/conda.sh
conda activate guard
export ETHICALGUARD_DATA_DIR="${ETHICALGUARD_DATA_DIR:-/mnt/zhangheng2025/EthicalGuard/data}"

echo "===== [0] 环境检查 $(date) ====="
curl -s --max-time 8 http://localhost:8003/v1/models > /dev/null \
  && echo "Qwen3-8B vLLM (8003) OK" \
  || { echo "错误：8003 未就绪，先起 python scripts/00_serve_local.py --port 8003"; exit 1; }

echo "===== [1] 15 底线敏感性（35 场景 × floor 0.15→0.30）$(date) ====="
if [ -s runs/floor_sensitivity.txt ] && grep -q "乘子激活场景" runs/floor_sensitivity.txt 2>/dev/null; then
  echo "已存在 runs/floor_sensitivity.txt → 跳过（想重跑请先删该文件）"
else
  python scripts/15_floor_sensitivity.py --config configs/config.yaml \
      --mode local --input data_cache/scenarios.jsonl --limit 35 \
      2>&1 | tee runs/floor_sensitivity.txt
  echo "[1 done] $(date)"
fi

echo "===== [2] 14 MCQ 三通道（rule+supervised 2182，llm_mcq 300）$(date) ====="
python scripts/14_mcq_eval.py --config configs/config.yaml \
    --channels rule,supervised,llm --checkpoint runs/detector.pt \
    --limit 2182 --llm-subset 300 --n-errors 5 \
    --out runs/mcq_eval_fixed2.json 2>&1 | tee runs/mcq_eval_fixed2.txt
echo "[2 done] $(date)"

echo "===== [3] 17 catfish 验证（20 真实 + 2 合谋，local）$(date) ====="
python scripts/17_catfish_gne.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --mode local --limit 20 \
    --include-collusion 2>&1 | tee runs/catfish_gne.txt
echo "[3 done] $(date)"

echo "===== [4] PrinciplismQA 随机抽样 30 议题（rule 映射秒级）$(date) ====="
# ⚠️ --mapping rule：local 模式默认走 dual（每场景 2 次 LLM 映射），300 场景会卡几小时。
# 抽样验证用 rule 映射（秒级）；协商阶段（02）仍是 local LLM，只影响四盒状态精度。
python scripts/01_prepare.py --config configs/config.yaml \
    --datasets principlismqa --limit 30 --shuffle --seed 42 --mapping rule \
    --out data_cache/scenarios_pqa30.jsonl 2>&1 | tail -2
python - <<'PY'
import json
scs = [json.loads(l) for l in open('data_cache/scenarios_pqa30.jsonl', encoding='utf-8')]
print('PQA30 场景数:', len(scs))
print('带 rubric keypoints:', sum(1 for s in scs if (s.get('reference') or {}).get('content', {}).get('keypoints')))
PY

echo "===== [5] 02 协商 30（local，预计 40-80min）$(date) ====="
python scripts/02_run_mane.py --config configs/config.yaml \
    --input data_cache/scenarios_pqa30.jsonl --mode local --limit 30 \
    --out runs/mane_pqa30.jsonl 2>&1 | tail -3
echo "[5 done] $(date)"

echo "===== [6] 07 三型评估（rubric + Judge + 分布）$(date) ====="
python scripts/07_eval_types.py --config configs/config.yaml \
    --input data_cache/scenarios_pqa30.jsonl --runs runs/mane_pqa30.jsonl \
    2>&1 | tee runs/eval_pqa30.txt
echo "[6 done] $(date)"

echo "===== [7] 04 协商质量汇总（闭环证据链）$(date) ====="
python scripts/04_eval.py --config configs/config.yaml \
    --runs runs/mane_pqa30.jsonl --scenarios data_cache/scenarios_pqa30.jsonl \
    2>&1 | tee runs/eval4_pqa30.txt
echo "[7 done] $(date)"

echo "===== 全部完成 $(date) ====="
echo "产物:"
echo "  runs/floor_sensitivity.txt      ([1] 底线激活曲线)"
echo "  runs/mcq_eval_fixed2.txt        ([2] MCQ 三通道含 llm_mcq)"
echo "  runs/catfish_gne.txt            ([3] catfish 真异议 + 路 A)"
echo "  runs/mane_pqa30.jsonl           ([5] PrinciplismQA 30 协商)"
echo "  runs/eval_pqa30.txt             ([6] rubric/Judge/分布)"
echo "  runs/eval4_pqa30.txt            ([7] 闭环证据链)"
