#!/bin/bash
# run_evidence2.sh — 数据一致性全链重跑 + E2 neutral 基线（P0-1 + P0-2）
# 目的：修复"韧性/消融/检测器还是旧 MEE-violation 动物题时代数据"的不一致，
#       全部换到 35 场景新数据（含 MEE-equilibrium 临床题）重跑。
# 用法: nohup bash run_evidence2.sh > logs/evidence2_all.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source /home/qluai/miniconda3/etc/profile.d/conda.sh
conda activate guard
export ETHICALGUARD_DATA_DIR="${ETHICALGUARD_DATA_DIR:-/mnt/zhangheng2025/EthicalGuard/data}"

echo "===== [0] 环境检查 $(date) ====="
curl -s --max-time 8 http://localhost:8003/v1/models > /dev/null \
  && echo "Qwen3-8B vLLM (8003) OK" \
  || { echo "错误：8003 未就绪"; exit 1; }

# 场景构成确认（应含 MEE-equilibrium 而非 MEE-violation）
echo "场景构成:"
python - <<'PY'
import json
from collections import Counter
scs = [json.loads(l) for l in open('data_cache/scenarios.jsonl', encoding='utf-8')]
print('  总数:', len(scs), dict(Counter(s['source'].get('dataset') for s in scs)))
mee = [s for s in scs if s['source'].get('dataset') == 'medethiceval']
print('  medethiceval kinds:', dict(Counter(s['source'].get('kind') for s in mee)))
PY

echo "===== [1] 05 检测器重训（新场景口径，principlismqa+vital+medethiceval）$(date) ====="
python scripts/05_train_detector.py --config configs/config.yaml \
    --sources principlismqa,vital,medethiceval \
    --epochs 30 --lr 1e-3 --batch 64 --w-ers 2 --out runs/detector_v2.pt 2>&1 | tee runs/detector_v2_train.log
export ETHICALGUARD_DETECTOR_CHECKPOINT=runs/detector_v2.pt
echo "[05 done] $(date)"

echo "===== [1b] 核对 02 产物（mane_results.jsonl 应为 35 场景新版，含 process_trace）====="
python - <<'PY'
import json
runs = [json.loads(l) for l in open('runs/mane_results.jsonl', encoding='utf-8')]
print('mane_results 场景数:', len(runs))
print('含 process_trace:', sum(1 for r in runs if r.get('process_trace')))
print('final_proposal agents:', sorted(set((r.get('final_proposal') or {}).get('agent') for r in runs)))
# 确认是 equilibrium 时代（无 MEE-violation）
ids = [r['scenario_id'] for r in runs]
bad = [i for i in ids if 'violation' in i]
print('含旧 MEE-violation 场景:', len(bad), '(应为 0)')
PY

echo "===== [3] 04 评估 + 闭环证据（35 新场景）$(date) ====="
python scripts/04_eval.py --config configs/config.yaml \
    --runs runs/mane_results.jsonl --scenarios data_cache/scenarios.jsonl 2>&1 | tee runs/eval_v2.txt
echo "[04 done] $(date)"

echo "===== [4] 06 基线对比（35 新场景，同口径）$(date) ====="
python scripts/06_baselines.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --runs runs/mane_results.jsonl 2>&1 | tee runs/baselines_v2.txt
echo "[06 done] $(date)"

echo "===== [5] 07 三型评估（35 新场景）$(date) ====="
python scripts/07_eval_types.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --runs runs/mane_results.jsonl 2>&1 | tee runs/eval_types_v2.txt
echo "[07 done] $(date)"

echo "===== [6] 03 韧性压力测试（35 新场景，local，repeat=1，预计 8-12h）$(date) ====="
# repeat=1 先跑通（成本可控）；repeat=3 的 ±std 后补
python scripts/03_stress.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --mode local --limit 35 --repeat 1 \
    --out runs/resilience_v2.jsonl 2>&1 | tee runs/resilience_v2.log
echo "[03 done] $(date)"

echo "===== [7] 08 消融（35 新场景，local，预计 3-4h）$(date) ====="
python scripts/08_ablations.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --mode local --limit 35 \
    --out runs/ablations_v2.txt 2>&1 | tee runs/ablations_v2.log
echo "[08 done] $(date)"

echo "===== [8] E2 真实 LLM 基线 v2（含 neutral_single_llm 对照，35 场景，预计 5-6h）$(date) ====="
python scripts/11_real_llm_baselines.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --mode local --limit 35 \
    --out runs/real_llm_baselines_v2.jsonl 2>&1 | tee runs/e2_baselines_v2.txt
echo "[E2v2 done] $(date)"

echo "===== [9] E1 底线激活（35 新场景复跑确认）$(date) ====="
python scripts/10_floor_activation.py --config configs/config.yaml \
    --mode local --floors-high --input data_cache/scenarios.jsonl 2>&1 | tee runs/e1_v2.txt
echo "[E1v2 done] $(date)"

echo "===== 全部完成 $(date) ====="
echo "产物:"
echo "  runs/resilience_v2.jsonl  (03 韧性，35 新场景)"
echo "  runs/ablations_v2.txt     (08 消融，35 新场景)"
echo "  runs/real_llm_baselines_v2.jsonl (E2 含 neutral 对照)"
echo "  runs/detector_v2.pt       (05 检测器重训)"
echo "  runs/eval_v2.txt / runs/baselines_v2.txt / runs/eval_types_v2.txt / runs/e1_v2.txt"
