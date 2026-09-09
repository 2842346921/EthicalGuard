#!/bin/bash
# run_evidence.sh — EthicalGuard 证据实验全流程：E1+E4+E2+跨模型，全部 local，同一 log
# 用法: nohup bash run_evidence.sh > logs/evidence_all.log 2>&1 &
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
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null \
  && echo "GPU 状态如上" || echo "(无 nvidia-smi，跨模型显存探测跳过)"

echo "===== [1] E1 底线激活实验（local, Qwen3-8B）$(date) ====="
python scripts/10_floor_activation.py --config configs/config.yaml \
    --mode local --floors-high --input data_cache/scenarios.jsonl 2>&1 | tee runs/e1_floor_activation.txt
echo "[E1 done] $(date)"

echo "===== [2] E4 指标校准（读 local 产物，纯分析）$(date) ====="
python scripts/12_metric_calibration.py --runs runs/mane_results.jsonl 2>&1 | tee runs/e4_calibration.txt
echo "[E4 done] $(date)"

echo "===== [3] E2 真实 LLM 基线（local, Qwen3-8B，预计 3-4h）$(date) ====="
python scripts/11_real_llm_baselines.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --mode local \
    --out runs/real_llm_baselines.jsonl 2>&1 | tee runs/e2_baselines.txt
echo "[E2 done] $(date)"

echo "===== [4] 跨模型：Llama-3.1-8B（MC4，需空闲显存 ≥16GB）$(date) ====="
# 生成 Llama 配置（若不存在）
[ -f configs/config_llama31.yaml ] || python scripts/13_model_config.py \
    --model llama31 --out configs/config_llama31.yaml

# 探测空闲 GPU（free > 16GB 的卡）
GPUS=$(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null)
TARGET=""
while IFS=',' read -r idx total free; do
  idx=$(echo "$idx" | xargs); free=$(echo "$free" | xargs)
  if [ -n "$free" ] && [ "$free" -gt 16000 ] 2>/dev/null; then
    TARGET="$idx"; break
  fi
done < <(echo "$GPUS")

if [ -z "$TARGET" ]; then
  echo "无空闲 GPU（需 >16GB）→ 跳过跨模型。"
  echo "手动方案：先停 8003（kill 对应 vllm 进程）→ 起 Llama 8012 → 单独跑 02/11。"
else
  echo "使用 GPU $TARGET 启动 Llama-3.1-8B（8012）"
  CUDA_VISIBLE_DEVICES="$TARGET" nohup python scripts/00_serve_local.py \
      --config configs/config_llama31.yaml --port 8012 > logs/vllm_llama31.log 2>&1 &
  for i in $(seq 1 36); do
    sleep 5
    curl -s --max-time 3 http://localhost:8012/v1/models > /dev/null 2>&1 && break
  done
  if curl -s --max-time 3 http://localhost:8012/v1/models > /dev/null 2>&1; then
    echo "Llama OK → 跑 02（同批场景子集 limit 10）"
    python scripts/02_run_mane.py --config configs/config_llama31.yaml \
        --input data_cache/scenarios.jsonl --mode local --limit 10 \
        --out runs/mane_llama31.jsonl
    echo "[Llama 02 done] $(date)"
    echo "Llama OK → 跑 E2 真实基线（同批 limit 10，跨模型对照）"
    python scripts/11_real_llm_baselines.py --config configs/config_llama31.yaml \
        --input data_cache/scenarios.jsonl --mode local --limit 10 \
        --out runs/real_llm_baselines_llama31.jsonl 2>&1 | tee runs/e2_llama31.txt
    echo "[Llama E2 done] $(date)"
  else
    echo "Llama 启动失败（见 logs/vllm_llama31.log）→ 跳过跨模型"
  fi
fi

echo "===== 全部完成 $(date) ====="
echo "产物: runs/e1_floor_activation.txt / runs/e2_baselines.txt / runs/e4_calibration.txt"
echo "       runs/real_llm_baselines.jsonl / runs/mane_llama31.jsonl / runs/real_llm_baselines_llama31.jsonl"
