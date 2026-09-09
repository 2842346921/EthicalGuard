#!/usr/bin/env bash
# ============================================================
# run_cross_model.sh —— 跨模型复现 E2（Llama-3.1-8B / Mistral-7B，各 35 场景 × 4 方法）
# 目的（R2）：证明"0% vs 50-90% 底线违反"不是 Qwen3-8B 特有 → 跨模型稳健。
# 前置：guard env；8003（Qwen）占 GPU0 勿动；本脚本把目标模型起在 GPU1（可用
#       CUDA_DEVICES 环境变量改，如 CUDA_DEVICES="1 2" 两卡并行两模型）。
# 每个模型：起 vLLM → 等就绪 → 31 跑 35 场景 → kill。单模型 ≈ 2-3h，两模型 ≈ 5-6h（建议 nohup 过夜）。
# ============================================================
set -u
cd "$(dirname "$0")"

INPUT="${INPUT:-data_cache/scenarios.jsonl}"
QWEN_RUNS="${QWEN_RUNS:-runs/real_llm_baselines.jsonl}"
# 每模型配置：模型目录名|端口|CUDA 设备|标签（可用环境变量 MODELS 覆盖）
MODELS_DEFAULT="Meta-Llama-3.1-8B-Instruct|8011|1|llama31,Mistral-7B-Instruct-v0.3|8012|1|mistral7b"
MODELS="${MODELS:-$MODELS_DEFAULT}"

wait_vllm() {  # $1=port
  for _ in $(seq 1 120); do
    if curl -s -m 3 "http://127.0.0.1:$1/v1/models" > /dev/null 2>&1; then return 0; fi
    sleep 5
  done
  return 1
}

echo "项目根: $(pwd)  开始: $(date '+%F %T')"
echo "模型配置: $MODELS"
IFS=',' read -ra ENTRIES <<< "$MODELS"
for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r NAME PORT DEV LABEL <<< "$entry"
  echo ""
  echo "================================================================"
  echo "[$LABEL] 启动 vLLM（/mnt/model/$NAME @ :$PORT, GPU$DEV） $(date '+%T')"
  echo "================================================================"
  CUDA_VISIBLE_DEVICES="$DEV" nohup vllm serve "/mnt/model/$NAME" \
      --served-model-name "$NAME" --port "$PORT" \
      --gpu-memory-utilization 0.9 --max-model-len 16384 \
      > "logs/vllm_${LABEL}.log" 2>&1 &
  if ! wait_vllm "$PORT"; then
    echo "[错误] $NAME 服务未就绪（看 logs/vllm_${LABEL}.log）——跳过该模型"
    continue
  fi
  echo "[$LABEL] 就绪，开始 35 场景 × 4 方法（≈2-3h）"
  python scripts/31_cross_model_e2.py --config configs/config.yaml \
      --input "$INPUT" \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --model "$NAME" --model-name "$LABEL" \
      --out "runs/cross_model_${LABEL}.jsonl" \
      --qwen-runs "$QWEN_RUNS" \
      2>&1 | tee "runs/cross_model_${LABEL}.txt"
  echo "[$LABEL done] $?"
  # 停掉该模型服务，释放显存给下一个
  pkill -f "port ${PORT}" 2>/dev/null
  sleep 10
done

echo ""
echo "全部完成 $(date '+%F %T')"
echo "产物：runs/cross_model_{$LABEL}.jsonl/.txt（每个模型）；对照表在 txt 尾部。"
echo "判读：MANE 违反场景率在 Llama/Mistral 上仍应 0%；基线仍显著 >0 → 协议增益跨模型稳健。"
