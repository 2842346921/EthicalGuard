#!/bin/bash
# run_floor_coupling.sh — 底线激活双实验统一跑（local）：
#   [0.5] 15_floor_sensitivity：35 混合场景 × floor 0.15→0.30（乘子激活曲线，~1-2h）
#   [1-2] 16_gne_modes 六格矩阵：PQA50 × coupling(collective/independent) × floor(def/0.22/0.25)（~8-12h）
# 回答：① 底线何时真激活（乘子证据）② 公共 v 让步 vs 教科书独立 谁守底线/谁平衡（标签错位）
# 用法: nohup bash run_floor_coupling.sh > logs/floor_coupling_all.log 2>&1 &
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

echo "===== [0.5] 底线敏感性：35 场景 × floor 0.15→0.30（乘子激活曲线）$(date) ====="
# 需先确认 scenarios.jsonl 是 35 场景新版（含 MEE-equilibrium）；用默认 grid 即可
python scripts/15_floor_sensitivity.py --config configs/config.yaml \
    --mode local --input data_cache/scenarios.jsonl --limit 35 \
    2>&1 | tee runs/floor_sensitivity.txt
echo "[0.5 done] $(date)"

# ===== [2] 16_gne_modes 六格已停（2026-09-05）=====
# 用户确认 FDBI 软惩罚+floor 硬约束为正常设计 → 16 的 collective/independent 对比不再必要。
# 其信息已被 [0.5] 底线敏感性 + 08 的 γ-sweep 覆盖。若仍想留"公共 v 让步贡献"实证，
# 手动跑简化版（省 ~70% 时间）：
#   python scripts/16_gne_modes.py --config configs/config.yaml \
#       --input data_cache/scenarios_pqa50.jsonl --mode local --limit 50 \
#       --matrix "collective@def,collective@0.25"
# ==============================================================

echo "===== 全部完成 $(date) ====="
echo "产物: runs/floor_sensitivity.txt"
