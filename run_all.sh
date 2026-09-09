#!/bin/bash
# =============================================================================
# EthicalGuard 一键全流程（服务器用，默认不含韧性评估）
# 前提：① 本地 vLLM 已启动（python scripts/00_serve_local.py --port 8003）
#       ② 数据集在 config.yaml 的 datasets.data_dir（或 ETHICALGUARD_DATA_DIR）
# 用法：
#   bash run_all.sh                                # 前台
#   nohup bash run_all.sh > logs/run_all.log 2>&1 &  # 后台挂起（推荐）
# 查看进度：tail -f logs/run_all.log
#
# 韧性开关：结果稳定后再跑韧性（03 每场景约 30 次协商，耗时最长）
#   RUN_STRESS=1 bash run_all.sh                   # 启用韧性
# =============================================================================
set -euo pipefail

# 韧性评估开关（默认关：耗时太长，等协商结果稳定后再开）
RUN_STRESS=${RUN_STRESS:-0}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs runs data_cache

# ---- conda 环境（按你的安装路径改）----
source /home/qluai/miniconda3/etc/profile.d/conda.sh
conda activate guard

# ---- 数据集根目录（默认 config.yaml 的 data_dir）----
export ETHICALGUARD_DATA_DIR=${ETHICALGUARD_DATA_DIR:-/mnt/zhangheng2025/EthicalGuard/data}

# 本地模型服务（config.yaml 的 llm.local.base_url）
LLM_URL=${ETHICALGUARD_LOCAL_BASE_URL:-http://localhost:8003/v1}

echo "===== [0/8] 检查本地模型服务: $LLM_URL ====="
if ! curl -s --max-time 8 "$LLM_URL/models" > /dev/null; then
    echo "错误：vLLM 未就绪。请先启动：python scripts/00_serve_local.py --port 8003"
    exit 1
fi

echo "===== [1/8] 01_prepare：5 数据集均分 + dual 映射 + 非临床题过滤 ====="
# 说明：部分数据集含非医疗伦理题（PrinciplismQA 研究/机构伦理、VITAL 社会舆论调查、
# MedEthicEval 职业伦理违规）——VITAL 适配器已只留临床道德困境；
# --min-clinical-signals 1 剔除四盒关键字段全 0 的"零状态"场景。
python scripts/01_prepare.py --config configs/config.yaml --mapping dual \
    --datasets principlismqa,medethiceval,vital,medethicsqa,llmevalmed \
    --limit 50 --report --min-clinical-signals 1

echo "===== [2/8] 05_train_detector：监督冲突检测器（w-ers 2 折中）====="
python scripts/05_train_detector.py --config configs/config.yaml \
    --sources principlismqa,vital,medethiceval \
    --epochs 30 --lr 1e-3 --batch 64 --w-ers 2 --out runs/detector.pt
export ETHICALGUARD_DETECTOR_CHECKPOINT=runs/detector.pt

echo "===== [3/8] 02_run_mane：LLM 五方协商（监督检测器 + 资源引导提示词）====="
python scripts/02_run_mane.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --mode local --limit 50

echo "===== [4/8] 04_eval：协商质量汇总 ====="
python scripts/04_eval.py --config configs/config.yaml --scenarios data_cache/scenarios.jsonl

echo "===== [5/8] 06_baselines：基线对比（按数据集分组）====="
python scripts/06_baselines.py --config configs/config.yaml --input data_cache/scenarios.jsonl

echo "===== [6/8] 07_eval_types：三型评估（答案/分布/评审，按数据集分组）====="
python scripts/07_eval_types.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --runs runs/mane_results.jsonl

echo "===== [7/8] 08_ablations：消融实验（5 数据集分组，local 模式；--limit 10 控制规模）====="
# local 模式验证全部机制（含 Catfish 的"历史文本→LLM 提案"路径）；
# 全量可去掉 --limit 10（29×5 次 LLM 协商，约 2-4 小时）
python scripts/08_ablations.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --mode local --limit 10 --out runs/ablations_local.txt

echo "===== [7b/8] 08 γ 敏感性：L_balance 权重扫描（§2.9 可解释参数）====="
python scripts/08_ablations.py --config configs/config.yaml \
    --input data_cache/scenarios.jsonl --mode local --limit 5 --gamma-sweep \
    --gamma-grid 0.0,0.2,0.4,0.6,0.8,1.0 --out runs/gamma_sweep.txt

echo "===== [8/8] 09_audit：实现完整性审计 ====="
python scripts/09_audit.py --config configs/config.yaml --scenarios data_cache/scenarios.jsonl

# ---- 韧性评估（可选，默认跳过）----
if [ "$RUN_STRESS" = "1" ]; then
    echo "===== [可选] 03_stress：韧性压力测试（--repeat 3 报 R_recover±std，调用量 ×3）====="
    python scripts/03_stress.py --config configs/config.yaml \
        --input data_cache/scenarios.jsonl --mode local --limit 10 --repeat 3
    python - <<'PY'
import json, numpy as np
reps = [json.loads(l) for l in open('runs/resilience.jsonl', encoding='utf-8')]
print('韧性场景数:', len(reps))
for r in reps:
    rec = r.get('l3_recoverability', 0)
    std = r.get('recovery_std')
    rec_s = f"{rec:.3f}±{std:.3f}" if std is not None else f"{rec:.3f}"
    print(f"  {r['scenario_id']:<12} C={r['l1_consistency']:.3f} R={r['l2_robustness']:.3f} "
          f"rec={rec_s} BSP={r.get('bsp')} ab={r.get('abandoned')}")
if reps:
    print('韧性均值: C=%.3f R_robust=%.3f R_rec=%.3f' % (
        np.mean([r['l1_consistency'] for r in reps]),
        np.mean([r['l2_robustness'] for r in reps]),
        np.mean([r['l3_recoverability'] for r in reps])))
PY
else
    echo "===== [跳过] 03_stress 韧性评估（RUN_STRESS=0）。结果稳定后：RUN_STRESS=1 bash run_all.sh ====="
fi

echo "===== 全部完成 ====="
echo "输出文件："
echo "  data_cache/scenarios.jsonl    # 01 场景"
echo "  runs/mane_results.jsonl       # 02 协商结果"
echo "  runs/resilience.jsonl         # 03 韧性结果（RUN_STRESS=1 时）"
echo "  runs/detector.pt              # 05 监督检测器"
echo "  runs/ablations_local.txt      # 08 消融表（local）"
echo "  runs/gamma_sweep.txt          # 08 γ 敏感性表"
echo "  logs/run_all.log              # 本脚本日志"
echo "  06_baselines / 07_eval_types / 09_audit 结果见上方终端输出"
