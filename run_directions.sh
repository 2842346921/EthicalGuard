#!/usr/bin/env bash
# ============================================================
# run_directions.sh —— 架构调整尝试（27）：A 现状 / B 议题感知 / C 议题+例外议程
# 目的：测试"提高内容性能（切题性）"与"增强可靠性（议题覆盖）"两个方向，
#       默认开关全关，主实验（已定稿数字）不受影响。
# 前置：conda guard 激活、8003（Qwen3-8B）在跑、代码已同步
#       （config.py / roles.py / engine.py + scripts/27 + sh）
# 耗时：B/C 各 30 场景现跑 ≈ 1.5-2.5h；judge（pqa+cov+clear+议题）≈ 40-60min → 全程 ~2.5-3.5h
# 断点：27 支持追加（文本/评分行按 (场景,配置) 去重）——中断后重跑同命令即可续。
# ============================================================
set -u
cd "$(dirname "$0")"
echo "项目根: $(pwd)  开始: $(date '+%F %T')"

echo ""
echo "================================================================"
echo "[27] 架构调整 A/B/C 对照（议题感知 + 例外议程；默认开关关，主实验不变） $(date '+%T')"
echo "================================================================"
python scripts/27_architecture_ablation.py --config configs/config.yaml \
    --input data_cache/scenarios_pqa30.jsonl \
    --texts runs/cross_method_texts.jsonl \
    --mode local \
    --texts-out runs/arch_texts.jsonl \
    --scores-out runs/arch_scores.jsonl 2>&1 | tee runs/arch_run.txt
echo "[27 done] $?"

echo ""
echo "全部完成 $(date '+%F %T')"
echo "产物：runs/arch_texts.jsonl（B/C 协商文本+终态向量+KKT）"
echo "      runs/arch_scores.jsonl（A/B/C × pqa/cov/clear/issue_cover/violation）"
echo "      runs/arch_run.txt（汇总与配对表）"
echo ""
echo "判读要点："
echo "  - B/C 的 pqa/cov/issue_cover 相对 A 显著升 → 议题感知是切题缺口主因；"
echo "  - C 在保密/自伤/强制报告类题上较 B 额外提升 → 例外议程补上'状态无保密维'缺口；"
echo "  - 违反场景率必须保持 0%（开关不得破坏保证层）。"
