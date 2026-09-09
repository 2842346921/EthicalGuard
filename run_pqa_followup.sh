#!/usr/bin/env bash
# =============================================================================
# run_pqa_followup.sh — PQA30 跨方法对比的补实验一键运行
#
# 依赖（先在服务器完成）：
#   1) cd /mnt/zhangheng2025/EthicalGuard  （本脚本放在仓库根目录）
#   2) 激活 guard 环境；LLM 服务已起（configs/config.yaml 的 llm.mode）
#   3) 把本仓库 scripts/ 下的以下文件同步到服务器同路径：
#        scripts/_x0_patch_missing_rows.py
#        scripts/_x1_judge_stability.py
#        scripts/_x2_safety_audit.py
#        scripts/_x3_export_blind_sample.py
#        scripts/23_pqa_official_contrast.py   （已修改：③ 声明口径改为动态）
#
# 用法：
#   ./run_pqa_followup.sh                # 全部阶段：x0 -> 23 -> x1 -> x2 -> x3
#   ./run_pqa_followup.sh x0             # 只补缺行 + 刷新配对统计
#   ./run_pqa_followup.sh x0 x2 x3       # 按需组合
#
# 环境变量（可选）：
#   PY=python3            # 解释器，默认 python
#   ROUNDS=3              # x1 评审重复轮数
#   JUDGE_TEMP=0.7        # x1 评审温度
#   JUDGE2=1              # x1 追加 API 第二评审基座（需 OPENAI_BASE_URL/KEY/MODEL）
#   AUDITOR2=1            # x2 追加 API 第二审计员（同上）
#   SKIP_JUDGE=0          # 即使默认跑也强制跳过 x1
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-python}"
ROUNDS="${ROUNDS:-3}"
JUDGE_TEMP="${JUDGE_TEMP:-0.7}"
LOG_DIR="runs/logs"
mkdir -p "$LOG_DIR"

CFG="configs/config.yaml"
INPUT="data_cache/scenarios_pqa30.jsonl"
TEXTS="runs/cross_method_texts.jsonl"
SCORES="runs/cross_method_scores.jsonl"

# 阶段集合：默认全跑；有参数则按参数跑（仅 x0/23/x1/x2/x3 合法）
if [ "$#" -gt 0 ]; then
  STAGES=("$@")
else
  STAGES=(x0 23 x1 x2 x3)
fi

banner(){ echo; echo "########## $* ##########"; }

for stage in "${STAGES[@]}"; do
  case "$stage" in
    x0)
      banner "x0 补 2 条空 rationale → n=30×4，并刷新配对统计"
      "$PY" scripts/_x0_patch_missing_rows.py \
          --config "$CFG" --input "$INPUT" \
          --texts "$TEXTS" --scores "$SCORES" \
          2>&1 | tee "$LOG_DIR/_x0.log"
      banner "刷新 23 汇总（口径已动态化）"
      "$PY" scripts/23_pqa_official_contrast.py \
          --texts "$TEXTS" --scores "$SCORES" \
          --out runs/pqa_official_contrast_v2.txt \
          2>&1 | tee "$LOG_DIR/_23_v2.log"
      ;;
    23)
      banner "23 汇总（用于单独重跑；通常在 x0 后自动执行）"
      "$PY" scripts/23_pqa_official_contrast.py \
          --texts "$TEXTS" --scores "$SCORES" \
          --out runs/pqa_official_contrast_v2.txt \
          2>&1 | tee "$LOG_DIR/_23_v2.log"
      ;;
    x1)
      if [[ "${SKIP_JUDGE:-0}" == "1" ]]; then banner "x1 已跳过（SKIP_JUDGE=1）"; continue; fi
      banner "x1 内容分评审稳定性（judge 噪声 / 可选第二基座）"
      extra=()
      if [[ "${JUDGE2:-0}" == "1" ]]; then extra+=(--judge2); fi
      "$PY" scripts/_x1_judge_stability.py \
          --config "$CFG" --texts "$TEXTS" --input "$INPUT" \
          --rounds "$ROUNDS" --temperature "$JUDGE_TEMP" \
          "${extra[@]}" \
          --out runs/judge_stability.txt \
          2>&1 | tee "$LOG_DIR/_x1.log"
      ;;
    x2)
      banner "x2 文本级独立安全审计（floor 之外的证据）"
      extra=()
      if [[ "${AUDITOR2:-0}" == "1" ]]; then extra+=(--auditor2); fi
      rm -f runs/safety_audit.jsonl   # 脚本为 append 模式，先清旧文件
      "$PY" scripts/_x2_safety_audit.py \
          --config "$CFG" --texts "$TEXTS" \
          "${extra[@]}" \
          --out-txt runs/safety_audit.txt \
          2>&1 | tee "$LOG_DIR/_x2.log"
      ;;
    x3)
      banner "x3 人工盲审抽样导出"
      "$PY" scripts/_x3_export_blind_sample.py \
          --texts "$TEXTS" \
          --out runs/audit_blind_sample.csv --key runs/audit_blind_key.csv \
          2>&1 | tee "$LOG_DIR/_x3.log"
      ;;
    *)
      echo "未知阶段：$stage（可选：x0 / 23 / x1 / x2 / x3）" >&2
      exit 2
      ;;
  esac
done

cat <<'EOF'

=========================== 全部完成 ===========================
关键产物：
  runs/pqa_stats_n30.txt              补行后的配对 Δ+95%CI+p（论文表）
  runs/pqa_official_contrast_v2.txt   23 汇总 v2（③ 口径动态化）
  runs/judge_stability.txt            x1 judge 噪声（可选用第二基座）
  runs/safety_audit.txt / .jsonl      x2 文本级安全审计 + McNemar
  runs/audit_blind_sample.csv/.key    x3 人工盲审样本 + 密钥
日志：runs/logs/
判读规则见本轮分析：内容层 p 不显著 → 写"无显著差异"；保证层需 x2 文本级证据支撑；
rubric 若仍 ~0.95 全饱和 → 降级为辅助列；neutral 官方分最高反直觉点主动进讨论。
EOF
