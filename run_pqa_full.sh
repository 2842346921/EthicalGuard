#!/usr/bin/env bash
# ============================================================
# run_pqa_full.sh —— PrinciplismQA 三子集充分利用：34(同批方法) → 35(competency) →
#                    36(by-principles 离线) → 37(Practice 全量基线官方分)
# 前置：guard env + 8003；34 需先跑（30 题同批 6 方法，LLM ~1-1.5h）；37 可按 --limit 调规模。
# ============================================================
set -u
cd "$(dirname "$0")"
echo "项目根: $(pwd)  开始: $(date '+%F %T')"
RUBRIC="data/PrinciplismQA/data/open-ended-rubric-principles.json"

echo ""
echo "================================================================"
echo "[34] 同批内容对照（30 题 × 6 方法，落盘逐条 scores+competency） $(date '+%T')"
echo "================================================================"
if [ -f runs/content_vs_baseline.jsonl ] && [ -s runs/content_vs_baseline.jsonl ]; then
    echo "[34 跳过] runs/content_vs_baseline.jsonl 已存在——如要重跑先删该文件"
else
    python scripts/34_content_vs_baseline_same_batch.py --config configs/config.yaml \
        --input data_cache/scenarios_pqa30.jsonl \
        --texts runs/cross_method_texts.jsonl \
        --arch-texts runs/arch_texts_v2.jsonl \
        --rubric "$RUBRIC" 2>&1 | tee runs/content_vs_baseline.txt
    echo "[34 done] $?"
fi

echo ""
echo "================================================================"
echo "[35] ACGME competency 分解（离线） $(date '+%T')"
echo "================================================================"
python scripts/35_competency_analyze.py \
    --in runs/content_vs_baseline.jsonl --out runs/competency_analysis.txt 2>&1 | tee runs/competency_analysis.log
echo "[35 done] $?"

echo ""
echo "================================================================"
echo "[36] By-Principles 分解 + Knowledge-Practice 对照（离线） $(date '+%T')"
echo "================================================================"
python scripts/36_principle_breakdown.py \
    --mcq runs/mcqa_official.jsonl \
    --content runs/content_vs_baseline.jsonl \
    --practice runs/practice_scale.jsonl \
    --rubric "$RUBRIC" \
    --out runs/principle_breakdown.txt 2>&1 | tee runs/principle_breakdown.log
echo "[36 done] $?"

echo ""
echo "================================================================"
echo "[37] Practice 全量/大子集基线官方分（LLM；--limit 0=全量 1466 ≈3-5h） $(date '+%T')"
echo "================================================================"
LIMIT="${PRACTICE_LIMIT:-300}"
python scripts/37_practice_scale.py --config configs/config.yaml \
    --qa data/PrinciplismQA/data/open-ended-qa.json \
    --rubric "$RUBRIC" \
    --methods single_llm_llm,neutral_single_llm \
    --limit "$LIMIT" --out runs/practice_scale.jsonl --skip-existing \
    2>&1 | tee runs/practice_scale.txt
echo "[37 done] $?（全量请用 PRACTICE_LIMIT=0 nohup 重跑，--skip-existing 续）"

echo ""
echo "全部完成 $(date '+%F %T')"
echo "产物：runs/content_vs_baseline.* / competency_analysis.* / principle_breakdown.* / practice_scale.*"
