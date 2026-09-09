"""01_prepare：把公开数据集适配为标准场景并缓存（支持规则/LLM/双通道映射）。

用法：
  python scripts/01_prepare.py --config configs/config.yaml \
      --datasets principlismqa,medethiceval --limit 50 \
      [--data-dir ...] [--dataset-paths ...] [--mapping rule|llm|dual] [--report]
说明：未给的参数全部取自 config.yaml（datasets.data_dir / datasets.selected / run.limit）。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethicalguard.config import Config
from ethicalguard.data import (
    DualChannelMapper,
    dump_scenarios,
    list_adapters,
    load_scenarios,
    mapping_quality_report,
)
from ethicalguard.llm import make_backend
from ethicalguard.utils import setup_logging

logger = setup_logging()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml", help="统一配置文件（YAML/JSON）")
    ap.add_argument("--data-dir", default=None, help="数据集统一根目录（默认取配置 datasets.data_dir）")
    ap.add_argument("--dataset-paths", default=None,
                    help="逐数据集根目录覆盖，逗号分隔 name=/path（默认取配置 datasets.paths）")
    ap.add_argument("--datasets", default=None, help="逗号分隔；可用：" + ",".join(list_adapters()))
    ap.add_argument("--out", default=None, help="输出 jsonl 路径（默认 config.run.cache_dir/scenarios.jsonl）")
    ap.add_argument("--kinds", default="scenarios")
    ap.add_argument("--limit", type=int, default=None, help="场景数上限（默认 config.run.limit）")
    ap.add_argument("--mapping", default=None, choices=["rule", "llm", "dual"],
                    help="文本→四盒映射：rule=规则通道；llm=仅LLM通道；dual=双通道交叉验证（默认取配置 llm.mode 推断）")
    ap.add_argument("--report", action="store_true", help="输出映射质量报告（分歧率）")
    ap.add_argument("--min-clinical-signals", type=int, default=0,
                    help="过滤临床信号过少的场景（≥1 时启用）：四盒关键字段"
                         "（severity/acuity/attitude_refuse/dnr/family_conflict/resource_pressure/insurance_stress/"
                         "religious_barrier）中 >0.05 的个数少于该值的场景被剔除。"
                         "PrinciplismQA 开放题含研究/机构伦理题（动物实验/IACUC/记录管理），"
                         "映射后落入零状态、协商无意义——此选项用于剔除它们")
    ap.add_argument("--shuffle", action="store_true",
                    help="抽样前随机打乱（避免'总取每集前 N 个'的选择偏差；配合 --seed）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（--shuffle 时生效）")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data_dir = args.data_dir or cfg.datasets.data_dir or r"E:\信息\论文\医疗诊断\多目标压力\论文\数据集"
    datasets = (args.datasets or ",".join(cfg.datasets.selected or ["principlismqa"]))
    datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    limit = args.limit if args.limit is not None else cfg.run.limit
    out = args.out or os.path.join(cfg.run.cache_dir, "scenarios.jsonl")

    # 逐数据集路径：CLI 优先，其次配置
    dataset_paths = dict(cfg.datasets.paths or {})
    if args.dataset_paths:
        for item in args.dataset_paths.split(","):
            item = item.strip()
            if item and "=" in item:
                k, v = item.split("=", 1)
                dataset_paths[k.strip()] = v.strip()

    # 映射器：rule（默认）/ llm / dual
    mapper = None
    mapping_mode = args.mapping or ("dual" if cfg.llm.mode in ("api", "local") else "rule")
    if mapping_mode in ("llm", "dual"):
        backend = make_backend(cfg.llm)
        if backend.mode == "rule":
            logger.warning("--mapping=%s 需要 LLM 后端（配置 llm.mode=api/local），当前回退为规则通道", mapping_mode)
        else:
            if mapping_mode == "dual":
                mapper = DualChannelMapper(backend)
            else:
                from ethicalguard.data.mapping_llm import LLMFourBoxMapper
                from ethicalguard.data.mapping import MappingResult
                _llm = LLMFourBoxMapper(backend)
                mapper = lambda t: MappingResult(state=_llm.map(t)[0], channel="llm", confidence=_llm.map(t)[1])  # noqa: E731

    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    scenarios = load_scenarios(datasets, data_dir, kinds=kinds, limit=limit,
                               mapper=mapper, dataset_paths=dataset_paths or None,
                               seed=args.seed, shuffle=args.shuffle)

    # 临床信号过滤（B 修复，可选）：剔除研究/机构伦理题坍缩成的"零状态"场景
    if args.min_clinical_signals > 0:
        clinical_keys = [("medical", "severity"), ("medical", "acuity"),
                         ("preference", "attitude_refuse"), ("preference", "dnr"),
                         ("context", "family_conflict"), ("context", "resource_pressure"),
                         ("context", "insurance_stress"), ("context", "religious_barrier")]
        before = len(scenarios)
        scenarios = [s for s in scenarios
                     if sum(1 for dim, k in clinical_keys
                            if getattr(s.state, dim).get(k, 0.0) > 0.05) >= args.min_clinical_signals]
        logger.info("临床信号过滤: %d -> %d 场景 (min_clinical_signals=%d)",
                    before, len(scenarios), args.min_clinical_signals)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    dump_scenarios(scenarios, out)
    logger.info("wrote %d scenarios -> %s", len(scenarios), out)

    if args.report:
        rep = mapping_quality_report(scenarios)
        print(f"[映射质量] 通道={rep.channel} 样本={rep.n_texts} "
              f"平均分歧率={rep.mean_disagreement:.3f} 最大分歧率={rep.max_disagreement:.3f}")
        for dim, d in rep.per_dim_disagreement.items():
            print(f"  - {dim}: 最大分歧 {d:.3f}")
        for ex in rep.flagged_examples[:5]:
            print(f"  ! 分歧示例: {ex}")


if __name__ == "__main__":
    main()
