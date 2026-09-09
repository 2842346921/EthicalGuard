"""统一加载入口（含双通道映射注入与质量报告）。"""
from __future__ import annotations

import json
import re
from typing import Callable, Iterable, Iterator, List, Optional

import numpy as np

from ..types import Scenario
from ..utils import setup_logging
from .mapping import MappingQualityReport, MappingResult, report_mapping
from .registry import get_adapter, list_adapters

logger = setup_logging()


def load_scenarios(datasets: Iterable[str], data_dir: str, *, kinds=("scenarios",),
                   limit: int = -1, mapper: Optional[Callable[[str], MappingResult]] = None,
                   dataset_paths: Optional[dict] = None, seed: Optional[int] = None,
                   shuffle: bool = False) -> List[Scenario]:
    """从指定数据集加载标准场景。

    kinds: 需要的条目类型，可为 "scenarios"（可协商）和/或 "knowledge"（知识检查）。
    limit: >0 时**按数据集均分截断**（ceil）——保证多数据集都能加载到，而不是第 1 个
           数据集独占全部名额（A 修复：旧逻辑在内层循环截断，--datasets 全选 5 个时
           实际只加载了第 1 个数据集的 limit 个）。
    mapper: 文本→四盒状态映射器（默认规则通道；可传 DualChannelMapper 走双通道）。
    dataset_paths: 逐数据集根目录覆盖 {name: path}——本地与服务器路径不同时用。
    seed + shuffle: >0 时先随机打乱再截断（公平抽样，避免"总取前 N 个"的选择偏差；
           审稿人"采样规则未说明"批评的修复——B 轨 PrinciplismQA 抽样 300 用）。
    """
    import random
    rng = random.Random(seed)
    out: List[Scenario] = []
    dataset_paths = dataset_paths or {}
    ds_list = [d for d in datasets if d and d.strip()]
    n_ds = max(1, len(ds_list))
    per = max(1, -(-limit // n_ds)) if limit > 0 else -1  # ceil 均分（limit 5、5 个数据集 → 每数据集 1 个）
    for name in ds_list:
        adapter = get_adapter(name, data_dir, mapper=mapper, path=dataset_paths.get(name))
        cnt = 0
        if "scenarios" in kinds:
            pool = list(adapter.load_scenarios())
            if shuffle:
                rng.shuffle(pool)
            for sc in pool:
                out.append(sc)
                cnt += 1
                if 0 < per <= cnt:
                    break
        if "knowledge" in kinds:
            pool = list(adapter.load_knowledge())
            if shuffle:
                rng.shuffle(pool)
            for sc in pool:
                out.append(sc)
                cnt += 1
                if 0 < per <= cnt:
                    break
    logger.info("loaded %d scenarios from %s (limit=%s, per-dataset=%s, shuffle=%s, seed=%s)",
                len(out), ds_list, limit, per, shuffle, seed)
    return out


def mapping_quality_report(scenarios: List[Scenario]) -> MappingQualityReport:
    """从已加载场景的 source.mapping 元信息聚合双通道质量报告。"""
    flagged: List[dict] = []
    dims: dict = {}
    rates: List[float] = []
    channel = "rule"
    for sc in scenarios:
        m = (sc.source or {}).get("mapping", {})
        channel = m.get("channel", "rule")
        rates.append(float(m.get("disagreement_rate", 0.0)))
        for f in m.get("flagged_fields", []):
            dim = f.split(".")[0]
            dims[dim] = max(dims.get(dim, 0.0), float(m.get("disagreement_rate", 0.0)))
            flagged.append({"scenario_id": sc.scenario_id, "field": f})
    return MappingQualityReport(
        n_texts=len(scenarios), channel=channel,
        mean_disagreement=float(np.mean(rates)) if rates else 0.0,
        max_disagreement=float(np.max(rates)) if rates else 0.0,
        per_dim_disagreement=dims, flagged_examples=flagged[:10],
    )


def dump_scenarios(scenarios: List[Scenario], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sc in scenarios:
            f.write(sc.model_dump_json() + "\n")


def load_scenarios_from_jsonl(path: str) -> Iterator[Scenario]:
    # errors="replace"：runs/jsonl 偶尔混入非 UTF-8 字节（LLM 输出的 mojibake，如 0xe9），
    # 严格解码会让整个 eval 管道崩溃（04/07 曾产出 traceback 残件）——容错解码保证产物可重出。
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Scenario.model_validate_json(line)


def iter_negotiation_records(path: str) -> List[dict]:
    """容错读取 MANE 结果 jsonl（04/06/07/12/20 统一入口）。

    容忍两类历史产物损坏：
    ① 非 UTF-8 字节（mojibake，errors='replace'）；
    ② 单行粘连/截断（中断或重复写入：record1 在 rationale 中途被切断后直接拼接了
       第二条 ``{"scenario_id":...}`` 完整 JSON）。
    策略：整行优先解析（干净行/仅乱码行直接过）；失败时按 ``{"scenario_id"`` 出现位置
    从后往前试"后缀完整记录"（record1 截断 → 真正的完整记录在后），仍失败才丢行。
    （注意：不能盲目按该子串切分——LLM rationale 常内嵌字面 JSON，会把好行切碎。）
    """
    out: dict = {}
    n_skip = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = None
            try:
                cand = json.loads(line)
                if isinstance(cand, dict) and "scenario_id" in cand:
                    obj = cand
            except Exception:  # noqa: BLE001
                obj = None
            if obj is None:
                # 截断/粘连行：从后往前找"以 {"scenario_id" 开头的完整 JSON"后缀
                for m in reversed(list(re.finditer(r'\{"scenario_id"\s*:', line))):
                    try:
                        cand = json.loads(line[m.start():])
                    except Exception:  # noqa: BLE001
                        continue
                    if isinstance(cand, dict) and "scenario_id" in cand:
                        obj = cand
                        break
            if obj is None:
                n_skip += 1
                continue
            out[obj["scenario_id"]] = obj
    if n_skip:
        import logging
        logging.getLogger("ethicalguard").warning(
            "iter_negotiation_records: 丢弃 %d 个无法恢复的行（%s）", n_skip, path)
    return list(out.values())
