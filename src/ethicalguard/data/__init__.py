"""工作①：统一场景适配层。"""
from __future__ import annotations

from .loaders import (
    dump_scenarios,
    load_scenarios,
    load_scenarios_from_jsonl,
    mapping_quality_report,
)
from .registry import get_adapter, list_adapters
from .mapping import (
    DualChannelMapper,
    MappingResult,
    MappingQualityReport,
    map_text_to_state,
    default_parties,
    default_constraints,
    report_mapping,
)
from .mapping_llm import LLMFourBoxMapper

__all__ = [
    "dump_scenarios",
    "load_scenarios",
    "load_scenarios_from_jsonl",
    "mapping_quality_report",
    "get_adapter",
    "list_adapters",
    "DualChannelMapper",
    "MappingResult",
    "MappingQualityReport",
    "map_text_to_state",
    "default_parties",
    "default_constraints",
    "report_mapping",
    "LLMFourBoxMapper",
]
