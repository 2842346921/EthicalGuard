"""数据集适配器抽象基类。"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Callable, Iterator, List, Optional

from ..types import Scenario
from .mapping import MappingResult, default_constraints, default_parties, map_text_to_state


class DatasetAdapter(ABC):
    """把单个公开数据集映射为标准场景（Scenario）的适配器。"""

    name: str = "base"
    folder: str = ""  # 数据集目录的实际文件夹名（注册名多为小写，文件夹保留原大小写）

    def __init__(self, data_dir: str, mapper: Optional[Callable[[str], MappingResult]] = None,
                 path: Optional[str] = None):
        self.data_dir = data_dir
        self.path = path  # 若指定，直接作为该数据集根目录（本地/服务器路径差异；否则用 data_dir/folder）
        # 文本→四盒状态映射器：默认规则通道；可注入 DualChannelMapper（双通道）
        self.mapper = mapper or (lambda t: MappingResult(state=map_text_to_state(t), channel="rule"))

    def _path(self, *parts: str) -> str:
        root = self.path or os.path.join(self.data_dir, self.folder or self.name)
        return os.path.join(root, *parts)

    def _load_json(self, *parts: str):
        with open(self._path(*parts), "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_scenario(self, scenario_id: str, raw_text: str, source: dict, reference=None) -> Scenario:
        """通用构造：文本 → 四盒状态（经映射器）→ 默认五方 + 默认约束 + 参照。"""
        mr = self.mapper(raw_text)
        state = mr.state
        # 记录映射元信息（通道/置信度/分歧率）进 source，便于质量审计
        source = dict(source)
        source["mapping"] = {
            "channel": mr.channel,
            "confidence": mr.confidence,
            "disagreement_rate": mr.disagreement_rate,
            "flagged_fields": mr.flagged_fields,
        }
        return Scenario(
            scenario_id=scenario_id,
            source=source,
            raw_text=raw_text,
            state=state,
            parties=default_parties(state),
            constraints=default_constraints(state),
            reference=reference or {"kind": "none", "content": {}},
        )

    @abstractmethod
    def load_scenarios(self) -> Iterator[Scenario]:
        """可协商场景（开放题/困境）。"""
        raise NotImplementedError

    def load_knowledge(self) -> List[Scenario]:
        """知识型（MCQA）条目——作为基座知识检查，不作协商。"""
        return []


# 适配器注册表
_ADAPTERS: dict = {}


def register(name: str):
    def deco(cls):
        _ADAPTERS[name] = cls
        cls.name = name
        return cls

    return deco


def get_adapter(name: str, data_dir: str, mapper=None, path: Optional[str] = None) -> DatasetAdapter:
    if name not in _ADAPTERS:
        raise KeyError(f"未注册的数据集适配器: {name}（可用: {sorted(_ADAPTERS)}）")
    return _ADAPTERS[name](data_dir, mapper=mapper, path=path)


def list_adapters() -> List[str]:
    return sorted(_ADAPTERS)
