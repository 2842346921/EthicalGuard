"""四盒状态 → 16 维特征（分组供四维注意力使用）。

分组：medical(3) + preference(5) + qol(2) + context(6) = 16。
顺序与 model.DIM_GROUPS 一致。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..types import FourBoxState

DIM_GROUPS: Dict[str, List[str]] = {
    "medical": ["severity", "rescue_available", "acuity"],
    "preference": ["dnr", "attitude_refuse", "clarity", "capacity", "info_completeness"],
    "qol": ["burden", "net_effect"],
    "context": ["resource_pressure", "insurance_stress", "family_conflict", "family_involvement",
                "religious_barrier", "legal_constraint"],
}

# 顺序拼接的字段索引（供模型 split 用）
_ORDERED = [(dim, f) for dim, fs in DIM_GROUPS.items() for f in fs]
_GROUP_SPLITS: Dict[str, slice] = {}
_start = 0
for _dim, _fs in DIM_GROUPS.items():
    _GROUP_SPLITS[_dim] = slice(_start, _start + len(_fs))
    _start += len(_fs)


def state_to_features(state: FourBoxState) -> np.ndarray:
    """四盒状态 → 16 维特征向量（缺失字段补 0）。"""
    vals = []
    for dim, f in _ORDERED:
        sub = getattr(state, dim, {})
        vals.append(float(sub.get(f, 0.0)))
    return np.asarray(vals, dtype=float)


def group_slices() -> Dict[str, slice]:
    return _GROUP_SPLITS
