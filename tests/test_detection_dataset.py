"""监督检测器数据构建测试（不依赖 torch）。"""
from __future__ import annotations

import os

import pytest

from ethicalguard.detection.dataset import (
    DetectorDataset,
    build_dataset,
    build_medethiceval_samples,
    build_principlism_samples,
    build_vital_samples,
    vrd_to_principles,
)
from ethicalguard.detection.rules_features import state_to_features


def test_vrd_mapping():
    assert vrd_to_principles("Autonomy and self-determination")[2] == 1.0
    assert vrd_to_principles("Minimizing harm")[1] == 1.0
    assert vrd_to_principles("Justice and fairness")[3] == 1.0
    assert vrd_to_principles("Right to life")[0] == 1.0


def test_features_shape():
    from ethicalguard.types import FourBoxState
    st = FourBoxState()
    f = state_to_features(st)
    assert f.shape == (16,)


def test_build_from_real_data():
    """若本地数据集存在，各源能构建出样本且标签在合理范围。"""
    data_dir = r"E:\信息\论文\医疗诊断\多目标压力\论文\数据集"
    if not os.path.isdir(data_dir):
        pytest.skip("本地数据集目录不存在")

    pqa = build_principlism_samples(data_dir)
    assert len(pqa) > 0
    assert all(s.principle is not None and s.principle.sum() > 0 for s in pqa)
    assert all(s.ers is None for s in pqa)

    vital = build_vital_samples(data_dir)
    assert len(vital) > 0
    assert all(s.principle is not None and s.principle.sum() > 0 for s in vital)

    mee = build_medethiceval_samples(data_dir)
    assert len(mee) > 0
    assert all(s.ers is not None and 0.0 <= s.ers <= 1.0 for s in mee)
    assert all(s.principle is None for s in mee)

    ds = build_dataset(data_dir)
    assert len(ds) > 0
    assert ds.features().shape[1] == 16
    assert ds.principle_mask().sum() >= len(pqa)
    assert ds.ers_mask().sum() == len(mee)
