"""监督检测器训练数据构建：第三方标注 → 监督样本。

数据来源（全部第三方，零自建金标 —— 直接回应"标签循环论证"攻击）：
- PrinciplismQA（ACL26）：2,182 MCQA 的 principlism 四原则布尔标注 +
  1,466 开放题的 rubric principles 列表 → 原则多标签（principle_label）
- VITAL（ACL25）：steerable_valuekaleidoscope 11,952 条 situation→vrd(价值/权利/义务)
  标注 → 经 vrd→四原则关键词映射 → 原则多标签
- MedEthicEval（NAACL25）：三类任务（明显违规/有明确倾向/平衡困境）→
  任务设计隐含的风险排序 → ERS 弱监督标签（0.85/0.62/0.55）

特征：病例文本经规则四盒通道提取的 16 维状态（医疗3+意愿5+QoL2+情境6），
分组供四维注意力使用（可扩展文本嵌入，见 model.py 注释）。
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .rules_features import DIM_GROUPS, state_to_features  # 四盒 → 16 维特征
from ..data.mapping_rules import map_text_to_state

# ---- vrd（价值/权利/义务）→ 四原则 关键词映射 ----
PRINCIPLE_KEYWORDS = {
    "beneficence": ["beneficence", "benefit", "best interest", "duty to save", "preservation of life",
                    "well-being", "fidelity", "care", "life", "health"],
    "nonmaleficence": ["nonmaleficence", "non-maleficence", "harm", "minimizing harm", "duty not to harm",
                       "do no harm", "safety", "right to safety"],
    "autonomy": ["autonomy", "self-determination", "informed consent", "choice", "freedom", "liberty",
                 "preference", "refusal", "consent"],
    "justice": ["justice", "fairness", "equity", "resource", "allocation", "impartiality",
                "right to", "dignity", "equality", "duty"],
}


def vrd_to_principles(vrd_text: str) -> np.ndarray:
    """把 VITAL 的 vrd 字符串映射为四原则 one-hot（可多标签）。"""
    t = (vrd_text or "").lower()
    out = np.zeros(4, dtype=float)
    for i, (principle, words) in enumerate(PRINCIPLE_KEYWORDS.items()):
        if any(w in t for w in words):
            out[i] = 1.0
    # 无命中 → 中性样本（不参与该样本的原则监督）
    return out


@dataclass
class Sample:
    text: str
    features: np.ndarray                       # (16,)
    principle: Optional[np.ndarray] = None     # (4,) 0/1 多标签（可 None）
    ers: Optional[float] = None                # 0-1 风险（可 None）


@dataclass
class DetectorDataset:
    samples: List[Sample] = field(default_factory=list)

    def features(self) -> np.ndarray:
        return np.array([s.features for s in self.samples])

    def principle_labels(self) -> Optional[np.ndarray]:
        """全量原则标签 (n,4)：无标签行填 0（配合 principle_mask 使用）。

        必须与 samples 等长且为 float32——训练脚本用全量索引（train_idx）切分，
        且 torch BCE 要求 float（float64 → RuntimeError: Found dtype Double）；
        压缩数组（只收集非 None）会导致索引错位（此前两个训练错误均已修复）。
        """
        if not any(s.principle is not None for s in self.samples):
            return None
        out = np.zeros((len(self.samples), 4), dtype=np.float32)
        for i, s in enumerate(self.samples):
            if s.principle is not None:
                out[i] = np.asarray(s.principle, dtype=np.float32)
        return out

    def principle_mask(self) -> np.ndarray:
        return np.array([s.principle is not None for s in self.samples])

    def ers_labels(self) -> Optional[np.ndarray]:
        """全量 ERS 标签 (n,) float32：无标签行填 0（配合 ers_mask 使用，与 principle_labels 同理）。"""
        if not any(s.ers is not None for s in self.samples):
            return None
        out = np.zeros(len(self.samples), dtype=np.float32)
        for i, s in enumerate(self.samples):
            if s.ers is not None:
                out[i] = float(s.ers)
        return out

    def ers_mask(self) -> np.ndarray:
        return np.array([s.ers is not None for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)


def _sample(text: str, principle: Optional[np.ndarray] = None, ers: Optional[float] = None) -> Sample:
    state = map_text_to_state(text)
    return Sample(text=text, features=state_to_features(state), principle=principle, ers=ers)


def fit_ers_temperature(logits: np.ndarray, targets: np.ndarray, lo: float = 0.5,
                        hi: float = 5.0, steps: int = 46) -> float:
    """ERS 温度缩放校准（P-7.1）：一维网格搜索最小化 MSE(σ(t·logit), y)。

    监督检测器的 ERS 头 sigmoid 输出易饱和不足（实测全挤在 0.5-0.7，弱监督标签
    0.85 的违规题被预测成 0.56），用温度 t>1 拉开预测分布后更贴近弱监督标签量纲。
    纯 numpy 实现（不依赖 torch），05 训练脚本在 eval 集拟合后把 t 存进 checkpoint。

    :param logits: 模型 logit（σ 逆），仅含有效 ERS 样本
    :param targets: 对应弱监督 ERS 标签
    :returns: 最优温度 t（默认 1.0 即不缩放）
    """
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(logits) < 5 or np.all(targets == targets[0]):
        return 1.0

    def mse(t: float) -> float:
        p = 1.0 / (1.0 + np.exp(-t * logits))
        return float(np.mean((p - targets) ** 2))

    best_t, best_mse = 1.0, mse(1.0)
    for cand in np.linspace(lo, hi, steps):
        m = mse(cand)
        if m < best_mse:
            best_t, best_mse = cand, m
    return float(best_t)


# ============================ 各数据集构建 ============================

def build_principlism_samples(data_dir: str) -> List[Sample]:
    """PrinciplismQA：MCQA 的 principlism 布尔标注 + 开放题的 rubric principles。"""
    out: List[Sample] = []
    mcqa_path = os.path.join(data_dir, "PrinciplismQA", "data", "knowledge-mcqa.json")
    if os.path.exists(mcqa_path):
        with open(mcqa_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for it in items:
            p = it.get("principlism", {})
            label = np.array([
                1.0 if p.get("beneficience") or p.get("beneficence") else 0.0,
                1.0 if p.get("nonmaleficience") or p.get("nonmaleficence") else 0.0,
                1.0 if p.get("autonomy") else 0.0,
                1.0 if p.get("justice") else 0.0,
            ])
            if label.sum() == 0:  # 无标注样本跳过（避免噪声）
                continue
            text = f"{it.get('question','')} {it.get('options',{})}"
            out.append(_sample(text, principle=label))

    rub_path = os.path.join(data_dir, "PrinciplismQA", "data", "open-ended-rubric-principles.json")
    if os.path.exists(rub_path):
        with open(rub_path, "r", encoding="utf-8") as f:
            rubrics = json.load(f)
        for r in rubrics:
            prins = r.get("principles", [])
            if not prins:
                continue
            label = np.array([
                1.0 if any(x in ("beneficence", "beneficience") for x in prins) else 0.0,
                1.0 if any(x in ("nonmaleficence", "nonmaleficience") for x in prins) else 0.0,
                1.0 if "autonomy" in prins else 0.0,
                1.0 if "justice" in prins else 0.0,
            ])
            if label.sum() == 0:  # principles 存在但无可识别原则名 → 跳过（避免全零噪声）
                continue
            out.append(_sample(r.get("question", ""), principle=label))
    return out


def build_vital_samples(data_dir: str) -> List[Sample]:
    """VITAL steerable_valuekaleidoscope：situation→vrd 标注 → 原则多标签。"""
    out: List[Sample] = []
    path = os.path.join(data_dir, "VITAL", "dataset", "vital_steerable_valuekaleidoscope.json")
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    for it in items:
        label = vrd_to_principles(it.get("vrd", ""))
        if label.sum() == 0:
            continue
        out.append(_sample(it.get("situation", "") or it.get("input", ""), principle=label))
    return out


def build_medethiceval_samples(data_dir: str) -> List[Sample]:
    """MedEthicEval：三类任务 → ERS 弱监督标签（任务设计隐含的风险排序）。"""
    out: List[Sample] = []
    base = os.path.join(data_dir, "MedEthicEval", "dataset")
    # (文件名, ERS 标签)：明显违规 > 有明确倾向的优先级困境 > 平衡困境
    for fname, ers in (("medical_ethics_detecting_violation.csv", 0.85),
                       ("medical_ethics_priority_dilemma.csv", 0.62),
                       ("medical_ethics_equilibrium_dilemma.csv", 0.55)):
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                text = f"{row.get('case','')} {row.get('query','')} {row.get('scenario','')}"
                out.append(_sample(text, ers=ers))
    return out


def build_dataset(data_dir: str, sources: Tuple[str, ...] = ("principlismqa", "vital", "medethiceval")) -> DetectorDataset:
    """按数据源构建监督数据集（各源只提供其具备的标签，另一头为 None）。"""
    ds = DetectorDataset()
    if "principlismqa" in sources:
        ds.samples += build_principlism_samples(data_dir)
    if "vital" in sources:
        ds.samples += build_vital_samples(data_dir)
    if "medethiceval" in sources:
        ds.samples += build_medethiceval_samples(data_dir)
    return ds
