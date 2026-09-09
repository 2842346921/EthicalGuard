"""监督检测器推理：加载训练好的四维注意力模型，输出 ConflictReport。"""
from __future__ import annotations

import os

import numpy as np

from ..types import ConflictReport, ConflictType, PrincipleVector, Scenario
from .base import ConflictDetector
from .rule import RuleConflictDetector
from .rules_features import state_to_features

try:
    import torch

    from .model import FourDimAttentionDetector
    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False


def action_for_ers(ers: float, threshold: float = 0.5) -> str:
    """按校准 ERS 分档行动（threshold 来自 config.detection.threshold，替代硬编码 0.5）：
    observe(<0.3) → review_24h(<threshold) → ethics_consult(<0.7) → intervene(≥0.7)。
    纯函数（不依赖 torch），保证 gate 与 action 口径一致。
    """
    if ers < 0.3:
        return "observe"
    if ers < threshold:
        return "review_24h"
    if ers < 0.7:
        return "ethics_consult"  # 类型为 preference_family 时上层可换 family_communication
    return "intervene"


class SupervisedConflictDetector(ConflictDetector):
    """四维注意力监督检测器（训练产物加载 + 推理）。

    使用模型输出的 ERS 与原则向量；冲突类型/强度/行动仍由四盒信号派生（与规则基线同构），
    保证输出结构一致。
    """

    channel = "supervised"

    def __init__(self, checkpoint: str, device: str = "cpu", threshold: float = 0.5):
        if not _TORCH_OK:
            raise RuntimeError("需要 torch>=2.0 才能使用监督检测器：pip install torch")
        self.checkpoint = checkpoint
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.threshold = float(threshold)  # ERS 门控阈值（config.detection.threshold）
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.model = FourDimAttentionDetector(**ckpt.get("model_args", {}))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        # ERS 温度缩放（P-7.1）：训练时在 eval 集拟合，拉开 sigmoid 饱和不足的分布
        self.ers_temperature = float(ckpt.get("ers_temperature", 1.0))
        self.fallback = RuleConflictDetector()

    def detect(self, scenario: Scenario) -> ConflictReport:
        # 类型/强度：复用四盒信号派生（与规则基线同构）
        base = self.fallback.detect(scenario)
        x = torch.from_numpy(state_to_features(scenario.state).astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(x)
        # 校准 ERS：σ(t·logit)
        p_clip = pred["ers"].clamp(1e-6, 1.0 - 1e-6)
        logit = torch.log(p_clip / (1.0 - p_clip))
        ers = float(torch.sigmoid(self.ers_temperature * logit).item())
        pv = pred["principle"][0].cpu().numpy()
        pv = pv / (pv.sum() + 1e-9)
        # 行动与校准 ERS + threshold 对齐（消除"ERS 高但 observe"的脱节）
        ctype = base.conflict_type
        action = action_for_ers(ers, self.threshold)
        if ctype == ConflictType.PREFERENCE_FAMILY and self.threshold <= ers < 0.7:
            action = "family_communication"
        if ers < 0.3:
            ctype = ConflictType.NONE
        report = ConflictReport(
            scenario_id=scenario.scenario_id,
            ers=float(np.clip(ers, 0.0, 1.0)),
            conflict_type=ctype,
            intensity=base.intensity,
            principle_vector=PrincipleVector.from_array(pv),
            action=action,
            rationale=f"监督检测器（{os.path.basename(self.checkpoint)}，t={self.ers_temperature:.2f}，"
                      f"thr={self.threshold:.2f}）：校准 ERS={ers:.3f}，原则向量={np.round(pv, 3)}",
            channel=self.channel,
        )
        return report
