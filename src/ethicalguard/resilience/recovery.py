"""恢复验证器：扰动撤销后，测量原则向量回到目标平衡点的动态指标（控制论标准量）。"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def recovery_metrics(trajectory: List[np.ndarray], target: np.ndarray, tau: float = 0.1) -> Dict[str, float]:
    """从恢复轨迹计算：settle_rounds / overshoot / steady_error。

    - settle_rounds: 首次进入目标 τ 邻域的轮数
    - overshoot: 恢复过程中偏离目标的最大距离
    - steady_error: 末轮与目标的残余距离
    """
    if not trajectory:
        return {"settle_rounds": -1, "overshoot": 0.0, "steady_error": 0.0}
    t = np.asarray(target, dtype=float)
    dists = [float(np.linalg.norm(np.asarray(v) - t)) for v in trajectory]
    settle = next((i + 1 for i, d in enumerate(dists) if d < tau), -1)
    return {
        "settle_rounds": float(settle),
        "overshoot": float(max(dists)),
        "steady_error": float(dists[-1]),
    }
