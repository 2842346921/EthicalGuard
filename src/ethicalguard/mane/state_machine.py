"""协商流程状态机（GoS 风格）：S0-S6 + R 回溯。

对应《项目架构规划》§3.2：
S0 事实收集 → S1 动态组队 → S2 提案+异议 → S3 对齐/共识检测 → S4 韧性施压
→ S5 仲裁收敛 → S6 恢复观测；R 回溯（Drift 越界时返回 S2/S3）。
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional


class State(str, Enum):
    S0 = "S0_fact_anchoring"
    S1 = "S1_assemble"
    S2 = "S2_propose"
    S3 = "S3_align"
    S4 = "S4_stress"
    S5 = "S5_arbitration"
    S6 = "S6_recovery"
    R = "R_backtrack"
    DONE = "DONE"


class NegotiationFSM:
    """协商状态机：记录状态轨迹，按收敛/漂移/仲裁标志驱动转移。"""

    def __init__(self, drift_backtrack_threshold: float = 0.3):
        self.state = State.S0
        self.history = [self.state]
        self.drift_backtrack_threshold = drift_backtrack_threshold

    def reset(self) -> None:
        self.state = State.S0
        self.history = [self.state]

    def advance(self, ctx: Dict[str, object]) -> State:
        """根据上下文标志返回下一状态。ctx 需含 flags（见下）。"""
        s = self.state
        f = ctx.get("flags", {})

        if s == State.S0:
            nxt = State.S1
        elif s == State.S1:
            nxt = State.S2
        elif s == State.S2:
            nxt = State.S3
        elif s == State.S3:
            if f.get("drift_exceeded"):
                nxt = State.R
            elif f.get("stress_active"):
                nxt = State.S4
            elif f.get("converged"):
                nxt = State.S5
            else:
                nxt = State.S2  # 继续一轮
        elif s == State.S4:
            nxt = State.S6
        elif s == State.S6:
            nxt = State.DONE
        elif s == State.S5:
            nxt = State.DONE
        elif s == State.R:
            nxt = State.S2  # 回溯后重提案
        else:
            nxt = State.DONE

        self.state = nxt
        self.history.append(nxt)
        return nxt
