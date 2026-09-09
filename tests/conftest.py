"""pytest 公共 fixtures。"""
from __future__ import annotations

import os
import sys

# 确保 src 在导入路径中（无需 pip install 即可跑测试）
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest

from ethicalguard.types import (
    Constraint,
    ConstraintKind,
    FourBoxState,
    Reference,
    Scenario,
)


def make_scenario(scenario_id: str = "SYN-1", resource_pressure: float = 0.2) -> Scenario:
    """构造一个合成的伦理冲突场景（ICU 邻近：脑死亡器官捐献家属反对）。"""
    state = FourBoxState(
        medical={"severity": 0.9, "rescue_available": 0.6, "acuity": 0.8},
        preference={"clarity": 0.4, "capacity": 0.2, "dnr": 0.0, "attitude_refuse": 0.6, "info_completeness": 0.6},
        qol={"burden": 0.7, "net_effect": -0.4},
        context={"resource_pressure": resource_pressure, "insurance_stress": 0.3,
                 "family_conflict": 0.8, "religious_barrier": 0.0, "legal_constraint": 0.5},
    )
    return Scenario(
        scenario_id=scenario_id,
        source={"dataset": "synthetic"},
        raw_text=("患者 Derek 因摩托车事故脑死亡，生前驾照登记为器官捐献者；其妻 Mrs. Polaski "
                  "强烈反对捐献，坚称'他还在呼吸'。ICU 床位紧张，医师需决定是否继续生命维持以等待器官获取。"),
        state=state,
        constraints=[
            Constraint(name="nonmaleficence_floor", kind=ConstraintKind.FLOOR, bound=0.5, direction="ge", value=0.7),
            Constraint(name="autonomy_floor", kind=ConstraintKind.FLOOR, bound=0.3, direction="ge", value=0.4),
            Constraint(name="resource_hard", kind=ConstraintKind.HARD, bound=0.6, direction="le", value=resource_pressure),
        ],
        reference=Reference(kind="rubric", content={"keypoints": ["尊重患者生前意愿", "与家属充分沟通", "法律合规"]}),
    )


@pytest.fixture
def scenario() -> Scenario:
    return make_scenario()
