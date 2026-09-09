"""仲裁层资源强制修正测试（P0-1）：Σρ(t) > cap → 强制降级仲裁 + GNE 精炼输入可行。

根因：GNE 求解器的治疗方案是固定输入、无法自愈，资源违约会导致资源约束
KKT 永不收敛（KKT 爆炸）。仲裁层负责 L1 资源否决（协商层强制修正）。
"""
from __future__ import annotations

import numpy as np
import pytest

from ethicalguard.config import Config
from ethicalguard.mane import MANEEngine
from ethicalguard.mane.arbitration import enforce_resource_cap, resource_usage
from ethicalguard.mane.gne_solver import scenario_resource_cap
from ethicalguard.types import PrincipleVector, Proposal


def _scenario(rp: float = 0.2):
    from conftest import make_scenario
    return make_scenario(resource_pressure=rp)


def _hot_proposals():
    """5 方高治疗提案：医师/家属积极治疗(3)、管理全面医疗(2)、委员会有限(1)、患者舒适(0)。"""
    w = PrincipleVector.from_array(np.ones(4))
    return {
        "physician": Proposal(agent="physician", treatment_level=3, principle_weights=w),
        "family": Proposal(agent="family", treatment_level=3, principle_weights=w),
        "patient": Proposal(agent="patient", treatment_level=0, principle_weights=w),
        "ethics_committee": Proposal(agent="ethics_committee", treatment_level=1, principle_weights=w),
        "hospital_admin": Proposal(agent="hospital_admin", treatment_level=2, principle_weights=w),
    }


def test_resource_usage_and_cap():
    props = _hot_proposals()
    assert resource_usage(list(props.values())) == 1.0 + 1.0 + 0.0 + 0.3 + 0.6  # 2.9
    # rp=0.2 → cap = 0.7*5*(1-0.4*0.2) = 3.22 → 2.9 可行，不做任何修正
    sc = _scenario(rp=0.2)
    assert scenario_resource_cap(sc, 5) == pytest.approx(3.22)
    fixed, lowered, note = enforce_resource_cap(sc, props)
    assert lowered == [] and note == ""
    assert resource_usage(list(fixed.values())) <= 3.22 + 1e-9
    # rp=0.5 → cap = 3.5*0.8 = 2.8 → 2.9 超限 → 强制降级
    sc5 = _scenario(rp=0.5)
    fixed5, lowered5, note5 = enforce_resource_cap(sc5, props)
    assert lowered5 and note5
    assert resource_usage(list(fixed5.values())) <= 2.8 + 1e-9
    for a, p in fixed5.items():
        assert p.treatment_level <= props[a].treatment_level  # 只降不升


def test_engine_resource_infeasible_triggers_arbitration():
    """高资源压力场景：Σρ > cap 且无硬约束违规 → 强制仲裁（资源否决）+ 精炼后可行 + KKT 收敛。"""
    cfg = Config()
    cfg.llm.mode = "rule"
    engine = MANEEngine(cfg)
    sc = _scenario(rp=0.5)  # cap=2.8；rule 提案 Σρ=2.9 → 超限；rp=0.5<0.6 不触发约束 veto
    result = engine.run(sc)
    assert result.arbitration_triggered is True
    assert result.arbitration_reason == "resource"  # O3：仲裁原因审计
    assert result.resource_feasible is True
    assert result.resource_used is not None and result.resource_used <= result.resource_cap + 1e-9
    assert result.kkt_residual is not None and result.kkt_residual < 1.0
    # 仲裁裁决应含资源否决说明
    ruling = result.trajectory[-1].proposals[-1]
    assert "资源" in ruling.rationale


def test_assemble_recruits_family_via_proxy_decision():
    """O2（A3 修复）：无家庭冲突/宗教/经济压力，但存在代理决策情境
    （患者容量缺失）→ family 也必须入队。"""
    from conftest import make_scenario
    cfg = Config()
    cfg.llm.mode = "rule"
    engine = MANEEngine(cfg)
    sc = make_scenario()
    # 清掉所有原家庭召回信号
    sc.state.context["family_conflict"] = 0.0
    sc.state.context["religious_barrier"] = 0.0
    sc.state.context["insurance_stress"] = 0.0
    sc.state.preference["capacity"] = 0.3  # 患者决策能力缺失 → 代理决策
    active = engine.assemble(sc)
    assert "family" in {a.spec.id for a in active}
    # 对照：完全无代理决策情境 → family 不入队
    sc2 = make_scenario()
    sc2.state.context["family_conflict"] = 0.0
    sc2.state.context["religious_barrier"] = 0.0
    sc2.state.context["insurance_stress"] = 0.0
    sc2.state.preference["capacity"] = 1.0
    sc2.state.preference["info_completeness"] = 0.9
    sc2.state.qol["burden"] = 0.2
    sc2.state.qol["net_effect"] = 0.3
    active2 = engine.assemble(sc2)
    assert "family" not in {a.spec.id for a in active2}


def test_decide_excludes_catfish_from_resource():
    """B2 回归：仲裁资源口径排除鲶鱼（与 GNE 5 方口径一致）——cap 不再按 6 方放大、
    Σρ 不再虚高、降级名单不混入 catfish。"""
    from ethicalguard.mane.arbitration import Arbitrator
    from ethicalguard.mane.agents import EthicsCommitteeAgent
    from ethicalguard.config import default_agent_specs
    w = PrincipleVector.from_array(np.ones(4))
    props = [
        Proposal(agent="physician", treatment_level=2, principle_weights=w),
        Proposal(agent="family", treatment_level=2, principle_weights=w),
        Proposal(agent="patient", treatment_level=2, principle_weights=w),
        Proposal(agent="ethics_committee", treatment_level=2, principle_weights=w),
        Proposal(agent="hospital_admin", treatment_level=2, principle_weights=w),
        Proposal(agent="catfish", treatment_level=2, principle_weights=w),  # 鲶鱼
    ]
    arb = Arbitrator(EthicsCommitteeAgent(default_agent_specs()["ethics_committee"], None))
    sc = _scenario(rp=0.5)  # cap(5方)=2.8；5×0.6=3.0 超限；6方口径 cap=3.36 也会超
    ruling = arb.decide(sc, props, None, {})
    assert "catfish" not in ruling.rationale
    assert "3.00" in ruling.rationale and "2.80" in ruling.rationale  # 5 方口径 Σρ/cap


def test_engine_resource_feasible_no_unnecessary_arbitration():
    """普通场景（资源可行）不触发资源仲裁，KKT 收敛，治疗方案不被改动。"""
    cfg = Config()
    cfg.llm.mode = "rule"
    engine = MANEEngine(cfg)
    sc = _scenario(rp=0.2)
    result = engine.run(sc)
    assert result.resource_feasible is True
    assert result.kkt_residual is not None and result.kkt_residual < 1.0
    # 可行场景的仲裁只可能由未收敛/满意度低触发（规则模式通常都不触发）
    assert result.arbitration_triggered in (True, False)


def test_high_resource_pressure_uses_refinement_not_veto():
    """第五轮修复：rp>0.6 的场景（resource_hard 约束 violation>0）不再被 veto 一刀切
    ——check_veto 排除资源类约束，资源问题统一由 enforce_resource_cap 精细降级。"""
    cfg = Config()
    cfg.llm.mode = "rule"
    engine = MANEEngine(cfg)
    sc = _scenario(rp=0.8)  # make_scenario 的 resource_hard: value=0.8 > bound=0.6 → 旧逻辑触发 veto
    result = engine.run(sc)
    assert result.arbitration_triggered is True
    assert result.arbitration_reason == "resource"
    ruling = result.trajectory[-1].proposals[-1]
    assert "违反硬约束" not in ruling.rationale
    assert "资源 L1 否决" in ruling.rationale
    assert result.resource_feasible is True
    assert result.kkt_residual is not None and result.kkt_residual < 1.0
