"""韧性层专项测试：压力注入通道 + 规则 Agent 压力响应 + 两档证据（基线一致性 vs 完整韧性）。

全部离线可跑（规则模式，无需 API）。
"""
from __future__ import annotations

import numpy as np
import pytest

from ethicalguard.config import Config, default_agent_specs
from ethicalguard.mane import MANEEngine
from ethicalguard.mane.agents import build_agent
from ethicalguard.resilience import ResilienceEvaluator
from ethicalguard.resilience.stress_engine import default_perturbations, inject
from ethicalguard.types import PRINCIPLES, Perturbation


def _scenario():
    from conftest import make_scenario
    return make_scenario()


def _agent(agent_id: str):
    from ethicalguard.mane.agents import AGENT_CLASSES
    spec = default_agent_specs()[agent_id]
    return AGENT_CLASSES[agent_id](spec, None)  # backend=None → 规则模式


def test_inject_records_stress_and_mutates_state():
    sc = _scenario()
    p = default_perturbations()[0]  # resource
    sc2 = inject(sc, p)
    assert sc2.stress is not None and sc2.stress.type == "resource"
    assert sc2.state.context["resource_pressure"] > sc.state.context["resource_pressure"]
    # 原场景不受污染
    assert sc.stress is None


def test_inject_scales_delta_by_intensity():
    """强度标定：实际注入量 = delta × intensity。"""
    from ethicalguard.resilience.stress_engine import scaled
    p = default_perturbations()[0]  # resource: context_delta={"resource_pressure": 0.5}
    sc = _scenario()
    rp0 = sc.state.context["resource_pressure"]  # 0.2
    half = inject(sc, scaled(p, 0.5))
    full = inject(sc, scaled(p, 1.0))
    assert half.state.context["resource_pressure"] == pytest.approx(rp0 + 0.25, abs=1e-9)
    assert full.state.context["resource_pressure"] == pytest.approx(rp0 + 0.5, abs=1e-9)
    # scaled() 保留类型与基准 delta，只换强度
    pp = scaled(p, 0.25)
    assert pp.type == p.type and pp.intensity == 0.25
    assert pp.context_delta == p.context_delta


def test_rule_agent_stress_response_deterministic():
    """规则 Agent 对压力的确定性响应：资源压力让医院管理降级（2→0），
    病情加重让医师行善权重上升。这是"基线一致性"的刺激通道。"""
    admin = _agent("hospital_admin")
    base = admin._rule_act(_scenario(), 1, [])
    p = next(p for p in default_perturbations() if p.type == "resource")
    stressed = admin._rule_act(inject(_scenario(), p), 1, [])
    assert stressed.treatment_level < base.treatment_level

    phys = _agent("physician")
    base_b = phys._rule_act(_scenario(), 1, []).principle_weights.as_array()
    p2 = Perturbation(type="disease", description="病情加重", medical_delta={"severity": 0.3})
    b_stress = phys._rule_act(inject(_scenario(), p2), 1, []).principle_weights.as_array()
    assert b_stress[0] > base_b[0]  # B 权重上升


def test_rule_agent_stress_response_scales_with_intensity():
    """规则 Agent 的压力响应随强度缩放：低强度治疗等级不变，满档才降级；
    权重修正量随强度线性放大。强度=1.0 时与旧行为一致。"""
    from ethicalguard.resilience.stress_engine import scaled
    admin = _agent("hospital_admin")
    p = next(p for p in default_perturbations() if p.type == "resource")
    light = admin._rule_act(inject(_scenario(), scaled(p, 0.25)), 1, [])
    full = admin._rule_act(inject(_scenario(), scaled(p, 1.0)), 1, [])
    assert light.treatment_level == 2   # 0.25 档：t_d=round(-0.25)=0，资源压力 0.325 未越 0.5 阈值
    assert full.treatment_level == 0    # 满档：降级

    phys = _agent("physician")
    p2 = Perturbation(type="disease", description="病情加重", medical_delta={"severity": 0.3})
    w_half = phys._rule_act(inject(_scenario(), scaled(p2, 0.5)), 1, []).principle_weights.as_array()
    w_full = phys._rule_act(inject(_scenario(), scaled(p2, 1.0)), 1, []).principle_weights.as_array()
    assert w_half[0] < w_full[0]  # B 权重修正随强度增大


def test_logical_trap_prompt_content_in_stress_block():
    """P1-2：logical_trap 压力类型带具体叙事内容，提示词压力块必须输出它
    （仅数值偏移 info_completeness=-0.2 模型不买账，故给论点原文）。"""
    from ethicalguard.resilience.stress_engine import scaled
    p = next(p for p in default_perturbations() if p.type == "logical_trap")
    assert p.prompt_content, "logical_trap 应配置 prompt_content（具体错误论点原文）"
    sc = inject(_scenario(), p)
    block = _agent("physician")._stress_block(sc)
    assert "压力情境详情" in block
    assert "病历显示患者从未" in block  # 论点原文进入提示词
    # scaled() 保留 prompt_content（强度网格扫描时叙事内容不丢失）
    assert scaled(p, 0.5).prompt_content == p.prompt_content


def test_resilience_report_rule_is_baseline_only():
    """rule 模式 → baseline_only=True：仅一致性基线，且逐压力记录齐全。"""
    cfg = Config()
    cfg.llm.mode = "rule"
    engine = MANEEngine(cfg)
    evaluator = ResilienceEvaluator(engine.run, cfg.resilience, mode="rule")
    report = evaluator.evaluate(_scenario(), max_perturbations=3)
    assert report.mode == "rule"
    assert report.baseline_only is True
    assert len(report.per_perturbation) == 3
    for rec in report.per_perturbation:
        assert rec["type"] in {p.type for p in default_perturbations()}
        assert 0.0 <= rec["consistency"] <= 1.0
        assert len(rec["v_shift"]) == 4
    # 规则模式一般不会真"放弃原则"（确定性微调不至于跌破底线）
    assert set(report.abandoned).issubset(set(PRINCIPLES))


def test_resilience_repeat_reports_recovery_std():
    """P1-1：repeat>1 时基线/恢复重跑 N 次，R_recover 报 mean±std（规则模式确定性 → std=0）。"""
    cfg = Config()
    cfg.llm.mode = "rule"
    cfg.resilience.repeat = 3
    engine = MANEEngine(cfg)
    evaluator = ResilienceEvaluator(engine.run, cfg.resilience, mode="rule")
    report = evaluator.evaluate(_scenario(), max_perturbations=2)
    assert report.recovery_std == 0.0  # 规则模式确定性重跑无波动
    assert 0.0 <= report.l3_recoverability <= 1.0
    assert report.overall >= 0.0


def test_resilience_report_llm_mode_flag():
    """mode=local → baseline_only=False（完整韧性档；标签由 03_stress 按真实后端传入）。"""
    cfg = Config()
    cfg.llm.mode = "rule"  # 只验证标签管道，不真的调 LLM
    engine = MANEEngine(cfg)
    evaluator = ResilienceEvaluator(engine.run, cfg.resilience, mode="local")
    report = evaluator.evaluate(_scenario(), max_perturbations=1)
    assert report.mode == "local"
    assert report.baseline_only is False


def test_stress_response_intensity_grid():
    """强度网格：stress_response 含每类压力×每档强度的曲线与临界强度。"""
    cfg = Config()
    cfg.resilience.intensity_grid = [0.5, 1.0]
    engine = MANEEngine(cfg)
    evaluator = ResilienceEvaluator(engine.run, cfg.resilience, mode="rule")
    report = evaluator.evaluate(_scenario(), max_perturbations=2)
    assert len(report.stress_response) == 2
    for rec in report.stress_response:
        assert rec["intensities"] == [0.5, 1.0]
        assert len(rec["consistency"]) == 2
        assert len(rec["abandoned_count"]) == 2
        assert len(rec["abandoned_principles"]) == 2
        # 临界强度要么未触发（None），要么落在网格内
        for crit in (rec["critical_consistency"], rec["critical_abandonment"]):
            assert crit is None or crit in (0.5, 1.0)
    # 主指标取满档切片 → per_perturbation 仍是"每压力一条"
    assert len(report.per_perturbation) == 2
    # abandonment_intensity 字段存在（rule 模式通常为空字典）
    assert isinstance(report.abandonment_intensity, dict)
    assert set(report.abandonment_intensity).issubset(set(PRINCIPLES))


def test_agent_abandoned_detects_floor_break():
    """_agent_abandoned：反事实放弃定义——基线上守住、压力下跌破底线才算放弃；
    基线本来就低于底线的（如患者天然低权重）不叫放弃。
    （helper 只检查多方协商轮，忽略仲裁轮的单方裁决，故构造两个提案。）"""
    from ethicalguard.resilience.evaluator import _agent_abandoned
    from ethicalguard.types import NegotiationResult, NegotiationRound, Proposal, PrincipleVector

    p_patient = Proposal(
        agent="patient", round=1, treatment_level=0,
        principle_weights=PrincipleVector(beneficence=0.3, nonmaleficence=0.3, autonomy=0.1, justice=0.3),
        confidence=0.5, rationale="压力下放弃自主",
    )
    p_family = Proposal(
        agent="family", round=1, treatment_level=1,
        principle_weights=PrincipleVector(beneficence=0.3, nonmaleficence=0.3, autonomy=0.2, justice=0.2),
        confidence=0.5, rationale="未跌破底线",
    )
    res = NegotiationResult(
        scenario_id="T", rounds=1, converged=True,
        trajectory=[NegotiationRound(index=1, proposals=[p_patient, p_family])],
    )
    floors = np.array([0.15, 0.20, 0.15, 0.20])
    # 基线：患者自主权严格守住 0.3（>0.15）→ 压力下 0.1 跌破 → 计为放弃
    base_w = {"patient": np.array([0.3, 0.3, 0.3, 0.3]),
              "family": np.array([0.3, 0.3, 0.3, 0.3])}
    out = _agent_abandoned(res, floors, base_w)
    assert out.get("patient") == ["autonomy"]  # 0.10 < 0.15
    assert "family" not in out
    # 反事实：基线自主权本来就 < 底线（0.1）→ 不算放弃
    base_low = {"patient": np.array([0.3, 0.3, 0.1, 0.3])}
    assert _agent_abandoned(res, floors, base_low).get("patient") is None
    # 反事实：基线恰好站在底线上（0.15）→ 不叫"守住"，也不计放弃
    base_on = {"patient": np.array([0.3, 0.3, 0.15, 0.3])}
    assert _agent_abandoned(res, floors, base_on).get("patient") is None


def test_negotiation_quality_summary():
    """闭环环 9：协商质量摘要（个体让步 vs 集体稳定）。"""
    from ethicalguard.resilience.negotiation_quality import (
        collective_volatility, concession_inequality, concessions, summary,
    )
    from ethicalguard.types import NegotiationRound, Proposal, PrincipleVector

    def prop(agent, w, rnd):
        return Proposal(agent=agent, round=rnd, treatment_level=2,
                        principle_weights=PrincipleVector.from_array(np.array(w, dtype=float)))

    # 两轮：每方从自己立场向集体靠拢（让步发生），集体从偏斜走向平衡
    r1 = NegotiationRound(index=1, collective_vector=PrincipleVector.from_array([0.5, 0.2, 0.2, 0.1]),
                          proposals=[prop("physician", [0.6, 0.2, 0.1, 0.1], 1),
                                     prop("patient", [0.1, 0.1, 0.7, 0.1], 1)])
    r2 = NegotiationRound(index=2, collective_vector=PrincipleVector.from_array([0.35, 0.25, 0.25, 0.15]),
                          proposals=[prop("physician", [0.45, 0.25, 0.15, 0.15], 2),
                                     prop("patient", [0.2, 0.15, 0.5, 0.15], 2)])
    c = concessions([r1, r2])
    assert c["physician"] > 0.0 and c["patient"] > 0.0
    assert 0.0 <= concession_inequality(c)
    assert collective_volatility([r1, r2]) > 0.0
    s = summary([r1, r2])
    assert s["concession_mean"] > 0.0
    assert s["n_agents_conceded"] == 2.0
    assert s["collective_volatility"] > 0.0
    # 单轮无让步：各方让步均为 0（键存在，幅度为 0）
    assert concessions([r1]) == {"physician": 0.0, "patient": 0.0}


def test_process_trace_reports_closed_loop():
    """闭环证据链：规则模式协商应产出 process_trace，
    且 catfish 异议进历史但未被消费（subsequent_collective_delta≈0，规则模式
    compromise 不读提案文本）——这恰是 rule/local 消融差异的过程级证据。"""
    cfg = Config()
    cfg.llm.mode = "rule"
    engine = MANEEngine(cfg)
    res = engine.run(_scenario())
    pt = res.process_trace
    assert pt["dissent_in_history"] is True  # Catfish 提案写入历史（use_catfish=True 默认）
    assert pt["catfish_rounds"]
    # 规则模式：异议内容不被消费（compromise 只读集体向量，不读提案文本）→ Δ 应极小。
    # 非严格 0：贝叶斯编排的可靠性权重演化也会微动集体向量（0.006 量级 ≪ 让步量级 0.1+），
    # 即该位移来自"谁可靠"而非"异议说了什么"——对比 local 模式的显著位移即为消融证据。
    assert pt["subsequent_collective_delta"] < 0.05
    assert "arbitration_triggered" in pt and "fusion_applied" in pt
    assert "resource_corrected" in pt and "kkt_residual" in pt and "floors_ok" in pt
    # 终态守住 L3 底线（仲裁/GNE 修复环生效）
    assert pt["floors_ok"] is True
