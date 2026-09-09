"""核心类型 + 映射 + 状态机 + GNE 求解器 + 指标 的单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from ethicalguard.types import PrincipleVector
from ethicalguard.data.mapping import map_text_to_state
from ethicalguard.mane import GNEProblem, NegotiationFSM
from ethicalguard.eval import metrics as M
from ethicalguard.resilience import metrics as R


def test_principle_vector_normalize():
    v = PrincipleVector.from_array([1.0, 1.0, 1.0, 1.0]).normalized()
    assert abs(v.as_array().sum() - 1.0) < 1e-6


def test_halved_drift_asymmetry():
    a = PrincipleVector(beneficence=0.3, nonmaleficence=0.7, autonomy=0.0, justice=0.0)
    b = PrincipleVector(beneficence=0.3, nonmaleficence=0.4, autonomy=0.0, justice=0.0)
    # 放弃 N（权重 3）比放弃其它原则漂移更大
    assert a.halved_drift(b) > 0.0


def test_mapping_extracts_state():
    text = "Patient is comatose with severe sepsis; family refuses DNR; ICU beds scarce."
    s = map_text_to_state(text)
    assert s.medical["severity"] > 0
    assert s.preference["attitude_refuse"] > 0
    assert s.context["resource_pressure"] > 0
    assert 0.0 <= s.mui() <= 1.0 and 0.0 <= s.cci() <= 1.0


def test_state_machine_backtrack():
    fsm = NegotiationFSM()
    assert fsm.advance({"flags": {}}) == "S1_assemble"
    assert fsm.advance({"flags": {}}) == "S2_propose"
    assert fsm.advance({"flags": {}}) == "S3_align"
    # 漂移越界 → 回溯
    assert fsm.advance({"flags": {"drift_exceeded": True}}) == "R_backtrack"
    assert fsm.advance({"flags": {}}) == "S2_propose"


def test_gne_solver_converges():
    n = 3
    sat = np.array([
        [0.8, 0.3, 0.2, 0.4],
        [0.2, 0.4, 0.8, 0.3],
        [0.3, 0.5, 0.3, 0.7],
    ])
    prob = GNEProblem(satisfaction=sat, balance_tau=0.85)
    sol = prob.solve(max_iter=2000)
    # 权重在单纯形上
    assert np.allclose(sol.weights.sum(axis=1), 1.0, atol=1e-6)
    # 集体向量也在单纯形上
    assert abs(sol.collective.sum() - 1.0) < 1e-6
    # 全 KKT 残差应较小（正则化 GNE 有内点解）
    assert sol.kkt.total < 0.5
    # 四项 KKT 分量均已报告
    for comp in (sol.kkt.stationarity, sol.kkt.primal_feasibility,
                 sol.kkt.dual_feasibility, sol.kkt.complementarity):
        assert comp >= 0.0


def test_gne_resource_constraint_treatment_levels():
    """资源约束升级：Σρ(t_i) ≤ cap，治疗等级携带资源用度。"""
    sat = np.array([[0.6, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5]])
    # 3 个 Agent 都选积极治疗(等级3) → ρ=1.0 每个 → 总用度 3.0
    prob_hot = GNEProblem(satisfaction=sat, treatments=np.array([3, 3, 3]),
                          treatment_resource=np.array([0.0, 0.3, 0.6, 1.0]),
                          resource_cap=2.0)
    sol_hot = prob_hot.solve(max_iter=500)
    assert sol_hot.resource_used == 3.0
    assert sol_hot.resource_feasible is False  # 3.0 > 2.0 违规

    # 全部舒适护理(等级0) → ρ=0 → 总用度 0.0 ≤ cap
    prob_cool = GNEProblem(satisfaction=sat, treatments=np.array([0, 0, 0]),
                           treatment_resource=np.array([0.0, 0.3, 0.6, 1.0]),
                           resource_cap=2.0)
    sol_cool = prob_cool.solve(max_iter=500)
    assert sol_cool.resource_used == 0.0
    assert sol_cool.resource_feasible is True


def test_fdbi_pci():
    balanced = np.array([0.25, 0.25, 0.25, 0.25])
    skewed = np.array([0.9, 0.03, 0.03, 0.04])
    assert M.fdbi(balanced) > M.fdbi(skewed)
    assert M.pci(skewed) > M.pci(balanced)


def test_resilience_consistency_detects_collapse():
    vn = np.array([0.5, 0.5, 0.5, 0.5])
    collapse = np.array([0.2, 0.2, 0.2, 0.2])  # 所有原则系统性坍缩
    same_shape = np.array([0.5, 0.5, 0.5, 0.5])
    assert R.consistency(vn, collapse) < R.consistency(vn, same_shape)


def test_rubric_alignment_bilingual():
    """修复回归：英文 keypoints + 中文 rationale 应 > 0（跨语言不匹配 bug）。"""
    kps = ["respect patient autonomy", "family refusal", "withhold futile treatment"]
    zh = "患者自主应被尊重；家属拒绝过度治疗；对于无效治疗应予以尊重并沟通"
    score = M.rubric_alignment(zh, kps)
    assert score > 0.0
    assert score <= 1.0
    # 中文 keypoint 直接命中（连续子串）
    assert M.rubric_alignment("我们尊重患者自主意愿", ["自主意愿"]) > 0.0
    # 完全无关 → 0
    assert M.rubric_alignment("天气很好，今天出门散步", kps) == 0.0
    # 空 keypoints → 0
    assert M.rubric_alignment("任何理由", []) == 0.0


def test_rubric_alignment_topic_sentences():
    """主题式匹配回归：英文**长句** keypoints（前 3 词是虚词）vs 中文 rationale。

    旧逻辑"取前 3 词"对长句必失败（physicians/should/after...）；主题法扫全句
    提取伦理主题（communication/continuity/cost/autonomy...）再匹配中文对应词。
    对应 PQ-1-2 实测：旧 0.167 → 新 1.0。
    """
    kps = [
        "Physicians should inform patients of changes in practice management "
        "through verbal or written communication and be prepared to address patients' questions.",
        "When considering changes in ownership, physicians must take patients' interests into account "
        "to ensure continuity and quality of care are not compromised.",
        "Patients should be informed that they always have the right to change their clinician.",
    ]
    zh = ("患者对医患关系连续性、成本与质量的潜在影响表现出明确关切；"
          "需通过结构化沟通进行伦理权衡，尊重患者知情权与自主决策，"
          "保障长期医患关系公正与患者利益。")
    score = M.rubric_alignment(zh, kps)
    assert score >= 2 / 3  # communication/continuity/autonomy 主题至少命中 2 条
    # 无关文本 → 0
    assert M.rubric_alignment("今天天气不错，出门散步", kps) == 0.0


def test_distribution_score_js_l1():
    p = [0.5, 0.3, 0.1, 0.1]
    q = [0.25, 0.25, 0.25, 0.25]
    d = M.distribution_score(p, q)
    assert 0.0 <= d["js"] <= 1.0
    assert d["l1"] == pytest.approx(np.abs(np.asarray(p) - np.asarray(q)).sum(), abs=1e-6)
    # 相同分布 → JS=0
    assert M.js_distance(p, p) == pytest.approx(0.0, abs=1e-9)


def test_ethical_composite_veto_gates():
    """ECS：底线违反/资源不可行/KKT 不达标 → 场景一票否决（0 分）。"""
    floors = [0.15, 0.20, 0.15, 0.20]
    # 场景 A 守住底线且可行/KKT 达标；场景 B 跌破底线
    va = np.array([0.25, 0.30, 0.25, 0.20])
    vb = np.array([0.25, 0.10, 0.30, 0.35])  # N=0.10 < 0.20 底线
    ecs = M.ethical_composite([va, vb], [0.9, 0.8], floors,
                              feasibles=[True, True], kkts=[1e-7, 1e-7])
    assert ecs["pass_rate"] == 0.5
    assert ecs["floor_viol_rate"] == 0.5
    # B 被否决 → 0 分；A 保留
    assert ecs["scores"][1] == 0.0
    assert ecs["scores"][0] > 0.0
    # 资源不可行 → 否决
    ecs2 = M.ethical_composite([va], [0.9], floors, feasibles=[False], kkts=[1e-7])
    assert ecs2["scores"][0] == 0.0
    # KKT 不达标 → 否决（kkt_gate=True 时）
    ecs3 = M.ethical_composite([va], [0.9], floors, feasibles=[True], kkts=[0.9])
    assert ecs3["scores"][0] == 0.0
    # w/o GNE（kkt=None + kkt_gate=False）→ 不因 KKT 否决
    ecs4 = M.ethical_composite([va], [0.9], floors, feasibles=[True], kkts=[None], kkt_gate=False)
    assert ecs4["scores"][0] > 0.0
    # 全否决 → mean 0
    ecs5 = M.ethical_composite([vb], [0.8], floors)
    assert ecs5["mean"] == 0.0
