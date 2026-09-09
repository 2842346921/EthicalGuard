"""双通道映射与 GNE 求解器接口测试（规则模式，无需 API）。"""
from __future__ import annotations

import numpy as np

from ethicalguard.data import DualChannelMapper, map_text_to_state, report_mapping
from ethicalguard.data.mapping import _merge_field
from ethicalguard.mane.gne_solver import GNEProblem, KKTResidual, solve_gne_from_proposals
from ethicalguard.config import Config
from ethicalguard.mane import MANEEngine


TEXT = ("Patient is comatose with severe sepsis; family refuses DNR; ICU beds scarce; "
        "family cannot afford expensive treatment; religious objection to blood products.")


def test_merge_field_semantic_first():
    """O1（A1 修复）：分歧 ≤ 阈值取均值；分歧 > 阈值取 LLM 值（语义通道优先）。

    规则通道的 0 多为"关键词未命中"而非"确信为 0"——旧策略分歧取规则值会把
    LLM 语义信息全部丢弃（实测平均分歧率 0.741 时退化为规则通道）。
    B3：LLM 中性默认（≈0.5）且规则无命中（≈0）→ 双方无证据 → 归零不采信。
    """
    assert abs(0.5 * (0.8 + 0.6) - 0.7) < 1e-9  # 分歧 0.2 ≤ 0.25 → 均值 0.7
    assert _merge_field(0.8, 0.6) == 0.7
    # 分歧 0.5 > 0.25 → 取 LLM 值（真实语义判断，非中性）
    assert _merge_field(0.0, 0.8) == 0.8
    assert _merge_field(0.9, 0.2) == 0.2
    # B3：LLM 中性 0.5 + 规则 0 → 无证据 → 取规则（0），不采信中性的 0.5
    assert _merge_field(0.0, 0.5) == 0.0
    assert _merge_field(0.0, 0.51) == 0.0
    # 规则有命中（0.3）时 LLM 中性 → 分歧 0.2 ≤ 0.25 → 均值 0.4
    assert abs(_merge_field(0.3, 0.5) - 0.4) < 1e-9


def test_rule_channel_only():
    """无 LLM 后端（rule 模式）→ 只走规则通道，分歧率为 0。"""
    mapper = DualChannelMapper(backend=None)
    r = mapper.map(TEXT)
    assert r.channel == "rule"
    assert r.disagreement_rate == 0.0
    assert r.flagged_fields == []
    assert r.state.medical["severity"] > 0
    assert r.state.context["resource_pressure"] > 0


def test_rule_channel_with_rule_backend():
    """显式传入 rule 模式后端 → 同样只走规则通道。"""
    cfg = Config()
    cfg.llm.mode = "rule"
    from ethicalguard.llm import make_backend
    mapper = DualChannelMapper(backend=make_backend(cfg.llm))
    r = mapper.map(TEXT)
    assert r.channel == "rule"


def test_report_mapping():
    """质量报告聚合。"""
    mapper = DualChannelMapper(backend=None)
    rep = report_mapping(mapper, [TEXT, "Patient wants full code, no resource issues."])
    assert rep.n_texts == 2
    assert rep.channel == "rule"
    assert rep.mean_disagreement == 0.0


def test_load_scenarios_round_robin_limit():
    """A 修复：limit 按数据集均分——--datasets 全选时每个数据集都能加载到，
    而不是第 1 个数据集独占全部名额（旧逻辑 principlismqa 先装满 50 个，后 4 个没跑）。"""
    import os
    import pytest
    from collections import Counter
    from ethicalguard.data import load_scenarios
    data_dir = r"E:\信息\论文\医疗诊断\多目标压力\论文\数据集"
    if not os.path.isdir(data_dir):
        pytest.skip("本地数据集目录不存在")
    ds = ["principlismqa", "medethiceval", "vital", "medethicsqa", "llmevalmed"]
    scs = load_scenarios(ds, data_dir, limit=10)  # 10 个 → 每数据集 ceil(10/5)=2 个
    dist = Counter(s.source.get("dataset") for s in scs)
    assert len(dist) >= 4, f"多数据集未都加载到: {dict(dist)}"
    assert sum(dist.values()) >= 10
    assert max(dist.values()) <= 2  # 均分：单数据集不超过 per 配额


def test_medethicsqa_bad_zip_diagnostic(tmp_path):
    """zip 损坏时给出可操作诊断（目录内容 + 修复建议），而非裸抛 BadZipFile。"""
    import pytest
    from ethicalguard.data.medethicsqa import MedEthicsQAAdapter
    folder = tmp_path / "MedEthicsQA"
    folder.mkdir()
    (folder / "MedEthicsQA_open.zip").write_bytes(b"definitely not a zip file")
    ad = MedEthicsQAAdapter(str(tmp_path))
    with pytest.raises(RuntimeError, match="损坏"):
        list(ad.load_scenarios())


def test_medethicsqa_missing_diagnostic(tmp_path):
    """数据文件缺失时给出目录内容诊断。"""
    import pytest
    from ethicalguard.data.medethicsqa import MedEthicsQAAdapter
    folder = tmp_path / "MedEthicsQA"
    folder.mkdir()
    (folder / "readme.txt").write_text("placeholder", encoding="utf-8")
    ad = MedEthicsQAAdapter(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="未找到"):
        list(ad.load_scenarios())


def test_gne_solver_api_complete():
    """完整 KKT 四分量 + 平衡度报告（solve_gne_from_proposals 接口）。"""
    from conftest import make_scenario
    cfg = Config()
    cfg.llm.mode = "rule"
    engine = MANEEngine(cfg)
    scenario = make_scenario()
    active = engine.assemble(scenario)
    proposals = {a.spec.id: a.act(scenario, 1, []) for a in active}
    sol = solve_gne_from_proposals(active, scenario, proposals,
                                   reliability=np.ones(len(active)) / len(active))
    assert isinstance(sol.kkt, KKTResidual)
    for comp in (sol.kkt.stationarity, sol.kkt.primal_feasibility,
                 sol.kkt.dual_feasibility, sol.kkt.complementarity, sol.kkt.total):
        assert comp >= 0.0
    assert np.allclose(sol.weights.sum(axis=1), 1.0, atol=1e-6)
    assert abs(sol.collective.sum() - 1.0) < 1e-6
    assert 0.0 <= sol.fdbi_achieved <= 1.0
    assert isinstance(sol.balance_met, bool)


def test_gne_problem_standalone():
    """独立构造 GNEProblem（Toy 3-agent 冲突游戏）能求解并给出 KKT。"""
    sat = np.array([[0.8, 0.3, 0.2, 0.4], [0.2, 0.4, 0.8, 0.3], [0.3, 0.5, 0.3, 0.7]])
    prob = GNEProblem(satisfaction=sat, balance_tau=0.75)
    sol = prob.solve(max_iter=2000)
    assert sol.collective.shape == (4,)
    assert abs(sol.collective.sum() - 1.0) < 1e-6
