"""端到端冒烟测试：MANE 协商 + 韧性评估（规则模式，无需 API）。"""
from __future__ import annotations

import os

import pytest

from ethicalguard.config import Config
from ethicalguard.mane import MANEEngine
from ethicalguard.resilience import ResilienceEvaluator


def test_mane_run_rule_mode():
    from conftest import make_scenario
    cfg = Config()
    cfg.llm.mode = "rule"
    engine = MANEEngine(cfg)
    result = engine.run(make_scenario())
    assert result.scenario_id.startswith("SYN")
    assert result.rounds >= 1
    assert result.final_proposal is not None
    assert result.final_vector is not None
    assert result.kkt_residual is not None
    # 至少组队了医师 + 伦理委员会
    agents = {p.agent for r in result.trajectory for p in r.proposals}
    assert "physician" in agents and "ethics_committee" in agents


def test_resilience_report():
    from conftest import make_scenario
    cfg = Config()
    cfg.llm.mode = "rule"
    engine = MANEEngine(cfg)
    evaluator = ResilienceEvaluator(engine.run, cfg.resilience, mode="rule")
    report = evaluator.evaluate(make_scenario(), max_perturbations=3)
    assert 0.0 <= report.overall <= 1.0
    assert report.verdict in ("ACCEPT", "RENEGOTIATE")
    assert report.bsp is not None
    # 规则模式 → 一致性基线档
    assert report.mode == "rule"
    assert report.baseline_only is True
    assert len(report.per_perturbation) == 3


def test_adapters_load_real_data():
    """若本地数据集目录存在，则各适配器能产出场景。"""
    data_dir = "/mnt/zhangheng2025/EthicalGuard/data"
    if not os.path.isdir(data_dir):
        pytest.skip("本地数据集目录不存在")

    from ethicalguard.data import get_adapter
    for name in ("principlismqa", "medethiceval", "vital", "medethicsqa", "llmevalmed"):
        adapter = get_adapter(name, data_dir)
        scenarios = list(adapter.load_scenarios())
        assert len(scenarios) > 0, f"{name} 未产出场景"
        assert scenarios[0].scenario_id
