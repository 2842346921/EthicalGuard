"""冲突识别（detection）模块测试。"""
from __future__ import annotations

import pytest

from ethicalguard.types import ConflictReport, ConflictType
from ethicalguard.detection import RuleConflictDetector, make_detector
from ethicalguard.config import Config


def test_rule_detector_high_risk():
    """医疗-意愿冲突：MUI 高 + VPI 低 → ERS 高、类型 medical_preference。"""
    from conftest import make_scenario
    sc = make_scenario()  # 合成场景：严重病情 + 拒绝 + 家庭冲突
    det = RuleConflictDetector()
    rep = det.detect(sc)
    assert isinstance(rep, ConflictReport)
    assert 0.0 <= rep.ers <= 1.0
    # 该场景 medical_preference 或 preference_family 信号应 > 0
    assert rep.ers >= 0.1
    assert rep.conflict_type in (ConflictType.MEDICAL_PREFERENCE,
                                 ConflictType.PREFERENCE_FAMILY,
                                 ConflictType.MEDICAL_RESOURCE,
                                 ConflictType.PREFERENCE_QOL,
                                 ConflictType.NONE)


def test_rule_detector_low_risk():
    """无冲突场景：ERS 应低、类型 none。"""
    from conftest import make_scenario
    sc = make_scenario(scenario_id="SYN-LOW", resource_pressure=0.0)
    # 把状态改为无冲突：低严重度、意愿清晰、无家庭冲突
    sc.state.medical["severity"] = 0.1
    sc.state.preference["clarity"] = 0.9
    sc.state.preference["capacity"] = 1.0
    sc.state.preference["attitude_refuse"] = 0.0
    sc.state.context["family_conflict"] = 0.0
    sc.state.context["resource_pressure"] = 0.0
    # make_scenario 默认带法律/经济压力 → 也清零（否则 CCI>0 抬高 ERS）
    sc.state.context["legal_constraint"] = 0.0
    sc.state.context["insurance_stress"] = 0.0
    # make_scenario 默认 QoL 损益为负（net_effect=-0.4）→ 高 VPI 会凑出 preference_qol 信号，清零
    sc.state.qol["net_effect"] = 0.0
    det = RuleConflictDetector()
    rep = det.detect(sc)
    assert rep.ers < 0.5


def test_make_detector_rule_mode():
    """rule 后端 → 规则检测器；api 后端 → LLM 检测器。"""
    cfg = Config()
    cfg.llm.mode = "rule"
    from ethicalguard.llm import make_backend
    det = make_detector(make_backend(cfg.llm))
    assert det.channel == "rule"

    cfg.llm.mode = "api"
    cfg.llm.api.base_url = "http://localhost:8000/v1"
    det2 = make_detector(make_backend(cfg.llm))
    assert det2.channel == "llm"


def test_llm_detector_prompt_threshold_replace_safe():
    """B1 回归：prompt 含 JSON 字面量花括号，threshold 注入必须用 replace 而非 format。
    format() 会把 {"beneficence":...} 当字段名抛 KeyError（曾致 LLM 检测器 10/10 回退规则）。"""
    from ethicalguard.detection.llm import _PROMPT_TMPL
    p = _PROMPT_TMPL.replace("{threshold}", "0.5")
    assert "ers<0.5" in p
    assert '{"beneficence' in p  # JSON 字面量花括号保留
    with pytest.raises((KeyError, IndexError, ValueError)):
        _PROMPT_TMPL.format(threshold=0.5)  # 回归防护：format 必然炸


def test_fit_ers_temperature_calibration():
    """P-7.1：温度校准——sigmoid 饱和不足的 logits（挤在 0.5-0.7）经 t>1 拉开后更贴近弱监督标签。"""
    import numpy as np
    from ethicalguard.detection.dataset import fit_ers_temperature
    # 模拟：弱监督标签 0.85/0.62/0.55，模型 logit 对应温和概率 0.56-0.65（挤在一起）
    p_mild = np.array([0.56, 0.60, 0.65, 0.58, 0.62, 0.55])
    logits = np.log(p_mild / (1.0 - p_mild))
    targets = np.array([0.85, 0.62, 0.55, 0.85, 0.62, 0.55])
    t = fit_ers_temperature(logits, targets)
    assert t > 1.0  # 模型过度温和 → 需要 t>1 拉开
    def mse(tt):
        p = 1.0 / (1.0 + np.exp(-tt * logits))
        return float(np.mean((p - targets) ** 2))
    assert mse(t) <= mse(1.0) + 1e-9
    # 样本不足 → 返回 1.0（不校准）
    assert fit_ers_temperature(logits[:2], targets[:2]) == 1.0


def test_action_for_ers_threshold_bands():
    """threshold 接线：action 分档跟随 config.detection.threshold（默认 0.5 行为不变）。"""
    from ethicalguard.detection.supervised import action_for_ers
    # 默认 threshold=0.5
    assert action_for_ers(0.2) == "observe"
    assert action_for_ers(0.4, 0.5) == "review_24h"
    assert action_for_ers(0.6, 0.5) == "ethics_consult"
    assert action_for_ers(0.8, 0.5) == "intervene"
    # 收紧 threshold=0.55：0.5-0.55 从 consult 降为 review_24h
    assert action_for_ers(0.52, 0.55) == "review_24h"
    assert action_for_ers(0.56, 0.55) == "ethics_consult"
    # 放松 threshold=0.45：0.45-0.5 从 review 升为 consult
    assert action_for_ers(0.48, 0.45) == "ethics_consult"


def test_engine_attaches_conflict_report():
    """MANE 引擎在协商结果上附带冲突识别报告。"""
    from conftest import make_scenario
    cfg = Config()
    cfg.llm.mode = "rule"
    from ethicalguard.mane import MANEEngine
    engine = MANEEngine(cfg)
    result = engine.run(make_scenario())
    assert result.conflict_report is not None
    assert 0.0 <= result.conflict_report.ers <= 1.0
