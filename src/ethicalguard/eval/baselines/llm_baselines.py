"""真实 LLM 基线（从 scripts/11_real_llm_baselines.py 迁移的可复用实现）。

三个"同一基座、同一场景、不同协议"的对照（E2 + 实验四内容对比共用）：
  llm_single          : 真·单 LLM（physician 角色，显式"守护行善"——角色偏置对照）
  neutral_single_llm  : 真·单 LLM（中性，无角色/原则偏置）——主基线
  medagents_style     : MedAgents 式五方 LLM 协作（无历史协商/无 GNE/无仲裁），等权平均
  MANE                : 完整框架（在 scripts/22 里用 MANEEngine 跑）

迁移原因：scripts/22_content_cross_method（实验四：同题同尺子跨方法内容对比）需要
与 11 共用同一份基线实现，避免双份代码漂移。11 改为从本模块 import（行为不变）。
"""
from __future__ import annotations

import numpy as np

from ...types import PrincipleVector, Proposal, Scenario, Syllogism

FLOORS = np.array([0.15, 0.20, 0.15, 0.20])


def _collective_mean(weights: list) -> PrincipleVector:
    w = np.mean(np.array(weights), axis=0)
    return PrincipleVector.from_array(w / w.sum())


def llm_single(scenario: Scenario, backend, agent) -> Proposal:
    """真·单 LLM：一个角色（医师，无协商）直接决策。

    ⚠️ physician 角色 prompt 显式"守护行善"——这是**角色偏置基线**；
    论文主基线应是 neutral_single_llm（11 docstring 对照逻辑：若 neutral 也违反 J →
    模型内在偏置；若 neutral 守住而 physician 违反 → 违反来自角色 prompt）。
    """
    return agent._llm_act(scenario, 1, [])


NEUTRAL_SYS = (
    "你是一名中立的临床伦理决策者。请基于临床伦理四原则（行善/不伤害/自主/公正）"
    "对以下病例输出结构化 JSON 决策："
    '{"treatment_level":0-3,"principle_weights":{"beneficence":x,"nonmaleficence":x,"autonomy":x,"justice":x},'
    '"confidence":0-1,"rationale":"...","syllogism":{"major_premise":"","minor_premise":"","conclusion":""}}'
    "（principle_weights 四项之和应为 1。你不代表任何单一角色的立场，须在四原则间独立权衡："
    "不因临床获益而忽视公正，不因尊重自主而忽视不伤害。治疗强度 0 仅用于临终/获益极低情境。）"
)


def _parse_proposal(text: str, agent_id: str, t_default: int = 2) -> Proposal:
    """把 LLM 文本解析为 Proposal（复用 base._extract_json）。"""
    from ...mane.agents.base import _extract_json, _safe_int
    try:
        data = _extract_json(text)
    except Exception:  # noqa: BLE001
        data = {"principle_weights": {}, "confidence": 0.2}
    w = data.get("principle_weights", {})
    syl = None
    _s = data.get("syllogism")
    if isinstance(_s, dict):
        # 容错：异构模型（Mistral 等）常缺 minor_premise/conclusion → 缺字段补空，不抛错
        # （2026-09-07：Mistral-7B 因 Syllogism 缺 minor_premise 抛 ValidationError 中断整个跨模型跑）
        try:
            syl = Syllogism(
                major_premise=str(_s.get("major_premise", "")),
                minor_premise=str(_s.get("minor_premise", "")),
                conclusion=str(_s.get("conclusion", "")),
            )
        except Exception:  # noqa: BLE001
            syl = None
    return Proposal(
        agent=agent_id, round=1,
        treatment_level=_safe_int(data.get("treatment_level"), t_default),
        principle_weights=PrincipleVector(
            beneficence=float(w.get("beneficence", 0.25)),
            nonmaleficence=float(w.get("nonmaleficence", 0.25)),
            autonomy=float(w.get("autonomy", 0.25)),
            justice=float(w.get("justice", 0.25)),
        ).normalized(),
        confidence=float(data.get("confidence", 0.5)),
        rationale=str(data.get("rationale", "")),
        syllogism=syl,
    )


def neutral_single_llm(scenario: Scenario, backend) -> Proposal:
    """真·单 LLM（中性）：无角色/原则偏置，直接平衡四原则。"""
    from ...mane.agents.base import _extract_json  # noqa: F401 (warm import)
    user_p = f"病例：{scenario.raw_text[:2000]}"
    state = scenario.state
    state_txt = "\n".join(f"[{dim}] " + ", ".join(f"{k}={v:.2f}" for k, v in d.items())
                          for dim, d in (("medical", state.medical), ("preference", state.preference),
                                         ("qol", state.qol), ("context", state.context)) if d)
    user_p += f"\n\n当前量化状态：\n{state_txt}"
    text = backend.complete(NEUTRAL_SYS, user_p)
    return _parse_proposal(text, "neutral_single_llm")


def medagents_style(scenario: Scenario, backend, specs: dict) -> Proposal:
    """MedAgents 式：五方 LLM 各提 1 轮案（无历史协商、无 GNE/仲裁），等权平均。"""
    from ...mane.agents import build_agent
    agents = {aid: build_agent(spec, backend) for aid, spec in specs.items() if aid != "catfish"}
    props = []
    for aid, agent in agents.items():
        p = agent._llm_act(scenario, 1, [])
        props.append(p)
    collective = _collective_mean([p.principle_weights.as_array() for p in props])
    # 用伦理委员会提案的 rationale 作为最终文本（与 MANE final_proposal 口径一致）
    ec = next((p for p in props if p.agent == "ethics_committee"), props[0])
    return Proposal(
        agent="medagents_style", round=1,
        treatment_level=int(np.median([p.treatment_level for p in props])),
        principle_weights=collective, confidence=0.5,
        rationale=ec.rationale,
        syllogism=Syllogism(major_premise="MedAgents 式多角色协作", minor_premise="",
                            conclusion=f"集体治疗强度 {int(np.median([p.treatment_level for p in props]))}"),
    )


def violation_rate(v: np.ndarray) -> float:
    """跌破 L3 底线的维度占比（[B,N,A,J]≥[0.15,0.20,0.15,0.20]）。"""
    return float((np.asarray(v, dtype=float) < FLOORS).mean())
