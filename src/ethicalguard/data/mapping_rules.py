"""规则通道的纯函数（供 mapping.py 与 mapping_llm.py 共用，避免循环导入）。"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..types import Constraint, ConstraintKind, FourBoxState, Party

# ---- 关键词词典（规则通道） ----
_MEDICAL_SEVERITY = [
    "critical", "severe", "life-threatening", "futile", "cardiac arrest",
    "shock", "hemorrhage", "brain dead", "brain death", "respiratory failure",
    "multi-organ", "sepsis", "septic", "comatose", "unresponsive",
    "危急", "危重", "濒死", "脑死亡", "休克", "大出血", "多器官衰竭", "脓毒症", "昏迷",
]
_MEDICAL_RESUE = ["ecmo", "ventilat", "intubat", "icu", "cpr", "resuscitat", "透析", "机械通气", "插管"]
_PREF_DNR = ["dnr", "do not resuscitate", "no cpr", "dni", "do not intubate", "放弃抢救", "不复苏"]
_PREF_REFUSE = ["refused", "refuse", "declined", "against", "withdraw", "comfort only", "拒绝", "不同意", "撤除"]
_PREF_WISH = ["wants", "wished", "requested", "advance directive", "living will", "full code", "希望", "要求", "预立指示", "生前预嘱"]
_PREF_CAPACITY_LOSS = ["unconscious", "sedat", "intubat", "dementia", "delirium", "无意识", "镇静", "痴呆", "谵妄"]
_QOL_BURDEN = ["pain", "suffering", "disability", "paralysis", "poor quality of life", "疼痛", "痛苦", "残疾", "瘫痪", "生活质量差"]
_CONTEXT_RESOURCE = ["icu bed", "scarce", "shortage", "no bed", "full", "waiting list", "床位", "紧张", "稀缺", "排队"]
_CONTEXT_ECON = ["uninsured", "self-pay", "cost", "expensive", "bankrupt", "自费", "无力承担", "昂贵", "因病返贫",
                 "financial", "insurance", "medicare", "purchase price", "payment", "economic", "医保", "费用", "经济负担"]
_CONTEXT_FAMILY = ["family conflict", "disagreement", "family insist", "difficult family", "家属冲突", "家属坚持", "意见分歧"]
_CONTEXT_RELIGION = ["jehovah", "no blood", "religious", "宗教", "拒输血", "耶和华"]
_CONTEXT_LEGAL = ["legal", "court", "guardian", "法律", "法院", "监护人", "诉讼",
                  "confidential", "confidentiality", "hipaa", "disclosure", "consent", "隐私", "保密", "泄露"]
# 通用临床伦理扩展：患者意愿/家属参与/知情同意（缓解规则通道的重症偏置）
_PREF_WISH_EXT = ["preference", "preferences", "wishes", "choice", "choose", "decide", "decision",
                  "意愿", "偏好", "选择", "决定"]
_PREF_CONSENT = ["informed consent", "consent", "disclosure", "disclose", "inform", "advised",
                 "告知", "知情同意", "同意", "披露"]
_FAMILY_INVOLVE = ["family", "spouse", "wife", "husband", "parent", "son", "daughter", "children",
                   "家属", "妻子", "丈夫", "父母", "子女", "亲人"]
_QOL_EXT = ["quality of life", "functional", "independence", "independent living", "burden of treatment",
            "生活质量", "功能状态", "独立生活", "治疗负担"]


def _score(text: str, words: List[str]) -> float:
    t = text.lower()
    hit = sum(1 for w in words if w in t)
    return float(np.clip(hit / max(2.0, len(words) * 0.25), 0.0, 1.0))


def map_text_to_state(text: str) -> FourBoxState:
    """规则通道：文本 → 四盒状态（各子字段 0-1 语义分数）。"""
    medical = {
        "severity": _score(text, _MEDICAL_SEVERITY),
        "rescue_available": min(1.0, _score(text, _MEDICAL_RESUE) * 0.5 + 0.3),
        "acuity": _score(text, _MEDICAL_SEVERITY),
    }
    preference = {
        "dnr": 1.0 if _score(text, _PREF_DNR) > 0 else 0.0,
        "attitude_refuse": 1.0 if _score(text, _PREF_REFUSE) > 0 else 0.0,
        "clarity": min(1.0, _score(text, _PREF_WISH + _PREF_WISH_EXT + _PREF_DNR + _PREF_CONSENT) * 0.5 + 0.3),
        "capacity": 1.0 - min(1.0, _score(text, _PREF_CAPACITY_LOSS)),
        "info_completeness": 0.6,
    }
    qol = {
        "burden": _score(text, _QOL_BURDEN + _QOL_EXT),
        "net_effect": -0.4 if _score(text, _QOL_BURDEN + _QOL_EXT) > 0 else 0.2,
    }
    context = {
        "resource_pressure": _score(text, _CONTEXT_RESOURCE),
        "insurance_stress": _score(text, _CONTEXT_ECON),
        "family_conflict": _score(text, _CONTEXT_FAMILY),
        "family_involvement": _score(text, _FAMILY_INVOLVE),
        "religious_barrier": _score(text, _CONTEXT_RELIGION),
        "legal_constraint": _score(text, _CONTEXT_LEGAL),
    }
    return FourBoxState(medical=medical, preference=preference, qol=qol, context=context)


def default_parties(state: FourBoxState) -> Dict[str, Party]:
    """根据四盒状态派生默认五方（信息不对称：各方只能看到相关子集）。"""
    return {
        "patient": Party(
            id="patient", name="患者",
            observed={"preference": state.preference, "qol": state.qol},
            guarded_constraints=["autonomy_floor"],
        ),
        "family": Party(
            id="family", name="家属",
            observed={"context": state.context, "preference": {"attitude_refuse": state.preference.get("attitude_refuse", 0.0)}},
            guarded_constraints=["family_soft"],
        ),
        "physician": Party(
            id="physician", name="医师",
            observed={"medical": state.medical, "qol": state.qol},
            guarded_constraints=["nonmaleficence_floor"],
        ),
        "ethics_committee": Party(
            id="ethics_committee", name="伦理委员会",
            observed={"medical": state.medical, "preference": state.preference, "qol": state.qol, "context": state.context},
            guarded_constraints=["all"],
        ),
        "hospital_admin": Party(
            id="hospital_admin", name="医院管理",
            observed={"context": state.context},
            guarded_constraints=["resource_hard"],
        ),
    }


def default_constraints(state: FourBoxState) -> List[Constraint]:
    """默认三层约束（L3 原则底线 + L1 资源上限）。

    底线数值与 GNE 求解器 / 韧性放弃检测的 floors 统一（[B,N,A,J]=[0.15,0.20,0.15,0.20]，
    gne_solver.GNEProblem 默认 floors 同源）——提示词显示的底线 = 求解器强制的底线，
    消除"LLM 认知底线与执行底线不一致"（设计 vs 实现审计项）。
    """
    floors = {"nonmaleficence": 0.20, "autonomy": 0.15, "beneficence": 0.15, "justice": 0.20}
    constraints: List[Constraint] = [
        Constraint(name="nonmaleficence_floor", kind=ConstraintKind.FLOOR,
                   description="不伤害底线 N≥0.20", bound=floors["nonmaleficence"], direction="ge", value=0.7),
        Constraint(name="autonomy_floor", kind=ConstraintKind.FLOOR,
                   description="自主底线 A≥0.15", bound=floors["autonomy"], direction="ge", value=0.4),
        Constraint(name="beneficence_floor", kind=ConstraintKind.FLOOR,
                   description="行善底线 B≥0.15", bound=floors["beneficence"], direction="ge", value=0.4),
        Constraint(name="justice_floor", kind=ConstraintKind.FLOOR,
                   description="公正底线 J≥0.20", bound=floors["justice"], direction="ge", value=0.5),
    ]
    rp = state.context.get("resource_pressure", 0.0)
    if rp > 0.3:
        constraints.append(Constraint(
            name="resource_hard", kind=ConstraintKind.HARD,
            description="资源可得性硬约束（床位/ECMO）", bound=0.6, direction="le", value=rp,
        ))
    return constraints
