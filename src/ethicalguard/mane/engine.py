"""MANE 引擎顶层：冲突识别门控 + KAMAC 风格动态组队 + 协商 + 仲裁。"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..config import Config, default_agent_specs
from ..detection import make_detector
from ..llm import make_backend
from ..types import NegotiationResult, Scenario
from ..utils import setup_logging
from .agents import CatfishAgent, build_agent
from .arbitration import Arbitrator
from .negotiation import run_negotiation
from .orchestration import BayesianOrchestrator

logger = setup_logging()


class MANEEngine:
    """五方 + 鲶鱼的多智能体协商引擎（MANE 3.0）。

    流程：冲突识别（ERS 门控信号）→ 动态组队 → 协商 → 仲裁 → GNE 精炼。
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.backend = make_backend(self.config.llm)
        specs = self.config.agents or default_agent_specs()
        self.all_agents = {aid: build_agent(spec, self.backend) for aid, spec in specs.items() if aid != "catfish"}
        self.catfish = CatfishAgent(specs["catfish"], self.backend)
        self.orchestrator = BayesianOrchestrator(list(self.all_agents.keys()))
        committee = self.all_agents["ethics_committee"]
        self.arbitrator = Arbitrator(committee, self.config.mane.satisfaction_floor)
        # 冲突检测器：监督权重（配置）> LLM 检测器（api/local）> 规则基线（rule）
        self.detector = make_detector(self.backend, self.config.detection.checkpoint,
                                      self.config.detection.threshold)

    def assemble(self, scenario: Scenario) -> List:
        """KAMAC 风格动态组队：约束缺口驱动招募（规则版）。"""
        active = []
        # 医师 + 伦理委员会始终在场
        active.append(self.all_agents["physician"])
        active.append(self.all_agents["ethics_committee"])
        # 患者：意愿维度相关（清晰度/能力/DNR/拒绝）
        p = scenario.state.preference
        if p.get("clarity", 0.0) > 0.2 or p.get("capacity", 1.0) < 0.9 or p.get("dnr", 0.0) > 0:
            active.append(self.all_agents["patient"])
        # 家属：家庭冲突/宗教/经济软约束 + 代理决策情境召回（O2/A3）：
        # QoL 负担高/损益为负、患者容量或信息缺失 → 存在"代理决策"张力，家属必须入队
        c = scenario.state.context
        q = scenario.state.qol
        proxy_decision = (q.get("burden", 0.0) > 0.5 or q.get("net_effect", 0.0) < 0.0
                          or p.get("capacity", 1.0) < 0.9 or p.get("info_completeness", 0.5) < 0.5)
        if (c.get("family_conflict", 0.0) > 0.2 or c.get("religious_barrier", 0.0) > 0.2
                or c.get("insurance_stress", 0.0) > 0.3 or proxy_decision):
            active.append(self.all_agents["family"])
        # 医院管理：资源硬约束
        if c.get("resource_pressure", 0.0) > 0.2 or c.get("legal_constraint", 0.0) > 0.2:
            active.append(self.all_agents["hospital_admin"])
        return active

    def run(self, scenario: Scenario) -> NegotiationResult:
        # 会话级状态重置：一次协商 = 独立贝叶斯先验（修复 orchestrator 跨 run 累积
        # 污染——规则模式重跑/跨场景评估被前次会话串扰）
        self.orchestrator.reset()
        # [27 架构实验] 委员会提示词开关（issue_aware / exception_protocol，默认关）：
        # 每次 run 按 config 刷新，保证多配置复用同一 engine 时互不串扰。
        ec_agent = self.all_agents.get("ethics_committee")
        if ec_agent is not None:
            ec_agent.issue_aware = bool(getattr(self.config.mane, "issue_aware", False))
            ec_agent.exception_protocol = bool(getattr(self.config.mane, "exception_protocol", False))
        # ① SEMA-RAG 证据锁定（F1 可计算前提，设计 Layer1.3）：协商前冻结客观事实维度，
        #    提示词对锁定字段标注 [锚定]——防幻觉产生的不可行方案
        scenario.evidence_locked = ["medical", "context"]
        # ② 冲突识别（ERS 门控信号；gate=intervene 表示 ≥threshold 需升级关注，否则常规流程）
        report = self.detector.detect(scenario)
        gate = "intervene" if report.ers >= self.config.detection.threshold else "routine"
        active = self.assemble(scenario)
        logger.info("[%s] ERS=%.2f type=%s action=%s gate=%s | 组队: %s",
                    scenario.scenario_id, report.ers, report.conflict_type.value,
                    report.action, gate, [a.spec.id for a in active])
        result = run_negotiation(scenario, active, self.catfish, self.config.mane,
                                 self.orchestrator, self.arbitrator, self.config.gne)
        result.conflict_report = report
        return result
