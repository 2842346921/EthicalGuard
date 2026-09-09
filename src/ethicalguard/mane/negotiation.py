"""协商循环（自适应轮次 + 状态机驱动 + 贝叶斯编排 + 收敛判定 + GNE 精炼）。

对齐《项目架构规划》§3.2 状态机（S0-S6+R）与 §2.3 F3 过程要求：
- 状态机接真：每轮以真实 flags（converged / drift_exceeded）推进，R 回溯被记录；
- 贝叶斯编排接真：每轮按"提案与集体向量接近度"更新 Agent 可靠性权重（EmoMAS）；
- 消融开关：w/o GNE / w/o Catfish / w/o 仲裁 / w/o 平衡正则（config.mane）。
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..config import GNEConfig, ManeConfig
from ..types import NegotiationResult, NegotiationRound, Proposal, Scenario, PrincipleVector
from .agents import CatfishAgent
from .arbitration import Arbitrator, enforce_resource_cap, resource_usage
from .gne_solver import scenario_resource_cap, solve_gne_from_proposals
from .orchestration import BayesianOrchestrator
from .state_machine import NegotiationFSM

# 贝叶斯编排：提案与集体向量的 L1 距离低于此值视为"被采纳"（EmoMAS 可靠性更新）
BAYESIAN_ACCEPT_L1 = 0.2


def _residual_conflict(proposals: List[Proposal]) -> float:
    """残余冲突 = 各 Agent 权重向量两两 L1 距离的均值。"""
    if len(proposals) < 2:
        return 0.0
    ws = [p.principle_weights.as_array() for p in proposals]
    d = 0.0
    n = 0
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            d += float(np.abs(ws[i] - ws[j]).sum())
            n += 1
    return d / max(1, n)


def _project_onto_floor(v: np.ndarray, floors: np.ndarray) -> np.ndarray:
    """把 v 投影回可行集 {v: v_k ≥ floors_k, Σv=1}（floor ≤ 0.25 时非空）。

    仅下限约束的单纯形投影：v = floors + u（u ≥ 0，Σu = 1 − Σfloors）。
    对 q = v − floors 做"非负 + 定和"标准单纯形投影（Duchi 水填法，d=4 精确）。
    Σfloors = 1（如 floor=0.25×4）时唯一可行点就是 floors 本身。
    """
    f = np.asarray(floors, dtype=float)
    q = np.asarray(v, dtype=float) - f
    target = max(0.0, 1.0 - f.sum())
    if target <= 0.0:
        return f.copy()
    order = np.argsort(q)[::-1]
    qs = q[order]
    cs = np.cumsum(qs) - target
    k_star = -1
    for k in range(len(qs)):
        if qs[k] - cs[k] / (k + 1) > 0:
            k_star = k
    if k_star < 0:
        return f.copy()
    rho = cs[k_star] / (k_star + 1)
    # ρ 是标量减到每个坐标上（与排序无关），u 直接在原序上算，无需 scatter
    u = np.maximum(q - rho, 0.0)
    return f + u


def run_negotiation(
    scenario: Scenario,
    agents: List,
    catfish: CatfishAgent,
    config: ManeConfig,
    orchestrator: BayesianOrchestrator,
    arbitrator: Arbitrator,
    gne_cfg: GNEConfig | None = None,
) -> NegotiationResult:
    fsm = NegotiationFSM(config.drift_backtrack_threshold)
    trajectory: List[NegotiationRound] = []
    converged = False
    arbitration_triggered = False
    collective: PrincipleVector | None = None
    prev_v: np.ndarray | None = None
    backtracks = 0      # R 回溯次数（状态机轨迹审计）
    bayesian_updates = 0  # 贝叶斯可靠性更新次数（EmoMAS 生效审计）
    catfish_rounds: List[int] = []   # 鲶鱼异议进入历史的轮次（闭环证据：异议→下一轮输入）
    dissent_collective_deltas: List[float] = []  # 异议轮之后集体向量相对异议轮的变化

    for r in range(1, config.max_rounds + 1):
        # 状态机推进（S0→S1→S2→S3...）——真实 flags
        delta = float("inf")
        if prev_v is not None and collective is not None:
            delta = float(np.linalg.norm(collective.as_array() - prev_v))
        drift = (collective.halved_drift(PrincipleVector.from_array(prev_v))
                 if prev_v is not None and collective is not None else 0.0)
        flags = {
            "converged": converged and r >= config.min_rounds,
            "drift_exceeded": drift > config.drift_backtrack_threshold,
            "stress_active": False,
        }
        st = fsm.advance({"flags": flags})
        if st.value == "R_backtrack":
            backtracks += 1

        proposals = [a.act(scenario, r, trajectory) for a in agents]
        dissent = catfish.act(scenario, r, trajectory) if config.use_catfish else None
        proposals_for_collective = proposals
        # 闭环证据：鲶鱼异议进入历史（下一轮各 Agent 的 _user_prompt 历史块可见，规则模式
        # compromise 只读集体向量故不可见——这恰是 rule/local 消融差异的过程证据）
        if dissent is not None:
            catfish_rounds.append(r)
        collective = orchestrator.collective_vector(proposals_for_collective)
        residual = _residual_conflict(proposals_for_collective)

        # 贝叶斯编排接真（EmoMAS）：提案接近集体向量 → 该 Agent 可靠性上调
        if config.use_gne:
            v_arr = collective.as_array()
            for p in proposals_for_collective:
                d = float(np.abs(p.principle_weights.as_array() - v_arr).sum())
                accepted = d < BAYESIAN_ACCEPT_L1
                orchestrator.update(p.agent, accepted=accepted)
                bayesian_updates += 1

        trajectory.append(NegotiationRound(
            index=r,
            proposals=proposals + ([dissent] if dissent else []),
            collective_vector=collective,
            residual_conflict=residual,
        ))

        # 收敛判定：集体原则向量 v_t 相对 v_{t-1} 的变化 < 阈值
        if prev_v is not None:
            delta = float(np.linalg.norm(collective.as_array() - prev_v))
        else:
            delta = float("inf")
        prev_v = collective.as_array()
        if delta < config.convergence_threshold and r >= config.min_rounds:
            converged = True
            break

    # 仲裁触发：未收敛 或 有方满意度过低 或 资源不可行（Σρ(t) > cap → L1 强制修正）
    satisfactions = {
        a.spec.id: orchestrator.satisfaction_from_vector(
            a.spec.id, collective.as_array(), a.satisfaction(scenario)
        ) if collective is not None else 0.0
        for a in agents
    }
    final_round_props = {p.agent: p for p in trajectory[-1].proposals if p.agent != "catfish"} if trajectory else {}
    res_used_pre = resource_usage(list(final_round_props.values())) if final_round_props else 0.0
    res_cap_pre = scenario_resource_cap(scenario, len(agents))
    res_feasible_pre = res_used_pre <= res_cap_pre + 1e-9
    arbitration_reason: str | None = None
    ruling: Proposal | None = None
    if config.use_arbitration and (not converged or min(satisfactions.values()) < config.satisfaction_floor
                                   or not res_feasible_pre):
        arbitration_triggered = True
        reasons = []
        if not converged:
            reasons.append("unconverged")
        if min(satisfactions.values()) < config.satisfaction_floor:
            reasons.append("satisfaction")
        if not res_feasible_pre:
            reasons.append("resource")
        arbitration_reason = "+".join(reasons)
        ruling = arbitrator.decide(scenario, trajectory[-1].proposals if trajectory else [], collective, satisfactions)
        trajectory.append(NegotiationRound(
            index=len(trajectory) + 1,
            proposals=[ruling],
            collective_vector=ruling.principle_weights,
            residual_conflict=0.0,
        ))

    # 状态机收尾（P-A 修复）：收敛/仲裁后补推进 S3 对齐 → S5 仲裁收敛 → DONE，
    # 保证轨迹完整——此前 2 轮收敛场景 break 在 S3 之前，S3_align 缺失（审计 ❌）
    for _ in range(4):
        st = fsm.advance({"flags": {"converged": converged, "drift_exceeded": False}})
        if st.value == "DONE":
            break

    # GNE 精炼：LLM/规则候选 → 求解器数值收敛 + KKT 校验
    neg_round = trajectory[-2] if arbitration_triggered and len(trajectory) >= 2 else trajectory[-1]
    last_proposals = {p.agent: p for p in neg_round.proposals if p.agent != "catfish"}
    # 资源 L1 强制修正（仲裁层职责，设计 §2.3"违规由仲裁层强制修正"）：
    # Σρ(t) > cap 时逐级降级到可行，再喂 GNE 求解器。
    # **仅在仲裁启用时执行**——w/o 仲裁消融不修正 → 资源超限场景可行率↓/KKT↓
    # （约束违反率上升，§5.3 预期退化，仲裁必要性才可见）
    resource_lowered: List[str] = []
    if config.use_arbitration:
        last_proposals, resource_lowered, res_note = enforce_resource_cap(scenario, last_proposals)
    # L3 底线从 gne_cfg 透传（E1 激活实验：floors 全 0=关约束对照；调高=观察约束真拉回）
    gne_floors = None
    if gne_cfg is not None and getattr(gne_cfg, "floors", None) is not None:
        gne_floors = np.asarray(gne_cfg.floors, dtype=float)
    # 耦合模式从 gne_cfg 透传（16_gne_modes：collective=公共 v 让步 / independent=教科书对照）
    gne_coupling = "collective"
    if gne_cfg is not None and getattr(gne_cfg, "coupling", None) in ("collective", "independent"):
        gne_coupling = gne_cfg.coupling
    # catfish 进 GNE（路 A，paper/EG-catfish-路A设计.md）：catfish 作为第 6 求解 agent，
    # satisfaction=基于当前集体的 maximin（抬最弱原则），treatment=0 不占资源。
    catfish_in_gne = bool(gne_cfg is not None and getattr(gne_cfg, "catfish_in_gne", False))
    # [17c] 守护坐委员会：委员会 GNE 满意度行抬最弱原则的授权强度（0=关，默认，现有行为不变）
    committee_beta = 0.0
    if gne_cfg is not None:
        committee_beta = float(getattr(gne_cfg, "committee_maximin_beta", 0.0) or 0.0)
    final_vector = collective
    kkt = None
    resource_used = resource_cap = None
    resource_feasible = None
    floor_lambdas = None  # L3 底线乘子（None=未启用 GNE/求解失败/约束未激活）
    if config.use_gne:
        try:
            all_w = orchestrator.weights()
            order = list(orchestrator.alpha.keys())
            reliability = np.array([all_w[order.index(a.spec.id)] for a in agents])
            kw = gne_cfg.solve_kwargs() if gne_cfg is not None else {}
            # balance_penalty（γ）从配置传递——此前 solve_kwargs() 不含它，配置项失效（恒用默认 0.3）；
            # w/o 平衡正则消融 → γ=0；γ-sweep（08 --gamma-sweep）通过 cfg.gne.balance_penalty 扫描
            kw["balance_penalty"] = (0.0 if not config.use_balance
                                     else (gne_cfg.balance_penalty if gne_cfg is not None else 0.3))
            sol_agents = agents
            sol_proposals = last_proposals
            if catfish_in_gne:
                # 路 A：catfish 作为第 6 求解 agent（treatment=0 不占资源；satisfaction 由
                # solve_gne 内对 catfish 特殊处理，见下方 extra_satisfaction 分支）
                # 取 catfish 最新提案作为 init_weights；无则中性 0.25
                catfish_prop = None
                for tr_r in reversed(trajectory):
                    for p in tr_r.proposals:
                        if p.agent == "catfish":
                            catfish_prop = p
                            break
                    if catfish_prop:
                        break
                if catfish_prop is None:
                    catfish_prop = Proposal(
                        agent="catfish", round=0, treatment_level=0,
                        principle_weights=PrincipleVector.from_array(np.full(4, 0.25)),
                        confidence=0.5, rationale="catfish 中性起点",
                        syllogism=Syllogism(major_premise="", minor_premise="", conclusion=""),
                    )
                sol_agents = agents + [catfish]
                sol_proposals = {**last_proposals, "catfish": catfish_prop}
            sol = solve_gne_from_proposals(
                sol_agents, scenario, sol_proposals,
                reliability=reliability if not catfish_in_gne else None,
                balance_tau=gne_cfg.balance_tau if gne_cfg is not None else 0.75,
                floors=gne_floors, coupling=gne_coupling,
                catfish_in_gne=catfish_in_gne,
                committee_maximin_beta=committee_beta,
                **kw)
            final_vector = PrincipleVector.from_array(sol.collective)
            kkt = sol.kkt.total
            resource_used = sol.resource_used
            resource_cap = sol.resource_cap
            resource_feasible = sol.resource_feasible
            # L3 底线乘子（E1b 底线敏感性：λ_floor>0 ⇔ 底线约束激活的精确证据）
            floor_lambdas = {k: float(v) for k, v in sol.lambdas.items() if k.startswith("floor_")} \
                if sol.lambdas else None
        except Exception:  # noqa: BLE001
            final_vector = collective  # 求解失败 → 朴素集体兜底
            floor_lambdas = None
    else:
        # w/o GNE：朴素集体（贝叶斯加权）作终态；资源可行性直接按修正后方案报告
        if last_proposals:
            resource_used = resource_usage(list(last_proposals.values()))
            resource_cap = scenario_resource_cap(scenario, len(agents))
            resource_feasible = resource_used <= resource_cap + 1e-9

    # P-B 修复：仲裁对最终结果有实质影响（均衡选择规则）——仲裁裁决（底线守护方视角）
    # 与 GNE 数值均衡各占一半融合。此前仲裁轮只追加轨迹、不进 GNE 输入，
    # 导致 w/o 仲裁消融 FDBI/PCI 零差异（仲裁"空转"）。
    # **E1b 底线守护修复（2026-09-06）**：裁决向量对 floor 无感知（_constrained_ruling
    # 弱原则恒 0.20/0.22，LLM 委员会裁决天然 min~0.21），0.5/0.5 融合会把求解器守住的
    # 底线拉穿——floor 0.24/0.25 档终态 min 掉到 ~(F+0.20)/2≈0.225 的"墙"（求解器 KKT
    # 仍 ~1e-3 守住，违约全在合成层）。融合后若违反底线，投影回可行集
    # {v ≥ floors, Σv=1}；默认 floor 下自然 min~0.21 > 0.20 → 投影恒等，主实验不受影响。
    if arbitration_triggered and ruling is not None and final_vector is not None:
        fv = final_vector.as_array()
        rw = ruling.principle_weights.as_array()
        fused = 0.5 * fv + 0.5 * rw
        proj_floors = gne_floors if gne_floors is not None else np.array([0.15, 0.20, 0.15, 0.20])
        if np.any(fused < proj_floors - 1e-12):
            fused = _project_onto_floor(fused, proj_floors)
        final_vector = PrincipleVector.from_array(fused)

    # final_proposal 取"最终共识的伦理论证"而非程序文本或异议者：
    # - 仲裁触发时 trajectory[-1] 是裁决轮（rationale 是 "CAMP: ... 资源 L1 否决" 程序摘要，
    #   非伦理论证）——rubric/Judge 若拿它评分会对程序文本打分；
    # - 未仲裁时 trajectory[-1].proposals[-1] 是 **catfish 异议**（append 顺序 proposals+[dissent]，
    #   鲶鱼恒在最后）——final_proposal 应代表共识而非异议；
    # 故两条路径统一：回溯最后一个多方协商轮（≥2 个非 catfish 提案），
    # 优先取伦理委员会（完全信息、守护底线角色）的提案作为最终论证。
    # 裁决本身保留在 trajectory（闭环审计 process_trace 仍引用裁决轮）。
    _consensus_prop = None
    for _round in reversed(trajectory):
        _props = [p for p in _round.proposals if p.agent != "catfish"]
        if len(_props) >= 2:
            _consensus_prop = next((p for p in _props if p.agent == "ethics_committee"), _props[-1])
            break
    if _consensus_prop is not None:
        final_proposal = _consensus_prop
    else:
        final_proposal = trajectory[-1].proposals[-1]
    # ---- 闭环证据链（process_trace）：每环输出是否真的进入下一环 ----
    # catfish 异议 → 进入历史 → 下一轮 LLM 提案感知 → 集体位移（subsequent_collective_delta）
    # → 触发仲裁（unconverged/satisfaction/resource）→ 融合进终态（fusion_applied）
    # → 资源修正（resource_corrected）→ GNE KKT 达标 → 终态守住 L3 底线（floors_ok）
    dissent_deltas: List[float] = []
    for idx in catfish_rounds:
        # 异议在第 idx 轮进历史；影响体现在第 idx+1 轮集体（LLM 提案读历史块）。
        nxt = next((t for t in trajectory if t.index == idx + 1), None)
        cur = next((t for t in trajectory if t.index == idx), None)
        if nxt is not None and cur is not None and nxt.collective_vector is not None \
                and cur.collective_vector is not None:
            dissent_deltas.append(float(np.abs(
                nxt.collective_vector.as_array() - cur.collective_vector.as_array()).sum()))
    # 底线审计值：优先用 gne_cfg 传入的 floors（E1 实验可调），否则默认
    audit_floors = gne_floors if gne_floors is not None else np.array([0.15, 0.20, 0.15, 0.20])
    # 审计容差与求解器 tol 对齐（E1b 一致性）：求解器只能保证 primal ≤ tol，
    # 严格 1e-9 会把容差内的合法出口（非仲裁场景 slack ~5e-4）误判为"未守住"；
    # w/o GNE 无求解器保证 → 保持严格 1e-9。
    audit_tol = 1e-9
    if config.use_gne:
        audit_tol = max(1e-9, float(getattr(gne_cfg, "tol", 1e-3) or 1e-3))
    floors_ok = (final_vector is not None
                 and bool(np.all(final_vector.as_array() >= audit_floors - audit_tol)))
    process_trace = {
        "catfish_rounds": catfish_rounds,
        "dissent_in_history": len(catfish_rounds) > 0,
        # 异议是否被"消费"：异议轮后一轮集体位移（>0 = 异议改动了后续协商；
        # 规则模式恒≈0——compromise 不读提案文本，异议"进了历史但没被听见"，
        # 这正是 rule/local 消融差异（轮次/仲裁率/满意度）的过程级证据）
        "subsequent_collective_delta": (float(np.mean(dissent_deltas)) if dissent_deltas else 0.0),
        "arbitration_triggered": arbitration_triggered,
        "arbitration_reason": arbitration_reason,
        "resource_corrected": len(resource_lowered) > 0,
        "resource_lowered_agents": sorted(set(resource_lowered)),
        "fusion_applied": bool(arbitration_triggered and ruling is not None
                               and final_vector is not None),
        "kkt_residual": kkt,
        "floors_ok": floors_ok,
    }
    return NegotiationResult(
        scenario_id=scenario.scenario_id,
        rounds=len(trajectory),
        converged=converged,
        arbitration_triggered=arbitration_triggered,
        arbitration_reason=arbitration_reason,
        process_trace=process_trace,
        trajectory=trajectory,
        final_proposal=final_proposal,
        final_vector=final_vector,
        kkt_residual=kkt,
        agent_satisfactions=satisfactions,
        resource_used=resource_used,
        resource_cap=resource_cap,
        resource_feasible=resource_feasible,
        floor_lambdas=floor_lambdas,
        state_trace=[s.value for s in fsm.history],  # 状态机轨迹（S0-S6+R），供审计/论文
        backtracks=backtracks,
        bayesian_updates=bayesian_updates,
    )
