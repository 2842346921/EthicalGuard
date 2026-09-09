"""Fair-GNE 风格的广义纳什均衡（GNE）求解器（完整原始-对偶 + 全 KKT 校验）。

问题建模（对应《项目架构规划》§2.2/§2.6）：
- 每个 Agent i 持有策略 x_i^t = (Treatment_i^t, α_i^t)：
   · Treatment_i^t ∈ {0,1,2,3}（治疗等级，携带资源用度 ρ(t)，见 treatment_resource）；
   · α_i ∈ Δ⁴（四原则权重）；集体原则向量 v = Σ w_i α_i / Σ w_i（w_i 为编排可靠性权重）；
- 个体目标 J_i(x_i) = α_i·s_i − soft_cost_i·α_i + γ·FDBI(v)（平衡收益，L_balance）
  − η Σ_k α_ik log α_ik（熵正则 → 内点解）；
- 共享（耦合）约束 g(x) ≤ 0，由全体 Agent 共同承担（GNE 的核心，区别于 NE）：
   1. 资源约束 g_res(x) = Σ_i ρ(t_i) − cap ≤ 0（各 Agent 资源用度总和，Fair-GNE 的
      "工作量总和"结构；ρ: 治疗等级 → 资源用度，如 0/1/2/3 级 → 0/0.3/0.6/1.0）；
   2. 原则底线 g_floor_k(v) = floor_k − v_k ≤ 0（作用于集体向量 v）；
- 拉格朗日 L = Σ J_i − λ_res g_res − Σ λ_floor_k g_floor_k；
- KKT 条件：平稳性 / 原始可行性 / 对偶可行性(λ≥0) / 互补松弛(λ⊙g=0)；
- 双时间尺度原始-对偶：
   · 内层：固定 λ，阻尼最佳响应（软max 闭式解）逼近原始不动点（平稳性≈0）；
   · 外层：对偶投影次梯度上升（0 ≤ λ ≤ λ_max，步长衰减，Fair-GNE 自适应对偶）；
   · 平衡目标以固定权重 γ 进目标而非对偶（非凸约束 FDBI 的稳定处理）；
   · 资源约束作用于协商产出（治疗等级），求解器做"可行性认证 + 影子价格报告"，
     违规由仲裁层（L1 否决）在协商层面强制修正——两层分工，见 arbitration.py。

LLM-候选 + 求解器精炼的混合架构：Agent 先给初值（LLM/规则，含治疗等级），
求解器在权重空间精炼 α 并做 KKT 校验；权重空间均衡与方案空间（治疗等级）通过
资源映射 ρ(t) 耦合。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..types import PrincipleVector, Scenario

# 治疗等级 → 资源用度 的临床映射（0 舒适护理 / 1 有限干预 / 2 全面医疗 / 3 积极治疗）
DEFAULT_TREATMENT_RESOURCE = np.array([0.0, 0.3, 0.6, 1.0])


def fdbi(v: np.ndarray) -> float:
    """四维度平衡指数 FDBI = 1 − σ/μ（值→1 平衡，→0 冲突）。"""
    v = np.asarray(v, dtype=float)
    mu = v.mean()
    if mu <= 0:
        return 0.0
    return float(1.0 - v.std() / mu)


def _fdbi_grad(v: np.ndarray) -> np.ndarray:
    """FDBI 对 v 的梯度 ∂FDBI/∂v_k —— 仅供报告/测试参考。

    注意：FDBI = 1 − 2·‖v − 0.25·1‖（v 在单纯形上时 σ = ‖v−0.25·1‖/2），其梯度在
    v→均匀时 0/0 爆炸，**不能**直接用于原始-对偶（会引起振荡）。平衡力请用
    ``_balance_force``（L_balance 方差惩罚的解析梯度，线性有界）。
    """
    v = np.asarray(v, dtype=float)
    mu = v.mean()
    s = v.std()
    if mu <= 1e-12 or s <= 1e-12:
        return np.zeros_like(v)
    dmu = 1.0 / v.size
    dsig = (1.0 / s) * dmu * (v - mu)          # ∂σ/∂v_k
    # ∂(σ/μ)/∂v_k = (dsig·μ − σ·dmu)/μ²；FDBI = 1 − σ/μ → 取负
    return -(dsig * mu - s * dmu) / (mu * mu)


def _balance_force(v: np.ndarray) -> np.ndarray:
    """平衡力：L_balance 方差惩罚 −γ·Var(v) 的解析梯度。

    在单纯形上 Var(v) = ‖v − 0.25·1‖²/4，故 ∂(−‖v−0.25‖²)/∂v_k = −2(v_k − 0.25)。
    线性有界，保证原始-对偶在平衡点附近稳定收敛（对应设计文档 EthicalLoss 的 L_balance）。
    """
    v = np.asarray(v, dtype=float)
    return -2.0 * (v - 0.25)


def _softmax_rowwise(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class KKTResidual:
    stationarity: float
    primal_feasibility: float
    dual_feasibility: float
    complementarity: float
    total: float


@dataclass
class GNESolution:
    weights: np.ndarray                 # (n,4) 各 Agent 精炼后的原则权重
    collective: np.ndarray              # (4,) 集体原则向量 v*
    lambdas: Dict[str, float]           # 收敛后的拉格朗日乘子（resource + floors）
    kkt: KKTResidual
    converged: bool
    iterations: int
    fdbi_achieved: float                # 求解后实际达成的平衡度
    balance_met: bool                   # FDBI ≥ balance_tau？
    resource_usage: np.ndarray = None   # (n,) 各 Agent 资源用度 ρ(t_i)
    resource_used: float = 0.0          # Σρ(t_i)
    resource_cap: float = 0.0           # 资源上限
    resource_feasible: bool = True      # Σρ(t_i) ≤ cap？


@dataclass
class GNEProblem:
    satisfaction: np.ndarray                              # (n,4)
    init_weights: Optional[np.ndarray] = None             # (n,4)
    reliability: Optional[np.ndarray] = None              # (n,)
    soft_cost: Optional[np.ndarray] = None                # (n,4) 线性软成本系数
    treatments: Optional[np.ndarray] = None               # (n,) 各 Agent 治疗等级 0-3（来自提案）
    treatment_resource: Optional[np.ndarray] = None       # (4,) 等级→资源用度
    resource_cap: float = 3.0                             # 资源上限（Σρ(t_i) ≤ cap）
    balance_tau: float = 0.75                             # FDBI 目标（作为报告指标，非对偶约束）
    balance_penalty: float = 0.6                          # γ：平衡进入目标的固定权重（L_balance）
    floors: Optional[np.ndarray] = None                   # (4,) 原则底线
    # 耦合模式（标签错位对比实验）：
    # "collective"  = 当前实现：公共 v 的平衡正则（γ·FDBI(v)）进每个 agent 目标 → agent 向共识让步
    # "independent" = 教科书 GNE 对照：agent 纯追自己 s_i（平衡项不依赖 v 的让步），只受 floor 约束
    # 对比回答："公共 v 让步机制"贡献多少（共识质量 vs 各自立场）——见 scripts/16_gne_modes.py
    coupling: str = "collective"

    _W: np.ndarray = field(init=False, default=None)      # 归一化可靠性

    def __post_init__(self) -> None:
        self.satisfaction = np.asarray(self.satisfaction, dtype=float)
        self.n, self.d = self.satisfaction.shape
        assert self.d == 4, "四原则，d 必须为 4"
        if self.coupling not in ("collective", "independent"):
            raise ValueError(f"coupling 必须为 collective/independent，收到 {self.coupling}")
        if self.init_weights is None:
            self.init_weights = np.full((self.n, 4), 0.25)
        self.init_weights = np.asarray(self.init_weights, dtype=float)
        self.init_weights = self.init_weights / self.init_weights.sum(axis=1, keepdims=True)
        if self.reliability is None:
            self.reliability = np.ones(self.n)
        self.reliability = np.asarray(self.reliability, dtype=float)
        self._W = self.reliability / (self.reliability.sum() + 1e-12)
        if self.soft_cost is None:
            self.soft_cost = np.zeros((self.n, 4))
        self.soft_cost = np.asarray(self.soft_cost, dtype=float)
        if self.treatments is None:
            self.treatments = np.zeros(self.n, dtype=int)
        self.treatments = np.asarray(self.treatments, dtype=int).clip(0, 3)
        if self.treatment_resource is None:
            self.treatment_resource = DEFAULT_TREATMENT_RESOURCE
        self.treatment_resource = np.asarray(self.treatment_resource, dtype=float)
        assert len(self.treatment_resource) == 4
        self.resource_usage = np.array([self.treatment_resource[t] for t in self.treatments])
        if self.floors is None:
            # 底线取保守值（"底线"语义：远低于平衡点 0.25，避免与 Agent 真实偏好冲突）
            self.floors = np.array([0.15, 0.20, 0.15, 0.20])
        self.floors = np.asarray(self.floors, dtype=float)

    # ---- 状态与约束 ----
    def _collective(self, W: np.ndarray) -> np.ndarray:
        v = (self._W[:, None] * W).sum(axis=0)
        s = v.sum()
        return v / s if s > 1e-12 else np.full(4, 0.25)

    def _constraint_values(self, v: np.ndarray) -> Dict[str, float]:
        """共享约束值：资源 = Σρ(t_i) − cap（治疗等级携带资源用度）；底线 = floor − v。"""
        return {
            "resource": float(self.resource_usage.sum() - self.resource_cap),
            **{f"floor_{k}": float(self.floors[k] - v[k]) for k in range(4)},
        }

    def _floor_coupling(self, lam: Dict[str, float]) -> np.ndarray:
        """底线约束对 v 的加权梯度 dg_floor = −λ_floor（资源约束作用于治疗方案，不作用于 v）。"""
        return -np.array([lam[f"floor_{k}"] for k in range(4)])

    def _best_response(self, v: np.ndarray, lam: Dict[str, float], eta: float) -> np.ndarray:
        """精确最佳响应（软max 闭式解）：

        α_i ∝ exp((s_i − soft_i + γ·balance_force(v)·(w_i/W) − dg_floor·(w_i/W))/η)
        —— 平衡收益项 +γ·(−‖v−0.25‖²) 进入目标（L_balance 方差惩罚，线性有界梯度）；
           底线约束 −λ_floor·g_floor 进入耦合；资源约束不直接作用于 v（由治疗方案承担）。
        coupling="independent"（教科书 GNE 对照）：去掉 balance 对公共 v 的让步
        （agent 纯追自己 s_i，只受 floor 约束）——测"公共 v 让步机制"的边际价值。
        """
        dg = self._floor_coupling(lam)
        if self.coupling == "independent":
            balance_term = np.zeros_like(v)
        else:
            balance_term = self.balance_penalty * _balance_force(v)
        arg = (self.satisfaction - self.soft_cost
               + self._W[:, None] * balance_term[None, :]
               - self._W[:, None] * dg[None, :]) / eta
        return _softmax_rowwise(arg)

    # ---- 主求解 ----
    def solve(self, *, eta: float = 0.6, lr_dual: float = 0.2, lr_dual_min: float = 0.005,
              lr_decay: float = 0.5, adaptive_dual: bool = True,
              rho: float = 0.0, damping: float = 0.5, inner_iters: int = 20,
              lam_max: float = 20.0, max_iter: int = 2000, tol: float = 1e-3) -> GNESolution:
        """双时间尺度原始-对偶：
        - 内层：固定 λ，阻尼最佳响应逼近原始不动点（平稳性 ≈ 0）；
        - 外层：对偶更新（0 ≤ λ ≤ λ_max）：
           · rho>0：method of multipliers（λ += ρ·g）；
           · rho=0：自适应步长投影次梯度上升。
        资源约束值在求解期间恒定（治疗方案固定），其可行性认证与影子价格在 KKT 中报告。

        **E1b 高 floor 收敛修复（2026-09-06）**：原实现步长按 lr_decay 指数衰减，
        当约束**真激活**（floor 调高、λ 需大幅爬升把解拉回可行域）时步长已衰减到
        爬不动（实测 λ 停滞在 0.004、违反量恒 0.0048、KKT 卡 2.7e-2）。改为**违反感知步长**：
        约束持续违反（g>0）时保持步长让 λ 爬升，满足后才按 lr_decay 衰减；
        且 rho>0 时直接 method of multipliers（λ += ρ·g）加速约束满足。
        """
        W = self.init_weights.copy()
        lam = {"resource": 0.0, **{f"floor_{k}": 0.0 for k in range(4)}}
        # 违反感知：记录哪些约束最近在违反（用于保持步长）
        _viol_recent = {key: False for key in lam}

        for it in range(max_iter):
            # 内层：原始不动点
            for _ in range(inner_iters):
                v = self._collective(W)
                br = self._best_response(v, lam, eta)
                W = (1.0 - damping) * W + damping * br

            # 外层：对偶更新（g 可正可负——约束满足时乘子应下降）
            v = self._collective(W)
            g = self._constraint_values(v)
            for key in lam:
                if rho > 0:
                    # method of multipliers：直接按违反量爬升（不依赖衰减步长）
                    lam[key] = min(lam_max, max(0.0, lam[key] + rho * g[key]))
                    continue
                gk = g[key]
                if adaptive_dual:
                    if gk > 1e-6:
                        # 约束违反中 → 保持较大步长（允许 λ 爬升拉回可行域）
                        step = max(lr_dual_min, lr_dual / 2.0)
                        _viol_recent[key] = True
                    elif _viol_recent[key]:
                        # 刚满足 → 短暂保持后衰减（避免刚拉回就掉）
                        step = max(lr_dual_min, lr_dual / (1.0 + it * lr_decay)) / 2.0
                        if gk <= 0.0:
                            _viol_recent[key] = False
                    else:
                        step = max(lr_dual_min, lr_dual / (1.0 + it * lr_decay)) / 2.0
                else:
                    step = max(lr_dual_min, lr_dual / (1.0 + it * lr_decay))
                lam[key] = min(lam_max, max(0.0, lam[key] + step * gk))

            kkt = self._kkt(W, v, g, lam, eta)
            if kkt.total < tol:
                return self._finalize(W, v, lam, kkt, it + 1)

        v = self._collective(W)
        g = self._constraint_values(v)
        kkt = self._kkt(W, v, g, lam, eta)
        return self._finalize(W, v, lam, kkt, max_iter)

    def _finalize(self, W, v, lam, kkt, iters) -> GNESolution:
        fdbi_val = fdbi(v)
        used = float(self.resource_usage.sum())
        return GNESolution(
            weights=W, collective=v, lambdas=lam, kkt=kkt,
            converged=(kkt.total < 1e-2), iterations=iters,
            fdbi_achieved=fdbi_val, balance_met=fdbi_val >= self.balance_tau,
            resource_usage=self.resource_usage.copy(),
            resource_used=used, resource_cap=self.resource_cap,
            resource_feasible=used <= self.resource_cap + 1e-9,
        )

    def _kkt(self, W, v, g, lam, eta) -> KKTResidual:
        # 平稳性：拉格朗日梯度（含平衡收益项）在单纯形切平面上的投影残差
        dg = self._floor_coupling(lam)
        if self.coupling == "independent":
            balance_term = np.zeros_like(v)
        else:
            balance_term = self.balance_penalty * _balance_force(v)
        grad = (self.satisfaction - self.soft_cost
                + self._W[:, None] * balance_term[None, :]
                - self._W[:, None] * dg[None, :]
                - eta * (1.0 + np.log(W + 1e-12)))
        grad_proj = grad - grad.mean(axis=1, keepdims=True)
        stationarity = float(np.mean(np.abs(grad_proj)))
        primal = float(max(0.0, g["resource"]) + sum(max(0.0, g[f"floor_{k}"]) for k in range(4)))
        dual = float(sum(max(0.0, -lam[k]) for k in lam))
        comp = float(abs(lam["resource"] * g["resource"])
                     + sum(abs(lam[f"floor_{k}"] * g[f"floor_{k}"]) for k in range(4)))
        return KKTResidual(stationarity=stationarity, primal_feasibility=primal,
                           dual_feasibility=dual, complementarity=comp,
                           total=stationarity + primal + dual + comp)


def agent_soft_cost(agent_id: str, scenario: Scenario) -> np.ndarray:
    """从场景上下文推导各 Agent 的线性软成本系数（L2 软约束进入个体目标）。"""
    c = scenario.state.context
    econ = c.get("insurance_stress", 0.0)
    rp = c.get("resource_pressure", 0.0)
    conflict = c.get("family_conflict", 0.0)
    if agent_id == "patient":
        return np.array([econ, 0.1, 0.0, 0.1])          # 行善(积极治疗)代价随经济压力上升
    if agent_id == "family":
        return np.array([0.1 + 0.4 * conflict, 0.1, 0.0, 0.2 + 0.3 * econ])
    if agent_id == "hospital_admin":
        return np.array([0.0, 0.0, 0.0, 0.0])           # 资源代价由共享约束承担
    if agent_id == "physician":
        return np.array([0.0, 0.0, 0.1, 0.0])
    if agent_id == "ethics_committee":
        return np.zeros(4)
    return np.zeros(4)


def scenario_resource_cap(scenario: Scenario, n_agents: int, base_per_capita: float = 0.7) -> float:
    """从场景资源压力推导资源上限：压力越大，人均预算越低。

    cap = base_per_capita · n · (1 − 0.4 · resource_pressure)
    （标定参数：正常场景占用 ~85% 而非顶格；资源紧张时收紧触发仲裁降级；论文中需做敏感性分析）
    """
    pressure = scenario.state.context.get("resource_pressure", 0.0)
    return float(base_per_capita * n_agents * (1.0 - 0.4 * min(1.0, pressure)))


def solve_gne_from_proposals(
    agents: List,
    scenario: Scenario,
    proposal_map: dict,
    reliability: Optional[np.ndarray] = None,
    balance_tau: float = 0.75,
    treatment_resource: Optional[np.ndarray] = None,
    floors: Optional[np.ndarray] = None,
    coupling: str = "collective",
    catfish_in_gne: bool = False,
    committee_maximin_beta: float = 0.0,
    **solver_kwargs,
) -> GNESolution:
    """便捷入口：从 Agent 对象 + 提案构造完整 GNE 问题并求解。

    治疗等级从提案中提取（treatments），资源上限从场景资源压力推导（scenario_resource_cap）。
    solver_kwargs 直接透传给 GNEProblem.solve（eta/balance_penalty/damping/max_iter 等，见 GNEConfig）。
    floors 透传给 GNEProblem（L3 底线；None=默认 [0.15,0.20,0.15,0.20]）。
    **E1 底线激活实验**：floors=全 0 或 None 时约束不激活（对照）；调高到 0.25+ 观察约束真正拉回。

    catfish_in_gne（路 A，paper/EG-catfish-路A设计.md）：catfish 作为第 6 求解 agent——
    satisfaction 用 maximin 方向（基于五方 init 集体的最弱原则，静态近似），treatment=0 不占
    资源（ρ=0），reliability 与五方并列。使"打破共谋"从文本偶然影响升格为博弈占优。
    [17c 架构修正] 路 A 保留仅供对照——六方 maximin 行在部分场景破坏五方收敛（KKT 0.36-0.8）
    且"非利益相关者当玩家"建模不干净。守护的推荐实现 = committee_maximin_beta>0：
    伦理委员会（本就在五方内、docstring 职责即"守护公正+底线"）的 GNE 满意度行改为
    0.5·1 + β·抬最弱原则（弱原则=初始集体 argmin，状态驱动、确定性、可 KKT 验证），
    使均衡层守护成为委员会的制度性偏好，而非凭空多一个玩家。
    """
    ids = [a.spec.id for a in agents]
    satisfaction = np.array([a.satisfaction(scenario) for a in agents])
    init_weights = np.array([proposal_map[i].principle_weights.as_array() for i in ids])
    soft_cost = np.array([agent_soft_cost(i, scenario) for i in ids])
    treatments = np.array([proposal_map[i].treatment_level for i in ids], dtype=int)
    if committee_maximin_beta > 0.0 and "ethics_committee" in ids:
        # 守护坐委员会：委员会满意度 = 中性 0.5 + β 抬"初始集体最弱原则"
        # （与 catfish satisfaction_maximin 同构但幅度温和；β∈(0,0.45)，避免行外维度过弱）
        ci = ids.index("ethics_committee")
        coll_init = init_weights.mean(axis=0)
        weakest = int(np.argmin(coll_init))
        row = np.full(4, 0.5)
        row[weakest] = 0.5 + float(np.clip(committee_maximin_beta, 0.0, 0.45))
        satisfaction[ci] = row
    if catfish_in_gne:
        # catfish 行：maximin satisfaction（基于五方 init 集体的最弱原则）
        catfish = next((a for a in agents if getattr(a, "spec", None) and a.spec.id == "catfish"), None)
        coll_init = init_weights.mean(axis=0) if len(init_weights) else np.full(4, 0.25)
        if catfish is not None and hasattr(catfish, "satisfaction_maximin"):
            cat_sat = catfish.satisfaction_maximin(coll_init)
        else:
            weakest = int(np.argmin(coll_init))
            cat_sat = np.full(4, 0.3)
            cat_sat[weakest] = 0.6  # 与规则 argmin 抬 0.6 一致
        satisfaction = np.vstack([satisfaction, cat_sat[None, :]])
        init_weights = np.vstack([init_weights, proposal_map["catfish"].principle_weights.as_array()[None, :]])
        soft_cost = np.vstack([soft_cost, np.zeros(4)[None, :]])
        treatments = np.append(treatments, 0)  # catfish 舒适护理，ρ=0 不占资源
    cap = scenario_resource_cap(scenario, len(ids))
    balance_penalty = solver_kwargs.pop("balance_penalty", 0.3)
    return GNEProblem(
        satisfaction=satisfaction,
        init_weights=init_weights,
        reliability=reliability,
        soft_cost=soft_cost,
        treatments=treatments,
        treatment_resource=treatment_resource,
        resource_cap=cap,
        balance_tau=balance_tau,
        balance_penalty=balance_penalty,
        floors=floors,
        coupling=coupling,
    ).solve(**solver_kwargs)
