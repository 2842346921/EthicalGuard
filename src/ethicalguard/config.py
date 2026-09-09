"""统一配置：configs/config.yaml（推荐）或 configs/default.json。

优先级（低 → 高）：
  1. 代码内 dataclass 默认值
  2. 配置文件（config.yaml / default.json）
  3. 环境变量（ETHICALGUARD_* / OPENAI_*）
  4. 命令行参数（脚本内覆盖）

结构：
  llm:      基座模型（mode/api/local + 地址/密钥/模型/温度/超时）
  datasets: 数据集（data_dir 统一根目录 + paths 逐数据集覆盖 + selected 默认选择）
  mane:     协商引擎（轮次/收敛/满意度/漂移/HALF 权重）
  gne:      GNE 求解器（eta/平衡/对偶/阻尼/迭代）
  resilience: 韧性（αβγ/阈值/σ/扰动数）
  run:      脚本运行默认（limit/seed/输出目录/缓存目录）
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:  # YAML 为可选依赖，缺失时回退 JSON
    import yaml as _yaml
except ImportError:  # pragma: no cover
    _yaml = None


def _load_config_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")) and _yaml is not None:
        return _yaml.safe_load(text) or {}
    if path.lower().endswith(".yaml"):
        raise RuntimeError("缺少 pyyaml 依赖：pip install pyyaml")
    return json.loads(text)


@dataclass
class AgentSpec:
    """五方/鲶鱼 Agent 的规格。"""

    id: str
    name: str
    principle: str  # 守护的原则（beneficence/nonmaleficence/autonomy/justice/balanced）
    constraint_levels: list = field(default_factory=list)  # L1/L2/L3
    full_info: bool = False
    persona: str = ""


@dataclass
class APIConfig:
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "gpt-4o"


@dataclass
class LocalModelConfig:
    base_url: str = "http://localhost:8000/v1"   # 本地模型服务地址（vLLM/Ollama）
    model: str = "Qwen/Qwen2.5-7B-Instruct"      # 服务上注册的模型名
    model_path: str = ""                          # 模型权重路径（供 00_serve_local.py 启动 vLLM 用）


@dataclass
class LLMConfig:
    mode: str = "rule"  # rule / api / local
    api: APIConfig = field(default_factory=APIConfig)
    local: LocalModelConfig = field(default_factory=LocalModelConfig)
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 120

    # ---- 便捷访问（兼容旧字段名） ----
    @property
    def base_url(self) -> Optional[str]:
        return self.api.base_url if self.mode == "api" else self.local.base_url

    @property
    def api_key(self) -> Optional[str]:
        return self.api.api_key

    @property
    def model(self) -> str:
        return self.api.model if self.mode == "api" else self.local.model


@dataclass
class DatasetConfig:
    data_dir: str = ""            # 统一根目录（为空则用脚本默认/环境变量）
    paths: Dict[str, str] = field(default_factory=dict)   # 逐数据集根目录覆盖
    selected: list = field(default_factory=list)          # 默认选择的数据集


@dataclass
class ManeConfig:
    max_rounds: int = 6
    min_rounds: int = 2
    convergence_threshold: float = 0.05
    satisfaction_floor: float = 0.3
    drift_backtrack_threshold: float = 0.3
    halved_weights: list = field(default_factory=lambda: [1.5, 3.0, 1.0, 2.0])
    # ---- 消融开关（§5.3）：w/o GNE / w/o Catfish / w/o 仲裁 / w/o 平衡正则 ----
    use_gne: bool = True          # False → 朴素集体（贝叶斯加权）作终态，KKT=None（w/o GNE 求解）
    use_catfish: bool = True      # False → 不注入鲶鱼异议（w/o Catfish，静默共识退化）
    use_arbitration: bool = True  # False → 不触发仲裁（w/o 伦理委员会）
    use_balance: bool = True      # False → GNE 平衡正则 γ·L_balance 关闭（w/o 平衡正则化）
    # ---- 架构实验开关（scripts/27_architecture_ablation；默认关 = 主实验行为不变）----
    issue_aware: bool = False           # 环3修复：委员会先声明"本题核心伦理议题"再权衡（LLM 模式）
    exception_protocol: bool = False    # 环1/2修复：安全例外（保密破例/自伤/强制报告）触发标准决策链议程


@dataclass
class GNEConfig:
    """GNE 求解器参数（传给 GNEProblem.solve）。"""

    eta: float = 0.6
    balance_penalty: float = 0.3
    balance_tau: float = 0.75
    # L3 原则底线 [B,N,A,J]（E1 激活实验：floors=全 0 关闭约束对照；调高观察激活）
    floors: Optional[list] = None
    # 耦合模式（16_gne_modes 标签错位对比）：collective=公共 v 让步（默认）/ independent=教科书对照
    coupling: str = "collective"
    # catfish 进 GNE（路 A）：catfish 作为第 6 求解 agent（maximin 守护最弱原则）
    catfish_in_gne: bool = False
    # [17c] 守护坐委员会：伦理委员会 GNE 满意度行 = 中性 0.5 + β·抬最弱原则
    # （β=0 关闭 = 现有行为完全不变；β∈{0.15,0.3} 试验）。五方不变，替代路 A 第 6 玩家
    # （路 A 收敛破损 + 玩家身份不干净：守护是委员会的授权偏好，不是独立利益相关者）。
    committee_maximin_beta: float = 0.0
    lr_dual: float = 0.2
    lr_dual_min: float = 0.005
    lr_decay: float = 0.5
    rho: float = 0.0
    damping: float = 0.5
    inner_iters: int = 20
    lam_max: float = 20.0
    max_iter: int = 2000
    tol: float = 1e-3

    def solve_kwargs(self) -> Dict[str, Any]:
        return {
            "eta": self.eta, "lr_dual": self.lr_dual, "lr_dual_min": self.lr_dual_min,
            "lr_decay": self.lr_decay, "rho": self.rho, "damping": self.damping,
            "inner_iters": self.inner_iters, "lam_max": self.lam_max,
            "max_iter": self.max_iter, "tol": self.tol,
        }


@dataclass
class DetectionConfig:
    """冲突识别检测器配置。"""

    checkpoint: str = ""      # 监督检测器权重路径（05_train_detector.py 产物）；空则用 LLM/规则
    threshold: float = 0.5    # ERS 门控阈值（≥此值视为高风险）


@dataclass
class ResilienceConfig:
    alpha: float = 0.35
    beta: float = 0.35
    gamma: float = 0.30
    verdict_threshold: float = 0.6
    sigma: float = 0.1
    n_perturbations: int = 7
    # L3 原则底线 [B,N,A,J]（放弃判定：Agent 提案中某原则权重 < floor 即视为"放弃该原则"，
    # 与 GNEProblem 的底线约束一致，见 gne_solver.py）
    floors: list = field(default_factory=lambda: [0.15, 0.20, 0.15, 0.20])
    # 压力强度网格：每类压力按这些强度档分别做反事实对照，产出压力-韧性曲线与
    # 临界强度 τ（[1.0] 即旧版单档行为；档数越多越能定位放弃临界点，调用量 ×档数）
    intensity_grid: list = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    # 基线/恢复重跑次数（P1-1：LLM 随机性下取均值、R_recover 报 mean±std；>1 时调用量 ×repeat，默认 1 不变）
    repeat: int = 1


@dataclass
class RunConfig:
    limit: int = 50
    seed: int = 42
    out_dir: str = "runs"
    cache_dir: str = "data_cache"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    mane: ManeConfig = field(default_factory=ManeConfig)
    gne: GNEConfig = field(default_factory=GNEConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    run: RunConfig = field(default_factory=RunConfig)
    agents: Dict[str, AgentSpec] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        cfg = cls()
        if path and os.path.exists(path):
            data = _load_config_file(path)
            _apply(cfg, data)
        # 环境变量覆盖
        _apply_env(cfg)
        return cfg


def _apply(cfg: Config, data: Dict[str, Any]) -> None:
    if "llm" in data:
        l = data["llm"]
        if isinstance(l.get("mode"), str):
            cfg.llm.mode = l["mode"]
        if isinstance(l.get("api"), dict):
            for k in ("base_url", "api_key", "model"):
                if k in l["api"]:
                    setattr(cfg.llm.api, k, l["api"][k])
        if isinstance(l.get("local"), dict):
            for k in ("base_url", "model", "model_path"):
                if k in l["local"]:
                    setattr(cfg.llm.local, k, l["local"][k])
        for k in ("temperature", "max_tokens", "timeout"):
            if k in l:
                setattr(cfg.llm, k, l[k])
    if "datasets" in data:
        d = data["datasets"]
        if isinstance(d.get("data_dir"), str):
            cfg.datasets.data_dir = d["data_dir"]
        if isinstance(d.get("paths"), dict):
            cfg.datasets.paths = {str(k): str(v) for k, v in d["paths"].items()}
        if isinstance(d.get("selected"), list):
            cfg.datasets.selected = [str(x) for x in d["selected"]]
    if "mane" in data:
        m = data["mane"]
        for k in ("max_rounds", "min_rounds", "convergence_threshold", "satisfaction_floor",
                  "drift_backtrack_threshold"):
            if k in m:
                setattr(cfg.mane, k, m[k])
        for k in ("use_gne", "use_catfish", "use_arbitration", "use_balance",
                  "issue_aware", "exception_protocol"):
            if k in m:
                setattr(cfg.mane, k, bool(m[k]))
        if "halved_weights" in m:
            cfg.mane.halved_weights = m["halved_weights"]
    if "gne" in data:
        g = data["gne"]
        for k in ("eta", "balance_penalty", "balance_tau", "lr_dual", "lr_dual_min", "lr_decay",
                  "rho", "damping", "inner_iters", "lam_max", "max_iter", "tol"):
            if k in g:
                setattr(cfg.gne, k, g[k])
        if "floors" in g and isinstance(g["floors"], list):
            cfg.gne.floors = [float(x) for x in g["floors"]]
        if "coupling" in g and g["coupling"] in ("collective", "independent"):
            cfg.gne.coupling = g["coupling"]
        if "catfish_in_gne" in g:
            cfg.gne.catfish_in_gne = bool(g["catfish_in_gne"])
        if "committee_maximin_beta" in g:
            cfg.gne.committee_maximin_beta = float(g["committee_maximin_beta"])
    if "detection" in data:
        d = data["detection"]
        if isinstance(d.get("checkpoint"), str):
            cfg.detection.checkpoint = d["checkpoint"]
        if isinstance(d.get("threshold"), (int, float)):
            cfg.detection.threshold = float(d["threshold"])
    if "resilience" in data:
        r = data["resilience"]
        for k in ("alpha", "beta", "gamma", "verdict_threshold", "sigma", "n_perturbations"):
            if k in r:
                setattr(cfg.resilience, k, r[k])
        if "floors" in r:
            cfg.resilience.floors = [float(x) for x in r["floors"]]
        if "intensity_grid" in r:
            cfg.resilience.intensity_grid = [float(x) for x in r["intensity_grid"]]
        if "repeat" in r:
            cfg.resilience.repeat = int(r["repeat"])
    if "run" in data:
        r = data["run"]
        for k in ("limit", "seed", "out_dir", "cache_dir"):
            if k in r:
                setattr(cfg.run, k, r[k])
    if "agents" in data:
        for aid, spec in data["agents"].items():
            cfg.agents[aid] = AgentSpec(
                id=spec.get("id", aid),
                name=spec.get("name", aid),
                principle=spec.get("principle", "balanced"),
                constraint_levels=spec.get("constraint_levels", []),
                full_info=spec.get("full_info", False),
                persona=spec.get("persona", ""),
            )


def _apply_env(cfg: Config) -> None:
    """环境变量覆盖（优先级高于配置文件）。"""
    if os.getenv("ETHICALGUARD_MODE"):
        cfg.llm.mode = os.getenv("ETHICALGUARD_MODE")
    if os.getenv("OPENAI_BASE_URL"):
        cfg.llm.api.base_url = os.getenv("OPENAI_BASE_URL")
    if os.getenv("OPENAI_API_KEY"):
        cfg.llm.api.api_key = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_MODEL"):
        cfg.llm.api.model = os.getenv("OPENAI_MODEL")
    if os.getenv("ETHICALGUARD_LOCAL_BASE_URL"):
        cfg.llm.local.base_url = os.getenv("ETHICALGUARD_LOCAL_BASE_URL")
    if os.getenv("ETHICALGUARD_LOCAL_MODEL"):
        cfg.llm.local.model = os.getenv("ETHICALGUARD_LOCAL_MODEL")
    if os.getenv("ETHICALGUARD_MODEL_PATH"):
        cfg.llm.local.model_path = os.getenv("ETHICALGUARD_MODEL_PATH")
    if os.getenv("ETHICALGUARD_DETECTOR_CHECKPOINT"):
        cfg.detection.checkpoint = os.getenv("ETHICALGUARD_DETECTOR_CHECKPOINT")
    if os.getenv("ETHICALGUARD_DATA_DIR"):
        cfg.datasets.data_dir = os.getenv("ETHICALGUARD_DATA_DIR")
    if os.getenv("ETHICALGUARD_DATASET_PATHS"):
        cfg.datasets.paths = {}
        for item in os.getenv("ETHICALGUARD_DATASET_PATHS").split(","):
            item = item.strip()
            if item and "=" in item:
                k, v = item.split("=", 1)
                cfg.datasets.paths[k.strip()] = v.strip()
    if os.getenv("ETHICALGUARD_LIMIT"):
        try:
            cfg.run.limit = int(os.getenv("ETHICALGUARD_LIMIT"))
        except ValueError:
            pass


def default_agent_specs() -> Dict[str, AgentSpec]:
    """五方 + 鲶鱼 Agent 的默认规格。"""
    return {
        "patient": AgentSpec("patient", "患者", "autonomy", ["L2", "L3"], False, "患者本人：表达自身痛苦、意愿与承受底线"),
        "family": AgentSpec("family", "家属", "autonomy", ["L2"], False, "家属：代理决策，携带照护能力与宗教文化约束"),
        "physician": AgentSpec("physician", "医师", "beneficence", ["L3"], False, "医师：基于循证给出疗效最优方案"),
        "ethics_committee": AgentSpec("ethics_committee", "伦理委员会", "justice", ["L1", "L3"], True, "伦理委员会：完全信息，守护底线，最终仲裁"),
        "hospital_admin": AgentSpec("hospital_admin", "医院管理", "justice", ["L1", "L2"], False, "医院管理：提供资源/政策约束"),
        "catfish": AgentSpec("catfish", "鲶鱼异议者", "balanced", [], True, "鲶鱼：注入结构化异议，打破沉默共识"),
    }
