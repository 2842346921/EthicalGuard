# EthicalGuard 架构说明

与 `项目架构规划.md`（v2.2）对应。本目录是**可运行的 Python 骨架**，实现"工作①适配层 → 工作②MANE 协商 → 工作③韧性评估"闭环。

## 模块 ↔ 理论要求映射（规划 §6.2）

| 理论要求 | 代码模块 |
|:---|:---|
| F1 可行性 | `data/mapping.py`（文本→四盒）+ `mane/arbitration.py`（L1 否决）+ `mane/agents/roles.py`(医院管理) |
| F2 均衡性 | `mane/gne_solver.py`（正则化 GNE 原始-对偶 + KKT） |
| F3 可审计 | `mane/agents/base.py`（Syllogism 三段论）+ `mane/state_machine.py` |
| F3 信息不对称 | `mane/agents/base.py`（observe 观察掩码）+ `mane/orchestration.py`（贝叶斯权重） |
| F3 过程 | `mane/negotiation.py`（自适应轮次）+ `mane/agents/roles.py`(Catfish) + `mane/engine.py`(动态组队) |
| F4 韧性 | `resilience/`（压力引擎 + 三层次度量 + 恢复验证） |
| 评估 | `eval/`（FDBI/PCI/分布距离/评审器/基线） |

## 已对齐的外部参考实现（`论文\相关模型`）

- **Aegle** `shared/data_models.py` → SOAP 结构化状态（我们的 `FourBoxState` + 观察掩码）
- **Graph-of-States** `belief/fsm.py` → 状态机转移（我们的 `NegotiationFSM`）
- **EmoMAS** `models/bayesian_multiagent.py` → 贝叶斯编排（我们的 `BayesianOrchestrator`）
- **MedLA** 逻辑树 → 我们的 `Syllogism` 三段论
- **ADEPT** `Personas/*.yaml` → 五方角色人格（`config.py` 的 AgentSpec.persona）
- **KAMAC** 动态招募 → 我们的 `MANEEngine.assemble()`
- **ConflictScope** 场景生成 → 我们的规则映射通道（LLM 通道预留）

## 当前为骨架：规则模式可端到端跑通，LLM 模式接口已就绪

- 规则模式：确定性、可复现、零依赖 API（`--mode rule`）
- API 模式：`OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`（`--mode api`）
- 本地模式：vLLM 占位（`llm/local.py`，自行起服务后改用 api 模式即可）
