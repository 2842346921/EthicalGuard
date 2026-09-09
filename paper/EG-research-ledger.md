# EG — 调研台账（研究 / baseline / 数据集详情）

> 生成：nativepaperskill · research ledger（每项含发表时间/论文/差异/可用性）
> 覆盖：三方向补充调研（多智能体伦理顶会 / GNE 先例 / 伦理底线形式化）+ 代码参考实现 + baseline + 数据集

---

## 一、多智能体伦理决策（2024-2026 顶会）

| 研究 | 发表时间 | 论文（venue/链接） | 方法一句话 | 与 EG 的差异 | 可用性 |
|---|---|---|---|---|---|
| **Many LLMs Are More Utilitarian Than One** | 2025 | **NeurIPS 2025**（papernotes/NeurIPS2025） | 多 LLM 群体伦理决策比单 LLM 更功利主义（聚合/投票） | 群体投票/聚合，**无博弈均衡、无共享底线约束、无韧性**；EG = GNE 均衡 + 底线否决 | 论文可读 |
| **Ethical issues in multi-agent AI systems for healthcare** | 2026 | Frontiers in Public Health（10.3389/fpubh.2026.1792627） | 多智能体医疗 AI 伦理问题综述（compound opacity 复合不透明） | 综述（问题提出）；**支撑 EG 的"可审计"需求**（compound opacity → 需可审计协商） | 综述可读 |
| Harmony（AI for Clinical Applications） | 2024 | ACM（10.1007/978-3-032-06004-4_10） | 临床 AI 多智能体协调 | 无博弈/无底线 | [待核实] |

## 二、GNE 与共享约束博弈（理论先例）

| 研究 | 发表时间 | 论文 | 方法一句话 | 与 EG 的关系 | 可用性 |
|---|---|---|---|---|---|
| **Facchinei & Kanzow: Generalized Nash Equilibrium Problems** | 2010 | 经典综述（Zbl 1211.91162） | GNE 问题理论奠基（共享耦合约束） | **理论支撑**：EG 的 GNE 形式化依据 | 理论文献 |
| **Games and teams with shared constraints** | 2017 | Royal Society Phil Trans A 375(2100) | 共享约束博弈与团队（GNE 结构分析） | **理论支撑**：共享约束博弈的稳定性分析 | 理论文献 |
| Mediation algorithms in MAS | 2024-2025 | emergentmind 主题 | 多智能体中介/调解算法 | 中介调解 vs EG 的均衡求解 | [待核实] |

## 三、伦理底线 / deontic 约束（agentic AI 先例）

| 研究 | 发表时间 | 论文 | 方法一句话 | 与 EG 的关系 | 可用性 |
|---|---|---|---|---|---|
| **Personalized Constitutionally-Aligned Agentic Superego** | 2025 | Matilda/Glos（eprints.glos.ac.uk/15288） | 宪法对齐的智能体"超我"（个人化伦理约束层） | **支撑"底线"概念**：deontic 约束层在 agentic AI 兴起；EG 是"多智能体协商中的底线" | 论文可读 |
| **A Governance Framework for Agentic AI** | 2025 | JICRCR（3549） | 多智能体架构的系统性风险治理框架 | 治理框架（高层）；EG 是机制层（博弈+底线） | 论文可读 |
| The Conscience Plug-in | 2025 | cognaptus 博客 | 按需注入伦理（插件式） | 单模型伦理插件 vs EG 多方协商 | 非学术（博客） |
| Hybrid Approaches for Moral Value Alignment: a Manifesto | 2024-2025 | Bohrium 论文 | 道德价值对齐混合方法宣言 | 对齐方法（训练层）；EG 是协商层 | 论文可读 |

## 四、代码参考实现（EG 架构已对齐，架构文档 §架构差异表）

| 参考 | 用途（EG 借用） | 差异（EG 独有） |
|---|---|---|
| Aegle（SOAP 状态） | FourBoxState + 观察掩码 | +GNE/底线否决/韧性 |
| EmoMAS（贝叶斯编排） | BayesianOrchestrator（Beta 权重） | +GNE |
| MedLA（逻辑树） | Syllogism 三段论 | +GNE |
| KAMAC（动态招募） | MANEEngine.assemble() | +GNE |
| Graph-of-States（状态机） | NegotiationFSM（S0-S6+R） | +GNE |
| ConflictScope（场景生成） | 规则映射通道 | +GNE |

## 五、baseline 候选

| baseline | 时间 | 论文 | 对比点 | 是否纳入 |
|---|---|---|---|---|
| 单 LLM 伦理推理 | — | — | 无多方/无底线（opacity） | ✅ 必纳 |
| Many LLMs 投票聚合 | 2025 | NeurIPS 2025 | 群体投票 vs 博弈均衡 | ✅ 必纳（最直接对照） |
| 宪法对齐智能体 | 2025 | Superego | 静态底线 vs 协商动态底线 | ✅ |
| 规则模式（EG 自身） | — | 本文 | 确定性基线 | ✅ |

## 六、数据集候选（EG 已自备）

| 数据集 | 时间 | 规模 | 用途 | 可用性 |
|---|---|---|---|---|
| MedEthicEval | 2025 | violation/equilibrium/priority/knowledge 4 组 | 违规检测+困境基准 | ✅ 自备 |
| MedEthicsQA | 2024 | MCQ + open-ended + taxonomy | 伦理问答 | ✅ 自备 |
| PrinciplismQA | 2024 | knowledge-mcqa + open-ended | 四原则评估 | ✅ 自备 |
| VITAL | 2025 | distributional + steerable | 价值分布/可引导 | ✅ 自备 |

## 七、我们的差异一句话（更新）

> 现有工作要么"群体投票无均衡"（NeurIPS 2025）、要么"静态底线无协商"（宪法对齐）、要么"理论 GNE 无 LLM 医疗实例"（Facchinei）——**EG 是首个把"GNE 共享约束=伦理底线"用于 LLM 多智能体医疗协商、且带可审计+韧性+防坏均衡的系统**。

## 八、待核实项（反幻觉）

- [ ] Harmony（ACM）具体方法与 venue
- [ ] Mediation algorithms 具体论文
- [ ] Superego 的正式 venue（Matilda/Glos eprints，可能 preprint）
