# EG-04 — EthicalGuard：实验验证（runs 结果整理成证据链 + 补缺）

> 生成：nativepaperskill · 流水线阶段 7-8（实验验证——runs 已有真实结果，整理 + 补缺）
> 原则：结果真实（反幻觉）、证据金字塔分层、补缺清单明确

---

## 1. 已有运行结果盘点（runs/，真实）

| 结果文件 | 内容 | 证据用途 |
|---|---|---|
| **mane_results.jsonl** | 5 角色协商轨迹（每轮：治疗强度/四原则权重/置信度/三段论）+ GNE 精炼（final_vector/KKT 残差 7.2e-8/agent_satisfactions 0.9+）+ residual_conflict + 冲突报告（ERS/type/intensity/action/channel）+ 资源否决（Σρ vs cap） | **主结果**：协商收敛 + 均衡质量 + 冲突管理 |
| ablations_local.txt | 消融（w/o GNE / w/o Catfish / w/o 仲裁 / w/o 平衡正则） | RQ2 消融 |
| gamma_sweep.txt | γ（平衡正则权重）敏感性 | RQ3 敏感性 |
| resilience.jsonl | 压力测试（stress 注入 → 恢复） | RQ4 韧性 |
| detector.pt | 监督违规检测器 | M0 检测评估 |

**实证特点**（从结果反推）：
- **GNE 均衡真实收敛**：KKT 残差 7.2e-8（数值收敛到均衡）
- **多方满意度高**：agent_satisfactions 0.9+（均衡是"无人强烈不满"的稳定协议）
- **底线否决真实生效**：资源 L1 否决（Σρ=4.0 > cap=2.8 → 强制降级）——"底线不可协商"的实证
- **冲突报告真实**：ERS 评分 + conflict_type + action（review_24h）+ 检测通道（supervised）
- **坏均衡防线存在**：catfish 角色 + residual_conflict 报告

## 2. RQ 设计（从结果反推，C5 映射贡献）

| RQ | 问题 | 证据 |
|---|---|---|
| RQ1 | GNE 协商能否在伦理底线约束内收敛到多方共识？ | mane_results（KKT 收敛 + 满意度 + residual_conflict） |
| RQ2 | 各组件贡献？（GNE/鲶鱼/仲裁/平衡正则） | ablations_local |
| RQ3 | 平衡正则 γ 的敏感性？ | gamma_sweep |
| RQ4 | 压力下协商的韧性（施压→恢复）？ | resilience.jsonl |
| RQ5 | 伦理底线否决是否正确触发（资源/原则）？ | mane_results 中的 veto/降级事件 |

## 3. 证据金字塔（当前覆盖）

| 层 | 状态 |
|---|---|
| 主结果（RQ1 协商收敛） | ✅ 有（mane_results） |
| 消融（RQ2） | ✅ 有（ablations） |
| 敏感性（RQ3 γ） | ✅ 有（gamma_sweep） |
| 韧性（RQ4） | ✅ 有（resilience） |
| 定性/案例（协商轨迹展示） | ✅ 有（mane_results 全轨迹可作案例） |
| **与金标准对照**（专家共识 vs GNE 输出） | ⬜ **缺**（关键补缺） |
| **坏均衡检测**（故意构造合谋场景） | ⬜ **缺**（核心卖点验证） |

## 4. 补缺清单（优先级）

| # | 缺 | 为什么 | 怎么做 |
|---|---|---|---|
| P0 | **与专家共识对照** | 审稿人必问"均衡=伦理正确？"——需金标准 | 请医学伦理专家对 N 场景给"应然治疗强度+原则权重"，与 GNE 输出算一致性（Kappa/距离） |
| P0 | **坏均衡验证** | 核心卖点"防坏均衡"需实证 | 构造合谋场景（多智能体共谋规避底线）→ 验证鲶鱼+底线否决能破坏/拦截 |
| P1 | 规则 vs LLM 双模式对比 | 区分"规则基线 vs LLM 语义"贡献 | 同一场景双模式跑，报告差异 |
| P1 | GNE 收敛统计 | 单一场景 KKT 7.2e-8 不够 | 多场景报告收敛残差分布 + 失败率 |
| P2 | seed 固定可复现 | 实验可复现性 | scripts 加固定 seed |

## 5. 数据集贡献（资源型卖点）

| 数据集 | 内容 | 用途 |
|---|---|---|
| MedEthicEval | detecting_violation / equilibrium_dilemma / priority_dilemma / knowledge | 伦理违规检测 + 困境基准 |
| MedEthicsQA | MCQ + open-ended + taxonomy | 伦理问答 |
| PrinciplismQA | 原则主义知识/开放 QA | 四原则评估 |
| VITAL | distributional / steerable 价值取向 | 价值导向评测 |

**→ 数据集本身可成为贡献 ③（资源）**：医疗伦理多维度评测基准（当前伦理 LLM 评测稀缺）。

## 6. 本阶段评估约束

- **G3 证据**：主结果/消融/敏感性/韧性全有真实数据 ✅；金标准/坏均衡缺（P0 补）
- **G9 创新**：实证支撑三件套（理论/方法/系统+资源）
- **G10 批判**：KKT 收敛≠伦理正确 → P0 金标准对照是必经防御；规则 vs LLM 需分离
- **台账**：补 EthicalGuard research ledger（相关工作/数据集/差异）

## 决策记录

| 决策 | 结论 |
|---|---|
| 主结果 | ✅ mane_results（KKT 收敛+满意度+底线否决+冲突报告） |
| 补缺 P0 | ⬜ 金标准对照 + 坏均衡验证（投稿前必补） |
| 数据集 | ✅ 4 套（资源型贡献 ③） |
