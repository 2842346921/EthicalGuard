# EG-03 — EthicalGuard：代码架构（code 模式审计 + 项目架构文档化）

> 生成：nativepaperskill · 流水线阶段 5-6（代码架构编写/审计——代码已存在，文档化 + 六维评估）
> 依据：code 模式协议（code-analysis.md）+ 实际代码（src/mane、scripts、tests、runs）

---

## 1. 代码盘点

```
EthicalGuard/
├── src/ethicalguard/
│   ├── config.py / types.py          # 配置（AgentSpec/约束/提案类型）+ 数据类型
│   ├── detection/                    # M0 冲突识别（rule/rules_features/supervised + detector.pt）
│   ├── mane/                         # M1-M4 核心
│   │   ├── engine.py                 #   MANEEngine 顶层（冲突门控+组队+协商+仲裁+GNE）
│   │   ├── agents/{base,roles}.py    #   五方+鲶鱼 Agent（规则确定性 / LLM 语义）
│   │   ├── negotiation.py            #   协商循环（状态机+贝叶斯编排+收敛判定+GNE 精炼）
│   │   ├── state_machine.py          #   S0-S6+R 状态机
│   │   ├── orchestration.py          #   贝叶斯可靠性权重（Beta(a,b)）
│   │   ├── arbitration.py            #   L1 硬否决 + 资源强制修正 + CAMP 仲裁
│   │   ├── gne_solver.py             #   GNE 求解器（原始-对偶 + KKT 校验）★理论核心
│   ├── resilience/                   # M5 韧性（stress_engine/recovery/metrics）
│   ├── eval/                         # 评估（FDBI/PCI/分布距离/judges/baselines）
│   └── llm/                          # 后端（api/base/local/registry）
├── scripts/                          # 流水线 01-09（serve→prepare→mane→stress→eval→detector→baselines→types→ablations→audit）
├── data/                             # MedEthicEval/MedEthicsQA/PrinciplismQA/VITAL 四套
├── runs/                             # mane_results/ablations/gamma_sweep/resilience/detector
└── tests/                            # 9 个测试（core/detection/arbitration/resilience/endtoend/mapping）
```

## 2. 代码 ↔ 论文架构一致性映射

| 论文架构（EG-02） | 代码模块 | 状态 |
|---|---|---|
| M0 冲突识别 | detection/ + engine 门控 | ✅ 实现 |
| M1 动态组队 | engine.py assemble | ✅ |
| M2 协商（状态机+贝叶斯） | negotiation + state_machine + orchestration | ✅ |
| M3 仲裁（底线否决+资源修正） | arbitration.py | ✅ |
| M4 GNE 精炼 | gne_solver.py（KKT） | ✅ ★ |
| M5 韧性 | resilience/ | ✅ |
| 评估 | eval/（FDBI/PCI） | ✅ |
| 数据 | data/ 四套 | ✅ |

**结论：论文架构 ↔ 代码实现一一对应，无孤儿模块**（C8 数据流显式、C4 组件齐全）。

## 3. 六维质量评估（code 模式）

| 维度 | 评估 | 发现 |
|---|---|---|
| 结构 | ✅ 模块单一职责（detection/mane/resilience/eval 分离） | 良好 |
| 可读性 | ✅ 中文 docstring 详实（每个文件头有理论说明） | 优秀（docstring 即论文素材） |
| 可复现性 | ✅ 规则模式零依赖可跑；配置外置（config.py） | 好；LLM 模式需 API key |
| 健壮性 | ✅ tests 9 个 + 资源否决/异常场景处理 | 好 |
| 测试 | ✅ test_core/arbitration/resilience/endtoend 等 | 较好（可补 GNE 收敛单测） |
| 效率 | △ 规则模式快；LLM 模式依赖调用 | 可接受 |

**改进建议**：①GNE 求解器单测（KKT 残差断言）②runs 结果的可复现脚本（固定 seed）③LLM 模式的缓存。

## 4. 项目架构文档化（即论文的 System/Method 章节素材）

```
输入：伦理困境场景（SEMA-RAG 锚定事实 + 压力注入）
流程：detection 门控（ERS）→ engine（组队）→ negotiation（状态机 S0-S6 + 贝叶斯编排 + 鲶鱼）
     → arbitration（L1 否决 + 资源修正）→ gne_solver（权重空间均衡 + KKT）
     → 输出（治疗强度/原则权重/满意度/residual_conflict/冲突报告）
评估：FDBI/PCI/分布距离 + 消融（w/o GNE/Catfish/仲裁/平衡正则）+ 韧性（压力→恢复）
```

## 5. 本阶段评估约束

- **G3 证据**：代码/结果真实存在 ✅（runs 有数据）
- **G10 批判**：规则模式="一致性基线"（代码注释已声明 baseline_only=True）——论文须区分"规则基线 vs LLM 语义"的贡献，避免夸大
- **待补**：GNE 收敛单测；seed 固定可复现；LLM 模式与规则模式的结果对比实验

## 决策记录

| 决策 | 结论 |
|---|---|
| 代码↔论文一致 | ✅ 一一对应，无孤儿 |
| 质量 | ✅ 结构/可读/可复现/测试良好（docstring 即论文素材） |
| 待改进 | GNE 单测 / seed 复现 / 双模式对比 |
