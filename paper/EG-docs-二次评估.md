# EG docs 二次评估报告（docs/ 文档 × 代码 × runs × paper/ 文档 四方交叉）

> 评估对象：`docs/architecture.md` + `docs/运行指南.md`
> 交叉材料：src 代码（gne_solver/arbitration/mapping/orchestration）、runs 真实结果（mane_results/resilience/ablations/gamma_sweep）、paper/（EG-evaluation、EG-critical-review、EG-01~04）
> 方法：①docs 声称 ↔ 代码实现逐项核验 ②runs 证据 ↔ paper 评估记录的进展对照 ③docs 内部一致性 ④批判审查🔴项在 docs/代码的落地状态

---

## 一、docs 声称 ↔ 代码核验（逐项）

| docs 声称 | 代码验证 | 结论 |
|---|---|---|
| 模块↔理论映射（F1 可行性/仲裁/角色） | `arbitration.py` L1 否决 ✅、`roles.py` 医院管理 ✅、`data/mapping.py` 四盒 ✅ | ✅ 一致 |
| F2 均衡性（正则化 GNE + KKT） | `gne_solver.py` 完整原始-对偶 + 全 KKT 校验（stationarity/primal/dual/complementarity）✅ | ✅ 一致 |
| F3 可审计（三段论 + 状态机） | `agents/base.py` Syllogism ✅、`state_machine.py` ✅ | ✅ 一致 |
| F3 信息不对称（观察掩码 + 贝叶斯权重） | `base.py` observe ✅、`orchestration.py` Beta 权重 ✅ | ✅ 一致 |
| F3 过程（自适应轮次/Catfish/动态组队） | `negotiation.py` ✅、`roles.py` Catfish ✅、`engine.py` assemble ✅ | ✅ 一致 |
| F4 韧性（压力引擎 + 三层次 + 恢复验证） | `resilience/` 四文件 ✅ | ✅ 一致 |
| rule/api/local 三模式 | `llm/` api.py + local.py + rule 路径 ✅；config.yaml `llm.mode` ✅ | ✅ 一致 |
| 运行指南 FAQ：P0-1 资源强制修正 | `arbitration.enforce_resource_cap` ✅（Σρ>cap 逐级降级） | ✅ 一致 |
| 运行指南 FAQ：O1 分歧>阈值取 LLM 值 | `mapping._merge_field` ✅（分歧≤阈取均值、>阈取 LLM、B3 无信息过滤） | ✅ 一致 |
| 韧性两档（rule=基线 / api/local=完整） | `roles.py` baseline_only=True ✅、resilience.jsonl `baseline_only:false`（local）✅ | ✅ 一致 |
| 压力强度网格 0.25/0.5/0.75/1.0 | config.yaml `intensity_grid` ✅、resilience.jsonl `stress_response` 四档 ✅ | ✅ 一致 |

**核验结论：docs 的技术声称 100% 有代码支撑，无虚构、无夸大。**

## 二、runs 真实证据 ↔ paper 评估记录（重大进展对照）

**这是二次评估最重要的发现：runs 中已有此前评估认为"未做"的证据。**

| paper/ 评估记录（EG-evaluation 等） | runs 实际已有 | 状态更新 |
|---|---|---|
| 🟠 "规则 vs LLM 双模式结果未对比" | `resilience.jsonl` 6 行 **mode=local、baseline_only=false**，含真实放弃证据（`abandoned:{justice:authority, autonomy:disease}`、`abandonment_intensity:{justice:0.25, autonomy:0.5}`） | ✅ **已做**（LLM 完整韧性已跑通） |
| 🟠 "GNE 收敛统计不足（单一 KKT 7.2e-8）" | `mane_results.jsonl` 29 行，每行含 `kkt_residual`（如 1.69e-7、1.16e-7），可做残差分布 | ✅ **已具备**（待聚合成统计） |
| 🟠 "消融"（C4 配对） | `ablations_local.txt`：MANE vs w/o GNE / w/o Catfish / w/o 仲裁 / w/o 平衡，含 FDBI/PCI/KKT率/底线违/满意min | ✅ **已跑**（local 模式） |
| 🟠 "权重敏感性"（RQ5 类） | `gamma_sweep.txt`：γ=0.0→1.0 扫描（FDBI 0.864→0.875 单调升、底线违恒 0） | ✅ **已跑** |
| 🟢 真实协商轨迹 | `mane_results.jsonl` 含完整 trajectory/三段论/仲裁原因/ERS | ✅ 充足 |

**二次评估结论：EG 的实证进展远超 paper/ 文档记录——local 模式（Qwen3-8B）已产出完整韧性 + 消融 + γ 扫描。EG-evaluation 的"待 P0 实证闭环"应更新为"P0 只剩金标准对照 + 坏均衡验证两项"。**

## 三、docs 内部一致性与配置实际状态（发现 4 处问题）

### 🔴 D1：config.yaml 的 data_dir 是服务器路径，与运行指南"默认配置什么都不改就能跑"冲突
- 实际：`config.yaml` `datasets.data_dir: "/mnt/zhangheng2025/EthicalGuard/data"`（Linux 服务器路径）；
- 运行指南声称"默认配置什么都不改就能跑（rule 模式）"、"本机默认：什么都不改，直接执行"；
- **冲突**：本机 Windows 直接跑 `01_prepare` 会按服务器路径找数据 → 失败；运行指南 §0/§4A 的"零修改可跑"陈述与配置实际不符；
- **建议**：运行指南明确"本机需设 `ETHICALGUARD_DATA_DIR` 指向 `data/` 或改 config"，或提供 `configs/config.local.yaml` 示例。

### 🟠 D2：docs/architecture.md 落后于实现（仍自称"骨架"）
- architecture.md 末段："当前为骨架：规则模式可端到端跑通，LLM 模式接口已就绪"；
- 实际：`llm/` 已实现 api+local，runs 已有 local 模式 29 行协商 + 6 行韧性结果——**早已不是骨架，LLM 模式已实际跑通**；
- 另 architecture.md 未提 `detection/` 监督检测器（config 有 `detection.checkpoint`、README 有 05_train_detector 章节、runs 有 detector.pt）——架构图缺 detection 模块。

### 🟠 D3：运行指南流水线只覆盖 01-04，未覆盖 05-09（论文写作必需）
- 运行指南只讲 `01_prepare→04_eval`；实际 scripts 有 05_train_detector / 06_baselines / 07_eval_types / 08_ablations / 09_audit；
- runs 里 `ablations_local.txt`（08 产物）、`gamma_sweep.txt` 已存在——**论文证据链依赖的脚本无使用说明**；
- 建议：运行指南补"第 3.6 步：消融/基线/审计脚本"。

### 🟢 D4：docs 未系统记录修复台账（P0-1/O1/B2/B3/P-D/O4）
- 运行指南 FAQ 零散提到 P0-1/O1；代码注释含更多（B2 鲶鱼不占资源、B3 无信息过滤、P-D 裁决底线权重、O4 ERS 门控阈值）；
- 建议：docs 增"修复台账"节（编号/问题/代码位置/影响），论文 appendix 可直接引用。

## 四、批判审查 🔴/🟠 项在 docs/代码层的落地状态（二次评估新观察）

| critical-review 项 | 优先级 | docs/代码现状 | 结论 |
|---|---|---|---|
| T1+T2 定理（底线保证/坏均衡排除） | 🔴 | docs 无；代码无证明（gne_solver 是数值求解非定理） | ⬜ 仍未落地（需在论文层补） |
| 博弈建模选择论证（GNE vs NBS/Stackelberg） | 🔴 | docs 无此节 | ⬜ 未落地 |
| 贝叶斯权重下限（w_i ≥ ε 防压制） | 🟠 | 代码无权重下限（orchestration 权重直接归一化） | ⬜ 未落地（公正风险仍开放） |
| 满意度指标反例分析 | 🟠 | 无 | ⬜ 未落地 |
| 鲶鱼防合谋理论支撑 | 🟢 | 无 | ⬜ 未落地 |

**但注意**：以上 🔴 项属**论文层**内容（定理证明/论证），不属于运行文档职责——docs 无它们是正常的；此处列出是提醒"论文正文仍缺"。

## 五、新发现的实证观察（二次评估独有，供论文参考）

1. **韧性数值的口径问题（需要解释）**：resilience.jsonl 中 `l2_robustness` 仅 0.10/0.05（很低），但 `overall` 0.82/0.77（高）——综合韧性主要由 l1/l3 贡献。论文需解释"l2 低是否说明系统对扰动敏感"或指标口径（l2 可能定义为 1−位移率，低=位移大=敏感性？需核对 metrics.py 定义，避免审稿人误解"韧性高但稳健性低"）；
2. **消融机制差异清晰（可进论文表）**：w/o GNE 底线违 0.125（GNE 确实在守底线）、w/o 仲裁 KKT率 30%（仲裁对收敛贡献）、w/o Catfish 仲裁率降（30%→70%？表里 w/o Catfish 仲裁率 30.00% vs 全量 70.00%——待确认行对齐）；
3. **γ 扫描单调性**：FDBI 随 γ 单调升（0.864→0.875）、底线违恒 0——平衡正则有效且不破坏底线，可直接作为敏感性表。

## 六、总评与修复优先级

```
docs 技术核验：✅ 100% 一致（声称全部有代码支撑）
进展对照：🟢 实证远超评估记录（local 完整韧性/消融/γ扫描已跑）
docs 自身问题：1 🔴（config 路径与"零修改可跑"冲突）+ 2 🟠（架构文档滞后/流水线缺 05-09）+ 1 🟢（修复台账缺）

优先动作：
  P1：docs 修正 config 路径说明（本机 vs 服务器）
  P1：paper/ 评估文档更新——"双模式对比/收敛统计/消融/敏感性"已由 runs 满足，P0 缩至金标准+坏均衡
  P2：architecture.md 升级（去掉"骨架"、补 detection 模块、补 05-09 脚本）
  P2：运行指南补 3.6 消融/基线/审计步骤
  P3：论文层补 T1/T2 定理 + 建模选择论证 + 贝叶斯权重下限（critical-review 🔴 未变）
```

---

## 附：本次评估依据清单

| 材料 | 关键内容 |
|---|---|
| docs/architecture.md | 模块↔理论映射、外部参考实现、三模式 |
| docs/运行指南.md | 环境/配置/流水线/韧性两档/强度网格/FAQ |
| src/ethicalguard/mane/arbitration.py | L1 否决 + CAMP + enforce_resource_cap + 裁决底线权重（P-D） |
| src/ethicalguard/mane/gne_solver.py | 原始-对偶 + 全 KKT + FDBI/平衡力（明确不用 FDBI 梯度防振荡） |
| src/ethicalguard/data/mapping.py | 双通道仲裁（O1）+ B3 无信息过滤 |
| configs/config.yaml | mode=local、intensity_grid、消融开关、detection.checkpoint |
| runs/mane_results.jsonl | 29 行：KKT 1.69e-7/1.16e-7、满意度 0.82-0.99、ERS/supervised、三段论 |
| runs/resilience.jsonl | 6 行 local：abandoned/abandonment_intensity/四档应力曲线 |
| runs/ablations_local.txt | 4 消融 × 7 指标 |
| runs/gamma_sweep.txt | γ 0-1.0 扫描 |
