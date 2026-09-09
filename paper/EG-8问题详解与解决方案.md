# EthicalGuard：8 个核心问题详解 + 解决方案 + 理论叙事优化方案

> 状态：2026-09-05。三审稿人 + 代码核查 → 方案 → **证据实验 E1/E2/E4 已跑完（local, Qwen3-8B）**。
> 目标：把"引擎完整但证据不足"升级为"可投 2026 顶会"。

---

# 实验结果更新（2026-09-05，local 全跑完）

## E2 真实 LLM 基线（审稿人 A1/T4 已解决——统计显著）

| 方法 | FDBI | 底线违例率 | 违反场景率 | 说明 |
|---|---|---|---|---|
| 真·单 LLM（同 Qwen3-8B 直接决策） | 0.669 | 0.121 | **45.7%** (16/35) | 1 次 LLM 调用 |
| MedAgents 式（五方 LLM 协作，无 GNE/仲裁/底线） | 0.803 | 0.057 | **22.9%** (8/35) | 5 次 LLM 调用 |
| **MANE** | **0.892** | **0.000** | **0.0%** | 完整机制 |

配对 t 检验（同 35 场景）：single vs MANE FDBI t=-12.04 p<0.0001；medagents vs MANE t=-6.29 p<0.0001。
- "基线稻草人"指控推翻：同模型真 LLM 直接决策，MANE 仍显著更好
- "多角色协作就够了"推翻：MedAgents 式违反率 22.9% vs MANE 0%——**增量在约束+仲裁，不在多角色**
- 违反全部集中在 **J（justice）<0.20**（0.10-0.15）——单 LLM/协作系统性忽视公正/资源分配，MANE 守住（领域上有意义）

## E1 底线激活（审稿人 T1 的修复证据，local 成立）

- **提案层 22/35 场景违反底线**（physician 20 / patient 12 / family 9 / ethics_committee 10 / hospital_admin 3）
- **终态层 0/35 违反**（MANE 全守住）
- 构造压力场景 OFF/ON 未跌破 0.15（默认底线对 LLM 协商偏低，终态被平衡正则拉向均匀）——需报 floor 敏感性
- 结论修正：不是"乘子激活"叙事，而是"**提案层天然偏执 → 系统兜底拉回**"的修复证据（更符合伦理语义）

## E4 指标校准（审稿人 A3/F 的指控被排除）

- FDBI 退化：MANE 0.889 vs 随机游走 0.438 / 单极 0 / 双极 0——非机械分
- 盒敏感性：clip 默认盒 FDBI 不变（0.8891=0.8891）——终态本在盒内，盒未作用，机械抬升排除
- 双轴主表：35 场景底线违 0、可行 100%、KKT ~1e-7

## 遗留

- 跨模型 Llama-3.1-8B 启动失败（MC4 非 blocking，E2 主证据已够）——待排查 vllm_llama31.log
- MedAgents 式是简化版（1 轮各自提案+平均），非忠实 5 步复现——必要时补

# E2 违反模式深挖：为什么违反全在 J？（发现 or 缺陷 的判定实验）## 现象
真·单 LLM（physician 角色）违反场景 16/35，**全部跌破 J（justice）**（0.10-0.15 < 0.20）；
MedAgents 式违反 8/35 也全在 J。MANE 无违反。

## 深挖发现（反直觉，是可发表发现的前置证据）
违反 vs 不违反场景的特征对比（35 场景状态字段均值）：

| 特征 | 违反场景 (16) | 不违反场景 (19) |
|---|---|---|
| resource_pressure | **0.284（低）** | 0.566（高） |
| family_conflict | **0.237（高）** | 0.0 |
| severity | 0.762（高） | 0.6 |

**违反不是发生在"资源分配"场景，恰恰相反**：违反场景是**单一患者、无资源竞争、高人际冲突**的
场景（囚犯患者、终止妊娠、肿瘤根治术、艾滋病生育、主动脉夹层放弃治疗）；不违反场景反而是
**多患者资源分配**场景（心脏移植/呼吸机分配——rp 高时模型主动考虑公平）。
含义：单 LLM 的 Justice 缺陷出现在"**无结构性分配压力、纯医患/家庭伦理张力**"的角落——
正是分配正义最容易被忽视之处（"分配正义不只稀缺时的分配，也包括对每个个体的平等对待"）。

## 必须排除的基线偏置（关键对照实验）
E2 的 single_llm_llm 用 physician 角色（persona="医师：基于循证给出疗效最优方案"、
principle="beneficence"、system prompt 显式"你守护的伦理原则是 行善"）——**行善偏置是显式的**。
审稿人会说"你让它当医师当然偏向行善"。

**判定实验**（已加入 11_real_llm_baselines.py）：
- `neutral_single_llm`：同一 Qwen3-8B、**中性 prompt**（无角色/原则偏置，要求四原则独立权衡）
  ——若 neutral 也违反 J → **模型内在偏置**（真发现，可发表："LLM 在人际伦理张力下系统性
  忽视公正，即使无角色引导"）；若 neutral 守住而 physician 违反 → 违反来自角色 prompt
  （基线偏置）→ 主基线改用 neutral，并把"角色化加剧偏置"写成发现。
- 两种结果都是论文信息：前者=模型层发现，后者=prompt 层发现。论文表格须含 4 列：
  physician 角色单 LLM / neutral 单 LLM / MedAgents 式 / MANE。

## 写作方向（无论结果如何）
1. 若模型内在偏置：R1-"LLM 伦理决策的公正盲区"——单 LLM 在无分配压力的临床伦理张力中
   系统性低估 justice，约束化协商（MANE）修复该盲区 → 与 MedAgents 等"提高 accuracy"
   的叙事彻底区分（我们修的是**原则覆盖完整性**）。
2. 若角色偏置：基线公平性修复 + "角色 prompt 如何塑造伦理权衡"的消融（physician vs neutral
   vs 委员会 prompt 的 justice 权重对比）。
3. 引 2603.13807（bounded-rational 聚合）做理论钩子：单决策者受角色认知约束 → 需要
   多角色 + 底线约束的聚合。

---

# 第一部分：评估有效性 4 问（审稿人 blocking——先修实验）

## 问题 A1：基线是硬编码规则，非真实 LLM——"优于单 LLM"无法归因

### 问题详解
代码证据（`eval/baselines/__init__.py`）：
```python
def single_llm_baseline(scenario):      # ← 名字叫 "single_llm" 但根本不是 LLM
    w = np.array([0.40, 0.35, 0.15, 0.10])   # 固定权重，从不随场景变化
    t = int(round(np.clip(sev, 0, 1) * 3))    # 只用了 severity 一个字段
    return Proposal(...)                      # rationale 是写死的字符串
```
harmony 同理（固定 pro/anti 两组权重）。我们声称"MANE FDBI 0.892 > single_llm 0.490"，但：
- 对手不会思考、不会看场景、不会回应其他方——等于"整台机器 vs 一块砖头"
- 无法回答审稿人第一问："你比真实 GPT-4/Qwen 单模型直接回答好多少？"
- 无法定位贡献来自哪：LLM 协商？贝叶斯加权？仲裁？GNE？——消融里删掉的每个机制，基线里都没有，所以"都有效"是必然

### 解决方案（实现层）
新增 `single_llm_baseline` 的真实 LLM 版 + MedAgents 式协作版：
1. **真·单 LLM 基线**：同一 Qwen3-8B、同一场景文本、同 max_tokens，让它直接输出治疗等级 + 四原则权重 + rationale（无协商、无其他 agent）——用现有 `_llm_act` 的单角色调用即可，约 10 行
2. **MedAgents 式协作基线**：复用五方角色 + LLM 后端，但**剥离 GNE/仲裁/底线机制**——每轮各 agent 提案后直接加权平均（不求解、不仲裁、不融合），作为"无保障机制的 LLM 协作"对照
3. **同 token 预算对照**：报告每方法每场景的 LLM 调用数/token，排除"多花算力换高分"
4. **同 29 场景、同口径**，报 ≥3 seeds 方差

### 论文叙事借用
2025-26 顶会（MedAgents ACL24 Findings、MDAgents、AgentBenchMedicine、TeamMedAgents 2025）全是"多角色 LLM 协作提高 QA 准确率"。我们的差异化定位是：**它们的评估指标是 answer accuracy；我们的任务是"约束下的方案生成"，指标是"约束满足率 + 底线守护 + 韧性"——评估维度的切换本身就是贡献**。但要成立，必须在**它们的场景/任务形态**上证明我们有增量（见理论问题 T4）。

---

## 问题 A2：生成与打分同一模型（Qwen3-8B），无人类金标——自评闭环

### 问题详解
评审型评估流程：`agent (Qwen3-8B) 生成 rationale` → `Judge (同一个 Qwen3-8B) 打分`。这不是普通自评，是"同一套偏好偏置同时负责生成与评分"。模型会对自己的措辞风格、约束格式、架构词汇给高分——系统性的自我表扬偏差，无外部校准。伦理领域尤其致命：**"裁决正确"在逻辑上无法与"模型自洽地自我表扬"区分**。

### 解决方案
1. **小规模人类锚定（最小成本，最高收益）**：2-3 名医生/临床伦理学家，对 10-15 个场景的 MANE 输出盲评三轴：可接受性（接受/拒绝/部分）、原则遵守（四原则各 0-1）、伤害风险（低/中/高）。报 Cohen's Kappa（人-人）+ 系统输出-人类判定一致性。
2. **异构 judge**：评审改用不同模型家族（如 GPT-4o/Claude 或至少 Qwen 更大档），报 judge 间一致性。
3. **公布评判 prompt 全文** + 修复前/后版本（我们 rubric 0.073→0.960 的历史就是可复现性证据，主动报告反而加分）。

### 论文叙事借用
2025-26 医疗 LLM 评估的主流（LLM-as-judge in Healthcare scoping review、Human-Machine Agreement in Medical Ethics 等）都是"LLM judge + 人类小样本校准"。我们照做即达标；"同模型自评"在当前文献里已被视为不可接受做法。

---

## 问题 A3：指标全自建（FDBI/ECS/nq），无外部锚，FDBI 受 L3 盒机械抬升

### 问题详解
**FDBI = 1 − σ/μ**（变异系数），只度量四原则权重的"离散度小"，不度量"决策对错"。更严重的是结构性缺陷：
- 代码证据：L3 底线约束 `floors=[0.15,0.20,0.15,0.20]` 把每方权重钉在盒内 → 盒的存在本身压缩 σ → FDBI 被机械抬高
- 数值证据：29 场景 final_vector 距底线平均间隙 [0.084, 0.047, 0.128, 0.041]——全部远离底线，但**正因为约束把向量压在盒内不散开**，FDBI 才高
- KKT 残差 0.000 只是求解器内部收敛检查（优化器跑对了≠解伦理正确）
- ECS 的 veto 维度（底线/可行/KKT）恰好是仲裁/GNE 负责强制的维度——准循环论证

### 解决方案
1. **退化系统校准**：对"恒定输出 0.25×4""纯规则""随机游走"跑同一指标管线，给出 FDBI 的机械基线——证明 0.892 不是低信息量输出也能拿到的分
2. **L3 盒敏感性**：floor 放宽到 0.05 / 收窄到 0.30，看 FDBI 变化——排除机械抬升
3. **主表换双轴**：保障轴（约束满足率/底线违反率/τ 临界强度）+ 满意度轴（min satisfaction）分表；FDBI 降为次要诊断指标
4. **单一外部锚**：至少一个维度接入既有可判分的伦理基准（如 MedEthicEval 的 knowledge/priority 有答案的子集）
5. ECS 补权重全域扫描（α∈[0,1]）+ veto 实现变体 + 人类盲评排序验证

### 论文叙事借用
"先满足硬约束、再优化软质量"的词典序立场在伦理领域可辩护（对应 Rawlsian 优先原则），但必须由外部锚兑现。关键叙事升级：**从"平衡度竞赛"转向"guarantee 竞赛"**——像 2025-26 的 safety/alignment 文献（constitutional agent governance 等）一样，把"约束满足率 100% + 底线违反 0"当作主结果，FDBI 只是辅助。

---

## 问题 A4：nq 条件差 0.111 < 跨机噪声 0.331——韧性主张无法与噪声区分

### 问题详解
我们声称"协商质量 nq（HEALTHY/DEGRADED）与恢复力 R_rec 相关"：
- nq HEALTHY 组 R_rec 均值 0.548（n=6）vs DEGRADED 组 0.437（n=4）→ 差 0.111
- 但项目自己记录：同一场景 PQ-4-10 服务器跑出 0.863、本地跑出 0.532 → 跨机波动 **0.331**
- **条件差只有自报噪声的 1/3** → "两类场景可区分"与"纯噪声"在统计上无法区分（n=10 太小）
- temperature=0 下出现 0.33 量级波动，说明 vLLM 批处理/浮点/配置层有不确定性——这首先污染韧性评估的基线（基线不是确定性的，却被当单点）

### 解决方案
1. **同机配对比较**：nq 的 HEALTHY/DEGRADED 判定与 R_rec 测量必须在**同一台机器、同一批运行**内配对完成——消除跨机噪声
2. **bootstrap CI**：n=10 太小时报点值无意义，给 95% CI
3. **机内 vs 跨机分离报告**：repeat=3 的 std 是机内重复性，0.863 vs 0.532 是跨机可复现性，两个量分开报
4. **补异构模型**：另一开源 7-8B 或商用 API 跑韧性子集，证明结论是框架层的
5. **固定 vLLM 版本/精度/批处理参数** + 排查跨机差异来源
6. 直接回答审稿人问题："该差异是否超过自身记录的可复现噪声？"——诚实报告：当前不超过，需重做实验

### 论文叙事借用
2025-26 的 LLM 韧性/压力测试基准（M2DE multi-stressor、multi-turn stress testing、adversarial stress testing of role-playing agents 2608.03166）都要求：多 stressor × 强度扫描 + 同机配对 + CI。我们的"7 压力×4 强度反事实协议"结构上对齐这些新基准，但必须补统计纪律。差异化机会：**把我们的协议表述为"首个针对伦理协商的韧性基准"**——他们测 role-play/工具可靠性，我们测"伦理底线在压力下的坚守"。

---

# 第二部分：理论/叙事 4 问（怎么修立场与论证）

## 问题 T1：底线约束从未被激活——"伦理底线=共享约束"核心主张无实证激活案例

### 问题详解
数值铁证（上面 A3 已列）：29 场景 final_vector 距底线平均间隙 [0.084, 0.047, 0.128, 0.041]，**任何场景任何维度都未触底（最小间隙 0.003）**。
含义：默认 floor=[0.15,0.20,0.15,0.20] 远低于各方自然平衡点（~0.25），约束从不生效。我们论文的核心创新主张是"伦理底线不可协商 = GNE 共享约束"，但**在全部实验中这条约束从未被激活**——就像写了条法律但从未有人违反过，审稿人必问："你的底线真的工作过吗？"（内部 G10b 早就指出：T1 底线保证在默认参数下是平凡的 KKT 推论）

### 解决方案
**补"底线激活场景"实验（这是最高优先级理论实验）**：
1. 构造压力场景让某方权重天然被压向 <0.15（如：某场景中 patient 强烈要求 autonomy 但 medical 客观事实强烈反对 → 无约束时 autonomy 会被压到 0.1 以下）
2. 关约束跑（baseline）→ 记录 violation；开约束跑（MANE）→ 验证 v 被拉回 ≥floor
3. 报告"约束激活前后对比"：同一场景，无 floor 时某原则跌破 0.15，有 floor 时被钉在 0.15——**这是"底线=硬约束"唯一的实证时刻**
4. 若默认 floor 太低导致很难激活 → 调 floor 到能激活的水平（如 0.20-0.25），或报告 floor 敏感性（0.05/0.15/0.20/0.25 各跑一遍，看何时开始约束生效）

### 论文叙事借用
"约束激活实验"在博弈论/约束优化文献是标准做法（KKT 互补松弛的激活集分析）。引用 2025-26 的 constitutional agent governance 类工作：它们也是"约束只在违规时显形"——我们的激活实验证明**约束不是装饰而是真正兜底**。这是把"底线=共享约束"从形式化主张升为**实证主张**的关键。

---

## 问题 T2：catfish 无博弈结构——"打破共谋"无法形式化

### 问题详解
代码证据：catfish 在三处被显式剔除——
- `arbitration.decide`: `proposals = [p for p in proposals if p.agent != "catfish"]`（不进仲裁）
- `negotiation`: `last_proposals = {... if p.agent != "catfish"}`（不进 GNE 求解）
- `engine.__init__`: catfish 独立于 `all_agents` 构建
它的唯一作用是：提案文本进入历史 → 下一轮 LLM 的 prompt 历史块可见 → 影响后续提案。
问题：我们声称"鲶鱼破坏坏均衡/打破共谋"（坏均衡排除论证 T2），但 catfish **没有战略空间、没有支付函数、不参与不动点分析**——它在博弈结构里根本不存在，"打破共谋"没有标准博弈论表述。消融证据（w/o Catfish 轮次↓仲裁↓满意↓）只能说明"异议文本有影响"，不能说明"打破共谋不动点"。

### 解决方案（两条路选一）
**路 A（形式化 catfish 为博弈者）**：给 catfish 一个真正的博弈角色——
- 定义 catfish 的支付：它代表"未被充分代表的极端立场"，支付 = 与集体向量的偏离度（越偏离越满意）——它是**逆势博弈者**（contrarian），天然反对收敛
- 把它放进 GNE 求解：catfish 的加入使集体向量必须兼顾它的极端诉求 → 均衡被"拉回"到不忽略任何方
- 理论论证：加 catfish 的均衡集 ⊆ 不加的均衡集（异议排除"忽略少数派"的伪均衡）——需要可证明或至少数值验证
**路 B（放弃"打破共谋"叙事，改"异议注入的实证效果"）**：不声称形式化，只报实证——w/o Catfish 消融显示轮次↓/仲裁↓/满意 min↓ = 异议让协商"慢而全"，用 process_trace 的异议被消费率做证据。诚实但弱。

### 论文叙事借用
2025-26 有 LLM 谈判协议机制研究（"Impact of Protocol Mechanics on Multilateral LLM Negotiations" JSAI 2026、CoopEval 2604.15267 社会困境合作机制）——它们研究"协议机制（含异议/否决）如何改变均衡结果"。路 A 把我们放进这个叙事：catfish = 协议层的 contrarian 机制，有可测量的均衡效应。路 B 则是"工程实证"，审稿人可能接受但 novelty 弱。

---

## 问题 T3：GNE 标签与教科书错位——实现是"共享公共变量的耦合最佳响应"，非标准 GNE

### 问题详解
教科书 GNE（Facchinei & Kanzow）：每个 agent 有**自己的策略 x_i**，仅共享约束 g(x)≤0；均衡 = 每个 agent 对自己的 x_i 最优、全体共享约束可行。
我们的实现（`gne_solver._best_response`）：所有 agent 共享**同一个集体向量 v**（`v = Σ w_i α_i`，`_collective`），每个 agent 的最佳响应都基于这个公共 v（`_best_response(v, ...)` 用同一个 v）。这是"共享公共变量的耦合最佳响应"——结构上更接近**潜在博弈/团队博弈**或"公共决策的协商机制"，不是教科书 GNE。
审稿人若深挖 KKT 推导会发现：策略空间/支付/约束的精确形式与"GNE"称谓需对齐说明。代码注释自己也承认耦合结构。

### 解决方案
**三个选项，按诚实度排序**：
- **选项 1（改标签）**：承认实现是"共享公共向量 v 的约束耦合优化"，把它表述为 **GNE 的"集体向量博弈"变体**或明确说"受 GNE 启发的约束均衡求解"——不叫 GNE，叫"约束协商均衡（Constrained Negotiation Equilibrium, CNE）"。风险：放弃 GNE 的品牌价值。
- **选项 2（改实现，对齐教科书 GNE）**：让每个 agent 有自己的策略权重向量 w_i（非共享 v），仅底线约束 g(v)=floor−v≤0 作用于**集体加权和** v=ΣW_i w_i——这才是真正的 GNE：每个 w_i 独立策略、集体约束共享。改造量中等（solver 的 best_response 从"对 v 响应"改成"对 w_i 响应 + 集体约束的乘子耦合"）。
- **选项 3（不改代码，改论证）**：写"建模选择论证"——为什么公共 v 而非独立策略是**伦理上更正确的结构**：伦理共识本来就是"一个集体立场"，不是各方私下策略；公共 v 的耦合协商 = "各方围绕同一个候选共识讨价还价"。这其实是伦理语义的优势，不是缺陷——但要在论文里正面论证，不能藏。

### 论文叙事借用
2025-26 LLM-Nash 系列（2507.08208 "Reasoning and Behavioral Equilibria in LLM-Nash Games"、2604.27167 "What Suppresses Nash Equilibrium Play in LLMs"）发现：LLM 在博弈中**行为均衡 ≠ 教科书均衡**，有系统性偏离。这反而支持我们：**LLM 协商本来就不收敛到教科书均衡，需要显式求解器**——"LLM 生成立场（发散、带偏见）+ 约束求解器收敛（保证性）"的混合架构，正是对"LLM 直接博弈不收敛"的回应。把 T3 从"标签缺陷"重写为"混合架构的正当性来源"。

---

## 问题 T4：无 MedAgents 等既有 LLM 医疗多智能体对比——RELATED WORK 定位落空

### 问题详解
paper/ 目录 grep "MedAgents" 零命中。MedAgents（ACL 2024 Findings，arXiv 2311.10537）是"多角色 LLM 医疗协作提高 QA 准确率"的开创工作，2025-26 有 MDAgents、TeamMedAgents、MDTeamGPT、AgentBenchMedicine 等一批。审稿人第一问："相比它们，你的增量是什么？"——我们当前无法回答，因为：
- 任务形态不同：它们做**医疗问答（QA）**（给症状问诊断），我们做**伦理困境的方案协商**
- 没有在同一批数据/任务上跑它们

### 解决方案
1. **任务映射论证**：把它们的 QA 形态与我们的协商形态对齐——伦理困境题本身就是"无唯一答案、需权衡"的 QA 变体，可以在 MedEthicEval/PrinciplismQA 的 open-ended 子集上跑它们（给 LLM 多角色协作生成答案）vs 我们的 MANE
2. **受控对比**：同场景、同模型、同预算，报告：约束满足率 / 底线违反率 / τ 临界强度 / 答案质量（若数据有金标）/ 轮次成本
3. **失败案例分析**：至少 3 个案例说明 MedAgents 式协作"无约束保障"会在什么场景产出违反底线的方案，而 MANE 守住——这是差异化的实证
4. **related work 补全**：MedAgents/MDAgents/TeamMedAgents/AgentBenchMedicine + 伦理多智能体（NeurIPS 2025 "Many LLMs More Utilitarian"、constitutional governance 类）全部进对比表

### 论文叙事借用
关键定位切换：**我们不是另一个"医疗多智能体提高准确率"的工作，而是"医疗多智能体 + 约束保障机制"的工作**。MedAgents 等证明了"多角色协作有帮助"，但没解决"协作产出可能违反伦理底线"——我们的 GNE 底线约束 + 仲裁 + 韧性正是补这个洞。叙事："多角色协作（他们）+ 底线保障（我们）→ 伦理安全的共识"。

---

# 综合：理论/叙事重写建议（回答"我们是什么"）

## 一句话新定位
**EthicalGuard：把临床伦理共识重新定义为"受不可协商底线约束的多方博弈均衡"，并用 LLM 协商 + 约束求解混合架构实现、用反事实压力韧性协议验收。**

## 三层叙事骨架（替换现有"15 模块"描述）
1. **问题层**：LLM 多智能体在伦理场景的协作已被证明有用（MedAgents 等），但①协作产出可能违反伦理底线 ②无保障机制的协作会静默共识（忽略少数派）③无系统性的"伦理韧性"验证——三个洞没人补
2. **机制层**：底线=可行域约束（GNE 耦合）而非提示词（Constitutional AI 把规范落在训练/事后分类器，我们在协商运行时强制）→ 仲裁=程序正义 + 最脆弱方（Rawlsian maximin 机制化）→ catfish=协议层 contrarian（打破静默共识）→ 混合架构=LLM 生成立场 + 求解器保证收敛（回应 LLM-Nash 发现的"LLM 不收敛到教科书均衡"）
3. **验收层**：约束满足率/底线违反率为主指标（guarantee 竞赛，对齐 2025-26 safety 文献）+ 7 压力×4 强度反事实韧性协议（对齐 M2DE/stress-testing 基准）+ 人类小样本 Kappa 锚定（对齐医疗 LLM 评估主流）

## 必须补的 6 个实验（按优先级）
| # | 实验 | 解决 | 成本 |
|---|---|---|---|
| E1 | **底线激活场景**（构造某方被压向 <floor，关/开约束对比） | T1 | 低-中（场景构造+协商） |
| E2 | **真·单 LLM 基线 + MedAgents 式协作基线**（同模型同预算） | A1/T4 | 中（需 vLLM 跑 2 组 ×29 场景） |
| E3 | **人类小样本盲评 + Kappa**（2-3 医生/伦理学家 × 10-15 场景） | A2 | 中（协调人） |
| E4 | **FDBI 退化校准 + L3 盒敏感性 + 双轴主表** | A3 | 低（纯分析） |
| E5 | **同机配对韧性重跑 + bootstrap CI + 异构模型** | A4 | 高（需 2 模型 × 韧性全量） |
| E6 | **catfish 形式化（路 A）或放弃共谋叙事（路 B）** | T2 | 高（路 A 需重设计求解） |

## 最诚实的路线图
- **投稿前第一关**：E1 + E2 + E4（纯实验，1-2 周）→ 主表可站
- **投稿前第二关**：E3（人类 Kappa）→ 打破自评闭环
- **投稿前第三关**：E5 + E6 视目标 venue——若投医学信息学（AMIA/AIME）E5 可缓；若冲 ACL/AAAI main 需要

---

# 第三部分：2025-2026 顶会锚点 + 理论叙事优化（调研版，2026-09-04）

> 来源：arXiv API/OpenAlex/出版社逐条核验（检索截止 2026-09-04）。含 ID 纠错——引用前务必用下面的正确 id。
> 完整调研报告：`C:\Users\zhangsan\Documents\ethicalguard_research\Q1-6_anchors_report.md`

## 新定位句（可直接进摘要/引言，英文）

*"Prior work treats constitutions as training signals (Constitutional AI), inference-time filters (Constitutional Classifiers), runtime policy languages (deontic policies), or institution-level governance graphs; none formalizes an ethical floor as a shared feasible-set constraint inside a solvable multi-party game. We show that treating the four-principles floor as shared GNE constraints — with a committee-selected equilibrium and a 7×4 counterfactual stress protocol — yields consensus that provably cannot violate the floor, and we anchor its quality to human ethicists via chance-corrected agreement."*

**一句话中文定位**：不是又一个医疗 QA 多智能体，也不是又一个伦理 LLM 评测——是回答"伦理底线如何成为多智能体系统里**可证明、可求解、可压测**的数学对象"的"约束-均衡-治理"三合一论文。

## 六大问题的锚点清单（正确 id + 怎么用）

### Q1 博弈均衡 vs LLM 协商
| 工作 | 正确 id/venue | 怎么用 |
|---|---|---|
| Zhu LLM-Nash Games | arXiv:2507.08208 | 引言立"LLM-博弈均衡"词汇（均衡在推理/prompt 空间）→ 转场"但无伦理约束、无共享约束求解器→GNE 的位置" |
| NegotiationArena | arXiv:2402.05863 | 负面证据：裸 LLM 谈判不收敛理性均衡 → 为什么不能靠自由协商 |
| Kwon et al. | EMNLP24 Findings, arXiv:2402.13550 | 同上（LLM 谈判能力系统性欠佳） |
| What Suppresses NE Play | arXiv:2604.27167 | 同上（机制证据） |
| **DICE** | arXiv:2606.08068 | 支撑：均衡多重性 ill-posed → 需要**选择装置** = 我们的 CAMP 仲裁（HQRE 对照） |
| Cooperate or Collapse | NeurIPS 2024, arXiv:2404.16698 | 无约束→社会困境自私崩坏 → 底线必须由系统施加 |
| ⚠️ 雷区 | arXiv:2604.11840 (solver-sampler) | 不说"模拟人类医生行为"，说"求解规范性可行共识" |
| ⚠️ 你引的"协议机制" | 实为 JSAI2026 国内会议，作者待核实 | 别当顶会引用 |

### Q2 伦理底线/宪法约束
| 工作 | 正确 id/venue | 怎么用 |
|---|---|---|
| Constitutional AI | arXiv:2212.08073 | 谱系起点：宪法=训练期软信号 |
| **CMAG（最近邻，必须划界）** | arXiv:2603.13189, AMSTA 2026 | 它已有"宪法+硬约束+伦理评分"——逐条划界：无博弈建模/无共享约束可行域语义（post-hoc 过滤非解空间边界）/无协商均衡/无仲裁/无压测/无人类锚定 |
| Constitutional Classifiers | arXiv:**2501.18837**（不是 2501.18867） | 硬过滤≠解空间 |
| Deontic Policies | arXiv:2606.19464 | 运行时策略语言族（企业合规） |
| Institutional AI | arXiv:2601.11369 | 制度层治理图（Cournot 合谋）——与我们"底线优先于偏好工程"同构 |
| AgentCity | arXiv:2604.07007 | 同上（代理经济治理） |
| Social Contract AI | arXiv:2310.17769 | 反例：规范习得路线 OOD 脆弱 → 支持显式编码底线 |
| NeMo Guardrails/Llama Guard | 2310.10501 / 2312.06674 | 只能拦或改，不能保证可行域内解存在 |

### Q3 为何 GNE（概念选择论证）
- **背景**：Rosen 1965 (Econometrica) + Facchinei-Kanzow 2010 (SIAM Review)——页码引用前复核
- **空白（双刃剑）**：截至 2026-09 arXiv/OpenAlex 未见 GNE+LLM 协商直接工作 = 组合新，但需**自证**
- **NBS-in-LLM 证据**（证明"替代概念已流行但用途不同"）：MUNBa arXiv:2411.15537（去学习）、MoE-Nash Merging arXiv:2510.16138 (ICLR'26)（模型合并）——都不是带底线约束的伦理协商
- **Stackelberg**：arXiv:2507.09407 (Zhu) —— 需写清"CAMP 是治理者不是收益玩家"，避免被说"你也是 Stackelberg"
- **可解性背书**：arXiv:2509.16826（受约束动态博弈）、arXiv:2410.15335（约束 MARL 原对偶）
- **必须做的概念消融**：同任务比 无约束 NE / 软惩罚(CMAG式) / 硬过滤(Classifier式) / GNE 共享约束 的底线违背率+共识达成率+稳定性——这组实验本身就是卖点

### Q4 医疗 LLM 多智能体（ID 纠错后）
| 工作 | 正确 id/venue | 差异 |
|---|---|---|
| MedAgents | **arXiv:2311.10537**, ACL24 Findings | QA 形态（病例→答案） |
| **MDAgents** | **arXiv:2404.15155**, NeurIPS24 oral（不是 2402.01771=BlackMamba!） | 自适应 solo/group 协作 |
| TeamMedAgents | arXiv:2508.08115 | Salas 团队理论→小模型机制 |
| MDTeamGPT | arXiv:2503.13856 | MDT 会诊+记忆演化 |
| AgentClinic | arXiv:2405.07960 | OSCE 四角色环境——可作沙盒 |
| MedAgentBench | arXiv:2501.14654 | 虚拟 EHR 环境 |
| npj DM 2026 (Kather) | DOI 10.1038/s41746-026-02443-6 | 评测方法学标杆：agent 临床决策要系统化对比 |
| ⚠️ AgentBenchMedicine | 仅 GitHub 套件，无独立论文 | 引用 GitHub 即可；"npj DM 配套"需核实（用户名吻合=高置信推断） |

**桥接实验**（写进论文）：① 同数据异指标：在 MedQA 等上把 MANE 输出压成答案测 accuracy（不主打，作"不输推理"下限）② 主指标矩阵：底线违例率（核心增量，现有方法无此指标）/共识率/κ/跨种子稳定性 ③ MedAgents+MDAgents 作"无硬约束基线"同病例跑，报其底线违例率（预期高于我们）→ 直接支撑"底线=约束而非角色提示"

### Q5 人类锚定
| 工作 | 正确 id/venue | 怎么用 |
|---|---|---|
| **Mugu et al. (医疗伦理人机一致性)** | JMIR Med Inform 2025, DOI 10.2196/77061 (e77061) | 我们 CAMP 校验的模板（患者自主案例→LLM 判定→人机一致性） |
| LLM-as-judge in Healthcare scoping | arXiv:2605.25273 | 报告规范缺口 → 我们按建议报 κ/AC1+样本量=显式贡献 |
| MedJUDGE scoping | arXiv:2604.25933 | pointwise 85.7%/GPT 族 73.5%/人类验证弱 → 我们补人类锚定即差异化 |
| **Decision Aggregation under QRE** | arXiv:2603.13807 | 定理：bounded-rational 专家下多数投票是最优鲁棒聚合 → 给 CAMP 委员会多数票理论背书（"共识质量=聚合规则性质而非谁的模型强"） |

### Q6 韧性/压力测试
| 工作 | 正确 id/venue | 怎么用 |
|---|---|---|
| **M2DE** | Pattern Recognition 178:113428 (2026) | 立 multi-stressor 家族：我们的协议=M2DE 压力源矩阵的**协商域实例化** |
| Adaptive Stress Testing | arXiv:2505.05665, ACL Findings 2026 | 最坏情形搜索原则（4 档强度=强度格搜索） |
| 2608.03166 | arXiv:2608.03166（⚠️ ADScAI 2026 斯里兰卡小会，8 页短文） | 先例但注明级别；差异化：测角色一致性 vs 我们测底线不可协商性 |
| NegotiationToM | EMNLP24 Findings, arXiv:2404.13627 | 多轮信念压力注入 |
| AgentHarm | ICLR 2025, arXiv:2410.09024 | 切割红队：我们测"善意压力下的道德滑移"非恶意攻击 |
| ⚠️ Validity Audit | arXiv:2607.28685 | 指标陷阱：不要只报通过率，报**违例类型化归因**（哪条底线/哪个压力/哪方先越界）+ scorer 审计 |

**一句话表述**："我们的韧性协议 = M2DE 的多压力源矩阵 × adaptive stress testing 的最坏情形搜索 × NegotiationToM 的多轮信念压力，输出 = [B,N,A,J] 底线违例的类型化归因而非单点通过率；并强调现有压测没有的**反事实干预 + 恢复性验证**（压力撤除后共识回稳）——这是协议级贡献。"

## 九条雷区（写论文时逐条对照）
1. 不说"LLM 自发收敛到均衡/共识"——只说"约束内均衡由求解器给出，CAMP 选均衡"
2. 不与 solver-sampler 批评 (2604.11840) 撞车——明说求解规范性共识，不声称行为拟真
3. 不把"7×4 压力"讲成全新发明而不提 M2DE/2608.03166/AST——主动写成"对 M2DE 家族的协商域扩展"
4. 主指标不只是 accuracy/通过率——核心是底线违例类型化归因 + κ + 跨种子稳定性
5. 人类锚定不可省（CAMP 自评必须配外部人类伦理评审的 κ/AC1）
6. AgentBenchMedicine 引用小心（无独立论文）
7. JSAI 2026 协议论文作者未核——给全标题+会议即可
8. 均衡概念消融不能省（GNE vs NBS vs Stackelberg vs 无约束 NE）
9. 经典博弈文献页码先复核（Rosen 1965 / Facchinei-Kanzow 2010 / McKelvey-Palfrey 1995）

---

# 第四部分：GNE 标签错位——数学重估与修正论证（2026-09-05）

## 一、重估结论：原"错位"判断部分错误，需修正

**原判断**："所有 agent 共享公共 v 的最佳响应 = 非教科书 GNE"。

**数学重估后修正**（看 `_best_response` 结构）：
```
α_i ∝ exp((s_i − soft_i + γ·balance_force(v)·(w_i/W) − dg_floor·(w_i/W)) / η)
```
- **每个 agent 的 α_i 是独立决策变量**（各自在 Δ⁴ 单纯形上 softmax），不是共享的
- v = Σ w_i α_i 是**集体量**（所有 α 的加权和），不是任何单 agent 的策略
- floor 约束 g_floor(v) = floor − v ≤ 0 是**共享耦合约束**（作用于集体量 v）

**这符合教科书 GNE 定义**：各 agent 独立策略 α_i + 共享耦合约束 g(v) ≤ 0（GNEP 的标准形式，
Facchinei-Kanzow 2010）。原判断"公共 v = 非 GNE"是**误读**——v 是策略的函数（聚合），
不是策略本身。

## 二、真正可辨的点：balance 正则进目标 vs 约束

代码注释自认："非凸约束 FDBI 的稳定处理"——**平衡正则 γ·FDBI(v) 是软目标惩罚
（加权进每个 agent 的 J_i），不是 GNE 硬约束**。这才是需要澄清的：

| 组件 | 数学地位 | 说明 |
|---|---|---|
| floor 底线 | ✅ GNE 共享约束（拉格朗日乘子 λ） | g_floor = floor − v ≤ 0，互补松弛验证 |
| balance 平衡 | ⚠️ 软目标正则（γ 进 J_i） | γ·(−‖v−0.25‖²)，非硬约束 |
| 熵正则 η | ⚠️ 正则化（进目标） | 保证内点解/收敛 |

**修正术语**：实现是"**带熵正则与平衡增强的 GNE（entropy-regularized GNE）**"——不是纯 GNE，
但**有明确的博弈论正当名称**。DICE (2606.08068) 的 HQRE 正是"熵正则稳定化多 LLM 协调"——
与 η 熵正则同思路，可作对接锚点（McKelvey-Palfrey QRE 1995 是理论源头）。

## 三、实证判定（scripts/16_gne_modes.py 已实现）

对比两模式（同场景、同求解器、唯一变量=coupling）：
- **collective**（当前）：γ·balance_force(v) 进 agent 目标 → agent 向公共 v 的平衡让步
- **independent**（教科书对照）：去掉 balance 让步，agent 纯追自己 s_i，只受 floor 约束

**已验证事实**：
- rule 模式：两模式**完全同结果**（差 +0.000）——因为规则提案终态 v≈0.25 均匀，
  balance_force(v)=−2(v−0.25)≈0，让步是"死代码"
- local 数据：34/35 场景终态偏离均匀 >0.05（梯度非零）→ **collective 让步在 local 真激活**

**判定逻辑**（local 跑后）：
- collective FDBI/满意min 显著高 → 公共 v 让步贡献真实共识质量 → 用"熵正则 GNE"术语可辩护
- 两者接近 → 让步机制边际价值低 → 需重新设计或诚实降调

## 四、论文写作指引（无论结果）

1. **术语**：用"entropy-regularized GNE / constrained collective equilibrium"而非裸 "GNE"；
   承认 FDBI 平衡是软正则（可解释参数 γ），floor 是硬约束（λ 乘子）——诚实分层
2. **对接顶会**：DICE (2606.08068) HQRE 熵正则、McKelvey-Palfrey QRE（理论源头）、
   Zhu LLM-Nash (2507.08208) 推理空间均衡——都支持"正则化均衡"词汇
3. **公共 v 的优势**（论文写）：可审计（process_trace 基于单一 v）、约束语义清晰
   （底线直接作用共识）、求解稳定（KKT 1e-7）、机构对齐（委员会产出单一建议）
4. **风险**：博弈论 hardcore 审稿人可能仍不接受——主动降级术语 + Related Work 承认渊源
