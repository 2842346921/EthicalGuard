# EG — 术语表（glossary，中英对照，全文唯一）

> 对应 LaTeX：EthicalGuard-paper-v2.tex；中文翻译：EthicalGuard-paper-v2.zh.md

| 英文 | 中文 | 定义/说明 |
|---|---|---|
| Generalized Nash Equilibrium (GNE) | 广义纳什均衡 | 各智能体在共享耦合约束下优化自身目标的均衡概念 |
| entropy-regularized GNE | 熵正则化 GNE | 本工作实际求解的对象：目标含熵正则项 η·Σαlogα，η→0 趋于经典 GNE |
| shared coupling constraint | 共享耦合约束 | 全体智能体共同承担的约束 g(x)≤0（资源可行性 + 四原则底线） |
| ethical floor | 伦理底线 | 四原则最低权重（默认 floor=[0.15,0.20,0.15,0.20]），不可协商 |
| floor-guarantee theorem | 底线保证定理 | 定理 1：收敛解满足 v\*_k ≥ floor_k − KKT_total |
| floor activation | 底线激活 | 约束在解处绑定（λ_k>0 且 v\*_k=floor_k），使定理非空洞 |
| shadow price | 影子价格 | KKT 乘子 λ_floor,k：放松底线的边际价值（"哪条原则被拉扯"） |
| primal–dual method | 原始-对偶方法 | 双时间尺度：内层阻尼最佳响应 + 外层对偶次梯度上升 |
| KKT residual | KKT 残差 | 平稳性/原始/对偶/互补松弛四项残差之和 |
| balance penalty (γ) | 平衡惩罚（γ） | FDBI 方差惩罚进入目标的权重 |
| FDBI | 四维平衡指数 | FDBI = 1 − σ/μ，值→1 平衡、→0 冲突 |
| reliability weight (r_i) | 可靠性权重（r_i） | 贝叶斯 Beta 后验均值，只用于触发仲裁与信息质量报告 |
| stance weight (w_i) | 立场权重（w_i） | 用于构成集体向量 v 的权重，带下限 ε（与 r_i 解耦） |
| weight floor (ε) | 权重下限（ε） | w_i ≥ ε，防低表达力方被压制（命题 2） |
| decoupling | 解耦 | 信息可靠性 ≠ 道德权重；可靠性不管立场代表权 |
| collective vector (v) | 集体向量（v） | v = Σw_i·α_i / Σw_i，底线约束的求值对象 |
| hard veto (L1) | 硬否决（L1） | 仲裁者强制拒绝违反底线的提案 |
| CAMP vote | CAMP 三值投票 | KEEP / REFUSE / NEUTRAL 玩家完备性投票 |
| catfish agent | 鲶鱼智能体 | 非策略异议者：针对最被忽视原则注入异议，打破沉默共识 |
| collusion | 合谋 | 多方协调压制某方立场（如四方压低患者自主） |
| ERS | 冲突风险评分 | 冲突检测门控信号（监督检测器/LLM/规则三轨） |
| resilience | 韧性 | 压力下的一致性（l1）/稳健性（l2）/可恢复性（l3） |
| abandonment | 原则放弃 | LLM 模式下压力导致某原则权重跌破底线的证据 |
| critical abandonment intensity (τ) | 临界放弃强度（τ） | 首次出现放弃的压力强度 |
| syllogism | 三段论 | 大前提/小前提/结论的可审计理由格式 |
| baseline_only | 基线一致性标记 | rule 模式的韧性报告只作一致性基线（true） |
| MedEthicEval | MedEthicEval | 违规/均衡/优先级困境数据集 |
| MedEthicsQA | MedEthicsQA | 选择题+开放题伦理 QA 数据集 |
| PrinciplismQA | PrinciplismQA | 原则锚定 QA 数据集 |
| VITAL | VITAL | 价值分布与可引导性数据集 |
| LLMEval-Med | LLMEval-Med | 医疗 LLM 评测数据集（立场质量检查） |
| four principles | 四原则 | 行善 B / 不伤害 N / 自主 A / 公正 J |
