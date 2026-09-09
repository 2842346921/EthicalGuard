# Catfish 路 A：从"文本异议者"到"博弈结构中的 contrarian"（形式化设计）

> 状态：2026-09-05 设计定稿，待实现。审稿人 T2 问题：catfish 被 5 处剔除（不进仲裁/协商轮
> GNE 输入/共识提取/engine 组队），无战略空间/支付/不动点分析——"打破共谋"无博弈论表述。

## 一、现状锚点（代码事实）

catfish 的**规则实现已经是 maximin 启发式**（roles.py L251-267）：
```python
v = history[-1].collective_vector.as_array()
weakest = int(np.argmin(v))          # 找当前集体最弱原则
w[weakest] = 0.6                      # 把最弱原则抬到 0.6
```
→ "挑战最被忽视的原则"。但它是**旁观者**：提案只进历史文本，不进 GNE 求解
（5 处 `!= "catfish"` 剔除），"抬升最弱"只靠 LLM 下一轮读到文本，无博弈保证。

## 二、路 A 形式化：catfish 作为第 6 个 GNE 参与者

### 设计：保持"不占资源/不投票"，但进均衡求解

catfish 与五方的本质差异要保留（审稿人会检查一致性）：
- **不占资源**：treatment=舒适护理(0)，ρ=0 → 不进资源约束
- **不投票仲裁**：无 CAMP 投票权（它是异议者非决策者）
- **但有博弈策略**：它的 α_catfish 是 Δ⁴ 上的决策变量，进集体向量 v

### catfish 的支付函数（maximin contrarian）

```
J_catfish(α_c) = min_k v_k          # 最大化集体向量最弱维度
其中 v = Σ_{i≠catfish} w_i α_i + w_catfish α_catfish
```

**博弈论正当性**：这是 **Rawlsian maximin 策略**（最弱原则最大化 = 保护最脆弱的价值维度）。
五方各自最大化自己的满意度 s_i·α_i（自利），catfish 最大化集体最弱维度（利他/守护）——
两股力在 GNE 里对抗：五方想把集体拉向自己偏好的原则，catfish 阻止任何原则被压到极低。

### 最佳响应（闭式）

catfish 的最佳响应：把权重全压在 v 当前最弱维度 k* = argmin_k v_k：
```
α_catfish = e_{k*}（或软化：softmax(v_min 方向的梯度)）
```
因为 v_k* 只受 α_catfish,k* 直接抬升 → 抬最弱维度是最大化 min_k v_k 的贪心最优。
这与现规则实现（argmin 抬 0.6）**完全一致**——规则实现是软化的 best response！

### 可证明的保证（T2 的博弈论版本）

**引理（最弱原则保障）**：设无 catfish 时 GNE 均衡集体为 v⁰，catfish 加入后
五方的最佳响应因 v 变化而调整，但 catfish 的 maximin 支付保证：存在 catfish
策略使 min_k v ≥ min_k v⁰（catfish 至少能守住当前最弱原则不进一步下跌）。
证明思路：catfish 把 α 全压 k*，v_k* 单调上升（其他 agent 权重和固定），
min 维度不降。这是"坏均衡排除"的可证版本：**五方合谋把某原则压到极低（伪共识）
时，catfish 的均衡策略必然抬升它**——共谋不动点被 catfish 打破。

### 与教科书博弈的关系
这不是标准 GNE（五方是"个人支付"，catfish 是"集体支付"）——是**混合动机博弈**
（mixed-motive game）：自利方 + 一个利他守护方。有明确的均衡概念（五方 KKT +
catfish maximin best response 的不动点），可数值验证。

## 三、实现方案

### 改动点
1. `gne_solver.py`：`solve_gne_from_proposals` 加 `include_catfish: bool = False`——
   True 时把 catfish agent 加入 agents 列表（satisfaction 用 maximin 专用函数）
2. `CatfishAgent`：加 `satisfaction_maximin()`——返回使 best response 指向最弱维度的
   satisfaction（或用现规则 argmin 逻辑作为 catfish 的确定性策略，绕过 satisfaction）
3. `negotiation.py`：catfish 提案**同时**进 GNE 求解的 init_weights（不再剔除），
   但仍不进仲裁/资源
4. 配置文件开关 `mane.catfish_in_gne: bool`（默认 False，消融对比用）

### 数值验证（catfish 博弈结构 vs 旁观者的差异）
同一批压力场景（含"某原则被五方一致压低"的合谋构造），对比：
- catfish 旁观（现实现）：最弱原则靠 LLM 文本偶然抬升
- catfish 进 GNE（路 A）：最弱原则被 maximin 策略**保证**抬升
指标：min_k v（最弱原则终值）、共谋场景的底线违反率、FDBI。

## 四、风险与对策
- **风险 1**：catfish 全压最弱维度 → α_catfish 接近 one-hot，v 可能过冲 → 用软化的
  best response（温度参数）限制单轮抬升幅度
- **风险 2**：6 方 GNE 求解稳定性 → 数值验证（现有 5 方 KKT 1e-7，6 方需复测）
- **风险 3**：审稿人问"catfish 也是自利方怎么办" → 明确它是**守护方**（satisfaction
  指向集体最弱，非个人偏好），这是设计声明非隐藏假设

## 五、替代（若 A 数值不稳定）
路 B 仍开放：放弃"打破共谋"叙事，catfish 只报实证（w/o Catfish 消融轮次↓仲裁↓
满意 min↓ + process_trace 异议消费率）。A 优先，A 失败回 B。
