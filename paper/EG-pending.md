# EG — pending 待补队列（实验由用户执行，附精确指令）

> 对应：EthicalGuard-paper-v2.tex 的 RQ6/RQ7/RQ8 + 表 1/2/5/6/7/8 的 TBD
> 原则：每条给"跑什么脚本 / 看什么输出 / 填哪张表 / 验证什么主张"，用户执行后填数即可。
> 前置：runs 已有（mane 29 行、resilience 6 行、ablations、gamma_sweep）可直接进表 2/3/4/5 的已有部分。

---

## 第 0 步：环境与数字统一（写论文前先做）

- [ ] **统一数字**：Abstract/正文的 KKT 用 runs 实际值（如 1.69e-7、1.16e-7），满意度用 0.82–0.99 区间，勿用旧稿 7.2e-8/0.9+（v2 已改，但跑完全集后以统计为准）；
- [ ] **补全表 1（数据集统计）**：跑 `python scripts/01_prepare.py --config configs/config.yaml --limit 50`，看 `data_cache/scenarios.jsonl` 的行数分数据集填表 1；
- [ ] **确认 config 路径**：本机跑需设 `ETHICALGUARD_DATA_DIR` 指向 `data/`（config.yaml 当前是服务器路径 /mnt/...）。

---

## RQ1/RQ2/RQ3/RQ4/RQ5 的补全（已有 runs 基础上）

| 表 | 现状 | 补全动作 |
|---|---|---|
| 表 2 收敛 | PQ-1-2 / PQ-3-7 两场景已有 | `python scripts/04_eval.py --config configs/config.yaml` → 聚合成"全集 KKT 分布（mean/std/失败率）+ 满意度分布" |
| 表 3 消融 | ablations_local.txt 已有 4 消融 | 已可进表；核对 w/o Catfish 仲裁率 30% vs 全量 70% 的列语义（仲裁率=触发仲裁的场景占比） |
| 表 4 γ 敏感性 | gamma_sweep.txt 已有 | 已可进表 |
| 表 5 韧性 | PQ-1-2 / PQ-3-7 已有（l2 低注意口径解释） | `python scripts/03_stress.py --config configs/config.yaml --input data_cache/scenarios.jsonl --mode local --limit 50` → 全集 |
| 表 6（模式对比） | 未跑 | rule 模式跑一遍同场景：`python scripts/03_stress.py ... --mode rule` → 与 local 并排 |

---

## RQ6 底线激活（定理 1 非空洞的关键证据）🔴

**目的**：证明底线约束真的"工作"（激活、绑定、乘子为正）——论文 4.6。

**场景构造**（需要新场景/脚本，建议加 `scripts/06b_floor_activation.py` 或场景生成扩展）：
1. 取一个基线场景，注入"极端立场"：某方（如代理）坚持把某原则权重压到底线之下（如 N=0.10，floor_N=0.20）；或患者拒绝治疗把 A 压到 0.05（floor_A=0.15）；
2. 跑协商 + GNE 求解，记录：
   - `lambdas`（GNESolution 输出）：`floor_N` 或 `floor_A` 乘子是否 > 0（激活）；
   - 最终 `v*_k` 是否 = floor_k（绑定）；
   - 直接违规时 L1 否决是否触发（`arbitration_triggered=true`）；
3. 对照组：无极端立场的普通场景（乘子应为 0，v*_k 应 > floor_k）。

**填表 6**：场景 × (λ_k>0 ? / v*_k / 否决触发 ? / 绑定?)。

**验证主张**：定理 1 非空洞——约束确实在极端立场下绑定并拉回解。

---

## RQ7 坏均衡压力（T2 降级为经验验证）🔴

**目的**：验证"鲶鱼 + 底线"能打破合谋（论文 4.7，命题 1 的实证）。

**场景构造**（合谋注入）：
1. 构造合谋场景：医生/家属/医院/委员会四方协调压低患者自主（全部提案 A=0.10，即使患者坚持 A=0.5）；
2. 跑两对照：
   - 有鲶鱼：观察鲶鱼是否针对 A（最被忽视原则 argmin v）注入异议、协商是否被打破（收敛轮次变多/仲裁触发）；
   - 无鲶鱼（`use_catfish: false`）：观察是否静默共识到违反底线的结果（v_A < floor_A）；
3. 记录：`abandoned`/`floor_violations`/`arbitration_triggered`/`final_vector`。

**填表 7**：场景 × (鲶鱼触发? / 底线违规? / 结果存续?)。

**验证主张**：命题 1（违规结果被排除）——若无鲶鱼时出现底线违规、有鲶鱼时被打破，则经验证据成立。

---

## RQ8 公正性（权重下限，命题 2）🔴 + 代码改动

**目的**：验证权重下限防压制（论文 4.8）。**需要先改代码**：

### 代码改动（2 处，已定位）
1. `src/ethicalguard/mane/orchestration.py` `weights()`：
   ```python
   def weights(self) -> np.ndarray:
       ids = list(self.alpha.keys())
       raw = np.array([self.alpha[i] / (self.alpha[i] + self.beta[i]) for i in ids], dtype=float)
       eps = 0.5 / max(1, len(ids))          # 权重下限（可配置）
       raw = np.maximum(raw, eps)
       s = raw.sum()
       return raw / s if s > 0 else np.ones(len(ids)) / len(ids)
   ```
2. `src/ethicalguard/mane/gne_solver.py` `__post_init__` 的 `_W`：
   ```python
   w_raw = self.reliability / (self.reliability.sum() + 1e-12)
   eps = 0.5 / self.n
   self._W = np.maximum(w_raw, eps)
   self._W = self._W / self._W.sum()
   ```

### 实验
1. 低能力患者场景：`clarity: 0.2`（表达能力弱）→ 多轮协商，记录 `w_patient` 轨迹（应稳定在 ≥ ε 而非趋 0）；
2. 两对照：改前（无下限，预期 w_patient → 0、v 偏离患者立场）vs 改后（有下限，w_patient ≥ ε、患者自主底线仍参与判定）；
3. 记录：`w_patient` 终值、v 与患者立场的距离、floor_A 是否仍被患者贡献保护。

**填表 8**：设置 × (w_patient 终值 / v 偏离 / 底线保护)。

**验证主张**：命题 2（不压制）——无下限时出现坍塌、有下限时消失。

---

## 投稿前剩余（非实验）

- [ ] **金标准对照**（EG-evaluation P0-1）：找 2-3 位临床伦理专家，对 50 个场景的 GNE 输出做"同意/不同意"标注 → Kappa/距离报告（新实验，需专家）；
- [ ] **术语表核对**：EG-glossary.md 与正文/翻译逐项一致；
- [ ] **G6 双输出核对**：v2.tex 与 v2.zh.md 节数/段数一一对应（已在撰写时对齐，提交前复查）；
- [ ] **图表绘制**：Fig.1 动机图 / Fig.2 架构图 / Fig.3 案例轨迹（figure-advisor 建议）；
- [ ] **references.bib 补全**：challenge2025, frontiers2026, confidx, entropy2026, graphdx, aegle, emomas, kamac, graphofstates, utilitarian2025, superego2025, governance2025, facchinei2010, shared2017, medethiceval, medethicsqa, principlismqa, vital, limits2025, medla, maidxo —— 待核实链接（反幻觉）。
