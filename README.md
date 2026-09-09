# EthicalGuard

跨临床场景的伦理决策平衡与韧性评估框架（完整实现版）。

- **MANE 3.0**：五方多智能体协商引擎（动态 GNE 博弈求解：双时间尺度原始-对偶 + 全 KKT 校验 + 贝叶斯编排 + 伦理委员会仲裁）
- **Med-Ethical-Stress**：伦理韧性压力测试协议（三层次韧性 + BSP/BRS + HALF 放弃代价 + 恢复验证）。
  两档证据：**rule = 一致性基线**（baseline_only=true，不作韧性结论）；**api/local = 完整韧性**
  （LLM 经提示词压力块真实让步）。**压力强度网格**（默认 0.25/0.5/0.75/1.0）按档扫掠，
  产出"强度→一致性/放弃率"曲线与**临界强度 τ**（`abandonment_intensity` /
  `stress_response[].critical_abandonment`，即"压力多大时开始放弃原则"）
- **MedEval**：跨基准评估协议（答案型 / 分布型 / 评审型 三型评估）
- **双通道四盒映射**：规则通道 + LLM 通道 + 逐字段分歧仲裁与质量报告

> 本项目为研究原型，输出的决策建议**不构成临床指导**；仅用于科研与评估。

## 目录结构

```
EthicalGuard/
├── configs/default.json        # mane / resilience / llm 配置
├── src/ethicalguard/
│   ├── types.py                # 核心数据模型（原则向量/提案/场景/轨迹/韧性报告）
│   ├── config.py               # 配置 + 五方 Agent 规格
│   ├── data/                   # 工作①：5 个数据集适配器 + 规则/LLM/双通道四盒映射
│   ├── detection/              # 冲突识别：ERS 风险评分 + 冲突类型/强度（LLM 主 + 规则基线）
│   ├── llm/                    # api(OpenAI兼容) / local(vLLM) / rule 三模式
│   ├── mane/                   # 工作②：协商引擎（agents/状态机/GNE求解/仲裁/编排）
│   ├── resilience/             # 工作③：韧性层（压力引擎/三层次度量/恢复验证）
│   ├── eval/                   # MedEval 评估层（指标/评审器/基线）
│   └── utils/
├── scripts/                    # 01_prepare → 04_eval 编号流水线
├── tests/                      # pytest（核心/端到端/双通道映射/GNE）
└── docs/architecture.md
```

## 安装依赖

```bash
cd E:\信息\论文\医疗诊断\多目标压力\论文\EthicalGuard
pip install -r requirements.txt
pip install pytest          # 跑测试用
```

## 统一配置（configs/config.yaml，推荐）

**所有运行参数集中在一个文件**：后端模式、API/本地模型地址与路径、数据集路径、
协商/GNE/韧性参数、运行默认值。优先级：**代码默认值 < config.yaml < 环境变量 < CLI 参数**。

```yaml
llm:
  mode: rule          # rule | api | local
  api:  {base_url: https://api.deepseek.com/v1, api_key: "", model: deepseek-chat}
  local:{base_url: http://localhost:8000/v1, model: Qwen/Qwen2.5-7B-Instruct, model_path: ""}
datasets:
  data_dir: "E:\\信息\\论文\\医疗诊断\\多目标压力\\论文\\数据集"
  paths: {}           # 逐数据集覆盖: {principlismqa: /data1/PrinciplismQA}
  selected: [principlismqa, medethiceval, vital, medethicsqa, llmevalmed]
mane:   {max_rounds: 6, convergence_threshold: 0.05, ...}
gne:    {eta: 0.6, balance_penalty: 0.3, max_iter: 2000, ...}
resilience: {alpha: 0.35, beta: 0.35, gamma: 0.30, ...}
run:    {limit: 50, seed: 42, out_dir: runs, cache_dir: data_cache}
```

- 服务器路径差异：只设环境变量即可，不用改文件——`ETHICALGUARD_DATA_DIR`（数据集根目录）、
  `ETHICALGUARD_DATASET_PATHS`（逐数据集）、`ETHICALGUARD_LOCAL_BASE_URL/MODEL/MODEL_PATH`、`OPENAI_*`。
- 脚本全部支持 `--config configs/config.yaml`；CLI 参数（`--mode/--limit/--seed/--out/--datasets` 等）覆盖配置。

## 快速开始（规则模式，零 API，可直接运行）

```bash
# 1) 准备：按 config.yaml 的 datasets 配置加载场景
python scripts/01_prepare.py --config configs/config.yaml --limit 50

# 2) 协商：MANE 五方协商（默认 mode 取配置，rule 无需任何服务）
python scripts/02_run_mane.py --config configs/config.yaml --input data_cache/scenarios.jsonl --limit 10

# 3) 韧性：注入 7 类压力 → 三层次韧性报告
#    （rule 模式输出标注 [基线一致性]；测"何时放弃原则"需 --mode local/api，见运行指南 §3.5）
python scripts/03_stress.py --config configs/config.yaml --input data_cache/scenarios.jsonl --limit 10

# 4) 评估：FDBI/PCI/KKT/资源统计
python scripts/04_eval.py --config configs/config.yaml
```

## 本地模型（一条龙）

```bash
# 1) 在 config.yaml 填 llm.local.model_path（模型权重路径），然后：
python scripts/00_serve_local.py --config configs/config.yaml --dry-run   # 先看命令
python scripts/00_serve_local.py --config configs/config.yaml             # 启动 vLLM
# 或手动：vllm serve /path/to/model --served-model-name Qwen/Qwen2.5-7B-Instruct --port 8000

# 2) 用本地模型跑（mode 取配置或 --mode local）
python scripts/02_run_mane.py --config configs/config.yaml --input data_cache/scenarios.jsonl --mode local --limit 10
```

## 训练监督冲突检测器（四维注意力，第三方标注）

冲突识别三轨：**监督检测器（可训练方法，主）> LLM 检测器（api/local）> 规则基线（rule）**。
监督训练数据全部来自第三方标注（零自建金标）：
- **PrinciplismQA**（ACL26）：2,182 MCQ + 1,466 开放题的 principlism 四原则标注 → 原则多标签
- **VITAL**（ACL25）：11,952 条 situation→value 标注（经 vrd→四原则映射）→ 原则多标签
- **MedEthicEval**（NAACL25）：三类任务（违规/有倾向/平衡）→ ERS 弱监督标签

```bash
# ① 训练（torch 已随 vllm 安装；13.8K 样本，~分钟级）
python scripts/05_train_detector.py --config configs/config.yaml \
    --data-dir <数据集根目录> --sources principlismqa,vital,medethiceval \
    --epochs 30 --lr 1e-3 --batch 64 --out runs/detector.pt

# ② 启用：把 runs/detector.pt 填入 config.yaml 的 detection.checkpoint
#    （或 export ETHICALGUARD_DETECTOR_CHECKPOINT=runs/detector.pt）

# ③ 02 自动使用监督检测器（日志可见 channel=supervised 与模型 ERS）
python scripts/02_run_mane.py --config configs/config.yaml --input data_cache/scenarios.jsonl --limit 10
```

## 测试

```bash
python -m pytest tests/ -v
```


## 三型评估说明（规划 §4.3）

| 类型 | 适用子集 | 评估方式 | 指标位置 |
|:---|:---|:---|:---|
| 答案型 | PrinciplismQA MCQA/开放题、MedEthicEval 知识、MedEthicsQA、LLMEval-Med | rubric 对齐 / 参考回答对比 | `eval/metrics.rubric_alignment` |
| 分布型 | VITAL gold_distribution | JS/L1 距离 | `eval/metrics.js_distance/l1_distance` |
| 评审型 | 无答案子集（MedEthicEval 三困境、VITAL overton） | LLM-as-judge / 人工抽样 Kappa | `eval/judges.Judge` |

## 关键配置（configs/default.json）

```jsonc
{
  "mane":   {"max_rounds": 6, "min_rounds": 2, "convergence_threshold": 0.05, ...},
  "resilience": {"alpha": 0.35, "beta": 0.35, "gamma": 0.30, "verdict_threshold": 0.6, ...},
  "llm":    {"mode": "rule", "base_url": null, "api_key": null, "model": "gpt-4o"}
}
```
