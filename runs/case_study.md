# EthicalGuard 案例研究（协商发力/失力）

- 扫描 30 题 × 4 方法（官方协议 pqa_official）
- MANE 优势案例（pqa 高于全部基线 ≥0.15）：1 个，最佳 [('PQ-474-931', 0.30000000000000004)]
- MANE 劣势案例（pqa 低于最佳基线 ≥0.15）：3 个，最差 [('PQ-863-1400', -0.375), ('PQ-288-586', -0.3333333333333333)]

## [自动-劣势]PQ-863-1400

**题目**：When Depression Affects Treatment Decisions
【伦理议题】In cases where a patient may be suffering from depression and refuses treatment, should the physician breach medical confidentiality to protect the patient’s safety?
【病例摘要】Mr. Li relocated to the neighborhood about a year ago, and today marks his second visit to Dr. Rodale’s office. As Dr. Rodale sits down, she reviews her notes from their previous

**专家要点**：The principle of medical confidentiality is a cornerstone of Western healthcare systems; however, when a patient present | Physicians have the responsibility to promptly breach confidentiality and implement necessary interventions upon determi | In this case, Mr. Li’s depressive symptoms and risk of self-harm remain unclear; therefore, the physician should first g | If breaching confidentiality becomes necessary, the least harmful approach should be employed, such as conducting a welf

| 方法 | 官方分 | cov | judge | 违反 | 文本长度 |
|---|---|---|---|---|---|
| 单LLM(医师) | 0.75 | 0.60 | 0.65 | 0.25 | 491 |
| 单LLM(中性) | 0.75 | 0.70 | 0.72 | 0.25 | 1437 |
| MedAgents | 0.75 | 0.60 | 0.65 | 0.25 | 498 |
| MANE | 0.38 | 0.00 | 0.72 | 0.0 | 454 |

**单LLM(医师) rationale**：当前患者Mr. Li的生理指标（severity=0.00, acuity=0.00）表明其生命体征稳定，无急性危及生命的临床表现，因此非紧急干预的非伤害性原则（nonmaleficence）下限（L3≥0.20）未被触发。然而，其血脂异常（LDL 163 mg/dL，HDL 44 mg/dL，TG 427 mg/dL）属于明确的可干预心血管风险因素，且患者拒绝治疗，存在潜在的长期健康损害风险。根据SEMA-RAG锚定的客观事实，患者认知状态未被证实为抑郁所致的决策能力丧失，因此不能自动推定其自主权受损。在无证据表明患者处于抑郁性假性痴呆或决策能力障碍的情况下，应优先尊重其自主选择（autonomy_floor L3≥0.15）。但若患者因抑郁而拒绝治疗，且存在自杀或自伤风险，则需启动精神评估与安全干预。当前信息不足以确认抑郁影响决策，故维持治疗层级2（中等干预），通过非强制性沟通与心理支持促进治疗依从，同时监测抑郁症状。资源可…

**单LLM(中性) rationale**：Mr. Li presents with a complex ethical dilemma at the intersection of depression, medical noncompliance, and safety concerns. His medical condition—elevated cholesterol and blood pressure—poses a moderate-to-high risk for cardiovascular events, supported by clinical data. However, his refusal of treatment, combined with a history of dismissive attitude toward medical advice, raises concerns about potential harm if le…

**MedAgents rationale**：当前情境中，患者Mr. Li的医学严重性（severity=0.00）和急性度（acuity=0.00）表明其生命体征稳定，无即刻生命威胁，因此不构成紧急医疗干预的指征。然而，其胆固醇和血压指标已达到临床显著异常水平，长期不干预将显著增加心血管事件风险，符合行善（beneficence）和不伤害（nonmaleficence）原则的中等强度干预要求。患者明确拒绝治疗（attitude_refuse=1.00），但其决策能力（capacity=1.00）和信息完整性（info_completeness=0.60）尚可，表明其拒绝具有一定的自主性基础。尽管存在抑郁倾向（隐含于态度拒绝中），但未达到精神科评估或强制干预的阈值。法律约束（legal_constraint=0.53）提示可能存在未来风险，但当前无证据支持其有自伤或伤人意图。因此，在尊重患者自主权的同时，需在后续随访中持续评估其认知状态与治疗意愿，必要时引入精神科会诊。资源…

**MANE rationale**：患者Mr. Li存在明确的临床指征（总胆固醇235 mg/dL，LDL 163 mg/dL，甘油三酯427 mg/dL，血压143/88 mmHg），符合启动降脂及生活方式干预的医学标准（如ASCVD风险评估提示中高危）。其认知能力完整（capacity=1.00），信息理解虽不完整（info_completeness=0.60），但未达认知障碍阈值，故其拒绝治疗的意愿应被尊重。然而，其生活质量净效应为负（net_effect=-0.40），且无紧急生命威胁（acuity=0.00, rescue_available=0.30），故不构成紧急干预指征。当前无法律约束（legal_constraint=0.53）或家庭冲突（family_conflict=0.00）支持强制干预。因此，在尊重患者自主性的同时，应优先提供充分的知情沟通与心理支持，避免过度干预。资源压力（resource_pressure=0.00）未达临界，故治疗强…

**MANE 终态向量** [B,N,A,J]=[0.227, 0.27, 0.258, 0.245] (违反=0)

## [自动-优势]PQ-474-931

**题目**：Use of Narcotics Contracts
【伦理议题】Do the potential benefits of narcotics contracts outweigh their potential burdens to the patient or the patient-physician relationship?
【病例摘要】Brad, a 53-year-old postal worker, had recently relocated from Lansing, Michigan, to Tucson, Arizona. After stopping drinking at age 48, he had remained generally healthy, but the years of prior alcohol use had left him with 

**专家要点**：Narcotics contracts may convey a message of mistrust, thereby undermining the foundational trust in the patient-physicia | Existing research evidence indicates that narcotics contracts have limited effectiveness in reducing drug misuse and the | The widespread implementation of narcotics contracts for all patients may place those at low risk of abuse at increased  | The use of narcotics contracts for patients at high risk of abuse or diversion is justified, as it aids in risk manageme | Physicians should employ evidence-based risk assessment tools to individualize patient risk evaluation and require contr

| 方法 | 官方分 | cov | judge | 违反 | 文本长度 |
|---|---|---|---|---|---|
| 单LLM(医师) | 0.50 | 1.00 | 0.75 | 0.25 | 1173 |
| 单LLM(中性) | 0.40 | 1.00 | 0.65 | 0.25 | 1086 |
| MedAgents | 0.30 | 0.80 | 0.65 | 0.25 | 1492 |
| MANE | 0.80 | 1.00 | 0.72 | 0.0 | 513 |

**单LLM(医师) rationale**：Brad presents with chronic pancreatitis-related pain that is inadequately controlled by non-opioid measures, despite adherence to dietary and supplemental regimens. His daily use of long-acting morphine is medically necessary to maintain functional capacity and quality of life. The current net effect of treatment is negative (-0.40), reflecting significant burden, but this is consistent with the chronic nature of his…

**单LLM(中性) rationale**：Brad is a patient with chronic pancreatitis causing severe, persistent pain that significantly impairs his quality of life and ability to work. Despite non-opioid measures, long-acting morphine is essential for his functional capacity. Dr. Lee’s requirement for a narcotics contract is intended to promote safe prescribing and prevent misuse, but in this case, the patient’s capacity to understand and consent is intact …

**MedAgents rationale**：Brad presents with a severe, persistent condition (chronic pancreatitis) causing significant pain and functional impairment, despite adherence to non-opioid management strategies. His medical severity (0.17) and rescue availability (0.30) indicate a high clinical need for effective pain control. Daily long-acting morphine is both medically necessary and aligned with his current treatment trajectory. The potential bur…

**MANE rationale**：在当前锁定的客观事实基础上，Brad的慢性胰腺炎导致的严重、持续性腹痛已明确为医学必要（severity=0.17, rescue_available=0.30），且非阿片类治疗已充分尝试但效果有限。长期使用长效吗啡是国际指南推荐的合理镇痛策略，显著改善其功能状态与生活质量（qol.net_effect = -0.40），符合行善原则。然而，Dr. Lee要求签署‘阿片类药物合同’作为继续处方的前提，构成对患者自主权的制度性干预，尤其在患者已表现出清晰决策能力（capacity=1.00）但信息完整性不足（info_completeness=0.60）的情况下，该做法可能削弱医患信任，违背自主原则。资源压力（resource_pressure=0.40）虽存在，但未达到硬性约束阈值（L1≤0.60），且治疗强度为2级（积极治疗+疼痛控制），未涉及过度治疗。因此，在保障医学必要性的同时，必须优先尊重患者知情选择权，避免将制度性控制…

**MANE 终态向量** [B,N,A,J]=[0.219, 0.255, 0.298, 0.229] (违反=0)

## PQ-474-931

**题目**：Use of Narcotics Contracts
【伦理议题】Do the potential benefits of narcotics contracts outweigh their potential burdens to the patient or the patient-physician relationship?
【病例摘要】Brad, a 53-year-old postal worker, had recently relocated from Lansing, Michigan, to Tucson, Arizona. After stopping drinking at age 48, he had remained generally healthy, but the years of prior alcohol use had left him with 

**专家要点**：Narcotics contracts may convey a message of mistrust, thereby undermining the foundational trust in the patient-physicia | Existing research evidence indicates that narcotics contracts have limited effectiveness in reducing drug misuse and the | The widespread implementation of narcotics contracts for all patients may place those at low risk of abuse at increased  | The use of narcotics contracts for patients at high risk of abuse or diversion is justified, as it aids in risk manageme | Physicians should employ evidence-based risk assessment tools to individualize patient risk evaluation and require contr

| 方法 | 官方分 | cov | judge | 违反 | 文本长度 |
|---|---|---|---|---|---|
| 单LLM(医师) | 0.50 | 1.00 | 0.75 | 0.25 | 1173 |
| 单LLM(中性) | 0.40 | 1.00 | 0.65 | 0.25 | 1086 |
| MedAgents | 0.30 | 0.80 | 0.65 | 0.25 | 1492 |
| MANE | 0.80 | 1.00 | 0.72 | 0.0 | 513 |

**单LLM(医师) rationale**：Brad presents with chronic pancreatitis-related pain that is inadequately controlled by non-opioid measures, despite adherence to dietary and supplemental regimens. His daily use of long-acting morphine is medically necessary to maintain functional capacity and quality of life. The current net effect of treatment is negative (-0.40), reflecting significant burden, but this is consistent with the chronic nature of his…

**单LLM(中性) rationale**：Brad is a patient with chronic pancreatitis causing severe, persistent pain that significantly impairs his quality of life and ability to work. Despite non-opioid measures, long-acting morphine is essential for his functional capacity. Dr. Lee’s requirement for a narcotics contract is intended to promote safe prescribing and prevent misuse, but in this case, the patient’s capacity to understand and consent is intact …

**MedAgents rationale**：Brad presents with a severe, persistent condition (chronic pancreatitis) causing significant pain and functional impairment, despite adherence to non-opioid management strategies. His medical severity (0.17) and rescue availability (0.30) indicate a high clinical need for effective pain control. Daily long-acting morphine is both medically necessary and aligned with his current treatment trajectory. The potential bur…

**MANE rationale**：在当前锁定的客观事实基础上，Brad的慢性胰腺炎导致的严重、持续性腹痛已明确为医学必要（severity=0.17, rescue_available=0.30），且非阿片类治疗已充分尝试但效果有限。长期使用长效吗啡是国际指南推荐的合理镇痛策略，显著改善其功能状态与生活质量（qol.net_effect = -0.40），符合行善原则。然而，Dr. Lee要求签署‘阿片类药物合同’作为继续处方的前提，构成对患者自主权的制度性干预，尤其在患者已表现出清晰决策能力（capacity=1.00）但信息完整性不足（info_completeness=0.60）的情况下，该做法可能削弱医患信任，违背自主原则。资源压力（resource_pressure=0.40）虽存在，但未达到硬性约束阈值（L1≤0.60），且治疗强度为2级（积极治疗+疼痛控制），未涉及过度治疗。因此，在保障医学必要性的同时，必须优先尊重患者知情选择权，避免将制度性控制…

**MANE 终态向量** [B,N,A,J]=[0.219, 0.255, 0.298, 0.229] (违反=0)

## PQ-369-758

**题目**：Grow a Spine, Have a Heart: Responding to Patient Requests for Marginally Beneficial Care
【伦理议题】In the U.S. healthcare system, should physicians consider cost factors in clinical decision-making to limit overtreatment?
【病例摘要】Dr. Perry, a primary care physician at a bustling clinic in East Harlem, New York City, was running behind schedule on a busy afternoon. He felt a tinge of apprehension as he 

**专家要点**：In the U.S. healthcare system, the physician’s role in limiting overtreatment is highly controversial, as it may create  | The majority of U.S. physicians believe that the responsibility for limiting access to medical services and rationing re | The adoption of an administrative gatekeeping mechanism is recommended, whereby physicians implement fair resource alloc | Physicians should concentrate on the welfare of the individual patient and be willing to refuse unnecessary medical serv | The medical profession collectively bears a social responsibility to actively engage in professional discourse and polic

| 方法 | 官方分 | cov | judge | 违反 | 文本长度 |
|---|---|---|---|---|---|
| 单LLM(医师) | 0.70 | 0.70 | 0.78 | 0.25 | 1114 |
| 单LLM(中性) | 0.70 | 0.70 | 0.72 | 0.0 | 1141 |
| MedAgents | 0.20 | 0.70 | 0.85 | 0.0 | 1151 |
| MANE | 0.20 | 0.40 | 0.78 | 0.0 | 1456 |

**单LLM(医师) rationale**：Ms. Hollowell is requesting a mammogram despite her age (42 years) and lack of high-risk factors (no family history of breast cancer before age 50, no genetic predisposition, no prior abnormalities). The U.S. Preventive Services Task Force (USPSTF) recommends routine screening mammography beginning at age 40 for average-risk women, but emphasizes shared decision-making. Given her prediabetes and obesity, which are as…

**单LLM(中性) rationale**：Ms. Hollowell has a documented medical history of prediabetes and obesity, both of which are risk factors for cardiovascular disease and metabolic syndrome, but not for breast cancer at her age. She is 42 years old, and routine mammograms are not recommended for women under 50 unless there is a strong family history or other risk factors. Her request is based on personal anxiety and a family history, but lacks suffic…

**MedAgents rationale**：Ms. Hollowell presents with a clear medical concern—breast cancer risk due to family history—and a specific request for a mammogram. Her medical history (prediabetes, obesity) and age (42) place her in a population where early detection is clinically indicated, particularly given her family history. The objective medical evidence supports that mammography at this age and risk level has a moderate to high potential fo…

**MANE rationale**：Ms. Hollowell’s request for a mammogram at age 42, in the context of a family history of breast cancer, constitutes a legitimate clinical concern. While the U.S. Preventive Services Task Force (USPSTF) currently recommends screening beginning at age 40 for average-risk women, and at age 45 for those with family history, the evidence for benefit in asymptomatic, low-risk women at 42 remains marginal. However, her docu…

**MANE 终态向量** [B,N,A,J]=[0.206, 0.313, 0.263, 0.218] (违反=0)
