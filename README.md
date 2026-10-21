# insight-agent

[![CI](https://github.com/cheneryang0616-bot/review-insight/actions/workflows/ci.yml/badge.svg)](https://github.com/cheneryang0616-bot/review-insight/actions/workflows/ci.yml)

> 一个骨架，三个场景的分析 Agent：**美团团购评论归因**、**信用风险建模与合规**，
> 以及 **设备健康与维护决策 PHM（Stage 1）**。
> 自然语言提问 → 领域工具集 → **多轮闭环（观察→修复/反问）** → 带出处的结论
> → 数字回查 + 业务断言双重质检。

```bash
python scripts/ask.py "为什么差评这么多？短板在哪？"            # 场景一
python scripts/ask.py "模型效果怎么样？" --domain credit        # 场景二
python scripts/ask.py "能不能用这个预测区间安排停机检修？" --domain phm --no-llm
```

PHM 已接通 8 个工具与 6 条工业闸门，支持单机健康、通道代理、代价情景和维护候选排期。真实预测的名义覆盖率为 95%，
实测仅 **56%**，因此明确拒绝据此安排停机；CLI 退出码 **2** 表示预期的业务阻断。
见 [Stage 1 验收报告](results/REPORT_phm.md) 与 [设计说明](docs/PHM_DESIGN.md)。

**它不是「让 LLM 看一眼数据然后写分析」**，而是：

> LLM 从头到尾看不到原始数据，只能读工具算完的结果；
> 算不出来时不会硬答，而是**放宽口径重算或反问用户**；
> 答案里每个数字都要回查到工具输出（查不到就重写或删句）；
> 每个领域再挂一组**可以机器判定的业务断言**（风控里是 KS 下限、禁忌字段、分箱单调性）。

---

## 为什么做成"一个框架 + 多个场景"

单独一个评论分析 Agent，或者单独一个风控 Agent，都只是一个项目。
不同领域的**失败形态不同**：

| | 场景一：评论归因 | 场景二：信用风险 | 场景三：PHM（Stage 1） |
|---|---|---|---|
| 错误长什么样 | 数字算错、把子集当全量 | **数字全对，但结论违反硬约束**（用了性别字段） | 区间过窄却用于停机决策 |
| 质检靠什么 | 数字回查 + 样本量门槛 | 数字回查 + **断言闸门**（KS/IV/单调性/合规字段） | 数字回查 + **覆盖率准入闸门** |
| 分析原语 | 负面表述率、log-odds 关键词、趋势异动 | WOE/IV 分箱、评分卡、KS/AUC、通过率权衡 | 时序预测 + 代价阈值 + 提前期候选排期 |

领域注册：`phm`：设备健康与维护决策（时序 + 决策优化；Stage 1 八工具与六闸门已完成）。

所以真正可复用的是**骨架**（规划→执行→合成→质检→修复），不是分析逻辑。
三个领域复用同一套规划、闭环、数字回查与闸门机制。

| 层 | 代码位置 | 是否领域无关 |
|---|---|---|
| 工具注册 + 参数校验 | `insight/core/registry.py` | ✅ |
| 通用规划器（意图表由领域注入） | `insight/core/planner.py` | ✅ |
| 编排闭环（执行→观察→修复/反问→合成→质检） | `insight/core/agent.py` | ✅ |
| 数字回查（grounding） | `insight/core/grounding.py` | ✅ |
| **断言闸门（gates）** | `insight/core/gates.py` | ✅ 机制通用，断言由领域定义 |
| 领域数据/统计/工具/模板/图表 | `insight/domains/<domain>/` | ❌ 各写各的 |

---

## 三层质量保障

**第 1 层：数字回查（通用）**
工具返回的 `facts` + `table` 是唯一合法证据池；答案里每个数字都要能查到（1% 相对误差）。
容差按答案的小数位数放宽，因为 `2.3` 相对 `2.2626` 是**取整**而不是编造 ——
不做这个区分，质检会把可信答案也拦下来，比不做更糟。

**第 2 层：断言闸门（领域）**

场景二配置了 8 条：

| 闸门 | 等级 | 断言 |
|---|---|---|
| `ks_floor` | ⛔ 阻断 | KS ≥ 0.3，否则模型不可交付 |
| `fair_lending` | ⛔ 阻断 | 模型不得使用性别/婚姻、外籍劳工字段 |
| `cutoff_monotone` | ⛔ 阻断 | 通过率越低，坏账率必须越低 |
| `iv_ceiling` | ⚠️ 告警 | IV > 0.5 先怀疑泄漏/极端值，别急着夸变量强 |
| `woe_monotonicity` | ⚠️ 告警 | WOE 分箱应单调，否则手工合并 |
| `sensitive_fields` | ⚠️ 告警 | 年龄等敏感字段入模需说明理由 |
| `forbidden_in_data` | ⚠️ 告警 | 数据集含禁忌字段，建模前必须处理 |
| `imbalance` | ⚠️ 告警 | 坏样本占比 < 2% 需重采样 |

场景一配置了数据时效性（"数据止于 2021-09，距今 1866 天"）与最小样本量两条。

**第 3 层：评测集**
每个领域一套问题（带期望工具、期望事实、期望切片、期望闸门状态），跑回归。

---

## 多轮闭环：失败之后会怎么办

这一层是后来补的，因为**单轮版本在这些场景上会给出「看起来正常」的错误答案**。
我先把这些故障固化成用例（`scripts/run_loop_eval.py`）并跑基线：

| 场景 | 单轮版本的行为 | 严重性 |
|---|---|---|
| 意图没听懂 | 静默返回「数据整体情况」 | 最高：用户以为得到了答案 |
| 必填参数缺失 | 工具抛异常，但答案照出 | 中 |
| 实体存在但规则过度改写工具链 | 问「某店的套餐排名」，答成「该店星级分布」 | 中 |
| 切片样本不足 | 只说「样本不足」就结束 | 中 |

**基线 4/9 → 闭环 9/9。** 三个修复动作 + 一次反问：

- **反问（clarification）**：缺必填参数、或意图完全没命中时，返回能力清单与候选，
  并明确说明「本轮未产出结论，所以不做质检」。这比猜一个答案诚实。
- **放宽切片**：从最窄条件开始去掉（年份 → 套餐 → 店铺），重算，
  **并在答案开头写明「已去掉 X 重新计算，以下结论适用于调整后的范围」** ——
  放宽口径不告诉用户，等于偷偷换问题。
- **剔除失败工具**：某个工具抛异常时，用剩下的结果继续，而不是把异常写进答案。

## 实测结果

### 场景一：美团团购评论（42,038 条，21 家店，2014–2021）

| 结论 | 数字 |
|---|---|
| 总体差评率 | **5.38%**（2,262 / 42,038） |
| 代金券 vs 套餐 | **7.39% vs 3.92%**（1.89 倍） |
| 同上，仅 2020 年 | **2.89% vs 3.16%** —— 差距反转，**是历史现象不是当前问题** |
| 第一短板（绝对量） | 口味出品：差评负面率 **21.13%** vs 好评 2.21% |
| 最强信号（区分度） | 服务体验：**6.14% vs 0.25%**（**24.5 倍**） |
| 食品安全信号 | 「拉肚子」差评 137 次 / 好评 61 次（z=17.02） |
| 门店 vs 时段 | 门店极差 7.0pt，星期极差仅 1.1pt |

### 场景二：信用风险（German Credit 1,000 条 / Give Me Some Credit 150,000 条）

| 结论 | 数字 |
|---|---|
| 违约率与不平衡 | 30.00%（German）/ **6.68%（1:13）**（GMC） |
| 变量异常 | GMC：**4 个变量 IV > 0.5**，其中额度使用率 **IV=1.11** 且非单调 → 先查极端值 |
| 模型表现 | German：AUC **0.7709** / KS **0.4202**；GMC 全量：AUC **0.7995** / KS **0.4482** |
| 阈值权衡 | German：通过率 90% → 坏账 25.19%；压到 50% → 坏账 13.33%，捕获 77.78% 坏客户 |
| **合规拦截** | 用默认模型问"效果怎么样" → **⛔ 阻断**：模型用了 `sex_marital` / `foreign_worker` |
| **修复后** | 剔除后 20 → 17 个特征，KS 0.4202 → **0.4271**，AUC 0.7709 → 0.7694 → **几乎不损失精度** |

最后一行是这个项目最想说的：**去掉歧视性变量几乎不损失区分度**，
把"合规"从口号变成了一个可执行的、有代价数字的取舍。

### 评测（`scripts/run_eval.py`）

| 领域 | 离线（规则规划 + 模板） | LLM 规划 + LLM 合成 |
|---|---|---|
| review（15 题） | **15/15** | 12/15（多次 10~12） |
| credit（German，9 题） | **9/9** | 6/9 |
| credit（GMC 抽样，8 题） | **8/8** | — |
| phm（Stage 1，12 题） | **12/12**（含 3 条预期业务阻断） | 未通过在线验收 |

> CI 里只有脱敏样例：`review` 15 条全部跳过（需要完整评论数据），
> `phm` 跳过 7 条（需要原始传感器序列 / model report），其余照常执行。
> **跳过而不是"假装通过"**——脚本会打印跳过清单，退出码仍为 0。

LLM 规划器的失败分三类，全部是跑出来的：

1. **漏工具**：问"数据整体情况"只调了 `dataset_overview`，漏掉星级分布 / IV 排序。
2. **粒度漂移**：问"哪个时段差评最多"自己选了按小时（22:00），而期望按星期。
3. **编数字**：写 `86.41%`（真实 68.50%）、`167`、`90` —— 被 groundedness 拦下。

最严重的一类出现在风控：**问"合规吗"，LLM 规划器没有调用 `forbidden_check`，
闸门因此根本没被触发**。指标的错可以查出来，**该做的检查没做查不出来** ——
所以合规、审批、发布这类强约束路径必须走规则规划，不能让模型自己决定要不要检查。

结论：**规则选工具、LLM 写解读**。

单元测试：**框架 22 + 评论 37 + 信贷 14 + 闭环 11 + PHM 104 = 188 项**；
2026-10-21 实测 184 项通过、4 项因缺少 German Credit 数据跳过，无失败。
闭环回归 **12/12**，其中 PHM 3/3（缺机器编号、领域外意图、序列不足）。

**资产指纹**：每轮回答的分析范围里都带着预测文件指纹（如 `npz:326dddb4`，
含 sha256 前 8 位 / 字节数 / mtime），并进入工具的 `facts` 与 `source_manifest` 交叉校验。
换一份 npz 指纹必变 —— 报告里写“运行 FD001_xxx_final”不再是一句空话，而是可核对的事实。

PHM 主验收问句“哪几台两周内必须修”：机器 34、81 成为候选，20/20 个数字回查通过。
默认备件 14 天到时两台均来不及，lead_time_gate 阻断；追问提前期改成 7 天后，
该约束通过。所有输出均是人工复核候选，不授予生产停机权限。

CI（`.github/workflows/ci.yml`）在 Python 3.9 / 3.11 双矩阵上跑：单测 + 三个领域评测 +
闭环评测 + 图表渲染冒烟，全程离线、无 API Key、无私有资产。
完整的 PHM Stage 1 验收需要原始序列与 model report，用 `scripts/verify_phm_stage1.py`
在本地执行（CI 里跑不了，也不该跑）。

PHM 的当前保证限于离线规则规划与模板合成。在线 LLM 路径存在漏必审工具的已知缺口，
尚未通过在线准入验收；不要把提示词或合成后闸门报告当成强制语义拦截。

---

## 快速开始

```bash
pip install -r requirements.txt

# 数据：仓库只带脱敏样例（见 data/README.md），完整数据自行放置
python3 scripts/ask.py "代金券和套餐哪个差评更多？"                    # 场景一
python3 scripts/ask.py "数据整体什么情况？" --domain credit            # 场景二
RI_CREDIT_DATA=german python3 scripts/ask.py "合规吗？" --domain credit

# 用真实 LLM（需要 DEEPSEEK_API_KEY，或写进 ~/.agentdoc.env）
python3 scripts/ask.py "为什么差评这么多？" --llm

# 图表与 CI 冒烟
python3 scripts/make_phm_figures.py                     # 生成 4 张图并嵌入报告
python3 scripts/make_phm_figures.py --out-dir /tmp/figs # 只渲染（CI 用的就是这个）

# 评测 / 测试 / 报告
python3 scripts/run_eval.py --domain review
python3 scripts/run_eval.py --domain credit
python3 scripts/run_eval.py --domain phm     # Stage 1：9 题，覆盖全部7意图
python3 scripts/run_loop_eval.py            # 自修正用例：失败之后的行为
python3 -m unittest discover -s tests -v
python3 scripts/build_report.py --domain review
RI_CREDIT_DATA=german python3 scripts/build_report.py --domain credit
```

PHM 默认读取外部 C-MAPSS 仓库，不复制 artifacts。可设置：

```bash
export RI_PHM_ROOT=/Users/zhuan/c-mapss_train
python3 scripts/ask.py "能不能用这个预测区间安排停机检修？" --domain phm --no-llm
# 本例退出码为 2：数字回查通过，但覆盖率闸门阻断
python3 scripts/make_phm_sample.py --root "$RI_PHM_ROOT"
# 一条命令复现全部 Stage 1 验收、回归测试及报告
python3 scripts/verify_phm_stage1.py --root "$RI_PHM_ROOT"
python3 scripts/ask.py "哪几台两周内必须修，备件7天到" --domain phm --no-llm
```

未设置 `RI_PHM_ROOT` 且默认资产目录不存在时，自动使用仓库自带的
`data/sample_phm_units.csv`（FD001 全部 100 台机器的真实预测摘要），独立 clone 也能运行。
显式配置错误会报错。摘要没有原始序列和模型报告，相关工具会返回 insufficient；
缺原始资产时 pipeline_selfcheck 阻断准入，样例只能做受限演示。
完整验收需要原始资产；默认每天1循环、提前期14天、维护窗口14天和成本比10:1均为示意假设，
可在问句中覆盖。RUL 循环数必须经显式速率换算，不能直接当成天。

示例输出（离线，数字全部可回查）：

```
> 领域：credit｜规划器：rule｜工具链：forbidden_check → perf_metrics｜合成器：template

- 合规检查：数据含 2 个硬阻断字段，其中 2 个已进入模型 —— 不可用，必须先剔除再建模。[T1]
### 结论
1. 模型当前不可用：它使用了硬阻断字段，指标再好在合规上也过不去，应先剔除再评估。[T1]

✅ 数字回查通过：12/12 个数字可追溯到工具输出
⛔ fair_lending：使用了禁忌字段：['foreign_worker', 'sex_marital']
```

---

## 项目结构

```
insight/
  core/                    # 框架（领域无关）
    registry.py  planner.py  agent.py  grounding.py  gates.py  llm.py  charts.py
  domains/
    review/                # 场景一：评论归因
      config.py data.py lexicon.py stats.py tools.py planner_spec.py bullets.py report.py eval/
    credit/                # 场景二：信用风险
      config.py data.py stats.py tools.py planner_spec.py bullets.py report.py eval/
    phm/                   # 场景三：设备健康（Stage 1）
      config.py data.py decision.py tools.py planner_spec.py bullets.py eval/
scripts/     ask.py / run_eval.py / build_report.py / make_samples.py
             make_phm_sample.py / verify_phm_stage1.py
tests/       test_core.py（22） / test_review.py（37） / test_credit.py（14） / test_loop.py（11） / test_phm*.py（95）
data/        sample_reviews.csv / sample_credit.csv / sample_phm_units.csv + README.md
figures/     review/（6 张） credit/（6 张）
results/     REPORT_review.md / REPORT_credit.md / REPORT_phm.md / phm_stage1.json / phm_phase0.json / eval_*.json
```

---

## 踩过的坑（都写进注释了）

**框架层**
1. **只有追问才继承上下文**。上一轮问"2020 年怎么样"，下一轮问"为什么差评多"，
   模型会忠实沿用 2020 —— 答案写"全量"，数字却是子集（服务倍差从 24.5 变 94.74）。
   修法不是让模型自觉，而是规则层判断"指代/省略式短问"才算追问。
2. **质检容差要按小数位放宽**，否则把取整当编造。
3. **引用原文与标签里的数字不参与回查**：「等了 20 分钟」是证据不是结论；
   `P99`、`Time30-59Days` 里的数字是字段名的一部分。
4. **百分数要按两种刻度入池**：表格里是 `6.42%`，facts 里是 `0.0642`。
5. **意图没命中时不要用默认序列硬答**：第一版遇到听不懂的问题会返回「数据整体情况」，
   用户看不出异常 —— 这比报错更危险。现在改成反问 + 能力清单。
6. **放宽口径必须写进答案**：否则用户看到「全站排名」却以为还是他问的那家店。
7. **澄清轮要跳过数字回查**：能力清单里有「PDO 50」这种示例数字，会被当成编造。

**场景一**
8. **方面归因第一版完全失效**：只用中性词（"味道""服务"）算提及率，
   差评与好评几乎一样（口味 61.8% vs 71.7%）—— 因为差评常说"锅底好吃但服务太差"。
   改成**负面表述率**才有区分度（服务 6.14% vs 0.25%）。
9. **通用词"套餐"不能当实体**：一旦当套餐名过滤，差评率最高的代金券类会被整体排除。
10. **有店铺实体时不能用 `shop_ranking`**：它是跨店排名，吃不到 shop 参数。

**场景二**
11. **German Credit 的标签是 1=好客户 / 0=坏客户**（不是 UCI 文档写的 1/2）。
   写错会导致坏样本数为 0、IV 全部变成 `inf` —— 有单测守着。
12. **通过率与坏账率的单调性方向容易判反**：通过率下降时坏账率应下降。
13. **合规修复要换整条链路**：只把模型换成"剔除版"，但阈值曲线还用旧模型，
    会出现指标与曲线口径不一致。

---

## 局限

- 两份信贷数据**都没有时间列**：只能随机切分，做不了 OOT 验证与跨期 PSI。报告里不会写"做了时间外验证"。
- 方面词典（场景一）是人工的，换品类必须重建；方面之间不正交。
- WOE 分箱用等频 10 箱，非单调变量需要手工合并，本项目没有自动合并策略。
- 全部是相关性/预测，不含因果；策略上线前需要 A/B。
- 质检对 ≤12 的整数与年份有意放宽，代价是"有 3 家店"这类小数字幻觉抓不到。

---

## 数据与合规

- 评论数据来自开源仓库 `wq624915051/meituan_text_minning`（无 LICENSE，仅个人研究使用）；
  信贷数据为 Kaggle Give Me Some Credit 与 UCI German Credit。
- **完整数据都不进仓库**：评论表含 27,686 个用户的昵称与 ID（个人信息），
  信贷数据受竞赛规则约束。仓库只放脱敏样例（`data/README.md` 说明获取方式）。
- 所有结论仅基于公开数据集，**不代表任何公司真实业务数据**。
