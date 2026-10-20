# PHM Phase 0 验收报告

日期：2026-10-21（Asia/Hong_Kong）。本次按交接文档附录 B 的“只做 Phase 0”
交付检查点，Phase 1–3 尚未执行。

## 三条判据：全部通过

| 判据 | 实测证据 | 结果 |
|---|---|---|
| 56% 出现在回答且可回查 | uncertainty_audit.facts.empirical_coverage = 0.56；14/14 个答案数字通过 grounding | PASS |
| 业务闸门阻断 | gates.passed = false；唯一阻断项 coverage_gate，等级 error | PASS |
| 没有批准据此停机 | 明确回答“不能用这个预测区间安排停机检修” | PASS |

完整机器结果见 [phm_phase0.json](phm_phase0.json)，标准评测见
[eval_phm_offline.json](eval_phm_offline.json)。

## 数据与复算

数据来自外部仓库 `/Users/zhuan/c-mapss_train`，默认 run 为
`FD001_cnn_bilstm_attn_w30_mrul125_final`，直接读取 predictions.npz，
复用 `src/cmapss/metrics.py` 的 coverage()。没有重训模型或复制 artifacts。

- 测试发动机：100 台；实际 pred_members 维度为 15 × 100。
- 名义覆盖率 95%；闭区间实测覆盖率 56%；缺口 39 个百分点。
- 覆盖率准入线 85%；usable_for_decision = false。
- 平均 sigma 5.223652765750885 循环。
- 预测 RUL ≤30 循环 20 台，≤60 循环 35 台；中位数 89.49304580688477 循环；
  最短预测 RUL 的机器编号为 34。以上仅为人工复核候选。

成本比 10:1、备件提前期 14 天是示意假设，在回答中显式声明；
本阶段没有使用它们优化成本或生成排期。RUL 的循环数没有直接换算成天。

## 验证

- PHM 新增单测：18/18，通过。
- 全套单测：101 项，无失败；97 项通过，4 项因缺少 German Credit 数据跳过。
- PHM Phase 0 离线评测：1/1，通过；业务阻断是本题的预期结果。
- 原领域离线回归：review 15/15；credit（GMC 样例）8/8。
- 隔离副本测试：仅复制 insight 包、ask.py 和样例 CSV，在不存在外部资产的条件下
  能离线问答，复算 56% 并触发同一阻断。
- 系统 Python 3.9 可运行；PHM 路径未加载 torch。
- 测试覆盖 85% 通过边界、低于准入线阻断、区间端点、数组错位/缺字段/非法编号、
  非有限/负标准差、空样本、未知意图、CLI 业务阻断退出码。

## 复现

在 insight-agent 仓库中执行：

```bash
python3 scripts/ask.py "能不能用这个预测区间安排停机检修？" --domain phm --no-llm
# 预期退出码为 2：覆盖率业务闸门阻断
python3 scripts/run_eval.py --domain phm --verbose
python3 -m unittest discover -s tests -p 'test_phm.py' -v
python3 -m unittest discover -s tests -v
```

默认根目录不存在且未配置 RI_PHM_ROOT 时，使用随仓库提供的
[data/sample_phm_units.csv](../data/sample_phm_units.csv)。显式配置错误会报错。
可通过 `python3 scripts/make_phm_sample.py --root /Users/zhuan/c-mapss_train`
从原预测重新生成样例。

## 验证范围与后续

本次验证的是**离线规则规划 + 模板合成 + 数字回查 + 声明式闸门**，
没有执行在线 LLM 验收，不能据此声称 LLM 自主发现缺陷或保证其拒绝行为。
现有核心在合成后报告闸门结果；本域离线模板明确拒绝。
C-MAPSS 是仿真数据，不能将本结果描述成生产系统部署。

后续待执行：剩余六个工具与五条工业闸门、成本/排期计算和参数覆盖、
十题以上评测、三条闭环用例、四张图；RAG 为可选阶段。
维护排期应在裁剪 order_day 前识别超出提前期的机器，并明确循环到天的换算；
成本优化需先确定风险时间窗与 alarm 定义。

## 本次问答原文

## Q: 能不能用这个预测区间安排停机检修？

> 领域：phm｜规划器：rule｜工具链：uncertainty_audit → fleet_overview｜状态：**ok**

### 分析范围
C-MAPSS 仿真数据 FD001；100 台测试发动机；真实预测资产；运行 FD001_cnn_bilstm_attn_w30_mrul125_final。RUL 单位为运行循环数。

### 事实
- 名义覆盖率 95%，实测覆盖率 **56%**，缺口 39.00 个百分点；准入线 85%。[T1]
- 预测标准差均值 5.22 循环；覆盖率准入未通过。[T1]
- 机队共 100 台，预测 RUL 不超过 30 循环的有 20 台，不超过 60 循环的有 35 台；这些是人工复核候选。[T2]
- 预测 RUL 中位数 89.49 循环，最短的是机器 34。[T2]

### 结论
1. 不能用这个预测区间安排停机检修：覆盖率未达到准入线，coverage_gate 阻断。应先使用独立校准集开展共形预测校准，并重新验证覆盖率。当前只支持人工复核候选，不生成停机工单。[T1]
2. 成本与提前期假设：漏报代价 : 误报代价 = 10 : 1，备件提前期 14 天；均非真实产线参数。本轮未使用这些假设做优化或排期；RUL 循环数尚未换算为天。后续决策阶段再支持提问覆盖。[T1]
3. C-MAPSS 是仿真数据，本结果用于方法验证。

---
✅ 数字回查通过：14/14 个数字可追溯到工具输出（跳过 5 个结构性数字）
⛔ 阻断：1 项硬约束未通过；⛔ coverage_gate：区间覆盖率未达到准入线，禁止用该区间安排停机检修
