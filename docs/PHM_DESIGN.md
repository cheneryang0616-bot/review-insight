# PHM Phase 0：区间决策准入原型

执行日期：2026-10-21。本次按交接文档附录 B 的“只做 Phase 0”完成检查点；
Phase 1 的剩余工具与工业约束、Phase 2 图表与完整评测、可选 Phase 3 RAG 均待后续执行。

## 目标与结果

对“能不能用这个预测区间安排停机检修？”返回可追溯的覆盖率审计，
并对欠覆盖的区间明确拒绝停机建议。三条验收判据均通过：

1. 答案出现 56%，可回查到 `uncertainty_audit.facts.empirical_coverage = 0.56`。
2. `gates.passed == False`，阻断项含 `coverage_gate`。
3. 答案明确写出“不能用这个预测区间安排停机检修”，没有批准停机。

详见 [验收报告](../results/REPORT_phm.md) 与 [完整结果](../results/phm_phase0.json)。
这里验证的是离线规则规划与模板合成的决策路径，不构成在线 LLM 自主推理能力的证明。

## 数据与计算口径

- 默认资产根目录：`/Users/zhuan/c-mapss_train`，可用 `RI_PHM_ROOT` 覆盖。
- 默认 run：`FD001_cnn_bilstm_attn_w30_mrul125_final`。交接文档中的
  Patch Transformer npz 是字段示例，其覆盖率为 50%；56% 属于本默认集成。
- `np.load(..., allow_pickle=False)` 只读 `predictions.npz`。实际成员数组为
  `(15, 100)`，加载器接受任意正成员数、沿成员轴求标准差。
- 一行一台机器：`unit / y_true / pred / sigma / member_std / end_cycle`。
  派生 `region`（按预测 ≤30、≤60、>60 分档）和 `lower / upper = pred ± 1.96σ`。
- 验证字段维度、相等长度、机器编号连续唯一、数值有限及标准差非负。
  这是加载时的基础校验，后续阶段仍需对原始 RUL 文件与 panel 做完整管道自检。
- 完整资产通过加载纯 numpy 的 `src/cmapss/metrics.py` 复用其 `coverage()`，
  不导入 torch、不加载模型。可移植样例用相同的闭区间覆盖公式；
  测试验证其与真实资产结果一致。
- 样例由脚本导出 FD001 全部 100 台机器，保留原始浮点精度。
  只有默认根目录不存在且未显式配置时才回退样例；配置错误或文件损坏不静默回退。

## 接口与闸门

新增 `PHMDomain`，在领域注册表增加 `phm`，复用核心的 Registry、PlanSpec、
Agent 闭环、grounding 和 gates。`insight/core/` 没有修改。

| 工具 | 主要 facts | 用途 |
|---|---|---|
| `fleet_overview` | units、need_service_30、need_service_60、median_rul、worst_unit | 点预测分档，仅列人工复核候选 |
| `uncertainty_audit` | nominal、empirical_coverage、gap_pt、mean_sigma、usable_for_decision | 带真实标签的测试集回顾性区间审计 |

`facts` 保留原始数值，`table` 中浮点数格式化成展示字符串。
`gap_pt = (nominal - empirical_coverage) × 100`，单位为百分点；
`usable_for_decision = empirical_coverage >= nominal - 0.10`。

通过现有 `fact_gate` 声明 `coverage_gate`，等级 `error`，检查
`empirical_coverage >= 0.85`。框架还会执行通用数字回查和工具健康闸门。
真实数据中覆盖率 0.56，单独触发覆盖率业务阻断。85% 边界会通过，
但输出仍明确说明尚需其他维护约束，本阶段不批准停机。

空 DataFrame 的两个工具返回 `insufficient`；审计返回不可用信号并触发阻断。
未知意图由现有闭环返回当前能力清单，不伪装成已实现单机诊断或排期。

## 诚实边界

- C-MAPSS 是仿真数据。这个结果是方法与工程闭环验证，不是工业部署。
- 成本比 10:1、备件提前期 14 天是示意假设；每次正常离线回答均明确声明。
  Phase 0 不计算成本、不生成工单、不进行天数换算；自然语言覆盖参数留给 Phase 1。
- 56% 是真实欠覆盖缺陷。修复方向是使用独立校准集开展共形预测并重新评估，
  本次没有校准，也没有利用测试集拟合校准器。
- 覆盖率检查通过只表示这一项准入检查通过，不能据此证明可部署。
- 本次三条验收在 `--no-llm` 模式完成。核心框架的闸门是在合成后报告阻断；
  离线 PHM 模板显式拒绝。在线 LLM 的选工具完整性、输出服从性和拒绝行为
  尚未验证，不能把提示词当作已实现的强制语义拦截。
- 当前 `status=ok` 表示闭环执行完成，业务是否允许必须看 `gates.passed`。
  CLI 在业务闸门阻断时退出码为 2，本例是预期行为。

## 验证与后续

`tests/test_phm.py` 含 18 项测试，覆盖真实资产与样例一致性、字段/维度/编号错误、
闭区间覆盖公式、85% 边界、空样本、工具展示格式、未知意图、CLI 退出码和三条验收。
Phase 0 评测仅一题；尚未声称满足 Phase 2 的至少十题评测或四张图要求。

Phase 1 需要先修正文档内两个实现歧义：维护排期必须在裁剪到零之前用原始
`pred - lead_time` 识别来不及到货的机器；RUL 循环数与天必须有显式换算口径。
成本阈值还需明确固定风险时间窗与 alarm 定义，避免不同阈值对应不同风险事件
而导致退化的期望代价目标。这些决策计算本次均未实现。
