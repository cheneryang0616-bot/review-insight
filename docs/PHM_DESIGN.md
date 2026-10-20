# PHM Stage 1：设备健康与维护候选决策

执行日期：2026-10-21。按新版 /Users/zhuan/agent/PHM_DATA_AGENT_PLAN.md 的 Phase 1，
在既有 Phase 0 之上补齐八工具、六条工业闸门、七种意图与可覆盖假设。
定位是可复现、可审计、可准入的工程交付，不包装为学术创新或生产落地。

## 一条命令与交付证据

在 insight-agent 仓库执行：

~~~bash
python3 scripts/verify_phm_stage1.py --root /Users/zhuan/c-mapss_train
~~~

该命令强制离线，运行完整单元测试、三个领域评测与闭环回归，重算 Stage 1 问答，
写出 results/REPORT_phm.md 与 results/phm_stage1.json。JSON 包含完整 facts/table、
工具调用参数、逐闸门结果、同会话追问、源文件路径及 SHA-256。
只读既有预测，不训练模型、不复制 artifacts、不调用在线 LLM。

本次 179 项单测中 175 项通过、4 项因缺少 German Credit 数据跳过；
PHM 95 项均通过。PHM 离线评测 9/9、闭环用例 3/3；
旧领域 review 15/15、credit(GMC样例)8/8、整体闭环12/12。

## 数据口径与来源定位

- RI_PHM_ROOT 默认 /Users/zhuan/c-mapss_train；显式配置错误不回退。
- 支持 FD001–FD004，默认使用各自 cnn_bilstm_attn_w30_mrul125_final 集成。
  56% 覆盖率属于 FD001 该集成；交接文档的 Patch Transformer 只是 npz 字段示例，
  不能混用其 50% 覆盖率。
- data.load(root=None,dataset=None) 返回一行一台机器：
  unit、y_true、pred、sigma、member_std、end_cycle、region、lower、upper。
  region 按预测 ≤30、≤60、>60 分档；区间为 pred ±1.96σ。
- npz 只通过 allow_pickle=False 读取；实际 FD001 成员维度为15×100，成员数不写死。
  单机数字定位为 predictions.npz::field[u_test == unit]，不按排期排序后的行号取标签。
- 原始 RUL_FD00X.txt 第 unit 行，与 y_true=min(raw_RUL,125) 核对；
  原始 test 文件按 unit/cycle 选择，末循环与 end_cycle 核对。
- pipeline_check 检查预测机器数、标签数、RUL行数、原始test唯一unit数，
  还检查编号集合、截断后标签及末循环的一致性。负预测保留给 RUL 闸门检查。
- panel() 缓存原始26列数据并返回独立副本；缓存键包括路径、修改时间、文件大小。
  sensor_stats() 只用训练集，不用测试机或真实RUL拟合标准化。
- runs() 读取五个正式模型的 report.json，并与 summary 核对；只允许相同数据集、
  窗口30、截断125的clip标签协议，不混入探针、联合训练或融合实验。
  CNN15成员与其他backbone3成员的差异保留为既有模型比较口径，不声称是等预算架构消融。
- 模型指标可定位至各 report.json 的 ensemble.rmse / ensemble.nasa_score，
  汇总定位至 summary_FD00X.json 的模型键。

数据不进 Agent 的语言模型上下文；语言模型只能看到工具结果。facts 保留原始浮点，
table 的浮点值格式化为展示字符串，最终答案数字由现有 grounding 回查。

## 八个工具

| 工具 | 实现与输出 | 边界 |
|---|---|---|
| fleet_overview | 机器数、预测分档数、中位数、最短RUL机器 | 人工复核候选 |
| unit_health | 点预测、标准差、区间、真实RUL、绝对误差、预测分区 | 真实值只作仿真离线复盘 |
| degradation_rate | 平滑代理健康指数末窗口斜率、前窗口斜率、近期下降、加速标志 | 单条传感器序列证据，不是RUL轨迹 |
| sensor_attribution | 每通道末窗口平均绝对z分数及归一化代理贡献 | 不是梯度、SHAP或故障因果 |
| uncertainty_audit | 名义/实测覆盖率、百分点缺口、标准差均值、准入标志 | 56% 欠覆盖时拒绝区间决策 |
| cost_threshold | 固定风险时间窗的告警线网格搜索、默认代价对比与成本比曲线 | 期望代价近似，仅情景分析 |
| maintenance_plan | 点预测升序的候选工单、原始/截零下单日、备件来不及数量 | 未验证实际库存、容量、停机窗口 |
| model_compare | RMSE最优、NASA最优、所选比较模型及NASA差值 | 比较不自动切换机队预测来源 |

### 退化与通道代理

训练各通道的总体均值/标准差用于 z 标准化。传感器2/7/11/12/15方向取训练整机
末30循环均值减首30循环均值的平均趋势；以定向 z 均值取负作健康指数，
因果5点平滑，比较最后两个30循环窗口的线性斜率。
近期负斜率的下降速度比前窗口增加20%以上才报告加速代理证据；
不足60条记录就返回 insufficient，不从点预测或误差推断退化加速。

通道贡献取末30循环每个非常数通道的平均绝对z分数，归一化后排序。
FD002/FD004 仍使用全训练集统计，未分工况标准化，工况变化可能混入偏离或趋势；
这些限制在工具note与答案中明确声明。

### 代价目标的必要澄清

原方案用同一个 t 表示风险事件 RUL<t 和告警线，搜索时会同时改变要防范的事件，
可能退化为零阈值。实现将风险时间窗固定为 H，只搜索告警线 t：

    p_i = P(RUL_i < H), RUL_i ~ N(pred_i, sigma_i)
    alarm_i(t) = 1[pred_i <= t]
    E(t) = Σ C_fn * p_i * (1 - alarm_i(t)) + C_fp * alarm_i(t)

默认 H=30循环，可在问句中覆盖。候选网格包含零、默认30、各预测值及最大预测值之外，
覆盖所有不同告警集合。严格同成本时取较小阈值。sigma=0按退化分布处理，
风险事件严格小于H，告警使用小于等于t。额外计算升序成本比曲线并检查阈值不下降。
compare_to_default = 默认告警线代价 - 最优代价，单位为机队总假设成本单位。

C_fp 在这个原方案近似中对每次告警付费，包含真告警，并非完整误报成本模型。
由于默认σ的实测覆盖率仅56%，这些概率未校准，成本阈值同时产出覆盖率信号，
由 coverage_gate 阻断实际区间/概率决策；不把最优数值包装成可上线阈值。

### 排期目标的必要澄清

RUL单位是循环，不是天。默认 cycles_per_day=1 仅作显式假设，支持提问覆盖。
先算 service_day=pred/cycles_per_day，选出维护窗口内候选并按预测升序：

    raw_order_day = service_day - lead_time_days
    missed = raw_order_day < 0
    order_day = max(0, raw_order_day)

必须在截零前识别备件来不及；恰好为零不算missed。没有候选时返回空清单，
不代表排除了风险。排期只基于点预测及规则，没有维修历史或实际生产排程优化证据。

## 六条工业闸门

沿用 core.gates 的 fact_gate / monotone_list_gate；没有另建闸门系统。

| 名称 | 级别 | 事实断言 |
|---|---|---|
| coverage_gate | error | empirical_coverage ≥ 0.85 |
| lead_time_gate | error | n_missed == 0 |
| rul_monotone | error | bad_units 为空 |
| threshold_monotone | warn | threshold_violations 为空 |
| model_selection | warn | nasa_gap ≤ 数值容差 |
| pipeline_selfcheck | error | pipeline_ok 为真 |

每条均有触发/不触发测试；管道未验证、空样本和摘要模式都不能伪装通过。
框架自带 numeric_trace 与 tool_health，故一次问答返回6条工业闸门及2条通用闸门。
无关指标的闸门按框架规则跳过；gates.passed 只表示本轮已产生指标的硬约束未失败，
不等于全部生产约束验证完成。status=ok 也只代表执行完成。
调用方至少同时检查 status、insufficient/error 和逐闸门结果；候选清单仍待人工复核。

## 规划、覆盖参数与闭环

七意图：机队、单机、原因、区间、阈值、排期、模型。显式机器编号、数据集、
成本值/倍数、备件提前期、维护窗口、循环/天、固定风险循环窗和比较模型均由问句抽取。
默认假设只由工具填充；所有工具都携带本轮有效假设，避免合并facts时被另一工具默认值覆盖。

“哪几台两周内必须修”走 maintenance_plan,fleet_overview；
“那提前期改成7天呢”继承维护窗口。全局区间审计不强求单机编号；
“这台还能飞多久”或“这台区间可信吗”缺编号时通过 required(unit) 触发澄清。
越界编号返回 insufficient+编号范围；原始序列不足时走框架 degraded，
不会伪称已诊断或自动生成有效工单。未知长意图返回领域能力清单。

共享核心保留原有规划与闭环机制，仅修复 _inherit 对 0 值的存在性判断，
避免有效的零提前期或零漏报成本在追问中被恢复为默认值。该兼容修复有共享回归测试，
且旧评论/信用领域评测通过；这是相对“core不动”约定的一处必要小修。

## 真实验收与诚实边界

默认 FD001，两周窗口、每日1循环、备件14天：
机器34、81成为候选，预计相对今天9.6612、10.8026天；
原始下单日-4.3388、-3.1974，虽建议下单日均截到0，n_missed仍为2，
lead_time_gate按预期阻断。提前期改7天后，原始下单日均非负，该约束通过。
主问句20/20、追问19/19个答案数字可回查。

Phase 0 仍保持覆盖率56%、coverage_gate阻断和明确拒绝区间停机建议。
覆盖率修复方向是独立校准集上的共形预测并重新评估；本次没有校准，也不在测试集上调校准器。

独立 clone 自带100机器摘要，能运行受限问答；但不含原始序列、训练统计或模型报告，
对应工具返回 insufficient，pipeline_selfcheck明确阻断不可验证的准入，
不把“能跑”说成“完整通过”。完整Stage1验收要求RI_PHM_ROOT指向原始资产。

当前承诺范围是离线规则规划与模板合成。在线 LLM 规划存在漏必审工具的已知路径，
且现有核心在合成之后报告闸门，不保证重写违反业务语义的语言结论；
未经在线强制计划校验与语义拒绝测试，不能声称在线LLM已经安全准入。
所有结果均基于仿真数据、显式成本与提前期假设，没有真实维修记录。
Phase 2 四张图、十题以上完整评测扩展及可选 RAG 不在本次交付范围。
