# -*- coding: utf-8 -*-
"""基于工具事实合成结论，明确拒绝未通过硬约束的建议。"""


def bullets(res, idx):
    f, name, cite = res.get("facts") or {}, res["tool"], f"[T{idx}]"
    if res.get("insufficient") or res.get("error"):
        return [f"- {res['note']}。{cite}"]
    if name == "uncertainty_audit":
        return [
            f"- 名义覆盖率 {f['nominal']:.0%}，实测覆盖率 **{f['empirical_coverage']:.0%}**，"
            f"缺口 {f['gap_pt']:.2f} 个百分点；准入线 {f['min_coverage']:.0%}。{cite}",
            f"- 预测标准差均值 {f['mean_sigma']:.2f} 循环；"
            f"覆盖率准入{'通过' if f['usable_for_decision'] else '未通过'}。{cite}",
        ]
    if name == "fleet_overview":
        return [
            f"- 机队共 {f['units']} 台，预测 RUL 不超过 {f['service_near']} 循环的有 "
            f"{f['need_service_30']} 台，不超过 {f['service_mid']} 循环的有 "
            f"{f['need_service_60']} 台；这些是人工复核候选。{cite}",
            f"- 预测 RUL 中位数 {f['median_rul']:.2f} 循环，最短的是机器 {f['worst_unit']}。{cite}",
        ]
    if name == "unit_health":
        return [
            f"- 机器 {f['unit']} 当前已运行 {f['end_cycle']} 循环，预测 RUL "
            f"{f['pred_rul']:.2f} 循环，预测分区 {f['in_region']}；标准差 {f['sigma']:.2f}。{cite}",
            f"- 未校准区间 [{f['lower']:.2f}, {f['upper']:.2f}] 循环仅供展示；"
            f"仿真离线真实 RUL {f['y_true']:.2f}，绝对误差 {f['abs_err']:.2f}。{cite}",
        ]
    if name == "degradation_rate":
        acceleration = "存在加速退化证据" if f["is_accelerating"] else "加速退化证据不足"
        return [
            f"- 机器 {f['unit']} 的代理健康指数，末 {f['window_cycles']} 循环窗口斜率 "
            f"{f['slope_per_cycle']:.6f}/循环，前窗口斜率 {f['previous_slope']:.6f}/循环，"
            f"近期下降 {f['recent_drop']:.6f}；{acceleration}。{cite}",
            f"- 依据全部训练集统计定向并平滑的通道均值；这是代理指数趋势，"
            f"不是 RUL 的变化或故障因果；多工况变化也可能影响趋势。{cite}",
        ]
    if name == "sensor_attribution":
        return [
            f"- 机器 {f['unit']} 的 {f['n_channels']} 个有效通道中，"
            f"偏离度最高为 {f['top_sensor']}，代理贡献 {f['top_sensor_contrib']:.2%}，"
            f"平均绝对 z 分数 {f['top_sensor_deviation']:.4f}。{cite}",
            f"- 这是相对全部训练集统计量的窗口 z 偏离代理，不是梯度或 SHAP 归因，"
            f"也不证明故障因果；多工况中可能混有工况偏离。{cite}",
        ]
    if name == "cost_threshold":
        return [
            f"- 固定风险时间窗 {f['risk_horizon_cycles']:g} 循环，期望代价近似最优告警线为 "
            f"{f['threshold_rul']:.4f} 循环，触发 {f['n_alarms']} 台告警。{cite}",
            f"- 机队总期望代价 {f['expected_cost']:.4f}，默认 {f['default_threshold']} 循环"
            f"告警线的代价 {f['default_cost']:.4f}，降低 {f['compare_to_default']:.4f} "
            f"假设成本单位；这是未校准正态分布下的情景分析，不准入实际决策。{cite}",
        ]
    if name == "maintenance_plan":
        lines = [
            f"- 未来 {f['window_days']:g} 天的点预测候选为 {f['n_orders']} 台，"
            f"备件来不及的有 {f['n_missed']} 台。{cite}",
        ]
        for row in res.get("orders", []):
            missed = "备件来不及，当前方案不可执行" if row["missed"] else "提前期检查通过，仍待人工复核"
            lines.append(
                f"- 候选机器 {row['unit']}：预计相对今天第 {row['service_day']:.2f} 天维护，"
                f"原始下单日 {row['raw_order_day']:.2f}，建议下单日 {row['order_day']:.2f}；"
                f"{missed}。{cite}")
        if not f["n_orders"]:
            lines.append(f"- 当前点预测与窗口假设下没有维护候选；不代表已排除故障风险。{cite}")
        return lines
    if name == "model_compare":
        return [
            f"- 现有报告中 RMSE 最优为 {f['best_by_rmse']}（{f['best_rmse']:.4f}），"
            f"NASA-Score 最优为 {f['best_by_nasa']}（{f['best_nasa']:.4f}）。{cite}",
            f"- 所选比较模型 {f['chosen']}：RMSE {f['chosen_rmse']:.4f}，NASA-Score "
            f"{f['chosen_nasa']:.4f}，比最优值多 {f['nasa_gap']:.4f}；"
            f"比较不切换机队预测来源，测试集排名不能直接用于上线选型。{cite}",
        ]
    return []


def conclusion(results):
    out = []
    usable = [(i, r) for i, r in enumerate(results, 1)
              if not r.get("insufficient") and not r.get("error")]
    audit = next(((i, r) for i, r in usable if r["tool"] == "uncertainty_audit"), None)
    if audit:
        idx, res = audit
        if not res["facts"].get("usable_for_decision"):
            out.append(f"不能用这个预测区间安排停机检修：覆盖率未达到准入线，"
                       f"coverage_gate 阻断。应先使用独立校准集开展共形预测校准，"
                       f"并重新验证覆盖率。[T{idx}]")
        else:
            out.append(f"覆盖率检查通过；仍需其他维护约束校验，本阶段不生成停机安排。[T{idx}]")
    elif any(r.get("facts", {}).get("usable_for_decision") is False for _, r in usable):
        out.append("区间未达到覆盖率准入要求；成本计算仅作情景分析，不授权停机。")
    else:
        out.append("本轮是点预测或代理指标复核；缺少完整约束与区间准入验证，不能据此授权生产停机。")

    plan = next(((i, r) for i, r in usable if r["tool"] == "maintenance_plan"), None)
    if plan:
        idx, res = plan
        f = res["facts"]
        if f["n_missed"]:
            out.append(f"lead_time_gate 阻断：{f['n_missed']} 台候选机器今天下单也来不及。"
                       f"需人工核实库存、供应提前期与维护窗口后重算；当前清单不可执行。[T{idx}]")
        else:
            out.append(f"当前候选清单的提前期约束通过；未验证维修容量与真实停机窗口，"
                       f"仍需人工复核。[T{idx}]")
    check = next(((i, r["facts"]) for i, r in enumerate(results, 1)
                  if "pipeline_ok" in r.get("facts", {})), None)
    if check:
        idx, f = check
        if not f["pipeline_ok"]:
            out.append(f"pipeline_selfcheck 阻断：原始标签与传感器序列对齐检查"
                       f"{'未通过' if f.get('alignment_checked') else '缺少资产、无法验证'}，不能准入。[T{idx}]")
        if f.get("bad_units"):
            out.append(f"rul_monotone 阻断：存在负 RUL，必须核对预测与标签对齐。[T{idx}]")
    assumption = next(((i, r["facts"]) for i, r in enumerate(results, 1)
                       if "c_fn" in r.get("facts", {})), None)
    if assumption:
        idx, f = assumption
        out.append(f"成本与提前期假设：漏报代价 : 误报代价 = {f['c_fn']:g} : {f['c_fp']:g}，"
                   f"备件提前期 {f['lead_time_days']:g} 天；每天 {f['cycles_per_day']:g} 循环、"
                   f"维护窗口 {f['window_days']:g} 天、成本风险时间窗 {f['risk_horizon_cycles']:g} 循环。"
                   f"这些均非真实产线参数，可在提问中覆盖。[T{idx}]")
    out.append("C-MAPSS 是仿真数据；排期基于规则与假设，没有维修记录支持，本结果用于可审计的工程方法验证。")
    return out
