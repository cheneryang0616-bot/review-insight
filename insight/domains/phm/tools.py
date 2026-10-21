# -*- coding: utf-8 -*-
"""八个领域工具；facts 原始数值，table 展示格式，现成闸门施加约束。"""
import math

import numpy as np

from ...core.registry import Registry
from . import config, data, decision

REGISTRY = Registry("phm")
tool = REGISTRY.tool
CONTEXT_PARAMS = {
    "dataset": "FD001..FD004，默认当前数据集",
    "c_fn": "漏报代价，非负示意值，默认10",
    "c_fp": "误报代价，正示意值，默认1",
    "lead_time_days": "备件提前期（天），默认14",
    "window_days": "维护候选时间窗（天），默认14",
    "cycles_per_day": "每天运行循环数，换算假设，默认1",
    "risk_horizon_cycles": "代价计算的固定风险时间窗（循环），默认30",
}
UNIT_PARAMS = {**CONTEXT_PARAMS, "unit": "机器编号（必填）"}


def _result(name, table, facts, note="", insufficient=False, params=None, **extra):
    return {"tool": name, "params": params or {}, "table": table, "facts": facts,
            "note": note, "insufficient": insufficient, **extra}


def _prepare(df, dataset=None, **context):
    if dataset and dataset.upper() != df.attrs.get("dataset"):
        df = data.load(**data.asset_context(df, dataset=dataset))
    defaults = {
        "c_fn": config.C_FN, "c_fp": config.C_FP, "lead_time_days": config.LEAD_TIME_DAYS,
        "window_days": config.WINDOW_DAYS, "cycles_per_day": config.CYCLES_PER_DAY,
        "risk_horizon_cycles": config.RISK_HORIZON_CYCLES,
    }
    settings = {}
    for key, default in defaults.items():
        value = context.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"{key} 不能为布尔值")
        value = float(value)
        positive = key in ("c_fp", "cycles_per_day", "risk_horizon_cycles")
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"{key} 必须是有限的{'正数' if positive else '非负数'}")
        settings[key] = value
    return df, settings


def _base(df, settings):
    # 所有工具均携带同一组有效参数，防止合并 facts 时默认值覆盖提问参数。
    # 资产指纹放在这里：任何一轮回答都能说清"这份结论来自哪个文件、哪一版"。
    info = data.asset_summary(df)
    facts = {
        "dataset": df.attrs.get("dataset", config.DATASET), **settings,
        "prediction_source": str(df.attrs.get("path", "in_memory_test_fixture")),
        "asset_id": info["asset_id"], "asset_bytes": info["asset_bytes"],
        "asset_mtime_ns": info["asset_mtime_ns"],
        "asset_mtime_iso": info["asset_mtime_iso"],
        "bad_units": [int(u) for u in df.loc[df["pred"] < 0, "unit"]],
    }
    facts.update(data.pipeline_check(df))
    return facts


def _params(df, settings, **extra):
    return {"dataset": df.attrs.get("dataset", config.DATASET), **settings, **extra}


def _missing(name, df, settings, note, **params):
    return _result(name, [], {**_base(df, settings), "n": 0}, note, True,
                   _params(df, settings, **params))


def _find_unit(df, unit):
    try:
        number = float(unit)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    rows = df.loc[df["unit"] == int(number)]
    return rows.iloc[0] if len(rows) else None


def _unit_note(df, unit):
    return f"机器编号 {unit} 不存在；请提供 {df['unit'].min()}..{df['unit'].max()} 内的编号"


@tool("fleet_overview", "机队 RUL 点预测总览与数据管道自检", CONTEXT_PARAMS)
def fleet_overview(df, dataset=None, **context):
    df, settings = _prepare(df, dataset, **context)
    if df.empty:
        return _result("fleet_overview", [], {"units": 0, **_base(df, settings)},
                       "无机器级预测可供分析", True)
    facts = {
        **_base(df, settings), "units": int(len(df)),
        "need_service_30": int((df["pred"] <= config.SERVICE_NEAR).sum()),
        "need_service_60": int((df["pred"] <= config.SERVICE_MID).sum()),
        "median_rul": float(np.median(df["pred"])),
        "worst_unit": int(df.loc[df["pred"].idxmin(), "unit"]),
        "service_near": config.SERVICE_NEAR, "service_mid": config.SERVICE_MID,
    }
    table = [{"预测资产": f"{facts['asset_id']}（{facts['asset_mtime_iso']}）",
              "测试机器数": facts["units"], "预测不超过30循环": facts["need_service_30"],
              "预测不超过60循环": facts["need_service_60"],
              "RUL中位数": f"{facts['median_rul']:.2f}", "最短RUL机器": facts["worst_unit"]}]
    return _result("fleet_overview", table, facts, "按预测分档，属于人工复核候选；C-MAPSS 是仿真数据。",
                   params=_params(df, settings))


@tool("unit_health", "单机剩余寿命、未校准区间与离线误差", UNIT_PARAMS, required=("unit",))
def unit_health(df, unit, dataset=None, **context):
    df, settings = _prepare(df, dataset, **context)
    row = _find_unit(df, unit)
    if row is None:
        return _missing("unit_health", df, settings, _unit_note(df, unit), unit=unit)
    facts = {
        **_base(df, settings), "unit": int(row["unit"]), "pred_rul": float(row["pred"]),
        "sigma": float(row["sigma"]), "lower": float(row["lower"]), "upper": float(row["upper"]),
        "y_true": float(row["y_true"]), "abs_err": abs(float(row["pred"]) - float(row["y_true"])),
        "in_region": str(row["region"]), "member_std": float(row["member_std"]),
        "end_cycle": int(row["end_cycle"]),
    }
    table = [{"机器": facts["unit"], "预测RUL(循环)": f"{facts['pred_rul']:.2f}",
              "标准差": f"{facts['sigma']:.2f}", "区间下界": f"{facts['lower']:.2f}",
              "区间上界": f"{facts['upper']:.2f}", "真实RUL(离线)": f"{facts['y_true']:.2f}",
              "绝对误差": f"{facts['abs_err']:.2f}", "预测分区": facts["in_region"]}]
    return _result("unit_health", table, facts,
                   "真实 RUL 与误差仅用于仿真离线复盘；区间未经校准，不作为停机授权。",
                   params=_params(df, settings, unit=facts["unit"]))


def _diagnostic_inputs(df, unit):
    asset = data.asset_context(df)
    return data.panel(unit, **asset), data.sensor_stats(**asset)


@tool("degradation_rate", "平滑健康指数末窗口斜率与加速证据", UNIT_PARAMS, required=("unit",))
def degradation_rate(df, unit, dataset=None, **context):
    df, settings = _prepare(df, dataset, **context)
    row = _find_unit(df, unit)
    if row is None:
        return _missing("degradation_rate", df, settings, _unit_note(df, unit), unit=unit)
    unit = int(row["unit"])
    try:
        panel, stats = _diagnostic_inputs(df, unit)
    except (FileNotFoundError, KeyError) as exc:
        return _missing("degradation_rate", df, settings, str(exc), unit=unit)
    window = config.DIAGNOSTIC_WINDOW
    if len(panel) < 2 * window:
        return _missing("degradation_rate", df, settings,
                        "传感器序列不足两个诊断窗口，无法比较近期与前期退化速度", unit=unit)
    sensors = [s for s in config.HEALTH_SENSORS if s in stats.index and stats.loc[s, "std"] > 1e-8]
    if not sensors:
        return _missing("degradation_rate", df, settings, "训练集无有效健康指数通道", unit=unit)
    z = (panel[sensors] - stats.loc[sensors, "mean"]) / stats.loc[sensors, "std"]
    direction = stats.loc[sensors, "direction"]
    health = -(z * direction).mean(axis=1)
    smooth = health.rolling(config.SMOOTHING_WINDOW, min_periods=1).mean()
    recent, previous = smooth.iloc[-window:], smooth.iloc[-2 * window:-window]
    cycles = panel["cycle"].to_numpy(dtype=float)
    slope = float(np.polyfit(cycles[-window:] - cycles[-window], recent, 1)[0])
    prior = float(np.polyfit(cycles[-2 * window:-window] - cycles[-2 * window], previous, 1)[0])
    facts = {
        **_base(df, settings), "unit": unit, "slope_per_cycle": slope,
        "previous_slope": prior, "recent_drop": float(recent.iloc[0] - recent.iloc[-1]),
        "is_accelerating": bool(slope < 0 and -slope > max(0., -prior) * (1 + config.ACCELERATION_MARGIN)
                                and -slope > 1e-8),
        "window_cycles": window, "n_channels": len(sensors),
    }
    table = [{"机器": unit, "末窗口斜率": f"{slope:.6f}", "前窗口斜率": f"{prior:.6f}",
              "近期健康指数下降": f"{facts['recent_drop']:.6f}",
              "退化加速证据": "有" if facts["is_accelerating"] else "不足"}]
    return _result("degradation_rate", table, facts,
                   "训练整机首末窗口趋势确定通道方向；训练 z 分数定向均值取负作代理健康指数，"
                   "因果平滑后比较相邻末窗口斜率。仅描述代理指数，不是 RUL 变化或故障因果；"
                   "多工况使用全部训练集统计，趋势可能混入工况变化。",
                   params=_params(df, settings, unit=unit))


@tool("sensor_attribution", "窗口通道 z 偏离度代理贡献，不是梯度或 SHAP 归因",
      UNIT_PARAMS, required=("unit",))
def sensor_attribution(df, unit, dataset=None, **context):
    df, settings = _prepare(df, dataset, **context)
    row = _find_unit(df, unit)
    if row is None:
        return _missing("sensor_attribution", df, settings, _unit_note(df, unit), unit=unit)
    unit = int(row["unit"])
    try:
        panel, stats = _diagnostic_inputs(df, unit)
    except (FileNotFoundError, KeyError) as exc:
        return _missing("sensor_attribution", df, settings, str(exc), unit=unit)
    if len(panel) < config.DIAGNOSTIC_WINDOW:
        return _missing("sensor_attribution", df, settings, "传感器序列不足一个诊断窗口", unit=unit)
    sensors = [s for s in stats.index if s in panel and stats.loc[s, "std"] > 1e-8]
    if not sensors:
        return _missing("sensor_attribution", df, settings, "训练集无非常数传感器", unit=unit)
    last = panel.iloc[-config.DIAGNOSTIC_WINDOW:]
    deviation = ((last[sensors] - stats.loc[sensors, "mean"]) / stats.loc[sensors, "std"]).abs().mean()
    order = sorted(sensors, key=lambda s: (-float(deviation[s]), s))
    total = float(deviation.sum())
    top = order[0]
    facts = {**_base(df, settings), "unit": unit, "top_sensor": top,
             "top_sensor_contrib": float(deviation[top] / total) if total > 0 else 0.0,
             "n_channels": len(sensors), "top_sensor_deviation": float(deviation[top])}
    table = [{"通道": s, "平均绝对z分数": f"{deviation[s]:.4f}",
              "代理贡献": f"{float(deviation[s] / total) if total > 0 else 0.:.2%}"} for s in order]
    return _result("sensor_attribution", table, facts,
                   "末窗口各通道相对全部训练集统计量的平均绝对 z 分数，归一化作贡献。"
                   "这是代理指标，不是梯度归因、SHAP 或故障因果；多工况可能混有工况偏离。",
                   params=_params(df, settings, unit=unit))


@tool("uncertainty_audit", "预测区间覆盖率审计与准入检查", CONTEXT_PARAMS)
def uncertainty_audit(df, dataset=None, **context):
    df, settings = _prepare(df, dataset, **context)
    if df.empty:
        return _result("uncertainty_audit", [], {"n": 0, "usable_for_decision": False,
                        "empirical_coverage": 0.0, **_base(df, settings)},
                       "无样本，不能评估区间可信度", True)
    empirical = data.coverage(df)
    facts = {
        **_base(df, settings), "n": int(len(df)), "nominal": config.NOMINAL,
        "empirical_coverage": empirical, "gap_pt": (config.NOMINAL - empirical) * 100,
        "mean_sigma": float(df["sigma"].mean()), "min_coverage": config.MIN_COVERAGE,
        "usable_for_decision": bool(empirical >= config.MIN_COVERAGE),
    }
    table = [{"名义覆盖率": f"{facts['nominal']:.0%}", "实测覆盖率": f"{empirical:.2%}",
              "缺口(百分点)": f"{facts['gap_pt']:.2f}", "平均标准差(循环)": f"{facts['mean_sigma']:.2f}",
              "准入覆盖率": f"{facts['min_coverage']:.0%}",
              "覆盖率准入": "通过" if facts["usable_for_decision"] else "阻断"}]
    return _result("uncertainty_audit", table, facts,
                   "带标签仿真测试集复算 pred ± 1.96σ 闭区间覆盖率；回顾性审计不是在线故障概率。",
                   params=_params(df, settings))


@tool("cost_threshold", "固定风险时间窗下的代价敏感阈值网格搜索", CONTEXT_PARAMS)
def cost_threshold(df, dataset=None, **context):
    df, settings = _prepare(df, dataset, **context)
    if df.empty:
        return _missing("cost_threshold", df, settings, "无预测，不能计算期望代价")
    computed = decision.cost_threshold(
        df, c_fn=settings["c_fn"], c_fp=settings["c_fp"],
        risk_horizon_cycles=settings["risk_horizon_cycles"], default_threshold=config.SERVICE_NEAR)
    empirical = data.coverage(df)
    facts = {
        **_base(df, settings),
        **{k: computed[k] for k in (
            "threshold_rul", "expected_cost", "default_cost", "compare_to_default", "n_alarms")},
        "default_threshold": config.SERVICE_NEAR,
        "threshold_violations": computed["threshold_violations"], "empirical_coverage": empirical,
        "usable_for_decision": bool(empirical >= config.MIN_COVERAGE),
    }
    table = [{"告警阈值(循环)": f"{r['threshold_rul']:.4f}",
              "机队期望代价": f"{r['expected_cost']:.4f}", "告警机器数": r["n_alarms"]}
             for r in computed["threshold_curve"]]
    return _result("cost_threshold", table, facts,
                   "这是期望代价近似：RUL ~ N(μ,σ)，固定风险事件 RUL < H，"
                   "alarm_i(t)=1[μ_i≤t]；总成本 Σ(C_fn·P(RUL_i<H)·(1−alarm_i)+C_fp·alarm_i)。"
                   "仅搜索告警线，避免阈值和风险事件同时变化导致零阈值退化。"
                   "C_fp 在此近似中作为每次告警成本支付，包含真告警。"
                   "compare_to_default 为默认30循环成本减最优成本；σ 未校准时仅作情景分析，不准入。",
                   params=_params(df, settings),
                   curves={"threshold": computed["threshold_curve"], "cost_ratio": computed["ratio_curve"]})


@tool("maintenance_plan", "点预测维护候选工单与备件提前期校验", CONTEXT_PARAMS)
def maintenance_plan(df, dataset=None, **context):
    df, settings = _prepare(df, dataset, **context)
    if df.empty:
        return _missing("maintenance_plan", df, settings, "无预测，无法生成维护候选清单")
    computed = decision.maintenance_plan(
        df, lead_time_days=settings["lead_time_days"], window_days=settings["window_days"],
        cycles_per_day=settings["cycles_per_day"])
    facts = {
        **_base(df, settings),
        **{k: computed[k] for k in (
            "n_orders", "n_missed", "earliest_unit", "latest_order_day", "missed_units")},
        "order_units": [int(r["unit"]) for r in computed["orders"]],
    }
    table = [{"机器": r["unit"], "预测RUL(循环)": f"{r['pred_rul']:.2f}",
              "候选维护日(相对今天)": f"{r['service_day']:.2f}", "原始下单日": f"{r['raw_order_day']:.2f}",
              "建议下单日": f"{r['order_day']:.2f}", "备件来不及": "是" if r["missed"] else "否"}
             for r in computed["orders"]]
    return _result("maintenance_plan", table, facts,
                   "按点预测升序生成规则候选工单，不是从历史工单学习。"
                   "service_day=pred/cycles_per_day，raw_order_day=service_day−lead_time_days；"
                   "先用 raw_order_day<0 统计来不及到货，再把建议下单日截到零。"
                   "未校验维修容量与真实停机窗口，不批准生产停机。",
                   params=_params(df, settings), orders=computed["orders"])


@tool("model_compare", "同数据集模型 RMSE 与 NASA-Score 比较",
      {**CONTEXT_PARAMS, "chosen": "所选比较模型，默认当前资产 cnn_bilstm_attn"})
def model_compare(df, chosen=None, dataset=None, **context):
    df, settings = _prepare(df, dataset, **context)
    try:
        rows = data.runs(**data.asset_context(df))
    except (FileNotFoundError, KeyError) as exc:
        return _missing("model_compare", df, settings, str(exc))
    if not rows:
        return _missing("model_compare", df, settings, "无同协议模型报告")
    chosen = chosen or config.CHOSEN_MODEL
    selected = next((r for r in rows if r["model"] == chosen), None)
    if selected is None:
        return _missing("model_compare", df, settings,
                        "所选模型不存在；可选：" + "、".join(r["model"] for r in rows), chosen=chosen)
    by_rmse = min(rows, key=lambda r: (r["rmse"], r["model"]))
    by_nasa = min(rows, key=lambda r: (r["nasa_score"], r["model"]))
    facts = {
        **_base(df, settings), "best_by_rmse": by_rmse["model"], "best_by_nasa": by_nasa["model"],
        "chosen": chosen, "nasa_gap": float(selected["nasa_score"] - by_nasa["nasa_score"]),
        "best_rmse": float(by_rmse["rmse"]), "best_nasa": float(by_nasa["nasa_score"]),
        "chosen_rmse": float(selected["rmse"]), "chosen_nasa": float(selected["nasa_score"]),
        "n_models": len(rows),
    }
    table = [{"模型": r["model"], "RMSE": f"{r['rmse']:.4f}", "NASA-Score": f"{r['nasa_score']:.4f}",
              "来源": r["source"]} for r in rows]
    return _result("model_compare", table, facts,
                   "只比较 summary 对应的 final backbone 与 GBDT，不混入探针/融合。"
                   "NASA-Score 对高估 RUL（晚报警）惩罚更重；chosen 仅指定比较对象，"
                   "不切换机队预测来源；测试集排名不是上线选型验证。",
                   params=_params(df, settings, chosen=chosen))
