# -*- coding: utf-8 -*-
"""Phase 0 两个工具；facts 保留原始数值，table 使用展示格式。"""
import numpy as np

from ...core.registry import Registry
from . import config, data

REGISTRY = Registry("phm")
tool = REGISTRY.tool


def _result(name, table, facts, note="", insufficient=False):
    return {"tool": name, "params": {}, "table": table, "facts": facts,
            "note": note, "insufficient": insufficient}


def _assumptions():
    return {"c_fn": config.C_FN, "c_fp": config.C_FP,
            "lead_time_days": config.LEAD_TIME_DAYS}


@tool("fleet_overview", "机队 RUL 点预测总览；不生成停机工单", {})
def fleet_overview(df):
    if df.empty:
        return _result("fleet_overview", [], {"units": 0}, "无机器级预测可供分析", True)
    facts = {
        "units": int(len(df)),
        "need_service_30": int((df["pred"] <= config.SERVICE_NEAR).sum()),
        "need_service_60": int((df["pred"] <= config.SERVICE_MID).sum()),
        "median_rul": float(np.median(df["pred"])),
        "worst_unit": int(df.loc[df["pred"].idxmin(), "unit"]),
        "service_near": config.SERVICE_NEAR, "service_mid": config.SERVICE_MID,
        **_assumptions(),
    }
    table = [{"测试机器数": facts["units"], "预测不超过30循环": facts["need_service_30"],
              "预测不超过60循环": facts["need_service_60"],
              "RUL中位数": f"{facts['median_rul']:.2f}", "最短RUL机器": facts["worst_unit"]}]
    return _result("fleet_overview", table, facts,
                   "按预测分档；计数是人工复核候选，不表示已经授权停机。C-MAPSS 是仿真数据。")


@tool("uncertainty_audit", "复算名义区间的实测覆盖率，判断是否达到决策准入线", {})
def uncertainty_audit(df):
    if df.empty:
        return _result("uncertainty_audit", [], {"n": 0, "usable_for_decision": False,
                        "empirical_coverage": 0.0}, "无样本，不能评估区间可信度", True)
    empirical = data.coverage(df)
    facts = {
        "n": int(len(df)), "nominal": config.NOMINAL, "empirical_coverage": empirical,
        "gap_pt": (config.NOMINAL - empirical) * 100,
        "mean_sigma": float(df["sigma"].mean()), "min_coverage": config.MIN_COVERAGE,
        "usable_for_decision": bool(empirical >= config.MIN_COVERAGE), **_assumptions(),
    }
    table = [{"名义覆盖率": f"{facts['nominal']:.0%}", "实测覆盖率": f"{empirical:.0%}",
              "缺口(百分点)": f"{facts['gap_pt']:.2f}", "平均标准差(循环)": f"{facts['mean_sigma']:.2f}",
              "准入覆盖率": f"{facts['min_coverage']:.0%}",
              "覆盖率准入": "通过" if facts["usable_for_decision"] else "阻断"}]
    return _result("uncertainty_audit", table, facts,
                   "在带真实标签的仿真测试集上复算 pred ± 1.96σ 的闭区间覆盖率；"
                   "属于回顾性审计，不是在线故障概率。覆盖率达标也不等于可直接安排停机。")
