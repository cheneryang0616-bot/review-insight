# -*- coding: utf-8 -*-
"""离线合成：必须明确拒绝未通过覆盖率准入的区间。"""


def bullets(res, idx):
    f = res.get("facts") or {}
    cite = f"[T{idx}]"
    if res.get("insufficient"):
        return [f"- {res['note']}。{cite}"]
    if res["tool"] == "uncertainty_audit":
        return [
            f"- 名义覆盖率 {f['nominal']:.0%}，实测覆盖率 **{f['empirical_coverage']:.0%}**，"
            f"缺口 {f['gap_pt']:.2f} 个百分点；准入线 {f['min_coverage']:.0%}。{cite}",
            f"- 预测标准差均值 {f['mean_sigma']:.2f} 循环；"
            f"覆盖率准入{'通过' if f['usable_for_decision'] else '未通过'}。{cite}",
        ]
    if res["tool"] == "fleet_overview":
        return [
            f"- 机队共 {f['units']} 台，预测 RUL 不超过 {f['service_near']} 循环的有 "
            f"{f['need_service_30']} 台，不超过 {f['service_mid']} 循环的有 "
            f"{f['need_service_60']} 台；这些是人工复核候选。{cite}",
            f"- 预测 RUL 中位数 {f['median_rul']:.2f} 循环，最短的是机器 "
            f"{f['worst_unit']}。{cite}",
        ]
    return []


def conclusion(results):
    out = []
    audit = next(((i, r) for i, r in enumerate(results, 1)
                  if r["tool"] == "uncertainty_audit"), None)
    if audit:
        idx, res = audit
        f = res["facts"]
        if not f.get("usable_for_decision"):
            out.append(f"不能用这个预测区间安排停机检修：覆盖率未达到准入线，"
                       f"coverage_gate 阻断。应先使用独立校准集开展共形预测校准，"
                       f"并重新验证覆盖率。当前只支持人工复核候选，不生成停机工单。[T{idx}]")
        else:
            out.append(f"覆盖率检查通过；停机还需要维护窗口、备件与工况约束校验，"
                       f"本阶段不生成停机安排。[T{idx}]")
    else:
        out.append("本轮缺少区间覆盖率审计，不能据此安排停机检修。")
    assumption = next(((i, r["facts"]) for i, r in enumerate(results, 1)
                       if "c_fn" in r.get("facts", {})), None)
    if assumption:
        idx, f = assumption
        out.append(f"成本与提前期假设：漏报代价 : 误报代价 = {f['c_fn']:g} : {f['c_fp']:g}，"
                   f"备件提前期 {f['lead_time_days']} 天；均非真实产线参数。"
                   f"本轮未使用这些假设做优化或排期；RUL 循环数尚未换算为天。"
                   f"后续决策阶段再支持提问覆盖。[T{idx}]")
    out.append("C-MAPSS 是仿真数据，本结果用于方法验证。")
    return out
