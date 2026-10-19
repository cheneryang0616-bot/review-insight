# -*- coding: utf-8 -*-
"""review 域 —— 模板合成器

把每个工具结果翻译成 1~3 条人话。所有数字都取自 facts / table，
因此模板合成天然不会出现幻觉（groundedness 在 LLM 路径上才真正起作用）。
"""
from typing import List

from ...core.agent import pct as _pct, pt as _pt
from . import config


def bullets(res: dict, idx: int) -> List[str]:
    t, f, tab = res["tool"], res.get("facts") or {}, res.get("table") or []
    tag = f"[T{idx}]"
    if not f:
        return [f"- {res.get('note') or '（无可用结果）'} {tag}"]

    if t == "dataset_overview":
        return [
            f"- 数据规模：**{f['rows']}** 条团购评论，覆盖 **{f['shops']}** 家店、"
            f"{f['deals_effective']} 个有效套餐（≥30 条评价），时间跨度 "
            f"{f['start']} ~ {f['end']}。{tag}",
            f"- 总体差评率 **{_pct(f['bad_rate'])}**（{f['bad_rows']} 条）；"
            f"文本可用率 {_pct(f['text_usable_rate'])} —— 短评（<{config.MIN_TEXT_LEN} 字）"
            f"占 {_pct(f['text_short_rate'])}，文本类分析已剔除。{tag}",
        ]

    if t == "star_distribution":
        lines = [f"- 当前范围共 **{f['n']}** 条评论，差评 **{f['bad_n']}** 条"
                 f"（**{_pct(f['bad_rate'])}**），平均星级 {f['mean_star']}。{tag}"]
        shares = [(k, v) for k, v in f.items() if k.startswith("share_")]
        if shares:
            top = sorted(shares, key=lambda kv: -kv[1])[:3]
            lines.append("- 星级结构（占比最高的三档）：" + "、".join(
                f"{k.replace('share_', '')} 星 {_pct(v)}" for k, v in top) + f"。{tag}")
        return lines

    if t == "shop_ranking":
        lines = [f"- 共 **{f['shops_compared']}** 家店达到样本量门槛；差评率最高的是"
                 f"店铺 **{f['worst_shop']}**（**{_pct(f['worst_bad_rate'])}**，"
                 f"{f['worst_n']} 条），最低 {_pct(f['best_bad_rate'])}，"
                 f"极差 {_pt(f['spread_pt'])}。{tag}"]
        if tab:
            lines.append("- 差评率前 3：" + "、".join(
                f"{r['店铺ID']}（{r['差评率']}）" for r in tab[:3]) + f"。{tag}")
        return lines

    if t == "deal_ranking":
        lines = [f"- 达标套餐 **{f['deals_compared']}** 个；差评率最高的是"
                 f"「**{f['worst_deal']}**」（**{_pct(f['worst_bad_rate'])}**，"
                 f"{f['worst_n']} 条），最低 {_pct(f['best_bad_rate'])}，"
                 f"极差 {_pt(f['spread_pt'])}。{tag}"]
        if tab:
            lines.append("- 差评率前 3：" + "、".join(
                f"{r['套餐']}（{r['差评率']}）" for r in tab[:3]) + f"。{tag}")
        return lines

    if t == "deal_type_compare":
        hi = f["higher_type"]
        return [
            f"- 代金券类差评率 **{_pct(f['coupon_bad_rate'])}**（{f['coupon_n']} 条），"
            f"套餐类 **{_pct(f['combo_bad_rate'])}**（{f['combo_n']} 条）；"
            f"{hi}类是另一类的 **{f['ratio']} 倍**。{tag}",
        ]

    if t == "aspect_gap":
        lines = [f"- 差评最集中的短板是「**{f['top_aspect']}**」：差评中负面表述率 "
                 f"**{_pct(f['top_aspect_neg_bad'])}**，好评中仅 "
                 f"{_pct(f.get('top_aspect_neg_good', 0))}，"
                 f"差 **{f['top_aspect_lift']} 倍**。{tag}"]
        if len(tab) >= 3:
            lines.append("- 短板排序（差评负面率）：" + "；".join(
                f"{r['方面']} {r['差评负面率']}（{r['倍差']}×）" for r in tab[:3])
                + f"。{tag}")
            lines.append(f"- 对应运营动作：{tab[0]['建议动作']}。{tag}")
        return lines

    if t == "keyword_diff":
        if not tab:
            return [f"- 该切片样本不足，未做词汇差异分析。{tag}"]
        words = "、".join(f"**{r['词']}**（z={r['z 值']}）" for r in tab[:6])
        return [f"- 差评相对好评显著更多的词：{words}。"
                f"（信息先验 log-odds，z>1.96 为 5% 水平显著）{tag}"]

    if t == "time_trend":
        lines = [f"- 按 {f['freq']} 共 **{f['periods']}** 期；最近一期 "
                 f"{f['last_period']} 差评率 **{_pct(f['last_bad_rate'])}**"
                 f"（{f['last_n']} 条）。{tag}"]
        if "last_z" in f:
            flag = "**异动**" if f["is_anomaly"] else "属正常波动"
            lines.append(f"- 历史基线均值 {_pct(f['baseline_mean'])}，"
                         f"z = {f['last_z']} → {flag}。{tag}")
        return lines

    if t == "time_slot":
        return [f"- 差评率最高时段 **{f['high_slot']}**（{_pct(f['high_bad_rate'])}），"
                f"最低 {f['low_slot']}（{_pct(f['low_bad_rate'])}），"
                f"极差 {_pt(f['spread_pt'])}。{tag}"]

    if t == "sample_comments":
        quotes = "；".join(f"「{r['评论'][:40]}」" for r in tab[:3])
        return [f"- 抽取 {f['n']} 条原话（候选池 {f['pool']} 条，平均长度 "
                f"{f['mean_len']} 字）：{quotes}。{tag}"]

    return [f"- {res.get('note', '')} {tag}"]


def conclusion(results: list) -> List[str]:
    """结论段：只做复述 + 排序，不引入任何新数字。"""
    out = []
    ag = next((r for r in results if r["tool"] == "aspect_gap"), None)
    dtc = next((r for r in results if r["tool"] == "deal_type_compare"), None)
    if ag and ag.get("facts"):
        i = results.index(ag) + 1
        out.append(f"**体验短板在「{ag['facts']['top_aspect']}」**，"
                   f"它是差评与好评差异最大的方面（{ag['facts']['top_aspect_lift']} 倍），"
                   f"优先修它。 [T{i}]")
    if dtc and dtc.get("facts"):
        i = results.index(dtc) + 1
        out.append(f"**{dtc['facts']['higher_type']}类团购是差评集中区**，"
                   f"差评率是另一类的 {dtc['facts']['ratio']} 倍，"
                   f"说明问题更可能出在履约规则而不是出品。 [T{i}]")
    return out
