# -*- coding: utf-8 -*-
"""credit 域 —— 模板合成器"""
from typing import List

from ...core.agent import pct as _pct, pt as _pt


def bullets(res: dict, idx: int) -> List[str]:
    t, f, tab = res["tool"], res.get("facts") or {}, res.get("table") or []
    tag = f"[T{idx}]"
    if not f:
        return [f"- {res.get('note') or '（无可用结果）'} {tag}"]
    # 样本不足时 facts 里只有 "n" / "xxx_compared" 之类，没有完整字段。
    # 不拦住就会在下面按完整字段取数时 KeyError —— 这个分支由 CI 在样例数据下暴露出来。
    if res.get("insufficient") and len(f) <= 2:
        return [f"- {res.get('note') or '当前切片样本不足，无法给出结论'} {tag}"]

    if t == "dataset_profile":
        lines = [
            f"- 数据集 **{f['dataset']}**：{f['rows']} 条样本、{f['features']} 个字段，"
            f"坏客户 **{f['bad_n']}** 条（**{_pct(f['bad_rate'])}**），"
            f"不平衡比约 **1:{(f['imbalance'] if f['imbalance'] is not None else 0):.1f}**。{tag}",
            f"- 缺失最严重字段缺失率 {_pct(f['missing_max'])}（{f['missing_cols']} 个字段有缺失），"
            f"{f['extreme_cols']} 个字段存在极端值（max 超过 P99 的 5 倍）。{tag}",
        ]
        if not f["has_time"]:
            lines.append("- **该数据集没有时间列**：只能随机切分，不能做 OOT 验证，"
                         f"也无法做跨期 PSI 监控 —— 这一点会影响结论的适用范围。{tag}")
        if f["forbidden_hard"]:
            lines.append(f"- 数据中含 **{f['forbidden_hard']} 个硬阻断字段**（性别/婚姻、外籍劳工），"
                         f"入模即不合规。{tag}")
        return lines

    if t == "iv_table":
        lines = [f"- 共 {f['features']} 个变量；IV 最高的是 **{f['iv_max_feature']}**"
                 f"（IV = **{f['iv_max']}**）。{tag}"]
        if f["iv_high"]:
            lines.append(f"- ⚠️ **{f['iv_high_count']} 个变量 IV 超过 {f['iv_suspicious_threshold']}**"
                         f"（{', '.join(f['iv_high'][:3])}）：这不是「变量太强」，"
                         f"而是**先要怀疑泄漏或极端值**。{tag}")
        if f["nonmono"]:
            lines.append(f"- **{f['nonmono_count']} 个变量 WOE 非单调**（{', '.join(f['nonmono'][:3])}），"
                         f"需要手工合并分箱后再入模。{tag}")
        top = "、".join(f"{r['变量']} {r['IV']}" for r in tab[:3])
        lines.append(f"- IV 前三：{top}。{tag}")
        return lines

    if t == "woe_table":
        mono = "单调" if f["monotone"] else "**非单调**"
        return [
            f"- 变量 **{f['feature']}**：IV = {f['iv']}，{f['bins']} 箱，WOE {mono}。{tag}",
            f"- 各箱坏账率区间 {_pct(f['bad_rate_min'])} ~ {_pct(f['bad_rate_max'])}，"
            f"WOE 区间 {f['woe_min']} ~ {f['woe_max']}。{tag}",
        ]

    if t == "perf_metrics":
        return [
            f"- **AUC {f['auc']}，KS {f['ks']}，Gini {f['gini']}，Lift@10% {f['lift_10']}**"
            f"（{f['n_features']} 个特征，训练/验证 {f['n_train']}/{f['n_test']}）。{tag}",
            f"- 验证集坏账率 {_pct(f['test_bad_rate'])}；KS 行业下限 {f['ks_floor']}，"
            f"{'达标' if f['ks'] >= f['ks_floor'] else '**未达标**'}。{tag}",
        ]

    if t == "cutoff_tradeoff":
        if f["keeps"] == 1:
            r = tab[0]
            return [f"- 若通过率压到 **{r['通过率']}**：实际坏账率 **{r['实际坏账率']}**"
                    f"（{r['通过人数']} 人通过），同时捕获 **{r['捕获坏客户']}** 的坏客户。{tag}"]
        return [
            f"- 通过率 **{_pct(f['loose_keep'])}** 时实际坏账率 **{_pct(f['full_bad_rate'])}**；"
            f"收紧到 **{_pct(f['strict_keep'])}** 时坏账率降到 **{_pct(f['strict_bad_rate'])}**，"
            f"同时捕获 **{_pct(f['strict_capture'])}** 的坏客户。{tag}",
        ]

    if t == "segment_bad_rate":
        return [
            f"- 按 **{f['feature']}** 分 {f['segments']} 群：最高群 **{f['high_segment']}**"
            f"（坏账率 **{_pct(f['high_bad_rate'])}**），最低群 {f['low_segment']}"
            f"（{_pct(f['low_bad_rate'])}），极差 {_pt(f['spread_pt'])}。{tag}",
        ]

    if t == "score_scale":
        return [
            f"- 评分刻度：基准分 {f['base']}，PDO {f['pdo']}，基准好坏比 {f['base_odds']}；"
            f"验证集分数中位数 **{f['score_median']}**（P10 {f['score_p10']}，P90 {f['score_p90']}）。{tag}",
        ]

    if t == "forbidden_check":
        lines = []
        if f["forbidden_present"]:
            used = f["forbidden_used_count"]
            lines.append(f"- 合规检查：数据含 **{f['forbidden_present']} 个硬阻断字段**，"
                         f"其中 **{used} 个已进入模型** —— "
                         f"{'**不可用**，必须先剔除再建模' if used else '当前未入模'}。{tag}")
        else:
            lines.append(f"- 合规检查：该数据集不含性别/婚姻、外籍劳工等硬阻断字段。{tag}")
        if f["sensitive_present"]:
            lines.append(f"- 另有 {f['sensitive_present']} 个敏感字段（如年龄），"
                         f"已入模 {f['sensitive_used_count']} 个，需在报告中说明使用理由。{tag}")
        return lines

    if t == "sample_cases":
        keys = [k for k in tab[0] if k != "标签"][:3] if tab else []
        sample = "；".join(
            f"{r['标签']}(" + ", ".join(f"{k}={r[k]}" for k in keys) + ")" for r in tab[:3])
        return [f"- 抽样 {f['n']} 条（候选池 {f['pool']} 条，占比 {_pct(f['pool_rate'])}）："
                f"{sample}。{tag}"]

    return [f"- {res.get('note', '')} {tag}"]


def conclusion(results: list) -> List[str]:
    out = []
    iv = next((r for r in results if r["tool"] == "iv_table"), None)
    perf = next((r for r in results if r["tool"] == "perf_metrics"), None)
    cut = next((r for r in results if r["tool"] == "cutoff_tradeoff"), None)
    forb = next((r for r in results if r["tool"] == "forbidden_check"), None)
    if forb and forb.get("facts", {}).get("forbidden_used_count"):
        i = results.index(forb) + 1
        out.append(f"**模型当前不可用**：它使用了硬阻断字段，指标再好在合规上也过不去，"
                   f"应先剔除再评估。 [T{i}]")
    if iv and iv.get("facts", {}).get("iv_high"):
        i = results.index(iv) + 1
        out.append(f"**先查数据再谈模型**：{iv['facts']['iv_high_count']} 个变量 IV 异常偏高，"
                   f"典型原因是极端值未处理或标签泄漏。 [T{i}]")
    if perf and perf.get("facts", {}).get("ks"):
        i = results.index(perf) + 1
        out.append(f"模型区分度 KS {perf['facts']['ks']}，"
                   f"{'达到' if perf['facts']['ks'] >= perf['facts']['ks_floor'] else '未达到'}"
                   f"行业下限 {perf['facts']['ks_floor']}。 [T{i}]")
    if cut and cut.get("facts", {}).get("strict_capture"):
        i = results.index(cut) + 1
        out.append(f"若把通过率压到 {_pct(cut['facts']['strict_keep'])}，"
                   f"坏账率可降到 {_pct(cut['facts']['strict_bad_rate'])}，"
                   f"代价是拒绝掉更多正常客户。 [T{i}]")
    return out
