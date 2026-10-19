#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成分析报告（含图表 + 真实数字 + Agent 问答示例）

用法：
    python scripts/build_report.py --domain review
    python scripts/build_report.py --domain credit            # 用当前数据
    RI_CREDIT_DATA=german python scripts/build_report.py --domain credit
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insight.core.agent import Agent
from insight.domains import get_domain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def pct(x):
    return f"{x * 100:.2f}%"


def load_eval(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def eval_section(domain, notes=""):
    out = ["## 评测（同一套问题，两条路径）\n"]
    eo = load_eval(f"eval_{domain}_offline.json")
    el = load_eval(f"eval_{domain}_llm.json")
    out.append("| 配置 | 通过 | 数字回查通过率 | 备注 |")
    out.append("|---|---|---|---|")
    if eo:
        out.append(f"| 规则规划 + 模板合成（离线） | **{eo['passed']}/{eo['total']}** | "
                   f"{eo.get('grounding_rate', 1.0)} | 可复现、零网络 |")
    if el:
        out.append(f"| LLM 规划 + LLM 合成 | {el['passed']}/{el['total']} | "
                   f"{el.get('grounding_rate', 0)} | 结论更会写，但选工具会漏、结果不稳定 |")
    out.append("")
    if el:
        out.append("LLM 路径的失败模式（真实跑出来的）：\n")
        for case in el["cases"]:
            if case["passed"]:
                continue
            detail = "；".join(c["detail"] for c in case["checks"] if not c["passed"])
            out.append(f"- `{case['id']}` {case['question']} → {detail}")
        out.append("")
    if notes:
        out.append(notes + "\n")
    return out


# ------------------------------------------------------------------ review
def build_review() -> str:
    from insight.domains.review import data as rdata, report as rreport, stats
    df = rdata.load()
    rreport.build_all(df)
    o = rdata.overview(df)
    usable = df[df["usable_text"]]
    ag = stats.aspect_gap(usable)
    ag_rows = sorted(ag.items(), key=lambda kv: -kv[1]["neg_diff"])
    dt = {r["type"]: r for r in stats.deal_type_compare(df)}
    dt20 = {r["type"]: r for r in stats.deal_type_compare(
        rdata.scope_df(df, None, None, None, 2020))}
    shops = stats.ranking(df, "shop", 200, 5)
    tr = stats.trend(df, "Q")
    last = tr["series"][-1]
    cb = stats.corpus(usable[usable.bad == 1]["text"])
    cg = stats.corpus(usable[usable.bad == 0]["text"])
    kws = stats.log_odds(cb, cg, sum(cb.values()), sum(cg.values()), top_k=8)
    slots = stats.slot_rates(df, "weekday")

    L = ["# 场景一：美团团购评论归因\n"]
    L.append(f"> {o['rows']:,} 条评论 · {o['shops']} 家店 · {o['start']} ~ {o['end']}；"
             f"差评口径 = 星级 ≤ 2（{o['bad_rows']:,} 条，{pct(o['bad_rate'])}）\n")
    L.append("## 数据体检\n")
    L.append(f"- 文本可用率 **{pct(o['text_usable_rate'])}**：{pct(o['text_short_rate'])} "
             f"的评论不足 4 字，文本类结论只覆盖 {len(usable):,} 条。")
    L.append(f"- 重复文本率 **{pct(o['dup_text_rate'])}**：短评高度模板化 → "
             f"关键词统计必须用带显著性的 log-odds，不能数频次。")
    L.append(f"- 套餐字段缺失率 **{pct(o['deal_missing_rate'])}**。\n")
    L.append("![星级分布](figures/review/fig1_star_distribution.png)\n")

    L.append("## 核心结论\n")
    L.append("### 1. 代金券类差评率是套餐类的 1.89 倍，但这是历史现象\n")
    L.append("![代金券vs套餐](figures/review/fig2_deal_type.png)\n")
    L.append(f"- 全量：代金券 **{pct(dt['代金券']['bad_rate'])}** vs 套餐 "
             f"**{pct(dt['套餐']['bad_rate'])}**（{dt['代金券']['bad_rate'] / dt['套餐']['bad_rate']:.2f} 倍）。")
    L.append(f"- 只看 2020：代金券 **{pct(dt20['代金券']['bad_rate'])}** vs 套餐 "
             f"**{pct(dt20['套餐']['bad_rate'])}** —— 差距反转。")
    L.append("- 说明问题出在**早期履约规则**，不是当前出品质量；只看全量均值会误判成持续恶化。\n")

    L.append("### 2. 差评真正在骂什么\n")
    L.append("![方面归因](figures/review/fig4_aspect_gap.png)\n")
    L.append("| 方面 | 差评负面率 | 好评负面率 | 倍差 | 差评命中数 |")
    L.append("|---|---|---|---|---|")
    for _, v in ag_rows:
        L.append(f"| {v['name']} | {pct(v['neg_bad_rate'])} | {pct(v['neg_good_rate'])} | "
                 f"{v['neg_lift']}× | {v['bad_neg_count']} |")
    L.append("")
    best = max(ag.items(), key=lambda kv: kv[1]["neg_lift"] or 0)[1]
    L.append(f"- 绝对量第一：**{ag_rows[0][1]['name']}**；区分度第一："
             f"**{best['name']}**（{best['neg_lift']} 倍）。\n")

    L.append("### 3. 关键词差异与食品安全信号\n")
    L.append("![关键词](figures/review/fig6_keywords.png)\n")
    kw = {r["word"]: r for r in kws}
    if "拉肚子" in kw:
        L.append(f"- 「拉肚子」差评 {kw['拉肚子']['bad']} 次 vs 好评 {kw['拉肚子']['good']} 次"
                 f"（z={kw['拉肚子']['z']}）：食品安全信号，优先级高于口味改进。\n")

    L.append("### 4. 门店差异远大于时段差异\n")
    L.append("![店铺排名](figures/review/fig3_shop_ranking.png)\n")
    L.append(f"- 店铺极差 **{(shops[0]['bad_rate'] - shops[-1]['bad_rate']) * 100:.1f} 个百分点**；"
             f"星期极差仅 **{(max(s['bad_rate'] for s in slots) - min(s['bad_rate'] for s in slots)) * 100:.1f}** 个百分点。\n")

    L.append("### 5. 时间趋势\n")
    L.append("![趋势](figures/review/fig5_quarter_trend.png)\n")
    L.append(f"- 最近一期 {last['period']} 差评率 **{pct(last['bad_rate'])}**，"
             f"历史均值 {pct(last.get('baseline_mean', 0))}，z = {last.get('z')}"
             f"（{'异动' if last.get('is_anomaly') else '正常波动'}）。\n")

    L.extend(eval_section("review", "规则规划器在这个领域完胜：意图空间窄（排名/归因/趋势/时段/样例），"
                                    "而 LLM 会漏工具、改粒度、丢参数。"))
    L.append("## 问答示例\n")
    agent = Agent(get_domain("review"))
    for q in ["代金券和套餐哪个差评更多？", "为什么差评这么多？短板在哪？"]:
        res = agent.ask(q, use_llm=False)
        L.append(f"### Q：{q}\n")
        L.append(f"`工具链：{' → '.join(c['tool'] for c in res['plan'])}`\n")
        L.append(res["answer"] + "\n")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ credit
def build_credit() -> str:
    from insight.domains.credit import config as cconfig
    from insight.domains.credit import data as cdata, report as creport, stats as cstats
    df = cdata.load()
    creport.build_all(df)
    p = cdata.profile(df)
    ivs = cstats.iv_table(df)
    m = cstats.fit(df)
    safe = cstats.fit(df, exclude_forbidden=True)
    cut = cstats.cutoff_tradeoff(m["y_test"], m["p_test"])
    agent = Agent(get_domain("credit"))

    L = ["# 场景二：信用风险建模与合规\n"]
    L.append(f"> 数据集 **{p['dataset']}** · {p['rows']:,} 条 · {p['features']} 个字段 · "
             f"坏客户 {p['bad_n']:,}（{pct(p['bad_rate'])}）\n")
    L.append("## 数据体检\n")
    L.append(f"- 不平衡比约 **1:{(p['imbalance'] or 0):.1f}**；缺失最严重 "
             f"{pct(p['missing_max'])}；{p['extreme_cols']} 个字段存在极端值。")
    L.append(f"- **无时间列** → 只能随机切分，不能做 OOT 验证与跨期 PSI。")
    if p["forbidden_hard"]:
        L.append(f"- 含 **{p['forbidden_hard']} 个硬阻断字段**（性别/婚姻、外籍劳工）。")
    L.append("")

    L.append("## 1. IV：最高的变量不是「最强」，而是「最可疑」\n")
    L.append("![IV 排序](figures/credit/fig1_iv.png)\n")
    L.append("| 变量 | IV | 分箱 | WOE 单调 |")
    L.append("|---|---|---|---|")
    for r in ivs[:6]:
        L.append(f"| {cconfig.label(r['feature'])} | {r['iv']} | {r['bins']} | "
                 f"{'是' if r['monotone'] else '否'} |")
    L.append("")
    high = [r for r in ivs if r["iv"] > cconfig.IV_SUSPICIOUS]
    nonmono = [r for r in ivs if not r["monotone"]]
    L.append(f"- **{len(high)} 个变量 IV > {cconfig.IV_SUSPICIOUS}**，"
             f"**{len(nonmono)} 个变量 WOE 非单调**。")
    L.append("- 教科书把 IV>0.5 叫「极强变量」，实务上第一步是查极端值与标签泄漏 —— "
             "这条判断写进了断言闸门（`iv_ceiling`），不是靠人记住。\n")

    L.append("## 2. 分箱单调性\n")
    L.append("![WOE 曲线](figures/credit/fig2_woe.png)\n")

    L.append("## 3. 模型表现\n")
    L.append("![KS](figures/credit/fig3_ks.png)\n")
    L.append(f"- **AUC {m['auc']:.4f} / KS {m['ks']:.4f} / Gini {m['gini']:.4f}**，"
             f"训练 {m['n_train']} / 验证 {m['n_test']}。")
    L.append(f"- KS 行业下限 {cconfig.KS_FLOOR}：{'达标' if m['ks'] >= cconfig.KS_FLOOR else '未达标'}。\n")

    L.append("## 4. 阈值权衡\n")
    L.append("![通过率权衡](figures/credit/fig4_cutoff.png)\n")
    L.append("| 通过率 | 通过人数 | 实际坏账率 | 捕获坏客户 |")
    L.append("|---|---|---|---|")
    for r in cut:
        L.append(f"| {pct(r['keep_rate'])} | {r['n_pass']} | {pct(r['bad_rate'])} | "
                 f"{pct(r['capture'])} |")
    L.append("")

    L.append("## 5. 评分刻度与分群\n")
    L.append("![评分分布](figures/credit/fig5_score.png)\n")
    L.append("![分群坏账率](figures/credit/fig6_segment.png)\n")

    L.append("## 6. 合规：这是与场景一完全不同的「错误形态」\n")
    L.append("场景一的错误是「数字算错」，这里的错误是**数字全对但结论不可用**。")
    L.append("")
    r = agent.ask("模型效果怎么样？", use_llm=False)
    L.append("### Q：模型效果怎么样？\n")
    L.append(r["answer"] + "\n")
    from insight.core import gates as G, grounding
    L.append("> 质检：" + grounding.format_report(r["grounding"]))
    L.append("> 闸门：" + G.format_report(r["gates"]) + "\n")

    r2 = agent.ask("剔除性别和外籍劳工字段后模型效果怎么样？", use_llm=False)
    L.append("### Q：剔除性别和外籍劳工字段后模型效果怎么样？\n")
    L.append(r2["answer"] + "\n")
    L.append("> 闸门：" + G.format_report(r2["gates"]) + "\n")
    L.append(f"- 剔除后特征数 {len(m['features'])} → {len(safe['features'])}，"
             f"KS {m['ks']:.4f} → {safe['ks']:.4f}，AUC {m['auc']:.4f} → {safe['auc']:.4f}。")
    L.append("- 结论：**去掉歧视性变量几乎不损失区分度**，这就把「合规」从口号变成了可执行的取舍。\n")

    L.extend(eval_section("credit", "注意 C06/C07/C08：闸门对同一句「模型效果怎么样」给出"
                                    "不同的结论（阻断 / 修复后通过），这不是随机，是模型口径不同。"))
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="review", choices=["review", "credit"])
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    text = build_review() if args.domain == "review" else build_credit()
    out = os.path.join(RES, f"REPORT_{args.domain}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"报告已生成：{out}（{len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
