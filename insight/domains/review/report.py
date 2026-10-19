# -*- coding: utf-8 -*-
"""reviewinsight.report —— 图表与报告生成

注意：MPLCONFIGDIR 必须在 import pyplot 之前设置，否则 matplotlib 会去写
~/.matplotlib（在沙箱/CI 里往往不可写）而报错。中文字体同理，必须显式指定，
否则所有中文变方框。
"""
import os

import pandas as pd

from ...core import charts
from ...core.charts import plt
from . import config, data as data_mod, stats


def setup_font() -> str:
    return charts.setup_font()


def _save(fig, name: str) -> str:
    return charts.save(fig, os.path.join(config.FIG_DIR, name))


def fig_star(df: pd.DataFrame) -> str:
    sd = stats.star_dist(df)
    keys = sorted(sd, key=float)
    vals = [sd[k]["share"] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = ["#c0392b" if float(k) <= 2 else "#f0ad4e" if float(k) == 3 else "#2e86c1"
              for k in keys]
    ax.bar([f"{float(k):.0f}★" for k in keys], vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_ylabel("占比 (%)")
    ax.set_title(f"星级分布（n = {len(df):,}，差评口径 = 1~2 星，共 "
                 f"{df['bad'].sum():,} 条）")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig1_star_distribution.png")


def fig_deal_type(df: pd.DataFrame) -> str:
    plan = [(y, data_mod.scope_df(df, None, None, None, y)) for y in (None, 2020)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (year, d) in zip(axes, plan):
        rs = stats.deal_type_compare(d)
        labels = [r["type"] for r in rs]
        vals = [r["bad_rate"] * 100 for r in rs]
        bars = ax.bar(labels, vals, color=["#c0392b", "#2e86c1"])
        for b, v, r in zip(bars, vals, rs):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%\n(n={r['n']:,})",
                    ha="center", fontsize=8)
        title = "全量 2014-2021" if year is None else f"仅 {year} 年"
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.35)
        ax.set_ylabel("差评率 (%)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("代金券 vs 套餐：全量差距 1.89 倍，但 2020 年后收敛")
    return _save(fig, "fig2_deal_type.png")


def fig_shop(df: pd.DataFrame, top: int = 10) -> str:
    rows = stats.ranking(df, "shop", config.MIN_N_SHOP, top)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [f"{r['key']} (n={r['n']:,})" for r in rows]
    vals = [r["bad_rate"] * 100 for r in rows]
    ax.barh(labels, vals, color="#c0392b")
    overall = df["bad"].mean() * 100
    ax.axvline(overall, color="#555", linestyle="--", linewidth=1)
    ax.text(overall + 0.1, -0.6, f"全站均值 {overall:.2f}%", fontsize=8, color="#555")
    for i, v in enumerate(vals):
        ax.text(v + 0.1, i, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("差评率 (%)")
    ax.set_title(f"店铺差评率 Top{top}（仅样本量 ≥ {config.MIN_N_SHOP} 的店）")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig3_shop_ranking.png")


def fig_aspect(df: pd.DataFrame) -> str:
    ag = stats.aspect_gap(df[df["usable_text"]])
    rows = sorted(ag.items(), key=lambda kv: kv[1]["neg_diff"], reverse=True)
    names = [v["name"] for _, v in rows]
    bad = [v["neg_bad_rate"] * 100 for _, v in rows]
    good = [v["neg_good_rate"] * 100 for _, v in rows]
    y = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([i + 0.2 for i in y], bad, height=0.38, color="#c0392b", label="差评中负面率")
    ax.barh([i - 0.2 for i in y], good, height=0.38, color="#2e86c1", label="好评中负面率")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    for i, (b, g, (_, v)) in enumerate(zip(bad, good, rows)):
        ax.text(b + 0.15, i + 0.2, f"{b:.2f}%", va="center", fontsize=8)
        ax.text(g + 0.15, i - 0.2, f"{g:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("负面表述率 (%)")
    ax.set_title("方面归因：7 个业务方面的负面表述率（差评 vs 好评）")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig4_aspect_gap.png")


def fig_trend(df: pd.DataFrame) -> str:
    t = stats.trend(df, "Q")
    periods = [s["period"] for s in t["series"]]
    vals = [s["bad_rate"] * 100 for s in t["series"]]
    ns = [s["n"] for s in t["series"]]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(periods, vals, marker="o", color="#2e86c1", linewidth=1.8)
    mean = sum(v for v in vals) / len(vals)
    ax.axhline(mean, color="#999", linestyle=":", linewidth=1)
    ax.text(0, mean + 0.1, f"均值 {mean:.2f}%", fontsize=8, color="#666")
    for i, (p, v, n) in enumerate(zip(periods, vals, ns)):
        if n >= 500 and i % 4 == 0:
            ax.annotate(f"{v:.1f}%", (p, v), textcoords="offset points",
                        xytext=(0, 7), fontsize=7, ha="center")
    ax.set_title("季度差评率趋势（共 %d 期）" % len(periods))
    ax.set_ylabel("差评率 (%)")
    step = max(1, len(periods) // 10)
    ax.set_xticks(range(0, len(periods), step))
    ax.set_xticklabels([periods[i] for i in range(0, len(periods), step)], rotation=45, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig5_quarter_trend.png")


def fig_keywords(df: pd.DataFrame, top: int = 12) -> str:
    d = df[df["usable_text"]]
    bad, good = d[d["bad"] == 1], d[d["bad"] == 0]
    cb, cg = stats.corpus(bad["text"]), stats.corpus(good["text"])
    rows = stats.log_odds(cb, cg, sum(cb.values()), sum(cg.values()), top_k=top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh([r["word"] for r in rows], [r["z"] for r in rows], color="#7d3c98")
    for i, r in enumerate(rows):
        ax.text(r["z"] + 0.1, i, f"z={r['z']}", va="center", fontsize=8)
    ax.axvline(1.96, color="#c0392b", linestyle="--", linewidth=1)
    ax.text(2.05, -0.7, "显著性阈值 1.96", fontsize=8, color="#c0392b")
    ax.set_xlabel("log-odds z 值（越大 = 越集中于差评）")
    ax.set_title("差评倾向关键词（信息先验 log-odds）")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig6_keywords.png")


def build_all(df: pd.DataFrame) -> dict:
    font = setup_font()
    paths = {
        "font": font,
        "star": fig_star(df),
        "deal_type": fig_deal_type(df),
        "shop": fig_shop(df),
        "aspect": fig_aspect(df),
        "trend": fig_trend(df),
        "keywords": fig_keywords(df),
    }
    return paths
