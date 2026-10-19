# -*- coding: utf-8 -*-
"""credit 域 —— 6 张图：IV 排序 / WOE 曲线 / KS / 通过率权衡 / 分数分布 / 分群坏账率"""
import os

import numpy as np
import pandas as pd

from ...core import charts
from ...core.charts import plt
from . import config, data as data_mod, stats


def _save(fig, name):
    return charts.save(fig, os.path.join(config.FIG_DIR, name))


def fig_iv(df: pd.DataFrame) -> str:
    rows = stats.iv_table(df)
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [config.label(r["feature"]) for r in rows]
    vals = [r["iv"] for r in rows]
    colors = ["#c0392b" if v > config.IV_SUSPICIOUS else "#2e86c1" for v in vals]
    ax.barh(labels, vals, color=colors)
    ax.axvline(config.IV_SUSPICIOUS, color="#c0392b", linestyle="--", linewidth=1)
    ax.text(config.IV_SUSPICIOUS + 0.02, -0.7, "IV=0.5：超过先查泄漏/极端值",
            fontsize=8, color="#c0392b")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("IV（信息值）")
    ax.set_title(f"变量 IV 排序（{df.attrs.get('dataset')}，n={len(df):,}）")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig1_iv.png")


def fig_woe(df: pd.DataFrame) -> str:
    rows = stats.iv_table(df)
    top = rows[0]["feature"]
    r = stats.feature_woe_iv(df[top], df["y"])
    tab = r["table"]
    x = np.arange(len(tab))
    fig, ax1 = plt.subplots(figsize=(8, 3.8))
    ax1.bar(x, tab["woe"].values, color="#7d3c98", alpha=0.85, label="WOE")
    ax1.set_ylabel("WOE")
    ax1.axhline(0, color="#999", linewidth=0.8)
    ax2 = ax1.twinx()
    ax2.plot(x, tab["bad_rate"].values * 100, color="#c0392b", marker="o", linewidth=1.6,
             label="坏账率")
    ax2.set_ylabel("坏账率 (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k)[:16] for k in tab.index], rotation=45, ha="right", fontsize=7)
    mono = "单调" if r["monotone"] else "非单调"
    ax1.set_title(f"{config.label(top)} 的分箱 WOE 与坏账率（IV={r['iv']:.3f}，{mono}）")
    ax1.spines[["top"]].set_visible(False)
    return _save(fig, "fig2_woe.png")


def fig_ks(df: pd.DataFrame) -> str:
    m = stats.fit(df, exclude_forbidden=False)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(m["y_test"], m["p_test"])
    ks_idx = int(np.argmax(tpr - fpr))
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr * 100, tpr * 100, color="#2e86c1", linewidth=2, label=f"AUC={m['auc']:.4f}")
    ax.plot([0, 100], [0, 100], color="#bbb", linestyle="--", linewidth=1)
    ax.vlines(fpr[ks_idx] * 100, fpr[ks_idx] * 100, tpr[ks_idx] * 100,
              color="#c0392b", linewidth=2, label=f"KS={m['ks']:.4f}")
    ax.set_xlabel("假正率 FPR (%)")
    ax.set_ylabel("真正率 TPR (%)")
    ax.set_title("验证集 ROC 与 KS")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig3_ks.png")


def fig_cutoff(df: pd.DataFrame) -> str:
    m = stats.fit(df, exclude_forbidden=False)
    rows = stats.cutoff_tradeoff(m["y_test"], m["p_test"],
                                 (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4))
    keeps = [r["keep_rate"] * 100 for r in rows]
    bad = [r["bad_rate"] * 100 for r in rows]
    cap = [r["capture"] * 100 for r in rows]
    fig, ax1 = plt.subplots(figsize=(7.5, 3.8))
    ax1.plot(keeps, bad, marker="o", color="#c0392b", label="通过客户的实际坏账率")
    ax1.set_xlabel("通过率 (%)")
    ax1.set_ylabel("坏账率 (%)", color="#c0392b")
    ax1.invert_xaxis()
    ax2 = ax1.twinx()
    ax2.plot(keeps, cap, marker="s", color="#2e86c1", label="捕获坏客户比例")
    ax2.set_ylabel("捕获坏客户 (%)", color="#2e86c1")
    ax1.set_title("通过率 vs 坏账率 vs 坏客户捕获率（验证集）")
    ax1.spines[["top"]].set_visible(False)
    return _save(fig, "fig4_cutoff.png")


def fig_score(df: pd.DataFrame) -> str:
    m = stats.fit(df, exclude_forbidden=False)
    p = m["p_test"]
    logits = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
    scores = np.array([stats.score_from_logit(l) for l in logits])
    y = m["y_test"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(scores[y == 0], bins=30, alpha=0.65, color="#2e86c1", label="好客户")
    ax.hist(scores[y == 1], bins=30, alpha=0.65, color="#c0392b", label="坏客户")
    ax.set_xlabel("信用分（base=600，PDO=50）")
    ax.set_ylabel("人数")
    ax.set_title("评分分布：好坏客户分离度")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig5_score.png")


def fig_segment(df: pd.DataFrame) -> str:
    feats = df.attrs.get("features", [])
    feat = "age" if "age" in feats else feats[0]
    rows = stats.segment_bad_rate(df, feat, 5)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels = [r["segment"] for r in rows]
    vals = [r["bad_rate"] * 100 for r in rows]
    ax.bar(range(len(rows)), vals, color="#f0ad4e")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{s}\n(n={r['n']})" for s, r in zip(labels, rows)],
                       rotation=30, ha="right", fontsize=7)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, f"{v:.2f}%", ha="center", fontsize=8)
    ax.set_ylabel("坏账率 (%)")
    ax.set_title(f"按 {config.label(feat)} 分群的坏账率")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig6_segment.png")


def build_all(df: pd.DataFrame) -> dict:
    font = charts.setup_font()
    return {"font": font, "iv": fig_iv(df), "woe": fig_woe(df), "ks": fig_ks(df),
            "cutoff": fig_cutoff(df), "score": fig_score(df), "segment": fig_segment(df)}
