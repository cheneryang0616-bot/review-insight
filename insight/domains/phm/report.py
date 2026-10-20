# -*- coding: utf-8 -*-
"""phm 域 —— 4 张交付图

四张图各自回答一个决策问题，而不是"好看"：
1. 机队紧急度：哪几台进了两周窗口
2. 单机区间：这台的点估计和区间有多宽（以及真值落在哪）
3. 校准曲线：这个区间到底能不能用（**最重要的一张**）
4. 代价曲线：告警线定在哪、相比默认值省多少

数据全部来自 tools/decision 的同一批计算，不另算一套，避免图文不一致。
"""
import math
import os

import numpy as np
import pandas as pd

from ...core import charts
from ...core.charts import plt
from . import config, data as data_mod, decision


def _save(fig, name):
    return charts.save(fig, os.path.join(config.FIG_DIR, name))


def _pred_arrays(df):
    pred = df["pred"].to_numpy(dtype=float)
    sigma = df["sigma"].to_numpy(dtype=float)
    y = df["y_true"].to_numpy(dtype=float)
    unit = df["unit"].to_numpy(dtype=int)
    return pred, sigma, y, unit


def fig_fleet(df: pd.DataFrame) -> str:
    pred, sigma, y, unit = _pred_arrays(df)
    order = np.argsort(pred)
    keep = order[:25]
    labels = [f"#{unit[i]}" for i in keep]
    values = pred[keep]
    colors = ["#c0392b" if pred[i] <= config.SERVICE_NEAR else "#e67e22" for i in keep]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    window_cycles = config.WINDOW_DAYS * config.CYCLES_PER_DAY
    for x, text, color in ((window_cycles, f"{config.WINDOW_DAYS} 天维护窗口（{window_cycles:g} 循环）", "#c0392b"),
                           (config.SERVICE_NEAR, f"{config.SERVICE_NEAR} 循环警戒线", "#555")):
        ax.axvline(x, color=color, linestyle="--", linewidth=1)
        ax.text(x + 0.4, -0.9, text, fontsize=8, color=color)
    for i, v in enumerate(values):
        ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=7.5)
    n_near = int((pred <= config.SERVICE_NEAR).sum())
    n_window = int((pred <= window_cycles).sum())
    ax.set_xlabel("预测剩余寿命 RUL（运行循环，1 循环 = 1 天为假设）")
    ax.set_title(f"机队紧急度 Top{len(keep)}（共 {len(pred)} 台；"
                 f"≤{config.SERVICE_NEAR} 循环 {n_near} 台，≤{window_cycles:g} 循环 {n_window} 台）")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig1_fleet_urgency.png")


def fig_unit_interval(df: pd.DataFrame, top: int = 10) -> str:
    pred, sigma, y, unit = _pred_arrays(df)
    order = np.argsort(pred)[:top]
    labels = [f"#{unit[i]}" for i in order]
    p, s, t = pred[order], sigma[order], y[order]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ypos = np.arange(len(order))
    ax.errorbar(p, ypos, xerr=config.INTERVAL_K * s, fmt="o", color="#2e86c1",
                ecolor="#5dade2", elinewidth=2, capsize=4, label=f"预测 ± {config.INTERVAL_K}σ")
    ax.scatter(t, ypos, marker="x", color="#c0392b", s=45, label="真实 RUL（离线复盘）", zorder=5)
    for i, (pv, tv) in enumerate(zip(p, t)):
        ax.plot([pv, pv], [i - 0.28, i + 0.28], color="#2e86c1", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    inside = int(np.sum((t >= p - config.INTERVAL_K * s) & (t <= p + config.INTERVAL_K * s)))
    ax.set_xlabel("剩余寿命 RUL（运行循环）")
    ax.set_title(f"最紧急 {len(order)} 台的预测区间：{inside}/{len(order)} 台真值落在区间内"
                 f"（全机队 {data_mod.coverage(df):.0%}，名义 95%）")
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig2_unit_interval.png")


def fig_calibration(df: pd.DataFrame) -> str:
    pred, sigma, y, _ = _pred_arrays(df)
    ks = np.linspace(0.25, 3.0, 40)
    nominal = np.array([math.erf(k / math.sqrt(2.0)) for k in ks])
    empirical = np.array([float(np.mean(np.abs(y - pred) <= k * sigma)) for k in ks])
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot([0, 100], [0, 100], linestyle="--", color="#999", label="理想校准（实测 = 名义）")
    ax.plot(nominal * 100, empirical * 100, color="#c0392b", linewidth=2.2, label="实测")
    k0 = config.INTERVAL_K
    nom0 = math.erf(k0 / math.sqrt(2.0)) * 100
    emp0 = float(np.mean(np.abs(y - pred) <= k0 * sigma)) * 100
    ax.scatter([nom0], [emp0], color="#c0392b", s=60, zorder=5)
    ax.annotate(f"k={k0}：名义 {nom0:.0f}% → 实测 {emp0:.0f}%\n缺口 {nom0 - emp0:.0f} 个百分点",
                xy=(nom0, emp0), xytext=(nom0 - 42, emp0 + 12), fontsize=9, color="#7b241c",
                arrowprops=dict(arrowstyle="->", color="#7b241c", linewidth=1))
    ax.axvline(config.MIN_COVERAGE * 100, color="#2e86c1", linestyle=":", linewidth=1)
    ax.text(config.MIN_COVERAGE * 100 + 1, 5, f"决策准入线 {config.MIN_COVERAGE:.0%}",
            fontsize=8, color="#2e86c1", rotation=90, va="bottom")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("名义覆盖率（%）")
    ax.set_ylabel("实测覆盖率（%）")
    ax.set_title("校准曲线：区间偏窄，名义 95% 实际只有 "
                 f"{emp0:.0f}%（{df.attrs.get('dataset')}，{len(df)} 台）")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig3_calibration.png")


def fig_cost_curve(df: pd.DataFrame) -> str:
    result = decision.cost_threshold(df, c_fn=config.C_FN, c_fp=config.C_FP,
                                    risk_horizon_cycles=config.RISK_HORIZON_CYCLES,
                                    default_threshold=config.SERVICE_NEAR)
    curve = pd.DataFrame(result["threshold_curve"])
    best_t = result["threshold_rul"]
    best_c = result["expected_cost"]
    default_c = result["default_cost"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(curve["threshold_rul"], curve["expected_cost"], color="#2e86c1", linewidth=1.6,
            marker="o", markersize=2.5, label="机队期望代价")
    ax.scatter([best_t], [best_c], color="#c0392b", s=60, zorder=5,
               label=f"最优告警线 {best_t:.2f}（代价 {best_c:.2f}）")
    ax.scatter([result["default_threshold"]], [default_c], color="#7f8c8d", s=50, zorder=5,
               label=f"默认 {result['default_threshold']:g}（代价 {default_c:.2f}）")
    ax.annotate(f"降低 {result['compare_to_default']:.2f} 假设成本单位\n"
                f"（漏报:误报 = {result['cost_ratio']:.0f}:1，未校准 σ 的情景分析）",
                xy=(best_t, best_c), xytext=(best_t + 12, best_c + 118),
                fontsize=9, color="#7b241c",
                arrowprops=dict(arrowstyle="->", color="#7b241c", linewidth=1))
    ax.set_xlabel("告警线：预测 RUL ≤ 阈值即告警（循环）")
    ax.set_ylabel("机队期望代价（假设成本单位）")
    ax.set_title(f"代价敏感告警线：风险事件 RUL < {config.RISK_HORIZON_CYCLES:g} 循环")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "fig4_cost_curve.png")


def build_all(df: pd.DataFrame) -> dict:
    font = charts.setup_font()
    return {"font": font,
            "fleet": fig_fleet(df),
            "interval": fig_unit_interval(df),
            "calibration": fig_calibration(df),
            "cost": fig_cost_curve(df)}
