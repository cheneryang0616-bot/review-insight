# -*- coding: utf-8 -*-
"""credit 域 —— 统计层

三件在风控里"必须有"的东西：
1. WOE 分箱 + IV（变量筛选的标准做法，比直接裸特征更能看出单调性）
2. WOE + 逻辑回归评分卡（可解释，系数能翻译成业务语言）
3. KS / AUC / Gini / Lift + 通过率-坏账率曲线（决策层只看这个）

一个必要的诚实说明：Give Me Some Credit **没有时间列**，
所以这里的"验证集"只能随机切分，**不是 OOT（时间外）验证**，
也无法做真正的 PSI 监控。代码里 PSI 只用于检查"切分是否同分布"。
"""
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from . import config

_MODEL_CACHE: Dict[tuple, dict] = {}


# ---------------------------------------------------------------- 分箱 / WOE
def _binned(s: pd.Series, edges: np.ndarray) -> pd.Series:
    return pd.cut(s, bins=edges, include_lowest=True, duplicates="drop")


def woe_from_bins(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    g = pd.DataFrame({"bin": bins, "y": y}).groupby("bin", observed=True)["y"].agg(["sum", "count"])
    g["bad"] = g["sum"].astype(float)
    g["good"] = g["count"] - g["bad"]
    tot_bad, tot_good = float(y.sum()), float(len(y) - y.sum())
    if tot_bad <= 0 or tot_good <= 0:
        raise ValueError("标签只有单一类别（坏样本数或好样本数为 0），无法计算 WOE/IV")
    # 0.5 平滑：避免某个箱里没有坏客户导致 WOE 无穷大（风控里的标准处理）
    g["bad_rate"] = g["bad"] / g["count"]
    g["woe"] = np.log(((g["bad"] + 0.5) / tot_bad) / ((g["good"] + 0.5) / tot_good))
    g["iv_contrib"] = ((g["bad"] + 0.5) / tot_bad - (g["good"] + 0.5) / tot_good) * g["woe"]
    g["woe"] = g["woe"].replace([np.inf, -np.inf], np.nan).clip(-5, 5).fillna(0.0)
    return g


def feature_woe_iv(s: pd.Series, y: pd.Series, bins: int = None) -> dict:
    """单变量：等频分箱 → WOE / IV / 单调性。"""
    bins = bins or config.DEFAULT_BINS
    xs = s.copy()
    fill = xs.median()
    xs = xs.fillna(fill)
    if xs.nunique() <= bins:                      # 离散小基数变量：每个取值一个箱
        edges = np.unique(np.concatenate([[-np.inf], np.sort(xs.unique())[:-1] + 0.5,
                                          [np.inf]]))
    else:
        _, edges = pd.qcut(xs, bins, retbins=True, duplicates="drop")
        edges[0], edges[-1] = -np.inf, np.inf
    b = _binned(xs, edges)
    tab = woe_from_bins(b, y)
    woe = tab["woe"].values
    mono = bool(len(woe) <= 2 or np.all(np.diff(woe) >= 0) or np.all(np.diff(woe) <= 0))
    return {
        "edges": edges, "table": tab, "iv": float(tab["iv_contrib"].sum()),
        "monotone": mono, "bins": int(len(tab)), "fill": float(fill),
    }


def iv_table(df: pd.DataFrame, features: Optional[List[str]] = None,
             bins: int = None) -> List[dict]:
    feats = features or df.attrs.get("features", [])
    y = df["y"]
    rows = []
    for c in feats:
        r = feature_woe_iv(df[c], y, bins)
        rows.append({"feature": c, "iv": round(r["iv"], 4), "bins": r["bins"],
                     "monotone": r["monotone"]})
    rows.sort(key=lambda r: -r["iv"])
    return rows


# ---------------------------------------------------------------- 评分卡模型
def _prepare(df: pd.DataFrame, exclude_forbidden: bool, features=None) -> dict:
    feats = list(features or df.attrs.get("features", []))
    if exclude_forbidden:
        drop = set(df.attrs.get("forbidden_hard", [])) | set(df.attrs.get("forbidden_soft", []))
        feats = [c for c in feats if c not in drop]
    return {"features": feats}


def fit(df: pd.DataFrame, exclude_forbidden: bool = False,
        features: Optional[List[str]] = None) -> dict:
    """WOE 变换 + 逻辑回归。结果缓存（同一数据集同一配置只算一次）。"""
    key = (df.attrs.get("dataset"), bool(exclude_forbidden), tuple(features or ()))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    feats = _prepare(df, exclude_forbidden, features)["features"]
    y = df["y"]
    X = df[feats]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=1 - config.TRAIN_RATIO,
        random_state=config.RANDOM_STATE, stratify=y)

    maps, Xtr_w, Xte_w = {}, pd.DataFrame(index=Xtr.index), pd.DataFrame(index=Xte.index)
    used = []
    for c in feats:
        s_tr = Xtr[c]
        if s_tr.notna().sum() < config.MIN_BIN_COUNT or s_tr.nunique() < 2:
            continue
        r = feature_woe_iv(s_tr, ytr)
        # 用训练集学到的切点，映射验证集（防止信息泄漏）
        b_tr = _binned(s_tr.fillna(r["fill"]), r["edges"])
        b_te = _binned(Xte[c].fillna(r["fill"]), r["edges"])
        woe_map = r["table"]["woe"]
        Xtr_w[c] = b_tr.map(woe_map).astype(float).replace([np.inf, -np.inf], 0.0)
        Xte_w[c] = b_te.map(woe_map).astype(float).replace([np.inf, -np.inf], 0.0)
        maps[c] = {"edges": r["edges"], "woe": woe_map, "iv": r["iv"],
                   "monotone": r["monotone"], "bins": r["bins"]}
        used.append(c)

    if not used:
        raise ValueError("没有可用特征（样本量或基数不足）")

    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=config.RANDOM_STATE)
    lr.fit(Xtr_w.fillna(0.0), ytr)
    p = lr.predict_proba(Xte_w.fillna(0.0))[:, 1]

    auc = float(roc_auc_score(yte, p)) if yte.nunique() > 1 else float("nan")
    fpr, tpr, _ = roc_curve(yte, p)
    ks = float(np.max(tpr - fpr))
    res = {
        "features": used,
        "coef": {c: float(b) for c, b in zip(used, lr.coef_[0])},
        "intercept": float(lr.intercept_[0]),
        "maps": maps,
        "auc": auc, "ks": ks, "gini": 2 * auc - 1,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "bad_test": float(yte.mean()),
        "y_test": yte.values, "p_test": p,
    }
    _MODEL_CACHE[key] = res
    return res


def lift_at(p: np.ndarray, y: np.ndarray, frac: float = 0.1) -> float:
    k = max(1, int(len(p) * frac))
    idx = np.argsort(-p)[:k]
    base = y.mean()
    return float(y[idx].mean() / base) if base > 0 else float("nan")


def cutoff_tradeoff(y: np.ndarray, p: np.ndarray,
                    keeps=(0.9, 0.8, 0.7, 0.6, 0.5)) -> List[dict]:
    """按预测坏概率从低到高放行：通过率 → 实际坏账率 / 捕获的坏客户比例。"""
    order = np.argsort(p)
    total_bad = float(y.sum())
    rows = []
    for keep in keeps:
        k = max(1, int(len(p) * keep))
        idx = order[:k]
        approved = y[idx]
        captured = (total_bad - float(approved.sum())) / total_bad if total_bad else 0.0
        rows.append({"keep_rate": keep, "n_pass": int(k),
                     "bad_rate": round(float(approved.mean()), 4),
                     "capture": round(float(captured), 4)})
    return rows


def score_from_logit(logit: float, base: float = 600, pdo: float = 50,
                     base_odds: float = 20) -> float:
    """标准分映射：score = base + PDO/ln2 * (ln(odds_good:bad) - ln(base_odds))。"""
    odds_good_bad = math.exp(-logit)
    return base + (pdo / math.log(2)) * (math.log(max(odds_good_bad, 1e-12)) - math.log(base_odds))


def segment_bad_rate(df: pd.DataFrame, feature: str, bins: int = 5) -> List[dict]:
    s = df[feature]
    if s.nunique() <= bins:
        b = s
    else:
        _, edges = pd.qcut(s.fillna(s.median()), bins, retbins=True, duplicates="drop")
        b = _binned(s.fillna(s.median()), edges)
    g = pd.DataFrame({"bin": b, "y": df["y"]}).groupby("bin", observed=True)["y"].agg(["count", "mean"])
    rows = []
    for k, r in g.iterrows():
        if int(r["count"]) < config.MIN_N_SEGMENT:
            continue
        rows.append({"segment": str(k), "n": int(r["count"]),
                     "bad_rate": round(float(r["mean"]), 4)})
    return rows


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI：这里只用于检查"训练/验证切分是否同分布"，不是时间维度监控。

    PSI < 0.1 稳定；0.1~0.25 需关注；> 0.25 显著漂移。
    """
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, bins=edges)[0] / max(len(cur), 1)
    r = np.clip(r, 1e-6, None)
    c = np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))
