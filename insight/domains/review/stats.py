# -*- coding: utf-8 -*-
"""reviewinsight.stats —— 统计层（纯计算，不碰 IO、不碰 agent）

这里放三个"有技术含量"的东西，也是回答和面试的核心：
1. 方面提及率对照（bad vs good）→ 回答"差评到底在骂什么"
2. Monroe 等的**信息先验对数几率比**（log-odds ratio with informative Dirichlet prior）
   → 回答"差评和好评在词汇上到底差在哪"，比 wordcloud / TF-IDF 严谨得多
3. 时间趋势 + 基线 z 值异动检测 → 回答"是不是最近变差了"
"""
import math
from collections import Counter
from typing import Dict, List, Optional

import jieba
import numpy as np
import pandas as pd

from . import config, lexicon, data as data_mod

_jieba_ready = False


def _ensure_jieba():
    global _jieba_ready
    if not _jieba_ready:
        for w in lexicon.JIEBA_WORDS:
            jieba.add_word(w)
        _jieba_ready = True


def tokens(text: str) -> List[str]:
    _ensure_jieba()
    out = []
    for w in jieba.cut(text):
        w = w.strip()
        if len(w) < 2:
            continue
        if w in lexicon.STOPWORDS:
            continue
        if w.isdigit():
            continue
        out.append(w)
    return out


def corpus(texts) -> Counter:
    c = Counter()
    for t in texts:
        c.update(tokens(lexicon.apply_noise(t)))
    return c


def log_odds(c_bad: Counter, c_good: Counter, n_bad: int, n_good: int,
             min_count: Optional[int] = None, top_k: int = 15) -> List[dict]:
    """Monroe, Colaresi & Quinn (2008) 的信息先验对数几率比。

    delta_i = log((y_i+a_i)/(n_y+A-y_i-a_i)) - log((x_i+a_i)/(n_x+A-x_i-a_i))
    var_i   = 1/(y_i+a_i) + 1/(x_i+a_i)
    z_i     = delta_i / sqrt(var_i)

    其中 y=差评语料, x=好评语料, a_i = A * (x_i+y_i)/(n_x+n_y) 是信息先验，
    A 取 5000（论文推荐的弱先验强度）。相比 TF-IDF，它自带显著性，
    不会把"只出现 2 次但全在差评里"的偶然词排到前面。
    """
    min_count = min_count or config.LOGO_ODDS_MIN_COUNT
    A = config.LOGO_ODDS_ALPHA0 or 5000.0
    total = n_bad + n_good
    vocab = set(c_bad) | set(c_good)
    rows = []
    for w in vocab:
        y, x = c_bad.get(w, 0), c_good.get(w, 0)
        if y + x < min_count:
            continue
        a_i = A * (x + y) / max(total, 1)
        p_y = (y + a_i) / max(n_bad + A - y - a_i, 1e-9)
        p_x = (x + a_i) / max(n_good + A - x - a_i, 1e-9)
        delta = math.log(max(p_y, 1e-12)) - math.log(max(p_x, 1e-12))
        var = 1.0 / (y + a_i) + 1.0 / (x + a_i)
        z = delta / math.sqrt(var)
        rows.append({
            "word": w, "bad": y, "good": x,
            "delta": round(delta, 3), "z": round(z, 2),
            "lift": round((y / max(n_bad, 1)) / (x / max(n_good, 1)), 2) if x else None,
        })
    rows.sort(key=lambda r: -r["z"])
    return rows[:top_k]


def aspect_gap(df: pd.DataFrame) -> Dict[str, dict]:
    """方面归因：负面表述率 + 提及率，差评 vs 好评双列对照。

    两个指标都要输出，因为它们的含义完全不同：
    - mention_rate：这个方面"被谈到"的比例（中性，区分度低）
    - negative_rate：这个方面"被负面表述"的比例（有区分度，用来定位短板）
    - lift = 差评负面率 / 好评负面率
    - diff = 差评负面率 - 好评负面率（百分点，用来排序"先修哪个"）
    """
    bad = df[df["bad"] == 1]
    good = df[df["bad"] == 0]
    nb, ng = len(bad), max(len(good), 1)

    bad_neg = [set(lexicon.negative_hits(t)) for t in bad["text"]]
    good_neg = [set(lexicon.negative_hits(t)) for t in good["text"]]
    bad_men = [set(lexicon.aspect_hits(t)) for t in bad["text"]]
    good_men = [set(lexicon.aspect_hits(t)) for t in good["text"]]

    out = {}
    for key, (name, _, negs, action) in lexicon.ASPECTS.items():
        cnb = sum(1 for s in bad_neg if key in s)
        cng = sum(1 for s in good_neg if key in s)
        cmb = sum(1 for s in bad_men if key in s)
        cmg = sum(1 for s in good_men if key in s)
        rnb, rng = cnb / nb if nb else 0.0, cng / ng
        rmb, rmg = cmb / nb if nb else 0.0, cmg / ng
        out[key] = {
            "name": name,
            "neg_bad_rate": round(rnb, 4),
            "neg_good_rate": round(rng, 4),
            "neg_lift": round(rnb / rng, 2) if rng else None,
            "neg_diff": round(rnb - rng, 4),
            "mention_bad_rate": round(rmb, 4),
            "mention_good_rate": round(rmg, 4),
            "bad_neg_count": cnb,
            "action": action,
        }
    return out


def star_dist(df: pd.DataFrame) -> Dict[str, dict]:
    vc = df["star"].value_counts().sort_index()
    total = int(vc.sum())
    return {str(k): {"count": int(v), "share": round(v / total, 4)} for k, v in vc.items()}


def ranking(df: pd.DataFrame, by: str, min_n: int, top_k: int = 10) -> List[dict]:
    """按差评率排名（只保留样本量达标的分组）。"""
    g = df.groupby(by)["bad"].agg(["sum", "count", "mean"])
    g = g[g["count"] >= min_n].sort_values("mean", ascending=False)
    total_bad = df["bad"].mean()
    return [{
        "key": str(k),
        "n": int(r["count"]),
        "bad": int(r["sum"]),
        "bad_rate": round(float(r["mean"]), 4),
        "vs_overall": round(float(r["mean"]) - total_bad, 4),
    } for k, r in g.head(top_k).iterrows()]


def trend(df: pd.DataFrame, freq: Optional[str] = None) -> dict:
    """时间趋势 + 最近一期相对历史基线的 z 值。"""
    freq = freq or config.TREND_FREQ
    key = {"Q": "quarter", "M": "month", "Y": "year"}[freq]
    g = df.groupby(key)["bad"].agg(["sum", "count", "mean"])
    g = g[g["count"] > 0]
    series = [{"period": str(k), "n": int(r["count"]),
               "bad_rate": round(float(r["mean"]), 4)} for k, r in g.iterrows()]
    if len(series) >= 3:
        hist = np.array([s["bad_rate"] for s in series[:-1]])
        last = series[-1]
        mu, sd = float(hist.mean()), float(hist.std(ddof=1))
        z = (last["bad_rate"] - mu) / sd if sd > 1e-9 else 0.0
        last["baseline_mean"] = round(mu, 4)
        last["z"] = round(float(z), 2)
        last["is_anomaly"] = bool(abs(z) >= config.ANOMALY_Z)
    return {"freq": freq, "series": series}


def slot_rates(df: pd.DataFrame, dim: str = "weekday") -> List[dict]:
    g = df.groupby(dim)["bad"].agg(["sum", "count", "mean"])
    g = g[g["count"] >= config.MIN_N_SLOT]
    names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    out = []
    for k, r in g.iterrows():
        out.append({
            "slot": names[int(k)] if dim == "weekday" else f"{int(k)}:00",
            "n": int(r["count"]),
            "bad_rate": round(float(r["mean"]), 4),
        })
    return out


def deal_type_compare(df: pd.DataFrame) -> List[dict]:
    """代金券 vs 套餐：本项目最强的一个业务结论来源。"""
    g = df[df["deal_type"].isin(["代金券", "套餐"])].groupby("deal_type")["bad"].agg(
        ["sum", "count", "mean"])
    out = []
    for k, r in g.iterrows():
        out.append({"type": str(k), "n": int(r["count"]), "bad": int(r["sum"]),
                    "bad_rate": round(float(r["mean"]), 4)})
    if len(out) == 2:
        hi = max(out, key=lambda r: r["bad_rate"])
        lo = min(out, key=lambda r: r["bad_rate"])
        if lo["bad_rate"] > 0:
            ratio = hi["bad_rate"] / lo["bad_rate"]
            for o in out:
                o["ratio_vs_other"] = round(ratio, 2) if o is hi else round(1 / ratio, 2)
    return out
