# -*- coding: utf-8 -*-
"""reviewinsight.tools —— 工具层

设计原则（这是整个项目最关键的架构决策）：

1. **工具是唯一的"事实来源"**。LLM 不许自己编数字，它只能读工具返回的 facts。
2. **每个工具返回结构化 facts（扁平数字字典）+ table（人看的表）+ note（口径说明）**。
   facts 是给 groundedness 质检回查用的，table 是给 LLM 写答案用的。
3. **切片参数统一**：shop / deal / deal_type / year，由 data.scope_df 统一实现，
   工具层禁止自己写 df[...] 过滤，否则口径一定会漂。
4. **样本量门槛写死在 config**：1 条评论的 100% 差评率不是洞察，是噪声。
"""
from typing import List, Optional

import pandas as pd

from ...core.registry import Registry
from . import config, data as data_mod, stats

REGISTRY = Registry("review")
tool = REGISTRY.tool            # 装饰器签名不变：@tool(name, desc, params)


def _result(tool_name: str, params: dict, table: List[dict], facts: dict,
            note: str = "") -> dict:
    return {"tool": tool_name, "params": params, "table": table,
            "facts": facts, "note": note}


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# --------------------------------------------------------------- 1. 数据体检
@tool("dataset_overview", "数据集体检：行数、店铺数、时间跨度、总体差评率、文本可用率、缺失率",
      {})
def dataset_overview(df: pd.DataFrame) -> dict:
    o = data_mod.overview(df)
    table = [{"指标": k, "值": v} for k, v in o.items()]
    return _result("dataset_overview", {}, table, o,
                   note=f"差评口径 = 星级 <= {config.BAD_STAR_MAX}；"
                        f"文本可用 = 清洗后长度 >= {config.MIN_TEXT_LEN}")


# --------------------------------------------------------------- 2. 星级分布
@tool("star_distribution", "星级分布（1~5 星各档的条数与占比）",
      {"shop": "店铺ID，可选", "deal": "套餐名，可选",
       "deal_type": "代金券/套餐，可选", "year": "年份，可选"})
def star_distribution(df: pd.DataFrame, shop=None, deal=None,
                      deal_type=None, year=None) -> dict:
    d = data_mod.scope_df(df, shop, deal, deal_type, year)
    if d.empty:
        return _result("star_distribution", dict(shop=shop, deal=deal), [],
                       {"n": 0}, note="该切片无数据")
    sd = stats.star_dist(d)
    table = [{"星级": k, "条数": v["count"], "占比": _pct(v["share"])}
             for k, v in sd.items()]
    bad_n = int(d["bad"].sum())
    facts = {"n": int(len(d)), "bad_n": bad_n,
             "bad_rate": round(float(d["bad"].mean()), 4),
             "mean_star": round(float(d["star"].mean()), 3)}
    facts.update({f"share_{k}": v["share"] for k, v in sd.items()})
    return _result("star_distribution", dict(shop=shop, deal=deal, year=year), table, facts,
                   note=f"范围：{data_mod.scope_desc(shop, deal, deal_type, year)}")


# ------------------------------------------------------------ 3. 店铺排名
@tool("shop_ranking", "按差评率给店铺排名（只统计样本量达标的店铺）",
      {"min_n": "最小样本量，默认 200", "top_k": "返回条数，默认 8"})
def shop_ranking(df: pd.DataFrame, min_n: int = None, top_k: int = 8) -> dict:
    min_n = min_n or config.MIN_N_SHOP
    rows = stats.ranking(df, "shop", min_n, top_k)
    if not rows:
        return _result("shop_ranking", {}, [], {"shops_compared": 0},
                       note=f"没有店铺达到样本量门槛 {min_n}")
    table = [{"店铺ID": r["key"], "评论数": r["n"], "差评数": r["bad"],
              "差评率": _pct(r["bad_rate"]),
              "高于平均": f"{r['vs_overall']*100:+.2f}pt"} for r in rows]
    stats_all = df.groupby("shop")["bad"].agg(["count", "mean"])
    facts = {"shops_total": int(df["shop"].nunique()),
             "shops_compared": len(stats_all[stats_all["count"] >= min_n]),
             "overall_bad_rate": round(float(df["bad"].mean()), 4)}
    if rows:
        facts["worst_shop"] = rows[0]["key"]
        facts["worst_bad_rate"] = rows[0]["bad_rate"]
        facts["worst_n"] = rows[0]["n"]
        facts["best_bad_rate"] = rows[-1]["bad_rate"]
        facts["spread_pt"] = round(rows[0]["bad_rate"] - rows[-1]["bad_rate"], 4)
    return _result("shop_ranking", {"min_n": min_n}, table, facts,
                   note=f"样本量门槛 {min_n} 条；共 {facts['shops_compared']} 家店参与排名")


# ------------------------------------------------------------ 4. 套餐排名
@tool("deal_ranking", "按差评率给团购套餐排名（可限定代金券/套餐）",
      {"deal_type": "代金券 或 套餐，可选", "min_n": "最小样本量，默认 300",
       "top_k": "返回条数，默认 8", "year": "年份，可选", "shop": "店铺ID，可选"})
def deal_ranking(df: pd.DataFrame, deal_type=None, min_n: int = None,
                 top_k: int = 8, year=None, shop=None) -> dict:
    min_n = min_n or config.MIN_N_DEAL
    d = data_mod.scope_df(df, shop, None, None, year)
    d = d[d["deal"].notna()]
    if deal_type:
        d = d[d["deal_type"] == deal_type]
    rows = stats.ranking(d, "deal", min_n, top_k)
    if not rows:
        return _result("deal_ranking", {}, [], {"deals_compared": 0},
                       note=f"没有套餐达到样本量门槛 {min_n}")
    table = [{"套餐": r["key"], "评论数": r["n"], "差评数": r["bad"],
              "差评率": _pct(r["bad_rate"])} for r in rows]
    all_g = d.groupby("deal")["bad"].agg(["count", "mean"])
    n_compared = int((all_g["count"] >= min_n).sum())
    facts = {"deals_compared": n_compared,
             "deals_total": int(d["deal"].nunique()),
             "overall_bad_rate": round(float(d["bad"].mean()), 4),
             "worst_deal": rows[0]["key"], "worst_bad_rate": rows[0]["bad_rate"],
             "worst_n": rows[0]["n"],
             "best_bad_rate": rows[-1]["bad_rate"],
             "spread_pt": round(rows[0]["bad_rate"] - rows[-1]["bad_rate"], 4)}
    return _result("deal_ranking", {"deal_type": deal_type, "min_n": min_n}, table, facts,
                   note=f"样本量门槛 {min_n} 条；缺套餐字段的行已剔除")


# ------------------------------------------------------- 5. 代金券 vs 套餐
@tool("deal_type_compare", "对比代金券类与套餐类团购的差评率（业务核心分组）",
      {"shop": "店铺ID，可选", "year": "年份，可选"})
def deal_type_compare(df: pd.DataFrame, shop=None, year=None) -> dict:
    d = data_mod.scope_df(df, shop, None, None, year)
    rows = stats.deal_type_compare(d)
    if len(rows) < 2:
        return _result("deal_type_compare", {}, [], {}, note="分组不足")
    table = [{"团购类型": r["type"], "评论数": r["n"], "差评数": r["bad"],
              "差评率": _pct(r["bad_rate"]),
              "相对另一类": f"{r.get('ratio_vs_other')}×"} for r in rows]
    by = {r["type"]: r for r in rows}
    facts = {}
    for t, r in by.items():
        tag = "coupon" if t == "代金券" else "combo"
        facts[f"{tag}_n"] = r["n"]
        facts[f"{tag}_bad_rate"] = r["bad_rate"]
    hi, lo = max(rows, key=lambda r: r["bad_rate"]), min(rows, key=lambda r: r["bad_rate"])
    facts["ratio"] = round(hi["bad_rate"] / lo["bad_rate"], 2)
    facts["higher_type"] = hi["type"]
    return _result("deal_type_compare", dict(shop=shop, year=year), table, facts,
                   note="代金券 = 面额券/抵用券；套餐 = 固定搭配餐；两者履约方式不同")


# ------------------------------------------------------------ 6. 方面归因
@tool("aspect_gap", "方面归因：7 个业务方面的负面表述率对照（差评 vs 好评），定位短板",
      {"shop": "店铺ID，可选", "deal": "套餐名，可选",
       "deal_type": "代金券/套餐，可选", "year": "年份，可选"})
def aspect_gap(df: pd.DataFrame, shop=None, deal=None, deal_type=None, year=None) -> dict:
    d = data_mod.scope_df(df, shop, deal, deal_type, year, usable_text=True)
    if d.empty or d["bad"].sum() < 5:
        return _result("aspect_gap", {}, [], {"n": 0},
                       note="该切片差评数不足 5 条，不做归因")
    ag = stats.aspect_gap(d)
    rows = sorted(ag.items(), key=lambda kv: -kv[1]["neg_diff"])
    table = [{"方面": v["name"], "差评负面率": _pct(v["neg_bad_rate"]),
              "好评负面率": _pct(v["neg_good_rate"]),
              "倍差": v["neg_lift"], "差评命中数": v["bad_neg_count"],
              "建议动作": v["action"]} for _, v in rows]
    facts = {"n": int(len(d)), "bad_n": int(d["bad"].sum())}
    for key, v in ag.items():
        facts[f"{key}_neg_bad"] = v["neg_bad_rate"]
        facts[f"{key}_neg_good"] = v["neg_good_rate"]
        if v["neg_lift"] is not None:
            facts[f"{key}_lift"] = v["neg_lift"]
    top = rows[0][1]
    facts["top_aspect"] = top["name"]
    facts["top_aspect_neg_bad"] = top["neg_bad_rate"]
    facts["top_aspect_neg_good"] = top["neg_good_rate"]
    facts["top_aspect_lift"] = top["neg_lift"]
    return _result("aspect_gap", dict(shop=shop, deal=deal, year=year), table, facts,
                   note=f"只统计文本可用评论（长度 >= {config.MIN_TEXT_LEN}）；"
                        f"负面率 = 命中该方面负面表述的评论占比")


# ------------------------------------------------------------ 7. 差异关键词
@tool("keyword_diff", "差评 vs 好评的词汇差异（信息先验 log-odds，带显著性 z 值）",
      {"shop": "店铺ID，可选", "deal": "套餐名，可选",
       "deal_type": "代金券/套餐，可选", "year": "年份，可选",
       "top_k": "返回词数，默认 12"})
def keyword_diff(df: pd.DataFrame, shop=None, deal=None, deal_type=None,
                 year=None, top_k: int = 12) -> dict:
    d = data_mod.scope_df(df, shop, deal, deal_type, year, usable_text=True)
    bad, good = d[d["bad"] == 1], d[d["bad"] == 0]
    if len(bad) < 10 or len(good) < 10:
        return _result("keyword_diff", {}, [], {"n": 0},
                       note="该切片某一类评论不足 10 条，不做词汇差异")
    cb, cg = stats.corpus(bad["text"]), stats.corpus(good["text"])
    nb, ng = sum(cb.values()), sum(cg.values())
    rows = stats.log_odds(cb, cg, nb, ng, top_k=top_k)
    table = [{"词": r["word"], "差评次数": r["bad"], "好评次数": r["good"],
              "z 值": r["z"], "倍数": r["lift"]} for r in rows]
    facts = {"n_bad_text": len(bad), "n_good_text": len(good),
             "tokens_bad": nb, "tokens_good": ng,
             "sig_z_threshold": 1.96, "sig_alpha": 0.05}
    for i, r in enumerate(rows[:5], 1):
        facts[f"kw{i}"] = r["word"]
        facts[f"kw{i}_z"] = r["z"]
        if r["lift"]:
            facts[f"kw{i}_lift"] = r["lift"]
    return _result("keyword_diff", dict(shop=shop, deal=deal, top_k=top_k), table, facts,
                   note="log-odds 用 Monroe et al. (2008) 的信息先验；z > 1.96 即在 5% 水平显著")


# ------------------------------------------------------------ 8. 时间趋势
@tool("time_trend", "差评率时间趋势 + 最近一期相对历史基线的 z 值异动检测",
      {"shop": "店铺ID，可选", "deal_type": "代金券/套餐，可选",
       "freq": "Y/M/Q，默认 Q", "year": "年份，可选"})
def time_trend(df: pd.DataFrame, shop=None, deal_type=None, freq: str = "Q",
               year=None) -> dict:
    d = data_mod.scope_df(df, shop, None, deal_type, year)
    if d.empty:
        return _result("time_trend", {}, [], {"n": 0}, note="该切片无数据")
    t = stats.trend(d, freq)
    table = [{"期": s["period"], "评论数": s["n"], "差评率": _pct(s["bad_rate"])}
             for s in t["series"]]
    last = t["series"][-1]
    facts = {"periods": len(t["series"]), "freq": freq,
             "last_period": last["period"], "last_n": last["n"],
             "last_bad_rate": last["bad_rate"]}
    if "z" in last:
        facts.update({"baseline_mean": last["baseline_mean"], "last_z": last["z"],
                      "is_anomaly": int(last["is_anomaly"])})
    return _result("time_trend", dict(shop=shop, freq=freq), table, facts,
                   note=f"最近一期与之前各期均值比较；|z| >= {config.ANOMALY_Z} 视为异动")


# ------------------------------------------------------------ 9. 时段分布
@tool("time_slot", "差评率的星期 / 小时分布（找高峰时段的体验塌陷）",
      {"dim": "weekday 或 hour，默认 weekday", "shop": "店铺ID，可选",
       "deal_type": "代金券/套餐，可选", "year": "年份，可选"})
def time_slot(df: pd.DataFrame, dim: str = "weekday", shop=None, deal_type=None,
              year=None) -> dict:
    d = data_mod.scope_df(df, shop, None, deal_type, year)
    rows = stats.slot_rates(d, dim)
    if not rows:
        return _result("time_slot", {}, [], {"n": 0}, note="样本量不足")
    table = [{"时段": r["slot"], "评论数": r["n"], "差评率": _pct(r["bad_rate"])}
             for r in rows]
    hi = max(rows, key=lambda r: r["bad_rate"])
    lo = min(rows, key=lambda r: r["bad_rate"])
    facts = {"dim": dim, "slots": len(rows),
             "high_slot": hi["slot"], "high_bad_rate": hi["bad_rate"],
             "low_slot": lo["slot"], "low_bad_rate": lo["bad_rate"],
             "spread_pt": round(hi["bad_rate"] - lo["bad_rate"], 4)}
    return _result("time_slot", dict(dim=dim, shop=shop), table, facts,
                   note=f"只保留样本量 >= {config.MIN_N_SLOT} 的时段")


# ------------------------------------------------------------ 10. 原始评论
@tool("sample_comments", "抽样原始评论作为证据（差评/好评各取 n 条）",
      {"shop": "店铺ID，可选", "deal": "套餐名，可选",
       "deal_type": "代金券/套餐，可选", "star": "bad 或 good，默认 bad", "n": "条数，默认 5"})
def sample_comments(df: pd.DataFrame, shop=None, deal=None, deal_type=None,
                    star: str = "bad", n: int = 5) -> dict:
    d = data_mod.scope_df(df, shop, deal, deal_type, usable_text=True)
    d = d[d["bad"] == (1 if star == "bad" else 0)]
    if d.empty:
        return _result("sample_comments", {}, [], {"n": 0}, note="无样本")
    # 优先取较长的评论（信息量大），并去重
    pick = d.drop_duplicates(subset=["text"]).sort_values("text_len", ascending=False).head(n)
    table = [{"星级": f"{r['star']:.1f}★", "套餐": r["deal"] or "-",
              "评论": r["text"][:120]} for _, r in pick.iterrows()]
    facts = {"n": int(len(pick)), "star_bucket": star,
             "pool": int(len(d)), "mean_len": round(float(pick["text_len"].mean()), 1)}
    return _result("sample_comments", dict(shop=shop, star=star), table, facts,
                   note="抽样规则：去重后按评论长度取 Top N（长评论信息量更大）")


# ------------------------------------------------------------------ 调度
def call(name: str, df: pd.DataFrame, **params) -> dict:
    return REGISTRY.call(name, df, **params)


def signatures() -> str:
    return REGISTRY.signatures()
