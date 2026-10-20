# -*- coding: utf-8 -*-
"""reviewinsight.data —— 数据层

把原始 xlsx 清洗成一个分析用的 DataFrame。所有"脏"都收敛在这里：
1. 星级 5 分制 ×10（含 15/25/35/45 半星）→ 除以 10
2. 差评口径 stars <= 2
3. 评论内容去掉「#套餐：xxx」模板前缀（否则模板词污染关键词统计）
4. 菜品字段 = 团购套餐名，按第一个逗号切分
5. 套餐类型：代金券 / 套餐 / 其他（这是本项目最核心的分组维度）
6. 文本长度 < 4 字的标记为 usable_text=False，文本类分析必须过滤
"""
import functools
import math
import os
import re
from typing import Optional

import pandas as pd

from . import config

# 「#套餐：男票甜蜜双人餐」这类模板前缀，占 0.55%，但会污染关键词
_TEMPLATE_RE = re.compile(r"^(?:「[^」]{0,60}」\s*)+")   # 连续多个「#菜品」标签
_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")

_COUPON_WORDS = ("代金券", "现金券", "抵用券", "优惠券", "券")
_COMBO_WORDS = ("套餐", "双人餐", "人餐", "人套餐", "餐")


def _is_missing(value) -> bool:
    """缺失判定要同时容忍 None / float NaN / 字符串 "nan"。

    pandas 2.x 里 `replace({"nan": None})` 得到的是 None，pandas 3.x 会优化成 float NaN；
    只判 `not deal` 会在 3.x 上直接 TypeError（`w in nan`）——CI 在 3.11 上抓到的就是这个。
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in ("nan", "none", "null")


def classify_deal(deal) -> str:
    """套餐名 → 业务类型。代金券和套餐的履约方式不同，差评机制也不同。"""
    if _is_missing(deal):
        return "未标注"
    deal = str(deal)
    if any(w in deal for w in _COUPON_WORDS):
        return "代金券"
    if any(w in deal for w in _COMBO_WORDS):
        return "套餐"
    return "其他"


def clean_text(s: str) -> str:
    s = _URL_RE.sub(" ", str(s or ""))
    s = _TEMPLATE_RE.sub("", s)
    s = _SPACE_RE.sub(" ", s)
    return s.strip()


def resolve_path(path: Optional[str] = None) -> tuple:
    """决定用哪份数据：显式指定 > 完整数据 > 脱敏样例。

    完整数据（含个人信息的原始 xlsx）不进仓库，见 data/README.md。
    """
    if path:
        return path, ("xlsx" if path.endswith((".xlsx", ".xls")) else "csv")
    for p, kind in ((config.DATA_FILE, "xlsx"), (config.SAMPLE_FILE, "csv")):
        if os.path.exists(p):
            return p, kind
    raise FileNotFoundError(
        "找不到数据文件。请把 meituan.xlsx 放到 data/meituan_reviews.xlsx，"
        "或使用仓库自带的 data/sample_reviews.csv。详见 data/README.md")


@functools.lru_cache(maxsize=1)
def load(path: Optional[str] = None) -> pd.DataFrame:
    """加载并清洗。带缓存，全流程只用一份内存副本。"""
    resolved, kind = resolve_path(os.environ.get("RI_DATA") or path)
    raw = pd.read_csv(resolved) if kind == "csv" else pd.read_excel(resolved)
    # 样例里可能缺列，补齐后再走同一套清洗，保证下游口径一致
    for col in ("评论用户姓名", "评论用户id"):
        if col not in raw.columns:
            raw[col] = None

    df = pd.DataFrame()
    df["shop"] = raw["评论店铺"].astype(str)
    df["star"] = raw["评论用户星级"].astype(float) / config.STAR_DIVISOR
    df["bad"] = (df["star"] <= config.BAD_STAR_MAX).astype(int)
    df["raw_text"] = raw["评论用户内容"].astype(str)
    df["text"] = df["raw_text"].map(clean_text)
    df["text_len"] = df["text"].str.len()

    deal = raw["评论用户菜品"].astype(str).str.split("，").str[0].str.strip()
    deal = deal.replace({"nan": None, "": None})
    df["deal"] = deal
    df["deal_type"] = deal.map(classify_deal)

    ts = pd.to_datetime(raw["评论时间"], unit="ms", errors="coerce")
    df["ts"] = ts
    df["year"] = ts.dt.year
    df["month"] = ts.dt.to_period("M").astype(str)
    df["quarter"] = ts.dt.to_period("Q").astype(str)
    df["weekday"] = ts.dt.weekday
    df["hour"] = ts.dt.hour

    df["usable_text"] = df["text_len"] >= config.MIN_TEXT_LEN
    df = df.reset_index(drop=True)
    df.attrs["source"] = "full" if kind == "xlsx" else "sample"
    df.attrs["path"] = resolved
    return df


def scope_df(df: pd.DataFrame, shop=None, deal=None, deal_type=None,
             year=None, usable_text: bool = False) -> pd.DataFrame:
    """统一的切片函数。工具层不许自己写 filter，否则口径会漂。"""
    d = df
    if shop is not None:
        d = d[d["shop"] == str(shop)]
    if deal is not None:
        d = d[d["deal"] == deal]
    if deal_type is not None:
        d = d[d["deal_type"] == deal_type]
    if year is not None:
        d = d[d["year"] == int(year)]
    if usable_text:
        d = d[d["usable_text"]]
    return d


def overview(df: pd.DataFrame) -> dict:
    """数据集体检：行数、店铺、时间跨度、文本可用率、缺失率。"""
    return {
        "rows": int(len(df)),
        "shops": int(df["shop"].nunique()),
        "deals": int(df["deal"].nunique()),
        "start": str(df["ts"].min())[:10],
        "end": str(df["ts"].max())[:10],
        "years": sorted(int(y) for y in df["year"].dropna().unique()),
        "bad_rate": round(float(df["bad"].mean()), 4),
        "bad_rows": int(df["bad"].sum()),
        "text_usable_rate": round(float(df["usable_text"].mean()), 4),
        "text_short_rate": round(1 - float(df["usable_text"].mean()), 4),
        "deal_missing_rate": round(float(df["deal"].isna().mean()), 4),
        "deals_effective": int((df["deal"].value_counts() >= 30).sum()),
        "deals_min_reviews": 30,
        "dup_text_rate": round(1 - df["text"].nunique() / len(df), 4),
    }


def scope_desc(shop=None, deal=None, deal_type=None, year=None) -> str:
    """把切片条件翻译成人话，用于答案里的"分析范围"段。"""
    parts = []
    if shop:
        parts.append(f"店铺 {shop}")
    if deal:
        parts.append(f"套餐「{deal}」")
    if deal_type:
        parts.append(f"{deal_type}类团购")
    if year:
        parts.append(f"{year} 年")
    return " · ".join(parts) if parts else "全量数据"
