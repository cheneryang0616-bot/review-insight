# -*- coding: utf-8 -*-
"""credit 域 —— 数据层

两份公开数据统一成同一套内部结构，工具层只认这个结构：

    df["y"]          1 = 坏客户（违约/严重逾期），0 = 好客户
    df[feature ...]  数值型特征（German 的类别字段本身就是数字编码）
    df.attrs         dataset / features / forbidden_hard / forbidden_soft / has_time

为什么强调 has_time：Give Me Some Credit **没有时间列**，
所以做不了 OOT 验证，也做不了 PSI 跨期监控 —— 这一点必须由数据层显式声明，
否则报告里很容易出现"我们做了时间外验证"这种站不住的话。
"""
import functools
import os
from typing import Optional

import pandas as pd

from . import config


def _load_gmc(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.drop(columns=[c for c in raw.columns if str(c).startswith("Unnamed")])
    df = pd.DataFrame({"y": (raw[config.TARGET_GMC] > 0).astype(int)})
    feats = [c for c in raw.columns if c != config.TARGET_GMC]
    for c in feats:
        df[c] = pd.to_numeric(raw[c], errors="coerce")
    df.attrs.update(dataset="give_me_some_credit", features=feats,
                    forbidden_hard=[c for c in feats if c in config.FORBIDDEN_HARD_COLS],
                    forbidden_soft=[c for c in feats if c in config.SENSITIVE_SOFT_COLS],
                    has_time=False)
    return df


def _load_german(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path)
    raw = raw.rename(columns=config.GERMAN_RENAME)
    df = pd.DataFrame({"y": (raw[config.TARGET_GERMAN] == config.GERMAN_BAD_VALUE).astype(int)})
    feats = [c for c in raw.columns if c != config.GERMAN_RENAME.get(
        config.TARGET_GERMAN, config.TARGET_GERMAN)]
    for c in feats:
        df[c] = pd.to_numeric(raw[c], errors="coerce")
    df.attrs.update(dataset="german_credit", features=feats,
                    forbidden_hard=[c for c in feats if c in config.FORBIDDEN_HARD_COLS],
                    forbidden_soft=[c for c in feats if c in config.SENSITIVE_SOFT_COLS],
                    has_time=False)
    return df


@functools.lru_cache(maxsize=2)
def _load_cached(path: str, kind: str) -> pd.DataFrame:
    return _load_gmc(path) if kind == "gmc" else _load_german(path)


def resolve(dataset: Optional[str] = None) -> tuple:
    """决定用哪份数据：显式指定 > 样例 > GMC 全量 > German。

    样例 csv 里带 `dataset` 列标记来源，因此可以只存在一份样例文件。
    """
    dataset = (dataset or os.environ.get("RI_CREDIT_DATA") or "").lower()
    if dataset in ("gmc", "give_me_some_credit"):
        return config.GMC_TRAIN, "gmc"
    if dataset in ("german", "german_credit"):
        return config.GERMAN, "german"
    if os.path.exists(config.SAMPLE):
        return config.SAMPLE, "sample"
    if os.path.exists(config.GMC_TRAIN):
        return config.GMC_TRAIN, "gmc"
    if os.path.exists(config.GERMAN):
        return config.GERMAN, "german"
    raise FileNotFoundError(
        "找不到信贷数据。请把 cs-training.csv / german_credit.xlsx 放到 data/，"
        "或使用仓库自带的 data/sample_credit.csv。详见 data/README.md")


def load(dataset: Optional[str] = None) -> pd.DataFrame:
    path, kind = resolve(dataset)
    if kind == "sample":
        raw = pd.read_csv(path)
        origin = str(raw["_dataset"].iloc[0]) if "_dataset" in raw.columns else "give_me_some_credit"
        raw = raw.drop(columns=[c for c in ["_dataset"] if c in raw.columns])
        if config.TARGET_GMC in raw.columns:
            df = pd.DataFrame({"y": (raw[config.TARGET_GMC] > 0).astype(int)})
            feats = [c for c in raw.columns if c != config.TARGET_GMC]
        else:
            df = _load_german(path)
            return df
        for c in feats:
            df[c] = pd.to_numeric(raw[c], errors="coerce")
        df.attrs.update(dataset=f"{origin}(sample)", features=feats,
                        forbidden_hard=[c for c in feats if c in config.FORBIDDEN_HARD_COLS],
                        forbidden_soft=[c for c in feats if c in config.SENSITIVE_SOFT_COLS],
                        has_time=False)
        return df
    return _load_cached(path, kind)


def profile(df: pd.DataFrame) -> dict:
    y = df["y"]
    feats = df.attrs.get("features", [])
    missing = {c: round(float(df[c].isna().mean()), 4) for c in feats if df[c].isna().any()}
    extremes = {}
    for c in feats:
        s = df[c].dropna()
        if len(s) and s.quantile(0.99) > 0:
            ratio = s.max() / s.quantile(0.99)
            if ratio >= 5:
                extremes[c] = round(float(ratio), 1)
    return {
        "dataset": df.attrs.get("dataset", "?"),
        "rows": int(len(df)),
        "features": len(feats),
        "bad_rate": round(float(y.mean()), 4),
        "bad_n": int(y.sum()),
        "imbalance": round(float((1 - y.mean()) / y.mean()), 2) if y.mean() > 0 else None,
        "missing_max": round(max(missing.values()), 4) if missing else 0.0,
        "missing_cols": len(missing),
        "extreme_cols": len(extremes),
        "has_time": int(bool(df.attrs.get("has_time"))),
        "forbidden_hard": len(df.attrs.get("forbidden_hard", [])),
        "forbidden_soft": len(df.attrs.get("forbidden_soft", [])),
    }


def scope_desc(dataset=None, feature=None, segment=None, bins=None,
               keep=None, kind=None) -> str:
    parts = []
    if kind:
        parts.append("坏客户案例" if kind == "bad" else "好客户案例")
    if keep:
        parts.append(f"通过率 {keep * 100:.0f}%")
    if dataset:
        parts.append(f"数据集 {dataset}")
    if feature:
        parts.append(f"变量 {feature}")
    if segment:
        parts.append(f"分群维度 {segment}")
    return " · ".join(parts) if parts else "当前数据集全量"
