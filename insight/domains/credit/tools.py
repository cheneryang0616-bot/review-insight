# -*- coding: utf-8 -*-
"""credit 域 —— 9 个工具

事实（facts）里除了业务指标，故意多放了几项**给闸门用的信号**：
iv_max / iv_high / nonmono / ks / cutoff_monotone / forbidden_used。

这样 gates 层可以完全声明式地断言业务约束，不用重新算一遍。
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from ...core.agent import pct as _pct
from ...core.registry import Registry
from . import config, data as data_mod, stats

REGISTRY = Registry("credit")
tool = REGISTRY.tool


def _result(tool_name: str, params: dict, table: List[dict], facts: dict,
            note: str = "") -> dict:
    return {"tool": tool_name, "params": params, "table": table, "facts": facts, "note": note}


# ------------------------------------------------------------ 1. 数据体检
@tool("dataset_profile", "数据体检：样本量、违约率、不平衡比、缺失、极端值、合规字段、有无时间列", {})
def dataset_profile(df: pd.DataFrame) -> dict:
    o = data_mod.profile(df)
    table = [{"指标": k, "值": v} for k, v in o.items()]
    note = "坏客户 = 违约/严重逾期；不平衡比 = 好客户数/坏客户数"
    if not o["has_time"]:
        note += "；该数据集无时间列 → 只能随机切分，不能做 OOT 验证或跨期 PSI"
    return _result("dataset_profile", {}, table, o, note)


# ------------------------------------------------------------ 2. IV 排序
@tool("iv_table", "变量 IV 排序（附单调性；IV>0.5 需先怀疑泄漏或极端值）",
      {"top_k": "返回变量数，默认 8", "bins": "分箱数，默认 10"})
def iv_table(df: pd.DataFrame, top_k: int = 8, bins: Optional[int] = None) -> dict:
    rows = stats.iv_table(df, bins=bins)
    if not rows:
        return _result("iv_table", {}, [], {}, "无可用特征")
    top = rows[:top_k]
    table = [{"变量": r["feature"], "IV": r["iv"], "分箱数": r["bins"],
              "WOE 单调": "是" if r["monotone"] else "否"} for r in top]
    high = [r["feature"] for r in rows if r["iv"] > config.IV_SUSPICIOUS]
    nonmono = [r["feature"] for r in rows if not r["monotone"]]
    facts = {
        "features": len(rows),
        "iv_max": round(max(r["iv"] for r in rows), 4),
        "iv_max_feature": max(rows, key=lambda r: r["iv"])["feature"],
        "iv_high_count": len(high),
        "iv_suspicious_threshold": config.IV_SUSPICIOUS,
        "nonmono_count": len(nonmono),
    }
    # 列表型 fact 供闸门断言
    facts["iv_high"] = high
    facts["nonmono"] = nonmono
    for i, r in enumerate(top[:3], 1):
        facts[f"top{i}_feature"] = r["feature"]
        facts[f"top{i}_iv"] = r["iv"]
    return _result("iv_table", {"top_k": top_k}, table, facts,
                   note="IV 0.02~0.1 弱、0.1~0.3 中、0.3~0.5 强；>0.5 先查数据再谈强变量")


# ------------------------------------------------------------ 3. 单变量分箱
@tool("woe_table", "单变量 WOE 分箱明细（每箱样本、坏账率、WOE），检查单调性",
      {"feature": "变量名（必需）", "bins": "分箱数，默认 10"},
      required=("feature",))
def woe_table(df: pd.DataFrame, feature: str, bins: Optional[int] = None) -> dict:
    feats = df.attrs.get("features", [])
    if feature not in feats:
        raise ValueError(f"未知变量：{feature}（可选：{feats[:8]}）")
    r = stats.feature_woe_iv(df[feature], df["y"], bins)
    tab = r["table"]
    table = [{"分箱": str(k), "样本": int(row["count"]),
              "坏账率": _pct(row["bad_rate"]), "WOE": round(float(row["woe"]), 3)}
             for k, row in tab.iterrows()]
    facts = {"feature": feature, "iv": round(r["iv"], 4), "bins": r["bins"],
             "monotone": int(r["monotone"]),
             "woe_min": round(float(tab["woe"].min()), 3),
             "woe_max": round(float(tab["woe"].max()), 3),
             "bad_rate_min": round(float(tab["bad_rate"].min()), 4),
             "bad_rate_max": round(float(tab["bad_rate"].max()), 4)}
    note = f"等频分箱；WOE 单调 = {'是' if r['monotone'] else '否'}"
    return _result("woe_table", {"feature": feature}, table, facts, note)


# ------------------------------------------------------------ 4. 模型指标
@tool("perf_metrics", "模型表现：AUC / KS / Gini / Lift@10%（WOE + 逻辑回归）",
      {"exclude_forbidden": "是否剔除禁忌字段，0/1，默认 0"})
def perf_metrics(df: pd.DataFrame, exclude_forbidden: int = 0) -> dict:
    m = stats.fit(df, exclude_forbidden=bool(exclude_forbidden))
    lift = stats.lift_at(m["p_test"], m["y_test"], 0.1)
    psi = stats.psi(m["p_test"][m["y_test"] == 0], m["p_test"][m["y_test"] == 1])
    table = [{"指标": "AUC", "值": round(m["auc"], 4)},
             {"指标": "KS", "值": round(m["ks"], 4)},
             {"指标": "Gini", "值": round(m["gini"], 4)},
             {"指标": "Lift@10%", "值": round(lift, 2)},
             {"指标": "特征数", "值": len(m["features"])},
             {"指标": "训练/验证", "值": f"{m['n_train']}/{m['n_test']}"}]
    facts = {"auc": round(m["auc"], 4), "ks": round(m["ks"], 4),
             "gini": round(m["gini"], 4), "lift_10": round(lift, 2),
             "n_features": len(m["features"]),
             "n_train": m["n_train"], "n_test": m["n_test"],
             "test_bad_rate": round(m["bad_test"], 4),
             "ks_floor": config.KS_FLOOR,
             "psi_good_vs_bad": round(psi, 4)}
    facts["features_used"] = list(m["features"])
    # 模型层面是否用了禁忌/敏感字段 —— 闸门真正断言的是这个
    present_hard = set(df.attrs.get("forbidden_hard", []))
    present_soft = set(df.attrs.get("forbidden_soft", []))
    facts["forbidden_used"] = sorted(set(m["features"]) & present_hard)
    facts["sensitive_used"] = sorted(set(m["features"]) & present_soft)
    facts["exclude_forbidden"] = int(bool(exclude_forbidden))
    coefs = sorted(m["coef"].items(), key=lambda kv: -abs(kv[1]))[:5]
    for i, (c, b) in enumerate(coefs, 1):
        facts[f"coef{i}_feature"] = c
        facts[f"coef{i}_value"] = round(b, 3)
    note = ("随机切分（非时间外验证，数据无时间列）；class_weight=balanced 处理不平衡")
    return _result("perf_metrics", {"exclude_forbidden": exclude_forbidden}, table, facts, note)


# ------------------------------------------------------------ 5. 通过率权衡
@tool("cutoff_tradeoff", "通过率 vs 实际坏账率 vs 捕获坏客户比例（决策阈值选择）",
      {"exclude_forbidden": "是否剔除禁忌字段，0/1，默认 0",
       "keep": "指定通过率（0~1），如 0.7；不传则给 90/80/70/60/50% 一组"})
def cutoff_tradeoff(df: pd.DataFrame, exclude_forbidden: int = 0,
                    keep: Optional[float] = None) -> dict:
    m = stats.fit(df, exclude_forbidden=bool(exclude_forbidden))
    keeps = (float(keep),) if keep else (0.9, 0.8, 0.7, 0.6, 0.5)
    rows = stats.cutoff_tradeoff(m["y_test"], m["p_test"], keeps)
    table = [{"通过率": _pct(r["keep_rate"]), "通过人数": r["n_pass"],
              "实际坏账率": _pct(r["bad_rate"]), "捕获坏客户": _pct(r["capture"])}
             for r in rows]
    bad_rates = [r["bad_rate"] for r in rows]
    # rows 按通过率从高到低排列，因此坏账率应当"不上升"
    monotone = all(bad_rates[i] >= bad_rates[i + 1] - 1e-9
                   for i in range(len(bad_rates) - 1))
    facts = {"keeps": len(rows), "cutoff_monotone": int(monotone),
             "full_bad_rate": rows[0]["bad_rate"],
             "strict_bad_rate": rows[-1]["bad_rate"],
             "strict_capture": rows[-1]["capture"],
             "loose_keep": rows[0]["keep_rate"], "strict_keep": rows[-1]["keep_rate"]}
    note = "通过率越低、坏账率越低才算正常；曲线非单调说明排序逻辑有问题"
    return _result("cutoff_tradeoff", {"exclude_forbidden": exclude_forbidden},
                   table, facts, note)


# ------------------------------------------------------------ 6. 分群坏账率
@tool("segment_bad_rate", "按某个变量分群的坏账率（找高风险客群）",
      {"feature": "变量名（必需）", "bins": "分群数，默认 5"},
      required=("feature",))
def segment_bad_rate(df: pd.DataFrame, feature: str, bins: int = 5) -> dict:
    feats = df.attrs.get("features", [])
    if feature not in feats:
        raise ValueError(f"未知变量：{feature}（可选：{feats[:8]}）")
    rows = stats.segment_bad_rate(df, feature, bins)
    if not rows:
        return _result("segment_bad_rate", {"feature": feature}, [], {}, "样本量不足")
    table = [{"分群": r["segment"], "样本": r["n"], "坏账率": _pct(r["bad_rate"])}
             for r in rows]
    hi = max(rows, key=lambda r: r["bad_rate"])
    lo = min(rows, key=lambda r: r["bad_rate"])
    facts = {"feature": feature, "segments": len(rows),
             "high_segment": hi["segment"], "high_bad_rate": hi["bad_rate"],
             "low_segment": lo["segment"], "low_bad_rate": lo["bad_rate"],
             "spread_pt": round(hi["bad_rate"] - lo["bad_rate"], 4)}
    return _result("segment_bad_rate", {"feature": feature, "bins": bins}, table, facts,
                   note=f"只保留样本量 >= {config.MIN_N_SEGMENT} 的分群")


# ------------------------------------------------------------ 7. 分数刻度
@tool("score_scale", "标准分映射：基准分 / PDO / 基准好坏比 → 分数区间与示例客户得分",
      {"base": "基准分，默认 600", "pdo": "PDO，默认 50", "base_odds": "基准好坏比，默认 20"})
def score_scale(df: pd.DataFrame, base: float = 600, pdo: float = 50,
                base_odds: float = 20) -> dict:
    m = stats.fit(df, exclude_forbidden=False)
    p = m["p_test"]
    logits = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
    scores = np.array([stats.score_from_logit(l, base, pdo, base_odds) for l in logits])
    table = [{"指标": "基准分", "值": base}, {"指标": "PDO", "值": pdo},
             {"指标": "基准好坏比", "值": base_odds},
             {"指标": "分数中位数", "值": round(float(np.median(scores)), 1)},
             {"指标": "P10", "值": round(float(np.percentile(scores, 10)), 1)},
             {"指标": "P90", "值": round(float(np.percentile(scores, 90)), 1)}]
    facts = {"base": base, "pdo": pdo, "base_odds": base_odds,
             "score_median": round(float(np.median(scores)), 1),
             "score_p10": round(float(np.percentile(scores, 10)), 1),
             "score_p90": round(float(np.percentile(scores, 90)), 1)}
    note = "score = base + PDO/ln2 × (ln(好坏比) − ln(base_odds))；分数越高越好"
    return _result("score_scale", {"base": base, "pdo": pdo}, table, facts, note)


# ------------------------------------------------------------ 8. 合规检查
@tool("forbidden_check", "合规检查：模型是否使用了性别/婚姻、外籍劳工等禁忌字段", {})
def forbidden_check(df: pd.DataFrame) -> dict:
    present_hard = list(df.attrs.get("forbidden_hard", []))
    present_soft = list(df.attrs.get("forbidden_soft", []))
    m = stats.fit(df, exclude_forbidden=False)
    used = set(m["features"])
    used_hard = [c for c in present_hard if c in used]
    used_soft = [c for c in present_soft if c in used]
    table = [
        {"类型": "硬阻断（禁止入模）", "字段": ", ".join(present_hard) or "无",
         "是否已入模": "是" if used_hard else "否"},
        {"类型": "敏感（需说明理由）", "字段": ", ".join(present_soft) or "无",
         "是否已入模": "是" if used_soft else "否"},
    ]
    facts = {"forbidden_present": len(present_hard),
             "sensitive_present": len(present_soft),
             "forbidden_used_count": len(used_hard),
             "sensitive_used_count": len(used_soft),
             # 列表型 fact 供闸门做集合断言（数据集层面）
             "forbidden_present_hard": present_hard,
             "sensitive_present_list": present_soft}
    note = "硬阻断字段一旦入模，模型即便指标再好也不可用（fair lending / 就业歧视）"
    return _result("forbidden_check", {}, table, facts, note)


# ------------------------------------------------------------ 9. 客户案例
@tool("sample_cases", "抽样客户案例（好/坏客户各看几条，含关键字段）",
      {"kind": "bad 或 good，默认 bad", "n": "条数，默认 5"})
def sample_cases(df: pd.DataFrame, kind: str = "bad", n: int = 5) -> dict:
    d = df[df["y"] == (1 if kind == "bad" else 0)]
    if d.empty:
        return _result("sample_cases", {}, [], {}, "无样本")
    feats = df.attrs.get("features", [])[:4]
    pick = d.head(n)
    table = []
    for _, r in pick.iterrows():
        row = {"标签": "坏客户" if r["y"] == 1 else "好客户"}
        for c in feats:
            row[config.label(c)] = (round(float(r[c]), 3) if pd.notna(r[c]) else "缺失")
        table.append(row)
    facts = {"n": int(len(pick)), "kind": kind, "pool": int(len(d)),
             "pool_rate": round(float(len(d) / len(df)), 4)}
    return _result("sample_cases", {"kind": kind, "n": n}, table, facts,
                   note="从数据头部抽样，不是随机抽样；仅用于观察字段尺度")


def call(name: str, df: pd.DataFrame, **params) -> dict:
    return REGISTRY.call(name, df, **params)


def signatures() -> str:
    return REGISTRY.signatures()
