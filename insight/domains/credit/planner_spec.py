# -*- coding: utf-8 -*-
"""credit 域 —— 规划配置

风控的问题空间比评论分析更窄，但更"术语化"：
IV / WOE / KS / 通过率 / 评分刻度 / 合规，每个词对应一组固定的工具。
"""
import re

import pandas as pd

from ...core.planner import PlanSpec

INTENTS = {
    "profile": ["概况", "整体", "违约率", "不平衡", "缺失", "样本", "数据情况", "什么样"],
    "iv": ["iv", "IV", "变量", "特征", "重要", "筛选", "区分"],
    "woe": ["分箱", "woe", "WOE", "单调", "箱"],
    "perf": ["auc", "AUC", "ks", "KS", "模型", "效果", "表现", "区分度", "gini", "lift"],
    "cutoff": ["通过率", "阈值", "拒绝", "截断", "放贷", "审批", "拒绝率"],
    "segment": ["分群", "客群", "分段", "哪个年龄", "哪个收入", "高风险人群"],
    "score": ["评分", "分数", "刻度", "pdo", "PDO", "打分"],
    "forbidden": ["合规", "禁忌", "性别", "歧视", "公平", "字段不能", "剔除"],
    "sample": ["案例", "客户", "例子", "抽样", "看看"],
}

SEQUENCE = {
    "profile": ["dataset_profile", "iv_table"],
    "iv": ["iv_table", "perf_metrics"],
    "woe": ["woe_table", "iv_table"],
    "perf": ["perf_metrics", "cutoff_tradeoff"],
    "cutoff": ["cutoff_tradeoff", "perf_metrics"],
    "segment": ["segment_bad_rate", "iv_table"],
    "score": ["score_scale", "perf_metrics"],
    "forbidden": ["forbidden_check", "perf_metrics"],
    "sample": ["sample_cases", "dataset_profile"],
}

ANAPHORA = ["这个变量", "该变量", "它", "那个", "这组", "同样", "还有"]

# 中文口语 → 英文字段名（真实数据字段很长，用户不会照着敲）
ALIASES = {
    "额度使用率": "RevolvingUtilizationOfUnsecuredLines",
    "使用率": "RevolvingUtilizationOfUnsecuredLines",
    "逾期": "NumberOfTime30-59DaysPastDueNotWorse",
    "30-59": "NumberOfTime30-59DaysPastDueNotWorse",
    "60-89": "NumberOfTime60-89DaysPastDueNotWorse",
    "90天": "NumberOfTimes90DaysLate",
    "年龄": "age",
    "负债": "DebtRatio",
    "负债率": "DebtRatio",
    "收入": "MonthlyIncome",
    "月收入": "MonthlyIncome",
    "信用线": "NumberOfOpenCreditLinesAndLoans",
    "房产": "NumberRealEstateLoansOrLines",
    "房贷": "NumberRealEstateLoansOrLines",
    "依赖": "NumberOfDependents",
    "账户余额": "account_balance",
    "余额": "account_balance",
    "期限": "duration_month",
    "分期": "instalment_pct",
    "贷款金额": "credit_amount",
    "金额": "credit_amount",
    "储蓄": "savings",
    "在职": "employment_length",
    "用途": "purpose",
    "还款": "payment_status_prev",
    "抚养": "NumberOfDependents",
}

LLM_HINTS = """- 问"模型效果"用 perf_metrics；问"阈值/通过率"用 cutoff_tradeoff
- 问某个变量的分箱细节，用 woe_table(feature=...)，feature 必须来自"可用对象"清单
- 问合规/性别/歧视，用 forbidden_check；若用户要"剔除后再看"，用
  perf_metrics(exclude_forbidden=1)
- 该数据集可能没有时间列：不要承诺 OOT 验证或跨期 PSI"""


def extract_entities(q: str, df: pd.DataFrame) -> dict:
    ent = {}
    feats = df.attrs.get("features", [])

    # ① 直接写英文字段名（忽略大小写）
    low = q.lower()
    hits = [f for f in feats if f.lower() in low]
    if hits:
        ent["feature"] = max(hits, key=len)

    # ② 中文口语别名
    if "feature" not in ent:
        for alias, feat in ALIASES.items():
            if alias in q and feat in feats:
                ent["feature"] = feat
                break

    # 通过率 / 拒绝率：把"压到 70%"这类口语换算成 0.7
    m = re.search(r"(\d{2})\s*%", q)
    if m and ("通过率" in q or "放行" in q or "审批" in q):
        v = int(m.group(1))
        if 10 <= v <= 99:
            ent["keep"] = round(v / 100.0, 2)
    m2 = re.search(r"拒绝\s*(\d{2})\s*%", q)
    if m2:
        v = 100 - int(m2.group(1))
        if 1 <= v <= 99:
            ent["keep"] = round(v / 100.0, 2)

    if "坏客户" in q or "违约案例" in q:
        ent["kind"] = "bad"
    elif "好客户" in q:
        ent["kind"] = "good"
    return ent


def adjust_seq(q: str, ent: dict, seq: list, intents: list) -> list:
    # ① 抽到具体变量且问"问题/为什么" → 先看分箱细节
    if ent.get("feature"):
        if any(w in q for w in ("问题", "为什么", "原因", "单调", "分箱", "怎么分")):
            seq = ["woe_table"] + [t for t in seq if t != "woe_table"]
        else:
            seq = [t for t in seq if t != "woe_table"] + ["woe_table"]

    # ② 合规修复：用户明确要"剔除/去掉"时，必须用排除禁忌字段的模型重算，
    #    否则会出现"你说剔除了，但指标还是含禁忌字段那版"的自相矛盾
    if any(w in q for w in ("剔除", "去掉", "排除", "不含", "修复")):
        # 整条链路都要换成"剔除后"的模型，否则会出现阈值曲线和模型指标口径不一致
        keep = [t for t in seq if t not in ("perf_metrics", "cutoff_tradeoff")]
        seq = keep + [("cutoff_tradeoff", {"exclude_forbidden": 1}),
                      ("perf_metrics", {"exclude_forbidden": 1})]

    seen, out = set(), []
    for t in seq:
        key = t[0] if isinstance(t, (tuple, list)) else t
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


PLAN_SPEC = PlanSpec(
    intents=INTENTS,
    sequence=SEQUENCE,
    default_sequence=["dataset_profile", "iv_table"],
    entity_fn=extract_entities,
    anaphora=ANAPHORA,
    adjust_seq=adjust_seq,
    scope_keys=("dataset", "feature", "segment", "keep", "kind"),
    llm_hints=LLM_HINTS,
)
