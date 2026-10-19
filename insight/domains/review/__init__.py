# -*- coding: utf-8 -*-
"""review 域 —— 美团团购评论归因（场景一）

这个文件只做"装配"：把数据、工具、规划配置、模板合成、业务断言拼成一个 Domain。
"""
import datetime

from ...core import gates as G
from ...core.agent import Domain
from . import bullets as bullets_mod
from . import data as data_mod
from . import planner_spec, tools

SYSTEM_PROMPT = (
    "你是美团团购评论的业务分析师。你只能使用给定工具结果表中的数字，"
    "禁止引入任何外部知识或估算。每条结论后面必须标注来源编号，形如 [T2]。"
    "输出 Markdown：先 3~5 条要点（每条一个事实 + 一句业务解读），"
    "再一段不超过 120 字的总判断。不要输出表格，不要复述表头。")


# ---------------------------------------------------------------- 业务断言
def _freshness(ctx):
    df = ctx.get("df")
    if df is None or not len(df):
        return True, "无数据，跳过"
    last = df["ts"].max()
    if last is None or str(last) == "NaT":
        return True, "无时间字段"
    days = (datetime.datetime.now() - last.to_pydatetime()).days
    ok = days <= 365
    return ok, (f"数据止于 {str(last)[:10]}，距今 {days} 天"
                + ("（结论只对该时期成立）" if not ok else ""))


GATES = [
    G.fact_gate("sample_size", "差评样本量必须足够支撑归因（≥30 条）",
                "bad_n", lambda v: v >= 30, G.WARN,
                "差评样本不足 30 条，方面归因结论不稳定"),
    G.Gate("freshness", "数据时效性：超过 1 年需提示结论不代表现状", G.WARN, _freshness),
]


CLARIFY_HINT = """- 数据概况与星级分布（「美团评论数据整体什么情况」）
- 店铺 / 套餐排名（「哪家店差评最多」「哪个套餐差评率最高」）
- 代金券与套餐对比（「代金券和套餐哪个差评更多」）
- 差评归因（「为什么差评这么多」「差评都在说什么」）
- 时间趋势与时段（「最近有没有变差」「哪个时段差评最多」）
- 原始评论抽样（「给我几条差评原话」）"""


class ReviewDomain(Domain):
    name = "review"
    registry = tools.REGISTRY
    plan_spec = planner_spec.PLAN_SPEC
    system_prompt = SYSTEM_PROMPT
    clarify_hint = CLARIFY_HINT
    gates = GATES

    def load(self):
        return data_mod.load()

    def bullets(self, res, idx):
        return bullets_mod.bullets(res, idx)

    def conclusion(self, results):
        return bullets_mod.conclusion(results)

    def scope_desc(self, shop=None, deal=None, deal_type=None, year=None):
        return data_mod.scope_desc(shop, deal, deal_type, year)

    def objects_hint(self, df):
        shops = sorted(set(df["shop"].dropna().unique()))[:20]
        deals = [d for d in df["deal"].dropna().unique()][:12]
        return ("店铺ID：" + ", ".join(shops) + "；"
                + "套餐示例：" + ", ".join(str(d) for d in deals))
