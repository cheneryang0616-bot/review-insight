# -*- coding: utf-8 -*-
"""review 域 —— 规划配置（意图表 / 实体抽取 / 序列调整）

这里只放"评论归因"这个领域特有的知识，机制全在 insight.core.planner。
"""
import re

import pandas as pd

from ...core.planner import PlanSpec

INTENTS = {
    "overview": ["概况", "总体", "整体", "多少条", "数据集", "样本", "总量",
                 "覆盖", "什么情况", "情况怎么样", "看看数据"],
    "star": ["星级", "星", "分布", "几星"],
    "shop_rank": ["哪家", "哪些店", "店铺", "门店", "排名", "最差", "最好", "对比店"],
    "deal_rank": ["套餐", "团购", "哪款", "产品", "代金券", "券"],
    "deal_type": ["代金券", "券", "套餐类", "两类", "类型"],
    "aspect": ["为什么", "原因", "短板", "问题", "抱怨", "骂", "差在哪", "归因",
               "体验", "不满", "痛点", "服务", "口味", "环境", "排队", "分量"],
    "keyword": ["关键词", "都说", "都在说", "说什么", "说了什么", "提到", "高频",
                "出现", "吐槽"],
    "trend": ["趋势", "最近", "变多", "变差", "下滑", "一个月", "这季度", "变化", "走势"],
    "slot": ["时段", "星期", "周末", "几点", "高峰", "工作日"],
    "sample": ["例子", "举例", "原话", "具体评论", "抽样", "看看评论"],
}

SEQUENCE = {
    "overview": ["dataset_overview", "star_distribution"],
    "star": ["star_distribution"],
    "shop_rank": ["shop_ranking", "time_slot"],
    "deal_rank": ["deal_ranking", "deal_type_compare"],
    "deal_type": ["deal_type_compare", "deal_ranking"],
    "aspect": ["aspect_gap", "keyword_diff", "sample_comments"],
    "keyword": ["keyword_diff", "sample_comments"],
    "trend": ["time_trend", "time_slot"],
    "slot": ["time_slot", "time_trend"],
    "sample": ["sample_comments"],
}

ANAPHORA = ["这家", "那家", "该店", "它", "这个店", "同样", "那", "还有"]

# 通用词不能当具体套餐名，否则"套餐和代金券哪个差"会被误判成单边过滤
GENERIC_DEALS = {"套餐", "代金券", "双人餐", "4人餐", "2人餐", "团购套餐"}

LLM_HINTS = """- 时间趋势默认按季度（freq="Q"），除非用户明确说按月/按年
- 问题里出现具体店铺ID时，要用能吃 shop 参数的工具（star_distribution / aspect_gap 等），
  不要用 shop_ranking（它是跨店排名，会忽略 shop 参数）
- 上一轮的分析对象（见"当前上下文"）在用户说"这家店/它/那呢"时继续沿用"""


def extract_entities(q: str, df: pd.DataFrame) -> dict:
    """从问题里抽切片实体：店铺 / 套餐 / 年份 / 团购类型。"""
    ent = {}

    shops = set(df["shop"].dropna().unique())
    for m in re.findall(r"\d{5,}", q):
        if m in shops:
            ent["shop"] = m
            break

    deals = [d for d in df["deal"].dropna().unique()
             if isinstance(d, str) and d not in GENERIC_DEALS and len(d) >= 3]
    hits = [d for d in deals if d in q]
    if hits:
        ent["deal"] = max(hits, key=len)
        ent["deal_type"] = df.loc[df["deal"] == ent["deal"], "deal_type"].iloc[0]

    years = df["year"].dropna().unique()
    for m in re.findall(r"(?<!\d)(20\d{2})(?!\d)", q):
        if float(m) in years:
            ent["year"] = int(m)
            break

    # 只在"明确指向某一类"且没做对比时才设 deal_type：
    # ① "代金券怎么样" → 只看券  ② "套餐类团购呢" → 只看套餐
    # ③ "券和套餐哪个差" → 两者都问，不能过滤成单边
    if "deal_type" not in ent:
        has_coupon = "代金券" in q or "抵用券" in q
        has_combo_word = "套餐" in q
        has_combo_type = "套餐类" in q or "套餐型" in q
        if has_coupon and not has_combo_word:
            ent["deal_type"] = "代金券"
        elif has_combo_type and not has_coupon:
            ent["deal_type"] = "套餐"
    return ent


def adjust_seq(q: str, ent: dict, seq: list, intents: list) -> list:
    """领域特有的序列修正。三条都对应踩过的坑，别删。"""
    # ① 归因 + 趋势复合问题，两条线都要查
    if "aspect" in intents and "trend" in intents:
        seq = ["time_trend", "aspect_gap", "keyword_diff", "sample_comments"]

    # ② 有具体店铺 + 问到套餐 → 用户要的是"这家店的套餐"，不是全站套餐排名
    if ent.get("shop") and "deal_rank" in intents:
        seq = ["deal_ranking", "star_distribution"]

    # ③ 有具体店铺实体时，"店铺排名"吃不到 shop 参数，必须换成单店口径
    if ent.get("shop"):
        # deal_ranking / deal_type_compare 都吃 shop 参数，不要一起换掉，
        # 否则"某店的套餐差评率排名"会被回答成星级分布（答非所问）。
        keep = [t for t in seq if t in ("aspect_gap", "keyword_diff", "sample_comments",
                                        "time_trend", "time_slot", "deal_ranking",
                                        "deal_type_compare")]
        seq = ["star_distribution"] + keep

    # ④ 抽到具体套餐名：问"问题在哪"就做归因，问"情况如何"才给排名
    if ent.get("deal"):
        if any(w in q for w in ("问题", "为什么", "骂", "差在哪", "原因", "短板",
                                "吐槽", "抱怨")):
            seq = ["aspect_gap", "keyword_diff", "sample_comments"]
        else:
            seq = ["deal_ranking"] + [t for t in seq
                                      if t in ("deal_type_compare", "aspect_gap")]

    seen = set()
    return [t for t in seq if not (t in seen or seen.add(t))]


PLAN_SPEC = PlanSpec(
    intents=INTENTS,
    sequence=SEQUENCE,
    default_sequence=["dataset_overview", "star_distribution"],
    entity_fn=extract_entities,
    anaphora=ANAPHORA,
    adjust_seq=adjust_seq,
    scope_keys=("shop", "deal", "deal_type", "year"),
    llm_hints=LLM_HINTS,
)
