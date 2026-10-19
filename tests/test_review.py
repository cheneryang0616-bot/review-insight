#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单元测试（标准库 unittest，不依赖 pytest）

覆盖四层：数据口径 / 统计口径 / 工具契约 / 质检与规划。
重点是 grounding —— 一个抓不到幻觉的质检层等于没有。
运行：python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insight.core import grounding, planner
from insight.core.agent import Agent
from insight.domains import get_domain
from insight.domains.review import data, planner_spec, stats, tools

# 完整数据（4.2 万行）不在仓库里，只有脱敏样例（225 行）。
# 依赖具体数字与规模的断言，在样例模式下自动跳过，而不是假装通过。
def _D():
    return get_domain("review")


try:
    _SRC = data.load().attrs.get("source", "sample")
except FileNotFoundError:
    _SRC = None
FULL = _SRC == "full"
needs_full = unittest.skipUnless(FULL, "需要完整数据（data/meituan_reviews.xlsx）")


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = data.load()

    @needs_full
    def test_shape(self):
        self.assertEqual(len(self.df), 42038)
        self.assertEqual(self.df["shop"].nunique(), 21)

    def test_star_scale(self):
        # 星级被压回 1~5
        self.assertAlmostEqual(self.df["star"].min(), 0.5)
        self.assertAlmostEqual(self.df["star"].max(), 5.0)

    @needs_full
    def test_bad_definition(self):
        self.assertEqual(int(self.df["bad"].sum()), 2262)
        self.assertAlmostEqual(self.df["bad"].mean(), 0.0538, places=4)

    def test_template_prefix_removed(self):
        # 「#套餐：xxx」前缀必须被清掉，否则污染关键词
        self.assertFalse(self.df["text"].str.startswith("「").any())

    @needs_full
    def test_scope_filter(self):
        d = data.scope_df(self.df, shop="4134768")
        self.assertEqual(len(d), 873)
        self.assertTrue((d["shop"] == "4134768").all())

    def test_deal_type(self):
        self.assertEqual(data.classify_deal("50元代金券1张"), "代金券")
        self.assertEqual(data.classify_deal("男票甜蜜双人餐"), "套餐")
        self.assertEqual(data.classify_deal(None), "未标注")


class TestStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = data.load()

    def test_star_share_sums_to_one(self):
        sd = stats.star_dist(self.df)
        self.assertAlmostEqual(sum(v["share"] for v in sd.values()), 1.0, places=3)

    def test_ranking_respects_min_n(self):
        rows = stats.ranking(self.df, "shop", min_n=10000, top_k=5)
        self.assertTrue(all(r["n"] >= 10000 for r in rows))

    @needs_full
    def test_log_odds_direction(self):
        d = self.df[self.df["usable_text"]]
        cb = stats.corpus(d[d["bad"] == 1]["text"])
        cg = stats.corpus(d[d["bad"] == 0]["text"])
        rows = {r["word"]: r for r in stats.log_odds(cb, cg, sum(cb.values()),
                                                      sum(cg.values()), top_k=200)}
        self.assertIn("难吃", rows)
        self.assertGreater(rows["难吃"]["z"], 0)      # 难吃应偏向差评
        self.assertGreater(rows["难吃"]["bad"], rows["难吃"]["good"])

    @needs_full
    def test_aspect_gap_direction(self):
        ag = stats.aspect_gap(self.df[self.df["usable_text"]])
        # 服务在差评里的负面率必须远高于好评
        self.assertGreater(ag["service"]["neg_bad_rate"], ag["service"]["neg_good_rate"] * 5)
        self.assertGreater(ag["service"]["neg_lift"], 5)

    @needs_full
    def test_deal_type_compare(self):
        rows = {r["type"]: r for r in stats.deal_type_compare(self.df)}
        self.assertGreater(rows["代金券"]["bad_rate"], rows["套餐"]["bad_rate"])
        self.assertAlmostEqual(rows["代金券"]["bad_rate"] / rows["套餐"]["bad_rate"], 1.89, places=2)


class TestTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = data.load()

    def test_registry_size(self):
        self.assertEqual(len(tools.REGISTRY), 10)

    def test_unknown_tool(self):
        with self.assertRaises(KeyError):
            tools.call("nope", self.df)

    def test_unknown_param(self):
        with self.assertRaises(ValueError):
            tools.call("dataset_overview", self.df, foo=1)

    def test_every_tool_returns_facts_and_table(self):
        base = [{}, {"shop": "4134768"}, {"year": 2019}, {"dim": "hour"},
                {"deal": "50元代金券（菜品）1张"}, {"top_k": 5}, {"min_n": 200}]
        for name, spec in tools.REGISTRY.tools.items():
            res, last = None, None
            for extra in base:
                params = {k: v for k, v in extra.items() if k in spec["params"]}
                try:
                    res = tools.call(name, self.df, **params)
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
            self.assertIsNotNone(res, f"{name} 无法用任何参数组合跑通：{last}")
            self.assertIn("facts", res)
            self.assertIn("table", res)

    def test_shop_ranking_excludes_small(self):
        res = tools.call("shop_ranking", self.df, min_n=5000)
        self.assertTrue(all(r["评论数"] >= 5000 for r in res["table"]))


class TestGrounding(unittest.TestCase):
    """质检层的测试用**构造的证据**，不依赖真实数据 —— 换数据不该影响质检是否可用。"""

    @classmethod
    def setUpClass(cls):
        cls.res = {
            "tool": "deal_type_compare",
            "note": "",
            "table": [{"团购类型": "代金券", "评论数": 14447, "差评率": "7.39%"},
                      {"团购类型": "套餐", "评论数": 16014, "差评率": "3.92%"}],
            "facts": {"coupon_bad_rate": 0.0739, "coupon_n": 14447,
                      "combo_bad_rate": 0.0392, "combo_n": 16014, "ratio": 1.89},
        }

    def test_catches_fabricated_number(self):
        g = grounding.check("代金券差评率 42.5%，是套餐的 1.89 倍。", [self.res])
        self.assertFalse(g["passed"])
        self.assertEqual([u["raw"] for u in g["unsupported"]], ["42.5"])

    def test_accepts_real_numbers_both_scales(self):
        g1 = grounding.check("代金券差评率 7.39%。", [self.res])
        g2 = grounding.check("代金券差评率 0.0739。", [self.res])
        self.assertTrue(g1["passed"])
        self.assertTrue(g2["passed"])

    def test_ratio_number(self):
        g = grounding.check("代金券类是套餐类的 1.89 倍。", [self.res])
        self.assertTrue(g["passed"])

    def test_skips_structural_and_years(self):
        g = grounding.check("从 7 个方面看，2019 到 2021 年，代金券差评率 7.39%。",
                            [self.res])
        self.assertTrue(g["passed"])
        self.assertGreaterEqual(g["skipped_structural"], 3)

    def test_extra_text_covers_scope(self):
        g = grounding.check("店铺 4134768 的差评率 7.39%。", [self.res], ["店铺 4134768"])
        self.assertTrue(g["passed"])

    def test_rounding_is_not_hallucination(self):
        # 真实值 2.2626 → 写 2.3 是取整，应通过；写 2.5 应失败
        res = {"tool": "x", "table": [], "facts": {"spread": 2.2626}}
        self.assertTrue(grounding.check("极差 2.3 个百分点", [res])["passed"])
        self.assertFalse(grounding.check("极差 2.5 个百分点", [res])["passed"])

    def test_tolerance(self):
        g = grounding.check("代金券差评率 7.4%。", [self.res])
        self.assertTrue(g["passed"])

    def test_strip_bad_sentences(self):
        text = "第一行 7.39% 是真的。\n第二行 99.9% 是编的。"
        out = grounding.strip_bad_sentences(text, [{"raw": "99.9"}])
        self.assertIn("7.39%", out)
        self.assertNotIn("99.9", out)


class TestPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = data.load()

    def test_year_entity(self):
        ent = planner_spec.extract_entities("2019年的情况", self.df)
        self.assertEqual(ent.get("year"), 2019)

    def test_shop_entity(self):
        ent = planner_spec.extract_entities("店铺 4134768 怎么样", self.df)
        self.assertEqual(ent.get("shop"), "4134768")

    def test_deal_entity_longest_match(self):
        ent = planner_spec.extract_entities("50元代金券（菜品）1张 有问题吗", self.df)
        self.assertEqual(ent.get("deal"), "50元代金券（菜品）1张")

    def test_generic_word_not_entity(self):
        ent = planner_spec.extract_entities("套餐和代金券哪个差", self.df)
        self.assertNotIn("deal", ent)
        self.assertNotIn("deal_type", ent)

    def test_anaphora_inherits_shop(self):
        plan = planner.rule_plan("那这家店差评在骂什么", self.df,
                                 planner_spec.PLAN_SPEC, tools.REGISTRY,
                                 prev_scope={"shop": "4134768"})
        self.assertEqual(plan["entities"].get("shop"), "4134768")

    def test_shop_entity_uses_single_shop_tool(self):
        plan = planner.rule_plan("店铺 4134768 的差评情况", self.df,
                                 planner_spec.PLAN_SPEC, tools.REGISTRY)
        names = [c["tool"] for c in plan["plan"]]
        self.assertIn("star_distribution", names)
        self.assertNotIn("shop_ranking", names)

    def test_deal_question_goes_to_attribution(self):
        plan = planner.rule_plan("50元代金券（菜品）1张 的问题在哪", self.df,
                                 planner_spec.PLAN_SPEC, tools.REGISTRY)
        names = [c["tool"] for c in plan["plan"]]
        self.assertIn("aspect_gap", names)


class TestAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = Agent(_D())

    @needs_full
    def test_offline_answer_passes_grounding(self):
        for q in ["代金券和套餐哪个差评更多？", "哪家店差评最多？", "为什么差评这么多"]:
            res = self.agent.ask(q, use_llm=False)
            self.assertTrue(res["grounding"]["passed"], f"{q} 未通过质检")
            self.assertEqual(res["composer"], "template")

    @needs_full
    def test_scope_memory(self):
        self.agent.ask("店铺 4134768 的差评情况", use_llm=False)
        self.assertEqual(self.agent.scope.get("shop"), "4134768")
        res = self.agent.ask("那这家店呢", use_llm=False)
        self.assertEqual(res["scope"].get("shop"), "4134768")

    @needs_full
    def test_new_question_does_not_inherit_scope(self):
        a = Agent(_D())
        a.ask("2020年代金券和套餐的差评率", use_llm=False)
        self.assertEqual(a.scope.get("year"), 2020)
        res = a.ask("为什么差评这么多？短板在哪？", use_llm=False)
        self.assertEqual(res["scope"], {}, "新问题不应继承上一轮的年份")

    @needs_full
    def test_followup_inherits_scope(self):
        a = Agent(_D())
        a.ask("店铺 4134768 的差评情况", use_llm=False)
        res = a.ask("那这家店呢", use_llm=False)
        self.assertEqual(res["scope"].get("shop"), "4134768")

    def test_history_accumulates(self):
        n0 = len(self.agent.history)
        self.agent.ask("数据整体情况", use_llm=False)
        self.assertEqual(len(self.agent.history), n0 + 1)

    @needs_full
    def test_answer_has_citations(self):
        res = self.agent.ask("代金券和套餐哪个差评更多？", use_llm=False)
        self.assertIn("[T1]", res["answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
