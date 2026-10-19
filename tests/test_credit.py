#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""credit 域测试：标签口径 / WOE-IV / 评分卡 / 合规闸门

注意：German Credit 的标签是 1=好客户、0=坏客户（不是 UCI 文档写的 1/2），
这里专门有一条测试守住这个口径——写错的话 IV 会全变成 inf。
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insight.core.agent import Agent
from insight.domains import get_domain
from insight.domains.credit import config, data, stats

try:
    _df0 = data.load()
except FileNotFoundError:
    _df0 = None

ACTIVE = _df0.attrs.get("dataset", "") if _df0 is not None else ""
IS_GERMAN = "german" in ACTIVE


def _D():
    return get_domain("credit")


@unittest.skipUnless(_df0 is not None, "缺少信贷数据")
class TestCreditData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = data.load()

    def test_shape_and_target(self):
        self.assertGreater(len(self.df), 500)
        self.assertEqual(set(self.df["y"].unique()) <= {0, 1}, True)
        self.assertGreater(self.df["y"].nunique(), 1, "标签只有单一类别")

    def test_profile_fields(self):
        p = data.profile(self.df)
        self.assertIn("bad_rate", p)
        self.assertGreater(p["bad_rate"], 0)
        self.assertLess(p["bad_rate"], 1)
        self.assertEqual(p["has_time"], 0)      # 两份数据都没有时间列

    @unittest.skipUnless(IS_GERMAN, "需要 German Credit")
    def test_german_label_mapping(self):
        # 1=好客户 700 条，0=坏客户 300 条
        self.assertEqual(len(self.df), 1000)
        self.assertEqual(int(self.df["y"].sum()), 300)
        self.assertAlmostEqual(self.df["y"].mean(), 0.3, places=6)

    @unittest.skipUnless(IS_GERMAN, "需要 German Credit")
    def test_forbidden_detected(self):
        self.assertIn("sex_marital", self.df.attrs["forbidden_hard"])
        self.assertIn("foreign_worker", self.df.attrs["forbidden_hard"])
        self.assertIn("age", self.df.attrs["forbidden_soft"])


@unittest.skipUnless(_df0 is not None, "缺少信贷数据")
class TestCreditStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = data.load()

    def test_woe_iv_finite(self):
        feat = self.df.attrs["features"][0]
        r = stats.feature_woe_iv(self.df[feat], self.df["y"])
        self.assertTrue(np.isfinite(r["iv"]))
        self.assertGreaterEqual(r["iv"], 0)

    def test_iv_table_sorted(self):
        rows = stats.iv_table(self.df, self.df.attrs["features"][:5])
        ivs = [r["iv"] for r in rows]
        self.assertEqual(ivs, sorted(ivs, reverse=True))

    def test_model_metrics_range(self):
        m = stats.fit(self.df)
        self.assertTrue(0.5 <= m["auc"] <= 1.0)
        self.assertTrue(0 <= m["ks"] <= 1.0)
        self.assertAlmostEqual(m["gini"], 2 * m["auc"] - 1, places=6)

    def test_cutoff_monotone_direction(self):
        m = stats.fit(self.df)
        rows = stats.cutoff_tradeoff(m["y_test"], m["p_test"])
        bad = [r["bad_rate"] for r in rows]
        # 通过率从高到低排列，坏账率不应上升
        self.assertTrue(all(bad[i] >= bad[i + 1] - 1e-9 for i in range(len(bad) - 1)))

    def test_score_monotone_in_logit(self):
        # 坏账概率越高（logit 越大），分数应越低
        s_low = stats.score_from_logit(-2.0)
        s_high = stats.score_from_logit(2.0)
        self.assertGreater(s_low, s_high)

    def test_exclude_forbidden_reduces_features(self):
        full = stats.fit(self.df, exclude_forbidden=False)
        safe = stats.fit(self.df, exclude_forbidden=True)
        self.assertLessEqual(len(safe["features"]), len(full["features"]))


@unittest.skipUnless(_df0 is not None, "缺少信贷数据")
class TestCreditAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = Agent(_D())

    def test_profile_question_grounded(self):
        r = self.agent.ask("数据整体什么情况？", use_llm=False)
        self.assertTrue(r["grounding"]["passed"])
        self.assertEqual(r["domain"], "credit")

    def test_cutoff_keep_param(self):
        r = self.agent.ask("通过率压到 70% 坏账率会怎样？", use_llm=False)
        self.assertEqual(r["scope"].get("keep"), 0.7)
        self.assertIn("cutoff_tradeoff", [c["tool"] for c in r["plan"]])

    @unittest.skipUnless(IS_GERMAN, "需要 German Credit")
    def test_forbidden_blocks_model(self):
        r = self.agent.ask("模型效果怎么样？", use_llm=False)
        names = {b["name"] for b in r["gates"]["blocking"]}
        self.assertIn("fair_lending", names)
        self.assertFalse(r["gates"]["passed"])

    @unittest.skipUnless(IS_GERMAN, "需要 German Credit")
    def test_after_exclusion_gate_passes(self):
        r = self.agent.ask("剔除性别和外籍劳工字段后模型效果怎么样？", use_llm=False)
        self.assertTrue(r["gates"]["passed"], r["gates"]["blocking"])
        facts = {}
        for res in r["results"]:
            facts.update(res.get("facts") or {})
        self.assertEqual(facts.get("exclude_forbidden"), 1)
        self.assertEqual(facts.get("forbidden_used"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
