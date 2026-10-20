#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""框架层测试：工具注册表 / 数字回查 / 断言闸门 / 通用规划器

这些都**不依赖真实数据**——质检是否可用，不该取决于今天加载了哪份数据。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insight.core import gates as G
from insight.core import grounding, planner
from insight.core.registry import Registry

RES = {
    "tool": "demo", "note": "",
    "table": [{"类型": "代金券", "评论数": 14447, "差评率": "7.39%"},
              {"类型": "套餐", "评论数": 16014, "差评率": "3.92%"}],
    "facts": {"coupon_bad_rate": 0.0739, "coupon_n": 14447,
              "combo_bad_rate": 0.0392, "combo_n": 16014, "ratio": 1.89},
}


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.r = Registry("test")
        self.r.tool("echo", "回显", {"x": "值"})(lambda df, x=1: {"tool": "echo", "table": [],
                                                              "facts": {"x": x}})

    def test_call(self):
        self.assertEqual(self.r.call("echo", None, x=5)["facts"]["x"], 5)

    def test_unknown_tool(self):
        with self.assertRaises(KeyError):
            self.r.call("nope", None)

    def test_unknown_param(self):
        with self.assertRaises(ValueError):
            self.r.call("echo", None, y=1)

    def test_signatures(self):
        self.assertIn("echo", self.r.signatures())


class TestGrounding(unittest.TestCase):
    def test_catches_fabricated_number(self):
        g = grounding.check("差评率 42.5%，是套餐的 1.89 倍。", [RES])
        self.assertFalse(g["passed"])
        self.assertEqual([u["raw"] for u in g["unsupported"]], ["42.5"])

    def test_both_scales(self):
        self.assertTrue(grounding.check("差评率 7.39%。", [RES])["passed"])
        self.assertTrue(grounding.check("差评率 0.0739。", [RES])["passed"])

    def test_ratio(self):
        self.assertTrue(grounding.check("是另一类的 1.89 倍。", [RES])["passed"])

    def test_structural_and_years_skipped(self):
        g = grounding.check("从 7 个方面看，2019 到 2021 年，差评率 7.39%。", [RES])
        self.assertTrue(g["passed"])
        self.assertGreaterEqual(g["skipped_structural"], 3)

    def test_rounding_is_not_hallucination(self):
        res = {"tool": "x", "table": [], "facts": {"spread": 2.2626}}
        self.assertTrue(grounding.check("极差 2.3 个百分点", [res])["passed"])
        self.assertFalse(grounding.check("极差 2.5 个百分点", [res])["passed"])

    def test_quoted_text_excluded(self):
        # 引用原文里的数字不算结论数字（"等了 20 分钟"）
        g = grounding.check("「等了 20 分钟」是原话。", [RES])
        self.assertTrue(g["passed"])

    def test_label_numbers_excluded(self):
        g = grounding.check("超过 P99 的 5 倍，KS 0.42。", [{"tool": "x", "table": [],
                                                          "facts": {"ks": 0.42}}])
        self.assertTrue(g["passed"])

    def test_strip_bad_sentences(self):
        out = grounding.strip_bad_sentences("真的 7.39%。\n假的 99.9%。", [{"raw": "99.9"}])
        self.assertIn("7.39%", out)
        self.assertNotIn("99.9", out)


class TestGates(unittest.TestCase):
    def _ctx(self, facts):
        return {"answer": "", "results": [{"tool": "t", "facts": facts}], "df": None}

    def test_fact_gate_pass_and_fail(self):
        gate = G.fact_gate("ks", "KS 下限", "ks", lambda v: v >= 0.3, G.ERROR)
        self.assertTrue(gate.run(self._ctx({"ks": 0.45})).passed)
        self.assertFalse(gate.run(self._ctx({"ks": 0.2})).passed)

    def test_gate_skipped_when_fact_missing(self):
        gate = G.fact_gate("ks", "KS 下限", "ks", lambda v: v >= 0.3, G.ERROR)
        r = gate.run(self._ctx({}))
        self.assertTrue(r.passed)
        self.assertIn("跳过", r.detail)

    def test_forbidden_intersection(self):
        gate = G.fact_in_gate("fair", "禁忌字段", "used", {"sex", "foreign"}, G.ERROR)
        self.assertTrue(gate.run(self._ctx({"used": []})).passed)
        self.assertFalse(gate.run(self._ctx({"used": ["sex"]})).passed)

    def test_run_all_blocking_vs_warning(self):
        specs = [G.fact_gate("a", "硬", "x", lambda v: v > 1, G.ERROR),
                 G.fact_gate("b", "软", "y", lambda v: v > 1, G.WARN)]
        res = G.run_all(specs, self._ctx({"x": 0, "y": 0}))
        self.assertFalse(res["passed"])
        self.assertEqual([b["name"] for b in res["blocking"]], ["a"])
        self.assertEqual([w["name"] for w in res["warnings"]], ["b"])

    def test_trace_gate_wraps_grounding(self):
        gate = G.trace_gate(lambda a, r: grounding.check(a, r))
        ok = gate.run({"answer": "7.39%", "results": [RES]})
        bad = gate.run({"answer": "99%", "results": [RES]})
        self.assertTrue(ok.passed)
        self.assertFalse(bad.passed)


class TestPlanner(unittest.TestCase):
    class FakeDF:
        attrs = {"features": ["a", "b"]}

    def _reg(self):
        r = Registry("x")
        for name in ("perf_metrics", "dataset_profile"):
            r.tool(name, "占位", {})(lambda df, _n=name: {"tool": _n, "facts": {},
                                                        "table": []})
        return r

    def _spec(self):
        return planner.PlanSpec(
            intents={"perf": ["效果"], "profile": ["概况"]},
            sequence={"perf": ["perf_metrics"], "profile": ["dataset_profile"]},
            default_sequence=["dataset_profile"],
            entity_fn=lambda q, df: {"feature": "a"} if "变量a" in q else {},
            anaphora=["这个变量"])

    def test_is_followup(self):
        self.assertTrue(planner.is_followup("那这个变量呢", ["这个变量"]))
        self.assertFalse(planner.is_followup("模型效果怎么样，请详细说说全部指标", ["这个变量"]))

    def test_intent_sequence(self):
        r = planner.rule_plan("模型效果怎么样", self.FakeDF(), self._spec(), self._reg())
        self.assertEqual([c["tool"] for c in r["plan"]], ["perf_metrics"])

    def test_scope_not_inherited_without_followup(self):
        r = planner.rule_plan("模型效果怎么样吧", self.FakeDF(), self._spec(),
                              self._reg(), prev_scope={"feature": "b"})
        self.assertNotIn("feature", r["entities"])

    def test_followup_preserves_zero_values_and_skips_none(self):
        spec = self._spec()
        spec.scope_keys = ("lead_time_days", "c_fn", "feature")
        r = planner.rule_plan("那这个变量的效果怎么样", self.FakeDF(), spec, self._reg(),
                              prev_scope={"lead_time_days": 0, "c_fn": 0, "feature": None})
        self.assertEqual(r["entities"]["lead_time_days"], 0)
        self.assertEqual(r["entities"]["c_fn"], 0)
        self.assertNotIn("feature", r["entities"])

    def test_tuple_entry_injects_params(self):
        reg = Registry("x")
        reg.tool("perf_metrics", "指标", {"exclude_forbidden": "0/1"})(
            lambda df, exclude_forbidden=0: {"tool": "perf_metrics",
                                             "facts": {"e": exclude_forbidden}, "table": []})
        spec = self._spec()
        spec.sequence["perf"] = [("perf_metrics", {"exclude_forbidden": 1})]
        r = planner.rule_plan("模型效果怎么样", self.FakeDF(), spec, reg)
        self.assertEqual(r["plan"][0]["params"]["exclude_forbidden"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
