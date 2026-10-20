#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHM 的离线规则规划，不依赖外部 C-MAPSS 文件或工具实现。"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from insight.core.planner import rule_plan
from insight.core.registry import Registry
from insight.domains.phm.planner_spec import PLAN_SPEC, extract_entities


def registry():
    result = Registry("phm_planner_test")
    contracts = {
        "fleet_overview": ("dataset",),
        "unit_health": ("unit", "dataset"),
        "degradation_rate": ("unit", "dataset"),
        "sensor_attribution": ("unit", "dataset"),
        "uncertainty_audit": ("dataset",),
        "cost_threshold": ("c_fn", "c_fp", "risk_horizon_cycles", "dataset"),
        "maintenance_plan": ("lead_time_days", "window_days", "cycles_per_day", "dataset"),
        "model_compare": ("chosen", "dataset"),
    }
    for name, params in contracts.items():
        required = ("unit",) if "unit" in params else ()
        result.tool(name, name, {key: key for key in params}, required=required)(lambda df, **kw: {})
    return result


class TestPHMPlanner(unittest.TestCase):
    def setUp(self):
        self.registry = registry()
        self.df = pd.DataFrame({"unit": [1, 2, 34], "pred": [10., 20., 30.]})

    def plan(self, question, previous=None):
        return rule_plan(question, self.df, PLAN_SPEC, self.registry, previous)

    @staticmethod
    def names(plan):
        return [call["tool"] for call in plan["plan"]]

    def test_seven_intents_route_to_contract_tools(self):
        examples = {
            "机队总览": ("fleet", ["fleet_overview", "uncertainty_audit"]),
            "34号还能飞多久": ("unit", ["unit_health", "degradation_rate"]),
            "34号为什么退化": ("why", ["sensor_attribution", "unit_health"]),
            "34号预测区间可信吗": ("uncertain", ["uncertainty_audit", "unit_health"]),
            "告警阈值怎么设": ("threshold", ["cost_threshold", "uncertainty_audit"]),
            "请给维修排期": ("plan", ["maintenance_plan", "fleet_overview"]),
            "哪个模型NASA更好": ("model", ["model_compare", "fleet_overview"]),
        }
        for question, (intent, names) in examples.items():
            with self.subTest(question=question):
                result = self.plan(question)
                self.assertEqual(result["intent"], intent)
                self.assertEqual(self.names(result), names)

    def test_acceptance_prioritizes_two_week_maintenance(self):
        result = self.plan("哪几台两周内必须修")
        self.assertEqual(result["intent"], "plan")
        self.assertEqual(self.names(result), ["maintenance_plan", "fleet_overview"])
        self.assertEqual(result["entities"], {"window_days": 14})
        self.assertEqual(result["plan"][0]["params"], {"window_days": 14})

    def test_phase_zero_global_audit_does_not_require_unit(self):
        for question in ("能不能用这个预测区间安排停机检修？", "这个区间可信吗？",
                         "机队预测覆盖率有多少"):
            with self.subTest(question=question):
                result = self.plan(question)
                self.assertEqual(self.names(result), ["uncertainty_audit", "fleet_overview"])
                self.assertFalse(any(self.registry.missing_required(c["tool"], c["params"])
                                     for c in result["plan"]))

    def test_deictic_single_machine_uncertainty_requires_unit(self):
        for question in ("这台区间可信吗", "单机预测区间可信么", "该机敢不敢停机"):
            with self.subTest(question=question):
                result = self.plan(question)
                self.assertEqual(self.names(result), ["uncertainty_audit", "unit_health"])
                self.assertEqual(self.registry.missing_required("unit_health", result["plan"][1]["params"]),
                                 ["unit"])
        inherited = self.plan("这台区间可信吗", {"unit": 34})
        self.assertEqual(inherited["plan"][1]["params"], {"unit": 34})

    def test_missing_unit_in_single_machine_and_sensor_questions_requires_clarification(self):
        for question in ("这台还能飞多久", "哪个传感器是原因", "单机健康怎么样"):
            with self.subTest(question=question):
                result = self.plan(question)
                self.assertNotIn("unit", result["entities"])
                self.assertTrue(any(self.registry.missing_required(c["tool"], c["params"])
                                    for c in result["plan"]))

    def test_explicit_unit_variants_and_out_of_range_are_preserved(self):
        for question, unit in (("发动机34健康怎样", 34), ("发动机编号为34的健康状况", 34),
                               ("机器编号：2还能飞多久", 2),
                               ("unit 1", 1), ("unit34", 34), ("第999号机器", 999),
                               ("0号", 0), ("机器-1", -1), ("34", 34)):
            with self.subTest(question=question):
                result = self.plan(question)
                self.assertEqual(result["entities"]["unit"], unit)
                self.assertIn("unit_health", self.names(result))
                self.assertEqual(next(c["params"]["unit"] for c in result["plan"]
                                      if c["tool"] == "unit_health"), unit)

    def test_numbers_without_machine_context_are_not_unit(self):
        for question in ("备件14天到", "漏报是误报的10倍", "每天2个循环", "FD001机队总览",
                         "未来30个循环内风险", "14天内必须修", "漏报代价100，误报代价2",
                         "机器14天内必须修"):
            with self.subTest(question=question):
                self.assertNotIn("unit", extract_entities(question, self.df))

    def test_cost_ratio_and_explicit_costs_override_defaults(self):
        cases = {
            "漏报是误报的10倍": {"c_fn": 10, "c_fp": 1},
            "漏报是误报的20倍": {"c_fn": 20, "c_fp": 1},
            "漏报是误报的十倍": {"c_fn": 10, "c_fp": 1},
            "漏报代价100，误报代价2": {"c_fn": 100, "c_fp": 2},
            "漏报的成本为100，误报的成本为2.5": {"c_fn": 100, "c_fp": 2.5},
            "误报代价2，漏报是误报的10倍": {"c_fn": 20, "c_fp": 2},
            "漏报代价100，漏报是误报的10倍": {"c_fn": 100, "c_fp": 10},
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                result = self.plan(question)
                self.assertEqual(result["entities"], expected)
                self.assertEqual(result["plan"][0]["params"], expected)

    def test_time_assumptions_are_explicit_and_convert_weeks_only(self):
        result = self.plan("备件14天到，每天2个循环，未来两周维修排期")
        self.assertEqual(result["entities"], {
            "lead_time_days": 14, "cycles_per_day": 2, "window_days": 14})
        self.assertEqual(result["plan"][0]["params"], result["entities"])
        self.assertEqual(self.plan("提前期一周、窗口三周的排期")["entities"],
                         {"lead_time_days": 7, "window_days": 21})
        self.assertEqual(self.plan("备件两周内到，三周内维修")["entities"],
                         {"lead_time_days": 14, "window_days": 21})
        self.assertEqual(self.plan("备件两周内到")["entities"], {"lead_time_days": 14})

    def test_risk_horizon_cycles_is_distinct_from_daily_rate_and_days(self):
        question = "30个循环内告警阈值，备件14天到，每天2个循环"
        result = self.plan(question)
        self.assertEqual(result["entities"], {
            "risk_horizon_cycles": 30, "lead_time_days": 14, "cycles_per_day": 2})
        threshold = self.plan("风险窗口30循环，c_fn=100,c_fp=2")["plan"][0]
        self.assertEqual(threshold["tool"], "cost_threshold")
        self.assertEqual(threshold["params"], {"risk_horizon_cycles": 30, "c_fn": 100, "c_fp": 2})

    def test_dataset_is_passed_to_every_tool_and_chosen_only_to_model(self):
        for dataset in ("FD001", "FD002", "FD003", "FD004"):
            with self.subTest(dataset=dataset):
                result = self.plan(f"{dataset.lower()}选择GBDT，哪个模型更好")
                self.assertEqual(result["entities"], {"dataset": dataset, "chosen": "gbdt"})
                for call in result["plan"]:
                    self.assertEqual(call["params"]["dataset"], dataset)
                self.assertEqual(result["plan"][0]["params"]["chosen"], "gbdt")
                self.assertNotIn("chosen", result["plan"][1]["params"])

    def test_unsupported_explicit_dataset_is_preserved_for_resolver(self):
        result = self.plan("使用FD005查询机队总览")
        self.assertEqual(result["entities"], {"dataset": "FD005"})
        self.assertTrue(all(c["params"]["dataset"] == "FD005" for c in result["plan"]))

    def test_no_synthetic_defaults_in_entities_or_parameters(self):
        result = self.plan("维修排期")
        self.assertEqual(result["entities"], {})
        self.assertTrue(all(c["params"] == {} for c in result["plan"]))

    def test_invalid_explicit_assumptions_reach_tools_for_validation(self):
        # 非法假设也必须显式保留，不能悄悄退回默认成本/提前期。
        result = self.plan("漏报代价-100，误报代价2，提前期-1天的排期")
        self.assertEqual(result["entities"], {"c_fn": -100, "c_fp": 2, "lead_time_days": -1})

    def test_followup_changes_lead_time_and_keeps_previous_assumptions(self):
        previous = {"dataset": "FD002", "unit": 34, "c_fn": 100, "c_fp": 2,
                    "lead_time_days": 14, "window_days": 21, "cycles_per_day": 2,
                    "risk_horizon_cycles": 30, "chosen": "gbdt"}
        result = self.plan("那提前期改成7天呢", previous)
        self.assertEqual(result["entities"], {**previous, "lead_time_days": 7})
        self.assertEqual(result["plan"][0]["params"], {
            "dataset": "FD002", "lead_time_days": 7, "window_days": 21, "cycles_per_day": 2})
        self.assertEqual(previous["lead_time_days"], 14)

    def test_new_question_does_not_inherit_previous_machine_or_costs(self):
        previous = {"unit": 34, "dataset": "FD003", "c_fn": 100, "window_days": 21}
        result = self.plan("请重新查看全部机队的预测概览", previous)
        self.assertEqual(result["entities"], {})
        self.assertTrue(all(c["params"] == {} for c in result["plan"]))
        self.assertNotIn("unit_health", self.names(result))

    def test_followup_explicit_costs_replace_inherited_costs(self):
        result = self.plan("那漏报代价100，误报代价2", {"c_fn": 10, "c_fp": 1, "dataset": "FD001"})
        self.assertEqual(result["entities"], {"c_fn": 100, "c_fp": 2, "dataset": "FD001"})

    def test_parameter_only_followup_routes_to_relevant_tool(self):
        previous = {"window_days": 14, "lead_time_days": 7, "cycles_per_day": 2}
        window = self.plan("那窗口改成三周呢", previous)
        self.assertEqual(self.names(window), ["maintenance_plan", "fleet_overview"])
        self.assertEqual(window["entities"], {**previous, "window_days": 21})
        rate = self.plan("那每天改成3个循环呢", previous)
        self.assertEqual(self.names(rate), ["maintenance_plan", "fleet_overview"])
        self.assertEqual(rate["entities"], {**previous, "cycles_per_day": 3})

    def test_unknown_long_question_remains_default_for_framework_capability_hint(self):
        result = self.plan("请帮我分析明天的股票行情")
        self.assertEqual(result["intent"], "default")
        self.assertEqual(result["entities"], {})

    def test_shutdown_and_interval_schedule_always_audit(self):
        for question in ("请根据预测区间给出维修排期", "请安排停机工单"):
            with self.subTest(question=question):
                result = self.plan(question)
                self.assertEqual(self.names(result), ["uncertainty_audit", "maintenance_plan", "fleet_overview"])
        self.assertEqual(self.names(self.plan("请给点预测维修排期")),
                         ["maintenance_plan", "fleet_overview"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
