#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHM Stage 1 integration: portable raw assets plus optional actual artifacts."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from insight.core import gates
from insight.core.agent import Agent
from insight.core.llm import OfflineStub
from insight.domains.phm import GATES, PHMDomain, config, data, tools


def make_agent(df):
    with patch.object(data, "load", return_value=df):
        return Agent(PHMDomain(), backend=OfflineStub())


def gate_named(result, name):
    return next(row for row in result["gates"]["results"] if row["name"] == name)


def write_fixture(root, dataset="FD001"):
    """A small complete dataset; source alignment and model reports stay real IO."""
    raw_path = Path(root) / "data"
    raw_path.mkdir(exist_ok=True)
    panel = []
    for unit in range(1, 4):
        for cycle in range(1, 61):
            sensors = [100. + index for index in range(1, 22)]
            for index in (2, 7, 11, 12, 15):
                sensors[index - 1] = float(cycle)
            panel.append([unit, cycle, 0, 0, 0] + sensors)
    panel = np.asarray(panel)
    np.savetxt(raw_path / ("train_%s.txt" % dataset), panel)
    np.savetxt(raw_path / ("test_%s.txt" % dataset), panel)
    np.savetxt(raw_path / ("RUL_%s.txt" % dataset), [10, 14, 80])
    artifacts = Path(root) / "artifacts"
    run = data.run_name(dataset)
    (artifacts / run).mkdir(parents=True, exist_ok=True)
    np.savez(artifacts / run / "predictions.npz", u_test=np.array([1, 2, 3]),
             y_test=np.array([10., 14., 80.]), pred_ensemble=np.array([9., 14., 80.]),
             sigma=np.array([1., 1., 2.]), end_cycle=np.array([60, 60, 60]),
             pred_members=np.array([[8., 13., 79.], [10., 15., 81.]]))
    summary = {}
    for index, model in enumerate(data.COMPARABLE_MODELS):
        metrics = {"n": 3, "rmse": 4. if model == "gbdt" else 5. + index,
                   "nasa_score": 6. if model == "gbdt" else 10. + index}
        summary[model] = metrics
        model_run = ("%s_cnn_bilstm_attn_w30_mrul125_gbdt" % dataset if model == "gbdt"
                     else data.run_name(dataset, model))
        model_path = artifacts / model_run
        model_path.mkdir(parents=True, exist_ok=True)
        report = {"run_name": model_run, "ensemble": metrics,
                  "config": {"data": {"dataset": dataset, "window": 30,
                                      "max_rul": 125, "rul_mode": "clip"}}}
        (model_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (artifacts / ("summary_%s.json" % dataset)).write_text(json.dumps(summary), encoding="utf-8")
    return data.load(root, dataset=dataset)


class TestPHMStageOnePortable(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.df = write_fixture(self.temporary.name)

    def test_eight_tools_produce_raw_facts_and_display_only_tables(self):
        calls = {
            "fleet_overview": {}, "unit_health": {"unit": 1},
            "degradation_rate": {"unit": 1}, "sensor_attribution": {"unit": 1},
            "uncertainty_audit": {}, "cost_threshold": {},
            "maintenance_plan": {}, "model_compare": {},
        }
        essential = {
            "fleet_overview": {"units", "need_service_30", "need_service_60", "median_rul", "worst_unit"},
            "unit_health": {"unit", "pred_rul", "sigma", "lower", "upper", "y_true", "abs_err", "in_region"},
            "degradation_rate": {"unit", "slope_per_cycle", "recent_drop", "is_accelerating"},
            "sensor_attribution": {"top_sensor", "top_sensor_contrib", "n_channels"},
            "uncertainty_audit": {"nominal", "empirical_coverage", "gap_pt", "mean_sigma", "usable_for_decision"},
            "cost_threshold": {"threshold_rul", "expected_cost", "compare_to_default", "threshold_violations"},
            "maintenance_plan": {"n_orders", "n_missed", "earliest_unit", "latest_order_day"},
            "model_compare": {"best_by_rmse", "best_by_nasa", "chosen", "nasa_gap"},
        }
        self.assertEqual(set(tools.REGISTRY.names()), set(calls))
        for name, params in calls.items():
            with self.subTest(tool=name):
                result = tools.REGISTRY.call(name, self.df, **params)
                self.assertTrue({"tool", "params", "table", "facts", "note", "insufficient"} <= set(result))
                self.assertFalse(result["insufficient"])
                self.assertFalse(result.get("error", False))
                self.assertTrue(essential[name] <= set(result["facts"]))
                self.assertTrue(result["facts"]["pipeline_ok"])
                self.assertTrue(result["table"])
                for row in result["table"]:
                    self.assertFalse(any(isinstance(value, (float, np.floating)) for value in row.values()))

    def test_each_industrial_gate_has_a_real_trigger_and_pass_boundary(self):
        cases = [
            ("coverage_gate", "empirical_coverage", .56, .85, gates.ERROR),
            ("lead_time_gate", "n_missed", 1, 0, gates.ERROR),
            ("rul_monotone", "bad_units", [1], [], gates.ERROR),
            ("threshold_monotone", "threshold_violations", [{"cost_ratio": 2}], [], gates.WARN),
            ("model_selection", "nasa_gap", 4., 0., gates.WARN),
            ("pipeline_selfcheck", "pipeline_ok", False, True, gates.ERROR),
        ]
        self.assertEqual({gate.name for gate in GATES}, {row[0] for row in cases})
        for name, key, invalid, valid, level in cases:
            for value, passed in ((invalid, False), (valid, True)):
                with self.subTest(gate=name, value=value):
                    result = gates.run_all(GATES, {"results": [{"facts": {key: value}}]})
                    outcome = next(row for row in result["results"] if row["name"] == name)
                    self.assertEqual(outcome["passed"], passed)
                    self.assertEqual(outcome["level"], level)
                    self.assertNotIn("异常", outcome["detail"])
                    self.assertNotIn("跳过", outcome["detail"])
                    self.assertEqual(result["passed"], passed or level == gates.WARN)

    def test_missing_unit_is_framework_clarification_before_tool_execution(self):
        agent = make_agent(self.df)
        result = agent.ask("请分析单机健康", use_llm=False)
        self.assertEqual(result["status"], "clarification")
        self.assertEqual(result["results"], [])
        self.assertIn("unit", result["answer"])
        self.assertIn("1..3", result["answer"])
        self.assertEqual(result["trajectory"][0]["action"], "clarify")

    def test_nonexistent_unit_is_insufficient_and_framework_degrades(self):
        for name in ("unit_health", "degradation_rate", "sensor_attribution"):
            with self.subTest(tool=name):
                result = tools.REGISTRY.call(name, self.df, unit=999)
                self.assertTrue(result["insufficient"])
                self.assertFalse(result.get("error", False))
                self.assertEqual(result["table"], [])
                self.assertIn("不存在", result["note"])
        result = make_agent(self.df).ask("机器999的健康", use_llm=False)
        self.assertEqual(result["status"], "degraded")
        self.assertTrue(all(row["insufficient"] for row in result["results"]))
        self.assertNotIn("预测 RUL 0", result["answer"])

    def test_summary_only_diagnostics_are_insufficient_without_asset_substitution(self):
        sample = data.load_sample()
        for name in ("degradation_rate", "sensor_attribution", "model_compare"):
            with self.subTest(tool=name):
                params = {"unit": 34} if name != "model_compare" else {}
                result = tools.REGISTRY.call(name, sample, **params)
                self.assertTrue(result["insufficient"])
                self.assertIn("样例", result["note"])
                self.assertFalse(result["facts"]["pipeline_ok"])
        result = make_agent(sample).ask("机器34健康", use_llm=False)
        self.assertEqual(result["status"], "degraded")
        self.assertTrue(any(row["tool"] == "unit_health" and not row["insufficient"]
                            for row in result["results"]))
        self.assertFalse(gate_named(result, "pipeline_selfcheck")["passed"])

    def test_unit_values_and_region_are_prediction_based(self):
        result = tools.unit_health(self.df, unit=1)
        facts = result["facts"]
        self.assertEqual(facts["pred_rul"], 9.)
        self.assertEqual(facts["y_true"], 10.)
        self.assertEqual(facts["abs_err"], 1.)
        self.assertEqual(facts["in_region"], "near")
        self.assertAlmostEqual(facts["lower"], 7.04)
        self.assertAlmostEqual(facts["upper"], 10.96)
        self.assertIn("离线", result["note"])
        self.assertIn("不作为停机授权", result["note"])

    def test_degradation_linear_proxy_has_independent_known_slope(self):
        panel = pd.DataFrame({"cycle": np.arange(1, 61), "sensor_2": np.arange(1, 61),
                              "sensor_7": np.arange(1, 61)})
        stats = pd.DataFrame({"mean": [0., 0.], "std": [1., 1.], "direction": [1., 1.]},
                             index=["sensor_2", "sensor_7"])
        with patch.object(tools, "_diagnostic_inputs", return_value=(panel, stats)):
            result = tools.degradation_rate(self.df, unit=1)
        self.assertAlmostEqual(result["facts"]["slope_per_cycle"], -1.)
        self.assertAlmostEqual(result["facts"]["recent_drop"], 29.)
        self.assertFalse(result["facts"]["is_accelerating"])
        self.assertIn("代理", result["note"])
        self.assertIn("不是 RUL", result["note"])

    def test_acceleration_requires_observed_later_window_evidence(self):
        cycles = np.arange(1, 61)
        values = np.where(cycles <= 30, cycles, 30 + 3 * (cycles - 30))
        panel = pd.DataFrame({"cycle": cycles, "sensor_2": values})
        stats = pd.DataFrame({"mean": [0.], "std": [1.], "direction": [1.]}, index=["sensor_2"])
        with patch.object(tools, "_diagnostic_inputs", return_value=(panel, stats)):
            result = tools.degradation_rate(self.df, unit=1)
        self.assertTrue(result["facts"]["is_accelerating"])
        self.assertLess(result["facts"]["slope_per_cycle"], -2.9)
        with patch.object(tools, "_diagnostic_inputs", return_value=(panel.iloc[:30], stats)):
            short = tools.degradation_rate(self.df, unit=1)
        self.assertTrue(short["insufficient"])
        self.assertNotIn("is_accelerating", short["facts"])

    def test_sensor_proxy_contribution_matches_hand_calculated_z_deviation(self):
        panel = pd.DataFrame({"cycle": np.arange(1, 61), "sensor_2": [8.] * 60,
                              "sensor_7": [1.] * 60})
        stats = pd.DataFrame({"mean": [0., 0.], "std": [2., 1.], "direction": [1., 1.]},
                             index=["sensor_2", "sensor_7"])
        with patch.object(tools, "_diagnostic_inputs", return_value=(panel, stats)):
            result = tools.sensor_attribution(self.df, unit=1)
        facts = result["facts"]
        self.assertEqual(facts["top_sensor"], "sensor_2")
        self.assertEqual(facts["n_channels"], 2)
        self.assertEqual(facts["top_sensor_deviation"], 4.)
        self.assertAlmostEqual(facts["top_sensor_contrib"], .8)
        self.assertIn("不是梯度归因", result["note"])
        self.assertIn("SHAP", result["note"])

    def test_explicit_cost_parameters_survive_audit_and_merged_facts(self):
        result = make_agent(self.df).ask("告警阈值，漏报代价20，误报代价2，风险窗口25循环", use_llm=False)
        self.assertEqual([row["tool"] for row in result["results"]],
                         ["cost_threshold", "uncertainty_audit"])
        self.assertTrue(result["grounding"]["passed"])
        for row in result["results"]:
            self.assertEqual(row["facts"]["c_fn"], 20.)
            self.assertEqual(row["facts"]["c_fp"], 2.)
            self.assertEqual(row["facts"]["risk_horizon_cycles"], 25.)
        combined = gates.merged_facts(result["results"])
        self.assertEqual(combined["c_fn"], 20.)
        self.assertEqual(combined["c_fp"], 2.)
        self.assertIn("20 : 2", result["answer"])
        self.assertIn("期望代价近似", result["answer"])

    def test_explicit_lead_window_and_rate_survive_fleet_fact_merge(self):
        result = make_agent(self.df).ask("维护工单，备件7天到，窗口14天，每天2循环", use_llm=False)
        self.assertEqual([row["tool"] for row in result["results"]],
                         ["maintenance_plan", "fleet_overview"])
        self.assertTrue(result["grounding"]["passed"])
        for row in result["results"]:
            self.assertEqual(row["facts"]["lead_time_days"], 7.)
            self.assertEqual(row["facts"]["window_days"], 14.)
            self.assertEqual(row["facts"]["cycles_per_day"], 2.)
        combined = gates.merged_facts(result["results"])
        self.assertEqual(combined["lead_time_days"], 7.)
        self.assertEqual(combined["cycles_per_day"], 2.)
        plan = next(row for row in result["results"] if row["tool"] == "maintenance_plan")
        self.assertEqual(plan["facts"]["order_units"], [1, 2])
        self.assertEqual([row["service_day"] for row in plan["orders"]], [4.5, 7.])
        self.assertEqual(plan["facts"]["n_missed"], 1)
        self.assertIn("每天 2 循环", result["answer"])

    def test_default_model_warns_and_chosen_gbdt_passes_without_switching_predictions(self):
        baseline = self.df["pred"].copy()
        default = make_agent(self.df).ask("哪个模型更好", use_llm=False)
        self.assertFalse(gate_named(default, "model_selection")["passed"])
        self.assertEqual(gate_named(default, "model_selection")["level"], gates.WARN)
        self.assertTrue(default["gates"]["passed"])
        selected = make_agent(self.df).ask("模型对比，选用GBDT", use_llm=False)
        self.assertTrue(gate_named(selected, "model_selection")["passed"])
        self.assertTrue(selected["grounding"]["passed"])
        comparison = next(row for row in selected["results"] if row["tool"] == "model_compare")
        self.assertEqual(comparison["facts"]["chosen"], "gbdt")
        self.assertEqual(comparison["facts"]["nasa_gap"], 0.)
        pd.testing.assert_series_equal(self.df["pred"], baseline)
        self.assertIn("不切换机队预测来源", selected["answer"])

    def test_fd002_is_loaded_for_every_tool_of_the_requested_plan(self):
        write_fixture(self.temporary.name, dataset="FD002")
        result = make_agent(self.df).ask("FD002未来两周检修工单，备件7天到", use_llm=False)
        self.assertEqual(result["scope"]["dataset"], "FD002")
        self.assertEqual(len(result["results"]), 2)
        for row in result["results"]:
            self.assertEqual(row["facts"]["dataset"], "FD002")
            self.assertTrue(row["facts"]["pipeline_ok"])
        self.assertTrue(result["grounding"]["passed"])
        self.assertTrue(result["gates"]["passed"])
        self.assertIn("FD002", result["answer"])

    def test_negative_rul_from_actual_tool_triggers_hard_gate(self):
        negative = self.df.copy()
        negative.loc[negative["unit"].eq(1), "pred"] = -2.
        result = make_agent(negative).ask("未来两周维护工单，备件0天到", use_llm=False)
        self.assertFalse(gate_named(result, "rul_monotone")["passed"])
        self.assertEqual(gate_named(result, "rul_monotone")["value"], [1])
        self.assertIn("存在负 RUL", result["answer"])

    def test_raw_label_corruption_is_detected_in_the_tool_pipeline_gate(self):
        corrupt = self.df.copy()
        corrupt.loc[corrupt["unit"].eq(1), "y_true"] = 11.
        result = make_agent(corrupt).ask("未来两周维护工单，备件7天到", use_llm=False)
        self.assertFalse(gate_named(result, "pipeline_selfcheck")["passed"])
        self.assertFalse(result["results"][0]["facts"]["labels_match"])
        self.assertIn("对齐检查未通过", result["answer"])


class TestPHMStageOneSampleAcceptance(unittest.TestCase):
    def test_two_week_candidates_are_units_34_and_81_and_default_lead_time_blocks(self):
        result = make_agent(data.load_sample()).ask("哪几台两周内必须修", use_llm=False)
        self.assertEqual([row["tool"] for row in result["results"]],
                         ["maintenance_plan", "fleet_overview"])
        self.assertTrue(result["grounding"]["passed"])
        plan = result["results"][0]
        self.assertEqual(plan["facts"]["order_units"], [34, 81])
        self.assertEqual(plan["facts"]["n_orders"], 2)
        self.assertEqual(plan["facts"]["n_missed"], 2)
        self.assertFalse(gate_named(result, "lead_time_gate")["passed"])
        self.assertIn("候选机器 34", result["answer"])
        self.assertIn("候选机器 81", result["answer"])
        self.assertIn("当前清单不可执行", result["answer"])
        self.assertIn("成本与提前期假设", result["answer"])

    def test_original_phase_zero_coverage_is_preserved_and_shutdown_is_refused(self):
        result = make_agent(data.load_sample()).ask("能不能用这个预测区间安排停机检修？", use_llm=False)
        self.assertTrue(result["grounding"]["passed"])
        self.assertIn("56%", result["answer"])
        self.assertFalse(gate_named(result, "coverage_gate")["passed"])
        self.assertIn("不能用这个预测区间安排停机检修", result["answer"])
        self.assertNotRegex(result["answer"], r"(可以|能够|可据此|建议立即).{0,12}停机")
        self.assertIn("仿真数据", result["answer"])


HAS_ACTUAL_ASSETS = (config.DEFAULT_ROOT / "artifacts" / config.RUN / "predictions.npz").is_file()


@unittest.skipUnless(HAS_ACTUAL_ASSETS, "独立 clone 不要求外部 C-MAPSS 原始资产")
class TestPHMStageOneActualAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = data.load(config.DEFAULT_ROOT)

    def test_actual_two_week_plan_block_and_seven_day_lead_time_pass(self):
        blocked = make_agent(self.df).ask("哪几台两周内必须修", use_llm=False)
        self.assertTrue(blocked["grounding"]["passed"])
        self.assertFalse(gate_named(blocked, "lead_time_gate")["passed"])
        self.assertTrue(gate_named(blocked, "pipeline_selfcheck")["passed"])
        passed = make_agent(self.df).ask("哪几台两周内必须修，备件7天到", use_llm=False)
        self.assertTrue(passed["grounding"]["passed"])
        self.assertTrue(passed["gates"]["passed"])
        self.assertTrue(gate_named(passed, "lead_time_gate")["passed"])
        plan = passed["results"][0]
        self.assertEqual(plan["facts"]["order_units"], [34, 81])
        self.assertEqual(plan["facts"]["n_missed"], 0)
        self.assertIn("仍需人工复核", passed["answer"])

    def test_actual_single_unit_and_sensor_queries_ground_every_number(self):
        for question in ("机器34健康", "机器34为什么退化，哪个传感器"):
            with self.subTest(question=question):
                result = make_agent(self.df).ask(question, use_llm=False)
                self.assertEqual(result["status"], "ok")
                self.assertTrue(result["grounding"]["passed"], result["grounding"])
                self.assertTrue(all(not row.get("insufficient") for row in result["results"]))
                self.assertTrue(gate_named(result, "pipeline_selfcheck")["passed"])

    def test_actual_model_nasa_warning_changes_when_gbdt_selected(self):
        default = make_agent(self.df).ask("哪个模型更好", use_llm=False)
        self.assertTrue(default["grounding"]["passed"])
        self.assertFalse(gate_named(default, "model_selection")["passed"])
        chosen = make_agent(self.df).ask("模型对比，选用GBDT", use_llm=False)
        self.assertTrue(chosen["grounding"]["passed"])
        self.assertTrue(gate_named(chosen, "model_selection")["passed"])


class TestPHMStageOneRegressions(unittest.TestCase):
    def test_zero_lead_time_survives_window_followup(self):
        with tempfile.TemporaryDirectory() as root:
            agent = make_agent(write_fixture(root))
            first = agent.ask("备件0天到，每天2个循环，未来两周维修排期", use_llm=False)
            self.assertEqual(first["results"][0]["facts"]["lead_time_days"], 0)
            followup = agent.ask("那窗口改成三周呢", use_llm=False)
        self.assertEqual(followup["scope"]["lead_time_days"], 0)
        self.assertEqual(followup["results"][0]["tool"], "maintenance_plan")
        self.assertEqual(followup["results"][0]["facts"]["window_days"], 21)
        self.assertEqual(followup["results"][0]["facts"]["cycles_per_day"], 2)
        self.assertTrue(followup["grounding"]["passed"])

    def test_empty_fleet_never_skips_pipeline_admission(self):
        empty = data.load_sample().iloc[:0]
        for tool in (tools.fleet_overview, tools.uncertainty_audit):
            result = tool(empty)
            self.assertTrue(result["insufficient"])
            g = gates.run_all(GATES, {"results": [result]})
            self.assertIn("pipeline_selfcheck", [row["name"] for row in g["blocking"]])

    def test_unsupported_dataset_degrades_without_composition_crash(self):
        result = make_agent(data.load_sample()).ask("使用FD005查询机队总览", use_llm=False)
        self.assertEqual(result["status"], "degraded")
        self.assertTrue(any(row.get("error") for row in result["results"]))
        self.assertFalse(result["gates"]["passed"])
        self.assertTrue(result["grounding"]["passed"])
        self.assertNotRegex(result["answer"], r"(可以|能够|建议立即).{0,12}停机")


if __name__ == "__main__":
    unittest.main(verbosity=2)
