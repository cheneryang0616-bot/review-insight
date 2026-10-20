#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHM Phase 0：真实缺陷、通过边界、输入对齐和离线拒绝行为。"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from insight.core import gates, grounding
from insight.core.agent import Agent
from insight.core.llm import OfflineStub
from insight.domains import get_domain
from insight.domains.phm import GATES, config, data, tools


def sample_agent():
    with patch.object(data, "load", return_value=data.load_sample()):
        return Agent(get_domain("phm"), backend=OfflineStub())


class TestPHMData(unittest.TestCase):
    def test_sample_has_unit_contract_and_derived_intervals(self):
        df = data.load_sample()
        self.assertEqual(len(df), 100)
        self.assertEqual(df["unit"].tolist(), list(range(1, 101)))
        self.assertTrue(set(data.BASE_COLUMNS + ("region", "lower", "upper")) <= set(df))
        np.testing.assert_allclose(df["lower"], df["pred"] - 1.96 * df["sigma"])
        np.testing.assert_allclose(df["upper"], df["pred"] + 1.96 * df["sigma"])
        self.assertTrue((df.loc[df["pred"] <= 30, "region"] == "near").all())
        self.assertEqual(df.attrs["source"], "sample")

    @staticmethod
    def _archive(root, **changes):
        values = {
            "u_test": np.array([1, 2, 3]), "y_test": np.array([10., 20., 30.]),
            "pred_ensemble": np.array([12., 22., 32.]), "sigma": np.array([1., 2., 3.]),
            "end_cycle": np.array([100, 101, 102]),
            "pred_members": np.array([[11., 21., 31.], [13., 23., 33.]]),
        }
        values.update(changes)
        path = Path(root) / "artifacts" / config.RUN / "predictions.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **values)
        return path

    def test_npz_member_axis_and_values(self):
        with tempfile.TemporaryDirectory() as root:
            self._archive(root)
            df = data.load(root)
        self.assertEqual(df["pred"].tolist(), [12., 22., 32.])
        np.testing.assert_allclose(df["member_std"], [1., 1., 1.])
        self.assertEqual(df.attrs["source"], "full")

    def test_npz_rejects_bad_shapes_uncertainties_and_ids(self):
        cases = [
            {"y_test": np.array([1., 2.])},
            {"u_test": np.array(1)},
            {"pred_members": np.ones((3, 2))},
            {"pred_members": np.full((2, 3), np.nan)},
            {"sigma": np.array([1., -1., 1.])},
            {"sigma": np.array([1., np.inf, 1.])},
            {"u_test": np.array([1, 1, 3])},
            {"u_test": np.array([1, 2, 4])},
            {"u_test": np.array([1., 2.5, 3.])},
        ]
        with tempfile.TemporaryDirectory() as root:
            for changes in cases:
                with self.subTest(changes=changes):
                    self._archive(root, **changes)
                    with self.assertRaises(ValueError):
                        data.load(root)

    def test_npz_missing_required_field(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._archive(root)
            np.savez(path, u_test=np.array([1]))
            with self.assertRaisesRegex(ValueError, "缺少字段"):
                data.load(root)

    def test_explicit_invalid_root_does_not_silently_use_sample(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(os.environ, {"RI_PHM_ROOT": str(Path(root) / "missing")}):
                with self.assertRaises(FileNotFoundError):
                    data.load()

    def test_missing_default_root_uses_portable_sample(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(config, "DEFAULT_ROOT", Path(root) / "missing"):
                    self.assertEqual(data.load().attrs["source"], "sample")

    @unittest.skipUnless(
        (config.DEFAULT_ROOT / "artifacts" / config.RUN / "predictions.npz").is_file(),
        "独立 clone 不要求外部 C-MAPSS 资产")
    def test_sample_matches_actual_asset_and_upstream_coverage(self):
        full, sample = data.load(config.DEFAULT_ROOT), data.load_sample()
        for column in data.BASE_COLUMNS:
            np.testing.assert_allclose(sample[column], full[column], rtol=1e-12, atol=1e-12)
        self.assertEqual(data.coverage(full), 0.56)
        self.assertEqual(data.coverage(sample), data.coverage(full))


class TestPHMToolsAndGates(unittest.TestCase):
    def test_fleet_counts_use_predictions_and_include_thresholds(self):
        df = data.load_sample().iloc[:4].copy()
        df["pred"], df["y_true"] = [20., 30., 60., 61.], [100.] * 4
        facts = tools.fleet_overview(df)["facts"]
        self.assertEqual(facts["need_service_30"], 2)
        self.assertEqual(facts["need_service_60"], 3)
        self.assertEqual(facts["median_rul"], 45.)
        self.assertEqual(facts["worst_unit"], 1)

    def test_coverage_uses_closed_interval_and_actual_labels(self):
        df = data.load_sample().iloc[:3].copy()
        df["pred"], df["y_true"], df["sigma"] = (
            [0., 10., 20.], [1.96, 11.96001, 17.], [1., 1., 2.])
        result = tools.uncertainty_audit(df)
        self.assertAlmostEqual(result["facts"]["empirical_coverage"], 2 / 3)
        self.assertFalse(result["facts"]["usable_for_decision"])

    def test_real_coverage_facts_are_raw_and_tables_are_formatted(self):
        df = data.load_sample()
        audit = tools.uncertainty_audit(df)
        self.assertEqual(audit["facts"]["empirical_coverage"], 0.56)
        self.assertAlmostEqual(audit["facts"]["gap_pt"], 39.)
        self.assertEqual(audit["facts"]["mean_sigma"], float(df["sigma"].mean()))
        self.assertNotEqual(audit["facts"]["mean_sigma"], round(df["sigma"].mean(), 2))
        for result in (audit, tools.fleet_overview(df)):
            for key in ("tool", "params", "table", "facts", "note", "insufficient"):
                self.assertIn(key, result)
            for row in result["table"]:
                self.assertFalse(any(isinstance(v, (float, np.floating)) for v in row.values()))

    def test_coverage_gate_blocks_and_passes_including_boundary(self):
        for coverage, want in ((0.56, False), (0.849, False), (0.85, True), (0.95, True)):
            with self.subTest(coverage=coverage):
                result = gates.run_all(GATES, {
                    "results": [{"facts": {"empirical_coverage": coverage}}],
                })
                self.assertEqual(result["passed"], want)
                self.assertEqual(result["results"][0]["name"], "coverage_gate")
                self.assertNotIn("异常", result["results"][0]["detail"])
                if not want:
                    self.assertEqual(result["blocking"][0]["level"], gates.ERROR)

    def test_gate_and_tool_agree_at_admission_boundary(self):
        df = data.load_sample().iloc[:20].copy()
        df["y_true"] = df["pred"]
        df.loc[df.index[:3], "y_true"] = df.loc[df.index[:3], "pred"] + 10000
        # 本测试单独验证覆盖率边界，管道对齐由 stage1 数据测试验证。
        with patch.object(data, "pipeline_check", return_value={"pipeline_ok": True}):
            audit = tools.uncertainty_audit(df)
        self.assertEqual(audit["facts"]["empirical_coverage"], 0.85)
        self.assertTrue(audit["facts"]["usable_for_decision"])
        self.assertTrue(gates.run_all(GATES, {"results": [audit]})["passed"])

    def test_empty_data_is_insufficient_and_cannot_pass_coverage(self):
        df = data.load_sample().iloc[:0]
        self.assertTrue(tools.fleet_overview(df)["insufficient"])
        audit = tools.uncertainty_audit(df)
        self.assertTrue(audit["insufficient"])
        self.assertFalse(gates.run_all(GATES, {"results": [audit]})["passed"])


class TestPHMPhaseZero(unittest.TestCase):
    def test_acceptance_all_three_conditions(self):
        result = sample_agent().ask("能不能用这个预测区间安排停机检修？", use_llm=False)
        self.assertIn("56%", result["answer"])
        self.assertTrue(result["grounding"]["passed"])
        self.assertTrue(result["grounding"]["cite_ok"])
        self.assertGreater(result["grounding"]["checked"], 0)
        audit = next(r for r in result["results"] if r["tool"] == "uncertainty_audit")
        self.assertEqual(audit["facts"]["empirical_coverage"], 0.56)
        self.assertTrue(grounding.check("实测覆盖率为 56%。", [audit])["passed"])
        self.assertFalse(grounding.check("实测覆盖率为 77.77%。", [audit])["passed"])
        self.assertFalse(result["gates"]["passed"])
        self.assertEqual({r["name"] for r in result["gates"]["blocking"]},
                         {"coverage_gate", "pipeline_selfcheck"})
        self.assertIn("不能用这个预测区间安排停机检修", result["answer"])
        self.assertNotRegex(result["answer"], r"(可以|能够|可据此|建议立即).{0,12}停机")
        self.assertIn("成本与提前期假设", result["answer"])
        self.assertIn("仿真数据", result["answer"])

    def test_fleet_question_also_audits_uncertainty(self):
        result = sample_agent().ask("机队总览", use_llm=False)
        self.assertEqual([r["tool"] for r in result["results"]],
                         ["fleet_overview", "uncertainty_audit"])
        self.assertTrue(result["grounding"]["passed"])
        self.assertFalse(result["gates"]["passed"])

    def test_unknown_intent_returns_current_capabilities(self):
        result = sample_agent().ask("请帮我分析明天的股票行情", use_llm=False)
        self.assertEqual(result["status"], "clarification")
        self.assertEqual(result["results"], [])
        self.assertIn("机队 RUL 总览", result["answer"])
        self.assertIn("维护候选排期", result["answer"])

    def test_passed_coverage_still_does_not_authorize_shutdown(self):
        agent = sample_agent()
        agent.df["y_true"] = agent.df["pred"]
        with patch.object(data, "pipeline_check", return_value={"pipeline_ok": True}):
            result = agent.ask("这个区间可信么？", use_llm=False)
        self.assertTrue(result["gates"]["passed"])
        self.assertTrue(result["grounding"]["passed"])
        self.assertIn("本阶段不生成停机安排", result["answer"])

    def test_cli_exit_code_two_means_expected_block(self):
        proc = subprocess.run(
            [sys.executable, "scripts/ask.py", "能不能用这个预测区间安排停机检修？",
             "--domain", "phm", "--no-llm", "--json"],
            cwd=str(ROOT), text=True, capture_output=True,
        )
        self.assertEqual(proc.returncode, 2, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertTrue(result["grounding"]["passed"])
        self.assertFalse(result["gates"]["passed"])
        self.assertEqual(result["gates"]["blocking"][0]["name"], "coverage_gate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
