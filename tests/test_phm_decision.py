#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent numeric checks for PHM cost and candidate scheduling rules."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from insight.domains.phm.decision import cost_threshold, maintenance_plan


def fleet(pred, sigma=None, units=None):
    values = {"unit": units if units is not None else range(1, len(pred) + 1),
              "pred": pred}
    if sigma is not None:
        values["sigma"] = sigma
    return pd.DataFrame(values)


class TestCostThreshold(unittest.TestCase):
    def test_gaussian_cost_matches_independent_hand_calculation(self):
        # At H=30, the two standard-normal probabilities are known independently.
        lower_tail = 0.15865525393145707
        df = fleet([20., 40.], [10., 10.])
        result = cost_threshold(df, c_fn=10, c_fp=1)
        self.assertEqual(result["threshold_rul"], 40.)
        self.assertEqual(result["n_alarms"], 2)
        self.assertAlmostEqual(result["expected_cost"], 2.)
        self.assertAlmostEqual(result["default_cost"], 1 + 10 * lower_tail)
        self.assertAlmostEqual(result["compare_to_default"], 10 * lower_tail - 1)
        by_threshold = {row["threshold_rul"]: row for row in result["threshold_curve"]}
        self.assertAlmostEqual(by_threshold[0.]["expected_cost"], 10.)
        self.assertAlmostEqual(by_threshold[20.]["expected_cost"], 1 + 10 * lower_tail)
        self.assertEqual(by_threshold[41.]["n_alarms"], 2)

    def test_fixed_horizon_avoids_trivial_zero_threshold(self):
        result = cost_threshold(fleet([10., 30., 50.], [0., 0., 0.]))
        self.assertEqual(result["threshold_rul"], 10.)
        self.assertEqual(result["expected_cost"], 1.)
        self.assertEqual(result["default_cost"], 2.)
        self.assertEqual(result["compare_to_default"], 1.)
        self.assertEqual(result["risk_horizon_cycles"], 30.)
        self.assertIn("fleet total", result["cost_unit"])

    def test_zero_sigma_uses_strict_risk_event_and_inclusive_alarm(self):
        result = cost_threshold(fleet([0., 29., 30.], [0., 0., 0.]), c_fn=10)
        curve = {row["threshold_rul"]: row for row in result["threshold_curve"]}
        self.assertEqual(curve[0.]["n_alarms"], 1)
        self.assertEqual(curve[0.]["expected_cost"], 11.)
        self.assertEqual(result["threshold_rul"], 29.)
        self.assertEqual(result["expected_cost"], 2.)
        self.assertEqual(result["default_cost"], 3.)

    def test_horizon_is_independent_of_default_alarm_line(self):
        result = cost_threshold(fleet([10., 30.], [0., 0.]),
                                risk_horizon_cycles=5, default_threshold=50)
        self.assertEqual(result["threshold_rul"], 0.)
        self.assertEqual(result["expected_cost"], 0.)
        self.assertEqual(result["default_cost"], 2.)
        self.assertEqual(result["risk_horizon_cycles"], 5.)

    def test_exact_tie_chooses_lowest_threshold(self):
        result = cost_threshold(fleet([10., 20.], [0., 0.]), c_fn=1, c_fp=1)
        self.assertEqual(result["threshold_rul"], 0.)
        self.assertEqual(result["expected_cost"], 2.)
        result = cost_threshold(fleet([10.], [1.]), c_fn=0)
        self.assertEqual(result["threshold_rul"], 0.)

    def test_ratio_curve_is_sorted_and_nondecreasing_for_heterogeneous_sigmas(self):
        result = cost_threshold(fleet([3., 17., 29., 31., 50., 75.],
                                      [0., 15., 0., 3., 40., 1.]),
                                c_fn=7, c_fp=2)
        ratios = [row["cost_ratio"] for row in result["ratio_curve"]]
        thresholds = [row["threshold_rul"] for row in result["ratio_curve"]]
        self.assertEqual(ratios, sorted(set(ratios)))
        self.assertIn(3.5, ratios)
        self.assertEqual(thresholds, sorted(thresholds))
        self.assertEqual(result["threshold_violations"], [])
        same_ratio = cost_threshold(fleet([3., 17., 29., 31., 50., 75.],
                                          [0., 15., 0., 3., 40., 1.]),
                                    c_fn=70, c_fp=20)
        self.assertEqual(result["threshold_rul"], same_ratio["threshold_rul"])
        self.assertAlmostEqual(same_ratio["expected_cost"], result["expected_cost"] * 10)

    def test_extra_thresholds_augment_complete_candidate_grid(self):
        result = cost_threshold(fleet([12., 22.], [0., 0.]), thresholds=[8, 8, 14])
        grid = [row["threshold_rul"] for row in result["threshold_curve"]]
        self.assertEqual(grid, [0., 8., 12., 14., 22., 23., 30.])

    def test_negative_predictions_are_preserved_for_gate(self):
        result = cost_threshold(fleet([-2., 10.], [0., 0.], units=[3, 8]))
        self.assertEqual(result["bad_units"], [3])
        self.assertEqual(result["threshold_curve"][0]["n_alarms"], 1)
        self.assertTrue(all(row["threshold_rul"] >= 0 for row in result["threshold_curve"]))

    def test_empty_fleet_has_explicit_zero_cost_result(self):
        result = cost_threshold(pd.DataFrame())
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["n_alarms"], 0)
        self.assertEqual(result["expected_cost"], 0.)
        self.assertEqual(result["default_cost"], 0.)
        self.assertEqual(result["threshold_violations"], [])
        self.assertEqual(result["bad_units"], [])

    def test_extreme_finite_predictions_and_sigmas_do_not_overflow_probability(self):
        result = cost_threshold(fleet([-1e308, 1e308], [1e-308, 1e-308]))
        self.assertTrue(math.isfinite(result["expected_cost"]))
        self.assertEqual(result["n_alarms"], 1)
        # An overflowing raw difference can still have a finite z score.
        result = cost_threshold(fleet([-1e308, 1e308], [1e308, 1e308]),
                                risk_horizon_cycles=1e308)
        self.assertTrue(math.isfinite(result["expected_cost"]))
        self.assertEqual(result["threshold_curve"][0]["expected_cost"], 6.)

    def test_invalid_cost_configuration_is_rejected(self):
        cases = [{"c_fn": -1}, {"c_fn": np.nan}, {"c_fn": True},
                 {"c_fp": 0}, {"c_fp": -1}, {"c_fp": np.inf},
                 {"risk_horizon_cycles": -1}, {"risk_horizon_cycles": np.nan},
                 {"default_threshold": -1}, {"default_threshold": np.inf},
                 {"thresholds": [np.nan]}, {"thresholds": [-1]},
                 {"thresholds": 10}, {"thresholds": "20"},
                 {"c_fn": 1e308, "c_fp": 1e-308}]
        for params in cases:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    cost_threshold(fleet([10.], [1.]), **params)

    def test_invalid_prediction_inputs_are_rejected(self):
        frames = [fleet([np.nan], [1.]), fleet([np.inf], [1.]),
                  fleet([10.], [-1.]), fleet([10.], [np.nan]),
                  fleet([10.], [np.inf]), fleet([10.]),
                  fleet([10., 20.], [1., 1.], units=[1, 1])]
        for df in frames:
            with self.subTest(values=df.to_dict()):
                with self.assertRaises(ValueError):
                    cost_threshold(df)


class TestMaintenancePlan(unittest.TestCase):
    def test_missed_is_computed_before_clipping_and_equality_is_on_time(self):
        result = maintenance_plan(fleet([13., 14., 15.]), lead_time_days=14,
                                  window_days=15)
        self.assertEqual([row["raw_order_day"] for row in result["orders"]], [-1., 0., 1.])
        self.assertEqual([row["order_day"] for row in result["orders"]], [0., 0., 1.])
        self.assertEqual([row["missed"] for row in result["orders"]], [True, False, False])
        self.assertEqual(result["n_missed"], 1)
        self.assertEqual(result["missed_units"], [1])
        self.assertEqual(result["latest_order_day"], 1.)

    def test_sorting_window_and_positive_cycles_conversion(self):
        result = maintenance_plan(fleet([40., 28., 10., 30.], units=[9, 2, 7, 5]),
                                  lead_time_days=3, window_days=14, cycles_per_day=2)
        self.assertEqual(result["n_orders"], 2)
        self.assertEqual([row["unit"] for row in result["orders"]], [7, 2])
        self.assertEqual([row["service_day"] for row in result["orders"]], [5., 14.])
        self.assertEqual([row["raw_order_day"] for row in result["orders"]], [2., 11.])
        self.assertEqual(result["earliest_unit"], 7)
        self.assertEqual(result["latest_order_day"], 11.)
        self.assertEqual(result["n_missed"], 0)
        self.assertEqual(result["assumptions"]["cycles_per_day"], 2.)

    def test_negative_predictions_remain_visible_as_missed_and_bad_units(self):
        result = maintenance_plan(fleet([8., -2., 0.], units=[4, 9, 2]))
        self.assertEqual([row["unit"] for row in result["orders"]], [9, 2, 4])
        self.assertEqual(result["bad_units"], [9])
        self.assertEqual(result["n_missed"], 3)
        self.assertEqual(result["orders"][0]["service_day"], -2.)
        self.assertEqual(result["orders"][0]["order_day"], 0.)

    def test_equal_rul_sorts_by_unit_and_does_not_depend_on_input_index(self):
        result = maintenance_plan(fleet([10., 10.], units=[8, 3]).set_index("unit", drop=False))
        self.assertEqual([row["unit"] for row in result["orders"]], [3, 8])

    def test_zero_lead_time_and_window_boundary(self):
        result = maintenance_plan(fleet([0., 1.]), lead_time_days=0, window_days=0)
        self.assertEqual(result["n_orders"], 1)
        self.assertEqual(result["n_missed"], 0)
        self.assertEqual(result["orders"][0]["raw_order_day"], 0.)

    def test_empty_fleet_and_no_candidate_window_have_explicit_empty_results(self):
        for df in (pd.DataFrame(), fleet([20., 30.])):
            with self.subTest(empty=df.empty):
                result = maintenance_plan(df)
                self.assertEqual(result["orders"], [])
                self.assertEqual(result["n_orders"], 0)
                self.assertEqual(result["n_missed"], 0)
                self.assertEqual(result["earliest_unit"], 0)
                self.assertEqual(result["latest_order_day"], 0.)

    def test_invalid_days_and_rate_are_rejected_even_for_empty_fleet(self):
        for key in ("lead_time_days", "window_days", "cycles_per_day"):
            values = [-1., np.nan, np.inf, True]
            if key == "cycles_per_day":
                values.append(0.)
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        maintenance_plan(pd.DataFrame(), **{key: value})

    def test_invalid_unit_prediction_and_conversion_overflow_are_rejected(self):
        frames = [fleet([np.nan]), fleet([np.inf]), fleet([1.], units=[1.5]),
                  fleet([1.], units=[0]), fleet([1., 2.], units=[3, 3]),
                  pd.DataFrame({"pred": [1.]}), pd.DataFrame({"unit": [1]})]
        for df in frames:
            with self.subTest(values=df.to_dict()):
                with self.assertRaises(ValueError):
                    maintenance_plan(df)
        with self.assertRaises(ValueError):
            maintenance_plan(fleet([1e308]), cycles_per_day=1e-308)


if __name__ == "__main__":
    unittest.main(verbosity=2)
