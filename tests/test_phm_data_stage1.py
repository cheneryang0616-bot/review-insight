#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHM Stage 1 的原始资产对齐、训练统计与正式模型比较契约。"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from insight.domains.phm import config, data


def write_assets(root, dataset="FD001"):
    root = Path(root)
    directory = root / "data"
    directory.mkdir(exist_ok=True)
    test = []
    for unit, end in enumerate((3, 4, 5), 1):
        for cycle in range(1, end + 1):
            test.append([unit, cycle, 0, 0, 1] + [float(unit + cycle + k) for k in range(21)])
    np.savetxt(directory / f"test_{dataset}.txt", test)
    train = []
    for unit in (1, 2):
        for cycle in range(1, 41):
            sensors = [7., float(unit + cycle), float(100 - unit - cycle)]
            sensors += [float(unit * 3 + cycle + k) for k in range(18)]
            train.append([unit, cycle, 0, 0, 1] + sensors)
    np.savetxt(directory / f"train_{dataset}.txt", train)
    np.savetxt(directory / f"RUL_{dataset}.txt", [10., 130., 30.])
    run = root / "artifacts" / data.run_name(dataset)
    run.mkdir(parents=True)
    np.savez(run / "predictions.npz", u_test=[1, 2, 3], y_test=[10., 125., 30.],
             pred_ensemble=[12., 120., 32.], sigma=[1., 2., 3.], end_cycle=[3, 4, 5],
             pred_members=[[11., 119., 31.], [13., 121., 33.]])
    return root


def write_reports(root, dataset="FD001"):
    root, summary = Path(root), {}
    for index, model in enumerate(data.COMPARABLE_MODELS):
        metrics = {"n": 3, "rmse": float(index + 1), "nasa_score": float(10 - index)}
        summary[model] = metrics
        run = (f"{dataset}_cnn_bilstm_attn_w30_mrul125_gbdt" if model == "gbdt"
               else data.run_name(dataset, model))
        path = root / "artifacts" / run / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "run_name": run, "ensemble": metrics,
            "config": {"data": {"dataset": dataset, "window": 30,
                                "max_rul": 125, "rul_mode": "clip"}},
        }), encoding="utf-8")
    path = root / "artifacts" / f"summary_{dataset}.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


class TestPHMStageOneData(unittest.TestCase):
    def test_dataset_selection_and_explicit_root_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_assets(directory, "FD002")
            with patch.dict(os.environ, {"RI_PHM_ROOT": str(root / "missing")}):
                frame = data.load(root=root, dataset="fd002")
                self.assertEqual(frame.attrs["dataset"], "FD002")
                self.assertEqual(frame.attrs["run"], data.run_name("FD002"))
                self.assertEqual(data.asset_context(frame)["root"], root.resolve())
                with self.assertRaises(FileNotFoundError):
                    data.load(dataset="FD002")
        with self.assertRaises(ValueError):
            data.load(dataset="FD999")

    def test_sample_cannot_supply_other_dataset_or_raw_asset_context(self):
        frame = data.load_sample()
        with self.assertRaises(FileNotFoundError):
            data.asset_context(frame)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(config, "DEFAULT_ROOT", Path(directory) / "missing"):
                    with self.assertRaises(FileNotFoundError):
                        data.load(dataset="FD002")
                    self.assertEqual(data.load().attrs["source"], "sample")

    def test_scope_description_uses_requested_dataset_from_same_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_assets(directory)
            write_assets(root, "FD002")
            frame = data.load(root)
            description = data.scope_desc(frame, dataset="FD002")
            self.assertIn("FD002", description)
            self.assertIn(data.run_name("FD002"), description)
            self.assertNotIn("FD001", description)
            self.assertEqual(frame.attrs["dataset"], "FD001")

    def test_sample_scope_description_reports_other_dataset_unavailable(self):
        frame = data.load_sample()
        with patch.object(data, "load") as loader:
            description = data.scope_desc(frame, dataset="FD002")
            loader.assert_not_called()
        self.assertIn("FD002 当前不可用", description)
        self.assertNotIn("100 台", description)

    def test_panel_is_raw_single_unit_sorted_and_returns_independent_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_assets(directory)
            frame = data.panel(2, root=root)
            self.assertEqual(tuple(frame.columns), data.PANEL_COLUMNS)
            self.assertEqual(frame["unit"].tolist(), [2] * 4)
            self.assertEqual(frame["cycle"].tolist(), [1, 2, 3, 4])
            frame.loc[0, "sensor_1"] = -999
            self.assertEqual(data.panel(2, root=root).loc[0, "sensor_1"], 3.)
            with self.assertRaises(ValueError):
                data.panel(77, root=root)

    def test_raw_parser_rejects_wrong_width_or_duplicate_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_assets(directory)
            path = root / "data" / "test_FD001.txt"
            np.savetxt(path, [[1, 1, 1]])
            with self.assertRaises(ValueError):
                data.panel(1, root=root)
            np.savetxt(path, [[1, 1] + [0] * 24] * 2)
            with self.assertRaises(ValueError):
                data.panel(1, root=root)

    def test_sensor_statistics_use_training_population_and_training_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_assets(directory)
            stats = data.sensor_stats(root=root)
            expected = np.array([u + c for u in (1, 2) for c in range(1, 41)])
            self.assertEqual(stats.index.tolist(), list(data.SENSOR_COLUMNS))
            self.assertAlmostEqual(stats.loc["sensor_2", "mean"], expected.mean())
            self.assertAlmostEqual(stats.loc["sensor_2", "std"], expected.std(ddof=0))
            self.assertEqual(stats.loc["sensor_1", "std"], 0.)
            self.assertEqual(stats.loc["sensor_2", "direction"], 1.)
            self.assertEqual(stats.loc["sensor_3", "direction"], -1.)
            self.assertIn("代理", stats.attrs["note"])

    def test_multiple_conditions_are_disclosed_and_missing_assets_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_assets(directory, "FD004")
            stats = data.sensor_stats(dataset="FD004", root=root)
            self.assertTrue(stats.attrs["is_multi_condition"])
            self.assertIn("未按工况", stats.attrs["note"])
            with self.assertRaises(FileNotFoundError):
                data.panel(1, dataset="FD003", root=root)
            with self.assertRaises(FileNotFoundError):
                data.sensor_stats(dataset="FD003", root=root)
            with self.assertRaises(FileNotFoundError):
                data.runs(dataset="FD003", root=root)

    def test_pipeline_verifies_clip_labels_and_terminal_cycles_by_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_assets(directory)
            frame = data.load(root)
            facts = data.pipeline_check(frame.iloc[::-1].copy())
            self.assertTrue(facts["pipeline_ok"])
            self.assertTrue(facts["alignment_checked"])
            for key in ("u_count", "y_count", "rul_count", "panel_unit_count"):
                self.assertEqual(facts[key], 3)
            frame.loc[0, "y_true"], frame.loc[2, "y_true"] = 30., 10.
            facts = data.pipeline_check(frame)
            self.assertFalse(facts["pipeline_ok"])
            self.assertFalse(facts["labels_match"])
            self.assertTrue(facts["alignment_checked"])

    def test_pipeline_rejects_end_cycle_and_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_assets(directory)
            frame = data.load(root)
            frame.loc[1, "end_cycle"] += 1
            self.assertFalse(data.pipeline_check(frame)["end_cycles_match"])
            self.assertFalse(data.pipeline_check(frame)["pipeline_ok"])
            frame = data.load(root)
            np.savetxt(root / "data" / "RUL_FD001.txt", [10., 130.])
            facts = data.pipeline_check(frame)
            self.assertEqual(facts["rul_count"], 2)
            self.assertFalse(facts["pipeline_ok"])

    def test_sample_and_missing_raw_assets_do_not_claim_alignment_was_checked(self):
        sample_facts = data.pipeline_check(data.load_sample())
        self.assertFalse(sample_facts["pipeline_ok"])
        self.assertFalse(sample_facts["alignment_checked"])
        self.assertEqual(sample_facts["rul_count"], -1)
        with tempfile.TemporaryDirectory() as directory:
            root = write_assets(directory)
            frame = data.load(root)
            (root / "data" / "RUL_FD001.txt").unlink()
            self.assertFalse(data.pipeline_check(frame)["alignment_checked"])

    def test_runs_use_only_five_formal_reports_and_crosscheck_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_assets(directory)
            summary_path = write_reports(root)
            probe = root / "artifacts" / "FD001_probe_better_rmse" / "report.json"
            probe.parent.mkdir()
            probe.write_text(json.dumps({"ensemble": {"rmse": 0, "nasa_score": 0}}))
            summary = json.loads(summary_path.read_text())
            summary["blend"] = {"n": 3, "rmse": 0, "nasa_score": 0}
            summary_path.write_text(json.dumps(summary))
            runs = data.runs(root=root)
            self.assertEqual([r["model"] for r in runs], list(data.COMPARABLE_MODELS))
            self.assertTrue(all(Path(r["source"]).is_file() for r in runs))
            self.assertEqual(min(runs, key=lambda r: r["rmse"])["model"], "cnn_bilstm_attn")
            self.assertEqual(min(runs, key=lambda r: r["nasa_score"])["model"], "cnn_transformer")
            summary["gbdt"]["rmse"] += 1
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "汇总不一致"):
                data.runs(root=root)

    def test_runs_reject_different_protocol_and_missing_formal_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_assets(directory)
            write_reports(root)
            path = root / "artifacts" / data.run_name() / "report.json"
            report = json.loads(path.read_text())
            report["config"]["data"]["max_rul"] = 150
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "比较协议"):
                data.runs(root=root)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                data.runs(root=root)

    @unittest.skipUnless(
        (config.DEFAULT_ROOT / "data" / "test_FD001.txt").is_file(),
        "独立 clone 不要求外部 C-MAPSS 资产")
    def test_optional_actual_assets_align_and_reports_match_all_datasets(self):
        for dataset in data.DATASETS:
            with self.subTest(dataset=dataset):
                frame = data.load(config.DEFAULT_ROOT, dataset=dataset)
                facts = data.pipeline_check(frame)
                self.assertTrue(facts["pipeline_ok"], facts)
                self.assertEqual(len(data.runs(root=config.DEFAULT_ROOT, dataset=dataset)), 5)
                self.assertEqual(data.panel(1, dataset, config.DEFAULT_ROOT)["cycle"].max(),
                                 frame.loc[frame["unit"].eq(1), "end_cycle"].item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
