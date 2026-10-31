"""Behavior checks for input leakage, daily cache fidelity and submission joins."""
from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from src.task1.staeformer_data import Day, load_day, model_inputs, panel_infos, synthetic_target_mask
from src.task2.catboost_queue import features
from src.task4.regularized_odme import solve
from src.submit.multitask import align_predictions, validate_frame


class PreprocessingTests(unittest.TestCase):
    def test_hidden_values_cannot_change_interpolation_or_inputs(self):
        rng = np.random.default_rng(42)
        speed = rng.uniform(10, 110, (288, 5)).astype(np.float32)
        flow = rng.uniform(10, 2000, (288, 5)).astype(np.float32)
        eligible = np.ones(speed.shape, bool)
        day = Day("test", "2030-06-01", speed, flow, eligible, ())
        target = synthetic_target_mask(day, .5, rng)
        original_x, original_b = model_inputs(day, ~target)
        changed = Day("test", day.day, speed.copy(), flow.copy(), eligible, ())
        changed.speed[target], changed.flow_lane[target] = 999999, -999999
        altered_x, altered_b = model_inputs(changed, ~target)
        np.testing.assert_array_equal(original_x, altered_x)
        np.testing.assert_array_equal(original_b, altered_b)
        self.assertTrue((original_x[..., :2][target] == 0).all())
        self.assertTrue((original_x[..., 2][target] == 0).all())

    def test_cache_matches_all_ten_original_panels(self):
        cache = Path("cache/preprocessed_smoke").resolve()
        if not (cache / "SUCCESS.json").exists():
            self.skipTest("Run preprocessing smoke first")
        infos = panel_infos()
        for panel, info in infos.items():
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("TFB_DAILY_CACHE", None)
                original = load_day(panel, "train", "2030-06-01", info, masked=False)
            with patch.dict(os.environ, {"TFB_DAILY_CACHE": str(cache)}):
                cached = load_day(panel, "train", "2030-06-01", info, masked=False)
            np.testing.assert_array_equal(original.speed, cached.speed)
            np.testing.assert_array_equal(original.flow_lane, cached.flow_lane)
            np.testing.assert_array_equal(original.eligible, cached.eligible)
            self.assertEqual(original.timestamps, cached.timestamps)

    def test_queue_lags_horizons_and_missingness(self):
        n = 3
        speed = np.repeat(np.arange(1, 13, dtype=np.float32)[:, None], n, axis=1)
        speed[:, 1] = np.nan
        history = {"speed": speed, "flow_lane": speed * 10, "occupancy": speed / 100,
                   "pct_observed": np.where(np.isfinite(speed), 100, 0)}
        net = {"panel_id": 0, "node_id": np.arange(n), "free_speed": np.ones(n),
               "capacity_lane": np.full(n, 100), "lanes": np.full(n, 3),
               "length": np.ones(n), "forward": True}
        x, names = features(history, net, 12, 0)
        np.testing.assert_array_equal(x[:, names.index("horizon_steps")], np.repeat(np.arange(1, 7), n))
        self.assertEqual(x[0, names.index("v_lag1")], 12)
        self.assertEqual(x[0, names.index("v_lag12")], 1)
        self.assertEqual(x[0, names.index("v_slope")], 1)
        self.assertTrue(np.isnan(x[1, names.index("v_last")]))
        self.assertEqual(x[1, names.index("observed_fraction")], 0)
        self.assertEqual(x[0, names.index("last_age")], 1)
        self.assertEqual(x[1, names.index("last_age")], 13)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.keys = pl.DataFrame({"submission_id": [10, 20], "split": ["validation", "private"],
                                  "panel": ["P", "P"], "timestamp": ["t", "t"],
                                  "station_id": ["S", "S"], "link_id": ["L", "L"], "mask_regime": ["R1", "R1"]})
        self.prediction = self.keys.drop("submission_id").with_columns(
            pl.Series("speed_kmh", [70., 80.]), pl.Series("flow_vph", [1000., 2000.]))

    def test_prediction_alignment_includes_split(self):
        result = align_predictions(self.keys, self.prediction.reverse(), "state").sort("submission_id")
        self.assertEqual(result["speed_kmh"].to_list(), [70., 80.])
        validate_frame(result, self.keys.select("submission_id").with_columns(pl.lit("state").alias("task")))

    def test_missing_and_duplicate_predictions_raise(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            align_predictions(self.keys, self.prediction.head(1), "state")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            align_predictions(self.keys, pl.concat([self.prediction, self.prediction.head(1)]), "state")

    def test_null_and_nonfinite_predictions_raise(self):
        for value in (None, float("nan"), float("inf"), -1.):
            with self.assertRaises(ValueError):
                align_predictions(self.keys, self.prediction.with_columns(pl.lit(value).alias("speed_kmh")), "state")

    def test_task_mapping_cannot_be_changed(self):
        result = align_predictions(self.keys, self.prediction, "state").sort("submission_id")
        template = result.select("submission_id", "task")
        with self.assertRaisesRegex(ValueError, "task mapping"):
            validate_frame(result.with_columns(pl.lit("queue").alias("task")), template)

    def test_regularized_nnls_obeys_active_nonnegative_bound(self):
        A = np.array([[1., 1.], [0., 1.]])
        prediction, diagnostic = solve(A, np.array([1., 10.]), np.zeros(2), lam=.05)
        np.testing.assert_allclose(prediction, [0., 11. / 2.05], atol=1e-5)
        self.assertTrue(diagnostic["success"])
        self.assertLess(diagnostic["objective_fit"], diagnostic["objective_prior"])


if __name__ == "__main__":
    unittest.main()
