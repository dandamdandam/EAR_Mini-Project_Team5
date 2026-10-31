"""Behavioral checks for hidden-value isolation and window coverage."""
import unittest
import numpy as np
from src.task1.staeformer_data import Day
from src.task1.upgrade import extended_inputs, window_starts


class UpgradeInputsTest(unittest.TestCase):
    def test_hidden_values_and_quality_cannot_affect_model_input(self):
        rng = np.random.default_rng(7)
        speed = rng.uniform(30, 100, (288, 4)).astype(np.float32)
        flow = rng.uniform(200, 1500, (288, 4)).astype(np.float32)
        visible = rng.random((288, 4)) > 0.5
        quality = np.full((288, 4), 90, dtype=np.uint8)
        a = Day('test', '2030-06-01', speed, flow, np.ones_like(visible), ())
        x, base = extended_inputs(a, visible, quality)
        speed2, flow2, quality2 = speed.copy(), flow.copy(), quality.copy()
        speed2[~visible], flow2[~visible], quality2[~visible] = 999, -1000, 0
        b = Day('test', '2030-06-01', speed2, flow2, a.eligible, ())
        x2, base2 = extended_inputs(b, visible, quality2)
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(base, base2)
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[..., 6:][visible], 0)
        np.testing.assert_array_equal(x[..., 5][~visible], 0)

    def test_whole_day_missing_node_has_capped_gaps(self):
        shape = (288, 2)
        visible = np.ones(shape, dtype=bool)
        visible[:, 1] = False
        day = Day('test', '2030-06-01', np.full(shape, 70.), np.full(shape, 1000.), visible, ())
        x, base = extended_inputs(day, visible, np.full(shape, 100))
        np.testing.assert_array_equal(x[:, 1, 6:], 1)
        self.assertTrue(np.isfinite(base).all())

    def test_overlapping_inference_covers_every_timestamp(self):
        for window, stride in [(24, 12), (48, 24)]:
            coverage = np.zeros(288, dtype=int)
            for start in window_starts(window, stride):
                self.assertGreaterEqual(start, 0)
                self.assertLessEqual(start + window, 288)
                coverage[start:start + window] += 1
            self.assertGreaterEqual(coverage.min(), 1)
            self.assertGreater(coverage.max(), 1)


if __name__ == '__main__':
    unittest.main()
