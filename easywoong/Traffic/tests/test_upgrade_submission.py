import tempfile
from pathlib import Path
import unittest
import polars as pl

from src.io.paths import PANELS
from src.submit.multitask import compare_frozen_tasks
from src.task1.upgrade_ensemble import align, blend, score


class UpgradeSubmissionTest(unittest.TestCase):
    def components(self):
        data = []
        for panel in PANELS:
            for regime in ("R1", "R2", "R3"):
                data.append(dict(panel=panel, timestamp="2031-01-01T00:00:00", station_id="s", link_id="l",
                                 mask_regime=regime, speed_kmh=50., flow_vph=1600., truth_speed=60.,
                                 truth_flow_lane=1000., lanes=2.))
        a = pl.DataFrame(data)
        b = a.with_columns(pl.lit(70.).alias("speed_kmh"), pl.lit(2400.).alias("flow_vph")).reverse()
        return a, b

    def test_ensemble_joins_coordinates_and_scores_per_lane(self):
        a, b = self.components()
        result = blend(align(a, b, True), 0.5, True)
        self.assertEqual(score(result)["s_state"], 1.)
        self.assertTrue(result["speed_kmh"].eq(60).all())
        self.assertTrue(result["flow_vph"].eq(2000).all())

    def test_ensemble_rejects_missing_duplicate_or_different_truth(self):
        a, b = self.components()
        for invalid in (b.head(-1), pl.concat([b, b.head(1)]), b.with_columns(pl.lit(61.).alias("truth_speed"))):
            with self.assertRaises(ValueError):
                align(a, invalid, True)

    def test_controlled_comparison_rejects_a_queue_change(self):
        original = pl.DataFrame({"submission_id": ["a", "b", "c"], "task": ["state", "queue", "odme"],
            "speed_kmh": [70., 0., 0.], "flow_vph": [1800., 0., 0.], "queue_pred": [0, 1, 0], "path_flow": [0., 0., 100.]})
        changed_state = original.with_columns(pl.when(pl.col("task") == "state").then(80.).otherwise(pl.col("speed_kmh")).alias("speed_kmh"))
        with tempfile.TemporaryDirectory() as folder:
            original.write_csv(Path(folder) / "submission.csv")
            self.assertTrue(compare_frozen_tasks(changed_state, folder)["queue_and_odme_identical"])
            changed_queue = changed_state.with_columns(pl.lit(0).cast(pl.Int64).alias("queue_pred"))
            with self.assertRaises(ValueError):
                compare_frozen_tasks(changed_queue, folder)
            with self.assertRaises(ValueError):
                compare_frozen_tasks(original, folder)


if __name__ == "__main__":
    unittest.main()
