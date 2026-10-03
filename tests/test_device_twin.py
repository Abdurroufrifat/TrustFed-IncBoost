import numpy as np
import pandas as pd
import unittest

from trustfed_incboost.device_twin import DeviceStateTwin, fuse_probabilities


def training_rows():
    rows = []
    for device, center in (("ward_a", 70.), ("ward_b", 90.)):
        for t in range(10):
            rows.append({"device_id": device, "timestamp": f"2026-01-01T00:{t:02d}:00Z",
                         "pulse": center + (t % 3) - 1, "label": 0})
    rows.append({"device_id": "ward_a", "timestamp": "2026-01-01T00:10:00Z",
                 "pulse": 900., "label": 1})
    return pd.DataFrame(rows)


def make_twin():
    return DeviceStateTwin.fit(
        training_rows(), device_column="device_id", time_column="timestamp",
        feature_columns=["pulse"], label_column="label", min_benign=5,
        detector_gate=.5, max_normal_deviation=4., alpha=.5,
    )


class DeviceTwinTests(unittest.TestCase):
    def test_attacked_reading_does_not_shift_state_or_another_device(self):
        twin = make_twin()
        initial_a = twin.states["ward_a"].copy()
        initial_b = twin.states["ward_b"].copy()
        attack = twin.observe("ward_a", "2026-01-01T00:11:00Z", [900.], .99)
        self.assertFalse(attack["state_updated"])
        np.testing.assert_array_equal(twin.states["ward_a"], initial_a)
        np.testing.assert_array_equal(twin.states["ward_b"], initial_b)
        safe = twin.observe("ward_b", "2026-01-01T00:11:00Z", [92.], .01)
        self.assertTrue(safe["state_updated"])
        self.assertFalse(np.array_equal(twin.states["ward_b"], initial_b))
        np.testing.assert_array_equal(twin.states["ward_a"], initial_a)

    def test_replay_rejects_duplicate_time_and_unknown_device(self):
        twin = make_twin()
        twin.observe("ward_a", "2026-01-01T00:11:00Z", [71.], .01)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            twin.observe("ward_a", "2026-01-01T00:11:00Z", [71.], .01)
        with self.assertRaisesRegex(ValueError, "unknown device"):
            twin.observe("new_patient", "2026-01-01T00:12:00Z", [71.], .01)

    def test_hybrid_keeps_detector_score_when_twin_is_quiet(self):
        original = np.array([.01, .2, .7])
        fused = fuse_probabilities(original, np.array([0., 1., .5]), twin_weight=.25)
        self.assertAlmostEqual(fused[0], original[0])
        self.assertTrue(np.all(fused >= original))
        self.assertTrue(np.all((fused >= 0) & (fused <= 1)))
        with self.assertRaisesRegex(ValueError, "same shape"):
            fuse_probabilities(original, np.array([.1]), twin_weight=.25)


if __name__ == "__main__":
    unittest.main()
