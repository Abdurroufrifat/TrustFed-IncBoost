import unittest
import pandas as pd

from trustfed_incboost.twin_study import validate_temporal_stream, episode_delays


class StudyValidationTests(unittest.TestCase):
    def test_random_split_is_not_presented_as_temporal_study(self):
        frame = pd.DataFrame({"device_id": ["a", "a"],
                              "timestamp": ["2026-01-01", "2026-01-02"],
                              "pulse": [70., 71.]})
        with self.assertRaisesRegex(ValueError, "temporal"):
            validate_temporal_stream(frame, {"split": {"method": "fingerprint_disjoint"}},
                                     "device_id", "timestamp", ["pulse"])

    def test_duplicate_device_time_is_rejected(self):
        frame = pd.DataFrame({"device_id": ["a", "a"],
                              "timestamp": ["2026-01-01", "2026-01-01"],
                              "pulse": [70., 71.]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_temporal_stream(frame, {"split": {"method": "temporal",
                                                     "time_column": "timestamp"}},
                                     "device_id", "timestamp", ["pulse"])

    def test_episode_delay_marks_missed_attack(self):
        frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=7, freq="min"),
                              "y_true": [0, 1, 1, 0, 1, 1, 0],
                              "alarm": [0, 0, 1, 0, 0, 0, 0]})
        result = episode_delays(frame, "alarm")
        self.assertEqual(result["attack_episodes"], 2)
        self.assertEqual(result["missed_episodes"], 1)
        self.assertEqual(result["detected_delay_minutes"], [1.])


if __name__ == "__main__":
    unittest.main()
