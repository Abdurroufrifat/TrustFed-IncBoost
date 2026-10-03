import unittest
import numpy as np

from trustfed_incboost.data.site_partition import site_group_partition


class SitePartitionTests(unittest.TestCase):
    def test_entire_site_stays_on_one_client(self):
        sites = np.array(["h1"]*4 + ["h2"]*4 + ["h3"]*4)
        labels = np.array([0, 0, 0, 1]*3)
        partitions, mapping = site_group_partition(sites, labels, n_clients=3,
                                                     min_samples_per_client=4)
        self.assertEqual(set(mapping.values()), {"h1", "h2", "h3"})
        for site in mapping.values():
            indices = partitions[next(key for key, value in mapping.items() if value == site)]
            self.assertEqual(set(sites[indices]), {site})

    def test_site_count_must_match_client_count(self):
        with self.assertRaisesRegex(ValueError, "site count"):
            site_group_partition(["h1", "h2"], [0, 1], n_clients=3)


if __name__ == "__main__":
    unittest.main()
