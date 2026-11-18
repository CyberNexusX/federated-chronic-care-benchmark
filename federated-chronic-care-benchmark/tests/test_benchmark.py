import unittest

import numpy as np

from benchmark import aggregate, federate, run, update


class BenchmarkTests(unittest.TestCase):
    def test_aggregation_uses_patient_counts(self):
        actual = aggregate([(np.array([0.0, 2.0]), 3), (np.array([4.0, 6.0]), 1)])
        np.testing.assert_allclose(actual, [1.0, 3.0])

    def test_one_local_step_matches_one_pooled_step(self):
        first = (np.array([[1.0, 0.0], [1.0, 2.0]]), np.array([0, 1]))
        second = (np.array([[1.0, 3.0]]), np.array([1]))
        pooled = (np.vstack([first[0], second[0]]), np.concatenate([first[1], second[1]]))
        initial = np.array([0.1, -0.2])
        actual = federate({"first": first, "second": second}, initial, rounds=1, local_steps=1)
        np.testing.assert_allclose(actual, update(initial, pooled, 1))

    def test_held_out_clinic_has_no_local_model(self):
        report = run(42)
        self.assertEqual(set(report["training_counts"]), {"north", "south"})
        coastal = report["evaluation"]["coastal"]
        self.assertEqual(coastal["split"], "held_out_site")
        self.assertEqual(set(coastal["models"]), {"pooled", "fedavg"})
        for site in ("north", "south"):
            self.assertIn("site_local", report["evaluation"][site]["models"])


if __name__ == "__main__":
    unittest.main()
