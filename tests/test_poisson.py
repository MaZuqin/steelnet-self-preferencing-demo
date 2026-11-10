import unittest

import numpy as np

from demand.poisson import PATH_LAMBDAS, generate


class PoissonDemandTests(unittest.TestCase):
    def test_path_and_time_prefixes_are_reproducible(self):
        full = generate(n_paths=20, horizon=366, seed=2026)
        small = generate(n_paths=3, horizon=30, seed=2026)
        np.testing.assert_array_equal(full[:3, :30], small)
        np.testing.assert_array_equal(full, generate(n_paths=20, horizon=366, seed=2026))
        self.assertFalse(np.array_equal(full, generate(n_paths=20, horizon=366, seed=2027)))
        self.assertEqual(full.dtype, np.float32)
        self.assertTrue((full >= 0).all())
        np.testing.assert_array_equal(full, np.floor(full))

    def test_fixed_rate_grid_and_poisson_moments(self):
        np.testing.assert_array_equal(PATH_LAMBDAS, np.linspace(92.0, 132.0, 20))
        # Large fixed-seed samples give a stable distribution check without
        # tying the test to implementation-specific individual random values.
        sample = generate(n_paths=20, horizon=50000, seed=58).astype(np.float64)
        np.testing.assert_allclose(sample.mean(axis=1), PATH_LAMBDAS, atol=0.25, rtol=0)
        np.testing.assert_allclose(sample.var(axis=1, ddof=1), PATH_LAMBDAS, rtol=0.025, atol=0)

    def test_invalid_bank_dimensions_are_rejected(self):
        for n_paths, horizon in ((0, 10), (21, 10), (1, 0)):
            with self.assertRaises(ValueError):
                generate(n_paths=n_paths, horizon=horizon)


if __name__ == "__main__":
    unittest.main()
