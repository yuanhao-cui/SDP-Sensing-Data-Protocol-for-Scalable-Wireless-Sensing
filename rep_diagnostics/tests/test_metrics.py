"""TDD tests for CKA / MMD2 representation-level diagnostics.

Pure-stdlib unittest so the suite runs anywhere numpy does
(no pytest / torch required). Run from repo root:

    python -m unittest discover -s rep_diagnostics/tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics import (  # noqa: E402
    biased_hsic,
    linear_cka,
    cka_across_seeds,
    median_heuristic_sigma,
    rbf_kernel,
    mmd2,
    class_balanced_subsample,
    summarize,
)


def make_rng(seed=0):
    return np.random.default_rng(seed)


class TestHSIC(unittest.TestCase):
    def test_hsic_symmetric(self):
        rng = make_rng(1)
        X = rng.normal(size=(40, 8))
        Y = rng.normal(size=(40, 8))
        kxy = X @ X.T
        kyy = Y @ Y.T
        self.assertAlmostEqual(biased_hsic(kxy, kyy), biased_hsic(kyy, kxy), places=12)

    def test_hsic_self_positive(self):
        rng = make_rng(2)
        X = rng.normal(size=(40, 8))
        k = X @ X.T
        self.assertGreater(biased_hsic(k, k), 0.0)


class TestLinearCKA(unittest.TestCase):
    def test_identical_representations_give_one(self):
        rng = make_rng(3)
        X = rng.normal(size=(60, 16))
        self.assertAlmostEqual(linear_cka(X, X), 1.0, places=10)

    def test_similar_representations_high(self):
        rng = make_rng(4)
        X = rng.normal(size=(60, 16))
        Y = X + 0.01 * rng.normal(size=(60, 16))
        self.assertGreater(linear_cka(X, Y), 0.99)

    def test_independent_representations_low(self):
        rng = make_rng(5)
        X = rng.normal(size=(500, 32))
        Y = rng.normal(size=(500, 32))
        self.assertLess(linear_cka(X, Y), 0.2)

    def test_requires_same_sample_count(self):
        rng = make_rng(6)
        X = rng.normal(size=(50, 8))
        Y = rng.normal(size=(40, 8))
        with self.assertRaises(ValueError):
            linear_cka(X, Y)


class TestCKAAcrossSeeds(unittest.TestCase):
    def test_all_pairs_used(self):
        # 5 seeds -> C(5,2) = 10 pairwise CKA values
        rng = make_rng(7)
        feats = [rng.normal(size=(40, 8)) for _ in range(5)]
        mean, std, pairs = cka_across_seeds(feats)
        self.assertEqual(len(pairs), 10)
        self.assertGreaterEqual(mean, 0.0)
        self.assertLessEqual(mean, 1.0)
        self.assertGreaterEqual(std, 0.0)

    def test_reproducible_models_high_cka(self):
        rng = make_rng(8)
        base = rng.normal(size=(40, 8))
        feats = [base + 0.01 * rng.normal(size=(40, 8)) for _ in range(5)]
        mean, _, _ = cka_across_seeds(feats)
        self.assertGreater(mean, 0.95)

    def test_needs_at_least_two_seeds(self):
        with self.assertRaises(ValueError):
            cka_across_seeds([np.zeros((10, 4))])

    def test_same_number_of_samples_required(self):
        rng = make_rng(9)
        feats = [rng.normal(size=(40, 8)), rng.normal(size=(41, 8))]
        with self.assertRaises(ValueError):
            cka_across_seeds(feats)


class TestRBFKernelAndMMD2(unittest.TestCase):
    def test_median_heuristic_positive_finite(self):
        rng = make_rng(10)
        X = rng.normal(size=(50, 8))
        sigma = median_heuristic_sigma(X, X)
        self.assertTrue(np.isfinite(sigma) and sigma > 0)

    def test_rbf_kernel_diagonal_is_one(self):
        rng = make_rng(11)
        X = rng.normal(size=(30, 8))
        k = rbf_kernel(X, X, sigma=1.0)
        np.testing.assert_allclose(np.diag(k), 1.0, atol=1e-12)

    def test_mmd2_identical_samples_is_zero(self):
        rng = make_rng(12)
        X = rng.normal(size=(80, 8))
        self.assertAlmostEqual(mmd2(X, X), 0.0, places=10)

    def test_mmd2_shifted_distributions_larger(self):
        rng = make_rng(13)
        X = rng.normal(size=(200, 8))
        Y_close = X + 0.1 * rng.normal(size=(200, 8))
        Y_far = X + 3.0
        self.assertGreater(mmd2(X, Y_far), mmd2(X, Y_close))

    def test_class_balanced_subsample(self):
        rng = make_rng(14)
        labels = np.repeat([0, 1, 2], [100, 20, 50])
        idx = class_balanced_subsample(labels, n_per_class=10, rng=rng)
        counts = np.bincount(labels[idx], minlength=3)
        np.testing.assert_array_equal(counts, [10, 10, 10])
        self.assertEqual(len(np.unique(idx)), 30)

    def test_class_balanced_rejects_over_request(self):
        labels = np.repeat([0, 1], [5, 100])
        with self.assertRaises(ValueError):
            class_balanced_subsample(labels, n_per_class=10, rng=make_rng(15))


class TestSummarize(unittest.TestCase):
    def test_mean_std(self):
        vals = [0.90, 0.91, 0.89, 0.90, 0.91]
        mean, std = summarize(vals)
        self.assertAlmostEqual(mean, 0.902, places=3)
        self.assertAlmostEqual(std, np.std(vals, ddof=1), places=12)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == "__main__":
    unittest.main()
