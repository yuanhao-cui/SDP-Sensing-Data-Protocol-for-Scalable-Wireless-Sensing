"""TDD tests for the end-to-end analysis pipeline (run_analysis).

Uses synthetic .npz feature files in the paper's layout so the full
pipeline is exercised without checkpoints or datasets:

    features_seed{seed}.npz  ->  X_test, y_test, X_train, y_train
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import run_analysis  # noqa: E402


def make_seed_features(base, rng, n_classes=3, n_train=100,
                       noise_scale=0.01):
    """Features that barely differ across seeds -> high CKA, tiny MMD2."""
    n_test = base.shape[0]
    X_test = base + noise_scale * rng.normal(size=base.shape)
    y_test = np.arange(n_test) % n_classes
    X_train = base[:n_train // 2].repeat(2, axis=0)[:n_train] + \
        noise_scale * rng.normal(size=(n_train, base.shape[1]))
    y_train = np.arange(n_train) % n_classes
    return X_test, y_test, X_train, y_train


class TestRunAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.features_dir = os.path.join(self.tmp.name, "features")
        os.makedirs(self.features_dir)
        rng = np.random.default_rng(0)
        base = rng.normal(size=(60, 16))
        self.seeds = [543, 785, 530, 558, 805]
        for s in self.seeds:
            X_test, y_test, X_train, y_train = make_seed_features(base, rng)
            np.savez(os.path.join(self.features_dir, f"features_seed{s}.npz"),
                     X_test=X_test, y_test=y_test,
                     X_train=X_train, y_train=y_train)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pipeline_produces_table3_style_row(self):
        out_dir = os.path.join(self.tmp.name, "results")
        row = run_analysis.run(self.features_dir, out_dir,
                               n_per_class=10, seeds=self.seeds)
        # CKA across seeds of near-identical features must be high
        self.assertGreater(row["cka_mean"], 0.9)
        self.assertGreaterEqual(row["cka_std"], 0.0)
        # class-balanced train/test MMD2 on same distribution must be tiny
        self.assertLess(row["mmd2_mean"], 0.05)
        self.assertGreaterEqual(row["mmd2_std"], 0.0)

        # results written where requested, in both json and csv
        with open(os.path.join(out_dir, "diagnostics.json")) as f:
            saved = json.load(f)
        self.assertAlmostEqual(saved["cka_mean"], row["cka_mean"])
        self.assertTrue(os.path.exists(os.path.join(out_dir, "diagnostics.csv")))

    def test_missing_seed_file_raises(self):
        os.remove(os.path.join(self.features_dir, "features_seed805.npz"))
        with self.assertRaises(FileNotFoundError):
            run_analysis.run(self.features_dir,
                             os.path.join(self.tmp.name, "r2"),
                             seeds=self.seeds)


if __name__ == "__main__":
    unittest.main()
