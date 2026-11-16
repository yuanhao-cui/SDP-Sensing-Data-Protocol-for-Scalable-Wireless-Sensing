"""End-to-end pipeline: per-seed feature files -> Table III style row.

Usage (after features_seed{seed}.npz files exist, e.g. produced by
feature_extraction.save_seed_features on the checkpoint machine):

    python rep_diagnostics/run_analysis.py \
        --features-dir path/to/features --out-dir rep_diagnostics/results \
        --seeds 543 785 530 558 805 --n-per-class 10

Each features_seed{s}.npz must contain X_test (features of the SAME
evaluation samples for every seed - e.g. the intersection of the
per-seed test splits), plus X_train/y_train/X_test/y_test for MMD2.
Outputs diagnostics.json / diagnostics.csv with cka_mean, cka_std,
mmd2_mean, mmd2_std - one row of Table III per (dataset, backbone).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics import cka_across_seeds, mmd2_class_balanced, summarize  # noqa: E402


def load_seed_features(features_dir, seed):
    path = os.path.join(features_dir, f"features_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing feature file: {path}")
    return np.load(path)


def run(features_dir, out_dir, n_per_class=10, seeds=None, seed_offset=0):
    """Compute one Table III row from per-seed .npz feature files."""
    if seeds is None:
        seeds = sorted(int(f[len("features_seed"):-len(".npz")])
                       for f in os.listdir(features_dir)
                       if f.startswith("features_seed") and f.endswith(".npz"))
    if len(seeds) < 2:
        raise ValueError("need features from at least two seeds")

    arrays = {s: load_seed_features(features_dir, s) for s in seeds}
    for s in seeds:
        for key in ("X_test", "X_train", "y_train", "y_test"):
            if key not in arrays[s]:
                raise KeyError(f"features_seed{s}.npz lacks {key!r}")

    n_test = arrays[seeds[0]]["X_test"].shape[0]
    for s in seeds:
        if arrays[s]["X_test"].shape[0] != n_test:
            raise ValueError(
                f"seed {s}: X_test has {arrays[s]['X_test'].shape[0]} samples, "
                f"expected {n_test} (same evaluation set for every seed)"
            )

    # CKA: latent reproducibility across independently trained seeds
    test_feats = [arrays[s]["X_test"] for s in seeds]
    cka_mean, cka_std, cka_pairs = cka_across_seeds(test_feats)

    # MMD2: class-balanced train--test discrepancy, per seed then summarized
    mmd2_per_seed = []
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(seed_offset + i)
        mmd2_per_seed.append(mmd2_class_balanced(
            arrays[s]["X_train"], arrays[s]["y_train"],
            arrays[s]["X_test"], arrays[s]["y_test"],
            n_per_class=n_per_class, rng=rng))
    mmd2_mean, mmd2_std = summarize(mmd2_per_seed)

    row = {
        "seeds": list(seeds),
        "n_eval_samples": int(n_test),
        "n_per_class": int(n_per_class),
        "cka_mean": cka_mean,
        "cka_std": cka_std,
        "mmd2_mean": mmd2_mean,
        "mmd2_std": mmd2_std,
        "cka_pairs": cka_pairs,
        "mmd2_per_seed": mmd2_per_seed,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "diagnostics.json"), "w") as f:
        json.dump(row, f, indent=2)
    with open(os.path.join(out_dir, "diagnostics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cka_mean", "cka_std",
                                               "mmd2_mean", "mmd2_std",
                                               "n_eval_samples",
                                               "n_per_class", "seeds"])
        writer.writeheader()
        writer.writerow({k: row[k] for k in writer.fieldnames})
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--n-per-class", type=int, default=10)
    args = parser.parse_args(argv)
    row = run(args.features_dir, args.out_dir,
              n_per_class=args.n_per_class, seeds=args.seeds)
    print(f"CKA : {row['cka_mean']:.4f} ± {row['cka_std']:.4f}")
    print(f"MMD2: {row['mmd2_mean']:.4f} ± {row['mmd2_std']:.4f}")
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
