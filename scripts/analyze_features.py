"""Milestone 2 driver: compute the competitive feature vector for every ingested map,
then summarize empirical distributions across the corpus.

Operates on cached engine dumps in dataset/<map>/_raw.* (no SC2 relaunch), so it can be
re-run instantly as the extractor evolves.

    python scripts/analyze_features.py            # all maps in dataset/
    python scripts/analyze_features.py A B C       # specific maps

Outputs:
    dataset/_features.json       per-map feature vectors
    dataset/_features.csv        same, tabular
    dataset/_distributions.json  per-feature summary stats (mean/std/percentiles)
    dataset/_distributions.png   histograms of key competitive parameters
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sc2mapgen.features import extract_features
from sc2mapgen.ingest.to_mapir import build_mapir

DATASET = Path("dataset")

# Non-competitive Melee test maps (empty/flat, no expansions) - excluded from the corpus.
EXCLUDE = {"Empty128", "Flat32", "Flat48", "Flat64", "Flat96", "Flat128"}

# curated features to plot as distributions
KEY_FEATURES = [
    "width",
    "playable_ratio",
    "n_bases",
    "n_generic_bases",
    "base_density",
    "rush_distance_min",
    "main_to_natural_mean",
    "rush_minwidth_min",
    "n_ramps",
    "n_regions",
    "high_ground_ratio",
    "symmetry_score",
]


def summarize(values: list[float]) -> dict:
    a = np.array([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 3),
        "std": round(float(a.std()), 3),
        "min": round(float(a.min()), 3),
        "p25": round(float(np.percentile(a, 25)), 3),
        "median": round(float(np.median(a)), 3),
        "p75": round(float(np.percentile(a, 75)), 3),
        "max": round(float(a.max()), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", nargs="*", help="map names (default: all in dataset/)")
    args = ap.parse_args()

    if args.maps:
        dirs = [DATASET / n for n in args.maps]
    else:
        dirs = sorted(
            p.parent for p in DATASET.glob("*/_raw.npz") if p.parent.name not in EXCLUDE
        )

    rows: list[dict] = []
    for d in dirs:
        name = d.name
        try:
            mapir = build_mapir(d)
            feats = extract_features(mapir)
            rows.append(feats)
            print(f"[features] {name}: bases={feats['n_bases']} rush_min={feats['rush_distance_min']} "
                  f"ramps={feats['n_ramps']} sym={feats['rot180_symmetry']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[features] FAILED {name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not rows:
        print("no maps analyzed", file=sys.stderr)
        raise SystemExit(1)

    (DATASET / "_features.json").write_text(json.dumps(rows, indent=2))

    # csv
    cols = sorted({k for r in rows for k in r})
    cols = ["map_name"] + [c for c in cols if c != "map_name"]
    with open(DATASET / "_features.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    # distributions (numeric columns only; categorical get value counts)
    def is_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    numeric_keys = [
        k for k in cols if k != "map_name" and any(is_num(r.get(k)) for r in rows)
    ]
    categorical_keys = [
        k for k in cols if k != "map_name" and k not in numeric_keys
    ]
    dist = {k: summarize([r.get(k) for r in rows if is_num(r.get(k))]) for k in numeric_keys}
    for k in categorical_keys:
        counts: dict = {}
        for r in rows:
            counts[r.get(k)] = counts.get(r.get(k), 0) + 1
        dist[k] = {"counts": counts}
    (DATASET / "_distributions.json").write_text(json.dumps(dist, indent=2))

    # plot key distributions
    keys = [k for k in KEY_FEATURES if any(r.get(k) is not None for r in rows)]
    cols_n = 4
    rows_n = (len(keys) + cols_n - 1) // cols_n
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 3.2, rows_n * 2.6))
    axes = np.array(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, k in zip(axes, keys):
        ax.axis("on")
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        ax.hist(vals, bins=min(12, max(3, len(set(vals)))), color="#3a86ff", edgecolor="black")
        ax.set_title(k, fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"Competitive feature distributions (n={len(rows)} maps)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(DATASET / "_distributions.png", dpi=100)
    plt.close(fig)

    print(f"\n[features] analyzed {len(rows)} maps")
    print(f"[features] -> {DATASET/'_features.json'}, {DATASET/'_features.csv'}")
    print(f"[features] -> {DATASET/'_distributions.json'}, {DATASET/'_distributions.png'}")


if __name__ == "__main__":
    main()
