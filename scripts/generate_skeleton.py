"""Milestone 3 driver: generate symmetric base skeletons and preview them.

    # one skeleton
    python scripts/generate_skeleton.py --seed 7

    # a grid of N skeletons into outputs/skeletons/ + a contact sheet
    python scripts/generate_skeleton.py --count 12

    # broaden the design space ("controlled weirdness", spec section 8)
    python scripts/generate_skeleton.py --count 12 --weirdness 0.5

    # use learned priors from a Milestone 2 feature dump (JSON list of feature dicts)
    python scripts/generate_skeleton.py --count 12 --features dataset/_features.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sc2mapgen.generate.preview import render
from sc2mapgen.generate.priors import DefaultPriors, LearnedPriors
from sc2mapgen.generate.skeleton import SkeletonGenerator, validate

OUT = Path("outputs/skeletons")


def load_priors(features: str | None, weirdness: float):
    if features:
        data = json.loads(Path(features).read_text())
        feats = data if isinstance(data, list) else data.get("features", [])
        print(f"[gen] learned priors from {len(feats)} maps in {features}")
        return LearnedPriors.from_features(feats, weirdness=weirdness)
    return DefaultPriors(weirdness=weirdness)


def contact_sheet(pngs: list[Path], out: Path) -> None:
    if not pngs:
        return
    cols = min(4, len(pngs))
    rows = (len(pngs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.6))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, pngs):
        ax.imshow(mpimg.imread(p))
    fig.tight_layout()
    fig.savefig(out, dpi=95)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=1, help="generate N skeletons (seeds seed..seed+N-1)")
    ap.add_argument("--weirdness", type=float, default=0.0)
    ap.add_argument("--features", default=None, help="JSON list of Milestone 2 feature dicts")
    args = ap.parse_args()

    priors = load_priors(args.features, args.weirdness)
    gen = SkeletonGenerator(priors=priors)
    OUT.mkdir(parents=True, exist_ok=True)

    pngs, n_valid = [], 0
    for i in range(args.count):
        seed = args.seed + i
        skel = gen.generate(seed)
        ok, issues = validate(skel)
        n_valid += ok
        skel.save(OUT / f"skeleton_{seed}.json")
        pngs.append(render(skel, OUT / f"skeleton_{seed}.png"))
        flag = "ok" if ok else f"INVALID {issues}"
        print(f"[gen] seed={seed}: {len(skel.bases)} bases, {len(skel.edges)} edges -> {flag}")

    if args.count > 1:
        contact_sheet(pngs, OUT / "_contact_sheet.png")
        print(f"[gen] {n_valid}/{args.count} valid; contact sheet: {OUT/'_contact_sheet.png'}")
    else:
        print(f"[gen] artifacts: {OUT}/skeleton_{args.seed}.json / .png")


if __name__ == "__main__":
    main()
