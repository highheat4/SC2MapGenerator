"""Milestone 4 driver: generate a skeleton, rasterize it to terrain, preview it.

    python scripts/rasterize_skeleton.py --seed 7
    python scripts/rasterize_skeleton.py --count 8 --weirdness 0.4
    python scripts/rasterize_skeleton.py --count 8 --features outputs/_features_test.json

Emits (per seed) in outputs/rasters/gen_<seed>/:
    terrain.npy / map.json   (a real MapIR - same type as ingested maps)
    preview.png              (rendered with the shared ingest renderer)
plus outputs/rasters/_contact_sheet.png for --count > 1.
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
import numpy as np

from sc2mapgen.generate.priors import DefaultPriors, LearnedPriors
from sc2mapgen.generate.rasterize import RasterConfig, rasterize
from sc2mapgen.generate.skeleton import SkeletonGenerator
from sc2mapgen.ingest.preview import render
from sc2mapgen.ir import BaseKind

OUT = Path("outputs/rasters")


def load_priors(features: str | None, weirdness: float):
    if features:
        data = json.loads(Path(features).read_text())
        feats = data if isinstance(data, list) else data.get("features", [])
        print(f"[m4] learned priors from {len(feats)} maps")
        return LearnedPriors.from_features(feats, weirdness=weirdness)
    return DefaultPriors(weirdness=weirdness)


def enrich_graph(mapir) -> None:
    """Best-effort: recompute region/connection graph via the ingest builders so the
    generated map is analyzed exactly like a real one (and the preview shows topology)."""
    try:
        from sc2mapgen.ingest.graph import build_connections, build_regions

        ramp_mask = np.zeros_like(mapir.walkable, dtype=bool)
        for r in mapir.ramps:
            for x, y in r.cells:
                ramp_mask[y, x] = True
        regions, region_labels = build_regions(mapir.walkable, ramp_mask, mapir.elevation)
        mapir.regions = regions
        mapir.connections = build_connections(mapir.ramps, region_labels)
    except Exception as exc:  # noqa: BLE001 - enrichment is optional
        print(f"[m4] graph enrichment skipped: {type(exc).__name__}: {exc}")


def check(mapir) -> dict:
    """Quick sanity checks (full validator is Milestone 5)."""
    from scipy import ndimage

    labels, _ = ndimage.label(mapir.walkable)
    mains = [b for b in mapir.bases if b.kind == BaseKind.MAIN]
    comp = {int(labels[int(round(b.y)), int(round(b.x))]) for b in mains}
    mains_connected = len(comp) == 1 and 0 not in comp
    walk_ratio = round(float(mapir.walkable.sum()) / (mapir.width * mapir.height), 3)
    return {
        "mains_connected": mains_connected,
        "walkable_ratio": walk_ratio,
        "n_ramps": len(mapir.ramps),
        "n_regions": len(mapir.regions),
    }


def contact_sheet(pngs, out) -> None:
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
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--weirdness", type=float, default=0.0)
    ap.add_argument("--features", default=None)
    args = ap.parse_args()

    priors = load_priors(args.features, args.weirdness)
    gen = SkeletonGenerator(priors=priors)
    rcfg = RasterConfig(weirdness=args.weirdness)
    OUT.mkdir(parents=True, exist_ok=True)

    pngs, all_ok = [], 0
    for i in range(args.count):
        seed = args.seed + i
        skel = gen.generate(seed)
        mapir = rasterize(skel, seed=seed, cfg=rcfg)
        enrich_graph(mapir)
        out_dir = OUT / mapir.map_name
        mapir.save(out_dir)
        pngs.append(render(mapir, out_dir / "preview.png"))
        c = check(mapir)
        all_ok += c["mains_connected"]
        print(f"[m4] seed={seed}: {mapir.width}x{mapir.height} "
              f"walk={c['walkable_ratio']} ramps={c['n_ramps']} regions={c['n_regions']} "
              f"mains_connected={c['mains_connected']}")

    if args.count > 1:
        contact_sheet(pngs, OUT / "_contact_sheet.png")
        print(f"[m4] {all_ok}/{args.count} with mains connected; "
              f"contact sheet: {OUT/'_contact_sheet.png'}")


if __name__ == "__main__":
    main()
