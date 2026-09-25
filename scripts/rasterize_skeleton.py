"""Milestone 4 driver: generate a skeleton, rasterize it to terrain, preview it.

    python scripts/rasterize_skeleton.py --seed 7
    python scripts/rasterize_skeleton.py --count 8 --weirdness 0.4
    python scripts/rasterize_skeleton.py --count 8 --features outputs/_features_test.json
    python scripts/rasterize_skeleton.py --count 100 --no-overlay

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

import numpy as np
from PIL import Image

from sc2mapgen.features import extract_features
from sc2mapgen.generate.plausibility import PlausibilityConfig, PlausibilityModel
from sc2mapgen.generate.priors import DefaultPriors, LearnedPriors
from sc2mapgen.generate.rasterize import RasterConfig, rasterize
from sc2mapgen.generate.skeleton import SYMMETRIES, GenConfig, SkeletonGenerator
from sc2mapgen.generate.validate import validate_map
from sc2mapgen.ingest.preview import render

OUT = Path("outputs/rasters")

# target openness band (walkable / playable-area): mid-fill with real structure/gaps
CORPUS_OPENNESS = (0.50, 0.62)


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




def contact_sheet(pngs, out, cols: int = 10, gap: int = 12) -> None:
    """Tile the previews at their native resolution. Pasting pixels instead of redrawing them
    through a figure keeps every tile as sharp as its own preview.png when zoomed in."""
    if not pngs:
        return
    tiles = [Image.open(p).convert("RGB") for p in pngs]
    tw = max(t.width for t in tiles)
    th = max(t.height for t in tiles)
    cols = min(cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap), "white")
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        x = gap + c * (tw + gap) + (tw - t.width) // 2
        y = gap + r * (th + gap) + (th - t.height) // 2
        sheet.paste(t, (x, y))
    sheet.save(out, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--weirdness", type=float, default=0.0)
    ap.add_argument("--features", default=None)
    ap.add_argument("--plausibility", default="dataset/_features.json",
                    help="corpus feature json for the M6 plausibility filter "
                         "(empty string disables it)")
    ap.add_argument("--symmetry", default="rot180",
                    choices=[*SYMMETRIES, "mixed"],
                    help="map symmetry: rot180 | mirror_lr | mirror_ud | mixed (all)")
    ap.add_argument("--no-overlay", action="store_true",
                    help="preview terrain, ramps, bases and resources only (no skeleton graph)")
    args = ap.parse_args()

    priors = load_priors(args.features, args.weirdness)
    syms = tuple(SYMMETRIES) if args.symmetry == "mixed" else (args.symmetry,)
    gen = SkeletonGenerator(priors=priors, config=GenConfig(symmetries=syms))
    rcfg = RasterConfig(weirdness=args.weirdness)
    OUT.mkdir(parents=True, exist_ok=True)

    # M6 plausibility filter: fit on the corpus once; the acceptance band widens with the same
    # weirdness knob so "controlled weirdness" maps survive.
    plaus = None
    if args.plausibility and Path(args.plausibility).exists():
        plaus = PlausibilityModel.from_file(
            args.plausibility, PlausibilityConfig(weirdness=args.weirdness))
        print(f"[m6] plausibility model fit on {plaus.corpus_std.shape[0]} maps, "
              f"features={list(plaus.features)}")

    pngs, valid_ok, mains_ok, band_ok, plaus_ok = [], 0, 0, 0, 0
    for i in range(args.count):
        seed = args.seed + i
        skel = gen.generate(seed)
        mapir = rasterize(skel, seed=seed, cfg=rcfg)
        enrich_graph(mapir)
        out_dir = OUT / mapir.map_name
        mapir.save(out_dir)

        rep = validate_map(mapir)                       # Milestone 5 validator
        pngs.append(render(mapir, out_dir / "preview.png", skeleton=skel,
                           subtitle="VALID" if rep.ok else "INVALID",
                           overlay=not args.no_overlay))
        m = rep.metrics
        valid_ok += rep.ok
        mains_ok += bool(m.get("mains_connected"))
        band_ok += CORPUS_OPENNESS[0] <= m.get("openness", 0) <= CORPUS_OPENNESS[1]
        tag = "VALID  " if rep.ok else "INVALID"
        print(f"[m5] seed={seed}: {tag} {mapir.width}x{mapir.height} "
              f"bases={m.get('n_bases')} resources={m.get('n_resources')} "
              f"ramps={m.get('ramps_clean')}/{m.get('n_ramps')}clean "
              f"rush={m.get('rush_distance')} openness={m.get('openness')} "
              f"sym={m.get('symmetry')}")
        for msg in rep.failures:
            print(f"        FAIL: {msg}")
        for msg in rep.warnings:
            print(f"        warn: {msg}")

        if plaus is not None:
            pr = plaus.score(extract_features(mapir))
            plaus_ok += pr.plausible
            print(f"        [m6] {pr.summary()}")
            for msg in pr.outliers:
                print(f"        outlier: {msg}")

    if args.count > 1:
        contact_sheet(pngs, OUT / "_contact_sheet.png")
        extra = f", {plaus_ok}/{args.count} plausible" if plaus is not None else ""
        print(f"[m5] {valid_ok}/{args.count} fully valid, {mains_ok}/{args.count} mains connected, "
              f"{band_ok}/{args.count} in openness band{extra}; "
              f"contact sheet: {OUT/'_contact_sheet.png'}")


if __name__ == "__main__":
    main()
