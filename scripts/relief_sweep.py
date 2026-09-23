"""Offline characterization of the rasterizer's RELIEF output across a seed sweep.

For each seed, rasterize (no export, no engine) and report the *engine-accurate* state:
    * LR = number of level-components (rasterize._level_components) covering the real bases
      (1 == every base is engine-traversably connected; >1 == sealed level-islands).
    * mains: whether both MAINs sit in one level-component.
    * ramps: total / clean (rasterize.ramp_cleanliness) / biggest ramp component (cells).
    * levels: distinct plateau levels present (relief variety).
    * valid: M5 validator ok.

    PYTHONPATH=src .venv/bin/python scripts/relief_sweep.py --seed 200 --count 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc2mapgen.generate.rasterize import (  # noqa: E402
    RasterConfig, _level_components, rasterize, ramp_cleanliness,
)
from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator  # noqa: E402
from sc2mapgen.generate.validate import validate_map  # noqa: E402
from sc2mapgen.ir import BaseKind  # noqa: E402

_REAL = {BaseKind.MAIN, BaseKind.NATURAL, BaseKind.BASE}


def ramp_mask(mapir) -> np.ndarray:
    m = np.zeros_like(mapir.walkable, dtype=bool)
    for r in mapir.ramps:
        for x, y in r.cells:
            m[y, x] = True
    return m


def analyze(mapir):
    walk = mapir.walkable
    elev = mapir.elevation.astype(int)
    ramp = ramp_mask(mapir)
    comp = _level_components(walk, elev, ramp)

    def cid(b):
        return int(comp[int(round(b.y)), int(round(b.x))])

    reals = [b for b in mapir.bases if b.kind in _REAL]
    mains = [b for b in mapir.bases if b.kind == BaseKind.MAIN]
    base_comps = {cid(b) for b in reals}
    main_comps = {cid(b) for b in mains}
    # biggest ramp component
    lab, n = ndimage.label(ramp, structure=np.ones((3, 3), bool))
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) if n else []
    biggest = int(max(sizes)) if len(sizes) else 0
    clean = sum(1 for *_x, ok in ramp_cleanliness(mapir) if ok)
    levels = sorted({int(v) for v in elev[walk]})
    return {
        "LR": len(base_comps),
        "mains_conn": len(main_comps) == 1,
        "n_ramps": len(mapir.ramps),
        "clean": clean,
        "biggest_blob": biggest,
        "levels": levels,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--count", type=int, default=40)
    args = ap.parse_args()

    gen = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
    rcfg = RasterConfig()

    lr1 = mains = valid = allclean = 0
    for i in range(args.count):
        seed = args.seed + i
        mapir = rasterize(gen.generate(seed), seed=seed, cfg=rcfg)
        a = analyze(mapir)
        rep = validate_map(mapir)
        lr1 += a["LR"] == 1
        mains += a["mains_conn"]
        valid += rep.ok
        allclean += a["clean"] == a["n_ramps"]
        print(f"seed={seed:>4} LR={a['LR']} mains={'Y' if a['mains_conn'] else '.'} "
              f"ramps={a['clean']}/{a['n_ramps']}clean big={a['biggest_blob']:>3} "
              f"levels={a['levels']} valid={'Y' if rep.ok else '.'}"
              + ("" if rep.ok else f"  fails={rep.failures[:2]}"))
    c = args.count
    print(f"\n[sweep] LR==1: {lr1}/{c}   mains_conn: {mains}/{c}   "
          f"all_ramps_clean: {allclean}/{c}   M5_valid: {valid}/{c}")


if __name__ == "__main__":
    main()
