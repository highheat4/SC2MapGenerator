"""EXPERIMENT: crop every level-changing ramp to a narrow straight choke band, re-export, and
report offline terrain connectivity. Then we probe in-engine to see if narrow ramps bridge
globally (unlike the wide emergent ones). Throwaway; informs the real rasterizer fix.

    PYTHONPATH=src .venv/bin/python scripts/narrow_experiment.py 234 [width]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator
from sc2mapgen.generate.rasterize import RasterConfig, rasterize, _snap8, ramp_cleanliness
from sc2mapgen.ir import Ramp
from sc2mapgen.export import ExportConfig, export_sc2map

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 234
WIDTH = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0


def narrow_ramps(mapir, width: float):
    """Crop each ramp component to a width-`width` straight band on its snapped low->high axis;
    wall (mark unwalkable) the cropped-off flank cells so they become cliff, and rebuild ramps."""
    walk = mapir.walkable
    elev = mapir.elevation.astype(int)
    h, w = walk.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    ramp = np.zeros_like(walk)
    for r in mapir.ramps:
        for x, y in r.cells:
            ramp[y, x] = True
    new_ramps = []
    for r in mapir.ramps:
        comp = np.zeros_like(walk)
        for x, y in r.cells:
            comp[y, x] = True
        ux, uy = (r.top[0] - r.bottom[0]), (r.top[1] - r.bottom[1])
        n = math.hypot(ux, uy) or 1.0
        ux, uy = _snap8(ux / n, uy / n)
        cx, cy = xx[comp].mean(), yy[comp].mean()
        perp = np.abs((xx - cx) * (-uy) + (yy - cy) * ux)
        drop = comp & (perp > width / 2.0)
        walk[drop] = False
        ramp[drop] = False
        comp2 = comp & ~drop
        cells = [(int(x), int(y)) for y, x in zip(*np.where(comp2))]
        if cells:
            new_ramps.append(Ramp(cells=cells, top=r.top, bottom=r.bottom,
                                  low_level=r.low_level, high_level=r.high_level, width=None))
    mapir.ramps = new_ramps
    # recompute buildable (drop no longer walkable)
    edt = ndimage.distance_transform_edt(walk)
    mapir.buildable = walk & (edt >= 2.0) & (~ramp)
    return mapir


def main() -> None:
    g = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
    mapir = rasterize(g.generate(SEED), seed=SEED, cfg=RasterConfig())
    print("before:", [(len(r.cells)) for r in mapir.ramps])
    mapir = narrow_ramps(mapir, WIDTH)
    print("after :", [(len(r.cells)) for r in mapir.ramps])
    clean = sum(1 for *_x, ok in ramp_cleanliness(mapir) if ok)
    print(f"clean ramps: {clean}/{len(mapir.ramps)}")
    name = f"gen_{SEED}"
    export_sc2map(mapir, f"outputs/export/{name}.SC2Map", ExportConfig(author_ramps=True))
    print(f"exported outputs/export/{name}.SC2Map")


if __name__ == "__main__":
    main()
