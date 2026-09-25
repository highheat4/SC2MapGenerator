"""Census of the cells channel_ramp_flanks voids, split by which plateau they belong to.

    PYTHONPATH=src .venv/bin/python scripts/_flank_census.py 0 30
"""
from __future__ import annotations

import sys

import numpy as np

from sc2mapgen.generate.rasterize import RasterConfig, engine_cliff_grid, rasterize
from sc2mapgen.generate.skeleton import GenConfig, SkeletonError, SkeletonGenerator
from sc2mapgen.generate.rasterize import RasterizeError
from sc2mapgen.ir import ramp_cell_cliffs, ramp_uphill


def main() -> None:
    a, b = int(sys.argv[1]), int(sys.argv[2])
    g = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
    tot_lo = tot_hi = 0
    for s in range(a, b):
        try:
            m = rasterize(g.generate(s), seed=s, cfg=RasterConfig())
        except (SkeletonError, RasterizeError) as e:
            print(f"seed {s}: skip ({type(e).__name__})")
            continue
        h, w = m.walkable.shape
        lev = np.clip(m.elevation.astype(np.int32), 0, None)
        pre = np.where(m.walkable, (lev + 1) * 64, 0).astype(np.int32)
        rm = np.zeros((h, w), bool)
        for r in m.ramps:
            if r.low_level is None or r.top is None:
                continue
            cells = [(int(x), int(y)) for x, y in r.cells]
            _, (ux, uy) = ramp_uphill(r)
            lo_c, hi_c = (int(r.low_level) + 1) * 64, (int(r.high_level) + 1) * 64
            for (x, y), v in zip(cells, ramp_cell_cliffs(cells, ux, uy, lo_c, hi_c)):
                pre[y, x] = v
                rm[y, x] = True
        post = engine_cliff_grid(m.walkable, m.elevation, m.ramps)
        voided = (pre > 0) & (post == 0)
        ys, xs = np.where(voided)
        n_lo = n_hi = 0
        for y, x in zip(ys, xs):
            v = pre[y, x]
            nb = [pre[y + dy, x + dx] for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= y + dy < h and 0 <= x + dx < w and rm[y + dy, x + dx]]
            if nb and v < min(nb):
                n_lo += 1
            else:
                n_hi += 1
        tot_lo += n_lo
        tot_hi += n_hi
        print(f"seed {s}: ramps={len(m.ramps)} voided={int(voided.sum())} "
              f"low-plateau={n_lo} high-plateau={n_hi}")
    print(f"TOTAL low-plateau={tot_lo} high-plateau={tot_hi}")


if __name__ == "__main__":
    main()
