"""Ad-hoc probe: inspect seed 24's ramp height data to explain 'divots'.

Reproduces the generated map, finds the ramp nearest the top-right base, and prints, per ramp
cell, the CLIF value and the derived SMAP/HMAP heights -- so we can see exactly which channel
produces the visible height dips and why.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc2mapgen.generate.rasterize import RasterConfig, rasterize  # noqa: E402
from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator  # noqa: E402
from sc2mapgen import ir as IR  # noqa: E402
from sc2mapgen.export import sc2map  # noqa: E402


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    gen = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
    mapir = rasterize(gen.generate(seed), seed=seed, cfg=RasterConfig())

    print(f"seed={seed} dims={mapir.width}x{mapir.height} ramps={len(mapir.ramps)}")
    # top-right base = max x, min y among real bases
    bases = [b for b in mapir.bases]
    for b in bases:
        lv = int(mapir.elevation[int(b.y), int(b.x)])
        print(f"  base kind={b.kind} pos=({b.x:.1f},{b.y:.1f}) elev={lv}")

    # pick the top-right-most base
    tr = min(bases, key=lambda b: (-b.x) + b.y)
    print(f"\ntop-right base: kind={tr.kind} pos=({tr.x:.1f},{tr.y:.1f})")

    # nearest ramp to that base
    def ramp_centroid(r):
        cs = np.array(r.cells, dtype=float)
        return cs[:, 0].mean(), cs[:, 1].mean()

    ramps = list(mapir.ramps)
    ramps.sort(key=lambda r: (ramp_centroid(r)[0] - tr.x) ** 2
                             + (ramp_centroid(r)[1] - tr.y) ** 2)
    r = ramps[0]
    cx, cy = ramp_centroid(r)
    print(f"nearest ramp: cells={len(r.cells)} centroid=({cx:.1f},{cy:.1f}) "
          f"lo={r.low_level} hi={r.high_level} top={getattr(r,'top',None)} "
          f"bottom={getattr(r,'bottom',None)}")

    # replicate the exporter gradient on this ramp's cells
    lo_cliff = (r.low_level + 1) * 64
    hi_cliff = (r.high_level + 1) * 64
    uvx = float(r.top[0] - r.bottom[0])
    uvy = float(r.top[1] - r.bottom[1])
    idx, (ux, uy) = IR.snap_uphill(uvx, uvy)
    print(f"lo_cliff={lo_cliff} hi_cliff={hi_cliff} snapped_dir={idx} u=({ux:.3f},{uy:.3f}) "
          f"span={IR.ramp_span(lo_cliff, hi_cliff)}")

    cells = [(int(x), int(y)) for (x, y) in r.cells]
    cvals = IR.ramp_cell_cliffs(cells, ux, uy, lo_cliff, hi_cliff)

    # lattice projection for each cell (the isoline coordinate)
    from sc2mapgen.ir import _lattice_dir
    ix, iy = _lattice_dir(ux, uy)
    proj = [x * ix + y * iy for (x, y) in cells]

    print("\nramp cells sorted by uphill lattice projection L (isoline coord):")
    print(" L   x   y   CLIF")
    order = sorted(range(len(cells)), key=lambda i: proj[i])
    for i in order:
        (x, y) = cells[i]
        print(f"{proj[i]:3d} {x:3d} {y:3d}   {cvals[i]:4d}")

    # Now build the full exported CLIF cell grid + derive SMAP/HMAP heights the way the exporter
    # does, and print a small window around the ramp so divots are visible as height values.
    print("\n--- deriving exported CLIF grid + heights around the ramp ---")
    n_tiers = int(mapir.elevation.max()) + 1
    # local cliff grid (plateaus), then overlay this ramp's gradient
    walk = mapir.walkable
    elev = np.clip(mapir.elevation.astype(np.int32), 0, n_tiers - 1)
    tier = np.where(walk, elev, -1)
    H, W = walk.shape
    clg = np.zeros((H, W), dtype=np.int32)
    for t in range(n_tiers):
        clg[tier == t] = (t + 1) * 64
    # overlay all ramps' gradients (like engine_cliff_grid)
    for rr in mapir.ramps:
        if rr.low_level is None or rr.high_level is None:
            continue
        lo_c = (rr.low_level + 1) * 64
        hi_c = (rr.high_level + 1) * 64
        if hi_c <= lo_c:
            continue
        uvx2 = float(rr.top[0] - rr.bottom[0]); uvy2 = float(rr.top[1] - rr.bottom[1])
        _, (u2x, u2y) = IR.snap_uphill(uvx2, uvy2)
        cc = [(int(x), int(y)) for (x, y) in rr.cells]
        for (x, y), cv in zip(cc, IR.ramp_cell_cliffs(cc, u2x, u2y, lo_c, hi_c)):
            clg[y, x] = cv

    xs = [x for (x, y) in cells]; ys = [y for (x, y) in cells]
    x0, x1 = min(xs) - 2, max(xs) + 3
    y0, y1 = min(ys) - 2, max(ys) + 3
    print(f"CLIF window x[{x0}:{x1}] y[{y0}:{y1}] (0=void):")
    for y in range(y0, y1):
        row = " ".join(f"{int(clg[y, x]):3d}" if 0 <= x < W and 0 <= y < H else "  ." for x in range(x0, x1))
        print(f"y{y:3d}: {row}")


if __name__ == "__main__":
    main()
