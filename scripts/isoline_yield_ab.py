"""A/B the OFFLINE STRICT yield of the isoline gradient vs the shipped rank-order gradient across a
range of carve runs, to show isoline holds yield at GOLD-SHORT runs where rank-order collapses.

Yield = fraction of seeds whose rasterized RELIEF map passes ``validate_map`` (the engine-accurate
<=8/4-connected oracle, ``engine_cliff_grid``). The gradient is chosen by the ``ISOLINE_RAMPS`` env
(read by ``ir.gradient_ranks``, shared by the exporter + the oracle so they can't drift). Carve run
is set via ``RasterConfig.ramp_run_range``. Also reports the median per-ramp cliff STEP (8 == a
clean single-8 staircase; 16 == a Delta16 wall packed into the band).

    PYTHONPATH=src .venv/bin/python scripts/isoline_yield_ab.py --seeds 60
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.generate.rasterize import RasterConfig, engine_cliff_grid, rasterize  # noqa: E402
from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator  # noqa: E402
from sc2mapgen.generate.validate import validate_map  # noqa: E402


def _median_ramp_step(mapir) -> float:
    """Median 4-connected cliff step ON ramp cells of the EXPORTED-faithful CLIF grid (8 = clean)."""
    walk = mapir.walkable
    elev = mapir.elevation.astype(int)
    cliff = engine_cliff_grid(walk, elev, mapir.ramps)
    rmask = np.zeros_like(walk)
    for r in mapir.ramps:
        for x, y in r.cells:
            if 0 <= y < rmask.shape[0] and 0 <= x < rmask.shape[1]:
                rmask[y, x] = True
    steps = []
    h, w = cliff.shape
    ys, xs = np.where(rmask & (cliff > 0))
    for y, x in zip(ys.tolist(), xs.tolist()):
        v = int(cliff[y, x])
        for dy, dx in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and cliff[ny, nx] > 0:
                steps.append(abs(int(cliff[ny, nx]) - v))
    return float(np.median(steps)) if steps else 0.0


def sweep(seed0: int, n: int, run_range, isoline: bool) -> tuple[int, float]:
    # Isoline is now the shipped DEFAULT (ir.gradient_ranks); the legacy even-spread gradient is the
    # opt-OUT via RANK_ORDER_RAMPS (findings §4.10). Set it for the rank-order rows, clear it for
    # isoline. (Older revisions used an opt-IN ISOLINE_RAMPS flag; that no longer selects anything.)
    if isoline:
        os.environ.pop("RANK_ORDER_RAMPS", None)
    else:
        os.environ["RANK_ORDER_RAMPS"] = "1"
    gen = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
    cfg = RasterConfig()
    cfg.ramp_run_range = run_range
    ok = 0
    steps = []
    for s in range(seed0, seed0 + n):
        mapir = rasterize(gen.generate(s), seed=s, cfg=cfg)
        if validate_map(mapir).ok:
            ok += 1
        steps.append(_median_ramp_step(mapir))
    return ok, float(np.median(steps)) if steps else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=60)
    args = ap.parse_args()
    configs = [
        ("rank-order", (11.0, 14.0), False),
        ("isoline   ", (11.0, 14.0), True),
        ("isoline   ", (8.0, 10.0), True),
        ("isoline   ", (6.0, 8.0), True),
        ("rank-order", (6.0, 8.0), False),
    ]
    print(f"seeds {args.seed}..{args.seed + args.seeds - 1} (n={args.seeds}), symmetry=rot180\n")
    print(f"{'gradient':<11} {'carve run':>12}   {'STRICT yield':>13}   {'med ramp step':>13}")
    print("-" * 58)
    for name, rr, iso in configs:
        ok, step = sweep(args.seed, args.seeds, rr, iso)
        print(f"{name:<11} {str(rr):>12}   {ok:>3}/{args.seeds:<9}   {step:>13.1f}")


if __name__ == "__main__":
    main()
