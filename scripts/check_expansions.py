"""Offline replica of python-sc2's _find_expansion_locations, run on MapIR (buildable +
elevation + resources). Reports, per seed, how many bases have NO valid townhall spot -- the
exact condition that crashes the bot -- so we can tune resource/pad geometry without launching
the game. Distances/coords are in cells (the export adds a constant offset, so this is faithful).
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.generate.rasterize import RasterConfig, rasterize  # noqa: E402
from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator  # noqa: E402
from sc2mapgen.ir import ResourceKind  # noqa: E402

OFFSETS = [(x, y) for x, y in itertools.product(range(-7, 8), repeat=2) if 4 < math.hypot(x, y) <= 8]


def check(mapir) -> tuple[int, int, list[str]]:
    place = mapir.buildable
    elev = mapir.elevation
    h, w = place.shape

    def hgt(x, y):
        xi, yi = int(round(x)), int(round(y))
        return int(elev[yi, xi]) if 0 <= xi < w and 0 <= yi < h else -99

    def buildable(x, y):
        xi, yi = int(round(x)), int(round(y))
        return 0 <= xi < w and 0 <= yi < h and bool(place[yi, xi])

    res = [(r.x, r.y, r.kind == ResourceKind.GEYSER) for r in mapir.resources]
    groups = [[r] for r in res]

    def ctr(g):
        return (sum(p[0] for p in g) / len(g), sum(p[1] for p in g) / len(g))

    merged = True
    while merged:
        merged = False
        for a, b in itertools.combinations(groups, 2):
            ca, cb = ctr(a), ctr(b)
            if math.hypot(ca[0] - cb[0], ca[1] - cb[1]) <= 10.5 and all(
                    abs(hgt(*ra[:2]) - hgt(*rb[:2])) <= 10 for ra in a for rb in b):
                groups.remove(a); groups.remove(b); groups.append(a + b); merged = True
                break

    fails: list[str] = []
    for g in groups:
        if len(g) > 12:
            continue
        cx = int(sum(p[0] for p in g) / len(g)) + 0.5
        cy = int(sum(p[1] for p in g) / len(g)) + 0.5
        ok = any(
            buildable(cx + ox, cy + oy) and all(
                math.hypot(cx + ox - r[0], cy + oy - r[1]) >= (7 if r[2] else 6) for r in g)
            for ox, oy in OFFSETS)
        if not ok:
            nb = sum(1 for ox, oy in OFFSETS if buildable(cx + ox, cy + oy))
            on_void = sum(1 for p in g if not buildable(p[0], p[1]))
            fails.append(f"cluster n={len(g)} gey={sum(p[2] for p in g)} c=({cx:.0f},{cy:.0f}) "
                         f"build-in-ring={nb} on_void={on_void}")
    return len(groups), len(fails), fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    gen = SkeletonGenerator(config=GenConfig())
    rcfg = RasterConfig()
    tot_clusters = tot_fails = clean_maps = 0
    for i in range(args.count):
        seed = args.seed + i
        mapir = rasterize(gen.generate(seed), seed=seed, cfg=rcfg)
        nc, nf, fails = check(mapir)
        tot_clusters += nc
        tot_fails += nf
        clean_maps += (nf == 0)
        flag = "OK " if nf == 0 else "XX "
        print(f"{flag}seed={seed}: {nc} clusters, {nf} FAIL")
        if args.verbose:
            for f in fails:
                print(f"      {f}")
    print(f"\n{clean_maps}/{args.count} maps fully placeable; "
          f"{tot_fails} failing clusters / {tot_clusters} total")


if __name__ == "__main__":
    main()
