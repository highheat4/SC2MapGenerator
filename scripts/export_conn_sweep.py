"""Sweep: export N seeds and check ENGINE-ACCURATE (4-connected) terrain connectivity of each.

The engine paths on 4-connectivity (no diagonal squeeze through a void corner) and treats two
adjacent non-void cells as traversable iff |Δcliff| <= 8. We export each seed's relief map, then
report whether the two start locations land in one 4-connected <=8-step component (main-to-main)
and whether every authored ramp climbs cleanly (no >8 interior step). Fast offline proxy for the
in-engine query_pathing bridging test.

    PYTHONPATH=src .venv/bin/python scripts/export_conn_sweep.py 30
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator  # noqa: E402
from sc2mapgen.generate.rasterize import RasterConfig, rasterize  # noqa: E402
from sc2mapgen.generate.validate import validate_map  # noqa: E402
from sc2mapgen.export import ExportConfig, export_sc2map, sc2map  # noqa: E402


def conn4_main_to_main(cliff: np.ndarray, starts, max_step: int = 8) -> bool:
    nz = cliff > 0
    h, w = cliff.shape
    parent = np.arange(h * w)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for dy, dx in ((0, 1), (1, 0)):
        m = nz & np.roll(np.roll(nz, dy, 0), dx, 1) & (
            np.abs(cliff - np.roll(np.roll(cliff, dy, 0), dx, 1)) <= max_step)
        if dy == 1:
            m[0, :] = False
        if dx == 1:
            m[:, 0] = False
        ys, xs = np.where(m)
        for y, x in zip(ys.tolist(), xs.tolist()):
            parent[find(y * w + x)] = find(((y - dy) % h) * w + (x - dx) % w)
    if len(starts) < 2:
        return False
    (x0, y0), (x1, y1) = starts[0], starts[1]
    return find(y0 * w + x0) == find(y1 * w + x1)


def starts_of(arch):
    x = arch.read_file("Objects").decode("utf-8", "replace")
    pts = []
    for m in re.finditer(r'Position="([^"]+)"[^>]*Type="StartLoc"', x):
        p = m.group(1).split(",")
        pts.append((int(float(p[0])), int(float(p[1]))))
    return pts


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ok = m5 = conn = 0
    fails = []
    for seed in range(100, 100 + n):
        g = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
        mapir = rasterize(g.generate(seed), seed=seed, cfg=RasterConfig())
        rep = validate_map(mapir)
        out = Path("outputs/export") / f"sweep_{seed}.SC2Map"
        tmpl = "gold_maps/FrostLE.SC2Map"
        export_sc2map(mapir, out, ExportConfig(author_ramps=True, template=tmpl))
        arch = MPQArchive(str(out))
        _v, cw, ch, cliff = sc2map.decode_cliff(arch.read_file("t3SyncCliffLevel"))
        c = conn4_main_to_main(cliff.astype(np.int32), starts_of(arch))
        ok += 1
        m5 += int(rep.ok)
        conn += int(c)
        if not c or not rep.ok:
            fails.append((seed, rep.ok, c, len(mapir.ramps)))
        out.unlink(missing_ok=True)
    print(f"seeds={ok}  M5_valid={m5}/{ok}  main_to_main_4conn={conn}/{ok}")
    for seed, v, c, nr in fails:
        print(f"  seed {seed}: M5={'ok' if v else 'FAIL'} conn={'ok' if c else 'FAIL'} ramps={nr}")


if __name__ == "__main__":
    main()
