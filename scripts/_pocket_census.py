"""Census of townhall pockets on exported relief maps.

For every real base: the 5x5 townhall footprint must be one non-void level in the exported CLIF,
its 1-cell ring must hold no higher or ramp cell (either bends a footprint vertex, so the engine
reads it unbuildable), and the footprint and resources must lie inside the template's playable
rect. Prints each violation with its distance to the IR grid and playable edges.

    PYTHONPATH=src .venv/bin/python scripts/_pocket_census.py 0 60 [rot180|mixed]
"""
from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc2mapgen.export import ExportConfig, TemplateFitError, export_sc2map, sc2map  # noqa: E402
from sc2mapgen.generate.rasterize import RasterConfig, RasterizeError, rasterize  # noqa: E402
from sc2mapgen.generate.skeleton import (  # noqa: E402
    SYMMETRIES, GenConfig, SkeletonError, SkeletonGenerator,
)
from sc2mapgen.ir import BaseKind  # noqa: E402

REAL = (BaseKind.MAIN, BaseKind.NATURAL, BaseKind.BASE)


def main() -> None:
    a, b = int(sys.argv[1]), int(sys.argv[2])
    sym = sys.argv[3] if len(sys.argv) > 3 else "rot180"
    syms = tuple(SYMMETRIES) if sym == "mixed" else (sym,)
    g = SkeletonGenerator(config=GenConfig(symmetries=syms))
    tmp = Path(tempfile.mkdtemp())
    n_bases = n_bad = n_maps_bad = n_nofit = 0
    for s in range(a, b):
        try:
            m = rasterize(g.generate(s), seed=s, cfg=RasterConfig())
        except (SkeletonError, RasterizeError):
            continue
        try:
            info = export_sc2map(m, tmp / f"s{s}.SC2Map", ExportConfig(author_ramps=True))
        except TemplateFitError as exc:
            n_nofit += 1
            print(f"seed {s}: NO TEMPLATE ({exc})")
            continue
        ox, oy = info["offset"]
        arch = MPQArchive(info["out_path"])
        _, cw, ch, cl = sc2map.decode_cliff(arch.read_file("t3SyncCliffLevel"))
        cl = cl.astype(np.int32)
        mi = arch.read_file("MapInfo")
        off, _, _ = sc2map._mapinfo_playable_offset(mi)
        pl, pb, pr, pt = struct.unpack_from("<4i", mi, off)
        ramp = sc2map.ramp_cell_mask(m, cw, ch, ox, oy)
        bad_here = []
        for base in m.bases:
            if base.kind not in REAL:
                continue
            n_bases += 1
            tx, ty = int(base.x) + ox, int(base.y) + oy
            probs = []
            fp = cl[ty - 2:ty + 3, tx - 2:tx + 3]
            if fp.shape != (5, 5) or (fp == 0).any() or len(set(fp.ravel().tolist())) != 1:
                probs.append(f"footprint levels={sorted(set(fp.ravel().tolist()))}")
            else:
                lv = int(fp[0, 0])
                ring = cl[ty - 3:ty + 4, tx - 3:tx + 4]
                rr = ramp[ty - 3:ty + 4, tx - 3:tx + 4]
                if (ring > lv).any() or rr.any():
                    probs.append(f"ring higher={int((ring > lv).sum())} ramp={int(rr.sum())}")
            if not (pl <= tx - 2 and tx + 3 <= pr and pb <= ty - 2 and ty + 3 <= pt):
                probs.append("footprint outside playable")
            res_out = [(r.x, r.y) for r in m.resources
                       if abs(r.x - base.x) <= 9 and abs(r.y - base.y) <= 9
                       and not (pl <= r.x + ox < pr and pb <= r.y + oy < pt)]
            if res_out:
                probs.append(f"{len(res_out)} resources outside playable")
            if probs:
                n_bad += 1
                edge_ir = min(base.x, base.y, m.width - base.x, m.height - base.y)
                edge_pl = min(tx - pl, ty - pb, pr - tx, pt - ty)
                bad_here.append(f"{base.kind.name}@({base.x:.0f},{base.y:.0f}) "
                                f"ir_edge={edge_ir:.0f} play_edge={edge_pl} " + "; ".join(probs))
        tpl = Path(info["template"]).stem
        if bad_here:
            n_maps_bad += 1
            print(f"seed {s}: {m.width}x{m.height} -> {tpl} off=({ox},{oy}) "
                  f"playable=({pl},{pb},{pr},{pt})")
            for line in bad_here:
                print(f"    {line}")
    print(f"TOTAL {n_bad}/{n_bases} bases bad on {n_maps_bad} maps; {n_nofit} seeds fit no template")


if __name__ == "__main__":
    main()
