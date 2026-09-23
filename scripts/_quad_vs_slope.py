"""Where does each <rampList> quad sit relative to its CLIF slope? (ours vs gold)

For every diagonal/cardinal ramp entry: step along the quad's uphill vector u from the base and mid
points and report the CLIF values under them, plus how many cells uphill of the slope's top edge
(the last sub-level cell before the plateau) the base sits. Gold quads sit ON the slope/interface.
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.export import sc2map  # noqa: E402

VEC = r"u\(([-\d.e+]+), ([-\d.e+]+)\) r\(([-\d.e+]+), ([-\d.e+]+)\) c=\(([-\d.e+]+), ([-\d.e+]+)\)"


def ramps(p):
    a = MPQArchive(p)
    C = sc2map.decode_cliff(a.read_file("t3SyncCliffLevel"))[3].astype(int)
    x = a.read_file("t3Terrain.xml").decode("utf-8", "replace")
    out = []
    for m in re.finditer(r'<ramp dir="(\d+)"[^>]*?base="' + VEC + r'[^"]*" mid="' + VEC, x):
        g = [float(v) for v in m.groups()[1:]]
        out.append((int(m.group(1)), (g[0], g[1]), (g[4], g[5]), (g[10], g[11])))
    return C, out


def cell(C, px, py):
    X, Y = int(np.floor(px)), int(np.floor(py))
    return int(C[Y, X]) if 0 <= Y < C.shape[0] and 0 <= X < C.shape[1] else -1


def depth_past_slope(C, base, u):
    """Walk downhill (-u) from the base until we hit a sub-level (non-multiple-of-64) cell."""
    for k in range(0, 15):
        v = cell(C, base[0] - u[0] * k, base[1] - u[1] * k)
        if v > 0 and v % 64:
            return k
    return None


if __name__ == "__main__":
    paths = sys.argv[1:] or ["outputs/export/gen_24.SC2Map", "outputs/export/gen_24_baseline.SC2Map",
                             *[f for f in glob.glob("gold_maps/*.SC2Map")
                               if any(k in f for k in ("FrostLE", "AutomatonAIE", "LostandFound"))]]
    for p in paths:
        C, rs = ramps(p)
        ds = [depth_past_slope(C, base, u) for d, u, base, mid in rs]
        print(f"{Path(p).name:28s} ramps={len(rs):2d}  cells from base down to first sloped cell: {ds}")
