"""Count 'ledge' cells per ramp: HIGH-plateau cells beside a ramp that lie DOWNHILL of its top row.

Gold ramps protrude from a straight high-plateau edge that runs along the slope's top row, so all
nearby high ground is at or uphill of the top row (lattice projection L >= L_top). High ground at
L < L_top runs alongside the slope: a ledge a unit can stand on above the ramp.

For each connected component of sloped cells (cliff not a multiple of 64): the uphill direction is
fitted from the cliff values and snapped to 8 directions -> integer lattice step (ix, iy), and
L = x*ix + y*iy. Reported per ramp: direction, sloped cells, ledge cells (hi plateau within
Chebyshev distance R of the slope with L < L_top).

    PYTHONPATH=src .venv/bin/python scripts/_ledge_count.py [MAP ...]
"""
from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import numpy as np
from mpyq import MPQArchive
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.export import sc2map  # noqa: E402
from sc2mapgen.ir import RAMP_DIR_U  # noqa: E402

R = 3
NAMES = ("N", "S", "W", "E", "NW", "NE", "SW", "SE")


def lattice(ux: float, uy: float) -> tuple[int, int, int]:
    best = max(range(8), key=lambda i: ux * RAMP_DIR_U[i][0] + uy * RAMP_DIR_U[i][1])
    vx, vy = RAMP_DIR_U[best]
    s = math.sqrt(2.0) if best >= 4 else 1.0
    return best, int(round(vx * s)), int(round(vy * s))


def ramp_ledges(C: np.ndarray) -> list[tuple[int, int, int, np.ndarray]]:
    h, w = C.shape
    sloped = (C > 0) & (C % 64 != 0)
    lab, n = ndimage.label(sloped, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 6:
            continue
        v = C[ys, xs].astype(float)
        A = np.c_[xs - xs.mean(), ys - ys.mean()]
        (a, b), *_ = np.linalg.lstsq(A, v - v.mean(), rcond=None)
        d, ix, iy = lattice(a, b)
        hi = (int(v.max()) // 64 + 1) * 64
        L_top = int((xs * ix + ys * iy).max())
        near = ndimage.binary_dilation(lab == i, structure=np.ones((3, 3)), iterations=R)
        gy, gx = np.mgrid[0:h, 0:w]
        ledge = near & (C == hi) & (gx * ix + gy * iy < L_top)
        out.append((d, len(xs), int(ledge.sum()), ledge))
    return out


if __name__ == "__main__":
    paths = sys.argv[1:] or ["outputs/export/gen_24.SC2Map",
                             *[f for f in glob.glob("gold_maps/*.SC2Map") if "Goldenaura512" in f]]
    for p in paths:
        C = sc2map.decode_cliff(MPQArchive(p).read_file("t3SyncCliffLevel"))[3].astype(int)
        rs = ramp_ledges(C)
        tot = sum(k for _d, _n, k, _m in rs)
        print(f"{Path(p).name}: ramps={len(rs)} total ledge cells={tot}")
        for d, n, k, _m in rs:
            print(f"   dir={NAMES[d]:2s} sloped={n:3d} ledge={k}")
