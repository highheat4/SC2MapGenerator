"""Gold DIAGONAL quads: declared width (base.w / sqrt2) vs the slope's top-row length (cell count).

The low corner markers sit at mid -/+ width*(1,1)-ish, so width decides whether they land on the
band's edges or overshoot onto the plateau beyond it.
"""
from __future__ import annotations

import collections
import glob
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _quad_vs_slope import VEC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mpyq import MPQArchive  # noqa: E402
from sc2mapgen.export import sc2map  # noqa: E402

pairs = collections.Counter()
for p in sorted(glob.glob("gold_maps/*.SC2Map")):
    try:
        a = MPQArchive(p)
        C = sc2map.decode_cliff(a.read_file("t3SyncCliffLevel"))[3].astype(int)
        x = a.read_file("t3Terrain.xml").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        continue
    top = (C > 0) & (C % 64 == 48)
    lab, _ = ndimage.label(top, structure=np.ones((3, 3)))
    ys, xs = np.nonzero(top)
    if not len(xs):
        continue
    for m in re.finditer(r'<ramp dir="(\d+)"[^>]*?base="' + VEC + r' w=([-\d.e+]+)', x):
        d = int(m.group(1))
        if d < 4:
            continue
        bx, by, bw = float(m.group(6)), float(m.group(7)), float(m.group(8))
        i = np.argmin((xs + 0.5 - bx) ** 2 + (ys + 0.5 - by) ** 2)
        n = int((lab == lab[ys[i], xs[i]]).sum())
        pairs[(n, round(bw / 2 ** 0.5, 2))] += 1

print("top-row cells -> declared width : count")
for (n, w), k in sorted(pairs.items()):
    print(f"  {n:3d} -> {w:5.2f} : {k}")

# CARDINAL quads: low-corner distance from mid (along r) and base/mid centring vs the top row.
LO = r'leftLo="' + VEC + r'[^"]*" leftHi="[^"]*" rightLo="' + VEC
card = collections.Counter()
for p in sorted(glob.glob("gold_maps/*.SC2Map")):
    try:
        a = MPQArchive(p)
        C = sc2map.decode_cliff(a.read_file("t3SyncCliffLevel"))[3].astype(int)
        x = a.read_file("t3Terrain.xml").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        continue
    top = (C > 0) & (C % 64 == 48)
    lab, _ = ndimage.label(top)
    ys, xs = np.nonzero(top)
    if not len(xs):
        continue
    for m in re.finditer(r'<ramp dir="([0-3])"[^>]*?' + LO + r'[^>]*?mid="' + VEC, x):
        g = [float(v) for v in m.groups()[1:]]
        lx, ly, rx, ry, mx, my = g[4], g[5], g[10], g[11], g[16], g[17]
        i = np.argmin((xs + 0.5 - mx) ** 2 + (ys + 0.5 - my) ** 2)
        cy, cx = np.nonzero(lab == lab[ys[i], xs[i]])
        n = len(cx)
        horiz = int(m.group(1)) in (0, 1)
        centre = (cx.mean() + 0.5) if horiz else (cy.mean() + 0.5)
        mlat = mx if horiz else my
        half = abs((lx - rx) if horiz else (ly - ry)) / 2
        card[(n, round(half, 1), round(mlat - centre, 1))] += 1
print("\nCARDINAL: top-row cells, corner half-spread from mid, mid offset from top-row centre : count")
for (n, h, off), k in sorted(card.items()):
    print(f"  {n:3d}  half={h:4.1f}  mid-offset={off:5.1f} : {k}")
