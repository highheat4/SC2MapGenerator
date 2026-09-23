"""Gold CARDINAL ramps: quad anchor vs the slope's top row, and the CLIF profile along uphill.

For each gold cardinal <ramp> (dir 0-3) prints aggregate stats of:
  * base offset from the centroid of the nearest top-row component (cells at the ramp's highest
    sub-level), along u and sideways;
  * the CLIF values sampled every cell downhill from base (the staircase profile);
  * the raw quad params (base w/h, mid w/h, run).
Pass map paths to inspect specific files (e.g. our export) instead of the gold corpus.
"""
from __future__ import annotations

import collections
import glob
import re
import sys
from pathlib import Path

import numpy as np
from mpyq import MPQArchive
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.export import sc2map  # noqa: E402

F = r"([-\d.e+]+)"
Q = rf'u\({F}, {F}\) r\({F}, {F}\) c=\({F}, {F}\) w={F} h={F}'


def entries(p):
    a = MPQArchive(p)
    C = sc2map.decode_cliff(a.read_file("t3SyncCliffLevel"))[3].astype(int)
    x = a.read_file("t3Terrain.xml").decode("utf-8", "replace")
    out = []
    for m in re.finditer(r'<ramp dir="(\d+)"[^>]*?base="' + Q + r'" mid="' + Q + '"', x):
        g = [float(v) for v in m.groups()[1:]]
        out.append(dict(dir=int(m.group(1)), u=(g[0], g[1]), base=(g[4], g[5]), bw=g[6], bh=g[7],
                        mid=(g[12], g[13]), mw=g[14], mh=g[15]))
    return C, out


def at(C, x, y):
    X, Y = int(np.floor(x)), int(np.floor(y))
    return int(C[Y, X]) if 0 <= Y < C.shape[0] and 0 <= X < C.shape[1] else -1


paths = sys.argv[1:] or sorted(glob.glob("gold_maps/*.SC2Map"))
offs, profiles, params = [], collections.Counter(), collections.Counter()
for p in paths:
    try:
        C, es = entries(p)
    except Exception:  # noqa: BLE001
        continue
    sub = (C > 0) & (C % 64 != 0)
    for e in es:
        if e["dir"] >= 4:
            continue
        u = e["u"]
        bx, by = e["base"]
        # profile: CLIF sampled at base - k*u (k = -2..8)
        prof = tuple(at(C, bx - u[0] * k, by - u[1] * k) % 64 if at(C, bx - u[0] * k, by - u[1] * k) > 0
                     else -1 for k in range(-2, 9))
        profiles[prof] += 1
        run = ((bx - e["mid"][0]) ** 2 + (by - e["mid"][1]) ** 2) ** 0.5
        params[(round(e["bw"], 2), round(e["bh"], 2), round(e["mw"], 2), round(e["mh"], 2),
                round(run, 2))] += 1
        # top-row: highest sub-level among sloped cells within 6 cells of base
        ys, xs = np.nonzero(sub)
        near = (xs + 0.5 - bx) ** 2 + (ys + 0.5 - by) ** 2 < 36
        if not near.any():
            continue
        vals = C[ys[near], xs[near]] % 64
        topv = vals.max()
        tx, ty = xs[near][vals == topv] + 0.5, ys[near][vals == topv] + 0.5
        r = (-u[1], u[0])
        dx, dy = bx - tx.mean(), by - ty.mean()
        offs.append((dx * u[0] + dy * u[1], dx * r[0] + dy * r[1], int(topv), len(tx)))

a = np.array(offs) if offs else np.zeros((0, 4))
print(f"cardinal ramps: {len(a)}")
if len(a):
    for k, n in ((0, "along u"), (1, "sideways")):
        print(f"  base offset from top-row centroid {n:9s}: median {np.median(a[:, k]):5.2f} "
              f"p10 {np.percentile(a[:, k], 10):5.2f} p90 {np.percentile(a[:, k], 90):5.2f}")
    print("  top sub-level (cliff % 64) counts:", collections.Counter(a[:, 2].astype(int).tolist()))
    print("  top-row length median:", np.median(a[:, 3]))
print("quad params (base.w, base.h, mid.w, mid.h, run) top:", params.most_common(6))
print("CLIF%64 profile from base (k=-2..8 cells downhill; -1 void, 0 plateau) top:")
for prof, n in profiles.most_common(8):
    print(f"  {n:4d}  {prof}")
