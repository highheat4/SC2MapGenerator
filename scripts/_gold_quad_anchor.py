"""Where do gold DIAGONAL rampList quads put `base`, relative to the slope's top row (hi-16 cells)?

Reports the base's offset from the centroid of the nearest top-row (cliff % 64 == 48) component,
split into along-uphill (u) and sideways (r) components, across the gold corpus.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _quad_vs_slope import ramps  # noqa: E402

rows = []
for p in sorted(glob.glob("gold_maps/*.SC2Map")):
    try:
        C, rs = ramps(p)
    except Exception:  # noqa: BLE001
        continue
    top = (C > 0) & (C % 64 == 48)
    lab, _ = ndimage.label(top, structure=np.ones((3, 3)))
    ys, xs = np.nonzero(top)
    if not len(xs):
        continue
    for d, u, base, _mid in rs:
        if d < 4:
            continue
        i = np.argmin((xs + 0.5 - base[0]) ** 2 + (ys + 0.5 - base[1]) ** 2)
        cy, cx = np.nonzero(lab == lab[ys[i], xs[i]])
        c = (cx.mean() + 0.5, cy.mean() + 0.5)
        r = (-u[1], u[0])
        dx, dy = base[0] - c[0], base[1] - c[1]
        rows.append((dx * u[0] + dy * u[1], dx * r[0] + dy * r[1]))

a = np.array(rows)
print("gold diagonal ramps:", len(a))
for k, name in ((0, "along uphill u"), (1, "sideways r")):
    print(f"  {name:15s} median {np.median(a[:, k]):6.2f}   p10 {np.percentile(a[:, k], 10):6.2f}"
          f"   p90 {np.percentile(a[:, k], 90):6.2f}")
