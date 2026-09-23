"""Render the exported seed-24 ramp region as (a) a CLIF map showing void seams and (b) a hillshade
of the SMAP/HMAP rendered-height surface, to visually locate the 'divots'."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpyq import MPQArchive

MAP = "outputs/export/gen_24.SC2Map"
a = MPQArchive(MAP)
names = {}
for n in a.files:
    key = n.decode() if isinstance(n, bytes) else n
    names[key.split("\\")[-1].lower()] = n
read = lambda s: a.read_file(names[s.lower()])

clif = read("t3SyncCliffLevel")
smap = read("t3SyncHeightMap")
csx, csy = struct.unpack_from("<II", clif, 8)
C = np.frombuffer(clif[32:32 + csx * csy * 2], dtype="<u2").reshape(csy, csx).astype(int)
ssx, ssy = struct.unpack_from("<II", smap, 8)
S = np.frombuffer(smap[64:64 + ssx * ssy * 4], dtype="<u2").reshape(ssy, ssx, 2)[..., 0]
S = S.astype(np.int16).astype(float) / 256.0

# broad window around the top-right ramp (template cells)
x0, x1, y0, y1 = 112, 145, 38, 68
Cw = C[y0:y1, x0:x1]
Sw = S[y0:y1, x0:x1]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
im0 = ax[0].imshow(Cw, cmap="viridis", interpolation="nearest",
                   extent=[x0, x1, y1, y0])
ax[0].set_title("t3SyncCliffLevel (CLIF)\n0=void(dark), 64=low, 128=high")
fig.colorbar(im0, ax=ax[0], fraction=0.046)
# mark void cells
vy, vx = np.where(Cw == 0)
ax[0].scatter(vx + x0 + 0.5, vy + y0 + 0.5, s=14, c="red", marker="s", label="void (CLIF=0)")
ax[0].legend(loc="upper left")

ls = LightSource(azdeg=315, altdeg=30)
shade = ls.hillshade(Sw, vert_exag=6.0, dx=1, dy=1)
ax[1].imshow(shade, cmap="gray", interpolation="bicubic", extent=[x0, x1, y1, y0])
c = ax[1].contour(np.linspace(x0, x1, Sw.shape[1]), np.linspace(y0, y1, Sw.shape[0]),
                  Sw, levels=12, colors="cyan", linewidths=0.4, alpha=0.7)
ax[1].set_title("SMAP rendered-height hillshade\n(what the player sees as relief)")

plt.tight_layout()
out = "outputs/export/_divot_seed24.png"
plt.savefig(out, dpi=120)
print("wrote", out)
