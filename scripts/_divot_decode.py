"""Decode the EXPORTED gen_24.SC2Map and print CLIF + SMAP(height) + HMAP(base) windows over the
top-right ramp, so the 'divots' are visible as actual rendered-height numbers.

IR ramp window (from _divot_probe): x 121..130, y 47..56. Template offset was (1,0), so template
X = ir_x + 1, Y = ir_y.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MAP = sys.argv[1] if len(sys.argv) > 1 else "outputs/export/gen_24.SC2Map"
OFF_X, OFF_Y = 1, 0

a = MPQArchive(MAP)
names = {}
for n in a.files:
    key = n.decode() if isinstance(n, bytes) else n
    names[key.split("\\")[-1].lower()] = n


def read(short):
    return a.read_file(names[short.lower()])


clif = read("t3SyncCliffLevel")
smap = read("t3SyncHeightMap")
hmap = read("t3HeightMap")

# CLIF: 32-byte header, u16[sy][sx]
csx, csy = struct.unpack_from("<II", clif, 8)
C = np.frombuffer(clif[32:32 + csx * csy * 2], dtype="<u2").reshape(csy, csx).astype(int)

# SMAP: 64-byte header, per-cell {int16 height; uint16 mask}, VERTEX grid (cells+1)
ssx, ssy = struct.unpack_from("<II", smap, 8)
sbody = np.frombuffer(smap[64:64 + ssx * ssy * 4], dtype="<u2").reshape(ssy, ssx, 2)
S_h = sbody[..., 0].astype(np.int16).astype(float) / 256.0   # gameplay height in "levels*?" units

# HMAP: 32-byte header, per-cell {u16 adjust; u16 base; u16 mask}, VERTEX grid
hsx, hsy = struct.unpack_from("<II", hmap, 8)
hbody = np.frombuffer(hmap[32:32 + hsx * hsy * 6], dtype="<u2").reshape(hsy, hsx, 3)
H_adj = hbody[..., 0].astype(int)
H_base = hbody[..., 1].astype(int)

print(f"CLIF {csx}x{csy}   SMAP {ssx}x{ssy}   HMAP {hsx}x{hsy}")

# window in TEMPLATE cell coords
x0, x1 = 121 + OFF_X - 2, 130 + OFF_X + 3
y0, y1 = 47 + OFF_Y - 2, 56 + OFF_Y + 3


def show(name, G, fmt):
    print(f"\n=== {name} (template cells x[{x0}:{x1}] y[{y0}:{y1}]) ===")
    hdr = "      " + " ".join(f"{x:>6d}" for x in range(x0, x1))
    print(hdr)
    for y in range(y0, y1):
        row = " ".join(fmt(G[y, x]) for x in range(x0, x1))
        print(f"y{y:3d}: {row}")


show("CLIF (0=void)", C, lambda v: f"{int(v):6d}")
# SMAP/HMAP are vertex grids; a cell (x,y) is bounded by verts (x,y),(x+1,y),(x,y+1),(x+1,y+1).
# Print the vertex value at (x,y) which is the top-left corner of the cell.
show("SMAP height (gameplay, /256)", S_h, lambda v: f"{v:6.2f}")
show("HMAP heightBase (render)", H_base, lambda v: f"{int(v):6d}")
show("HMAP heightAdjust (render dents)", H_adj, lambda v: f"{int(v):6d}")
