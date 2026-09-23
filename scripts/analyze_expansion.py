"""Replicate python-sc2's expansion finder on the exported map + engine grids to find exactly
which resource cluster crashes (min() empty) and why (height-split cluster vs no buildable ring).
"""
from __future__ import annotations

import itertools
import math
import re
import sys
from pathlib import Path

import numpy as np
from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

place = np.load("outputs/export/eng_place.npy")     # [y_up][x], 1==buildable
height = np.load("outputs/export/eng_height.npy")   # [y_up][x] byte
H, W = place.shape


def hgt(x, y):
    return int(height[int(y), int(x)])


def buildable(x, y):
    xi, yi = int(x), int(y)
    return 0 <= xi < W and 0 <= yi < H and place[yi, xi] == 1


# parse resources (world coords) from the exported Objects XML
_MAP = sys.argv[1] if len(sys.argv) > 1 else "gen_213"
xml = MPQArchive(f"outputs/export/{_MAP}.SC2Map").read_file("Objects").decode()
res = []  # (x, y, is_geyser)
for m in re.finditer(r'Position="([0-9.]+),([0-9.]+),0"[^>]*UnitType="(MineralField|VespeneGeyser)"', xml):
    res.append((float(m.group(1)), float(m.group(2)), m.group(3) == "VespeneGeyser"))
print(f"resources parsed: {len(res)}  (geysers={sum(r[2] for r in res)})")

# cluster: start each resource its own group, merge if center dist<=10.5 AND all pairwise height<=10
groups = [[r] for r in res]


def center(g):
    return (sum(p[0] for p in g) / len(g), sum(p[1] for p in g) / len(g))


merged = True
while merged:
    merged = False
    for a, b in itertools.combinations(groups, 2):
        ca, cb = center(a), center(b)
        if math.hypot(ca[0] - cb[0], ca[1] - cb[1]) <= 10.5 and all(
                abs(hgt(ra[0], ra[1]) - hgt(rb[0], rb[1])) <= 10 for ra in a for rb in b):
            groups.remove(a); groups.remove(b); groups.append(a + b); merged = True
            break

offsets = [(x, y) for x, y in itertools.product(range(-7, 8), repeat=2) if 4 < math.hypot(x, y) <= 8]
print(f"clusters: {len(groups)}")
fails = 0
for g in sorted(groups, key=lambda g: center(g)[1]):
    cx = int(sum(p[0] for p in g) / len(g)) + 0.5
    cy = int(sum(p[1] for p in g) / len(g)) + 0.5
    ngey = sum(p[2] for p in g)
    pts = []
    for ox, oy in offsets:
        px, py = ox + cx, oy + cy
        if buildable(px, py) and all(
                math.hypot(px - r[0], py - r[1]) >= (7 if r[2] else 6) for r in g):
            pts.append((px, py))
    tag = "OK  " if pts else "FAIL"
    if not pts:
        fails += 1
        nb = sum(1 for ox, oy in offsets if buildable(ox + cx, oy + cy))
        print(f"  {tag} cluster n={len(g)} gey={ngey} center=({cx},{cy}) "
              f"buildable-in-ring={nb} heights={sorted({hgt(p[0], p[1]) for p in g})}")
print(f"\n{fails} FAILING clusters -> these cause the min() empty crash")
