"""Read an EXPORTED .SC2Map's CLIF grid and compute the engine's terrain connectivity on it.

Engine rule (findings S4.4): two adjacent non-void cells are ground-traversable iff their cliff
values differ by <= 8 (a bigger jump is an impassable cliff wall; an 8-step is a walkable ramp
gradation). We label cells into <=8-step components (8-connected) and report whether the two start
locations land in one component -- i.e. is the TERRAIN itself connected main-to-main, independent of
the <ramp> quad coverage. If terrain is connected here but the engine says mains are NOT connected,
the break is the quad; if terrain is disconnected here, the break is the export gradient/channeling.

    PYTHONPATH=src .venv/bin/python scripts/export_conn.py gen_234
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from mpyq import MPQArchive
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.export import sc2map  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "gen_234"
MAP = f"outputs/export/{NAME}.SC2Map"


def _starts(arch):
    x = arch.read_file("Objects").decode("utf-8", "replace")
    pts = []
    for m in re.finditer(r'Type="StartLoc"[^>]*Position="([^"]+)"', x):
        p = m.group(1).split(",")
        pts.append((int(float(p[0])), int(float(p[1]))))
    for m in re.finditer(r'Position="([^"]+)"[^>]*Type="StartLoc"', x):
        p = m.group(1).split(",")
        pts.append((int(float(p[0])), int(float(p[1]))))
    return pts


def step_components(cliff: np.ndarray, max_step: int = 8) -> np.ndarray:
    """8-connected components where an edge exists iff both cells non-void and |Δcliff|<=max_step."""
    h, w = cliff.shape
    nz = cliff > 0
    idx = -np.ones((h, w), dtype=np.int64)
    ids = np.where(nz.ravel())[0]
    idx.ravel()[ids] = np.arange(len(ids))
    parent = np.arange(len(ids))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    c = cliff.astype(np.int32)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = nz & np.roll(np.roll(nz, dy, 0), dx, 1)
        step = np.abs(c - np.roll(np.roll(c, dy, 0), dx, 1)) <= max_step
        m = a & step
        # avoid wrap edges
        if dy == 1:
            m[0, :] = False
        if dx == 1:
            m[:, 0] = False
        if dx == -1:
            m[:, -1] = False
        ys, xs = np.where(m)
        for y, x in zip(ys.tolist(), xs.tolist()):
            py, px = (y - dy) % h, (x - dx) % w
            parent[find(idx[y, x])] = find(idx[py, px])
    out = -np.ones((h, w), dtype=np.int64)
    roots = np.array([find(i) for i in range(len(ids))])
    out.ravel()[ids] = roots
    return out


def main() -> None:
    arch = MPQArchive(MAP)
    _v, cw, ch, cliff = sc2map.decode_cliff(arch.read_file("t3SyncCliffLevel"))
    cliff = cliff.astype(np.int32)
    starts = _starts(arch)
    comp = step_components(cliff)
    nz = cliff > 0
    ncomp = len({int(v) for v in comp[nz]})
    print(f"{NAME}: grid={cw}x{ch} nonvoid_cells={int(nz.sum())} <=8-step_components={ncomp}")
    print(f"cliff values present: {sorted({int(v) for v in np.unique(cliff)})}")
    if len(starts) >= 2:
        (x0, y0), (x1, y1) = starts[0], starts[1]
        c0, c1 = int(comp[y0, x0]), int(comp[y1, x1])
        print(f"start0=({x0},{y0}) comp={c0}  start1=({x1},{y1}) comp={c1}  "
              f"MAIN_TO_MAIN_TERRAIN_CONNECTED={c0 == c1 and c0 >= 0}")
    # biggest components
    labs, counts = np.unique(comp[nz], return_counts=True)
    order = np.argsort(-counts)[:6]
    print("top components (id:cells):", [(int(labs[o]), int(counts[o])) for o in order])


if __name__ == "__main__":
    main()
