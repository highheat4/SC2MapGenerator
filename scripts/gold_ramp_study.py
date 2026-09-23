"""Measure gold-standard ramp geometry to derive the traversable-ramp invariants (findings §4.7).

Two data sources:
  A) gold_maps/*.SC2Map      -> decoded t3SyncCliffLevel + <rampList> (authored ground truth)
  B) dataset/*/_raw.npz      -> engine pathing/placement/height grids (walkable ground truth)

Reports, across maps:
  - CLIF vocabulary (plateaus 64/128/192; sub-levels 72..112, skip 120; +16 top step)
  - per-axis-cell step (8 CLIF == 2 engine-height units)
  - authored band RUN (fixed, 6 sub-levels/level) vs WIDTH (free, ~5..15)
  - engine WALKABLE run vs width
  - rampList quad width/run (tiny: width 1..5, run 2) + Pnp openness

Usage: PYTHONPATH=src python scripts/gold_ramp_study.py
"""
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from mpyq import MPQArchive
from scipy import ndimage
from scipy.ndimage import maximum_filter, minimum_filter

from sc2mapgen.export import sc2map

GOLD = ("AcropolisAIE", "AutomatonAIE", "FrostLE")


def _pca_dims(ys, xs):
    """Return (major, minor) extents (major = along isolines = WIDTH, minor = perp = RUN)."""
    pts = np.stack([xs, ys], 1).astype(float)
    pts -= pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    proj = pts @ vt.T
    return np.ptp(proj[:, 0]) + 1, np.ptp(proj[:, 1]) + 1


def study_authored():
    print("=== A) AUTHORED CLIF band + rampList (gold_maps/*.SC2Map) ===")
    for name in GOLD:
        a = MPQArchive(f"gold_maps/{name}.SC2Map")
        _, cw, ch, cliff = sc2map.decode_cliff(a.read_file("t3SyncCliffLevel"))
        inter = (cliff % 64 != 0) & (cliff > 0)
        vocab = sorted(np.unique(cliff[inter]).tolist())
        lbl, n = ndimage.label(inter, structure=np.ones((3, 3)))
        widths, runs, nsub = [], [], []
        for i in range(1, n + 1):
            ys, xs = np.where(lbl == i)
            if len(xs) < 4:
                continue
            mj, mn = _pca_dims(ys, xs)
            widths.append(mj)
            runs.append(mn)
            nsub.append(len(np.unique(cliff[ys, xs])))
        widths, runs = np.array(widths), np.array(runs)
        # rampList quad width/run
        xml = a.read_file("t3Terrain.xml").decode("utf-8", "replace")
        qw, qr, dirs, levels = [], [], set(), set()
        for r in re.findall(r"<ramp\b[^>]*/>", xml):
            m = re.search(r'base="u\([^)]*\) r\([^)]*\) c=\([^)]*\) w=([-0-9.eE+]+)', r)
            b = re.search(r'base=".*?c=\(([^)]*)\)', r)
            mid = re.search(r'mid=".*?c=\(([^)]*)\)', r)
            if m:
                qw.append(float(m.group(1)))
            if b and mid:
                bx, by = map(float, b.group(1).split(","))
                mx, my = map(float, mid.group(1).split(","))
                qr.append(((bx - mx) ** 2 + (by - my) ** 2) ** 0.5)
            dd = re.search(r'dir="(\d+)"', r)
            lo = re.search(r'lo="(\d+)"', r)
            hi = re.search(r'hi="(\d+)"', r)
            if dd:
                dirs.add(int(dd.group(1)))
            if lo and hi:
                levels.add(int(hi.group(1)) - int(lo.group(1)))
        print(f"\n  {name}: {n} bands")
        print(f"     CLIF sub-levels: {vocab}")
        print(f"     WIDTH(isoline) med={np.median(widths):.1f} range[{widths.min():.1f},{widths.max():.1f}] std={widths.std():.1f}")
        print(f"     RUN(perp)      med={np.median(runs):.1f} range[{runs.min():.1f},{runs.max():.1f}] std={runs.std():.1f}")
        print(f"     #sub-levels per ramp: {sorted(set(nsub))}   dirs used: {sorted(dirs)}   level jumps: {sorted(levels)}")
        print(f"     quad width(base.w): {min(qw):.1f}..{max(qw):.1f}   quad run: {min(qr):.1f}..{max(qr):.1f}")


def study_walkable():
    print("\n=== B) ENGINE WALKABLE ramps (dataset/*/_raw.npz) ===")
    W, R, steps = [], [], []
    nmaps = 0
    for p in glob.glob("dataset/*/_raw.npz"):
        if any(s in p for s in ("Flat", "Empty", "Simple")):
            continue
        try:
            z = np.load(p)
        except Exception:
            continue
        P = z["pathing"] > 0
        H = z["height"].astype(int)
        lo = minimum_filter(H, 3)
        hi = maximum_filter(H, 3)
        slope = P & ((hi - lo) > 0) & ((hi - lo) <= 12)
        if slope.sum() < 50:
            continue
        nmaps += 1
        for dy, dx in ((0, 1), (1, 0)):
            m = slope & np.roll(np.roll(slope, dy, 0), dx, 1)
            d = np.abs(H[m] - np.roll(np.roll(H, dy, 0), dx, 1)[m])
            steps += d[d > 0].tolist()
        lbl, n = ndimage.label(slope, structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            ys, xs = np.where(lbl == i)
            if len(xs) < 8:
                continue
            mj, mn = _pca_dims(ys, xs)
            if mj >= 4 and mn >= 2 and mj / mn < 8:
                W.append(mj)
                R.append(mn)
    W, R, steps = np.array(W), np.array(R), np.array(steps)
    import collections
    c = collections.Counter(steps.tolist())
    print(f"  maps={nmaps}  ramps={len(W)}")
    print(f"  per-axis-cell height step: " + ", ".join(f"{k}:{100*c[k]/len(steps):.0f}%" for k in sorted(c)[:5]))
    print(f"  WIDTH med={np.median(W):.1f} std={W.std():.1f} p10/p90={np.percentile(W,10):.1f}/{np.percentile(W,90):.1f}")
    print(f"  RUN   med={np.median(R):.1f} std={R.std():.1f} p10/p90={np.percentile(R,10):.1f}/{np.percentile(R,90):.1f}")
    print(f"  CoV width={W.std()/W.mean():.2f} run={R.std()/R.mean():.2f}  corr(w,r)={np.corrcoef(W,R)[0,1]:.2f}")


if __name__ == "__main__":
    study_authored()
    study_walkable()
