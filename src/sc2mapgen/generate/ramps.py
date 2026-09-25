"""Ramp geometry, and the ramp plan the skeleton hands the rasterizer.

Every level change is one straight staircase cut through the cliff wall between two terraces
(findings §3-4). This module owns what such a cut looks like (``paint_planned_ramp``), when it is
well formed (``ramp_shape_problems``, ``is_gold_main_ramp``, ``ramp_plateau_contacts``), and where
the skeleton puts them (``plan_ramps``). Planning paints the same pre-cut ground the rasterizer
paints (``paint_ground``: flat lanes, rooms, pads, base cores + rings, every seam walled), then
decides each cut there, so the rasterizer replays the plan instead of searching.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from sc2mapgen.generate.footprint import is_nat_out, node_centre, painted_width
from sc2mapgen.ir import (
    BaseKind, _lattice_dir, cardinal_profile, ramp_cell_cliffs, ramp_span, snap_uphill,
)

_REAL = (BaseKind.MAIN, BaseKind.NATURAL, BaseKind.BASE)
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _paint_box(walk, elev, ramp, center, rx, ry, angle, level, canvas) -> None:
    """Paint a flat walkable rectangular room (rotated by ``angle``); clears ramp under it."""
    cx, cy = center
    rad = math.hypot(rx, ry)
    h, w = walk.shape
    xmin = max(0, int(np.floor(cx - rad - 1)))
    xmax = min(w - 1, int(np.ceil(cx + rad + 1)))
    ymin = max(0, int(np.floor(cy - rad - 1)))
    ymax = min(h - 1, int(np.ceil(cy + rad + 1)))
    if xmin > xmax or ymin > ymax:
        return
    ys, xs = np.mgrid[ymin : ymax + 1, xmin : xmax + 1].astype(float)
    ca, sa = math.cos(angle), math.sin(angle)
    dx, dy = xs - cx, ys - cy
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    m = (np.abs(u) <= rx) & (np.abs(v) <= ry) & canvas[ymin : ymax + 1, xmin : xmax + 1]
    walk[ymin : ymax + 1, xmin : xmax + 1][m] = True
    elev[ymin : ymax + 1, xmin : xmax + 1][m] = level
    ramp[ymin : ymax + 1, xmin : xmax + 1][m] = False


def _paint_corridor(walk, elev, ramp, p0, p1, width, l0, l1, canvas,
                    ramp_t0: float = 0.0, ramp_t1: float = 1.0, passage=None) -> None:
    """Paint a width-thick *rectangular* (flat-ended) lane from p0 to p1.

    The lane is flat at ``l0`` for the parametric span ``t < ramp_t0`` (extending land
    mass 0), transitions l0->l1 as a RAMP within ``[ramp_t0, ramp_t1]``, then is flat at
    ``l1`` for ``t > ramp_t1`` (extending land mass 1). Keeping the ramp band short (and
    inside the gap between the two rooms) makes ramps read as a simple interface between
    two plateaus instead of a long tube from one room's core to another's. If the ends
    share a level, the lane is uniformly flat (no ramp)."""
    (x0, y0), (x1, y1) = p0, p1
    r = width / 2.0
    h, w = walk.shape
    xmin = max(0, int(np.floor(min(x0, x1) - r - 1)))
    xmax = min(w - 1, int(np.ceil(max(x0, x1) + r + 1)))
    ymin = max(0, int(np.floor(min(y0, y1) - r - 1)))
    ymax = min(h - 1, int(np.ceil(max(y0, y1) + r + 1)))
    if xmin > xmax or ymin > ymax:
        return
    ys, xs = np.mgrid[ymin : ymax + 1, xmin : xmax + 1].astype(float)
    dx, dy = x1 - x0, y1 - y0
    seg = dx * dx + dy * dy or 1.0
    slen = math.sqrt(seg)
    t = ((xs - x0) * dx + (ys - y0) * dy) / seg          # projection param along segment
    perp = np.abs((xs - x0) * (-dy) + (ys - y0) * dx) / slen  # perpendicular distance
    m = (t >= 0.0) & (t <= 1.0) & (perp <= r) & canvas[ymin : ymax + 1, xmin : xmax + 1]
    walk[ymin : ymax + 1, xmin : xmax + 1][m] = True
    # A flat lane (l0==l1) is a same-level PASSAGE: record it so cliff_cleanup protects it from
    # being pinched off where it routes past a different-level plateau (it walls the plateau side
    # instead). Ramp bands are protected separately via the ``ramp`` mask.
    if passage is not None and l0 == l1:
        passage[ymin : ymax + 1, xmin : xmax + 1][m] = True
    # piecewise height: flat l0 -> short ramp -> flat l1
    span = max(ramp_t1 - ramp_t0, 1e-6)
    f = np.clip((t - ramp_t0) / span, 0.0, 1.0)
    ev = np.rint(l0 + f * (l1 - l0)).astype(elev.dtype)
    elev[ymin : ymax + 1, xmin : xmax + 1][m] = ev[m]
    if l0 != l1:
        band = m & (t >= ramp_t0) & (t <= ramp_t1)
        ramp[ymin : ymax + 1, xmin : xmax + 1][band] = True



# 8 ramp directions -- MUST match the exporter's _RAMP_DIR_U (sc2map.py) so the straight ramp we
# rasterize is aligned to the same axis the exporter snaps its gradient + <ramp> quad to.
_S2 = 0.7071067811865476
_DIRS8 = (
    (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0),          # N, E, S, W (cardinal)
    (-_S2, -_S2), (_S2, -_S2), (-_S2, _S2), (_S2, _S2),        # diagonals
)


def _snap8(ux: float, uy: float) -> tuple[float, float]:
    """Snap a unit vector to the nearest of the 8 ramp directions (cardinal + diagonal)."""
    best, bestdot = _DIRS8[1], -2.0
    for vx, vy in _DIRS8:
        dot = ux * vx + uy * vy
        if dot > bestdot:
            best, bestdot = (vx, vy), dot
    return best



def _diff_boundary(elev: np.ndarray, walk: np.ndarray) -> np.ndarray:
    """Walkable cells adjacent (4-neigh) to a walkable cell of a different elevation."""
    diff = np.zeros_like(walk, dtype=bool)
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        diff |= walk & np.roll(walk, sh, axis=ax) & (elev != np.roll(elev, sh, axis=ax))
    return diff



def _higher_boundary(elev: np.ndarray, walk: np.ndarray) -> np.ndarray:
    """Walkable cells strictly HIGHER than some walkable 4-neighbour -- i.e. the top edge of a
    cliff. Walling only these (not the lower side) turns a level seam into a clean cliff while
    keeping the lower floor continuous, so low passages aren't pinched off."""
    hi = np.zeros_like(walk, dtype=bool)
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        hi |= walk & np.roll(walk, sh, axis=ax) & (elev > np.roll(elev, sh, axis=ax))
    return hi



def ramp_plateau_contacts(comp, walk, elev, ramp):
    """Plateau levels one ramp component touches, the number of distinct plateau components at
    each, and whether that is exactly one low and one high plateau a level apart."""
    plat = walk & ~ramp
    nb = ndimage.binary_dilation(comp) & plat
    levels = sorted({int(v) for v in elev[nb]}) if nb.any() else []
    counts: dict[int, int] = {}
    for L in levels:
        slab, _ = ndimage.label(plat & (elev == L), structure=_STRUCT8)
        counts[L] = len({int(v) for v in slab[nb & (elev == L)]} - {0})
    clean = (
        len(levels) == 2
        and levels[1] - levels[0] == 1
        and counts.get(levels[0]) == 1
        and counts.get(levels[1]) == 1
    )
    return levels, counts, clean


RAMP_MIN_STRAIGHT_WIDTH = {"diagonal": 2, "cardinal": 3}


# Every gold main (94/94 single-ramp 1v1 maps) enters through the same diagonal staircase: 8 lattice
# ranks along the uphill axis (plateau, 6 sloped isolines, plateau) and 2 lattice steps either side
# of it, so the sloped isolines hold 3,2,3,2,3,2 cells from the bottom up.
GOLD_MAIN_RAMP_RANKS = 8
GOLD_MAIN_RAMP_HALF = 2
GOLD_MAIN_RAMP_ISOLINES = (3, 2, 3, 2, 3, 2)


def _stamp_gold_main_ramp(walk, elev, ramp, canvas, centre, u, lo: int, hi: int) -> None:
    ix, iy = _lattice_dir(u[0], u[1])
    lc = centre[0] * ix + centre[1] * iy
    pc = centre[0] * -iy + centre[1] * ix
    l0 = int(round(lc - (GOLD_MAIN_RAMP_RANKS - 1) / 2.0))
    p0 = int(round(pc))
    if (p0 - (l0 + 1)) % 2:
        p0 += 1
    h, w = walk.shape
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    r = GOLD_MAIN_RAMP_RANKS
    for y in range(max(0, cy - r), min(h, cy + r + 1)):
        for x in range(max(0, cx - r), min(w, cx + r + 1)):
            k = x * ix + y * iy - l0
            if 0 <= k < GOLD_MAIN_RAMP_RANKS and abs(x * -iy + y * ix - p0) <= GOLD_MAIN_RAMP_HALF \
                    and canvas[y, x]:
                walk[y, x] = True
                ramp[y, x] = True
                elev[y, x] = lo if k < GOLD_MAIN_RAMP_RANKS // 2 else hi


def ramp_axis(comp, walk, elev, ramp):
    """``(bottom, top, lo_level, hi_level)`` of a ramp component as the exporter will see it, or
    None if it touches only one level.

    A clean ramp is painted at ONE elevation but *bridges* two plateaus, so its span is the set of
    plateau levels it touches, not its own cells' elevation; bottom/top are the centroids of its
    low / high plateau neighbours, which set the direction the exporter climbs."""
    nb = ndimage.binary_dilation(comp, structure=np.ones((3, 3), bool)) & walk & ~ramp
    levels = sorted({int(v) for v in elev[nb]}) if nb.any() else []
    if len(levels) <= 1:
        return None
    lo, hi = levels[0], levels[-1]
    lys, lxs = np.where(nb & (elev == lo))
    hys, hxs = np.where(nb & (elev == hi))
    return (float(lxs.mean()), float(lys.mean())), (float(hxs.mean()), float(hys.mean())), lo, hi


def is_gold_main_ramp(cells, ux: float, uy: float, lo_level: int, hi_level: int) -> bool:
    """True iff a ramp's sloped cells form exactly the gold main staircase."""
    if snap_uphill(ux, uy)[0] < 4:
        return False
    lo, hi = (lo_level + 1) * 64, (hi_level + 1) * 64
    per: dict[int, int] = {}
    for c in ramp_cell_cliffs(list(cells), ux, uy, lo, hi):
        if lo < c < hi:
            per[c] = per.get(c, 0) + 1
    return tuple(per[k] for k in sorted(per)) == GOLD_MAIN_RAMP_ISOLINES


def ramp_shape_problems(cells, ux: float, uy: float, lo_level: int, hi_level: int) -> list[str]:
    """Why a ramp's cells are not a gold-shaped straight band (empty == fine).

    The engine only detects a ramp whose sloped cells (strictly between the two plateau cliffs,
    per the exporter's own ``ramp_cell_cliffs``) form every isoline of the gradient -- 6 for a
    diagonal, 3 for a cardinal -- and share one straight strip across all of them. A band clipped
    by a base core or bent round a corner fails the strip test even when its cliff steps are legal.
    Strip width is counted in cells along the isolines (lattice units for a diagonal)."""
    lo, hi = (lo_level + 1) * 64, (hi_level + 1) * 64
    diagonal = snap_uphill(ux, uy)[0] >= 4
    ix, iy = _lattice_dir(ux, uy)
    iso: dict[int, list[int]] = {}
    for (x, y), c in zip(cells, ramp_cell_cliffs(list(cells), ux, uy, lo, hi)):
        if lo < c < hi:
            iso.setdefault(x * ix + y * iy, []).append(x * -iy + y * ix)
    want = ramp_span(lo, hi) - 1 if diagonal else len(cardinal_profile(lo, hi)) - 2
    if len(iso) != want:
        return [f"{len(iso)} sloped isolines (want {want})"]
    a = max(min(v) for v in iso.values())
    b = min(max(v) for v in iso.values())
    step = 2 if diagonal else 1
    width = (b - a) // step + 1 if b >= a else 0
    need = RAMP_MIN_STRAIGHT_WIDTH["diagonal" if diagonal else "cardinal"]
    if width < need:
        return [f"straight width {width} < {need}"]
    return []


# --------------------------------------------------------------------------- #
# symmetry
# --------------------------------------------------------------------------- #
def mirror_point(p, sym: str, centre) -> tuple[float, float]:
    (x, y), (cx, cy) = p, centre
    if sym == "mirror_lr":
        return (2 * cx - x, y)
    if sym == "mirror_ud":
        return (x, 2 * cy - y)
    return (2 * cx - x, 2 * cy - y)


def _partner_cells(ys, xs, pa, sym: str):
    """Integer mirror partner of cells: an exact involution about the doubled playable centre
    ``(2*pa.x + pa.width, 2*pa.y + pa.height)`` the skeleton mirrors about."""
    msx, msy = 2 * pa.x + pa.width, 2 * pa.y + pa.height
    if sym == "mirror_lr":
        return ys, msx - xs
    if sym == "mirror_ud":
        return msy - ys, xs
    return msy - ys, msx - xs


def symmetrize(grids, pa, sym: str, only=None) -> None:
    """Make ``grids`` pixel-exact mirrors in place. Each cell pair takes the value of its lower
    raster-index cell; with ``only``, just the pairs touching a cell of that mask, taking the value
    from the cell inside it (so a change painted on one side is copied onto the other)."""
    h, w = grids[0].shape
    if only is None:
        ys, xs = np.mgrid[0:h, 0:w]
        ys, xs = ys.ravel(), xs.ravel()
    else:
        ys, xs = np.nonzero(only)
    my, mx = _partner_cells(ys, xs, pa, sym)
    inb = (mx >= 0) & (mx < w) & (my >= 0) & (my < h)
    ys, xs, my, mx = ys[inb], xs[inb], my[inb], mx[inb]
    lower = my * w + mx < ys * w + xs
    if only is None:
        sel = ~lower
    else:
        sel = ~(only[my, mx] & lower)
    for g in grids:
        g[my[sel], mx[sel]] = g[ys[sel], xs[sel]]


def mirror_dir(c, u, sym: str, centre) -> tuple[float, float]:
    a = mirror_point(c, sym, centre)
    b = mirror_point((c[0] + u[0], c[1] + u[1]), sym, centre)
    return (b[0] - a[0], b[1] - a[1])


# --------------------------------------------------------------------------- #
# a planned staircase
# --------------------------------------------------------------------------- #
@dataclass
class PlannedRamp:
    """One straight staircase: a ``run``-long, ``width``-wide band centred on ``centre`` climbing
    along ``u`` from level ``lo`` to ``hi``, tied back to its two nodes by flat approach lanes that
    start at ``p_lo`` / ``p_hi``. A gold main ramp is the fixed main stamp instead of a band."""
    lo_node: int
    hi_node: int
    lo: int
    hi: int
    p_lo: tuple[float, float]
    p_hi: tuple[float, float]
    centre: tuple[float, float]
    u: tuple[float, float]
    width: float
    run: float
    gold_main: bool = False
    nat_out: bool = False

    def mirrored(self, mirror: list[int], sym: str, centre) -> "PlannedRamp":
        return PlannedRamp(
            mirror[self.lo_node], mirror[self.hi_node], self.lo, self.hi,
            mirror_point(self.p_lo, sym, centre), mirror_point(self.p_hi, sym, centre),
            mirror_point(self.centre, sym, centre), mirror_dir(self.centre, self.u, sym, centre),
            self.width, self.run, self.gold_main, self.nat_out)

    def to_json_dict(self) -> dict:
        return {"lo_node": self.lo_node, "hi_node": self.hi_node, "lo": self.lo, "hi": self.hi,
                "centre": [round(v, 2) for v in self.centre], "u": [round(v, 4) for v in self.u],
                "width": round(self.width, 2), "run": round(self.run, 2),
                "gold_main": self.gold_main, "nat_out": self.nat_out}


RAMP_LANDING = 3.0


def paint_planned_ramp(walk, elev, ramp, canvas, r: PlannedRamp) -> None:
    """Paint one staircase with a guaranteed full run, plus its two flat approach lanes. The band
    is never clamped to how close the endpoints are (a 2-cell gap gave a 16-24 cliff step per
    cell, an in-engine wall)."""
    (cxp, cyp), (ux, uy) = r.centre, r.u
    if r.gold_main:
        half = GOLD_MAIN_RAMP_RANKS / (2.0 * math.sqrt(2.0))
    else:
        half = r.run / 2.0
    lo_end = (cxp - half * ux, cyp - half * uy)
    hi_end = (cxp + half * ux, cyp + half * uy)
    if r.gold_main:
        _paint_corridor(walk, elev, ramp, r.p_lo, lo_end, r.width, r.lo, r.lo, canvas)
        _paint_corridor(walk, elev, ramp, hi_end, r.p_hi, r.width, r.hi, r.hi, canvas)
    else:
        # each end opens onto a straight on-axis landing before its lane turns toward the node,
        # so both plateaus meet the band squarely and the exporter climbs along the band's axis
        lo_land = (lo_end[0] - RAMP_LANDING * ux, lo_end[1] - RAMP_LANDING * uy)
        hi_land = (hi_end[0] + RAMP_LANDING * ux, hi_end[1] + RAMP_LANDING * uy)
        _paint_corridor(walk, elev, ramp, r.p_lo, lo_land, r.width, r.lo, r.lo, canvas)
        _paint_corridor(walk, elev, ramp, lo_land, lo_end, r.width, r.lo, r.lo, canvas)
        _paint_corridor(walk, elev, ramp, hi_end, hi_land, r.width, r.hi, r.hi, canvas)
        _paint_corridor(walk, elev, ramp, hi_land, r.p_hi, r.width, r.hi, r.hi, canvas)
    if r.gold_main:
        _stamp_gold_main_ramp(walk, elev, ramp, canvas, r.centre, r.u, r.lo, r.hi)
    else:
        _paint_corridor(walk, elev, ramp, lo_end, hi_end, r.width, r.lo, r.hi, canvas,
                        ramp_t0=0.0, ramp_t1=1.0)


def ramp_placements(p_lo, p_hi, diagonal_only: bool = False):
    """Band centre + uphill axis to try, best first: the crossing's midpoint on the nearest snapped
    axis, then sideways / along-axis shifts, then the second-nearest axis. A base core or a
    neighbouring plateau can clip the band at the midpoint."""
    dx, dy = p_hi[0] - p_lo[0], p_hi[1] - p_lo[1]
    slen = math.hypot(dx, dy) or 1.0
    dirs = _DIRS8[4:] if diagonal_only else _DIRS8
    ranked = sorted(dirs, key=lambda v: -(v[0] * dx + v[1] * dy) / slen)
    mx, my = (p_lo[0] + p_hi[0]) / 2.0, (p_lo[1] + p_hi[1]) / 2.0
    for u in ranked[:2]:
        if u[0] * dx + u[1] * dy <= 0.0:
            continue
        rx, ry = -u[1], u[0]
        for side, along in ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2), (4, 0), (-4, 0),
                            (2, 2), (-2, 2), (2, -2), (-2, -2), (6, 0), (-6, 0)):
            yield (mx + side * rx + along * u[0], my + side * ry + along * u[1]), u


def pad_edge(bases, i: int, toward, pad_half: float) -> tuple[float, float]:
    """Where a cut's lane leaves node ``i``: a real base's pad edge toward the crossing (a lane
    through the centre would bisect the buildable pad), else the node's centre."""
    bx, by = node_centre(bases[i])
    if bases[i].kind not in _REAL:
        return (bx, by)
    d = math.hypot(toward[0] - bx, toward[1] - by) or 1.0
    off = pad_half + 1.0
    return (bx + off * (toward[0] - bx) / d, by + off * (toward[1] - by) / d)


# --------------------------------------------------------------------------- #
# terraces: the pre-cut ground, seam walling, and ramp-aware connectivity
# --------------------------------------------------------------------------- #
def wall_seams(walk, elev, pad_level, locked) -> None:
    """Wall the higher side of every different-level seam (cliff top), protecting only pads and
    locked staircases (+halo). A free cell abutting a protected cell of another level is walled
    regardless of height, since the protected side can't be. Iterates to a fixpoint."""
    for _ in range(4):
        padprot = (pad_level >= 0) & (elev == pad_level)
        prot = padprot | ndimage.binary_dilation(locked, iterations=1)
        npf = np.zeros_like(walk)
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            npf |= np.roll(prot, sh, axis=ax) & (elev != np.roll(elev, sh, axis=ax))
        kill = _diff_boundary(elev, walk) & ~prot & (_higher_boundary(elev, walk) | npf)
        if not kill.any():
            break
        walk[kill] = False


def terrace_labels(walk, elev, ramp) -> np.ndarray:
    """Engine-accurate terrace connectivity: same-level ground (4-connected, as the engine oracle)
    joined only through a ramp component whose own cells span exactly one level step, from its
    low-plateau terrace to its high-plateau terrace. A staircase that merely sits next to a third
    plateau connects nothing there (the engine walls that seam), and a multi-level blob bridges
    nothing. Per-cell terrace root on ground cells, -1 elsewhere."""
    h, w = walk.shape
    ground = walk & ~ramp
    tl = np.full((h, w), -1, dtype=np.int64)
    nid = 0
    for L in (np.unique(elev[ground]) if ground.any() else []):
        m = ground & (elev == L)
        lab_, nn = ndimage.label(m)
        tl[m] = lab_[m] - 1 + nid
        nid += nn
    parent = np.arange(max(nid, 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    rlab, rn = ndimage.label(ramp, structure=_STRUCT8)
    for cid in range(1, rn + 1):
        comp = rlab == cid
        own = sorted({int(v) for v in elev[comp]})
        if len(own) < 2 or own[-1] - own[0] != 1:
            continue
        lo, hi = own[0], own[-1]
        nb = ndimage.binary_dilation(comp, _STRUCT8) & ground
        loT = {int(v) for v in tl[nb & (elev == lo)] if v >= 0}
        hiT = {int(v) for v in tl[nb & (elev == hi)] if v >= 0}
        for a in loT:
            for b in hiT:
                parent[find(a)] = find(b)
    out = np.full((h, w), -1, dtype=np.int64)
    wm = tl >= 0
    if nid:
        roots = np.array([find(i) for i in range(nid)], dtype=np.int64)
        out[wm] = roots[tl[wm]]
    return out


# Ground this close to a staircase's sides is not a passage in-engine (seed 2: a lane squeezed
# along a ramp's flank read connected offline and sealed two bases in the client).
RAMP_FLANK_MARGIN = 2

# Empty cells required between two staircases. Closer bands compete in-engine and only one
# registers as a ramp, and which one wins is not mirror-symmetric (seed 12: a 1-cell gap sealed
# one natural's out-ramp and its twin's neighbour).
RAMP_MIN_GAP = 3


def flank_halo(walk, elev, ramp, margin: int = RAMP_FLANK_MARGIN) -> np.ndarray:
    """Ground within ``margin`` cells of a staircase's sides: beside the band, level with it along
    its uphill axis (the cells beyond its two ends are its landings, not its flanks)."""
    h, w = walk.shape
    lab, n = ndimage.label(ramp, structure=_STRUCT8)
    out = np.zeros_like(walk)
    if not n:
        return out
    ys, xs = np.mgrid[:h, :w]
    for c in range(1, n + 1):
        comp = lab == c
        ax = ramp_axis(comp, walk, elev, ramp)
        if ax is None:
            continue
        _, (ux, uy) = snap_uphill(ax[1][0] - ax[0][0], ax[1][1] - ax[0][1])
        p = xs * ux + ys * uy
        pc = p[comp]
        out |= (ndimage.binary_dilation(comp, iterations=margin) & ~comp
                & (p >= pc.min()) & (p <= pc.max()))
    return out


def passable_terraces(walk, elev, ramp):
    """``(labels, ground)``: ``terrace_labels`` on the ground a unit can really walk, i.e. without
    the staircases' flank halos. Query nodes with ``node_terrace(labels, ground, ...)``."""
    ground = walk & ~flank_halo(walk, elev, ramp)
    return terrace_labels(ground, elev, ramp), ground


def node_terrace(lab, walk, elev, ramp, pt, level: int) -> int:
    """A node's terrace: the terrace of the nearest ground cell at the node's own level."""
    own = walk & ~ramp & (elev == level)
    iy, ix = int(round(pt[1])), int(round(pt[0]))
    if own[iy, ix]:
        return int(lab[iy, ix])
    if not own.any():
        return -1
    jdx = ndimage.distance_transform_edt(~own, return_indices=True)[1]
    return int(lab[jdx[0][iy, ix], jdx[1][iy, ix]])


def base_rings(skel, shape):
    """``(core_level, ring_level)``: each real base's immutable core and editable ring, as the
    base's level (-1 elsewhere). A cell in a core is never in a ring."""
    core_level = np.full(shape, -1, dtype=np.int16)
    ring_level = np.full(shape, -1, dtype=np.int16)
    h, w = shape
    for b in skel.bases:
        lay = b.layout
        if lay is None:
            continue
        own = np.zeros(shape, dtype=bool)
        for x, y in lay.footprint():
            if 0 <= x < w and 0 <= y < h:
                own[y, x] = True
        ring = ndimage.binary_dilation(own, _STRUCT8, iterations=lay.ring) & ~own
        ring_level[ring] = lay.level
        core_level[own] = lay.level
    ring_level[core_level >= 0] = -1
    return core_level, ring_level


@dataclass(frozen=True)
class Ground:
    """How the pre-staircase ground is painted. Recorded on the skeleton with its plan, so the
    rasterizer repaints exactly the ground the plan was decided on."""
    pad_half: float          # half-extent of every real base's flat pad
    border: int              # canvas inset from the playable rect
    min_width: float         # floor on painted lane width
    ramp_choke_width: float  # cap on a level-changing edge's lane width

    @classmethod
    def from_config(cls, cfg) -> "Ground":
        return cls(cfg.pad_half, cfg.canvas_border, cfg.min_edge_paint_width, cfg.ramp_choke_width)

    def to_json_dict(self) -> dict:
        return {"pad_half": self.pad_half, "border": self.border, "min_width": self.min_width,
                "ramp_choke_width": self.ramp_choke_width}


def paint_ground(skel):
    """The terrain before any staircase is cut, painted per ``skel.ground``: flat lanes for
    same-level edges, every room box, each real base's pad, ring and core, and every level seam
    walled.

    Returns ``(walk, elev, ramp, pad_level, canvas)``; ``canvas`` is where painting may happen
    (the inset playable rect minus the base cores)."""
    g = skel.ground
    pad_half, border, min_width, ramp_choke_width = g.pad_half, g.border, g.min_width, g.ramp_choke_width
    bases, edges, pa = skel.bases, skel.edges, skel.playable
    h, w = skel.grid_h, skel.grid_w
    pts = [node_centre(b) for b in bases]
    core_level, ring_level = base_rings(skel, (h, w))
    core = core_level >= 0
    canvas = np.zeros((h, w), dtype=bool)
    canvas[pa.y + border : pa.y + pa.height - border, pa.x + border : pa.x + pa.width - border] = True
    canvas[core] = False

    walk = np.zeros((h, w), dtype=bool)
    elev = np.zeros((h, w), dtype=np.int16)
    ramp = np.zeros((h, w), dtype=bool)
    deg = [0] * len(bases)
    for e in edges:
        deg[e.a] += 1
        deg[e.b] += 1
        if bases[e.a].level == bases[e.b].level:
            _paint_corridor(walk, elev, ramp, pts[e.a], pts[e.b],
                            painted_width(bases, e, min_width, ramp_choke_width),
                            bases[e.a].level, bases[e.a].level, canvas)
    for i, b in enumerate(bases):
        if b.kind == BaseKind.JUNCTION and deg[i] == 0:
            continue
        _paint_box(walk, elev, ramp, pts[i], b.rx, b.ry, b.angle, b.level, canvas)

    pad_level = np.full((h, w), -1, dtype=np.int16)
    pr = int(math.ceil(pad_half))
    for i, b in enumerate(bases):
        if b.kind in _REAL:
            _paint_box(walk, elev, ramp, pts[i], pad_half, pad_half, 0.0, b.level, canvas)
            iy, ix = int(round(pts[i][1])), int(round(pts[i][0]))
            pad_level[max(0, iy - pr):iy + pr + 1, max(0, ix - pr):ix + pr + 1] = b.level
    ring = (ring_level >= 0) & canvas
    walk[ring] = True
    elev[ring] = ring_level[ring]
    ramp[ring] = False
    pad_level[ring] = ring_level[ring]
    walk[core] = True
    elev[core] = core_level[core]
    ramp[core] = False
    pad_level[core] = core_level[core]
    symmetrize((walk, elev, ramp, pad_level), pa, skel.symmetry)
    wall_seams(walk, elev, pad_level, np.zeros_like(walk))
    return walk, elev, ramp, pad_level, canvas


def meets_one_terrace_per_side(comp, walk, elev, ramp) -> bool:
    """A staircase meets exactly one low and one high terrace, counted with the engine's
    4-connectivity (two floors touching only at a corner are separate, and the engine never lets
    a unit enter a ramp from its flank to get between them)."""
    plat = walk & ~ramp
    nb = ndimage.binary_dilation(comp) & plat
    levels = sorted({int(v) for v in elev[nb]}) if nb.any() else []
    if len(levels) != 2 or levels[1] - levels[0] != 1:
        return False
    for L in levels:
        lab, _ = ndimage.label(plat & (elev == L))
        if len(set(np.unique(lab[nb & (elev == L)]).tolist()) - {0}) != 1:
            return False
    return True


# --------------------------------------------------------------------------- #
# cutting a planned pair
# --------------------------------------------------------------------------- #
def _groups(walk, elev, ramp, keep) -> list[int]:
    lab, gw = passable_terraces(walk, elev, ramp)
    return [node_terrace(lab, gw, elev, ramp, pt, lv) for pt, lv in keep]


def cut_pair(walk, elev, ramp, locked, pad_level, canvas, halves: list[PlannedRamp],
             pa, sym: str, keep=None) -> list[str]:
    """Cut a staircase and its mirror, lock them, and wall the seams. Kept only if both halves come
    out gold-shaped, don't touch another locked staircase (two stacked staircases need a real flat
    landing between them, else they fuse into a 0->2 blob), don't paint over a third level, and
    each meets exactly one low and one high plateau. With ``keep`` (``[(point, level)]`` nodes), it
    must also leave every two of them that shared a passable terrace still sharing one: a cut may
    join terraces, never pinch an existing passage. Returns why it was refused (arrays restored),
    or [] with the cut applied."""
    snap = (walk.copy(), elev.copy(), ramp.copy(), locked.copy())
    before_groups = _groups(walk, elev, ramp, keep) if keep else None

    def refuse(msg: str) -> list[str]:
        walk[:], elev[:], ramp[:], locked[:] = snap
        return [msg]

    before = snap[2]
    paint_planned_ramp(walk, elev, ramp, canvas, halves[0])
    grew1 = ramp & ~before
    changed = (walk != snap[0]) | (elev != snap[1]) | (ramp != snap[2])
    symmetrize((walk, elev, ramp), pa, sym, only=changed)
    grew = ramp & ~before
    grews = [grew1, grew & ~grew1]
    for r, g in zip(halves, grews):
        if not g.any():
            continue
        cells = [(int(x), int(y)) for y, x in zip(*np.nonzero(g))]
        bad = ramp_shape_problems(cells, r.u[0], r.u[1], r.lo, r.hi)
        if bad:
            return refuse(bad[0])
    if not grews[0].any():
        return refuse("no ramp cells")
    if (ndimage.binary_dilation(grew, _STRUCT8, iterations=RAMP_MIN_GAP) & (locked & ~grew)).any():
        return refuse("too close to a locked staircase")
    if grews[1].any():
        near_twin = ndimage.binary_dilation(grews[0], _STRUCT8, iterations=RAMP_MIN_GAP) & grews[1]
        touching = ndimage.binary_dilation(grews[0], _STRUCT8) & grews[1]
        if near_twin.any() and not touching.any():
            return refuse("too close to its mirror twin")
    walk0, elev0 = snap[0], snap[1]
    seam = ndimage.binary_dilation(grew, _STRUCT8, iterations=2)
    lane_x = walk0 & ~ramp & (elev != elev0) & ~seam
    lo, hi = halves[0].lo, halves[0].hi
    band_x = grew & walk0 & (elev0 != lo) & (elev0 != hi)
    if (lane_x | band_x).any():
        hit = sorted({int(v) for v in elev0[lane_x | band_x]})
        return refuse(f"crosses {int((lane_x | band_x).sum())} cells of level {hit}")
    locked[grew] = True
    wall_seams(walk, elev, pad_level, locked)
    # walling this cut's seams must not split the plateau under an earlier staircase
    near = ndimage.binary_dilation(grew, _STRUCT8, iterations=12)
    olab, on = ndimage.label(ramp & locked & ~grew, structure=_STRUCT8)
    for cid in set(np.unique(olab[near]).tolist()) - {0}:
        if not meets_one_terrace_per_side(olab == cid, walk, elev, ramp):
            return refuse("splits the plateau under an earlier staircase")
    for r, g in zip(halves, grews):
        if not g.any():
            continue
        if not meets_one_terrace_per_side(g & ramp, walk, elev, ramp):
            return refuse("touches more than one terrace per level")
        # the exporter climbs along the plateau-contact axis, so it must be the band's own axis
        ax = ramp_axis(g & ramp, walk, elev, ramp)
        if ax is None or snap_uphill(ax[1][0] - ax[0][0], ax[1][1] - ax[0][1])[0] \
                != snap_uphill(*r.u)[0]:
            return refuse("exporter would climb it along another axis")
        if r.gold_main:
            if not is_gold_main_ramp(
                    [(int(x), int(y)) for y, x in zip(*np.nonzero(g & ramp))],
                    ax[1][0] - ax[0][0], ax[1][1] - ax[0][1], ax[2], ax[3]):
                return refuse("not the gold main staircase")
    if keep:
        after = _groups(walk, elev, ramp, keep)
        first: dict[int, int] = {}
        for k, g0 in enumerate(before_groups):
            if g0 < 0:
                continue
            j = first.setdefault(g0, k)
            if after[k] != after[j]:
                return refuse("pinches a passage between two connected nodes")
    return []


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def planned_nodes(skel) -> list[tuple[tuple[float, float], int]]:
    """``(centre, level)`` of every node that paints a room: what a cut must not disconnect."""
    deg = [0] * len(skel.bases)
    for e in skel.edges:
        deg[e.a] += 1
        deg[e.b] += 1
    return [(node_centre(b), int(b.level)) for i, b in enumerate(skel.bases)
            if not (b.kind == BaseKind.JUNCTION and deg[i] == 0)]


def plan_ramps(skel, mirror: list[int], cfg) -> tuple[list[PlannedRamp], list[str]]:
    """Decide every staircase on the skeleton. Returns ``(ramps, problems)``: ramps come in mirror
    pairs ``[r, r']`` (a self-mirror crossing's partner covers the same nodes), in cut order.

    Cross-level edges are cut main-first, then shortest first, one per mirror pair, skipping an
    edge whose two terraces some earlier cut already joined. A main's ramp is the gold stamp; the
    natural's out-ramp is exactly ``cfg.natural_out_width`` wide; other widths and runs are drawn
    from ``cfg.ramp_width_range`` / ``cfg.ramp_run_range``. ``problems`` names every main or
    natural-out crossing that couldn't be cut (or was already bypassed), and every real base the
    finished plan leaves unreachable from the first main: the skeleton rejects the attempt."""
    bases, edges, pa = skel.bases, skel.edges, skel.playable
    sym = skel.symmetry
    cen = (pa.x + pa.width / 2.0, pa.y + pa.height / 2.0)
    pts = [node_centre(b) for b in bases]
    skel.ground = Ground.from_config(cfg)
    walk, elev, ramp, pad_level, canvas = paint_ground(skel)
    pa_ = skel.playable
    locked = np.zeros_like(walk)
    rng = np.random.default_rng(skel.seed * 2 + 13)
    level = [int(b.level) for b in bases]

    def pkey(e):
        return tuple(sorted([tuple(sorted((e.a, e.b))), tuple(sorted((mirror[e.a], mirror[e.b])))]))

    def is_main(e):
        return BaseKind.MAIN in (bases[e.a].kind, bases[e.b].kind)

    cross = sorted({pkey(e): e for e in edges if level[e.a] != level[e.b]}.values(),
                   key=lambda e: (not is_main(e), math.hypot(pts[e.a][0] - pts[e.b][0],
                                                             pts[e.a][1] - pts[e.b][1])))
    out: list[PlannedRamp] = []
    problems: list[str] = []
    sep = int(math.ceil(cfg.ramp_run_range[1]))
    keep = planned_nodes(skel)
    for e in cross:
        a, b = e.a, e.b
        main, nat = is_main(e), is_nat_out(bases, e)
        what = "main ramp" if main else "natural out-ramp" if nat else None
        mx, my = (pts[a][0] + pts[b][0]) / 2.0, (pts[a][1] + pts[b][1]) / 2.0
        iy, ix = int(round(my)), int(round(mx))
        if what is None and locked[max(0, iy - sep):iy + sep + 1, max(0, ix - sep):ix + sep + 1].any():
            continue
        lab, gw = passable_terraces(walk, elev, ramp)
        if node_terrace(lab, gw, elev, ramp, pts[a], level[a]) \
                == node_terrace(lab, gw, elev, ramp, pts[b], level[b]) >= 0:
            if what:
                problems.append(f"{what} {a}-{b}: its terraces are already joined elsewhere")
            continue
        if abs(level[a] - level[b]) != 1:
            continue
        cw = float(rng.uniform(*cfg.ramp_width_range))
        rn = float(rng.uniform(*cfg.ramp_run_range))
        if nat:
            cw = cfg.natural_out_width
        lo_n, hi_n = (a, b) if level[a] < level[b] else (b, a)
        p_lo = pad_edge(bases, lo_n, pts[hi_n], skel.ground.pad_half)
        p_hi = pad_edge(bases, hi_n, pts[lo_n], skel.ground.pad_half)
        why = []
        for c, u in ramp_placements(p_lo, p_hi, diagonal_only=main):
            r = PlannedRamp(lo_n, hi_n, level[lo_n], level[hi_n], p_lo, p_hi, c, u, cw, rn,
                            gold_main=main, nat_out=nat)
            pair = [r, r.mirrored(mirror, sym, cen)]
            bad = cut_pair(walk, elev, ramp, locked, pad_level, canvas, pair, pa_, sym, keep)
            if not bad:
                out += pair
                break
            why.append(bad[0])
        else:
            if what:
                problems.append(f"{what} {a}-{b}: no clean placement ({sorted(set(why))})")
    lab, gw = passable_terraces(walk, elev, ramp)
    main1 = next(i for i, b in enumerate(bases) if b.kind == BaseKind.MAIN)
    root = node_terrace(lab, gw, elev, ramp, pts[main1], level[main1])
    for i, b in enumerate(bases):
        if b.kind in _REAL and node_terrace(lab, gw, elev, ramp, pts[i], level[i]) != root:
            problems.append(f"{b.kind.name} {i}: unreachable from the first main")
    return out, problems
