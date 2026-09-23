"""Milestone 4: geometry rasterizer (boxy rooms + rectangular chokes).

Real competitive maps read as a set of distinct terraced *rooms* (bases, expansions,
interior junctions/plazas) at varied heights, connected by *rectangular chokes/ramps*, so
moving through the map is a rhythm of "narrow corridor (choke) -> wide room (open) ->
narrow corridor ...". Circular plateaus joined by rounded tubes can't produce that; this
version realizes the symmetric skeleton as:

    1. BOXES: paint a walkable rectangular room at every node (MAIN/NATURAL/BASE/ROOM),
       with independently-sampled half-extents (aspect variety) at a sampled angle. Height
       = the node's level (MAIN high, NATURAL mid, generic/ROOM at varied levels).
    2. CORRIDORS: paint a rectangular (flat-ended) lane along every skeleton edge; a lane
       that spans two levels interpolates height and becomes a RAMP. Widths are SHAPED:
       the main's single exit is the narrowest choke, interior lanes vary. Rooms are wide,
       so corridor->room reads as choke->open.
    3. CLIFF-CLEANUP: any non-ramp different-level adjacency becomes a thin wall -> every
       level change is legally a ramp or a cliff (yields rectangular cliff faces).
    4. REPAIR connectivity (guarantee a ground path between the two mains), mirror-safe.
    5. AUTO-TUNE the room-growth multiplier so playable_ratio lands in a mid band.

Everything is driven off the symmetric skeleton (a 180-degree rotation maps a box of a
given orientation to a box of the same orientation, so the box model stays symmetric).
Output is a real ``MapIR`` (same type as ingestion).
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from sc2mapgen.generate.skeleton import Skeleton
from sc2mapgen.ir import (
    BaseKind, BaseNode, MapIR, Ramp, Rect, Resource, ResourceKind,
    ramp_cell_cliffs, snap_uphill,
)

# real resource bases (need a buildable pad); ROOM/JUNCTION are routing-only pseudo-nodes.
_REAL_BASES = {BaseKind.MAIN, BaseKind.NATURAL, BaseKind.BASE}


@dataclass
class RasterConfig:
    # canvas: inset (tiles) from the playable rect so walkable never touches the map edge
    border: int = 2

    # Room geometry now derives from each node's scalar ``width`` (from the skeleton): the
    # room's short cross-section == node width, its long side == width * aspect. This keeps
    # "the node is the widest part locally" true, since every incident edge <= node width.
    aspect: tuple[float, float] = (1.0, 1.7)   # long/short room ratio
    box_cap: float = 18.0                       # cap a room half-extent even after growth
    junction_scale: float = 0.8                 # junctions are modest plazas, not full rooms
    anchor_growth: float = 1.5                  # fixed MAIN/NATURAL growth (not openness-tuned)

    # elevation levels are OWNED BY THE SKELETON now (generate/skeleton.GenConfig: main_level /
    # natural_level / generic_levels / generic_level_weights). The rasterizer reads each node's tier
    # from ``SkelBase.level`` and only PAINTS it -- it no longer draws or nudges levels.

    # corridor widths come straight from the skeleton edges; floor only:
    min_width: float = 3.0

    # A level change is rendered as a NARROW CHOKE: one SC2 <ramp> quad covers only a narrow band, so
    # a corridor-wide ramp is partly covered and its flanks stay walls, sealing the map (findings
    # §4.4). Cap any level-changing edge to this width; same-level lanes keep full skeleton width.
    ramp_choke_width: float = 4.0
    # A level change is painted as a CLEAN STRAIGHT STAIRCASE snapped to one of the 8 ramp directions,
    # flanked by same-level approach lanes, aligned to the direction the exporter snaps to so the
    # +8/cell gradient and the <ramp> quad cover it fully (findings §4.4/§4.5). ``ramp_run`` is the
    # staircase length (cells) along the uphill axis.
    ramp_run: float = 6.0

    # TERRACE-mode per-cut ramp size variety (width, run). With the default true-isoline gradient
    # (findings §4.10) the cliff steps +8 per axis-cell, so a one-level climb is a fixed ~8-cell run;
    # a diagonal band needs >= ~8 isolines, hence run (8,10) (STRICT yield 65/120; seeds 5 & 22
    # connect in-engine). The authored band WIDTH is held MODERATE: a quad wider than the true band
    # balloons the footprint and spawns phantom ramps that seal the exit, and widening the band re-
    # breaks traversability offline-invisibly, so we hold (4,6) (findings §4.6/§4.8/§4.9).
    ramp_choke_range: tuple[float, float] = (4.0, 6.0)
    ramp_run_range: tuple[float, float] = (8.0, 10.0)

    # main->natural ramp length (tiles): the main's entrance is a short slope at the main's
    # (high) edge rather than a long tube. MUST be >= 8: a one-level (64 cliff) change is only
    # walkable if the gradient steps <= 8 per cell, so it needs >= 8 cells of run (SC2 engine
    # constraint, verified via scripts/step_probe.py). All other lanes keep the full-length ramp.
    ramp_len: float = 9.0

    # a walkable cell is buildable if this far (tiles) from the nearest wall
    build_clearance: float = 2.0

    # every real base (MAIN/NATURAL/BASE) is guaranteed a flat buildable *pad* of this
    # half-extent (tiles) at its own level, so the nexus/CC + mineral line + geyser always
    # fit and sit on the same floor. ROOM/JUNCTION nodes are pure routing, no pad needed.
    base_pad_half: float = 9.5
    # assertion: each real base must have at least this buildable clearance (radius, tiles)
    base_min_clear: float = 4.0

    # openness auto-tune: grow ROOMS (not corridors) until playable_ratio lands in this band
    target_openness: tuple[float, float] = (0.50, 0.62)
    growth_lo: float = 0.7
    growth_hi: float = 3.6
    growth_iters: int = 10

    # --- resources (M5): a deterministic, legal mineral line + geysers per real base. The
    # mineral field is a shallow arc facing *away* from the base's corridors (the classic
    # "mineral line at the back"); geysers flank it. All patches sit on the base's own floor
    # and are mirror-symmetric with the paired base. ---
    res_minerals: int = 8               # mineral patches per base
    res_geysers: int = 2                # vespene geysers per base
    # python-sc2's expansion finder requires the townhall be >=6 from every mineral and >=7
    # from every geyser; a townhall sits in the pocket ~base-center, so rings must clear that.
    res_mineral_ring: float = 6.5       # mineral arc radius from base center (tiles)
    res_gas_ring: float = 7.5           # geyser radius from base center (tiles)
    res_mineral_arc_deg: float = 120.0  # angular spread of the mineral arc
    res_gas_flank_deg: float = 18.0     # geyser offset just outside each mineral-arc end
    res_min_gap: float = 1.5            # min center spacing between any two placed patches
    res_townhall_half: float = 2.5      # keep patches off the townhall footprint (tiles)
    # min radius (tiles) a patch may sit from the base center; matches python-sc2's townhall
    # clearance (>=6 from minerals, >=7 from geysers) so the pocket is always placeable.
    res_townhall_clear: float = 6.0
    res_mineral_amount: int = 1800
    res_gas_amount: int = 2250

    weirdness: float = 0.0

    # TERRACE-FIRST ramp model (findings §4.4): don't paint ramps per edge; paint solid same-level
    # terraces, WALL every level boundary, then cut ONE choke-width straight staircase per terrace-pair
    # that must connect (locked), so clustered crossings can't fuse into unauthorable blobs. A post-
    # extraction redundant-crossing prune (in ``rasterize``; opt-out NO_PRUNE_CROSSINGS) walls the
    # surplus fusion-prone crossings in mirror pairs without dropping offline yield (findings §4.11).
    # Default ON; force the legacy per-edge path with NO_TERRACE=1.
    terrace_mode: bool = True


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pair_mirror(bases) -> list[int]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(bases):
        groups[b.pair].append(i)
    mirror = list(range(len(bases)))
    for idxs in groups.values():
        if len(idxs) == 2:
            a, b = idxs
            mirror[a], mirror[b] = b, a
    return mirror


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


def _paint_ramp_edge(walk, elev, ramp, p_a, p_b, la: int, lb: int, width: float,
                     canvas, run: float, passage=None) -> None:
    """Paint a level-changing connection as a CLEAN STRAIGHT STAIRCASE (snapped to one of the 8
    ramp directions) flanked by same-level flat approach lanes -- the gold-standard ramp shape.

    Unlike slicing a bent corridor (which yields an irregular ramp the exporter's straight quad +
    1D gradient can't match, sealing the map -- see findings S4.4), the staircase here is a straight
    rectangle aligned to the SAME 8-way direction the exporter snaps to, so its cross-section lines
    are exactly the gradient's isolines: seamless with both plateaus, fully covered by one quad.

    Geometry: uphill axis u = snap8(low->high). The ``run``-long ramp rectangle sits centred in the
    gap on the segment, oriented along u; a flat lane at the LOW level runs from the low node to the
    ramp's low end, and a flat lane at the HIGH level from the ramp's high end to the high node. If
    the ends share a level this degenerates to a plain flat corridor (no ramp).
    """
    if la == lb:
        _paint_corridor(walk, elev, ramp, p_a, p_b, width, la, lb, canvas, passage=passage)
        return
    (p_low, l0), (p_high, l1) = ((p_a, la), (p_b, lb)) if la < lb else ((p_b, lb), (p_a, la))
    dx, dy = p_high[0] - p_low[0], p_high[1] - p_low[1]
    slen = math.hypot(dx, dy) or 1.0
    ux, uy = _snap8(dx / slen, dy / slen)
    mx, my = (p_low[0] + p_high[0]) / 2.0, (p_low[1] + p_high[1]) / 2.0   # gap midpoint
    hr = min(run, slen) / 2.0
    lo_end = (mx - hr * ux, my - hr * uy)
    hi_end = (mx + hr * ux, my + hr * uy)
    # flat approach lanes (same-level, no ramp) from each node to the staircase ends
    _paint_corridor(walk, elev, ramp, p_low, lo_end, width, l0, l0, canvas, passage=passage)
    _paint_corridor(walk, elev, ramp, hi_end, p_high, width, l1, l1, canvas, passage=passage)
    # the straight staircase itself (whole span is ramp, gradient l0->l1 along u)
    _paint_corridor(walk, elev, ramp, lo_end, hi_end, width, l0, l1, canvas,
                    ramp_t0=0.0, ramp_t1=1.0)


def _box_extent(rx: float, ry: float, box_ang: float, phi: float) -> float:
    """Half-extent of a rotated rectangle (half-sizes rx,ry at angle box_ang) measured
    along direction phi -> how far the room reaches from its center toward the corridor."""
    th = box_ang - phi
    return abs(rx * math.cos(th)) + abs(ry * math.sin(th))


def _ramp_band(p0, p1, ext0: float, ext1: float, ramp_len: float,
               higher: int) -> tuple[float, float]:
    """Parametric [t0,t1] for a short ramp placed right at the *higher* plateau's edge, so
    the high ground stays compact and the lower plateau simply extends to meet the ramp
    (the natural main->natural look)."""
    slen = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) or 1.0
    t_a = min(ext0 / slen, 0.48)          # corridor leaves room 0 here
    t_b = max(1.0 - ext1 / slen, 0.52)    # corridor enters room 1 here
    if t_b <= t_a:                        # rooms overlap along the corridor -> tiny center ramp
        return 0.46, 0.54
    h = ramp_len / slen
    if higher == 0:                       # room 0 is the high plateau -> ramp at its edge
        lo, hi = t_a, min(t_b, t_a + h)
    else:                                 # room 1 is the high plateau -> ramp at its edge
        lo, hi = max(t_a, t_b - h), t_b
    return (t_a, t_b) if hi <= lo else (lo, hi)


def _diff_boundary(elev: np.ndarray, walk: np.ndarray) -> np.ndarray:
    """Walkable cells adjacent (4-neigh) to a walkable cell of a different elevation."""
    diff = np.zeros_like(walk, dtype=bool)
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        diff |= walk & np.roll(walk, sh, axis=ax) & (elev != np.roll(elev, sh, axis=ax))
    return diff


def _symmetrize_rot180(walk: np.ndarray, elev: np.ndarray, ramp: np.ndarray, pa):
    """Force EXACT rot180 symmetry by copying each grid's canonical (lower raster-index) half onto
    its mirror partner. ``partner(x, y) = (msX - x, msY - y)`` with ``msX = 2*pa.x + pa.width`` and
    ``msY = 2*pa.y + pa.height`` (the doubled continuous playable centre the skeleton mirrors about)
    is an exact involution on the integer grid, so copying the low-index half yields a pixel-exact
    mirror. Cells whose partner falls outside the grid (a single unpaired edge row/col) are left as
    is. Operates in place on ``walk`` / ``elev`` / ``ramp`` with the SAME mapping so they stay
    mutually consistent; returns them for convenience."""
    h, w = walk.shape
    msx = 2 * pa.x + pa.width
    msy = 2 * pa.y + pa.height
    ys, xs = np.mgrid[0:h, 0:w]
    mxs, mys = msx - xs, msy - ys
    inb = (mxs >= 0) & (mxs < w) & (mys >= 0) & (mys < h)
    # canonical source = cells whose own raster index is below their partner's; each partner is then
    # written exactly once, so the result is a clean mirror of the canonical half.
    src = inb & (ys * w + xs < mys * w + mxs)
    sy, sx = np.where(src)
    ty, tx = msy - sy, msx - sx
    for grid in (walk, elev, ramp):
        grid[ty, tx] = grid[sy, sx]
    return walk, elev, ramp


def _level_components(walk: np.ndarray, elev: np.ndarray, ramp: np.ndarray) -> np.ndarray:
    """Label walkable cells into ENGINE-TRAVERSABLE components: two adjacent walkable cells are
    connected iff they share an elevation OR one of them is a ramp cell -- exactly how the SC2
    pathing engine treats terrain (a level seam is an impassable cliff; only an authored ramp
    crosses it). Returns an int component id per cell (-1 where not walkable).

    NOTE: plain ``ndimage.label(walk)`` is WRONG for connectivity here -- it ignores elevation and
    counts a cliff as passable, so it reports the map connected when the engine sees several
    disconnected level-islands. All connectivity decisions must use THIS function.
    """
    h, w = walk.shape
    ground = walk & ~ramp
    # region id per cell: same-level ground components first, ramp components after (offset by nid)
    region = np.full((h, w), -1, dtype=np.int64)
    nid = 0
    for L in np.unique(elev[ground]) if ground.any() else []:
        m = ground & (elev == L)
        lab, n = ndimage.label(m, structure=_STRUCT8)
        region[m] = lab[m] - 1 + nid          # 0-indexed (matches the ramp ids below)
        nid += n
    rlab, rn = ndimage.label(ramp, structure=_STRUCT8)
    region[ramp] = rlab[ramp] - 1 + nid
    parent = np.arange(nid + rn)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # union every adjacency that the engine can traverse: it always keeps same-region cells
    # together, and a ramp cell bridges to any walkable neighbour (vectorised over 8 shifts).
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1),):
        a, b = region, np.roll(region, sh, axis=ax)
        rr = ramp | np.roll(ramp, sh, axis=ax)
        m = walk & np.roll(walk, sh, axis=ax) & rr & (a >= 0) & (b >= 0) & (a != b)
        if not m.any():
            continue
        for u, v in {(int(p), int(q)) for p, q in zip(a[m].tolist(), b[m].tolist())}:
            parent[find(u)] = find(v)
    # diagonal ramp bridges (ramps can be diagonal staircases)
    for dy, dx in ((1, 1), (1, -1)):
        a = region
        b = np.roll(np.roll(region, dy, axis=0), dx, axis=1)
        rr = ramp | np.roll(np.roll(ramp, dy, axis=0), dx, axis=1)
        ww = walk & np.roll(np.roll(walk, dy, axis=0), dx, axis=1)
        m = ww & rr & (a >= 0) & (b >= 0) & (a != b)
        if not m.any():
            continue
        for u, v in {(int(p), int(q)) for p, q in zip(a[m].tolist(), b[m].tolist())}:
            parent[find(u)] = find(v)
    roots = np.array([find(i) for i in range(nid + rn)], dtype=np.int64)
    out = np.full((h, w), -1, dtype=np.int64)
    wm = region >= 0
    out[wm] = roots[region[wm]]
    return out


def engine_cliff_grid(walk: np.ndarray, elev: np.ndarray, ramps) -> np.ndarray:
    """Predict the EXPORTED ``t3SyncCliffLevel`` grid (in IR space) so connectivity can be judged the
    way the SC2 engine actually paths -- WITHOUT loading a template.

    Reproduces ``export.sc2map.build_cliff_cells`` + ``author_ramp_cliffs`` under the fixed SC2 cliff
    encoding (a plateau at level L has cliff ``(L+1)*64``; ramps subdivide a level into 8-unit
    sub-steps). A void cell is 0; a plateau cell is ``(L+1)*64``; a ramp cell gets the SAME
    GOLD-STANDARD gradient the exporter writes (``ir.ramp_cell_cliffs``: 72,80,...,112 then +16 to
    128, skipping 120) between its low/high plateau cliffs. The +16 top step is quad-covered and thus
    walkable, so ``engine_components`` joins it (a diff of exactly 16). Only Δcliff matters for
    traversability, so the absolute base (64) is irrelevant, and the
    generator caps levels at 2 (tiers 64/128/192), matching the default FrostLE palette.

    This is the honest connectivity oracle: ``ndimage.label(walk)`` ignores cliffs entirely, and the
    older ``_level_components`` optimistically assumed every ramp cell bridges its neighbours even
    when the exported gradient would top out short of the high plateau (a >8 seam) -- this grid
    instead carries the real gradient, so a ramp that can't climb its level shows up as disconnected.
    """
    h, w = walk.shape
    lev = np.clip(elev.astype(np.int32), 0, None)
    cliff = np.where(walk, (lev + 1) * 64, 0).astype(np.int32)
    for r in ramps:
        if r.low_level is None or r.high_level is None or r.top is None or r.bottom is None:
            continue
        lo, hi = int(r.low_level), int(r.high_level)
        if hi <= lo:
            continue
        cells = [(int(x), int(y)) for (x, y) in r.cells if 0 <= int(x) < w and 0 <= int(y) < h]
        if not cells:
            continue
        lo_cliff = (lo + 1) * 64
        hi_cliff = (hi + 1) * 64
        _, (ux, uy) = snap_uphill(r.top[0] - r.bottom[0], r.top[1] - r.bottom[1])
        for (x, y), cval in zip(cells, ramp_cell_cliffs(cells, ux, uy, lo_cliff, hi_cliff)):
            cliff[y, x] = cval
    # Mirror the exporter's flank channeling (export.sc2map.channel_ramp_flanks): it VOIDS plateau
    # cells abutting a ramp flank that differ by >8 (findings §4.4), so a ramp is crossable only at
    # its lo/hi ends. The offline oracle MUST apply the same voids or it keeps cells the exporter
    # deletes and over-reports connectivity -- e.g. seed 5's ramp2 read as bridging L1<->L2 offline
    # while in-engine it is a walled dead-end pocket (its low end's L1 plateau cells were channeled
    # away). Verified: applying this drops offline_walk to the exact exported walkable set.
    from sc2mapgen.export.sc2map import channel_ramp_flanks
    ramp_mask = np.zeros((h, w), dtype=bool)
    for r in ramps:
        for (x, y) in r.cells:
            ix, iy = int(x), int(y)
            if 0 <= ix < w and 0 <= iy < h:
                ramp_mask[iy, ix] = True
    channel_ramp_flanks(cliff, ramp_mask)
    return cliff


def cardinal_ramp_bridge(ramps, shape: tuple[int, int]) -> np.ndarray | None:
    """Per-cell ramp id (``engine_components`` ``bridge``) for gold-profile CARDINAL ramps only.

    Their staircase steps +16/+24 (``ir.cardinal_profile``), which the strict ``<=8`` rule reads as
    walls although the quad makes them walkable in-engine. Diagonal ramps keep the strict rule (a
    blanket bridge false-accepted seed 22's diagonal, see validate_map). None if there are none.
    """
    from sc2mapgen.ir import is_gold_cardinal

    h, w = shape
    rid = np.zeros((h, w), dtype=np.int32)
    for i, r in enumerate(ramps, start=1):
        if r.top is None or r.bottom is None:
            continue
        if not is_gold_cardinal(r.top[0] - r.bottom[0], r.top[1] - r.bottom[1]):
            continue
        for (x, y) in r.cells:
            ix, iy = int(x), int(y)
            if 0 <= ix < w and 0 <= iy < h:
                rid[iy, ix] = i
    return rid if rid.any() else None


def engine_components(cliff: np.ndarray, max_step: int = 8,
                      bridge: np.ndarray | None = None) -> np.ndarray:
    """4-connected components of a cliff grid: two adjacent non-void cells are joined iff
    ``|Δcliff| <= max_step`` -- the SC2 ground-traversal rule -- OR both cells belong to the SAME
    ramp (``bridge`` carries a per-cell ramp id: 0 = not a ramp cell, >0 = that ramp's id).

    4-connectivity (no diagonals) matches the engine, which will not squeeze a unit through a
    one-cell void corner. The ``bridge`` (same-ramp) rule models the VERIFIED ramp physics (findings
    §4.5, ``scripts/rampstep_probe.py`` + ``rampwide_probe.py``): a ``<rampList>`` quad makes the
    transition it covers walkable REGARDLESS of the cliff-gradient step, and the engine floods that
    walkability across the whole contiguous single-level band -- so a ramp's own cells are mutually
    connected even when a short/steep gradient steps >8/cell WITHIN the band.

    Crucially the bridge is scoped to *one ramp*, NOT "any adjacency touching a ramp cell" (the old
    rule): a ramp's low/high END cells already equal the low/high plateau cliffs (rank 0 == lo_cliff,
    rank span == hi_cliff), so the ramp<->plateau joins at the matching-level edges fall out of
    ``step_ok`` for free -- while its FLANKS (which differ from the abutting plateau by >8) stay
    walled, exactly as the engine channels them. Scoping to one id also stops two *different* ramps
    whose cells happen to touch from fusing into a false cross-level path. Without ``bridge`` this
    reduces to the strict ``|Δcliff| <= 8`` rule. Returns an int label per cell, ``-1`` where void.
    Component ids are arbitrary but stable within a call."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    h, w = cliff.shape
    nz = cliff > 0
    c = cliff.astype(np.int32)
    if bridge is None:
        rid = np.zeros((h, w), dtype=np.int32)
    elif bridge.dtype == bool:            # back-compat: a boolean mask == all ramp cells share id 1
        rid = bridge.astype(np.int32)
    else:
        rid = bridge.astype(np.int32)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    # Border rows/cols are masked off before shifting, so no wrap-around edges are created.
    for dy, dx in ((0, 1), (1, 0)):
        diff = np.abs(c - np.roll(np.roll(c, dy, 0), dx, 1))
        # GOLD-STANDARD ramp top step: the top sub-level (hi-16 == 112) abuts the high plateau
        # (128) with a +16 axis step (120 is skipped, findings §4.7). That step is walkable because
        # the <rampList> quad covers the high-plateau interface, so join it too. After
        # channel_ramp_flanks the ONLY surviving axis pair with diff==2*max_step is a ramp top exit
        # (flank walls are voided or differ by >=24), so this can't bless a genuine cliff wall.
        step_ok = (diff <= max_step) | (diff == 2 * max_step)
        rid_sh = np.roll(np.roll(rid, dy, 0), dx, 1)
        bridged = (rid > 0) & (rid == rid_sh)     # both cells belong to the SAME ramp
        m = nz & np.roll(np.roll(nz, dy, 0), dx, 1) & (step_ok | bridged)
        if dy == 1:
            m[0, :] = False
        if dx == 1:
            m[:, 0] = False
        ys, xs = np.where(m)
        rows.append(ys * w + xs)
        cols.append((ys - dy) * w + (xs - dx))
    r = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    cc = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    graph = coo_matrix(
        (np.ones(r.size, dtype=np.int8), (r, cc)), shape=(h * w, h * w)).tocsr()
    _, labels = connected_components(graph, directed=False)
    out = np.full((h, w), -1, dtype=np.int64)
    out[nz] = labels.reshape(h, w)[nz]
    return out


def _higher_boundary(elev: np.ndarray, walk: np.ndarray) -> np.ndarray:
    """Walkable cells strictly HIGHER than some walkable 4-neighbour -- i.e. the top edge of a
    cliff. Walling only these (not the lower side) turns a level seam into a clean cliff while
    keeping the lower floor continuous, so low passages aren't pinched off."""
    hi = np.zeros_like(walk, dtype=bool)
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        hi |= walk & np.roll(walk, sh, axis=ax) & (elev > np.roll(elev, sh, axis=ax))
    return hi


def _adj_to_mask(mask: np.ndarray) -> np.ndarray:
    """Cells 4-adjacent to any True cell of ``mask``."""
    out = np.zeros_like(mask, dtype=bool)
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        out |= np.roll(mask, sh, axis=ax)
    return out


_STRUCT8 = np.ones((3, 3), dtype=bool)


def _ramp_neighbor_levels(comp: np.ndarray, walk: np.ndarray, elev: np.ndarray,
                          ramp: np.ndarray) -> list[int]:
    """Sorted distinct plateau levels a ramp component touches (its non-ramp walkable nbrs)."""
    nb = ndimage.binary_dilation(comp, _STRUCT8) & walk & ~ramp
    return sorted({int(v) for v in elev[nb]}) if nb.any() else []


def _wall_multilevel_ramps(walk: np.ndarray, elev: np.ndarray, ramp: np.ndarray) -> int:
    """A ramp component whose OWN cells span >1 level (a merged staircase with no mid landing,
    e.g. a 0->2 blob) can't be authored as a single walkable ramp. Remove it (wall it off) so
    the two floors are cleanly separated; downstream connectivity-repair then reconnects
    whatever that islands via a proper single-level link (or the map is regenerated).

    We judge by the ramp's OWN painted elevations, NOT its neighbours: with distinct per-node
    levels packed together, a perfectly valid 0->1 ramp often sits beside an unrelated level-2
    room, so a neighbour-span test would falsely wall it (and sever the map). A real single-step
    ramp is painted with a 0->1 gradient (own span == 1); only genuine >1-span blobs are walled.
    Returns the number of cells walled."""
    lab, n = ndimage.label(ramp, structure=_STRUCT8)
    walled = 0
    for cid in range(1, n + 1):
        comp = lab == cid
        own = elev[comp]
        if own.size and int(own.max()) - int(own.min()) > 1:
            walk[comp] = False
            ramp[comp] = False
            walled += int(comp.sum())
    return walled


def _moat_ramp_flanks(walk: np.ndarray, elev: np.ndarray, ramp: np.ndarray,
                      pad_level: np.ndarray) -> int:
    """Void a moat along each ramp's two long FLANKS so the ramp is a clean isolated bridge that
    touches exactly ONE plateau at its low end and ONE at its high end -- the gold-standard choke.

    Without this, a plateau wraps around the ramp's side (the ramp then borders several plateaus of
    the same level -- the validator's ``plateaus/level={2:4}`` failure). The exporter's flank
    channeling (`channel_ramp_flanks`) then punches POROUS voids through the transition wherever a
    mid-ramp cell (cliff well below the plateau) abuts that wrap, pinching/severing the crossing.
    By removing the flank-adjacent plateau cells HERE (before export, before IR ramp detection) the
    ramp keeps clean plateau contact only at its two ends. Never eats a base pad. Returns cells
    voided. Deterministic + symmetric-in => symmetric-out.
    """
    yy, xx = np.mgrid[0 : elev.shape[0], 0 : elev.shape[1]].astype(np.float64)
    is_pad = pad_level >= 0
    lab, n = ndimage.label(ramp, structure=_STRUCT8)
    towall = np.zeros_like(walk)
    for cid in range(1, n + 1):
        comp = lab == cid
        nb = ndimage.binary_dilation(comp, _STRUCT8) & walk & ~ramp
        levels = _ramp_neighbor_levels(comp, walk, elev, ramp)
        if len(levels) < 2 or not nb.any():
            continue
        lo, hi = levels[0], levels[-1]
        lo_nb, hi_nb = nb & (elev == lo), nb & (elev == hi)
        if not lo_nb.any() or not hi_nb.any():
            continue
        # flow axis u: low-plateau centroid -> high-plateau centroid
        ux = xx[hi_nb].mean() - xx[lo_nb].mean()
        uy = yy[hi_nb].mean() - yy[lo_nb].mean()
        nrm = math.hypot(ux, uy) or 1.0
        ux, uy = ux / nrm, uy / nrm
        proj = xx * ux + yy * uy
        pc = proj[comp]
        pmin, pmax = pc.min(), pc.max()
        # a plateau neighbour whose projection falls in the ramp's MID span (not near either end)
        # is a flank -> void it. Keep a 2-cell end margin so the end approach lanes stay attached.
        flank = nb & (proj > pmin + 2.0) & (proj < pmax - 2.0) & ~is_pad
        towall |= flank
    walk[towall] = False
    return int(towall.sum())


def _straighten_ramps(walk: np.ndarray, elev: np.ndarray, ramp: np.ndarray,
                      choke_w: float) -> int:
    """Crop every ramp component to a CLEAN STRAIGHT BAND along its snapped 8-way uphill axis.

    Two ramps that meet at a shared node fuse into an L/T/blob (own-span still 1, so the multilevel
    wall leaves it) that the exporter's single straight <ramp> quad + 1D gradient can't cover -> the
    choke won't author walkable. Here we keep only the cells within ``choke_w/2`` of the centreline
    (through the component centroid, along the low->high axis snapped to a cardinal/diagonal), and
    WALL the off-axis arm. That leaves one authorable band; whatever the walled arm islanded is
    reconnected by the downstream connectivity repair (cleanly, via a protected passage or a fresh
    single staircase). Deterministic + symmetric-in => symmetric-out. Returns cells removed."""
    h, w = walk.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    lab, n = ndimage.label(ramp, structure=_STRUCT8)
    removed = 0
    for cid in range(1, n + 1):
        comp = lab == cid
        if int(comp.sum()) < 6:
            continue
        own = elev[comp]
        lo, hi = int(own.min()), int(own.max())
        if hi == lo:
            continue                                    # flat 'ramp' -> left to the regularizer
        lo_c, hi_c = comp & (elev == lo), comp & (elev == hi)
        if not lo_c.any() or not hi_c.any():
            continue
        ux = xx[hi_c].mean() - xx[lo_c].mean()
        uy = yy[hi_c].mean() - yy[lo_c].mean()
        nrm = math.hypot(ux, uy) or 1.0
        ux, uy = _snap8(ux / nrm, uy / nrm)
        cx, cy = xx[comp].mean(), yy[comp].mean()
        perp = np.abs((xx - cx) * (-uy) + (yy - cy) * ux)   # distance to the centreline
        drop = comp & (perp > choke_w / 2.0 + 0.5)
        if drop.any():
            walk[drop] = False
            ramp[drop] = False
            removed += int(drop.sum())
    return removed


def _ramp_bfs_run(comp: np.ndarray, lo_nb: np.ndarray, hi_nb: np.ndarray) -> int:
    """8-connected cell-distance THROUGH the ramp from the low-plateau seam to the high-plateau
    seam -- exactly the run the exporter's 8/cell gradient climbs. This is the metric that
    matters (a wide-but-shallow blob has a big bounding box yet a tiny run)."""
    seed = ndimage.binary_dilation(lo_nb, _STRUCT8) & comp
    if not seed.any():
        return 0
    INF = 1 << 30
    dist = np.full(comp.shape, INF, dtype=np.int32)
    dist[seed] = 0
    frontier = seed.copy()
    d = 0
    reach = ndimage.binary_dilation(hi_nb, _STRUCT8) & comp
    while frontier.any():
        if (frontier & reach).any():
            return d
        d += 1
        nxt = ndimage.binary_dilation(frontier, _STRUCT8) & comp & (dist == INF)
        dist[nxt] = d
        frontier = nxt
    return 0                                            # low seam never reaches high seam


def _lengthen_ramps(walk: np.ndarray, elev: np.ndarray, ramp: np.ndarray,
                    pad_level: np.ndarray, run_min: int = 8) -> None:
    """Grow every single-level ramp until its BFS run (low seam -> high seam through the ramp)
    reaches ``run_min`` cells, so the exporter's fixed 8/cell gradient climbs a full 64-unit
    level without a >8 seam at the top. Growth deepens ONLY the two flow-axis ENDS (the cells at
    the extreme low/high projection) one cell into the adjacent plateau -- this lengthens the run
    while keeping the ramp a clean narrow channel (NOT a blob, which would break the rampList quad
    alignment, see findings S4.4). Base pads are never eaten (pad-pad lanes too short to fit a ramp
    are flattened upstream by the skeleton's level-constrained edge builder). Deterministic,
    symmetric-in -> symmetric-out.
    Ramps boxed in by pads/void that can't reach run_min are left for the relief validator."""
    yy, xx = np.mgrid[0 : elev.shape[0], 0 : elev.shape[1]].astype(np.float64)
    is_pad = pad_level >= 0
    for _ in range(run_min + 6):                      # bounded; converges well before this
        lab, n = ndimage.label(ramp, structure=_STRUCT8)
        changed = False
        for cid in range(1, n + 1):
            comp = lab == cid
            csz = int(comp.sum())
            if csz < 3 or csz > 90:
                continue                              # too small to matter / already big (don't blob)
            levels = _ramp_neighbor_levels(comp, walk, elev, ramp)
            if len(levels) != 2 or (levels[1] - levels[0]) != 1:
                continue                              # flat / multi-level: not our job here
            lo, hi = levels
            nb = ndimage.binary_dilation(comp, _STRUCT8) & walk & ~ramp
            lo_nb, hi_nb = nb & (elev == lo), nb & (elev == hi)
            if not lo_nb.any() or not hi_nb.any():
                continue
            if _ramp_bfs_run(comp, lo_nb, hi_nb) >= run_min:
                continue
            # flow axis: low-plateau centroid -> high-plateau centroid
            ux = xx[hi_nb].mean() - xx[lo_nb].mean()
            uy = yy[hi_nb].mean() - yy[lo_nb].mean()
            nrm = math.hypot(ux, uy) or 1.0
            ux, uy = ux / nrm, uy / nrm
            proj = xx * ux + yy * uy
            pc = proj[comp]
            pmin, pmax = pc.min(), pc.max()
            grow = ndimage.binary_dilation(comp, _STRUCT8)
            # deepen the LOW end into low plateau (cells at/below the low projection extreme) and the
            # HIGH end into high plateau (at/above the high extreme); only the ends -> no widening.
            ext_lo = grow & lo_nb & (proj <= pmin + 0.5) & ~is_pad
            ext_hi = grow & hi_nb & (proj >= pmax - 0.5) & ~is_pad
            ext = ext_lo | ext_hi
            if not ext.any():
                continue                              # boxed in by pads/void -> validator rejects
            ramp[ext] = True
            changed = True
        if not changed:
            break


def _rectangularize_ramps(walk: np.ndarray, elev: np.ndarray, ramp: np.ndarray,
                          pad_level: np.ndarray, choke_w: float) -> int:
    """Crop every single-level ramp to a CLEAN UNIFORM-WIDTH straight band along its snapped
    uphill axis, so the exporter's fixed-width <ramp> quad + 1D gradient cover it EXACTLY.

    ``_lengthen_ramps`` deepens each end by grabbing the *full-width* plateau seam (all cells at the
    low/high projection extreme), which re-flares the ends that ``_straighten_ramps`` had narrowed --
    leaving an HOURGLASS: wide mouths, pinched waist (see the seed-234 central ramp: 12 cells at the
    ends, 4 at the waist). A rampList quad is a fixed-width rectangle, so the engine cannot reconcile
    a 12-wide quad with a 4-wide waist: it treats the un-covered cells as cliff, producing the
    one-cell untraversable/unbuildable seam observed in-game (findings §4.4). This FINAL pass (run
    after lengthening) keeps only the cells within ``choke_w/2`` of the centreline -- through the
    component centroid, along the snapped low->high axis -- and RE-FLOORS the off-band flare cells to
    their nearer plateau (low near the low mouth, high near the high mouth). That leaves a solid
    rectangle flanked by solid ground (no void holes, still buildable) whose cross-section lines are
    exactly the gradient's isolines. Base pads are never touched. Run length is preserved (we only
    crop perpendicular). Deterministic + symmetric-in => symmetric-out. Returns cells re-floored."""
    h, w = walk.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    is_pad = pad_level >= 0
    lab, n = ndimage.label(ramp, structure=_STRUCT8)
    refloored = 0
    for cid in range(1, n + 1):
        comp = lab == cid
        levels = _ramp_neighbor_levels(comp, walk, elev, ramp)
        if len(levels) != 2 or (levels[1] - levels[0]) != 1:
            continue                                    # not a clean single-step ramp -> leave it
        lo, hi = levels
        nb = ndimage.binary_dilation(comp, _STRUCT8) & walk & ~ramp
        lo_nb, hi_nb = nb & (elev == lo), nb & (elev == hi)
        if not lo_nb.any() or not hi_nb.any():
            continue
        # snapped uphill axis (low-plateau centroid -> high-plateau centroid), same snap the
        # exporter uses for the quad and the 1D gradient.
        ux = xx[hi_nb].mean() - xx[lo_nb].mean()
        uy = yy[hi_nb].mean() - yy[lo_nb].mean()
        nrm = math.hypot(ux, uy) or 1.0
        ux, uy = _snap8(ux / nrm, uy / nrm)
        # DIAGONAL ramps only: skip. Their cells are sqrt2 apart, so a full 64-unit climb needs a
        # ~11.3-cell projection run; a diagonal band already tapers to its choke and cropping its
        # end corners would shorten that run below what the exporter's 8/cell gradient needs to
        # reach the high plateau (severing it). The hourglass flare is a wide-CARDINAL-ramp problem.
        if ux != 0.0 and uy != 0.0:
            continue
        cx, cy = xx[comp].mean(), yy[comp].mean()
        perp = np.abs((xx - cx) * (-uy) + (yy - cy) * ux)   # distance to the centreline
        proj = xx * ux + yy * uy
        pmid = float(proj[comp].mean())
        off = comp & (perp > choke_w / 2.0 + 0.5) & ~is_pad
        if not off.any():
            continue
        # re-floor off-band cells to the nearer plateau: low-mouth flares -> low level (adjacent to
        # the low plateau they fan out of, step 0), high-mouth flares -> high level. This keeps the
        # ground solid (buildable, no void hole) and creates no >8 seam (a low cell only ever abuts
        # the low mouth's <=72 band cell; a high cell only the >=120 band cell).
        to_lo = off & (proj <= pmid)
        to_hi = off & (proj > pmid)
        elev[to_lo] = lo
        elev[to_hi] = hi
        ramp[off] = False
        refloored += int(off.sum())
    return refloored


def _canonicalize_ramps(walk: np.ndarray, elev: np.ndarray, ramp: np.ndarray,
                        pad_level: np.ndarray, locked: np.ndarray, canvas: np.ndarray,
                        choke_w: float, run: float, min_run: int = 4) -> bool:
    """Replace every ramp component with ONE clean narrow staircase between the two DOMINANT
    plateau fragments it bridges, walling the rest -- the definitive blob killer (findings §4.4).

    A blob is a FUSED JUNCTION FRONT: several staircases + grown rooms + repair-carves converge, so
    the component is wide (perpW≫choke), its low seam never reaches its high seam *through* the ramp
    (BFS ``run==0``), and it abuts several disjoint low-plateau fragments. One straight ``<ramp>``
    quad cannot cover that, and post-hoc perpendicular cropping only re-floors cells that
    ``cliff_cleanup`` then walls (measured: drops yield) -- you cannot crop a broad front into a
    valid bridge. Instead, for each not-yet-canonical component we (1) find the two dominant
    (largest) plateau fragments on the low/high side, (2) WALL the whole blob, and (3) carve a fresh
    clean straight staircase (via ``_paint_ramp_edge``, choke-wide, ``run``-long, snapped to the same
    8-way axis the exporter uses) between the dominant fragments' nearest cells -- guaranteeing one
    plateau per end and a full gradient the quad covers. Carved staircase cells are LOCKED so later
    cleanup/repair/regularize passes never re-widen or demote them. Components that already read as a
    clean narrow single-step staircase are locked in place and skipped (idempotent). Components that
    are not a single-step interface at all (flat, or >1-level span) are walled for the connectivity
    repair to reconnect. Runs INSIDE ``paint`` (before the final ``_symmetrize_rot180``), so any
    per-half asymmetry it introduces is erased by the mirror-copy. Returns True if it changed geometry
    (walled/carved), so the caller can iterate cleanup+repair to a fixpoint."""
    h, w = walk.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    lab, n = ndimage.label(ramp, structure=_STRUCT8)
    changed = False
    for cid in range(1, n + 1):
        comp = lab == cid
        if (comp & locked).any():
            continue                                    # already a locked canonical staircase
        if int(comp.sum()) < 3:
            ramp[comp] = False                          # tiny fragment -> demote to plateau
            changed = True
            continue
        nb = ndimage.binary_dilation(comp, _STRUCT8) & walk & ~ramp
        levels = sorted({int(v) for v in elev[nb]}) if nb.any() else []
        if len(levels) != 2 or (levels[1] - levels[0]) != 1:
            walk[comp] = False                          # not a single-step interface -> wall it
            ramp[comp] = False
            changed = True
            continue
        lo, hi = levels
        lo_nb, hi_nb = nb & (elev == lo), nb & (elev == hi)
        lo_lab, ln = ndimage.label(lo_nb, structure=_STRUCT8)
        hi_lab, hn = ndimage.label(hi_nb, structure=_STRUCT8)
        if not ln or not hn:
            continue
        # snapped uphill axis + perpendicular width of the current component
        ux = xx[hi_nb].mean() - xx[lo_nb].mean()
        uy = yy[hi_nb].mean() - yy[lo_nb].mean()
        nrm = math.hypot(ux, uy) or 1.0
        ux, uy = _snap8(ux / nrm, uy / nrm)
        cx, cy = xx[comp].mean(), yy[comp].mean()
        perp = np.abs((xx[comp] - cx) * (-uy) + (yy[comp] - cy) * ux)
        pw = float(perp.max()) * 2.0
        run_cur = _ramp_bfs_run(comp, lo_nb, hi_nb)
        if ln == 1 and hn == 1 and pw <= choke_w + 2.5 and run_cur >= min_run:
            locked[comp] = True                         # already clean -> lock in place, no churn
            continue
        # rebuild: orient along the dominant (largest) plateau fragments' axis and carve a staircase
        # CENTRED on the blob that extends run/2 into EACH plateau -- a fixed physical run (not
        # clamped to how close the plateaus abut), so the exporter's 8/cell gradient always has
        # enough depth to climb the full level (a run==0 staircase is a 64-unit wall; findings §4.4).
        lo_dom = lo_lab == (int(np.bincount(lo_lab.ravel())[1:].argmax()) + 1)
        hi_dom = hi_lab == (int(np.bincount(hi_lab.ravel())[1:].argmax()) + 1)
        ux = xx[hi_dom].mean() - xx[lo_dom].mean()
        uy = yy[hi_dom].mean() - yy[lo_dom].mean()
        nrm = math.hypot(ux, uy) or 1.0
        ux, uy = _snap8(ux / nrm, uy / nrm)
        half = max(run, min_run + 1) / 2.0
        lo_end = (cx - half * ux, cy - half * uy)       # reaches into the low plateau
        hi_end = (cx + half * ux, cy + half * uy)       # reaches into the high plateau
        walk[comp] = False                              # erase the blob
        ramp[comp] = False
        before = ramp.copy()
        _paint_corridor(walk, elev, ramp, lo_end, hi_end, choke_w, lo, hi, canvas,
                        ramp_t0=0.0, ramp_t1=1.0)
        locked |= ramp & ~before                        # lock the freshly carved staircase
        changed = True
    return changed


def base_clearances(mapir: MapIR) -> list[tuple[BaseNode, float, int]]:
    """For each real base, the best buildable-clearance radius (tiles) available in a small
    window around its center, plus the floor (elevation) the pad sits on. Used to assert
    that every base can host a nexus/CC + mineral line + geyser on a single floor."""
    edt_build = ndimage.distance_transform_edt(mapir.buildable)
    out: list[tuple[BaseNode, float, int]] = []
    h, w = mapir.buildable.shape
    for b in mapir.bases:
        if b.kind not in _REAL_BASES:
            continue
        iy, ix = int(round(b.y)), int(round(b.x))
        y0, y1 = max(0, iy - 3), min(h, iy + 4)
        x0, x1 = max(0, ix - 3), min(w, ix + 4)
        win = edt_build[y0:y1, x0:x1]
        best = float(win.max()) if win.size else 0.0
        out.append((b, best, int(mapir.elevation[iy, ix])))
    return out


def validate_base_pads(mapir: MapIR, min_clear: float = 4.0) -> list[str]:
    """Return a list of human-readable failures: any real base whose buildable pad is
    smaller than ``min_clear`` (radius, tiles). Empty list == every base is placeable."""
    fails: list[str] = []
    for b, clear, floor in base_clearances(mapir):
        if clear < min_clear:
            fails.append(
                f"{b.kind.value} @({b.x:.0f},{b.y:.0f}) buildable clearance "
                f"{clear:.1f} < {min_clear:.1f} (floor {floor})"
            )
    return fails


def ramp_cleanliness(mapir: MapIR):
    """For each ramp, report the plateau levels it touches and how many distinct plateau
    *components* touch it at each level, plus whether it is a clean single ascending strip.
    A ramp is CLEAN iff it touches exactly two levels one step apart and exactly one plateau
    component at each -- i.e. one low plateau <-> one high plateau, which is the only thing
    the SC2 exporter can turn into a single valid ramp."""
    walk = mapir.walkable
    elev = mapir.elevation.astype(int)
    ramp = np.zeros_like(walk)
    for r in mapir.ramps:
        for x, y in r.cells:
            ramp[y, x] = True
    plat = walk & ~ramp
    struct8 = np.ones((3, 3), dtype=bool)
    out = []
    for r in mapir.ramps:
        comp = np.zeros_like(walk)
        for x, y in r.cells:
            comp[y, x] = True
        nb = ndimage.binary_dilation(comp) & plat
        levels = sorted({int(v) for v in elev[nb]}) if nb.any() else []
        counts: dict[int, int] = {}
        for L in levels:
            slab, _ = ndimage.label(plat & (elev == L), structure=struct8)
            counts[L] = len({int(v) for v in slab[nb & (elev == L)]} - {0})
        clean = (
            len(levels) == 2
            and levels[1] - levels[0] == 1
            and counts.get(levels[0]) == 1
            and counts.get(levels[1]) == 1
        )
        out.append((r, levels, counts, clean))
    return out


def validate_ramps(mapir: MapIR) -> list[str]:
    """Return a list of human-readable failures: any ramp that is not a clean single
    ascending strip (one low plateau <-> one high plateau). Empty list == all ramps clean."""
    fails: list[str] = []
    for r, levels, counts, clean in ramp_cleanliness(mapir):
        if not clean:
            x, y = r.top
            fails.append(
                f"ramp @~({x:.0f},{y:.0f}) levels={levels} plateaus/level={counts} "
                f"(want 2 levels 1 apart, 1 plateau each)"
            )
    return fails


def _place_resources(skel, level, elev, ramp, canvas, pa, cfg):
    """Deterministic, legal resource layout for every real base.

    Each MAIN/NATURAL/BASE gets ``cfg.res_minerals`` mineral fields in a shallow arc plus
    ``cfg.res_geysers`` geysers flanking it, all on the base's own floor, off its townhall
    footprint, and not overlapping another patch. The arc faces *away* from the base's
    corridors (mineral line at the back). Returns ``(resources, res_idx)`` where
    ``res_idx[skel_index] = [indices into resources]``."""
    bases = skel.bases
    cx0, cy0 = pa.x + pa.width / 2.0, pa.y + pa.height / 2.0
    h, w = elev.shape

    adj: dict[int, list[int]] = {i: [] for i in range(len(bases))}
    for e in skel.edges:
        adj[e.a].append(e.b)
        adj[e.b].append(e.a)

    def orient(i: int) -> float:
        # point the mineral line AWAY from the mean direction of this base's corridors
        vx = vy = 0.0
        for j in adj[i]:
            dx, dy = bases[j].x - bases[i].x, bases[j].y - bases[i].y
            d = math.hypot(dx, dy) or 1.0
            vx += dx / d
            vy += dy / d
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:      # no edges: face away from map center
            vx, vy = bases[i].x - cx0, bases[i].y - cy0
        return math.atan2(-vy, -vx)

    occupied: set[tuple[int, int]] = set()

    def reserve(ix: int, iy: int) -> None:
        occupied.add((ix, iy))

    def far_enough(ix: int, iy: int, gap: float) -> bool:
        g = int(math.ceil(gap))
        for oy in range(iy - g, iy + g + 1):
            for ox in range(ix - g, ix + g + 1):
                if (ox, oy) in occupied and math.hypot(ox - ix, oy - iy) < gap:
                    return False
        return True

    def valid_cell(ix: int, iy: int, lv: int) -> bool:
        return (
            0 <= ix < w and 0 <= iy < h
            and canvas[iy, ix] and not ramp[iy, ix]
            and int(elev[iy, ix]) == lv
        )

    def snap(x: float, y: float, lv: int, gap: float,
             bx: float = 0.0, by: float = 0.0, min_r: float = 0.0):
        # nearest valid, unoccupied cell to (x,y) within a small spiral, but never closer than
        # ``min_r`` to the base center -- python-sc2 rejects a townhall within 6 of any mineral /
        # 7 of any geyser, so a patch that snaps inward into the pad would make the base unplaceable
        best = None
        for r in range(0, 5):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    ix, iy = int(round(x)) + dx, int(round(y)) + dy
                    if min_r and math.hypot(ix - bx, iy - by) < min_r:
                        continue
                    if valid_cell(ix, iy, lv) and far_enough(ix, iy, gap):
                        d = math.hypot(ix - x, iy - y)
                        if best is None or d < best[0]:
                            best = (d, ix, iy)
            if best is not None:
                return best[1], best[2]
        return None

    def layout(i: int) -> list[tuple[ResourceKind, int, int]]:
        bx, by = bases[i].x, bases[i].y
        # use the ACTUAL floor the townhall will sit on (elevation at the base center), not the
        # intended level -- cleanup/repair may have settled the center a level off, and the
        # mineral line must share the townhall's floor.
        lv = int(elev[int(round(by)), int(round(bx))])
        th0 = orient(i)
        # reserve townhall footprint so patches never sit on the nexus/CC
        tr = int(math.ceil(cfg.res_townhall_half))
        for oy in range(-tr, tr + 1):
            for ox in range(-tr, tr + 1):
                reserve(int(round(bx)) + ox, int(round(by)) + oy)
        out: list[tuple[ResourceKind, int, int]] = []
        arc = math.radians(cfg.res_mineral_arc_deg)
        n = max(1, cfg.res_minerals)
        for k in range(n):
            frac = (k / (n - 1) - 0.5) if n > 1 else 0.0
            ang = th0 + frac * arc
            # outward-only ladder: never place a patch *inside* the ring (that would put it
            # closer than 6 to the pocket and fail the finder's townhall-clearance check). If the
            # patch's exact angle is boxed in (terrain edge / ramp / occupied) at every radius,
            # retry with a few small angle jitters before giving up -- otherwise a single blocked
            # patch drops the line to 7/8 and fails M5 (e.g. seed 34).
            placed = False
            for dang in (0.0, math.radians(4.0), math.radians(-4.0),
                         math.radians(8.0), math.radians(-8.0)):
                for rr in (cfg.res_mineral_ring, cfg.res_mineral_ring + 1.0,
                           cfg.res_mineral_ring + 2.0, cfg.res_mineral_ring + 3.0):
                    s = snap(bx + rr * math.cos(ang + dang), by + rr * math.sin(ang + dang), lv,
                             cfg.res_min_gap, bx, by, cfg.res_townhall_clear)
                    if s:
                        reserve(*s)
                        out.append((ResourceKind.MINERAL, s[0], s[1]))
                        placed = True
                        break
                if placed:
                    break
        flank = arc / 2.0 + math.radians(cfg.res_gas_flank_deg)
        for gi in range(cfg.res_geysers):
            side = -1 if gi % 2 == 0 else 1
            ang = th0 + side * flank
            for rr in (cfg.res_gas_ring, cfg.res_gas_ring + 1.0, cfg.res_gas_ring + 2.0):
                s = snap(bx + rr * math.cos(ang), by + rr * math.sin(ang), lv,
                         cfg.res_min_gap + 0.5, bx, by, cfg.res_townhall_clear + 1.0)
                if s:
                    reserve(*s)
                    out.append((ResourceKind.GEYSER, s[0], s[1]))
                    break
        return out

    resources: list[Resource] = []
    res_idx: dict[int, list[int]] = {}

    def emit(kind: ResourceKind, x: float, y: float) -> int:
        if kind == ResourceKind.MINERAL:
            resources.append(Resource(ResourceKind.MINERAL, "MineralField",
                                      float(x), float(y), cfg.res_mineral_amount))
        else:
            resources.append(Resource(ResourceKind.GEYSER, "VespeneGeyser",
                                      float(x), float(y), cfg.res_gas_amount))
        return len(resources) - 1

    # Place each real base independently, validity-checked against its OWN floor. (Mirroring a
    # canonical base's patch positions is tempting for exact symmetry, but the rasterized
    # terrain is not pixel-perfectly symmetric, so a mirrored patch can land one cell onto a
    # different floor; per-base placement keeps every patch legal, and the layout is
    # near-symmetric anyway because the geometry and corridor orientation are symmetric.)
    for i, b in enumerate(bases):
        if b.kind not in _REAL_BASES:
            continue
        res_idx[i] = [emit(k, x, y) for k, x, y in layout(i)]
    return resources, res_idx


def _bases_connected(walk: np.ndarray, bases) -> tuple[bool, np.ndarray]:
    lab, _ = ndimage.label(walk)
    ids = set()
    for b in bases:
        v = int(lab[int(round(b.y)), int(round(b.x))])
        if v == 0:
            return False, lab
        ids.add(v)
    return (len(ids) == 1), lab


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def rasterize(skel: Skeleton, seed: int | None = None, cfg: RasterConfig | None = None) -> MapIR:
    cfg = cfg or RasterConfig()
    rng = np.random.default_rng((seed if seed is not None else skel.seed) * 2 + 1)

    h, w = skel.grid_h, skel.grid_w
    pa = skel.playable
    bases = skel.bases
    mirror = _pair_mirror(bases)
    pts = [(b.x, b.y) for b in bases]

    # box orientation must reflect for lateral-symmetry maps (rot180 preserves angle)
    sym = getattr(skel, "symmetry", "rot180")

    def mirror_angle(a: float) -> float:
        if sym == "mirror_lr":
            return math.pi - a
        if sym == "mirror_ud":
            return -a
        return a

    # ---- levels + edges are OWNED BY THE SKELETON now (generate/skeleton._assign_levels) ----
    # The skeleton decided each node's tier and rebuilt its graph under the level/ramp rules; here
    # we just read them. Nullified cross-level junctions survive in ``bases`` as degree-0 pseudo-
    # nodes (no incident edges), so we detect them from the edge set and skip painting them.
    level = [int(b.level) for b in bases]
    edges = skel.edges
    _deg = [0] * len(bases)
    for e in edges:
        _deg[e.a] += 1
        _deg[e.b] += 1
    dead_junctions = {i for i, b in enumerate(bases)
                      if b.kind == BaseKind.JUNCTION and _deg[i] == 0}

    # ---- per-pair box geometry (rx, ry, angle), identical for a node and its mirror ----
    # rooms derive from the node's scalar width: short half-extent = width/2, long side is
    # width/2 * aspect. Capped nodes (MAIN/NATURAL) never exceed box_cap even after growth.
    rx0 = [0.0] * len(bases)
    ry0 = [0.0] * len(bases)
    ang = [0.0] * len(bases)
    is_capped = [False] * len(bases)

    done: set[int] = set()
    for i, b in enumerate(bases):
        if i in done:
            rx0[i], ry0[i] = rx0[mirror[i]], ry0[mirror[i]]
            ang[i] = mirror_angle(ang[mirror[i]])
            is_capped[i] = is_capped[mirror[i]]
            continue
        if b.kind in (BaseKind.MAIN, BaseKind.NATURAL):
            is_capped[i] = True
        short = max(cfg.min_width, b.width) / 2.0
        if b.kind == BaseKind.JUNCTION:
            short *= cfg.junction_scale
        a_ratio = float(rng.uniform(*cfg.aspect))
        rx0[i], ry0[i] = short, short * a_ratio
        ang[i] = float(rng.uniform(0.0, math.pi))
        done.add(i)
        done.add(mirror[i])

    # ---- per-edge corridor width comes straight from the (symmetric) skeleton edge ----
    def edge_key(a, b):
        ma, mb = mirror[a], mirror[b]
        return tuple(sorted([tuple(sorted((a, b))), tuple(sorted((ma, mb)))]))

    width_by_key: dict = {}
    for e in edges:
        k = edge_key(e.a, e.b)
        ew = max(cfg.min_width, e.width)
        if level[e.a] != level[e.b]:
            # level change -> render as a narrow choke so one <ramp> quad fully bridges it
            # (a corridor-wide ramp only gets partial coverage; see findings S4.4).
            ew = min(ew, cfg.ramp_choke_width)
        width_by_key.setdefault(k, ew)

    # each node's widest incident corridor (so we can guarantee "node wider than its edges")
    max_edge_w = [0.0] * len(bases)
    for e in edges:
        ew = width_by_key[edge_key(e.a, e.b)]
        max_edge_w[e.a] = max(max_edge_w[e.a], ew)
        max_edge_w[e.b] = max(max_edge_w[e.b], ew)

    # canvas: walkable is only ever painted inside this inset playable rect
    b0 = cfg.border
    canvas = np.zeros((h, w), dtype=bool)
    canvas[pa.y + b0 : pa.y + pa.height - b0, pa.x + b0 : pa.x + pa.width - b0] = True

    def room_size(i: int, mult: float) -> tuple[float, float]:
        # MAIN/NATURAL are fixed-size bases: they use a modest fixed growth (not the
        # openness multiplier) so they stay compact and leave a real gap for a clean short
        # ramp instead of ballooning into their neighbor. Interior rooms grow to fill space.
        m = cfg.anchor_growth if is_capped[i] else mult
        rx, ry = rx0[i] * m, ry0[i] * m
        if is_capped[i]:
            rx, ry = min(rx, cfg.box_cap), min(ry, cfg.box_cap)
        floor = max(rx0[i], max_edge_w[i] / 2.0)
        return max(rx, floor), max(ry, floor)

    terrace = (cfg.terrace_mode or bool(os.environ.get("TERRACE_MODE"))) \
        and not os.environ.get("NO_TERRACE")

    def paint(mult: float):
        """Render the whole map at room-growth ``mult``; returns (walk, elev, ramp)."""
        walk = np.zeros((h, w), dtype=bool)
        elev = np.zeros((h, w), dtype=np.int16)
        ramp = np.zeros((h, w), dtype=bool)
        passage = np.zeros((h, w), dtype=bool)   # same-level flat lanes (protected in cleanup)
        locked = np.zeros((h, w), dtype=bool)     # canonical staircase cells (never re-widened)

        # 1) corridors first (chokes/ramps). Same-level lanes are flat corridors; a level change
        #    is a CLEAN STRAIGHT STAIRCASE snapped to one of the 8 ramp directions, flanked by
        #    same-level flat approach lanes (gold-standard ramp shape -- see _paint_ramp_edge /
        #    findings S4.4), so the exporter's straight quad + 1D gradient bridges it fully.
        for e in edges:
            if terrace and level[e.a] != level[e.b]:
                continue     # TERRACE mode: cross-level crossings are cut later as isolated
                             # staircases through the walls, never painted per-edge here.
            _paint_ramp_edge(
                walk, elev, ramp, pts[e.a], pts[e.b], level[e.a], level[e.b],
                width_by_key[edge_key(e.a, e.b)], canvas, cfg.ramp_run, passage=passage,
            )
        # 2) rooms on top (reclaim flat ends, clearing stray ramp flags). Grow only rooms
        #    (corridors stay fixed => chokes). Floor each half-extent so the node is never
        #    narrower than its widest incident corridor (keeps "node is widest locally").
        for i in range(len(bases)):
            if i in dead_junctions:
                continue                         # nullified cross-level junction -> not a region
            rx, ry = room_size(i, mult)
            _paint_box(walk, elev, ramp, pts[i], rx, ry, ang[i], level[i], canvas)

        # 2b) guarantee a flat buildable PAD at every real base (MAIN/NATURAL/BASE): an
        #     axis-aligned square at the base's level, clearing any stray ramp under it, so
        #     the nexus/CC + mineral line + geyser always fit on one floor. ROOM/JUNCTION are
        #     routing-only (no pad). ``pad_mask`` marks pad cells so cliff-cleanup never
        #     erodes a pad -- it walls the *other* (non-pad) side of a pad seam instead.
        pad = cfg.base_pad_half
        pr = int(math.ceil(pad))
        pad_level = np.full((h, w), -1, dtype=np.int16)  # a pad cell's floor, else -1

        def stamp_pads() -> None:
            for i in range(len(bases)):
                if bases[i].kind in _REAL_BASES:
                    _paint_box(walk, elev, ramp, pts[i], pad, pad, 0.0, level[i], canvas)
                    iy, ix = int(round(bases[i].y)), int(round(bases[i].x))
                    pad_level[max(0, iy - pr):iy + pr + 1,
                              max(0, ix - pr):ix + pr + 1] = level[i]

        stamp_pads()

        # helpers for the cleanup/repair convergence below -------------------
        def cliff_cleanup() -> None:
            # Any non-ramp adjacency between two *different* elevations becomes a wall, so two
            # floors never touch as walkable (that renders as high terrain covering the lower
            # floor). A ramp (plus a 2-cell halo) is the only legal seam. A base pad is never
            # eroded (else repeated cleanups next to higher ground would eat it away): a pad/
            # neighbour seam is resolved by walling the *other* side. Only cells still sitting
            # at their pad's level are protected -- a lane repainted through a pad is not.
            protect = ndimage.binary_dilation(ramp, iterations=2)
            padprot = (pad_level >= 0) & (elev == pad_level)
            # same-level passages are protected too (only while still at their painted level -- a
            # passage overwritten by a later room/pad is no longer a passage): cleanup walls the
            # abutting plateau side instead of pinching the lane.
            passprot = passage & walk
            # locked canonical staircases (plus a 1-cell halo) are immutable -- a late cleanup must
            # never pinch a blob-killer staircase back into a seam.
            lockprot = ndimage.binary_dilation(locked, iterations=1)
            prot = protect | padprot | passprot | lockprot
            free = ~prot
            db = _diff_boundary(elev, walk)
            # Default: wall the HIGHER side of a seam (the cliff top) so the lower floor stays
            # continuous and low passages aren't pinched off. Exception: where a free cell sits
            # next to a PROTECTED cell of a different level (a pad/ramp edge), we can't wall the
            # protected side, so wall the free cell regardless of which is higher.
            higher = _higher_boundary(elev, walk)
            # a free db cell adjacent to a PROTECTED cell of a different elevation must be walled
            # (the protected side can't be), even if it's the lower side.
            npf = np.zeros_like(walk, dtype=bool)
            for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
                npf |= np.roll(prot, sh, axis=ax) & (elev != np.roll(elev, sh, axis=ax))
            kill = db & free & (higher | npf)
            walk[kill] = False

        def assert_main_chokes(skip_locked: bool = False) -> None:
            # Re-open each MAIN's one legit choke (main->natural) after a cleanup so a close
            # natural / grown rooms can never wall the main's only exit. It's the main's sole
            # edge, so this never adds a second exit. Once a canonical staircase already serves the
            # choke (``skip_locked`` in the canonicalize loop), re-painting a fresh UNLOCKED ramp
            # over it only fuses a blob back onto the locked core (which canonicalize then can't
            # touch); skip those -- the locked staircase already keeps the exit open.
            for e in edges:
                if bases[e.a].kind != BaseKind.MAIN and bases[e.b].kind != BaseKind.MAIN:
                    continue
                if skip_locked:
                    mx = int(round((pts[e.a][0] + pts[e.b][0]) / 2.0))
                    my = int(round((pts[e.a][1] + pts[e.b][1]) / 2.0))
                    r = int(math.ceil(cfg.ramp_run))
                    y0, y1 = max(0, my - r), min(h, my + r + 1)
                    x0, x1 = max(0, mx - r), min(w, mx + r + 1)
                    if locked[y0:y1, x0:x1].any():
                        continue
                _paint_ramp_edge(
                    walk, elev, ramp, pts[e.a], pts[e.b], level[e.a], level[e.b],
                    width_by_key[edge_key(e.a, e.b)], canvas, cfg.ramp_run, passage=passage,
                )

        # helpers for connectivity repair (mirror-safe) ----------------------
        def nearest_walkable_idx():
            return ndimage.distance_transform_edt(~walk, return_indices=True)[1]

        def base_cell(idx, i):
            iy, ix = int(round(bases[i].y)), int(round(bases[i].x))
            return int(idx[1][iy, ix]), int(idx[0][iy, ix])  # (x, y)

        cx0, cy0 = pa.x + pa.width / 2.0, pa.y + pa.height / 2.0

        def mirror_pt(x, y):
            if sym == "mirror_lr":
                return (2 * cx0 - x, y)
            if sym == "mirror_ud":
                return (x, 2 * cy0 - y)
            return (2 * cx0 - x, 2 * cy0 - y)

        wv = max(cfg.min_width, 4.0)

        def repair_one() -> bool:
            # Carve one link from a disconnected base to the mains' component. Prefer the
            # nearest root cell on the base's *own* level (a flat, same-level passage); fall
            # back to the nearest cell of any level (a real ramp) only when a same-level
            # target is much farther / absent. Source is the base's own center (never a
            # foreign room). Returns True if it carved a link (made progress).
            lab = _level_components(walk, elev, ramp)   # ENGINE-traversable components
            idx = nearest_walkable_idx()
            root = lab[base_cell(idx, 0)[1], base_cell(idx, 0)[0]]  # main1's component
            if all(lab[base_cell(idx, i)[1], base_cell(idx, i)[0]] == root
                   for i in range(len(bases))):
                return False
            root_mask = lab == root
            dt_any, ridx = ndimage.distance_transform_edt(
                ~root_mask, return_distances=True, return_indices=True)
            for i in range(len(bases)):
                sx, sy = base_cell(idx, i)
                if lab[sy, sx] == root:
                    continue
                bl = int(level[i])
                bx, by = bases[i].x, bases[i].y
                same = root_mask & (elev == bl)
                dx, dy, lv = int(ridx[1][sy, sx]), int(ridx[0][sy, sx]), None
                if same.any():
                    dt_s, sidx = ndimage.distance_transform_edt(
                        ~same, return_distances=True, return_indices=True)
                    if dt_s[sy, sx] <= 1.6 * dt_any[sy, sx] + 12.0:
                        dx, dy, lv = int(sidx[1][sy, sx]), int(sidx[0][sy, sx]), bl
                if lv is None:
                    # cross-level: this link is a real ramp. Aim it at a root cell that is NOT
                    # adjacent to an existing ramp, so the fresh staircase doesn't fuse into an L/T
                    # with a nearby one (an unauthorable blob). Only if a ramp-free target isn't
                    # much farther; otherwise fall back to the nearest root cell.
                    # only carve toward a root cell at most ONE level away -- a 2-level target would
                    # paint an unauthorable 0->2 ramp that gets walled next pass (oscillation). Prefer
                    # a ramp-free, <=1-level target so the fresh staircase is clean and single-step.
                    near = root_mask & (np.abs(elev.astype(np.int32) - bl) <= 1)
                    if not near.any():
                        continue                     # no reachable <=1-level target -> skip this base
                    rampfree = near & ~ndimage.binary_dilation(ramp, iterations=3)
                    target = rampfree if rampfree.any() else near
                    dt_t, tidx = ndimage.distance_transform_edt(
                        ~target, return_distances=True, return_indices=True)
                    dx, dy = int(tidx[1][sy, sx]), int(tidx[0][sy, sx])
                    lv = int(elev[dy, dx])
                # Start the lane at the base's pad *edge* (toward the target), never through
                # its center -- otherwise a repair ramp bisects the pad and halves its
                # buildable area. ROOM/JUNCTION have no pad, so they start at their center.
                off = (pad + 1.0) if bases[i].kind in _REAL_BASES else 0.0
                seg = math.hypot(dx - bx, dy - by)
                if seg > off + 1.0:
                    ux, uy = (dx - bx) / seg, (dy - by) / seg
                    sx0, sy0 = bx + off * ux, by + off * uy
                else:
                    sx0, sy0 = bx, by
                # cross-level repair -> clean straight staircase (matches the exporter's quad);
                # same-level -> plain flat lane. _paint_ramp_edge handles both.
                _paint_ramp_edge(walk, elev, ramp, (sx0, sy0), (dx, dy), bl, lv,
                                 min(wv, cfg.ramp_choke_width), canvas, cfg.ramp_run,
                                 passage=passage)
                _paint_ramp_edge(walk, elev, ramp, mirror_pt(sx0, sy0), mirror_pt(dx, dy),
                                 bl, lv, min(wv, cfg.ramp_choke_width), canvas, cfg.ramp_run,
                                 passage=passage)
                return True
            return False

        # ============================ TERRACE-FIRST PATH ============================
        # Solid same-level terraces are already painted (cross-level edges were skipped in step 1).
        # WALL every level boundary, then cut ONE choke-wide straight staircase per terrace-pair the
        # graph says must connect (well-separated, LOCKED). Ramp geometry is thus decoupled from edge
        # count -- clustered crossings can't fuse. Repair remaining splits by cutting more isolated
        # staircases, never by growing terrain. See findings §4.4.
        if terrace:
            # per-cut size variety (deterministic + identical for a cut and its mirror since both
            # halves are painted in one terrace_cut call from a single sample). Reseeded per paint()
            # so the auto-tune passes stay reproducible.
            size_rng = np.random.default_rng((seed if seed is not None else skel.seed) * 2 + 13)
            cw_lo, cw_hi = cfg.ramp_choke_range
            rn_lo, rn_hi = cfg.ramp_run_range

            def sample_size():
                return (float(size_rng.uniform(cw_lo, cw_hi)),
                        float(size_rng.uniform(rn_lo, rn_hi)))

            def wall_all() -> None:
                # Wall the HIGHER side of every different-level seam (cliff top), protecting only pads
                # and locked staircases (+halo). A free cell abutting a PROTECTED different-level cell
                # is walled regardless of height (can't wall the protected side). Iterate to a fixpoint.
                for _ in range(4):
                    padprot = (pad_level >= 0) & (elev == pad_level)
                    lockprot = ndimage.binary_dilation(locked, iterations=1)
                    prot = padprot | lockprot
                    npf = np.zeros_like(walk)
                    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
                        npf |= np.roll(prot, sh, axis=ax) & (elev != np.roll(elev, sh, axis=ax))
                    kill = _diff_boundary(elev, walk) & ~prot & (_higher_boundary(elev, walk) | npf)
                    if not kill.any():
                        break
                    walk[kill] = False

            def conn_labels() -> np.ndarray:
                # ENGINE-ACCURATE terrace connectivity. Label same-level GROUND (walk & ~ramp) into
                # terraces, then union two terraces ONLY through a ramp component that actually serves
                # them: a clean single-level staircase (own cells span exactly lo..lo+1) bridges its
                # low-plateau terrace to its high-plateau terrace, and NOTHING else. A staircase that
                # merely sits next to a third-level plateau does NOT connect it (the engine walls that
                # Δcliff>8 seam) -- which is exactly where _level_components over-reports. Multi-level
                # (fused) ramp components bridge nothing (unauthorable). Returns a per-cell terrace-root
                # label on ground cells (-1 elsewhere).
                ground = walk & ~ramp
                tl = np.full((h, w), -1, dtype=np.int64)
                nid = 0
                for L in (np.unique(elev[ground]) if ground.any() else []):
                    m = ground & (elev == L)
                    lab_, nn = ndimage.label(m)          # 4-connectivity, matching the engine oracle
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
                        continue                            # flat or multi-level -> bridges nothing
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

            def node_comp(i, lab):
                # a node's terrace = the terrace of the nearest GROUND cell at the node's own level
                own = walk & ~ramp & (elev == level[i])
                iy, ix = int(round(bases[i].y)), int(round(bases[i].x))
                if own[iy, ix]:
                    return int(lab[iy, ix])
                if not own.any():
                    return -1
                jdx = ndimage.distance_transform_edt(~own, return_indices=True)[1]
                return int(lab[jdx[0][iy, ix], jdx[1][iy, ix]])

            def pad_edge(i, toward):
                # Start a cut at a real base's PAD EDGE (toward the crossing), never its centre --
                # a lane through the centre bisects the buildable pad. Pad-less nodes start at centre.
                bx, by = bases[i].x, bases[i].y
                if bases[i].kind not in _REAL_BASES:
                    return (bx, by)
                d = math.hypot(toward[0] - bx, toward[1] - by) or 1.0
                off = pad + 1.0
                return (bx + off * (toward[0] - bx) / d, by + off * (toward[1] - by) / d)

            def _cut_one(pa_pt, pb_pt, la, lb, cw, rn) -> None:
                # One staircase with a GUARANTEED full run: place an rn-long ramp band centred on the
                # crossing (the endpoints' midpoint), oriented along the snapped low->high axis, so the
                # exporter's 8/cell gradient always has >=rn cells (>=8 ranks) to climb the level --
                # NEVER clamped to how close the two endpoints happen to be (a 2-cell gap gave a 16-24
                # cliff step per cell = an in-engine wall). Flat approach lanes at each plateau level
                # tie the band's ends back to the two endpoints (same-level, so they carve open ground).
                (p_lo, lo), (p_hi, hi) = ((pa_pt, la), (pb_pt, lb)) if la < lb else ((pb_pt, lb), (pa_pt, la))
                dx, dy = p_hi[0] - p_lo[0], p_hi[1] - p_lo[1]
                slen = math.hypot(dx, dy) or 1.0
                ux, uy = _snap8(dx / slen, dy / slen)
                cxp, cyp = (p_lo[0] + p_hi[0]) / 2.0, (p_lo[1] + p_hi[1]) / 2.0
                half = rn / 2.0
                lo_end = (cxp - half * ux, cyp - half * uy)
                hi_end = (cxp + half * ux, cyp + half * uy)
                _paint_corridor(walk, elev, ramp, p_lo, lo_end, cw, lo, lo, canvas)
                _paint_corridor(walk, elev, ramp, hi_end, p_hi, cw, hi, hi, canvas)
                _paint_corridor(walk, elev, ramp, lo_end, hi_end, cw, lo, hi, canvas,
                                ramp_t0=0.0, ramp_t1=1.0)

            def terrace_cut(pa_pt, pb_pt, la, lb) -> bool:
                # Carve a clean full-run straight staircase (+ its rot180 mirror) and LOCK the ramp
                # cells so wall_all never touches them. A cut whose ramp cells would TOUCH an existing
                # locked staircase is UNDONE: two stacked staircases (a 0->1 and a 1->2 across a level-1
                # boundary) must be separated by a real flat landing, else post-paint labelling fuses
                # them into one 0->2 component the engine can't climb (Δcliff>8). Does NOT re-stamp pads
                # (that would erase locked cells and re-trigger cuts forever). True iff it carved.
                if abs(la - lb) != 1:
                    return False          # only single-step crossings; a 0->2 must route via level 1
                cw, rn = sample_size()
                snap = (walk.copy(), elev.copy(), ramp.copy(), locked.copy())
                before = ramp.copy()
                _cut_one(pa_pt, pb_pt, la, lb, cw, rn)
                _cut_one(mirror_pt(*pa_pt), mirror_pt(*pb_pt), la, lb, cw, rn)
                grew = ramp & ~before
                if not grew.any():
                    return False
                if (ndimage.binary_dilation(grew, iterations=2) & (locked & ~grew)).any():
                    walk[:], elev[:], ramp[:], locked[:] = snap   # would fuse -> undo
                    if os.environ.get("TERRACE_DEBUG"):
                        print(f"  [cut UNDO fusion] L{la}->{lb}")
                    return False
                if os.environ.get("TERRACE_DEBUG"):
                    print(f"  [cut OK] {tuple(round(v) for v in pa_pt)}->"
                          f"{tuple(round(v) for v in pb_pt)} L{la}->{lb} cells={int(grew.sum())}")
                locked[grew] = True
                wall_all()
                return True

            stamp_pads()
            wall_all()

            # --- plan cuts from the cross-level edges, shortest first, deduped by terrace-pair ---
            def pkey_e(e):
                ma, mb = mirror[e.a], mirror[e.b]
                return tuple(sorted([tuple(sorted((e.a, e.b))), tuple(sorted((ma, mb)))]))

            cross = sorted(
                {pkey_e(e): e for e in edges if level[e.a] != level[e.b]}.values(),
                key=lambda e: math.hypot(pts[e.a][0] - pts[e.b][0], pts[e.a][1] - pts[e.b][1]))
            for e in cross:
                a, b = e.a, e.b
                mid = ((pts[a][0] + pts[b][0]) / 2.0, (pts[a][1] + pts[b][1]) / 2.0)
                # well-separation: skip if a locked staircase already sits within ~run of this crossing
                iy, ix = int(round(mid[1])), int(round(mid[0]))
                r = int(math.ceil(rn_hi))
                if locked[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1].any():
                    continue
                lab = conn_labels()
                if node_comp(a, lab) == node_comp(b, lab) >= 0:
                    continue                                  # terraces already joined -> no cut
                terrace_cut(pad_edge(a, pts[b]), pad_edge(b, pts[a]), level[a], level[b])

            # --- connectivity repair: cut extra isolated staircases until every base reaches main1 ---
            main1 = next(i for i, b in enumerate(bases) if b.kind == BaseKind.MAIN)
            for _ in range(2 * len(bases)):
                lab = conn_labels()
                root = node_comp(main1, lab)
                stragglers = [i for i in range(len(bases)) if bases[i].kind in _REAL_BASES
                              and node_comp(i, lab) != root]
                if not stragglers:
                    break
                rootmask = lab == root
                if not rootmask.any():
                    break
                # cut from the straggler nearest the root component toward its closest root cell
                # that is <=1 level away (a clean single-step staircase or a flat same-level lane)
                progressed = False
                dt_root, ridx = ndimage.distance_transform_edt(
                    ~rootmask, return_distances=True, return_indices=True)
                stragglers.sort(key=lambda i: dt_root[int(round(bases[i].y)), int(round(bases[i].x))])
                for i in stragglers:
                    iy, ix = int(round(bases[i].y)), int(round(bases[i].x))
                    near = rootmask & (np.abs(elev.astype(np.int32) - level[i]) <= 1)
                    if not near.any():
                        continue
                    ndx = ndimage.distance_transform_edt(~near, return_indices=True)[1]
                    ty, tx = int(ndx[0][iy, ix]), int(ndx[1][iy, ix])
                    if terrace_cut(pad_edge(i, (float(tx), float(ty))), (float(tx), float(ty)),
                                   level[i], int(elev[ty, tx])):
                        progressed = True
                        break
                if not progressed:
                    break

            return walk, elev, ramp
        # ============================ END TERRACE-FIRST PATH ========================

        # 3) converge to a state that is BOTH clean (no two floors touch) and connected.
        #    Each pass: wall illegal seams, re-open the main chokes, wall any seam the choke
        #    repaint introduced, then -- if still split -- carve one repair link. A repair
        #    lands on a same-level cell (flat) or via a protected ramp, so the next pass's
        #    cleanup never re-walls it: the loop settles (clean + connected) in a few passes.
        #    Each pass restamps pads (healing any lane a prior repair drove across one), then
        #    cleans seams, re-opens the main chokes, cleans again, and finally repairs one
        #    link if still split. On the pass where nothing needs repair the map is left
        #    clean (no illegal seam), connected, and with intact buildable pads.
        n_repairs = 0
        for _ in range(30):
            stamp_pads()
            cliff_cleanup()
            assert_main_chokes()
            cliff_cleanup()
            if not repair_one():
                break
            n_repairs += 1

        # final settle: wall EVERY remaining illegal seam, even at a pad edge (main pad next
        # to its natural pad) or a ramp shoulder, using a tight 1-cell ramp halo so two
        # plateaus never abut at a ramp corner. This can occasionally pinch a just-wide-enough
        # repair ramp, so re-check connectivity and re-carve if needed -- a repair lands on a
        # same-level cell (flat, no seam) or a wider ramp, so the loop settles at zero flat
        # overlaps and fully connected.
        for _ in range(8):
            protect = ndimage.binary_dilation(ramp, iterations=1)
            padprot = (pad_level >= 0) & (elev == pad_level)
            passprot = passage & walk
            prot = protect | padprot | passprot | ndimage.binary_dilation(locked, iterations=1)
            # wall the HIGHER side of each seam (cliff top) except where the free cell abuts a
            # protected cell of a different level (then wall the free cell regardless of height).
            npf = np.zeros_like(walk, dtype=bool)
            for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
                npf |= np.roll(prot, sh, axis=ax) & (elev != np.roll(elev, sh, axis=ax))
            db = _diff_boundary(elev, walk)
            walk[db & ~prot & (_higher_boundary(elev, walk) | npf)] = False
            if not repair_one():
                break

        # 4) RAMP REGULARIZATION: guarantee every surviving ramp is a single ascending strip
        #    connecting exactly one low plateau to one high plateau (one level step) -- the
        #    only thing the SC2 exporter can turn into a valid ramp. This is a no-op on ramps
        #    that are already clean, so it never disturbs the good ones. Any *unclean* ramp
        #    component (same-level bridge, >1-level span, a blob touching several plateaus, or
        #    a sub-min-size fragment) is DEMOTED to plain plateau (its ramp flag cleared,
        #    elevation kept). The existing cliff-cleanup then walls the level seams the demoted
        #    blob exposes, and connectivity repair reconnects whatever that split off --
        #    preferring a same-level flat passage (so e.g. two same-level plateaus that were
        #    wrongly bridged by a ramp get a proper flat passage instead) and otherwise a
        #    fresh single-step ramp. We iterate demote -> cleanup -> repair to a fixpoint that
        #    is all-ramps-clean AND connected; residual unclean maps are caught by the
        #    validator (and regenerated by the generation loop).
        struct8 = np.ones((3, 3), dtype=bool)

        def regularize_ramps() -> bool:
            # Non-destructive demotion only: a ramp component that is NOT actually a level
            # transition -> clear its ramp flag so it reads as a flat passage. "Not a
            # transition" means the component's own cells are all one elevation AND every
            # plateau neighbour is that same level AND it does not abut another ramp (so it is
            # not one step of a staircase). Under those conditions all its neighbours already
            # share its level, so clearing the flag creates no new seam and cannot disconnect
            # anything. Genuine ramps (own cells span two levels, or feeding another ramp) are
            # left untouched.
            dil = ndimage.binary_dilation
            lab_r, nr = ndimage.label(ramp)
            changed = False
            for cid in range(1, nr + 1):
                comp = lab_r == cid
                if (comp & locked).any():
                    continue                              # canonical staircase -> never demote
                comp_levels = {int(v) for v in elev[comp]}
                if len(comp_levels) != 1:
                    continue                              # spans levels -> real ramp
                grow = dil(comp)
                if (grow & ramp & ~comp).any():
                    continue                              # abuts another ramp -> staircase
                nb = grow & walk & ~ramp
                if nb.any() and {int(v) for v in elev[nb]} - comp_levels:
                    continue                              # touches a different level -> real
                ramp[comp] = False                        # spurious flat "ramp" -> flatten
                changed = True
            return changed

        # Flattening a spurious flat "ramp" cannot create a new level seam (all its
        # neighbours already share its level) and cannot disconnect anything, so a single
        # pass leaves the map exactly as clean/connected as before, with the spurious ramp
        # flags removed.
        regularize_ramps()

        # 5) RELIEF NORMALIZATION for a walkable export. The exporter climbs a level with a fixed
        #    8-cliff/cell gradient, so (a) a ramp may bridge at most ONE level, and (b) a one-level
        #    ramp needs >= 8 cells of run. First wall every >1-level ramp (no single gradient can
        #    span it) and reconnect what that islands via same-level/single-step repair; then
        #    lengthen every one-level ramp along its flow axis until its run >= 8. Iterate to a
        #    fixpoint. Residual failures are caught by the relief validator and regenerated.
        for _pass in range(6):
            straightened = _straighten_ramps(walk, elev, ramp, cfg.ramp_choke_width)
            walled = _wall_multilevel_ramps(walk, elev, ramp)
            stamp_pads()
            cliff_cleanup()
            assert_main_chokes()
            cliff_cleanup()
            # isolate every ramp to a clean bridge (one plateau per end) so the exporter's flank
            # channeling can't punch porous voids through the crossing (see _moat_ramp_flanks).
            moated = _moat_ramp_flanks(walk, elev, ramp, pad_level)
            progressed = False
            for _ in range(len(bases)):          # bounded: at most one carve per base
                if not repair_one():
                    break
                progressed = True
            regularize_ramps()
            if not straightened and not walled and not moated and not progressed:
                break
        # FINAL geometry + connectivity settle. Two edits happen here that earlier passes don't:
        #   * _lengthen_ramps deepens each single-level ramp's ENDS along its flow axis (never pads)
        #     until its run reaches run_min cells, so the exporter's fixed 8/cell gradient climbs the
        #     full 64-unit level. A level is 64/8 = 8 sub-steps, so the ramp needs run_min = 9 (=8+1)
        #     distinct projection ranks for every consecutive step to be <=8; run_min=8 yields only 8
        #     ranks (denom 7), stepping by 16 at one seam -> an in-engine >8 cliff wall INSIDE the
        #     ramp (findings §4.4).
        #   * _rectangularize_ramps crops the hourglass flare lengthening re-introduced (wide mouths,
        #     pinched waist) back to a uniform-width band the rampList quad covers exactly.
        # BOTH edit terrain with NO connectivity repair after them, and the trailing cliff_cleanup
        # walls fresh seams -- so a link the earlier passes had joined can be severed here, leaving a
        # map disconnected even by its own optimistic oracle (measured: ~half of the disconnected
        # rejects). Iterate edit -> clean -> repair to a fixpoint: each pass lengthens/crops, heals
        # seams, then carves one repair link if still split (re-cropped next pass), until the geometry
        # is stable AND connected. Residual failures are caught by the (engine-accurate) validator.
        for _ in range(8):
            _lengthen_ramps(walk, elev, ramp, pad_level, run_min=9)
            refloored = _rectangularize_ramps(walk, elev, ramp, pad_level, cfg.ramp_choke_width)
            stamp_pads()
            cliff_cleanup()
            regularize_ramps()
            progressed = repair_one()
            if not progressed and not refloored:
                break

        # 6) CANONICALIZE: rebuild any surviving blob (fused junction front, run==0, wide, multiple
        #    plateau fragments) into ONE locked clean narrow staircase between its two dominant
        #    plateaus, walling the rest; clean staircases are locked in place and skipped. Iterate
        #    canonicalize -> restamp -> clean -> repair to a fixpoint that is all-ramps-clean AND
        #    connected. Locked staircases are immutable to cleanup/regularize, so this cannot
        #    oscillate; residual failures are caught by the (engine-accurate) validator.
        for _ in range(12):
            canon = _canonicalize_ramps(walk, elev, ramp, pad_level, locked, canvas,
                                        cfg.ramp_choke_width, cfg.ramp_run)
            stamp_pads()
            cliff_cleanup()
            assert_main_chokes()
            cliff_cleanup()
            regularize_ramps()
            progressed = repair_one()
            if not canon and not progressed:
                break

        if os.environ.get("RAST_DEBUG") and n_repairs:
            print(f"[rast] seed={seed}: {n_repairs} connectivity repair carve(s)")
        return walk, elev, ramp

    def openness(walk) -> float:
        sub = walk[pa.y : pa.y + pa.height, pa.x : pa.x + pa.width]
        return float(sub.mean())

    # ---- auto-tune growth multiplier to hit the target openness band (bisection) ----
    lo, hi = cfg.growth_lo, cfg.growth_hi
    walk, elev, ramp = paint(hi)
    best = (walk, elev, ramp)
    if openness(walk) >= cfg.target_openness[0]:
        for _ in range(cfg.growth_iters):
            mid = 0.5 * (lo + hi)
            walk, elev, ramp = paint(mid)
            o = openness(walk)
            best = (walk, elev, ramp)
            if o < cfg.target_openness[0]:
                lo = mid
            elif o > cfg.target_openness[1]:
                hi = mid
            else:
                break
    walk, elev, ramp = best

    # ---- EXACT rot180 symmetry (fairness + connectivity) ----
    # The skeleton is pixel-exact rot180-symmetric, but the paint/cleanup/repair passes drift by a
    # handful of cells (~0.4%). That tiny divergence is amplified at ramps -- a mirror pair can
    # rasterize to 92 vs 87 cells -- enough to flip a single <ramp> quad from full to partial
    # coverage, so ONE spawn's crossing works while its mirror doesn't (in-engine verified on
    # seed 25: MAIN1 half connects, MAIN2 half doesn't). It is also a fairness defect (a 1v1 map
    # must be pixel-symmetric). Snap the terrain back to an exact mirror BEFORE deriving buildable /
    # ramp objects / resources, so both halves -- and every downstream layer -- are identical.
    walk, elev, ramp = _symmetrize_rot180(walk, elev, ramp, pa)

    # ---- ramp objects, prune, buildable, MapIR ----
    ramps: list[Ramp] = []
    labeled, n = ndimage.label(ramp)
    for lab_id in range(1, n + 1):
        comp = labeled == lab_id
        rys, rxs = np.where(comp)
        if len(rxs) < 6:
            ramp[comp] = False                        # tiny fragment -> demote to plateau
            continue
        # A clean ramp is painted at ONE elevation but *bridges* two plateaus; its true span is
        # the set of plateau levels it touches, NOT its own cells' elevation. Derive lo/hi from the
        # adjacent (non-ramp) plateaus so a real 0->1 ramp strip isn't mislabelled 0->0.
        grow = ndimage.binary_dilation(comp, structure=np.ones((3, 3), bool))
        nb = grow & walk & ~ramp
        nb_levels = sorted({int(v) for v in elev[nb]}) if nb.any() else []
        if len(nb_levels) <= 1:
            ramp[comp] = False                        # same-level bridge -> flat passage, not a ramp
            continue
        lo_lv, hi_lv = nb_levels[0], nb_levels[-1]
        # orient bottom/top toward the low / high plateau centroids so the exporter climbs correctly
        low_nb = nb & (elev == lo_lv)
        hi_nb = nb & (elev == hi_lv)
        lys, lxs = np.where(low_nb)
        hys, hxs = np.where(hi_nb)
        cells = [(int(x), int(y)) for x, y in zip(rxs, rys)]
        ramps.append(
            Ramp(
                cells=cells,
                bottom=(float(lxs.mean()), float(lys.mean())),
                top=(float(hxs.mean()), float(hys.mean())),
                low_level=lo_lv,
                high_level=hi_lv,
                width=None,
            )
        )

    # ---- PRUNE REDUNDANT CROSSINGS (source-side terrace fix, findings §4.4 "one/two well-separated
    #      staircases per terrace pair") -------------------------------------------------------------
    # The terrace carve cuts ~one staircase per cross-level graph edge, so a fragmented multi-terrace
    # seed ends up with FAR more ramps than connectivity needs (measured: seed 42 carves 14, needs 7).
    # Every extra crossing is in-engine detection/fusion SURFACE: the offline oracle models CLIF
    # connectivity but NOT ramp detection/quad coverage, so a blobby many-ramp seed reads connected
    # offline yet seals a main in-engine ("17 blobby ramps" -- §4.10 seed 42). Seeds whose only
    # crossings are the minimal articulation set (5/22/2: exactly 2 mirror pairs) connect in-engine.
    # So greedily WALL the most fusion-prone crossings -- wide + DIAGONAL first, since cardinals detect
    # reliably in every orientation while wide leftward diagonals are the marginal class (§4.6) -- IN
    # MIRROR PAIRS, keeping every real base connected under the EXACT exporter-faithful oracle
    # validate_map uses (engine_cliff_grid + engine_components), so offline yield can never drop. The
    # survivors are the minimal, well-separated, cardinal-leaning set gold maps use.
    real_bases = [b for b in bases if b.kind in _REAL_BASES]
    if ramps and real_bases and not os.environ.get("NO_PRUNE_CROSSINGS"):
        cx0v, cy0v = pa.x + pa.width / 2.0, pa.y + pa.height / 2.0

        def _mir(x, y):
            if sym == "mirror_lr":
                return (2 * cx0v - x, y)
            if sym == "mirror_ud":
                return (x, 2 * cy0v - y)
            return (2 * cx0v - x, 2 * cy0v - y)

        def _all_connected(keep, wmask):
            cliff = engine_cliff_grid(wmask, elev, keep)
            comp = engine_components(cliff, bridge=cardinal_ramp_bridge(keep, cliff.shape))

            def _lab(b):
                return int(comp[int(round(b.y)), int(round(b.x))])

            root = _lab(real_bases[0])
            return root >= 0 and all(_lab(b) == root for b in real_bases)

        # Only prune a map that is ALREADY connected under the oracle (else it's rejected anyway and
        # walling could mask a real fragmentation bug).
        if _all_connected(ramps, walk):
            cents = [(sum(x for x, _ in r.cells) / len(r.cells),
                      sum(y for _, y in r.cells) / len(r.cells)) for r in ramps]
            # pair each ramp with its rot180/mirror partner (nearest mirrored centroid); a center
            # ramp self-pairs. Terrain is pixel-exact symmetric, so partners align to well under 1 cell.
            partner = [-1] * len(ramps)
            for i in range(len(ramps)):
                if partner[i] != -1:
                    continue
                mx, my = _mir(*cents[i])
                best, bd = -1, 9.0
                for j in range(len(ramps)):
                    if j == i or partner[j] != -1:
                        continue
                    d = (cents[j][0] - mx) ** 2 + (cents[j][1] - my) ** 2
                    if d < bd:
                        bd, best = d, j
                if best != -1:
                    partner[i] = best
                    partner[best] = i
                elif (mx - cents[i][0]) ** 2 + (my - cents[i][1]) ** 2 <= 9.0:
                    partner[i] = i          # genuinely self-symmetric (center) crossing
                else:
                    partner[i] = -2         # no mirror partner (asymmetric map) -> never prune it

            def _prio(i):
                r = ramps[i]
                idx, _ = snap_uphill(r.top[0] - r.bottom[0], r.top[1] - r.bottom[1])
                xs = [c[0] for c in r.cells]
                ys = [c[1] for c in r.cells]
                width = min(max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
                return (0 if idx >= 4 else 1, -width)   # diagonal + wide -> tried for removal first

            removed: set[int] = set()
            changed = True
            while changed:
                changed = False
                for i in sorted((k for k in range(len(ramps)) if k not in removed), key=_prio):
                    if i in removed or partner[i] == -2:
                        continue                        # unpartnered -> pruning it would desync mirror
                    grp = {i, partner[i]} - removed
                    if not grp:
                        continue
                    keep = [ramps[k] for k in range(len(ramps)) if k not in removed and k not in grp]
                    tw = walk.copy()
                    for k in grp:
                        for (x, y) in ramps[k].cells:
                            tw[y, x] = False
                    if _all_connected(keep, tw):
                        for k in grp:
                            for (x, y) in ramps[k].cells:
                                walk[y, x] = False
                                ramp[y, x] = False
                        removed |= grp
                        changed = True
                        break
            if removed:
                ramps = [ramps[k] for k in range(len(ramps)) if k not in removed]

    # ---- buildable (after pruning walled the redundant crossings) ----
    edt = ndimage.distance_transform_edt(walk)
    buildable = walk & (edt >= cfg.build_clearance) & (~ramp)

    # ---- resources (M5): deterministic legal mineral line + geysers per real base ----
    resources, res_idx = _place_resources(skel, level, elev, ramp, canvas, pa, cfg)

    # ROOM/JUNCTION nodes are interior junctions, not resource bases: they are painted as
    # plateaus above, but must not appear as bases/start locations in the MapIR.
    ir_bases = [
        BaseNode(kind=b.kind, x=b.x, y=b.y, resource_idx=res_idx.get(i, []))
        for i, b in enumerate(bases)
        if b.kind in _REAL_BASES
    ]
    starts = [(b.x, b.y) for b in bases if b.kind == BaseKind.MAIN]

    return MapIR(
        map_name=f"gen_{skel.seed}",
        width=w,
        height=h,
        playable_area=Rect(pa.x, pa.y, pa.width, pa.height),
        walkable=walk,
        buildable=buildable,
        elevation=elev,
        bases=ir_bases,
        resources=resources,
        ramps=ramps,
        regions=[],
        connections=[],
        start_locations=starts,
    )
