"""Encoders for the SC2 internal files we author, plus the template-fit helpers.

Formats are the community-decoded ones (SC2Mapster wiki) confirmed by round-tripping the
real maps in ``gold_maps/``:

    t3SyncCliffLevel (CLIF): 32-byte header {magic, version, sizeX, sizeY, zero[16]}
                             followed by USHORT[sizeX*sizeY], row-major with row == y.
                             value 0 == unplayable/void; playable == CLIFF_BASE + CLIFF_STEP*level.

    Objects: XML. Start locations are ObjectPoint Type="StartLoc"; resources are ObjectUnit
             with UnitType MineralField / VespeneGeyser.
"""

from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass

import numpy as np

from sc2mapgen.ir import (BaseKind, MapIR, ResourceKind, cardinal_profile, is_gold_cardinal,
                          ramp_cell_cliffs, ramp_uphill)

# CLIF: void/unplayable -> 0. Real maps use full terrain TIERS at multiples of 64 (64, 128, 192);
# the in-between values (72, 80, ...) are ramp gradations, NOT flat buildable ground. We therefore
# map our discrete elevations onto the harvested tiers rather than assuming a fixed formula.
CLIFF_VOID = 0

_CLIF_MAGIC = b"CLIF"

# heightmaps are per-VERTEX (verts = cells+1):
#   t3SyncHeightMap (SMAP): 64-byte header + u32[w*h], 20.12 fixed point.
#   t3HeightMap     (HMAP): 32-byte header + (u16,u16,u16)[w*h] packing (ref, height, tier).
# The community formulae are lossy/per-map, so instead of synthesizing values we HARVEST real,
# engine-valid flat-vertex encodings from the template (one per terrain tier + void) and stamp
# them. Identical tuples across a plateau guarantee flatness (== buildable); different tuples
# between tiers form genuine cliffs. The game reads HMAP for terrain, so authoring it is required.
_SMAP_HEADER = 64
_HMAP_HEADER = 32


@dataclass
class Palette:
    """Engine-valid encodings harvested from a template.

    ``tiers`` are the flat, buildable plateau levels (cliff multiples of 64). ``enc`` maps EVERY
    cliff value the template uses -- void (0), flat tiers, AND ramp sub-levels (72, 80, ... the
    8-steps between tiers) -- to a real (smap_raw, hmap_tuple). Ramp sub-levels are what make a
    cross-level transition a *walkable* ramp instead of a wall, so we harvest and re-stamp them
    rather than synthesising heights.
    """
    void_cliff: int
    void_smap: int
    void_hmap: tuple[int, int, int]
    tiers: list[tuple[int, int, tuple[int, int, int]]]  # (cliff, smap_raw, hmap) ascending height
    enc: dict[int, tuple[int, tuple[int, int, int]]]    # cliff value -> (smap_raw, hmap_tuple)

    @property
    def n_tiers(self) -> int:
        return len(self.tiers)

    @property
    def tier_cliffs(self) -> list[int]:
        return [c for c, _s, _h in self.tiers]

    def sublevels_between(self, lo_cliff: int, hi_cliff: int) -> list[int]:
        """Ascending GOLD-STANDARD ramp sub-level cliff values between two tiers (72,80,...,112).

        Gold maps SKIP the sub-level just below the high plateau (``hi_cliff - 8`` == 120 for a
        64->128 climb) so the top axis step is +16, not +8 -- measured on every ramp of FrostLE /
        AcropolisAIE / AutomatonAIE (findings ?4.7). Heights are interpolated linearly from the tier
        heights for any cliff value (see ``_interp_heights``), so any 8-step value is valid; the
        skipped 120 is a gold *convention*, not a limitation. The actual per-cell gradient is written
        by ``ir.ramp_cell_cliffs``; this list is the sub-level vocabulary (and a non-empty guard).
        """
        return list(range(lo_cliff + 8, hi_cliff - 8, 8))


def _flat_vertex(cl: np.ndarray, level: int) -> tuple[int, int] | None:
    """First vertex (vy, vx) whose four surrounding cells all equal ``level`` (a flat interior)."""
    ch, cw = cl.shape
    for vy in range(1, ch):
        row_a, row_b = cl[vy - 1], cl[vy]
        for vx in range(1, cw):
            if (row_a[vx - 1] == level and row_a[vx] == level
                    and row_b[vx - 1] == level and row_b[vx] == level):
                return vy, vx
    return None


def flat_levels(cl: np.ndarray) -> set[int]:
    """Cliff values that have at least one flat interior vertex (see :func:`_flat_vertex`)."""
    a = cl[:-1, :-1]
    flat = (a == cl[:-1, 1:]) & (a == cl[1:, :-1]) & (a == cl[1:, 1:])
    return {int(v) for v in np.unique(a[flat])}


def palette_capacity(cl: np.ndarray) -> int:
    """Flat tiers :func:`harvest_palette` can harvest from this CLIF grid, or 0 if it has no flat
    void vertex (such a template can't be harvested at all)."""
    levels = flat_levels(cl)
    if 0 not in levels:
        return 0
    return sum(1 for v in levels if v > 0 and v % 64 == 0)


def _vertex_cliff_max(cliff_cell: np.ndarray, sw: int, sh: int) -> np.ndarray:
    """Per-vertex cliff = max cliff over the (up to four) surrounding cells.

    Taking the max keeps each plateau flat right to its edge and drops the height step onto the
    lower side, which is what real cliffs (and the low end of a ramp) do. On a ramp, whose cells
    step by 8, this yields a monotone per-vertex gradient -- a genuine slope.
    """
    ch, cw = cliff_cell.shape
    if (sw, sh) != (cw + 1, ch + 1):
        raise ValueError(f"heightmap verts {sw}x{sh} != cliff cells+1 {cw+1}x{ch+1}")
    verts = np.zeros((sh, sw), dtype=np.int32)
    verts[0:ch, 0:cw] = np.maximum(verts[0:ch, 0:cw], cliff_cell)
    verts[1:sh, 0:cw] = np.maximum(verts[1:sh, 0:cw], cliff_cell)
    verts[0:ch, 1:sw] = np.maximum(verts[0:ch, 1:sw], cliff_cell)
    verts[1:sh, 1:sw] = np.maximum(verts[1:sh, 1:sw], cliff_cell)
    return verts


def _vertex_cliff_ramp_aware(cliff_cell: np.ndarray, ramp_mask: np.ndarray,
                             sw: int, sh: int, max_rise: int = 16) -> np.ndarray:
    """:func:`_vertex_cliff_max`, except a vertex touching a ramp ignores cells more than
    ``max_rise`` above its highest ramp cell. Plain max lets a high-plateau corner beside a ramp's
    lower sub-levels spike that ramp vertex up to the plateau height -- a visible notch in the
    slope. ``max_rise=16`` keeps the gold +16 top exit (112 -> 128) flush with the plateau.
    """
    verts = _vertex_cliff_max(cliff_cell, sw, sh)
    ch, cw = cliff_cell.shape
    rc = np.where(ramp_mask, cliff_cell, 0).astype(np.int32)
    rmax = np.zeros((sh, sw), dtype=np.int32)
    corners = ((slice(0, ch), slice(0, cw)), (slice(1, sh), slice(0, cw)),
               (slice(0, ch), slice(1, sw)), (slice(1, sh), slice(1, sw)))
    for vs in corners:
        rmax[vs] = np.maximum(rmax[vs], rc)
    capped = np.zeros((sh, sw), dtype=np.int32)
    cc = cliff_cell.astype(np.int32)
    for vs in corners:
        ok = cc <= rmax[vs] + max_rise
        capped[vs] = np.maximum(capped[vs], np.where(ok, cc, 0))
    return np.where(rmax > 0, capped, verts)


def harvest_palette(cliff: bytes, smap: bytes, hmap: bytes) -> Palette:
    """Build a :class:`Palette` from a template's CLIF / SMAP / HMAP blobs.

    Flat tiers (cliff multiples of 64) and void are taken from real flat interior vertices.
    Ramp sub-levels (72, 80, ... 8-steps) are harvested from the ramp vertices themselves, keyed
    by the per-vertex max cliff -- exactly how we later derive vertices when authoring -- so a
    stamped ramp reproduces heights the engine already treats as a walkable slope.
    """
    _, cw, ch, cl = decode_cliff(cliff)
    w = struct.unpack_from("<I", smap, 8)[0]
    h = struct.unpack_from("<I", smap, 12)[0]
    n = w * h
    sm = np.frombuffer(smap[_SMAP_HEADER:_SMAP_HEADER + n * 4], dtype="<u4").reshape(h, w)
    hm = np.frombuffer(hmap[_HMAP_HEADER:_HMAP_HEADER + n * 6], dtype="<u2").reshape(h, w, 3)

    def encode_at(vy: int, vx: int) -> tuple[int, tuple[int, int, int]]:
        return int(sm[vy, vx]), (int(hm[vy, vx, 0]), int(hm[vy, vx, 1]), int(hm[vy, vx, 2]))

    # void + flat tiers from flat interior vertices (guaranteed uniform -> buildable pads)
    vpos = _flat_vertex(cl, 0)
    if vpos is None:
        raise ValueError("template has no flat void vertex")
    v_smap, v_hmap = encode_at(*vpos)

    enc: dict[int, tuple[int, tuple[int, int, int]]] = {0: (v_smap, v_hmap)}
    tiers: list[tuple[int, int, tuple[int, int, int]]] = []
    for lv in sorted(v for v in set(cl.flatten().tolist()) if v > 0 and v % 64 == 0):
        pos = _flat_vertex(cl, lv)
        if pos is None:
            continue
        s, hh = encode_at(*pos)
        tiers.append((lv, s, hh))
        enc[lv] = (s, hh)
    if not tiers:
        raise ValueError("template has no primary (multiple-of-64) flat tiers")

    # ramp sub-levels: key by the per-vertex max cliff (matches build_smap/build_hmap derivation)
    vc = _vertex_cliff_max(cl.astype(np.int32), w, h)
    for lv in sorted(v for v in set(cl.flatten().tolist()) if v > 0 and v % 64 != 0):
        if lv in enc:
            continue
        pos = np.argwhere(vc == lv)
        if len(pos):
            vy, vx = int(pos[0][0]), int(pos[0][1])
            enc[lv] = encode_at(vy, vx)
    return Palette(CLIFF_VOID, v_smap, v_hmap, tiers, enc)


def decode_cliff(data: bytes) -> tuple[int, int, int, np.ndarray]:
    """Return (version, sizeX, sizeY, uint16 array [sizeY, sizeX]) from a t3SyncCliffLevel blob."""
    magic = data[0:4]
    if magic != _CLIF_MAGIC:
        raise ValueError(f"not a CLIF file (magic={magic!r})")
    version = struct.unpack_from("<I", data, 4)[0]
    sx = struct.unpack_from("<I", data, 8)[0]
    sy = struct.unpack_from("<I", data, 12)[0]
    body = np.frombuffer(data[32 : 32 + sx * sy * 2], dtype="<u2").reshape(sy, sx)
    return version, sx, sy, body


def encode_cliff(levels: np.ndarray, version: int = 100) -> bytes:
    """Serialize a uint16 cliff-level grid [sizeY, sizeX] into a t3SyncCliffLevel blob."""
    sy, sx = levels.shape
    header = _CLIF_MAGIC + struct.pack("<3I", version, sx, sy) + b"\x00" * 16
    return header + np.ascontiguousarray(levels, dtype="<u2").tobytes()


# uphill unit vector per ramp `dir`. 0-3 CARDINAL (unit spacing), 4-7 DIAGONAL (sqrt2 spacing).
# ? The integer `dir` attribute we emit MUST match SC2's OWN convention or the engine's ramp
# detector mislabels the ramp and DROPS it (undetected == a walled cliff line). The canonical
# mapping below was recovered by tabulating base.u vs dir across all 106 gold maps:
#     dir 0 -> (0,-1) N   dir 1 -> (0,+1) S   dir 2 -> (-1,0) W   dir 3 -> (+1,0) E
#     dir 4 -> NW         dir 5 -> NE         dir 6 -> SW         dir 7 -> SE
# The diagonals (4-7) already matched, which is why diagonal ramps always detected; the cardinals
# were previously SCRAMBLED (1=E,2=S,3=W), so e.g. a west ramp emitted dir=3 (== SC2 *East*, 180?
# opposite its u) and the engine never registered it (seed 42's dir-west cardinals; findings ?4.6).
# +y is downward (SC2 screen coords).
# Gold-standard rampList quad width (cells). Gold maps use 1-5 (median 4); the engine expands this
# small quad across the full cliff band. A quad wider than the band balloons/mis-anchors the ramp
# (findings ?4.7/?4.8, scripts/_rampmut.py). Quad `run` stays make_ramp_entry's default (2 cells).
GOLD_QUAD_W = 4.0
_DIAG_SHOULDER_ANCHOR = bool(os.environ.get("DIAG_SHOULDER_ANCHOR"))

# Gold-standard CARDINAL trapezoid quad params (measured Goldenaura512AIE dir=0/2). base is a wide
# lip (w=5, h=1) that reaches UP into the high plateau; mid is narrower (w=3, h=2) one `run` cell
# downhill; low corners are w=1 h=2 markers at ?off along `r`. Gold ships run=1 for its SHORT/STEEP
# 3-cell cardinal gradient; our isoline staircase is LONG/GENTLE (~6 cells), so `CARD_RUN` is bumped
# so the quad reaches onto the gradient (not just the plateau lip) ? env-tunable for in-engine A/B.
_CARD_BASE_W = float(os.environ.get("CARD_BASE_W", 5.0))
_CARD_BASE_H = float(os.environ.get("CARD_BASE_H", 1.0))
_CARD_MID_W = float(os.environ.get("CARD_MID_W", 3.0))
_CARD_MID_H = float(os.environ.get("CARD_MID_H", 2.0))
_CARD_RUN = float(os.environ.get("CARD_RUN", 2.0))
_CARD_CORNER_OFF = float(os.environ.get("CARD_CORNER_OFF", 4.0))

_RAMP_DIR_U = {
    0: (0.0, -1.0),          # north  (SC2 dir0)
    1: (0.0, 1.0),           # south  (SC2 dir1)
    2: (-1.0, 0.0),          # west   (SC2 dir2)
    3: (1.0, 0.0),           # east   (SC2 dir3)
    4: (-0.7071066, -0.7071069),   # NW
    5: (0.7071069, -0.7071066),    # NE
    6: (-0.7071066, 0.7071069),    # SW
    7: (0.7071069, 0.7071066),     # SE
}
_SQRT2 = 1.4142135623730951


def _ef(x: float) -> str:
    """Format a float like the Galaxy Editor: mantissa + 3-digit signed exponent (1.000000e+000)."""
    s = f"{x:.6e}"
    mant, exp = s.split("e")
    sign = "+" if exp[0] not in "+-" else exp[0]
    digits = exp.lstrip("+-")
    return f"{mant}e{sign}{int(digits):03d}"


def _tf(u, r, c, w, h) -> str:
    return (f"u({_ef(u[0])}, {_ef(u[1])}) r({_ef(r[0])}, {_ef(r[1])}) "
            f"c=({_ef(c[0])}, {_ef(c[1])}) w={_ef(w)} h={_ef(h)}")


def make_ramp_entry(base_c, direction: int, width: float, lo: int, hi: int,
                    run: float = 2.0, corner_off: float | None = None,
                    top_len: float | None = None) -> str:
    """Synthesize one t3Terrain <ramp> entry, cardinal (dir 0-3) or diagonal (dir 4-7).

    base_c: (x,y) center of the *uphill* edge (on the high plateau boundary).
    direction: 0-3 cardinal (unit spacing) or 4-7 diagonal (sqrt2 spacing); sets uphill `u`.
    width: full ramp width in cells along the edge (FrostLE uses 3-4).
    lo,hi: cliff levels (e.g. 1,2). run: cells of run (FrostLE uses 2).
    Geometry (FrostLE ground truth): base = uphill edge, mid = base - run*u (downhill center),
    corners = mid -/+ halfw*r, r = rotate(u,-90). Diagonal cells are sqrt2 apart so the run/width/
    corner offsets carry a sqrt2 factor; cardinal cells are unit-spaced (factor 1) -- proven via
    scripts/axis_ramp_probe.py.
    """
    ux, uy = _RAMP_DIR_U[direction]
    rx, ry = (uy, -ux)  # rotate u by -90deg
    sp = _SQRT2 if direction >= 4 else 1.0             # diagonal cells are sqrt2 apart
    bx, by = float(base_c[0]), float(base_c[1])
    if direction < 4:
        # ---- CARDINAL: gold-standard TRAPEZOID quad (Goldenaura512AIE dir=0/2, findings ?4.12) ----
        # A plain rectangle (base.h=0) either floats on the flat clamped shoulder (drift, seals a
        # base) or, if anchored at the gradient top, no longer TOUCHES the high plateau -> the engine
        # spawns phantoms. Gold cardinals instead use a trapezoid: a WIDE base (w=5) with a lip
        # `base.h`>0 reaching UP into the high plateau, narrowing to a smaller mid (w=3, h=2) one
        # `run` cell downhill, corners w=1 h=2 at ?off along `r`. Anchored at the ramp's GRADIENT TOP
        # (build_ramp_list), the base.h lip sits exactly on the 128<->112 interface so the engine
        # detects the ramp AND expands it down our long/gentle isoline staircase without phantoms.
        # ? The user's earlier "copy gold's trapezoid" attempt regressed because it kept the SHOULDER
        # anchor + gold's run=1; the working recipe is trapezoid + gradient-top anchor (+ a run long
        # enough to reach onto the gradient). Params are env-tunable for A/B (see CARD_* below).
        runlen, base_w, mid_w = _CARD_RUN, _CARD_BASE_W, _CARD_MID_W
        off = _CARD_CORNER_OFF
        if top_len is not None:
            # Gold cardinal sizing from the top sub-level row length L (71 ramps, exact):
            # base.w = L/2+1, mid.w = L/2-1, low corners at +-L/2 from mid, run 1.
            half = top_len / 2.0
            runlen, base_w, mid_w, off = 1.0, half + 1.0, max(1.0, half - 1.0), half
        if corner_off is not None:
            off = corner_off
        mx, my = bx - runlen * ux, by - runlen * uy
        llx, lly = mx - off * rx, my - off * ry
        rlx, rly = mx + off * rx, my + off * ry
        base = _tf((ux, uy), (rx, ry), (bx, by), base_w, _CARD_BASE_H)
        mid = _tf((ux, uy), (rx, ry), (mx, my), mid_w, _CARD_MID_H)
        left_lo = _tf((ux, uy), (rx, ry), (llx, lly), 1.0, 2.0)
        right_lo = _tf((ux, uy), (rx, ry), (rlx, rly), 1.0, 2.0)
        left_hi = _tf((0, 0), (0, 0), (0.0, -2.0), 0.0, 0.0)
        right_hi = _tf((0, 0), (0, 0), (0.0, 4.0), 0.0, 0.0)
        return (f'<ramp dir="{direction}" hi="{hi}" lo="{lo}" '
                f'leftLo="{left_lo}" leftHi="{left_hi}" '
                f'rightLo="{right_lo}" rightHi="{right_hi}" '
                f'base="{base}" mid="{mid}" cid="1" '
                f'leftLoVar="0" leftHiVar="4294967295" '
                f'rightLoVar="0" rightHiVar="4294967295"/>')
    # ---- DIAGONAL (4-7): rectangle quad, engine auto-expands (unchanged; detects + bridges) ----
    runlen = run * sp
    mx, my = bx - runlen * ux, by - runlen * uy       # downhill center
    halfw = width * sp                                 # corner offset from mid == base.w
    llx, lly = mx - halfw * rx, my - halfw * ry        # leftLo corner
    rlx, rly = mx + halfw * rx, my + halfw * ry        # rightLo corner
    base = _tf((ux, uy), (rx, ry), (bx, by), width * sp, 0.0)
    mid = _tf((ux, uy), (rx, ry), (mx, my), width * sp, runlen)
    # corners are small 2x2 markers whose frame is the ramp's u turned 45deg onto the nearest axis
    # (gold, 1930 diagonals: dir4 u(0,-1), dir5 u(1,0), dir6 u(-1,0), dir7 u(0,1); r = u turned
    # -90deg). One fixed frame only mirrors correctly for one direction. Hi corners unused
    # (sentinel, Var=0xffffffff).
    cu = (round((ux - uy) / _SQRT2), round((ux + uy) / _SQRT2))
    cr = (cu[1], -cu[0])
    left_lo = _tf(cu, cr, (llx, lly), 2.0, 2.0)
    right_lo = _tf(cu, cr, (rlx, rly), 2.0, 2.0)
    left_hi = _tf((0, 0), (0, 0), (178.0, 178.0), 0.0, 0.0)
    right_hi = _tf((0, 0), (0, 0), (174.0, 174.0), 0.0, 0.0)
    return (f'<ramp dir="{direction}" hi="{hi}" lo="{lo}" '
            f'leftLo="{left_lo}" leftHi="{left_hi}" '
            f'rightLo="{right_lo}" rightHi="{right_hi}" '
            f'base="{base}" mid="{mid}" cid="0" '
            f'leftLoVar="0" leftHiVar="4294967295" '
            f'rightLoVar="0" rightHiVar="4294967295"/>')


def _snap_dir(ux: float, uy: float) -> int:
    """Snap an uphill vector to the nearest ramp `dir` over all 8 (cardinal + diagonal), so the
    quad matches the ramp's true orientation (axis-aligned ramps need a cardinal quad)."""
    best, bestdot = 1, -2.0
    for d, (vx, vy) in _RAMP_DIR_U.items():
        dot = ux * vx + uy * vy
        if dot > bestdot:
            best, bestdot = d, dot
    return best


# back-compat alias (older callers/tests)
_snap_diag_dir = _snap_dir


def channel_ramp_flanks(cliff_cell: np.ndarray, ramp_mask: np.ndarray, max_step: int = 8) -> int:
    """Void high-plateau cells that abut a ramp sub-level below its top one, in place.

    Gold ramps keep their flanks as plateau (findings ?3.2): the sub-levels ``lo+8..lo+40`` are
    flanked by the low plateau and walled purely by the cliff step, so low-plateau flank cells are
    left alone. Only the top sub-level (``hi-16``) may touch the high plateau, with the walkable +16
    exit the quad covers. Where our cut leaves the high plateau wrapped around a lower sub-level
    (``delta >= +24``), gold has no equivalent: kept as plateau, the ramp-aware vertex heights would
    notch the plateau corner down to the slope, and lowering the cell would leave a one-cell pocket,
    so that cell is voided. A plateau cell above a ramp cell is always its high plateau, and a +16
    is always the top exit, so ``delta > 2*max_step`` selects exactly these cells. Ramp cells are
    never voided. Returns the number of cells walled. Symmetric input -> symmetric output.
    """
    ch, cw = cliff_cell.shape
    d = cliff_cell.astype(np.int32)
    walk = cliff_cell > 0
    towall = np.zeros((ch, cw), dtype=bool)
    ys, xs = np.where(ramp_mask & walk)
    for y, x in zip(ys.tolist(), xs.tolist()):
        v = int(d[y, x])
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < ch and 0 <= nx < cw and walk[ny, nx] and not ramp_mask[ny, nx]:
                if int(d[ny, nx]) - v > 2 * max_step:
                    towall[ny, nx] = True
    cliff_cell[towall] = 0
    return int(towall.sum())


def ramp_cell_mask(mapir, cw: int, ch: int, off_x: int, off_y: int) -> np.ndarray:
    """Template-sized boolean mask of ramp cells (offset-applied), matching author_ramp_cliffs."""
    m = np.zeros((ch, cw), dtype=bool)
    for r in mapir.ramps:
        for (x, y) in r.cells:
            X, Y = int(x) + off_x, int(y) + off_y
            if 0 <= X < cw and 0 <= Y < ch:
                m[Y, X] = True
    return m


def ramp_cell_levels(mapir) -> list[tuple[int, int, int | None]]:
    """``(x, y, level)`` per IR ramp cell: ``None`` on the slope, else the plateau level the cell's
    gradient clamps to. The isoline gradient flattens cells past its span onto a plateau cliff
    (about half of all ramp cells), so those play and should look like that plateau. The IR
    elevation of a clamped cell is not reliably its plateau, hence the gradient."""
    out: list[tuple[int, int, int | None]] = []
    for r in mapir.ramps:
        cells = [(int(x), int(y)) for (x, y) in r.cells]
        if r.low_level is None or r.high_level is None or r.top is None or r.bottom is None:
            out.extend((x, y, None) for (x, y) in cells)
            continue
        lo, hi = int(r.low_level), int(r.high_level)
        lo_c, hi_c = (lo + 1) * 64, (hi + 1) * 64
        for (x, y), v in zip(cells, ramp_cell_cliffs(cells, *ramp_uphill(r)[1], lo_c, hi_c)):
            out.append((x, y, lo if v == lo_c else hi if v == hi_c else None))
    return out


def ramp_slope_mask(mapir, cw: int, ch: int, off_x: int, off_y: int) -> np.ndarray:
    """Template-sized mask of the ramp cells that actually slope (see :func:`ramp_cell_levels`)."""
    m = np.zeros((ch, cw), dtype=bool)
    for x, y, lv in ramp_cell_levels(mapir):
        X, Y = x + off_x, y + off_y
        if lv is None and 0 <= X < cw and 0 <= Y < ch:
            m[Y, X] = True
    return m


def author_ramp_diagonal(cliff_cell: np.ndarray, base_xy, direction: int, width: float,
                         lo_cliff: int, hi_cliff: int, subs: list[int]) -> None:
    """Paint a clean diagonal staircase gradient (lo->hi) into the cliff grid, in place.

    Steps the cliff value by one sub-level (8) per cell along the diagonal uphill axis `u`, over a
    band `width` cells wide (along the perpendicular `p`). The full value ramp is
    ``[lo_cliff] + subs + [hi_cliff]`` (consecutive diff == 8, which the engine treats as walkable);
    the band is centered on ``base_xy`` and extends symmetrically so its low end merges with the low
    plateau and its high end with the high plateau. Co-authoring the gradient from the same
    (base, dir, width) geometry as make_ramp_entry guarantees the rampList quad sits over the slope.
    """
    ch, cw = cliff_cell.shape
    ux, uy = _RAMP_DIR_U[direction]
    px, py = (uy, -ux)                     # perpendicular
    values = [lo_cliff] + list(subs) + [hi_cliff]
    n = len(values)
    cx, cy = float(base_xy[0]), float(base_xy[1])
    half = (n - 1) / 2.0
    hw = max(1, int(round(width)))
    for j, val in enumerate(values):
        sc = (j - half)                    # signed step offset along u (in diagonal cells)
        ctr_x, ctr_y = cx + sc * ux, cy + sc * uy
        for k in range(-hw, hw + 1):
            X = int(round(ctr_x + k * px))
            Y = int(round(ctr_y + k * py))
            if 0 <= X < cw and 0 <= Y < ch:
                cliff_cell[Y, X] = val


def build_ramp_list(mapir, palette, off_x: int, off_y: int) -> list[str]:
    """Synthesize a <ramp> entry per IR ramp (see make_ramp_entry / ?4 of the format findings).

    Each IR ramp carries ``top`` (high-plateau centroid), ``bottom`` (low-plateau centroid),
    ``low_level``/``high_level``, ``cells`` and, for generated maps, the planned ``direction``
    (``ir.ramp_uphill``; an ingested ramp's low->high vector is snapped to the nearest of the 8
    directions instead). Tier indices map to cliff levels (cliff//64). DIAGONAL ramps
    (4-7) anchor the quad at the ramp's furthest-uphill cell (the engine expands their rectangle
    quad cleanly down the band). CARDINAL ramps (0-3) instead anchor at the GRADIENT TOP and emit a
    gold-style trapezoid (make_ramp_entry) -- the isoline clamp gives cardinals a big flat high
    shoulder, and a shoulder-anchored rectangle drifts/seals a base while a gradient-top rectangle
    spawns phantoms; the trapezoid's `base.h` lip touches the plateau interface so the ramp detects
    tightly AND bridges (fixes seed 42's last sealed base; findings ?4.12).
    """
    entries: list[str] = []
    for r in mapir.ramps:
        if r.top is None or r.bottom is None or r.low_level is None or r.high_level is None:
            continue
        lo, hi = int(r.low_level), int(r.high_level)
        if hi <= lo or hi >= palette.n_tiers or not r.cells:
            continue
        lo_cliff = palette.tiers[lo][0]
        hi_cliff = palette.tiers[hi][0]
        lo_lvl = lo_cliff // 64
        hi_lvl = hi_cliff // 64
        # the planned uphill direction (snapped from the plateau centroids for ingested ramps)
        direction, _ = ramp_uphill(r)
        dux, duy = _RAMP_DIR_U[direction]
        cells_t = [(cx + off_x, cy + off_y) for (cx, cy) in r.cells]
        # base_c = ramp's GRADIENT TOP: the highest-projection cell still carrying a sub-level
        # BELOW hi_cliff, i.e. the cell that meets the high plateau at the 128<->112 interface --
        # NOT simply the furthest-uphill cell. The isoline gradient (findings ?4.10) CLAMPS every
        # cell beyond `span` lattice-steps to hi_cliff, producing a flat high shoulder that merges
        # into the plateau. Picking the max-projection cell lands the quad's uphill edge several
        # cells INSIDE that flat shoulder (measured seed 42: base_c on a 128 cell 3 cells uphill of
        # the real interface, 15/50 ramp cells clamped), so the quad floats on the plateau instead
        # of sitting on the ramp; the engine's ramp detector then anchors ambiguously and drifts
        # (parity-unstable), sealing a mirror base. Gold cardinal ramps place base.c exactly on the
        # top gradient cell adjacent to the plateau (scripts/_rampcmp.py: Goldenaura512AIE dir=2/0).
        # Using the SNAPPED (dux,duy) + ir.ramp_cell_cliffs keeps this identical to the gradient the
        # exporter actually writes (author_ramp_cliffs), so quad and staircase can't drift.
        # DIAGONAL ramps (4-7) already detect + bridge reliably with the shoulder anchor (the engine
        # expands their quad cleanly), so only re-anchor CARDINALS (0-3), the marginal class.
        cell_cliffs = ramp_cell_cliffs([(int(cx), int(cy)) for (cx, cy) in cells_t],
                                       dux, duy, lo_cliff, hi_cliff)
        gold_card = direction < 4 and is_gold_cardinal(dux, duy)
        top_cliff = cardinal_profile(lo_cliff, hi_cliff)[-2] if gold_card else hi_cliff - 16
        top_row = [c for c, cv in zip(cells_t, cell_cliffs) if cv == top_cliff]
        gold_diag = direction >= 4 and bool(top_row) and not _DIAG_SHOULDER_ANCHOR
        if gold_diag:
            # Gold diagonal quads (1961 ramps, scripts/_gold_quad_anchor.py): base.c = centroid of
            # the top sub-level row + sqrt2 uphill, zero sideways offset. The furthest-uphill cell
            # lies 4-5 cells into the flat clamped shoulder, and the engine then draws its ramp
            # mesh over flat plateau (visible humps at both ends of the painted ramp).
            base_c = (sum(c[0] for c in top_row) / len(top_row) + 0.5 + dux * _SQRT2,
                      sum(c[1] for c in top_row) / len(top_row) + 0.5 + duy * _SQRT2)
        else:
            anchor = cells_t
            if direction < 4:
                anchor = [c for c, cv in zip(cells_t, cell_cliffs) if cv < hi_cliff] or cells_t
            hx, hy = max(anchor, key=lambda c: c[0] * dux + c[1] * duy)
            base_c = (hx + 0.5, hy + 0.5)
        corner_off = None
        top_len = None
        if gold_card and top_row:
            # Gold cardinal quads (71 ramps, scripts/_gold_cardinal.py): base.c = centroid of the
            # top sub-level row + 0.5 uphill (the plateau edge), zero sideways offset; the quad is
            # sized from that row's length inside make_ramp_entry.
            n = len(top_row)
            base_c = (sum(c[0] for c in top_row) / n + 0.5 + 0.5 * dux,
                      sum(c[1] for c in top_row) / n + 0.5 + 0.5 * duy)
            top_len = float(n)
        elif direction < 4 and top_row and not _DIAG_SHOULDER_ANCHOR:
            # max() picks ONE tied cell (the leftmost), so the quad sat off-centre and its fixed
            # +-4 corners overshot the flank void onto the plateau beyond: a phantom ramp region
            # a worker can walk into (seed 24's cardinal at (62,62)). Centre on the top row
            # sideways and put the corners on its outer edges, as gold diagonals do.
            n = len(top_row)
            if duy:
                base_c = (sum(c[0] for c in top_row) / n + 0.5, base_c[1])
            else:
                base_c = (base_c[0], sum(c[1] for c in top_row) / n + 0.5)
            corner_off = n / 2.0
        # GOLD-FAITHFUL quad width: SMALL and fixed (~4), never wider than the true band. Gold maps
        # (Acropolis/Automaton/Frost) all declare quad width 1-5 (base.w 1.4-7.1) for bands 4.5-15 and
        # let the engine AUTO-EXPAND the small quad across the full contiguous band (findings ?4.7).
        # Emitting a quad WIDER than the band is actively harmful: the engine balloons the ramp
        # footprint (37->148 cells), drifts its center off the true interface, and spawns a PHANTOM
        # ramp where the quad overshoots into the plateau -- which seals the real main exit (seed 22).
        # Proven single-variable in scripts/_rampmut.py (findings ?4.8). So: quad width = min(band, 4).
        rvx, rvy = duy, -dux
        perp = [c[0] * rvx + c[1] * rvy for c in cells_t]
        band_w = (max(perp) - min(perp)) + 1.0 if len(cells_t) > 1 else 4.0
        width = max(2.0, min(band_w, GOLD_QUAD_W))
        if gold_diag:
            # Gold diagonal quads declare width = top-row cells / 2 (1961 ramps, no exceptions,
            # scripts/_gold_quad_width.py), which puts the low corner markers on the slope's two
            # ends. A wider quad drops them past the flank void onto the plateau beyond, where the
            # engine grows a phantom ramp region and unpathable blocks (seed 24's "nook").
            width = max(1.0, len(top_row) / 2.0)
        _wmax = os.environ.get("QUAD_WMAX")           # experiment override (see _rampmut.py)
        if _wmax:
            width = min(band_w, float(_wmax))
        # RUN stays at make_ramp_entry's default (~2): the quad only needs to sit at the high-plateau
        # interface in the right orientation -- the engine AUTO-EXPANDS it down the contiguous ramp
        # band (verified scripts/rampwide_probe.py). Manually stretching run to the ramp's full extent
        # is HARMFUL: the far edge overshoots past the low plateau into wall cells and BREAKS ramp
        # detection (verified: seed 22 natural->main ramp goes dead with a stretched quad, works with
        # the default). See findings ?4.5 "Quad stretch is harmful".
        entries.append(make_ramp_entry(base_c, direction, width, lo_lvl, hi_lvl,
                                       corner_off=corner_off, top_len=top_len))
    return entries


def set_ramp_list(terrain_xml: bytes, entries: list[str]) -> bytes:
    """Replace the <rampList> in t3Terrain.xml with the given <ramp> entries."""
    text = terrain_xml.decode("utf-8", "replace")
    body = "\n".join("            " + e for e in entries)
    new = f'<rampList num="{len(entries)}">\n{body}\n        </rampList>'
    text2 = re.sub(r"<rampList\b[^>]*>.*?</rampList>", new, text, count=1, flags=re.DOTALL)
    if text2 == text:
        text2 = re.sub(r'<rampList\b[^>]*?/>', new, text, count=1)
    return text2.encode("utf-8")


def author_pnp(template_blob: bytes, walkable: np.ndarray) -> bytes:
    """Author CellAttribute_Pnp (the editor's painted pathing/placement layer).

    Per-cell bitfield: ``0x00`` = fully open, ``0xff`` = fully blocked (FrostLE ground truth). Ramp
    cells are ``0x00`` in real maps -- the small per-cell cliff step (8) is what makes them walkable,
    not a pathing flag. We author ``0`` where ``walkable`` is true and ``0xff`` elsewhere, preserving
    the template's small header so we do not inherit stray blockers on our own layout.
    """
    sy, sx = walkable.shape
    hdr = len(template_blob) - sx * sy
    if hdr < 0:
        raise ValueError("Pnp template smaller than grid")
    body = np.where(walkable, 0, 0xFF).astype(np.uint8)
    return template_blob[:hdr] + body.tobytes()


def clear_cell_flags(blob: bytes) -> bytes:
    """Zero a t3CellFlags (LFCT) body so no cells punch a hole through the world.

    Cell-flag value ``0x03`` "punches a hole" in the world terrain (the engine does this at the
    template's original cliffs so cliff *models* can show through without terrain poking out). We
    inherit the template's flags unchanged, so on a flattened map those holes remain and render as
    black / deep-lava voids that appear/disappear with the camera angle. Our flat terrain has no
    cliff models, so every cell should be plain ground (``0x00``): keep the 32-byte header, zero the
    per-cell body.
    """
    if blob[:4] != b"LFCT":
        raise ValueError("not a t3CellFlags (LFCT) blob")
    sx, sy = struct.unpack_from("<II", blob, 24)
    hdr = len(blob) - sx * sy
    if hdr < 0:
        raise ValueError("unexpected t3CellFlags size")
    return blob[:hdr] + b"\x00" * (sx * sy)


def strip_render_relief(terrain_xml: bytes) -> bytes:
    """Zero out the template's precomputed ramp meshes in t3Terrain.xml.

    The editor bakes explicit ramp geometry (``<rampList num="N">...``) at fixed coordinates that
    render regardless of the (flattened) cliff-level grid. On a flat interim map these show up as
    "ghost" cliffs/ramps sitting on flat, fully-walkable ground. Collapsing the list to ``num="0"``
    removes that stale geometry so the flat map renders as flat. Cliff *set* definitions (texture
    references) are left intact -- with a flat cliff grid no cliffs are generated from them anyway.
    """
    text = terrain_xml.decode("utf-8", "replace")
    # <rampList num="16"> ... </rampList>  ->  <rampList num="0"/>
    text = re.sub(r"<rampList\b[^>]*>.*?</rampList>", '<rampList num="0"/>',
                  text, count=1, flags=re.DOTALL)
    # also handle a self-closing form just in case
    text = re.sub(r'<rampList\b[^>]*?/>', '<rampList num="0"/>', text, count=1)
    return text.encode("utf-8")


def parse_texture_info(blob: bytes) -> tuple[int, int, int, list[str], int]:
    """(ver, width, height, texture_names, cell_region_offset) for a t3SyncTextureInfo blob.

    Layout: 16-byte header {magic RTXT, ver, w, h}, a u32, then N null-terminated ASCII texture
    names, then w*h records of 8 bytes each where byte 0 is the base-texture index into ``names``.
    """
    if blob[:4] != b"RTXT":
        raise ValueError("not a t3SyncTextureInfo (RTXT) blob")
    ver, w, h = struct.unpack_from("<3I", blob, 4)
    off = 20  # skip magic(4)+ver(4)+w(4)+h(4)+u32(4)
    names: list[str] = []
    while off < len(blob):
        end = blob.index(b"\x00", off)
        s = blob[off:end]
        if s and all(32 <= c < 127 for c in s):
            names.append(s.decode("ascii"))
            off = end + 1
        else:
            break  # start of the 8-byte-per-cell region
    if len(blob) - off != w * h * 8:
        raise ValueError(f"unexpected RTXT cell region: {len(blob) - off} != {w*h*8}")
    return ver, w, h, names, off


def build_texture_class_grid(tier_grid: np.ndarray, mapir: "MapIR", n_tex: int,
                             off_x: int, off_y: int) -> np.ndarray:
    """Template-sized [ch, cw] uint8 of base-texture indices keyed by *intended* level.

    Each terrain tier gets its own solid texture, sloped ramp cells get a dedicated one, so levels
    stay visually distinct even on the flat interim (where geometry is collapsed but the IR still
    knows every cell's intended elevation). A ramp's flat clamped ends take their plateau's texture.
    ``tier_grid`` is the intended (pre-flatten) tier grid.
    """
    ramp_idx = n_tex - 1                                   # last texture reserved for ramps
    avail = [i for i in range(n_tex) if i != ramp_idx] or [0]
    tier_choices = avail[0::2] + avail[1::2]               # spread so adjacent tiers differ more
    grid = np.zeros(tier_grid.shape, dtype=np.uint8)       # void -> texture 0
    for tier in range(int(tier_grid.max()) + 1 if tier_grid.max() >= 0 else 0):
        grid[tier_grid == tier] = tier_choices[tier % len(tier_choices)]
    ch, cw = tier_grid.shape
    for x, y, lv in ramp_cell_levels(mapir):
        tx, ty = x + off_x, y + off_y
        if 0 <= ty < ch and 0 <= tx < cw:
            grid[ty, tx] = ramp_idx if lv is None else tier_choices[lv % len(tier_choices)]
    return grid


def author_texture_info(blob: bytes, class_grid: np.ndarray) -> bytes:
    """Rewrite each cell's base-texture index (byte 0) from ``class_grid`` ([h, w] uint8).

    Byte 0 is the dominant/base texture index into the name table (bytes 1-7 stay 0 as in stock
    maps). This is the "sync" texture record; the *visible* per-cell texture is driven by the MASK
    (see :func:`author_texture_mask`), which we key from the same ``class_grid`` so both agree.
    """
    ver, w, h, names, off = parse_texture_info(blob)
    n_tex = len(names)
    cg = np.clip(class_grid.astype(np.int32), 0, max(0, n_tex - 1)).astype(np.uint8)
    cells = np.zeros((h, w, 8), dtype=np.uint8)
    cells[:, :, 0] = cg
    return blob[:off] + cells.tobytes()


def author_texture_mask(blob: bytes, class_grid: np.ndarray) -> bytes:
    """Author a t3TextureMasks (MASK) blob so each cell shows exactly one texture by level.

    MASK layout (SC2Mapster-confirmed and round-trip-verified against the templates):
      * 64-byte header {magic "MASK", u32 ver, u32 unk, u32 sizeX, u32 sizeY, ...}.
      * body = ``n_layers`` layers, one **per texture** (layer i == texture i). Each layer is a
        ``sizeX*sizeY`` grid of **4-bit** alphas packed 2 per byte (**high nibble first**), tiled
        in **64x64-pixel blocks** ordered +x then +y. ``n_layers = (len-64)/(sizeX*sizeY/2)``.
      * resolution is 8x the cell grid (``sizeX == 8*cellW``), i.e. an 8x8 sub-cell block per cell.

    We set the chosen layer's alpha to full (0xF) over each cell's 8x8 block and every other layer
    to 0 -> one solid texture per cell, no blending, no purple. ``class_grid`` is [cellH, cellW].
    """
    if blob[:4] != b"MASK":
        raise ValueError("not a t3TextureMasks (MASK) blob")
    _ver, _unk, sx, sy = struct.unpack_from("<4I", blob, 4)
    hdr = 64
    layer_sz = sx * sy // 2
    n_layers = (len(blob) - hdr) // layer_sz
    cw, ch = sx // 8, sy // 8
    cg = class_grid[:ch, :cw]
    if cg.shape != (ch, cw):
        raise ValueError(f"class_grid {class_grid.shape} smaller than cell dims {(ch, cw)}")
    bpr, bpc = sx // 64, sy // 64
    out = bytearray(blob[:hdr])
    for L in range(n_layers):
        full = np.where(np.repeat(np.repeat(cg == L, 8, axis=0), 8, axis=1), 0x0F, 0).astype(np.uint8)
        # (sy,sx) -> blocks ordered (block_row, block_col), each 64x64 row-major
        blocks = full.reshape(bpc, 64, bpr, 64).transpose(0, 2, 1, 3).reshape(-1)
        packed = ((blocks[0::2] << 4) | blocks[1::2]).astype(np.uint8)   # hi nibble first
        out += packed.tobytes()
    return bytes(out)


_TGA_HEADER = 18
_MINIMAP_RAMP_RGB = (26, 204, 230)
_CLIFF_EDGE_STEP = 32          # larger than any in-ramp step (max +24), smaller than a 64 wall
_MOTTLE_SIGMA = 1.2
_MOTTLE_OUTLIER = 3.0         # luminance MADs kept around a look's median
_MOTTLE_MIN_POOL = 16
_VOID_DARKEN = 0.65            # void = darkest painted texture's median colour, a few shades down


def _minimap_frame(blob: bytes) -> tuple[int, int, int]:
    """(pixel offset, width, height) of a template ``Minimap.tga``.

    Every template stores an uncompressed 24-bit BGR, top-left-origin TGA (18-byte header, 26-byte
    TGA 2.0 footer) whose sides are the playable size rounded up to a power of two. One pixel is
    one cell; the playable rect sits centred (floor) in the image and the top row is the highest y.
    """
    if blob[2] != 2 or blob[16] != 24 or not blob[17] & 0x20:
        raise ValueError("Minimap.tga is not an uncompressed 24-bit top-left TGA")
    iw, ih = struct.unpack_from("<HH", blob, 12)
    return _TGA_HEADER + blob[0], iw, ih


def _minimap_window(iw: int, ih: int, playable) -> tuple[int, int, int, int, int, int]:
    """(x0, y0, sx0, sy0, w, h): the playable rect's pixel origin, the first visible playable
    row/col, and the visible extent (clipped to the image)."""
    l, b, r, t = playable
    pw, ph = r - l, t - b
    x0, y0 = (iw - pw) // 2, (ih - ph) // 2
    sx0, sy0 = max(0, -x0), max(0, -y0)
    return x0, y0, sx0, sy0, min(pw - sx0, iw - x0 - sx0), min(ph - sy0, ih - y0 - sy0)


def read_minimap_cells(blob: bytes, playable, cw: int, ch: int) -> tuple[np.ndarray, np.ndarray]:
    """Template minimap resampled onto the cell grid: ``(rgb [ch, cw, 3], valid [ch, cw])``."""
    start, iw, ih = _minimap_frame(blob)
    img = np.frombuffer(blob[start:start + iw * ih * 3], dtype=np.uint8).reshape(ih, iw, 3)[..., ::-1]
    l, b, r, t = playable
    x0, y0, sx0, sy0, w, h = _minimap_window(iw, ih, playable)
    win = img[y0 + sy0:y0 + sy0 + h, x0 + sx0:x0 + sx0 + w][::-1]
    rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
    valid = np.zeros((ch, cw), dtype=bool)
    Y0 = t - sy0 - h
    rgb[Y0:Y0 + h, l + sx0:l + sx0 + w] = win
    valid[Y0:Y0 + h, l + sx0:l + sx0 + w] = True
    return rgb, valid


def write_minimap_cells(blob: bytes, rgb: np.ndarray, playable) -> bytes:
    """Return ``blob`` with the playable window painted from ``rgb`` ([ch, cw, 3], cell grid) and
    everything outside it black. Header, footer and image size are kept."""
    start, iw, ih = _minimap_frame(blob)
    l, b, r, t = playable
    x0, y0, sx0, sy0, w, h = _minimap_window(iw, ih, playable)
    img = np.zeros((ih, iw, 3), dtype=np.uint8)
    Y0 = t - sy0 - h
    img[y0 + sy0:y0 + sy0 + h, x0 + sx0:x0 + sx0 + w] = rgb[Y0:Y0 + h, l + sx0:l + sx0 + w][::-1]
    return blob[:start] + img[..., ::-1].tobytes() + blob[start + iw * ih * 3:]


def author_minimap(blob: bytes, walkable: np.ndarray, tier_grid: np.ndarray,
                   ramp_mask: np.ndarray, playable: tuple[int, int, int, int]) -> bytes:
    """Preview-coloured ``Minimap.tga``: void black, walkable grey by level, ramps cyan."""
    l, b, r, t = playable
    walk = walkable[b:t, l:r]
    tiers = tier_grid.astype(np.float64)
    rgb = np.zeros(walkable.shape + (3,), dtype=np.uint8)
    if walk.any():
        emin, emax = tiers[walkable].min(), tiers[walkable].max()
        shade = 0.35 + 0.65 * (tiers - emin) / max(emax - emin, 1.0)
        rgb[walkable] = np.rint(255 * shade[walkable])[:, None].astype(np.uint8)
    rgb[ramp_mask & walkable] = _MINIMAP_RAMP_RGB
    return write_minimap_cells(blob, rgb, playable)


def decode_texture_mask(blob: bytes) -> np.ndarray:
    """Per-cell mean alpha (0-15) of every t3TextureMasks layer: ``[n_layers, cellH, cellW]``.
    Inverse of the packing in :func:`author_texture_mask`."""
    if blob[:4] != b"MASK":
        raise ValueError("not a t3TextureMasks (MASK) blob")
    _ver, _unk, sx, sy = struct.unpack_from("<4I", blob, 4)
    layer_sz = sx * sy // 2
    n_layers = (len(blob) - 64) // layer_sz
    out = np.empty((n_layers, sy // 8, sx // 8), dtype=np.float32)
    for L in range(n_layers):
        packed = np.frombuffer(blob[64 + L * layer_sz:64 + (L + 1) * layer_sz], dtype=np.uint8)
        px = np.empty(packed.size * 2, dtype=np.uint8)
        px[0::2], px[1::2] = packed >> 4, packed & 0x0F
        full = px.reshape(sy // 64, sx // 64, 64, 64).transpose(0, 2, 1, 3).reshape(sy, sx)
        out[L] = full.reshape(sy // 8, 8, sx // 8, 8).mean(axis=(1, 3))
    return out


def dominant_texture(alpha: np.ndarray) -> np.ndarray:
    """Visible texture per cell: the topmost layer at least half opaque (later layers paint over
    earlier ones), else the most opaque layer."""
    top = np.argmax(alpha, axis=0)
    for L in range(alpha.shape[0]):
        top = np.where(alpha[L] >= 8, L, top)
    return top.astype(np.int32)


def _minimap_classes(tex: np.ndarray, cliff: np.ndarray, n_tex: int) -> np.ndarray:
    """Look class per cell: texture index, then ``n_tex`` void, ``n_tex+1`` cliff top (a 4-
    neighbour drops by a wall step), ``n_tex+2`` cliff foot (a 4-neighbour rises by one)."""
    c = cliff.astype(np.int32)
    pad = np.pad(c, 1, mode="edge")
    nbrs = [pad[1:-1, :-2], pad[1:-1, 2:], pad[:-2, 1:-1], pad[2:, 1:-1]]
    top = np.zeros(c.shape, dtype=bool)
    foot = np.zeros(c.shape, dtype=bool)
    for n in nbrs:
        top |= n <= c - _CLIFF_EDGE_STEP
        foot |= n >= c + _CLIFF_EDGE_STEP
    cls = tex.astype(np.int32).copy()
    cls[c == 0] = n_tex
    cls[foot] = n_tex + 2
    cls[top & (c > 0)] = n_tex + 1
    return cls


def author_minimap_textured(blob: bytes, template_cliff: np.ndarray, template_alpha: np.ndarray,
                            cliff: np.ndarray, class_grid: np.ndarray,
                            playable: tuple[int, int, int, int], seed: int = 0) -> bytes:
    """Editor-look ``Minimap.tga`` rebuilt from the template's own editor render.

    The template minimap already shows how the editor draws each of its textures, its cliff tops
    and feet, and its void under this tileset's lighting. Each template cell is labelled with one
    of those looks (:func:`_minimap_classes` over its dominant texture and CLIF), and each look
    keeps its pixels minus luminance outliers (doodads, shadows) as a pool sorted dark to light.
    Our cells get the same labels from the exported cliff grid and texture ``class_grid``; a
    smooth random field picks from the pool by rank, giving mottled ground with no visible tiling.
    Void is one flat colour a few shades darker than the darkest texture painted on our map.
    """
    from scipy import ndimage

    ch, cw = cliff.shape
    n_tex = template_alpha.shape[0]
    t_rgb, valid = read_minimap_cells(blob, playable, cw, ch)
    t_cls = _minimap_classes(dominant_texture(template_alpha)[:ch, :cw], template_cliff, n_tex)
    cls = _minimap_classes(np.clip(class_grid, 0, n_tex - 1), cliff, n_tex)

    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.standard_normal((ch, cw)), sigma=_MOTTLE_SIGMA)
    rank = (np.argsort(np.argsort(field, axis=None)).reshape(ch, cw) + 0.5) / field.size
    lum_w = np.array([0.299, 0.587, 0.114])
    ground = t_rgb[valid & (template_cliff > 0)]
    fallback = ground.mean(axis=0) if len(ground) else np.full(3, 96.0)

    rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
    darkest = None
    for k in np.unique(cls):
        if k == n_tex:
            continue
        where = cls == k
        pool = t_rgb[(t_cls == k) & valid].astype(np.float64)
        if len(pool):
            lum = pool @ lum_w
            med = np.median(lum)
            mad = np.median(np.abs(lum - med)) + 1.0
            keep = np.abs(lum - med) <= _MOTTLE_OUTLIER * mad
            pool, lum = pool[keep], lum[keep]
        if len(pool) < _MOTTLE_MIN_POOL:
            scale = 0.5 if k >= n_tex else 1.0
            rgb[where] = np.clip(fallback * scale, 0, 255).astype(np.uint8)
            continue
        order = np.argsort(lum)
        pool, lum = pool[order], lum[order]
        if k < n_tex:
            mid = pool[len(pool) // 2]
            if darkest is None or lum[len(pool) // 2] < darkest[0]:
                darkest = (lum[len(pool) // 2], mid)
        q = rank[where]
        rgb[where] = pool[np.minimum((q * len(pool)).astype(int), len(pool) - 1)]
    base = darkest[1] if darkest is not None else fallback
    rgb[cls == n_tex] = np.clip(np.rint(base * _VOID_DARKEN), 0, 255).astype(np.uint8)
    return write_minimap_cells(blob, rgb, playable)


def build_tier_grid(mapir: MapIR, cw: int, ch: int, off_x: int, off_y: int,
                    n_tiers: int) -> np.ndarray:
    """Template-sized (ch x cw) cell grid of tier indices: -1 == void, else clamped elevation.

    The MapIR grid is placed at (off_x, off_y); anything outside the template is cropped.
    """
    grid = np.full((ch, cw), -1, dtype=np.int16)
    h, w = mapir.walkable.shape
    sx0 = max(0, -off_x)
    sy0 = max(0, -off_y)
    sx1 = min(w, cw - off_x)
    sy1 = min(h, ch - off_y)
    if sx1 <= sx0 or sy1 <= sy0:
        return grid
    walk = mapir.walkable[sy0:sy1, sx0:sx1]
    elev = np.clip(mapir.elevation[sy0:sy1, sx0:sx1].astype(np.int32), 0, n_tiers - 1)
    tier = np.where(walk, elev, -1)
    dy, dx = off_y + sy0, off_x + sx0
    grid[dy : dy + (sy1 - sy0), dx : dx + (sx1 - sx0)] = tier.astype(np.int16)
    return grid


def build_cliff_cells(tier_grid: np.ndarray, palette: Palette) -> np.ndarray:
    """Map a tier-index cell grid to a uint16 cliff-level grid (void 0, else the tier cliff).

    Plateaus only -- ramp gradients are overlaid afterwards by :func:`author_ramp_cliffs`.
    """
    grid = np.zeros(tier_grid.shape, dtype=np.uint16)
    for idx, (cliff, _s, _h) in enumerate(palette.tiers):
        grid[tier_grid == idx] = cliff
    return grid


# backwards-compatible alias (older callers / tests)
encode_cliff_from_tiers = build_cliff_cells


def author_ramp_cliffs(cliff_cell: np.ndarray, tier_grid: np.ndarray, ramps,
                       palette: Palette, off_x: int, off_y: int) -> int:
    """Overlay walkable ramp gradients onto a plateau cliff grid (in place).

    For each IR ramp (a single-step transition between two tiers) we replace its cells' cliff
    values with the harvested ramp SUB-LEVELS (72, 80, ...) stepping from the low tier up to the
    high tier, ordered by each cell's fractional distance between the two plateaus. Consecutive
    cells then differ by one 8-step, which the engine treats as a traversable ramp (a direct
    64-jump is a wall). Returns the number of ramps successfully authored.
    """
    from scipy import ndimage

    ch, cw = cliff_cell.shape
    all_ramp = np.zeros((ch, cw), dtype=bool)
    per_ramp: list[list[tuple[int, int]]] = []
    for r in ramps:
        cells = []
        for (x, y) in r.cells:
            X, Y = int(x) + off_x, int(y) + off_y
            if 0 <= X < cw and 0 <= Y < ch:
                all_ramp[Y, X] = True
                cells.append((X, Y))
        per_ramp.append(cells)

    plateau = ~all_ramp
    authored = 0
    for r, cells in zip(ramps, per_ramp):
        lo, hi = r.low_level, r.high_level
        if lo is None or hi is None or not cells:
            continue
        lo, hi = int(lo), int(hi)
        if hi <= lo or hi >= palette.n_tiers:
            continue
        lo_cliff = palette.tiers[lo][0]
        hi_cliff = palette.tiers[hi][0]
        subs = palette.sublevels_between(lo_cliff, hi_cliff)
        if not subs:
            continue
        R = np.zeros((ch, cw), dtype=bool)
        for (X, Y) in cells:
            R[Y, X] = True
        dil = ndimage.binary_dilation(R)
        low_b = dil & plateau & (cliff_cell == lo_cliff)
        high_b = dil & plateau & (cliff_cell == hi_cliff)
        if not low_b.any() or not high_b.any():
            low_b = dil & plateau & (tier_grid == lo)
            high_b = dil & plateau & (tier_grid == hi)
        if not low_b.any() or not high_b.any():
            continue
        # LINEAR staircase along the ramp's snapped flow direction: parallel isolines
        # (perpendicular to u) stepping +8 per cell, exactly like a real FrostLE ramp. This matches
        # the rampList quad's 1D slope. (A BFS-from-plateau gradient makes CURVED isolines that
        # follow the plateau outline, so the quad's fixed direction only lines up in patches and the
        # engine won't path most of the ramp.) direction MUST match build_ramp_list's quad exactly,
        # so both take it from ramp_uphill.
        direction, _ = ramp_uphill(r)
        ux, uy = _RAMP_DIR_U[direction]
        # Staircase gradient (shared with the validator's engine_cliff_grid via ir.ramp_cell_cliffs
        # so export + oracle never drift). GOLD-STANDARD: the isoline gradient steps +8 per lattice
        # axis-cell up to hi-16 (112) then jumps +16 to the high plateau (128), SKIPPING hi-8 (120) --
        # exactly what every gold ramp authors (findings ?4.7). The +16 top step is walkable because
        # the <rampList> quad covers the high-plateau interface.
        for (X, Y), cval in zip(cells, ramp_cell_cliffs(cells, ux, uy, lo_cliff, hi_cliff)):
            cliff_cell[Y, X] = cval
        authored += 1
    return authored


def _interp_heights(vc: np.ndarray, palette: Palette) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex (smap_raw, hmap_tuple) by interpolating tier heights over the vertex's cliff level.

    Ramp cells carry cliff sub-levels (72, 80, ...) between two tier cliffs, so interpolating the
    flat tier heights across them yields a genuine, monotone height slope -- which is what the
    engine's terrain analysis needs to treat the cells as a walkable ramp rather than a wall.
    Void (cliff 0) and flat tiers reproduce their exact harvested encodings (flat == buildable).

    ? ``t3SyncHeightMap`` packs TWO independent value types into each u32 vertex: ``int16 height``
    (low word) + ``uint16 mask`` (high word). Interpolating the PACKED word blends the mask's bit-16
    steps into the gameplay height, producing garbage ramp heights (measured: our ramp SMAP came out
    -30466..27325 instead of gold's clean 2045->2557 monotone slope; the engine then never sees a
    walkable slope so the ramp fails). So SPLIT the fields: interpolate the SIGNED height linearly and
    STEP the mask to the LOWER tier (a ramp vertex is flagged at its low plateau's level, exactly as
    gold maps encode it). ``t3HeightMap``'s three columns (adjust, base, mask) are already SEPARATE
    u16 fields, so adjust/base interpolate cleanly; only its level-index mask must likewise STEP.
    """
    cliffs = np.array([c for c, _s, _h in palette.tiers], dtype=np.float64)
    smaps_packed = np.array([s for _c, s, _h in palette.tiers], dtype=np.uint32)
    hcols = np.array([list(h) for _c, _s, h in palette.tiers], dtype=np.float64)  # (n,3)

    heights = (smaps_packed & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.float64)
    masks = (smaps_packed >> 16).astype(np.uint16)

    vcf = vc.astype(np.float64)
    lo_idx = np.clip(np.searchsorted(cliffs, vcf, side="right") - 1, 0, len(cliffs) - 1)

    smap_h = np.clip(np.rint(np.interp(vcf, cliffs, heights)), -32768, 32767)
    smap_h = smap_h.astype(np.int64).astype(np.int16).view(np.uint16).astype(np.uint32)
    smap = (masks[lo_idx].astype(np.uint32) << 16) | smap_h

    hmap = np.empty(vc.shape + (3,), dtype=np.float64)
    hmap[..., 0] = np.interp(vcf, cliffs, hcols[:, 0])   # adjustments (fine dents) -- linear
    hmap[..., 1] = np.interp(vcf, cliffs, hcols[:, 1])   # base plateau height     -- linear
    hmap[..., 2] = hcols[lo_idx, 2]                       # level-index mask        -- STEP

    void = vc < int(cliffs[0])            # cliff 0 (and anything below the lowest tier) == void
    smap[void] = palette.void_smap
    hmap[void] = np.array(palette.void_hmap, dtype=np.float64)
    return smap.astype("<u4"), np.rint(hmap).astype("<u2")


def _vertex_cliff(cliff_cell: np.ndarray, sw: int, sh: int,
                  ramp_mask: np.ndarray | None) -> np.ndarray:
    cc = cliff_cell.astype(np.int32)
    if ramp_mask is None:
        return _vertex_cliff_max(cc, sw, sh)
    return _vertex_cliff_ramp_aware(cc, ramp_mask, sw, sh)


def build_smap(cliff_cell: np.ndarray, template_smap: bytes, palette: Palette,
               ramp_mask: np.ndarray | None = None) -> bytes:
    """Author t3SyncHeightMap by interpolating tier heights over each vertex's (max) cliff level."""
    sw = struct.unpack_from("<I", template_smap, 8)[0]
    sh = struct.unpack_from("<I", template_smap, 12)[0]
    vc = _vertex_cliff(cliff_cell, sw, sh, ramp_mask)
    smap, _hmap = _interp_heights(vc, palette)
    return template_smap[:_SMAP_HEADER] + smap.tobytes()


def build_hmap(cliff_cell: np.ndarray, template_hmap: bytes, palette: Palette,
               ramp_mask: np.ndarray | None = None) -> bytes:
    """Author t3HeightMap (rendered layer) by interpolating tier tuples over each vertex's cliff."""
    sw = struct.unpack_from("<I", template_hmap, 8)[0]
    sh = struct.unpack_from("<I", template_hmap, 12)[0]
    vc = _vertex_cliff(cliff_cell, sw, sh, ramp_mask)
    _smap, hmap = _interp_heights(vc, palette)
    return template_hmap[:_HMAP_HEADER] + hmap.tobytes()


# --------------------------------------------------------------------------- #
# Objects XML
# --------------------------------------------------------------------------- #
_UNIT_TYPE = {ResourceKind.MINERAL: "MineralField", ResourceKind.GEYSER: "VespeneGeyser"}


def _oid(*parts) -> int:
    """Deterministic 31-bit object id from its defining fields."""
    h = 2166136261
    for p in parts:
        for b in str(p).encode():
            h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h & 0x7FFFFFFF


def build_objects_xml(mapir: MapIR, off_x: float, off_y: float) -> str:
    """Emit a PlacedObjects XML with our start locations + resources (no doodads).

    Cell (x, y) maps to SC2 world point (x + off_x + 0.5, y + off_y + 0.5) -- cell centres.
    """
    lines = ['<?xml version="1.0" encoding="utf-8"?>', '<PlacedObjects Version="27">']

    def point(px: float, py: float, ptype: str, name: str) -> str:
        oid = _oid(ptype, name, round(px, 2), round(py, 2))
        return (f'    <ObjectPoint Id="{oid}" Position="{px:g},{py:g},0" Scale="1,1,1" '
                f'Type="{ptype}" Name="{name}" Color="0,0,0,0"/>')

    def unit(px: float, py: float, utype: str) -> str:
        oid = _oid(utype, round(px, 2), round(py, 2))
        return (f'    <ObjectUnit Id="{oid}" Position="{px:g},{py:g},0" Scale="1,1,1" '
                f'UnitType="{utype}">\n'
                f'        <Flag Index="ForcePlacement" Value="1"/>\n'
                f'    </ObjectUnit>')

    # start locations: the two MAIN bases
    mains = [b for b in mapir.bases if b.kind == BaseKind.MAIN]
    for i, b in enumerate(mains, start=1):
        lines.append(point(b.x + off_x + 0.5, b.y + off_y + 0.5, "StartLoc",
                           f"Start Location {i:03d}"))

    # resources
    for r in mapir.resources:
        if r.kind not in _UNIT_TYPE:
            continue
        utype = r.unit_type or _UNIT_TYPE[r.kind]
        lines.append(unit(r.x + off_x + 0.5, r.y + off_y + 0.5, utype))

    lines.append("</PlacedObjects>")
    return "\n".join(lines) + "\n"


def count_objects(xml: str) -> tuple[int, int]:
    """(n start locations, n resource units) in an Objects XML string."""
    return xml.count('Type="StartLoc"'), xml.count("<ObjectUnit ")


# --------------------------------------------------------------------------- #
# MapInfo (playable area)
# --------------------------------------------------------------------------- #
def _mapinfo_playable_offset(data: bytes) -> tuple[int, int, int]:
    """(offset of the 4x int32 playable bounds, grid width, grid height) in a MapInfo blob.

    Layout (version 39): 44-byte fixed header, then two null-terminated strings (tileset,
    map name), then playable bounds as int32 left, bottom, right, top. We anchor on the two
    string terminators, so the exact fixed-header length past 44 doesn't matter.
    """
    if data[:4] != b"IpaM":   # "MapI" stored little-endian
        raise ValueError(f"not a MapInfo file (magic={data[:4]!r})")
    width, height = struct.unpack_from("<2I", data, 16)
    off = data.index(b"\x00", 44) + 1          # end of tileset string
    off = data.index(b"\x00", off) + 1         # end of map-name string
    return off, width, height


def patch_mapinfo_playable(data: bytes, left: int, bottom: int,
                           right: int, top: int) -> bytes:
    """Rewrite MapInfo's playable bounds (left, bottom, right, top), clamped inside the grid.

    Our generated maps are larger than the template's original playable region, so the bounds
    must be widened or every base past the old edge sits in void (unplayable == unbuildable).
    The IR carries its own void-bordered playable region, so we map that in rather than using
    the full grid (which would leave no border and fail to load).
    """
    off, gw, gh = _mapinfo_playable_offset(data)
    left = max(1, left)
    bottom = max(1, bottom)
    right = min(gw - 1, right)
    top = min(gh - 1, top)
    return data[:off] + struct.pack("<4i", left, bottom, right, top) + data[off + 16:]


# --- Map name / rename (safe: name strings are decoupled from the MapInfo bounds trap, ?7) -----
#
# The name the game shows on the loading screen / lobby is NOT in ``MapInfo`` -- it is a localized
# game-string ``DocInfo/Name`` stored in two places, both of which we can edit safely (they carry
# no coupled camera/playable-area fields):
#   * ``DocumentHeader`` (binary): a localized-string table. Fixed pre-table header, then a u32
#     record count, then ``count`` records of {u16 keyLen, key, char[4] locale, u16 valLen, value}.
#     The locale tag is the locale code REVERSED (enUS -> "SUne", frFR -> "RFrf", deDE -> "EDed").
#     Renaming replaces the value of every ``DocInfo/Name`` record; the record count is unchanged
#     so no other field moves. Verified to parse to EOF with count-match on every template.
#   * ``<locale>.SC2Data\LocalizedData\GameStrings.txt`` (UTF-8, one ``key=value`` per line): the
#     per-locale text table the client actually loads at runtime; rename the ``DocInfo/Name`` line.
# We rewrite BOTH so the new name shows regardless of the client's locale. This also avoids the
# template's asset-cache collision noted in ?2 (the map no longer shares the ladder map's name).

_DOC_NAME_KEY = b"DocInfo/Name"


def _docheader_table_start(blob: bytes) -> tuple[int, int]:
    """Return ``(table_start, record_count)`` for a ``DocumentHeader`` localized-string table.

    The pre-table region (magic ``H2CS`` + attributes + dependency strings) varies in length, so
    we locate the table by finding the earliest offset whose records parse cleanly to EOF and whose
    preceding u32 equals the parsed record count.
    """
    # candidate table starts sit just before the first localized key (keys begin "DocInfo/" or
    # "MapInfo/"); the u32 count is the 4 bytes immediately before the first record.
    firsts = [i for i in (blob.find(b"DocInfo/"), blob.find(b"MapInfo/")) if i != -1]
    if not firsts:
        raise ValueError("DocumentHeader: no localized-string keys found")
    upper = min(firsts)
    for t in range(8, upper):
        n = _docheader_parse_count(blob, t)
        if n is not None and t >= 4 and struct.unpack_from("<I", blob, t - 4)[0] == n:
            return t, n
    raise ValueError("DocumentHeader: could not locate localized-string table")


def _docheader_parse_count(blob: bytes, start: int) -> int | None:
    """Number of records if the table starting at ``start`` parses exactly to EOF, else None."""
    o = start
    n = 0
    end = len(blob)
    while o < end:
        if o + 2 > end:
            return None
        klen = struct.unpack_from("<H", blob, o)[0]
        if klen == 0 or klen > 256:
            return None
        p2 = o + 2 + klen
        if p2 + 6 > end:
            return None
        key = blob[o + 2:p2]
        if not all(32 <= c < 127 for c in key):
            return None
        vlen = struct.unpack_from("<H", blob, p2 + 4)[0]
        vstart = p2 + 6
        if vstart + vlen > end:
            return None
        o = vstart + vlen
        n += 1
    return n if o == end else None


def set_document_header_name(blob: bytes, new_name: str,
                             key: bytes = _DOC_NAME_KEY) -> bytes:
    """Return ``DocumentHeader`` with every ``key`` record's value replaced by ``new_name``.

    Rebuilds the localized-string table (record count unchanged), leaving the pre-table header and
    all other records byte-identical.
    """
    start, count = _docheader_table_start(blob)
    val = new_name.encode("utf-8")
    out = bytearray(blob[:start])
    o = start
    for _ in range(count):
        klen = struct.unpack_from("<H", blob, o)[0]
        k = blob[o + 2:o + 2 + klen]
        p2 = o + 2 + klen
        loc = blob[p2:p2 + 4]
        vlen = struct.unpack_from("<H", blob, p2 + 4)[0]
        vstart = p2 + 6
        v = val if k == key else blob[vstart:vstart + vlen]
        out += struct.pack("<H", klen) + k + loc + struct.pack("<H", len(v)) + v
        o = vstart + vlen
    out += blob[o:]                                    # trailing bytes (normally none)
    return bytes(out)


def set_gamestrings_name(data: bytes, new_name: str,
                         key: str = "DocInfo/Name") -> bytes:
    """Return a ``GameStrings.txt`` blob with the ``key=...`` line's value set to ``new_name``.

    Preserves the file's UTF-8 BOM and line-ending style; a no-op if the key is absent.
    """
    bom = b"\xef\xbb\xbf" if data.startswith(b"\xef\xbb\xbf") else b""
    text = data[len(bom):].decode("utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"
    prefix = key + "="
    lines = text.split(nl)
    changed = False
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = prefix + new_name
            changed = True
    if not changed:
        return data
    return bom + nl.join(lines).encode("utf-8")
