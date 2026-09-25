"""Base layouts: every real base's townhall cell and gold resource formation, decided on the
skeleton before anything is painted, so room footprints and the rasterizer both see the exact core
each base reserves."""
from __future__ import annotations

import math
from dataclasses import dataclass

from sc2mapgen.ir import BaseKind, ResourceKind

_REAL = (BaseKind.MAIN, BaseKind.NATURAL, BaseKind.BASE)


# Gold L-shaped base formations (findings §7), as offsets from the townhall cell with the mineral
# corner in the (-x, -y) quadrant: a 5-patch arm on the -x side and a 3-patch arm on the -y side,
# one geyser capping each arm. Minerals are (dx, dy, is_750); a 2x1 mineral field sits on a
# half-integer x, a 3x3 geyser on integer x and y. Only axis flips are legal orientations -- a
# transpose would turn the 2x1 fields into 1x2.
_L_FORMATIONS = {
    "A": (
        ((-7.5, -1, True), (-6.5, -4, True), (-6.5, -2, False), (-6.5, 0, False),
         (-5.5, -5, True), (-2.5, -6, False), (-0.5, -7, True), (0.5, -6, False)),
        ((-7, 3), (4, -7)),
    ),
    "B": (
        ((-7.5, -1, True), (-6.5, -4, True), (-6.5, -2, False), (-6.5, 0, False),
         (-5.5, -5, False), (-4.5, -6, True), (-1.5, -7, True), (-0.5, -6, False)),
        ((-7, 3), (3, -7)),
    ),
}
_TOWNHALL_HALF = 2   # 5x5 footprint


def base_townhall_cell(b) -> tuple[int, int]:
    """The integer cell nearest a base's position. The exporter adds +0.5 per axis, giving the
    .5-aligned world centre gold townhalls use. ``floor(v + 0.5)`` rather than ``round`` so the
    snap commutes with the ``msX - x`` cell mirror."""
    return int(math.floor(b.x + 0.5)), int(math.floor(b.y + 0.5))


def mineral_cells(x: float, y: float) -> list[tuple[int, int]]:
    return [(int(x - 0.5), int(y)), (int(x + 0.5), int(y))]


def geyser_cells(x: float, y: float) -> list[tuple[int, int]]:
    return [(int(x) + ox, int(y) + oy) for oy in (-1, 0, 1) for ox in (-1, 0, 1)]


def townhall_cells(tx: int, ty: int) -> list[tuple[int, int]]:
    r = _TOWNHALL_HALF
    return [(tx + ox, ty + oy) for oy in range(-r, r + 1) for ox in range(-r, r + 1)]


@dataclass(frozen=True)
class BaseLayout:
    """A real base's townhall cell plus the gold L formation around it. ``flip`` multiplies the
    canonical (-x, -y)-cornered offsets of ``_L_FORMATIONS[formation]``."""

    town: tuple[int, int]
    formation: str
    flip: tuple[int, int]
    level: int
    town_margin: int = 1
    res_margin: int = 1
    ring: int = 3

    def resources(self) -> list[tuple[ResourceKind, float, float, bool]]:
        (tx, ty), (fx, fy) = self.town, self.flip
        minerals, geysers = _L_FORMATIONS[self.formation]
        out = [(ResourceKind.MINERAL, tx + fx * dx, ty + fy * dy, is750)
               for dx, dy, is750 in minerals]
        out += [(ResourceKind.GEYSER, tx + fx * dx, ty + fy * dy, False) for dx, dy in geysers]
        return out

    def resource_cells(self) -> list[tuple[int, int]]:
        return [c for k, x, y, _ in self.resources()
                for c in (mineral_cells(x, y) if k == ResourceKind.MINERAL else geyser_cells(x, y))]

    def footprint(self) -> set[tuple[int, int]]:
        """The immutable CORE: every cell that must be flat ground at ``level`` -- the townhall
        grown by ``town_margin`` and each resource grown by ``res_margin``. The ``ring`` of cells
        around it is flat base-level ground where ramps, passages and cliffs attach."""
        tx, ty = self.town
        r = _TOWNHALL_HALF + self.town_margin
        res_margin = self.res_margin
        cells = {(tx + ox, ty + oy) for oy in range(-r, r + 1) for ox in range(-r, r + 1)}
        for x, y in self.resource_cells():
            for oy in range(-res_margin, res_margin + 1):
                for ox in range(-res_margin, res_margin + 1):
                    cells.add((x + ox, y + oy))
        return cells


def plan_base_layouts(skel, mirror, cfg) -> dict[int, BaseLayout]:
    """Decide every real base's townhall cell, formation and corner from the skeleton alone,
    before any terrain is painted, so the rasterizer can reserve the exact footprint.

    The mineral corner is picked, in priority order, so the core doesn't overlap or touch another
    base's core, its resources stay clear of same-level neighbours' (python-sc2 grouping), it
    stays out of its own corridors, and it faces away from them (mineral line at the back). The
    townhall is clamped so the whole core lies inside the painted canvas. One base of each mirror
    pair is planned and its partner is its exact mirror image."""
    bases = skel.bases
    level = [b.level for b in bases]
    pa = skel.playable
    sym = getattr(skel, "symmetry", "rot180")
    margins = dict(town_margin=cfg.base_town_margin, res_margin=cfg.base_res_margin,
                   ring=cfg.base_ring)
    cx0, cy0 = pa.x + pa.width / 2.0, pa.y + pa.height / 2.0
    msx, msy = 2 * pa.x + pa.width, 2 * pa.y + pa.height
    # inclusive cell bounds, symmetric under the msX - x mirror, one cell inside the canvas
    x_lo, x_hi = pa.x + cfg.canvas_border + 1, pa.x + pa.width - cfg.canvas_border - 1
    y_lo, y_hi = pa.y + cfg.canvas_border + 1, pa.y + pa.height - cfg.canvas_border - 1

    adj: dict[int, list[int]] = {i: [] for i in range(len(bases))}
    for e in skel.edges:
        adj[e.a].append(e.b)
        adj[e.b].append(e.a)

    def orient(i: int) -> float:
        vx = vy = 0.0
        for j in adj[i]:
            dx, dy = bases[j].x - bases[i].x, bases[j].y - bases[i].y
            d = math.hypot(dx, dy) or 1.0
            vx += dx / d
            vy += dy / d
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:      # no edges: face away from map center
            vx, vy = bases[i].x - cx0, bases[i].y - cy0
        return math.atan2(-vy, -vx)

    def formation_for(i: int) -> str:
        key = f"{skel.seed}:{i}"
        hsh = 2166136261
        for ch in key.encode():
            hsh = ((hsh ^ ch) * 16777619) & 0xFFFFFFFF
        return "A" if (hsh % 1000) < cfg.res_formation_a_share * 1000 else "B"

    def clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    plans: dict[int, BaseLayout] = {}
    for i, b in enumerate(bases):
        if b.kind not in _REAL or i in plans:
            continue
        th0 = orient(i)
        tx, ty = base_townhall_cell(b)
        name = formation_for(i)
        taken: set[tuple[int, int]] = set()
        for k, lay_k in plans.items():
            taken |= lay_k.footprint()
        tr = _TOWNHALL_HALF + cfg.base_town_margin
        for k, other in enumerate(bases):
            if k != i and k not in plans and other.kind in _REAL:
                ox, oy = base_townhall_cell(other)
                taken |= {(ox + dx, oy + dy) for dx in range(-tr, tr + 1) for dy in range(-tr, tr + 1)}
        corridors = []
        for j in adj[i]:
            dx, dy = bases[j].x - b.x, bases[j].y - b.y
            d = math.hypot(dx, dy) or 1.0
            corridors.append((dx / d, dy / d, d))

        def corner_key(c):
            # core cells lying in any incident corridor's band (where its staircase will be cut)
            lay_c = BaseLayout((tx, ty), name, (-c[0], -c[1]), int(level[i]), **margins)
            cells_c = lay_c.footprint()
            # cores may neither overlap nor touch (a cliff wall must fit between two floors)
            clash = sum(1 for x, y in cells_c
                        if any((x + ox, y + oy) in taken for ox in (-1, 0, 1) for oy in (-1, 0, 1)))
            blocked = 0
            for x, y in cells_c:
                px, py = x - tx, y - ty
                for ux, uy, d in corridors:
                    along = px * ux + py * uy
                    if 0.0 < along < d and abs(px * uy - py * ux) < cfg.base_corridor_half:
                        blocked += 1
                        break
            # python-sc2 greedily merges nearby same-height resources into one expansion, so keep
            # this base's resources clear of every same-level neighbour's (an unplanned neighbour's
            # resources can reach ~8.5 from its townhall in any direction)
            rs = [(r[1], r[2]) for r in lay_c.resources()]
            merges = 0
            for k, other in enumerate(bases):
                if k == i or other.kind not in _REAL or int(level[k]) != int(level[i]):
                    continue
                if k in plans:
                    gap = min(math.hypot(x - r[1], y - r[2])
                              for x, y in rs for r in plans[k].resources())
                else:
                    gap = min(math.hypot(x - other.x, y - other.y) for x, y in rs) - 8.5
                merges += gap < cfg.res_group_clearance
            a = math.atan2(c[1], c[0])
            return (clash, merges, blocked, abs(math.remainder(a - th0, math.tau)))

        sx, sy = min(((sx, sy) for sx in (-1, 1) for sy in (-1, 1)), key=corner_key)
        lay = BaseLayout((tx, ty), name, (-sx, -sy), int(level[i]), **margins)
        fp = lay.footprint()
        dxs = [x - tx for x, _ in fp]
        dys = [y - ty for _, y in fp]
        tx = clamp(tx, x_lo - min(dxs), x_hi - max(dxs))
        ty = clamp(ty, y_lo - min(dys), y_hi - max(dys))
        lay = BaseLayout((tx, ty), lay.formation, lay.flip, lay.level, **margins)
        plans[i] = lay
        j = mirror[i]
        if j != i:
            (fx, fy) = lay.flip
            if sym == "mirror_lr":
                plans[j] = BaseLayout((msx - tx, ty), lay.formation, (-fx, fy), int(level[j]), **margins)
            elif sym == "mirror_ud":
                plans[j] = BaseLayout((tx, msy - ty), lay.formation, (fx, -fy), int(level[j]), **margins)
            else:
                plans[j] = BaseLayout((msx - tx, msy - ty), lay.formation, (-fx, -fy),
                                      int(level[j]), **margins)
    return plans


def layout_problems(skel, cfg) -> list[str]:
    """Why the planned layouts can't all be built (empty == fine): two cores that overlap or touch
    (a cliff wall must fit between two floors), or two same-level bases whose resources are close
    enough for python-sc2's expansion finder to merge them into one group."""
    lays = [(i, b.layout) for i, b in enumerate(skel.bases) if b.layout is not None]
    grown = {i: {(x + ox, y + oy) for x, y in lay.footprint() for ox in (-1, 0, 1) for oy in (-1, 0, 1)}
             for i, lay in lays}
    probs = []
    for n, (i, a) in enumerate(lays):
        fa = a.footprint()
        ra = [(r[1], r[2]) for r in a.resources()]
        for j, b in lays[n + 1:]:
            if fa & grown[j]:
                probs.append(f"base {i} core touches base {j} core")
            elif a.level == b.level:
                gap = min(math.hypot(x - r[1], y - r[2]) for x, y in ra for r in b.resources())
                if gap < cfg.res_group_clearance:
                    probs.append(f"base {i} resources {gap:.1f} from base {j}'s")
    return probs
