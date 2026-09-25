"""Room footprints: the final painted box of every skeleton node, decided on the skeleton.

Each node becomes a rotated rectangle (``SkelBase.rx``/``ry`` half-extents, ``angle``). Interior
rooms grow by one shared multiplier until the map's walkable fraction lands in
``GenConfig.target_openness``; MAIN/NATURAL use a fixed modest growth. A room never grows over a
corridor on another level that it isn't an endpoint of, and never shrinks below its own width or
its widest corridor. The rasterizer paints these boxes as given.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from sc2mapgen.ir import BaseKind

_REAL = (BaseKind.MAIN, BaseKind.NATURAL, BaseKind.BASE)


def box_mask(shape, center, rx, ry, angle) -> np.ndarray:
    """Cells of a rotated ``2rx x 2ry`` rectangle centred on ``center``."""
    h, w = shape
    cx, cy = center
    rad = math.hypot(rx, ry)
    out = np.zeros(shape, dtype=bool)
    xmin, xmax = max(0, int(np.floor(cx - rad - 1))), min(w - 1, int(np.ceil(cx + rad + 1)))
    ymin, ymax = max(0, int(np.floor(cy - rad - 1))), min(h - 1, int(np.ceil(cy + rad + 1)))
    if xmin > xmax or ymin > ymax:
        return out
    ys, xs = np.mgrid[ymin : ymax + 1, xmin : xmax + 1].astype(float)
    ca, sa = math.cos(angle), math.sin(angle)
    dx, dy = xs - cx, ys - cy
    out[ymin : ymax + 1, xmin : xmax + 1] = (
        (np.abs(dx * ca + dy * sa) <= rx) & (np.abs(-dx * sa + dy * ca) <= ry))
    return out


def corridor_mask(shape, p0, p1, width) -> np.ndarray:
    """Cells of a flat-ended ``width``-thick lane from ``p0`` to ``p1``."""
    h, w = shape
    (x0, y0), (x1, y1) = p0, p1
    r = width / 2.0
    out = np.zeros(shape, dtype=bool)
    xmin, xmax = max(0, int(np.floor(min(x0, x1) - r - 1))), min(w - 1, int(np.ceil(max(x0, x1) + r + 1)))
    ymin, ymax = max(0, int(np.floor(min(y0, y1) - r - 1))), min(h - 1, int(np.ceil(max(y0, y1) + r + 1)))
    if xmin > xmax or ymin > ymax:
        return out
    ys, xs = np.mgrid[ymin : ymax + 1, xmin : xmax + 1].astype(float)
    dx, dy = x1 - x0, y1 - y0
    seg = dx * dx + dy * dy or 1.0
    t = ((xs - x0) * dx + (ys - y0) * dy) / seg
    perp = np.abs((xs - x0) * (-dy) + (ys - y0) * dx) / math.sqrt(seg)
    out[ymin : ymax + 1, xmin : xmax + 1] = (t >= 0.0) & (t <= 1.0) & (perp <= r)
    return out


def is_nat_out(bases, e) -> bool:
    """The natural's one edge to the rest of the map (not its main's ramp)."""
    kinds = {bases[e.a].kind, bases[e.b].kind}
    return BaseKind.NATURAL in kinds and BaseKind.MAIN not in kinds


def painted_width(bases, e, min_width: float, ramp_choke_width: float) -> float:
    """Width an edge is painted at: its skeleton width, floored, and narrowed to a ramp choke when
    it changes level (one <ramp> quad only bridges a narrow band, findings §4.4) -- except the
    natural's out-edge, which keeps its standard width either way."""
    w = max(min_width, e.width)
    if bases[e.a].level != bases[e.b].level and not is_nat_out(bases, e):
        w = min(w, ramp_choke_width)
    return w


def node_centre(b) -> tuple[float, float]:
    """Where a node's room and corridors are anchored: a real base's townhall cell."""
    if b.layout is not None:
        return float(b.layout.town[0]), float(b.layout.town[1])
    return b.x, b.y


def node_footprint(shape, b, rx: float, ry: float, pad_half: float) -> np.ndarray:
    """A node's painted ground: its room box, plus for a real base its flat pad and its core
    grown by its ring."""
    c = node_centre(b)
    m = box_mask(shape, c, rx, ry, b.angle)
    if b.kind in _REAL:
        m |= box_mask(shape, c, pad_half, pad_half, 0.0)
        if b.layout is not None:
            h, w = shape
            core = np.zeros(shape, dtype=bool)
            for x, y in b.layout.footprint():
                if 0 <= x < w and 0 <= y < h:
                    core[y, x] = True
            m |= ndimage.binary_dilation(core, np.ones((3, 3), bool), iterations=b.layout.ring)
    return m


def natural_zones(skel, cfg, sizes=None) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Per natural: ``(natural, out_target, pocket, choke)``.

    ``pocket`` is the natural's footprint grown by ``cfg.pocket_margin``: no other ground on the
    natural's level may enter it. ``choke`` is the first ``cfg.natural_choke_len`` tiles of the
    out-corridor beyond the natural's footprint: no other ground on any level may cover it, so it
    stays the natural's single ``natural_out_width`` opening. ``sizes`` overrides the natural's
    current ``rx/ry``."""
    bases, edges = skel.bases, skel.edges
    shape = (skel.grid_h, skel.grid_w)
    out = []
    for n, b in enumerate(bases):
        if b.kind != BaseKind.NATURAL:
            continue
        outs = [e for e in edges if n in (e.a, e.b) and is_nat_out(bases, e)]
        if len(outs) != 1:
            continue
        e = outs[0]
        t = e.b if e.a == n else e.a
        rx, ry = sizes[n] if sizes else (b.rx, b.ry)
        own = node_footprint(shape, b, rx, ry, cfg.pad_half)
        pocket = ndimage.binary_dilation(own, iterations=cfg.pocket_margin)
        lane = corridor_mask(shape, node_centre(b), node_centre(bases[t]),
                             painted_width(bases, e, cfg.min_edge_paint_width, cfg.ramp_choke_width))
        reach = ndimage.binary_dilation(own, iterations=int(math.ceil(cfg.natural_choke_len)))
        out.append((n, t, pocket, lane & ~own & reach))
    return out


def natural_pocket_problems(skel, cfg) -> list[str]:
    """Why a sized skeleton's naturals are not closed pockets with one standard choke (empty ==
    fine): any other node or corridor over the choke, or on the natural's level inside the pocket."""
    bases, edges = skel.bases, skel.edges
    shape = (skel.grid_h, skel.grid_w)
    probs = []
    for n, t, pocket, choke in natural_zones(skel, cfg):
        lv = bases[n].level
        for k, o in enumerate(bases):
            if k == n or o.rx <= 0:
                continue
            fp = node_footprint(shape, o, o.rx, o.ry, cfg.pad_half)
            if (fp & choke).any():
                probs.append(f"natural {n}: node {k} ({o.kind.name}) covers its choke")
            elif o.level == lv and (fp & pocket).any():
                probs.append(f"natural {n}: node {k} ({o.kind.name}) enters its pocket")
        for e in edges:
            if n in (e.a, e.b):
                continue
            lm = corridor_mask(shape, node_centre(bases[e.a]), node_centre(bases[e.b]),
                               painted_width(bases, e, cfg.min_edge_paint_width, cfg.ramp_choke_width))
            same = bases[e.a].level == bases[e.b].level
            if same and (lm & choke).any():
                probs.append(f"natural {n}: corridor {e.a}-{e.b} covers its choke")
            elif same and bases[e.a].level == lv and (lm & pocket).any():
                probs.append(f"natural {n}: corridor {e.a}-{e.b} enters its pocket")
    return probs


def _mirror_angle(a: float, sym: str) -> float:
    if sym == "mirror_lr":
        return math.pi - a
    if sym == "mirror_ud":
        return -a
    return a


def size_rooms(skel, mirror: list[int], cfg) -> None:
    """Decide every node's final box and write it onto ``SkelBase.rx/ry/angle``.

    Draws each mirror pair's aspect and angle from a dedicated stream (``seed * 2 + 1``), then
    bisects the shared growth multiplier on the union of rooms, same-level corridors and base pads
    until the walkable fraction of the playable rect lands in ``cfg.target_openness``."""
    bases, edges = skel.bases, skel.edges
    n = len(bases)
    h, w = skel.grid_h, skel.grid_w
    pa = skel.playable
    pts = [node_centre(b) for b in bases]
    rng = np.random.default_rng(skel.seed * 2 + 1)

    rx0, ry0, ang = [0.0] * n, [0.0] * n, [0.0] * n
    capped = [b.kind in (BaseKind.MAIN, BaseKind.NATURAL) for b in bases]
    done: set[int] = set()
    for i, b in enumerate(bases):
        if i in done:
            rx0[i], ry0[i] = rx0[mirror[i]], ry0[mirror[i]]
            ang[i] = _mirror_angle(ang[mirror[i]], skel.symmetry)
            continue
        short = max(cfg.min_edge_paint_width, b.width) / 2.0
        if b.kind == BaseKind.JUNCTION:
            short *= cfg.junction_scale
        a_ratio = float(rng.uniform(*cfg.room_aspect))
        rx0[i], ry0[i] = short, short * a_ratio
        ang[i] = float(rng.uniform(0.0, math.pi))
        done.update((i, mirror[i]))
    for i, b in enumerate(bases):
        b.angle = ang[i]

    deg = [0] * n
    max_edge_w = [0.0] * n
    for e in edges:
        pw = painted_width(bases, e, cfg.min_edge_paint_width, cfg.ramp_choke_width)
        for k in (e.a, e.b):
            deg[k] += 1
            max_edge_w[k] = max(max_edge_w[k], pw)
    dead = {i for i, b in enumerate(bases) if b.kind == BaseKind.JUNCTION and deg[i] == 0}

    # cross-level edges are cut as isolated staircases, so only same-level edges paint corridors
    flat = [e for e in edges if bases[e.a].level == bases[e.b].level]
    lanes = [(e, corridor_mask((h, w), pts[e.a], pts[e.b],
                               painted_width(bases, e, cfg.min_edge_paint_width, cfg.ramp_choke_width)))
             for e in flat]
    foreign: list[np.ndarray | None] = []
    for i in range(n):
        m = None
        for e, lm in lanes:
            if i not in (e.a, e.b) and bases[e.a].level != bases[i].level:
                lm = ndimage.binary_dilation(lm)
                m = lm if m is None else (m | lm)
        foreign.append(m)

    def fitted(i: int, mult: float) -> tuple[float, float]:
        g = cfg.anchor_growth if capped[i] else mult
        rx, ry = rx0[i] * g, ry0[i] * g
        if capped[i]:
            rx, ry = min(rx, cfg.box_cap), min(ry, cfg.box_cap)
        fx = max(rx0[i], max_edge_w[i] / 2.0)
        fy = max(ry0[i], max_edge_w[i] / 2.0)
        rx, ry = max(rx, fx), max(ry, fy)
        if foreign[i] is None:
            return rx, ry
        s = 1.0
        while (box_mask((h, w), pts[i], max(rx * s, fx), max(ry * s, fy), ang[i])
               & foreign[i]).any() and (rx * s > fx or ry * s > fy):
            s *= 0.9
        return max(rx * s, fx), max(ry * s, fy)

    # a natural's size never depends on the growth multiplier; once known, every other room is
    # kept off its choke, and off its pocket if it shares the natural's level
    sizes0 = [fitted(i, 1.0) if bases[i].kind == BaseKind.NATURAL else None for i in range(n)]
    for nat, _, pocket, choke in natural_zones(skel, cfg, sizes0):
        for i in range(n):
            if i == nat:
                continue
            zone = choke | pocket if bases[i].level == bases[nat].level else choke
            foreign[i] = zone if foreign[i] is None else (foreign[i] | zone)

    canvas = np.zeros((h, w), dtype=bool)
    bd = cfg.canvas_border
    canvas[pa.y + bd : pa.y + pa.height - bd, pa.x + bd : pa.x + pa.width - bd] = True
    base = np.zeros((h, w), dtype=bool)
    for e, lm in lanes:
        base |= lm
    for i, b in enumerate(bases):
        if b.kind in _REAL:
            base |= node_footprint((h, w), b, 0.0, 0.0, cfg.pad_half)

    def layout(mult: float):
        sizes = [fitted(i, mult) for i in range(n)]
        walk = base.copy()
        for i in range(n):
            if i not in dead:
                walk |= box_mask((h, w), pts[i], sizes[i][0], sizes[i][1], ang[i])
        walk &= canvas
        return sizes, float(walk[pa.y : pa.y + pa.height, pa.x : pa.x + pa.width].mean())

    lo, hi = cfg.growth_range
    sizes, o = layout(hi)
    if o >= cfg.target_openness[0]:
        for _ in range(cfg.growth_iters):
            mid = 0.5 * (lo + hi)
            sizes, o = layout(mid)
            if o < cfg.target_openness[0]:
                lo = mid
            elif o > cfg.target_openness[1]:
                hi = mid
            else:
                break
    for i, b in enumerate(bases):
        b.rx, b.ry = (0.0, 0.0) if i in dead else sizes[i]
