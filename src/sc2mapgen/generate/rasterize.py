"""Milestone 4: geometry rasterizer.

Turns an abstract Skeleton (base nodes + connection edges) into a concrete, playable
raster - the same ``MapIR`` type produced by ingestion - so generated maps flow through
the exact validation/feature/export path as real maps.

Pipeline per map:
    1. sample region + path + elevation geometry PER SYMMETRIC PAIR (so the result stays
       180-degree symmetric by construction),
    2. paint variable-width corridors along edges, then base-region blobs on top,
    3. derive elevation terraces; a corridor between two different levels becomes a ramp,
    4. compute buildability from local clearance (distance transform),
    5. emit a MapIR (walkable / buildable / elevation channels + bases + ramps).

Resources (Milestone 5) and export (Milestone 7) build on top of this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from sc2mapgen.generate.skeleton import Skeleton
from sc2mapgen.ir import BaseKind, BaseNode, MapIR, Ramp, Rect


@dataclass
class RasterConfig:
    # region blob radii (tiles), sampled per pair
    main_radius: tuple[float, float] = (9.0, 12.0)
    natural_radius: tuple[float, float] = (8.0, 11.0)
    generic_radius: tuple[float, float] = (6.0, 10.0)

    # corridor widths (tiles) by edge kind; min_path_width is a HARD floor
    natural_width: tuple[float, float] = (4.0, 6.0)
    standard_width: tuple[float, float] = (7.0, 13.0)
    flank_width: tuple[float, float] = (5.0, 9.0)
    min_path_width: float = 4.0

    # elevation levels
    main_level: int = 2
    natural_level: int = 1
    p_generic_highground: float = 0.25  # chance a generic base sits on high ground

    # a walkable cell is buildable if this far (tiles) from the nearest wall
    build_clearance: float = 2.0

    weirdness: float = 0.0


def _pair_mirror(bases) -> list[int]:
    """mirror[i] = index of the 180-degree partner of base i (i itself if self-paired)."""
    from collections import defaultdict

    groups: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(bases):
        groups[b.pair].append(i)
    mirror = list(range(len(bases)))
    for _pid, idxs in groups.items():
        if len(idxs) == 2:
            a, b = idxs
            mirror[a], mirror[b] = b, a
    return mirror


def _widen(rng, lo, hi, weirdness):
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * (1.0 + weirdness)
    return float(rng.uniform(mid - half, mid + half))


def _paint_capsule(walk, elev, ramp, p0, p1, width, l0, l1):
    """Paint a width-thick segment from p0 to p1. Interpolates elevation; if the two
    endpoints differ in level the whole segment is flagged as a ramp."""
    h, w = walk.shape
    (x0, y0), (x1, y1) = p0, p1
    r = width / 2.0
    xmin = max(0, int(np.floor(min(x0, x1) - r - 1)))
    xmax = min(w - 1, int(np.ceil(max(x0, x1) + r + 1)))
    ymin = max(0, int(np.floor(min(y0, y1) - r - 1)))
    ymax = min(h - 1, int(np.ceil(max(y0, y1) + r + 1)))
    if xmin > xmax or ymin > ymax:
        return
    ys, xs = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    dx, dy = x1 - x0, y1 - y0
    seg_len2 = dx * dx + dy * dy or 1.0
    t = ((xs - x0) * dx + (ys - y0) * dy) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    projx = x0 + t * dx
    projy = y0 + t * dy
    dist = np.hypot(xs - projx, ys - projy)
    mask = dist <= r
    if not mask.any():
        return
    sub_walk = walk[ymin : ymax + 1, xmin : xmax + 1]
    sub_elev = elev[ymin : ymax + 1, xmin : xmax + 1]
    sub_ramp = ramp[ymin : ymax + 1, xmin : xmax + 1]
    sub_walk[mask] = True
    if l0 == l1:
        sub_elev[mask] = l0
    else:
        lvl = np.rint(l0 + (l1 - l0) * t).astype(np.int16)
        sub_elev[mask] = lvl[mask]
        sub_ramp[mask] = True  # corridor across an elevation change == ramp


def _paint_disk(walk, elev, ramp, center, radius, level):
    h, w = walk.shape
    cx, cy = center
    xmin = max(0, int(np.floor(cx - radius - 1)))
    xmax = min(w - 1, int(np.ceil(cx + radius + 1)))
    ymin = max(0, int(np.floor(cy - radius - 1)))
    ymax = min(h - 1, int(np.ceil(cy + radius + 1)))
    ys, xs = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    mask = np.hypot(xs - cx, ys - cy) <= radius
    walk[ymin : ymax + 1, xmin : xmax + 1][mask] = True
    elev[ymin : ymax + 1, xmin : xmax + 1][mask] = level
    ramp[ymin : ymax + 1, xmin : xmax + 1][mask] = False  # plateau, not ramp


def rasterize(skel: Skeleton, seed: int | None = None, cfg: RasterConfig | None = None) -> MapIR:
    cfg = cfg or RasterConfig()
    # offset the seed so raster sampling is independent of the skeleton's own RNG stream
    rng = np.random.default_rng((seed if seed is not None else skel.seed) * 2 + 1)

    h, w = skel.grid_h, skel.grid_w
    walk = np.zeros((h, w), dtype=bool)
    elev = np.zeros((h, w), dtype=np.int16)
    ramp = np.zeros((h, w), dtype=bool)

    bases = skel.bases
    mirror = _pair_mirror(bases)

    # ---- per-pair geometry (identical for a base and its mirror => symmetric) ----
    radius = [0.0] * len(bases)
    level = [0] * len(bases)
    done: set[int] = set()
    for i, b in enumerate(bases):
        if i in done:
            radius[i] = radius[mirror[i]]
            level[i] = level[mirror[i]]
            continue
        if b.kind == BaseKind.MAIN:
            radius[i] = _widen(rng, *cfg.main_radius, cfg.weirdness)
            level[i] = cfg.main_level
        elif b.kind == BaseKind.NATURAL:
            radius[i] = _widen(rng, *cfg.natural_radius, cfg.weirdness)
            level[i] = cfg.natural_level
        else:
            radius[i] = _widen(rng, *cfg.generic_radius, cfg.weirdness)
            level[i] = 1 if rng.random() < cfg.p_generic_highground else 0
        done.add(i)
        done.add(mirror[i])

    # ---- per-edge widths (shared across an edge and its mirror) ----
    def edge_key(a, b):
        ma, mb = mirror[a], mirror[b]
        return tuple(sorted([tuple(sorted((a, b))), tuple(sorted((ma, mb)))]))

    width_by_key: dict = {}
    for e in skel.edges:
        k = edge_key(e.a, e.b)
        if k in width_by_key:
            continue
        if e.kind == "natural":
            wv = _widen(rng, *cfg.natural_width, cfg.weirdness)
        elif e.kind == "flank":
            wv = _widen(rng, *cfg.flank_width, cfg.weirdness)
        else:
            wv = _widen(rng, *cfg.standard_width, cfg.weirdness)
        width_by_key[k] = max(cfg.min_path_width, wv)

    # ---- paint corridors, then base blobs on top ----
    pts = [(b.x, b.y) for b in bases]
    for e in skel.edges:
        wv = width_by_key[edge_key(e.a, e.b)]
        _paint_capsule(walk, elev, ramp, pts[e.a], pts[e.b], wv, level[e.a], level[e.b])
    for i, b in enumerate(bases):
        _paint_disk(walk, elev, ramp, pts[i], radius[i], level[i])

    # ---- buildability from local clearance ----
    edt = ndimage.distance_transform_edt(walk)
    buildable = walk & (edt >= cfg.build_clearance) & (~ramp)

    # ---- ramp objects from the ramp mask ----
    ramps: list[Ramp] = []
    labels, n = ndimage.label(ramp)
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        if len(xs) < 6:
            continue
        cells = [(int(x), int(y)) for x, y in zip(xs, ys)]
        levels = elev[ys, xs]
        lo_i = int(np.argmin(levels))
        hi_i = int(np.argmax(levels))
        ramps.append(
            Ramp(
                cells=cells,
                bottom=(float(xs[lo_i]), float(ys[lo_i])),
                top=(float(xs[hi_i]), float(ys[hi_i])),
                low_level=int(levels.min()),
                high_level=int(levels.max()),
                width=None,
            )
        )

    ir_bases = [BaseNode(kind=b.kind, x=b.x, y=b.y) for b in bases]
    starts = [(b.x, b.y) for b in bases if b.kind == BaseKind.MAIN]
    pa = skel.playable

    return MapIR(
        map_name=f"gen_{skel.seed}",
        width=w,
        height=h,
        playable_area=Rect(pa.x, pa.y, pa.width, pa.height),
        walkable=walk,
        buildable=buildable,
        elevation=elev,
        bases=ir_bases,
        resources=[],
        ramps=ramps,
        regions=[],
        connections=[],
        start_locations=starts,
    )
