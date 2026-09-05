"""Ramp detection + semantic region/connection graph.

Operates purely on the cached engine grids (pathing / placement / height), so it can be
re-run without relaunching SC2 (``ingest_batch --skip-dump``).

Ramp detection mirrors python-sc2's algorithm: a cell that is pathable but not buildable
is a *ramp* only if its 3x3 terrain-height neighbourhood varies; if it is flat it is a
*vision blocker*, not a ramp. This is the fix for the earlier false-positive ramps.

Regions are plateaus: connected components of walkable, non-ramp ground. Because
elevation changes happen on ramps (which we exclude), each component sits at ~constant
height. Ramps then become the Connections between regions -> a clean elevation topology.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage

from sc2mapgen.ir import Connection, Ramp, Region

# 8-connectivity structuring element
_STRUCT8 = np.ones((3, 3), dtype=int)


def _real_to_level(height_u8: np.ndarray, walkable: np.ndarray) -> np.ndarray:
    real = -16.0 + height_u8.astype(np.float64) / 255.0 * 32.0
    level = np.rint(real).astype(np.int16)
    if walkable.any():
        level -= int(level[walkable].min())
    return level


def find_ramp_cells(
    pathing: np.ndarray,
    placement: np.ndarray,
    height_u8: np.ndarray,
    playable: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ramp_mask, vision_blocker_mask), both bool [y, x].

    ramp/vision candidates = pathable & not buildable, inside the playable area.
    A candidate is a ramp if its 3x3 height neighbourhood varies, else a vision blocker.
    """
    h, w = pathing.shape
    candidate = (pathing == 1) & (placement == 0)

    # restrict to playable area
    px, py, pw, ph = playable["x"], playable["y"], playable["width"], playable["height"]
    area = np.zeros_like(candidate)
    area[py : py + ph, px : px + pw] = True
    candidate &= area

    # height variation in a 3x3 window: max - min > 0
    hmax = ndimage.maximum_filter(height_u8, size=3, mode="nearest")
    hmin = ndimage.minimum_filter(height_u8, size=3, mode="nearest")
    varies = (hmax - hmin) > 0

    ramp_mask = candidate & varies
    vision_mask = candidate & ~varies
    return ramp_mask, vision_mask


def _label_components(mask: np.ndarray, min_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Label 8-connected components; drop those smaller than min_size.

    Returns (labels array with 0 = background, list of boolean masks per kept component).
    """
    labels, n = ndimage.label(mask, structure=_STRUCT8)
    kept: list[np.ndarray] = []
    out = np.zeros_like(labels)
    next_id = 0
    for lab in range(1, n + 1):
        comp = labels == lab
        if comp.sum() >= min_size:
            next_id += 1
            out[comp] = next_id
            kept.append(comp)
    return out, kept


def build_ramps(
    ramp_mask: np.ndarray,
    elevation: np.ndarray,
    min_size: int = 8,
) -> tuple[list[Ramp], np.ndarray]:
    """Group ramp cells into Ramp objects. Returns (ramps, ramp_label_array)."""
    labels, comps = _label_components(ramp_mask, min_size)
    ramps: list[Ramp] = []
    for comp in comps:
        ys, xs = np.where(comp)
        cells = list(zip(xs.tolist(), ys.tolist()))
        levels = elevation[ys, xs]
        lo, hi = int(levels.min()), int(levels.max())
        lo_cells = comp & (elevation == lo)
        hi_cells = comp & (elevation == hi)
        lys, lxs = np.where(lo_cells)
        hys, hxs = np.where(hi_cells)
        bottom = (float(lxs.mean()), float(lys.mean())) if len(lxs) else None
        top = (float(hxs.mean()), float(hys.mean())) if len(hxs) else None
        # approx width = size / length, length = |top - bottom|
        length = 1.0
        if top and bottom:
            length = max(1.0, ((top[0] - bottom[0]) ** 2 + (top[1] - bottom[1]) ** 2) ** 0.5)
        ramps.append(
            Ramp(
                cells=cells,
                top=top,
                bottom=bottom,
                low_level=lo,
                high_level=hi,
                width=round(len(cells) / length, 2),
            )
        )
    return ramps, labels


def build_regions(
    walkable: np.ndarray,
    ramp_mask: np.ndarray,
    elevation: np.ndarray,
    min_size: int = 25,
) -> tuple[list[Region], np.ndarray]:
    """Plateaus = connected components of walkable & ~ramp. Returns (regions, label array)."""
    region_cells = walkable & ~ramp_mask
    labels, comps = _label_components(region_cells, min_size)
    regions: list[Region] = []
    for rid, comp in enumerate(comps, start=1):
        ys, xs = np.where(comp)
        levels = elevation[ys, xs]
        # dominant level (mode)
        vals, counts = np.unique(levels, return_counts=True)
        level = int(vals[int(np.argmax(counts))])
        regions.append(
            Region(
                id=rid,
                level=level,
                area=int(comp.sum()),
                centroid=(float(xs.mean()), float(ys.mean())),
            )
        )
    return regions, labels


def clearance(walkable: np.ndarray) -> np.ndarray:
    """Distance transform: tiles to the nearest non-walkable cell. width ~= 2 * clearance."""
    return ndimage.distance_transform_edt(walkable)


def _snap_to_walkable(walkable: np.ndarray, xy: tuple[float, float]) -> tuple[int, int] | None:
    """Nearest walkable cell to a float position (base centers sit on townhall footprints)."""
    h, w = walkable.shape
    x0, y0 = int(round(xy[0])), int(round(xy[1]))
    if 0 <= y0 < h and 0 <= x0 < w and walkable[y0, x0]:
        return x0, y0
    best = None
    best_d = 1e18
    for r in range(1, 8):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = x0 + dx, y0 + dy
                if 0 <= nx < w and 0 <= ny < h and walkable[ny, nx]:
                    d = dx * dx + dy * dy
                    if d < best_d:
                        best_d, best = d, (nx, ny)
        if best is not None:
            return best
    return None


_NEI = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def dijkstra(walkable: np.ndarray, src: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Single-source Dijkstra over walkable cells (8-connectivity, octile cost).

    Returns (dist [y,x] with inf for unreachable, pred flat-index array for path rebuild).
    """
    h, w = walkable.shape
    dist = np.full((h, w), np.inf, dtype=np.float64)
    pred = np.full(h * w, -1, dtype=np.int64)
    sx, sy = src
    dist[sy, sx] = 0.0
    pq: list[tuple[float, int, int]] = [(0.0, sx, sy)]
    while pq:
        d, x, y = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for dx, dy, cost in _NEI:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and walkable[ny, nx]:
                nd = d + cost
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    pred[ny * w + nx] = y * w + x
                    heapq.heappush(pq, (nd, nx, ny))
    return dist, pred


def bottleneck_from(walkable: np.ndarray, edt: np.ndarray, src: tuple[int, int]) -> np.ndarray:
    """Widest-path (maximin) clearance from src to every cell.

    best[cell] = max over paths of (min clearance along the path). The value at a target
    is the clearance of the *narrowest* point on the *widest* corridor connecting them -
    i.e. the true choke you must pass through. width = 2 * best.
    """
    h, w = walkable.shape
    best = np.zeros((h, w), dtype=np.float64)
    sx, sy = src
    best[sy, sx] = edt[sy, sx]
    # max-heap via negative keys
    pq: list[tuple[float, int, int]] = [(-edt[sy, sx], sx, sy)]
    while pq:
        neg_b, x, y = heapq.heappop(pq)
        b = -neg_b
        if b < best[y, x]:
            continue
        for dx, dy, _cost in _NEI:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and walkable[ny, nx]:
                nb = min(b, edt[ny, nx])
                if nb > best[ny, nx]:
                    best[ny, nx] = nb
                    heapq.heappush(pq, (-nb, nx, ny))
    return best


def rebuild_path(pred: np.ndarray, w: int, dst: tuple[int, int]) -> list[tuple[int, int]]:
    dx, dy = dst
    idx = dy * w + dx
    path: list[tuple[int, int]] = []
    while idx != -1:
        path.append((idx % w, idx // w))
        idx = pred[idx]
    path.reverse()
    return path


def path_metrics(
    walkable: np.ndarray,
    edt: np.ndarray,
    a: tuple[float, float],
    b: tuple[float, float],
    dijkstra_cache: dict | None = None,
) -> dict | None:
    """Ground-path metrics between two positions: length, min/mean width, curvature.

    ``min_width`` is the bottleneck (widest-path) choke, ``mean_width`` is the mean
    clearance along the shortest path, both from the distance transform (2 * clearance).
    """
    sa = _snap_to_walkable(walkable, a)
    sb = _snap_to_walkable(walkable, b)
    if sa is None or sb is None:
        return None
    h, w = walkable.shape
    if dijkstra_cache is not None and sa in dijkstra_cache:
        dist, pred = dijkstra_cache[sa]
    else:
        dist, pred = dijkstra(walkable, sa)
        if dijkstra_cache is not None:
            dijkstra_cache[sa] = (dist, pred)
    if not np.isfinite(dist[sb[1], sb[0]]):
        return None
    length = float(dist[sb[1], sb[0]])
    path = rebuild_path(pred, w, sb)
    # mean width along the shortest path (finite cells only; base disks are +inf)
    finite = [2.0 * float(edt[y, x]) for x, y in path if np.isfinite(edt[y, x])]

    # min width = bottleneck (widest-path) clearance, cached per source
    if dijkstra_cache is not None:
        bkey = ("bneck", sa)
        bneck = dijkstra_cache.get(bkey)
        if bneck is None:
            bneck = bottleneck_from(walkable, edt, sa)
            dijkstra_cache[bkey] = bneck
    else:
        bneck = bottleneck_from(walkable, edt, sa)
    bval = bneck[sb[1], sb[0]]
    # fall back to shortest-path finite min if the whole corridor was masked (very close bases)
    min_width = 2.0 * float(bval) if np.isfinite(bval) else (min(finite) if finite else 0.0)

    euclid = math.hypot(sa[0] - sb[0], sa[1] - sb[1]) or 1.0
    return {
        "path_length": round(length, 2),
        "min_width": round(min_width, 2),
        "mean_width": round(float(np.mean(finite)), 2) if finite else 0.0,
        "curvature": round(length / euclid, 3),
    }


def build_connections(
    ramps: list[Ramp],
    region_labels: np.ndarray,
) -> list[Connection]:
    """Each ramp connects the region touching its low side to the region on its high side."""
    h, w = region_labels.shape
    connections: list[Connection] = []
    for ramp in ramps:
        # collect region labels adjacent to this ramp's cells, split by low/high proximity
        low_regions: dict[int, int] = {}
        high_regions: dict[int, int] = {}
        for x, y in ramp.cells:
            cell_level = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        lab = int(region_labels[ny, nx])
                        if lab == 0:
                            continue
                        # assign to low/high bucket by proximity to ramp endpoints
                        if ramp.bottom and ramp.top:
                            d_lo = (nx - ramp.bottom[0]) ** 2 + (ny - ramp.bottom[1]) ** 2
                            d_hi = (nx - ramp.top[0]) ** 2 + (ny - ramp.top[1]) ** 2
                            bucket = low_regions if d_lo <= d_hi else high_regions
                        else:
                            bucket = low_regions
                        bucket[lab] = bucket.get(lab, 0) + 1
        if not low_regions or not high_regions:
            continue
        src = max(low_regions, key=low_regions.get)
        tgt = max(high_regions, key=high_regions.get)
        if src == tgt:
            continue
        connections.append(
            Connection(
                source_region=src,
                target_region=tgt,
                kind="ramp",
                path_length=round(
                    (
                        (ramp.top[0] - ramp.bottom[0]) ** 2
                        + (ramp.top[1] - ramp.bottom[1]) ** 2
                    )
                    ** 0.5,
                    2,
                )
                if (ramp.top and ramp.bottom)
                else 0.0,
                minimum_width=ramp.width or 0.0,
                mean_width=ramp.width or 0.0,
                curvature=1.0,
                elevation_changes=int((ramp.high_level or 0) - (ramp.low_level or 0)),
            )
        )
    return connections
