"""Normalize a raw engine dump (_raw.npz + _raw.json) into MapIR.

Everything SC2-specific stops here. If our interpretation of an engine grid is wrong,
this is the only file that changes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sc2mapgen.ir import (
    BaseKind,
    BaseNode,
    MapIR,
    Ramp,
    Rect,
    Resource,
    ResourceKind,
)


def _real_height(height_u8: np.ndarray) -> np.ndarray:
    """SC2 encodes terrain height as uint8; convert to real world units."""
    return -16.0 + height_u8.astype(np.float64) / 255.0 * 32.0


def _connected_components(mask: np.ndarray, min_size: int = 10) -> list[list[tuple[int, int]]]:
    """4-connectivity flood-fill labeling over a boolean mask. Returns cell lists (x, y)."""
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or visited[sy, sx]:
                continue
            stack = [(sy, sx)]
            visited[sy, sx] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((x, y))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) >= min_size:
                comps.append(cells)
    return comps


def _nearest(pt: tuple[float, float], candidates: list[tuple[float, float]]) -> int:
    return min(
        range(len(candidates)),
        key=lambda i: (candidates[i][0] - pt[0]) ** 2 + (candidates[i][1] - pt[1]) ** 2,
    )


def build_mapir(dump_dir: str | Path) -> MapIR:
    d = Path(dump_dir)
    raw = np.load(d / "_raw.npz")
    meta = json.loads((d / "_raw.json").read_text())

    pathing = raw["pathing"]      # [y, x], 1 = pathable
    placement = raw["placement"]  # [y, x], 1 = buildable
    height_u8 = raw["height"]     # [y, x], 0..255

    walkable = pathing.astype(bool)
    buildable = placement.astype(bool)

    # Discrete elevation terraces: round real height to integer levels, offset so the
    # lowest *walkable* terrace is level 0. Ramps become intermediate gradients.
    real = _real_height(height_u8)
    level = np.rint(real).astype(np.int16)
    if walkable.any():
        level -= int(level[walkable].min())
    elevation = level

    # --- resources ---
    resources = [
        Resource(
            kind=ResourceKind(r["kind"]),
            unit_type=r["unit_type"],
            x=r["x"],
            y=r["y"],
            amount=r.get("amount"),
        )
        for r in meta["resources"]
    ]

    # --- bases: derive MAIN / NATURAL / BASE from expansions + start locations ---
    expansions: list[tuple[float, float]] = [tuple(e) for e in meta["expansions"]]
    # NOTE: game_info.start_locations excludes the observing bot's own start, so we
    # merge in own_start_location to recover the full set of player starts.
    starts: list[tuple[float, float]] = [tuple(s) for s in meta["start_locations"]]
    own = meta.get("own_start_location")
    if own is not None and tuple(own) not in starts:
        starts.append(tuple(own))

    kinds: list[BaseKind] = [BaseKind.BASE] * len(expansions)
    main_idxs: set[int] = set()
    if expansions:
        # MAIN = expansion nearest each start location
        for s in starts:
            mi = _nearest(s, expansions)
            kinds[mi] = BaseKind.MAIN
            main_idxs.add(mi)
        # NATURAL = for each main, nearest non-main expansion (Euclidean proxy for the
        # spike; ground-path distance is a later refinement)
        for mi in main_idxs:
            others = [i for i in range(len(expansions)) if i not in main_idxs]
            if not others:
                continue
            m = expansions[mi]
            ni = min(others, key=lambda i: (expansions[i][0] - m[0]) ** 2 + (expansions[i][1] - m[1]) ** 2)
            if kinds[ni] == BaseKind.BASE:
                kinds[ni] = BaseKind.NATURAL

    # assign each resource to nearest expansion cluster
    bases: list[BaseNode] = []
    base_of_resource: list[int] = []
    if expansions:
        for ri, r in enumerate(resources):
            base_of_resource.append(_nearest((r.x, r.y), expansions))
    for bi, (ex, ey) in enumerate(expansions):
        res_idx = [ri for ri, b in enumerate(base_of_resource) if b == bi]
        bases.append(BaseNode(kind=kinds[bi], x=ex, y=ey, resource_idx=res_idx))

    # --- ramps: pathable but not buildable, minus resource footprints ---
    ramp_mask = walkable & ~buildable
    # knock out cells right under destructibles/resources to reduce false positives
    for r in resources:
        rx, ry = int(round(r.x)), int(round(r.y))
        y0, y1 = max(0, ry - 2), min(ramp_mask.shape[0], ry + 3)
        x0, x1 = max(0, rx - 2), min(ramp_mask.shape[1], rx + 3)
        ramp_mask[y0:y1, x0:x1] = False

    ramps: list[Ramp] = []
    for cells in _connected_components(ramp_mask, min_size=12):
        ys = [c[1] for c in cells]
        xs = [c[0] for c in cells]
        # split endpoints by elevation extremes within the component
        levels = [int(elevation[y, x]) for x, y in cells]
        lo_i = int(np.argmin(levels))
        hi_i = int(np.argmax(levels))
        ramps.append(
            Ramp(
                cells=cells,
                bottom=(float(cells[lo_i][0]), float(cells[lo_i][1])),
                top=(float(cells[hi_i][0]), float(cells[hi_i][1])),
            )
        )

    pa = meta["playable_area"]
    return MapIR(
        map_name=meta["map_name"],
        width=meta["width"],
        height=meta["height"],
        playable_area=Rect(**pa),
        walkable=walkable,
        buildable=buildable,
        elevation=elevation,
        bases=bases,
        resources=resources,
        ramps=ramps,
        start_locations=[tuple(s) for s in starts],
    )
