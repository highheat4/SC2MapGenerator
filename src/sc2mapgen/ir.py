"""Normalized internal representation (MapIR) for SC2 maps.

Two layers, as agreed in the design:

    SC2RawMap  -> mirrors the game's own representation (engine grids + placed units)
    MapIR      -> our normalized interpretation (walkable/buildable/elevation + bases)

This module defines the *normalized* MapIR plus the intermediate SC2RawMap that the
engine-dump harness produces. The generator (later milestones) only ever depends on
MapIR, so if our interpretation of an engine grid turns out to be wrong we fix the
SC2RawMap -> MapIR step without touching downstream code.

Grid conventions (inherited from the SC2 engine via python-sc2):
    * arrays are indexed [y, x]
    * origin is bottom-left (row 0 == y == 0)
    * one grid cell == one SC2 placement tile (we deliberately do NOT rescale maps)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Ramp gradient geometry (shared single source of truth)
#
# Both the exporter (export.sc2map.author_ramp_cliffs, which writes the real t3SyncCliffLevel) and
# the validator (generate.rasterize.engine_cliff_grid, which PREDICTS that grid to check engine
# traversability) must assign the SAME sub-level rank to each ramp cell, or the validator would
# bless maps the exporter then breaks. Keeping the rank math here guarantees they never drift.
# --------------------------------------------------------------------------- #

# Uphill unit vector per ramp direction: 0-3 CARDINAL (unit-spaced cells), 4-7 DIAGONAL (sqrt2).
# +y is downward (SC2 screen coords). Mirrors export.sc2map._RAMP_DIR_U -- indices follow SC2's OWN
# convention (recovered from all gold maps): 0=N, 1=S, 2=W, 3=E, 4=NW, 5=NE, 6=SW, 7=SE. Used here
# only to snap the uphill vector for the gradient lattice axis (index-agnostic), but kept identical
# to the exporter's table so the emitted <ramp> `dir` label and this gradient never disagree.
RAMP_DIR_U: tuple[tuple[float, float], ...] = (
    (0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0),
    (-0.7071066, -0.7071069), (0.7071069, -0.7071066),
    (-0.7071066, 0.7071069), (0.7071069, 0.7071066),
)


def snap_uphill(ux: float, uy: float) -> tuple[int, tuple[float, float]]:
    """Snap an uphill vector to the nearest of the 8 ramp directions; return (index, unit vector)."""
    n = math.hypot(ux, uy) or 1.0
    ux, uy = ux / n, uy / n
    best, bestdot = 1, -2.0
    for i, (vx, vy) in enumerate(RAMP_DIR_U):
        d = ux * vx + uy * vy
        if d > bestdot:
            best, bestdot = i, d
    return best, RAMP_DIR_U[best]


def ramp_gradient_ranks(cells: list[tuple[int, int]], ux: float, uy: float,
                        span: int) -> list[int]:
    """Sub-level rank in ``[0, span]`` for each ramp cell (aligned to ``cells``), RANK-ORDER style.

    Sort the cells by projection onto the snapped uphill axis ``(ux, uy)`` and spread the sub-levels
    EVENLY over the DISTINCT projection levels: the lowest projection -> rank 0 (low-plateau cliff),
    the highest -> rank ``span`` (high-plateau cliff). This always REACHES both plateaus and never
    SKIPS a sub-level (adjacent cells differ by <=1 rank == <=8 cliff) as long as the ramp has
    ``>= span+1`` distinct projection levels -- see the §4.4 ramp fix in the format-findings doc.
    """
    projs = [x * ux + y * uy for (x, y) in cells]
    uniq = sorted({round(p, 3) for p in projs})
    order = {p: k for k, p in enumerate(uniq)}
    denom = (len(uniq) - 1) or 1
    return [max(0, min(span, math.floor(order[round(p, 3)] * span / denom + 0.5)))
            for p in projs]


def _lattice_dir(ux: float, uy: float) -> tuple[int, int]:
    """Snap an uphill vector to the nearest of the 8 ramp directions and return it as an INTEGER
    lattice step in {-1,0,1}^2. Cardinals -> a unit axis step; diagonals -> a (±1,±1) step. Each
    4-connected axis neighbour then changes the lattice projection ``x*ix + y*iy`` by exactly 1."""
    idx, (vx, vy) = snap_uphill(ux, uy)
    s = math.sqrt(2.0) if idx >= 4 else 1.0
    return int(round(vx * s)), int(round(vy * s))


def isoline_gradient_ranks(cells: list[tuple[int, int]], ux: float, uy: float,
                           span: int) -> list[int]:
    """Sub-level rank in ``[0, span]`` for each ramp cell, TRUE-45°-ISOLINE style (gold-standard).

    Assign the rank as the cell's LATTICE PROJECTION along the snapped uphill axis: ``rank =
    clamp(L - Lmin, 0, span)`` where ``L = x*ix + y*iy`` and ``(ix,iy)`` is the integer lattice step
    (§4.7 invariant #2). Because every 4-connected AXIS neighbour changes ``L`` by exactly 1, the
    cliff steps EXACTLY +8 per axis-cell (walkable) regardless of run; the diagonal neighbour jumps
    +16 but 4-connected pathing ignores it. A full level is therefore always ``span`` axis-cells of
    run -- gold's short FIXED run -- and width (extent along the isolines) is free.

    Robust to LONG bands (cells beyond ``span`` clamp to ``span`` == hi plateau -> a flat high
    shoulder, no wall) but NOT to bands shorter than ``span`` isolines (the top cell then sits below
    the high plateau -> a >8 seam). Verified in-engine walkable at run=8 anti-diagonals across
    widths 4/8/12 by ``scripts/isoline_ramp_probe.py``.
    """
    ix, iy = _lattice_dir(ux, uy)
    ls = [x * ix + y * iy for (x, y) in cells]
    lmin = min(ls) if ls else 0
    return [max(0, min(span, L - lmin)) for L in ls]


def gradient_ranks(cells: list[tuple[int, int]], ux: float, uy: float, span: int) -> list[int]:
    """Dispatch to the true-isoline (DEFAULT) or the legacy rank-order gradient.

    Both the exporter (``author_ramp_cliffs``) and the offline oracle (``engine_cliff_grid``) call
    THIS, so they always agree. ISOLINE is now the shipped default (findings §4.10): it locks the
    cliff step at +8 per lattice axis-cell, so a full level is a FIXED short run (gold-faithful) and
    width is free. Verified beats the old rank-order gradient on STRICT yield (65 vs 59/120 at the
    gold-shorter run (8,10) vs (11,14)) and fixes in-engine dead-ends (seeds 5, 22). Set
    ``RANK_ORDER_RAMPS`` to restore the legacy even-spread gradient (needs the longer >=11 run).
    """
    if os.environ.get("RANK_ORDER_RAMPS"):
        return ramp_gradient_ranks(cells, ux, uy, span)
    return isoline_gradient_ranks(cells, ux, uy, span)


def ramp_span(lo_cliff: int, hi_cliff: int) -> int:
    """Number of +8 axis sub-steps a ramp climbs from its low to high plateau, GOLD-STANDARD.

    Measured on every ramp of FrostLE / AcropolisAIE / AutomatonAIE: a ramp SKIPS the sub-level
    immediately below the high plateau (``hi_cliff - 8`` -- i.e. 120 for a 64->128 climb, 184 for
    128->192) so the TOP axis step is +16, not +8. The staircase is therefore
    ``lo, lo+8, ..., hi-16, hi`` and a one-level climb is ``span == 7`` axis-cells (NOT 8: the old
    full +8 staircase that emitted 120 is not what gold authors). ``ramp_cell_cliffs`` maps the top
    rank straight to ``hi_cliff``; the +16 top step is walkable because the ``<rampList>`` quad
    covers the high-plateau interface (findings §4.7 #3, verified in-engine on gold maps).
    """
    return max(1, (hi_cliff - lo_cliff) // 8 - 1)


def ramp_cell_cliffs(cells: list[tuple[int, int]], ux: float, uy: float,
                     lo_cliff: int, hi_cliff: int) -> list[int]:
    """Per-cell CLIF value for one ramp (aligned to ``cells``): the SINGLE source of truth shared by
    the exporter (``export.sc2map.author_ramp_cliffs``, which writes the real ``t3SyncCliffLevel``)
    and the offline oracle (``generate.rasterize.engine_cliff_grid``, which predicts it) so the two
    can never drift.

    Ranks come from the shared ``gradient_ranks`` (isoline by default). The GOLD-STANDARD mapping
    sends the top rank straight to ``hi_cliff`` and every lower rank to ``lo_cliff + 8*rank`` -- the
    ``lo, lo+8, ..., hi-16, hi`` staircase that SKIPS ``hi-8`` (the +16 top step, see ``ramp_span``).
    """
    if is_gold_cardinal(ux, uy):
        profile = cardinal_profile(lo_cliff, hi_cliff)
        return [profile[min(r, len(profile) - 1)]
                for r in isoline_gradient_ranks(cells, ux, uy, len(profile) - 1)]
    span = ramp_span(lo_cliff, hi_cliff)
    return [hi_cliff if r >= span else lo_cliff + 8 * r
            for r in gradient_ranks(cells, ux, uy, span)]


GOLD_CARDINAL_RAMPS = not os.environ.get("LONG_CARDINAL_RAMPS")


def is_gold_cardinal(ux: float, uy: float) -> bool:
    """True if this uphill vector snaps to a cardinal direction and gold cardinal ramps are on."""
    return GOLD_CARDINAL_RAMPS and snap_uphill(ux, uy)[0] < 4


def cardinal_profile(lo_cliff: int, hi_cliff: int) -> list[int]:
    """Gold CARDINAL staircase, low plateau first: ``lo, lo+8, lo+24, lo+40, hi``.

    Every gold cardinal ramp (71 across the corpus, scripts/_gold_cardinal.py) is a 3-cell slope
    stepping +8/+16/+16 then +24 into the high plateau -- not the 6-cell +8 diagonal staircase. The
    engine's cardinal ramp mesh is sized for that short slope; a longer one pokes out of it as humps.
    The >8 steps are walkable because the quad covers the ramp (findings §4.5 #4).
    """
    return [lo_cliff, lo_cliff + 8, lo_cliff + 24, lo_cliff + 40, hi_cliff]


# --------------------------------------------------------------------------- #
# Small value types
# --------------------------------------------------------------------------- #
class BaseKind(str, Enum):
    """Only MAIN and NATURAL are semantic. Everything else is a generic BASE.

    Whether a BASE is a "third", "safe expansion", etc. must emerge from geometry,
    never from a label (per the design spec).
    """

    MAIN = "MAIN"
    NATURAL = "NATURAL"
    BASE = "BASE"
    # ROOM is a non-base interior node (junction/plaza): no resources, just a region that
    # routes pass through. Used by the generator's skeleton; never emitted by ingestion.
    ROOM = "ROOM"
    # JUNCTION is a midpoint pseudo-node at (A+B)/2 where a third corridor taps the A-B
    # link (a Y-split). Generator-only; never emitted as an IR base/start (like ROOM).
    JUNCTION = "JUNCTION"


class ResourceKind(str, Enum):
    MINERAL = "MINERAL"
    GEYSER = "GEYSER"
    DESTRUCTIBLE = "DESTRUCTIBLE"
    WATCHTOWER = "WATCHTOWER"


@dataclass
class Resource:
    kind: ResourceKind
    unit_type: str  # engine unit type name, e.g. "MineralField750"
    x: float
    y: float
    amount: int | None = None  # mineral/vespene contents if known


@dataclass
class BaseNode:
    kind: BaseKind
    x: float
    y: float
    # tags of resources that belong to this base cluster (indices into MapIR.resources)
    resource_idx: list[int] = field(default_factory=list)


@dataclass
class Ramp:
    """A pathable elevation transition (python-sc2 style: pathable, non-buildable,
    with local height variation - distinguishing it from a flat vision blocker)."""

    cells: list[tuple[int, int]] = field(default_factory=list)
    top: tuple[float, float] | None = None
    bottom: tuple[float, float] | None = None
    low_level: int | None = None
    high_level: int | None = None
    width: float | None = None  # approx tiles across (size / length); the planned width if generated
    # SC2 ``dir`` (index into RAMP_DIR_U) the ramp climbs along. Set from the plan for generated
    # maps; None for ingested ramps, whose direction is snapped from ``bottom -> top``.
    direction: int | None = None


def ramp_uphill(r: Ramp) -> tuple[int, tuple[float, float]]:
    """``(dir, unit uphill vector)`` a ramp is authored along: the planned direction when there is
    one, else the low->high plateau-centroid vector snapped to the 8 ramp directions."""
    if r.direction is not None:
        return r.direction, RAMP_DIR_U[r.direction]
    return snap_uphill(r.top[0] - r.bottom[0], r.top[1] - r.bottom[1])


@dataclass
class Region:
    """A plateau: a connected patch of walkable, non-ramp ground at ~constant elevation.

    Ramps are excluded from region cells and instead become Connections between regions,
    which yields a clean elevation-topology graph.
    """

    id: int
    level: int
    area: int  # cells
    centroid: tuple[float, float]


@dataclass
class Connection:
    """An edge of the semantic graph between two regions.

    V1 connections come from ramps (ground links across an elevation change). Fields
    mirror the design's Connection type; curvature is deferred (paths within a single
    ramp are short/straight) and defaults to 1.0.
    """

    source_region: int
    target_region: int
    kind: str  # "ramp"
    path_length: float
    minimum_width: float
    mean_width: float
    curvature: float
    elevation_changes: int


@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int


# --------------------------------------------------------------------------- #
# SC2RawMap: the engine's own view, dumped verbatim
# --------------------------------------------------------------------------- #
@dataclass
class SC2RawMap:
    """Verbatim engine dump for one map. Arrays are [y, x].

    * pathing:   1 == pathable, 0 == blocked
    * placement: 1 == buildable, 0 == not buildable
    * height:    uint8 0..255 (real height = -16 + v/255*32)
    """

    map_name: str
    width: int
    height: int
    playable_area: Rect
    pathing: np.ndarray
    placement: np.ndarray
    height_map: np.ndarray
    start_locations: list[tuple[float, float]]
    resources: list[Resource]
    expansions: list[tuple[float, float]]  # engine-clustered expansion centers


# --------------------------------------------------------------------------- #
# MapIR: our normalized interpretation
# --------------------------------------------------------------------------- #
@dataclass
class MapIR:
    map_name: str
    width: int
    height: int
    playable_area: Rect

    # Boolean/scalar channels, all [y, x], all cropped/aligned to full grid.
    walkable: np.ndarray  # bool
    buildable: np.ndarray  # bool
    elevation: np.ndarray  # int8 discrete level (0,1,2,...) derived from height

    bases: list[BaseNode]
    resources: list[Resource]
    ramps: list[Ramp]
    regions: list[Region]
    connections: list[Connection]

    start_locations: list[tuple[float, float]]

    # ------------------------------ persistence ---------------------------- #
    def channels(self) -> np.ndarray:
        """Stack the raster channels into a single (C, H, W) array for terrain.npy."""
        return np.stack(
            [
                self.walkable.astype(np.uint8),
                self.buildable.astype(np.uint8),
                self.elevation.astype(np.int16),
            ]
        ).astype(np.int16)

    def to_json_dict(self) -> dict:
        return {
            "map_name": self.map_name,
            "width": self.width,
            "height": self.height,
            "playable_area": asdict(self.playable_area),
            "channels": ["walkable", "buildable", "elevation"],
            "bases": [
                {"kind": b.kind.value, "x": b.x, "y": b.y, "resource_idx": b.resource_idx}
                for b in self.bases
            ],
            "resources": [
                {
                    "kind": r.kind.value,
                    "unit_type": r.unit_type,
                    "x": r.x,
                    "y": r.y,
                    "amount": r.amount,
                }
                for r in self.resources
            ],
            "ramps": [
                {
                    "top": r.top,
                    "bottom": r.bottom,
                    "num_cells": len(r.cells),
                    "low_level": r.low_level,
                    "high_level": r.high_level,
                    "width": r.width,
                    "direction": r.direction,
                }
                for r in self.ramps
            ],
            "regions": [
                {"id": rg.id, "level": rg.level, "area": rg.area, "centroid": rg.centroid}
                for rg in self.regions
            ],
            "connections": [asdict(c) for c in self.connections],
            "start_locations": self.start_locations,
        }

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "terrain.npy", self.channels())
        (out / "map.json").write_text(json.dumps(self.to_json_dict(), indent=2))
