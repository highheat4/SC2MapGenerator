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
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

import numpy as np


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
    width: float | None = None  # approx tiles across (size / length)


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
