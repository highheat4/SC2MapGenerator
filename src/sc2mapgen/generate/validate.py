"""Milestone 5: the map validator.

A single entry point, :func:`validate_map`, runs every hard-rule and plausibility check the
design spec (section 9) requires on a generated ``MapIR`` and returns a structured
``ValidationReport``. Hard failures mean the map is invalid and must be rejected/regenerated;
warnings mean the map is valid but unusual (outside typical competitive ranges) -- the
per-match loop may keep or down-rank it.

Checks
------
Hard (``report.failures``):
  * exactly two MAIN and two NATURAL bases;
  * a ground path connects the two mains (and every real base is reachable);
  * every real base has a buildable pad big enough for a townhall (reuses
    :func:`rasterize.validate_base_pads`);
  * every ramp is a clean single ascending strip (reuses :func:`rasterize.validate_ramps`);
  * resources are legal: each base has the expected minerals + geysers, all on the base's
    own floor, inside the playable area, not on a ramp, and non-overlapping.

Soft (``report.warnings``): rush distance, minimum critical-path width, terrain symmetry,
base spacing, and openness outside their broad V1 bands.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from sc2mapgen.generate.rasterize import (
    cardinal_ramp_bridge, engine_cliff_grid, engine_components, geyser_cells, mineral_cells,
    townhall_cells, validate_base_pads, validate_ramps,
)
from sc2mapgen.ir import BaseKind, MapIR, ResourceKind

_REAL_BASES = {BaseKind.MAIN, BaseKind.NATURAL, BaseKind.BASE}


@dataclass
class ValidateConfig:
    min_spacing: float = 12.0                 # min base center-to-center distance (tiles)
    min_path_width: float = 3.0               # min clearance along the main<->main path
    rush_band: tuple[float, float] = (45.0, 240.0)   # main<->main ground distance (tiles)
    symmetry_min: float = 0.95                # min walkable-grid symmetry agreement
    openness_band: tuple[float, float] = (0.50, 0.62)
    expect_minerals: int = 8
    expect_geysers: int = 2
    base_min_clear: float = 4.0


@dataclass
class ValidationReport:
    ok: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        tag = "VALID" if self.ok else "INVALID"
        return (f"{tag}: {len(self.failures)} failure(s), {len(self.warnings)} warning(s)")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cell(b) -> tuple[int, int]:
    return int(round(b.x)), int(round(b.y))


def _bfs(walk: np.ndarray, src: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """4-connected BFS over walkable cells from ``src`` (x, y). Returns (dist, prev) where
    dist is step count (-1 = unreachable) and prev encodes the predecessor as y*W+x."""
    h, w = walk.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    prev = np.full((h, w), -1, dtype=np.int64)
    sx, sy = src
    if not (0 <= sx < w and 0 <= sy < h and walk[sy, sx]):
        return dist, prev
    dist[sy, sx] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        d = dist[y, x]
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and walk[ny, nx] and dist[ny, nx] == -1:
                dist[ny, nx] = d + 1
                prev[ny, nx] = y * w + x
                q.append((nx, ny))
    return dist, prev


def _sym_score(a: np.ndarray) -> tuple[str, float]:
    """Best agreement fraction of ``a`` with its rot180 / lr / ud reflection.

    The generator mirrors about the continuous playable centre, so on an even-sized grid the
    discrete reflection is offset by half a cell; we therefore compare against both the plain
    reflection and a one-cell-shifted reflection and take the better alignment, so a truly
    symmetric map scores ~1.0 rather than ~0.92."""
    def agree(b: np.ndarray) -> float:
        best = float((a == b).mean())
        for sy in (0, 1):
            for sx in (0, 1):
                if sy == 0 and sx == 0:
                    continue
                best = max(best, float((a == np.roll(np.roll(b, sy, 0), sx, 1)).mean()))
        return best

    cands = {
        "rot180": a[::-1, ::-1],
        "mirror_lr": a[:, ::-1],
        "mirror_ud": a[::-1, :],
    }
    best, score = "rot180", -1.0
    for name, b in cands.items():
        s = agree(b)
        if s > score:
            best, score = name, s
    return best, score


def _bottleneck_width(walk: np.ndarray, src: tuple[int, int], dst: tuple[int, int]) -> float:
    """Widest-path bottleneck between src and dst: the maximum over all ground paths of the
    minimum corridor width along the path (~2x the min wall-clearance). Unlike the shortest
    path (which hugs walls and always reports ~2), this reflects the true rush-lane choke."""
    import heapq

    h, w = walk.shape
    edt = ndimage.distance_transform_edt(walk)
    sx, sy = src
    dx, dy = dst
    if not (walk[sy, sx] and walk[dy, dx]):
        return 0.0
    best = np.full((h, w), -1.0)
    best[sy, sx] = float(edt[sy, sx])
    pq = [(-best[sy, sx], sx, sy)]
    while pq:
        negcap, x, y = heapq.heappop(pq)
        cap = -negcap
        if cap < best[y, x]:
            continue
        if (x, y) == (dx, dy):
            return 2.0 * cap
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and walk[ny, nx]:
                ncap = min(cap, float(edt[ny, nx]))
                if ncap > best[ny, nx]:
                    best[ny, nx] = ncap
                    heapq.heappush(pq, (-ncap, nx, ny))
    return 2.0 * float(best[dy, dx]) if best[dy, dx] >= 0 else 0.0


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def validate_map(mapir: MapIR, cfg: ValidateConfig | None = None) -> ValidationReport:
    cfg = cfg or ValidateConfig()
    fails: list[str] = []
    warns: list[str] = []
    metrics: dict = {}

    walk = mapir.walkable
    elev = mapir.elevation.astype(int)
    pa = mapir.playable_area
    real = [b for b in mapir.bases if b.kind in _REAL_BASES]
    mains = [b for b in mapir.bases if b.kind == BaseKind.MAIN]
    nats = [b for b in mapir.bases if b.kind == BaseKind.NATURAL]

    # --- base counts -------------------------------------------------------
    if len(mains) != 2:
        fails.append(f"expected 2 mains, got {len(mains)}")
    if len(nats) != 2:
        fails.append(f"expected 2 naturals, got {len(nats)}")
    metrics["n_bases"] = len(real)

    # --- connectivity: ENGINE-ACCURATE (predicts the exported t3SyncCliffLevel and paths it the
    #     way SC2 does: 4-connected, two adjacent cells traversable iff |Δcliff| <= 8, with ramps
    #     carrying the real 8-step gradient). ndimage.label(walk) is elevation-blind, so it blessed
    #     maps the engine actually sees as several disconnected level-islands -- see findings §4.4.
    ecliff = engine_cliff_grid(walk, elev, mapir.ramps)
    # engine_cliff_grid now mirrors the exporter EXACTLY (including channel_ramp_flanks), so the
    # STRICT |Δcliff|<=8 / 4-connected metric (findings §4.4) is the oracle. The only bridge is for
    # gold-profile CARDINAL ramps (+16/+24 steps by design, findings §4.13). We deliberately do NOT
    # bridge every ramp: it was tried (bridge every same-ramp adjacency, modelling "a quad makes
    # the whole ramp walkable at ANY step") and it FALSE-ACCEPTED -- seed 22 read connected offline
    # but its short/steep diagonal natural->main ramp is a walled dead-end in-engine (user-confirmed
    # in-game + scripts/mainconn_probe.py). The quad-is-sole-controller result only holds for AXIS-
    # uniform staircases, not our rank-order DIAGONAL ramps, so the honest oracle is the strict rule:
    # with adequate run (ramp_run_range >= ~11, a 1-level diagonal climb steps <=8/cell) legit ramps
    # already connect under it. Residual over-reports on quad-uncoverable ramps are backstopped by
    # the validate_ramps hard-gate below and rejection-sampled by --valid-only (§4.4 "Current
    # status"). The real cure is the source-side terrace redesign, not post-hoc oracle tuning.
    ecomp = engine_components(ecliff, bridge=cardinal_ramp_bridge(mapir.ramps, ecliff.shape))

    def _bcomp(b) -> int:
        cx, cy = _cell(b)
        if 0 <= cy < ecomp.shape[0] and 0 <= cx < ecomp.shape[1]:
            return int(ecomp[cy, cx])
        return -1

    comps = set()
    for b in real:
        v = _bcomp(b)
        if v < 0:
            fails.append(f"{b.kind.value} @({b.x:.0f},{b.y:.0f}) is not on traversable ground")
        comps.add(v)
    comps.discard(-1)
    main_comps = {_bcomp(m) for m in mains}
    mains_connected = len(mains) == 2 and len(main_comps) == 1 and -1 not in main_comps
    metrics["mains_connected"] = mains_connected
    metrics["base_components"] = len(comps)
    if not mains_connected:
        fails.append("no ground path between the two mains (engine cliff-step connectivity)")
    elif len(comps) > 1:
        fails.append(f"{len(comps)} disconnected base components (bases unreachable in-engine)")

    # --- rush distance + minimum critical-path width -----------------------
    if mains_connected:
        m0, m1 = _cell(mains[0]), _cell(mains[1])
        # measure on the ACTUAL traversable component (ramps included), not the elevation-blind
        # walkable mask, so the rush lane can't "tunnel" through a cliff.
        trav = ecomp == _bcomp(mains[0])
        dist, _ = _bfs(trav, m0)
        rush = int(dist[m1[1], m1[0]])
        metrics["rush_distance"] = rush
        if not (cfg.rush_band[0] <= rush <= cfg.rush_band[1]):
            warns.append(f"rush distance {rush} outside band {cfg.rush_band}")
        # Measure the contested rush-lane choke between the two NATURALS, not the mains: each
        # main's exit is an intentionally-tight defensive choke and would dominate (and falsely
        # flag) a main<->main measurement. The natural<->natural lane is the open field both
        # players fight over, where a 1-tile pinch really is a defect.
        w0, w1 = (_cell(nats[0]), _cell(nats[1])) if len(nats) == 2 else (m0, m1)
        min_w = _bottleneck_width(trav, w0, w1)
        metrics["min_path_width"] = round(min_w, 1)
        if min_w < cfg.min_path_width:
            warns.append(f"natural<->natural rush-lane choke {min_w:.1f} < {cfg.min_path_width}")

    # --- base spacing ------------------------------------------------------
    min_pair = min(
        (np.hypot(a.x - b.x, a.y - b.y)
         for i, a in enumerate(real) for b in real[i + 1:]),
        default=1e9,
    )
    metrics["min_base_spacing"] = round(float(min_pair), 1)
    if min_pair < cfg.min_spacing:
        warns.append(f"min base spacing {min_pair:.1f} < {cfg.min_spacing}")

    # --- terrain symmetry --------------------------------------------------
    sub = walk[pa.y:pa.y + pa.height, pa.x:pa.x + pa.width]
    sym_name, sym_score = _sym_score(sub)
    metrics["symmetry"] = f"{sym_name}:{sym_score:.3f}"
    if sym_score < cfg.symmetry_min:
        warns.append(f"terrain symmetry {sym_score:.3f} ({sym_name}) < {cfg.symmetry_min}")

    # --- openness ----------------------------------------------------------
    openness = float(sub.mean())
    metrics["openness"] = round(openness, 3)
    if not (cfg.openness_band[0] <= openness <= cfg.openness_band[1]):
        warns.append(f"openness {openness:.3f} outside band {cfg.openness_band}")

    # --- base pads (hard) --------------------------------------------------
    fails.extend(validate_base_pads(mapir, min_clear=cfg.base_min_clear))

    # --- ramps (hard) ------------------------------------------------------
    ramp_fails = validate_ramps(mapir)
    metrics["n_ramps"] = len(mapir.ramps)
    metrics["ramps_clean"] = len(mapir.ramps) - len(ramp_fails)
    fails.extend(ramp_fails)

    # --- resources (hard) --------------------------------------------------
    fails.extend(_check_resources(mapir, elev, walk, cfg))
    metrics["n_resources"] = len(mapir.resources)

    ok = len(fails) == 0
    return ValidationReport(ok=ok, failures=fails, warnings=warns, metrics=metrics)


def _check_resources(mapir: MapIR, elev, walk, cfg: ValidateConfig) -> list[str]:
    fails: list[str] = []
    h, w = elev.shape
    ramp_mask = np.zeros((h, w), dtype=bool)
    for r in mapir.ramps:
        for x, y in r.cells:
            ramp_mask[y, x] = True
    pa = mapir.playable_area
    seen: dict[tuple[int, int], int] = {}

    def in_playable(ix, iy) -> bool:
        return pa.x <= ix < pa.x + pa.width and pa.y <= iy < pa.y + pa.height

    for bi, b in enumerate(mapir.bases):
        if b.kind not in _REAL_BASES:
            continue
        mins = geys = 0
        floor = int(elev[_cell(b)[1], _cell(b)[0]])
        th = townhall_cells(*_cell(b))
        if not all(in_playable(x, y) and mapir.buildable[y, x] and int(elev[y, x]) == floor
                   for x, y in th):
            fails.append(f"{b.kind.value} @({b.x:.0f},{b.y:.0f}) townhall footprint not buildable")
        for ri in b.resource_idx:
            r = mapir.resources[ri]
            if r.kind == ResourceKind.MINERAL:
                mins += 1
                cells = mineral_cells(r.x, r.y)
            elif r.kind == ResourceKind.GEYSER:
                geys += 1
                cells = geyser_cells(r.x, r.y)
            else:
                continue
            for ix, iy in cells:
                if not in_playable(ix, iy):
                    fails.append(f"{b.kind.value} @({b.x:.0f},{b.y:.0f}) resource off playable area")
                elif ramp_mask[iy, ix]:
                    fails.append(f"{b.kind.value} @({b.x:.0f},{b.y:.0f}) resource on a ramp")
                elif int(elev[iy, ix]) != floor:
                    fails.append(
                        f"{b.kind.value} @({b.x:.0f},{b.y:.0f}) resource on floor "
                        f"{int(elev[iy, ix])} != base floor {floor}"
                    )
                key = (ix, iy)
                if key in seen:
                    fails.append(f"resource overlap at ({ix},{iy})")
                seen[key] = ri
        if mins != cfg.expect_minerals:
            fails.append(
                f"{b.kind.value} @({b.x:.0f},{b.y:.0f}) has {mins} minerals "
                f"(want {cfg.expect_minerals})"
            )
        if geys != cfg.expect_geysers:
            fails.append(
                f"{b.kind.value} @({b.x:.0f},{b.y:.0f}) has {geys} geysers "
                f"(want {cfg.expect_geysers})"
            )
    return fails
