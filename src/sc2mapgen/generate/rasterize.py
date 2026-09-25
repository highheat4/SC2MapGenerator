"""Milestone 4: compile the skeleton's plan into a MapIR, and the offline oracles that check it.

The skeleton decides everything: node levels, room boxes, base layouts and every staircase
(``Skeleton.ramps``, from ``ramps.plan_ramps``). ``rasterize`` repaints the exact ground the plan
was decided on (``ramps.paint_ground``), replays each planned staircase pair through the same
``cut_pair`` the planner used, and derives ramps, buildable cells and resources. It repairs
nothing: a plan that doesn't replay raises ``RasterizeError``.

The rest of this module is the engine-faithful connectivity oracle (``engine_cliff_grid`` /
``engine_components``) and the MapIR checks the validator runs.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from sc2mapgen.generate.skeleton import Skeleton
from sc2mapgen.generate.layout import (  # noqa: F401  (re-exported for validate)
    BaseLayout, geyser_cells, mineral_cells, townhall_cells,
)
from sc2mapgen.generate.ramps import (  # noqa: F401  (re-exported for validate / scripts)
    GOLD_MAIN_RAMP_HALF, GOLD_MAIN_RAMP_ISOLINES, GOLD_MAIN_RAMP_RANKS, RAMP_MIN_STRAIGHT_WIDTH,
    PlannedRamp, _DIRS8, cut_pair, pad_edge, paint_ground, _diff_boundary, _higher_boundary,
    _paint_box, _paint_corridor, _snap8, _stamp_gold_main_ramp, is_gold_main_ramp, mirror_dir,
    mirror_point, node_terrace, paint_planned_ramp, passable_terraces, planned_nodes, ramp_axis,
    symmetrize, ramp_placements, ramp_plateau_contacts, ramp_shape_problems, terrace_labels,
    wall_seams,
)
from sc2mapgen.ir import (  # noqa: F401
    BaseKind, BaseNode, MapIR, Ramp, Rect, Resource, ResourceKind,
    _lattice_dir, cardinal_profile, ramp_cell_cliffs, ramp_span, ramp_uphill, snap_uphill,
)

# real resource bases (need a buildable pad); ROOM/JUNCTION are routing-only pseudo-nodes.
_REAL_BASES = {BaseKind.MAIN, BaseKind.NATURAL, BaseKind.BASE}


@dataclass
class RasterConfig:
    # How the ground is painted comes from the skeleton's plan (``Skeleton.ground``); these only
    # shape what is derived from it.
    # a walkable cell is buildable if this far (tiles) from the nearest wall
    build_clearance: float = 2.0
    # assertion: each real base must have at least this buildable clearance (radius, tiles)
    base_min_clear: float = 4.0
    res_mineral_amount: int = 1800
    res_mineral750_amount: int = 900
    res_gas_amount: int = 2250
    weirdness: float = 0.0


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
        _, (ux, uy) = ramp_uphill(r)
        for (x, y), cval in zip(cells, ramp_cell_cliffs(cells, ux, uy, lo_cliff, hi_cliff)):
            cliff[y, x] = cval
    # Mirror the exporter's flank channeling (export.sc2map.channel_ramp_flanks), which voids high-
    # plateau cells beside a ramp's lower sub-levels. The offline oracle MUST apply the same voids or
    # it keeps cells the exporter deletes and over-reports connectivity.
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
        if not is_gold_cardinal(*ramp_uphill(r)[1]):
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
        c_sh = np.roll(np.roll(c, dy, 0), dx, 1)
        diff = np.abs(c - c_sh)
        # GOLD-STANDARD ramp top step: the top sub-level (hi-16 == 112) abuts the high plateau
        # (128) with a +16 axis step (120 is skipped, findings §3.2). That step is walkable because
        # the <rampList> quad covers the high-plateau interface, so join it too. Plateaus are
        # multiples of 64 and sub-levels are not, so a +16 is a top exit only when the HIGHER cell
        # is the plateau; a low-plateau flank 16 below a sub-level (lo -> lo+16) stays a wall.
        hi_c, lo_c = np.maximum(c, c_sh), np.minimum(c, c_sh)
        top_exit = (diff == 2 * max_step) & (hi_c % 64 == 0) & (lo_c % 64 != 0)
        step_ok = (diff <= max_step) | top_exit
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


_STRUCT8 = np.ones((3, 3), dtype=bool)


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
    out = []
    for r in mapir.ramps:
        comp = np.zeros_like(walk)
        for x, y in r.cells:
            comp[y, x] = True
        levels, counts, clean = ramp_plateau_contacts(comp, walk, elev, ramp)
        out.append((r, levels, counts, clean))
    return out


def main_plateaus(walk, elev, ramp, mains) -> list[np.ndarray]:
    """Each MAIN's plateau: the 4-connected same-level ground holding its townhall cell."""
    out = []
    for b in mains:
        x, y = int(round(b.x)), int(round(b.y))
        ground = walk & ~ramp & (elev == elev[y, x])
        lab, _ = ndimage.label(ground)
        out.append(lab == lab[y, x] if lab[y, x] else np.zeros_like(walk))
    return out


def validate_ramps(mapir: MapIR) -> list[str]:
    """Return a list of human-readable failures: any ramp that is not a clean single ascending
    strip (one low plateau <-> one high plateau) with a gold straight-band shape, and any MAIN
    whose plateau has other than one ramp (its single entrance) or whose ramp is not the gold main
    staircase. Empty list == all ramps clean."""
    fails: list[str] = []
    for r, levels, counts, clean in ramp_cleanliness(mapir):
        x, y = r.top
        if not clean:
            fails.append(
                f"ramp @~({x:.0f},{y:.0f}) levels={levels} plateaus/level={counts} "
                f"(want 2 levels 1 apart, 1 plateau each)"
            )
            continue
        for msg in ramp_shape_problems(r.cells, *ramp_uphill(r)[1], r.low_level, r.high_level):
            fails.append(f"ramp @~({x:.0f},{y:.0f}) malformed: {msg}")
    ramp = np.zeros_like(mapir.walkable)
    rid = np.zeros(mapir.walkable.shape, dtype=np.int32)
    for k, r in enumerate(mapir.ramps, start=1):
        for cx, cy in r.cells:
            ramp[cy, cx] = True
            rid[cy, cx] = k
    mains = [b for b in mapir.bases if b.kind == BaseKind.MAIN]
    for b, plat in zip(mains, main_plateaus(mapir.walkable, mapir.elevation, ramp, mains)):
        touching = set(np.unique(rid[ndimage.binary_dilation(plat) & ramp]).tolist()) - {0}
        if len(touching) != 1:
            fails.append(f"main @({b.x:.0f},{b.y:.0f}) plateau has {len(touching)} ramps (want 1)")
            continue
        r = mapir.ramps[touching.pop() - 1]
        if r.low_level is not None and r.high_level is not None and not is_gold_main_ramp(
                r.cells, *ramp_uphill(r)[1], r.low_level, r.high_level):
            fails.append(f"main @({b.x:.0f},{b.y:.0f}) ramp is not the gold main staircase")
    return fails


def base_layout_violations(walk, elev, ramp, buildable, layouts, cfg) -> list[str]:
    """Human-readable breaches of the base-core invariant: every core cell is walkable, off
    ramps and at the base's level; the townhall is buildable; no two cores overlap."""
    fails: list[str] = []
    h, w = walk.shape
    owner: dict[tuple[int, int], int] = {}
    for i, lay in layouts.items():
        bad = 0
        cells = lay.footprint()
        for x, y in cells:
            if not (0 <= x < w and 0 <= y < h) or not walk[y, x] or ramp[y, x] \
                    or int(elev[y, x]) != lay.level:
                bad += 1
                continue
            if owner.setdefault((x, y), i) != i:
                fails.append(f"base @{lay.town} core overlaps base @{layouts[owner[(x, y)]].town}")
                break
        if bad:
            fails.append(f"base @{lay.town} core has {bad} cell(s) off its level-{lay.level} floor")
        if not all(buildable[y, x] for x, y in townhall_cells(*lay.town)):
            fails.append(f"base @{lay.town} townhall not buildable")
    return fails


def _emit_resources(layouts: dict[int, BaseLayout], cfg):
    """``(resources, res_idx)`` for the planned layouts, ``res_idx[skel_index] = [indices]``."""
    resources: list[Resource] = []
    res_idx: dict[int, list[int]] = {}
    for i in sorted(layouts):
        idx = []
        for kind, x, y, is750 in layouts[i].resources():
            if kind == ResourceKind.GEYSER:
                resources.append(Resource(kind, "VespeneGeyser", float(x), float(y),
                                          cfg.res_gas_amount))
            elif is750:
                resources.append(Resource(kind, "MineralField750", float(x), float(y),
                                          cfg.res_mineral750_amount))
            else:
                resources.append(Resource(kind, "MineralField", float(x), float(y),
                                          cfg.res_mineral_amount))
            idx.append(len(resources) - 1)
        res_idx[i] = idx
    return resources, res_idx


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
class RasterizeError(RuntimeError):
    """The skeleton's plan did not compile: a planned staircase refused to replay, or a painted
    ramp came out as something other than one planned staircase."""


def rasterize(skel: Skeleton, seed: int | None = None, cfg: RasterConfig | None = None) -> MapIR:
    """Compile the skeleton's plan: paint the ground it was decided on, replay its staircases in
    cut order, and emit the MapIR. Nothing is repaired here; a plan that doesn't compile raises."""
    cfg = cfg or RasterConfig()
    h, w = skel.grid_h, skel.grid_w
    pa = skel.playable
    bases = skel.bases
    sym = skel.symmetry
    layouts = {i: b.layout for i, b in enumerate(bases) if b.layout is not None}

    if skel.ground is None:
        raise RasterizeError(f"seed {skel.seed}: skeleton has no ramp plan")
    walk, elev, ramp, pad_level, canvas = paint_ground(skel)
    locked = np.zeros_like(walk)
    keep = planned_nodes(skel)
    for k in range(0, len(skel.ramps), 2):
        bad = cut_pair(walk, elev, ramp, locked, pad_level, canvas, skel.ramps[k:k + 2],
                       pa, sym, keep)
        if bad:
            raise RasterizeError(f"seed {skel.seed}: planned staircase {k // 2} "
                                 f"did not replay ({bad[0]})")

    ramps: list[Ramp] = []
    labeled, n = ndimage.label(ramp)
    for lab_id in range(1, n + 1):
        comp = labeled == lab_id
        ax = ramp_axis(comp, walk, elev, ramp)
        if ax is None:
            ys, xs = np.nonzero(comp)
            raise RasterizeError(f"seed {skel.seed}: ramp near ({xs.mean():.0f},{ys.mean():.0f}) "
                                 f"joins no two levels")
        bottom, top, lo_lv, hi_lv = ax
        rys, rxs = np.nonzero(comp)
        near = ndimage.binary_dilation(comp, _STRUCT8, iterations=2)
        planned = [r for r in skel.ramps
                   if 0 <= int(round(r.centre[1])) < h and 0 <= int(round(r.centre[0])) < w
                   and near[int(round(r.centre[1])), int(round(r.centre[0]))]]
        dirs = {snap_uphill(*r.u)[0] for r in planned}
        if len(dirs) != 1:
            raise RasterizeError(f"seed {skel.seed}: ramp near ({rxs.mean():.0f},{rys.mean():.0f}) "
                                 f"matches {len(planned)} planned staircase(s) with dirs {sorted(dirs)}")
        ramps.append(Ramp(cells=[(int(x), int(y)) for x, y in zip(rxs, rys)], bottom=bottom,
                          top=top, low_level=lo_lv, high_level=hi_lv,
                          width=float(planned[0].width), direction=dirs.pop()))

    edt = ndimage.distance_transform_edt(walk)
    buildable = walk & (edt >= cfg.build_clearance) & (~ramp)

    resources, res_idx = _emit_resources(layouts, cfg)
    if os.environ.get("RAST_DEBUG"):
        for msg in base_layout_violations(walk, elev, ramp, buildable, layouts, cfg):
            print(f"[rast] seed={seed}: {msg}")

    # ROOM/JUNCTION nodes are routing-only; real bases sit on their townhall cell
    ir_bases = [
        BaseNode(kind=b.kind, x=float(layouts[i].town[0]), y=float(layouts[i].town[1]),
                 resource_idx=res_idx[i])
        for i, b in enumerate(bases)
        if b.kind in _REAL_BASES
    ]
    starts = [(b.x, b.y) for b in ir_bases if b.kind == BaseKind.MAIN]

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
