"""Milestone 2: feature extraction.

Turns a MapIR into a flat vector of ~scalar competitive measurements. These are the
quantities whose empirical distributions the generator will later sample from (Milestone
6) and whose outliers the plausibility filter will reject.

Distance / width features use ground pathfinding on the walkable grid (see graph.py).
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import ndimage

from sc2mapgen.ingest.graph import clearance, path_metrics
from sc2mapgen.ir import BaseKind, MapIR, ResourceKind


def _summ(prefix: str, vals: list[float]) -> dict:
    if not vals:
        return {f"{prefix}_mean": None, f"{prefix}_min": None, f"{prefix}_max": None}
    a = np.array(vals, dtype=float)
    return {
        f"{prefix}_mean": round(float(a.mean()), 2),
        f"{prefix}_min": round(float(a.min()), 2),
        f"{prefix}_max": round(float(a.max()), 2),
    }


def extract_features(m: MapIR) -> dict:
    walkable = m.walkable
    pa = m.playable_area
    playable_cells = max(1, pa.width * pa.height)
    walk_cells = int(walkable.sum())

    f: dict = {"map_name": m.map_name}

    # --- global geometry ---
    f["width"] = m.width
    f["height"] = m.height
    f["aspect"] = round(m.width / m.height, 3)
    f["playable_width"] = pa.width
    f["playable_height"] = pa.height
    f["playable_ratio"] = round(walk_cells / playable_cells, 3)
    f["walkable_cells"] = walk_cells

    # --- bases ---
    mains = [b for b in m.bases if b.kind == BaseKind.MAIN]
    nats = [b for b in m.bases if b.kind == BaseKind.NATURAL]
    generic = [b for b in m.bases if b.kind == BaseKind.BASE]
    f["n_bases"] = len(m.bases)
    f["n_mains"] = len(mains)
    f["n_naturals"] = len(nats)
    f["n_generic_bases"] = len(generic)
    # base density: bases per 10k walkable tiles
    f["base_density"] = round(len(m.bases) / walk_cells * 10000, 3) if walk_cells else None

    # --- resources ---
    minerals = [r for r in m.resources if r.kind == ResourceKind.MINERAL]
    geysers = [r for r in m.resources if r.kind == ResourceKind.GEYSER]
    f["n_minerals"] = len(minerals)
    f["n_geysers"] = len(geysers)
    f["minerals_per_base"] = round(len(minerals) / len(m.bases), 2) if m.bases else None
    f["geysers_per_base"] = round(len(geysers) / len(m.bases), 2) if m.bases else None
    f["n_watchtowers"] = sum(1 for r in m.resources if r.kind == ResourceKind.WATCHTOWER)
    f["n_destructibles"] = sum(1 for r in m.resources if r.kind == ResourceKind.DESTRUCTIBLE)

    # --- elevation ---
    elev = m.elevation
    walk_levels = elev[walkable]
    f["n_elevation_levels"] = int(len(np.unique(walk_levels))) if walk_cells else 0
    f["high_ground_ratio"] = (
        round(float((walk_levels > walk_levels.min()).mean()), 3) if walk_cells else None
    )

    # --- regions ---
    areas = [rg.area for rg in m.regions]
    f["n_regions"] = len(m.regions)
    f.update(_summ("region_area", [float(a) for a in areas]))

    # --- ramps / connections ---
    f["n_ramps"] = len(m.ramps)
    f.update(_summ("ramp_width", [r.width for r in m.ramps if r.width is not None]))
    f["n_connections"] = len(m.connections)
    f.update(_summ("conn_elev_change", [float(c.elevation_changes) for c in m.connections]))

    # --- ground distances & path widths (pathfinding) ---
    edt = clearance(walkable)
    # Neutralize clearance within a disk around every base center: a base's own mineral
    # line / townhall footprint sits at clearance ~1 and would otherwise cap every route's
    # bottleneck at width 2. We want the choke of the corridor *between* bases.
    edt_paths = edt.copy()
    h_, w_ = edt.shape
    R = 6
    yy, xx = np.ogrid[:h_, :w_]
    for b in m.bases:
        disk = (xx - b.x) ** 2 + (yy - b.y) ** 2 <= R * R
        edt_paths[disk] = np.inf
    edt = edt_paths
    cache: dict = {}

    def pos(b):
        return (b.x, b.y)

    # main -> its own-side natural (nearest natural to each main)
    mn_len, mn_minw = [], []
    for mb in mains:
        if not nats:
            break
        nat = min(nats, key=lambda n: (n.x - mb.x) ** 2 + (n.y - mb.y) ** 2)
        pm = path_metrics(walkable, edt, pos(mb), pos(nat), cache)
        if pm:
            mn_len.append(pm["path_length"])
            mn_minw.append(pm["min_width"])
    f.update(_summ("main_to_natural", mn_len))
    f.update(_summ("main_natural_minwidth", mn_minw))

    # main <-> main rush distance (min over pairs = closest-enemy rush; plus mean)
    mm_len, mm_minw = [], []
    for a, b in itertools.combinations(mains, 2):
        pm = path_metrics(walkable, edt, pos(a), pos(b), cache)
        if pm:
            mm_len.append(pm["path_length"])
            mm_minw.append(pm["min_width"])
    f["rush_distance_min"] = round(min(mm_len), 2) if mm_len else None
    f.update(_summ("main_to_main", mm_len))
    f.update(_summ("rush_minwidth", mm_minw))

    # generic base-to-base mean distance (spatial spread of expansions)
    bb_len = []
    all_bases = m.bases
    for a, b in itertools.combinations(all_bases, 2):
        pm = path_metrics(walkable, edt, pos(a), pos(b), cache)
        if pm:
            bb_len.append(pm["path_length"])
    f.update(_summ("base_to_base", bb_len))

    # --- symmetry QC ---
    # Competitive maps use different symmetry types: 180-degree rotational (point) OR
    # reflection (mirror) across a horizontal/vertical axis. We score each transform on
    # the playable sub-grid and keep the best, recording the detected type.
    sub = walkable[pa.y : pa.y + pa.height, pa.x : pa.x + pa.width]
    sym_scores = {
        "rot180": float((sub == np.flip(sub)).mean()),
        "mirror_lr": float((sub == np.flip(sub, axis=1)).mean()),
        "mirror_ud": float((sub == np.flip(sub, axis=0)).mean()),
    }
    # diagonal (reflection across a diagonal) requires a square playable area
    if sub.shape[0] == sub.shape[1]:
        sym_scores["diagonal"] = float((sub == sub.T).mean())
        sym_scores["antidiagonal"] = float((sub == np.flip(np.flip(sub, 0), 1).T).mean())
    best_type = max(sym_scores, key=sym_scores.get)
    f["rot180_symmetry"] = round(sym_scores["rot180"], 3)
    f["symmetry_score"] = round(sym_scores[best_type], 3)
    f["symmetry_type"] = best_type

    return f
