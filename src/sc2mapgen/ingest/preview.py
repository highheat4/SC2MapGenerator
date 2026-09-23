"""Render a MapIR to a debug PNG.

The visual sanity check is the gate for Milestone 1: if preview.png does not obviously
represent the same strategic geometry as the real map, do not proceed to modeling.

Color scheme:
    black       = unreachable / non-walkable
    grey ramp   = elevation shading on walkable ground (dark = low, light = high)
    cyan cells  = ramps (walkable, non-buildable transitions)
    white dots  = start locations
    red  ring   = MAIN base
    orange ring = NATURAL base
    yellow ring = generic BASE
    light-blue  = mineral fields
    green       = geysers
    magenta     = destructibles
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sc2mapgen.ir import BaseKind, MapIR, ResourceKind


def render(mapir: MapIR, out_path: str | Path) -> Path:
    walkable = mapir.walkable
    elevation = mapir.elevation.astype(float)

    h, w = walkable.shape
    rgb = np.zeros((h, w, 3), dtype=float)

    # elevation shading on walkable ground: normalize to [0.35, 1.0]
    if walkable.any():
        emin = elevation[walkable].min()
        emax = elevation[walkable].max()
        span = max(emax - emin, 1.0)
        shade = 0.35 + 0.65 * (elevation - emin) / span
        for c in range(3):
            rgb[..., c] = np.where(walkable, shade, 0.0)

    # ramps in cyan
    ramp_mask = np.zeros((h, w), dtype=bool)
    for ramp in mapir.ramps:
        for x, y in ramp.cells:
            ramp_mask[y, x] = True
    rgb[ramp_mask] = np.array([0.1, 0.8, 0.9])

    # infer non-base nodes from where ramps lead (exits not pointing at a nearby base)
    from sc2mapgen.ingest.graph import detect_nonbase_nodes

    nonbase_nodes = detect_nonbase_nodes(
        [(b.x, b.y) for b in mapir.bases],
        mapir.ramps,
        walkable,
        ramp_mask,
        main_xy=[(b.x, b.y) for b in mapir.bases if b.kind == BaseKind.MAIN],
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    # origin lower: SC2 (0,0) is bottom-left; array row 0 is y=0
    ax.imshow(rgb, origin="lower", interpolation="nearest")

    # region/connection graph overlay: region centroids + ramp connections as edges
    region_centroid = {rg.id: rg.centroid for rg in mapir.regions}
    for conn in mapir.connections:
        a = region_centroid.get(conn.source_region)
        b = region_centroid.get(conn.target_region)
        if a and b:
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#ff2fd0", lw=1.2, alpha=0.8, zorder=3)
    if mapir.regions:
        rx = [rg.centroid[0] for rg in mapir.regions]
        ry = [rg.centroid[1] for rg in mapir.regions]
        ax.scatter(rx, ry, c="#ff2fd0", s=22, marker="D", edgecolors="black", linewidths=0.4, zorder=4)

    # inferred NON-BASE nodes (from converging ramp exits); size grows with convergence
    if nonbase_nodes:
        nx = [nd["point"][0] for nd in nonbase_nodes]
        ny = [nd["point"][1] for nd in nonbase_nodes]
        ax.scatter(nx, ny, s=90, c="#8a2be2", marker="s", edgecolors="white",
                   linewidths=1.4, zorder=6, label="non-base node")

    # resources
    def scatter(kind, color, marker, size):
        pts = [(r.x, r.y) for r in mapir.resources if r.kind == kind]
        if pts:
            xs, ys = zip(*pts)
            # edgecolors only apply to filled markers; 'x'/'*' are unfilled
            kw = {} if marker in ("x", "*") else {"edgecolors": "black", "linewidths": 0.3}
            ax.scatter(xs, ys, c=color, s=size, marker=marker, **kw)

    scatter(ResourceKind.MINERAL, "#7ec8ff", "s", 10)
    scatter(ResourceKind.GEYSER, "#33cc44", "o", 28)
    scatter(ResourceKind.DESTRUCTIBLE, "#cc33cc", "x", 18)
    scatter(ResourceKind.WATCHTOWER, "#ffffff", "*", 60)

    # bases
    base_style = {
        BaseKind.MAIN: ("red", 260),
        BaseKind.NATURAL: ("orange", 200),
        BaseKind.BASE: ("yellow", 150),
    }
    for b in mapir.bases:
        color, size = base_style[b.kind]
        ax.scatter([b.x], [b.y], s=size, facecolors="none", edgecolors=color, linewidths=2.0)

    # start locations
    if mapir.start_locations:
        sx, sy = zip(*mapir.start_locations)
        ax.scatter(sx, sy, c="white", s=40, marker="P", edgecolors="black", linewidths=0.5)

    ax.set_title(
        f"{mapir.map_name}  ({mapir.width}x{mapir.height}, "
        f"{len(mapir.bases)} bases, {len(mapir.ramps)} ramps, "
        f"{len(nonbase_nodes)} non-base nodes)"
    )
    ax.set_xlim(-1, w)
    ax.set_ylim(-1, h)
    ax.set_aspect("equal")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out
