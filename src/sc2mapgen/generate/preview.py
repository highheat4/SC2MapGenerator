"""Render a generated Skeleton to a debug PNG (nodes + connections + symmetry center)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sc2mapgen.generate.skeleton import Skeleton
from sc2mapgen.ir import BaseKind

_EDGE_STYLE = {
    "natural": ("orange", "-"),
    "out": ("#e67e22", "-"),
    "standard": ("#4a90d9", "-"),
    "junction": ("#16a085", "-"),
}
_NODE_STYLE = {
    BaseKind.MAIN: ("red", 260),
    BaseKind.NATURAL: ("orange", 180),
    BaseKind.BASE: ("gold", 110),
    BaseKind.ROOM: ("#9b59b6", 90),      # non-base interior room/plaza
    BaseKind.JUNCTION: ("#16a085", 45),  # midpoint Y-split pseudo-node
}


def render(skel: Skeleton, out_path: str | Path) -> Path:
    pa = skel.playable
    c = skel.center()
    fig, ax = plt.subplots(figsize=(8, 8))

    # playable rect + grid bounds
    ax.add_patch(plt.Rectangle((pa.x, pa.y), pa.width, pa.height,
                               fill=False, edgecolor="#888", lw=1.0, ls=":"))
    ax.add_patch(plt.Rectangle((0, 0), skel.grid_w, skel.grid_h,
                               fill=False, edgecolor="#444", lw=1.0))

    pts = [(b.x, b.y) for b in skel.bases]
    # edge line width scales with the corridor width so chokes read as thin lines
    for e in skel.edges:
        col, ls = _EDGE_STYLE.get(e.kind, ("#4a90d9", "-"))
        (x0, y0), (x1, y1) = pts[e.a], pts[e.b]
        ax.plot([x0, x1], [y0, y1], color=col, lw=0.6 + 0.5 * e.width, ls=ls, zorder=1)

    # node marker area scales with node width (widest part of its locale)
    for b in skel.bases:
        col, base_size = _NODE_STYLE[b.kind]
        size = base_size if b.kind == BaseKind.JUNCTION else base_size + 8.0 * b.width
        ax.scatter([b.x], [b.y], s=size, c=col, edgecolors="black", linewidths=0.7, zorder=2)

    # symmetry center
    ax.scatter([c[0]], [c[1]], marker="+", c="black", s=120, zorder=3)

    n_main = sum(1 for b in skel.bases if b.kind == BaseKind.MAIN)
    n_nat = sum(1 for b in skel.bases if b.kind == BaseKind.NATURAL)
    n_room = sum(1 for b in skel.bases if b.kind == BaseKind.ROOM)
    n_junc = sum(1 for b in skel.bases if b.kind == BaseKind.JUNCTION)
    ax.set_title(
        f"skeleton seed={skel.seed} priors={skel.priors} sym={skel.symmetry}\n"
        f"{skel.grid_w}x{skel.grid_h}  {n_main}M/{n_nat}N/{n_room}room/{n_junc}junc  "
        f"{len(skel.edges)} edges"
    )
    ax.set_xlim(-2, skel.grid_w + 2)
    ax.set_ylim(-2, skel.grid_h + 2)
    ax.set_aspect("equal")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out
