"""Render a generated Skeleton to a debug PNG (nodes + connections + symmetry center)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sc2mapgen.generate.skeleton import Skeleton
from sc2mapgen.ir import BaseKind

_EDGE_STYLE = {
    "natural": ("orange", 2.4, "-"),
    "standard": ("#4a90d9", 1.6, "-"),
    "flank": ("#c0392b", 1.6, "--"),
}
_NODE_STYLE = {
    BaseKind.MAIN: ("red", 220),
    BaseKind.NATURAL: ("orange", 150),
    BaseKind.BASE: ("gold", 90),
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
    for e in skel.edges:
        col, lw, ls = _EDGE_STYLE.get(e.kind, ("#4a90d9", 1.6, "-"))
        (x0, y0), (x1, y1) = pts[e.a], pts[e.b]
        ax.plot([x0, x1], [y0, y1], color=col, lw=lw, ls=ls, zorder=1)

    for b in skel.bases:
        col, size = _NODE_STYLE[b.kind]
        ax.scatter([b.x], [b.y], s=size, c=col, edgecolors="black", linewidths=0.7, zorder=2)

    # symmetry center
    ax.scatter([c[0]], [c[1]], marker="+", c="black", s=120, zorder=3)

    n_main = sum(1 for b in skel.bases if b.kind == BaseKind.MAIN)
    n_nat = sum(1 for b in skel.bases if b.kind == BaseKind.NATURAL)
    ax.set_title(
        f"skeleton seed={skel.seed} priors={skel.priors}\n"
        f"{skel.grid_w}x{skel.grid_h}  {len(skel.bases)} bases "
        f"({n_main}M/{n_nat}N)  {len(skel.edges)} edges"
    )
    ax.set_xlim(-2, skel.grid_w + 2)
    ax.set_ylim(-2, skel.grid_h + 2)
    ax.set_aspect("equal")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out
