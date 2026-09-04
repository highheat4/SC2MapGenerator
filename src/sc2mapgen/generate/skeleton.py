"""Milestone 3: symmetric skeleton generator.

Produces a semantic graph of bases (MAIN / NATURAL / generic BASE) plus connections,
under 180-degree rotational symmetry about the playable-area center, sampled from priors.
This is the pre-raster skeleton; Milestone 4 turns it into terrain.

Design choices worth noting:
    * Connectivity uses a Relative Neighborhood Graph (RNG) over the base points. The RNG
      contains the Euclidean MST, so the graph is always connected, and because it is a
      pure function of the (symmetric) point set it is automatically symmetric - no manual
      edge mirroring needed.
    * Hard rules (2 mains, symmetry, min spacing, in-bounds, connectivity, natural gap)
      are enforced by the validator; only parameters are sampled. Invalid candidates are
      rejected and regenerated (spec section 7, step 12).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from sc2mapgen.generate.priors import DefaultPriors, Priors
from sc2mapgen.ir import BaseKind, Rect


@dataclass
class SkelBase:
    kind: BaseKind
    x: float
    y: float
    pair: int  # bases sharing a pair id are 180-degree mirrors (center base pairs w/ self)


@dataclass
class SkelEdge:
    a: int
    b: int
    kind: str  # "natural" | "standard" | "flank"


@dataclass
class Skeleton:
    grid_w: int
    grid_h: int
    playable: Rect
    bases: list[SkelBase]
    edges: list[SkelEdge]
    seed: int
    priors: str = "default"

    # --------------------------------------------------------------------- #
    def center(self) -> tuple[float, float]:
        return (self.playable.x + self.playable.width / 2.0,
                self.playable.y + self.playable.height / 2.0)

    def to_json_dict(self) -> dict:
        return {
            "grid_w": self.grid_w,
            "grid_h": self.grid_h,
            "playable": asdict(self.playable),
            "seed": self.seed,
            "priors": self.priors,
            "bases": [{"kind": b.kind.value, "x": round(b.x, 2), "y": round(b.y, 2),
                       "pair": b.pair} for b in self.bases],
            "edges": [{"a": e.a, "b": e.b, "kind": e.kind} for e in self.edges],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json_dict(), indent=2))
        return p


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _rot180(p: tuple[float, float], c: tuple[float, float]) -> tuple[float, float]:
    return (2 * c[0] - p[0], 2 * c[1] - p[1])


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _rng_graph(pts: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Relative Neighborhood Graph edges. Contains the MST (=> connected); invariant
    under isometries of the point set (=> symmetric for a symmetric point set)."""
    n = len(pts)
    d = [[_dist(pts[i], pts[j]) for j in range(n)] for i in range(n)]
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dij = d[i][j]
            blocked = any(
                k != i and k != j and d[i][k] < dij and d[j][k] < dij for k in range(n)
            )
            if not blocked:
                edges.append((i, j))
    return edges


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
@dataclass
class GenConfig:
    margin: float = 6.0        # keep bases this far inside the playable rect
    center_keepout: float = 6.0  # generic bases must be at least this far from dead center
    p_flank: float = 0.5       # chance to add a symmetric extra "flank" edge (loop/motif)
    max_attempts: int = 300


class SkeletonGenerator:
    def __init__(self, priors: Priors | None = None, config: GenConfig | None = None):
        self.priors = priors or DefaultPriors()
        self.cfg = config or GenConfig()

    # ------------------------------------------------------------------ #
    def generate(self, seed: int) -> Skeleton:
        rng = np.random.default_rng(seed)
        for _ in range(self.cfg.max_attempts):
            skel = self._attempt(rng, seed)
            ok, _issues = validate(skel)
            if ok:
                return skel
        # last attempt returned even if imperfect; surface issues to caller via validate()
        return skel

    # ------------------------------------------------------------------ #
    def _attempt(self, rng: np.random.Generator, seed: int) -> Skeleton:
        dims = self.priors.sample_dims(rng)
        pa = dims.playable
        c = (pa.x + pa.width / 2.0, pa.y + pa.height / 2.0)
        spacing = self.priors.sample_min_spacing(rng)
        total = self.priors.sample_total_bases(rng)

        def in_bounds(p: tuple[float, float]) -> bool:
            m = self.cfg.margin
            return (pa.x + m <= p[0] <= pa.x + pa.width - m
                    and pa.y + m <= p[1] <= pa.y + pa.height - m)

        # 1) P1 main in a corner of the playable rect (canonical: lower-left quadrant)
        fx, fy = self.priors.sample_main_fraction(rng)
        main1 = (pa.x + fx * pa.width, pa.y + fy * pa.height)

        # 2) natural: offset from main toward center by sampled distance/angle
        ndist, nang = self.priors.sample_natural_offset(rng)
        ndist = max(ndist, spacing * 1.05)  # ensure natural respects min spacing
        to_c = math.atan2(c[1] - main1[1], c[0] - main1[0])
        nat1 = (main1[0] + ndist * math.cos(to_c + nang),
                main1[1] + ndist * math.sin(to_c + nang))

        # 3) mirror to P2
        main2 = _rot180(main1, c)
        nat2 = _rot180(nat1, c)

        anchors = [main1, main2, nat1, nat2]
        bases: list[SkelBase] = [
            SkelBase(BaseKind.MAIN, *main1, pair=0),
            SkelBase(BaseKind.MAIN, *main2, pair=0),
            SkelBase(BaseKind.NATURAL, *nat1, pair=1),
            SkelBase(BaseKind.NATURAL, *nat2, pair=1),
        ]

        # 4) generic BASE nodes in symmetric pairs, rejection-sampled for spacing
        n_generic = max(0, total - 4)
        n_pairs = n_generic // 2
        placed: list[tuple[float, float]] = list(anchors)
        pair_id = 2
        tries = 0
        while len([b for b in bases if b.kind == BaseKind.BASE]) < 2 * n_pairs and tries < 4000:
            tries += 1
            p = (rng.uniform(pa.x + self.cfg.margin, pa.x + pa.width - self.cfg.margin),
                 rng.uniform(pa.y + self.cfg.margin, pa.y + pa.height - self.cfg.margin))
            if _dist(p, c) < self.cfg.center_keepout + spacing / 2:
                continue
            q = _rot180(p, c)
            if not (in_bounds(p) and in_bounds(q)):
                continue
            if any(_dist(p, e) < spacing or _dist(q, e) < spacing for e in placed):
                continue
            if _dist(p, q) < spacing:
                continue
            bases.append(SkelBase(BaseKind.BASE, *p, pair=pair_id))
            bases.append(SkelBase(BaseKind.BASE, *q, pair=pair_id))
            placed.extend([p, q])
            pair_id += 1

        # optional self-symmetric center base if an odd generic count was requested
        if n_generic % 2 == 1 and all(_dist(c, e) >= spacing for e in placed):
            bases.append(SkelBase(BaseKind.BASE, c[0], c[1], pair=pair_id))
            placed.append(c)

        # 5) connections: RNG graph over all base points (symmetric + connected)
        pts = [(b.x, b.y) for b in bases]
        rng_edges = _rng_graph(pts)

        # classify edges; tag main<->its-natural as "natural"
        nat_of_main = {0: 2, 1: 3}  # base indices
        edges: list[SkelEdge] = []
        edge_set = set()
        for a, b in rng_edges:
            kind = "standard"
            if (a in nat_of_main and nat_of_main[a] == b) or (
                b in nat_of_main and nat_of_main[b] == a
            ):
                kind = "natural"
            edges.append(SkelEdge(a, b, kind))
            edge_set.add((a, b))

        # ensure each main actually links to its natural (add if RNG missed it)
        for mi, ni in nat_of_main.items():
            if (min(mi, ni), max(mi, ni)) not in edge_set:
                edges.append(SkelEdge(mi, ni, "natural"))
                edge_set.add((min(mi, ni), max(mi, ni)))

        # 6) optional symmetric flank/loop motif
        if rng.random() < self.cfg.p_flank and len(bases) >= 6:
            self._add_flank(rng, bases, edges, edge_set, c)

        priors_name = type(self.priors).__name__
        return Skeleton(dims.grid_w, dims.grid_h, pa, bases, edges, seed, priors_name)

    # ------------------------------------------------------------------ #
    def _add_flank(self, rng, bases, edges, edge_set, c) -> None:
        """Add one extra edge plus its 180-degree mirror, creating an alternate route."""
        n = len(bases)
        pts = [(b.x, b.y) for b in bases]

        def mirror_index(i: int) -> int | None:
            m = _rot180(pts[i], c)
            for j in range(n):
                if _dist(pts[j], m) < 1e-6:
                    return j
            return None

        candidates = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if (i, j) not in edge_set and _dist(pts[i], pts[j]) < 0.45 * max(
                self._span(pts), 1.0
            )
        ]
        rng.shuffle(candidates)
        for i, j in candidates:
            mi, mj = mirror_index(i), mirror_index(j)
            if mi is None or mj is None:
                continue
            e1 = (min(i, j), max(i, j))
            e2 = (min(mi, mj), max(mi, mj))
            if e1 in edge_set or e2 in edge_set:
                continue
            edges.append(SkelEdge(*e1, "flank"))
            edge_set.add(e1)
            if e2 != e1:
                edges.append(SkelEdge(*e2, "flank"))
                edge_set.add(e2)
            return

    @staticmethod
    def _span(pts: list[tuple[float, float]]) -> float:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


# --------------------------------------------------------------------------- #
# validation (hard rules)
# --------------------------------------------------------------------------- #
def _connected(n: int, edges: list[SkelEdge]) -> bool:
    if n == 0:
        return True
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for e in edges:
        adj[e.a].append(e.b)
        adj[e.b].append(e.a)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def validate(skel: Skeleton, min_spacing_tol: float = 0.85) -> tuple[bool, list[str]]:
    issues: list[str] = []
    bases = skel.bases
    c = skel.center()
    pa = skel.playable

    mains = [b for b in bases if b.kind == BaseKind.MAIN]
    nats = [b for b in bases if b.kind == BaseKind.NATURAL]
    if len(mains) != 2:
        issues.append(f"expected 2 mains, got {len(mains)}")
    if len(nats) != 2:
        issues.append(f"expected 2 naturals, got {len(nats)}")

    # in-bounds
    for b in bases:
        if not (pa.x <= b.x <= pa.x + pa.width and pa.y <= b.y <= pa.y + pa.height):
            issues.append(f"base out of playable bounds at ({b.x:.1f},{b.y:.1f})")
            break

    # symmetry: every base has a 180-degree partner
    pts = [(b.x, b.y) for b in bases]
    for i, p in enumerate(pts):
        m = _rot180(p, c)
        if not any(_dist(m, q) < 1.5 for q in pts):
            issues.append(f"base {i} has no symmetric partner")
            break

    # spacing (use the tightest natural gap as reference floor)
    min_pair = min(
        (_dist(pts[i], pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts))),
        default=1e9,
    )
    # infer intended spacing from main<->natural distance
    if mains and nats:
        ref = min(_dist((mb.x, mb.y), (nb.x, nb.y)) for mb in mains for nb in nats)
        if min_pair < min_spacing_tol * min(ref, 16.0):
            issues.append(f"bases too close: min pair distance {min_pair:.1f}")

    # connectivity (and therefore main<->main reachability)
    if not _connected(len(bases), skel.edges):
        issues.append("base graph is not connected")

    # edge symmetry
    def mirror_idx(i):
        m = _rot180(pts[i], c)
        for j in range(len(pts)):
            if _dist(pts[j], m) < 1.5:
                return j
        return None

    eset = {(min(e.a, e.b), max(e.a, e.b)) for e in skel.edges}
    for e in skel.edges:
        ma, mb = mirror_idx(e.a), mirror_idx(e.b)
        if ma is None or mb is None or (min(ma, mb), max(ma, mb)) not in eset:
            issues.append("edge set is not symmetric")
            break

    return (len(issues) == 0, issues)
