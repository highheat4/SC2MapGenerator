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
    width: float = 8.0  # scalar node width (tiles); every incident edge is <= this
    # for JUNCTION pseudo-nodes: indices of the two real nodes whose midpoint this is
    parents: tuple[int, int] | None = None


@dataclass
class SkelEdge:
    a: int
    b: int
    kind: str  # "natural" | "out" | "standard" | "junction"
    width: float = 4.0  # corridor width (tiles); <= min(width[a], width[b])


@dataclass
class Skeleton:
    grid_w: int
    grid_h: int
    playable: Rect
    bases: list[SkelBase]
    edges: list[SkelEdge]
    seed: int
    priors: str = "default"
    symmetry: str = "rot180"  # one of SYMMETRIES

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
            "symmetry": self.symmetry,
            "bases": [{"kind": b.kind.value, "x": round(b.x, 2), "y": round(b.y, 2),
                       "pair": b.pair, "width": round(b.width, 2),
                       "parents": b.parents} for b in self.bases],
            "edges": [{"a": e.a, "b": e.b, "kind": e.kind,
                       "width": round(e.width, 2)} for e in self.edges],
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


# supported symmetry modes: 180-degree rotational, or lateral (mirror) reflection across
# the vertical (left-right) or horizontal (up-down) center axis.
SYMMETRIES = ("rot180", "mirror_lr", "mirror_ud")


def _mirror(p: tuple[float, float], c: tuple[float, float], mode: str) -> tuple[float, float]:
    if mode == "mirror_lr":
        return (2 * c[0] - p[0], p[1])          # reflect across the vertical axis
    if mode == "mirror_ud":
        return (p[0], 2 * c[1] - p[1])          # reflect across the horizontal axis
    return (2 * c[0] - p[0], 2 * c[1] - p[1])   # rot180 (default)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mirror_indices(pts, c, mode, tol: float = 1.5) -> list[int]:
    """For each point, the index of its symmetric partner (itself if self-symmetric)."""
    n = len(pts)
    out = list(range(n))
    for i in range(n):
        m = _mirror(pts[i], c, mode)
        best, bd = i, 1e9
        for j in range(n):
            d = _dist(pts[j], m)
            if d < bd:
                best, bd = j, d
        out[i] = best if bd < tol else i
    return out


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
    max_attempts: int = 300
    # non-base ROOM nodes scattered in the interior so routes thread
    # base -> room -> room -> base (multiple chokepoints) instead of direct base->base spokes
    room_pairs: tuple[int, int] = (3, 6)   # how many symmetric interior room pairs to add
    room_spacing_frac: float = 0.62        # rooms may pack tighter than bases (x min-spacing)
    p_center_room: float = 0.5             # chance of a self-symmetric central room
    # symmetry modes to draw from per map (uniformly). Default is rotational only; pass
    # e.g. ("rot180", "mirror_lr", "mirror_ud") to also generate lateral-symmetry maps.
    symmetries: tuple[str, ...] = ("rot180",)

    # --- node widths (tiles). A node's scalar width bounds every incident edge; the
    # mirror of a node shares its width. Rooms are the widest (open) nodes. ---
    main_width: tuple[float, float] = (8.0, 12.0)
    natural_width: tuple[float, float] = (7.0, 11.0)
    base_width: tuple[float, float] = (8.0, 13.0)
    room_width: tuple[float, float] = (7.0, 16.0)

    # --- edge widths (tiles) ---
    main_edge_width: float = 3.0             # main's single defensible choke (small, fixed)
    natural_out_width: tuple[float, float] = (4.0, 7.0)  # wider than main edge, capped
    min_edge_width: float = 3.0
    # interior edges sample uniformly in [min_edge_width, min(width[a], width[b])]

    # --- midpoint Y-junctions ---
    p_midpoint: float = 0.35    # chance to convert an eligible interior edge into a Y-split
    max_midpoints: int = 4      # cap on junction motifs (per symmetric map)


class SkeletonGenerator:
    def __init__(self, priors: Priors | None = None, config: GenConfig | None = None):
        self.priors = priors or DefaultPriors()
        self.cfg = config or GenConfig()

    # ------------------------------------------------------------------ #
    def generate(self, seed: int) -> Skeleton:
        rng = np.random.default_rng(seed)
        sym = str(rng.choice(self.cfg.symmetries))  # fixed per map
        for _ in range(self.cfg.max_attempts):
            skel = self._attempt(rng, seed, sym)
            ok, _issues = validate(skel)
            if ok:
                return skel
        # last attempt returned even if imperfect; surface issues to caller via validate()
        return skel

    # ------------------------------------------------------------------ #
    def _attempt(self, rng: np.random.Generator, seed: int, sym: str) -> Skeleton:
        def mirror(p: tuple[float, float]) -> tuple[float, float]:
            return _mirror(p, c, sym)

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

        # 3) mirror to P2 (per the chosen symmetry mode)
        main2 = mirror(main1)
        nat2 = mirror(nat1)

        def sample_w(band: tuple[float, float]) -> float:
            return float(rng.uniform(band[0], band[1]))

        main_w = sample_w(self.cfg.main_width)
        nat_w = sample_w(self.cfg.natural_width)
        anchors = [main1, main2, nat1, nat2]
        bases: list[SkelBase] = [
            SkelBase(BaseKind.MAIN, *main1, pair=0, width=main_w),
            SkelBase(BaseKind.MAIN, *main2, pair=0, width=main_w),
            SkelBase(BaseKind.NATURAL, *nat1, pair=1, width=nat_w),
            SkelBase(BaseKind.NATURAL, *nat2, pair=1, width=nat_w),
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
            q = mirror(p)
            if not (in_bounds(p) and in_bounds(q)):
                continue
            if any(_dist(p, e) < spacing or _dist(q, e) < spacing for e in placed):
                continue
            if _dist(p, q) < spacing:
                continue
            bw = sample_w(self.cfg.base_width)
            bases.append(SkelBase(BaseKind.BASE, *p, pair=pair_id, width=bw))
            bases.append(SkelBase(BaseKind.BASE, *q, pair=pair_id, width=bw))
            placed.extend([p, q])
            pair_id += 1

        # optional self-symmetric center base if an odd generic count was requested
        if n_generic % 2 == 1 and all(_dist(c, e) >= spacing for e in placed):
            bases.append(SkelBase(BaseKind.BASE, c[0], c[1], pair=pair_id,
                                  width=sample_w(self.cfg.base_width)))
            placed.append(c)
            pair_id += 1

        # 4b) non-base ROOM/JUNCTION nodes in the interior (symmetric pairs). These have
        # no resources; they exist to become interior plateaus/junctions that routes pass
        # through, so the connectivity graph reads base->room->room->base.
        mains_xy = [(bases[0].x, bases[0].y), (bases[1].x, bases[1].y)]
        n_room_pairs = int(rng.integers(self.cfg.room_pairs[0], self.cfg.room_pairs[1] + 1))
        room_spacing = spacing * self.cfg.room_spacing_frac
        rooms = 0
        tries = 0
        while rooms < n_room_pairs and tries < 6000:
            tries += 1
            p = (rng.uniform(pa.x + self.cfg.margin, pa.x + pa.width - self.cfg.margin),
                 rng.uniform(pa.y + self.cfg.margin, pa.y + pa.height - self.cfg.margin))
            q = mirror(p)
            if not (in_bounds(p) and in_bounds(q)):
                continue
            # keep rooms out of the main pockets (preserve the main's single entrance)
            if any(_dist(p, m) < spacing or _dist(q, m) < spacing for m in mains_xy):
                continue
            if any(_dist(p, e) < room_spacing or _dist(q, e) < room_spacing for e in placed):
                continue
            if _dist(p, q) < room_spacing:
                continue
            rw = sample_w(self.cfg.room_width)
            bases.append(SkelBase(BaseKind.ROOM, *p, pair=pair_id, width=rw))
            bases.append(SkelBase(BaseKind.ROOM, *q, pair=pair_id, width=rw))
            placed.extend([p, q])
            pair_id += 1
            rooms += 1

        # optional self-symmetric central room (a contested middle plaza/junction)
        if rng.random() < self.cfg.p_center_room and all(_dist(c, e) >= room_spacing for e in placed):
            bases.append(SkelBase(BaseKind.ROOM, c[0], c[1], pair=pair_id,
                                  width=sample_w(self.cfg.room_width)))
            placed.append(c)
            pair_id += 1

        # 5) connections: constructive builder enforcing the degree rules --------
        #   MAIN(deg 1) -> NATURAL(deg 2) -> [single wider out-edge] -> interior net
        #   interior nodes (BASE/ROOM) get degree >= 2; widths obey edge <= min(node).
        edges, edge_set = self._build_edges(rng, bases, c, sym)

        # 6) midpoint Y-junctions: tap some interior edges into a third branch --------
        self._insert_midpoints(rng, bases, edges, edge_set, c, sym)

        priors_name = type(self.priors).__name__
        return Skeleton(dims.grid_w, dims.grid_h, pa, bases, edges, seed, priors_name, sym)

    # ------------------------------------------------------------------ #
    def _build_edges(self, rng, bases, c, sym):
        """Build the symmetric connection graph obeying the per-type degree rules."""
        n = len(bases)
        pts = [(b.x, b.y) for b in bases]
        mirror_of = _mirror_indices(pts, c, sym)

        edges: list[SkelEdge] = []
        edge_set: set[tuple[int, int]] = set()

        def edge_w(a: int, b: int) -> float:
            hi = min(bases[a].width, bases[b].width)
            lo = min(self.cfg.min_edge_width, hi)
            return float(rng.uniform(lo, hi))

        def add_sym(a: int, b: int, kind: str, width: float | None = None) -> None:
            ma, mb = mirror_of[a], mirror_of[b]
            if width is None:
                width = edge_w(a, b)
            for x, y in {(min(a, b), max(a, b)), (min(ma, mb), max(ma, mb))}:
                if x != y and (x, y) not in edge_set:
                    edges.append(SkelEdge(x, y, kind, width))
                    edge_set.add((x, y))

        nat_of_main = {0: 2, 1: 3}
        interior = [i for i in range(n) if bases[i].kind in (BaseKind.BASE, BaseKind.ROOM)]

        # (1) each MAIN -> its NATURAL, the small fixed choke (mirror handled by add_sym)
        add_sym(0, 2, "natural", width=self.cfg.main_edge_width)

        # (2) each NATURAL's single wider outgoing edge -> nearest interior node
        if interior:
            x = min(interior, key=lambda j: _dist(pts[2], pts[j]))
            w_out = float(rng.uniform(*self.cfg.natural_out_width))
            w_out = max(w_out, self.cfg.main_edge_width + 0.5)          # wider than main edge
            w_out = min(w_out, bases[2].width, bases[x].width)          # <= node widths (cap)
            add_sym(2, x, "out", width=w_out)
        else:
            add_sym(2, 3, "out")  # degenerate: no interior -> naturals link directly

        # (3) interior network: RNG (=> connected + symmetric) over interior nodes
        if len(interior) >= 2:
            inter_pts = [pts[i] for i in interior]
            for ia, ib in _rng_graph(inter_pts):
                add_sym(interior[ia], interior[ib], "standard")

        # (4) raise every interior node to degree >= 2 (nearest non-adjacent interior)
        def adjacency() -> dict[int, set[int]]:
            adj: dict[int, set[int]] = {i: set() for i in range(n)}
            for e in edges:
                adj[e.a].add(e.b)
                adj[e.b].add(e.a)
            return adj

        for _ in range(2 * n):
            adj = adjacency()
            low = [i for i in interior if len(adj[i]) < 2]
            if not low:
                break
            i = low[0]
            cand = [j for j in interior if j != i and j not in adj[i]]
            if not cand:
                break
            j = min(cand, key=lambda k: _dist(pts[i], pts[k]))
            add_sym(i, j, "standard")

        return edges, edge_set

    # ------------------------------------------------------------------ #
    def _insert_midpoints(self, rng, bases, edges, edge_set, c, sym) -> None:
        """Convert some interior edges (A,B) into Y-junctions: add M=(A+B)/2 (JUNCTION),
        replace A-B with A-M + B-M, and add a third branch M-X to a nearby interior node.
        Both A and B stay linked through M (each is credited the connection). Mirror-safe."""
        def is_interior(i: int) -> bool:
            return bases[i].kind in (BaseKind.BASE, BaseKind.ROOM)

        # eligible = interior-interior edges (skip anything touching a main/natural/junction)
        eligible = [
            e for e in list(edges)
            if is_interior(e.a) and is_interior(e.b)
        ]
        rng.shuffle(eligible)

        made = 0
        for e in eligible:
            if made >= self.cfg.max_midpoints:
                break
            if rng.random() >= self.cfg.p_midpoint:
                continue
            key = (min(e.a, e.b), max(e.a, e.b))
            if key not in edge_set:
                continue  # already consumed by an earlier motif

            pts = [(b.x, b.y) for b in bases]
            mirror_of = _mirror_indices(pts, c, sym)
            a, b = e.a, e.b
            ma, mb = mirror_of[a], mirror_of[b]
            mkey = (min(ma, mb), max(ma, mb))
            if mkey not in edge_set:
                continue

            # choose the third-branch target: nearest interior node that is not a, b (or
            # their mirrors), so the junction reads as a genuine 3-way split.
            mid = ((bases[a].x + bases[b].x) / 2.0, (bases[a].y + bases[b].y) / 2.0)
            avoid = {a, b, ma, mb}
            cand = [i for i in range(len(bases))
                    if is_interior(i) and i not in avoid]
            if not cand:
                continue
            x = min(cand, key=lambda i: _dist(mid, (bases[i].x, bases[i].y)))

            self_sym = (mkey == key)  # edge lies on the symmetry axis/center (self-mirror)

            def make_junction(a1: int, b1: int, x1: int, pair_id: int) -> None:
                m = ((bases[a1].x + bases[b1].x) / 2.0,
                     (bases[a1].y + bases[b1].y) / 2.0)
                jw = min(bases[a1].width, bases[b1].width)
                jidx = len(bases)
                bases.append(SkelBase(BaseKind.JUNCTION, m[0], m[1], pair=pair_id,
                                      width=jw, parents=(a1, b1)))
                # drop the direct A-B edge, wire A-M, B-M, X-M (all <= junction width)
                k = (min(a1, b1), max(a1, b1))
                if k in edge_set:
                    edge_set.discard(k)
                    edges[:] = [ee for ee in edges
                                if (min(ee.a, ee.b), max(ee.a, ee.b)) != k]
                for u in (a1, b1):
                    w = min(jw, bases[u].width, self._rw(rng, jw))
                    edges.append(SkelEdge(min(u, jidx), max(u, jidx), "junction", w))
                    edge_set.add((min(u, jidx), max(u, jidx)))
                wx = min(jw, bases[x1].width, self._rw(rng, jw))
                edges.append(SkelEdge(min(x1, jidx), max(x1, jidx), "junction", wx))
                edge_set.add((min(x1, jidx), max(x1, jidx)))

            # shared pair id so a junction and its mirror are recognized as a pair
            j_pair = max(bb.pair for bb in bases) + 1
            make_junction(a, b, x, j_pair)
            if not self_sym:
                # mirror the whole motif; recompute mirror target for x
                pts2 = [(bb.x, bb.y) for bb in bases]
                mof = _mirror_indices(pts2, c, sym)
                make_junction(ma, mb, mof[x], j_pair)
            made += 1

    @staticmethod
    def _rw(rng, hi: float) -> float:
        return float(rng.uniform(min(3.0, hi), hi))


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

    # symmetry: every base has a partner under the skeleton's symmetry mode
    pts = [(b.x, b.y) for b in bases]
    for i, p in enumerate(pts):
        m = _mirror(p, c, skel.symmetry)
        if not any(_dist(m, q) < 1.5 for q in pts):
            issues.append(f"base {i} has no symmetric partner")
            break

    # spacing (use the tightest natural gap as reference floor); ROOM/JUNCTION nodes are
    # allowed to pack tighter than resource bases, so they are excluded from this floor.
    base_pts = [(b.x, b.y) for b in bases
                if b.kind not in (BaseKind.ROOM, BaseKind.JUNCTION)]
    min_pair = min(
        (_dist(base_pts[i], base_pts[j])
         for i in range(len(base_pts)) for j in range(i + 1, len(base_pts))),
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

    # degree rules: main == 1 (to its natural), natural == 2 (main + one out),
    # every other real node >= 2. Junctions are pseudo-nodes (not degree-constrained).
    adj: dict[int, set[int]] = {i: set() for i in range(len(bases))}
    for e in skel.edges:
        adj[e.a].add(e.b)
        adj[e.b].add(e.a)
    nat_kinds = {BaseKind.NATURAL}
    for i, b in enumerate(bases):
        deg = len(adj[i])
        if b.kind == BaseKind.MAIN:
            if deg != 1:
                issues.append(f"main {i} degree {deg} != 1")
                break
            if not any(bases[j].kind in nat_kinds for j in adj[i]):
                issues.append(f"main {i} not connected to a natural")
                break
        elif b.kind == BaseKind.NATURAL:
            if deg != 2:
                issues.append(f"natural {i} degree {deg} != 2")
                break
            if not any(bases[j].kind == BaseKind.MAIN for j in adj[i]):
                issues.append(f"natural {i} not connected to a main")
                break
        elif b.kind in (BaseKind.BASE, BaseKind.ROOM):
            if deg < 2:
                issues.append(f"node {i} ({b.kind.value}) degree {deg} < 2")
                break

    # width hierarchy: no edge is wider than either endpoint node
    for e in skel.edges:
        cap = min(bases[e.a].width, bases[e.b].width)
        if e.width > cap + 1e-6:
            issues.append(f"edge {e.a}-{e.b} width {e.width:.1f} > node cap {cap:.1f}")
            break

    # edge symmetry
    def mirror_idx(i):
        m = _mirror(pts[i], c, skel.symmetry)
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
