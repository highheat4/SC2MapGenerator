"""PROTOTYPE: can we author GOLD-SHORT DIAGONAL ramps headless with a locked-8 TRUE-ISOLINE gradient?

Motivation (findings §4.5/§4.7, and the user's question): gold ramps use a FIXED short run and vary
only in WIDTH. Our shipped exporter instead uses a RANK-ORDER gradient (`ir.ramp_gradient_ranks`)
that spreads `span` sub-levels evenly over however many distinct projections the (messy) band has, so
it needs a LONG run (>=11 diagonal) to avoid a >8 (wall) step -- the opposite of gold.

TRUE 45deg isolines fix this: assign ``cliff = lo + 8 * (x+y - a_lo)`` so each anti-diagonal (x+y =
const) is one sub-level and every 4-connected AXIS neighbour steps EXACTLY +8 (walkable) regardless
of run; the diagonal neighbour jumps +16 but 4-connected pathing ignores it (§4.7 invariant #2). A
full level (64) is then always 8 axis-cells / ~5.7 cells along the uphill axis -- gold's short run.

This probe builds a synthetic 2-plateau map whose ONLY low<->high crossing is a code-authored
locked-8 diagonal band (flanks channelled to void), emits the gold-style SMALL quad (make_ramp_entry,
width<=4, run 2) and asks the two engine ground-truths (findings §4.5): the derived
`game_info.pathing_grid` component labels + a REAL Marine ordered across. Width is the free knob; run
is fixed. A `--rankorder` variant authors the SAME band cells with the shipped rank-order gradient
for a direct A/B; `--noquad` is the negative control; `--range N` sweeps the run (N anti-diagonals).

    SC2PATH="/Applications/StarCraft II" PYTHONPATH=src .venv/bin/python scripts/isoline_ramp_probe.py build 8
    SC2PATH="/Applications/StarCraft II" PYTHONPATH=src .venv/bin/python scripts/isoline_ramp_probe.py probe 8
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np
from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.export import sc2map  # noqa: E402
from sc2mapgen.ir import ramp_gradient_ranks  # noqa: E402
from sc2mapgen.export.stormlib_mpq import export_via_stormlib  # noqa: E402

TEMPLATE = "gold_maps/FrostLE.SC2Map"
_SQRT2 = math.sqrt(2.0)


def _flags():
    return ("--rankorder" in sys.argv, "--noquad" in sys.argv,
            int(next((a.split("=")[1] for a in sys.argv if a.startswith("--range=")), 8)))


def _map_name(width: int) -> str:
    rank, noquad, arange = _flags()
    return f"iso_w{width}_r{arange}{'_rank' if rank else ''}{'_q0' if noquad else ''}"


def _geom(width: int):
    """Return (arch, version, cw, ch, region bounds, a_lo, a_hi, b0, centre, lo/hi sample cells)."""
    arch = MPQArchive(TEMPLATE)
    version, cw, ch, _ = sc2map.decode_cliff(arch.read_file("t3SyncCliffLevel"))
    mi = arch.read_file("MapInfo")
    poff, _, _ = sc2map._mapinfo_playable_offset(mi)
    pl, pb, pr, pt = struct.unpack_from("<4i", mi, poff)
    x0, x1 = pl + 6, pr - 6
    y0, y1 = pb + 6, pt - 6
    cx0, cy0 = (x0 + x1) // 2, (y0 + y1) // 2
    _rank, _nq, arange = _flags()
    a_mid = cx0 + cy0
    a_lo, a_hi = a_mid - arange // 2, a_mid + (arange - arange // 2)   # a_hi - a_lo == arange
    b0 = cx0 - cy0                                                     # centreline: x - y == b0
    lo_xy = (cx0 - (arange // 2 + 6), cy0 - (arange // 2 + 6))         # deep on low plateau
    hi_xy = (cx0 + (arange // 2 + 6), cy0 + (arange // 2 + 6))         # deep on high plateau
    return arch, version, cw, ch, x0, x1, y0, y1, a_lo, a_hi, b0, cx0, cy0, lo_xy, hi_xy


def _build_cliff(width: int):
    (arch, version, cw, ch, x0, x1, y0, y1,
     a_lo, a_hi, b0, cx0, cy0, lo_xy, hi_xy) = _geom(width)
    rank, _nq, _ar = _flags()
    cliff = np.zeros((ch, cw), dtype=np.uint16)
    band_cells: list[tuple[int, int]] = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            a = x + y
            if a <= a_lo:
                cliff[y, x] = 64                       # low plateau
            elif a >= a_hi:
                cliff[y, x] = 128                      # high plateau
            else:
                pdist = abs((x - y) - b0) / _SQRT2      # perp distance from centreline
                if pdist <= width / 2.0:
                    band_cells.append((x, y))          # ramp cell (value filled below)
                else:
                    cliff[y, x] = 0                    # channelled flank -> void wall
    if rank:
        # A/B: author the SAME band cells with the SHIPPED rank-order gradient at this (short) run.
        ux, uy = sc2map._RAMP_DIR_U[7]
        span = (128 - 64) // 8
        for (x, y), rk in zip(band_cells, ramp_gradient_ranks(band_cells, ux, uy, span)):
            cliff[y, x] = 64 + 8 * rk
    else:
        # TRUE 45deg isolines: value locked to the x+y anti-diagonal, +8 per isoline.
        for (x, y) in band_cells:
            cliff[y, x] = 64 + 8 * (x + y - a_lo)
    return (arch, version, cw, ch, x0, x1, y0, y1,
            a_lo, a_hi, b0, cx0, cy0, lo_xy, hi_xy, cliff, band_cells)


def build(width: int) -> None:
    (arch, version, cw, ch, x0, x1, y0, y1, a_lo, a_hi, b0, cx0, cy0,
     lo_xy, hi_xy, cliff, band_cells) = _build_cliff(width)
    rank, noquad, arange = _flags()

    # OFFLINE sanity: every 4-connected walkable step must be <=8 (else the engine sees a wall).
    walk = cliff > 0
    bad = 0
    for dy, dx in ((1, 0), (0, 1)):
        a, b = cliff.astype(int), np.roll(cliff.astype(int), (-dy, -dx), (0, 1))
        m = walk & np.roll(walk, (-dy, -dx), (0, 1))
        bad += int((m & (np.abs(a - b) > 8)).sum())
    tmpl_smap = arch.read_file("t3SyncHeightMap")
    tmpl_hmap = arch.read_file("t3HeightMap")
    palette = sc2map.harvest_palette(arch.read_file("t3SyncCliffLevel"), tmpl_smap, tmpl_hmap)

    # gold-style SMALL quad at the band's HIGH end (a == a_hi centre), uphill dir 7 (+x,+y).
    hx = (a_hi + b0) / 2.0
    hy = (a_hi - b0) / 2.0
    entry = sc2map.make_ramp_entry((hx + 0.5, hy + 0.5), direction=7,
                                   width=min(float(width), sc2map.GOLD_QUAD_W), lo=1, hi=2)
    ramp_entries = [] if noquad else [entry]

    lxc, lyc = x0 + 2, y0 + 2
    rxc, ryc = x1 - 2, y1 - 2
    objects = (
        '<?xml version="1.0" encoding="utf-8"?>\n<PlacedObjects Version="27">\n'
        f'    <ObjectPoint Id="1001" Position="{lxc}.5,{lyc}.5,0" Scale="1,1,1" '
        'Type="StartLoc" Name="Start Location 001" Color="0,0,0,0"/>\n'
        f'    <ObjectPoint Id="1002" Position="{rxc}.5,{ryc}.5,0" Scale="1,1,1" '
        'Type="StartLoc" Name="Start Location 002" Color="0,0,0,0"/>\n'
        '</PlacedObjects>\n'
    )
    out = Path(f"outputs/export/{_map_name(width)}.SC2Map")
    out.parent.mkdir(parents=True, exist_ok=True)
    export_via_stormlib(TEMPLATE, out, {
        "t3SyncCliffLevel": sc2map.encode_cliff(cliff, version=version),
        "t3SyncHeightMap": sc2map.build_smap(cliff, tmpl_smap, palette),
        "t3HeightMap": sc2map.build_hmap(cliff, tmpl_hmap, palette),
        "CellAttribute_Pnp": sc2map.author_pnp(arch.read_file("CellAttribute_Pnp"), walk),
        "t3CellFlags": sc2map.clear_cell_flags(arch.read_file("t3CellFlags")),
        "t3Terrain.xml": sc2map.set_ramp_list(arch.read_file("t3Terrain.xml"), ramp_entries),
        "Objects": objects.encode("utf-8"),
    })
    import shutil
    shutil.copy(out, f"/Applications/StarCraft II/maps/{_map_name(width)}.SC2Map")
    vals = sorted({int(cliff[y, x]) for (x, y) in band_cells})
    run_u = arange / _SQRT2
    print(f"built {_map_name(width)}: gradient={'RANK-ORDER' if rank else 'ISOLINE'} "
          f"band_cells={len(band_cells)} width={width} run={arange}anti-diag (~{run_u:.1f} along u) "
          f"band_vals={vals} OFFLINE_bad_steps(>8)={bad} lo={lo_xy} hi={hi_xy} noquad={noquad}")


def probe(width: int) -> None:
    from scipy import ndimage
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.bot_ai_internal import BotAIInternal
    from sc2.data import Difficulty, Race
    from sc2.ids.unit_typeid import UnitTypeId
    from sc2.main import run_game
    from sc2.player import Bot, Computer
    from sc2.position import Point2

    BotAIInternal._find_expansion_locations = lambda self: None
    (arch, version, cw, ch, x0, x1, y0, y1, a_lo, a_hi, b0, cx0, cy0,
     lo_xy, hi_xy) = _geom(width)
    rank, noquad, arange = _flags()
    tag = _map_name(width)
    lo_pt = Point2((lo_xy[0] + 0.5, lo_xy[1] + 0.5))
    hi_pt = Point2((hi_xy[0] + 0.5, hi_xy[1] + 0.5))
    a_hi_reach = a_hi + 2

    class P(BotAI):
        def __init__(self):
            super().__init__()
            self._t = 0
            self._utag = None
            self._maxa = None

        async def on_start(self):
            gi = self.game_info
            path = gi.pathing_grid.data_numpy
            hgt = gi.terrain_height.data_numpy
            lbl, _n = ndimage.label(path)

            def comp(x, y):
                if path[y, x]:
                    return int(lbl[y, x])
                ys, xs = np.where(path > 0)
                i = np.argmin((ys - y) ** 2 + (xs - x) ** 2)
                return int(lbl[ys[i], xs[i]])

            cl, cr = comp(*lo_xy), comp(*hi_xy)
            print(f"PATHGRID {tag} ramps_detected={len(gi.map_ramps)} "
                  f"lo_comp={cl} hi_comp={cr} CONNECTED={cl == cr} "
                  f"h_lo={hgt[lo_xy[1], lo_xy[0]]:.2f} h_hi={hgt[hi_xy[1], hi_xy[0]]:.2f}")
            await self.client.debug_create_unit([[UnitTypeId.MARINE, 1, lo_pt, 1]])

        async def on_step(self, it):
            self._t += 1
            marines = self.units(UnitTypeId.MARINE)
            u = None
            if self._utag is not None:
                u = next((x for x in self.units if x.tag == self._utag), None)
            elif marines:
                u = marines.closest_to(lo_pt)
            if u is None:
                if self._t > 40:
                    print(f"UNITMOVE {tag} RESULT=NO_UNIT")
                    await self.client.leave()
                return
            if self._utag is None:
                self._utag = u.tag
                self._maxa = u.position.x + u.position.y
            u.move(hi_pt)
            self._maxa = max(self._maxa, u.position.x + u.position.y)
            reached = (u.position.x + u.position.y) >= a_hi_reach
            if reached or self._t > 900:
                hgt = self.game_info.terrain_height.data_numpy
                ux2, uy2 = int(u.position.x), int(u.position.y)
                print(f"UNITMOVE {tag} reached_high={reached} "
                      f"max_a={self._maxa:.1f} a_hi={a_hi} "
                      f"final=({u.position.x:.1f},{u.position.y:.1f}) "
                      f"unit_terrain_h={hgt[uy2, ux2]:.2f}")
                await self.client.leave()

    run_game(maps.get(_map_name(width)),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    mode, width = sys.argv[1], int(sys.argv[2])
    (build if mode == "build" else probe)(width)
