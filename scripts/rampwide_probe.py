"""Can we make WIDER ramps? Test single-wide-quad vs TWO quads vs a narrow quad over a wide band.

Build a 2-plateau (64->128) AXIS staircase whose transition is a ramp band W rows tall (the rest of
the low/high boundary is a VOID wall, so units MUST cross through the band). Cover it three ways:
  wide1  : ONE quad of the full width W
  split2 : TWO quads of width W/2, side by side (does multi-quad tile a wider ramp?)
  narrow1: ONE width-4 quad (baseline: shows partial coverage of a wide band)
Then measure COVERAGE ACROSS THE WIDTH: spawn a row of Marines at several perpendicular offsets on
the low plateau, order each across, and report how many reach the high plateau. Also report engine
pathing_grid connectivity sampled at each offset.

    SC2PATH="/Applications/StarCraft II" PYTHONPATH=src .venv/bin/python scripts/rampwide_probe.py build wide1 16
    SC2PATH="/Applications/StarCraft II" PYTHONPATH=src .venv/bin/python scripts/rampwide_probe.py probe wide1 16
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.export import sc2map  # noqa: E402
from sc2mapgen.export.stormlib_mpq import export_via_stormlib  # noqa: E402

TEMPLATE = "gold_maps/FrostLE.SC2Map"
STEP = 8  # gentle 8-step gradient so the band is unambiguous; the quad is what we vary


def _name(mode: str, W: int) -> str:
    return f"rwide_{mode}_{W}"


def _ax_quad(base_c, lo, hi, width, run, direction=1):
    """Axis quad, uphill east (+x). width == half-width corner offset (as in make_ramp_entry)."""
    ux, uy = 1.0, 0.0
    rx, ry = 0.0, -1.0
    bx, by = float(base_c[0]), float(base_c[1])
    mx, my = bx - run * ux, by - run * uy
    llx, lly = mx - width * rx, my - width * ry
    rlx, rly = mx + width * rx, my + width * ry
    tf = sc2map._tf
    base = tf((ux, uy), (rx, ry), (bx, by), width, 0.0)
    mid = tf((ux, uy), (rx, ry), (mx, my), width, run)
    cu, cr = (0.0, -1.0), (-1.0, 0.0)
    left_lo = tf(cu, cr, (llx, lly), 2.0, 2.0)
    right_lo = tf(cu, cr, (rlx, rly), 2.0, 2.0)
    left_hi = tf((0, 0), (0, 0), (178.0, 178.0), 0.0, 0.0)
    right_hi = tf((0, 0), (0, 0), (174.0, 174.0), 0.0, 0.0)
    return (f'<ramp dir="{direction}" hi="{hi}" lo="{lo}" '
            f'leftLo="{left_lo}" leftHi="{left_hi}" rightLo="{right_lo}" rightHi="{right_hi}" '
            f'base="{base}" mid="{mid}" cid="0" '
            f'leftLoVar="0" leftHiVar="4294967295" rightLoVar="0" rightHiVar="4294967295"/>')


def _geom(W: int):
    arch = MPQArchive(TEMPLATE)
    version, cw, ch, _ = sc2map.decode_cliff(arch.read_file("t3SyncCliffLevel"))
    mi = arch.read_file("MapInfo")
    poff, _, _ = sc2map._mapinfo_playable_offset(mi)
    pl, pb, pr, pt = struct.unpack_from("<4i", mi, poff)
    x0, x1 = pl + 6, pr - 6
    y0, y1 = pb + 6, pt - 6
    band = list(range(64 + STEP, 128, STEP))           # 7 interior columns
    midx = (x0 + x1) // 2
    ramp_x0 = midx - len(band) // 2
    hi_edge = ramp_x0 + len(band)
    yc = (y0 + y1) // 2
    yb0, yb1 = yc - W // 2, yc + W // 2                 # ramp band rows [yb0, yb1)
    return (arch, version, cw, ch, x0, x1, y0, y1, band, ramp_x0, hi_edge, yc, yb0, yb1)


def build(mode: str, W: int) -> None:
    (arch, version, cw, ch, x0, x1, y0, y1,
     band, ramp_x0, hi_edge, yc, yb0, yb1) = _geom(W)
    tmpl_smap = arch.read_file("t3SyncHeightMap")
    tmpl_hmap = arch.read_file("t3HeightMap")
    palette = sc2map.harvest_palette(arch.read_file("t3SyncCliffLevel"), tmpl_smap, tmpl_hmap)

    cliff = np.zeros((ch, cw), dtype=np.uint16)
    for y in range(y0, y1):
        for x in range(x0, x1):
            if x < ramp_x0:
                cliff[y, x] = 64                        # low plateau (all rows)
            elif x >= hi_edge:
                cliff[y, x] = 128                       # high plateau (all rows)
            elif yb0 <= y < yb1:
                cliff[y, x] = band[x - ramp_x0]         # ramp band (only W rows)
            else:
                cliff[y, x] = 0                          # VOID wall between plateaus

    walkable = cliff > 0
    run = float(len(band) + 1)
    if mode == "wide1":
        entries = [_ax_quad((hi_edge + 0.5, yc + 0.5), 1, 2, width=float(W) / 2.0, run=run)]
    elif mode == "split2":
        q1c = yc - W // 4
        q2c = yc + W // 4
        entries = [_ax_quad((hi_edge + 0.5, q1c + 0.5), 1, 2, width=float(W) / 4.0, run=run),
                   _ax_quad((hi_edge + 0.5, q2c + 0.5), 1, 2, width=float(W) / 4.0, run=run)]
    else:  # narrow1
        entries = [_ax_quad((hi_edge + 0.5, yc + 0.5), 1, 2, width=2.0, run=run)]

    lxc, rxc = x0 + 2, x1 - 2
    objects = (
        '<?xml version="1.0" encoding="utf-8"?>\n<PlacedObjects Version="27">\n'
        f'    <ObjectPoint Id="1001" Position="{lxc}.5,{yc}.5,0" Scale="1,1,1" '
        'Type="StartLoc" Name="Start Location 001" Color="0,0,0,0"/>\n'
        f'    <ObjectPoint Id="1002" Position="{rxc}.5,{yc}.5,0" Scale="1,1,1" '
        'Type="StartLoc" Name="Start Location 002" Color="0,0,0,0"/>\n'
        '</PlacedObjects>\n'
    )
    out = Path(f"outputs/export/{_name(mode, W)}.SC2Map")
    export_via_stormlib(TEMPLATE, out, {
        "t3SyncCliffLevel": sc2map.encode_cliff(cliff, version=version),
        "t3SyncHeightMap": sc2map.build_smap(cliff, tmpl_smap, palette),
        "t3HeightMap": sc2map.build_hmap(cliff, tmpl_hmap, palette),
        "CellAttribute_Pnp": sc2map.author_pnp(arch.read_file("CellAttribute_Pnp"), walkable),
        "t3CellFlags": sc2map.clear_cell_flags(arch.read_file("t3CellFlags")),
        "t3Terrain.xml": sc2map.set_ramp_list(arch.read_file("t3Terrain.xml"), entries),
        "Objects": objects.encode("utf-8"),
    })
    import shutil
    shutil.copy(out, f"/Applications/StarCraft II/maps/{_name(mode, W)}.SC2Map")
    print(f"built {_name(mode, W)}: band rows [{yb0},{yb1}) (W={W}) quads={len(entries)} "
          f"ramp_x0={ramp_x0} hi_edge={hi_edge} yc={yc} run={run}")


def probe(mode: str, W: int) -> None:
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
    (_a, _v, _cw, _ch, x0, x1, y0, y1,
     band, ramp_x0, hi_edge, yc, yb0, yb1) = _geom(W)
    # perpendicular offsets across the band width to sample coverage
    offs = sorted(set([o for o in range(-(W // 2) + 1, W // 2, max(1, W // 6))] + [0]))
    lo_x, hi_x = ramp_x0 - 6, hi_edge + 6

    class P(BotAI):
        def __init__(self):
            super().__init__()
            self._t = 0
            self._spawned = False
            self._targets = {}   # tag -> (offset, hi_target_x)
            self._done = {}

        async def on_start(self):
            gi = self.game_info
            path = gi.pathing_grid.data_numpy
            lbl, _ = ndimage.label(path)

            def comp(x, y):
                if path[y, x]:
                    return int(lbl[y, x])
                return -1
            conn = []
            for o in offs:
                cl, cr = comp(lo_x, yc + o), comp(hi_x, yc + o)
                conn.append(f"off{o:+d}:{'Y' if (cl == cr and cl > 0) else 'n'}")
            print(f"PATHGRID {mode} W={W} ramps_detected={len(gi.map_ramps)} "
                  f"coverage[{' '.join(conn)}]")

        async def on_step(self, it):
            self._t += 1
            if not self._spawned and self._t > 2:
                cmds = [[UnitTypeId.MARINE, 1, Point2((lo_x + 0.5, yc + o + 0.5)), 1] for o in offs]
                await self.client.debug_create_unit(cmds)
                self._spawned = True
                return
            if self._spawned and self._t < 8:
                return
            marines = self.units(UnitTypeId.MARINE)
            for u in marines:
                if u.tag not in self._targets:
                    # assign by nearest offset at spawn
                    o = min(offs, key=lambda o: abs(u.position.y - (yc + o + 0.5)))
                    self._targets[u.tag] = o
                u.move(Point2((hi_x + 0.5, yc + self._targets[u.tag] + 0.5)))
                if u.position.x >= hi_edge + 2:
                    self._done[u.tag] = self._targets[u.tag]
            if self._t > 800 or len(self._done) >= len(offs):
                crossed = sorted(self._done.values())
                print(f"UNITMOVE {mode} W={W} crossed {len(self._done)}/{len(offs)} "
                      f"offsets_that_crossed={crossed}")
                await self.client.leave()

    run_game(maps.get(_name(mode, W)),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    m, w = sys.argv[2], int(sys.argv[3])
    (build if sys.argv[1] == "build" else probe)(m, w)
