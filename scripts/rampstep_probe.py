"""Find the STEEPEST (shortest) walkable ramp headless, using the RIGHT oracles.

Earlier `query_pathing`-only probes were ambiguous: `query_pathing` can reflect authored layers.
This probe instead uses the two engine GROUND-TRUTHS that units actually obey:
  1. `game_info.pathing_grid` (engine-DERIVED at load from terrain height + rampList) -- are the low
     and high plateau sample cells in the SAME connected component?  + `game_info.map_ramps`.
  2. A REAL unit move: spawn a Marine on the low plateau, order it across the ramp, and watch whether
     it actually reaches the high plateau (max-x progress + terrain height climbed).

Build a clean 2-plateau (cliff 64 -> 128) AXIS-ALIGNED staircase whose columns step by S, with a
covering <rampList> quad (known-good axis geometry from axis_ramp_probe). A one-level (64) change
then spans 64/S ramp cells, so the largest S that stays walkable == the shortest ramp we can author.

    SC2PATH="/Applications/StarCraft II" PYTHONPATH=src .venv/bin/python scripts/rampstep_probe.py build 16
    SC2PATH="/Applications/StarCraft II" PYTHONPATH=src .venv/bin/python scripts/rampstep_probe.py probe 16
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


def _map_name(step: int, noquad: bool = False) -> str:
    return f"rstep_{step}{'q0' if noquad else ''}"


def _ax_ramp_entry(base_c, lo, hi, width, direction, run):
    """Axis-aligned <ramp> quad: uphill u=(+1,0) east, right r=(0,-1). (from axis_ramp_probe)."""
    ux, uy = 1.0, 0.0
    rx, ry = 0.0, -1.0
    bx, by = float(base_c[0]), float(base_c[1])
    mx, my = bx - run * ux, by - run * uy
    halfw = width
    llx, lly = mx - halfw * rx, my - halfw * ry
    rlx, rly = mx + halfw * rx, my + halfw * ry
    tf = sc2map._tf
    base = tf((ux, uy), (rx, ry), (bx, by), width, 0.0)
    mid = tf((ux, uy), (rx, ry), (mx, my), width, run)
    cu, cr = (0.0, -1.0), (-1.0, 0.0)
    left_lo = tf(cu, cr, (llx, lly), 2.0, 2.0)
    right_lo = tf(cu, cr, (rlx, rly), 2.0, 2.0)
    left_hi = tf((0, 0), (0, 0), (178.0, 178.0), 0.0, 0.0)
    right_hi = tf((0, 0), (0, 0), (174.0, 174.0), 0.0, 0.0)
    return (f'<ramp dir="{direction}" hi="{hi}" lo="{lo}" '
            f'leftLo="{left_lo}" leftHi="{left_hi}" '
            f'rightLo="{right_lo}" rightHi="{right_hi}" '
            f'base="{base}" mid="{mid}" cid="0" '
            f'leftLoVar="0" leftHiVar="4294967295" '
            f'rightLoVar="0" rightHiVar="4294967295"/>')


def _geom(step: int):
    arch = MPQArchive(TEMPLATE)
    version, cw, ch, _ = sc2map.decode_cliff(arch.read_file("t3SyncCliffLevel"))
    mi = arch.read_file("MapInfo")
    poff, _, _ = sc2map._mapinfo_playable_offset(mi)
    pl, pb, pr, pt = struct.unpack_from("<4i", mi, poff)
    x0, x1 = pl + 6, pr - 6
    y0, y1 = pb + 6, pt - 6
    band = list(range(64 + step, 128, step))       # interior columns
    midx = (x0 + x1) // 2
    ramp_x0 = midx - max(1, len(band)) // 2
    hi_edge = ramp_x0 + len(band)
    ly = (y0 + y1) // 2
    # sample points close to the ramp so a unit only has to cross the ramp itself
    lo_xy = (ramp_x0 - 8, ly)
    hi_xy = (hi_edge + 8, ly)
    return arch, version, cw, ch, x0, x1, y0, y1, band, ramp_x0, hi_edge, ly, lo_xy, hi_xy


def build(step: int) -> None:
    (arch, version, cw, ch, x0, x1, y0, y1,
     band, ramp_x0, hi_edge, ly, lo_xy, hi_xy) = _geom(step)
    tmpl_smap = arch.read_file("t3SyncHeightMap")
    tmpl_hmap = arch.read_file("t3HeightMap")
    palette = sc2map.harvest_palette(arch.read_file("t3SyncCliffLevel"), tmpl_smap, tmpl_hmap)

    cliff = np.zeros((ch, cw), dtype=np.uint16)
    for y in range(y0, y1):
        for x in range(x0, x1):
            if x < ramp_x0:
                cliff[y, x] = 64
            elif x >= hi_edge:
                cliff[y, x] = 128
            else:
                cliff[y, x] = band[x - ramp_x0]

    noquad = "--noquad" in sys.argv
    walkable = np.zeros((ch, cw), dtype=bool)
    walkable[y0:y1, x0:x1] = True
    run = float(max(2, len(band) + 1))
    entry = _ax_ramp_entry((hi_edge + 0.5, ly + 0.5), lo=1, hi=2, width=4.0, direction=1, run=run)
    ramp_entries = [] if noquad else [entry]

    lxc, rxc = x0 + 2, x1 - 2
    objects = (
        '<?xml version="1.0" encoding="utf-8"?>\n<PlacedObjects Version="27">\n'
        f'    <ObjectPoint Id="1001" Position="{lxc}.5,{ly}.5,0" Scale="1,1,1" '
        'Type="StartLoc" Name="Start Location 001" Color="0,0,0,0"/>\n'
        f'    <ObjectPoint Id="1002" Position="{rxc}.5,{ly}.5,0" Scale="1,1,1" '
        'Type="StartLoc" Name="Start Location 002" Color="0,0,0,0"/>\n'
        '</PlacedObjects>\n'
    )
    out = Path(f"outputs/export/{_map_name(step, noquad)}.SC2Map")
    export_via_stormlib(TEMPLATE, out, {
        "t3SyncCliffLevel": sc2map.encode_cliff(cliff, version=version),
        "t3SyncHeightMap": sc2map.build_smap(cliff, tmpl_smap, palette),
        "t3HeightMap": sc2map.build_hmap(cliff, tmpl_hmap, palette),
        "CellAttribute_Pnp": sc2map.author_pnp(arch.read_file("CellAttribute_Pnp"), walkable),
        "t3CellFlags": sc2map.clear_cell_flags(arch.read_file("t3CellFlags")),
        "t3Terrain.xml": sc2map.set_ramp_list(arch.read_file("t3Terrain.xml"), ramp_entries),
        "Objects": objects.encode("utf-8"),
    })
    import shutil
    shutil.copy(out, f"/Applications/StarCraft II/maps/{_map_name(step, noquad)}.SC2Map")
    print(f"built {_map_name(step, noquad)}: {len(band)} interior cols ({64 // step} ramp cells) "
          f"ramp_x0={ramp_x0} hi_edge={hi_edge} y={ly} run={run} lo={lo_xy} hi={hi_xy} noquad={noquad}")


def probe(step: int) -> None:
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
    noquad = "--noquad" in sys.argv
    tag = f"{step}{'q0' if noquad else ''}"
    (_a, _v, _cw, _ch, x0, x1, y0, y1,
     band, ramp_x0, hi_edge, ly, lo_xy, hi_xy) = _geom(step)
    lo_pt = Point2((lo_xy[0] + 0.5, lo_xy[1] + 0.5))
    hi_pt = Point2((hi_xy[0] + 0.5, hi_xy[1] + 0.5))

    class P(BotAI):
        def __init__(self):
            super().__init__()
            self._t = 0
            self._utag = None
            self._x0 = None
            self._maxx = None
            self._reported = False

        async def on_start(self):
            gi = self.game_info
            path = gi.pathing_grid.data_numpy
            hgt = gi.terrain_height.data_numpy
            lbl, n = ndimage.label(path)

            def comp(x, y):
                if path[y, x]:
                    return int(lbl[y, x])
                ys, xs = np.where(path > 0)
                i = np.argmin((ys - y) ** 2 + (xs - x) ** 2)
                return int(lbl[ys[i], xs[i]])

            cl, cr = comp(*lo_xy), comp(*hi_xy)
            cells = 64 // step
            print(f"PATHGRID STEP={tag} ({cells} ramp cells) ramps_detected={len(gi.map_ramps)} "
                  f"lo_comp={cl} hi_comp={cr} CONNECTED={cl == cr} "
                  f"h_lo={hgt[lo_xy[1], lo_xy[0]]} h_hi={hgt[hi_xy[1], hi_xy[0]]}")
            # spawn a marine on the LOW plateau for the real move test
            await self.client.debug_create_unit([[UnitTypeId.MARINE, 1, lo_pt, 1]])

        async def on_step(self, it):
            self._t += 1
            u = None
            marines = self.units(UnitTypeId.MARINE)
            if self._utag is not None:
                u = next((x for x in self.units if x.tag == self._utag), None)
            elif marines:
                u = marines.closest_to(lo_pt)
            if u is None:
                if self._t > 40:
                    print(f"UNITMOVE STEP={step} RESULT=NO_UNIT (spawn failed)")
                    await self.client.leave()
                return
            if self._utag is None:
                self._utag = u.tag
                self._x0 = u.position.x
                self._maxx = u.position.x
                u.move(hi_pt)
                return
            u.move(hi_pt)
            self._maxx = max(self._maxx, u.position.x)
            reached = u.position.x >= (hi_edge + 4)
            if reached or self._t > 900:
                hgt = self.game_info.terrain_height.data_numpy
                ux, uy = int(u.position.x), int(u.position.y)
                print(f"UNITMOVE STEP={tag} ({64 // step} cells) reached_high={reached} "
                      f"start_x={self._x0:.1f} max_x={self._maxx:.1f} final=({u.position.x:.1f},"
                      f"{u.position.y:.1f}) ramp_x0={ramp_x0} hi_edge={hi_edge} "
                      f"unit_terrain_h={hgt[uy, ux]}")
                await self.client.leave()

    run_game(maps.get(_map_name(step, noquad)),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    mode, step = sys.argv[1], int(sys.argv[2])
    (build if mode == "build" else probe)(step)
