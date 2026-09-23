"""Controlled experiment: does an authored cliff-substep staircase make SC2 treat a transition
as a WALKABLE ramp? Build a trivial 2-plateau map (left tier0, right tier1) joined by a vertical
ramp band that steps 64->72->...->128, export it, then (separately) probe engine pathing to see
if the two plateaus land in one pathable component.

    PYTHONPATH=src .venv/bin/python scripts/ramp_probe.py build      # write outputs/export/ramp_test.SC2Map
    PYTHONPATH=src .venv/bin/python scripts/ramp_probe.py probe      # launch SC2 + report connectivity
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
OUT = Path("outputs/export/ramp_test.SC2Map")


def build() -> None:
    arch = MPQArchive(TEMPLATE)
    cliff_blob = arch.read_file("t3SyncCliffLevel")
    tmpl_smap = arch.read_file("t3SyncHeightMap")
    tmpl_hmap = arch.read_file("t3HeightMap")
    version, cw, ch, _ = sc2map.decode_cliff(cliff_blob)
    palette = sc2map.harvest_palette(cliff_blob, tmpl_smap, tmpl_hmap)

    mi = arch.read_file("MapInfo")
    poff, _, _ = sc2map._mapinfo_playable_offset(mi)
    pl, pb, pr, pt = struct.unpack_from("<4i", mi, poff)
    print("playable", pl, pb, pr, pt, "grid", cw, ch)

    lo_cliff = palette.tiers[0][0]        # 64
    hi_cliff = palette.tiers[1][0]        # 128
    subs = palette.sublevels_between(lo_cliff, hi_cliff)   # 72..112 (template's harvested set)
    # FORCE a full diff-8 staircase so BOTH junctions step by exactly 8 (no 16-jump at the top):
    if len(sys.argv) > 2 and sys.argv[2] == "full8":
        subs = list(range(lo_cliff + 8, hi_cliff, 8))      # 72,80,...,120
    print("tiers", palette.tier_cliffs, "subs", subs)

    cliff = np.zeros((ch, cw), dtype=np.uint16)
    # a big playable rectangle, inset a few cells from the playable bounds
    x0, x1 = pl + 6, pr - 6
    y0, y1 = pb + 6, pt - 6
    midx = (x0 + x1) // 2
    band = subs                                   # one column per sub-level
    ramp_x0 = midx - len(band) // 2
    for y in range(y0, y1):
        for x in range(x0, x1):
            if x < ramp_x0:
                cliff[y, x] = lo_cliff
            elif x >= ramp_x0 + len(band):
                cliff[y, x] = hi_cliff
            else:
                cliff[y, x] = band[x - ramp_x0]   # staircase 72,80,...,112

    cliff_bytes = sc2map.encode_cliff(cliff, version=version)
    # walkable mask = the whole authored plateau/ramp rectangle
    walkable = np.zeros((ch, cw), dtype=bool)
    walkable[y0:y1, x0:x1] = True
    # minimal Objects: one start location centred on each plateau so the map is valid + playable
    ly = (y0 + y1) // 2
    lxc, rxc = (x0 + ramp_x0) // 2, (ramp_x0 + len(band) + x1) // 2
    objects = (
        '<?xml version="1.0" encoding="utf-8"?>\n<PlacedObjects Version="27">\n'
        f'    <ObjectPoint Id="1001" Position="{lxc}.5,{ly}.5,0" Scale="1,1,1" '
        'Type="StartLoc" Name="Start Location 001" Color="0,0,0,0"/>\n'
        f'    <ObjectPoint Id="1002" Position="{rxc}.5,{ly}.5,0" Scale="1,1,1" '
        'Type="StartLoc" Name="Start Location 002" Color="0,0,0,0"/>\n'
        '</PlacedObjects>\n'
    )
    replacements = {
        "t3SyncCliffLevel": cliff_bytes,
        "t3SyncHeightMap": sc2map.build_smap(cliff, tmpl_smap, palette),
        "t3HeightMap": sc2map.build_hmap(cliff, tmpl_hmap, palette),
        "CellAttribute_Pnp": sc2map.author_pnp(arch.read_file("CellAttribute_Pnp"), walkable),
        "t3CellFlags": sc2map.clear_cell_flags(arch.read_file("t3CellFlags")),
        # strip FrostLE's 16 phantom render ramps so map_ramps only reflects OUR staircase
        "t3Terrain.xml": sc2map.strip_render_relief(arch.read_file("t3Terrain.xml")),
        "Objects": objects.encode("utf-8"),
    }
    export_via_stormlib(TEMPLATE, OUT, replacements)
    import shutil
    shutil.copy(OUT, "/Applications/StarCraft II/maps/ramp_test.SC2Map")
    print("wrote", OUT, "left plateau x<", ramp_x0, " right x>=", ramp_x0 + len(band),
          " sample cells: left", (x0 + 2, (y0 + y1) // 2), "right", (x1 - 2, (y0 + y1) // 2))
    print("LEFT_XY", x0 + 2, (y0 + y1) // 2, "RIGHT_XY", x1 - 2, (y0 + y1) // 2)


def probe() -> None:
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.bot_ai_internal import BotAIInternal
    from sc2.data import Difficulty, Race
    from sc2.main import run_game
    from sc2.player import Bot, Computer
    from scipy import ndimage

    BotAIInternal._find_expansion_locations = lambda self: None

    class P(BotAI):
        async def on_start(self):
            gi = self.game_info
            path = gi.pathing_grid.data_numpy
            place = gi.placement_grid.data_numpy
            hgt = gi.terrain_height.data_numpy
            print(f"pathable={int(path.sum())} buildable={int(place.sum())} "
                  f"height_uniq={sorted(set(hgt.flatten().tolist()))[:12]}")
            if int(path.sum()) == 0:
                print("NO PATHABLE CELLS - authored terrain is non-pathable")
                await self.client.leave()
                return
            lbl, n = ndimage.label(path)
            # left/right sample cells passed via env-ish constants recomputed here
            arch = MPQArchive(str(OUT))
            _, cw, ch, _ = sc2map.decode_cliff(arch.read_file("t3SyncCliffLevel"))
            mi = arch.read_file("MapInfo")
            poff, _, _ = sc2map._mapinfo_playable_offset(mi)
            pl, pb, pr, pt = struct.unpack_from("<4i", mi, poff)
            x0, x1 = pl + 6, pr - 6
            y0, y1 = pb + 6, pt - 6
            ly, lx = (y0 + y1) // 2, x0 + 2
            ry, rx = (y0 + y1) // 2, x1 - 2

            def comp(x, y):
                if path[y, x]:
                    return int(lbl[y, x])
                ys, xs = np.where(path > 0)
                i = np.argmin((ys - y) ** 2 + (xs - x) ** 2)
                return int(lbl[ys[i], xs[i]])

            cl = comp(lx, ly)
            cr = comp(rx, ry)
            print(f"LEFT({lx},{ly}) comp={cl}  RIGHT({rx},{ry}) comp={cr}  "
                  f"CONNECTED(rawpath)={cl == cr}  pathable={int(path.sum())} components={n}")
            # DEFINITIVE signal: did the engine register a walkable ramp at our staircase?
            print(f"MAP_RAMPS_DETECTED={len(gi.map_ramps)}")
            for r in gi.map_ramps:
                try:
                    print(f"  ramp cells={len(r.points)} top={r.top_center} bot={r.bottom_center}")
                except Exception as e:  # noqa: BLE001
                    print("  ramp (no center):", len(r.points), e)
            # ramp-aware connectivity: treat ramp cells (path==0 & place==0) as passable
            rampcells = (path == 0) & (place == 0)
            passable = (path > 0) | rampcells
            lbl2, n2 = ndimage.label(passable)

            def comp2(x, y):
                if passable[y, x]:
                    return int(lbl2[y, x])
                ys, xs = np.where(passable)
                i = np.argmin((ys - y) ** 2 + (xs - x) ** 2)
                return int(lbl2[ys[i], xs[i]])

            print(f"CONNECTED(rampaware)={comp2(lx, ly) == comp2(rx, ry)} components={n2}")
            # GROUND TRUTH: ask the engine's own pathfinder to route a ground unit L->R.
            from sc2.position import Point2
            for (ax, ay), (bx, by), tag in [((lx, ly), (rx, ry), "L->R across staircase"),
                                            ((lx, ly), (lx + 3, ly), "L->L (sanity, same plateau)")]:
                d = await self.client.query_pathing(Point2((ax + 0.5, ay + 0.5)),
                                                    Point2((bx + 0.5, by + 0.5)))
                print(f"ENGINE_PATH {tag}: dist={d}  WALKABLE={d is not None}")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get("ramp_test"),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    (build if sys.argv[1] == "build" else probe)()
