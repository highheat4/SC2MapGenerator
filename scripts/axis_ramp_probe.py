"""DECISIVE experiment: is an AXIS-ALIGNED ramp walkable? Build the same 2-plateau axis-aligned
full-8-step staircase as ramp_probe, but ADD an axis-aligned <ramp> quad (uphill=+x, u=(1,0)) over
it. If the engine detects the ramp and routes a ground unit L->R, axis-aligned ramps work and we can
keep boxy geometry. If not (FrostLE has only 45deg ramps; SC2 ramps are famously diagonal-only),
we must author diagonal ramp corridors.

    PYTHONPATH=src .venv/bin/python scripts/axis_ramp_probe.py build [dir]
    PYTHONPATH=src .venv/bin/python scripts/axis_ramp_probe.py probe
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
OUT = Path("outputs/export/axis_ramp_test.SC2Map")


def _ax_ramp_entry(base_c, lo: int, hi: int, width: float, direction: int, run: float = 2.0) -> str:
    """An axis-aligned <ramp> entry: uphill u=(+1,0) (east), right r=(0,-1). Mirrors the diagonal
    make_ramp_entry geometry but with cardinal unit vectors and unit (not sqrt2) spacing."""
    ux, uy = 1.0, 0.0
    rx, ry = 0.0, -1.0
    bx, by = float(base_c[0]), float(base_c[1])
    mx, my = bx - run * ux, by - run * uy               # downhill center
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


def build() -> None:
    direction = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    arch = MPQArchive(TEMPLATE)
    cliff_blob = arch.read_file("t3SyncCliffLevel")
    tmpl_smap = arch.read_file("t3SyncHeightMap")
    tmpl_hmap = arch.read_file("t3HeightMap")
    version, cw, ch, _ = sc2map.decode_cliff(cliff_blob)
    palette = sc2map.harvest_palette(cliff_blob, tmpl_smap, tmpl_hmap)

    mi = arch.read_file("MapInfo")
    poff, _, _ = sc2map._mapinfo_playable_offset(mi)
    pl, pb, pr, pt = struct.unpack_from("<4i", mi, poff)

    lo_cliff = palette.tiers[0][0]        # 64
    hi_cliff = palette.tiers[1][0]        # 128
    band = list(range(lo_cliff + 8, hi_cliff, 8))          # 72..120 (full 8-step)

    cliff = np.zeros((ch, cw), dtype=np.uint16)
    x0, x1 = pl + 6, pr - 6
    y0, y1 = pb + 6, pt - 6
    midx = (x0 + x1) // 2
    ramp_x0 = midx - len(band) // 2
    for y in range(y0, y1):
        for x in range(x0, x1):
            if x < ramp_x0:
                cliff[y, x] = lo_cliff
            elif x >= ramp_x0 + len(band):
                cliff[y, x] = hi_cliff
            else:
                cliff[y, x] = band[x - ramp_x0]

    cliff_bytes = sc2map.encode_cliff(cliff, version=version)
    walkable = np.zeros((ch, cw), dtype=bool)
    walkable[y0:y1, x0:x1] = True
    ly = (y0 + y1) // 2
    lxc, rxc = (x0 + ramp_x0) // 2, (ramp_x0 + len(band) + x1) // 2

    # axis-aligned quad centered on the staircase, uphill east onto the high (128) plateau
    hi_edge_x = ramp_x0 + len(band)                        # first high-plateau column
    entry = _ax_ramp_entry((hi_edge_x + 0.5, ly + 0.5), lo=1, hi=2, width=4.0, direction=direction)
    terrain = sc2map.set_ramp_list(arch.read_file("t3Terrain.xml"), [entry])

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
        "t3Terrain.xml": terrain,
        "Objects": objects.encode("utf-8"),
    }
    export_via_stormlib(TEMPLATE, OUT, replacements)
    import shutil
    shutil.copy(OUT, "/Applications/StarCraft II/maps/axis_ramp_test.SC2Map")
    print(f"wrote axis ramp dir={direction} at hi_edge_x={hi_edge_x} y={ly}")
    print("LEFT_XY", x0 + 2, ly, "RIGHT_XY", x1 - 2, ly)


def probe() -> None:
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.bot_ai_internal import BotAIInternal
    from sc2.data import Difficulty, Race
    from sc2.main import run_game
    from sc2.player import Bot, Computer
    from sc2.position import Point2

    BotAIInternal._find_expansion_locations = lambda self: None
    arch = MPQArchive(str(OUT))
    _, cw, ch, _ = sc2map.decode_cliff(arch.read_file("t3SyncCliffLevel"))
    mi = arch.read_file("MapInfo")
    poff, _, _ = sc2map._mapinfo_playable_offset(mi)
    pl, pb, pr, pt = struct.unpack_from("<4i", mi, poff)
    x0, x1 = pl + 6, pr - 6
    y0, y1 = pb + 6, pt - 6
    ly = (y0 + y1) // 2

    class P(BotAI):
        async def on_start(self):
            gi = self.game_info
            print(f"MAP_RAMPS_DETECTED={len(gi.map_ramps)}")
            d = await self.client.query_pathing(Point2((x0 + 2 + 0.5, ly + 0.5)),
                                                Point2((x1 - 2 + 0.5, ly + 0.5)))
            print(f"ENGINE_PATH L->R across AXIS ramp: dist={d} WALKABLE={d is not None}")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get("axis_ramp_test"),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    (build if sys.argv[1] == "build" else probe)()
