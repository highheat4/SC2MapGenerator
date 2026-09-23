"""Find the MAX walkable cliff step. Build a synthetic diagonal 2-plateau map (low 64, high 128)
joined by a diagonal ramp whose gradient steps by S per cell, plus a matching rampList entry + Pnp.
Probe query_pathing across it for S in {8,16,32,64}. The largest S that stays WALKABLE tells us how
short (steep) a 1-level ramp can be: a 64 change needs 64/S cells.

    PYTHONPATH=src .venv/bin/python scripts/step_probe.py build
    PYTHONPATH=src .venv/bin/python scripts/step_probe.py probe
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
STEPS = [8, 16, 32, 64]


def _one(step: int):
    """Return (out_path, low_xy, high_xy) for a map with a ramp stepping by `step`."""
    arch = MPQArchive(TEMPLATE)
    cliff_blob = arch.read_file("t3SyncCliffLevel")
    tmpl_smap = arch.read_file("t3SyncHeightMap")
    tmpl_hmap = arch.read_file("t3HeightMap")
    version, cw, ch, _ = sc2map.decode_cliff(cliff_blob)
    palette = sc2map.harvest_palette(cliff_blob, tmpl_smap, tmpl_hmap)
    mi = arch.read_file("MapInfo")
    poff, _, _ = sc2map._mapinfo_playable_offset(mi)
    pl, pb, pr, pt = struct.unpack_from("<4i", mi, poff)
    x0, x1 = pl + 8, pr - 8
    y0, y1 = pb + 8, pt - 8

    cliff = np.zeros((ch, cw), dtype=np.uint16)
    # diagonal split: low (64) where (x - y) < 0, high (128) where > 0, ramp band around 0
    lo_c, hi_c = 64, 128
    ncells = max(1, (hi_c - lo_c) // step)                 # gradient cells across the band
    for y in range(y0, y1):
        for x in range(x0, x1):
            s = (x - x0) - (y - y0)                        # diagonal coordinate
            if s < -ncells:
                cliff[y, x] = lo_c
            elif s > ncells:
                cliff[y, x] = hi_c
            else:
                k = int(round((s + ncells) / 2))           # 0..ncells
                cliff[y, x] = min(hi_c, lo_c + k * step)
    walkable = np.zeros((ch, cw), dtype=bool)
    walkable[y0:y1, x0:x1] = True

    # ramp entry centered on the diagonal midline (dir 7 = uphill toward +x,+y)
    cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
    entry = sc2map.make_ramp_entry((cxm, cym), 7, 4, 1, 2)
    lx, ly = x0 + 6, y0 + 20          # deep on low side
    hx, hy = x1 - 6, y1 - 20          # deep on high side
    objects = (
        '<?xml version="1.0" encoding="utf-8"?>\n<PlacedObjects Version="27">\n'
        f'    <ObjectPoint Id="1001" Position="{lx}.5,{ly}.5,0" Scale="1,1,1" '
        'Type="StartLoc" Name="Start Location 001" Color="0,0,0,0"/>\n'
        f'    <ObjectPoint Id="1002" Position="{hx}.5,{hy}.5,0" Scale="1,1,1" '
        'Type="StartLoc" Name="Start Location 002" Color="0,0,0,0"/>\n'
        '</PlacedObjects>\n')
    out = Path(f"outputs/export/step_{step}.SC2Map")
    export_via_stormlib(TEMPLATE, out, {
        "t3SyncCliffLevel": sc2map.encode_cliff(cliff, version=version),
        "t3SyncHeightMap": sc2map.build_smap(cliff, tmpl_smap, palette),
        "t3HeightMap": sc2map.build_hmap(cliff, tmpl_hmap, palette),
        "CellAttribute_Pnp": sc2map.author_pnp(arch.read_file("CellAttribute_Pnp"), walkable),
        "t3CellFlags": sc2map.clear_cell_flags(arch.read_file("t3CellFlags")),
        "t3Terrain.xml": sc2map.set_ramp_list(arch.read_file("t3Terrain.xml"), [entry]),
        "Objects": objects.encode("utf-8"),
    })
    import shutil
    shutil.copy(out, f"/Applications/StarCraft II/maps/step_{step}.SC2Map")
    return out, (lx, ly), (hx, hy)


def build() -> None:
    for s in STEPS:
        _one(s)
        print(f"built step_{s}")


def probe() -> None:
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.bot_ai_internal import BotAIInternal
    from sc2.data import Difficulty, Race
    from sc2.main import run_game
    from sc2.player import Bot, Computer
    from sc2.position import Point2

    BotAIInternal._find_expansion_locations = lambda self: None
    # rebuild to recover the low/high sample points deterministically
    coords = {s: _one(s)[1:] for s in STEPS}

    for step in STEPS:
        (lx, ly), (hx, hy) = coords[step]

        class P(BotAI):
            _step = step
            _lo = (lx, ly)
            _hi = (hx, hy)

            async def on_start(self):
                d = await self.client.query_pathing(
                    Point2((self._lo[0] + 0.5, self._lo[1] + 0.5)),
                    Point2((self._hi[0] + 0.5, self._hi[1] + 0.5)))
                cells = 64 // self._step
                print(f"STEP={self._step:>2} ({cells} ramp cells) dist={d} WALKABLE={d is not None}")
                await self.client.leave()

            async def on_step(self, it):
                pass

        run_game(maps.get(f"step_{step}"),
                 [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
                 realtime=False)


if __name__ == "__main__":
    (build if sys.argv[1] == "build" else probe)()
