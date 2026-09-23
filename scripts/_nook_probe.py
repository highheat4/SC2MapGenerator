"""Dump what the ENGINE says about a window of cells next to our exported CLIF grid.

    SC2PATH="/Applications/StarCraft II" PYTHONPATH=src .venv/bin/python scripts/_nook_probe.py 24 36 54 104 122

Per cell prints `CLIF/code` where code is: P=pathable, B=buildable (placement), R=inside a detected
map_ramp, .=none. Rows are printed top = highest y (matches the in-game screen). Map must already be
in the SC2 maps folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc2 import maps  # noqa: E402
from sc2.bot_ai import BotAI  # noqa: E402
from sc2.bot_ai_internal import BotAIInternal  # noqa: E402
from sc2.data import Difficulty, Race  # noqa: E402
from sc2.main import run_game  # noqa: E402
from sc2.player import Bot, Computer  # noqa: E402

from sc2mapgen.export import sc2map  # noqa: E402

seed, x0, x1, y0, y1 = (int(v) for v in sys.argv[1:6])
name = f"gen_{seed}"
C = sc2map.decode_cliff(MPQArchive(f"outputs/export/{name}.SC2Map").read_file("t3SyncCliffLevel"))[3]
BotAIInternal._find_expansion_locations = lambda self: None


class Probe(BotAI):
    async def on_start(self):
        gi = self.game_info
        ramp_pts = {(int(p[0]), int(p[1])) for r in gi.map_ramps for p in r.points}
        print(f"window x[{x0},{x1}) y[{y0},{y1}]  cell = CLIF/code  (P pathable, B buildable, R ramp)")
        for y in range(y1, y0 - 1, -1):
            row = []
            for x in range(x0, x1):
                code = ("P" if gi.pathing_grid[(x, y)] else "") + \
                       ("B" if gi.placement_grid[(x, y)] else "") + \
                       ("R" if (x, y) in ramp_pts else "")
                row.append(f"{int(C[y, x]):>3d}/{code or '.':<2s}")
            print(f"y{y:3d} " + " ".join(row))
        print("     " + " ".join(f"{x:>6d}" for x in range(x0, x1)))
        print(f"map_ramps detected: {len(gi.map_ramps)}")
        for r in gi.map_ramps:
            xs = [p[0] for p in r.points]
            ys = [p[1] for p in r.points]
            print(f"  ramp centre=({sum(xs)/len(xs):.1f},{sum(ys)/len(ys):.1f}) cells={len(xs)}")
        await self.client.leave()

    async def on_step(self, it):
        pass


run_game(maps.get(name), [Bot(Race.Terran, Probe()), Computer(Race.Terran, Difficulty.VeryEasy)],
         realtime=False)
