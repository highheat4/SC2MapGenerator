"""Correct connectivity probe: SC2 marks RAMP cells as pathing_grid==0 (that's how python-sc2
detects ramps: pathing==0 & placement==0). So plateaus joined by a walkable ramp look like
separate components if you BFS on pathing>0 alone. Here we build a passable grid = pathing>0
UNION ramp cells, and re-check main<->natural + all-base connectivity.

    PYTHONPATH=src .venv/bin/python scripts/conn_probe.py gen_213_relief
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2 import maps  # noqa: E402
from sc2.bot_ai import BotAI  # noqa: E402
from sc2.bot_ai_internal import BotAIInternal  # noqa: E402
from sc2.data import Difficulty, Race  # noqa: E402
from sc2.main import run_game  # noqa: E402
from sc2.player import Bot, Computer  # noqa: E402

BotAIInternal._find_expansion_locations = lambda self: None
MAP = sys.argv[1] if len(sys.argv) > 1 else "gen_213_relief"


class P(BotAI):
    async def on_start(self):
        gi = self.game_info
        path = gi.pathing_grid.data_numpy          # 1==pathable (RAMPS==0 here!)
        place = gi.placement_grid.data_numpy
        ramps = gi.map_ramps
        h, w = path.shape

        passable = path > 0
        ramp_cells = 0
        for r in ramps:
            for p in r.points:
                x, y = int(p.x), int(p.y)
                if 0 <= y < h and 0 <= x < w:
                    passable[y, x] = True
                    ramp_cells += 1
        print(f"pathing>0={int((path>0).sum())}  +ramp_cells={ramp_cells}  "
              f"passable={int(passable.sum())}  ramps={len(ramps)}")

        lbl_p, np_ = ndimage.label(path > 0)
        lbl, n = ndimage.label(passable)
        print(f"components: pathing-only={np_}  ramp-inclusive={n}")

        def cid(x, y):
            x, y = int(round(x)), int(round(y))
            if passable[y, x]:
                return int(lbl[y, x])
            ys, xs = np.where(passable)
            i = np.argmin((ys - y) ** 2 + (xs - x) ** 2)
            return int(lbl[ys[i], xs[i]])

        starts = [(s.x, s.y) for s in gi.start_locations]
        # cluster geysers into base anchors
        gy = [(g.position.x, g.position.y) for g in self.vespene_geyser]
        used = [False] * len(gy)
        anchors = []
        for i, g in enumerate(gy):
            if used[i]:
                continue
            grp = [g]
            used[i] = True
            for j in range(i + 1, len(gy)):
                if not used[j] and abs(gy[j][0] - g[0]) + abs(gy[j][1] - g[1]) < 16:
                    grp.append(gy[j]); used[j] = True
            anchors.append((float(np.mean([p[0] for p in grp])), float(np.mean([p[1] for p in grp]))))

        m0 = starts[0]
        mc = cid(*m0)
        print(f"main0 {tuple(round(v) for v in m0)} comp={mc}")
        conn = sum(1 for a in anchors if cid(*a) == mc)
        print(f"base anchors connected to main0: {conn}/{len(anchors)} "
              f"(ramp-inclusive)")
        sizes = sorted([int((lbl == i).sum()) for i in range(1, n + 1)], reverse=True)
        print(f"biggest ramp-inclusive comp = {max(sizes)/int(passable.sum()):.1%} of passable")
        await self.client.leave()

    async def on_step(self, it):
        pass


run_game(maps.get(MAP),
         [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
         realtime=False)
