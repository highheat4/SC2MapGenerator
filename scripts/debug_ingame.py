"""Launch gen_213 headless, skip the crashing expansion finder, and dump what the ENGINE
actually computed: placement grid, pathing grid, terrain height, and resource positions.

    PYTHONPATH=src .venv/bin/python scripts/debug_ingame.py
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.bot_ai_internal import BotAIInternal
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

# neuter the call that crashes so we get into on_start
BotAIInternal._find_expansion_locations = lambda self: None


class Probe(BotAI):
    async def on_start(self):
        gi = self.game_info
        place = gi.placement_grid.data_numpy      # 1 == buildable
        path = gi.pathing_grid.data_numpy         # 1 == pathable
        height = gi.terrain_height.data_numpy     # engine height (byte)
        print("map size:", gi.map_size, "playable:", gi.playable_area)
        print("placement buildable cells:", int(place.sum()), "/", place.size)
        print("pathing pathable cells:", int(path.sum()))
        print("height unique:", sorted(set(height.flatten().tolist()))[:12])

        mfs = self.mineral_field
        gas = self.vespene_geyser
        print(f"minerals={len(mfs)} geysers={len(gas)}")

        # ramps the engine detects (issue 1: main->natural must be a usable ramp)
        print(f"engine map_ramps detected: {len(self.game_info.map_ramps)}")
        for rp in self.game_info.map_ramps[:8]:
            try:
                print(f"  ramp size={len(rp.points)} top={rp.top_center} bottom={rp.bottom_center}")
            except Exception:
                print(f"  ramp size={len(rp.points)} (no top/bottom -> not a clean ramp)")

        # buildability near each base (issue 3). Use geysers as base anchors (2/base) and
        # start locations (mains). Report buildable fraction in the 5x5 townhall pocket a few
        # tiles back from the resources, plus the biggest buildable window near the anchor.
        def build_report(cx, cy, tag):
            best = 0
            for oy in range(-6, 7):
                for ox in range(-6, 7):
                    y0, x0 = cy + oy - 2, cx + ox - 2
                    foot = place[y0:y0 + 5, x0:x0 + 5]
                    if foot.shape == (5, 5):
                        best = max(best, int(foot.sum()))
            here = place[cy - 2:cy + 3, cx - 2:cx + 3]
            print(f"  {tag} ({cx},{cy}) here={int(here.sum())}/25 best5x5nearby={best}/25")

        print("buildability near mains (start locations):")
        for s in gi.start_locations:
            build_report(int(s.x), int(s.y), "main")
        print("buildability near geysers (base anchors):")
        for g in gas:
            build_report(int(g.position.x), int(g.position.y), "gas")

        # render placement grid with resources marked
        h, w = place.shape
        img = np.zeros((h, w, 3), np.uint8)
        img[path > 0] = (40, 40, 40)
        img[place > 0] = (60, 160, 60)
        for m in mfs:
            img[int(m.position.y), int(m.position.x)] = (80, 160, 255)
        for g in gas:
            img[int(g.position.y), int(g.position.x)] = (0, 255, 0)
        for s in gi.start_locations:
            img[int(s.y), int(s.x)] = (255, 0, 0)
        Image.fromarray(np.flipud(img)).resize((w * 4, h * 4), Image.NEAREST).save(
            "outputs/export/gen_213_engine_grids.png")
        np.save("outputs/export/eng_place.npy", place)
        np.save("outputs/export/eng_path.npy", path)
        np.save("outputs/export/eng_height.npy", height)
        print("wrote outputs/export/gen_213_engine_grids.png + npy grids")
        await self.client.leave()


MAP = sys.argv[1] if len(sys.argv) > 1 else "gen_213"
run_game(maps.get(MAP),
         [Bot(Race.Terran, Probe()), Computer(Race.Terran, Difficulty.Easy)],
         realtime=False)
