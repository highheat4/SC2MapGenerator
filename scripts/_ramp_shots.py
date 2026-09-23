"""Launch an exported map and screenshot the game window centred on every ramp.

    PYTHONPATH=src .venv/bin/python scripts/_ramp_shots.py 24 [MAP_NAME] [OUT_DIR]

The macOS client returns all-zero RGB render observations, so this moves the in-game camera to
each ramp and grabs the screen with macOS `screencapture` (needs Screen Recording permission for
the terminal). MAP_NAME defaults to gen_<seed> (must already be in the SC2 maps folder).
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc2 import maps  # noqa: E402
from sc2.bot_ai import BotAI  # noqa: E402
from sc2.bot_ai_internal import BotAIInternal  # noqa: E402
from sc2.data import Difficulty, Race  # noqa: E402
from sc2.main import run_game  # noqa: E402
from sc2.player import Bot, Computer  # noqa: E402
from sc2.position import Point2  # noqa: E402

from sc2mapgen.generate.rasterize import RasterConfig, rasterize  # noqa: E402
from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 24
map_name = sys.argv[2] if len(sys.argv) > 2 else f"gen_{seed}"
out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else f"outputs/shots/{map_name}")
out_dir.mkdir(parents=True, exist_ok=True)
OFF = (1, 0)  # template offset the exporter chose for seed 24 (printed by export_map.py)

mapir = rasterize(SkeletonGenerator(config=GenConfig(symmetries=("rot180",))).generate(seed),
                  seed=seed, cfg=RasterConfig())
targets = []
for i, r in enumerate(mapir.ramps):
    c = np.array(r.cells, dtype=float)
    targets.append((f"ramp{i}_L{r.low_level}-{r.high_level}",
                    Point2((c[:, 0].mean() + OFF[0] + 0.5, c[:, 1].mean() + OFF[1] + 0.5))))

BotAIInternal._find_expansion_locations = lambda self: None


class Shooter(BotAI):
    async def on_start(self):
        await self.client.debug_show_map()

    async def on_step(self, it):
        if it < 20:
            return
        for name, pt in targets:
            await self.client.move_camera(pt)
            await self.client.step(8)
            await asyncio.sleep(1.5)
            p = out_dir / f"{name}_{pt.x:.0f}_{pt.y:.0f}.png"
            subprocess.run(["screencapture", "-x", str(p)], check=False)
            print("saved", p)
        await self.client.leave()


run_game(maps.get(map_name), [Bot(Race.Terran, Shooter()), Computer(Race.Terran, Difficulty.VeryEasy)],
         realtime=False)
