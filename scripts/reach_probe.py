"""Query main1 -> every base center on an ALREADY-EXPORTED map (no re-export). Reveals the engine's
component split. Base coords come from re-running the rasterizer (deterministic) + the export offset.

    PYTHONPATH=src .venv/bin/python scripts/reach_probe.py 234 [off_x off_y]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 234
OX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OY = int(sys.argv[3]) if len(sys.argv) > 3 else 0
NAME = f"gen_{SEED}"


def main() -> None:
    import shutil
    shutil.copy(f"outputs/export/{NAME}.SC2Map", f"/Applications/StarCraft II/maps/{NAME}.SC2Map")
    from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator
    from sc2mapgen.generate.rasterize import RasterConfig, rasterize
    g = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
    mapir = rasterize(g.generate(SEED), seed=SEED, cfg=RasterConfig())
    bases = [(b.kind.name, int(round(b.x)) + OX, int(round(b.y)) + OY) for b in mapir.bases]

    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.bot_ai_internal import BotAIInternal
    from sc2.data import Difficulty, Race
    from sc2.main import run_game
    from sc2.player import Bot, Computer
    from sc2.position import Point2

    BotAIInternal._find_expansion_locations = lambda self: None
    mains = [b for b in bases if b[0] == "MAIN"]
    m1 = mains[0]

    class P(BotAI):
        async def on_start(self):
            src = Point2((m1[1] + 0.5, m1[2] + 0.5))
            print(f"MAP_RAMPS_DETECTED={len(self.game_info.map_ramps)}")
            for kind, x, y in bases:
                d = await self.client.query_pathing(src, Point2((x + 0.5, y + 0.5)))
                print(f"  MAIN1 -> {kind:8} ({x:>3},{y:>3}) REACH={d is not None} "
                      f"dist={None if d is None else round(d,1)}")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get(NAME),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    main()
