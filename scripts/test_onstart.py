"""Export a batch of seeds and, for each, launch SC2 with an idle bot to confirm on_start (the
expansion finder) succeeds -- the exact thing that used to crash. Prints PASS/FAIL per seed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2 import maps  # noqa: E402
from sc2.bot_ai import BotAI  # noqa: E402
from sc2.data import Difficulty, Race  # noqa: E402
from sc2.main import run_game  # noqa: E402
from sc2.player import Bot, Computer  # noqa: E402

from sc2mapgen.export import ExportConfig, export_sc2map  # noqa: E402
from sc2mapgen.generate.rasterize import RasterConfig, rasterize  # noqa: E402
from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator  # noqa: E402

MAPS_DIR = Path("/Applications/StarCraft II/Maps")
OUT = Path("outputs/export")


class Idle(BotAI):
    async def on_start(self):
        print(f"    PASS expansions={len(self.expansion_locations_list)} "
              f"start={self.start_location}")
        await self.client.leave()

    async def on_step(self, it):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    gen = SkeletonGenerator(config=GenConfig())
    for s in args.seeds:
        mapir = rasterize(gen.generate(s), seed=s, cfg=RasterConfig())
        info = export_sc2map(mapir, OUT / f"{mapir.map_name}.SC2Map", ExportConfig())
        shutil.copy(info["out_path"], MAPS_DIR / f"{mapir.map_name}.SC2Map")
        print(f"seed {s}: {mapir.map_name} -> {Path(info['template']).name} off{info['offset']}")
        try:
            run_game(maps.get(mapir.map_name),
                     [Bot(Race.Terran, Idle()), Computer(Race.Terran, Difficulty.Easy)],
                     realtime=False)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAIL {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
