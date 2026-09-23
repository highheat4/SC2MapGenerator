"""CONTROL: repack FrostLE (nearly) unmodified via StormLib, then ask the ENGINE for its own ramp
centers and query_pathing across one. Isolates two hypotheses:
  * if FrostLE's real ramps stay WALKABLE after our repack -> repack is fine; our headless authoring
    is missing the rampList (editor-baked ramp definition).
  * if they become UNWALKABLE -> the StormLib repack itself breaks ramp pathing.

    PYTHONPATH=src .venv/bin/python scripts/control_probe.py build
    PYTHONPATH=src .venv/bin/python scripts/control_probe.py probe
"""
from __future__ import annotations

import sys
from pathlib import Path

from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.export.stormlib_mpq import export_via_stormlib  # noqa: E402

TEMPLATE = "gold_maps/FrostLE.SC2Map"
OUT = Path("outputs/export/control_frost.SC2Map")


def build() -> None:
    arch = MPQArchive(TEMPLATE)
    # Repack with NO gameplay changes at all: pass through the original Objects untouched.
    replacements = {"Objects": arch.read_file("Objects")}
    export_via_stormlib(TEMPLATE, OUT, replacements)
    import shutil
    shutil.copy(OUT, "/Applications/StarCraft II/maps/control_frost.SC2Map")
    print("wrote", OUT)


def probe() -> None:
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.bot_ai_internal import BotAIInternal
    from sc2.data import Difficulty, Race
    from sc2.main import run_game
    from sc2.player import Bot, Computer
    from sc2.position import Point2

    BotAIInternal._find_expansion_locations = lambda self: None

    class P(BotAI):
        async def on_start(self):
            gi = self.game_info
            ramps = gi.map_ramps
            print(f"MAP_RAMPS_DETECTED={len(ramps)}")
            tested = 0
            for r in ramps:
                try:
                    top = Point2(r.top_center)
                    bot = Point2(r.bottom_center)
                except Exception:  # noqa: BLE001
                    continue
                d = await self.client.query_pathing(top, bot)
                print(f"  ramp top={tuple(round(v,1) for v in top)} "
                      f"bot={tuple(round(v,1) for v in bot)} dist={d} WALKABLE={d is not None}")
                tested += 1
                if tested >= 6:
                    break
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get("control_frost"),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    (build if sys.argv[1] == "build" else probe)()
