"""SYNTHESIS PROOF: keep FrostLE terrain intact, but replace its rampList with a SINGLE ramp entry
that WE generate from scratch (make_ramp_entry) for ramp #6's location. If the engine routes a
ground unit across it, our synthesized rampList entries produce walkable ramps -> headless ramps.

    PYTHONPATH=src .venv/bin/python scripts/synth_ramp_probe.py build
    PYTHONPATH=src .venv/bin/python scripts/synth_ramp_probe.py probe
"""
from __future__ import annotations

import sys
from pathlib import Path

from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.export import sc2map  # noqa: E402
from sc2mapgen.export.stormlib_mpq import export_via_stormlib  # noqa: E402

TEMPLATE = "gold_maps/FrostLE.SC2Map"
OUT = Path("outputs/export/synth_ramp.SC2Map")
# ramp #6 ground-truth location: uphill edge center (48,58), dir 4, width 4, lo1->hi2
RAMP = dict(base_c=(48, 58), direction=4, width=4, lo=1, hi=2)


def build() -> None:
    arch = MPQArchive(TEMPLATE)
    entry = sc2map.make_ramp_entry(**RAMP)
    terrain = sc2map.set_ramp_list(arch.read_file("t3Terrain.xml"), [entry])
    export_via_stormlib(TEMPLATE, OUT, {"t3Terrain.xml": terrain,
                                        "Objects": arch.read_file("Objects")})
    import shutil
    shutil.copy(OUT, "/Applications/StarCraft II/maps/synth_ramp.SC2Map")
    print("wrote", OUT, "with 1 GENERATED ramp entry at", RAMP["base_c"])


def probe() -> None:
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.bot_ai_internal import BotAIInternal
    from sc2.data import Difficulty, Race
    from sc2.main import run_game
    from sc2.player import Bot, Computer
    from sc2.position import Point2

    BotAIInternal._find_expansion_locations = lambda self: None
    ux, uy = sc2map._RAMP_DIR_U[RAMP["direction"]]
    cx, cy = RAMP["base_c"]

    class P(BotAI):
        async def on_start(self):
            gi = self.game_info
            print(f"MAP_RAMPS_DETECTED={len(gi.map_ramps)}")
            for r in gi.map_ramps:
                try:
                    print(f"  detected ramp top={tuple(round(v,1) for v in r.top_center)} "
                          f"bot={tuple(round(v,1) for v in r.bottom_center)}")
                except Exception:  # noqa: BLE001
                    pass
            hi = Point2((cx + 3.0 * ux + 0.5, cy + 3.0 * uy + 0.5))   # high plateau side
            lo = Point2((cx - 3.0 * ux + 0.5, cy - 3.0 * uy + 0.5))   # low plateau side
            d = await self.client.query_pathing(lo, hi)
            print(f"ENGINE_PATH across GENERATED ramp: lo={tuple(round(v,1) for v in lo)} "
                  f"hi={tuple(round(v,1) for v in hi)} dist={d} WALKABLE={d is not None}")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get("synth_ramp"),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    (build if sys.argv[1] == "build" else probe)()
