"""Copy FrostLE, mutate ONE ramp's rampList quad (single variable = quad WIDTH), repack, and ask the
engine whether that ramp is still detected + walkable. Isolates: does an oversized quad (what our
generator emits: quad width = full band extent) break detection on an otherwise-valid gold ramp?

  PYTHONPATH=src python scripts/_rampmut.py build  control|w4|w8|w12|w16|remove
  SC2PATH=... PYTHONPATH=src python scripts/_rampmut.py probe control|w4|w8|w12|w16|remove
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.export import sc2map  # noqa: E402
from sc2mapgen.export.stormlib_mpq import export_via_stormlib  # noqa: E402

TEMPLATE = "gold_maps/FrostLE.SC2Map"
K = 7                      # FrostLE ramp index to mutate: dir5 1->2, base.c=(136,58), gold width 4
BASE_C = (136.0, 58.0)
DIRECTION = 5
LO, HI = 1, 2             # cliff LEVELS (not tier idx) -- FrostLE lo=1 hi=2
MAPS = Path("/Applications/StarCraft II/maps")


def build(variant: str) -> None:
    a = MPQArchive(TEMPLATE)
    xml = a.read_file("t3Terrain.xml").decode("utf-8", "replace")
    ramps = re.findall(r"<ramp\b[^>]*/>", xml)
    orig = ramps[K]
    if variant == "control":
        new = orig
    elif variant == "remove":
        new = ""
    else:
        w = {"w4": 4.0, "w8": 8.0, "w12": 12.0, "w16": 16.0}[variant]
        new = sc2map.make_ramp_entry(BASE_C, DIRECTION, w, LO, HI)
    # rebuild rampList
    new_ramps = [r for j, r in enumerate(ramps) if j != K] + ([new] if new else [])
    body = "\n".join("            " + e for e in new_ramps)
    block = f'<rampList num="{len(new_ramps)}">\n{body}\n        </rampList>'
    xml2 = re.sub(r"<rampList\b[^>]*>.*?</rampList>", block, xml, count=1, flags=re.DOTALL)
    out = Path("outputs/export") / f"frostmut_{variant}.SC2Map"
    export_via_stormlib(TEMPLATE, out, {"t3Terrain.xml": xml2.encode("utf-8")})
    import shutil
    shutil.copy(out, MAPS / f"frostmut_{variant}.SC2Map")
    print(f"built {variant}: rampList now {len(new_ramps)} entries -> {out}")


def probe(variant: str) -> None:
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.bot_ai_internal import BotAIInternal
    from sc2.data import Difficulty, Race
    from sc2.main import run_game
    from sc2.player import Bot, Computer
    from sc2.position import Point2

    BotAIInternal._find_expansion_locations = lambda self: None
    target = Point2((BASE_C[0], BASE_C[1]))

    class P(BotAI):
        async def on_start(self):
            gi = self.game_info
            ramps = gi.map_ramps
            print(f"[{variant}] MAP_RAMPS_DETECTED={len(ramps)}")
            # dump ALL ramp centers (sorted) so we can diff variants offline to locate ramp K
            cents = []
            for r in ramps:
                try:
                    tc = r.top_center; bc = r.bottom_center
                except Exception:  # noqa: BLE001
                    continue
                mid = ((tc[0]+bc[0])/2, (tc[1]+bc[1])/2)
                walk = await self.client.query_pathing(Point2(tc), Point2(bc))
                cents.append((round(mid[0],1), round(mid[1],1), len(r.points), walk is not None))
            for c in sorted(cents):
                print(f"[{variant}] RAMP mid=({c[0]},{c[1]}) npts={c[2]} walkable={c[3]}")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get(f"frostmut_{variant}"),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    (build if sys.argv[1] == "build" else probe)(sys.argv[2])
