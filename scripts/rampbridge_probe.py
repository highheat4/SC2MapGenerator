"""Per-ramp GLOBAL bridging test. For each synthesized rampList quad, query_pathing across it at
increasing radii along its uphill axis. A ramp that passes at r=3 (local) but fails at larger r
does NOT actually bridge its two plateaus for the engine -> partial quad coverage on a wide/irregular
ramp. This localizes exactly which ramps seal the map.

    PYTHONPATH=src .venv/bin/python scripts/rampbridge_probe.py gen_213
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

NAME = sys.argv[1] if len(sys.argv) > 1 else "gen_213"
MAP = f"/Applications/StarCraft II/maps/{NAME}.SC2Map"
RADII = (3, 5, 7, 10, 14)


def _ramp_bases(path: str):
    t = MPQArchive(path).read_file("t3Terrain.xml").decode("utf-8", "replace")
    out = []
    for e in re.findall(r"<ramp [^>]*?/>", t):
        d = dict(re.findall(r'(\w+)="([^"]*)"', e))
        bu = re.search(r"u\(([^)]+)\)", d["base"]).group(1).split(",")
        bc = re.search(r"c=\(([^)]+)\)", d["base"]).group(1).split(",")
        out.append(((float(bc[0]), float(bc[1])), (float(bu[0]), float(bu[1]))))
    return out


def main() -> None:
    import shutil
    shutil.copy(f"outputs/export/{NAME}.SC2Map", MAP)
    bases = _ramp_bases(MAP)
    print(f"rampList entries={len(bases)}")

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
            print(f"MAP_RAMPS_DETECTED={len(self.game_info.map_ramps)} of {len(bases)}")
            for i, ((cx, cy), (ux, uy)) in enumerate(bases):
                row = []
                for r in RADII:
                    lo = Point2((cx - r * ux + 0.5, cy - r * uy + 0.5))
                    hi = Point2((cx + r * ux + 0.5, cy + r * uy + 0.5))
                    d = await self.client.query_pathing(lo, hi)
                    row.append("Y" if d is not None else ".")
                print(f"  ramp#{i:>2} base=({cx:.0f},{cy:.0f}) "
                      f"[{' '.join(f'{r}:{c}' for r, c in zip(RADII, row))}]")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get(NAME),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    main()
