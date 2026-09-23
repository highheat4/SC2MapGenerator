"""Probe an exported RELIEF map (real levels + synthesized ramps): report engine-detected ramps,
per-ramp walkability across each synthesized entry, and main-to-main connectivity through them.

    PYTHONPATH=src .venv/bin/python scripts/relief_probe.py gen_213
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

NAME = sys.argv[1] if len(sys.argv) > 1 else "gen_213"
MAP = f"/Applications/StarCraft II/maps/{NAME}.SC2Map"


def _ramp_bases(path: str):
    t = MPQArchive(path).read_file("t3Terrain.xml").decode("utf-8", "replace")
    out = []
    for e in re.findall(r"<ramp [^>]*?/>", t):
        d = dict(re.findall(r'(\w+)="([^"]*)"', e))
        bu = re.search(r"u\(([^)]+)\)", d["base"]).group(1).split(",")
        bc = re.search(r"c=\(([^)]+)\)", d["base"]).group(1).split(",")
        out.append(((float(bc[0]), float(bc[1])), (float(bu[0]), float(bu[1]))))
    return out


def _starts(path: str):
    x = MPQArchive(path).read_file("Objects").decode("utf-8", "replace")
    pts = []
    for m in re.finditer(r'Type="StartLoc"[^>]*Position="([^"]+)"', x):
        p = m.group(1).split(",")
        pts.append((float(p[0]), float(p[1])))
    if not pts:  # attribute order can vary
        for m in re.finditer(r'Position="([^"]+)"[^>]*Type="StartLoc"', x):
            p = m.group(1).split(",")
            pts.append((float(p[0]), float(p[1])))
    return pts


def main() -> None:
    import shutil
    shutil.copy(f"outputs/export/{NAME}.SC2Map", MAP)
    bases = _ramp_bases(MAP)
    starts = _starts(MAP)
    print(f"rampList entries={len(bases)} startlocs={len(starts)}")

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
            print(f"MAP_RAMPS_DETECTED={len(gi.map_ramps)} of {len(bases)} authored")
            walk = 0
            for i, ((cx, cy), (ux, uy)) in enumerate(bases):
                lo = Point2((cx - 3 * ux + 0.5, cy - 3 * uy + 0.5))
                hi = Point2((cx + 3 * ux + 0.5, cy + 3 * uy + 0.5))
                d = await self.client.query_pathing(lo, hi)
                walk += d is not None
                print(f"  ramp#{i:>2} base=({cx:.0f},{cy:.0f}) dist={d} WALKABLE={d is not None}")
            print(f"RAMPS_WALKABLE={walk}/{len(bases)}")
            if len(starts) >= 2:
                d = await self.client.query_pathing(Point2(starts[0]), Point2(starts[1]))
                print(f"MAIN_TO_MAIN dist={d} CONNECTED={d is not None}")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get(NAME),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    main()
