"""ABLATION: keep FrostLE terrain 100% intact, but remove most <ramp> entries from t3Terrain.xml's
rampList (keep only a chosen subset). Then probe the ENGINE: which of the 16 original ramp locations
are still walkable? If kept ramps stay walkable and removed ones become blocked -> the rampList entry
is what makes a ramp walkable (terrain gradient alone is not enough), so we CAN author ramps headless.

    PYTHONPATH=src .venv/bin/python scripts/ablation_probe.py build   # keep ramps {0,6}
    PYTHONPATH=src .venv/bin/python scripts/ablation_probe.py probe
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sc2mapgen.export.stormlib_mpq import export_via_stormlib  # noqa: E402

TEMPLATE = "gold_maps/FrostLE.SC2Map"
OUT = Path("outputs/export/ablation_frost.SC2Map")
KEEP = {0, 6}
META = Path("outputs/export/ablation_meta.json")


def _parse_tf(s: str):
    u = re.search(r"u\(([^)]+)\)", s).group(1).split(",")
    c = re.search(r"c=\(([^)]+)\)", s).group(1).split(",")
    return [float(u[0]), float(u[1])], [float(c[0]), float(c[1])]


def build() -> None:
    arch = MPQArchive(TEMPLATE)
    t = arch.read_file("t3Terrain.xml").decode("utf-8", "replace")
    entries = re.findall(r"<ramp [^>]*?/>", t)
    meta = []
    for i, e in enumerate(entries):
        d = dict(re.findall(r'(\w+)="([^"]*)"', e))
        u, c = _parse_tf(d["base"])
        meta.append({"idx": i, "u": u, "c": c, "kept": i in KEEP})
    META.write_text(json.dumps(meta))
    kept = [e for i, e in enumerate(entries) if i in KEEP]
    new_list = f'<rampList num="{len(kept)}">\n' + "\n".join(kept) + "\n</rampList>"
    t2 = re.sub(r"<rampList[^>]*>.*?</rampList>", new_list, t, count=1, flags=re.DOTALL)
    assert t2 != t and f'num="{len(kept)}"' in t2
    export_via_stormlib(TEMPLATE, OUT, {"t3Terrain.xml": t2.encode("utf-8"),
                                        "Objects": arch.read_file("Objects")})
    import shutil
    shutil.copy(OUT, "/Applications/StarCraft II/maps/ablation_frost.SC2Map")
    print(f"kept ramps {sorted(KEEP)} of {len(entries)}; wrote {OUT}")


def probe() -> None:
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.bot_ai_internal import BotAIInternal
    from sc2.data import Difficulty, Race
    from sc2.main import run_game
    from sc2.player import Bot, Computer
    from sc2.position import Point2

    BotAIInternal._find_expansion_locations = lambda self: None
    meta = json.loads(META.read_text())

    class P(BotAI):
        async def on_start(self):
            gi = self.game_info
            print(f"MAP_RAMPS_DETECTED={len(gi.map_ramps)} (kept {sorted(KEEP)})")
            for m in meta:
                cx, cy = m["c"]
                ux, uy = m["u"]
                lo = Point2((cx - 2.5 * ux + 0.5, cy - 2.5 * uy + 0.5))
                hi = Point2((cx + 2.5 * ux + 0.5, cy + 2.5 * uy + 0.5))
                d = await self.client.query_pathing(lo, hi)
                print(f"  ramp#{m['idx']:>2} kept={int(m['kept'])} "
                      f"c=({cx:.0f},{cy:.0f}) dist={d} WALKABLE={d is not None}")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get("ablation_frost"),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    (build if sys.argv[1] == "build" else probe)()
