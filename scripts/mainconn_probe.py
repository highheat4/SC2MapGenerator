"""Localize a relief connectivity break: from MAIN 1's start, ask the engine to route a ground unit
to MAIN 2, to the map center, and to every base center. Prints reachability so we can see WHICH link
is broken (not just per-ramp local walkability).

    PYTHONPATH=src .venv/bin/python scripts/mainconn_probe.py 213
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 213
NAME = f"gen_{SEED}"


def main() -> None:
    from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator
    from sc2mapgen.generate.rasterize import RasterConfig, rasterize
    from sc2mapgen.export import ExportConfig, export_sc2map

    import os
    cfg = RasterConfig()
    gcfg = GenConfig(symmetries=("rot180",))
    if os.environ.get("RAMP_RUN"):
        lo, hi = (float(x) for x in os.environ["RAMP_RUN"].split(","))
        gcfg.ramp_run_range = (lo, hi)
    if os.environ.get("RAMP_CHOKE"):
        lo, hi = (float(x) for x in os.environ["RAMP_CHOKE"].split(","))
        gcfg.ramp_width_range = (lo, hi)
    g = SkeletonGenerator(config=gcfg)
    mapir = rasterize(g.generate(SEED), seed=SEED, cfg=cfg)
    info = export_sc2map(mapir, f"outputs/export/{NAME}.SC2Map", ExportConfig(author_ramps=True))
    ox, oy = info["offset"]
    import shutil
    shutil.copy(f"outputs/export/{NAME}.SC2Map", f"/Applications/StarCraft II/maps/{NAME}.SC2Map")

    bases = [(b.kind.name, int(b.x) + ox, int(b.y) + oy) for b in mapir.bases]

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
            # Every base centre is a standardized townhall cell inside the immutable, buildable
            # base core, so it is probed exactly. The one exception is a MAIN, whose centre is
            # covered by the start townhall the engine places there (pathing 0): its query point
            # is the first pathable cell just outside that 5x5 footprint, still inside the core.
            pg = self.game_info.pathing_grid.data_numpy

            def query_cell(kind, x, y):
                if kind != "MAIN":
                    return x, y
                for dx, dy in ((0, -3), (0, 3), (-3, 0), (3, 0)):
                    if pg[y + dy, x + dx] > 0:
                        return x + dx, y + dy
                return x, y

            sx, sy = query_cell(*m1)
            src = Point2((sx + 0.5, sy + 0.5))
            print(f"MAP_RAMPS_DETECTED={len(self.game_info.map_ramps)} src=({sx},{sy})")

            # AUTHORITATIVE engine connectivity: flood the engine's OWN pathing_grid, adding its
            # detected ramp cells as passable (pathing_grid marks ramp cells 0, same as walls --
            # findings §4a). This uses only engine-derived data (no query_pathing source/target
            # artifacts), so base-component membership here is ground truth for terrain traversal.
            import numpy as _np
            from scipy import ndimage as _ndi
            pass_ = pg > 0
            for rp in self.game_info.map_ramps:
                for (rx, ry) in rp.points:
                    ix, iy = int(rx), int(ry)
                    if 0 <= iy < pass_.shape[0] and 0 <= ix < pass_.shape[1]:
                        pass_[iy, ix] = True
            # the start townhalls are units, not terrain: open their footprints for the flood
            for kind, x, y in bases:
                if kind == "MAIN":
                    pass_[y - 2:y + 3, x - 2:x + 3] = True
            lbl, _ = _ndi.label(pass_)

            def _pcomp(x, y):
                return int(lbl[y, x])
            mc = _pcomp(m1[1], m1[2])
            print(f"PATHGRID_CONN main1_comp={mc}")
            for kind, x, y in bases:
                bc = _pcomp(x, y)
                print(f"  [grid] MAIN1 ~ {kind:8} ({x},{y}) comp={bc} SAME={bc == mc and bc > 0}")
            for kind, x, y in bases:
                qx, qy = query_cell(kind, x, y)
                same = (qx, qy) == (sx, sy)
                d = None if same else await self.client.query_pathing(
                    src, Point2((qx + 0.5, qy + 0.5)))
                print(f"  MAIN1 -> {kind:8} ({x},{y}) q=({qx},{qy}) "
                      f"pg={int(pg[qy, qx])} dist={None if d is None else round(d, 1)} "
                      f"REACH={d is not None or same}{' (self)' if same else ''}")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get(NAME),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    main()
