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
    if os.environ.get("RAMP_RUN"):
        lo, hi = (float(x) for x in os.environ["RAMP_RUN"].split(","))
        cfg.ramp_run_range = (lo, hi)
    if os.environ.get("RAMP_CHOKE"):
        lo, hi = (float(x) for x in os.environ["RAMP_CHOKE"].split(","))
        cfg.ramp_choke_range = (lo, hi)
    g = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
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
            # SOURCE must be a PATHABLE cell: the raw MAIN1 center is often a townhall/cliff cell,
            # and query_pathing from a non-pathable source returns None for EVERY target (a false
            # all-unreachable). Snap to the nearest pathable cell (pathing_grid[y,x] > 0).
            pg = self.game_info.pathing_grid.data_numpy
            sx, sy, bd = m1[1], m1[2], 1 << 30
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    x, y = m1[1] + dx, m1[2] + dy
                    if 0 <= y < pg.shape[0] and 0 <= x < pg.shape[1] and pg[y, x] > 0:
                        d = dx * dx + dy * dy
                        if d < bd:
                            bd, sx, sy = d, x, y
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
            lbl, _ = _ndi.label(pass_)

            def _pcomp(x, y):
                bestc = 0
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < lbl.shape[0] and 0 <= xx < lbl.shape[1] and lbl[yy, xx] > 0:
                            return int(lbl[yy, xx])
                return bestc
            mc = _pcomp(m1[1], m1[2])
            print(f"PATHGRID_CONN main1_comp={mc}")
            for kind, x, y in bases:
                bc = _pcomp(x, y)
                print(f"  [grid] MAIN1 ~ {kind:8} ({x},{y}) comp={bc} SAME={bc == mc and bc > 0}")
            for kind, x, y in bases:
                # Query a small ring around the base center, not just the center: the exact
                # townhall cell is often non-pathable, giving a FALSE 'unreachable' even though the
                # base is connected (verified: unit walked there anyway). REACH iff ANY nearby
                # cell is reachable.
                best = None
                for dx in (0, -4, 4, -6, 6):
                    for dy in (0, -4, 4, -6, 6):
                        d = await self.client.query_pathing(src, Point2((x + dx + 0.5, y + dy + 0.5)))
                        if d is not None and (best is None or d < best):
                            best = d
                same = abs(x - m1[1]) < 2 and abs(y - m1[2]) < 2
                print(f"  MAIN1 -> {kind:8} ({x},{y}) dist={None if best is None else round(best, 1)} "
                      f"REACH={best is not None or same}{' (self)' if same else ''}")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get(NAME),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    main()
