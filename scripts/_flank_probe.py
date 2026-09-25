"""In-engine check of plateau (un-voided) ramp flanks for one seed.

Reports detected vs authored ramps, MAIN1 reachability to every base, and a side-entry test: for
each ramp, the engine path distance from a low-plateau flank cell beside a mid sub-level to the
high plateau just past the top exit, against the offline shortest path (which may only enter the
ramp at its bottom). An engine distance far below the offline one means units climb the flank.

    SC2PATH="/Applications/StarCraft II" PYTHONPATH=src .venv/bin/python scripts/_flank_probe.py 0
"""
from __future__ import annotations

import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NAME = f"gen_{SEED}"


def offline_dist(cliff, rid, src, max_step=8):
    """4-connected BFS distance under engine_components' step rule."""
    h, w = cliff.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    sx, sy = src
    dist[sy, sx] = 0
    q = deque([src])
    while q:
        x, y = q.popleft()
        a = int(cliff[y, x])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h) or dist[ny, nx] >= 0 or cliff[ny, nx] == 0:
                continue
            b = int(cliff[ny, nx])
            d = abs(a - b)
            hi, lo = max(a, b), min(a, b)
            ok = d <= max_step or (d == 2 * max_step and hi % 64 == 0 and lo % 64 != 0)
            ok = ok or (rid[y, x] > 0 and rid[y, x] == rid[ny, nx])
            if ok:
                dist[ny, nx] = dist[y, x] + 1
                q.append((nx, ny))
    return dist


def _old_channel_ramp_flanks(cliff_cell, ramp_mask, max_step=8):
    """The previous rule: void every plateau flank cell off by >8, except the +16 top exit."""
    ch, cw = cliff_cell.shape
    d = cliff_cell.astype(np.int32)
    walk = cliff_cell > 0
    towall = np.zeros((ch, cw), dtype=bool)
    for y, x in zip(*np.where(ramp_mask & walk)):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < ch and 0 <= nx < cw and walk[ny, nx] and not ramp_mask[ny, nx]:
                delta = int(d[ny, nx]) - int(d[y, x])
                if abs(delta) > max_step and delta != 2 * max_step:
                    towall[ny, nx] = True
    cliff_cell[towall] = 0
    return int(towall.sum())


def main() -> None:
    import os

    from sc2mapgen.export import ExportConfig, export_sc2map, sc2map
    from sc2mapgen.generate.rasterize import (
        RasterConfig, cardinal_ramp_bridge, engine_cliff_grid, rasterize,
    )
    from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator

    g = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
    mapir = rasterize(g.generate(SEED), seed=SEED, cfg=RasterConfig())
    info = export_sc2map(mapir, f"outputs/export/{NAME}.SC2Map", ExportConfig(author_ramps=True))
    ox, oy = info["offset"]
    shutil.copy(f"outputs/export/{NAME}.SC2Map", f"/Applications/StarCraft II/maps/{NAME}.SC2Map")

    cliff = engine_cliff_grid(mapir.walkable, mapir.elevation, mapir.ramps)
    h, w = cliff.shape
    rid = cardinal_ramp_bridge(mapir.ramps, cliff.shape)
    rid = rid if rid is not None else np.zeros((h, w), dtype=np.int32)
    ramp_any = np.zeros((h, w), dtype=bool)
    for r in mapir.ramps:
        for x, y in r.cells:
            ramp_any[int(y), int(x)] = True

    # per ramp: (flank cell, target cell, offline distance), IR coords
    tests = []
    for i, r in enumerate(mapir.ramps):
        lo_c, hi_c = (int(r.low_level) + 1) * 64, (int(r.high_level) + 1) * 64
        cells = {(int(x), int(y)) for x, y in r.cells}
        flank = target = None
        for (x, y) in sorted(cells):
            v = int(cliff[y, x])
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h) or ramp_any[ny, nx]:
                    continue
                nv = int(cliff[ny, nx])
                fx, fy = nx + 2 * dx, ny + 2 * dy
                if (flank is None and nv == lo_c and lo_c + 24 <= v <= hi_c - 24
                        and 0 <= fx < w and 0 <= fy < h and cliff[fy, fx] == lo_c
                        and cliff[ny + dy, nx + dx] == lo_c):
                    flank = (fx, fy)
                if (target is None and v >= hi_c - 16 and nv == hi_c
                        and 0 <= fx < w and 0 <= fy < h and cliff[fy, fx] == hi_c):
                    target = (fx, fy)
        if flank is None or target is None:
            tests.append((i, flank, target, None))
            continue
        d = offline_dist(cliff, rid, flank)[target[1], target[0]]
        tests.append((i, flank, target, int(d)))

    if os.environ.get("OLD_FLANKS"):
        sc2map.channel_ramp_flanks = _old_channel_ramp_flanks
    info = export_sc2map(mapir, f"outputs/export/{NAME}.SC2Map", ExportConfig(author_ramps=True))
    ox, oy = info["offset"]
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
    m1 = next(b for b in bases if b[0] == "MAIN")

    class P(BotAI):
        async def on_start(self):
            pg = self.game_info.pathing_grid.data_numpy

            def query_cell(kind, x, y):
                if kind != "MAIN":
                    return x, y
                for dx, dy in ((0, -3), (0, 3), (-3, 0), (3, 0)):
                    if pg[y + dy, x + dx] > 0:
                        return x + dx, y + dy
                return x, y

            print(f"RESULT seed={SEED} authored={len(mapir.ramps)} "
                  f"detected={len(self.game_info.map_ramps)}")
            sx, sy = query_cell(*m1)
            src = Point2((sx + 0.5, sy + 0.5))
            n_reach = 0
            for kind, x, y in bases:
                qx, qy = query_cell(kind, x, y)
                d = 0.0 if (qx, qy) == (sx, sy) else await self.client.query_pathing(
                    src, Point2((qx + 0.5, qy + 0.5)))
                n_reach += d is not None
                if d is None:
                    print(f"RESULT   UNREACHABLE {kind} ({x},{y})")
            print(f"RESULT reach={n_reach}/{len(bases)}")
            for i, flank, target, doff in tests:
                if doff is None:
                    print(f"RESULT   ramp{i}: no flank/target test (flank={flank} target={target})")
                    continue
                fx, fy = flank[0] + ox, flank[1] + oy
                tx, ty = target[0] + ox, target[1] + oy
                de = await self.client.query_pathing(
                    Point2((fx + 0.5, fy + 0.5)), Point2((tx + 0.5, ty + 0.5)))
                flag = "SIDE-ENTRY?" if de is not None and doff > 0 and de < 0.6 * doff else "ok"
                print(f"RESULT   ramp{i}: flank=({fx},{fy}) pg={int(pg[fy, fx])} "
                      f"target=({tx},{ty}) engine={None if de is None else round(de, 1)} "
                      f"offline4={doff} {flag}")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get(NAME),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    main()
