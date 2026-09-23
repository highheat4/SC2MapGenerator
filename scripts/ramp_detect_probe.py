"""For one seed, dump every AUTHORED ramp and whether the ENGINE actually detected a ramp there.

An authored ramp with no nearby ``game_info.map_ramps`` entry is an UNDETECTED ramp == a walled
cliff line (findings §4.6): the offline oracle can't see this (it models CLIF connectivity, not
detection). Pinpoints WHICH crossing seals a mirror half.

    SC2PATH="/Applications/StarCraft II" PYTHONPATH=src .venv/bin/python scripts/ramp_detect_probe.py 42
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
NAME = f"gen_{SEED}"


def main() -> None:
    from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator
    from sc2mapgen.generate.rasterize import RasterConfig, rasterize
    from sc2mapgen.export import ExportConfig, export_sc2map
    from sc2mapgen.export.sc2map import _snap_dir, _RAMP_DIR_U

    g = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
    mapir = rasterize(g.generate(SEED), seed=SEED, cfg=RasterConfig())
    info = export_sc2map(mapir, f"outputs/export/{NAME}.SC2Map", ExportConfig(author_ramps=True))
    ox, oy = info["offset"]
    import shutil
    shutil.copy(f"outputs/export/{NAME}.SC2Map", f"/Applications/StarCraft II/maps/{NAME}.SC2Map")

    authored = []
    for r in mapir.ramps:
        if r.top is None or r.bottom is None:
            continue
        ux, uy = (r.top[0] - r.bottom[0]), (r.top[1] - r.bottom[1])
        nrm = math.hypot(ux, uy) or 1.0
        d = _snap_dir(ux / nrm, uy / nrm)
        # HIGH-end cell (where the quad sits) in template coords
        cells_t = [(cx + ox, cy + oy) for (cx, cy) in r.cells]
        dux, duy = _RAMP_DIR_U[d]
        hx, hy = max(cells_t, key=lambda c: c[0] * dux + c[1] * duy)
        authored.append((hx, hy, d, int(r.low_level), int(r.high_level), len(r.cells)))

    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.bot_ai_internal import BotAIInternal
    from sc2.data import Difficulty, Race
    from sc2.main import run_game
    from sc2.player import Bot, Computer

    BotAIInternal._find_expansion_locations = lambda self: None

    class P(BotAI):
        async def on_start(self):
            det = []
            for rp in self.game_info.map_ramps:
                try:
                    cx, cy = rp.top_center
                    det.append((float(cx), float(cy)))
                except Exception:  # noqa: BLE001
                    pass
            print(f"AUTHORED={len(authored)} DETECTED={len(self.game_info.map_ramps)}")
            for (hx, hy, d, lo, hi, nc) in authored:
                # nearest detected ramp to this authored high-end
                best = min((math.hypot(hx - dx, hy - dy) for dx, dy in det), default=1e9)
                flag = "OK " if best <= 5.0 else "MISS"
                print(f"  {flag} authored high=({hx},{hy}) dir={d} {lo}->{hi} cells={nc} "
                      f"nearest_detected={best:.1f}")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get(NAME),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    main()
