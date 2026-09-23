"""Experiment: can the StarCraft II *Editor* bake walkable ramps into a headless-authored map?

The Galaxy Editor has no headless/CLI mode, so this drives its GUI via AppleScript:
    open the .SC2Map -> wait for load -> Cmd+S (save, which finalizes terrain) -> Cmd+Q.

REQUIREMENTS (macOS):
  * System Settings -> Privacy & Security -> Accessibility: enable the app running this
    (Cursor / Terminal), or `osascript` keystrokes are silently dropped.
  * The machine must be left alone while the editor processes (it takes over focus).

USAGE:
    # 1) export a NON-flat map (real cliffs/ramps for the editor to bake):
    PYTHONPATH=src .venv/bin/python scripts/editor_bake.py export --seed 213
    # 2) drive the editor to open+save it:
    PYTHONPATH=src .venv/bin/python scripts/editor_bake.py bake --seed 213 --load-wait 40 --save-wait 30
    # 3) probe whether the natural now connects to the main across the ramp:
    PYTHONPATH=src .venv/bin/python scripts/editor_bake.py probe --seed 213
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

EDITOR = "StarCraft II Editor"
OUT = Path("outputs/export")
MAPS = Path("/Applications/StarCraft II/maps")


def do_export(seed: int) -> Path:
    from sc2mapgen.export import ExportConfig, export_sc2map
    from sc2mapgen.generate.rasterize import RasterConfig, rasterize
    from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator

    gen = SkeletonGenerator(config=GenConfig())
    ir = rasterize(gen.generate(seed), seed=seed, cfg=RasterConfig())
    name = f"gen_{seed}_relief"
    # flatten_terrain=False -> keep the real multi-level cliffs/ramps for the editor to finalize
    info = export_sc2map(ir, OUT / f"{name}.SC2Map", ExportConfig(flatten_terrain=False))
    dst = MAPS / f"{name}.SC2Map"
    subprocess.run(["cp", info["out_path"], str(dst)], check=True)
    print(f"exported non-flat {dst} (ramps_authored={info['n_ramps_authored']}, offset={info['offset']})")
    return dst


def _osa(script: str) -> None:
    subprocess.run(["osascript", "-e", script], check=False)


def do_bake(seed: int, load_wait: float, save_wait: float) -> None:
    mp = MAPS / f"gen_{seed}_relief.SC2Map"
    if not mp.exists():
        raise SystemExit(f"{mp} missing - run `export` first")
    print(f"opening {mp} in the editor ...")
    subprocess.run(["open", "-a", EDITOR, str(mp)], check=True)
    print(f"waiting {load_wait}s for load ...")
    time.sleep(load_wait)
    # focus editor, save (finalizes terrain), wait, quit
    _osa(f'tell application "{EDITOR}" to activate')
    time.sleep(2)
    _osa('tell application "System Events" to keystroke "s" using command down')
    print(f"sent Cmd+S; waiting {save_wait}s for terrain finalize+save ...")
    time.sleep(save_wait)
    # dismiss any dialog with Return (e.g. overwrite / keep format), then quit
    _osa('tell application "System Events" to key code 36')  # Return
    time.sleep(2)
    _osa(f'tell application "{EDITOR}" to quit')
    time.sleep(3)
    _osa('tell application "System Events" to key code 36')  # confirm quit if prompted
    print("done. now run: probe")


def do_probe(seed: int) -> None:
    import numpy as np
    from scipy import ndimage
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.bot_ai_internal import BotAIInternal
    from sc2.data import Difficulty, Race
    from sc2.main import run_game
    from sc2.player import Bot, Computer

    from sc2mapgen.generate.rasterize import RasterConfig, rasterize
    from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator

    BotAIInternal._find_expansion_locations = lambda self: None
    gen = SkeletonGenerator(config=GenConfig())
    ir = rasterize(gen.generate(seed), seed=seed, cfg=RasterConfig())

    class P(BotAI):
        async def on_start(self):
            path = self.game_info.pathing_grid.data_numpy
            lbl, n = ndimage.label(path)
            # recover offset from a main start vs our IR main
            sl = [(s.x, s.y) for s in self.game_info.start_locations]
            mains = [b for b in ir.bases if b.kind.value == "MAIN"]
            nats = [b for b in ir.bases if b.kind.value == "NATURAL"]

            def near(bx, by):
                # brute best offset by matching a main to a start location
                return None

            # infer offset from closest start-loc to any main (integer)
            best = None
            for (sx, sy) in sl:
                for m in mains:
                    ox, oy = round(sx - 0.5 - m.x), round(sy - 0.5 - m.y)
                    if best is None:
                        best = (ox, oy)
            ox, oy = best or (0, 0)

            def cid(x, y):
                x, y = int(round(x)) + ox, int(round(y)) + oy
                if 0 <= y < path.shape[0] and 0 <= x < path.shape[1] and path[y, x]:
                    return int(lbl[y, x])
                ys, xs = np.where(path > 0)
                i = np.argmin((ys - y) ** 2 + (xs - x) ** 2)
                return int(lbl[ys[i], xs[i]])

            print(f"offset~({ox},{oy}) components={n} pathable={int(path.sum())}")
            for m in mains:
                nn = min(nats, key=lambda b: (b.x - m.x) ** 2 + (b.y - m.y) ** 2)
                a, b = cid(m.x, m.y), cid(nn.x, nn.y)
                print(f"  main({m.x:.0f},{m.y:.0f})comp={a}  nat({nn.x:.0f},{nn.y:.0f})comp={b}  "
                      f"RAMP_WALKABLE={a == b}")
            await self.client.leave()

        async def on_step(self, it):
            pass

    run_game(maps.get(f"gen_{seed}_relief"),
             [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "bake", "probe"])
    ap.add_argument("--seed", type=int, default=213)
    ap.add_argument("--load-wait", type=float, default=40)
    ap.add_argument("--save-wait", type=float, default=30)
    a = ap.parse_args()
    if a.cmd == "export":
        do_export(a.seed)
    elif a.cmd == "bake":
        do_bake(a.seed, a.load_wait, a.save_wait)
    else:
        do_probe(a.seed)


if __name__ == "__main__":
    main()
