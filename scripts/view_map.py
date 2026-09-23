"""Export a generated map and open it in StarCraft II so you can inspect it yourself.

    # export relief (multi-level + ramps) seed 234, copy into SC2, render a preview PNG:
    PYTHONPATH=src .venv/bin/python scripts/view_map.py 234

    # flat interim instead of relief:
    PYTHONPATH=src .venv/bin/python scripts/view_map.py 234 --flat

    # also LAUNCH the map in realtime so you can walk a unit around and test ramps:
    PYTHONPATH=src .venv/bin/python scripts/view_map.py 234 --play

Then to inspect the TERRAIN visually, open the copied map in the StarCraft II Editor:
    /Applications/StarCraft II/Support/../<Editor>   (File > Open, pick gen_<seed>.SC2Map)
or just use --play to spectate/scout it live.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator  # noqa: E402
from sc2mapgen.generate.rasterize import RasterConfig, rasterize  # noqa: E402
from sc2mapgen.generate.validate import validate_map  # noqa: E402
from sc2mapgen.export import ExportConfig, export_sc2map  # noqa: E402
from sc2mapgen.ingest.preview import render  # noqa: E402

SC2_MAPS = Path("/Applications/StarCraft II/maps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int, nargs="?", default=234)
    ap.add_argument("--flat", action="store_true", help="flat interim export (default: relief)")
    ap.add_argument("--play", action="store_true", help="launch the map in realtime to walk it")
    args = ap.parse_args()

    seed = args.seed
    name = f"gen_{seed}"
    g = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
    mapir = rasterize(g.generate(seed), seed=seed, cfg=RasterConfig())
    rep = validate_map(mapir)
    print(f"{name}: {mapir.width}x{mapir.height}  M5={'VALID' if rep.ok else 'INVALID'} "
          f"ramps={len(mapir.ramps)}  bases={len([b for b in mapir.bases])}")
    for f in rep.failures[:6]:
        print(f"   FAIL: {f}")

    out = Path("outputs/export") / f"{name}.SC2Map"
    info = export_sc2map(mapir, out, ExportConfig(author_ramps=not args.flat))
    print(f"exported -> {out}  (template={Path(info['template']).name}, "
          f"ramps_authored={info['n_ramps_authored']}, mode={'flat' if args.flat else 'relief'})")

    # top-down preview PNG (walkable / elevation / ramps / bases)
    png = render(mapir, out.with_suffix(".png"))
    print(f"preview  -> {png}")

    if SC2_MAPS.is_dir():
        dst = SC2_MAPS / f"{name}.SC2Map"
        shutil.copy(out, dst)
        print(f"copied   -> {dst}")
        print("Open it in the StarCraft II Editor (File > Open Map...) to inspect terrain/ramps.")
    else:
        print(f"(SC2 maps folder not found at {SC2_MAPS}; open {out} in the Editor manually)")

    if args.play:
        from sc2 import maps
        from sc2.bot_ai import BotAI
        from sc2.bot_ai_internal import BotAIInternal
        from sc2.data import Difficulty, Race
        from sc2.main import run_game
        from sc2.player import Bot, Computer

        BotAIInternal._find_expansion_locations = lambda self: None

        class Watch(BotAI):
            async def on_start(self):
                print(f"IN-GAME: map_ramps detected by engine = {len(self.game_info.map_ramps)}")

            async def on_step(self, it):
                pass

        print("launching SC2 in realtime -- scout with your worker; close the game window to exit.")
        run_game(maps.get(name),
                 [Bot(Race.Terran, Watch()), Computer(Race.Terran, Difficulty.VeryEasy)],
                 realtime=True)


if __name__ == "__main__":
    main()
