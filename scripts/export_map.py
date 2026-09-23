"""Milestone 7 driver: generate -> validate -> export a fresh .SC2Map, and verify it.

    python scripts/export_map.py --seed 200 --count 5
    python scripts/export_map.py --seed 213 --template gold_maps/CatalystLE.SC2Map --components

For each seed it builds a skeleton, rasterizes it, runs the M5 validator, compiles it into a
.SC2Map on a real template shell, and re-opens the archive with an independent reader to
confirm the authored layers survived. Invalid maps are still exported (flagged) so the output
can be inspected; the per-match loop (M8) is where rejection/retry will live.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc2mapgen.export import ExportConfig, export_sc2map  # noqa: E402
from sc2mapgen.export.exporter import verify_export  # noqa: E402
from sc2mapgen.generate.rasterize import RasterConfig, rasterize  # noqa: E402
from sc2mapgen.generate.skeleton import SYMMETRIES, GenConfig, SkeletonGenerator  # noqa: E402
from sc2mapgen.generate.validate import validate_map  # noqa: E402

OUT = Path("outputs/export")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--weirdness", type=float, default=0.0)
    ap.add_argument("--symmetry", default="rot180", choices=[*SYMMETRIES, "mixed"])
    ap.add_argument("--template", default=None, help="explicit template .SC2Map (else auto-pick)")
    ap.add_argument("--templates-dir", default="gold_maps")
    ap.add_argument("--components", action="store_true",
                    help="also emit the authored files as a .SC2Components folder")
    ap.add_argument("--author-ramps", action="store_true",
                    help="author real multi-level terrain + walkable ramps (else flat interim)")
    ap.add_argument("--valid-only", action="store_true",
                    help="scan seeds and export ONLY M5-valid maps (engine-traversable) until "
                         "--count are found or --max-attempts is hit")
    ap.add_argument("--max-attempts", type=int, default=500,
                    help="seed budget for --valid-only search")
    args = ap.parse_args()

    syms = tuple(SYMMETRIES) if args.symmetry == "mixed" else (args.symmetry,)
    gen = SkeletonGenerator(config=GenConfig(symmetries=syms))
    rcfg = RasterConfig(weirdness=args.weirdness)
    ecfg = ExportConfig(templates_dir=args.templates_dir, template=args.template,
                        emit_components=args.components, author_ramps=args.author_ramps)
    OUT.mkdir(parents=True, exist_ok=True)

    # In --valid-only mode we search the seed line for M5-valid (== engine-traversable) maps and
    # export only those; otherwise we export the first --count seeds regardless (inspection mode).
    if args.valid_only:
        seeds, attempts = [], 0
        while len(seeds) < args.count and attempts < args.max_attempts:
            s = args.seed + attempts
            attempts += 1
            mapir = rasterize(gen.generate(s), seed=s, cfg=rcfg)
            if validate_map(mapir).ok:
                seeds.append(s)
        print(f"[m7] --valid-only: found {len(seeds)}/{args.count} traversable map(s) "
              f"in {attempts} attempt(s): seeds={seeds}")
    else:
        seeds = [args.seed + i for i in range(args.count)]

    exported_ok = valid_ok = 0
    for seed in seeds:
        mapir = rasterize(gen.generate(seed), seed=seed, cfg=rcfg)
        rep = validate_map(mapir)
        valid_ok += rep.ok

        try:
            info = export_sc2map(mapir, OUT / f"{mapir.map_name}.SC2Map", ecfg)
            ok, probs = verify_export(info["out_path"], mapir)
            exported_ok += ok
            tpl = os.path.basename(info["template"])
            vtag = "VALID  " if rep.ok else "INVALID"
            etag = "verified" if ok else f"VERIFY-FAIL {probs}"
            print(f"[m7] seed={seed}: {vtag} {mapir.width}x{mapir.height} -> {tpl}"
                  f"{info['template_dims']} off{info['offset']} "
                  f"starts={info['n_start_locations']} res={info['n_resources']} "
                  f"ramps={info['n_ramps_authored']} "
                  f"{info['size_bytes']//1024}KB  {etag}")
        except Exception as exc:  # noqa: BLE001
            print(f"[m7] seed={seed}: EXPORT ERROR {type(exc).__name__}: {exc}")

    if len(seeds) > 1 or args.valid_only:
        n = len(seeds)
        print(f"[m7] {exported_ok}/{n} exported+verified, "
              f"{valid_ok}/{n} passed M5 validation; output in {OUT}/")


if __name__ == "__main__":
    main()
