"""Milestone 1 spike driver: ingest ONE .SC2Map into the normalized artifacts.

    python scripts/ingest_map.py "AcropolisAIE"
    python scripts/ingest_map.py "AcropolisAIE" --out outputs/AcropolisAIE

Produces in the output dir:
    _raw.npz / _raw.json   (verbatim engine dump)
    terrain.npy            (stacked walkable/buildable/elevation channels)
    map.json               (MapIR metadata: bases, resources, ramps, starts)
    preview.png            (visual sanity check)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make src/ importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc2mapgen.ingest.engine_dump import dump_map
from sc2mapgen.ingest.preview import render
from sc2mapgen.ingest.to_mapir import build_mapir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("map_name", help="map stem, e.g. AcropolisAIE (no .SC2Map)")
    ap.add_argument("--out", default=None, help="output dir (default outputs/<map_name>)")
    ap.add_argument(
        "--skip-dump",
        action="store_true",
        help="reuse existing _raw.* and only rebuild MapIR + preview",
    )
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / args.map_name

    if not args.skip_dump:
        print(f"[ingest] launching SC2 engine dump for {args.map_name} ...")
        dump_map(args.map_name, out_dir)

    print("[ingest] normalizing raw dump -> MapIR ...")
    mapir = build_mapir(out_dir)
    mapir.save(out_dir)

    print("[ingest] rendering preview ...")
    png = render(mapir, out_dir / "preview.png")

    print(
        f"[ingest] done: {mapir.map_name} "
        f"({mapir.width}x{mapir.height}) "
        f"bases={len(mapir.bases)} resources={len(mapir.resources)} ramps={len(mapir.ramps)}"
    )
    print(f"[ingest] artifacts in {out_dir}/  (preview: {png})")


if __name__ == "__main__":
    main()
