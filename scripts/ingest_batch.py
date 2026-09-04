"""Batch ingestion + verification harness.

Runs the engine-dump -> MapIR -> preview pipeline over many maps, tolerating per-map
failures (validate-on-load: a map that won't open or dumps degenerate data is skipped
and recorded, not fatal). Serves the 5 -> 20 -> all ramp from the design plan.

    # explicit list
    python scripts/ingest_batch.py AcropolisAIE AbyssalReefAIE "(4)DarknessSanctuaryLE"

    # everything in gold_maps/
    python scripts/ingest_batch.py --all

    # rebuild MapIR+preview from existing dumps (no SC2 relaunch)
    python scripts/ingest_batch.py --all --skip-dump

Outputs:
    dataset/<map>/{_raw.*, terrain.npy, map.json, preview.png}
    dataset/_summary.json      per-map metrics + status
    dataset/_contact_sheet.png montage of previews for quick visual verification
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sc2mapgen.ingest.engine_dump import dump_map
from sc2mapgen.ingest.preview import render
from sc2mapgen.ingest.to_mapir import build_mapir

CORPUS = Path("gold_maps")
DATASET = Path("dataset")


def map_names_from_corpus() -> list[str]:
    return sorted(p.name[: -len(".SC2Map")] for p in CORPUS.glob("*.SC2Map"))


def ingest_one(name: str, skip_dump: bool) -> dict:
    out_dir = DATASET / name
    rec: dict = {"map_name": name, "status": "ok"}
    try:
        if not skip_dump or not (out_dir / "_raw.npz").exists():
            dump_map(name, out_dir)
        mapir = build_mapir(out_dir)
        mapir.save(out_dir)
        render(mapir, out_dir / "preview.png")

        kinds = {"MAIN": 0, "NATURAL": 0, "BASE": 0}
        for b in mapir.bases:
            kinds[b.kind.value] += 1
        rec.update(
            width=mapir.width,
            height=mapir.height,
            n_bases=len(mapir.bases),
            n_mains=kinds["MAIN"],
            n_naturals=kinds["NATURAL"],
            n_resources=len(mapir.resources),
            n_ramps=len(mapir.ramps),
            n_starts=len(mapir.start_locations),
        )
        # cheap sanity flags (not fatal): expect symmetric mains/naturals for ladder maps
        rec["warn"] = []
        if kinds["MAIN"] < 2:
            rec["warn"].append("fewer_than_2_mains")
        if kinds["MAIN"] != kinds["NATURAL"]:
            rec["warn"].append("mains_ne_naturals")
    except Exception as exc:  # noqa: BLE001 - per-map isolation is the point
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["trace"] = traceback.format_exc()[-800:]
        print(f"[batch] FAILED {name}: {rec['error']}", file=sys.stderr)
    return rec


def contact_sheet(records: list[dict], out: Path) -> None:
    ok = [r for r in records if r["status"] == "ok" and (DATASET / r["map_name"] / "preview.png").exists()]
    if not ok:
        return
    n = len(ok)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, ok):
        img = mpimg.imread(DATASET / r["map_name"] / "preview.png")
        ax.imshow(img)
        warn = ("  ⚠" + ",".join(r.get("warn", []))) if r.get("warn") else ""
        ax.set_title(f"{r['map_name']}\n{r['n_bases']}b {r['n_mains']}M/{r['n_naturals']}N{warn}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", nargs="*", help="map stems (omit with --all)")
    ap.add_argument("--all", action="store_true", help="ingest every map in gold_maps/")
    ap.add_argument("--skip-dump", action="store_true", help="reuse existing _raw.* dumps")
    args = ap.parse_args()

    names = map_names_from_corpus() if args.all else args.maps
    if not names:
        ap.error("provide map names or --all")

    DATASET.mkdir(parents=True, exist_ok=True)
    records = []
    for i, name in enumerate(names, 1):
        print(f"[batch] ({i}/{len(names)}) {name}")
        records.append(ingest_one(name, args.skip_dump))

    (DATASET / "_summary.json").write_text(json.dumps(records, indent=2))
    contact_sheet(records, DATASET / "_contact_sheet.png")

    ok = sum(1 for r in records if r["status"] == "ok")
    warned = sum(1 for r in records if r.get("warn"))
    print(f"\n[batch] done: {ok}/{len(records)} ok, {warned} with warnings")
    for r in records:
        if r["status"] != "ok":
            print(f"  ERROR {r['map_name']}: {r.get('error')}")
        elif r.get("warn"):
            print(f"  warn  {r['map_name']}: {','.join(r['warn'])}")
    print(f"[batch] summary: {DATASET/'_summary.json'}  contact sheet: {DATASET/'_contact_sheet.png'}")


if __name__ == "__main__":
    main()
