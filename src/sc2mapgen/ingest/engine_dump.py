"""Option B ingestion: load a real .SC2Map in the SC2 engine and dump its ground-truth
terrain grids + placed resources.

We deliberately do NOT decode the MPQ terrain binary ourselves. Instead we launch the
retail SC2 client via python-sc2, let the engine report `pathing_grid`, `placement_grid`,
`terrain_height`, `playable_area`, `start_locations`, and the placed resource units, then
leave the game immediately.

Output (written to ``out_dir``):
    _raw.npz   pathing / placement / height uint8 arrays, shape [y, x]
    _raw.json  scalar metadata + resource / expansion / start-location lists

Run standalone:
    python -m sc2mapgen.ingest.engine_dump "AcropolisAIE" outputs/AcropolisAIE
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _classify(name: str) -> str | None:
    """Map an engine unit *type name* to one of our ResourceKind names, or None to ignore."""
    lname = name.lower()
    if "mineralfield" in lname or lname.startswith("richmineralfield") or "mineral" in lname:
        return "MINERAL"
    if "vespene" in lname or "geyser" in lname:
        return "GEYSER"
    if "watchtower" in lname or "xelnaga" in lname:
        return "WATCHTOWER"
    if (
        "destructible" in lname
        or "rock" in lname
        or "debris" in lname
        or "mineralbarrier" in lname
        or "unbuildable" in lname
    ):
        return "DESTRUCTIBLE"
    return None


def dump_map(map_name: str, out_dir: str | Path) -> Path:
    """Launch SC2 on ``map_name`` and write raw engine data to ``out_dir``.

    Returns the output directory path. Raises on failure to load.
    """
    # Imported lazily so the rest of the package doesn't hard-depend on a running SC2.
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.data import Difficulty, Race
    from sc2.main import run_game
    from sc2.player import Bot, Computer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    class DumpBot(BotAI):
        async def on_step(self, iteration: int):
            if iteration != 0:
                await self.client.leave()
                return

            gi = self.game_info

            pathing = gi.pathing_grid.data_numpy.astype(np.uint8)      # 1 = pathable
            placement = gi.placement_grid.data_numpy.astype(np.uint8)  # 1 = buildable
            height = gi.terrain_height.data_numpy.astype(np.uint8)     # 0..255

            np.savez_compressed(
                out / "_raw.npz",
                pathing=pathing,
                placement=placement,
                height=height,
            )

            # Placed resources / objects. Resolve type names via the running game's
            # game_data (knows every raw id) rather than the static UnitTypeId enum,
            # which can be missing ids from newer map/game patches (e.g. raw id 2046).
            def type_name(unit) -> str:
                raw_id = unit._proto.unit_type
                data = self.game_data.units.get(raw_id)
                return data.name if data is not None else f"Unit{raw_id}"

            resources = []
            for unit in list(self.mineral_field) + list(self.vespene_geyser) + list(
                self.destructables
            ) + list(self.watchtowers):
                name = type_name(unit)
                kind = _classify(name)
                if kind is None:
                    continue
                resources.append(
                    {
                        "kind": kind,
                        "unit_type": name,
                        "x": float(unit.position.x),
                        "y": float(unit.position.y),
                        "amount": int(getattr(unit, "mineral_contents", 0))
                        or int(getattr(unit, "vespene_contents", 0))
                        or None,
                    }
                )

            # Engine-clustered expansion centers (our "resource cluster -> base candidate").
            try:
                expansions = [
                    [float(p.x), float(p.y)] for p in self.expansion_locations_list
                ]
            except Exception as exc:  # noqa: BLE001 - expansions are best-effort
                print(f"[engine_dump] expansion_locations failed: {exc}", file=sys.stderr)
                expansions = []

            pa = gi.playable_area
            meta = {
                "map_name": map_name,
                "width": int(gi.map_size.x),
                "height": int(gi.map_size.y),
                "playable_area": {
                    "x": int(pa.x),
                    "y": int(pa.y),
                    "width": int(pa.width),
                    "height": int(pa.height),
                },
                "start_locations": [
                    [float(p.x), float(p.y)] for p in gi.start_locations
                ],
                "own_start_location": [
                    float(self.start_location.x),
                    float(self.start_location.y),
                ],
                "resources": resources,
                "expansions": expansions,
            }
            (out / "_raw.json").write_text(json.dumps(meta, indent=2))
            print(
                f"[engine_dump] dumped {map_name}: grid {pathing.shape}, "
                f"{len(resources)} resources, {len(expansions)} expansions"
            )
            await self.client.leave()

    run_game(
        maps.get(map_name),
        [Bot(Race.Terran, DumpBot()), Computer(Race.Terran, Difficulty.VeryEasy)],
        realtime=False,
    )

    if not (out / "_raw.npz").exists():
        raise RuntimeError(f"engine dump produced no output for {map_name}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python -m sc2mapgen.ingest.engine_dump <MapName> <out_dir>")
        raise SystemExit(2)
    dump_map(sys.argv[1], sys.argv[2])
