"""Deterministic SC2 exporter: normalized ``MapIR`` -> StarCraft II ``.SC2Map``.

Compiles a ``MapIR`` into a real ``.SC2Map`` (an MPQ archive) by editing a known-good ladder
map in-place as a **template shell** (per the design spec's "template / components" strategy).
The container is never built from scratch -- a hand-rolled MPQ is structurally valid but the SC2
client silently rejects it, so we overwrite only the handful of internal files we author with
StormLib (which preserves the header/hash-table/encryption invariants the client validates) and
leave everything else byte-identical. All findings below are grounded in
``docs/SC2_MAP_FORMAT_FINDINGS.md`` (decoded from real maps + verified in-engine with ``python-sc2``).

What we AUTHOR (overwrite):
  * ``t3SyncCliffLevel`` (CLIF)   -- discrete cliff/level grid; 0 = void (§3a).
  * ``t3SyncHeightMap`` (SMAP)    -- gameplay heightmap; drives ``terrain_height`` (§3b).
  * ``t3HeightMap`` (HMAP)        -- render heightmap; cosmetic (§3c).
  * ``t3SyncTextureInfo`` (RTXT)  -- per-cell base texture index (§6a).
  * ``t3TextureMasks`` (MASK)     -- the actual render texturing, one solid texture per cell (§6b/§6d).
  * ``t3Terrain.xml``             -- ``<rampList>`` is AUTHORITATIVE for ramp walkability (§4/§5);
                                     zeroed on flat maps to kill ghost cliffs.
  * ``Objects`` (XML)             -- start locations + mineral/geyser layout (§8).
  * ``CellAttribute_Pnp``         -- painted pathing/placement bitfield (relief only; §10/§13g).
  * ``t3CellFlags`` (LFCT)        -- cleared so inherited world-holes don't render as void (§10).

What we INHERIT verbatim: ``MapInfo`` (patching the playable area breaks loading -- we do
playable-aware template selection + centering instead, §7), ``t3FluffDoodad``, ``t3VertCol``,
``t3Water``, the ``Base.SC2Data`` game-data catalogs (touching them flips the map Melee->Arcade,
§13a), doodad ``.m3`` models, etc.

Two export modes (``ExportConfig``):
  * **Flat (default, ``flatten_terrain=True``)** -- collapse every walkable cell onto one tier, weld
    the cross-level void gaps, and strip render relief. This is the reliably-playable path: it loads,
    is fully walkable, and every base sits on buildable ground. Terrain is still textured by the
    IR's *intended* level so tiers stay visually distinct (§6e).
  * **Relief (``author_ramps=True``)** -- real multi-level terrain with walkable ramps. This IS
    authorable fully headless (no Galaxy Editor bake): author a per-ramp diagonal/axis cliff gradient
    (+8/cell) AND emit a matching ``<rampList>`` entry -- the piece that actually makes the engine
    treat a cliff boundary as crossable (§4). Correctness is gated on the engine-accurate connectivity
    metric, but per-seed yield of fully-traversable relief is still low, so flat remains the default.

The displayed map name is NOT in ``MapInfo`` -- it is the localized ``DocInfo/Name`` game-string in
``DocumentHeader`` (binary) and each locale's ``GameStrings.txt``. Both are decoupled from the
MapInfo bounds trap (§7), so we rename the export to the IR's ``map_name`` (``ExportConfig.rename_map``,
on by default) instead of shipping the template's name -- which also avoids sharing the ladder
template's asset cache (§2).
"""

from sc2mapgen.export import sc2map
from sc2mapgen.export.exporter import (
    ExportConfig,
    TemplateFitError,
    export_sc2map,
    pick_template,
    verify_export,
)

__all__ = ["ExportConfig", "TemplateFitError", "export_sc2map", "pick_template", "verify_export",
           "sc2map"]
