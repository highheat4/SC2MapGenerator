# SC2 `.SC2Map` Internals — Reference for Headless Authoring

What we know about the StarCraft II map format, as needed by the exporter (`src/sc2mapgen/export/`).
Everything here was established by decoding the gold maps (`gold_maps/`, `dataset/*/_raw.npz`, 106
maps), round-tripping our own exports, and probing them in-engine with `python-sc2`. Byte offsets are
little-endian. "Cell" = one game tile; "vertex" = cell corner (vertex grids are cells+1 per side).

References: SC2Mapster wiki (<https://sc2mapster.wiki.gg/wiki/File_Formats/Maps>, behind Cloudflare,
often wrong in details) and S2 Editor Guides (<https://s2editor-guides.readthedocs.io>, good for the
editor's conceptual model). Always confirm by round-tripping a real map.

---

## 1. Container: edit a real map in place, never write an MPQ from scratch

- `.SC2Map` is an MPQ archive of internal files addressed by path (`t3SyncCliffLevel`,
  `Base.SC2Data\GameData\...`). Read with `mpyq`; write with StormLib
  (`export/stormlib_mpq.py`, `ctypes`).
- A from-scratch MPQ is structurally valid (readers round-trip it) but the **SC2 client silently
  rejects it**. StormLib in-place editing preserves the header/hash-table/encryption invariants the
  client checks.
- So the exporter takes a real ladder map as a **template shell**, overwrites only the files below,
  and leaves everything else byte-identical. Editing only terrain/objects keeps the map **Melee**;
  any data-catalog change would flip it to Arcade.

| Internal file | Magic | Authored? | Contents |
|---|---|---|---|
| `t3SyncCliffLevel` | `CLIF` | yes | discrete cliff-level grid (§2a) |
| `t3SyncHeightMap` | `SMAP` | yes | gameplay height (§2b) |
| `t3HeightMap` | `HMAP` | yes | render height (§2c) |
| `t3Terrain.xml` | xml | yes | `<rampList>` — authoritative ramp walkability (§3) |
| `CellAttribute_Pnp` | raw | yes | painted pathing flags; we write `0x00` (open) everywhere walkable (§5) |
| `t3CellFlags` | `LFCT` | yes (cleared) | `0x03` punches black holes in terrain; `clear_cell_flags` zeroes it |
| `t3SyncTextureInfo` | `RTXT` | yes | per-cell sync texture index (§6) |
| `t3TextureMasks` | `MASK` | yes | visible texture alpha layers (§6) |
| `Objects` | xml | yes | start locations + resources (§7) |
| `DocumentHeader`, `*.SC2Data\LocalizedData\GameStrings.txt` | `H2CS` / text | yes | map name (§8) |
| `MapInfo` | bin | **never** | playable area / camera bounds (§8) |
| `t3SyncPathingInfo` | `PATH` | no | derived by the engine; editing it has no effect |
| `t3FluffDoodad`, `t3VertCol`, `t3Water`, game data, models | — | no | inherited |

---

## 2. Elevation: three coupled layers

Gameplay (buildability, pathing, resource clustering) reads **CLIF + SMAP**; HMAP is visual only.
All three must agree, or you get e.g. minerals "on the wrong height" or bumpy-looking flat ground.

### 2a. `t3SyncCliffLevel` (CLIF)
- Header 32 B: `"CLIF"`, `u32 version=100`, `u32 sizeX`, `u32 sizeY`, 16 zero bytes. Body
  `u16[sizeY][sizeX]`, row-major, row = y. World **y points up** (IR row y == world y); flip rows to
  compare with screenshots.
- `0` = void (unplayable, unbuildable, blocks). Plateaus are `64 / 128 / 192 / 256` (levels 0–3).
  **There are exactly 4 cliff levels**; the generator must never emit more.
- Ramp cells carry sub-levels between plateaus (§3.2).
- Pathing is **4-connected**, and two adjacent walkable cells are mutually traversable by terrain
  only if `|Δcliff| ≤ 8`. A 64 step is a cliff wall. Larger in-ramp steps are fine where a ramp quad
  covers them (§3.1).

### 2b. `t3SyncHeightMap` (SMAP, ver 102) — gameplay height
- Header 64 B: `"SMAP"`, `u32 version`, `u32 sizeX`, `u32 sizeY`, pad. Body per **vertex**:
  `{int16 height; u16 mask}`; `height/256` = level. Vertex grid is e.g. 169×169 for 168² cells —
  don't read it as a cell grid.
- This is what `python-sc2` reports as `terrain_height`, and what its expansion finder reads.
- Heights are linear in cliff value: FrostLE `64→16.50`, `128→32.62`, `192→48.75`, void `→16.0`
  (~16 height units per level, so one +8 CLIF step ≈ 2.0 height).

### 2c. `t3HeightMap` (HMAP, ver 101) — render height
- Header 32 B: `"HMAP"`, version, sizeX, sizeY, pad. Body per vertex:
  `{u16 heightAdjustments; u16 heightBase; u16 mask}` — fine jitter, plateau height, level index
  (`0` void, `1` tier 0, …).
- Purely cosmetic, but it must match SMAP or the map looks different from how it plays.

### 2d. How we author them
- **Palette harvesting** (`harvest_palette`): the engine only treats a plateau as flat/buildable if
  its `(cliff, smap, hmap)` triple is one it understands. We copy real flat-vertex tuples for each
  tier (and void) out of the template instead of computing them; synthesizing heights produced
  non-buildable plateaus.
- `build_smap` / `build_hmap` interpolate those tier tuples over each vertex's cliff value, so ramp
  heights follow the CLIF gradient exactly.
- Vertex cliff = max of its surrounding cells (plateaus stay flat to the edge, the drop lands on the
  low side), **except** at ramps: `_vertex_cliff_ramp_aware` ignores cells more than 16 above the
  vertex's highest ramp cell, so a high-plateau corner doesn't spike a ramp-edge vertex.

---

## 3. Ramps

### 3.1 What makes a level change walkable

A walkable ramp needs **both**:
1. a CLIF **gradient** between the two plateaus (§3.2), and
2. a **`<ramp>` entry in `t3Terrain.xml`'s `<rampList>`** whose quad sits at the ramp's high edge.

The quad is the controlling factor. Proven by ablation on intact FrostLE terrain: deleting an entry
walls that ramp; a from-scratch entry (`make_ramp_entry`) makes a ramp walkable; a gentle staircase
with no entry is a wall; and a fully quad-covered transition is walkable even as a 1-cell 64 jump.
The engine **auto-expands a small quad** across the whole contiguous gradient band (one width-2 quad
makes a 32-wide band walkable), so one small quad per ramp is enough at any width.

Headless ramps need no Galaxy Editor bake. (Editor ramp *doodads* bake pathing differently, but gold
ladder maps use the same cliff-gradient + quad paradigm we do.)

### 3.2 Gold-standard ramp geometry (measured across 106 gold maps)

- **Single level only:** every ramp goes `lo → lo+1`.
- **Diagonal ramps** (the common case; dir 4–7): sub-levels `lo+8, 16, 24, 32, 40, 48` (6 cells),
  then **+16** into the high plateau (`hi−8`, e.g. 120, is always skipped). Isolines run at 45°, so
  every 4-connected neighbour steps exactly +8 and only diagonal neighbours jump +16 (irrelevant to
  4-connected pathing). A level therefore takes a **fixed ~7–8 axis-cell run**.
- **Cardinal ramps** (71 in the corpus; dir 0–3): a short 3-cell slope `lo+8, lo+24, lo+40`, then
  **+24** into the plateau. The engine's cardinal ramp mesh is sized for this profile.
- **Run is fixed; width is the only free knob.** Authored band width along the isolines varies
  4.5–15 on the same map. Engine-walkable ramps are ~2× the authored band (median ~10 × 8).
- **Flanks never touch void.** Sub-levels `lo+8..lo+40` are flanked by the low plateau; the top
  sub-level (`lo+48`) by the high plateau. The flank is walled purely by the cliff step.
- **`Pnp` = `0x00`** on ramp cells.

### 3.3 The `<ramp>` entry

```
<ramp dir hi lo leftLo leftHi rightLo rightHi base mid cid leftLoVar leftHiVar rightLoVar rightHiVar/>
```

Each transform is `u(x,y) r(x,y) c=(x,y) w h`, floats formatted like `1.000000e+000`. `u` is the
uphill unit vector, `r = rotate(u, −90°)`; `hi`/`lo` are cliff levels (`cliff // 64`). `leftHi` /
`rightHi` are unused sentinels (`w=h=0`, `Var=4294967295`).

**`dir` must match SC2's convention** (recovered from all 106 gold maps). The engine keys detection off
`dir`; a `dir` that disagrees with `u` makes the ramp vanish or mis-register.

| dir | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `u` | N (0,−1) | S (0,+1) | W (−1,0) | E (+1,0) | NW | NE | SW | SE |

Cell spacing along `u` is 1 for cardinals and √2 for diagonals. Gold quad recipes:

| | diagonal (dir 4–7) | cardinal (dir 0–3) |
|---|---|---|
| anchor `base.c` | centroid of top sub-level row + √2·u, no sideways offset | centroid of top sub-level row + 0.5·u (plateau edge) |
| width | `L/2` (L = top-row cells); `base.w = width·√2` | `base.w = L/2+1`, `mid.w = L/2−1` |
| `base.h` / `mid.h` | 0 / `run·√2` | 1 / 2 (the `base.h` lip reaches into the plateau) |
| run (`mid = base − run·spacing·u`) | 2 | 1 |
| low corners | 2×2 markers at `mid ∓ base.w·r` | 1×2 markers at `mid ∓ (L/2)·r` |

Walking downhill from `base.c` hits the first sloped cell after exactly 2 steps on every gold diagonal.

### 3.4 Quad failure modes (all verified in-engine)

- **Oversized quad → corrupted ramp.** Widening one FrostLE quad from 4 to 8/12/16 (`_rampmut.py`)
  inflates the engine ramp footprint (37 → 148 cells), drifts its centre off the interface, and spawns
  a **phantom ramp** where the quad overshoots into plateau. This sealed mains on generated maps.
  Endpoint `query_pathing` still says "walkable", so it is invisible to simple probes.
- **Stretched run → dead ramp.** Extending `run` to the full ramp length pushes the quad's far edge
  into wall cells and breaks detection. Leave run at the gold value; the engine expands it.
- **Quad on the flat shoulder → render humps and drift.** A quad anchored uphill of the slope makes
  the engine draw ramp mesh over flat plateau (walkable, buildable humps) and anchor ambiguously.
- **Low corners on plateau → phantom "nook".** A quad wider than `L/2` drops its corner markers past
  the flank onto the plateau, where the engine grows a phantom ramp region with unpathable blocks.
- **Rectangle at the gradient top without a lip → phantoms** (cardinals); the trapezoid's
  `base.h = 1` fixes it.

---

## 4. How the exporter authors ramps

Single source of truth: `ir.ramp_cell_cliffs` computes each ramp cell's CLIF value and is called by
both the exporter (`author_ramp_cliffs`, which writes CLIF) and the offline oracle
(`rasterize.engine_cliff_grid`), so they cannot drift.

- **Diagonal gradient** — `ir.isoline_gradient_ranks`: `cliff = lo + 8·(L − Lmin)` with
  `L = x·ix + y·iy` on the snapped lattice axis, clamped at `span = (hi−lo)/8 − 1 = 7`; the top rank
  maps to `hi` (the gold +16 step). Cells past the span clamp to `hi` (a flat shoulder, never a wall);
  a band shorter than `span` isolines tops out short and walls, hence `ramp_run_range=(8,10)`.
- **Cardinal gradient** — `ir.cardinal_profile`: `lo, lo+8, lo+24, lo+40, hi`.
- **Direction** — the low→high plateau-centroid vector snapped to the nearest of the 8 `dir`s
  (`_snap_dir`).
- **Quad** — `build_ramp_list` → `make_ramp_entry` with the gold anchors and sizes from §3.3 (a
  fallback `min(band, 4)` width applies when no top row is found).
- **Flanks** — `channel_ramp_flanks` voids plateau cells beside a ramp whose step exceeds 8, except
  the +16 top exit. This differs from gold, whose flanks are plateau, not void (open item, §10).
- **Pnp** — `author_pnp` writes `0x00` on all walkable cells, ramps included.

Env opt-outs, for A/B only: `RANK_ORDER_RAMPS` (legacy even-spread gradient; needs run ≥ 11),
`LONG_CARDINAL_RAMPS` (6-cell cardinal staircase + old trapezoid), `DIAG_SHOULDER_ANCHOR` (old
furthest-uphill diagonal anchor), `QUAD_WMAX`, `CARD_*` (cardinal quad params).

---

## 5. Pathing flags: `CellAttribute_Pnp`

4-byte header + `sizeX*sizeY` bytes, `0x00` = open … `0xFF` = blocked. The editor paints four types
(No Pathing, Ground, No Building, No Burrowing), so this byte is almost certainly those flags OR'd.
We only write `0x00`. Necessary for walkability but not sufficient — ramps still need the quad. Decode
the individual bits if we ever want no-build decoration or drop-proof high ground.

---

## 6. Texturing: `t3SyncTextureInfo` (RTXT) + `t3TextureMasks` (MASK)

**RTXT** — header `"RTXT"`, `u32 version=101`, sizeX, sizeY, one `u32` (observed `1`; preserve it).
Then a texture-name table: null-terminated ASCII strings with **no count field**. Scan until the
bytes stop being printable; the per-cell region starts with indices `0x00..0x07`. Then
`sizeX*sizeY*8` bytes: byte 0 = the cell's base texture index, bytes 1–7 always 0.

**MASK** — the visible texturing. Header 64 B: `"MASK"`, `u32 version=102`, `u32 unk`, sizeX, sizeY,
pad. `sizeX = 8 × cell width` (8×8 sub-cells per cell). Body = `n_layers` layers, one per texture in
name-table order, `n_layers = (fileSize − 64) / (sizeX·sizeY/2)`. Each layer is 4-bit alpha, 2 per
byte (high nibble first), tiled in 64×64-pixel blocks ordered +x then +y.

- All-zero MASK renders **purple** (no layer covers anything); all-`0xF` gives a muddy blend.
- Recipe (`author_texture_mask` + `author_texture_info`): per cell, the chosen layer = `0xF`, all
  others 0, and RTXT byte 0 = the same index. That is the editor's "Uniform Texture".
- A texture set has 8 textures; a map may combine up to 4 sets / 16 textures. Parse counts
  dynamically; don't reorder the primary set. Texture names are template-specific, so pick by index.
- `build_texture_class_grid` textures by **intended** level (even on flat exports), with a spread so
  adjacent tiers don't get look-alike textures, and a reserved ramp texture.

---

## 7. `Objects` and resource placement

- XML: start locations are `ObjectPoint Type="StartLoc"`; resources are `ObjectUnit` with
  `MineralField` / `VespeneGeyser`. Template doodads are dropped.
- `python-sc2`'s expansion finder groups resources that share terrain height and lie within 10.5 of
  each other, then needs a buildable townhall spot 4–8 from the centroid, ≥ 6 from every mineral and
  ≥ 7 from every geyser. If none exists, bot init crashes with `min() arg is an empty sequence`.
- Values that satisfy it: minerals on a 6.5 arc, geysers at 7.5, base pad half-width 9.5, nothing
  snapped inside the townhall clearance. `scripts/check_expansions.py` replicates the finder offline.

---

## 8. `MapInfo` and the map name

**Never edit `MapInfo`.** Playable area, full grid and camera bounds are coupled; enlarging the
playable rect alone made the client refuse to load ("A game has not been started yet"). Instead,
`exporter.pick_template` chooses the smallest template whose existing playable area contains our
content, and centres it there. Read the rect with `_mapinfo_playable_offset` then
`struct.unpack_from("<4i", mi, off)` → `(l, b, r, t)`.

**Rename** (`set_document_header_name`, `set_gamestrings_name`): the displayed name is the localized
string `DocInfo/Name`, stored in two places that carry no bounds fields:
- `DocumentHeader`: `"H2CS"` + fixed attributes + dependency strings, then `u32 recordCount` and
  records `{u16 keyLen; key; char locale[4]; u16 valLen; value}`. The locale is reversed (`enUS` →
  `"SUne"`). Replace every locale's value; no size or checksum to fix.
- Each locale's `GameStrings.txt`: the `DocInfo/Name=` line.

Renaming also avoids sharing the template's client cache. Still restart SC2 when iterating on one
template, to clear stale render/minimap caches.

---

## 9. Buildability, flat exports, and verification

- A cell is buildable iff CLIF ≠ 0 and its SMAP neighbourhood is flat (a harvested tier). Causes of
  "can't build here": void in the pad (bases too close to the template edge), or the base being
  **unreachable**, not unbuildable.
- **Flat export** (`flatten_terrain=True`, the default): collapse everything to one tier. Also
  `strip_render_relief` sets `<rampList num="0"/>`, otherwise the template's ramps render as ghost
  cliffs on flat ground. `_heal_flat_terrain` closes the leftover cliff-cleanup gaps
  (`binary_closing` + `binary_fill_holes`; filled cells take the nearest intended level).
- **Grid limits:** 32–256 per side in steps of 8; minimum playable area 15×9. Images (e.g. a preview)
  must be TGA; a preview must be square 24-bit.

**Oracles** — which checks can be trusted:
- `query_pathing(a, b)` is valid on properly authored maps. Don't query a townhall centre (often
  unpathable); `mainconn_probe.py` queries a small ring instead. It is **not** a slope oracle on maps
  whose Pnp/cell flags were flattened to open.
- `game_info.pathing_grid` reports ramp cells as 0, the same as walls. Real passability = pathing > 0
  or cell ∈ a detected ramp (`game_info.map_ramps`).
- The offline oracle (`engine_cliff_grid` + `engine_components`, 4-connected `|Δcliff| ≤ 8`, plus a
  same-ramp bridge for gold-profile cardinals only, `cardinal_ramp_bridge`) models CLIF connectivity.
  It is **blind to ramp detection**, so a map can pass offline and still seal a base in-engine. The
  `ramp_cleanliness` gate (exactly one plateau per side) catches most of these; in-engine probes are
  the final word.
- The macOS client returns all-zero render frames, so in-game visual checks need a human.

---

## 10. Open items

- **Flank voids:** `channel_ramp_flanks` still voids ramp flanks; gold flanks are plateau. Switching
  would be gold-faithful but has not been implemented.
- **Cardinal ramp width:** ours are ~5 wide (`ramp_choke_range=(4,6)`), gold 8–12, so our cardinal
  quads are smaller than any gold one. The cardinal render fix still needs a visual check in-game.
- **Ramp texture** covers the flat clamped cells at each ramp end (cosmetic).
- **Offline over-acceptance:** blobby/fused many-ramp seeds can read valid offline and split
  in-engine. The redundant-crossing prune and terrace carve reduce this; the durable fix is fully
  decoupling ramps from graph edges.
- **Oversized seeds** can exceed every template's playable area; **edge expansions** occasionally have
  a partly-void townhall pocket; a **custom preview image** is not authored.

---

## 11. Tools

Key probes (run with `PYTHONPATH=src .venv/bin/python scripts/<name>`):
- `export_map.py` — generate → validate → export → re-verify; `--author-ramps`, `--valid-only`.
- `mainconn_probe.py <seed>` — in-engine reachability from MAIN1 to every base (the real connectivity
  gate). `ramp_detect_probe.py <seed>` — authored vs engine-detected ramps, phantoms.
- `debug_ingame.py` — dump engine grids, buildable pockets, detected ramps.
  `check_expansions.py` — offline expansion-finder replica.
- `gold_ramp_study.py`, `_gold_quad_anchor.py`, `_gold_quad_width.py`, `_gold_cardinal.py`,
  `_quad_vs_slope.py`, `_rampcmp.py` — gold-map measurements behind §3.2–3.3.
- `_rampmut.py` — copy a gold map, mutate one ramp entry, read `map_ramps` back (single-variable test).
- `_symcheck.py` — check an exported map's terrain channels for mirror symmetry.
- `_divot_probe.py` / `_divot_decode.py` / `_divot_render.py`, `_nook_probe.py` — inspect CLIF/SMAP/
  HMAP/quad and engine codes around one ramp.
- Historical experiments that established §3.1 (kept for reproduction): `ramp_probe.py`,
  `control_probe.py`, `ablation_probe.py`, `synth_ramp_probe.py`, `axis_ramp_probe.py`,
  `rampstep_probe.py`, `rampwide_probe.py`, `isoline_ramp_probe.py`, `isoline_yield_ab.py`.
