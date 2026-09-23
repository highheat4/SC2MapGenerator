# SC2 `.SC2Map` Internals — Reverse-Engineered Findings

Practical, hard-won notes on the StarCraft II map format as it pertains to **headless map
authoring** (no Galaxy Editor). Everything here was discovered by decoding real ladder maps in
`gold_maps/`, round-tripping our own exports, and probing them in-engine with `python-sc2`.
Community docs (SC2Mapster wiki) got us started but were incomplete/ambiguous; the notes below
are what actually held true in practice.

> Scope: what the exporter (`src/sc2mapgen/export/`) needs. Byte offsets are little-endian.
> "cell" = terrain cell (1 per game tile). "vertex" = cell-corner grid (dims are cells+1 in the
> engine, but our authored grids are cell-sized and the engine tolerates it — see caveats).

> **References:**
> - SC2Mapster wiki (file formats): <https://sc2mapster.wiki.gg/wiki/File_Formats/Maps> — documents
>   most files (partial/occasionally-wrong, but a great start). Behind Cloudflare, so a
>   browser/`WebFetch` gets a JS challenge — use the **MediaWiki API** instead (not gated):
>   `curl -A '<browser-UA>' 'https://sc2mapster.wiki.gg/api.php?action=parse&page=File_Formats/Maps/t3TextureMasks&prop=wikitext&format=json&formatversion=2'`
>   (also `action=query&list=search&srsearch=...`). Always cross-check by round-tripping a real map
>   — that's how the MASK layout above got corrected.
> - S2 Editor Guides (concepts / editor model): <https://s2editor-guides.readthedocs.io> — plain
>   ReadTheDocs (no Cloudflare, but occasionally times out; just retry). Best for *why* the format
>   is shaped the way it is (map types, texture sets, map bounds, terrain layers). See §13.
> - SC2Mapster mkdocs "Getting Started": <https://sc2mapster.github.io/mkdocs/setup/> — an
>   **editor-workflow** guide (sc2edit + VSCode + Galaxy/`.SC2Layout`/FontStyles scaffolding). Almost
>   none of it applies to headless authoring; the only useful bits are conceptual (`.SC2Map` =
>   archived MPQ vs `.SC2Components` = the *same* files unarchived in a folder — handy for diffing a
>   template's internals) and the Melee↔Arcade note (§13a).
> - **Cloudflare update:** as of this pass the wiki's **MediaWiki API is ALSO gated** (`api.php`
>   returns the "Just a second…" JS challenge, not JSON). `curl` with a browser UA no longer works.
>   What *did* work: fetching the rendered page through our isolated fetcher (returns Cloudflare's
>   cached HTML). Cross-check everything by round-tripping a real map regardless.

---

## 0. The single most important lesson

**Do not write MPQ archives from scratch.** A hand-rolled MPQ writer produces archives that are
*structurally* valid (independent readers like `mpyq` round-trip them) but the **SC2 client
silently rejects them** ("A game has not been started yet" / protocol errors, or just won't load).

The only reliable path found: **take a real map as a template shell and edit files in-place with
StormLib** (`ctypes` binding, `src/sc2mapgen/export/stormlib_mpq.py`). StormLib preserves the
archive's header/hash-table/encryption invariants the client validates. We overwrite only the
handful of files we author and leave everything else byte-identical.

Corollary: everything below is about **which internal files to overwrite and how to encode
them** — never about building the container.

---

## 1. Archive = MPQ ("MoPaQ")

- `.SC2Map` is an MPQ archive. Internal files are addressed by path strings (e.g.
  `t3SyncCliffLevel`, `Base.SC2Data\GameData\TerrainTexData.xml`).
- Read with `mpyq` for inspection; write with StormLib for anything the client must load.
- `MPQArchive(path).files` lists internal names (bytes or str depending on version — normalize).

---

## 2. Template-shell strategy (what we author vs inherit)

We **author** (overwrite): `t3SyncCliffLevel`, `t3SyncHeightMap`, `t3HeightMap`, `Objects`,
`t3SyncTextureInfo`, `t3TextureMasks`, and (for flat maps) `t3Terrain.xml`.

We **inherit** verbatim: `MapInfo` (see §7 — do NOT patch it), `t3FluffDoodad`, `t3VertCol`,
`t3Water`, `Base.SC2Data\...` game-data XMLs, doodad `.m3` models, etc.

We also **author** the map's displayed name (`DocumentHeader` + every locale's `GameStrings.txt`,
see §7) so the export no longer shows the template's name (a map built on `FrostLE.SC2Map` used to
show "FrostLE" everywhere). Renaming also avoids colliding with the real ladder map's asset cache;
still, when iterating on the same template, quit + relaunch SC2 to clear stale render/minimap
caches. (The name lives in `DocumentHeader`/`GameStrings`, **not** in the `MapInfo` we inherit.)

---

## 3. Terrain elevation — three coupled files

SC2 separates **gameplay elevation** (buildability/vision/cluster detection) from **render
elevation** (what you see) from **discrete cliff levels**. All three must agree or you get
mismatches (e.g. minerals detected on the "wrong" height).

### 3a. `t3SyncCliffLevel` (magic `CLIF`) — discrete cliff/level grid
- Header (32 bytes): `magic[4]="CLIF"`, `u32 version` (100), `u32 sizeX`, `u32 sizeY`, `zero[16]`.
- Body: `u16[sizeY][sizeX]`, row-major, `row == y`.
- Value `0` = **unplayable/void** (renders purple-ish, unbuildable, blocks). Playable values are
  `CLIFF_BASE + CLIFF_STEP*level`. In our templates the flat tiers are multiples of 64
  (`64, 128, 192, ...`); ramp *sub-levels* sit between them (`72, 80, ... 112/120`).
- This is the grid the engine uses to decide **where cliffs are** for gameplay.
- **Hard cap: only 4 cliff levels exist.** The Terrain Layer guide (§13f) states there are exactly
  *four* distinct cliff levels, and the Cliff brush's Raise/Lower refuses to go past the highest/
  lowest. So the full playable tier set is just `64 / 128 / 192 / 256` (levels 0–3) — the generator
  must never emit more than 4 tiers, and the per-tier texture spread (§6e) has at most 4 buckets.

### 3b. `t3SyncHeightMap` (magic `SMAP`, ver 102) — synced/gameplay heightmap
- Header 64 bytes: `magic[4]="SMAP"`, `u32 version`, `u32 sizeX`, `u32 sizeY`, padding to 64.
- Body: per cell `{ int16 height; uint16 mask; }` (4 bytes). `height/256` = level; `mask` is a flag
  field. (We initially misread the 4 bytes as one `u32` 20.12 — the *tuples* still round-trip fine
  because we harvest & copy exact values rather than compute them, see §3d, but the correct
  breakdown is height(int16)+mask(uint16).)
- **This is what `python-sc2`'s `terrain_height` reports** and what resource-cluster/expansion
  detection reads. If this disagrees with where you placed resources, the bot's expansion finder
  breaks (see §8).

### 3c. `t3HeightMap` (magic `HMAP`, ver 101) — render heightmap
- Header 32 bytes: `magic[4]="HMAP"`, `u32 version`, `u32 sizeX`, `u32 sizeY`, padding to 32.
- Body: per cell `{ uint16 heightAdjustments; uint16 heightBase; uint16 mask; }` (6 bytes):
  - `heightAdjustments`: fine "dents/jitter",
  - `heightBase`: plateau height (what you mostly see; e.g. void `3642`, tier0 `32781`),
  - `mask`: `0..3` (level index; `0` void, `1` tier0, ...).
- Cliff model heights: `0` floor, `8` bottom, `10` middle, `12` highest. Level formula (needs
  `quantizeScale`, `quantizeBias`, `standardHeight` from `t3Terrain.xml`):
  `level = (heightAdjustments + heightBase) * quantizeScale - quantizeBias - standardHeight - 1`.
- Purely cosmetic for pathing (units path/build off SMAP+CLIF) — but flatten SMAP without HMAP and
  the map looks bumpy while playing flat (and vice-versa).

### 3d. Palette harvesting (the buildable-flat trick)
The engine only treats a plateau as truly flat/buildable if its SMAP/HMAP/CLIF triple matches an
encoding it "understands". Rather than guess, we **harvest** valid `(cliff, smap_raw, hmap_tuple)`
tuples straight from the template for each tier (and void), then stamp those exact values onto our
tier grid. See `sc2map.harvest_palette` / `build_smap` / `build_hmap`. Deriving heights by naive
interpolation instead of harvested values produced non-flat / non-buildable plateaus.

---

## 4. Ramps DO work headless — the missing ingredient is the `rampList` entry (BREAKTHROUGH)

> ⚠️ **This overturns the earlier "ramps impossible headless" conclusion.** Authoring a CLIF
> sub-level staircase + SMAP/HMAP alone is *not* enough — but that was never the whole recipe. The
> engine derives ramp walkability from the **`<rampList>` entry in `t3Terrain.xml`**, not from the
> height/cliff gradient. Add a valid entry and the ramp becomes walkable headless. No editor bake.

### 4.1 How we proved it (four experiments, `query_pathing` = ground truth)
The reliable oracle is the engine's own pathfinder: `await client.query_pathing(a, b)` returns a
finite distance iff a ground unit can walk `a→b` (returns `None` if unreachable). Do **not** judge
ramps from `pathing_grid` — see §4a.

1. **`scripts/ramp_probe.py`** — synthetic 2-plateau map, perfect +8 cliff staircase, locked
   heights, `Pnp=0`, cliff-flags cleared, `rampList` stripped → `query_pathing` L→R = `None`
   (blocked), sanity same-plateau = `3.0` (works). So gradient alone is **not** enough.
2. **`scripts/control_probe.py`** — repack FrostLE via StormLib with **no** gameplay change → all
   16 real ramps stay **walkable** (dist ≈ 5). ⇒ our StormLib repack does *not* break ramps.
3. **`scripts/ablation_probe.py`** — keep FrostLE terrain 100% intact, delete all but 2 `rampList`
   entries → `MAP_RAMPS_DETECTED` drops to exactly the 2 kept; **10 of 14 removed ramps become
   `None` (blocked)**; the few still reachable only via long detours through kept ramps. Same
   terrain, entry removed ⇒ ramp blocked. So the **`rampList` entry is the controlling factor**.
4. **`scripts/synth_ramp_probe.py`** — keep FrostLE terrain intact, replace its `rampList` with a
   **single entry we generate from scratch** (`sc2map.make_ramp_entry`) for ramp #6's location →
   `MAP_RAMPS_DETECTED=1` at our location and `query_pathing` across it = **6.0, WALKABLE=True**.
   ⇒ we can *synthesize* walkable ramps.

### 4.2 Anatomy of a walkable ramp (FrostLE ground truth)
All 16 FrostLE ramps are **diagonal (45°)** — `dir ∈ {4,5,6,7}`, uphill unit vector
`u = (±0.7071, ±0.7071)`. There are **no** axis-aligned ramps (our first failed staircase was
vertical). A ramp is defined by three coupled things:

- **CLIF grid gradient** (`t3SyncCliffLevel`): plateaus at multiples of 64 (level·64); the ramp is a
  short diagonal band stepping the cliff value by **+8 per cell** (64→72→…→128). Small steps (8) are
  walkable; a full-level jump (64) is an impassable cliff wall.
- **Heights locked to cliff**: SMAP is a **linear function of the cliff value** — measured FrostLE
  tiers are `cliff 64→16.500`, `128→32.624`, `192→48.749` (≈**16.1 gameplay-height units per level**,
  ⇒ ~2.0/cell at an 8-step). `build_smap`/`build_hmap` derive heights from the cliff grid by
  interpolating these tiers, so SMAP slope ∝ CLIF step — see §4.5 for the full height stack and why
  this caps how short a cliff-gradient ramp can be. *(An earlier note here gave `2045 + (cliff−64)·8`;
  that was a stale/unit-confused figure — the tier measurements above supersede it.)*
- **`<rampList>` entry** (the missing piece): a diagonal quad. Fields (see `make_ramp_entry`):
  `dir`, `hi`, `lo`; `base` = uphill-edge center (`u` uphill, `r = rotate(u,−90°)`, `w = width·√2`);
  `mid` = downhill center = `base − run·√2·u`, with `mid.h = run·√2` (FrostLE run = 2 cells);
  `leftLo/rightLo` = `mid ∓ (width·√2)·r` (small 2×2 corner markers); `leftHi/rightHi` are unused
  sentinels (`w=h=0`, `Var=0xFFFFFFFF`). Our generator reproduces FrostLE #6 to the digit.
- **`CellAttribute_Pnp = 0`** on ramp + plateau cells (ramp cells are *open*, not flagged).

### 4.3 What this unlocks
Multi-level terrain with real ramps is authorable **fully headless** — no Galaxy Editor bake
(`scripts/editor_bake.py` is obsolete). The exporter authors, per ramp, a cliff gradient (§4.4) + a
matching `<rampList>` entry (`set_ramp_list`) + `Pnp`, wired behind `ExportConfig.author_ramps`
(`scripts/export_map.py --author-ramps`). Flat stays the default — see §4.4 for the connectivity/
yield state that keeps it that way.

### 4.4 Making relief actually CONNECTED (the walkability rules the export must satisfy)
A relief map loads and shows real levels+ramps, but for units to actually traverse it the CLIF grid
must obey these rules. Verify headless with `query_pathing` and a `|Δcliff| ≤ 8` BFS
(`scripts/relief_probe.py`); the IR is already one connected component, so any fragmentation is an
*export/geometry* bug, not a generator bug.

- **Every adjacent pair of walkable cells must differ by ≤ 8** in cliff value. A plateau step is
  64 (=one level) — an impassable wall — so a level change is crossable *only* through a ramp whose
  gradient steps +8/cell. A single stray >8 step severs a plateau.
- **Ramp gradient authoring** (`author_ramp_cliffs`): use a **fixed 8/cell slope** that reaches both
  plateau values (`lo_cliff`…`hi_cliff`), NOT a span-normalized fraction. The current method is a
  **rank-order** spread along the snapped uphill axis (see below); a `d_lo/(d_lo+d_hi)` fraction and
  an 8-connected-BFS distance were both tried and abandoned (fraction made short ramps steep; BFS
  isolines curve so a straight quad can't follow them). Heights interpolate for any cliff value, so
  emit the **complete** 8-step sequence `range(lo+8, hi, 8)` — don't restrict to template-harvested
  sub-levels (FrostLE lacks 120).
- **Ramp level must be derived from the connected plateaus, not the ramp's own cells.** A clean ramp
  strip is painted at ONE elevation but *bridges* two levels; `elev.min()/max()` over its own cells
  mislabels a real 0→1 ramp as 0→0, and the exporter then skips it (leaving the high plateau
  walled). Fix in `rasterize.py`: `lo/hi = min/max plateau level among the ramp's non-ramp
  4-neighbours`. Same pass demotes same-level "ramps" (one neighbour level) to flat passages and
  drops <6-cell fragments.
- **Ramps ≥ 8 cells long? — YES for OUR diagonal ramps (nuanced; see §4.5).** `rampstep_probe.py`
  showed a *fully-quad-covered AXIS* staircase crosses at any run (even 1 cell) — the quad, not the
  gradient, gates walkability THERE. But the generator's ramps are rank-order **diagonal** staircases
  and the exporter quad sits only at the top (the engine auto-expands it; stretching it is HARMFUL,
  §4.5). So a diagonal 1-level climb still needs `run ≥ 8·√2 ≈ 11.3` to keep every cell ≤8/cell;
  shorter packs Δ16 jumps that island the high plateau (user-confirmed in-game on seed 22 @ run 5–7).
  **Generator ships `run 11–14`.** Two-level ramps (lo0→hi2) are still split into chained single-level
  ramps via a level-1 landing (a single quad has one lo + one hi edge).
- **Ramps must be channeled** — flanked by cliff/void on both sides, open only at the low and high
  ends (as FrostLE does). If a ramp's side abuts a plateau of the *other* level, that side is a 64
  jump (wall) and can also read as "overlapping levels". Requires generation-side geometry.
- `build_tier_grid` already preserves the IR's cliff-cleanup void walls (walk=False → tier −1), so
  different-level plateaus are separated by void *except* where the generator intends a ramp.

**Anatomy of a REAL FrostLE ramp (measured from `t3SyncCliffLevel`).** A ramp climbing one level
(64→128) is a **clean, narrow, DIAGONAL staircase**: each diagonal line of cells holds a constant
cliff value and consecutive lines step by exactly 8 —
`64 · 72 · 80 · 88 · 96 · 104 · 112 · (128)` — an **8-cell run at 8/cell**, flanked on both sides by
the plateau. The downhill axis is a 45° diagonal. This is the shape the engine's ramp detector and
its `<ramp>` quad expect.

```
128 128 112 104  96  88  80  72  64  64      <- one diagonal line per cliff value,
128 112 104  96  88  80  72  64  64  64         stepping 8 across ~8 cells (a clean
112 104  96  88  80  72  64  64  64  64         diagonal staircase, not a blob)
```

**A walkable level transition needs BOTH gates.** (1) the CLIF gradient steps `≤8` per cell with an
`≥8-cell run` low→high (a 64 jump is a wall; only a +8/cell staircase is walkable) — *necessary*;
and (2) a `<rampList>` quad that COVERS the ramp (§4/§5) — *authoritative*. A quad is a fixed-width
straight rectangle, so it only bridges a ramp that is itself a **clean narrow straight band**
(≈choke-width, exactly one plateau per end). That "clean strip" shape is the whole game.

**Axis-aligned ramps ARE walkable (proven with an engine-accurate probe).**
`scripts/axis_ramp_probe.py` builds a 2-plateau map with a horizontal full-8-step staircase + an
**axis-aligned** quad (`dir=1`, `u=(1,0)`, **unit** — not √2 — spacing): the engine detects the ramp
and `query_pathing` routes a ground unit across it; the same staircase with `rampList` stripped is
NOT walkable. So SC2 ramps are **not** diagonal-only (FrostLE just happens to use only diagonals) —
the quad must match the ramp's real direction: cardinal `u=(±1,0)/(0,±1)` with **unit** spacing, or
diagonal `u=(±.707,±.707)` with **√2** spacing. **Keep the boxy geometry; author each quad in the
ramp's real orientation** (8-way snap of the low→high vector). `build_ramp_list` does this. The old
~5/10 walkability came purely from forcing every quad to a diagonal even on axis-aligned corridors.

**★ The metric bug that invalidated months of "connected" claims.** Connectivity was originally
checked with `ndimage.label(walk)` — **elevation-blind**: it treats a cliff between two walkable
cells as passable. The engine does NOT; a level seam is an impassable wall crossed only by an
authored ramp. So grids read "connected" while the engine saw several disconnected level-islands.
Fix: `rasterize.engine_cliff_grid(walk, elev, ramps)` PREDICTS the exported CLIF (plateau
`(level+1)*64`; ramps carrying the real rank-order 8-step gradient via `ir.ramp_gradient_ranks` — the
*same* helper the exporter uses, so they can't drift), and `engine_components()` labels it under the
true rule: **4-connectivity with `|Δcliff| ≤ 8`**. `validate_map` gates main-to-main and all-base
connectivity on this. **Caveat (in-engine verified, seed 25):** this models the per-cell CLIF but
NOT quad coverage, so it over-reports on UNCLEAN ramps — hence the `ramp_cleanliness` hard-gate
(exactly one plateau component per side) MUST stay as the backstop that catches quad-uncoverable
ramps the CLIF model blesses.

**★ Exporter gradient — RANK-ORDER, not arithmetic (two encoder bugs fixed).** `author_ramp_cliffs`
sorts the ramp's cells by projection onto the snapped uphill axis and spreads the sub-levels EVENLY
over the distinct projection levels (low→`lo_cliff`, high→`hi_cliff`). This always reaches both
plateaus and never skips a sub-level. Two earlier schemes were traced to in-game one-cell walls:
- **Banker's rounding → 16-jumps.** `round((proj−proj0)/spacing)` is round-half-**even** in Python,
  so half-integer seams snapped ranks to even values → `64,64,80,80,…` (steps of 16 = an impassable
  cliff) instead of `64,72,80,88,…`.
- **"+8/cell from proj0" capped SHORT ramps** below the high plateau — a diagonal Δ64 climb (cells √2
  apart) needs a projection run ≥ `8·√2 ≈ 11.3`; ramps a hair short topped out at `176` and islanded
  the high plateau.

**★ Clean straight staircases, not blobs (the last-mile geometry fix).** The killer failure is
**blob ramps** — `perpW 7–16`, **`run=0`** (the low seam never reaches the high seam *through* the
component), up to 3 plateau fragments per side. They are FUSED JUNCTION FRONTS (a union of
staircases + grown rooms + repair-carves), which no single quad covers. Post-hoc cropping does NOT
fix them (a broad front can't be cropped into a valid bridge — the re-floored cells just get walled,
disconnecting further). The transition must be a single narrow choke at *carve* time. Implemented:
- **`_canonicalize_ramps`** — rebuild every non-clean ramp component into ONE choke-wide straight
  staircase between its two dominant plateaus, carved with a *fixed physical run* (extends `run/2`
  into each plateau so the gradient has depth to climb the full level), and LOCK those cells so
  cleanup/repair can't re-widen or demote them. Max ramp-component size on valid maps: 92 → ≤48.
- **`_nullify_cross_level_junctions`** — a Y-junction whose two parents land on different levels
  forces both parent→junction edges to ramp into one 3-way tap (an unauthorable blob). Delete such
  junctions + their edges and let `_build_constrained_edges` restitch the survivors with clean
  same-level / single-step edges. Preserves relief (level mix ~42/37/21); yield 7/60 → 11/60.
  *Rejected alternative:* equalising the junction's parents' levels — it cascaded downhill, flattened
  the map to 57% one level, and *lowered* yield.

**★ Exact rot180 symmetry (fairness + robustness).** paint/cleanup/repair drifted ~0.4% from the
(exactly symmetric) skeleton — enough to desync a mirror ramp pair (92 vs 87 cells) so one spawn
crossed and its mirror didn't (a 1v1 fairness defect). Source: raster-order `ndimage.label`
tie-breaks, `argmin/argmax` picks, per-half repair ordering. Fix: `_symmetrize_rot180` copies one
canonical half onto its partner (walk, elev, AND ramps) about the true playable centre
`partner(x,y) = (2·pa.x+pa.width−x, 2·pa.y+pa.height−y)` as the final paint step. Verified 0-cell
mirror disagreement (and 0-diff exported CLIF). **Residual (engine-side, can't author around):**
wide/irregular blob ramps still bridge asymmetrically because the ENGINE's own quad-expansion
flood-fill is parity-unstable on blobs — yet another reason to keep ramps narrow.

**★ PRNG lesson.** Generic node levels (`rng.choice((0,1,2), p=…)`) were drawn INTERLEAVED with
per-node geometry draws, so a `choice()` landed every 3rd value of the PCG64 stream; some preambles
give long constant runs (one seed: `[1,1,1,1,1,1,1,1,0]`) → the map collapsed to ~85% one level.
Fix: draw categorical fields in ONE vectorised call on a DEDICATED rng. **Never strided-interleave a
`choice(p=…)` with other draws on the same Generator.**

**Current status.** Relief+ramps are authorable fully headless and correctness is gated on the
engine-accurate metric (valid ⇒ main-to-main connected in the exported CLIF, verified on scans). The
remaining lever is **yield**: only a minority of raw seeds (single-digit to ~18% depending on the
sweep) rasterize into a fully clean-ramp-connected map — some LAYOUTS simply can't be built with only
clean single-step ramps — so `scripts/export_map.py --valid-only` rejection-samples the seed line
(offline oracles: `export_conn.py` / `rampbridge_probe.py`). Raising yield needs the source-side
redesign this doc keeps pointing at: **decouple ramps from graph edges** — between two adjacent
terraces lay a cliff wall with one/two well-separated choke-width straight staircases (quad + gradient
co-derived) and route every crossing through those + same-level passages, keeping transitions off
junctions. Fighting fusion after the fact (straighten / moat / crop / repair) is proven whack-a-mole.
The **flat** export (`flatten_terrain=True`, default) remains the shipped playable path meanwhile.

> **NOT a goal: a single-level rush path.** Gold maps are TERRACED — usually a ramp DOWN from main to
> natural, and the rush path crosses several levels. Connectivity must come from clean authorable
> chokes, NOT from flattening the map toward one level.

> *Dead code noted while instrumenting:* `_clamp_levels`, `_enforce_ramp_gaps`, `_break_peaks_pits`
> are defined but never called; the only live node-level logic is the initial level draw plus
> `_build_constrained_edges` (pure graph logic, zero pixel deps), so moving level ownership to the
> skeleton is a medium lift and the natural home for junction rules like the nullify one above.

### 4.5 How ramps ACTUALLY work — the height stack, real dimensions, and why OUR runs are longer
*Deep dive prompted by "real maps look like `run≈6`, ours is `10–12` — why, and can we cap width?"
Everything below is **measured from files** (`gold_maps/FrostLE.SC2Map` CLIF/SMAP, `dataset/
AutomatonAIE/_raw.npz` game-API grids) — no guessing.*

**(1) The height stack: CLIF drives a coarse level, SMAP is the gameplay height, and SMAP is a
LINEAR function of CLIF.** Two separate grids encode elevation:
- `t3SyncCliffLevel` (CLIF): the discrete level grid. Plateaus at multiples of **64** (`64·level`);
  ramp cells carry the in-between 8-steps (`72,80,…`).
- `t3SyncHeightMap` (SMAP, 20.12 fixed): the **gameplay height** that the engine and `python-sc2`
  expose as `terrain_height`. Measured FrostLE tier heights: `cliff 64→16.500`, `128→32.624`,
  `192→48.749`, void `→16.000` ⇒ **≈16.1 gameplay-height units per level**. `build_smap`/
  `_interp_heights` produce SMAP by **linearly interpolating the tier heights over each vertex's
  cliff value**, so **SMAP slope is exactly proportional to the CLIF step**:
  `ΔSMAP/cell ≈ 16.1 × (ΔCLIF/cell)/64 ≈ 2.0 × (ΔCLIF/cell)/8`.
  So our 8-CLIF/cell staircase = **~2.0 gameplay-height/cell**; a 64-jump (wall) = ~16/cell.
- (`t3HeightMap`/HMAP is the render layer only; pathing uses SMAP+CLIF, not HMAP.)

**(2) ⚠ CORRECTED: real maps use ~8 CLIF/cell GENTLE staircases (run ~8), NOT steep/short ramps.**
An earlier pass claimed AutomatonAIE ramps were "run ~4.5, step ~4 height/cell, 2× steeper than our
8/cell rule." **That was a measurement error** — it took the ramp's LONG axis as "width" and a short
sub-segment as "run", and overstated the per-cell step. Re-measured from `dataset/AutomatonAIE/
_raw.npz` **along each ramp's own height-gradient direction** (ramp signature `pathing=1 &
placement=0` with a real slope; n=22 ramps):

| map | ramps | RUN along gradient (p25/med/p75) | WIDTH ⊥ (p25/med/p75) | height/cell → CLIF/cell |
|-----|-------|----------------------------------|-----------------------|-------------------------|
| AutomatonAIE (shipped) | 22 | 6.0 / **7.6** / 7.9 | 7.2 / **9.5** / 11.6 | **2.09 → ~8.4** |
| our generator (now) | — | **11 – 14** (diagonal) | 8 – 12 (med ~9) | 1.0–2.0 → ~4–8 |

Plateau heights 191/207/223 are Δ16/level (16 height units/level, the same scale as FrostLE's
~16.1). A one-level (16-height) climb at **~2.0 height/cell = ~8 CLIF/cell** therefore takes **~8
cells of run** — i.e. AutomatonAIE obeys the SAME `≤8/cell, ≥8-cell run` rule as §4.4, and its ramps
are ~9.5 WIDE (a bit wider than their run). Real ramps are **not** steeper than ours. Our runs are a
touch longer (11–14) only because ours are 45° DIAGONAL (cells √2 apart, so 8 sub-steps need
`8·√2≈11.3` cells) whereas AutomatonAIE's are closer to axis-aligned; matching its run would mean
authoring more cardinal ramps. Our width ~9 already matches its ~9.5.

**(3) Two ramp-authoring paradigms — and paradigm (a) does NOT license short runs (see #4).**
- **(a) Cliff-gradient staircase** — what WE author and what FrostLE uses: the level change is a
  CLIF sub-level staircase (`64→…→128`), SMAP is derived from it, and a `<rampList>` quad COVERS the
  band. A quad only bridges a ramp whose gradient is ALSO a clean ≤8/cell climb over its full run
  (§4.4 "BOTH gates"); a short diagonal run packs 2 sub-levels into a cell (Δ16) and tops out short,
  islanding the high plateau. So this paradigm IS effectively run-limited (~≥11 cells diagonal).
- **(b) Editor ramp doodad** — how AAA maps (AutomatonAIE) are made: the Galaxy Editor places a ramp
  *model* and **bakes** the pathing + ramp data at build time, giving very smooth **rendered** slopes
  and wide footprints. We cannot run the editor bake headless, but per (4) we don't need it for
  *traversal* — (a)+quad already gives short walkable ramps; (b) would only improve the rendered
  look / allow wider ramps.

**(4) ★ VERIFIED: the `<rampList>` quad — NOT the cliff-gradient run — is the SOLE controller of
walkability.** `scripts/rampstep_probe.py` builds a clean 2-plateau (64→128) axis staircase whose
columns step by `S ∈ {8,16,32,64}` (= `64/S` ramp cells) with a covering quad, then checks BOTH
engine ground-truths — the DERIVED `game_info.pathing_grid` component labels **and a real Marine
ordered across it** — plus a no-quad negative control:

| cliff step | ramp cells | quad? | engine `pathing_grid` | Marine crosses (h 191→207)? |
|---|---|---|---|---|
| 8  | 8 | ✅ | CONNECTED | ✅ |
| 16 | 4 | ✅ | CONNECTED | ✅ |
| 32 | 2 | ✅ | CONNECTED | ✅ |
| 64 | **1** | ✅ | CONNECTED | ✅ |
| 8  | 8 | ❌ | **disconnected** | ❌ (stuck at h=201) |
| 64 | 1 | ❌ | **disconnected** | ❌ (stuck at h=191) |

So **a FULLY-quad-covered transition is traversable at any gradient step (even a 1-cell 64-jump), and
a gentle 8-step staircase with NO quad is a wall.**

**⚠ CORRECTION — this does NOT license short generator runs (the "runs can be short" over-claim).**
The subtlety this probe hid: **the engine floods walkability from a quad across a >8 step only where
the quad PHYSICALLY COVERS that step.** `rampstep_probe` made the quad cover the ENTIRE run, so every
steep cell was covered → walkable. But the generator's exporter (`build_ramp_list` → `make_ramp_entry`)
emits a quad at the ramp's HIGH end with the **default `run=2`**, covering only the top ~2 cells:
- A **≤8/cell** gradient floods DOWN from that top quad across the whole clean staircase (each ≤8
  step is terrain-traversable once the ramp is detected) → walkable. This is why a long-run
  (≥~11 diagonal) clean ramp works (verified in-engine: seed 32 fully connected).
- A **>8/cell (Δ16)** step below the covered top is NOT covered → the flood stops there → the ramp
  **tops out short and islands the high plateau** (verified: seed 5's MAIN1 exit is a walled
  dead-end, a Marine stuck 10 cells short). Short diagonal runs pack exactly these Δ16 jumps.

So runs are free to be short ONLY if the quad covers the full steep run. We tried to force that by
**stretching the exporter's quad** to the ramp's whole extent — and it is **✗ VERIFIED HARMFUL, not
an open item**:

> **Quad stretch is harmful — the engine already auto-expands a top-anchored quad.** Per #7
> (`rampwide_probe.py`) the engine floods a single narrow quad placed at the high-plateau interface
> across the *entire contiguous ramp band* on its own. Manually sizing the quad's `run` to the ramp's
> full projected extent (`run = extent/√2` for diagonals, i.e. `make_ramp_entry`'s `run` arg instead
> of the default `~2`) pushes the quad's FAR edge past the low plateau into wall cells, which
> **breaks the engine's ramp detection.** A/B in-engine at `run=(11,14)` (`scripts/mainconn_probe.py`,
> `pathing_grid`+`map_ramps` flood): stretch **ON** isolates seed 22's MAIN1 from the entire map
> (natural→main ramp dead) and severs seed 2's natural link, while stretch **OFF** connects both;
> it only "helped" seed 13 and was neutral on 32/5 — a per-seed wash that can silently regress maps
> the offline oracle can't see. At short runs it also drives a `bridge` oracle to **false-accept**
> (seed 22 reads VALID offline but its natural→main is a 1-tile walled diagonal dead-end in-engine —
> **user-confirmed in-game**). Conclusion: **do NOT stretch; leave `run` at `make_ramp_entry`'s
> default and let the engine expand it.** The real lever is the CLIF gradient (≤8/cell), i.e. run
> length, not quad size.

The robust, real-map-matching path is therefore the **gentle ≤8/cell staircase**, which per (2) is
what AutomatonAIE itself uses (run ~8 axis / ~11 diagonal). Empirical yield confirms:
`run=(4,6)` → median ramp step 16, STRICT connectivity 6/60; `run=(11,14)` → median step 8, 42/60.
**We ship `ramp_run_range=(11,14)`.** Paradigm (b)'s doodad bakes Pnp so it needs no gradient at all;
we can't run it headless, but per (2) we don't need it — a gentle staircase matches real dimensions.
The residual case that survives all of the above is a **short/steep DIAGONAL ramp**, e.g. seed 22's
natural→main: even at `run=(11,14)` its detection is marginal (`MAP_RAMPS_DETECTED` flips 4↔5 across
otherwise-identical engine runs), so the strict oracle over-accepts it. This is the diagonal-ramp
limitation the source-side terrace redesign (§4.4) is meant to cure, not oracle/quad tuning.

**(5) Width is a free realism knob, not a traversal limit (see #7).** Ramp width is measured as the
perpendicular extent of the ramp cells, which **inflates on diagonals** (pixel staggering makes a
5-wide diagonal read as ~7). We can go as wide as we like for *traversal* (#7); the only cost is
*yield* (a wider diagonal band is likelier to sprawl onto a 2nd plateau → unclean → rejected). Real
maps sit at width ~8–12 (median 11), reachable by paradigm (a) alone — no doodad needed.

**(6) Corrections to earlier claims + the probe methodology that resolved this.**
- **FrostLE is NOT "≤8/cell everywhere."** Its ramps carry only **6 distinct sub-levels
  (`72,80,88,96,104,112`)** — it **lacks `120`**, so the top edge is a **`112→128` = +16 step** and
  the ramp still works. Consistent with (4): the quad, not the per-cell step, gates walkability.
- **`query_pathing` alone is NOT a slope oracle — use `game_info.pathing_grid` + a unit move.** An
  earlier step-sweep (`_step_axis`, since deleted) reported EVERY step — even a 1-cell 64-drop and a
  no-`rampList` control — as "walkable" because it read only `query_pathing` on a map whose `Pnp`
  had been flattened to all-open (`0x00`) with `t3CellFlags` cleared. `query_pathing` consults those
  authored pathing layers, so flattening them defeats the test. The correct oracles (used by
  `rampstep_probe.py`) are the engine-DERIVED `game_info.pathing_grid` (re-derived from terrain +
  rampList; ramp-only maps show gradient-only staircases as disconnected without a quad) and an
  actual `debug_create_unit` + `Unit.move`. NB `query_pathing` IS valid on a *properly-authored*
  exported map (real void walls block it) — but query the TOWNHALL CENTER at your peril: it is often
  non-pathable, giving a false "unreachable" (fixed in `mainconn_probe.py` by querying a small ring).
- **Offline oracle uses the STRICT `≤8` metric + exporter-faithful grid (a ramp `bridge` was tried
  and dropped).** `engine_cliff_grid` now applies the exporter's `channel_ramp_flanks`, so the
  offline grid is byte-for-byte the exported `t3SyncCliffLevel` (verified `VOIDED_by_export=0`).
  `validate_map` then labels it with `engine_components` under the plain `≤8`/4-connected rule (§4.4)
  — **no** ramp bridge. A bridge (join adjacencies touching a ramp cell, or same-ramp-id cells) was
  added to rescue short/steep ramps but it FALSE-ACCEPTS quad-uncoverable ramps the engine walls
  (in-engine: seeds 2/5 "valid" but a Marine can't cross); with adequate run (#4) legit ramps are
  ≤8 and connect under strict anyway, so the bridge only added false-accepts. Residual over-report
  on unclean ramps → the `ramp_cleanliness` hard-gate + `--valid-only` rejection sampling (see #8).

**(7) ★ VERIFIED: one quad makes an ENTIRE clean band walkable at ANY width — wider ramps need a
wider BAND, not more quads.** `scripts/rampwide_probe.py` builds a 2-plateau axis staircase whose
transition is a ramp band `W` rows tall (the rest of the boundary is a void wall, so units MUST
cross through the band), covers it three ways — ONE wide quad, TWO half-width quads, or ONE width-2
quad — then spawns a *row* of Marines across the full width and orders each across:

| band width `W` | covering | Marines that crossed (offsets across width) |
|---|---|---|
| 16 | 1× width-2 quad | **9/9** (±7) |
| 16 | 1× width-8 quad | 9/9 |
| 16 | 2× width-4 quads | 9/9 |
| **32** | **1× width-2 quad** | **7/7** (±15) |
| 32 | 2× width-8 quads | 7/7 |

So the engine **expands the detected ramp across the whole contiguous single-level band**: a lone
narrow quad sitting on the transition floods walkability to the band's full width (≥32 cells, well
past real maps' ~19 max). **Multiple quads are therefore redundant for width** — they work but buy
nothing over one quad. (The earlier "a wide ramp with a width-4 quad was only *partly* walkable" was
about *irregular/unclean* bands whose cells were not one contiguous single-level gradient — not
about width.) Practical upshot: the engine expands a *small* quad across the band, so a small quad is
sufficient for any width. ⚠ **SUPERSEDED (see §4.8/§4.9):** the earlier claim here — "to make ramps
wider just widen the band; `build_ramp_list` sizes the quad to the ramp's full cross-section" — was
WRONG. Sizing the quad to the full band *corrupts* the ramp (balloons + phantom), and widening the
authored band re-breaks traversability offline-invisibly. `build_ramp_list` now emits a **small fixed
quad** (`min(band,4)`, run 2) and the band stays moderate; see §4.9 for the shipped values.

**Ramp WIDTH knob — ⚠ SUPERSEDED by §4.9.** This paragraph originally shipped
`ramp_choke_range=(7.0,10.0)` on the theory that a full-cross-section quad made the band the only width
control. That was wrong (§4.8): the full-band quad ballooned/phantom-split ramps. The **actually
shipped** value is `ramp_choke_range=(4.0,6.0)` with `ramp_run_range=(11.0,14.0)` and a small fixed
quad — this already yields ~16×14 *walkable* ramps (wider than gold) while keeping traversability;
widening the band to `(7,10)`+ drops yield and re-breaks main→natural links (§4.9). For historical
context only, the earlier STRICT-connectivity sweep at run=(11,14) read choke `(3,5)`→46/60,
`(5,7)`→48, `(7,10)`→42; those numbers predate the small-quad exporter fix and the exporter-faithful
oracle, so they are not comparable to §4.9's `(4,6)`→32/60. The run must stay ≥~11 (diagonal) or ramps
island the high plateau (#4).

**(8) ⚠ OPEN: the STRICT oracle still slightly over-accepts on quad-uncoverable ramps (documented
limitation, not a new one).** With clean long runs (#4) the strict `≤8` metric matches the engine
for the vast majority of ramps, and in-engine spot checks (`mainconn_probe.py`, using the engine's
own `pathing_grid`+`map_ramps` flood and confirmed by real `debug_create_unit`+`Unit.move`) verified
seeds like 32 flip from broken→fully-connected once runs are adequate. But a residual class remains:
a ramp with a clean ≤8 gradient whose diagonal band still isn't cleanly coverable by one straight
quad reads connected offline yet is a walled dead-end in-engine (seeds 2, 5). This is exactly the
§4.4 caveat ("models per-cell CLIF, not quad coverage → over-reports on UNCLEAN ramps"); the
`ramp_cleanliness` hard-gate catches most but not all, and `--valid-only` rejection-samples the rest.
The durable fix is NOT more oracle tuning (proven whack-a-mole) but the **source-side terrace
redesign** already prescribed in §4.4 "Current status" (co-derive one clean straight quad+gradient
per terrace crossing). A bridge variant was tried and dropped (it only added false-accepts, above).

### 4a. `pathing_grid` gotcha that faked us out
`python-sc2` reports **ramp cells as `pathing_grid == 0`** (same value as walls). Early
connectivity BFS treated all `0` as impassable and wrongly concluded ramps were disconnected. Real
passability = `pathing > 0` **OR** cell ∈ a detected ramp. `scripts/conn_probe.py` does this
correctly. **Even so, don't judge ramp walkability from grids** — a walkable ramp and an
impassable cliff wall are *both* `pathing==0 & placement==0`. Use `query_pathing` (§4.1) as the
oracle instead.

### 4.6 ★★ The engine's ramp DETECTOR is orientation-biased on WIDE ramps (root cause of the mirror-asymmetric "one main sealed" bug)

> ⚠️ **Largely SUPERSEDED by §4.12.** Later work (seed 42) showed wide DIAGONAL ramps detect fine and
> the real systematic dropout was CARDINAL ramps whose emitted `<ramp dir>` label disagreed with SC2's
> convention (`dir` scrambled for cardinals). Fixing the `dir` table (§4.12) restored cardinal detection
> and reconnected the mains. The genuine residual below (engine quad-expansion parity on *marginal*
> ramps) is real but small; the width-cap `ramp_choke_range=(4,6)` still ships for the reasons in §4.9.

**Symptom.** On rot180 maps, one main's exit ramp is a walkable slope while its MIRROR twin is a
1-tile-wide walled diagonal dead-end — the SCV can't leave (user-reported, seed 22). Two *identical*
mirror ramps behave differently, which "makes no sense" if the map is symmetric.

**What it is NOT — every exported channel is PROVABLY rot180-symmetric.** We read the exported
`.SC2Map` back and tested each terrain layer for rot180 symmetry *at the two mirror main ramps*
(`scripts/_symcheck.py`, seed 22):

| channel | rot180 symmetry at the ramp cells |
|---|---|
| `t3SyncCliffLevel` (CLIF) | **1.0000** (exported grid == offline `engine_cliff_grid`, 0 differing cells) |
| `t3SyncHeightMap` (SMAP — what the engine paths on) | **exact**: 0/119 cell mismatch, ramp1 vs ramp4 |
| `t3HeightMap` (HMAP) | **1.0000** |
| `CellAttribute_Pnp` | **1.0000** |
| `t3CellFlags` | 1.0000 (zeroed) |

The `rampList` quad polygons for every mirror pair are exact rot180 mirrors too (uniform +1 cell
even-grid offset). So there is **no hidden asymmetric channel** — the engine reads mirror-identical
local data for the two ramps. (Beware: SMAP/HMAP are **169×169 vertex** grids, `u4`/`u2×3`; reading
them as the 168² cell grid gives garbage that falsely looks ~50% asymmetric. Compare mirror **cells**
directly, not a global grid shift.)

**What it IS — the detector heuristic is not rot180-invariant, and only fails when the band is WIDE.**
`game_info.map_ramps` on seed 22 held only **4 of our 6** ramps. Census of all 6 (each an exact
mirror-pair member):

| width (perp extent) | dir (uphill) | detected? |
|---|---|---|
| 7.1 | dir4 = NW (left) | ✅ |
| 7.1 | dir7 = SE (right) | ✅ |
| 8.5 | dir5 = NE (right) | ✅ |
| 8.5 | dir6 = SW (left) | ❌ |
| 9.2 | dir7 = SE (right) | ✅ |
| **9.2** | **dir4 = NW (left)** | ❌ ← MAIN1's exit → main sealed |

Pattern: **wide** (≳8) diagonal ramps whose uphill points **leftward** (dir4/dir6) are NOT
registered as ramps; their right-pointing mirror twins are, and **narrow** (≲7) ramps register in
**every** orientation. An unregistered ramp = the engine never opens the cliff line at its top, so
the whole band is pathable-slope-but-sealed from the plateau above (a walled diagonal seam exactly
one cliff-line thick — matches the in-game screenshot).

**Interim fix (mechanism was wrong — real fix in §4.8/§4.9): cap ramp width narrow.**
`ramp_choke_range=(4.0,6.0)` → measured width ~6–7 → all 6 ramps detected, seed 22 MAIN1↔MAIN2 fully
connected in-engine, offline STRICT yield 22→32/60. **⚠ This was a workaround with the WRONG
mechanism.** (The shipped config keeps `(4,6)`, but for the *quad* reason in §4.9, not a
detector-width reason — with the small quad the same band width now detects a clean 6 ramps, not 8.) The trigger isn't "wide *band*" — it's our **oversized `rampList` quad** (we set the
quad width to the band's full extent). Gold maps prove wide cliff-gradient ramps detect fine when the
quad is SMALL: they author width~11 bands with a width~3 quad and the engine expands it (§4.7). So the
real fix is to **decouple the quad from the band** (small fixed quad + wide band), *not* to shrink the
band. Gold-map wide ramps are NOT doodads — they are the same cliff-gradient paradigm we use.

**Consequence — the offline oracle is BLIND to ramp detection.** `validate_map`/`engine_cliff_grid`
model CLIF *connectivity* (`|Δcliff|≤8`, 4-conn), never whether the engine will *detect* the ramp.
So a wide-ramp map reads VALID offline yet seals a main in-engine (why seed 22 passed offline). The
width cap is a *source-side* guarantee; the oracle cannot catch this class, only rejection-sampling
against the real engine (or the width cap) can. Residual: some seeds (e.g. 5, 2) still split the two
mirror halves for a *separate* reason — a center-link/`lo0→hi2` two-level ramp — tracked in §12.

### 4.7 ★★★ Gold-standard ramp invariants — the definitive recipe (measured across 106 gold maps)

Direct measurement of the actual gold files settles how real ramps are built and what makes them
traversable. Sources: `gold_maps/{Acropolis,Automaton,Frost}.SC2Map` (decoded CLIF + `rampList`) and
`dataset/*/_raw.npz` (engine `pathing`/`placement`/`height` grids, 106 maps). Analysis:
`scripts/gold_ramp_study.py`.

**The two "widths" that were being conflated.** A ramp has an *authored cliff band* and an
*engine-expanded walkable region*, and they differ by ~2×:

| dimension | authored CLIF band | engine WALKABLE ramp |
|---|---|---|
| **WIDTH** (units abreast, along isolines) | med 9.5, range **4.5–15** | med **10.1**, p10–p90 8–13, up to ~20 |
| **RUN** (uphill, low→high) | **~4.5 perp / ~8 axis-cells**, FIXED | med 8.1 (incl. plateau shoulders) |

So gold ramps ARE ~10–12 wide — that's the "width 12" you see. It is produced by a **narrow authored
band + a tiny quad that the engine expands**, NOT by authoring a 12-wide quad.

**The invariants (hold across all/most gold maps):**

1. **Cliff vocabulary is universal.** Plateaus at `64 / 128 / 192` (levels 1/2/3). Ramp sub-levels
   are exactly `72, 80, 88, 96, 104, 112` (steps of **8**), then a **+16 step to the plateau**
   (112→128); **sub-level 120 is skipped**. Every gold ramp uses all **6** sub-levels.
2. **Step = 8 CLIF (= 2 engine-height units) per AXIS cell.** 92.5% of 4-connected neighbour steps on
   ramp cells are exactly 2 height units; one level = 16 height = 64 CLIF ⇒ **~8 axis-cells of run per
   level**. Pathing is 4-connected, so only axis neighbours matter — the *diagonal* neighbour jumps 16
   and that's fine. (This retro-justifies the ≤8/cell rule *on axis neighbours*, and shows the old
   "diagonal ramps need run ≥ 11.3" worry was moot: 45° isolines give 8/axis-cell at short run.)
3. **The +16 top step works because the quad covers it.** FrostLE has no sub-level 120, so its top
   edge is a 112→128 = +16 axis step; still walkable because the `rampList` quad sits over the
   high-plateau interface (run 2 from the top).
4. **Single-level only.** Every gold ramp is `lo → lo+1` (1→2 or 2→3). None skip a level (no `lo0→hi2`).
5. **Diagonal orientation** (dir 4–7), isolines at 45°. All three sampled maps use only dir 4–7.
6. **The `rampList` quad is TINY and uniform**: `run` = 2 cells (**every** gold ramp: base→mid = 2.83
   = 2·√2 diagonal), quad `width` param = **1–5 cells** (`base.w` = 1.41–7.07). Placed at the HIGH
   edge. The engine then **auto-expands** the small quad down and across the contiguous cliff band to
   the full ~10–12 walkable width. Corner markers are 2×2 sentinels; hi-corners are `Var=0xffffffff`.
7. **Pnp at ramp cells is open (`0x00`).** FrostLE: 100% `0x00`. Acropolis/Automaton: mostly `0x00`
   with a few edge-flag cells (cosmetic/doodad seams). Nothing "pathing" is baked — walkability comes
   from the CLIF gradient + the quad, exactly our headless paradigm.

**What allows VARIABILITY between ramps (and what does not):**

- **RUN is essentially FIXED, quantised by level count.** With step locked at 8 CLIF/cell and 64 per
  level, a single-level ramp is *always* 6 sub-levels ≈ 4.5 perp / ~8 axis-cells. Measured authored
  run: med 4.5, range 3.8–4.5, **std 0.2** — no real variability. Run only grows if the ramp climbs
  more levels (which gold maps never do). **Run is not a free design knob.**
- **WIDTH is the free continuous variable.** Authored band width varies **4.5–15** (std 2–3); walkable
  width CoV ≈ 0.34. Ramps differ almost entirely in how wide the band is along the isolines. Narrow
  chokes (~4.5) and wide ramps (~15) coexist on the same map.
- Weak positive corr(width, run) ≈ 0.55 in the *walkable* measure only — an artifact of shoulder
  inclusion and the occasional multi-cell merge, not of the authored geometry.

**⇒ Generator recipe (gold ideal vs shipped compromise).** The gold ideal is: author a WIDE band
(width ~5–13 along the isolines) at a FIXED short run (6 sub-levels/level, step 8 CLIF/axis-cell, +16
top step), single-level, 45° diagonal, and emit a SMALL fixed quad (width ≈ 3–4, run 2) — let the
engine expand it. We shipped the SMALL QUAD half of this (see §4.9), which fixed detection: seed 22 now
detects a clean 6 ramps instead of the phantom-split 8 the old "quad = full band" produced. We did NOT
adopt the wide-band + short-run half: our rank-order diagonal gradient needs run ≥ ~11 to stay ≤8/cell
(it lacks gold's true 45° isolines), and widening the authored band re-breaks traversability
offline-invisibly (§4.9). Shipped values are therefore `ramp_choke_range=(4,6)`, `ramp_run_range=
(11,14)`, small quad — which already expand to ~16×14 walkable ramps. Closing the aspect-ratio gap
(gold-tight wide+short ramps) is the outstanding gradient-rework follow-up (§4.9, §12).

### 4.8 ★★ Copy-and-mutate experiment — what an oversized quad ACTUALLY does (single-variable, in-engine)

`scripts/_rampmut.py` copies FrostLE, replaces exactly ONE ramp's `<ramp>` entry (index 7: dir5,
1→2, gold width 4 at cell (136,58)) with our own `make_ramp_entry` at increasing quad widths, repacks
via StormLib, and reads `game_info.map_ramps` back. Everything else (the CLIF gradient, the narrow
~4.5-cell band) is byte-identical. Result — the ramp's engine footprint (`npts`) and center:

| quad width | ramp @ (132,57)? | footprint npts | phantom ramp | total ramps |
|---|---|---|---|---|
| control (gold, w4) | present | 37 | — | 16 |
| w4 (our `make_ramp_entry`) | present, identical | 37 | — | 16 (reproduces gold exactly) |
| w8 | drifts →(130,56) | **60** | **+(139,65)** | 17 |
| w12 | drifts →(127,50) | **148** | **+(140.8,66.8)** | 16 |
| w16 | drifts →(125.8,52) | **119** | **+(142.6,68.5)** | 16 |
| remove entry | **gone** | — | — | 15 |

**What makes a ramp valid vs invalid:**
1. A ramp needs its `<rampList>` entry — deleting it drops the ramp (16→15), the band becomes a wall.
2. `make_ramp_entry` at the gold width **reproduces the gold ramp exactly** (w4 == control) — our
   synthesis geometry is correct.
3. **An oversized quad does NOT delete the ramp — it CORRUPTS it.** When the quad is wider than the
   true cliff band, the engine (a) *balloons* the ramp footprint (37→148 cells, grabbing plateau it
   shouldn't), (b) *drifts* the ramp center off the real interface, and (c) spawns a **phantom ramp**
   where the quad's far edge overshoots into the plateau/wall. `query_pathing(top,bottom)` still says
   "walkable" at the ramp's own endpoints, so this is invisible to endpoint probes — but the ramp is
   now anchored at the WRONG place and swallows plateau cells, which is what seals the true main exit /
   breaks connectivity on our generated maps (seed 22). So "quad ≤ true band width, small (~3–4)" is
   the operative rule; quad > band ⇒ balloon + phantom.

**Compared to what WE currently emit** (`build_ramp_list`, seed 22): quad width = the ramp's **full
band extent** (6–9+), band run 12.5 (gold ~4.5). The full-band quad is exactly the w8–w16 failure mode
above. Fix = emit the gold-style small fixed quad (width ~3–4, run 2) and let the engine expand it.

---

### 4.9 ★★ SHIPPED FIX — small fixed quad, and why we DON'T widen the authored band

Implemented in `export/sc2map.py build_ramp_list`: the declared `<rampList>` quad width is now
`min(band_extent, GOLD_QUAD_W=4)` (was the full band extent), run stays `make_ramp_entry`'s default
`~2`. This is the gold recipe from §4.7/§4.8. In-engine verification (`scripts/mainconn_probe.py`,
default choke `(4,6)`):

- **Seed 22** (the sealed-main case): now detects a **clean 6 ramps** (was 8 — the extra 2 were the
  phantoms from §4.8) and **MAIN1↔MAIN2 fully connects**. Seed 32 also fully connects.
- The change is purely an exporter tweak, so **STRICT offline yield is unchanged (32/60)** — the quad
  never entered the offline CLIF oracle.

**Walkable width is already gold-plus.** Measuring the engine's `pathing_grid`/`terrain_height` slope
bands, our authored `6.6 × 12.5` (width × run) band expands to a **~16.5 × 14.2 WALKABLE** ramp —
*wider than gold* (`~10.1 × 8.1`) in both axes. So the small quad already gives wide, traversable
ramps; there is no need to widen the authored band to hit "width ~12".

**Widening the AUTHORED band is harmful — do not.** Sweeping `ramp_choke_range`:

| choke | STRICT yield | authored width med/p90/max |
|---|---|---|
| **(4,6) ← shipped** | **32/60** | 6.6 / 9.1 / 13.7 |
| (6,10) | 21/60 | 10.3 / 13.4 / 16.1 |
| (8,12) | 16/60 | 12.5 / 15.2 / 16.9 |

Wider bands not only drop yield, they **re-break traversability offline-invisibly**: at `(6,10)`, of 5
offline-VALID seeds probed in-engine, 3 fully connect but seeds 11 & 13 lose their own **main→natural**
ramp (`SAME=False`) even though the offline oracle passed them — the wider diagonal band creates a
`Δcliff>8` / flank-void the CLIF oracle doesn't model. So `(4,6)` stays the default.

**Open follow-up (aesthetics only, not traversability).** Our ramps were wide *and long/blobby*
(`16.5 × 14.2`, aspect ~1.16) vs gold's tighter `10.1 × 8.1`. Getting gold-TIGHT ramps means a shorter
run, which the *rank-order* diagonal gradient couldn't do without exceeding `≤8/cell` (needs run ≥
8·√2 ≈ 11.3). Gold packs a level into ~8 axis-cells via **45°-oriented isolines**. ✅ **DONE — see
§4.10:** the exporter+oracle now default to a **true-isoline gradient**, which shortens the run to
gold's ~8 axis-cells AND raises yield.

---

### 4.10 ★★★ SHIPPED: true-isoline gradient (gold-short runs, now the DEFAULT)

The rank-order gradient (§4.4/§4.5) spread `span` sub-levels evenly over a diagonal band's DISTINCT
projection levels, so a one-level climb needed run ≥ 8·√2 ≈ 11.3 or it packed a Δ16 wall — the
"our runs are 2× gold" problem. The fix is to author the gradient the way gold maps do: **lock the
cliff value to the cell's anti-diagonal (`cliff = lo + 8·(L − Lmin)`, `L = x·ix + y·iy` on the
snapped integer axis)**. Then every 4-connected AXIS neighbour steps EXACTLY +8 (walkable) and the
diagonal neighbour jumps +16 (4-connected pathing ignores it, §4.7 #2), so a full level is ALWAYS
`span` axis-cells of run — gold's short FIXED run — and WIDTH is free.

**Single source of truth.** `ir.isoline_gradient_ranks` (+ the `ir.gradient_ranks` dispatcher) is
called by BOTH `author_ramp_cliffs` (writes `t3SyncCliffLevel`) and `engine_cliff_grid` (the offline
oracle), so they can never drift. **Default = isoline;** set `RANK_ORDER_RAMPS` to restore the legacy
even-spread gradient. Shipped run is now `ramp_run_range=(8,10)` (was `(11,14)`).

**Proven headless first** (`scripts/isoline_ramp_probe.py`): a code-authored locked-8 diagonal band
at run = 8 anti-diagonals (~5.7 along `u`, ≈ HALF the old run) is walkable in-engine (derived
`pathing_grid` component + a real Marine) at widths **4, 8, and 12** — fixed short run, free width.
A no-quad control is a wall (0 ramps); run < 8 isolines tops out short (needs ≥ span isolines).
Robust to LONG bands (extra cells clamp to the high plateau = a flat shoulder, never a wall), only
SHORT bands (< span isolines) wall — which is why run must stay ≥ ~8.

**A/B (`scripts/isoline_yield_ab.py`, `mainconn_probe.py`):**

| gradient | carve run | STRICT yield | in-engine |
|---|---|---|---|
| rank-order (old default) | (11,14) | 59/120 | seeds 5 walled |
| **isoline (new default)** | **(8,10)** | **65/120** | **seeds 5 & 22 fully connect** |
| isoline | (11,14) | 32/60 (== rank-order) | — |
| isoline | (6,8) | 20/60 (< 8 isolines → top-out) | — |
| rank-order | (8,10) | 33/60 | — |

So isoline at the gold-SHORTER run (8,10) **beats** rank-order at the longer (11,14) on yield AND
fixes real in-engine dead-ends (seed 5 — a documented walled dead-end — now reaches every base;
seed 22 stays connected with 6 clean ramps). The valid set nearly supersets the old one (+7/−1 over
seeds 0–59). Verified with **pure defaults** (no env): seed 5 MAIN1→MAIN2 dist=223.2.

**⚠ Does NOT fix the mirror-split over-report class (§4.8/§12).** The offline oracle still models CLIF
*connectivity*, not ramp *detection*/quad coverage, so a blobby many-ramp seed can read VALID offline
yet split in-engine — e.g. **seed 42** (offline-valid under isoline (8,10), 17 fused ramps, MAIN1→MAIN2
unreachable in-engine). That is the junction-fusion problem the **source-side terrace carve** targets
(§4.4 "Current status", §11/§12), not a gradient issue. Isoline shortens/cleans the gradient; it does not
straighten a fused band. — **partially addressed in §4.11 (redundant-crossing prune).**

---

### 4.11 ★★★ SHIPPED: redundant-crossing prune (source-side, for the seed-42 over-report class)

The terrace carve (`RasterConfig.terrace_mode`, §4.4 "Current status": wall every level boundary, cut
ONE clean straight staircase per terrace-pair the graph says must connect) already emits *clean*
single-step staircases — but it cuts **~one per cross-level graph edge**, so a fragmented multi-terrace
seed carries FAR more crossings than connectivity needs. **Measured: seed 42 carves 14 ramps but only
7 are needed** to keep all 12 real bases connected under the exporter-faithful oracle — a ~2× surplus.
Every extra crossing is pure **in-engine detection/fusion surface**: the offline oracle (§4.4
`engine_cliff_grid` + `engine_components`) models CLIF *connectivity* but NOT ramp detection/quad
coverage, so a blobby many-ramp seed reads VALID offline yet seals a main in-engine (seed 42's "17
blobby ramps"). Seeds whose crossings are already the minimal articulation set (5/22/2: exactly 2
mirror pairs) connect in-engine.

**The prune** (`generate.rasterize.rasterize`, after ramp extraction; opt-out `NO_PRUNE_CROSSINGS`):
greedily WALL the most fusion-prone crossings **in MIRROR PAIRS**, keeping every real base connected
under the *exact* oracle `validate_map` uses — so offline yield can never drop:
- **Order = wide + DIAGONAL first.** Cardinal ramps detect reliably in every orientation; wide
  leftward-uphill diagonals are the marginal class (§4.6). Removing the fusion-prone ones first biases
  the SURVIVORS toward reliably-detected cardinals.
- **Mirror-paired removal** (partner = nearest rot180/mirror centroid; a center ramp self-pairs) keeps
  the map pixel-exact symmetric (verified 0-cell mirror disagreement).
- Only prunes a map already connected under the oracle (else it's rejected anyway).

**Offline A/B (120 seeds, rot180, isoline (8,10)):** STRICT yield **65 → 67/120** (+2 — removing a
redundant *unclean* ramp also clears the `ramp_cleanliness`/relief gates), ramps/map **mean 7.7 → 6.8,
median 8 → 6**. Seed 42: **14 → 8 ramps** (6 cardinal, 2 diagonal), still connected + symmetric. The
survivors are the minimal, well-separated, cardinal-leaning set gold maps use.

> **In-engine gate (run to confirm):** `PYTHONPATH=src .venv/bin/python scripts/mainconn_probe.py 42`
> — expect MAIN1→MAIN2 to now connect (far fewer crossings to fuse). The prune is a *source-side*
> guarantee the offline oracle is blind to (it can't see ramp detection), so mainconn_probe / the
> engine remains the only true gate for this class.

**Not a full cure.** A maximally-fragmented seed can still legitimately need many crossings (sweep max
ramps is still 14); those stay. Fully eliminating fragmentation is the deeper skeleton-level level-
ownership redesign (§4.4 architectural note, §12) — the prune is the bounded, offline-verifiable
first cut at the seed-42 class.

---

### 4.12 ★★★ SHIPPED: the `<ramp dir="…">` attribute must match SC2's OWN convention (real seed-42 root cause)

**Symptom.** On seed 42 (offline-valid, isoline (8,10)), MAIN1↔MAIN2 was unreachable in-engine with
CARDINAL ramps dropping out: the census showed the **west** cardinal ramps (and, less severely, other
non-north cardinals) were NOT in `game_info.map_ramps`, plus a few **phantom** ramps appeared. Wide
DIAGONAL ramps detected fine — overturning the earlier "wide leftward diagonal" detector-bias theory
(§4.6). Every terrain channel for a failing west ramp was CLEAN and a correct mirror image of a
detected east ramp (verified `scripts/_rampcmp.py window`, CLIF/SMAP/HMAP/Pnp all gold-faithful:
`64·72·80·88·96·104·112·128` staircase, monotone SMAP slope, `Pnp=0` on ramp cells, flanks walled).

**Root cause — our `dir` label contradicted the `u` vector.** The `<ramp>` entry carries BOTH a `dir`
integer AND the `base`/`mid` `u` uphill vectors. The engine's ramp detector keys off `dir`; if `dir`
disagrees with `u`, it mis-registers (or drops) the ramp. Tabulating `base.u` vs `dir` across **all 106
gold maps** (`scripts/_rampcmp.py`) recovers SC2's canonical convention:

| dir | `u` | compass |
|---|---|---|
| 0 | (0,−1) | N |
| 1 | (0,+1) | S |
| 2 | (−1,0) | **W** |
| 3 | (+1,0) | **E** |
| 4 | (−.707,−.707) | NW |
| 5 | (+.707,−.707) | NE |
| 6 | (−.707,+.707) | SW |
| 7 | (+.707,+.707) | SE |

Our `_RAMP_DIR_U` (export/`sc2map.py`) and `ir.RAMP_DIR_U` had the **cardinals scrambled** (`1=E, 2=S,
3=W`). Diagonals (4–7) already matched — *the sole reason diagonal ramps always detected*. But a west
ramp emitted `dir=3` (= SC2 **East**, 180° opposite its `u=(−1,0)`) → the detector dropped it entirely;
`dir=1`(east, 90° off) survived marginally; `dir=0`(north) is the one cardinal that accidentally
matched. Exactly the observed census. **Fix: reorder both tables to `0=N,1=S,2=W,3=E,4–7` diagonals.**
This is a *label-only* change to the emitted `dir` (the gradient snaps on the `u` vector, index-agnostic,
so the offline CLIF oracle / STRICT yield are byte-unchanged). Verified in-engine on seed 42:

- `ramp_detect_probe.py 42`: AUTHORED=8 → **DETECTED=8, no phantoms** (was fewer + phantoms), the
  previously-dead west ramp now `OK`.
- `mainconn_probe.py 42`: **MAIN1↔MAIN2 dist=207.8 REACH=True** (was unreachable), both naturals + 11/12
  bases reachable. One residual mirror base still split (see §12).

**★ Cardinal QUAD geometry — the trapezoid IS needed, but only *with* a gradient-top anchor (real fix for seed 42's last sealed base).**
Gold cardinal ramps use a distinct **trapezoid** quad (`base.w=5/h=1, mid.w=3/h=2, run=1, corner w=1`)
vs our diagonal-style rectangle (`base.h=0, run=2, base.w=mid.w, corner w=2`). A *naive* copy (keeping our
existing SHOULDER anchor + gold's `run=1`) tightened detection (all 8 `OK`) **but REGRESSED traversability —
MAIN1↔MAIN2 disconnected**, so it was first reverted with the note "the `dir` label was the fix."

That reverted attempt missed the **anchor**. Two coupled bugs made our cardinals marginal:

1. **Anchor sat on the flat clamped shoulder.** The isoline gradient (§4.10) CLAMPS every ramp cell beyond
   `span` lattice-steps to `hi_cliff`, so a cardinal ramp grows a big flat high shoulder (measured seed 42:
   **15 of 50 cells** clamped to 128). `build_ramp_list` anchored the quad at the *furthest-uphill* cell —
   i.e. **3 cells INTO that flat plateau**, off the actual `128↔112` gradient interface. A rectangle floating
   on the plateau makes the engine's detector anchor ambiguously → **parity-unstable drift** that seals one
   mirror base (seed 42 `BASE(74,149)`, detect drift 5.4 vs its tight twin 2.1).
2. **A gradient-top rectangle then spawns phantoms.** Moving the anchor to the gradient top with a plain
   rectangle (`base.h=0`) makes the quad *not touch* the high plateau → the engine spawns phantom ramps
   (seed 42: DETECTED 8→12, MAIN1↔MAIN2 dead). Gold avoids this with a **`base.h` lip** that reaches UP into
   the plateau.

**The working recipe = gold trapezoid + gradient-top anchor** (`make_ramp_entry` cardinal branch +
`build_ramp_list` cardinal anchor). For cardinals only (diagonals keep their shoulder-anchored rectangle,
which already detects+bridges): anchor `base_c` at the highest-projection ramp cell whose exported cliff is
`< hi_cliff` (the gradient top adjacent to the plateau, computed from the SAME `ir.ramp_cell_cliffs` the
exporter writes), and emit a trapezoid `base.w=5/h=1 → mid.w=3/h=2`, corners `w=1/h=2`. The `base.h=1` lip
sits on the interface so the ramp **detects tightly AND bridges**. Gold ships `run=1` for its SHORT/STEEP
3-cell cardinal gradient; our LONG/GENTLE 6-cell isoline staircase gets `CARD_RUN=2` (one cell more coverage
onto the gradient; `run=1` and `run=2` both connect seed 42 in-engine, `run=2` is the safer margin). All
`CARD_*` params are env-tunable for A/B.

**In-engine, seed 42 (pure defaults):** `ramp_detect_probe.py 42` → AUTHORED=8, all 8 `OK` (the two problem
cardinals N@(34,87) 5.1→2.2 and W@(59,147) 5.4→3.2), phantoms 4→1. `mainconn_probe.py 42` → **every base
reachable incl. `BASE(74,149)` (dist 198.4, was `None`)**, MAIN1↔MAIN2 198.8, and several base paths got
*shorter* (better anchoring opens direct routes). No regression on seeds 5/22/32 (all still fully connected).
The change is exporter-only (the quad never enters the offline CLIF oracle) so **STRICT yield is unchanged**.
Residual: **1 benign phantom** on seed 42 (connectivity holds); the durable cure for phantom-prone blobby
seeds is still the source-side terrace redesign (§4.4 "Current status").

---

## 5. `t3Terrain.xml` — ghost cliffs on flat maps

Plain XML: tileset, dims, `<cliffSetList>` (texture-set refs), and **`<rampList num="N">`** — a
list of **ramp geometry entries**. This is *not* just render art: it is the **authoritative
definition of where ground units may cross a cliff boundary** (see §4 — removing an entry blocks the
ramp; adding a synthesized one enables it). We author these via `sc2map.make_ramp_entry` +
`sc2map.set_ramp_list`.

**Gotcha:** these ramp meshes render **regardless of your (flattened) CLIF grid**. On a flat
interim map you'll see the template's original cliffs/ramps as "ghost" geometry floating on flat,
fully-walkable ground — looks like a high cliff, but SCVs walk right up it because the actual
terrain is flat. Fix: rewrite `<rampList ...>...</rampList>` to `<rampList num="0"/>` when
flattening (`sc2map.strip_render_relief`). Leave `<cliffSetList>` alone (harmless with a flat CLIF
grid; it's just texture references).

---

## 6. Texturing — `t3SyncTextureInfo` (RTXT) + `t3TextureMasks` (MASK)

This was the least-documented part. Model that actually works:

### 6a. `t3SyncTextureInfo` (magic `RTXT`) — per-cell texture layers
- Header: `magic[4]="RTXT"`, `u32 version` (101), `u32 sizeX`, `u32 sizeY`, then **one `u32`**
  (observed value `1`; purpose unclear — preserve it).
- **Texture name table:** N null-terminated ASCII strings back-to-back. There is **no reliable
  count field** — parse by reading consecutive printable null-terminated strings until the bytes
  stop looking like names (the per-cell region begins with small indices `0..7` = `0x00..0x07`,
  which are non-printable and cleanly terminate the scan). FrostLE has 8:
  `ZhakulDasGrass, IceWorld7, IceWorld4, Korhal8, ZhakulDasTiles, HavenRockSmooth,
  UlaanRoughRock, Typhon1`.
- **Per-cell region:** exactly `sizeX*sizeY*8` bytes = **8 bytes per cell**. In every stock map only
  **byte 0 varies** — it's the cell's base/dominant texture index into the name table; bytes 1–7 are
  `0`. RTXT is the "sync" texture record; the *visible* texturing is the MASK (§6b), which we key
  from the same class grid so they agree. (This is separate from the MASK's 8 layers — do not
  conflate the RTXT 8 bytes with the MASK 8 layers.) We set only byte 0.

> **8 is not a hard limit — it's FrostLE's one Terrain Type.** The wiki Map Properties page says a
> map may use up to **4 Terrain Types combining to as many as 16 textures** (one set = 8 textures;
> "can be maxed with only 2 sets"). FrostLE has exactly 8 because it uses a single texture set. So
> `n_layers` (MASK) and the RTXT name-table length can be **8–16 depending on the template** — good
> thing both are parsed dynamically (name-table scan for RTXT, `(fileSize-64)/(sizeX*sizeY/2)` for
> MASK) rather than hard-coded to 8. Also note the **Primary Texture Set** is the one the engine's
> terrain validator, creep visuals, and gravity key off of — don't reorder it.

### 6b. `t3TextureMasks` (magic `MASK`) — the ACTUAL render texturing (decode carefully!)
This drives what you *see*. Layout (SC2Mapster + round-trip-verified against templates):
- Header 64 bytes: `magic[4]="MASK"`, `u32 version` (102), `u32 unk`, `u32 sizeX`, `u32 sizeY`,
  padding to 64. `sizeX == sizeY == 8 * cellDim` (e.g. `1472 = 8*184`) → **8×8 sub-cells per cell**.
- Body: **`n_layers` layers, one per texture** (layer *i* == texture *i* in the `t3Terrain.xml`
  `<textureList>` / RTXT name order). `n_layers = (fileSize - 64) / (sizeX*sizeY/2)` (8 for FrostLE).
- Each layer is a `sizeX*sizeY` grid of **4-bit alpha** values (0..15) packed **2 per byte, high
  nibble first**, tiled in **64×64-pixel blocks** ordered +x then +y (`sizeX/64` blocks per row).
- **Verified:** in stock maps, layer *L*'s alpha ≈ 15 exactly where that cell's base texture is *L*,
  ≈ 0 elsewhere. The engine renders, per cell, whichever layer has full alpha.

> ⚠️ Earlier wrong guess: "4 interleaved `u8` channels." It is **not** — it's 8 nibble-packed,
> block-tiled layers. Decoding it as 4 `u8` channels *happens* to give the right byte count but
> scrambles everything.

### 6c. **Purple = no texture coverage** (the failure we hit)
Setting RTXT byte0 per cell and **zeroing the entire MASK** made the whole map render **purple**:
with every layer's alpha `0`, **no layer covers anything**, so the engine shows its purple
"no-texture" fallback. Conversely, filling the whole MASK with `0xFF` makes **all 8 textures fully
opaque everywhere** → a muddy blend, not per-level. Neither is right.

### 6d. Solid one-texture-per-cell recipe (what we ship)
Author the MASK directly: for each cell, set the chosen layer's alpha to `0xF` over its 8×8 block
and **every other layer to 0**. That yields exactly one solid texture per cell — no purple, no
blend. Also set RTXT byte0 to the same index so the "sync" record agrees with the render. See
`sc2map.author_texture_mask` + `sc2map.author_texture_info`. Round-trip check: re-decode the MASK,
take each cell's max-alpha layer, and assert it equals the RTXT byte0 grid.

### 6e. Texture-by-*intended*-level
Even on a flat export we texture by the IR's **intended** elevation (kept as `tiers_intended`
before the flatten collapse) so levels stay visually distinct: each tier → its own texture, ramp
cells → a dedicated texture (last index reserved). Indices are chosen with a spread
(`avail[0::2]+avail[1::2]`) so adjacent tiers don't both land on look-alike textures (e.g. two ice
variants). See `sc2map.build_texture_class_grid`. Texture *names* are template-specific, so map by
**position in the harvested name table**, not by name.

---

## 7. `MapInfo` — do NOT patch the playable area

`MapInfo` (binary) holds grid dims, playable-area rect `(l, b, r, t)`, players, etc. We tried to
**enlarge `playable_area`** so oversized maps fit — the client then **refused to load**
("A game has not been started yet"). The playable rect is **coupled to other internal fields**
(camera bounds and more) that validate against it; patching one in isolation corrupts the map.

This is corroborated by the editor guide (008 Map Properties): the editor exposes **Map Bounds**
and **Camera Bounds** as *two separate, lockable* things, and "Playable size = Full size − a
hard-coded buffer on each side." So playable area, full grid, and camera bounds are three linked
quantities — editing one raw offset without the others is what corrupts loading.

**Working approach instead — playable-aware template selection** (`exporter.pick_template`):
choose the smallest template whose *existing playable area* already contains our content, then
**center our content inside that playable region** via `off_x/off_y`. This guarantees every base
lands on legitimately playable (buildable) ground without touching `MapInfo`.

Reading the playable offset: `sc2map._mapinfo_playable_offset(mapinfo)` then
`struct.unpack_from("<4i", mi, off)` → `(l, b, r, t)`.

**Map name / rename — ✅ SOLVED.** The "FrostLE" text you see is a localized game-string
`DocInfo/Name` (008 Map Properties → Map Info **Name**), **not** in `MapInfo` and **not** the file
name. It lives in two places, both **decoupled from the coupled-field trap above** (they carry no
camera/playable-area fields), so editing them is safe:
- **`DocumentHeader`** (binary): a localized-string table. Layout: `magic "H2CS"` + fixed
  attributes + dependency strings, then **`u32 recordCount`**, then `recordCount` records of
  `{ u16 keyLen; char key[keyLen]; char locale[4]; u16 valLen; char value[valLen] }`. The
  `locale` tag is the locale code **reversed** (`enUS`→`"SUne"`, `frFR`→`"RFrf"`, `deDE`→`"EDed"`).
  Every locale gets its own `DocInfo/Name` record. Renaming = replace those values (record count
  unchanged, so nothing else moves; there is no total-size/checksum field to fix). Verified to
  parse to EOF with count-match on every template, and a rebuild touching nothing is byte-identical.
- **`<locale>.SC2Data\LocalizedData\GameStrings.txt`** (UTF-8, `key=value` per line): the per-locale
  text table the client loads at runtime — set the `DocInfo/Name=` line.

We rewrite **both**, for **all** locales, to the IR's `map_name` (`ExportConfig.rename_map`, on by
default; `map_name` overrides the displayed name). See `sc2map.set_document_header_name` /
`sc2map.set_gamestrings_name`. This also avoids sharing the template's asset cache (§2).

---

## 8. `Objects` (XML) + resource placement for `python-sc2`

- `Objects` is XML: start locations are `ObjectPoint Type="StartLoc"`; resources are `ObjectUnit`
  with `UnitType` `MineralField` / `VespeneGeyser`.
- `python-sc2`'s expansion/townhall finder is picky. A townhall spot must be **≥6 tiles from every
  mineral, ≥7 from every geyser**, and 4–8 tiles from the cluster centroid. If resources crowd the
  intended townhall pocket, `_find_expansion_locations` throws `min() arg is an empty sequence` and
  the bot crashes on init (map still *loads*).
- Values that worked (`rasterize.py`): mineral ring `6.5`, gas ring `7.5`, base pad half-width
  `9.5`, with a `min_r` snap constraint keeping resources ≥ townhall clearance (6 mineral / 7 gas)
  from the center, and outward-only fallback search.

---

## 9. Buildability = flat SMAP + non-void CLIF (reachability matters too)

A cell is buildable in-engine iff its CLIF ≠ 0 (not void) **and** its SMAP neighborhood is flat
(matches a harvested tier). Two real causes of "can't build here" we hit:
1. **Void in the pad:** base pad overlapping CLIF-0 cells (edge bases near the template void, or
   IR non-walkable cells inside the pad). Fix = keep bases well inside playable area (§7) and
   guarantee a flat buildable pad radius per base in the rasterizer.
2. **Unreachable, not unbuildable:** naturals appeared "unbuildable" because a cross-level wall
   made them unreachable; the ground itself was fine. The flat interim (§4) resolves this class.
3. **Ghost void holes/gaps on flat maps:** the rasterizer's cliff-cleanup carves walls to separate
   different elevations. Once the export flattens everything to one tier those walls are
   meaningless but linger as void holes *and thin gaps* that fragment the playable area (and *look*
   like unbuildable terrain). Enclosed holes fill trivially, but most gaps are 1–3 cells wide and
   *connect to the outer void*, so plain hole-filling misses them. Fix = morphological **closing**
   (`exporter._heal_flat_terrain`, `scipy.ndimage.binary_closing` with a small disk, then
   `binary_fill_holes`) → welds gaps up to ~2·radius wide without touching the large outer void or
   pinching chokes. Closing only adds cells (never disconnects) and a symmetric SE on a symmetric
   map preserves symmetry. Filled cells inherit the *nearest intended level* so per-level texturing
   stays consistent (no stray grass in a tiled plateau).

Verify buildability the way the engine does, not via our IR grid: `scripts/debug_ingame.py`
reports the buildable 5×5 townhall pocket near every base from `game_info.placement_grid`. A
misleading gotcha: measuring "buildability" in a fixed radius around a base center in the *IR* can
capture a neighbouring plateau across a wall and look bad even when the base's own pad is a clean,
fully-buildable single level — trust the engine's `placement_grid`.

---

## 10. Quick reference — files we touch

| Internal file | Magic | We author? | Notes |
|---|---|---|---|
| `t3SyncCliffLevel` | `CLIF` | yes | u16 level grid; 0 = void |
| `t3SyncHeightMap` | `SMAP` | yes | `int16 height`+`u16 mask`/cell (height/256=level); drives `terrain_height` |
| `t3HeightMap` | `HMAP` | yes | u16×3 (adjust/base/mask) render height; cosmetic |
| `t3SyncTextureInfo` | `RTXT` | yes | 8 B/cell; byte 0 = base texture index (rest 0) |
| `t3TextureMasks` | `MASK` | yes | 8 layers, 4-bit alpha, 64×64-block-tiled @ 8× res |
| `t3Terrain.xml` | (xml) | yes | **`<rampList>` is authoritative for ramp walkability** (§4); `make_ramp_entry`+`set_ramp_list`. Zero it (`strip_render_relief`) on flat maps to kill ghost cliffs |
| `CellAttribute_Pnp` | (raw) | yes | painted pathing/placement **bitfield**, `0x00`=open … `0xff`=blocked, 4-byte header + `sizeX*sizeY` body; ramp cells are `0`. `author_pnp`. The editor's Pathing Layer exposes 4 paint types (No-Pathing / Ground / No-Building / No-Burrowing) → this byte is almost certainly those flags OR'd together, not a plain open/blocked scalar (§13g) |
| `Objects` | (xml) | yes | start locs + resources |
| `MapInfo` | (bin) | **NO** | patching playable area breaks loading |
| `t3SyncPathingInfo` | `PATH` | **NO** | `0x00` pass / `0x02` ramp / `0x23` block, but editing it has **no effect** in-game (pathing is derived) |
| `t3CellFlags` | `LFCT` | yes | `0x03` "punches a hole" in world terrain → renders as black/deep-lava voids on flat maps; `clear_cell_flags` zeroes the body |
| `DocumentHeader` | `H2CS` | yes | localized-string table (`u32` count + `{keyLen,key,locale[4]=rev,valLen,val}` records); rename via `DocInfo/Name` (§7). `set_document_header_name` |
| `GameStrings.txt` (per locale) | (utf-8) | yes | `key=value` text table; rename via `DocInfo/Name=` line (§7). `set_gamestrings_name` |
| `t3FluffDoodad`/`t3VertCol`/`t3Water` | — | no | inherited; fine to leave |

---

## 11. Diagnostic scripts

- `scripts/ramp_probe.py` — minimal 2-plateau map; showed gradient-only ramps don't path (§4.1 #1).
- `scripts/control_probe.py` — repack FrostLE unmodified; proves repack keeps ramps walkable (#2).
- `scripts/ablation_probe.py` — delete `rampList` entries on intact terrain; proves the entry
  controls walkability (#3).
- `scripts/synth_ramp_probe.py` — replace `rampList` with a from-scratch generated entry; proves we
  can synthesize walkable ramps (#4).
- `scripts/rampstep_probe.py` — sweep the cliff STEP (8/16/32/64 = 8/4/2/1 ramp cells) with a
  covering quad + a `--noquad` control; uses the engine `pathing_grid` + a real Marine move to prove
  the quad (not the gradient run) controls walkability, so ramps can be short (§4.5 #4).
- `scripts/rampwide_probe.py` — cover a wide ramp band (W=16/32) with ONE wide quad vs TWO quads vs
  one narrow quad; spawns a row of Marines across the full width to prove one quad floods walkability
  to any band width, so wider ramps need a wider band not more quads (§4.5 #7).
- `scripts/conn_probe.py` — connectivity BFS that correctly counts ramp cells as passable.
- `scripts/_rampmut.py` — copy FrostLE, mutate ONE ramp's `rampList` quad width (single variable),
  repack via StormLib, read `map_ramps` back. Proved an oversized quad balloons the ramp + spawns a
  phantom (§4.8); `make_ramp_entry` at gold width reproduces the gold ramp exactly.
- `scripts/isoline_ramp_probe.py` — synthesize a code-authored locked-8 TRUE-ISOLINE diagonal
  staircase (short fixed run, variable width) + gold small quad; in-engine `pathing_grid` + Marine
  move + no-quad/rank-order/run-sweep controls. Proved the §4.10 gradient walkable at gold-short run.
- `scripts/isoline_yield_ab.py` — A/B the STRICT offline yield of the isoline vs rank-order gradient
  across carve runs (shows isoline (8,10) 65/120 > rank-order (11,14) 59/120). Honors `RANK_ORDER_RAMPS`.
- `scripts/gold_ramp_study.py` — measure gold ramp geometry (authored CLIF band + `rampList` from
  `gold_maps/*.SC2Map`, walkable run/width + per-cell step from `dataset/*/_raw.npz`, 106 maps) to
  derive the traversable-ramp invariants and the run(fixed)/width(free) variability split (§4.7).
- `scripts/_symcheck.py` — read the EXPORTED `.SC2Map` back and test each terrain channel (CLIF/
  SMAP/HMAP/Pnp) for rot180 symmetry, incl. a direct mirror-CELL SMAP comparison at a ramp pair.
  Proved every channel is mirror-symmetric, isolating the asymmetry to the engine detector (§4.6).
  NB decode SMAP/HMAP as **169² vertex** grids (`u4` / `u2×3`), not the 168² cell grid.
- `scripts/mainconn_probe.py` — MAIN1 reachability to every base via engine `pathing_grid`+`map_ramps`
  flood (authoritative) and `query_pathing`; honors `RAMP_CHOKE`/`RAMP_RUN` env to A/B ramp sizing.
- `scripts/debug_ingame.py` — dump engine grids, buildable pockets, engine-detected ramps.
- `scripts/editor_bake.py` — (obsolete) prototype editor automation; no longer needed (§4.3).
- `scripts/export_map.py` — generate → validate → export → independent re-verify.

---

## 12. Open items / not yet solved headless

- **Walkable ramps / real high ground** — ✅ **SOLVED headless** (§4): author diagonal cliff
  gradients + a synthesized `<rampList>` entry per ramp. Remaining work is *pipeline* (wire into the
  exporter: turn off `flatten_terrain`, emit an entry per skeleton ramp), not research.
- **Map rename** — ✅ **SOLVED** (§7): the export is renamed to the IR's `map_name` by editing the
  localized `DocInfo/Name` in `DocumentHeader` + every `GameStrings.txt` (both decoupled from the
  `MapInfo` bounds trap). On by default (`ExportConfig.rename_map`).
- **Relief-export yield** — relief+ramps are correct when they pass, but only a minority of raw
  seeds rasterize into a fully clean-ramp-connected map, so relief relies on rejection sampling
  (`--valid-only`); flat is the shipped default (§4.4).
- **Headless ramp-slope ceiling** — ✅ **RESOLVED by the true-isoline gradient (§4.10).** The old
  rank-order DIAGONAL ramps needed `run ≥ 8·√2 ≈ 11.3` (shorter → Δ16 islands the high plateau); the
  now-default isoline gradient locks +8/axis-cell so a full level is a FIXED ~8-cell run, and we ship
  `ramp_run_range=(8,10)` (STRICT yield 65/120 vs the old 59/120; seeds 5 & 22 connect in-engine).
  Stretching the exporter quad is still HARMFUL (§4.5); offline oracle is still the STRICT `≤8` rule.
- **Wide-ramp detection failure** — ✅ **ROOT-CAUSED + fixed** (§4.6): the engine's ramp DETECTOR is
  orientation-biased and drops WIDE (≳8) diagonal ramps in leftward-uphill orientations, sealing one
  mirror main even though every exported channel is provably rot180-symmetric. Fixed source-side by
  capping width (`ramp_choke_range=(4,6)`); the offline oracle cannot see this (it models CLIF
  connectivity, not ramp detection).
- **Mirror-half split on some seeds** — a few seeds connect each main to its own natural/half but not
  the two halves to each other. The isoline gradient (§4.10) fixed several (seed 5 now connects); the
  **redundant-crossing prune (§4.11)** attacks the blobby/many-ramp class; and the **`dir`-attribute fix
  (§4.12)** fixed the real seed-42 root cause — cardinal (esp. west) ramps were dropped because the
  emitted `<ramp dir>` disagreed with the `u` vector, so **MAIN1↔MAIN2 now connects in-engine** (seed 42
  dist=207.8) with all 8 ramps detected and no phantoms.
  - **Residual (seed 42) — ✅ FIXED via the cardinal trapezoid + gradient-top anchor (§4.12).** The last
    split base `BASE(74,149)` (served by a marginally-anchored WEST cardinal ramp) now reaches (dist 198.4)
    and MAIN1↔MAIN2 connects (198.8) with all 8 ramps detected. Root cause was NOT engine parity but our
    exporter anchoring the cardinal quad on the flat clamped shoulder (drift) + a rectangle that phantoms if
    moved to the gradient top; the gold-style trapezoid `base.h` lip at the gradient top fixes both. One
    benign phantom remains (connectivity holds). Durable end-state for phantom-prone blobby seeds is still
    the **skeleton-level level-ownership redesign** (§4.4). In-engine gate: `scripts/mainconn_probe.py 42`.
- **Oversized seeds** — can still exceed the largest template's playable area (§7).
- **Edge-adjacent expansions** — occasional partly-void townhall pockets (§9).
- **Custom preview image** — still shows the template's; a headless preview needs a square 24-bit
  TGA (§14). (Map rename itself is done, §7.)

---

## 13. Galaxy Editor mental model (from S2 Editor Guides) — why the format is shaped this way

Concept-level notes from <https://s2editor-guides.readthedocs.io> that explain constraints above
and inform strategy. These are about the *editor*, but they tell us what the file layers mean.

### 13a. Map types: keep ours **Melee** (004 Map Types)
A map bundles terrain + data + code in one file. **Melee** maps carry only terrain/base-layout plus
a stock melee ruleset; **any data change flips the map to Arcade.** Implication: our template-shell
approach (edit only terrain/objects layers of a real melee map, never its data/triggers) keeps the
output a *valid melee map* — exactly what we want for competitive 1v1. Don't touch data catalogs.

### 13b. Texture Set = 8 textures **+ cliffs + lighting + sounds** (005, 008)
The "Texture Set" (a.k.a. Terrain Type) is a fixed palette of **8 ground textures** and *also*
defines the cliff styles, creep visuals, lighting and ambient sound for that tileset. This is why
(§6) there are exactly 8 texture layers and why cliff look is template-bound. "Initial Texture"
paints the whole map one texture at creation; "Base Height" + optional random height set the
starting surface. Our per-level texturing (§6e) is essentially re-painting from that palette.

### 13c. Map/Camera bounds + hard-coded buffer (008) → see §7
Playable size = full size − a hard-coded per-side buffer; camera bounds are a *separate* lockable
rect. Confirms why raw `MapInfo` playable-area edits break loading (coupled fields).

### 13d. Terrain is **7 layers**; cliffs/ramps live *on top of* the heightmap (018)
The Terrain Editor layers: (1) **Terrain** — raise/lower height, paint ground, and *add water,
cliffs, and ramps on top*; (2) Units/buildings/destructibles; (3) Doodads; (4) Points;
(5) Regions; (6) Cameras; (7) **Pathing Layer**. So cliffs/ramps are an editor construct applied
over the heightmap. The ramp "construct" the editor writes is the **`<rampList>` entry** in
`t3Terrain.xml` (§4) — and we now reproduce it headless, so no editor bake is required after all.

### 13e. RESOLVED: the authoritative ramp data is the `<rampList>` entry, not a painted layer
Earlier lead (kept for history): we suspected a `PaintedPathingLayer`/`CellAttribute_Pnp` file was
the authoritative pathing, since editing `t3SyncPathingInfo` (PATH) had no effect. We *did* decode
`CellAttribute_Pnp` (a per-cell open/blocked bitfield, now authored via `author_pnp`), but the
ablation/synthesis experiments in §4 showed **`Pnp=0` is necessary-but-not-sufficient**: the piece
that actually makes a cliff boundary walkable is the **`<rampList>` entry in `t3Terrain.xml`**.
Removing an entry blocks an otherwise-identical ramp; adding a from-scratch generated entry
(`make_ramp_entry`) makes one walkable (`query_pathing` = 6.0). The §4 negative conclusion is
**overturned** — headless ramps work.

### 13f. Terrain Layer brushes (020) — confirms our elevation/texture model
The Terrain Layer's brushes each map onto a file layer we author, and their described behaviour
corroborates the encodings above:
- **Height brush ≠ Cliff brush.** "All height features made with the Height brush remain **pathable**
  regardless of how outlandish they appear; you make pathing distinctions with the cliff and pathing
  tools." This is exactly the SMAP/HMAP (render/gameplay height, *not* barriers) vs CLIF+`rampList`
  (the actual cliffs/ramps) split from §3/§4 — smooth height never blocks movement; only a cliff
  level jump does.
- **Cliff brush = 4 levels, one step at a time.** Raise/Lower moves terrain **one** cliff level and
  clamps at the top/bottom → only **4 levels** exist (see §3a cap). "Add Ramp creates a ramp between
  terrain levels **if possible**" — i.e. only between adjacent levels, matching our single-level-ramp
  rule (§4.4). "Remove Ramp" is the inverse of stripping a `<rampList>` entry (§4/§5).
- **Texture brush = alpha-mixed 8-texture palette; "Uniform Texture" = locked alpha.** Confirms the
  MASK's per-layer alpha model (§6b) and that our solid one-texture-per-cell recipe (§6d, every layer
  0 except the chosen one at `0xF`) is literally the editor's "Uniform Texture" op — no variation.
- **Terrain Objects / Water / Foliage / Creep / Lighting / Roads** are separate brushes writing other
  layers (terrain-object set-pieces sorted by cliff level; water regions at a set height; creep &
  no-creep zones; 4 paintable lighting regions). We **inherit** all of these from the template
  (§2/§10) and don't author them — noted only so we know what the untouched layers are.

### 13g. Pathing Layer (026) — the 4 pathing types behind `CellAttribute_Pnp`
The editor paints **four** pathing types as colored overlays, plus dynamic zones:
- **No Pathing** (red) — nothing may path through (used e.g. to stop units being *dropped* onto
  decorative high ground / tower tops).
- **Ground** (green) — flagged as ground for certain ability/data rules.
- **No Building** (yellow) — blocks placing structures (but units still walk).
- **No Burrowing** (blue) — blocks burrow.
- Dynamic: **No-Fly zones** (affect flyers) and a flood-fill No-Pathing "Pathing Fill".

Implication for us: `CellAttribute_Pnp` (§10) is very likely a **per-cell OR of these flags**, not a
plain open/blocked scalar — which fits the observed `0x00`→`0xff` range. Today we only ever author
`0x00` (fully open, incl. ramp cells), so we've never needed to separate the bits; but if we ever
want **no-build decoration**, **buildable-but-unwalkable pads**, or **drop-proof high ground** headless,
this is the file and these are the bits to decode. (Reminder from §10: editing `t3SyncPathingInfo`
/`PATH` has *no* in-game effect — pathing is derived from CLIF+`rampList`+Pnp, not from PATH.)

---

## 14. Engine/editor hard limits & size constraints (wiki.gg Map Properties + guide 005)

Concrete numeric limits that constrain **template selection** (§7) and **generator validation** (§9).
These are editor/engine limits, so they hold headless too:

- **Grid dimensions:** 32–256 per side, in **increments of 8** (guide 005). Our authored dims must
  land on that lattice; templates already do.
- **Map size bounds:** max **256×256**, min **32×32**. Min **camera bounds** = 7 (left/right), 4
  (top/bottom); min **playable area** = **15×9** (wiki Map Properties → Bounds).
- **Reducing Map Bounds deletes any terrain outside the new bounds and clears the undo buffer.** Yet
  another reason our pipeline **never edits `MapInfo` bounds** (§7) and instead does playable-aware
  template selection + centering. Enlarging bounds is the trap that broke loading; shrinking would
  silently discard authored terrain.
- **Textures:** up to **4 Terrain Types → up to 16 textures** total (8 per set); the **Primary
  Texture Set** drives the terrain validator, creep, and gravity (see the §6a note). Don't assume 8.
- **Melee vs Arcade & multiplayer data (guide 008 / §13a):** Publishing Options set the map to
  Melee/Custom vs Arcade; a Melee map's **"Automatically Add Multiplayer Data"** pulls the melee
  ruleset dependencies **at game launch**. Our template-shell inherits the template's melee data
  verbatim and touches no data catalog, so the output stays a valid melee map — reconfirming "never
  edit data/triggers" (§13a).
- **Assets are TGA, not PNG/JPG (wiki File Formats).** In-game images (incl. the map **Preview
  Image**, which must be a *square 24-bit* TGA) use `.tga`; `.png`/`.jpg` are rejected by the engine
  (lone exception: loading-screen layouts can reference JPEG). Relevant to the still-open custom
  **preview image** work (§12) — it would need a TGA, not a PNG. (Map rename itself is done, §7.)
- **Description limits (wiki):** basic description ≤ 79 chars, extended ≤ 300 chars — if we ever
  author map metadata headless.

> Not useful for us (recorded so we don't re-chase them): the `File_Formats` wiki page is just a
> high-level catalog (MPQ / SC2Map / SC2Replay / dds / m3 / m3a / ogv / tga) with no new byte-level
> detail beyond §1; and the entire mkdocs "Getting Started" is editor/VSCode/Galaxy-scripting setup
> (Data module, `.SC2Layout` UI, FontStyles, trigger bootstrapping) that has no bearing on headless
> file authoring.
