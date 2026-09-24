# StarCraft II Generative Map Generator — V1

A constrained, data-informed generator for one-use competitive 1v1 maps: learn the shape of normal
competitive maps, generate a fresh valid map per match, and export it into StarCraft II.

> **Core principle:** the training maps define the *center* of the design space, not its boundary.
> V1 produces mostly recognizable competitive-style maps with moderate structural variation. Hard
> gameplay and serialization rules are enforced in code; the SC2 exporter is deterministic and works
> from a normalized internal representation.

Format/engine details live in [`docs/SC2_MAP_FORMAT_FINDINGS.md`](docs/SC2_MAP_FORMAT_FINDINGS.md).
Section 15 below describes the current implementation.

## 1. Goals

Generate a new 1v1 map for every match that players immediately recognize as a valid SC2 map, but
whose timings and expansion patterns can't be memorized. Priors come from ~350 historical and current
competitive maps.

- Anchor each player with a familiar main and natural.
- Every later expansion is a generic BASE (no "third", "fourth", …); how it's used emerges from
  geometry.
- Learn distances, region sizes, path widths, base density and topology from the corpus; allow
  controlled deviation from those norms.
- Guarantee hard constraints in code, not through learned behavior.

## 2. Non-goals for V1

No end-to-end neural generator; no strategic "interesting choice" scoring; no asymmetric spawns; no
special features (islands, gold bases, vision blockers, unusual destructibles); no procedural art
pass (tileset/doodads/lighting stay template-driven); the model never learns the file format.

## 3. Architecture

```
Competitive SC2 maps (~350)
  → extract + normalize
  → learn priors / distributions
  → procedural topology + geometry generator
  → hard-constraint / repair pass
  → validation + plausibility check
  → deterministic SC2 exporter
  → fresh .SC2Map for the match
```

Generation is hierarchical: skeleton → region geometry → paths → elevation → resources. It never
generates individual terrain cells independently.

## 4. Internal representation (MapIR)

Two views of the same map, independent of SC2 serialization.

**Semantic graph**

```
MapGraph:   main_p1, natural_p1, main_p2, natural_p2, bases[], regions[], connections[]
BaseNode:   position, elevation, region_area, resource_layout, distance_from_own_main,
            distance_from_enemy_main, approach_count, approach_widths
Connection: source_region, target_region, path_length, minimum_width, mean_width,
            curvature, elevation_changes
```

**Raster channels:** walkable, elevation level, ramp mask, buildable, base locations, mineral
locations, geyser locations. No SC2 terrain values appear here; the exporter translates them.

## 5. Dataset processing

For each corpus map: load terrain/pathing and objects; identify starts, mains, naturals, later bases,
ramps, elevations, regions and connections; normalize orientation and verify symmetry; convert to
the graph + raster; extract a tabular feature vector. Features include map size and playable ratio,
main/natural/base areas, main→natural distance and offset, rush distance, base count and spacing,
path length/width/curvature by context, region sizes, elevation proportions, and graph-motif
frequencies (forks, loops, alternate routes, center connections).

## 6. Hard rules vs. learned priors

| Feature | Treatment |
|---|---|
| Two mirrored player starts; map symmetry | Hardcode |
| Main / natural geometry, main–natural wall-gap | Mostly hardcode (limited learned variation) |
| Later expansion labels | Not encoded — all generic BASE |
| Number/location of later bases, path widths, map size, region sizes | Learn + sample (within hard minimums / supported ranges) |
| Connectivity, ramp/cliff legality, resource legality | Hard validate |
| SC2 serialization | Hardcode, never learned |

## 7. Runtime generation

1. Sample global parameters (dimensions, openness, base count) from learned distributions.
2. Place the P1 main near an edge/corner; place the natural from learned distance/angle priors.
3. Mirror P2.
4. Sample and place generic BASE nodes (mirrored).
5. Build the connectivity graph from corpus motifs (direct paths, forks, loops, center, flanks).
6. Assign region geometry; rasterize edges into corridors of sampled width/length/curvature.
7. Union into the walkable surface; derive boundaries.
8. Assign elevation; insert ramps only where connectivity needs a level change.
9. Place resources (deterministic legal layouts).
10. Validate and repair; otherwise reject and regenerate.
11. Plausibility filter against the corpus (wider band than the corpus itself).
12. Export.

## 8. Controlled weirdness

`competitive prior → sample → broaden by weirdness factor → validate`. Acceptable variation: a more
open or constrained center, an unusually wide/narrow non-critical lane, a base closer to center, an
uncommon base count, an unusual fork/loop, a larger/smaller elevation region. V1 doesn't score
whether deviations are strategically interesting — only that they stay valid and comprehensible.

## 9. Validation requirements

Valid mains and naturals for both players; symmetric sides; a ground path between players; every
base reachable with enough buildable space; no critical path under the minimum width; legal
main/natural entrance geometry; no resource overlaps or resources on illegal terrain; ramps join
adjacent levels only; no accidental traps or islands; key metrics (rush distance, playable area,
base density) within broad bounds; the exported map loads in SC2.

## 10. Plausibility model

With ~350 maps, use simple statistics: a standardized 50–150-feature vector, empirical distributions
for key scalars, conditional sampling where dependencies are obvious, and a nearest-neighbour
outlier score. The runtime acceptance band is deliberately wider than the corpus.

## 11. Export strategy

The exporter is a deterministic compiler from MapIR onto a known-good SC2 map used as a shell:
walkability/elevation/ramps → terrain/pathing/cliff data; bases/resources → Objects; everything the
generator doesn't model is inherited. It supports a finite terrain vocabulary: void, walkable
levels, ramps between adjacent levels, and buildable vs. non-buildable walkable cells.

## 12. Success criteria

A fresh map is generated automatically before each match, loads and plays in SC2, has a familiar
main/natural that supports standard openings, has later bases and routes not tied to memorized
templates, reads as competitive-style, is sometimes noticeably unusual without being broken, and
invalid candidates are rejected fast enough for per-match use. The architecture must accept later
extensions (strategic scoring, mutation operators, special features) without a rewrite.

## 13. Milestones

1. Dataset extractor 2. Feature analysis 3. Skeleton generator 4. Geometry rasterizer
5. Validator 6. Plausibility filter 7. SC2 exporter 8. Per-match generation loop

## 14. Deferred beyond V1

Strategic scoring, race-specific balance, bot simulation, learned quality models / GNNs, aggressive
mutation operators, asymmetric maps, special bases/islands/destructibles/towers/vision blockers,
procedural visuals, telemetry-driven adaptation.

---

## 15. As-built (current implementation)

Authoritative where it differs from the spec above.

### 15.1 Status

| Milestone | Status |
|---|---|
| M1 Dataset extractor | done |
| M2 Feature analysis | done — 109-map feature vectors in `dataset/_features.json` |
| M3 Skeleton generator | done |
| M4 Geometry rasterizer | done |
| M5 Validator + resource placement | done |
| M6 Plausibility filter | done — broadened per-feature bands + nearest-neighbour distance |
| M7 SC2 exporter | done — loads and plays in SC2 (flat default; relief opt-in) |
| M8 Per-match generation loop | pending |

Learned priors are wired but optional; the generator currently runs on default priors plus a
weirdness factor.

### 15.2 Symmetry

One of `rot180`, `mirror_lr`, `mirror_ud` per map (or `mixed` across a batch). Every stage is
symmetry-aware. After rasterization the terrain (walkable, elevation, ramps) is snapped to a
pixel-exact mirror about the playable centre; otherwise small drift can make one spawn's ramp work
while its mirror doesn't.

### 15.3 Skeleton (`generate/skeleton.py`)

A constructive symmetric graph. Node kinds: MAIN, NATURAL, BASE (real bases); ROOM (routing-only
plaza); JUNCTION (a Y-split pseudo-node at the midpoint of an A–B link, tapped by a third corridor).
ROOM/JUNCTION are never emitted as IR bases.

- Each node has one scalar width; an edge is no wider than its narrower endpoint.
- MAIN has exactly one edge (a narrow choke to its NATURAL); NATURAL has two; every other node ≥ 2.
- **Levels are owned here** (`_assign_levels`): MAIN = 2, NATURAL = 1, others drawn from {0,1,2} in
  one vectorized call on a dedicated RNG. (Interleaving `choice(p=…)` with other draws on one
  generator produced long constant runs.) Edges are rebuilt so every linked pair differs by ≤ 1 level.
  JUNCTIONs whose parents land on different levels are deleted and the survivors reconnected, because
  such a junction forces two ramps into one unauthorable blob. The rasterizer only reads these levels.

### 15.4 Rasterizer (`generate/rasterize.py`)

- Boxy terrain: each node is a rectangular room; each edge a flat-ended corridor, giving the
  choke → room → choke rhythm. Interior rooms grow until walkable/playable ≈ 0.50–0.62; MAIN/NATURAL
  stay compact.
- **Terrace carve** (`terrace_mode`, default on): every level boundary is a cliff wall, with one clean
  straight staircase cut per terrace pair the graph needs connected. Same level → flat passage; one
  level apart → exactly one ramp. Ramp cells are locked so cleanup and repair can't re-widen them.
- **Ramp sizes:** `ramp_choke_range=(4,6)` width, `ramp_run_range=(8,10)` run. That run is the gold
  fixed run for the isoline gradient. Widening the band lowered yield and broke main→natural links
  in ways the offline oracle couldn't see.
- **Redundant-crossing prune:** remove the most fusion-prone ramps (wide diagonals first) in mirror
  pairs while every base stays connected under the validator's oracle. Fewer ramps means less
  in-engine detection risk. Opt-out `NO_PRUNE_CROSSINGS`.
- Every real base gets a protected flat buildable pad. A mirror-safe repair pass guarantees a ground
  path to every base (carving from pad edges, preferring same-level links). Cliff cleanup walls any
  non-ramp contact between different levels.
- Resources: 8 minerals on a 6.5 arc facing away from the base's corridors, and 2 geysers at 7.5, all
  on the townhall's floor. This satisfies `python-sc2`'s expansion finder (findings §7).

### 15.5 Validator (`generate/validate.py`)

`validate_map(mapir) → ValidationReport`, split into hard failures (reject) and soft warnings.

- **Hard:** two mains and two naturals; main-to-main and all-base connectivity under the
  engine-accurate oracle (predicted CLIF, 4-connected `|Δcliff| ≤ 8`, same-ramp bridge for
  gold-profile cardinals); buildable pads; ramp cleanliness (one plateau per side); legal resources.
- **Soft:** rush distance, natural↔natural choke width, terrain symmetry, base spacing, openness.

Current STRICT offline yield is ~36/60 on seeds 0–59. The oracle can't see ramp *detection*, so final
confirmation is in-engine (`scripts/mainconn_probe.py`).

### 15.6 Exporter (`export/`)

A template-shell compiler: pick the smallest real template whose playable area contains the map,
centre the map in it, overwrite the authored layers in place with StormLib, and repack. Output is
re-read with `mpyq` to verify. See the findings doc for every file format.

- **Flat (default, `flatten_terrain=True`):** one tier, walkable geometry preserved, textured by
  intended level. This is the reliably playable path.
- **Relief (`author_ramps=True`):** real levels and walkable ramps, fully headless — a CLIF gradient
  plus one gold-style `<rampList>` quad per ramp (findings §3–4). Correct when it validates; pair with
  `scripts/export_map.py --valid-only` to rejection-sample seeds.
- Also authored: SMAP/HMAP from harvested template tuples, textures, Pnp, start locations and
  resources, and the map name. `MapInfo` is never touched.

### 15.7 Known gaps

- M8 per-match loop.
- Relief yield: some layouts can't be built with clean single-step ramps. The durable fix is fully
  decoupling ramps from graph edges.
- Offline oracle can over-accept blobby seeds that split in-engine.
- Cardinal ramps narrower than gold (~5 vs 8–12); ramp flanks are voided where gold uses plateau; the
  ramp texture paints the flat ends.
- Oversized seeds that exceed every template; occasional partly-void townhall pockets at map edges;
  no custom preview image.
