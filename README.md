# StarCraft II Generative Map Generator V1 Design Specification

A constrained, data-informed generator for one-use competitive 1v1 maps

**Version 1 scope: learn the shape of normal competitive maps, generate a fresh valid map per match, and export it into StarCraft II.**

> **Core principle: the training maps define the center of the design space, not its boundary. V1 should produce mostly recognizable competitive-style maps while permitting moderate structural variation. The SC2 exporter remains deterministic; the generator works on a normalized internal representation.**

## 1. Product Goal

**V1 should generate a new 1v1 StarCraft II map for every match.** Players should immediately recognize the map as a valid SC2 environment, but should not be able to rely on memorized map-specific timings or expansion patterns. The system should learn broad competitive-map conventions from an available corpus of approximately 350 historical and current competitive maps, then sample and recombine those conventions procedurally.

- Anchor the player with a familiar main base and natural expansion.
- Generate all later expansions as generic BASE nodes rather than “third,” “fourth,” etc.
- Learn normal distances, region sizes, path widths, base densities, and topology tendencies from the competitive corpus.
- Allow controlled deviations from those learned norms so each match can require adaptation.
- Guarantee hard gameplay and serialization constraints through code, not through learned behavior.

## 2. Explicit Non-Goals for V1

- No end-to-end neural map generator such as a diffusion model.
- No requirement to optimize for “interesting choice” or formal strategic diversity scoring.
- No asymmetric spawn geometry; V1 should use 180-degree rotational symmetry.
- No complex special features such as islands, gold bases, vision blockers, unusual destructibles, or exotic resource rules unless added later as simple extensions.
- No procedural art pass. Tilesets, doodads, lighting, and other visual presentation can remain template-driven or minimal.
- No need for the model to learn SC2 file-format details. Export is deterministic.

## 3. High-Level Architecture

Competitive SC2 Maps (~350)         |         v Extract + Normalize         |         v Learn Priors / Distributions         |         v Procedural Topology + Geometry Generator         |         v Hard Constraint / Repair Pass         |         v Validation + Competitive-Plausibility Check         |         v Deterministic SC2 Exporter         |         v Fresh .SC2Map for the match

**The generator should be hierarchical.** It first decides the map skeleton, then region geometry, then paths, then elevation, then resource/object placement. It should not start by generating individual terrain cells independently.

## 4. Normalized Internal Representation

The internal representation should be independent of SC2 serialization. V1 needs two complementary views of the same map:

- A semantic graph for bases, regions, and connections.
- A raster/grid representation for exact walkability, elevation, ramps, buildability, and later SC2 export.

### 4.1 Semantic graph

MapGraph   main_p1: BaseNode   natural_p1: BaseNode   main_p2: BaseNode   natural_p2: BaseNode   bases: BaseNode[]   regions: Region[]   connections: Connection[]

BaseNode   position   elevation   region_area   resource_layout   distance_from_own_main   distance_from_enemy_main   approach_count   approach_widths

Connection   source_region   target_region   path_length   minimum_width   mean_width   curvature   elevation_changes

**Important:** Only MAIN and NATURAL are semantic expansion types. Every expansion beyond the natural is simply BASE. Whether players treat one as their third, fourth, safe expansion, aggressive expansion, etc. should emerge from the generated geometry.

### 4.2 Raster/grid representation

Use several channels rather than one overloaded tile value. A first-pass tensor could be:

channel 0: walkable / non-walkable channel 1: elevation level channel 2: ramp mask channel 3: buildable / non-buildable channel 4: base locations channel 5: mineral locations channel 6: geyser locations

The exact SC2 terrain values should not appear here. The exporter translates this normalized representation into the required terrain, pathing, cliff, ramp, and object records.

## 5. Dataset Processing

For each competitive map in the corpus, extract and normalize structural information. The goal is not to train directly on screenshots; it is to build a statistical description of how good SC2 maps are structured.

1. Unpack or load the SC2 map and read terrain/pathing and object placement.
2. Identify player starts, main bases, naturals, generic later bases, ramps, elevations, walkable regions, and major path connections.
3. Normalize orientation so Player 1 is consistently represented, then verify 180-degree symmetry where expected.
4. Convert the map into the semantic graph plus raster channels.
5. Extract map-level and feature-level measurements into a tabular dataset.
Useful learned measurements include:

- Map width, height, playable-area ratio, and starting-position placement.
- Main area, natural area, and generic base area distributions.
- Main-to-natural path distance and geometric offset.
- Main-to-main ground rush distance.
- Number of generic bases per map and their spatial density.
- Base-to-base distance distributions.
- Path length, minimum width, mean width, and curvature by connection context.
- Region size and open-space distributions.
- Elevation proportions and common elevation relationships.
- Frequency of graph motifs such as forks, loops, alternate routes, and central connections.

## 6. Hard Rules vs. Learned Priors

V1 should deliberately separate constraints that define a valid SC2 match from parameters that can vary.

| **Feature** | **V1 Treatment** | **Notes** |
| --- | --- | --- |
| Two valid player starts | Hardcode | Always present and mirrored. |
| 180° rotational symmetry | Hardcode | Removes spawn-position unfairness in V1. |
| Main base geometry | Mostly hardcode | Standardized anchor with limited learned variation. |
| Natural geometry | Mostly hardcode | Standardized anchor with limited learned variation. |
| Main/natural wall-gap rules | Hardcode | Use measured canonical SC2 units after extraction; do not assume raw grid width until verified. |
| Later expansion labels | Do not encode | All are generic BASE nodes. |
| Number/location of later bases | Learn + sample | Derived from corpus distributions. |
| Normal path widths | Learn + sample | Subject to a hard minimum pathability constraint. |
| Map dimensions | Learn + sample | Constrain to supported ranges. |
| Region sizes | Learn + sample | Main/natural can have narrower ranges. |
| Connectivity | Hard validate | Required regions must remain reachable. |
| Ramp/cliff legality | Hard validate/export | Generator specifies intent; exporter builds legal SC2 structures. |
| Resource legality | Hard validate | No overlap; valid townhall/mineral/geyser spacing. |
| SC2 serialization | Hardcode | Never learned. |

## 7. V1 Runtime Generation Process

A new map should be generated from scratch for each match using a staged process. Each stage works at a higher semantic level before being rasterized.

**1. Sample global map parameters:** Sample dimensions, playable-area target, broad openness, and base-count target from the learned competitive distributions.

**2. Place Player 1 main:** Choose a legal main position near an edge/corner according to the learned start-position prior.

**3. Generate the natural:** Place the natural relative to the main using learned distance/angle distributions while enforcing standardized main/natural gameplay rules.

**4. Mirror Player 2:** Create Player 2 main and natural through 180-degree rotation.

**5. Generate generic BASE nodes:** Sample the number of additional expansions and place candidate base nodes according to learned spatial distributions. Mirror paired features where appropriate.

**6. Generate the region/connectivity graph:** Connect regions using graph motifs sampled from the corpus: direct paths, forks, loops, center connections, flank routes, etc.

**7. Assign region geometry:** Sample region area, eccentricity, orientation, and spacing; turn abstract nodes into playable regions.

**8. Generate paths:** Rasterize graph edges into corridors whose width, length, and curvature are sampled from learned priors.

**9. Generate walkable terrain and boundaries:** Union the generated regions and paths into the playable surface; derive walls/non-walkable boundaries from that surface.

**10. Assign elevation and ramps:** Sample broad elevation structure, then insert valid ramps only where graph connectivity requires an elevation transition.

**11. Place resources and gameplay objects:** Use deterministic legal layouts for the main/natural and valid layouts for all generic bases.

**12. Validate and repair:** Run hard validators. Repair simple defects such as narrow passages or local overlaps; otherwise reject and regenerate.

**13. Plausibility filter:** Compare the generated feature vector with the competitive corpus. Reject extreme outliers, but allow a wider range than the training maps themselves.

**14. Export:** Translate the normalized map into a valid SC2 map using the deterministic exporter/template pipeline.

## 8. Controlled Weirdness in V1

V1 should support variation outside the exact competitive distribution, but keep it simple. The generator should sample from a broadened version of learned distributions rather than add arbitrary tile noise.

competitive prior -> sample -> broaden by weirdness factor -> validate

Examples of acceptable V1 variation:

- A center that is noticeably more open or more constrained than average.
- An unusually wide or narrow non-critical attack lane, while respecting minimum legal width.
- A generic base located somewhat closer to the center than normal.
- An uncommon number of later bases within the supported range.
- A slightly unusual path fork or loop structure.
- An elevation region that is larger or smaller than typical.
**V1 should not attempt to score whether these deviations create sophisticated strategic tradeoffs.** It only needs to ensure they remain valid, comprehensible, and not wildly outside the competitive-map design space.

## 9. Validation Requirements

Every candidate map must pass hard validation before export. At minimum:

- Both players have valid main and natural bases.
- The two player sides are rotationally symmetric.
- There is a valid ground path between the players.
- All required bases are reachable and have enough legal buildable space.
- No critical path falls below the configured minimum width.
- Main and natural entrance geometry satisfies the standardized walling rules.
- No resources overlap illegal terrain or each other.
- Ramps connect valid adjacent elevation levels.
- No accidental one-tile traps or disconnected walkable islands are created unless intentionally supported later.
- Rush distance, playable area, base density, and other key metrics stay within broad V1 bounds.
- The final exported SC2 map successfully opens/loads in the Galaxy Editor or game validation path.

## 10. Competitive-Plausibility Model

With only ~350 complete maps, V1 should use simple statistics rather than a large neural model. The training corpus is mainly a source of distributions, conditional relationships, and an outlier detector.

Recommended first implementation:

- Standardize a feature vector of approximately 50–150 map-level measurements.
- Store empirical distributions for important scalar parameters.
- Use conditional sampling where dependencies are obvious (for example, dimensions vs. rush distance or region size vs. path structure).
- Use nearest-neighbor distance or another simple outlier metric to reject maps that are structurally too far from the competitive corpus.
- Deliberately make the runtime acceptance band wider than the training-set center so unusual but valid maps can survive.

## 11. SC2 Export Strategy

The exporter is a deterministic compiler from the normalized map into StarCraft II structures. V1 should use an existing known-good SC2 map or SC2Components project as a shell and preserve metadata that the generator does not need to modify.

Normalized Map IR     |     +-- walkability/elevation/ramp intent -> terrain/pathing/cliff data     |     +-- bases/resources -> Objects data     |     +-- dimensions/settings -> required map metadata     |     v Known-good SC2 template / components     |     v Generated .SC2Map

The exporter should support only a finite V1 terrain vocabulary, for example:

- Non-walkable / outside playable space.
- Walkable low ground.
- Walkable high ground.
- Valid low-to-high ramp.
- Buildable vs. explicitly non-buildable walkable cells if required.
This keeps the difficult SC2-specific logic bounded: each normalized construct maps to a known valid SC2 construction rather than asking the generator to synthesize arbitrary SC2 terrain records.

## 12. V1 Success Criteria

V1 is successful when the system can repeatedly produce maps that satisfy all of the following:

- A fresh map can be generated automatically before a match.
- The generated map is valid and loadable in StarCraft II.
- The main and natural feel familiar enough that standard openings remain possible.
- Later expansion locations and route geometry are not tied to memorized map templates.
- The map is recognizably drawn from competitive SC2 design conventions.
- Some generated maps are noticeably unusual without being obviously broken.
- Generation can reject and retry invalid candidates quickly enough for per-match use.
- The same architecture can later accept additional strategic scoring, mutation operators, special terrain features, and richer balance analysis without rewriting the core system.

## 13. Recommended Implementation Milestones

**1. Dataset extractor:** Parse a small set of existing competitive maps into the normalized graph/grid representation and verify visually.

**2. Feature analysis:** Compute distributions for map size, bases, paths, region sizes, elevations, and distances across the full corpus.

**3. Skeleton generator:** Generate valid symmetric graphs containing MAIN, NATURAL, generic BASE nodes, and connections.

**4. Geometry rasterizer:** Turn graphs into playable regions and variable-width paths on a normalized grid.

**5. Validator:** Implement connectivity, spacing, width, symmetry, base-clearance, ramp, and resource checks.

**6. Plausibility filter:** Add empirical sampling and simple outlier rejection from the competitive dataset.

**7. SC2 exporter:** Compile a generated normalized map into a template SC2 project and validate loading.

**8. Per-match generation loop:** Generate multiple candidates, select the first acceptable result or best plausible candidate, export, and launch/use it for the match.

## 14. Deferred Beyond V1

- Formal “interesting choice” / expansion tradeoff scoring.
- Race-specific balance prediction.
- Bot simulation or automated gameplay evaluation.
- Learned map-quality neural networks or graph neural networks.
- More aggressive semantic mutation operators.
- Asymmetric maps.
- Special base/resource types, islands, destructibles, watch towers, vision blockers, unusual air-space design, and other advanced features.
- Procedural visual theming, doodad placement, and polish.
- Player feedback or telemetry-driven adaptation of the generator over time.

## 15. Implementation Addendum (V1, As-Built)

This section records how the V1 pipeline is actually implemented as of the current milestones, and the concrete design decisions made while building it. Where it differs from or refines the specification above, this addendum is authoritative for the current build. The core principles (learned center of the design space, hard constraints enforced in code, deterministic export) are unchanged.

### 15.1 Symmetry options

The specification assumed 180-degree rotational symmetry only. The implementation generalizes this because part of the competitive corpus is laterally symmetric. Supported symmetries are:

- rot180 - 180-degree rotational symmetry (the original default).
- mirror_lr - left/right mirror.
- mirror_ud - up/down mirror.
A single symmetry is chosen per map (or 'mixed' to draw randomly across a batch). Every downstream stage - node pairing, box orientation, ramp placement, resource layout, and connectivity repair - is symmetry-aware so the two sides remain images.

### 15.2 Skeleton model (Milestone 3)

The skeleton is an abstract, symmetric graph of nodes and edges built by a constructive builder (not pure random graph sampling) so degree and width rules can be enforced exactly. Node kinds:

- MAIN, NATURAL, BASE - real resource bases (BASE is the single generic expansion type; no ordinal 'third/fourth').
- ROOM - a routing-only interior region (a junction/plaza). Rasterized as a plateau but never emitted as an IR base or start location.
- JUNCTION - a Y-split pseudo-node at the midpoint of two nodes A and B: a third corridor taps the A-B link so A and B become linked through the junction, each gaining one effective degree. Generator-only, like ROOM.
Widths and degrees:

- Each node carries a single scalar width; every incident edge is no wider than the minimum width of its two endpoints (the node is the widest part locally).
- MAIN has exactly one edge (a narrow choke to its NATURAL); NATURAL has exactly two (the main choke plus one wider outgoing edge, capped); every other node has degree >= 2.
- Node pairs sharing an id are mirror images; a center node pairs with itself.

### 15.3 Geometry rasterizer (Milestone 4)

The rasterizer realizes the skeleton as boxy, terraced terrain (rooms + rectangular chokes) rather than circular plateaus joined by rounded tubes, giving the competitive rhythm of choke -> open room -> choke:

- Each node becomes an axis-variable rectangular room sized from its scalar width and an aspect ratio, at a sampled orientation.
- Each edge becomes a rectangular (flat-ended) corridor whose width comes straight from the edge, so corridor->room reads as choke->open.
- Elevation: MAIN = high (2), NATURAL = mid (1), generic BASE/ROOM sampled from {0,1,2}; a JUNCTION takes the average level of its two parents.
- Openness auto-tune: interior rooms are grown (corridors and MAIN/NATURAL anchors are not) until the walkable/playable ratio lands in a mid band (~0.50-0.62). MAIN/NATURAL stay compact so the main entrance stays a clean short ramp.

### 15.4 Elevation, ramps, and the single-step invariant

Ramps are tuned to read like real SC2 transitions, and are constrained so the exporter can turn each one into a single valid ramp:

- Same level -> flat passage (never a ramp); one level apart -> exactly one ramp.
- Single-step invariant: every corridor-linked pair of plateaus differs by at most one level. This is enforced at level-assignment time by a symmetric clamp; MAIN/NATURAL levels are fixed and JUNCTION levels are the rounded mean of their parents.
- A genuine two-level gap (0<->2) is only allowed THROUGH a JUNCTION, which supplies the mid-level landing - i.e. a staircase of two single-step ramps ('ramp into a ramp').
- The MAIN->NATURAL lane uses a short ramp placed at the high (main) edge, so the entrance is a short slope at the boundary of the two land masses rather than a long tube.
No-overlap invariant: two different elevations never touch as walkable except through a ramp; any non-ramp different-level adjacency is walled into a cliff by a cliff-cleanup pass. After painting, a non-destructive ramp regularizer clears any spurious flat 'ramp' (both sides the same level) so it reads as a flat passage. A ramp validator (validate_ramps) flags any ramp that is not a clean one-low-plateau <-> one-high-plateau strip; residual unclean ramps are caught there and regenerated by the per-match loop.

### 15.5 Guaranteed base pads

Every real base (MAIN/NATURAL/BASE) is guaranteed a flat, buildable pad on a single floor so the townhall/CC, mineral line, and geyser fit and sit on the same level. ROOM/JUNCTION get no pad. The pad is protected from erosion during cliff-cleanup, and validate_base_pads asserts every real base has at least a minimum buildable clearance radius.

### 15.6 Connectivity repair

A repair pass guarantees a ground path from the mains' component to every base, mirror-safe and non-destructive: it carves from a base's pad edge (never through the pad center), prefers wiring to the nearest same-level cell (a flat passage) and only ramps when the target is genuinely a different level. Repair and cliff-cleanup run together in a convergence loop, so the final map is simultaneously connected, free of illegal level overlaps, and pad-intact.

### 15.7 Resource placement (Milestone 5)

The generator now places a deterministic, legal resource layout at every real base:

- Eight mineral fields in a shallow arc plus two vespene geysers flanking it (configurable).
- The arc faces AWAY from the base's corridors (the classic mineral line at the back), derived from the mean direction of the base's skeleton edges.
- Every patch sits on the townhall's actual floor, off the townhall footprint, off ramps, inside the playable area, and non-overlapping. MapIR.resources and BaseNode.resource_idx are populated (ROOM/JUNCTION get none).

### 15.8 The validator (Milestone 5)

A single entry point, validate_map(mapir) -> ValidationReport, runs every hard-rule and plausibility check the spec (section 9) requires and splits results into hard failures (invalid -> reject/regenerate) and soft warnings (valid but unusual). Hard checks: exactly two mains and two naturals; a ground path between the mains with every real base reachable; buildable base pads; clean single-step ramps; and legal resources (expected counts, on the base's floor, off ramps, in-bounds, non-overlapping). Soft checks: rush distance, the contested natural<->natural rush-lane choke (measured as a widest-path bottleneck), terrain symmetry (half-cell tolerant), base spacing, and openness. The M4 driver now runs the validator and reports VALID/INVALID with metrics per map.

### 15.9 Milestone status

- M1 Dataset extractor - implemented.
- M2 Feature analysis - implemented (109-map corpus feature vectors in dataset/_features.json).
- M3 Skeleton generator - implemented (ROOM/JUNCTION nodes, scalar widths, degree rules, midpoint junctions, all three symmetries).
- M4 Geometry rasterizer - implemented (boxy rooms + chokes, elevation and ramps, the single-step + no-overlap invariants, guaranteed base pads, openness auto-tune, mirror-safe connectivity repair).
- M5 Validator - implemented (resource placement + unified validate_map with connectivity, spacing, path-width, symmetry, base-clearance, ramp, and resource checks).
- M6 Plausibility filter - implemented (broadened per-feature corpus bands + nearest-neighbour distance; wired into the M4/M5 driver).
- M7 SC2 exporter - implemented and confirmed loadable + PLAYABLE in StarCraft II. Template-shell compiler that edits a real map in-place with StormLib, authoring the cliff-level layer (t3SyncCliffLevel), both heightmaps (t3SyncHeightMap + t3HeightMap), per-level + ramp textures (t3SyncTextureInfo + t3TextureMasks), the placed-object layer (start locations + full mineral/geyser lines), and the map name (DocInfo/Name). Verified headlessly (python-sc2): representative maps load in-game, are fully connected, every base is reachable and buildable, and the expansion finder initialises. Two modes (ExportConfig): the DEFAULT flat export (flatten_terrain, on) collapses to a single tier - walkable geometry preserved, no high ground - and the relief export (author_ramps) authors real multi-level terrain + walkable ramps fully headless (no editor bake); relief is correct when it passes but low-yield, so flat is the shipped default (see 15.11/15.12).
- M8 Per-match generation loop - pending.

### 15.10 Known gaps / next steps

- A minority of maps still have unclean ramps (merged junction staircases and single-step multi-plateau blobs); the validator flags them precisely for regeneration. This has since been substantially hardened - see 15.12.
- Learned priors are wired but optional; the generator currently runs on default priors plus a controlled-weirdness factor.

### 15.11 SC2 exporter (Milestone 7)

The exporter is a deterministic compiler from MapIR to a .SC2Map (an MPQ archive), built on the spec's template/components strategy. It takes a known-good real competitive map as a shell, overwrites only the layers that can be authored reliably, reuses everything else, and repacks the archive.

What is authored vs. inherited (grounded in the community-decoded format and confirmed by round-tripping the real maps in gold_maps/):

- t3SyncCliffLevel (CLIF) - our terraced high/low ground. Encoding, verified against engine height on real maps: void/unplayable = 0, playable level L = 64 + 8*L, as USHORT per cell in a 32-byte-headed grid. Our elevation grid is embedded into the template-sized grid.
- Objects (XML) - our two start locations (ObjectPoint Type=StartLoc) and the full mineral/geyser layout (ObjectUnit). Template doodads are dropped. This placed-object layer is the reliably-writable, editor-verified part of the format.
- Heightmaps (t3SyncHeightMap + t3HeightMap) - AUTHORED. The community height formulae are lossy/per-map, so rather than synthesise values we harvest real, engine-valid flat-vertex encodings from the template (one tuple per terrain tier at cliff levels 64/128/192, plus void) and stamp them per vertex from our tier grid. Identical tuples across a plateau guarantee flatness (== buildable pad); different tuples between tiers form genuine cliffs. Everything else (tileset, textures, water, lighting) is inherited from the template.
Template selection: the generated map is matched to the smallest real template whose cliff grid fully contains it (falling back to the largest, with a centre-crop). The MapIR grid is centre-embedded and all object coordinates are translated by the same offset, so the two player sides stay symmetric within the shell.

MPQ packaging: the container is edited IN-PLACE with StormLib (a ctypes binding), which preserves the header/hash-table/encryption invariants the SC2 client validates; only the authored files are overwritten and everything else stays byte-identical. A from-scratch MPQ is structurally valid but the client silently rejects it, so the bundled pure-Python writer is kept only as a structural fallback.

Verification: exports are re-opened with an independent reader (mpyq) to confirm the authored layers survived (cliff terraces present and non-void; exactly two start locations; resource count matches MapIR; the DocInfo/Name rename applied), and representative maps are additionally confirmed to LOAD and be PLAYABLE in StarCraft II via python-sc2 (fully connected, every base reachable/buildable, the expansion finder initialises).

Known scope / remaining gaps (honest): (1) The playable area is not rewritten (SC2 rejects an edited MapInfo), so we pick a template whose existing playable region covers the map and centre content inside it - a minority of over-sized seeds still exceed every template. (2) A few expansion bases near the walkable-mask edge have a partly-void townhall pocket; a placement inset is the follow-up. (3) Relief + walkable ramps ARE authorable headless (see 15.11/15.13), but only a minority of seeds rasterize into a fully clean-ramp-connected map, so the flat export remains the shipped default.

Buildability + expansion finder (as-built). python-sc2 groups a base's resources only when they share terrain level and lie within 10.5 of each other, then places the townhall at an offset 4-8 from the resource centroid that is buildable and >=6 from every mineral / >=7 from every geyser. To satisfy this deterministically the rasterizer places minerals on a 6.5-tile arc and geysers at 7.5, never snaps a patch inside the townhall-clearance radius, and guarantees each real base a flat buildable pad (base_pad_half). Template selection is playable-area-aware (the map's void-bordered content is centred inside the template's own playable region) so edge bases are not stranded in void. An offline replica of the finder (scripts/check_expansions.py) and a headless engine probe (scripts/debug_ingame.py + analyze_expansion.py) are used to verify placeability without opening the editor.

### 15.12 Connectivity hardening & yield (post-V1 iteration)

Follow-up work on the unclean-ramp gap in 15.10, driven by in-engine testing of generated maps. The goal was to raise the fraction of generated maps that pass the M5 validator ("yield") by eliminating the ramp shapes the SC2 exporter cannot turn into a single valid ramp. Net result: yield rose from ~2/60 to 11/60 on a fixed 60-seed sweep, with exact symmetry now guaranteed.

- Exact rot180 symmetry: terrain (walkable, elevation, and the ramp set) is now snapped to a pixel-exact mirror about the true playable centre as the final rasterization step. The skeleton was already symmetric, but paint/cleanup/repair drifted by ~0.4%, enough to desync a mirror ramp pair and make one spawn's crossing work while its mirror's didn't - a fairness defect on a 1v1 map. Verified 0-cell mirror disagreement across all 60 seeds.
- Blob = fused junction front (root cause): the ramps that seal a map are not wide single ramps but the UNION of several staircases + grown rooms + repair-carves converging at a junction, so the low and high plateaus never actually connect through the ramp (BFS run==0) and it touches multiple plateau fragments. One straight <ramp> quad cannot cover that, and cropping the width after the fact only disconnects it further.
- Ramp canonicalization: a final pass rebuilds every blob into ONE clean narrow staircase between its two dominant plateaus - carved with a fixed physical run (extends into each plateau) so the exporter's gradient always has depth - and LOCKS those cells so later cleanup/repair passes cannot re-widen or demote them. Max ramp-component size on valid maps dropped from 92 to <=48.
- Nullify cross-level junctions: a Y-junction whose two parents end up on different levels forces a level change at the 3-way tap (both parent->junction edges ramp into one point = an unauthorable blob). Such junctions and all their edges are deleted and the surviving valid nodes are reconnected by the existing constrained-edge logic with clean same-level / single-step edges. This lifted yield 7/60 -> 11/60 while preserving relief variety (level mix ~42/37/21). An alternative that instead equalised the junction's parents' levels was measured and REJECTED: it flattened the map (57% on one level) and lowered yield to 5/60.
- Engine-side residual (documented, not a data bug): even with pixel-exact-symmetric terrain and quads, the SC2 engine's own ramp-quad EXPANSION over wide/irregular ramps is parity/orientation-unstable, so a mirror blob pair can still bridge differently in-engine. Clean narrow staircases bridge deterministically on both halves - another reason to keep ramps narrow. A minor quad-CENTRE parity artifact (the <ramp> c uses cell-centre while CLIF mirrors by cell index, a <=1-cell offset) remains cosmetic.
- Architectural note: the heavy level-assignment helpers (_clamp_levels, _enforce_ramp_gaps, _break_peaks_pits) are dead code; the only live node-level logic is the initial draw plus _build_constrained_edges, which is pure graph logic with no pixel dependencies. Relocating level ownership from the rasterizer to the skeleton is therefore a medium lift and is the natural home for topology rules like the junction rule above.

### 15.13 Ramp mechanics deep dive (how SC2 ramps really work)

Investigation prompted by the observation that shipped maps use short ramps (run ~4-5 cells) while our generator's minimum is run 10-12. Everything below is measured from files (gold_maps/FrostLE.SC2Map CLIF+SMAP; dataset/AutomatonAIE game-API grids). Full detail lives in docs/SC2_MAP_FORMAT_FINDINGS.md section 4.5.

- Height stack: elevation is encoded twice. t3SyncCliffLevel (CLIF) is the coarse level grid (plateaus at multiples of 64; ramp cells carry the 8-steps 72,80,...). t3SyncHeightMap (SMAP) is the gameplay height the engine paths on; measured FrostLE tiers are 16.5 / 32.6 / 48.7 units (about 16 gameplay-height units per level). Our exporter builds SMAP by LINEARLY interpolating those tiers over the CLIF value, so SMAP slope is exactly proportional to the CLIF step: an 8-CLIF/cell staircase is about 2.0 height/cell, a 64-jump wall is about 16/cell.
- Real maps use steeper, shorter, wider ramps. AutomatonAIE plateaus read 191/207/223 (delta 16 per level, same scale) and its ramps step about 4 height/cell over about 4 cells (walkable, non-buildable) - roughly 16 CLIF/cell, twice as steep as our 8/cell rule. Measured ramp dims: AutomatonAIE width 4-19 (median 11), run about 4.5; FrostLE width ~4.5, run 4.5-12.3; ours width <=~7.5, run 10-12.
- Two authoring paradigms. (a) Cliff-gradient staircase - what we and FrostLE use: a CLIF sub-level staircase plus a rampList quad that must COVER the band. (b) Editor ramp doodad - how AutomatonAIE gets short/wide ramps: the Galaxy Editor places a ramp model and BAKES the pathing (CellAttribute_Pnp) and ramp-detection data at build time, allowing steeper slopes and wider footprints. We cannot run the editor bake headless, so (b) is currently unavailable; we approximate it with (a).
- Why our floor is run 10-12 (empirical, not the engine minimum). Our ramps are diagonal, so a one-level (64) climb needs a projection run >= 8*sqrt(2) ~= 11.3 for the rank-order gradient to place all 8 sub-steps without a >8 skip (a skip reads as a wall); the rank-order scheme also needs span+1 distinct projection levels. Connectivity yield sweeps over 60 seeds: run(8,10)=31, run(9,11)=36, run(10,12)=38. So ramp_run_range=(10,12) is tuned for our headless method; it is NOT the SC2 physical minimum.
- Width cap (⚠ SUPERSEDED — see §4.9 / the small-quad paragraph below). This originally shipped ramp_choke_range=(7.0,10.0) on the theory that one full-cross-section quad made the band the sole width control. That theory was wrong: a quad sized to the full band corrupts the ramp (balloons + phantom). The actually shipped value is ramp_choke_range=(4.0,6.0) with a small fixed quad; the engine still expands this into ~16x14 walkable ramps.
- Corrections and probe caveat. FrostLE is not '<=8/cell everywhere' - its ramps carry only 6 sub-levels (72..112, it lacks 120), so the top edge is a 112->128 = +16 step that still works, i.e. the engine tolerates a >8 step at a quad-covered plateau edge. Also, query_pathing reads the AUTHORED Pnp / t3CellFlags layers, not the raw terrain slope: a synthetic step-sweep that flattened Pnp to all-open reported every step (even a 1-cell 64-drop, even with no rampList) as walkable, so it is not a slope oracle. RESOLVED via scripts/rampstep_probe.py (engine pathing_grid + a real Marine move + a no-quad control): the rampList QUAD is the SOLE controller of walkability - a quad-covered transition traverses at ANY cliff step (even a 1-cell 64-jump), and a gentle 8-cell gradient with no quad is a wall. There is no >=8-cell run floor.
- SHIPPED (final, reconciled): ramp_run_range is (11,14) and ramp_choke_range is (4,6), paired with a SMALL fixed rampList quad (min(band,4), run 2) — see the small-quad paragraph below and findings §4.9. Earlier drafts of this doc said the shipped choke was (7,10); that predates the small-quad exporter fix and is superseded. (A separate interim edit had also lowered run to (5,7) on the theory that 'the quad gates walkability so runs can be short' — that was wrong for our diagonal ramps, which need run ≥ ~11 to stay ≤8 CLIF/cell; run stays (11,14).)
- WIDTH is the free knob: one quad covers ANY clean band width. scripts/rampwide_probe.py walls a ramp band W rows tall (void elsewhere) and sends a row of Marines across; a single width-2 quad makes the WHOLE band walkable at W=16 (+/-7) and W=32 (+/-15), so wider ramps need a wider band, not more quads (2 quads work but are redundant). Re-measuring AutomatonAIE ALONG each ramp's height gradient (the earlier 'run 4.5 / width 11 / ~4 height per cell, 2x steeper' figure had run and width swapped and overstated the step) gives run ~7.6, width ~9.5, ~2.09 height/cell ~= 8.4 CLIF/cell: real maps use the SAME gentle <=8/cell staircase we do (~8-cell run for one level; ours are ~11 only because they are 45-degree diagonal). We ship width ~9 (choke 7,10), matching AutomatonAIE's ~9.5.
- OPEN (documented limitation, not new): with clean long runs the STRICT <=8 / 4-connected metric matches the engine for most ramps, and engine_cliff_grid now applies the exporter's channel_ramp_flanks so the offline grid equals the exported CLIF exactly. A ramp 'bridge' oracle was tried and dropped - it only added false-accepts. A residual class still slips through: a clean-<=8 diagonal band that one straight quad cannot cleanly cover reads connected offline but is a walled dead-end in-engine (seeds 2, 5); the ramp_cleanliness gate catches most, --valid-only rejection-samples the rest. The durable fix is the source-side terrace redesign (co-derive one clean straight quad+gradient per crossing), not further oracle tuning.
- **Quad stretch is harmful (verified; do not do it).** We tried sizing each rampList quad to cover the ramp's FULL extent (passing the real run to make_ramp_entry) to rescue short runs. It does NOT work: the engine already auto-expands a top-anchored quad across the whole contiguous ramp band (rampwide_probe), so stretching only pushes the quad's far edge past the low plateau into wall cells and BREAKS ramp detection. In-engine A/B at run(11,14) (mainconn_probe, pathing_grid+map_ramps flood): stretch ON isolates seed 22's MAIN1 from the entire map (natural->main ramp dead) and cuts seed 2's natural link, while stretch OFF connects both; it only helped seed 13 and was neutral on 32/5 - a per-seed wash that silently regresses maps the offline oracle cannot see. At short runs it also makes a bridge oracle false-accept: seed 22 reads VALID offline but its natural->main is a 1-tile walled diagonal dead-end in-engine (user-confirmed in game). Conclusion: leave run at make_ramp_entry's default and let the engine expand the quad; the real lever is the CLIF gradient (<=8/cell), i.e. run length, not quad size. The surviving hard case is a short/steep DIAGONAL ramp (seed 22 natural->main): even at run(11,14) its detection is marginal (MAP_RAMPS_DETECTED flips 4<->5 across identical engine runs), so the strict oracle over-accepts it - a diagonal-ramp limitation the source-side terrace redesign is meant to cure, not oracle/quad tuning.
- **Wide-ramp 'detection failure' (⚠ MECHANISM CORRECTED — see the small-quad paragraph below / findings §4.8–4.9). We first attributed seed 22's sealed main to an engine detector orientation bias on wide diagonal ramps. A single-variable copy-and-mutate on FrostLE (scripts/_rampmut.py) disproved that: the real cause is our OVERSIZED rampList quad. A quad wider than the true cliff band balloons the ramp footprint and spawns a phantom ramp that seals the exit; a small quad on the same wide band detects cleanly. Fixed by the small fixed quad (below).**
- **Gold ramp recipe: small quad, engine-expanded wide band (findings §4.7).** Real ramps have TWO widths that were being conflated: an authored cliff band (~4.5 perp run, ~9.5 wide along isolines) and an engine-EXPANDED walkable region (~2x: median width 10.1, run 8.1). Gold ramps ARE ~10-12 wide - but that width comes from a NARROW authored band plus a TINY rampList quad (run=2 for every gold ramp; width param 1-5, base.w 1.4-7.1) that the engine auto-expands across the contiguous band. Pnp at ramp cells is open (0x00; FrostLE 100%). This means our previous 'cap band width to 4-6' fix (15.x / findings §4.6) used the WRONG mechanism: the detector chokes on our OVERSIZED quad (we set quad width = full band extent), not on wide bands. The gold-faithful fix is to DECOUPLE quad from band - author a wide band and emit a small fixed quad (width ~3-4, run 2) at the high edge - giving traversable width-10-12 ramps that also detect reliably. Gold wide ramps are the same cliff-gradient paradigm we use, NOT editor doodads.
- **Gold-standard ramp invariants + variability (findings §4.7, measured across 106 gold maps).** RUN is FIXED (quantised by level count), WIDTH is the only free knob. With step locked at 8 CLIF (=2 engine-height units) per AXIS cell and 64 CLIF per level, a single-level ramp is ALWAYS 6 sub-levels (72,80,88,96,104,112, skip 120, then +16 to the plateau) ~ 4.5 perp / ~8 axis-cells of run: measured authored run med 4.5, range 3.8-4.5, std 0.2 (no variability). Run only grows if the ramp climbs more levels, which gold maps never do (all transitions are lo->lo+1). WIDTH along the isolines varies freely 4.5-15 (std 2-3; walkable CoV 0.34). So the generator should VARY RAMPS BY WIDTH ONLY (sample ~5-13) and keep run/step constant. Pathing is 4-connected so only axis neighbours matter; the diagonal neighbour jumps 16 and is harmless, which is why 45-degree isolines give 8/axis-cell at a short run (the old 'diagonal ramps need run>=11.3' worry was moot). Reproduce with scripts/gold_ramp_study.py (106 maps).
- **Copy-and-mutate experiment confirms the quad is the culprit (findings §4.8).** scripts/_rampmut.py copies FrostLE and replaces exactly one ramp's rampList quad (single variable = quad width) via make_ramp_entry, repacks with StormLib, and reads game_info.map_ramps back. At gold width (4) our synthesis reproduces the gold ramp EXACTLY. Widening the quad to 8/12/16 does NOT delete the ramp - it CORRUPTS it: the engine balloons the ramp footprint (37 -> 148 cells, swallowing plateau), drifts the ramp center off the true interface, and spawns a PHANTOM ramp where the quad's far edge overshoots into the plateau/wall. query_pathing at the ramp's own endpoints still returns walkable, so endpoint probes miss it - but the ramp is anchored in the wrong place and eats plateau cells, which is what seals the main exit on our generated maps (seed 22). Our build_ramp_list emits quad width = the ramp's FULL band extent (6-9+), i.e. exactly this failure mode. Operative rule: quad width <= true band, small (~3-4), run 2; let the engine expand it. This supersedes the earlier 'narrow the band' workaround.
Shipped fix — small fixed rampList quad (see findings §4.9). The declared ramp quad width is now min(band, 4) with run≈2 (gold recipe), instead of the full band extent. A quad wider than the true cliff band does not fail silently: it balloons the engine ramp footprint (37→148 cells), drifts the ramp center off the real interface, and spawns a phantom ramp into the plateau that seals the true main exit — proven single-variable on a copied FrostLE in scripts/_rampmut.py. With the small quad, seed 22 now detects a clean 6 ramps (was 8, the extra 2 were phantoms) and MAIN1↔MAIN2 fully connects; seed 32 also connects; STRICT offline yield is unchanged at 32/60 (the quad never enters the offline CLIF oracle). Measuring the engine pathing/height grids, our authored 6.6×12.5 band expands to a ~16.5×14.2 WALKABLE ramp — already wider than gold’s ~10.1×8.1 in both axes — so we do NOT widen the authored band: sweeping ramp_choke_range to (6,10)/(8,12) drops yield to 21/16 of 60 and, worse, re-breaks main→natural traversability offline-invisibly (verified seeds 11,13). Default band stays (4,6). Remaining follow-up is aesthetic only: our ramps are wide-and-long/blobby vs gold’s tighter wide-and-short; shortening the run needs 45°-isoline gradient rework (diagonal ramps otherwise need run ≥ 8·√2 ≈ 11.3 to stay ≤8 CLIF/cell).

### 15.14 True-isoline ramp gradient (gold-short runs) — shipped default

Follow-up to 15.13, resolving the deferred “our ramps are longer/blobbier than gold” aesthetic gap and, as a bonus, raising yield. The exporter and the offline connectivity oracle now author the ramp cliff gradient with TRUE 45-degree ISOLINES instead of the previous rank-order even-spread. Detail lives in docs/SC2_MAP_FORMAT_FINDINGS.md section 4.10.

Mechanism. A ramp cell's cliff value is locked to its anti-diagonal: cliff = lo + 8*(L - Lmin) with L = x*ix + y*iy on the snapped integer uphill axis. Because pathing is 4-connected, every axis neighbour then steps exactly +8 (walkable) and only the diagonal neighbour jumps +16 (ignored), so a one-level climb is ALWAYS ~8 axis-cells of run — gold's short FIXED run — while WIDTH stays free. The old rank-order gradient spread the sub-levels over the band's distinct projections, so a diagonal one-level climb needed run >= 8*sqrt(2) ~= 11.3 or it packed a delta-16 wall.

Single source of truth + opt-out. ir.isoline_gradient_ranks (via the ir.gradient_ranks dispatcher) is called by BOTH export.sc2map.author_ramp_cliffs (which writes t3SyncCliffLevel) and generate.rasterize.engine_cliff_grid (the offline oracle), so they cannot drift. Isoline is now the default; set RANK_ORDER_RAMPS to restore the legacy gradient. Shipped ramp_run_range is now (8,10), down from (11,14).

Evidence. Proven headless first with scripts/isoline_ramp_probe.py: a code-authored locked-8 diagonal band at run = 8 anti-diagonals (~5.7 along the uphill axis, about half the old run) is walkable in-engine (derived pathing_grid component + a real Marine) at widths 4, 8 and 12; a no-quad control is a wall and run below 8 isolines tops out short. A/B over 120 seeds (scripts/isoline_yield_ab.py): isoline (8,10) gives STRICT yield 65/120 versus the old rank-order (11,14) at 59/120 — shorter runs AND higher yield. In-engine (scripts/mainconn_probe.py, pure defaults): seed 5 (a previously walled dead-end) now reaches every base (MAIN1->MAIN2 dist=223.2) and seed 22 stays connected with 6 clean ramps.

Scope / still open. The isoline gradient does NOT fix the mirror-half split over-report class: the offline oracle models CLIF connectivity, not the engine's ramp DETECTION / quad coverage, so a blobby many-ramp seed can read valid offline yet split in-engine (e.g. seed 42: 17 fused ramps, MAIN1->MAIN2 unreachable). That is junction-fusion, addressed by the source-side terrace carve (the next work item), not by the gradient.
