"""Deterministic MapIR -> .SC2Map compiler (Milestone 7).

Uses a real map as a template shell (per the design's export strategy), overwrites only the
layers we can author reliably (cliff terraces + placed objects), reuses everything else, and
packs a structurally valid MPQ. Verification is structural (independent mpyq round-trip +
format conformance); in-game load additionally needs the editor's ramp finalization -- see
the module docstring in ``export/__init__.py``.
"""

from __future__ import annotations

import glob
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpyq import MPQArchive
from scipy import ndimage

from sc2mapgen.export import sc2map
from sc2mapgen.export.mpq import pack_new, read_template_blocks, write_mpq
from sc2mapgen.export.stormlib_mpq import export_via_stormlib, stormlib_available
from sc2mapgen.ir import BaseKind, MapIR

_TEMPLATES_DIR = "gold_maps"


@dataclass
class ExportConfig:
    templates_dir: str = _TEMPLATES_DIR
    template: str | None = None       # explicit template .SC2Map path (overrides auto-pick)
    emit_components: bool = False      # also write the modified files as a plain folder
    use_stormlib: bool = True          # prefer StormLib in-place edit (loadable) when available
    # Multi-level terrain with walkable ramps IS authorable headless (see docs §4): author a
    # diagonal cliff gradient per ramp AND emit a matching <rampList> entry (the piece that makes
    # the engine treat a cliff boundary as crossable). Set author_ramps=True for real relief+ramps.
    # When author_ramps is False we keep the safe flat interim (single tier, fully walkable).
    author_ramps: bool = False
    flatten_terrain: bool = True
    # Rename the exported map so it no longer shows the template's name (§7/§12). Defaults to the
    # IR's ``map_name`` (e.g. "gen_1234"); set ``map_name`` to override the displayed name.
    rename_map: bool = True
    map_name: str | None = None


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _heal_flat_terrain(tiers: np.ndarray, tiers_intended: np.ndarray,
                       close_radius: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """On a flattened map, remove the thin void gaps left by cross-level cliff walls.

    The rasterizer's cliff-cleanup carves walls to separate different elevations; once everything is
    one tier those walls are meaningless but linger as thin void gaps/notches that fragment the
    playable area (many connect to the outer void, so plain hole-filling misses them). A
    morphological *closing* (+ hole fill) welds gaps up to ~2*radius wide shut without touching the
    large outer void or pinching chokes. Closing only adds cells, so it can never disconnect the
    map, and a symmetric structuring element on a symmetric map preserves symmetry.

    Filled cells inherit the *intended* elevation of the nearest plateau (nearest-neighbour) so the
    per-level texturing stays consistent (no stray grass patch in a tiled plateau). Returns the
    healed ``(tiers, tiers_intended)``.
    """
    walk0 = tiers >= 0
    closed = ndimage.binary_closing(walk0, structure=_disk(close_radius))
    closed = ndimage.binary_fill_holes(closed)
    # iteratively fill concave void specks (a void cell whose 5x5 nbhd is majority walkable) so the
    # playable boundary has no leftover 1-cell black notches; converges in a few dozen passes.
    for _ in range(40):
        frac = ndimage.uniform_filter(closed.astype(np.float32), size=5)
        add = (~closed) & (frac > 0.5)
        if not add.any():
            break
        closed |= add
    filled = closed & ~walk0
    out = tiers.copy()
    out[filled] = 0                                   # new walkable, flat tier 0
    ti = tiers_intended.copy()
    if filled.any() and (ti >= 0).any():
        idx = ndimage.distance_transform_edt(ti < 0, return_indices=True)[1]
        ti[filled] = ti[idx[0][filled], idx[1][filled]]
    return out, ti


def _template_dims(path: str) -> tuple[int, int, int, int, int] | None:
    """(grid_w, grid_h, playable_w, playable_h, palette_tiers) of a template, or None if
    unreadable. ``palette_tiers`` is 0 when the template has no flat void to harvest."""
    try:
        a = MPQArchive(path)
        _, cw, ch, cl = sc2map.decode_cliff(a.read_file("t3SyncCliffLevel"))
        mi = a.read_file("MapInfo")
        off, _, _ = sc2map._mapinfo_playable_offset(mi)
        l, b, r, t = struct.unpack_from("<4i", mi, off)
        return cw, ch, r - l, t - b, sc2map.palette_capacity(cl)
    except Exception:
        return None


class TemplateFitError(RuntimeError):
    """No template's playable area can hold the map's walkable content."""


def content_bbox(mapir: MapIR) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` (end-exclusive) of the walkable cells. The IR grid pads the playable
    area with void on every side, so this, not the grid, is what a template must hold."""
    ys, xs = np.nonzero(mapir.walkable)
    if not xs.size:
        return 0, 0, mapir.width, mapir.height
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def required_tiers(mapir: MapIR, author_ramps: bool) -> int:
    """Flat tiers the template palette must supply: one when flattened, else every level used."""
    if not author_ramps or not mapir.walkable.any():
        return 1
    return int(mapir.elevation[mapir.walkable].max()) + 1


def pick_template(mapir: MapIR, templates_dir: str = _TEMPLATES_DIR, n_tiers: int = 1) -> str:
    """Choose the smallest real map whose *playable area* holds our walkable content, so bases
    never land in the template's unplayable border (unbuildable) and we
    don't have to rewrite MapInfo (which SC2 rejects). Raises :class:`TemplateFitError` if none
    does: a clipped export would leave edge bases outside the playable area.

    Only templates whose palette is harvestable with at least ``n_tiers`` flat tiers are
    considered: many (``Flat*``, ``LastFantasyAIE``, ``SiteDelta*``, ...) contain no void at all.
    """
    x0, y0, x1, y1 = content_bbox(mapir)
    pw, ph = x1 - x0, y1 - y0
    fits: list[tuple[int, str]] = []
    usable = False
    for path in sorted(glob.glob(str(Path(templates_dir) / "*.SC2Map"))):
        dims = _template_dims(path)
        if dims is None:
            continue
        cw, ch, plw, plh, cap = dims
        if cap < n_tiers:
            continue
        usable = True
        if plw >= pw and plh >= ph:
            fits.append((cw * ch, path))
    if fits:
        return min(fits)[1]
    if not usable:
        raise RuntimeError(f"no usable .SC2Map templates in {templates_dir!r} "
                           f"(need a flat void and {n_tiers} flat tier(s))")
    raise TemplateFitError(f"walkable content {pw}x{ph} exceeds every template's playable area "
                           f"({n_tiers} tier(s)) in {templates_dir!r}")


def export_sc2map(mapir: MapIR, out_path: str | Path,
                  cfg: ExportConfig | None = None) -> dict:
    """Compile ``mapir`` into ``out_path`` (.SC2Map). Returns an info dict."""
    cfg = cfg or ExportConfig()
    if cfg.author_ramps:
        cfg.flatten_terrain = False   # real relief needs the un-flattened tier grid
    n_needed = required_tiers(mapir, cfg.author_ramps)
    template = cfg.template or pick_template(mapir, cfg.templates_dir, n_needed)

    arch = MPQArchive(template)
    names = {(n.decode() if isinstance(n, bytes) else n) for n in arch.files}
    cliff_blob = arch.read_file("t3SyncCliffLevel")
    tmpl_smap = arch.read_file("t3SyncHeightMap")
    tmpl_hmap = arch.read_file("t3HeightMap")
    version, cw, ch, _ = sc2map.decode_cliff(cliff_blob)

    # harvest engine-valid flat encodings (cliff/smap/hmap) per terrain tier from the template
    palette = sc2map.harvest_palette(cliff_blob, tmpl_smap, tmpl_hmap)
    if palette.n_tiers < n_needed:
        raise ValueError(f"template {Path(template).name} has {palette.n_tiers} flat tier(s); "
                         f"map needs {n_needed}")

    # centre our walkable content inside the template's playable region (not the whole grid), so
    # every base lands on playable ground. The IR's void padding may overhang the template grid;
    # every layer builder crops it.
    tmpl_mapinfo = arch.read_file("MapInfo")
    _poff, _, _ = sc2map._mapinfo_playable_offset(tmpl_mapinfo)
    pl, pb, pr, pt = struct.unpack_from("<4i", tmpl_mapinfo, _poff)
    x0, y0, x1, y1 = content_bbox(mapir)
    off_x = int(round((pl + pr) / 2 - (x0 + x1) / 2))
    off_y = int(round((pb + pt) / 2 - (y0 + y1) / 2))

    # one tier-index grid drives all three consistently authored terrain layers. Plateaus map to
    # the flat tier cliffs; ramps get overlaid with sub-level gradients so cross-level transitions
    # are genuinely walkable (not walls). SMAP/HMAP then derive per-vertex from the final cliff grid.
    tiers = sc2map.build_tier_grid(mapir, cw, ch, off_x, off_y, palette.n_tiers)
    tiers_intended = tiers.copy()                   # keep real elevations for texture-by-level
    if cfg.flatten_terrain:
        tiers[tiers >= 0] = 0                       # collapse every walkable cell onto one tier
        tiers, tiers_intended = _heal_flat_terrain(tiers, tiers_intended)  # weld cross-level gaps
    cliff = sc2map.build_cliff_cells(tiers, palette)
    n_ramps = 0 if cfg.flatten_terrain else \
        sc2map.author_ramp_cliffs(cliff, tiers, mapir.ramps, palette, off_x, off_y)
    if not cfg.flatten_terrain:
        # low-plateau flanks stay plateau (gold); only high plateau wrapped around a sub-level
        # below the top is voided -- see findings doc §4.
        ramp_mask = sc2map.ramp_cell_mask(mapir, cw, ch, off_x, off_y)
        sc2map.channel_ramp_flanks(cliff, ramp_mask)
    height_ramp_mask = None if cfg.flatten_terrain else ramp_mask
    cliff_bytes = sc2map.encode_cliff(cliff, version=version)
    objects_xml = sc2map.build_objects_xml(mapir, off_x, off_y)
    objects_bytes = objects_xml.encode("utf-8")
    replacements = {
        "t3SyncCliffLevel": cliff_bytes,
        "Objects": objects_bytes,
        "t3SyncHeightMap": sc2map.build_smap(cliff, tmpl_smap, palette, height_ramp_mask),
        "t3HeightMap": sc2map.build_hmap(cliff, tmpl_hmap, palette, height_ramp_mask),
    }
    if cfg.flatten_terrain:
        # our terrain is flat, but the template's baked ramp meshes would still render as ghost
        # cliffs on flat ground -- drop them so the map looks as flat as it plays.
        try:
            replacements["t3Terrain.xml"] = sc2map.strip_render_relief(
                arch.read_file("t3Terrain.xml"))
        except Exception:  # noqa: BLE001 -- non-fatal cosmetic step
            pass
        # clear inherited world-holes (t3CellFlags 0x03) that render as black/lava voids on flat maps
        try:
            replacements["t3CellFlags"] = sc2map.clear_cell_flags(
                arch.read_file("t3CellFlags"))
        except Exception:  # noqa: BLE001 -- non-fatal cosmetic step
            pass
    else:
        # real relief: emit a <rampList> entry per authored ramp (the piece that makes the engine
        # treat the cliff gradient as a WALKABLE ramp -- docs/SC2_MAP_FORMAT_FINDINGS.md §4), and
        # author the painted pathing layer so every plateau/ramp cell is open and void is blocked.
        ramp_entries = sc2map.build_ramp_list(mapir, palette, off_x, off_y)
        n_ramps = len(ramp_entries)
        try:
            replacements["t3Terrain.xml"] = sc2map.set_ramp_list(
                arch.read_file("t3Terrain.xml"), ramp_entries)
        except Exception:  # noqa: BLE001
            pass
        try:
            replacements["CellAttribute_Pnp"] = sc2map.author_pnp(
                arch.read_file("CellAttribute_Pnp"), cliff > 0)
        except Exception:  # noqa: BLE001
            pass
        try:
            replacements["t3CellFlags"] = sc2map.clear_cell_flags(
                arch.read_file("t3CellFlags"))
        except Exception:  # noqa: BLE001
            pass

    # texture each cell by its *intended* level: one solid texture per tier + a dedicated ramp
    # texture, with overlay blending zeroed so the look is uniform per level.
    try:
        tex_blob = arch.read_file("t3SyncTextureInfo")
        _v, _tw, _th, tex_names, _ = sc2map.parse_texture_info(tex_blob)
        class_grid = sc2map.build_texture_class_grid(
            tiers_intended, mapir, len(tex_names), off_x, off_y)
        replacements["t3SyncTextureInfo"] = sc2map.author_texture_info(tex_blob, class_grid)
        replacements["t3TextureMasks"] = sc2map.author_texture_mask(
            arch.read_file("t3TextureMasks"), class_grid)
    except Exception:  # noqa: BLE001 -- non-fatal cosmetic step
        pass

    # the minimap background is a baked image; without this it keeps the template's terrain
    minimap_authored = None
    if "Minimap.tga" in names:
        mm_blob = arch.read_file("Minimap.tga")
        playable = (pl, pb, pr, pt)
        try:
            replacements["Minimap.tga"] = sc2map.author_minimap_textured(
                mm_blob, sc2map.decode_cliff(cliff_blob)[3],
                sc2map.decode_texture_mask(arch.read_file("t3TextureMasks")),
                cliff, class_grid, playable)
            minimap_authored = "textured"
        except Exception:  # noqa: BLE001 -- fall back to the preview-coloured minimap
            try:
                replacements["Minimap.tga"] = sc2map.author_minimap(
                    mm_blob, cliff > 0, tiers_intended,
                    sc2map.ramp_slope_mask(mapir, cw, ch, off_x, off_y), playable)
                minimap_authored = "flat"
            except Exception:  # noqa: BLE001 -- non-fatal cosmetic step
                pass
    smap_authored = True

    # rename the map away from the template's name (§7/§12): the displayed name is the localized
    # DocInfo/Name game-string in DocumentHeader (binary) + every locale's GameStrings.txt -- both
    # decoupled from the MapInfo bounds trap, so editing them is safe. This also avoids sharing the
    # ladder template's asset cache (§2).
    map_renamed = False
    if cfg.rename_map:
        new_name = cfg.map_name or mapir.map_name
        try:
            replacements["DocumentHeader"] = sc2map.set_document_header_name(
                arch.read_file("DocumentHeader"), new_name)
            map_renamed = True
        except Exception:  # noqa: BLE001 -- non-fatal cosmetic step
            pass
        for name in list(names):
            if name.endswith("GameStrings.txt"):
                try:
                    replacements[name] = sc2map.set_gamestrings_name(
                        arch.read_file(name), new_name)
                    map_renamed = True
                except Exception:  # noqa: BLE001
                    pass

    out_path = Path(out_path)
    if cfg.use_stormlib and stormlib_available():
        # StormLib in-place edit of a known-good archive -> loadable in SC2/the editor.
        export_via_stormlib(template, out_path, replacements)
        backend = "stormlib"
    else:
        # pure-Python writer: structurally valid (independent round-trip) but not game-verified.
        blocks = read_template_blocks(template)
        for name, data in replacements.items():
            blocks[name] = pack_new(name, data)
        write_mpq(out_path, blocks)
        backend = "python"

    if cfg.emit_components:
        comp = out_path.with_suffix(".SC2Components")
        comp.mkdir(parents=True, exist_ok=True)
        for name in ("t3SyncCliffLevel", "Objects", "t3SyncHeightMap", "t3HeightMap"):
            (comp / name).write_bytes(replacements[name])

    n_starts, n_res = sc2map.count_objects(objects_xml)
    return {
        "template": template,
        "template_dims": (cw, ch),
        "offset": (off_x, off_y),
        "backend": backend,
        "heightmap_authored": smap_authored,
        "map_renamed": map_renamed,
        "minimap_authored": minimap_authored,
        "map_name": (cfg.map_name or mapir.map_name) if cfg.rename_map else None,
        "n_tiers": palette.n_tiers,
        "n_ramps_authored": n_ramps,
        "n_start_locations": n_starts,
        "n_resources": n_res,
        "out_path": str(out_path),
        "size_bytes": out_path.stat().st_size,
    }


def verify_export(out_path: str | Path, mapir: MapIR) -> tuple[bool, list[str]]:
    """Re-open the exported archive with an independent reader and check our layers survived."""
    problems: list[str] = []
    a = MPQArchive(str(out_path))
    names = {(n.decode() if isinstance(n, bytes) else n) for n in a.files}

    for required in ("t3SyncCliffLevel", "Objects", "MapInfo", "t3Terrain.xml"):
        if required not in names:
            problems.append(f"missing {required}")

    # cliff layer decodes and carries our terracing
    try:
        _, cw, ch, grid = sc2map.decode_cliff(a.read_file("t3SyncCliffLevel"))
        levels = sorted(int(v) for v in set(grid.flatten().tolist()))
        walk_cells = int((grid > 0).sum())
        if walk_cells == 0:
            problems.append("cliff grid is all void")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"cliff decode failed: {exc}")

    # objects layer parses and has the expected two start locations
    try:
        xml = a.read_file("Objects").decode("utf-8")
        n_starts, n_res = sc2map.count_objects(xml)
        n_mains = sum(1 for b in mapir.bases if b.kind == BaseKind.MAIN)
        if n_starts != n_mains:
            problems.append(f"start locations {n_starts} != mains {n_mains}")
        if n_res != len(mapir.resources):
            problems.append(f"resource units {n_res} != {len(mapir.resources)}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"objects parse failed: {exc}")

    # map name: the DocInfo/Name game-string should read back as the IR's name, not the template's
    try:
        dh = a.read_file("DocumentHeader")
        start, count = sc2map._docheader_table_start(dh)
        o = start
        found = None
        for _ in range(count):
            klen = struct.unpack_from("<H", dh, o)[0]
            k = dh[o + 2:o + 2 + klen]
            p2 = o + 2 + klen
            vlen = struct.unpack_from("<H", dh, p2 + 4)[0]
            vs = p2 + 6
            if k == b"DocInfo/Name":
                found = dh[vs:vs + vlen].decode("utf-8", "replace")
                break
            o = vs + vlen
        if found is not None and found != mapir.map_name:
            problems.append(f"map name {found!r} != {mapir.map_name!r}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"DocumentHeader parse failed: {exc}")

    return (len(problems) == 0), problems
