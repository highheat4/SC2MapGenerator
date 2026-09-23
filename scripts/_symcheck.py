"""Read back the EXPORTED seed-22 .SC2Map and test each terrain channel for rot180 symmetry,
to find the asymmetric channel that makes mirror ramps detect differently."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from mpyq import MPQArchive
from sc2mapgen.export import sc2map

PATH = "outputs/export/gen_22.SC2Map"


def best_rot180(a):
    b = a[::-1, ::-1]
    best = -1; bs = None
    for sy in (0, 1):
        for sx in (0, 1):
            s = float((a == np.roll(np.roll(b, sy, 0), sx, 1)).mean())
            if s > best:
                best = s; bs = (sy, sx)
    return best, bs


def main():
    a = MPQArchive(PATH)
    names = {(n.decode() if isinstance(n, bytes) else n) for n in a.files}
    ver, cw, ch, cliff = sc2map.decode_cliff(a.read_file("t3SyncCliffLevel"))
    print(f"grid {cw}x{ch}")
    r, s = best_rot180(cliff)
    print(f"t3SyncCliffLevel : rot180 agreement={r:.4f} shift={s}")

    def grid_from(blob, elemsize, dtype):
        hdr = len(blob) - cw * ch * elemsize
        if hdr < 0:
            return None
        return np.frombuffer(blob[hdr:hdr + cw*ch*elemsize], dtype=dtype).reshape(ch, cw)

    for fname, elem, dt in [("t3SyncHeightMap", 2, "<u2"),
                            ("t3HeightMap", 4, "<f4"),
                            ("CellAttribute_Pnp", 1, "u1"),
                            ("t3CellFlags", 1, "u1")]:
        if fname not in names:
            print(f"{fname}: (absent)"); continue
        blob = a.read_file(fname)
        g = grid_from(blob, elem, dt)
        if g is None:
            print(f"{fname}: size mismatch (len={len(blob)})"); continue
        r, s = best_rot180(g)
        extra = ""
        if fname == "CellAttribute_Pnp":
            extra = f"  nonzero={int((g!=0).sum())}"
        print(f"{fname:20}: rot180 agreement={r:.4f} shift={s}{extra}")

    # ---- compare exported CLIF vs my SYMMETRIC offline prediction, and check symmetry AT RAMP CELLS
    from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator
    from sc2mapgen.generate.rasterize import RasterConfig, rasterize, engine_cliff_grid
    g = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
    mapir = rasterize(g.generate(22), seed=22, cfg=RasterConfig())
    off_x, off_y = 0, 1   # seed 22 offset (from exporter)
    offline = engine_cliff_grid(mapir.walkable, mapir.elevation.astype(int), mapir.ramps)
    # place offline into grid coords
    off_grid = np.zeros_like(cliff)
    H, W = offline.shape
    off_grid[off_y:off_y+H, off_x:off_x+W] = offline
    ro, so = best_rot180(off_grid)
    print(f"\noffline CLIF (placed) : rot180 agreement={ro:.4f} shift={so}")
    diffs = int((off_grid != cliff).sum())
    print(f"exported CLIF vs offline: {diffs} differing cells")

    # ---- test each channel in CONTENT space (crop to IR region) ----
    print("\n--- symmetry about CONTENT center (cropped to IR region) ---")
    def crop(gg):
        return gg[off_y:off_y+H, off_x:off_x+W]
    for fname, elem, dt in [("t3SyncCliffLevel", 2, "<u2"),
                            ("t3SyncHeightMap", 2, "<u2"),
                            ("t3HeightMap", 4, "<f4"),
                            ("CellAttribute_Pnp", 1, "u1")]:
        if fname not in names:
            continue
        gg = grid_from(a.read_file(fname), elem, dt)
        if gg is None:
            continue
        rr, ss = best_rot180(crop(gg))
        print(f"  {fname:20}: content rot180 = {rr:.4f} shift={ss}")

    # ramp-cell mask in grid coords
    rmask = np.zeros_like(cliff, dtype=bool)
    for rr in mapir.ramps:
        for x, y in rr.cells:
            gx, gy = int(x)+off_x, int(y)+off_y
            if 0 <= gy < cliff.shape[0] and 0 <= gx < cliff.shape[1]:
                rmask[gy, gx] = True
    # rot180 symmetry restricted to ramp cells: compare cliff & smap at ramp cell vs its mirror
    smap = grid_from(a.read_file("t3SyncHeightMap"), 2, "<u2")
    b_cliff = cliff[::-1, ::-1]
    b_smap = smap[::-1, ::-1]
    for name, gg, bb in [("CLIF@ramps", cliff, b_cliff), ("SMAP@ramps", smap, b_smap)]:
        best = -1; bs = None
        for sy in (0, 1):
            for sx in (0, 1):
                bshift = np.roll(np.roll(bb, sy, 0), sx, 1)
                m = rmask | np.roll(np.roll(rmask[::-1, ::-1], sy, 0), sx, 1)
                agree = float((gg[m] == bshift[m]).mean()) if m.any() else 1.0
                if agree > best:
                    best = agree; bs = (sy, sx)
        print(f"{name}: rot180 agreement over ramp cells = {best:.4f} shift={bs}")


if __name__ == "__main__":
    main()
