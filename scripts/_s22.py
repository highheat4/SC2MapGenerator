"""One-off: overlay OFFLINE model vs IN-ENGINE pathing for seed 22's MAIN1 exit ramp,
to localize the 1-tile untraversable obstacle."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sc2mapgen.generate.skeleton import GenConfig, SkeletonGenerator
from sc2mapgen.generate.rasterize import RasterConfig, rasterize, engine_cliff_grid
from sc2mapgen.export.sc2map import channel_ramp_flanks
from sc2mapgen.ir import BaseKind
from sc2mapgen.export import ExportConfig, export_sc2map

SEED = 22
NAME = f"gen_{SEED}"
# window in IR coords
X0, X1, Y0, Y1 = 36, 55, 44, 66


def main():
    g = SkeletonGenerator(config=GenConfig(symmetries=("rot180",)))
    mapir = rasterize(g.generate(SEED), seed=SEED, cfg=RasterConfig())
    walk = mapir.walkable
    elev = mapir.elevation.astype(int)
    ecliff = engine_cliff_grid(walk, elev, mapir.ramps)   # includes channel voids
    rampmask = np.zeros_like(walk, dtype=bool)
    for r in mapir.ramps:
        for x, y in r.cells:
            rampmask[int(y), int(x)] = True

    info = export_sc2map(mapir, f"outputs/export/{NAME}.SC2Map", ExportConfig(author_ramps=True))
    ox, oy = info["offset"]
    import shutil
    shutil.copy(f"outputs/export/{NAME}.SC2Map", f"/Applications/StarCraft II/maps/{NAME}.SC2Map")

    mains = [b for b in mapir.bases if b.kind == BaseKind.MAIN]
    m1 = mains[0]
    print(f"MAIN1 IR=({int(m1.x)},{int(m1.y)}) offset=({ox},{oy})")

    def dump(title, grid, fmt):
        print(f"\n=== {title} (IR x {X0}..{X1}, y {Y0}..{Y1}) ===")
        header = "    " + "".join(f"{x%10}" for x in range(X0, X1))
        print(header)
        for y in range(Y0, Y1):
            row = f"{y:3d} "
            for x in range(X0, X1):
                row += fmt(grid, x, y)
            print(row)

    # OFFLINE: level digit, ramp='r', void='.'
    def f_off(_, x, y):
        if not walk[y, x]:
            return " "
        if rampmask[y, x]:
            return "r"
        return str(elev[y, x])
    dump("OFFLINE walk/elev (r=ramp cell)", None, f_off)

    def f_cliff(_, x, y):
        c = ecliff[y, x]
        return "." if c == 0 else chr(ord('a') + (int(c)//8 - 16)) if 0 <= int(c)//8-16 < 26 else "#"
    dump("OFFLINE exported CLIF (.=void; a=128,b=136,...,i=192)", None, f_cliff)

    # ---- engine side ----
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.bot_ai_internal import BotAIInternal
    from sc2.data import Difficulty, Race
    from sc2.main import run_game
    from sc2.player import Bot, Computer

    BotAIInternal._find_expansion_locations = lambda self: None

    class P(BotAI):
        async def on_start(self):
            from scipy import ndimage as ndi
            pg = self.game_info.pathing_grid.data_numpy
            H, W = pg.shape
            print(f"\npathing_grid shape={pg.shape}  MAP_RAMPS={len(self.game_info.map_ramps)}")
            rampset = set()
            for rp in self.game_info.map_ramps:
                for (rx, ry) in rp.points:
                    rampset.add((int(rx), int(ry)))

            # resolve y-orientation: engine(x,y) maps to IR via one of two forms. Pick the one whose
            # pathable mask best agrees with the OFFLINE plateau-walkable mask over the window.
            plate = walk & ~rampmask
            def eng_at(x, y, flip):
                ax = x + ox
                ay = (H - 1 - (y + oy)) if flip else (y + oy)
                if 0 <= ay < H and 0 <= ax < W:
                    return ax, ay
                return None
            best = None
            for flip in (False, True):
                agree = tot = 0
                for y in range(Y0, Y1):
                    for x in range(X0, X1):
                        cell = eng_at(x, y, flip)
                        if cell is None:
                            continue
                        ax, ay = cell
                        pe = pg[ay, ax] > 0 or (ax, ay) in rampset
                        agree += int(pe == bool(plate[y, x])); tot += 1
                r = agree/max(tot, 1)
                print(f"orient flip={flip}: agreement with offline plateau = {r:.3f}")
                if best is None or r > best[0]:
                    best = (r, flip)
            flip = best[1]
            print(f"--> using flip={flip}")

            # flood engine-pathable (pg>0 or map_ramp) from MAIN1
            passable = (pg > 0)
            for (rx, ry) in rampset:
                if 0 <= ry < H and 0 <= rx < W:
                    passable[ry, rx] = True
            lbl, _ = ndi.label(passable)
            mcell = eng_at(int(m1.x), int(m1.y), flip)
            mc = 0
            # snap to nearest pathable near MAIN1
            for r in range(0, 8):
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        c = eng_at(int(m1.x)+dx, int(m1.y)+dy, flip)
                        if c and lbl[c[1], c[0]] > 0:
                            mc = int(lbl[c[1], c[0]]); break
                    if mc: break
                if mc: break
            print(f"MAIN1 engine comp={mc}")

            def f_eng(_, x, y):
                c = eng_at(x, y, flip)
                if c is None:
                    return "?"
                ax, ay = c
                inramp = (ax, ay) in rampset
                if lbl[ay, ax] == mc and mc > 0:
                    return "R" if inramp else "M"   # reachable from MAIN1
                if passable[ay, ax]:
                    return "r" if inramp else "o"    # pathable but NOT reachable from main
                return "."                            # blocked
            dump("ENGINE reachability (M=reach-from-main, o=pathable-not-reached, .=blocked, R/r=ramp)",
                 None, f_eng)

            # ---- WHY detection fails: compare IR ramps vs engine-detected map_ramps ----
            from sc2mapgen.export.sc2map import _snap_dir, _RAMP_DIR_U, _SQRT2
            print("\n=== IR ramps (exporter quad geometry) ===")
            ir_info = []
            for i, r in enumerate(mapir.ramps):
                if r.top is None or r.bottom is None:
                    continue
                ux, uy = (r.top[0]-r.bottom[0]), (r.top[1]-r.bottom[1])
                n = (ux*ux+uy*uy)**0.5 or 1.0
                ux, uy = ux/n, uy/n
                d = _snap_dir(ux, uy)
                sux, suy = _RAMP_DIR_U[d]
                cells = [(int(x), int(y)) for x, y in r.cells]
                along = [c[0]*sux+c[1]*suy for c in cells]
                perp = [c[0]*suy-c[1]*sux for c in cells]
                run = (max(along)-min(along))
                wid = (max(perp)-min(perp))
                cx = sum(c[0] for c in cells)/len(cells); cy = sum(c[1] for c in cells)/len(cells)
                diag = "DIAG" if d >= 4 else "AXIS"
                ir_info.append((i, cx, cy))
                print(f" ramp {i}: lo{r.low_level}->hi{r.high_level} dir={d}({diag}) "
                      f"ncells={len(cells)} run={run:.1f} width={wid:.1f} "
                      f"centroid=({cx:.0f},{cy:.0f})")

            print("\n=== ENGINE-detected map_ramps ===")
            for j, rp in enumerate(self.game_info.map_ramps):
                pts = list(rp.points)
                cx = sum(p[0] for p in pts)/len(pts) - ox
                cy = sum(p[1] for p in pts)/len(pts) - oy
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                # nearest IR ramp centroid
                best = min(ir_info, key=lambda t: (t[1]-cx)**2+(t[2]-cy)**2)
                print(f" map_ramp {j}: npts={len(pts)} centroid_IR=({cx:.0f},{cy:.0f}) "
                      f"bbox={max(xs)-min(xs)}x{max(ys)-min(ys)} -> nearest IR ramp {best[0]} "
                      f"@({best[1]:.0f},{best[2]:.0f})")
            try:
                await self.client.leave()
            except Exception:
                pass

        async def on_step(self, it):
            pass

    run_game(maps.get(NAME), [Bot(Race.Terran, P()), Computer(Race.Terran, Difficulty.Easy)],
             realtime=False)


if __name__ == "__main__":
    main()
