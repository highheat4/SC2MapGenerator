"""Decode + compare ramp channels between maps (gold vs our export).

Parses every <ramp> entry in t3Terrain.xml into structured fields, classifies its orientation
from base.u, and can dump local windows of every terrain channel (CLIF/SMAP/HMAP/Pnp/PATH/
CellFlags) around a ramp. Used to diagnose why an authored ramp fails engine detection by
comparing it field-for-field and channel-for-channel against a gold-map ramp of the same dir.

    PYTHONPATH=src .venv/bin/python scripts/_rampcmp.py census gold_maps/Goldenaura512AIE.SC2Map
"""
from __future__ import annotations

import math
import re
import struct
import sys
from pathlib import Path

import numpy as np
from mpyq import MPQArchive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _pv(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


_FIELD_RE = re.compile(
    r'u\(([^)]*)\)\s*r\(([^)]*)\)\s*c=\(([^)]*)\)\s*w=([-0-9.e+]+)\s*h=([-0-9.e+]+)')


def parse_field(s: str) -> dict:
    m = _FIELD_RE.search(s)
    u = _pv(m.group(1)); r = _pv(m.group(2)); c = _pv(m.group(3))
    return {"u": u, "r": r, "c": c, "w": float(m.group(4)), "h": float(m.group(5))}


def parse_ramps(xml: bytes) -> list[dict]:
    text = xml.decode("utf-8", "replace")
    m = re.search(r"<rampList\b[^>]*>(.*?)</rampList>", text, re.DOTALL)
    if not m:
        return []
    out = []
    for ent in re.findall(r"<ramp\b[^>]*/>", m.group(0)):
        d = {"raw": ent}
        d["dir"] = int(re.search(r'dir="(\d+)"', ent).group(1))
        d["lo"] = int(re.search(r'lo="(\d+)"', ent).group(1))
        d["hi"] = int(re.search(r'hi="(\d+)"', ent).group(1))
        for fld in ("leftLo", "leftHi", "rightLo", "rightHi", "base", "mid"):
            fm = re.search(fr'{fld}="([^"]*)"', ent)
            d[fld] = parse_field(fm.group(1))
        out.append(d)
    return out


def orient(base_u: tuple[float, float]) -> str:
    ux, uy = base_u
    if abs(ux) < 1e-3 and abs(uy) < 1e-3:
        return "?"
    if abs(abs(ux) - abs(uy)) < 0.2:  # diagonal
        ns = "S" if uy > 0 else "N"
        ew = "E" if ux > 0 else "W"
        return ns + ew
    if abs(ux) > abs(uy):
        return "E" if ux > 0 else "W"
    return "S" if uy > 0 else "N"


CHANNELS = {
    "CLIF": "t3SyncCliffLevel",
    "SMAP": "t3SyncHeightMap",
    "HMAP": "t3HeightMap",
    "Pnp": "CellAttribute_Pnp",
    "PATH": "t3SyncPathingInfo",
    "LFCT": "t3CellFlags",
}


def load_channels(path: str) -> dict:
    a = MPQArchive(path)
    names = set(a.files) if isinstance(next(iter(a.files)), str) else {
        n.decode() for n in a.files}
    out = {}
    for key, fname in CHANNELS.items():
        try:
            out[key] = a.read_file(fname)
        except Exception:
            out[key] = None
    out["terrain"] = a.read_file("t3Terrain.xml")
    out["ramps"] = parse_ramps(out["terrain"])
    return out


def decode_cliff(blob: bytes) -> np.ndarray:
    sx, sy = struct.unpack_from("<II", blob, 8)
    return np.frombuffer(blob[32:32 + sx * sy * 2], dtype="<u2").reshape(sy, sx)


def decode_smap(blob: bytes):
    """Returns (height int16 [sh,sw], mask u16 [sh,sw]) on the VERTEX grid."""
    sw, sh = struct.unpack_from("<II", blob, 8)
    body = np.frombuffer(blob[64:64 + sw * sh * 4], dtype="<u4").reshape(sh, sw)
    height = (body & 0xFFFF).astype(np.uint16).view(np.int16)
    mask = (body >> 16).astype(np.uint16)
    return height, mask


def decode_hmap(blob: bytes):
    sw, sh = struct.unpack_from("<II", blob, 8)
    body = np.frombuffer(blob[32:32 + sw * sh * 6], dtype="<u2").reshape(sh, sw, 3)
    return body  # (adjust, base, mask)


def decode_cellbytes(blob: bytes, sx: int, sy: int) -> np.ndarray:
    """A per-cell u8 body with an unknown-length header inferred from sx*sy."""
    if blob is None:
        return None
    hdr = len(blob) - sx * sy
    if hdr < 0:
        return None
    return np.frombuffer(blob[hdr:hdr + sx * sy], dtype=np.uint8).reshape(sy, sx)


def census(path: str) -> None:
    ch = load_channels(path)
    print(f"== {Path(path).name} : {len(ch['ramps'])} ramps ==")
    hdr = f"{'#':>2} {'dir':>3} {'orient':>6} {'lo>hi':>5} " \
          f"{'base.w':>7} {'base.h':>7} {'mid.w':>7} {'mid.h':>7} {'run':>5} " \
          f"{'llo.w':>5} {'llo.h':>5}"
    print(hdr)
    for i, r in enumerate(ch["ramps"]):
        b, m = r["base"], r["mid"]
        run = math.hypot(b["c"][0] - m["c"][0], b["c"][1] - m["c"][1])
        o = orient(b["u"])
        print(f"{i:>2} {r['dir']:>3} {o:>6} {r['lo']}>{r['hi']:>3} "
              f"{b['w']:>7.3f} {b['h']:>7.3f} {m['w']:>7.3f} {m['h']:>7.3f} {run:>5.2f} "
              f"{r['leftLo']['w']:>5.2f} {r['leftLo']['h']:>5.2f}")


def _grid_str(g: np.ndarray, fmt: str = "{:4d}") -> str:
    return "\n".join("".join(fmt.format(int(v)) for v in row) for row in g)


def window(path: str, x0: int, x1: int, y0: int, y1: int, margin: int = 2) -> None:
    """Dump every channel over a cell box [x0..x1]x[y0..y1] (+margin). SMAP/HMAP are vertex grids."""
    ch = load_channels(path)
    clif = decode_cliff(ch["CLIF"])
    sy, sx = clif.shape
    ax0, ax1 = max(0, x0 - margin), min(sx, x1 + margin + 1)
    ay0, ay1 = max(0, y0 - margin), min(sy, y1 + margin + 1)
    print(f"== {Path(path).name}  cells x[{ax0}:{ax1}] y[{ay0}:{ay1}] ==")
    print("-- CLIF (cliff value per cell) --")
    print(_grid_str(clif[ay0:ay1, ax0:ax1]))

    sh, sw = decode_smap(ch["SMAP"])[0].shape
    smh, smm = decode_smap(ch["SMAP"])
    print("-- SMAP height (int16, vertex grid) --")
    print(_grid_str(smh[ay0:ay1, ax0:ax1], "{:7d}"))
    print("-- SMAP mask (u16, vertex grid) --")
    print(_grid_str(smm[ay0:ay1, ax0:ax1], "{:6d}"))

    hm = decode_hmap(ch["HMAP"])
    print("-- HMAP base (col1, vertex grid) --")
    print(_grid_str(hm[ay0:ay1, ax0:ax1, 1], "{:7d}"))
    print("-- HMAP mask (col2, vertex grid) --")
    print(_grid_str(hm[ay0:ay1, ax0:ax1, 2], "{:4d}"))

    for key in ("Pnp", "PATH", "LFCT"):
        cg = decode_cellbytes(ch[key], sx, sy)
        if cg is None:
            print(f"-- {key}: (absent) --")
            continue
        print(f"-- {key} (per-cell u8) --")
        print(_grid_str(cg[ay0:ay1, ax0:ax1], "{:4d}"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "census":
        census(sys.argv[2])
    elif cmd == "window":
        p = sys.argv[2]
        x0, x1, y0, y1 = (int(v) for v in sys.argv[3:7])
        window(p, x0, x1, y0, y1)
