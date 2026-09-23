"""StormLib-backed MPQ modification (the loadable-in-SC2 path).

SC2's map loader is strict: it expects a full format-version 3/4 archive (extended header,
HET/BET tables, MD5 block checksums). Synthesizing that from scratch is exactly what nobody
maintains a pure-Python writer for. The robust, community-standard route is to start from a
known-good real map and let StormLib splice in the couple of files we author -- StormLib keeps
every archive invariant, so the result loads.

We ctypes-bind the system StormLib (Homebrew: ``brew install stormlib``). If it isn't present,
the caller can fall back to the pure-Python writer (structurally valid, not game-verified).
"""

from __future__ import annotations

import ctypes
import os
import shutil
from ctypes.util import find_library
from pathlib import Path

# StormLib flags
MPQ_FILE_COMPRESS = 0x00000200
MPQ_FILE_SINGLE_UNIT = 0x01000000
MPQ_FILE_REPLACEEXISTING = 0x80000000
MPQ_COMPRESSION_ZLIB = 0x02

# repo-local build (see .tools/); parents: export -> sc2mapgen -> src -> repo root
_REPO_LIB = Path(__file__).resolve().parents[3] / ".tools" / "libstorm.dylib"

_CANDIDATES = [
    os.environ.get("STORMLIB_PATH", ""),
    str(_REPO_LIB),
    "/opt/homebrew/lib/libstorm.dylib",
    "/usr/local/lib/libstorm.dylib",
    "libstorm.dylib",
    "libstorm.so",
]


def _load_stormlib() -> ctypes.CDLL | None:
    for cand in _CANDIDATES:
        if not cand:
            continue
        try:
            return ctypes.CDLL(cand)
        except OSError:
            continue
    found = find_library("storm")
    if found:
        try:
            return ctypes.CDLL(found)
        except OSError:
            return None
    return None


_STORM = _load_stormlib()


def stormlib_available() -> bool:
    return _STORM is not None


def _bind(storm: ctypes.CDLL) -> None:
    H = ctypes.c_void_p
    storm.SFileOpenArchive.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.POINTER(H)]
    storm.SFileOpenArchive.restype = ctypes.c_int
    storm.SFileCloseArchive.argtypes = [H]
    storm.SFileCloseArchive.restype = ctypes.c_int
    storm.SFileCreateFile.argtypes = [H, ctypes.c_char_p, ctypes.c_ulonglong, ctypes.c_uint,
                                      ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(H)]
    storm.SFileCreateFile.restype = ctypes.c_int
    storm.SFileWriteFile.argtypes = [H, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
    storm.SFileWriteFile.restype = ctypes.c_int
    storm.SFileFinishFile.argtypes = [H]
    storm.SFileFinishFile.restype = ctypes.c_int
    storm.SFileRemoveFile.argtypes = [H, ctypes.c_char_p, ctypes.c_uint]
    storm.SFileRemoveFile.restype = ctypes.c_int


if _STORM is not None:
    _bind(_STORM)


def export_via_stormlib(template_path: str | Path, out_path: str | Path,
                        replacements: dict[str, bytes]) -> None:
    """Copy ``template_path`` to ``out_path`` and replace ``replacements`` (name->bytes) in it.

    Raises RuntimeError on any StormLib failure (so the caller can fall back).
    """
    if _STORM is None:
        raise RuntimeError("StormLib not available")
    storm = _STORM

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_path, out)

    handle = ctypes.c_void_p()
    if not storm.SFileOpenArchive(str(out).encode(), 0, 0, ctypes.byref(handle)):
        raise RuntimeError(f"SFileOpenArchive failed for {out}")

    try:
        for name, data in replacements.items():
            cname = name.encode()
            # remove any existing copy first, then re-create (belt & suspenders vs. REPLACE flag)
            storm.SFileRemoveFile(handle, cname, 0)
            fhandle = ctypes.c_void_p()
            flags = MPQ_FILE_COMPRESS | MPQ_FILE_SINGLE_UNIT | MPQ_FILE_REPLACEEXISTING
            if not storm.SFileCreateFile(handle, cname, 0, len(data), 0, flags,
                                         ctypes.byref(fhandle)):
                raise RuntimeError(f"SFileCreateFile failed for {name}")
            buf = ctypes.create_string_buffer(data, len(data))
            if not storm.SFileWriteFile(fhandle, buf, len(data), MPQ_COMPRESSION_ZLIB):
                storm.SFileFinishFile(fhandle)
                raise RuntimeError(f"SFileWriteFile failed for {name}")
            if not storm.SFileFinishFile(fhandle):
                raise RuntimeError(f"SFileFinishFile failed for {name}")
    finally:
        storm.SFileCloseArchive(handle)
