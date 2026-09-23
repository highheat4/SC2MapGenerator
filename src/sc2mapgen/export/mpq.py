"""Minimal MPQ writer for SC2 map export, with verbatim block copy-through.

There is no maintained pure-Python MPQ *writer* (mpyq is read-only; the ``mpq`` package is
StormLib bindings that explicitly don't write). We only need a small, well-formed subset.

Key idea: to modify a real ``.SC2Map`` we copy every *unchanged* file's raw archived block
(still compressed, sector tables intact) straight through -- no decode needed, so templates
using compression codecs mpyq can't decode still work -- and freshly store only the couple of
files we author. Reading/parsing the template is delegated to mpyq; the crypto here mirrors
mpyq's exactly so the hash/block tables round-trip.

Emitted archive (classic format-version 0, 32-byte header, no user-data shunt):

    [0]                  MPQ header (32 bytes)
    [32]                 file data blocks (verbatim passthrough or freshly stored)
    [hash_table_offset]  encrypted hash table
    [block_table_offset] encrypted block table
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from mpyq import MPQArchive

MPQ_FILE_EXISTS = 0x80000000
MPQ_FILE_SINGLE_UNIT = 0x01000000
MPQ_FILE_COMPRESS = 0x00000200
MPQ_FILE_ENCRYPTED = 0x00010000
_COMPRESSION_ZLIB = 0x02  # StormLib/mpyq compression mask byte for zlib/deflate

_HASH_TYPES = {"TABLE_OFFSET": 0, "HASH_A": 1, "HASH_B": 2, "TABLE": 3}


@dataclass
class Block:
    """One archived file: its name, the bytes to store, MPQ flags, and real (unpacked) size."""

    name: str
    stored: bytes
    flags: int
    size: int


def _build_encryption_table() -> dict[int, int]:
    seed = 0x00100001
    table: dict[int, int] = {}
    for i in range(256):
        index = i
        for _ in range(5):
            seed = (seed * 125 + 3) % 0x2AAAAB
            temp1 = (seed & 0xFFFF) << 0x10
            seed = (seed * 125 + 3) % 0x2AAAAB
            temp2 = seed & 0xFFFF
            table[index] = temp1 | temp2
            index += 0x100
    return table


_ENC = _build_encryption_table()


def _hash(string: str, hash_type: str) -> int:
    seed1 = 0x7FED7FED
    seed2 = 0xEEEEEEEE
    t = _HASH_TYPES[hash_type]
    for ch in string.upper():
        value = _ENC[(t << 8) + ord(ch)]
        seed1 = (value ^ (seed1 + seed2)) & 0xFFFFFFFF
        seed2 = (ord(ch) + seed1 + seed2 + (seed2 << 5) + 3) & 0xFFFFFFFF
    return seed1


def _encrypt(data: bytes, key: int) -> bytes:
    """Inverse of mpyq._decrypt: seed2 advances on the *plaintext* value."""
    assert len(data) % 4 == 0
    seed1 = key
    seed2 = 0xEEEEEEEE
    out = bytearray()
    for i in range(len(data) // 4):
        seed2 = (seed2 + _ENC[0x400 + (seed1 & 0xFF)]) & 0xFFFFFFFF
        value = struct.unpack_from("<I", data, i * 4)[0]
        enc = (value ^ ((seed1 + seed2) & 0xFFFFFFFF)) & 0xFFFFFFFF
        seed1 = (((~seed1 << 0x15) + 0x11111111) | (seed1 >> 0x0B)) & 0xFFFFFFFF
        seed2 = (value + seed2 + (seed2 << 5) + 3) & 0xFFFFFFFF
        out += struct.pack("<I", enc)
    return bytes(out)


def read_template_blocks(path: str | Path) -> dict[str, Block]:
    """Read every listed file's *raw* archived block (verbatim, still compressed) from an MPQ.

    Copying these through preserves whatever codec/sectoring the template used, so we never
    have to decode cosmetic files. Encrypted blocks are refused (their key is offset-bound and
    would break when moved); SC2 map files are not normally encrypted.
    """
    a = MPQArchive(str(path))
    base = a.header["offset"]
    out: dict[str, Block] = {}
    for raw_name in a.files:
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else raw_name
        if name == "(listfile)":
            continue
        he = a.get_hash_table_entry(name)
        if he is None:
            continue
        be = a.block_table[he.block_table_index]
        if not be.flags & MPQ_FILE_EXISTS:
            continue
        if be.flags & MPQ_FILE_ENCRYPTED:
            raise RuntimeError(f"template file {name!r} is encrypted; not supported")
        a.file.seek(be.offset + base)
        blob = a.file.read(be.archived_size)
        out[name] = Block(name=name, stored=blob, flags=be.flags, size=be.size)
    return out


def pack_new(name: str, data: bytes) -> Block:
    """Freshly store a file as a single-unit block (zlib-compressed only if it helps)."""
    flags = MPQ_FILE_EXISTS | MPQ_FILE_SINGLE_UNIT
    stored = data
    packed = bytes([_COMPRESSION_ZLIB]) + zlib.compress(data, 9)
    if len(packed) < len(data):
        stored = packed
        flags |= MPQ_FILE_COMPRESS
    return Block(name=name, stored=stored, flags=flags, size=len(data))


def _next_pow2(n: int) -> int:
    p = 4
    while p < n:
        p <<= 1
    return p


def write_mpq(path: str | Path, blocks: dict[str, Block]) -> None:
    """Write the given blocks (name->Block) into a classic MPQ archive."""
    items = list(blocks.values())
    listfile = ("\r\n".join(b.name for b in items) + "\r\n").encode("utf-8")
    items = items + [pack_new("(listfile)", listfile)]

    header_size = 32
    data_blob = bytearray()
    block_rows: list[tuple[int, int, int, int]] = []  # offset, csize, size, flags
    for blk in items:
        offset = header_size + len(data_blob)
        block_rows.append((offset, len(blk.stored), blk.size, blk.flags))
        data_blob += blk.stored

    hash_size = _next_pow2(len(items) * 2)
    hash_off = header_size + len(data_blob)
    block_off = hash_off + hash_size * 16

    EMPTY = [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF, 0xFFFF, 0xFFFFFFFF]
    slots = [list(EMPTY) for _ in range(hash_size)]
    for block_index, blk in enumerate(items):
        start = _hash(blk.name, "TABLE_OFFSET") & (hash_size - 1)
        a_hash = _hash(blk.name, "HASH_A")
        b_hash = _hash(blk.name, "HASH_B")
        i = start
        while slots[i][4] != 0xFFFFFFFF:
            i = (i + 1) & (hash_size - 1)
        slots[i] = [a_hash, b_hash, 0, 0, block_index]
    hash_raw = b"".join(struct.pack("<2I2HI", *s) for s in slots)
    hash_enc = _encrypt(hash_raw, _hash("(hash table)", "TABLE"))

    block_raw = b"".join(struct.pack("<4I", *row) for row in block_rows)
    block_enc = _encrypt(block_raw, _hash("(block table)", "TABLE"))

    archive_size = block_off + len(block_enc)
    header = struct.pack(
        "<4s2I2H4I",
        b"MPQ\x1a",
        header_size,
        archive_size,
        0,     # format_version 0 (32-byte header)
        3,     # sector_size_shift (irrelevant for single-unit files)
        hash_off,
        block_off,
        hash_size,
        len(items),
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(header)
        fh.write(data_blob)
        fh.write(hash_enc)
        fh.write(block_enc)
