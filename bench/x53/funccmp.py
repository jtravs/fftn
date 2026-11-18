#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Has a change left the code a case runs byte-identical? (experiment X53,
dev/experiments/X53-two-row-leaves.md; AGENTS.md section 6, code placement.)

    funccmp.py OBJ_A OBJ_B SYMBOL_REGEX...

For each function defined in OBJ_A whose symbol matches a regex: whether its bytes and the
relocations inside it (offset from the function's start, type, target) are the same in
OBJ_B. Uses nm and readelf (binutils) and the standard library only."""

import re
import subprocess
import sys


def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def functions(obj):
    out = {}
    for line in run("nm", "-S", "--defined-only", obj).splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in "Tt":
            out[parts[3]] = (int(parts[0], 16), int(parts[1], 16))
    return out


def section_of(obj, sym):
    # symbol table: section index per symbol
    for line in run("readelf", "-sW", obj).splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[-1] == sym:
            return parts[6]
    return None


def section_bytes(obj, idx):
    hdr = run("readelf", "-SW", obj)
    for line in hdr.splitlines():
        m = re.match(r"\s*\[\s*(\d+)\]\s+(\S+)\s+\S+\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)", line)
        if m and m.group(1) == idx:
            name, off, size = m.group(2), int(m.group(3), 16), int(m.group(4), 16)
            with open(obj, "rb") as f:
                return name, f.read()[off : off + size]
    return None, b""


def relocs(obj, secname):
    out = []
    cur = None
    for line in run("readelf", "-rW", obj).splitlines():
        m = re.match(r"Relocation section '\.rela(\S+)'", line)
        if m:
            cur = m.group(1)
            continue
        if cur == secname:
            parts = line.split()
            if len(parts) >= 5 and re.match(r"^[0-9a-f]{8,}$", parts[0]):
                out.append((int(parts[0], 16), parts[2], " ".join(parts[4:])))
    return out


def describe(obj, sym, f):
    start, size = f
    idx = section_of(obj, sym)
    name, data = section_bytes(obj, idx)
    body = data[start : start + size]
    rel = [(o - start, t, tgt) for o, t, tgt in relocs(obj, name) if start <= o < start + size]
    return body, rel


a, b = sys.argv[1], sys.argv[2]
fa, fb = functions(a), functions(b)
for rx in sys.argv[3:]:
    for sym in sorted(s for s in fa if re.search(rx, s)):
        if sym not in fb:
            print("ONLY-A", sym)
            continue
        ba, ra = describe(a, sym, fa[sym])
        bb, rb = describe(b, sym, fb[sym])
        same_bytes = ba == bb
        same_rel = ra == rb
        verdict = "SAME" if same_bytes and same_rel else "DIFF"
        print(verdict, f"bytes={same_bytes} relocs={same_rel} size={len(ba)}", sym)
        if not same_rel:
            for x, y in zip(ra, rb, strict=False):
                if x != y:
                    print("   ", x, "|", y)
            if len(ra) != len(rb):
                print("    reloc counts", len(ra), len(rb))
