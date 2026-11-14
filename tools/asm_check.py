#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Disassembly checks of FFTN's ISA regions (DESIGN.md 6.4, 15.3), run as a CTest test.

    tools/asm_check.py --objdump <objdump> <group> [<group> ...]

A group is an object file, a static library, or several of them joined by ';' (as CMake's
$<TARGET_OBJECTS> writes them). Functions are assigned to a target by their mangled
namespace, fftn::<version>::detail::<target>::; everything else is baseline code. Checks:

- each group contains the scalar target and the SIMD target of its architecture (avx2 on
  x86-64, neon_fcma on AArch64), and that target's code uses its defining instructions
  (an FMA and a ymm register; FCMLA and FCADD);
- no function of a target calls or tail-calls an FFTN or standard-library function: the
  vocabulary is always inlined, and nothing standard runs inside a region. One shape is
  not a call: identical-code folding (GCC's IPA-ICF, seen in sanitizer builds) turns one
  of two instantiations with identical code into a single jump to the other, so a
  function whose whole body is one unconditional jump to another instantiation of the
  same function in the same target is an alias, and the jump's target is checked as a
  function in its own right (identical_code_alias);
- x86-64: no VEX or EVEX instruction and no ymm or zmm register outside avx2, and no
  EVEX feature (zmm, mask registers, registers 16-31) inside it;
- AArch64: no FCMLA or FCADD outside neon_fcma; in neon_fcma no LDAPR or FJCVTZS, which
  "arch=armv8.3-a" also allows; pointer authentication outside the hint space only in
  neon_fcma, which is dispatched only where the processor has it (the hint forms, such as
  PACIASP, are NOPs on Armv8.0 and are what branch protection emits everywhere);
- --baseline-has relaxes the baseline checks for what the build's baseline already has:
  Apple's compilers target Apple silicon by default, which has FCMA, so scalar code there
  may legitimately use FCADD;
- an object named after a target (avx2.cpp.o, probe_avx2.cpp.o) defines FFTN functions of
  that target only, and target code is defined in no other object. A linked executable
  (what an LTO build leaves to check) gets every other check.

Works with llvm-objdump (any object format) and GNU objdump (ELF), and with Itanium and
Microsoft name mangling (the latter is untested on Windows so far). Standard library only.
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

TARGETS = ("scalar", "avx2", "neon_fcma")
SIMD_TARGET = {"x86": "avx2", "arm": "neon_fcma"}

FILE_FORMAT = re.compile(r"^(?P<name>.*?):\s+file format\s+(?P<fmt>.+)$")
LABEL = re.compile(r"^(?P<addr>[0-9a-f]+) <(?P<name>.+)>:$")
RELOC = re.compile(
    r"^\s*[0-9a-f]+:\s+(?:R_|X86_64_RELOC_|ARM64_RELOC_|IMAGE_REL_)\S*\s+(?P<sym>\S+)"
)
INSN = re.compile(r"^\s*(?P<addr>[0-9a-f]+):\s+(?P<mn>\S+)\s*(?P<ops>.*)$")
TARGET_REF = re.compile(r"(?P<addr>[0-9a-f]+) <(?P<sym>[^>+]+)(?:\+0x[0-9a-f]+)?>")
RELOC_TARGET = re.compile(r"^(?P<base>.+?)(?P<addend>[+-]0x[0-9a-f]+)?$")
# Itanium names (_Z, with Mach-O's extra underscore) and Microsoft names (?), which list a
# name's namespaces innermost first, ending in "@@": ?f@?A0x1@avx2@detail@v0@fftn@@...
MANGLED = re.compile(r"^(?:_{1,2}Z|\?)")
MSVC_TARGET = re.compile(r"@(\w+)@detail@v\d+@fftn@@")

X86_CALLS = ("call", "callq", "jmp", "jmpq")
ARM_CALLS = ("bl", "b")
FMA = re.compile(r"^vfn?m(?:add|sub)(?:sub|add)?\d*[ps][sd]$")
EVEX = re.compile(r"%zmm|%k[0-7]\b|%[xy]mm(?:1[6-9]|2\d|3[01])\b|\{")
PAUTH_OTHER = {"retaa", "retab", "eretaa", "eretab", "ldraa", "ldrab"}
PAUTH_HINTS = {
    "paciasp", "pacibsp", "autiasp", "autibsp", "paciaz", "pacibz", "autiaz", "autibz",
    "pacia1716", "pacib1716", "autia1716", "autib1716", "xpaclri",
}  # fmt: skip


@dataclass
class Call:
    mnemonic: str
    symbol: str  # from the relocation, or as the disassembler printed the operand
    addr: int | None  # the resolved target address when no relocation names it
    # A relocation against a section symbol (an ELF local function): section and offset.
    section_ref: tuple[str, int] | None = None


@dataclass
class Function:
    names: list[str]
    obj: str
    section: str
    addr: int
    insns: list[tuple[str, str]] = field(default_factory=list)  # (mnemonic, operands)
    calls: list[Call] = field(default_factory=list)
    end: int = 1 << 64

    @property
    def name(self) -> str:
        for n in self.names:
            if MANGLED.match(n):
                return n
        return self.names[0]


def components(mangled: str) -> list[str]:
    """The leading source names of an Itanium nested name: _ZN4fftn2v06detail4avx2...

    A local entity (_ZZN..., a lambda or block-scope static) yields the names of the
    function it belongs to, so a lambda in a region counts as that target's code."""
    m = re.match(r"^_*ZZ?N[rVKRO]*", mangled)
    if not m:
        return []
    out, i = [], m.end()
    while i < len(mangled) and mangled[i].isdigit():
        j = i
        while j < len(mangled) and mangled[j].isdigit():
            j += 1
        n = int(mangled[i:j])
        out.append(mangled[j : j + n])
        i = j + n
        while i < len(mangled) and mangled[i] == "L":  # internal-linkage marker
            i += 1
    return out


def is_fftn(sym: str) -> bool:
    if sym.startswith("?"):
        return "@fftn@@" in sym
    return components(sym)[:1] == ["fftn"]


def is_std(sym: str) -> bool:
    if sym.startswith("?"):
        return "@std@@" in sym
    return re.match(r"^_*Z(?:N[rVKRO]*)?St", sym) is not None


def target_of(sym: str) -> str | None:
    if sym.startswith("?"):
        m = MSVC_TARGET.search(sym)
        return m.group(1) if m and m.group(1) in TARGETS else None
    c = components(sym)
    if len(c) >= 4 and c[0] == "fftn" and c[2] == "detail" and c[3] in TARGETS:
        return c[3]
    return None


def object_target(obj: str) -> str | None:
    # Archive members may be named by a Windows path (llvm-lib keeps the path it was given).
    m = re.search(r"(?:^|[_/\\(])(scalar|avx2|neon_fcma)\.cpp\.(?:o|obj)\)?$", obj)
    return m.group(1) if m else None


def is_object(obj: str) -> bool:
    return re.search(r"\.(?:o|obj)\)?$", obj) is not None


def llvm_flavour(objdump: str) -> bool:
    out = subprocess.run([objdump, "--version"], capture_output=True, text=True, check=False)
    return "LLVM" in out.stdout


def section_call(call: Call, target: str) -> None:
    """Records a call relocated against target, which may be a section symbol.

    ELF assemblers relocate calls to internal-linkage functions against their section.
    With -ffunction-sections that section is named after the function (.text._ZN...,
    .text.unlikely._ZN...), and the name is taken from it; otherwise (.text,
    .text.unlikely, .text.startup) the section and offset are resolved later."""
    m = RELOC_TARGET.match(target)
    base, addend = (m.group("base"), m.group("addend")) if m else (target, None)
    call.addr = None
    if not base.startswith("."):
        call.symbol = base
        return
    for prefix in (".text.unlikely.", ".text.hot.", ".text.startup.", ".text.exit.", ".text."):
        rest = base.removeprefix(prefix)
        if rest != base and MANGLED.match(rest):
            call.symbol = rest
            return
    call.symbol = ""
    call.section_ref = (base, int(addend, 16) if addend else 0)


def disassemble(objdump: str, path: str) -> tuple[str, list[Function]]:
    args = [objdump, "-d", "-r", "--no-show-raw-insn"]
    if llvm_flavour(objdump):
        args.append("--show-all-symbols")
    out = subprocess.run([*args, path], capture_output=True, text=True, check=True).stdout
    arch, obj, section = "", path, ""
    funcs: list[Function] = []
    cur: Function | None = None
    pending: Call | None = None  # a call whose relocation line may follow
    for line in out.splitlines():
        if m := FILE_FORMAT.match(line):
            obj, fmt = m.group("name"), m.group("fmt")
            arch = "x86" if re.search(r"x86-64|x86_64", fmt) else arch
            arch = "arm" if re.search(r"arm64|aarch64", fmt) else arch
            cur = None
        elif line.startswith("Disassembly of section "):
            section = line.removeprefix("Disassembly of section ").rstrip(":")
            cur = None
        elif m := LABEL.match(line):
            addr = int(m.group("addr"), 16)
            if cur is not None and not cur.insns and cur.addr == addr:
                cur.names.append(m.group("name"))  # another symbol for the same code
            else:
                cur = Function([m.group("name")], obj, section, addr)
                funcs.append(cur)
            pending = None
        elif cur is None:
            continue
        elif m := RELOC.match(line):
            if pending is not None:
                section_call(pending, m.group("sym"))
                pending = None
        elif m := INSN.match(line):
            mn, ops = m.group("mn"), m.group("ops")
            cur.insns.append((mn, ops))
            pending = None
            if mn in X86_CALLS or mn in ARM_CALLS:
                ref = TARGET_REF.search(ops)
                pending = Call(mn, ref.group("sym"), int(ref.group("addr"), 16)) if ref else None
                if pending is not None:
                    cur.calls.append(pending)
    resolve(funcs, arch)
    return arch, funcs


def same_function(a: str, b: str) -> bool:
    """b is a or a part GCC split from it (a.cold, a.cold.1)."""
    return a == b or b.startswith((a + ".cold", a.split(".")[0] + ".cold"))


def resolve(funcs: list[Function], arch: str) -> None:
    """Sets each function's end, and names the target of calls the assembler resolved.

    A resolved branch prints the nearest symbol before its target, which may be a
    section-start label (ltmp0) or the calling function itself; the address decides. A
    target that is no function's start is not a call: a branch within the function, or a
    jump table embedded in the code (Mach-O x86-64) decoded as instructions."""
    starts: dict[tuple[str, str], list[Function]] = {}
    for f in funcs:
        starts.setdefault((f.obj, f.section), []).append(f)
    for group in starts.values():
        group.sort(key=lambda f: f.addr)
        for a, b in itertools.pairwise(group):
            a.end = b.addr
    for f in funcs:
        for call in f.calls:
            if call.section_ref is not None:
                section, addend = call.section_ref
                # x86-64 rel32 addends count from the end of the 4-byte field.
                target = addend + (4 if arch == "x86" else 0)
                owner = [g for g in starts.get((f.obj, section), []) if g.addr <= target < g.end]
                call.symbol = owner[0].name if owner else f"{section}+{target:#x} (unresolved)"
            elif call.addr is not None:
                callee = [g for g in starts[(f.obj, f.section)] if g.addr == call.addr]
                call.symbol = callee[0].name if callee and callee[0] is not f else ""
            if same_function(f.name, call.symbol) or same_function(call.symbol, f.name):
                call.symbol = ""


# Instructions a compiler may put before a function's first real instruction.
LANDING_PADS = {"endbr64", "endbr32", "bti", "nop"}
JUMPS = {"jmp", "jmpq", "b"}
# Alignment padding a compiler may put after a function's last instruction (GCC on x86-64
# pads a folded alias with `data16 nopw %cs:...`); it is never executed after an
# unconditional jump.
PADDING = {"nop", "nopw", "nopl", "data16", "int3", "xchg"}


def template_name(sym: str) -> list[str] | None:
    """The nested name of an Itanium-mangled function without its template arguments and
    parameters, or None where the mangling is not understood (Microsoft names)."""
    if sym.startswith("?"):
        return None
    return components(sym) or None


def identical_code_alias(f: Function, call: Call, target: str) -> bool:
    """Whether f is nothing but a jump to call's target, another instantiation of the same
    function (or overload) in the same target: the form identical-code folding leaves.
    Such a jump transfers to code that is checked itself and moves no vector across a
    boundary that the folded copy would not have moved; any other call still counts."""
    body = [mn for mn, _ in f.insns if mn.split(".")[0] not in LANDING_PADS]
    while len(body) > 1 and body[-1] in PADDING and body[0] in JUMPS:
        body.pop()
    if len(body) != 1 or body[0] not in JUMPS or f.calls != [call]:
        return False
    mine, theirs = template_name(f.name), template_name(call.symbol)
    return (
        mine is not None
        and mine == theirs
        and target_of(call.symbol) == target
        and call.symbol != f.name
    )


def call_errors(target: str, f: Function) -> list[str]:
    """The forbidden calls of f, a function of target."""
    errors = []
    for call in f.calls:
        sym = call.symbol
        if sym.endswith("(unresolved)"):
            errors.append(f"{target}: {f.name} calls {sym}, which cannot be checked")
        elif sym and (is_fftn(sym) or is_std(sym)) and not identical_code_alias(f, call, target):
            errors.append(f"{target}: {f.name} calls {sym} ({call.mnemonic}); must inline")
    return errors


def is_pauth(base: str) -> bool:
    """A pointer-authentication instruction outside the hint space."""
    if base in PAUTH_HINTS:
        return False
    if base.startswith(("pac", "aut", "xpac")):
        return True
    return base in PAUTH_OTHER or re.match(r"^bl?ra[ab]z?$", base) is not None


def check_group(objdump: str, group: list[str], baseline_has: set[str]) -> list[str]:
    errors: list[str] = []
    arch = ""
    funcs: list[Function] = []
    for path in group:
        a, f = disassemble(objdump, path)
        arch = arch or a
        funcs += f
    if arch not in SIMD_TARGET:
        return [f"{group}: unknown architecture (need x86-64 or AArch64 objects)"]
    simd = SIMD_TARGET[arch]
    by_target: dict[str, list[Function]] = {t: [] for t in TARGETS}
    baseline: list[Function] = []
    for f in funcs:
        t = target_of(f.name)
        (by_target[t] if t else baseline).append(f)
        where = object_target(f.obj)
        if t is not None and where != t and is_object(f.obj):
            errors.append(f"{f.obj}: defines {t} code outside the {t} unit: {f.name}")
        if where is not None and is_fftn(f.name) and t != where:
            errors.append(f"{f.obj}: defines FFTN code outside the {where} namespace: {f.name}")

    for t in ("scalar", simd):
        if not by_target[t]:
            errors.append(f"no function of target {t} in {';'.join(group)}")
    for t, fs in by_target.items():
        for f in fs:
            errors += call_errors(t, f)

    mnemonics = {t: {mn for f in fs for mn, _ in f.insns} for t, fs in by_target.items()}
    if arch == "x86":
        avx2 = by_target["avx2"]
        if avx2 and not any(FMA.match(mn) for mn in mnemonics["avx2"]):
            errors.append("avx2: no FMA instruction in the avx2 code")
        if avx2 and not any("%ymm" in ops for f in avx2 for _, ops in f.insns):
            errors.append("avx2: no ymm register in the avx2 code")
        for f in avx2:
            for mn, ops in f.insns:
                if EVEX.search(ops):
                    errors.append(f"avx2: EVEX feature in {f.name}: {mn} {ops}")
        for f in by_target["scalar"] + baseline:
            for mn, ops in f.insns:
                vex = mn.startswith("v") and mn not in ("verr", "verw")
                vex = (vex or "%ymm" in ops) and "avx" not in baseline_has
                if vex or ("%zmm" in ops and "avx512" not in baseline_has):
                    errors.append(f"baseline: VEX or EVEX code in {f.name}: {mn} {ops}")
    else:
        fcma = mnemonics["neon_fcma"]
        for want in ("fcmla", "fcadd"):
            if by_target["neon_fcma"] and not any(mn.startswith(want) for mn in fcma):
                errors.append(f"neon_fcma: no {want.upper()} in the neon_fcma code")
        for f in by_target["scalar"] + baseline:
            for mn, ops in f.insns:
                if mn.startswith(("fcmla", "fcadd")) and "fcma" not in baseline_has:
                    errors.append(f"baseline: FCMA instruction in {f.name}: {mn} {ops}")
        for f in by_target["neon_fcma"]:
            for mn, ops in f.insns:
                base = mn.split(".")[0]
                if base.startswith("ldapr") or base == "fjcvtzs":
                    errors.append(f"Armv8.3 instruction outside FCMA in {f.name}: {mn} {ops}")
        for f in by_target["scalar"] + baseline:
            for mn, _ in f.insns:
                if is_pauth(mn.split(".")[0]) and "pauth" not in baseline_has:
                    errors.append(f"pointer authentication outside neon_fcma in {f.name}: {mn}")

    for t in TARGETS:
        fs = by_target[t]
        if fs:
            n = sum(len(f.insns) for f in fs)
            print(f"   {t:<10} {len(fs):3d} functions {n:6d} instructions")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--objdump", required=True, help="llvm-objdump or GNU objdump")
    ap.add_argument(
        "--baseline-has",
        action="append",
        default=[],
        choices=["avx", "avx512", "fcma", "pauth"],
        help="the build's baseline already has this (as when the compiler targets Apple"
        " silicon, which has FCMA): baseline code may use it",
    )
    ap.add_argument("groups", nargs="+", help="objects or archives; ';' joins one group")
    opts = ap.parse_args()
    for feature in opts.baseline_has:
        print(f"   the baseline has {feature}: baseline code may use it")
    errors: list[str] = []
    for arg in opts.groups:
        group = [p for p in arg.split(";") if p]
        missing = [p for p in group if not Path(p).is_file()]
        if missing:
            errors.append(f"not found: {', '.join(missing)}")
            continue
        print(f"== {', '.join(Path(p).name for p in group)}")
        errors += check_group(opts.objdump, group, set(opts.baseline_has))
    for e in errors:
        print(f"asm_check: {e}")
    if errors:
        print(f"asm_check: {len(errors)} problem(s)")
        return 1
    print("asm_check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
