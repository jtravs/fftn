# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""What is generated: the radices, variants and directions, and the output files.

This is the only list of radices in FFTN: the C++ side reads src/gen/codelets.hpp, which
is written from it (DESIGN.md 7.2, step 7). Adding a radix: gen/README.md.
"""

from __future__ import annotations

from .codelet import (
    SIGNS,
    SPLIT_VARIANTS,
    VARIANTS,
    Codelet,
    SplitCodelet,
    build,
    build_split,
)
from .emit_cpp import codelet_inl, metadata_hpp, split_codelet_inl
from .staged import build_staged

# Every size 2 to 16; the primes 17, 19 and 23, the leaves of those lengths and the passes
# of those factors in composites, which otherwise went through Bluestein (experiment X27);
# 20, a pass radix only (never a leaf), which FFTW's plans use at 160-20000 points and
# which saves a pass there (experiment X38); 25, a pass radix only, which FFTW's plans use
# for odd parts 5^4 and 5^5 (X39, experiment X42); then the powers of two that
# whole-transform leaves (experiment X8) and few-big-pass chains (task 1.6, experiment X3)
# use. Ascending: the C++ tables list the rows in this order.
RADICES = (*range(2, 18), 19, 20, 23, 25, 32, 64)

# The radices whose plain and dif codelets also have a two-level form, R = n1 x n2 staged
# through memory (staged.py), which targets with 16 vector registers compile instead of the
# flat form (codelet_registers in src/isa/vocabulary.inl): the flat forms peak at 29, 35
# and 67 live values (experiment X42). dit keeps its flat form: no pass calls it.
STAGED = {25: (5, 5), 32: (8, 4), 64: (8, 8)}
STAGED_VARIANTS = ("plain", "dif")

# The split lowering's radices (task 1.5, experiment X1): the lane passes of a chain, which
# the planner prices up to 16. A split codelet holds about twice the live registers of a
# complex one (radix 16: 34 to 40 reals against 18 complex), so 32 and 64 would spill
# heavily.
SPLIT_RADICES = tuple(range(2, 17))


def all_codelets() -> list[Codelet]:
    return [build(r, v, s) for r in RADICES for v in VARIANTS for s in SIGNS]


def all_staged_codelets() -> list[Codelet]:
    return [
        build_staged(r, n1, n2, v, s)
        for r, (n1, n2) in STAGED.items()
        for v in STAGED_VARIANTS
        for s in SIGNS
    ]


def all_split_codelets() -> list[SplitCodelet]:
    return [build_split(r, v) for r in SPLIT_RADICES for v in SPLIT_VARIANTS]


def generate(
    codelets: list[Codelet] | None = None,
    split: list[SplitCodelet] | None = None,
    staged: list[Codelet] | None = None,
) -> dict[str, str]:
    """File name (under src/gen/) to contents."""
    if codelets is None:
        codelets = all_codelets()
    if staged is None:
        staged = all_staged_codelets()
    if split is None:
        split = all_split_codelets()
    return {
        "complex_codelets.inl": codelet_inl(codelets, staged, STAGED),
        "split_codelets.inl": split_codelet_inl(split),
        "codelets.hpp": metadata_hpp(codelets, split),
    }
