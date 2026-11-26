# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""What is generated: the radices, variants and directions, and the output files.

This is the only list of radices in FFTN: the C++ side reads src/gen/codelets.hpp, which
is written from it (DESIGN.md 7.2, step 7). Adding a radix: gen/README.md.
"""

from __future__ import annotations

from .codelet import (
    PRUNED_VARIANTS,
    SIGNS,
    SPLIT_VARIANTS,
    VARIANTS,
    Codelet,
    SplitCodelet,
    build,
    build_split,
)
from .emit_cpp import codelet_inl, metadata_hpp, split_codelet_inl

# Every size 2 to 16; the primes 17, 19 and 23, the leaves of those lengths and the passes
# of those factors in composites, which otherwise went through Bluestein (experiment X27);
# 20, a pass radix only (never a leaf), which FFTW's plans use at 160-20000 points and
# which saves a pass there (experiment X38); 25, a pass radix only, which FFTW's plans use
# for odd parts 5^4 and 5^5 (X39, experiment X42); then the powers of two that
# whole-transform leaves (experiment X8) and few-big-pass chains (task 1.6, experiment X3)
# use. Ascending: the C++ tables list the rows in this order.
RADICES = (*range(2, 18), 19, 20, 23, 25, 32, 64)

# The split lowering's radices (task 1.5, experiment X1): the lane passes of a chain, which
# the planner prices up to 16. A split codelet holds about twice the live registers of a
# complex one (radix 16: 34 to 40 reals against 18 complex), so 32 and 64 would spill
# heavily.
SPLIT_RADICES = tuple(range(2, 17))

# The pruned codelets' radices (DESIGN.md 12.1, task 5.1e): the even codelet radices, whose
# halves h = R/2 a stored run of half a line fills exactly (the valid half of a zero-padded
# convolution's row). The edge passes of a line with stored runs take them: the chain's first
# pass reads only legs 0 .. h - 1 (dif_head) when the input's runs lie in the first h m
# elements of the line's n = R m, and its last pass writes only legs 0 .. h - 1 (plain_head)
# when the output's lie in the first h s, n = R s. Halves only: `--table` prints the counts of
# both variants against their full codelets, which is how the set was chosen (gen/README.md,
# "Pruned codelets").
PRUNED_RADICES = tuple(r for r in RADICES if r % 2 == 0)


def all_codelets() -> list[Codelet]:
    return [build(r, v, s) for r in RADICES for v in VARIANTS for s in SIGNS]


def all_pruned_codelets() -> list[Codelet]:
    return [build(r, v, s) for r in PRUNED_RADICES for v in PRUNED_VARIANTS for s in SIGNS]


def all_split_codelets() -> list[SplitCodelet]:
    return [build_split(r, v) for r in SPLIT_RADICES for v in SPLIT_VARIANTS]


def generate(
    codelets: list[Codelet] | None = None,
    split: list[SplitCodelet] | None = None,
    pruned: list[Codelet] | None = None,
) -> dict[str, str]:
    """File name (under src/gen/) to contents."""
    if codelets is None:
        codelets = all_codelets()
    if split is None:
        split = all_split_codelets()
    if pruned is None:
        pruned = all_pruned_codelets()
    return {
        "complex_codelets.inl": codelet_inl(codelets + pruned),
        "split_codelets.inl": split_codelet_inl(split),
        "codelets.hpp": metadata_hpp(codelets, split, pruned),
    }
