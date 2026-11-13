# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The reference table of operation counts (DESIGN.md 7.3) and the comparison with it.

Counts are real additions and real multiplications of a complex DFT without fused
operations, a complex multiply being 4 multiplies and 2 additions: the convention of the
published tables. The plain counts are the best published for each size: split radix for
2^k (3n log2 n - 3n + 4 flops in all; Johnson and Frigo 2007, table 1), and for the other
sizes the counts of FFTW's generated codelets (Frigo, PLDI 1999; FFTW 3.3), where 13 uses
Rader's algorithm. A twiddle variant adds R - 1 complex multiplies.
"""

from __future__ import annotations

from .codelet import Codelet

PLAIN: dict[int, tuple[int, int]] = {
    2: (4, 0),
    3: (12, 4),
    4: (16, 0),
    5: (32, 12),
    6: (36, 8),
    7: (60, 36),
    8: (52, 4),
    9: (80, 40),
    10: (84, 24),
    11: (140, 100),
    12: (96, 16),
    13: (176, 68),
    14: (148, 72),
    15: (156, 56),
    16: (144, 24),
}

# Sizes where the generator does worse than the reference, with its counts for the plain
# variant and the reason. Any change to these (better or worse) fails the tests, so a
# gap is never widened silently and a closed gap is noticed.
GAPS: dict[int, tuple[tuple[int, int], str]] = {
    13: (
        (192, 134),
        "the symmetric-pair form costs about m^2 real multiplies per part for m = 6 "
        "pairs; the reference uses Rader's algorithm (a length-12 cyclic convolution), "
        "which is outside task 1.1",
    ),
}


def reference(radix: int, variant: str) -> tuple[int, int]:
    adds, muls = PLAIN[radix]
    if variant != "plain":
        adds, muls = adds + 2 * (radix - 1), muls + 4 * (radix - 1)
    return adds, muls


def compare(c: Codelet) -> str | None:
    """None if c reaches the reference, or the recorded gap exactly; otherwise a message."""
    ours = (c.counts.flops_adds, c.counts.flops_muls)
    ref = reference(c.radix, c.variant)
    if c.radix in GAPS:
        plain, _ = GAPS[c.radix]
        extra = (0, 0) if c.variant == "plain" else (2 * (c.radix - 1), 4 * (c.radix - 1))
        expected = (plain[0] + extra[0], plain[1] + extra[1])
        if ours != expected:
            return f"radix {c.radix} {c.variant}: {ours}, recorded gap says {expected}"
        return None
    if ours[0] > ref[0] or ours[1] > ref[1]:
        return f"radix {c.radix} {c.variant}: {ours} adds/muls, reference {ref}"
    return None


def table(codelets: list[Codelet]) -> str:
    """Markdown table of the forward codelets against the reference."""
    lines = [
        "| R | variant | adds | muls | reference | vector ops | fmadd_r | peak live |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in codelets:
        if c.sign > 0:
            continue
        ref = reference(c.radix, c.variant)
        k = c.counts
        lines.append(
            f"| {c.radix} | {c.variant} | {k.flops_adds} | {k.flops_muls} | {ref[0]} / {ref[1]} "
            f"| {k.vector_ops} | {k.fma_ops} | {c.live} |"
        )
    return "\n".join(lines)
