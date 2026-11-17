# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The reference table of operation counts (DESIGN.md 7.3) and the comparison with it.

Counts are real additions and real multiplications of a complex DFT without fused
operations, a complex multiply being 4 multiplies and 2 additions: the convention of the
published tables. A twiddle variant adds R - 1 complex multiplies to the plain count.

Each entry says where its numbers come from:
- published: the operation counts stated in the headers of FFTW 3.3's generated
  no-twiddle codelets n1_N (genfft; Frigo, "A fast Fourier transform compiler", PLDI
  1999), and for powers of two also the split-radix count, 4n log2 n - 6n + 8 real
  operations in all (n = 16: 144 additions and 24 multiplications, 168; Johnson and
  Frigo, "A modified split-radix FFT with fewer arithmetic operations", 2007). Twiddle
  variants match the t1_N headers (for example t1_16: 174 and 84; t1_32: 434 and 208;
  t1_64: 1038 and 500);
- derived: no count is cited for the size, so the reference is the sum of its parts
  under the algorithm the generator uses, from the entries above (good_thomas());
- none: no reference, with the reason; such a radix is generated without a check.
"""

from __future__ import annotations

from dataclasses import dataclass

from .codelet import Codelet


@dataclass(frozen=True, slots=True)
class Reference:
    counts: tuple[int, int] | None  # plain variant: (additions, multiplications)
    source: str  # "published: ...", "derived: ..." or "none: <reason>"


def published(adds: int, muls: int, codelet: str) -> Reference:
    return Reference((adds, muls), f"published: FFTW 3.3 {codelet}")


def good_thomas(n1: int, n2: int) -> Reference:
    """n2 DFTs of size n1 and n1 of size n2, with no factor between them."""
    a1, m1 = PLAIN[n1].counts
    a2, m2 = PLAIN[n2].counts
    return Reference((n2 * a1 + n1 * a2, n2 * m1 + n1 * m2), f"derived: Good-Thomas {n1} x {n2}")


def cooley_tukey(n1: int, n2: int) -> Reference:
    """n2 DFTs of size n1 and n1 of size n2, and (n1 - 1)(n2 - 1) general complex multiplies
    between them (for odd n1 and n2 no internal factor is a quarter turn or an eighth)."""
    a1, m1 = PLAIN[n1].counts
    a2, m2 = PLAIN[n2].counts
    t = (n1 - 1) * (n2 - 1)
    return Reference(
        (n2 * a1 + n1 * a2 + 2 * t, n2 * m1 + n1 * m2 + 4 * t),
        f"derived: Cooley-Tukey {n1} x {n2}",
    )


PLAIN: dict[int, Reference] = {
    2: published(4, 0, "n1_2"),
    3: published(12, 4, "n1_3"),
    4: published(16, 0, "n1_4"),
    5: published(32, 12, "n1_5"),
    7: published(60, 36, "n1_7"),
    8: published(52, 4, "n1_8"),
    9: published(80, 40, "n1_9"),
    11: published(140, 100, "n1_11"),
    12: published(96, 16, "n1_12"),
    13: published(176, 68, "n1_13 (Rader's algorithm)"),
    15: published(156, 56, "n1_15"),
    16: published(144, 24, "n1_16"),
    32: published(372, 84, "n1_32"),
    64: published(912, 248, "n1_64"),
}
# FFTW 3.3 generates no codelet for these primes, and no other count is cited; the
# symmetric-pair form costs m(m - 1) + 1 multiplies per part for m = (p - 1)/2 pairs, where
# Rader's algorithm would cost fewer (13's gap below).
for _p in (17, 19, 23):
    PLAIN[_p] = Reference(None, "none: no published count for this prime")
PLAIN[6] = good_thomas(2, 3)
PLAIN[10] = good_thomas(2, 5)
PLAIN[14] = good_thomas(2, 7)
PLAIN[20] = good_thomas(4, 5)
PLAIN[25] = cooley_tukey(5, 5)

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


class MissingReference(KeyError):
    pass


def _twiddle_extra(radix: int, variant: str) -> tuple[int, int]:
    return (0, 0) if variant == "plain" else (2 * (radix - 1), 4 * (radix - 1))


def reference(
    radix: int, variant: str, plain: dict[int, Reference] = PLAIN
) -> tuple[int, int] | None:
    """The reference counts, or None for an entry marked "none"."""
    if radix not in plain:
        raise MissingReference(
            f"radix {radix} has no entry in reference.PLAIN: add a published count, a derived "
            'one or none with a reason (gen/README.md, "Adding a radix")'
        )
    counts = plain[radix].counts
    if counts is None:
        return None
    extra = _twiddle_extra(radix, variant)
    return counts[0] + extra[0], counts[1] + extra[1]


def compare(c: Codelet, plain: dict[int, Reference] = PLAIN) -> str | None:
    """None if c reaches the reference, matches its recorded gap exactly, or has no
    reference by declaration; otherwise a message."""
    ours = (c.counts.flops_adds, c.counts.flops_muls)
    ref = reference(c.radix, c.variant, plain)
    if c.radix in GAPS:
        gap, _ = GAPS[c.radix]
        extra = _twiddle_extra(c.radix, c.variant)
        expected = (gap[0] + extra[0], gap[1] + extra[1])
        if ours != expected:
            return f"radix {c.radix} {c.variant}: {ours}, recorded gap says {expected}"
        return None
    if ref is not None and (ours[0] > ref[0] or ours[1] > ref[1]):
        return f"radix {c.radix} {c.variant}: {ours} adds/muls, reference {ref}"
    return None


def table(codelets: list[Codelet], plain: dict[int, Reference] = PLAIN) -> str:
    """Markdown table of the forward codelets against the reference."""
    lines = [
        "| R | variant | adds | muls | reference | source | vector ops | fmadd_r | peak live |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in codelets:
        if c.sign > 0:
            continue
        ref = reference(c.radix, c.variant, plain)
        shown = "none" if ref is None else f"{ref[0]} / {ref[1]}"
        k = c.counts
        lines.append(
            f"| {c.radix} | {c.variant} | {k.flops_adds} | {k.flops_muls} | {shown} "
            f"| {plain[c.radix].source} | {k.vector_ops} | {k.fma_ops} | {c.live} |"
        )
    return "\n".join(lines)
