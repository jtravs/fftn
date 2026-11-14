# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Codelets: a radix, a variant and a direction, built, lowered, ordered and verified.

Variants (DESIGN.md 7.3), with w_k the forward twiddle of leg k (k = 1 .. R-1) and
F the R-point DFT of the codelet's direction:
  plain  y = F x
  dit    y = F (x_0, x_1 t_1, ..., x_{R-1} t_{R-1})      (multiply inputs)
  dif    y_0 = (F x)_0,  y_k = (F x)_k t_k               (multiply outputs)
with t_k = w_k forward and conj(w_k) backward: twiddle tables hold forward roots only
(DESIGN.md decision log), and a backward pass multiplies by their conjugates (mulc).
"""

from __future__ import annotations

from dataclasses import dataclass

from .algorithms import dft
from .ir import Graph
from .lower import Op, lower
from .realify import realify
from .schedule import peak_live, schedule
from .split import SPLIT_FUSED, expand, lower_split
from .verify import verify, verify_split

VARIANTS = ("plain", "dit", "dif")
SIGNS = (-1, 1)


@dataclass(frozen=True, slots=True)
class Counts:
    """Operation counts of one codelet (gen/README.md, "Metadata").

    Vocabulary operations: add-type (add, sub, addj, subj), scale, fmadd_r, cmul (mul or
    mulc by a twiddle). Real arithmetic on a fused target, cmuls apart: adds, muls,
    fmas. Real arithmetic without fused operations and with each cmul as 4 multiplies and
    2 additions (the convention of the published counts): flops_adds, flops_muls.
    """

    add_ops: int
    scale_ops: int
    fma_ops: int
    cmuls: int

    @property
    def adds(self) -> int:
        return 2 * self.add_ops

    @property
    def muls(self) -> int:
        return 2 * self.scale_ops

    @property
    def fmas(self) -> int:
        return 2 * self.fma_ops

    @property
    def vector_ops(self) -> int:
        return self.add_ops + self.scale_ops + self.fma_ops + self.cmuls

    @property
    def flops_adds(self) -> int:
        return self.adds + self.fmas + 2 * self.cmuls

    @property
    def flops_muls(self) -> int:
        return self.muls + self.fmas + 4 * self.cmuls


@dataclass(frozen=True, slots=True)
class Codelet:
    radix: int
    variant: str
    sign: int
    ops: tuple[Op, ...]
    counts: Counts
    live: int

    @property
    def twiddles(self) -> int:
        return 0 if self.variant == "plain" else self.radix - 1


def count(ops: list[Op] | tuple[Op, ...]) -> Counts:
    kinds = [op.kind for op in ops]
    return Counts(
        add_ops=sum(k in ("add", "sub", "addj", "subj") for k in kinds),
        scale_ops=kinds.count("scale"),
        fma_ops=kinds.count("fmadd_r"),
        cmuls=kinds.count("mul") + kinds.count("mulc"),
    )


def build_ops(radix: int, variant: str, sign: int, real: bool = True) -> list[Op]:
    """The codelet's operations in topological order, before ordering.

    real: lower the real-coefficient form (realify.py), which every generated codelet
    uses; False lowers the DAG as built (kept for the tests that compare the two).
    """
    g = Graph(radix, twiddles=radix - 1)
    x = [g.input(j) for j in range(radix)]
    conj = sign > 0
    if variant == "dit":
        x = [x[0]] + [g.tw_mul(x[j], g.twiddle(j), conj) for j in range(1, radix)]
    y = dft(g, x, sign)
    if variant == "dif":
        y = [y[0]] + [g.tw_mul(y[k], g.twiddle(k), conj) for k in range(1, radix)]
    if real:
        g, y = realify(g, y, radix)
    return lower(g, y)


def build(radix: int, variant: str, sign: int) -> Codelet:
    """Builds and verifies one codelet; raises VerificationError if it is wrong."""
    ops = schedule(build_ops(radix, variant, sign))
    verify(ops, radix, variant, sign)
    return Codelet(radix, variant, sign, tuple(ops), count(ops), peak_live(ops))


# Variants of the split lowering: the ones the passes call (src/passes/split.inl). Forward
# only: the backward transform exchanges real and imaginary parts (split.py).
SPLIT_VARIANTS = ("plain", "dif")


@dataclass(frozen=True, slots=True)
class SplitCounts:
    """Real-register operations of a split codelet, each on W reals (gen/README.md,
    "Metadata"): add and sub; mul and mulk; the fused forms; neg."""

    adds: int
    muls: int
    fmas: int
    negs: int

    @property
    def vector_ops(self) -> int:
        return self.adds + self.muls + self.fmas + self.negs

    @property
    def flops_adds(self) -> int:
        return self.adds + self.fmas

    @property
    def flops_muls(self) -> int:
        return self.muls + self.fmas


@dataclass(frozen=True, slots=True)
class SplitCodelet:
    radix: int
    variant: str
    ops: tuple[Op, ...]
    counts: SplitCounts
    live: int

    @property
    def twiddles(self) -> int:
        return 0 if self.variant == "plain" else self.radix - 1


def count_split(ops: list[Op] | tuple[Op, ...]) -> SplitCounts:
    kinds = [op.kind for op in ops]
    return SplitCounts(
        adds=sum(k in ("add", "sub") for k in kinds),
        muls=sum(k in ("mul", "mulk") for k in kinds),
        fmas=sum(k in SPLIT_FUSED for k in kinds),
        negs=kinds.count("neg"),
    )


def build_split_ops(radix: int, variant: str) -> list[Op]:
    """The forward split codelet's operations in topological order, before ordering."""
    g = Graph(radix, twiddles=radix - 1)
    x = [g.input(j) for j in range(radix)]
    y = dft(g, x, -1)
    if variant == "dif":
        y = [y[0]] + [g.tw_mul(y[k], g.twiddle(k), False) for k in range(1, radix)]
    h, parts = expand(g, y)
    return lower_split(h, parts)


def build_split(radix: int, variant: str) -> SplitCodelet:
    """Builds and verifies one forward split codelet."""
    ops = schedule(build_split_ops(radix, variant))
    verify_split(ops, radix, variant)
    return SplitCodelet(radix, variant, tuple(ops), count_split(ops), peak_live(ops))
