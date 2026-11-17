# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Verification (DESIGN.md 7.2, step 6): the emitted operations against the exact DFT.

The operation list (the same objects the emitter prints, one statement each) is evaluated
in 256-bit arithmetic on every basis input, which checks every column of the codelet's
matrix, and on random inputs, each with random twiddles. The reference is the definition
of the variant with roots of unity whose angles are reduced exactly. A codelet with any
output off by more than 2^-200 relative is an error and is never written.
"""

from __future__ import annotations

import random

import mpmath

from .lower import Op


class VerificationError(Exception):
    pass


_TOLERANCE = mpmath.mpf(2) ** -200


def evaluate(ops: list[Op] | tuple[Op, ...], x: list, w: list) -> list:
    """Runs the operations on inputs x and twiddles w (w[k] for leg k; w[0] unused)."""
    env: dict[int, mpmath.mpc] = {}
    out: dict[int, mpmath.mpc] = {}
    stage: dict[int, mpmath.mpc] = {}  # the slots of a two-level codelet (staged.py)
    for op in ops:
        a = [env[v] for v in op.args]
        match op.kind:
            case "load":
                env[op.dst] = x[op.index]
            case "stage_store":
                if op.index in stage:
                    raise VerificationError(f"slot {op.index} stored twice")
                stage[op.index] = a[0]
            case "stage_load":
                env[op.dst] = stage[op.index]
            case "load_tw":
                env[op.dst] = w[op.index]
            case "add":
                env[op.dst] = a[0] + a[1]
            case "sub":
                env[op.dst] = a[0] - a[1]
            case "addj":
                env[op.dst] = a[0] + 1j * a[1]
            case "subj":
                env[op.dst] = a[0] - 1j * a[1]
            case "scale":
                env[op.dst] = op.const * a[0]
            case "fmadd_r":
                env[op.dst] = a[0] + op.const * a[1]
            case "mul":
                env[op.dst] = a[0] * a[1]
            case "mulc":
                env[op.dst] = a[0] * mpmath.conj(a[1])
            case "store":
                if op.index in out:
                    raise VerificationError(f"output {op.index} stored twice")
                out[op.index] = a[0]
            case _:
                raise VerificationError(f"unknown operation {op.kind}")
    return [out[k] for k in range(len(out))]


def reference(radix: int, variant: str, sign: int, x: list, w: list) -> list:
    def t(k: int) -> mpmath.mpc:
        return mpmath.conj(w[k]) if sign > 0 else w[k]

    if variant == "dit":
        x = [x[0]] + [x[j] * t(j) for j in range(1, radix)]
    y = [
        mpmath.fsum(
            x[j] * mpmath.expjpi(mpmath.mpf(2 * sign * ((j * k) % radix)) / radix)
            for j in range(radix)
        )
        for k in range(radix)
    ]
    if variant == "dif":
        y = [y[0]] + [y[k] * t(k) for k in range(1, radix)]
    return y


def check_shape(ops: list[Op] | tuple[Op, ...], radix: int, variant: str) -> None:
    """The accessor contract (gen/README.md, "Calling convention"): every input leg loaded
    once, every twiddle leg 1 .. R-1 once (none for plain), every output stored once.

    Checked apart from the values because a stray access can be numerically harmless: a
    load of twiddle 0, which the tables do not hold, multiplies by whatever lies before
    the table, and the evaluation below would have to guess a value for it.
    """
    where = f"radix {radix} {variant}"
    expected = {
        "load": list(range(radix)),
        "load_tw": [] if variant == "plain" else list(range(1, radix)),
        "store": list(range(radix)),
    }
    for kind, want in expected.items():
        got = sorted(op.index for op in ops if op.kind == kind)
        if got != want:
            raise VerificationError(f"{where}: {kind} indices {got}, expected {want} once each")


def verify(
    ops: list[Op] | tuple[Op, ...], radix: int, variant: str, sign: int, seed: int = 7
) -> None:
    rng = random.Random(seed)

    def rand() -> mpmath.mpc:
        return mpmath.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1))

    check_shape(ops, radix, variant)
    cases = [[mpmath.mpc(int(j == i)) for j in range(radix)] for i in range(radix)]
    cases += [[rand() for _ in range(radix)] for _ in range(3)]
    for x in cases:
        w = [None] + [rand() for _ in range(radix - 1)]  # leg 0 has no twiddle
        got = evaluate(ops, x, w)
        want = reference(radix, variant, sign, x, w)
        scale = max(mpmath.mpf(1), *(abs(v) for v in want))
        for k, (a, b) in enumerate(zip(got, want, strict=True)):
            if abs(a - b) > _TOLERANCE * scale:
                raise VerificationError(
                    f"radix {radix} {variant} sign {sign}: output {k} is {mpmath.nstr(a, 12)}, "
                    f"expected {mpmath.nstr(b, 12)}"
                )


def evaluate_split(ops: list[Op] | tuple[Op, ...], x: list, w: list) -> list:
    """Runs split operations (split.py) on complex inputs x and twiddles w."""
    env: dict[int, mpmath.mpf] = {}
    out: dict[int, mpmath.mpc] = {}
    for op in ops:
        a = [env[v] for v in op.args]
        match op.kind:
            case "load_re":
                env[op.dst] = mpmath.re(x[op.index])
            case "load_im":
                env[op.dst] = mpmath.im(x[op.index])
            case "load_twr":
                env[op.dst] = mpmath.re(w[op.index])
            case "load_twi":
                env[op.dst] = mpmath.im(w[op.index])
            case "add":
                env[op.dst] = a[0] + a[1]
            case "sub":
                env[op.dst] = a[0] - a[1]
            case "mul":
                env[op.dst] = a[0] * a[1]
            case "mulk":
                env[op.dst] = op.const * a[0]
            case "fmaddk":
                env[op.dst] = op.const * a[0] + a[1]
            case "fnmaddk":
                env[op.dst] = a[1] - op.const * a[0]
            case "fmsubk":
                env[op.dst] = op.const * a[0] - a[1]
            case "fmadd":
                env[op.dst] = a[0] * a[1] + a[2]
            case "fnmadd":
                env[op.dst] = a[2] - a[0] * a[1]
            case "fmsub":
                env[op.dst] = a[0] * a[1] - a[2]
            case "neg":
                env[op.dst] = -a[0]
            case "store":
                if op.index in out:
                    raise VerificationError(f"output {op.index} stored twice")
                out[op.index] = mpmath.mpc(a[0], a[1])
            case _:
                raise VerificationError(f"unknown split operation {op.kind}")
    return [out[k] for k in range(len(out))]


def check_shape_split(ops: list[Op] | tuple[Op, ...], radix: int, variant: str) -> None:
    """The split accessor contract: both parts of every input leg loaded once, both parts of
    every twiddle leg 1 .. R-1 once (none for plain), every output stored once."""
    where = f"split radix {radix} {variant}"
    legs = list(range(radix))
    twiddle_legs = [] if variant == "plain" else list(range(1, radix))
    expected = {
        "load_re": legs,
        "load_im": legs,
        "load_twr": twiddle_legs,
        "load_twi": twiddle_legs,
        "store": legs,
    }
    for kind, want in expected.items():
        got = sorted(op.index for op in ops if op.kind == kind)
        if got != want:
            raise VerificationError(f"{where}: {kind} indices {got}, expected {want} once each")


def verify_split(ops: list[Op] | tuple[Op, ...], radix: int, variant: str, seed: int = 7) -> None:
    """A forward split codelet against the definition, as verify() does for the complex
    form; the basis inputs include the imaginary units, which the real parts do not reach."""
    rng = random.Random(seed)

    def rand() -> mpmath.mpc:
        return mpmath.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1))

    check_shape_split(ops, radix, variant)
    cases = [[mpmath.mpc(int(j == i)) for j in range(radix)] for i in range(radix)]
    cases += [[mpmath.mpc(0, int(j == i)) for j in range(radix)] for i in range(radix)]
    cases += [[rand() for _ in range(radix)] for _ in range(3)]
    for x in cases:
        w = [None] + [rand() for _ in range(radix - 1)]
        got = evaluate_split(ops, x, w)
        want = reference(radix, variant, -1, x, w)
        scale = max(mpmath.mpf(1), *(abs(v) for v in want))
        for k, (a, b) in enumerate(zip(got, want, strict=True)):
            if abs(a - b) > _TOLERANCE * scale:
                raise VerificationError(
                    f"split radix {radix} {variant}: output {k} is {mpmath.nstr(a, 12)}, "
                    f"expected {mpmath.nstr(b, 12)}"
                )
