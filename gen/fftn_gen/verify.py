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
_MI = mpmath.mpc(0, -1)


def evaluate(ops: list[Op] | tuple[Op, ...], x: list, w: list) -> list:
    """Runs the operations on inputs x and twiddles w (w[k] for leg k; w[0] unused)."""
    env: dict[int, mpmath.mpc] = {}
    out: dict[int, mpmath.mpc] = {}
    for op in ops:
        a = [env[v] for v in op.args]
        match op.kind:
            case "load":
                env[op.dst] = x[op.index]
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
            case "neg":
                env[op.dst] = -a[0]
            case "rot_m":
                env[op.dst] = _MI * a[0]
            case "rot_p":
                env[op.dst] = -_MI * a[0]
            case "store":
                if op.index in out:
                    raise VerificationError(f"output {op.index} stored twice")
                out[op.index] = a[0]
            case _:
                raise VerificationError(f"unknown operation {op.kind}")
    return [out[k] for k in range(len(out))]


def reference(radix: int, variant: str, sign: int, x: list, w: list) -> list:
    def t(k: int) -> mpmath.mpc:
        if k == 0:
            return mpmath.mpc(1)
        return mpmath.conj(w[k]) if sign > 0 else w[k]

    if variant == "dit":
        x = [x[j] * t(j) for j in range(radix)]
    y = [
        mpmath.fsum(
            x[j] * mpmath.expjpi(mpmath.mpf(2 * sign * ((j * k) % radix)) / radix)
            for j in range(radix)
        )
        for k in range(radix)
    ]
    if variant == "dif":
        y = [y[k] * t(k) for k in range(radix)]
    return y


def verify(
    ops: list[Op] | tuple[Op, ...], radix: int, variant: str, sign: int, seed: int = 7
) -> None:
    rng = random.Random(seed)

    def rand() -> mpmath.mpc:
        return mpmath.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1))

    stores = sorted(op.index for op in ops if op.kind == "store")
    if stores != list(range(radix)):
        raise VerificationError(f"radix {radix}: stores {stores}, expected each output once")
    cases = [[mpmath.mpc(int(j == i)) for j in range(radix)] for i in range(radix)]
    cases += [[rand() for _ in range(radix)] for _ in range(3)]
    for x in cases:
        w = [mpmath.mpc(1)] + [rand() for _ in range(radix - 1)]
        got = evaluate(ops, x, w)
        want = reference(radix, variant, sign, x, w)
        scale = max(mpmath.mpf(1), *(abs(v) for v in want))
        for k, (a, b) in enumerate(zip(got, want, strict=True)):
            if abs(a - b) > _TOLERANCE * scale:
                raise VerificationError(
                    f"radix {radix} {variant} sign {sign}: output {k} is {mpmath.nstr(a, 12)}, "
                    f"expected {mpmath.nstr(b, 12)}"
                )
