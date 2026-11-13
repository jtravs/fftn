# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""DFT algorithms as DAG builders (DESIGN.md 7.2, step 1).

Every builder maps a list of input terms x_0 .. x_{n-1} to the output terms
X_k = sum_j x_j exp(sign 2 pi i jk / n); sign = -1 is the forward transform. Builders call
dft() for their sub-transforms, so algorithms nest: a Good-Thomas 12 uses split radix for
its 4-point and the symmetric form for its 3-point transforms.

The algorithm for a size is chosen by choose(); see gen/README.md for how to add one.
"""

from __future__ import annotations

from math import gcd

import mpmath

from .ir import ZERO, Graph, Term, rotate


def factor(n: int) -> list[tuple[int, int]]:
    """Prime factorisation as (prime, exponent) pairs, smallest prime first."""
    out, p = [], 2
    while p * p <= n:
        e = 0
        while n % p == 0:
            n, e = n // p, e + 1
        if e:
            out.append((p, e))
        p += 1
    if n > 1:
        out.append((n, 1))
    return out


def choose(n: int) -> tuple[str, int, int]:
    """(algorithm, n1, n2) for size n; n1 * n2 = n for the two-factor algorithms."""
    if n <= 2:
        return ("trivial", n, 1)
    primes = factor(n)
    if len(primes) == 1:
        p, e = primes[0]
        if p == 2:
            return ("split_radix", n, 1)
        if e == 1:
            return ("prime", n, 1)
        return ("cooley_tukey", p, n // p)
    # Coprime split: the power of the smallest prime against the rest.
    p, e = primes[0]
    return ("good_thomas", p**e, n // p**e)


def dft(g: Graph, x: list[Term], sign: int) -> list[Term]:
    n = len(x)
    match choose(n):
        case ("trivial", 1, _):
            return list(x)
        case ("trivial", 2, _):
            return [g.add(x[0], x[1]), g.sub(x[0], x[1])]
        case ("split_radix", _, _):
            return split_radix(g, x, sign)
        case ("prime", _, _):
            return prime(g, x, sign)
        case ("good_thomas", n1, n2):
            return good_thomas(g, x, sign, n1, n2)
        case ("cooley_tukey", n1, n2):
            return cooley_tukey(g, x, sign, n1, n2)
    raise ValueError(f"no algorithm for size {n}")


def _i_sign(sign: int) -> int:
    """The rotation k with (-i)^k = i^sign: exp(sign i pi / 2)."""
    return 1 if sign < 0 else 3


def split_radix(g: Graph, x: list[Term], sign: int) -> list[Term]:
    """Conjugate-pair split radix, decimation in time (n a power of two, n >= 4).

    X_k = U_k + (w^k Z_k + w^-k Z'_k) with U the half-length DFT of the even inputs and
    Z, Z' the quarter-length DFTs of x_{4m+1} and x_{4m-1}, w = exp(sign 2 pi i / n).
    Using w^-k on the second quarter (rather than w^3k) makes the two twiddles of each k
    a conjugate pair; the trivial ones (k = 0, and k = n/8 as (1 -+ i) sqrt(1/2)) cost
    nothing or one scale (Bernstein; Johnson and Frigo 2007 give the operation counts).
    """
    n = len(x)
    q = n // 4
    u = dft(g, x[0::2], sign)
    z = dft(g, [x[4 * m + 1] for m in range(q)], sign)
    zc = dft(g, [x[(4 * m - 1) % n] for m in range(q)], sign)
    out = [ZERO] * n
    for k in range(q):
        a = g.root_mul(z[k], -sign * k, n)
        b = g.root_mul(zc[k], sign * k, n)
        s = g.add(a, b)
        d = rotate(g.sub(a, b), _i_sign(sign))
        out[k] = g.add(u[k], s)
        out[k + 2 * q] = g.sub(u[k], s)
        out[k + q] = g.add(u[k + q], d)
        out[k + 3 * q] = g.sub(u[k + q], d)
    return out


def good_thomas(g: Graph, x: list[Term], sign: int, n1: int, n2: int) -> list[Term]:
    """Prime-factor (Good-Thomas) mapping for coprime n1, n2: no twiddle multiplies.

    Input j = (n2 j1 + n1 j2) mod n; output k is the index with k = k1 mod n1 and
    k = k2 mod n2 (Chinese remainder theorem). Then X_k is an n1-point DFT over j1 of the
    n2-point DFTs over j2, with no factor between them (Good 1958, Thomas 1963).
    """
    n = n1 * n2
    assert gcd(n1, n2) == 1
    inner = [dft(g, [x[(n2 * j1 + n1 * j2) % n] for j2 in range(n2)], sign) for j1 in range(n1)]
    out = [ZERO] * n
    for k2 in range(n2):
        col = dft(g, [inner[j1][k2] for j1 in range(n1)], sign)
        for k1 in range(n1):
            k = next(k for k in range(k2, n, n2) if k % n1 == k1)
            out[k] = col[k1]
    return out


def cooley_tukey(g: Graph, x: list[Term], sign: int, n1: int, n2: int) -> list[Term]:
    """Cooley-Tukey decimation in time with constant internal twiddles.

    Input j = n2 j1 + j2, output k = k1 + n1 k2: n1-point DFTs over j1, a factor
    w^(j2 k1) with w = exp(sign 2 pi i / n), then n2-point DFTs over j2.
    """
    n = n1 * n2
    inner = [dft(g, [x[n2 * j1 + j2] for j1 in range(n1)], sign) for j2 in range(n2)]
    out = [ZERO] * n
    for k1 in range(n1):
        col = [g.root_mul(inner[j2][k1], -sign * j2 * k1, n) for j2 in range(n2)]
        for k2, y in enumerate(dft(g, col, sign)):
            out[k1 + n1 * k2] = y
    return out


def _chain(g: Graph, start: Term, terms: list[tuple[Term, mpmath.mpf]]) -> Term:
    """start + sum c_j t_j, left to right, so that each step can become an fmadd_r."""
    acc = start
    for t, c in terms:
        acc = g.add(acc, g.scale(t, c))
    return acc


def prime(g: Graph, x: list[Term], sign: int) -> list[Term]:
    """Odd prime p, symmetric-pair form.

    With s_j = x_j + x_{p-j} and d_j = x_j - x_{p-j} (j = 1 .. m, m = (p-1)/2):
      X_0 = x_0 + sum s_j,   X_k, X_{p-k} = A_k +- i^sign B_k,
      A_k = x_0 + sum_j cos(2 pi jk / p) s_j,   B_k = sum_j sin(2 pi jk / p) d_j.
    The cosines of each row sum to -1/2, so with t = sum s_j and D_kj = cos + 1/(2m):
      A_k = (x_0 - t / (2m)) + sum_{j<m} D_kj (s_j - s_m),
    which needs m(m-1) + 1 multiplies instead of m^2 and the same number of additions.
    For p = 5 the two D entries are opposite, so one multiply serves both rows.
    """
    p = len(x)
    m = (p - 1) // 2
    s = [g.add(x[j], x[p - j]) for j in range(1, m + 1)]
    d = [g.sub(x[j], x[p - j]) for j in range(1, m + 1)]
    t = s[0]
    for sj in s[1:]:
        t = g.add(t, sj)
    out = [ZERO] * p
    out[0] = g.add(x[0], t)
    u = g.add(x[0], g.scale(t, -mpmath.mpf(1) / (2 * m)))
    e = [g.sub(s[j], s[m - 1]) for j in range(m - 1)]
    two_pi = 2 * mpmath.pi
    for k in range(1, m + 1):
        cos = [
            mpmath.cos(two_pi * ((j * k) % p) / p) + mpmath.mpf(1) / (2 * m) for j in range(1, m)
        ]
        a = _chain(g, u, list(zip(e, cos, strict=True)))
        sin = [mpmath.sin(two_pi * ((j * k) % p) / p) for j in range(1, m + 1)]
        b = _chain(g, g.scale(d[0], sin[0]), list(zip(d[1:], sin[1:], strict=True)))
        rb = rotate(b, _i_sign(sign))
        out[k] = g.add(a, rb)
        out[p - k] = g.sub(a, rb)
    return out
