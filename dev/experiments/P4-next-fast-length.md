# P4: are `next_fast_length`'s lengths fast?

- **Status:** done (rule 1 fails on x86 at short lengths; rule 2 applied twice: composites with 11, 17, 19 or 23 left the fast set)
- **Owner:** task/2.5-next-fast-length session, 2026-11-17
- **Phase / task:** 2.5 (`next_fast_length` and published per-length cost)

## Question

`next_fast_length(n)` (include/fftn/lengths.hpp) returns the smallest length whose row the
reference cost table (AVX2 double, X2's measured one) prices per point within 1.25 times the
power of two at or above it. Is it fast on real machines?

The fast set's composites with 11 and 13 rest on columns scaled from radix 10 and 12 rather
than measured, and those with 17, 19 and 23 on a scaling of the radix-16 pass. Should the
set be restricted to fewer prime factors before the API is published?

## Hypothesis

Measured per point, a fast length is within 1.3 times the power of two above it, and well
below the awkward length it replaces. Composites with 11–23 may be slower than the table
believes.

## Change under test

None: a measurement of the new function's choices. One arm, `N` = `fftn` at the task's
commit, with fbench's identical-arm control.

## Method

- Cases: `bench/catalogues/p4.toml`, single c2c rows in double. 26 awkward lengths n (primes
  and odd lengths from 37 to 250007), each with `next_fast_length(n)` and the power of two at
  or above n.
- Machines: the Mac (Clang 23, `neon_fcma`), the EPYC 7763 (GCC 16, `avx2`) and the Xeon 6240
  (GCC 14, `avx512`). One thread.
- Metrics, per pair with m = `next_fast_length(n)` ≠ n:
  1. ns per point of m over ns per point of the power of two P at or above n;
  2. the time of m over the time of n.
- Decision rule, written before running:
  1. On every machine, at least 90 % of the pairs have metric 1 at most 1.3, and every
     pair whose n has a prime factor above 23 has metric 2 below 1.
  2. If the misses on two machines or more share a prime factor among 11, 13, 17, 19 and 23,
     that factor leaves the fast set (in composites; its single codelet stays), the test's
     golden list is regenerated, and the change is recorded in the decision log.
  3. Otherwise the tolerance of 1.25 and the fast set stay as built.

## Results

### First pass (commit 5750d70)

| Machine | Run | Load, A/A band | Metric 1 within 1.3 | Metric 2 (n with a factor above 23) |
|---|---|---|---|---|
| Mac M5 Max, Clang 23 | 20261117-043206-8762 | 4 of 6 rounds loaded, ±5.16 % | 22 of 23 (96 %) | all below 1 (0.17–0.48, 250 → 252 0.96, 1001 → 1008 0.75) |
| EPYC 7763, GCC 16 | 20261117-043640-9251 | 6 of 6 loaded, ±2.13 % | 18 of 23 (78 %) | all below 1 (0.14–0.86) |
| Xeon 6240, GCC 14 | 20261117-044629-3274 | 0 of 6 loaded, ±1.74 % | 18 of 23 (78 %) | all below 1 (0.11–0.76) |

Misses (the pick's ns per point over its power of two's):
- **Mac:** 1020 (2² · 3 · 5 · 17), 1.38.
- **EPYC:**
  - 132 (2² · 3 · 11), 1.40;
  - 252 (2² · 3² · 7), 1.38;
  - 304 (2⁴ · 19), 1.51;
  - 150144 (2⁷ · 3 · 17 · 23), 1.47;
  - 250240 (2⁷ · 5 · 17 · 23), 1.56.
- **Xeon:**
  - 100 (2² · 5²), 1.66;
  - 112 (2⁴ · 7), 1.43;
  - 132, 1.68;
  - 252, 1.54;
  - 1020, 1.32.

Rule 2: the misses shared by two machines with a factor among 11–23 are 132 (11: EPYC, Xeon)
and the lengths with 17 (1020 on the Mac and the Xeon, 150144 and 250240 on the EPYC). So 11
and 17 left the fast set in composites (commit 9947c53).

### Confirmation of the changed picks (commit 9947c53)

Runs: Mac 20261117-045149-51e4, EPYC 20261117-045502-77a5, Xeon 20261117-045439-910f.

| Pick | Mac | EPYC | Xeon |
|---|---|---|---|
| 131 → 140 | 1.00 | 1.11 | 1.76 |
| 1009 → 1024 | 1.00 | 1.00 | 1.00 |
| 10007 → 10051 (19 · 23²) | 1.41 | 1.66 | 1.10 |
| 15013 → 15120 | 0.77 | 0.94 | 0.89 |
| 30011 → 30240 | 0.78 | 0.83 | 0.91 |
| 150001 → 150176 (2⁵ · 13 · 19²) | 0.93 | 1.37 | 1.14 |
| 250007 → 250880 | 0.73 | 1.14 | 0.89 |

Every pick ran at 0.12–0.42 of the awkward length's time. 10051 missed on two machines, with
19 and 23 as its factors, so by the same rule both left the fast set in composites.

## Verdict

1. **Metric 2 holds everywhere.** Every next fast length ran at 0.11–0.96 of the time of the
   awkward length it replaces, primes included.
2. **Rule 2, applied twice.** Composites with a factor 11, 17, 19 or 23 are no longer fast,
   and the primes alone are; the golden list follows. A fast length is now a product of 2,
   3, 5, 7 and 13, or 11, 17, 19 or 23 alone.
3. **Rule 1 fails on x86, and the final set is not measured as a whole.** The misses that
   remain are short lengths below 300: 100, 112, 140 and 252 on the Xeon, and 252 on the
   EPYC. Powers of two that short are exceptionally fast on x86 (pair leaves, X28b; two-pass
   256 and 512, X30), so AVX2's double table, the reference, underrates them there. The
   choices for the maintainer:
   - a tighter tolerance for short lengths;
   - an x86-specific reference, which gives up machine-independent grids;
   - accept it, since `relative_cost` shows each machine's own view.
4. **13 stays.** Its composites measured fine (10010 and 30030 at 0.93–1.10, with 11 in
   them as well), but 13 has no composite pick of its own here.
