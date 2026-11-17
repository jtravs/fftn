# P4: are `next_fast_length`'s lengths fast?

- **Status:** running
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

Pending.

## Verdict

Pending.
