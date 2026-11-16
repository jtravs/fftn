# X36b: what X36 left of the prime tail (EPYC 1018, the Xeon's long Bluestein rows)

- **Status:** running
- **Owner:** X36 agent, 2026-11-16
- **Phase / task:** PLAN Status, phase 1 exit check, cause (1), after
  [X36](X36-large-primes.md)

## Question

Can FFTN take the two exit-check primes X36 left failing under FFTW's time: 1018 = 2 · 509
on the EPYC (1.51) and 131071, 999983 on the Xeon (1.28, 1.10)? The answer changes the
few-copy rule of the Bluestein stage (`chain_builder::radices`) and, for the Xeon, how a
long convolution is computed.

## Hypothesis

- **EPYC 1018.** A Bluestein stage with s = C = 2 copies (AVX2 double) runs its convolution
  of 2 × 1024 interleaved points at 16.3 ns per point against 10.0 for one copy of 509
  (X36's run 20261116-203815-9d19): with s = C every unit is whole, but the chains' passes
  over two copies are lane passes with s = 2 (no grouped first pass, a twiddle load per
  unit of two). FFTW loops one copy at a time. Running the copies one at a time needs a
  program loop over a scratch-resident line, which the program cannot express (DESIGN 8.5);
  one Bluestein stage over the whole line of 1018 (X36's arm 3, extended from s < C to
  s ≤ C) runs one copy of M = 2048 at about 1019's 12.0 ns per point: about 0.73 of the
  time, 1.1 of FFTW's.
- **Xeon long rows.** The Xeon runs long rows as grids (X32: chains lose there); a
  Bluestein convolution is always a chain: 2¹⁸ as a chain took 8.98 ns per point against
  the grid's 6.86 for c2c 262144 (runs 20261116-183101-fc67, 20261116-161123-f848), so a
  grid convolution would take 131071 to about 0.8 of its time (about FFTW's).

## Change under test

- **Arm 1** (EPYC 1018): the few-copy rule covers 1 < s ≤ C (was s < C), still only where
  the whole line's convolution is no longer than the copies' together. Changes AVX2 double
  and NEON float (C = 2, s = 2), AVX2 float and AVX-512 double (s = 4), AVX-512 float
  (s = 8). A = `main`, B = arm 1.
- **Arm 2** (Xeon): a long convolution as a grid. Before measuring: whether the program
  can express it (below).

Bits: arm 1 changes the result bits of the lengths whose plans it changes (within the
derived tolerances).

## Method

- Cases: `bench/catalogues/x36.toml` tier `tracking` (X36's cases) and its `x36b` group
  (lengths with 2–8 copies of a large prime: 1018, 2036, 2038, 4072, 8186, 262142, 1527).
  Collateral: the standard catalogue's tracking tier.
- Machines: EPYC 7763 (`avx2`, GCC 16), Xeon Gold 6240 (`avx512`, GCC 14, with MKL), both
  in fbench_remote's `--serial` queue; the Mac (`neon_fcma`, Clang 23) for NEON float.
- Metric and decision rule (written before running), per target and precision: arm 1 is
  adopted if the geometric mean of B ÷ A over the cases whose plan it changes is at most
  0.95 with the A/A band below 3 % and none of them is slower beyond the band; a slower
  case keeps the old route by a rule for its shape measured in the same run, or the arm is
  rejected on that target; no unchanged tracking case slower by more than 3 % beyond the
  band.

## Results

(running)

## Verdict

(running)
