# X36b: what X36 left of the prime tail (EPYC 1018, the Xeon's long Bluestein rows)

- **Status:** done: arm 1 adopted with a shape rule; arm 2 needs a program-model change (not built)
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
  can express it. It cannot without a change to the program model: the grid
  (`grid_builder::c2c_row`) runs a row between the movable slots `in` and `out` (and the
  workspace `block` for its intermediate), stepping them through the program's loops,
  with scratch lines a and b holding one block of columns; a Bluestein convolution needs
  an array of M > n points for its input and another for the grid's output or
  intermediate, and the only movable array of that size is `block` (`in` must be
  preserved, `out` holds n). So a grid convolution needs a second movable workspace slot
  (or per-stage element offsets), through the driver, the loops' step arrays (moving_slots),
  the work graph and describe(). Measured instead, as the size of the prize and a cheaper
  stopgap: the Xeon's chains of 2¹⁸–2²⁰ points with the rules' radices, X32's radix-8 and
  radix-4 sequences, against its grid (run x36b-len-xeon).

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

One thread, hot, A/A control; A = `main` cddd342, B = arm 1 (cc5d2ea, s ≤ C everywhere),
F = FFTW, M = MKL; cases the filtered x36 tracking tier.

| machine, run, band | changed cases B ÷ A | FFTN ÷ FFTW |
|---|---|---|
| EPYC (`--serial`), 20261117-002835-5d41, ±1.0 %, all rounds loaded | double 1018 0.733, 8186 0.935, 262142 0.730; float 2036 0.801 | 1018 1.53 → 1.12 (11.96 against 10.70 ns per point) |
| Xeon (`--serial`), 20261117-001452-0084, ±1.83 % | double 2036 0.813; float 4072 0.952 | 1018 unchanged (s = 2 < C, X36's rule) |
| Mac, 20261116-232501-14c7, ±1.0 % | float 1018 0.966, 2038 0.993, 8186 0.987, 262142 **1.227** | – |

No changed case was slower except NEON float 262142 = 2 × 131071 (one stage of 552960,
whose power-of-two part is 4096, against two copies of 276480). Rule 2: s = C takes the
whole line only while the line is not long (`long_convolution_bytes`: 128 KiB on the Mac,
256 KiB on the EPYC, 4 MiB on the Xeon), ce007be; that keeps every measured win but
EPYC 262142 (0.730), which becomes a follow-up. No case of the standard tracking tier has a
large prime with more than one copy, so the rule changes no plan there (checked from the
catalogue's lengths).

Arm 2, the size of the prize: the Xeon's c2c rows as its grid (G) against forced chains
of the rules' radices (R), one round only (run 20261117-002122-58ba: its radix-4 arm had a
bad option and stopped the run after round 0; the repeat x36b-len2-xeon, Slurm 16199, was
queued behind four jobs), ns per point: 262144 7.07 against 7.41, 524288 10.05 against
9.78, 1048576 11.63 against 13.80, 2073600 11.83 against 16.26; float 262144 3.57 against
4.78, 1048576 4.94 against 6.41. With X36's split convolutions the chain is now within 5 %
of the grid at 2¹⁸ in double, so a grid convolution would take 131071 (M = 2¹⁸) to about
0.97 of its time, not under FFTW's (1.28): most of its gap is the long transform itself
(c2c 262144 is 1.22 of FFTW's time on the Xeon, cause 5 of the exit check). It would take
999983 (M = 2073600) to about 0.78, under FFTW's (1.10 → about 0.86), and float long rows
to 0.75–0.77.

## Verdict

- **Arm 1 adopted** with the shape rule (s = C only for lines that are not long): EPYC
  1018 0.73 (1.53 → 1.12 of FFTW: better, still over 1.05), 8186 0.94, float 2036 0.80;
  Xeon 2036 0.81, float 4072 0.95; Mac float 1018 0.97. The remaining EPYC 1018 gap is the
  whole-line stage's M = 2048 against FFTW's two loops of 1024; matching it needs the copies
  run one at a time, a program loop over a scratch-resident line, which the program cannot
  express (DESIGN 8.5).
- **Arm 2 not built:** a grid convolution needs a second movable workspace slot (or stage
  offsets) in the program model; the screen says it would fix the Xeon's 999983 and float
  long rows but not 131071, whose gap is the Xeon's long transforms (exit cause 5).
- Follow-ups: the second movable workspace slot (it would serve both the grid
  convolution and a per-copy Bluestein loop); EPYC 262142 (0.73 as one stage, excluded by
  the shape rule because of the Mac's float loss).

Done: DESIGN 8.5 and 11.2 (s ≤ C), the decision log, PLAN index and Status.
