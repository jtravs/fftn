# X7: lanes against in-row kernels for batches of short rows

- **Status:** done (adopted)
- **Owner:** lanes session, 2026-11-15
- **Phase / task:** 2.2

## Question

For a batch of short rows, is the lanes strategy (DESIGN.md 8.5: one row per vector lane,
the row's own chain on a tile of B rows) faster than the chain run row by row, by length,
target, precision and batch? The answer sets the planner's threshold
`target_profile::lanes_up_to` per target and precision (`src/planner/rules.cpp`,
`lanes_up_to_of`) and the block rule (`lanes_block`, `src/core/plan_core.cpp`).

## Hypothesis

Two costs of the row-by-row chain fall on short rows: the leaf (n ≤ 16, 32 on NEON double)
broadcasts its values and stores lane 0, idling C − 1 lanes, and a grouped first pass with
m < C runs as tails; and every row pays the driver and its passes' set-up (FFTN ÷ NEON FFTW
3.2 at 4096 rows of 8 points on the Mac, the per-row overhead on C = 1, where no lane idles,
so the fixed cost of a row is about 6 ns against FFTW's 2.3 ns for the whole row). The
lanes remove both, at the price of a gather and a scatter (C × C transposes) where a run of
C rows is not one vector access (C > 1 and rows not adjacent: the usual case on x86 and in
float on the Mac). Expected: lanes win everywhere on C = 1 (no gather), and on C > 1 up to
the length where the chain's passes fill every lane and the gather and scatter, two extra
passes over the block, cost more than the overhead they save: about 16–64 on NEON float,
higher on AVX-512 (C = 4 and 8), where the chain's first passes run as tails up to n = C²
and FFTW and MKL vectorise across the batch too (X18).

## Change under test

The strategy only, forced per arm with fbench's `STRATEGY` (an internal build setting,
`build_settings::strategy`, so no switch in `dev/KNOBS.toml`): arm C `STRATEGY=chain`
(row by row), arm L `STRATEGY=lanes` (the rule's block: the largest power of two times C,
at most 64 rows, whose two scratch tiles fit half of L1). Both give every row the same bits
(tests/core/lanes_test.cpp), so the arms differ in time only. References: FFTW 3.3.11 (the
machine's SIMD build, MEASURE) and on the Xeon oneMKL 2026.1, double only (fbench times
competitors in double).

## Method

- Cases: `bench/catalogues/x7.toml`, tier `full`: c2c of 8, 12, 16, 24, 32, 48, 64, 100,
  128 and 256 points in batches of 64 and 1024 (and 4096 for 8, 16, 64, 128), r2c and c2r
  of 16 to 256 points in batches of 64 and 1024; out of place, double and float.
- Machines: Apple M5 Max (`neon_fcma`; Homebrew Clang 23); Xeon Gold 6240 (`avx512` and
  `avx2` forced; GCC 14; Slurm `-c 8`, shared); EPYC 7763 (`avx2`; GCC 16; DMOG `-c 8
  --hint=nomultithread`, shared). One thread, Release.
- Metric: the median over rounds of the paired ratio L ÷ C per case with its 95 % interval,
  and the A/A band of the run's control arm.
- Decision rule, written before the decision runs (the Xeon and EPYC runs on commit
  9a4b8ed and the Mac run on the branch head): per target and precision, `lanes_up_to` is
  the largest catalogue length L such that no case of length at most L (c2c, r2c and c2r,
  every batch) is slower with lanes (no interval wholly above 1 + band), and the geometric
  mean of L ÷ C over those cases is at most 0.97. If no length qualifies, 0 (never). The
  rule needs a batch of at least C rows. The block rule stays unless a forced-block arm
  (`LANE_BLOCK`) beats it by more than the band on the target cases.

Commands (`F="uv run --frozen python bench/fbench.py"`; remote runs through
`scripts/remote/fbench_remote.py submit <profile> <arms> -- <fbench options>`):

```bash
$F run C=fftn+STRATEGY=chain L=fftn+STRATEGY=lanes F=fftw \
  --catalogue bench/catalogues/x7.toml --tier full                       # Mac
# Xeon (profile with MKL; arms built from the commit)
submit xeonbase8 C=fftn:<c>+STRATEGY=chain L=fftn:<c>+STRATEGY=lanes F=fftw:<c> M=mkl:<c> \
  C2=fftn:<c>+ISA=avx2+STRATEGY=chain L2=fftn:<c>+ISA=avx2+STRATEGY=lanes -- \
  --catalogue bench/catalogues/x7.toml --tier full
submit epyc C=fftn:<c>+STRATEGY=chain L=fftn:<c>+STRATEGY=lanes F=fftw:<c> -- \
  --catalogue bench/catalogues/x7.toml --tier full
```

## Results

Exploratory runs on the Mac while building (3 rounds each, not decision runs): the first
version, with a gather and a scatter on every target, beat the chain on NEON double only up
to 24 points (0.65–0.89) and lost beyond (1.13–1.66 at 32–256); reading and writing the
rows in the first and last passes where a run of C rows is one access (C = 1), and running
those passes rows outermost, turned that into 0.26 (8 points) to 0.96 (256), and real rows
0.49–0.97 (c2r of 256: 1.05).

A first pass reading the rows through a stack tile per element on C = 2 (NEON float),
instead of the gather, was slower (0.56 against 0.39 ns per point at 8 points, 1.4–1.6 times
the chain from 64 on) and was dropped.

Decision runs (median of L ÷ C over rounds; FFTN ÷ competitor from the same runs' medians):

- **Xeon Gold 6240**, run 20261115-004715-ca5d (commit 9a4b8ed; 8 rounds, band ±3.0 %,
  load 7.5 on 72 CPUs). AVX-512, L ÷ C: double 0.29–0.45 at 8–16 points, 0.33 at 24,
  0.63–0.75 at 32, slower from 48 (b1024 1.14); float 0.13–0.26 at 8–24, 0.53–0.98 up to
  128 (none slower), 1.12–1.26 at 256; real rows 0.16–0.79 up to 128 in float, double
  r2c 128 b1024 1.38. AVX2 forced (L2 ÷ C2, medians only): double 0.43–0.69 at 8–16,
  0.99–1.00 at 24, 1.13–1.23 at 32; float 0.23–0.40 at 8–24, 0.72 at 32, 0.89 at 48, 1.04–1.13
  at 64. Against the competitors, AVX-512 double: c2c lanes ÷ FFTW 0.88–1.53 at 8–16
  points (1.3–1.5 at batch 64), 0.80 at 24, 1.2–1.3 at 32–256; lanes ÷ MKL 1.0–2.2 (MKL
  0.49 ns per point at 16 points against 1.07); r2c 0.70–0.92 of FFTW up to 64, c2r
  0.54–0.77 of FFTW and 0.54–1.12 of MKL up to 64.
- **EPYC 7763**, run 20261115-004702-22c7 (commit 9a4b8ed; 8 rounds, band ±1.05 %, no round
  loaded). AVX2 double 0.38–0.59 at 8–16, 0.78–0.93 at 24–32, 0.98–1.03 at 48–100 (none
  slower at 48), slower from 64 (b1024 1.07); float 0.21–0.29 at 8–24, 0.55–0.84 at 32–100,
  1.05 at 128 b4096; real rows 0.32–0.98 throughout. Lanes ÷ FFTW (double): c2c 1.7 at 8–16
  points, 1.1–1.6 above; r2c and c2r 0.73–1.04 (r2c 16 0.91, 64 0.91).
- **M5 Max**, run 20261115-010429-28d5 (branch commit ccf09d2; 8 rounds, all loaded: load
  14 on 18 CPUs, band ±4.7 %). `neon_fcma` double (C = 1, no gather): 0.26–0.46 at 8–16,
  0.55–0.76 at 24–48, 0.85–0.99 at 64–128, no case slower up to 256 (c2c 256 1.00–1.07,
  real 256 0.99–1.04, inside the band); lanes ÷ NEON FFTW 0.86–1.01 at 8–16, 0.75–0.82 at
  24 and 48, 0.78–0.98 at 100–256, 1.06–1.16 at 32 and 64; r2c 0.81–1.09, c2r 0.62–1.15.
  Float (C = 2, gather): 0.84–0.97 at 24–32, slower from 48 (b1024 1.15).

Thresholds by the rule (the length before the first slower case, the minimum over the
machines sharing a target): AVX-512 32 (double) and 128 (float); AVX2 24 (the EPYC alone
allowed 48) and 48 (the EPYC 100); NEON 256 where C = 1 and 32 in float. Every geometric
mean up to its threshold is below 0.97 (0.33–0.70). The block rule was not varied: no
forced-block arm was run (a follow-up if the x86 gather is replaced).

## Verdict

Adopted: the lanes are the rules' choice for a batch of at least C rows up to
`lanes_up_to` per target and C (`src/planner/rules.cpp`, commit after 5dba687 for NEON
float's 32), DESIGN.md 8.5 as built, lesson 4.7. Task 2.2's acceptance (awkward short rows
on AVX-512 at or below FFTW and MKL) is met for real rows but not for c2c in double: the
lanes cut FFTN's time to a third at 8–16 points, but FFTW stays 1.3–1.5 times and MKL up to
2.2 times faster there. What remains on x86 is the gather and scatter (two passes over the
block, which FFTW avoids by building its vectors from 128-bit inserts in the codelet's
loads) and the passes' own speed (X18); both are follow-ups, not part of this verdict.

