# X7: lanes against in-row kernels for batches of short rows

- **Status:** running
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

(Decision runs: pending.)

## Verdict

(Pending.)
