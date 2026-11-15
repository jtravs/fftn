# X28: single short rows

- **Status:** running
- **Owner:** x28 session, 2026-11-15
- **Phase / task:** 1 (exit criteria); PLAN lead 4; follows X20 and X25

## Question

What is left of the cost of one short row (≤ 120 points, batch 1) after X20, and which of
three changes closes it: a lighter entry, 128-bit leaves on x86, or a flat path for chains
of two passes? Each one that wins becomes the default on the targets where it wins.

## Hypothesis

Baseline (tracking tier, FFTN ÷ FFTW): single rows ≤ 64 points 1.14 on the Mac, 1.38 on
the EPYC, 1.51 on the Xeon; 8 points 1.80–1.83 on the Xeon; 60 points 1.60 on the Xeon and
1.42 on the EPYC; 120 points 1.62 on the Xeon (runs 20261115-094953-38c5,
20261115-101319-f98f, 20261115-114105-ea43). X25 split the gap:

- at 8 points 23 of the 25 cycles of the Xeon's gap (14 of 15 on the EPYC) are fixed cost
  per call. X20's harness puts only about 2 ns between the API and the leaf on the Xeon
  (8 points: `api` 13.0 ns, the leaf alone through its pointer 10.9 ns), so most of that
  fixed cost is in the leaf's own call: its set-up of 16 leg offsets for any stride, six
  saved registers and `vzeroupper`, and on the Xeon a 512-bit leaf that runs at frequency
  licence 1 where FFTW's 128-bit `n2fv_8_sse2` runs at licence 0 (1.62 times the cycles,
  1.84 times the time);
- at 60 points on the EPYC 37 of 79 gap cycles are the two-pass chain's driver
  (`run_flat_one`, `moved_in_nest`): each stage copies its 144-byte `pass_args`, reaches its
  stage through a `std::variant` and its slots through switches;
- at 60 points on the Xeon the partial-unit radix-4 grouped pass takes 57 % of the time
  (left to X23b, which changes those partial units).

Expected: (a) a few per cent at 2–16 points; (b) on the Xeon most of the 8–16-point gap
(licence 0, no broadcasts of 512 bits), little on the EPYC (Zen 3 runs 256-bit operations at
full rate); (c) 5–15 % at 24–120 points on all machines.

## Change under test

Arm A: `main` (3f90275). Then one commit per arm, each on top of the one before:

- (a) the lighter entry: what a measurement of each piece (X20's harness with the new
  `public` and `copy` levels, `--pmu` counts) shows is worth removing between
  `plan::execute` and the kernel of a one-row plan.
- (b) 128-bit leaves on x86: the leaves of n ≤ 16 compiled with a 128-bit vocabulary (one
  complex double per register, as on NEON) inside the `avx2` region, and used by `avx2` and
  `avx512` plans whose program is rows of leaves. The vocabulary repeats `avx2`'s operations
  at half width, so every lane rounds as before: bits unchanged.
- (c) a flat path for chains of two passes: the stages of a flat program prepared so that
  one transform runs its passes without the variant walk and with the least copying the
  calling convention allows.

No switch in `dev/KNOBS.toml`: each arm is a commit.

## Method

- Cases: catalogue `x28` (`bench/catalogues/x28.toml`): c2c rows 8, 12, 16, 24, 32, 48,
  60, 64, 120 at batch 1 in double and float (targets), the same at batches 64 and 4096 in
  double, r2c and c2r 16, 32, 64 at batch 1. Collateral: the standard catalogue's tracking
  tier, final arm against A.
- Machines: Xeon Gold 6240 (GCC 14, `avx512` and forced `avx2`, FFTW 3.3.11 with AVX-512
  codelets, MKL); EPYC 7763 (DMOG, GCC 16, `avx2`); Apple M5 Max (Clang, `neon_fcma`, NEON
  FFTW), as the brief allows for this lead. fbench interleaved rounds with the A/A control;
  Slurm `-c 8`, shared, `--serial`.
- Diagnosis: X20's harness (`bench/x20/x20_overhead`) per arm, ns per transform at the
  levels `public`, `api`, `driver`, `copy`, `kernel`, `once`.
- Metric: fbench paired ratio arm ÷ A per case; weighted geometric mean over the batch-1
  c2c cases (targets), and FFTN ÷ FFTW.
- Decision rule, written before running, per arm and per machine-target, on the release
  toolchains (GCC on x86-64 Linux, Clang on macOS):
  1. every test passes; arms (a) and (c) keep results bit-identical (same kernels on the
     same arguments), arm (b) too (same operations per lane), held by
     `Contracts.BatchedRowsMatchSingleRows` and the forced-target suites;
  2. adopt where the geometric mean over the target cases is below 0.97 against the arm
     before it, with its interval clear of the A/A band;
  3. and no case of `x28` (batches included) or of the tracking tier is slower than the
     arm before it by more than the band; a case just past the band on a loaded machine is
     re-run before it decides. A win on one target and a loss on another is adopted per
     target.

## Results

(pending)

## Verdict

(pending)
