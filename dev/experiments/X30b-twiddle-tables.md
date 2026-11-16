# X30b: twiddle tables on x86 at 1024–4096 points: alignment, storage per pass, three passes

- **Status:** running
- **Owner:** x30b session, 2026-11-16
- **Phase / task:** 1 (exit criteria); PLAN lead 3; follows X30

## Question

Do 64-byte aligned twiddle tables, and the half-size (wr, wi) twiddle storage chosen only
for the passes whose tables are large, close part of the EPYC's gap to FFTW at 1024–4096
points in double without the short-row losses that rejected X30's arm T? The answer
changes how every target allocates its tables, the `avx2` and `avx512` vocabularies'
twiddle storage, and possibly the `avx2` double plan of 4096.

## Hypothesis

X30 left the x86 chains at 1024–4096 refilling L1 1.4–3.8 times as often as FFTW, with
twiddle tables twice FFTW's bytes and IPC 2–2.5 on the EPYC (so not at the dispatch limit).
Its evidence for each arm here:

1. **Aligned tables.** Tables start where `std::vector` puts them (16 bytes into a line on
   glibc), so half of a per-lane pass's 32-byte twiddle loads cross a cache line (main
   crossed lines on 863–3046 loads per transform at 1024–4096). Aligning the tables alone
   took 0.97–0.98 of main's cycles at 1024–4096 (counter diagnostics, single processes,
   branch `diag/x30-ma`). It changes no instruction and no bit, so it should cost nothing
   anywhere (the Mac's 128-byte lines included).
2. **(wr, wi) storage per pass.** Arm T (all tables (wr, wi), read with duplicating loads)
   won where the tables overflow L1 (EPYC double 384–4096 0.95–0.97, float 0.80–0.94) and
   lost where they are small (EPYC 128 1.040, 192 1.035; Xeon `avx512` 16–17 1.06–1.10):
   the duplicating loads cannot be folded into the multiplies, and the imaginary parts'
   load at p + 1 crossed a line in half the units. Choosing the storage per pass by the
   size of its table keeps the pre-arranged (duplicated) form for short rows and small
   passes and halves the large tables; building the imaginary parts with an in-lane
   shuffle of the aligned load (no crossing; diagnostic `diag/x30-tb`, 0.95–0.97 of T's
   cycles) should keep T's wins. Not everything is expected: T lost 3072 (1.052, a radix-12
   grouped pass over a 88 KiB table), so the size of the table is not the whole story.
3. **Three passes at 4096 (`avx2` double).** FFTW runs 4096 in three passes, FFTN's `avx2`
   double in four (8 × 8 × 8 × 8, X18's override; X18 found the radix-16 passes losing on
   AVX2 at 4096, 18 live values in 16 registers). `avx512` double already plans 16 × 16 × 16.
   A three-pass sequence with a radix-64 or radix-16 pass may win now that the passes have
   changed (X19, X28, X30).

## Change under test

Arms are commits of branch `exp/x30b-twiddle-tables`:

- **A** = `origin/main` (e1879cb, with X30's P).
- **L** (arm 1): every plan table (`plan_core::tables_f64`, `tables_f32`, so the chain's,
  the grid's and Bluestein's) allocated on a 64-byte boundary (`platform/aligned_allocator.hpp`).
  All targets; bits unchanged.
- **C** (arm 2, on L): the `avx2` and `avx512` vocabularies hold a twiddle register pair as
  (wr, wr) and (wi, wi) and multiply with FMADDSUB/FMSUBADD (X30's arithmetic, which rounds
  as the pre-arranged form did, so bits are unchanged); tables may be stored either
  pre-arranged ((wr, wr), (wi, wi) per lane; 4 reals per twiddle) or compact ((wr, wi), 2
  reals): grouped first passes and twiddled lane passes (no edge operation) get a compact
  instantiation, and the chain builder takes it for a pass whose pre-arranged table would
  exceed `machine_profile::twiddle_compact_bytes`, by default half of the L1 data cache
  (16 KiB on both x86 machines). Compact per-lane loads: double r by `vmovddup`, i by
  `vpermilpd` of the same aligned load (compilers emit one load and two shuffles; the
  diagnostic `diag/x30-tb` measured this form); float r and i by `vmovsldup` from p and
  p + 1 (load ports only, X30's T form; the one-load form became a load and two shuffles,
  so it was not used). Compact broadcasts: two scalar broadcasts. Amended before any
  timing: written as above when the arms were built. Targets other than `avx2` and
  `avx512` have no compact form and are unchanged. The threshold is a machine-profile field,
  so fbench's `TWIDDLE_COMPACT_BYTES` sets it per arm (`never`: every table pre-arranged;
  `1`: every table compact, X30's T with aligned tables and, in double, the shuffle).
- **R** (arm 3, screened before any commit): `avx2` double 4096 forced (`RADICES`) to
  16 × 16 × 16, 4 × 16 × 64, 8 × 8 × 64, 16 × 4 × 64 and 64 × 64 against the planner's
  8 × 8 × 8 × 8; a winner becomes the override in `src/planner/rules.cpp`.

No library switch: arms are commits, the threshold is a machine-profile field (private
build settings, like the grid's sizes), `dev/KNOBS.toml` is untouched.

## Method

- **Cases**: `bench/catalogues/x30.toml` (c2c rows of 128–4096 in double and float, batches
  of 64, r2c and c2r of 512–4096, planes 256² and 512²); collateral: the standard
  catalogue's tracking tier.
- **Runs per machine** (one thread, fbench interleaved process rounds, A/A control,
  `--serial`): x30 catalogue and tracking tier, arms A, L, C, C8 (`C+TWIDDLE_COMPACT_BYTES=8192`),
  C1 (`=1`), CN (`=never`). C, C8 and C1 are judged against L: the runs are re-read with L
  as the baseline (its labels swapped with A's in a copy of the run record, then
  `fbench show`), so their intervals are paired by round as fbench's are; the A/A band is
  still A's.
- **Machines**: EPYC 7763 (DMOG, GCC 16.2, `avx2`), the main machine; Xeon Gold 6240
  (GCC 14.4, `avx512`, and `ISA=avx2` in the same job); Mac M5 Max (Homebrew Clang,
  `neon_fcma`) for L only, x30 and tracking with A and L. Clang on x86 not measured
  (not a release toolchain).
- **Counters**: `run --pmu` on the EPYC for A, L, C and C1 at 1024, 2048, 4096 and 3072
  (double, batch 1): cycles, instructions, L1 data-cache fills (as X25 counted them) and
  misaligned loads (PMCx047, as X30), to confirm that fills and line crossings fall where
  the time does.
- **Metric**: fbench's median paired ratio per case; geometric means of x30's double and
  float cases; the run's A/A band.
- **Decision rule**, written before any arm is timed. Judged on GCC on x86 and Clang on the
  Mac, per machine-target:
  - **L** is adopted on every target if (a) x30's double geomean, L ÷ A, is at most 1.00
    on each x86 target and at most 1 + band on the Mac, (b) no case of x30 or of the
    tracking tier is slower than 1 + band + 2 % on any machine-target, and (c) x30's float
    geomean is at most 1 + band everywhere. L is deleted otherwise.
  - **C**, judged against L, per target (`avx2`: EPYC and the Xeon's `avx2`; `avx512`: the
    Xeon), for each threshold measured (C, C8, C1): (a) x30's double geomean at most 0.99
    with its interval clear of the band on the deciding machine (EPYC for `avx2`, the
    Xeon for `avx512`), and at most 1 + band on the Xeon's `avx2`; (b) no case of x30 or of
    the tracking tier slower than 1 + band + 2 %; (c) x30's float geomean at most 1 + band.
    Of the thresholds that pass, the one with the best double geomean on the deciding
    machine becomes the default (as a fraction of L1); if none passes, C is deleted. A
    case past (b) on a loaded machine is re-run once and the re-run decides. The choice
    among thresholds uses the same data it is judged on; the tracking tier at the chosen
    threshold is the check against that optimism.
  - **R**: a sequence becomes `avx2` double 4096's override if it is at most 0.97 of the
    planner's at batch 1 and 64 with intervals clear of the band, and then passes (b) on
    the tracking tier (EPYC and the Xeon's `avx2`).
  - Nothing is iterated past these three arms.

## Results

(pending)

## Verdict

(pending)
