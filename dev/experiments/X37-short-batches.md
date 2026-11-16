# X37: short rows and small batches in the phase 1 tail

- **Status:** running
- **Owner:** X37 agent, 2026-11-16
- **Phase / task:** 1 (exit criteria: the 5 % tail rule of DESIGN 1.3); causes (2) and (3)
  of the exit check in PLAN Status; follows X28b and X24

## Question

Which of the phase 1 exit check's failing cases among single short rows, tiny batches and
batches of 16 points are fixed costs per call and which are kernels, and do pair leaves for
the lengths that have none (18, 22, 26, 30 on x86, 18–30 on NEON) and for the batched
leaves of 8–16 points on x86 bring them within 1.05 of FFTW (and MKL on the Xeon)?

## Hypothesis

The exit check's runs (FFTN ÷ FFTW, double; 20261116-161123-f848 Xeon `avx512` with MKL,
20261116-161226-416e EPYC `avx2`, 20261116-160815-6deb Mac `neon_fcma`):

- **Composites without a pair leaf.** Xeon 18, 22, 26, 30 points 1.22–1.33, at 1.99–2.63
  ns per point against 0.65–0.90 for the lengths that have a pair leaf (20, 24, 32);
  C = 4 divides none of them, so `avx512` has no pair leaf and runs a chain of two passes.
  `avx2`'s table has 20, 24, 30, 32 but not 18 = 9 × 2, 22 = 11 × 2, 26 = 13 × 2 (EPYC 18
  1.08, 26 1.12). On NEON (C = 1 in double) there are no pair leaves at all: 20 points is a
  chain 4 × 5 at 14.5 ns against FFTW's 8.1 (1.78; FFTW has a codelet of 20), 22 and 26
  1.05–1.06. A pair leaf holds n / C registers: 9–15 of `avx2`'s 16 at C = 2, 18–30 of
  NEON's 32 at C = 1. Expected: these rows 0.5–0.8 of their time, as X28b's pair leaves
  gave 20–48 points (0.42–0.76), and below FFTW.
- **Batches of 16 points on x86.** A batch of 8–16-point rows runs the 128-bit leaf once
  per row (X28): one complex double per register, so `avx2` uses half and `avx512` a
  quarter of its width. FFTW at 16 × 256 does 0.65–0.73 ns per point and MKL 0.44 against
  FFTN's 0.76–0.77 (EPYC 1.18, Xeon ÷ MKL 1.72); FFTN's own pair leaves of 32 points run
  0.50–0.54 ns per point there. A pair leaf 4 × 4 on `avx512` (one unit of four
  butterflies, a 4 × 4 transpose, one block) or 8 × 2 on `avx2` uses whole registers.
  Expected: 16 × 16–4096 0.6–0.8 of the time on both machines; 8 and 12 points less (fewer
  operations per transpose).
- **Tiny rows (2–6 points) and 2 × 3.** Fixed cost per call: X20's harness on the Mac puts
  the 2-point row at 2.0 ns through `plan::execute` against fbench's 2.5, and FFTW's at 2.1.
  fbench's FFTN backend calls `transform_count` and then `execute_core` (two calls into the
  library) where `plan::execute` makes one (`execute_all`, X28). Expected: correcting the
  backend moves 2–6 points by 0.2–0.5 ns per call, on every arm alike.
- **Backward batches of 2 (EPYC 15 × 2 1.27, 64 × 2 1.31).** FFTW is faster per row at
  batch 2 than at batch 1 (15 points 0.88 against 1.46 ns per point), consistent with two
  rows in the lanes of one register; FFTN runs them as two single rows (single rows 0.78
  and 0.96 of FFTW). Pair leaves do not reach them (C does not divide 15; 64 is too long
  for AVX2's registers). Recorded, not targeted by an arm here.

## Change under test

Arm A: `origin/main` e8c2639. One commit per arm, each on top of the one before:

0. **Benchmark correction** (not a library change): fbench's FFTN backend executes a serial
   whole batch through `execute_all`, the function `plan::execute` calls, instead of
   `transform_count` followed by `execute_core`. Timed against A to show its size; the arms
   below are judged against it.
1. **Short pair leaves**: `avx2`'s pair table gains 18 = 9 × 2, 22 = 11 × 2 and 26 = 13 × 2
   (double); plans of the `avx512` target take `avx2`'s pair leaves for lengths their own
   table lacks (18, 22, 26, 30: each pair set carries its lanes and its twiddle packing, so
   the leaf gets a table in its own storage); NEON gets pair leaves (C = 1 in double) for
   18, 20, 22, 24, 26, 28, 30 with the factors of the chain the rules choose, so their
   results keep the bits they have.
2. **Pair leaves for short batches on x86**: 8 = 2 × 4, 12 = 3 × 4, 16 = 4 × 4 on `avx512`
   and 8 = 4 × 2, 12 = 6 × 2, 16 = 8 × 2 on `avx2` (double), taken before the single-codelet
   leaf for rows of stride 1 in a batch (and for single rows where the run shows they win).

Bits: arm 0 changes no library code; arms 1 and 2 give the bits of the chain r1 × m of
the leaf's own target (the test `Passes.PairLeavesMatchTheForcedChain`), so a plan's
results change only where its chain had other factors or the leaf comes from `avx2` in an
`avx512` plan, within the derived tolerances. No switch in `dev/KNOBS.toml`
(`build_settings::pair_leaves` stays the tests' forcing parameter).

## Method

- Cases: catalogue `x37` (`bench/catalogues/x37.toml`, double): single c2c rows of 2–30
  points; batches of 3 of 2–16 points out of and in place; the backward batches of 2 of 8,
  15 and 64 points; batches of 16, 256 and 4096 of 8–64 points; batches of 64 and 1024 of
  12 and of 18–30 points (collateral for the pair leaves). The failing exit-check cases
  are a subset; they are also judged with the exit check's catalogue and filter
  (`standard`, `^c2c-[0-9]+-b`, tier `full`) on the final arm. Collateral: the standard
  catalogue's tracking tier, final arm against A.
- Machines: Xeon Gold 6240 (GCC 14, `avx512`, FFTW 3.3.11 with `PLAN_NOTE=1`, oneMKL), EPYC
  7763 (DMOG, GCC 16, `avx2`, FFTW with `PLAN_NOTE=1`), Apple M5 Max (Clang 23,
  `neon_fcma`, NEON FFTW). fbench interleaved rounds with the A/A control; Slurm `-c 8`
  shared, `--serial`; on the Mac one run at a time. Where the time goes: fbench `--pmu`
  (cycles, instructions) on the x86 machines for the target cases, FFTW's printed plans.
- Metric: fbench paired ratio of each arm against the arm before it; geometric mean over
  each arm's target cases (the cases whose plan or kernel the arm changes); FFTN ÷ FFTW
  and ÷ MKL per case.
- Library size: text of `libfftn.a` (`size`) per arm on x86-64 (GCC, the remote build
  cache) and arm64.
- Decision rule, written before running, per arm and per machine-target on the release
  toolchains (GCC on x86-64 Linux, Clang on macOS), double:
  1. every test passes; arm 0 changes no library code; arms 1 and 2 give the bits of the
     chain r1 × m of the leaf's target (a test holds it);
  2. an arm is adopted on a machine-target where the geometric mean over its target cases
     there is at most 0.97 of the arm before it, with its interval clear of the A/A band;
  3. and no x37 case (or, for the final arm, tracking-tier case) is slower than the arm
     before it beyond the band. A pair-leaf length that is slower beyond the band at some
     batch leaves that target's table rather than rejecting the arm; where it loses only
     on single rows (or only in batches) it is limited to the other. A case just past the
     band on a loaded machine is re-run before it decides; a win on one target and a loss
     on another is adopted per target;
  4. arm 0 is kept whatever it measures if rule 1 holds: it makes fbench time what a
     caller of `plan::execute` runs. Its effect is reported next to the arms'.

## Results

(pending)

## Verdict

(pending)
