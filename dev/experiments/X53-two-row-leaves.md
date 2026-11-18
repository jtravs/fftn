# X53: batch leaves (C rows per register) for tiny batches

- **Status:** running
- **Owner:** X53 agent, 2026-11-18
- **Phase / task:** 1 (exit criteria: the tail rule of DESIGN 1.3); exit-check cause 2

## Question

Do leaves that run C short rows of a batch side by side, row u + l in lane l of every
register, close the short backward pairs that still take 1.15–1.3 of FFTW's time on x86
(EPYC 15 × 2 and 64 × 2, Xeon 15 × 2; X49), and for which lengths and batch counts should
the planner take them?

## Hypothesis

X49's counters (EPYC 7763, Slurm 512563): 15 × 2 backward takes 94 cycles and 376
instructions per call against FFTW's 87 and 321, 64 × 2 443 and 1946 against 354 and 1374.
FFTN runs 15 × 2 as the 128-bit leaf of 15 twice (one complex double per `xmm`) and 64 × 2
as the chain 4 × 16 twice; FFTW runs both rows in the two halves of one `ymm` with one
codelet call (`n1bv_15_avx2`, `n1bv_64_avx2` in its printed plan). FFTN's lanes strategy,
which does fill the lanes with rows, gathers them into a tile in memory first and is 1.6–10
times slower on these cases.

A batch leaf loads C consecutive elements of each of C rows as whole vectors, transposes the
C × C block in registers (on `avx2` in double two `vperm2f128` per two elements), runs the
codelet once for C rows, and transposes back before whole-vector stores; where C does not
divide n the last block overlaps the one before it, so no access is partial. Per row that is
half the codelet's arithmetic of the 128-bit leaf (one `ymm` operation for two rows where the
leaf issues one `xmm` operation per row) for one shuffle per element on each side. Expected:
15 × 2 and 64 × 2 at or below FFTW on the EPYC and the Xeon; every short codelet length gains
in batches of two or more where the plan is a 128-bit leaf today, and possibly where it is a
pair leaf or a short chain (8, 12, 16, 32, 64), from the screen. Each lane does what the leaf
does to its row, so a batch leaf gives the leaf's bits (the narrow leaf's on x86, which rounds
alike), whatever group a row falls in.

## Change under test

Arm A: `main` 4d75c48. Arm B: the batch leaves (`src/passes/batch_leaf.inl`; tables in
`src/isa/batch_*.cpp` for `avx2`, `avx512`, `neon`, `neon_fcma`; the leaf of every codelet
radix, both directions, the scale on either edge) for the rows of a c2c plan of one axis and
one batch dimension of at least two contiguous rows, chosen by a rule from the screen below
(`build_settings::batch_leaves`, plan_core.cpp `batch_leaves_of`). fbench's FFTN backend
forces them with `BATCH_LEAVES=never|own|narrower` (narrower: `avx2`'s two rows per `ymm` in
an `avx512` plan); no library switch, so nothing in `dev/KNOBS.toml`.

## Method

- **Screen** (forced arms of the infrastructure commit, one binary: `BATCH_LEAVES=never`
  against `own` and, on the Xeon, `narrower`): catalogue `x53` (every codelet length 2–64 in
  batches of 2, 3, 4 both directions, 8, 64, 1024 forward, and in-place batches of 2 and 3),
  double, on the EPYC 7763 (DMOG, GCC 16, `avx2`) and the Xeon Gold 6240 (GCC 14,
  `avx512`); `x53` in float (`f32`) on the M5 Max (Clang 23, `neon_fcma`, C = 2), the only
  precision NEON has batch leaves in. FFTW in the run for the ratios. The screen decides the
  rule; it is not the verdict.
- **Rule** (written into `batch_leaves_of` after the screen, before the judging run): per
  target and precision, the lengths and the smallest batch count from which `own` (or
  `narrower`) is faster than `never` beyond the screen's band on every case of that length
  from that count up; elsewhere none.
- **Judging run**: arm A against arm B (the rule's commit) with fbench `--serial`, catalogues
  `x53` and `x37` and the standard tracking tier, EPYC and Xeon (FFTW; MKL on the Xeon); the
  Mac in float (`x53` f32) if its rule takes any case.
- Metric: fbench paired ratio B ÷ A per case; the geometric mean over the cases whose plan
  the rule changes; FFTN ÷ FFTW (÷ MKL on the Xeon) per case.
- Decision rule, written before running, per machine-target, on the release toolchains (GCC
  on x86-64 Linux, Clang on macOS):
  1. every test passes; a batch leaf gives the leaf's bits row by row (tests/core
     `BatchLeaves.EqualTheLeafRowByRow` on every target that runs; `avx512` and `avx2` on the
     Xeon and the EPYC, NEON on the Mac);
  2. B is adopted on a target if the geometric mean over the cases its rule changes is at most
     0.97 of A with the interval clear of the A/A band, 15 × 2 backward is at most 0.90 of A
     (and 64 × 2 backward on the EPYC), and no case of `x53`, `x37` or the tracking tier is
     slower than A beyond the band (a case just past it on a loaded machine is re-run
     before it decides);
  3. a target whose rule takes no case, or that misses rule 2, keeps no batch leaves: its
     tables are deleted.

## Results

Not yet run.

## Verdict

Pending.
