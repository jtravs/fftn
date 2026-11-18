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
  4. Added after the screens, before any judging run (the rule above names 64 × 2, which no
     arm closed): **arm R** is the rule's commit, judged by fresh `--serial` runs under rule
     2 without the 64 × 2 clause; 64 × 2 is reported, not judged. On NEON (float only) the
     15 × 2 clause does not apply.

## Results

Toolchains: GCC 16.2 on the EPYC 7763 (DMOG, `-c 8` shared, every round loaded by other
jobs on the node), GCC 14.4 on the Xeon Gold 6240 (`-c 8` shared), Clang 23.1.4 on the M5 Max
(`neon_fcma`, float); Release, one thread. Screens are `fbench_remote.py submit` without
`--serial` (DMOG and the Xeon) and local runs on the Mac; ratios are the forced arm ÷
`BATCH_LEAVES=never` of the same commit.

### Arms

- **B** (18a3d1b): C × C block transposes of all R inputs before the codelet and of the
  outputs after it; a last block that overlaps where C does not divide n; a partial group
  for the rows left over.
- **B2** (f06050c): each input gathered when the codelet asks for it (C broadcasts,
  transposed) and each output stored lane by lane. Clang emits FFTW's form on avx2 (a 128-bit
  load and a `vinsertf128` from memory per element); GCC 16 broadcasts and shuffles.
  **Rejected**: slower than B on every machine (EPYC 1.05-1.4 of B's time where B wins, run
  20261118-024008-3317; Xeon with avx2's leaves 1.05-1.25, run 20261118-024044-e8cf; M5 Max
  float 1.1-1.2, run 20261118-023555-b948), except the EPYC's 64 (0.91 of the chain against
  B's 1.00).
- **B3** (ec43a2d, on B2): a single row left over runs in the leaf's form (broadcast, lane
  0 stored) instead of a partial group, which cost as much as a whole group (B's batches of
  three: 15 × 3 1.00 of the leaf where 15 × 2 and 15 × 4 were 0.74).
- **B4** (9f03409): B's blocks with B3's last row. **Kept**: on the Xeon (quiet run, band
  ±2.76 %, 20261118-025722-2bc0) batches of three moved from 1.0-1.3 to 0.87-1.05 of the
  leaf at 9-25 points; elsewhere as B.
- **B5** (f0ae7d6): 16, 32 and 64 as the chains 4 × 4, 4 × 8, 8 × 8 through a stack tile
  with C rows per register (64's codelet over two rows spilled about a thousand registers
  with GCC 16). **Rejected** (deleted, c5d9292): EPYC 64 × 2 0.97 of the chain, 16 and 32
  1.2-1.6 (run 20261118-032128-4316); Xeon 1.7-3.8 of avx512's pair leaves (run
  20261118-032040-94a1); M5 Max float 16 0.70 against the codelet's 0.55, 64 1.05-1.31 (run
  20261118-031708-11d0).
- avx512's own leaves (C = 4) on batches of 2 and 3: 1.4-4.3 of the leaf with B, 4-15 with
  B2 (partial groups of 2 and 3 rows in one `zmm`); from 4 rows they beat avx2's at 11 and
  13-25 points (15: 0.61-0.67 against 0.73-0.78; runs 20261118-022102-90c2,
  20261118-032040-94a1).

### Screen of B4 (the rule's evidence)

B4 ÷ the leaf, pair leaf or chain of `main`, per length (rows across: 2, 3, 4 both
directions, 8, 64, 1024, in place 2 and 3). EPYC (run 20261118-025728-609b, band ±1.28 %):
lengths 2-8 lose at 2-4 rows (1.0-1.2; in place 1.1-1.85) and gain little from 8 rows
(0.90-0.96); 9-15, 17, 19, 23 gain 0.65-0.95, 25 0.45-0.56; 16, 20, 32 lose (1.0-1.4: their
rows are pair leaves or chains already on whole registers); 64 1.00. Xeon with avx2's leaves
(run 20261118-025722-2bc0, band ±2.76 %): 9-15, 17, 19, 23 0.67-0.96, 25 0.43-0.55, 2-8 lose
at 2-4 rows, 16, 32, 64 1.4-3.0 (avx512's pair leaves). M5 Max float (run
20261118-025350-b624, band ±1.00 %): 4-23 0.53-0.92 at every batch, 2 loses (1.0-1.16), 64
1.42-1.73.

Targets, FFTN ÷ FFTW from the B4 screens (`main` → B4): EPYC 15 × 2 backward 1.24 → 0.91,
64 × 2 backward 1.29 → 1.30 (no arm closes it: B2 0.91, B5 0.97 of the chain); Xeon 15 × 2
backward 1.14 → 0.88 (÷ MKL 0.78 → 0.60).

The rule (rules.cpp, `batch_rows_of`; arm R): per length the smallest batch from which
every screened case ran below 0.97 of `main` with none slower, restricted to lengths whose
row alone is the leaf of n, so that a batch's rows keep the bits of a plan of one row
(tests/api `BatchedRowsMatchSingleRows`): this leaves out 12 on x86 (0.86-0.98), 8 on avx512,
20 and 25 (never leaves; 25 gains 0.43-0.56 on x86) and 32 in float on NEON.

## Verdict

Pending.
