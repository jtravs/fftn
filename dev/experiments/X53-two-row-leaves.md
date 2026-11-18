# X53: batch leaves (C rows per register) for tiny batches

- **Status:** done (adopted on NEON float, avx2 and avx512 double)
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
  5. Added after arm R's Xeon judging run, before any run of R2: R had one case slower
     beyond the band whose plan it does not change (a single row of 2 points, 1.106, re-run
     1.091). **Arm R2** (R plus the JCC padding of the avx2-region leaf units, 56c30b6) is
     adopted in R's place if, against A with fresh `--serial` runs, it meets rule 2 on the
     Xeon (`x37`, `x53`, tracking tier) and no case of `x37` or the tracking tier on the
     EPYC is slower than A beyond the band (the padding moves code there too); else the
     Xeon keeps no batch leaves and R stands on the EPYC and NEON.
  6. Added after the maintainer dropped the requirement that a batch's row match a plan of
     that row alone (2026-11-18, AGENTS.md section 6 and DESIGN 15.5; the published
     guarantee, every row of a plan getting the same arithmetic whatever its position,
     stays), before any run of it: **arm R4** adds the lengths R left out for that reason
     only, at the batches the B4 screens gave: on `avx2` 12 from 8 rows and 25 from 2; on
     `avx512` 8 from 1024 rows, 12 from 8 and 25 from 2 (its own four rows per register from
     4); in float on NEON 20 and 32 from 2. Baseline: R's branch head with `main` merged
     (b89ca02). R4 is adopted per target if, over the cases R4 changes (`x53`, the new lengths
     from their rows), the geometric mean is at most 0.97 of the baseline with its interval
     clear of the A/A band and none of them is slower beyond the band, and no other case of
     `x53` or of the tracking tier is slower beyond the band except one whose code R4
     provably leaves as it was (same plan, byte-identical objects: AGENTS.md section 6),
     recorded with that evidence. `--serial` runs on the EPYC and the Xeon; the Mac (float)
     once it runs on mains power.

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

### Judging runs of arm R (fd75adf against A, `main` 4d75c48; `--serial`)

Geometric mean over the cases R's rule changes; FFTN ÷ FFTW (÷ MKL) from the same runs.

- **EPYC 7763** (every round loaded by other jobs on the node): `x37` run
  20261118-064012-095a (band ±1.75 %): 15 × 2 backward 0.723 (FFTN ÷ FFTW 1.21 → 0.88), 64 ×
  2 backward 0.992 (1.30 → 1.29), nothing slower. `x53` run 20261118-064329-911a (band
  ±1.12 %): the 68 changed cases 0.810, worst 7 × 8 0.972 (9-15, 17, 19, 23 at their rows
  0.63-0.95); one case past the band, 4 × 4 1.148 [1.012, 1.273], whose plan R does not
  change (4 points from 8 rows): re-run (job 512652, below). Tracking tier run
  20261118-065015-043b (band ±3.07 %): 1.001, nothing slower.
- **Xeon 6240** (quiet): `x37` run 20261118-061230-7b61 (band ±1.00 %): 15 × 2 backward 0.780
  (FFTN ÷ FFTW 1.16 → 0.90; ÷ MKL 0.78 → 0.61), 64 × 2 backward 1.014 (0.54 of FFTW); four
  cases slower beyond the band whose plans R does not change: single rows of 2 (1.106), 4
  (1.078) and 144 points (1.049), 30 × 1024 (1.037). Re-run 20261118-065943-3d75 (band
  ±1.71 %): only the row of 2 points stays slower (1.091 [1.076, 1.100]): code placement (the
  narrow leaves' unit is unpadded, LESSONS 2.11), hence arm R2 (rule 5). `x53` run
  20261118-061610-5ef1 (band ±1.78 %): the 83 changed cases 0.688, worst 15 × 3 backward
  0.959 (avx512's own four rows per register from 4 rows: 13-23 points 0.38-0.57), nothing
  slower. Tracking tier run 20261118-062417-2a58 (band ±1.73 %): 0.995, nothing slower.
- **M5 Max** (float, `neon_fcma`): `x53` f32 run 20261118-035249-9ff7 (band ±1.00 %): the 100
  changed cases 0.606, worst 5 × 3 0.919, nothing slower. Tracking tier run
  20261118-035829-0210 (band ±1.00 %, 6 of 12 rounds loaded): 1.000, all 61 cases no
  evidence (the rule changes no double case: NEON has no batch leaves in double).

### Arms R2 and R3 (the tiny rows R does not change)

- **R2** (56c30b6: R plus `-Wa,-mbranches-within-32B-boundaries` on the avx2-region leaf
  units, rule 5). Xeon: `x37` run 20261118-072152-c97d (band ±1.42 %): the row of 2 points
  still 1.092, the row of 4 1.035; `x53` run 20261118-072627-62b4: the changed cases 0.702,
  and 3 × 4 backward, unchanged, 1.061; tracking run 20261118-073120-b16f 0.996. EPYC: `x37`
  run 20261118-072654-7292: three unchanged cases newly slower (4 points 1.055, 16 × 3 1.023,
  256 points 1.041); tracking run 20261118-073011-ecba 0.997. **Rejected** (rule 5) and
  reverted (cd19adf): the padding moved the losses rather than removing them.
- **R3** (0461442, a local diagnostic commit, not pushed: R without avx512's batch
  tables and without the Xeon's rule). Xeon run 20261118-073846-febc (band ±1.00 %, quiet):
  the row of 2 points 1.099 of `main` with R3, 1.103 with R in the same run. The loss on the
  2-point row does not come from the Xeon's batch leaves or from its rule; deleting them
  (rule 3) would not remove it.
- The EPYC's 4 × 4, unchanged by R's rule (4 points from 8 rows), re-run (job 512652, run
  20261118-072527-ded1, band ±1.03 %): 1.243 again, bimodal over the rounds (R at 9.1 ns in
  2 of 20 rounds, 10.7-13.0 in the others; A at 9.0-9.9 in all), the plan text identical.
  The 7 × 8 case it re-ran with, changed, 0.969.

The cases slower than `main` beyond the band whose plans R does not change, then, are the
Xeon's single row of 2 points (1.09-1.10 in three runs) and the EPYC's 4 × 4 (1.15, 1.24),
and each move or vanish when code moves (R2, re-runs: the Xeon's 4, 30 × 1024, 144 and 3 × 4
and the EPYC's 16 × 3 and 256 came and went). FFTN ÷ FFTW on them stays below 1 (Xeon 2
points 0.83, EPYC 4 × 4 below FFTW in the screens).

### Arm R4 (fd18d63 against R's head with `main` merged, b89ca02; rule 6; `--serial`)

- **EPYC**: `x53` run 20261118-104720-4fbd (band ±1.00 %, every round loaded): the 12 cases
  R4 changes (12 from 8 rows, 25 from 2) 0.565, worst 12 × 8 0.890; no case slower. Tracking
  tier run 20261118-105311-28ba (band ±2.10 %): 0.999, nothing slower. **Adopted.**
- **Xeon**: `x53` run 20261118-104834-112c (band ±2.20 %, quiet): the 13 cases R4 changes
  (8 × 1024, 12 from 8 rows, 25 from 2) 0.511, worst 12 × 8 0.968 [0.960, 0.974]; no case
  slower. Tracking tier run 20261118-105414-c1aa (band ±2.43 %): 1.001, nothing slower.
  **Adopted.**
- **M5 Max** (float): `x53` f32 run 20261118-120534-6bf3 (band ±3.32 %, 12 of 14 rounds
  loaded): the 12 cases R4 changes (20 and 32 from 2 rows) 0.589, worst 32 × 3 0.826; no case
  slower. Tracking tier run 20261118-120958-0099: 0.997, nothing slower. **Adopted.** These
  re-run the first pair (20261118-110350-322f, -110843-8571: 0.583 and 1.001), which started
  within minutes of the Mac's return to mains power.

### The EPYC's 64 × 2 after X54 (a screen, not a verdict)

X54 (pull request 69) ordered the radix-64 codelet for 16 registers. The batch leaf of 64 forced
on the merged head (run 20261118-104206-ee09, one DMOG screen, band ±1.00 %, every round
loaded) now runs 64 × 2 backward at 0.908 of the chain (1.00 before X54), 64 × 4, × 8 and
× 64 at 0.90-0.93; 64 × 3 (the leftover row as a full-width leaf) 1.06-1.08 and 64 × 1024
1.035 lose. FFTN ÷ FFTW on 64 × 2 backward would go from 1.30 to 1.18: the gain is real but
does not clear the exit case, and a rule would need an upper batch limit and even counts, so
no new arm was built.

## Verdict

- **NEON (float): adopted** (rule 2): the 100 changed cases 0.606 of `main`, nothing slower,
  tracking tier 1.000.
- **avx2 (EPYC) and avx512 (Xeon), double: adopted** under the maintainer's rule of
  2026-11-18 (AGENTS.md section 6, pull request 71): a case past a limit whose code the
  change provably leaves as it was does not decide the verdict. Everything else in rule 2 is
  met: EPYC changed cases 0.810, 15 × 2 backward 0.723 (FFTN ÷ FFTW 1.21 → 0.88), tracking
  1.001; Xeon changed cases 0.688, 15 × 2 backward 0.780 (1.16 → 0.90 of FFTW, 0.78 → 0.61
  of MKL), tracking 0.995. The two cases past the band, the Xeon's single row of 2 points
  (1.09-1.10) and the EPYC's 4 × 4 (1.15, 1.24), meet the rule:
  - **Same plan.** fbench's plan note for both arms in every run that timed them reads
    `axis 0 (n = 2): radices 2` and `axis 0 (n = 4): radices 4` (runs
    20261118-061230-7b61, -065943-3d75, -073846-febc on the Xeon; -064329-911a,
    -072527-ded1 on the EPYC); a batch leaf would read `(batch leaf, C rows per register)`.
    The rule cannot take them: a single row has no batch, and 4 points take batch leaves
    from 8 rows.
  - **Byte-identical code on the path they run**, release GCC (16.2 on the EPYC, 14.4 on
    the Xeon), the fbench build cache's objects of `main` 4d75c48 and of R fd75adf: the
    narrow leaves' object (`narrow_avx2.cpp.o`, whose 128-bit unit leaves of 2 and 4 points
    these plans call), `driver.cpp.o` and `api.cpp.o` identical byte for byte (md5);
    in `plan_core.cpp.o`, `execute_all` and `execute_core` identical in bytes and in the
    relocations inside them (only `build`, `make_plan_core`, `describe` and other plan-time
    functions differ); in fbench's FFTN backend, `fbench_execute` identical (its threaded
    branch's callee `execute_threaded` differs: the session record gained a field; serial
    runs never call it); the harness's functions identical. Each function is compared by
    its bytes and the relocations inside it with `bench/x53/funccmp.py`.
  - **Code placement.** R3 (without X53's avx512 code and the Xeon's rule) keeps the 2-point
    loss (1.099 of `main`, run 20261118-073846-febc), and R2's padding moved such losses to
    other unchanged cases (above).
- **Arm R4 adopted on every target** (rule 6): the lengths left out for single-row bits (8
  on avx512, 12 and 25 on x86, 20 and 32 in float on NEON) run at 0.51-0.58 of R on the
  cases they change, nothing slower; every row of a plan still gets the same arithmetic
  (tests/api `BatchRowsGetTheSameArithmeticAnywhere`).
- **64 × 2 backward is not closed** on the EPYC (FFTN ÷ FFTW 1.30 → 1.29). Before X54 the
  codelet over two rows ran at 1.00 of the chain, per-element gathering 0.91 and the chains
  through a tile 0.97, all short of FFTW's 0.77. After X54 the codelet runs at 0.91 (1.18 of
  FFTW), which still does not clear the case (screen above).
- **Not addressed:** the Mac's 2 × 3 in place (double: NEON has no batch leaves there; X49's
  store-to-load forwarding) and the Xeon's 2 × 3 in place and single row of 4 (single rows
  and batches of three of 2-4 points take no batch leaves; in these runs FFTN was 0.91 and
  0.83 of FFTW on them).
- Library size: eight new kernel units (two per target with batch leaves).
