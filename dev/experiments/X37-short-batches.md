# X37: short rows and small batches in the phase 1 tail

- **Status:** done (arms 0–2 adopted on x86 and NEON where they apply, arm 2 without `avx512`'s 8 and 12; arm 3 adopted on AArch64 only)
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

3. **A lighter path for leaf rows** (added after the counters below, before its runs):
   the leaves take their batch distances in bytes as given (no signed division into
   reals: six instructions per call), and `execute_core`'s path for leaf rows no longer
   shares a frame with the general path (that moves out of line). Target cases: the
   single rows and batches that are one leaf (2–16 points, every batch); bit-identical.

Bits: arm 0 changes no library code; arm 3 is bit-identical; arms 1 and 2 give the bits of the chain r1 × m of
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

Toolchains: GCC 14.4 on the Xeon (`avx512`), GCC 16.2 on the EPYC (DMOG, `-c 8`, shared),
Clang 23 on the Mac; Release, one thread, double. Arms: A e8c2639 (`main`), B dd61535 (arm
0), P b5457ac (arm 1), Q 6a305b3 (arm 2 as first written), R 3abec91 (arm 3 as first
written), S df90c48 (the final branch: arm 2 without `avx512`'s 8 and 12, arm 3 without the
split entry). Ratios are medians over rounds of the paired per-round ratio.

### Where the time goes

Counters (fbench `--pmu cycles,instructions`, user mode, three process rounds; FFTW with
shared wisdom and `PLAN_NOTE=1`; Xeon job 16107, EPYC job 511080), ns / instructions per
call, A → Q against FFTW:

| case | Xeon A | Xeon Q | Xeon FFTW (plan) | EPYC A | EPYC Q | EPYC FFTW (plan) |
|---|---|---|---|---|---|---|
| 2 × 1 | 7.3 / 101 | 6.4 / 91 | 5.9 / 73 (`n2fv_2_sse2`) | 7.2 / 95 | 5.5 / 77 | 7.5 / 71 |
| 18 × 1 | 45.5 / 506 | 25.0 / 213 | 37.0 / 393 (9 × 2, `t1fv_9`, `n2fv_2`) | 33.5 / 469 | 17.4 / 199 | 30.8 / 382 |
| 22 × 1 | 61.6 / 641 | 32.9 / 260 | 46.4 / 500 (2 × 11, `n1fv_11` on 2 columns) | 44.6 / 583 | 23.0 / 248 | 41.7 / 499 |
| 16 × 1 | 20.2 / 228 | 16.7 / 134 | 21.6 / 218 (`n2fv_16_avx2_128`) | 15.8 / 221 | 12.5 / 160 | 16.1 / 216 |
| 16 × 16 | 208 / 2298 | 123 / 884 | 184 / 1962 (`n1fv_16_avx2`, 2 rows per register) | 197 / 2291 | 146 / 1555 | 157 / 1961 |
| 16 × 256 | 3322 / 35418 | 1942 / 12884 | 3154 / 30282; MKL 1906 / 12749 | 3145 / 35411 | 2388 / 23875 | 2741 / 30281 |
| 15 × 2 bwd | 38.9 / 392 | 37.4 / 382 | 32.0 / 321 (`n1bv_15_avx2`, 2 rows) | 34.9 / 385 | 31.8 / 367 | 27.1 / 321 |
| 64 × 2 bwd | 73.5 / 598 | 72.2 / 588 | 143 / 1382 | 143.6 / 1951 | 142.1 / 1936 | 110.9 / 1374 (`n1bv_64_avx2`, 2 rows) |

So: the composites without a pair leaf and the batches of 16 were kernel cost (2–3 times
FFTW's instructions; the 128-bit leaf does a whole radix-16 codelet per row on one complex
per register); the tiny rows were fixed cost (Xeon 2 points: 18 more instructions than FFTW
per call, the extra library call of fbench's backend and the leaf's division of its
distances); the EPYC's backward batches of 2 are FFTW running two rows in the lanes of one
`ymm` with a whole codelet (15, 64), which no arm here addresses (FFTN's 64 on `avx2` is a
chain of 1.4 times FFTW's instructions). The Mac's in-place 2 × 3 (1.39–1.44) is a latency
chain: in place with one buffer set, each execution reads what the previous one stored, and
FFTN's 2-point leaf (4 instructions) still takes 3.8 ns per call there against 3.1 out of
place in a microbenchmark; a hand-written load-add-store loop on the same address measures
3.1–3.3 ns per iteration with either paired or single 128-bit accesses, under FFTW's 2.6 ns
per call. Not resolved.

### Arms 0–2 (runs 20261116-190621-c3ce Xeon, 80 rounds, 19 loaded, band ±1.00 %, Slurm 16097, builds 16095; 20261116-190423-3869 EPYC, 80 rounds, all loaded, band ±1.00 %, Slurm 511074, builds 511073; 20261116-182910-c795 Mac, 12 rounds, band ±1.26 %)

| arm (targets) | Xeon `avx512` | EPYC `avx2` | Mac `neon_fcma` |
|---|---|---|---|
| 0, bench (all x37; single rows 2–6) | 0.970; 0.889 (2 points 0.88, 6 0.86) | 0.949; 0.808 (2 points 0.80) | 0.995; 0.963 |
| 1, short pairs (18, 22, 26, 30 x86; 18–30 NEON) | 0.468 (0.395–0.537) | 0.519 (0.486–0.543) | 0.718 (0.576–0.843; 20 points 0.66) |
| 2, pairs of 8–16 (8, 12, 16) | 16: 0.661 (16 × 16 0.595, 16 × 256 0.589); 8 and 12: 1.009, in place 8 × 3 2.15, 12 × 3 1.55 | 0.852 (0.747–0.967) | does not apply |

Rule 3 removed `avx512`'s 8 = 2 × 4 and 12 = 3 × 4 (r1 < C: the second pass stores partial
runs, and an in-place batch whose next execution reads them back stalls); their
out-of-place cases were 0.85–0.98. `avx512` plans take `avx2`'s 4 × 2 and 6 × 2 for them
instead (find_more_pair_leaves), timed in the final runs. Other cases slower than the arm
before beyond the band: Xeon c2c 28 × 1 1.035–1.049 in arms 0–2 (its 4 × 7 chain is
unchanged; code placement of the entry), 22 and 26 × 1 1.014–1.017 in arm 0 only; EPYC 64 ×
256 1.022 in arm 2 (plan unchanged).

### Arm 3 (Mac run 20261116-185336-400d, 12 rounds, band ±2.23 %; EPYC run 20261116-201125-8509, 21 rounds, all loaded, band ±1.18 %, Slurm 511093)

As first written (R): on the leaf rows (2–16 points, every batch) Mac R ÷ P 0.963 (single
rows 2–6 0.83–0.89), EPYC R ÷ Q 0.969 (0.849–1.016). But on the EPYC single rows that
take the general path got slower: 31, 60, 61, 100, 144, 192, 256 points and float 97 × 2 at
1.03–1.13 of Q; the out-of-line general path (rule 3). The final arm (S) keeps the byte
distances and drops the split.

### Arm 3 without the split entry (S df90c48), and the final branch (U e7063ff)

S kept the leaves' byte distances and dropped the split. On the EPYC that was worse than
arm 2 on its own targets: run 20261116-233121-5385 (Slurm 511237, builds 511236; Q, S,
FFTW; 24 rounds, all loaded, band ±1.00 %): S ÷ Q over the leaf rows 1.042 (single rows of
2–8 points 1.10–1.20, 2 × 3 1.13), the general-path rows 1.019 (61 points 1.13). With the
split (R) the same leaf rows had been 0.85–0.95 of Q. On the Xeon S gained (below), on the
Mac the split is worth 6–9 % at 2–6 points by itself. So arm 3 is kept on AArch64 only
(e7063ff: the leaves' byte distances and the out-of-line general path under
`FFTN_ARCH_AARCH64`; x86 builds compile arm 2's text), and the final branch U is arm 2 (with
rule 3's table change) on x86 and arm 3 on arm64.

- Xeon x37, run 20261116-225820-974d (Slurm 16147, build 16146; Q, S, FFTW, MKL; 24
  rounds, all loaded, band ±2.66 %): S ÷ Q on arm 3's leaf targets (2–7, 15, 16 points)
  0.961 (2 × 1 0.96, 4 × 1 0.89, 16 × 1 0.84), slower 3 × 3 in place 1.079 (a latency
  chain, still 1.01 of FFTW); 8 and 12 points with `avx2`'s pair leaves against
  `avx512`'s 2 × 4 and 3 × 4: 0.904 (in place 8 × 3 0.41, 12 × 3 0.62; 12 × 256 and
  12 × 3 out of place 1.06–1.07, still 0.82–0.91 of FFTW); the general-path rows 31–256
  0.996. FFTN ÷ FFTW over the x37 cases 0.721, over 1.05 only 28 × 1 (1.09, a 4 × 7 chain)
  and 15 × 2 backward (1.13); ÷ MKL 0.576, over 1.05 24 × 64 and 24 × 1024 (1.10–1.15) and
  32 × 256 (1.07).
- Xeon tracking tier, run 20261116-230356-f5f4 (Slurm 16149, build 16148; A, S; 16
  rounds, 9 loaded, band ±1.87 %): S ÷ A 0.983, 5 faster, nothing slower.
- Xeon exit catalogue, run 20261116-202515-4a68 (Slurm 16118; A, Q, R, FFTW, MKL; 15
  rounds, 2 loaded, band ±2.25 %): cases over 1.05 of FFTW, double, A 30 of 193 → Q 25 (Q
  still had `avx512`'s 2 × 4 and 3 × 4: 8 × 3 and 12 × 3 in place 1.88 and 1.24, which S
  takes to 0.79–0.80 of FFTW, so the final branch has 23); of MKL A 32 → Q 29 (final 27).
- Mac exit catalogue, run 20261116-185638-1733 (A, R, FFTW; 10 rounds, band ±2.26 %): over
  1.05 of FFTW A 24 of 193 → R 19; R ÷ A 0.983, nothing slower. Mac tracking tier, run
  20261116-191129-ab4d (A, R; band ±2.94 %): 0.988, 4 faster, nothing slower. Mac x37, run
  20261116-211646-3571 (R, S, FFTW; band ±1.00 %): without the split entry (S) single rows
  of 2–6 points are 1.06–1.09 of R and the general-path rows 1.00, hence the split on
  AArch64.

Library text (`size` of `libfftn.a`): x86-64 GCC 14 A 47.82 MB, P +0.06 %, Q +0.10 %, R
−0.23 % (the leaves lose their divisions); arm64 Clang A 19.25 MB, P +0.73 %, R +0.20 %.

## Verdict

Rule 1 holds: the whole suite passes natively on the Xeon (GCC 14 `-Werror`, `avx512` and
`avx2`, 1797 tests, Slurm 16099; one test fixed for arm 2, 66696ad), on the Mac (1850 tests
after merging `main`) and in CI; `Passes.PairLeavesMatchTheForcedChain` now also checks the
`avx2` leaves an `avx512` plan takes. Arm 3 is bit-identical; NEON's pair leaves keep the
rules' factors, so NEON results do not change.

| arm | Xeon `avx512` | EPYC `avx2` | Mac `neon_fcma` |
|---|---|---|---|
| 0, bench calls `execute_all` | kept (rule 4): single rows of 2–6 0.889 | kept: 0.808 | kept: 0.963 |
| 1, short pair leaves | adopted: 0.468 | adopted: 0.519 | adopted: 0.718 |
| 2, pair leaves of 8–16 | adopted for 16 (0.661); 8 and 12 take `avx2`'s 4 × 2 and 6 × 2 (rule 3: `avx512`'s 2 × 4 and 3 × 4 lost in place, 1.5–2.1) | adopted: 0.852 | does not apply |
| 3, lighter leaf path | rejected on x86 (Xeon 0.961 without the split but 3 × 3 in place 1.08; EPYC 1.042 without the split, general path 1.03–1.13 with it) | rejected | adopted: 0.963 (single rows 2–6 0.83–0.89) |

The exit check's failing cases of causes 2 and 3, FFTN ÷ FFTW (÷ MKL on the Xeon), before →
after (exit catalogue runs above; A = `main` e8c2639):

- Xeon: 2, 4, 6 points 1.25, 1.23, 1.13 → 1.10, 1.11, 0.98; 18, 22, 26, 30 1.24–1.32 →
  0.64–0.71; 2 × 3 in place 1.18 → 1.01; 16 × 16 1.13 → 0.67; ÷ MKL 16 × 16 1.57 → 0.94,
  16 × 256 1.71 → 1.01, 16 × 3 1.19–1.28 → 0.74–0.77. Left: 2 and 4 points (1.10–1.11, a
  few cycles of entry: 91 instructions per call against FFTW's 73), 15 × 2 backward
  (1.19: FFTW runs both rows in one register), ÷ MKL 512 × 16 (1.18, not short).
- EPYC: 18 and 26 points 1.10–1.14 → 0.53–0.57; 16 × 16, 256, 4096 1.26, 1.19, 1.16 →
  0.93, 0.90, 0.89; 12 × 64 and 1024 → 0.77–0.91. Left: 15 × 2 and 64 × 2 backward (1.22,
  1.30) and 64 × 16 (1.15): FFTW's two-row codelets (`n1bv_15`, `n1bv_64`, `n1fv_64` on
  `ymm`), which pair leaves do not reach (C divides neither 15 nor, in registers, 64).
- Mac: 20 points 1.55 → 1.01; 2 and 4 points 1.23, 1.20 → 0.97, 0.98; 2 × 3 out of place
  1.09 → 0.94, 3 × 3 1.11 → 0.99; 22 and 26 0.86–0.91. Left: 2 × 3 in place 1.39 (a memory
  latency chain between executions, above), 7 × 3 in place 1.09.

Cases over 1.05 in the exit catalogue (double, 193 cases): Xeon ÷ FFTW 30 → 23, ÷ MKL 32 →
27 (arm 2's run, less 8 × 3 and 12 × 3 in place, which the final branch fixes); EPYC ÷ FFTW 36 → 30 (arm 2's run; on the EPYC the final branch compiles arm 2's code);
Mac ÷ FFTW 24 → 19. Tracking tier against `main`: Mac 0.988 (arm 3 run) and 0.993 (final
against `main` 1fe3fe7), nothing slower; Xeon 0.983 (S), nothing slower. Exceptions under
rule 3, recorded: Xeon 28 × 1 1.035–1.049 from arm 0 on (plan unchanged), EPYC 64 × 256
1.022 in arm 2 (plan unchanged). Library: x86-64 +0.10 %, arm64 +0.2–0.7 %.

DESIGN.md 6.1, 8.2's table and the decision log, LESSONS 1.11 and 4.14 updated.
