# X30b: twiddle tables on x86 at 1024–4096 points: alignment, storage per pass, three passes

- **Status:** done
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

*Amendment 1 (after the counter screen below, before any fbench run finished).* The
first runs (Slurm 510261 EPYC, 15971 and 15973 Xeon; 510264) were cancelled unfinished when
the counters showed what the size threshold misses: a compact per-lane table (grouped
pass) cost 1.06–1.13 of the cycles at 2048 and 4096 in double whatever its size, while
compact broadcast tables (lane passes), which cost no extra operation, gave the best
cycles at 1024 (0.92 of A against 0.97 for the size rule). Arm C becomes **C2** (commit on
C): one limit per table kind, `machine_profile::compact_grouped_bytes` (per-lane, default
never) and `compact_shared_bytes` (broadcast, default 0: every lane pass), fbench's
`COMPACT_GROUPED_BYTES` and `COMPACT_SHARED_BYTES`. Arms: A, L, CN (C2, both never: the
FMADDSUB arithmetic alone), **CS** (C2's default: broadcast tables compact), CSG (CS with
per-lane tables above 16 KiB compact, the size rule for the grouped pass), C1 (every
table compact). The decision rule for C applies to CS, CSG and C1 in place of C, C8, C1.

## Method

- **Cases**: `bench/catalogues/x30.toml` (c2c rows of 128–4096 in double and float, batches
  of 64, r2c and c2r of 512–4096, planes 256² and 512²); collateral: the standard
  catalogue's tracking tier.
- **Runs per machine** (one thread, fbench interleaved process rounds, A/A control,
  `--serial`): x30 catalogue and tracking tier, arms A, L and C's variants (amendment 1:
  CN, CS, CSG, C1). The variants are judged against L: the runs are re-read with L
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

### Counter screen (EPYC, before any timing run)

EPYC 7763 node17, GCC 16.2, `avx2`, single processes, medians of three, batch 1 (Slurm
510271, `fftn-x30b/pmu.sh` on DMOG; builds A e1879cb, L b5d3d6d, C 88cd9c3 with C's size
rule at 16 KiB, CN = never, C1 = every table). Cycles per transform as a ratio to A;
misaligned loads (PMCx047) and L1 fills (PMCx044) per transform:

| n (double) | L | CN | C (16 KiB) | C1 | misaligned A → L | L1 fills A / L / C1 |
|---|---|---|---|---|---|---|
| 256 | 1.015 | 0.966 | 0.955 | 1.027 | 122 → 0 | 0 |
| 512 | 0.994 | 0.985 | 0.988 | 0.997 | 0 | 523 / 522 / 200 |
| 1024 | 0.979 | 0.962 | 0.966 | 0.920 | 859 → 0 | 1861 / 1853 / 1512 |
| 2048 | 1.010 | 1.014 | 1.130 | 1.111 | 0 | 5157 / 5136 / 5463 |
| 3072 | 0.900 | 0.902 | 0.958 | 0.947 | 4135 → 1 | 7768 / 7386 / 7669 |
| 4096 | 1.000 | 0.995 | 1.060 | 1.052 | 1 → 1 | 12708 / 12694 / 13406 |

Float: L 0.88–1.03 (128 0.880, 2048 0.941); C1 1024 0.889, 512 0.954, but 4096 1.040.
Aligned tables remove every misaligned twiddle load where main's tables happened to start
off a line (256, 1024, 3072 here; where `std::vector` puts a table varies by run). The
compact per-lane tables (C at 1024–4096, C1) add instructions (+4 % at 2048–4096) and do
not cut L1 fills at 2048–4096 (they rise); the compact broadcast tables (C1 against C at
1024, where only the lane passes differ) cut fills and cycles. Hence amendment 1.

### Counter screen of C2 (EPYC)

Same method (Slurm 510287), build C2 ead29f2: CN both kinds pre-arranged, CS (C2's default:
broadcast tables compact), C1 every table compact. Cycles ÷ A:

| n | L | CN | CS | C1 | L1 fills L / CS / C1 |
|---|---|---|---|---|---|
| double 512 | 0.948 | 0.932 | 0.923 | 0.927 | 517 / 512 / 194 |
| double 1024 | 0.987 | 0.976 | 0.972 | 0.935 | 1810 / 1691 / 1468 |
| double 2048 | 0.978 | 0.984 | 0.986 | 1.102 | 5201 / 5205 / 5515 |
| double 3072 | 0.893 | 0.909 | 0.890 | 0.941 | 7318 / 7160 / 7580 |
| double 4096 | 0.996 | 1.001 | 0.995 | 1.060 | 12694 / 12710 / 13424 |
| float 512 | 0.995 | 1.001 | 0.927 | 0.965 | 0 |
| float 1024 | 0.964 | 0.964 | 0.889 | 0.901 | 520 / 493 / 215 |
| float 4096 | 1.004 | 1.003 | 1.014 | 1.022 | 5913 / 5970 / 5942 |

The FMADDSUB arithmetic alone (CN) costs nothing against L. Compact broadcast tables gain
where the lane passes' tables matter (float 512–1024 0.89–0.93 against L's 0.96–1.00) and
cost nothing elsewhere; compact per-lane tables gain only at 1024 (radix-4 first pass,
whose table halves from 24 to 12 KiB beside a 32 KiB line) and lose at 2048–4096, where the
lines alone overflow L1.

### Arm R: three passes at 4096, `avx2` double (EPYC, run 20261116-072518-0755)

EPYC 7763, GCC 16.2, main e1879cb with `RADICES` forced, c2c 4096 double at batch 1 and
64, 6 rounds (all loaded; every case resolved), A/A band ±1.00 %. Forced ÷ the planner's
8 × 8 × 8 × 8:

| sequence | batch 1 | batch 64 |
|---|---|---|
| 8 × 8 × 64 | 1.301 | 1.305 |
| 4 × 16 × 64 | 1.495 | 1.520 |
| 64 × 64 | 1.669 | 1.654 |
| 16 × 16 × 16 | 1.744 | 1.576 |
| 16 × 4 × 64 | 1.877 | 1.583 |

Every three-pass (and the two-pass) sequence is 1.3–1.9 times slower: on AVX2 the radix-16
and radix-64 passes at these strides cost more than the pass they save (X18's finding at
4096; X30's lane radix-64 loop spills, 407 stack references per unit). R is rejected; no
override changes.

### Timing runs

All on the pre-X29 base (A e1879cb, L b5d3d6d, C2 ead29f2), GCC on x86, one thread, fbench
interleaved process rounds with the A/A control, `--serial`. C2's variants are read against
L by swapping L's labels with A's in a copy of the record (no control then; the band
given is A's). Geometric means of the median paired ratios; "over" is past 1 + band + 2 %.

| Run | Machine, target | Band | L ÷ A double / float | CN ÷ L | CS ÷ L | CSG ÷ L | C1 ÷ L |
|---|---|---|---|---|---|---|---|
| x30 20261116-073236-20e5 | EPYC `avx2`, 80 rounds all loaded | 1.00 % | 0.986 / 0.971, none over | 0.999 / 1.003 | 0.995 / 0.968 | **0.986 / 0.962** | 1.002 / 0.979 |
| tracking 20261116-081805-ee1f | EPYC `avx2`, 80 rounds all loaded | 1.00 % | 0.994, none over (worst 2²⁰ 1.016) | 0.997 | 0.994 | 0.989 | 1.001 |
| x30 20261116-073810-5ff4 | Xeon `avx512`, 80 rounds, none loaded | 1.00 % | 0.989 / 0.981, none over | – | 1.001 / 1.002 | – | 0.991 / 1.007 |
| same run | Xeon `avx2` | 1.00 % | 0.996 / 0.999, none over | – | 0.996 / 0.919 | – | 1.002 / 0.917 |
| tracking 20261116-082153-5575 | Xeon `avx512`, 77 rounds | 1.38 % | 0.994; c2r 2²⁰ 1.035 [0.992, 1.063] | – | 1.001 | – | 0.985 |
| same run | Xeon `avx2` | 1.38 % | 0.990, none over | – | 0.999 | – | 1.001 |

Cases past the limit, C2 against L:

- EPYC x30: float c2c 256 × 64 for every C2 variant (CN 1.038, CS 1.037, CSG 1.035, C1
  1.050); C1 also 128 1.068, 192 1.059, 256 × 64 1.061, r2c 512 1.039, float 192 1.069.
  Re-run (20261116-101921-03d8, 80 rounds, loaded, band 1.00 %): float 256 × 64 CN 1.042,
  CS 1.038, CSG 1.039; c2r 512 1.013–1.016 (not past). The re-run decides: past.
- EPYC tracking: none for CN, CS, CSG; C1 ten (64–128 points 1.04–1.06, 2048 × 36 1.045).
- Xeon x30: none for CS; C1 ten on `avx512` (192 1.076, 256 1.064, float 512 1.058) and
  three on `avx2` (192 1.080, 3072 1.046).
- Xeon tracking: c2c 16 for CS (`avx512` 1.054, `avx2` 1.050) and C1; C1 also 60–128.

Per case, CSG against L on the EPYC (x30, double): 768 0.964, 1024 0.954, 1536 0.955, 2048
0.971, 4096 0.977, 1024 × 64 0.962, 4096 × 64 0.967, c2r 4096 0.971; float 512–1024
0.92–0.93, 4096 0.908; tracking 16384 0.903, 10403 0.914, 1024 0.957. The compact per-lane
tables that lost cycles at 2048–4096 in the single-process counters won there in fbench.

Float 256 × 64 runs a 16 × 16 chain (`avx2` float), whose tables are below every limit
tried: its passes are those of L in every C2 variant, CN included, except for the FMADDSUB
form of the twiddle product (and a larger library). The loss there is not the compact
storage's.

Not run: CSG on the Xeon (its run, Slurm 15991 `x30b3-xeon-x30`, was cancelled once the
EPYC re-run had decided C); a re-run of Xeon c2c 16 and of `avx512` c2r 2²⁰ (Slurm 16005,
`x30b2-xeon-rerun`), run 20261116-112332-7993 (80 rounds, 1 loaded, band 1.02 %): L ÷ A
`avx512` c2r 2²⁰ 0.983 [0.963, 0.996], c2c 16 0.998; c2c 16 against L: CS 1.045 and CSG
1.053 (`avx512`), CS 1.060 and CSG 1.066 (`avx2`), so the 16-point loss is real (CSG's
other cases there: `avx512` c2r 1000 1.032, `avx2` 1009 0.971).

## Verdict

- **L adopted** (all targets): plan tables start on a 64-byte boundary. (a) x30 double
  L ÷ A EPYC 0.986, Xeon `avx512` 0.989, `avx2` 0.996, Mac 0.999; (b) no case past the limit
  on the EPYC, the Xeon's `avx2` or the Mac; on the Xeon's `avx512` c2r 2²⁰ 1.035 with an
  interval containing 1 in the tracking run, 0.983 on the re-run (20261116-112332-7993); (c) float 0.971, 0.981, 0.999, 1.000. Bits unchanged.
- **C rejected** on both x86 targets by (b), and on `avx512` also by (a), and reverted
  (b54a7f4): its best form, CSG (broadcast tables compact, per-lane tables above 16 KiB),
  took the EPYC's x30 cases to 0.986 (double) and 0.962 (float) of L and the tracking tier
  to 0.989, but float 256 × 64 stayed 1.035–1.042 on the re-run in every C2 variant,
  including the one with no compact table (CN), so the loss belongs to the FMADDSUB
  arithmetic or to code placement, not to the storage. On the Xeon the broadcast form gave
  nothing in double on `avx512` (1.001) and 0.919 in float on `avx2`; c2c 16, a leaf C2 does
  not touch, ran 1.05 in the tracking tier there (X30's T showed the same, 1.057–1.066).
  A follow-up that keeps the pre-arranged (wr, wr), (−wi, wi) arithmetic and builds
  (−wi, wi) from compact storage only where the table is large, with the 16-point and
  256 × 64 cases checked first, would test whether the gains survive without that loss.
- **R rejected**: every three-pass (and the two-pass) `avx2` double 4096 sequence is
  1.30–1.88 times the planner's 8 × 8 × 8 × 8.

## Slurm jobs

EPYC (DMOG): builds 510237, 510260, 510263, 510283, 510285; timing 510238 (failed: no case
selected), 510266 (R), 510261 and 510264 (cancelled by amendment 1), 510284 (x30), 510286
(tracking), 510382 (re-run); counters and tests 510269, 510271, 510287, 510294, 510295,
510301. Xeon: builds 15970, 15972, 15974, 15976, 15990; timing 15971 and 15973 (cancelled
by amendment 1), 15975 (x30), 15977 (tracking), 15991 (cancelled), 16005 (re-run); tests
15978 (cancelled, wrong path), 15979.
