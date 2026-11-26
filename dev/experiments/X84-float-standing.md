# X84: float's standing against FFTW's float build

- **Status:** done: float stands at 0.77–0.82 of FFTW's float build over the catalogue (c2c rows 0.69–0.70); 54, 81 and 94 cases above 1.10 on the Mac, the EPYC and the Xeon, listed below for the maintainer (Q23)
- **Owner:** task 6.11 session, 2026-11-26
- **Phase / task:** 6.11, part 2 ([PLAN.md](../PLAN.md), phase 6 table); DESIGN 18.2 "Phase 6", Q23

## Question

Where does FFTN's single precision stand against FFTW's single-precision build at one thread,
case by case over the standard catalogue, and which cases are above 1.10 of FFTW, for the
maintainer to name for tuning (Q23)?

## Hypothesis

Every phase 1 to 4 experiment judged double; float shares the planner rules, the generator's
codelets and the strategies, but a `float` vector holds twice the elements, so a radix or a leaf
chosen for double's register budget and cache footprints may be wrong for float. Expected: a
geometric mean near double's standing on each machine (X56 on the c2c rows: Mac 0.737, EPYC
0.788, Xeon 0.658 of FFTW), with a tail of cases where the footprint limits, measured for double
bytes, fall at the wrong lengths (rows near the L2 limits, real rows near X75's L2/8 limit) and
small batches where float's wider vectors make FFTW's SIMD batch loops relatively stronger.

## Change under test

None: a measurement of `main`'s library. The arms are built from this branch, whose library is
`main` 09e0d2b5's and whose fbench FFTW backend takes single precision (commit 0fb24876: the
backend links `libfftw3f`; `--precision f32` takes the catalogue's cases in float). No switch.

Arms: `N` = FFTN, with fbench's identical-arm control `N~`; `F` = FFTW 3.3.11's single-precision
library with its SIMD codelets, MEASURE, fresh wisdom per run.

- **Mac M5 Max:** the NEON build of FFTW (`fftw-3.3.11-neon`, whose `libfftw3f` has NEON
  codelets), Homebrew Clang 23, `neon_fcma`.
- **EPYC 7763 (DMOG):** FFTW 3.3.11 built with GCC 16 and SSE2/AVX2 codelets (its `libfftw3f`
  reports `sse2-avx2-avx2_128`), GCC 16.2.0, `avx2`; Slurm, 8 CPUs of a shared node, never
  `--exclusive`.
- **Xeon Gold 6240:** FFTW 3.3.11 built with GCC 16 and SSE2/AVX/AVX2/AVX-512 codelets; its
  `libfftw3f` was built for this experiment with the double library's configure flags plus
  `--enable-float` (Slurm job 17383; version string `sse2-avx-avx2-avx2_128-avx512`, as the
  double one's), GCC 14.4.0, `avx512`; Slurm `--serial`, 8 CPUs.

## Method

- **Cases:** `--catalogue standard --tier full --precision f32`: the standard catalogue's 437
  cases in single precision (the catalogue's c2c, r2c and c2r rows, batches, planes and volumes,
  in and out of place). One thread, hot, median statistic, fbench's defaults otherwise.
- **Commands:**
  - Xeon: `scripts/remote/fbench_remote.py submit xeon --serial N=fftn:<sha> F=fftw:<sha> --
    --catalogue standard --tier full --precision f32 --budget 5400`.
  - EPYC: the same with `epyc` and without `--serial`.
  - Mac: `bench/fbench.py run N=fftn:<build> F=fftw:<build> --catalogue standard --tier full
    --precision f32 --budget 1800 --max-rounds 12`, on AC power, no other fbench running,
    `pmset -g batt` before and after (a run that overlaps battery or sleep is invalid), by day.
- **Metric:** per case, FFTN ÷ FFTW (paired bootstrap interval, 95 %) and the run's A/A band.
- **Decision rules** (written before any run; no adoption follows, Q23 is the maintainer's):
  1. **The tuning list.** A case is *above 1.10* on a machine when the whole interval of its
     FFTN ÷ FFTW lies above 1.10 × (1 + the run's band). A case whose median ratio is above 1.10
     but whose interval reaches below that bound is a near miss: it gets one `--serial` re-run on
     that machine (filtered to the near misses; on the Mac a short run on AC, by day), and the
     re-run decides. The verdict lists every case above 1.10, per machine, with its ratio.
  2. **The standing** (informative, against DESIGN 1.3's double rows: ≤ 0.85 Mac and Xeon,
     ≤ 0.90 EPYC, at most 5 % of cases beyond 1.05 of FFTW beyond noise): per machine, the
     geometric mean of FFTN ÷ FFTW over the cases both arms timed and the count of cases whose
     whole interval lies above 1.05 × (1 + band); the same over the c2c rows alone
     (`^c2c-[0-9]+-b`), for comparison with X56's double figures.
  3. A case that fails fbench's accuracy check in either arm is reported as a failure, not
     timed.

**Amendment 1** (2026-11-26 15:06 UTC, before the EPYC's deciding re-run). The EPYC's near-miss
re-run (run 20261126-150451-f8fa, Slurm 520153) took six rounds in 18 s and its intervals span
0.33–0.97 of FFTW ÷ FFTN for every case: too few rounds to decide anything. It is void as
deciding; the deciding re-run repeats it with `--min-rounds 24`. The Mac's and the Xeon's
re-runs (12 rounds each, intervals of a few per cent) stand.

## Results

One thread, hot, median statistic; FFTN ÷ FFTW = 1 ÷ fbench's "F vs N" ratio, its interval
[1 ÷ upper, 1 ÷ lower]. Every case passed fbench's accuracy check in both arms on every machine.

| Machine | Run | Rounds (under load) | Band | Geomean, all 437 | c2c rows (201) | Beyond 1.05 (1 + band): all, c2c rows | Above 1.10 (1 + band) |
|---|---|---|---|---|---|---|---|
| Mac M5 Max | 20261126-132858-4724 | 12 (12) | ±1.24 % | **0.819** | 0.693 | 58 (13.3 %), 0 | 52 + 2 of 4 re-run = **54** |
| EPYC 7763 | 20261126-133150-792a | 31 (5) | ±1.00 % | **0.796** | 0.701 | 90 (20.6 %), 29 (14.4 %) | 80 + 1 of 7 re-run = **81** |
| Xeon 6240 | 20261126-133155-bc03 | 23 (12) | ±3.13 % | **0.774** | 0.697 | 101 (23.1 %), 29 (14.4 %) | 92 + 2 of 9 re-run = **94** |

Near-miss re-runs (rule 1), each deciding: Mac 20261126-135209-184d (12 rounds, ±2.32 %; r2c
32 × 30 × 2 out of place 1.37 and c2r 2048 × 36 in place 1.21 above, c2r 32 × 30 × 2 out of place
1.10 and c2c 32 × 32 × 16 1.11 not); EPYC 20261126-150623-9a50 (24 rounds, ±1.00 %, after
amendment 1; c2c 16 × 4096 1.13 above, the other six 1.06–1.10 not); Xeon 20261126-150442-f67d
(12 rounds, ±2.25 %; c2c 64 × 16 1.16 and r2c 2048 × 36 out of place 1.17 above, the other
seven 1.03–1.17 with intervals reaching below 1.125 not). A first Mac re-run,
20261126-135123-ef7c, ran on a tree with uncommitted test files (the binaries were 8eefad78's)
and was repeated clean; the EPYC's void six-round run is 20261126-150451-f8fa.

By family, FFTN ÷ FFTW (geometric means; "batch" is more than three transforms):

| Family | Mac | EPYC | Xeon |
|---|---|---|---|
| c2c rows | 0.664 | 0.650 | 0.645 |
| c2c row batches | 0.830 | 0.966 | 0.965 |
| c2c planes | 0.789 | 0.853 | 0.766 |
| c2c volumes | 0.776 | 0.790 | 0.865 |
| real rows | 1.000 | 0.881 | 0.850 |
| real row batches | 0.801 | 0.657 | 0.488 |
| real planes | 0.971 | 0.945 | 0.920 |
| real volumes | 1.088 | 1.013 | 1.111 |

For comparison, double's c2c rows against FFTW's double build (X56, 2026-11-18): Mac 0.737,
EPYC 0.788, Xeon 0.658, with 3, 6 and 1 of 193 beyond noise. Float's c2c rows stand as well on
the mean, but the x86 machines carry 29 of 201 beyond noise each.

Validity and load. The Mac stayed on AC power from 13:28 to 13:50 UTC with no sleep (pmset and
the power log); its battery read 99 % at the start and 90 % at the end with "AC attached; not
charging" throughout, macOS holding the charge below its limit, not running on the battery.
Every Mac round counted as loaded (other agents' work, 2.6 cores median). The Xeon's band is
wide: its rounds shared the node with this step's own test builds (Slurm 17394, 17397, 17400)
and other jobs. EPYC: DMOG node of Slurm 519947, shared, 5 of 31 rounds loaded.
Slurm jobs: Xeon FFTW float build 17383, timing 17384 (build), 17385, 17405; EPYC 519946
(build), 519947, 520153 (void), 520154.

## Verdict

**Done; no code change** (Q23: tuning only for the cases the maintainer names). Float stands at
0.82 (Mac), 0.80 (EPYC) and 0.77 (Xeon) of FFTW's float build over the standard catalogue,
inside DESIGN 1.3's double targets on the mean (≤ 0.85, ≤ 0.90, ≤ 0.85) but far outside its
tail rule (13–23 % of cases beyond 1.05, against 5 %). The cases above 1.10, by family, for the
maintainer to choose from:

1. **Tiny real transforms** (all three machines, most of the list): r2c and c2r of 2, 3, 8, 15 and
   16 points in batches of two (2–10 × FFTW), and the smoke tier's small real planes (5 × 7,
   8 × 6, 16 × 12, 16 × 16, 32 × 30, 32 × 32: 1.13–4.0). FFTW has a codelet per size; FFTN's
   real route pays a pairing pass and a call per row. Double was not measured here; X6 (real-input
   codelets) is the standing proposal.
2. **Real volumes** (all three): 16³, 32³, 48³ (Mac), 4 × 6 × 8 and 3 × 5 × 4, 1.13–2.2.
3. **Real planes with a long last axis**: 2048 × 36 (all three, 1.12–1.24), 2048² in place (Xeon,
   1.40 and 1.52), 1536-point c2r rows (Xeon 1.46).
4. **x86 c2c short-row batches**: 8, 12, 16, 32 and 64 points in batches of 16–4096 (EPYC 1.13–1.94,
   Xeon 1.15–1.64), 96 × 1024 and 96 × 64 (Xeon), 1024 × 1–16 (EPYC 1.16–1.22).
5. **x86 c2c rows of smooth mid lengths**: EPYC 80, 120, 160, 240, 384 (1.23–1.37); Xeon 20, 22,
   26, 160, 256, 131071 (1.19–1.51); 15 × 2 backward (both).
6. **x86 small c2c planes and volumes**: 5 × 7, 8 × 6, 16 × 12, 16 × 16 (1.22–1.64), 4 × 6 × 8,
   3 × 5 × 4, 16³ (EPYC).

The Mac has no c2c case above 1.10: its list is families 1–3.

The full lists (FFTN ÷ FFTW, median), from the runs above with the re-runs' decisions:

- **Mac (54):** r2c-3-b2 ip 9.90, op 9.61; c2r-3-b2 op 6.28, ip 4.92; c2r-15-b2 op 3.98, ip 3.74;
  r2c-15-b2 ip 3.83, op 3.66; c2r-2-b2 op 3.61, ip 3.58; c2r-5x7-b2 ip 3.58, op 3.31;
  r2c-5x7-b2 ip 3.01, op 3.00; r2c-2-b2 op 2.93, ip 2.89; r2c-32x32x32 2.20; c2r-32x32x32 2.16;
  c2r-8-b2 op 1.98, ip 1.79; r2c-8-b2 op 1.80, ip 1.69; r2c-32x32-b16 1.70; r2c-32x32 ip 1.70,
  op 1.68; c2r-32x32 op 1.59, ip 1.59; c2r-16-b2 ip 1.45, op 1.34; r2c-16x16x16 1.41;
  r2c-48x48x48 1.41; c2r-16x16x16 1.40; c2r-48x48x48 1.39; r2c-32x30-b2 op 1.37 (re-run);
  c2r-8x6-b2 op 1.30, ip 1.27; r2c-16-b2 ip 1.29, op 1.19; r2c-2048x36 op 1.24, ip 1.21;
  c2r-2048x36 op 1.23, ip 1.21 (re-run); c2r-16x12-b2 ip 1.24, op 1.23; r2c-16x12-b2 op 1.22,
  ip 1.22; c2r-32x30-b2 ip 1.18; r2c-16x16-b2 op 1.17, ip 1.17; r2c-32x30-b2 ip 1.17;
  r2c-8x6-b2 ip 1.16, op 1.15; c2r-16x16-b2 op 1.14, ip 1.13.
- **EPYC (81):** c2r-3-b2 op 4.71, ip 4.71; r2c-3-b2 op 4.25, ip 4.20; r2c-5x7-b2 op 3.96,
  ip 3.93; c2r-5x7-b2 ip 3.75, op 3.74; c2r-2-b2 ip 3.09, op 3.07; r2c-2-b2 op 2.66, ip 2.56;
  c2r-15-b2 ip 2.63, op 2.61; r2c-15-b2 ip 2.24, op 2.20; c2c-8-b16 1.94; c2r-8-b2 op 1.88,
  ip 1.83; r2c-8x6-b2 op 1.88, ip 1.88; c2r-8x6-b2 ip 1.86, op 1.57; r2c-8-b2 op 1.86, ip 1.82;
  c2r-16x16x16 1.72; r2c-16x12-b2 ip 1.69, op 1.67; r2c-16x16x16 1.69; c2r-16x12-b2 ip 1.68,
  op 1.68; c2c-8x6-b2 op 1.64, ip 1.63; c2c-16-b16 1.48; r2c-16-b2 ip 1.43, op 1.42;
  r2c-60-b2 op 1.41; c2c-32-b4096 1.41; c2c-32-b16 1.40; c2c-16x16-b2 ip 1.39, op 1.38;
  r2c-4x6x8 1.38; c2c-240 1.37; c2c-15-b2 bwd 1.37; c2c-80 1.35; c2c-32-b256 1.35;
  r2c-16x16-b2 op 1.34, ip 1.33; c2c-160 1.33; c2r-16-b2 ip 1.32, op 1.31; c2c-8-b4096 1.28;
  c2r-16x16-b2 op 1.28, ip 1.28; c2c-26 1.28; c2r-4x6x8 1.27; c2c-384 1.26; c2c-22 1.26;
  c2c-16x12-b2 ip 1.25, op 1.25; c2c-8-b256 1.25; c2c-120 1.23; c2c-5x7-b2 ip 1.23, op 1.22;
  c2r-60-b2 op 1.22; c2c-1024 1.22; c2c-1024-b3 1.21; c2r-3x5x4 1.21; c2c-1024-b2 1.20;
  c2c-1024-b16 1.20; r2c-2048x36 op 1.19; c2c-16x16x16 1.18; c2r-2048x36 op 1.18;
  c2c-1024 bwd 1.16; c2c-8-b2 ip 1.15; c2c-4x6x8 1.14; r2c-3x5x4 1.13; c2c-16-b4096 1.13
  (re-run); c2c-16-b3 op 1.13, ip 1.13; c2r-2048x36 ip 1.12.
- **Xeon (94):** c2r-3-b2 ip 5.15, op 5.12; c2r-2-b2 ip 4.20, op 4.08; r2c-3-b2 op 4.13, ip 4.00;
  r2c-2-b2 ip 3.10, op 3.05; c2r-15-b2 op 2.51, ip 2.51; r2c-5x7-b2 op 2.50, ip 2.49;
  c2r-5x7-b2 op 2.39, ip 2.38; r2c-15-b2 op 2.25, ip 2.22; c2r-8x6-b2 op 2.23, ip 2.21;
  c2c-3x5x4 2.12; c2r-8-b2 op 2.11, ip 2.06; c2r-3x5x4 2.11; r2c-8x6-b2 op 2.10, ip 2.05;
  c2r-16x12-b2 ip 2.08, op 2.04; r2c-8-b2 op 2.04, ip 2.02; c2r-16x16x16 2.01; r2c-16x12-b2
  ip 1.99, op 1.98; r2c-16x16x16 1.90; c2r-16-b2 ip 1.85, op 1.82; c2r-16x16-b2 op 1.83, ip 1.81;
  c2c-4x6x8 1.82; r2c-16-b2 ip 1.81, op 1.77; c2c-15-b2 bwd 1.68; r2c-16x16-b2 op 1.67, ip 1.66;
  c2c-8-b16 1.64; r2c-4x6x8 1.57; c2r-4x6x8 1.56; c2c-8-b2 op 1.54, ip 1.48; c2r-2048x2048 ip
  1.52; c2c-22 1.51; c2c-96-b1024 1.48; c2c-8-b2 bwd 1.47; c2c-5x7-b2 ip 1.46, op 1.45;
  c2r-1536 1.46; c2c-12-b1024 1.45; c2c-16-b3 op 1.43, ip 1.36; c2c-64-b4096 1.43; c2c-160 1.40;
  c2c-12-b64 1.40; r2c-2048x2048 ip 1.40; c2c-20 1.38; c2c-26 1.37; c2c-16x12-b2 op 1.36,
  ip 1.35; c2c-16x16-b2 ip 1.34, op 1.31; c2c-8x6-b2 ip 1.34, op 1.33; c2r-32x32x32 1.30;
  c2r-16 1.28; r2c-16 1.28; c2c-32-b4096 1.28; r2c-32x32x32 1.26; c2c-12-b3 ip 1.25, op 1.25;
  r2c-3x5x4 1.25; c2c-8-b256 1.24; r2c-2048x2048 1.23; r2c-32x32 op 1.23, ip 1.22;
  c2r-2048x36 op 1.21, ip 1.15; c2c-256 1.19; c2c-131071 1.19; r2c-32x32-b16 1.19; c2r-32x32
  op 1.19, ip 1.16; c2c-8-b3 ip 1.17, op 1.17; r2c-2048x36 op 1.17 (re-run); c2c-96-b64
  1.17; c2c-64-b16 1.16 (re-run); c2c-8-b4096 1.15.

Cases are out of place and one transform unless marked (`ip` in place, `-bN` a batch of N,
`bwd` backward). The lists are where FFTN loses most, not a measure of the double tuning's reach:
rules measured in double (cache-footprint limits, batch-leaf tables) apply to float unchanged.
Lesson 1.14 in [LESSONS.md](../LESSONS.md).
