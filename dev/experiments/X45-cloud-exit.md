# X45: the phase 1 exit check on recent cloud cores (Zen 5, Granite Rapids, Axion)

- **Status:** done (measurement only. Every mean target met: ÷ FFTW Zen 5 0.70, Granite Rapids 0.77, Axion 0.71; Granite Rapids ÷ MKL 0.75. The tail rule met on Axion only, at its edge (9 of 193 cases, 4.7 %); Zen 5 7.8 %, Granite Rapids 13.0 % (MKL 11.4 %); see Verdict)
- **Owner:** x45-cloud session, 2026-11-17
- **Phase / task:** the phase 1 exit ([PLAN.md](../PLAN.md) "Status", "Phase 1 exit check")

## Question

Does `main` 509d123 meet the phase 1 exit (DESIGN.md 1.3: the geometric mean target, and
at most 5 % of cases slower than the reference by more than 5 %) on the three recent server
cores it was never tuned on, as it is judged on the Mac, the EPYC 7763 and the Xeon 6240?
And how far has `main` moved on them since the last cloud runs (X31, X31b, X35)?

## Hypothesis

The means are met with room, as on the local machines (Mac 0.78, EPYC 0.84, Xeon 0.76 of
FFTW on 2026-11-16, before X36–X39): X31 measured the one-dimensional double cases at 0.80
(Zen 5), 0.88 (Granite Rapids) and 0.74 (Axion), and X35 took Granite Rapids' long rows
0.59–0.98 of their time. The tail rule fails, as on every local machine, on the same
causes: primes outside Rader's range, short single rows and tiny batches, batches of 16
points on x86, and Granite Rapids' long mixed rows (100000, 46080) and in-place 65536.

## Change under test

None (a measurement). Arms, all fbench Release builds of `main` 509d123:

- `N` = `fftn:509d123`, with fbench's identical-arm control `N~`;
- `F` = FFTW 3.3.11 built on the machine with its SIMD and FMA codelets (x86-64: SSE2, AVX,
  AVX2, AVX-512; AArch64: NEON), MEASURE, fresh wisdom;
- `M` = oneMKL, the `mkl-devel` wheel (Granite Rapids only).

## Method

- **Cases:** the exit-check set of PLAN "Status": `--catalogue standard --tier full
  --filter '^c2c-[0-9]+-b'` (201 one-dimensional c2c rows; the metric takes the double rows
  every arm timed).
- **Machines:** Google Cloud Spot VMs in us-central1-a, set up as in X31 and X35:
  `c4d-standard-16` (AMD Zen 5) and `c4-standard-16` with minimum platform Granite Rapids
  (Intel Xeon 6), each with `--threads-per-core=1` (8 whole cores), and `c4a-standard-16`
  (Google Axion, Neoverse V2, 16 cores). Ubuntu 24.04, GCC 14, CMake, Ninja; FFTN Release,
  its targets chosen at run time (`avx512`, `neon_fcma`). One thread; fbench pinned to CPU 2
  with `taskset`; hot mode, median statistic; 6–12 rounds within 90 minutes (the local exit
  runs took 12).
- **Metric:** per case FFTN's time ÷ the reference's (each arm's median over rounds of its
  process medians, as in PLAN "Status"); the geometric mean over the cases; the share of
  cases above 1.05; the worst ten.
- **Decision rule, written before running** (a measurement: the exit-check metric). A
  machine passes the exit against a reference when the geometric mean is at or below its
  DESIGN.md 1.3 target (Zen 5 and Granite Rapids ÷ FFTW 0.85, Granite Rapids ÷ MKL 1.10 for
  the phase 1 interim, 1.00 at the end; Axion ÷ FFTW 0.90) and at most 5 % of the cases are
  above 1.05.
- **Comparison with the last cloud runs:** on the cases the earlier tracking-tier runs share
  with this set, FFTN ÷ FFTW then and now (each within its own run), and FFTN's time now ÷
  then (across VMs of the same type, so only differences beyond a few per cent count).
- **Budget:** at most USD 6 (campaign cap USD 20, USD 3.51 spent), ledger in
  [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md), section X45.

## Results

### Setup and runs

- **VMs** (ledger in [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md), section X45): Google
  Cloud Spot, us-central1-a, Ubuntu 24.04, kernel 7.0.0-1017-gcp, GCC 14.2.0; sources from a
  bundle of `origin/main` 509d123. None was preempted.
  - `fftn-x45-c4d`: `c4d-standard-16`, AMD EPYC 9B45 (Zen 5), 8 cores, SMT off; L2 1 MiB,
    L3 32 MiB for CPUs 0–7; FFTN `avx512`, FFTW `sse2-avx-avx2-avx2_128-avx512`.
  - `fftn-x45-c4`: `c4-standard-16`, Intel Xeon 6985P-C (Granite Rapids), 8 cores, SMT off;
    L2 2 MiB, L3 504 MiB for CPUs 0–7; FFTN `avx512`, FFTW as above; oneMKL wheel 2026.2.1
    (reports 2026.0.2, AVX-512 path).
  - `fftn-x45-c4a`: `c4a-standard-16`, Google Axion (Neoverse V2), 16 cores; L2 2 MiB,
    L3 80 MiB; FFTN `neon_fcma`, FFTW `neon`.
- **Runs.** fbench's smoke passed on all three. Each run reached the 12-round cap with every
  round valid and none under load:

| Machine | Run | Arms | Rounds | Band (A/A) |
|---|---|---|---|---|
| Zen 5 | 20261117-144317-7119 | N F | 12 | ±2.16 % |
| Granite Rapids | 20261117-144359-0a8f | N F M | 12 | ±1.10 % |
| Axion | 20261117-144321-6196 | N F | 12 | ±1.00 % |

  The stores also hold one-case build runs that are not results (Zen 5 -143827-2659 and
  -144310-5edf, Granite Rapids -143846-9b4e and -144350-7ea6, Axion -143816-bec8 and
  -144312-1ba8; the first of each pair stopped on a script error after building). The runs
  are in the main checkout's `bench-out/runs`.

### The exit-check metric

The 193 double rows that every arm timed (of the 201; as on the local machines), FFTN's
time ÷ the reference's:

| Machine | Reference | Geomean (target) | Cases > 1.05 | Share (limit 5 %) |
|---|---|---|---|---|
| Zen 5 | FFTW | **0.695** (0.85) | 15 | **7.8 %** |
| Granite Rapids | FFTW | **0.770** (0.85) | 25 | **13.0 %** |
| Granite Rapids | MKL | **0.747** (1.10 interim, 1.00 end) | 22 | **11.4 %** |
| Axion | FFTW | **0.714** (0.90) | 9 | **4.7 %** |

Axion passes at the edge: 5 % of 193 allows 9 cases, and the next case, 64 × 2 backward,
is at 1.048 (paired interval of FFTW ÷ FFTN [0.950, 0.957]), so a re-run could tip it.

Worst ten, FFTN ÷ reference:

- **Zen 5 ÷ FFTW:** 2²⁴ 1.17, 100000 1.17, 2 1.17, 6 1.16, 4 1.14, 2²³ 1.13, 2 × 3 in place
  1.12, 2 × 3 1.11, 4096 1.11, 2048 1.09. The other five over 1.05: 12 × 64 1.09, 8 1.08,
  46080 1.07, 240 1.06, 12 × 1024 1.06.
- **Granite Rapids ÷ FFTW:** 46080 1.60, 65536 × 3 in place 1.55, 65536 1.53, 65536 in place
  1.50, 100000 1.39, 65536 backward 1.35, 65536 × 3 1.33, 131072 1.33, 262144 1.33,
  131071 1.29. Then 160 1.26, 32768 1.22, 240 1.14, 1024 forward and backward 1.14, 100 1.12,
  2²² 1.10, 1024 × 3 and × 16 1.10, 512 1.10, 100 × 64 1.09, 256 × 1024 1.08, 384 1.06,
  12 1.06, 2 × 3 in place 1.06.
- **Granite Rapids ÷ MKL:** 65536 in place 1.58, 46080 1.48, 65536 × 3 in place 1.44, 131071
  1.38, 999983 1.32, 65536 × 3 1.32, 65536 backward 1.30, 65536 1.30, 256 × 1024 1.19, 3125
  1.19. Then 1024 × 3 in place 1.17, 1024 × 16 1.11, 32768 1.11, 1024 × 3 1.09, 2²³ 1.09,
  4096 × 1024 1.09, 4096 × 256 1.08, 1000 1.07, 15625 1.06, 128 × 4096 1.06, 1024 in place
  1.05, 4096 × 16 1.05.
- **Axion ÷ FFTW:** 2 × 3 in place 1.49, 4 1.44, 2 × 3 1.43, 2 1.41, 6 1.35, 3 × 3 1.25,
  8 1.19, 65537 1.09, 20 1.08 (all nine over 1.05), then 64 × 2 backward 1.048.

By the exit check's causes (PLAN "Status"): on Zen 5 the tail is short single rows and tiny
batches (cause 2: 6 cases), long rows (2²³, 2²⁴, 100000, 46080; 2048 and 4096), two short
batches of 12 and 240; on Granite Rapids it is the long rows of cause 5 and the in-place 65536 of
cause 6 (11 cases), mid rows 100–1024 (7), batches (1024 × 3 and × 16, 100 × 64, 256 × 1024),
primes (131071) and short rows (12, 2 × 3 in place); against MKL also
primes (131071, 999983, 3125, 15625) and batches of 1024–4096 points; on Axion it is
cause 2 alone, short single rows and tiny batches of 2–8 points, plus 65537 (Rader) and 20.

### Against the last cloud runs

On the 35 c2c rows the tracking tier shares with this set: FFTN ÷ FFTW then and now (each
within its own run), and FFTN's and FFTW's times now ÷ then (across VMs of one type):

| Machine | Earlier run (FFTN arm) | ÷ FFTW then → now | FFTN time now ÷ then | FFTW time now ÷ then |
|---|---|---|---|---|
| Zen 5 | X31 20261116-115045-ed43 (`main` 4ebcee5) | 0.755 → 0.679 | 0.903 | 1.005 |
| Zen 5 | X31b 20261116-151405-7d87 (`main` 5b02f07) | 0.700 → 0.679 | 0.967 | 0.998 |
| Granite Rapids | X31 20261116-115322-2631 (`main` 4ebcee5) | 0.822 → 0.784 (÷ MKL 0.812 → 0.771) | 0.950 | 0.996 |
| Granite Rapids | X35 20261116-192413-6457 (arm B9, as merged) | 0.778 → 0.784 (÷ MKL 0.774 → 0.771) | 0.939 | 0.933 |
| Axion | X31 20261116-114945-a77e (`main` 4ebcee5) | 0.709 → 0.703 | 0.980 | 0.989 |
| Axion | X35 20261116-171150-dd31 (`main` 2eea943) | 0.703 → 0.703 | 0.986 | 0.987 |

- **Zen 5** moved most: 16 points 0.71 of its X31b time (÷ FFTW 0.94 → 0.68), 8 × 4096 0.79
  (1.18 → 0.92), 4093 0.75 (0.75 → 0.60, X36), 65537 0.92.
- **Granite Rapids:** this VM ran both libraries about 6 % faster than X35's, so only the
  ratios compare. They are level with X35 (0.778 → 0.784): 16 points 0.98 → 0.67 and 8 × 4096
  0.95 → 0.88 gained (X37); 65536 1.24 → 1.53 and 2²² 0.76 → 1.10 lost because FFTW was
  faster in this run (0.80 and 0.46 of its X35 time; FFTN 0.99 and 0.66), which looks like
  FFTW's MEASURE choosing different plans, not FFTN moving.
- **Axion:** level overall; 4093 1.11 → 0.80 (FFTN 0.72 of its time, X36), 8 1.24 → 1.19.
  Single rows of 16 and 17 points took 1.08 of their X35 time with FFTW unchanged (÷ FFTW
  0.86 → 0.93, 0.44 → 0.48); X37's lighter AArch64 leaf path merged in between. This is a
  cross-VM comparison and needs a same-run A/B before it counts.

## Verdict

Measurement only; no library change. On recent cloud cores `main` 509d123 meets every mean
target with room, better than on the local machines' last exit check (Mac 0.78, EPYC 0.84,
Xeon 0.76 at 304a8c5): ÷ FFTW Zen 5 0.70, Granite Rapids 0.77, Axion 0.71; Granite Rapids
÷ MKL 0.75, already below the end target of 1.00. The 5 % tail rule is met only on Axion,
by the narrowest margin (9 of 193 allowed and taken). Zen 5 has 15 cases over (7.8 %),
Granite Rapids 25 against FFTW (13.0 %) and 22 against MKL (11.4 %).

What stands between each machine and the exit:

1. **Granite Rapids' long rows and in-place 65536** (11 of its 25): 65536 in every form
   1.33–1.55, 46080 1.60, 100000 1.39, 131072 and 262144 1.33, 32768 1.22. X35's profile
   took the worst of them down (100000 2.16 → 1.39 since X31) but 46080 (1.60 in X31) and
   65536 (1.40 in X31) did not move. This is the cloud form of the Xeon's causes 5 and 6.
2. **Short single rows and tiny batches (cause 2)** on all three, and the whole tail on
   Axion: 2–8 points 1.19–1.49 there (with GCC 14; X31 already had 8 points at 1.24 and found
   FFTN's time per point twice the Mac's), 1.08–1.17 on Zen 5.
3. **Granite Rapids' mid rows 100–1024 and some batches** (160 1.26, 240 1.14, 1024
   1.14, 512 1.10), as on the EPYC before X38; against MKL, primes (131071 1.38, 999983 1.32,
   3125 1.19) and batches of 1024–4096 points (1.05–1.19).
4. **Zen 5's longest powers of two** (2²³ 1.13, 2²⁴ 1.17; not in the tracking tier) and
   100000 1.17.

Since the last cloud runs `main` gained 3–10 % on Zen 5 (the shared rows at 0.90 of X31's
time, 0.97 of X31b's), held its ratios on Granite Rapids and Axion, and 4093 gained on Zen 5 and
Axion (X36; Axion 1.11 → 0.80 of FFTW). The one possible step back, Axion's 16 and 17 points at 1.08 of their
X35 time, is across VMs and is left for a same-run check (LESSONS 1.12 added: across VMs
compare ratios, not times). Cost USD 0.58 of the USD 6 allowed
([ledger](cloud-ledger-2026-11.md)).
