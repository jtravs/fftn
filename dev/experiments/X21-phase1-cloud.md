# X21: phase 1 exit cloud checks (recent Intel, Zen 5, Neoverse V2)

- **Status:** done (check passed; three follow-ups)
- **Owner:** phase1-cloud session, 2026-11-15
- **Phase / task:** phase 1 exit ([PLAN.md](../PLAN.md), phase 1 "Exit"; DESIGN.md 1.3)

## Question

Is anything badly off on the machine families FFTN has not been timed on (a recent Intel
Xeon with MKL, AMD Zen 5 with AVX-512, an AArch64 Neoverse), and does the instruction-set
preference (`avx512` over `avx2`) hold off Cascade Lake? Tuning these machines is phase 2
and 6 work; this check decides only whether something needs a closer look before then.

## Hypothesis

FFTN ÷ FFTW lands near the local x86 figures (Xeon Gold 6240 about 1.16 over the tracking
c2c rows, EPYC 7763 about 1.35, X1) on C4 and C4D, and near the Mac's on Axion. Without
Cascade Lake's 512-bit clock licences, `avx512` beats `avx2` on both precisions on the
recent Intel and on Zen 5 (lesson 2.3 found float 1.11 slower on Cascade Lake).

## Change under test

None: competitor baselines at main 6844ad0 (X1 merge). Arms: `A=fftw` (MEASURE, fresh
wisdom), `B=fftn` (the default target), `C=mkl` (C4 only), `D=fftn+ISA=avx2` (x86) or
`D=fftn+ISA=neon` (Axion, when FCMA is present). Supplementary: `A=fftn+ISA=avx2
B=fftn` (x86) or `A=fftn+ISA=neon B=fftn` (Axion) over c2c rows in both precisions (a catalogue of 40 cases: 16–65536, 60, 1000, 97,
batches 1 and 64; the tracking tier has double only).

## Method

- Cases: the standard catalogue's tracking tier (double, 61 cases), default round budget
  (an hour; a run stops earlier once every case is resolved), one thread, hot mode.
- Machines: Google Cloud Spot VMs in us-central1, 16 vCPUs each: `c4-standard-16` (Intel),
  `c4d-standard-16` (AMD Zen 5), `c4a-standard-16` (Google Axion, Neoverse V2). Ubuntu
  24.04, GCC 14, CMake 3.28, Ninja; FFTW 3.3.11 from source with its SIMD and FMA codelets
  (x86: SSE2, AVX, AVX2, AVX-512, FMA; AArch64: NEON, FMA), double and single; oneMKL from
  the `mkl-devel` wheel on C4. FFTN Release; gate stages `build bench` before timing.
  Ledger: [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md).
- Metric: per case the median paired ratio over rounds; weighted geomean per arm; per-group
  geomeans (smooth c2c rows, batches/in-place/backward, primes and awkward lengths, long
  rows of 2^18 and more, real, planes and volumes).
- Decision rule, written before running: "badly off" is a test or output failure, or a
  group geomean of FFTN ÷ FFTW more than 0.15 above the same group on the nearest local
  machine (C4 against the Xeon, C4D against the EPYC, Axion against the Mac). Such a group
  opens a follow-up; otherwise the check passes. `avx512` against `avx2`: a geomean of
  avx2 ÷ avx512 below 0.97 in either precision on either machine is recorded as a
  `best_target` question for phase 2's cost model.

## Results

Machines (Google Cloud Spot, us-central1-a, 16 vCPUs, Ubuntu 24.04, kernel 7.0.0-1017-gcp,
GCC 14.2.0, CMake 3.28.3, Ninja 1.11.1; source 6844ad0 exported with `git archive`):

| Machine | CPU (model string) | Cores | L2 / L3 | `fftn_platform_dump` | vPMU |
|---|---|---|---|---|---|
| C4 `c4-standard-16` | INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz (Emerald Rapids; AMX, AVX-512 FP16) | 8 × 2 SMT | 2 MiB per core / 260 MiB | avx2, fma, avx512f/dq/bw/vl; best_target avx512 | standard: `perf stat` counts cycles |
| C4D `c4d-standard-16` | AMD EPYC 9B45 (Zen 5) | 8 × 2 SMT | 1 MiB per core / 32 MiB | the same; best_target avx512 | not offered for C4D (API refuses the flag) |
| C4A `c4a-standard-16` | Neoverse-V2 (Google Axion) | 16 | 2 MiB per core / 80 MiB | neon 1, fcma 1, pauth 1, sve_bits 128; best_target neon_fcma | standard: `perf stat` counts cycles |

Gate stages `build bench` (Release, `-Werror`, every kernel variant the CPU has, then the
fbench smoke with the competitors) passed everywhere: 1610 tests on C4 and C4D (scalar,
avx2, avx512), 1611 on C4A (scalar, neon, neon_fcma). Competitors: FFTW 3.3.11 built on each
machine (`fftw-3.3.11-sse2-avx-avx2-avx2_128-avx512` on x86, `fftw-3.3.11-neon` on Axion);
oneMKL from the `mkl-devel` 2026.2.1 wheel, which reports itself as 2026.0.2 build 20260915
on the AVX-512 + AMX code path.

Tracking runs (bands from the A/A control; no round loaded; the C4 and C4D runs shared their
first four minutes, 00:46–00:50 UTC, with a supplementary run started too early by the
controller, which fbench recorded as one other busy core and absorbed in the band):

| Machine | Run | Rounds | Band | FFTN ÷ FFTW weighted | FFTN `avx2` (or `neon`) ÷ FFTW | MKL ÷ FFTW | FFTN ÷ MKL |
|---|---|---|---|---|---|---|---|
| C4 | 20261115-004454-df45 | 80 | ±1.00 % | 1.266 [1.263, 1.270] | 1.457 | 0.951 | 1.332 |
| C4D | 20261115-004121-e653 | 80 | ±1.00 % | 1.257 [1.254, 1.260] | 1.410 | — | — |
| C4A | 20261115-013952-9897 | 22 | ±1.00 % | 0.757 [0.755, 0.758] | 0.883 | — | — |

FFTN ÷ FFTW by group (geometric mean of the per-case ratios; range in brackets). Smooth rows
are single out-of-place forward 7-smooth rows below 2^18 points; the second group is the
batches, in-place and backward c2c rows; primes and awkward lengths have a factor above 7;
long rows are 2^18–2^22:

| Group (cases) | C4 | C4D | C4A | C4: FFTN ÷ MKL |
|---|---|---|---|---|
| all 35 c2c rows | 1.166 | 1.120 | 0.810 | 1.188 |
| smooth rows (16) | 1.316 [0.94–2.58] | 1.334 [0.74–2.95] | 0.775 [0.50–3.05] | 1.228 |
| batches, in place, backward (8) | 1.260 [0.88–3.17] | 1.355 [0.87–4.69] | 0.805 [0.53–3.27] | 1.423 |
| primes and awkward (8) | 0.866 [0.51–1.72] | 0.682 [0.41–1.56] | 0.994 [0.56–2.82] | 1.000 |
| long rows (3) | 1.100 [0.81–1.57] | 0.996 [0.86–1.12] | 0.606 [0.54–0.68] | 0.971 |
| real (11) | 1.167 [1.00–1.38] | 1.039 [0.88–1.38] | 0.707 [0.58–0.87] | 1.105 |
| planes and volumes (15) | 1.628 [0.94–2.32] | 1.895 [1.04–5.02] | 0.678 [0.44–0.88] | 1.998 |

The worst cases are the same on all three machines and are known open work: rows of 8 and
16 points (2.0–3.0) and 4096 rows of 8 (3.2–4.7), the fixed cost per call (X20); a row of
17 (1.6–2.8). On x86 also 60 and 120 points (1.46–1.72), 100000 (1.61–1.84), and c2c planes
and volumes (phase 4; on C4D 2.8–5.0 for 256², 1024², 480 × 640 and 64³, against 1.6–2.0
on C4).

`avx2` ÷ `avx512` (FFTN's own default against the forced AVX2 target, same runs): tracking
1.151 [1.148, 1.155] on C4 and 1.122 [1.119, 1.124] on C4D, `avx512` ahead in every group
but planes (1.02, 1.00). On Axion `neon` ÷ `neon_fcma` 1.167, slower in all 61 cases.

Supplementary runs, `avx512` ÷ `avx2` (or `neon_fcma` ÷ `neon`) over c2c rows in both
precisions, 40 cases (16, 64, 256, 1024, 4096, 16384, 65536, 60, 1000, 97; batches 1 and
64), geometric means:

| Machine | Run | Band | double b1 | double b64 | float b1 | float b64 | per-length extremes |
|---|---|---|---|---|---|---|---|
| C4 | 20261115-013950-b0e1 (80 rounds) | ±1.00 % | 0.828 | 0.856 | 0.952 | 0.993 | float: 1024 0.52, 60 2.92–3.08, 64 1.21–1.23 |
| C4D | 20261115-011736-8034 (80 rounds) | ±1.00 % | 0.868 | 0.873 | 1.033 | 1.074 | float: 1024 0.66, 60 3.03–3.13, 16384–65536 1.28–1.63 |
| C4A | 20261115-015056-55ee (6 rounds) | ±1.00 % | 0.837 | 0.841 | 0.849 | 0.849 | 0.78–0.94, every case faster |

Local machines, for comparison (FFTN ÷ FFTW over the same 35 tracking c2c rows, FFTW at
MEASURE): Xeon Gold 6240 `avx512` 1.162 and EPYC 7763 1.354 at the X1 merge
([X1](X1-split-lowering.md), runs 20261114-183454-f63e and -182358-c6ff), Mac M5 Max 1.05
([X19](X19-generator-rotations.md), run 20261114-152124-de44); MKL ÷ FFTW on the Xeon 0.95
([X18](X18-x86-kernels.md)), 0.951 on C4. Per-group figures for the local machines are not
in the tracked documents (their stores are on the remote machines), so the group-level rule
was applied to the aggregate.

## Verdict

The check passes: every test and every output check passed on all three machines, and no
machine is badly off against its nearest local one. C4 (Emerald Rapids) matches the Xeon
Gold 6240 (1.166 against 1.162 over the c2c rows), C4D (Zen 5) is well ahead of the EPYC
7763 (1.120 against 1.354), and Axion is FFTN's best machine so far (0.757 weighted, 0.81 over
the c2c rows, against 1.05 on the Mac): DESIGN 1.3's Neoverse target of 0.90 against FFTW is
already met there, although FFTW's NEON codelets on Linux AArch64 may be weaker than on
Apple silicon (not measured here). The phase 1 targets themselves are not met on x86 (1.17
and 1.12 over the c2c rows against 0.85; 1.19 of MKL against the interim 1.10), which is the
known x86 gap. Nothing was changed; follow-ups:

1. `best_target` per precision and length. `avx512` wins in double on both recent x86
   machines (0.83–0.87) and in float on Emerald Rapids (0.95–0.99), but loses in float on
   Zen 5 (1.03–1.07, so avx2 ÷ avx512 is 0.93–0.97 and the decision rule records it) and
   loses badly at 60 points in float on both (2.9–3.1 times avx2's time; 1.1–1.3 in double).
   This is phase 2's cost model pricing `avx2` and `avx512` side by side; the 60-point float
   case is worth a look first (a tail-tile or leaf choice).
2. c2c planes and volumes on Zen 5 (2.8–5.0 against FFTW, against 1.6–2.0 on C4): a
   phase 4 input; check whether the column pass's strided accesses or the AVX-512 tiles are
   the cause before tuning planes for Zen 5.
3. Single rows of 65536 (C4D 1.63, C4 1.31) and 100000 points (1.61–1.84) on x86, and
   262144 on C4 (1.57; MKL is slow there too, 1.50 of FFTW).

Lesson 2.3 in [LESSONS.md](../LESSONS.md) updated (AVX-512 off Cascade Lake), lesson 6.12 added
(the oneMKL wheel's library names). Cost USD 1.36 of the USD 20 cap
([ledger](cloud-ledger-2026-11.md)).
