# X48: the phase 1 exit check on cloud cores again (Zen 5, Granite Rapids, Axion), `main` 8c18dd2

- **Status:** done (measurement only. Every mean target met: ÷ FFTW Zen 5 0.69, Granite Rapids 0.76, Axion 0.71; Granite Rapids ÷ MKL 0.73. Tail rule beyond noise met on Zen 5 (3.1 %) and Axion (4.7 %), not on Granite Rapids (÷ FFTW 9.3 %, ÷ MKL 5.7 %); see Verdict)
- **Owner:** exit-1118 session, 2026-11-18
- **Phase / task:** the phase 1 exit ([PLAN.md](../PLAN.md) "Status", "Phase 1 exit check
  (2026-11-18)"); repeats [X45](X45-cloud-exit.md) after X42, X46, X47, the `avx512` branch
  padding and task 4.2

## Question

Does `main` 8c18dd2 meet the phase 1 exit (DESIGN.md 1.3: the geometric-mean target, and at
most 5 % of cases slower than 1.05 of the reference beyond noise) on the three cloud cores
X45 measured at 509d123, and which cases still fail there?

## Hypothesis

The means stay met. Since X45, X42 (radix-25 codelets) moved `avx2`/`avx512` double
500–25000, X46's depth-first chain changed only the Xeon 6240 and Apple profiles (Granite
Rapids' 2 MiB L2 and Zen 5 keep their plans), X47 changed two-pass chains on two lines, and
the `avx512` branch padding (pull request 62) targeted the Xeon's small rows. So Zen 5 and
Granite Rapids should gain on 500–25000 (3125, 15625, 25000) and otherwise hold; Axion
should not move; the tail rule beyond noise (DESIGN 1.3 since 2026-11-17: X45 had Zen 5 11,
Granite Rapids 22 / MKL 17, Axion 9) stays failed on Granite Rapids' long rows and in-place
65536.

## Change under test

None (a measurement). Arms, fbench Release builds of `main` 8c18dd2 on each VM, as X45:

- `N` = `fftn:8c18dd2`, with fbench's identical-arm control `N~`;
- `F` = FFTW 3.3.11 built on the VM with its SIMD and FMA codelets (x86-64: SSE2, AVX,
  AVX2, AVX-512; AArch64: NEON), MEASURE, fresh wisdom;
- `M` = oneMKL from the `mkl-devel` wheel (Granite Rapids only).

## Method

As X45 in every respect: `--catalogue standard --tier full --filter '^c2c-[0-9]+-b'` (201
one-dimensional c2c rows, 193 with every arm); Google Cloud Spot VMs in us-central1-a,
`c4d-standard-16` (Zen 5) and `c4-standard-16` with minimum platform Granite Rapids, both
with one thread per core (8 cores), and `c4a-standard-16` (Axion, 16 cores); Ubuntu 24.04,
GCC 14.2; fbench pinned to CPU 2 with `taskset`, one thread, hot, median statistic, 6–12
rounds. Scored by the exit-check rule of DESIGN 1.3: geometric mean of FFTN ÷ reference
over the 193 double rows; a case counts against the tail when its whole interval lies above
1.05 × (1 + the run's A/A band); the plain count above 1.05 is given too.

- **Decision rule, written before running** (a measurement): a machine passes against a
  reference when the geometric mean is at or below its DESIGN 1.3 target (Zen 5 and Granite
  Rapids ÷ FFTW 0.85, Granite Rapids ÷ MKL 1.00, Axion ÷ FFTW 0.90) and at most 5 % of the
  cases (9 of 193) count beyond noise.
- **Budget:** at most USD 6 (campaign cap USD 20, USD 4.09 spent before), ledger in
  [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md), section X48.

## Results

### Setup and runs

VMs as X45 (ledger section X48): `fftn-x48-c4d` AMD EPYC 9B45, 8 cores, L2 1 MiB, L3 32 MiB;
`fftn-x48-c4` Intel Xeon 6985P-C, 8 cores, L2 2 MiB, L3 504 MiB, oneMKL wheel 2026.2.1
(reports 2026.0.2, AVX-512 path); `fftn-x48-c4a` Neoverse V2, 16 cores, L2 2 MiB, L3 80 MiB.
Kernel 7.0.0-1017-gcp, GCC 14.2.0; FFTN `avx512` / `neon_fcma`. Sources from a bundle of
`origin/main` 8c18dd2. fbench's smoke passed on all three; no VM was preempted. Each run
reached 12 rounds, all valid:

| Machine | Run | Arms | Rounds under load | Band (A/A) |
|---|---|---|---|---|
| Zen 5 | 20261118-003054-e82e | N F | 1 of 12 | ±1.38 % |
| Granite Rapids | 20261118-003118-2144 | N F M | 1 of 12 | ±1.87 % |
| Axion | 20261118-002820-6df2 | N F | 0 of 12 | ±1.00 % |

The stores also hold one-case build runs that are not results (-002637-445e, -002640-fceb,
-002640-8ac0). The runs are in the main checkout's `bench-out/runs`.

### The exit-check metric

The 193 double rows every arm timed, FFTN's time ÷ the reference's (X45 at 509d123 in
brackets):

| Machine | Reference | Geomean (target) | Beyond noise | Plain > 1.05 |
|---|---|---|---|---|
| Zen 5 | FFTW | **0.686** (0.85) [0.695] | **6, 3.1 %** [11, 5.7 %] | 15 [15] |
| Granite Rapids | FFTW | **0.758** (0.85) [0.770] | **18, 9.3 %** [22, 11.4 %] | 23 [25] |
| Granite Rapids | MKL | **0.734** (1.00) [0.747] | **11, 5.7 %** [17, 8.8 %] | 16 [22] |
| Axion | FFTW | **0.714** (0.90) [0.714] | **9, 4.7 %** [9, 4.7 %] | 10 [9] |

Cases beyond noise (FFTN ÷ reference, point estimate):

- **Zen 5 ÷ FFTW (6):** 6 points 1.13, 4096 1.13, 2048 1.13, 2²⁴ 1.12, 2²³ 1.08,
  12 × 64 1.07. Above 1.05 but within noise: 100000 1.13, 46080 1.10, 2 × 3 1.07, rows of
  2, 4, 8 points 1.06, 12 × 1024 1.06, 4096 × 16 and 2 × 3 in place 1.05.
- **Granite Rapids ÷ FFTW (18):** 46080 1.62, 65536 × 3 in place 1.51, 65536 backward 1.51,
  65536 in place 1.47, 65536 1.38, 100000 1.37, 131071 1.29, 131072 1.29, 65536 × 3 1.27,
  160 1.26, 262144 1.26, 32768 1.20, 240 1.18, 2²² 1.12, 1024 (× 1, backward, × 3, × 16)
  1.10.
- **Granite Rapids ÷ MKL (11):** 65536 in place 1.57, 46080 1.46, 131071 1.36, 65536 × 3 in
  place 1.36, 999983 1.29, 65536 × 3 1.27, 65536 backward 1.25, 65536 1.23, 1024 × 3 in place
  1.18, 1024 × 16 1.10, 1024 × 3 1.08.
- **Axion ÷ FFTW (9):** 2 × 3 in place 1.46, 4 1.42, 2 × 3 1.41, 2 1.40, 6 1.34, 3 × 3 1.24,
  8 1.19, 20 1.08, 65537 1.08. Next: 64 × 4096 1.10 within noise.

### Against X45 (`main` 509d123, same VM types, a day apart)

Ratios compare within each run; times across VMs only beyond a few per cent (LESSONS 1.12).

- **Zen 5:** ÷ FFTW 0.695 → 0.686, FFTN's time 0.98 of X45's (FFTW 0.99). Gains: 100 and
  100 × 64 (0.82–0.87 of their time), 3125, 500, 15625, 25000 (0.81–0.89, X42), 32768 0.86,
  16 points 0.93. Lost ratio: 31, 37, 45, 61 points (FFTN 1.11–1.14 of its X45 time, FFTW
  unchanged) and 65536 (0.85 → 0.92). Out of the tail since X45 (now above 1.05 within
  noise): 100000, rows of 2 and 4 points, 2 × 3 in and out of place; still in it: 2048, 4096,
  2²³, 2²⁴, 6 points, 12 × 64.
- **Granite Rapids:** ÷ FFTW 0.770 → 0.758, ÷ MKL 0.747 → 0.734; FFTN's time 1.02 of X45's,
  FFTW's 1.03, so this VM was a little slower. Gains: 3125 (FFTN 0.58 of its time; ÷ MKL 1.19
  → 0.67), 15625 0.78, 100 and 100 × 64 0.86, 10000, 500 and 25000 0.87–0.93 (X42). Lost:
  16384 (FFTN 1.16 of its time; ÷ FFTW 0.83 → 0.99, ÷ MKL 1.02 → 1.12) and 45 points 1.10,
  although no merged change targets either on this profile (X46 does not apply to a 2 MiB
  L2): a same-run A/B of 509d123 against 8c18dd2 is needed before calling it a regression.
  The long rows and in-place 65536 are where they were (46080 1.60 → 1.62, 65536 in place 1.50
  → 1.47).
- **Axion:** level (0.714 both; FFTN and FFTW 1.008 of their X45 times). 2²⁰ and 10⁶ lost
  ratio (FFTN 1.14–1.16 of its time); 262144 gained (0.67 → 0.60). The tail is the same nine
  cases: the fixed cost of calls of 2–8 points.

## Verdict

Measurement only; no library change. On the cloud cores `main` 8c18dd2 meets every mean
target with room (÷ FFTW Zen 5 0.69, Granite Rapids 0.76, Axion 0.71; Granite Rapids ÷ MKL
0.73), and the tail rule beyond noise is now met on **Zen 5** (6 of 193, 3.1 %; 11 at X45)
and still on **Axion** (9, 4.7 %, at the limit). **Granite Rapids fails** it against FFTW
(18, 9.3 %) and against MKL (11, 5.7 %, two cases over the limit). What stands in the way:

1. **Granite Rapids' long rows and in-place 65536** (12 of the 18 against FFTW, 8 of the 11
   against MKL): 46080, 65536 in every form, 100000, 131072, 262144, 32768, 2²². X46's
   depth-first rule is not enabled for its 2 MiB L2 (X46 "Left open"); measuring it there is
   the obvious next arm (X51).
2. **Granite Rapids' mid rows** 160, 240 and 1024 (single, backward, × 3, × 16) at
   1.10–1.26 of FFTW; 1024 × 3 in place and × 16 against MKL.
3. **Primes against MKL on Granite Rapids:** 131071 1.36, 999983 1.29 (X50's subject).
4. **Fixed cost of tiny calls** (X49's subject): all nine of Axion's cases, and 6 points on
   Zen 5.
5. **Zen 5's 2048, 4096, 2²³, 2²⁴** at 1.08–1.13.

Two cross-VM changes need a same-run check before they count: Granite Rapids 16384 (FFTN
1.16 of its X45 time) and Zen 5's 31–61-point primes (1.11–1.14). Cost USD 0.63 of the
USD 6 allowed ([ledger](cloud-ledger-2026-11.md)).
