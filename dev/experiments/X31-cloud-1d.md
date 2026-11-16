# X31: one-dimensional double on recent server chips (Zen 5, Granite Rapids, Axion)

- **Status:** done (one-dimensional double FFTN ÷ FFTW: Zen 5 0.80, Granite Rapids 0.88 (MKL 0.85), Axion 0.74; divergences listed under Verdict)
- **Owner:** x31-cloud session, 2026-11-16
- **Phase / task:** phase 1 one-dimensional double push ([PLAN.md](../PLAN.md) "Status",
  lead (d))

## Question

After the one-dimensional double push, where does FFTN stand against FFTW (and MKL on Intel)
on recent server cores it has not been tuned on, and does any of them diverge from the
EPYC 7763, the Xeon Gold 6240 and the Mac strongly enough that per-core rules (lead (d))
become worth their cost?

## Hypothesis

The push's gain carries over: FFTN ÷ FFTW over the 46 one-dimensional double cases lands
near the local machines' 0.81–0.94 (X21 measured Emerald Rapids, Zen 5 and Axion before the
push at 1.17, 1.12 and 0.81 over the tracking c2c rows), and the per-group pattern is the
local one: long powers of two and long real rows worst, primes and prime products best.

## Change under test

None (a measurement). Arms: `N=fftn:<main>` (the baseline A, with fbench's identical-arm
control `N~`), `O=fftn:ad6acbe` (`main` before the push, so the push's gain shows in the
same run), `F=fftw` (3.3.11 built on the machine with its SIMD and FMA codelets, MEASURE,
fresh wisdom), `M=mkl` (Intel only, the oneMKL `mkl-devel` wheel). `F` and `M` are built
from the `N` commit's fbench.

## Method

- Cases: the standard catalogue's tracking tier (61 cases, double); the analysis takes the
  46 one-dimensional cases and groups them as PLAN "Status" does (powers of two ≤ 64,
  128–4096, ≥ 16384; mixed radix; primes; prime products; batches; in place; r2c and c2r).
- Machines: Google Cloud Spot VMs, us-central1, 16 vCPUs on x86 with
  `--threads-per-core=1` (8 whole cores), 16 cores on Axion: `c4d-standard-16` (AMD Zen 5),
  `c4-standard-16` with minimum platform Granite Rapids (Intel Xeon 6), `c4a-standard-16`
  (Google Axion, Neoverse V2). Ubuntu 24.04, GCC 14, CMake, Ninja; FFTN Release (the build a
  user gets). One thread; fbench and its processes pinned to core 2 with `taskset`.
- fbench: `--tier tracking --min-rounds 8 --max-rounds 40`, at most 45 minutes of rounds;
  hot mode, median statistic. Builds are made first by a one-case run, unpinned.
- Metric: per case B's time ÷ A's (each arm's median over rounds of its process medians,
  as in PLAN "Status"); per-group geometric means; the paired ratio and its interval from
  fbench's report for single cases.
- Divergence rule, written before running: a group whose FFTN ÷ FFTW lies more than 0.15
  outside the range of the same group on the local machines of its architecture (x86: the
  EPYC 7763 and the Xeon Gold 6240; AArch64: the Mac M5 Max; their 2026-11-16 runs in PLAN
  "Status"), or a case among a machine's worst ten that is above 1.2 there and below 1.0
  on every local machine of its architecture, is a divergence and is listed for lead (d).

## Results

Machines (Google Cloud Spot, us-central1-a, Ubuntu 24.04, kernel 7.0.0-1017-gcp, GCC
14.2.0; sources from a bundle of `origin/main`; ledger in
[cloud-ledger-2026-11.md](cloud-ledger-2026-11.md)):

| Machine | CPU | Cores used | L2 / L3 | FFTN target | FFTW build |
|---|---|---|---|---|---|
| C4D `c4d-standard-16` | AMD EPYC 9B45 (Zen 5) | 8, SMT off | 1 MiB per core / 32 MiB | `avx512` | `fftw-3.3.11-sse2-avx-avx2-avx2_128-avx512` |
| C4 `c4-standard-16` | Intel Xeon 6985P-C (Granite Rapids) | 8, SMT off | 2 MiB per core / 504 MiB | `avx512` | the same; oneMKL 2026.2.1 wheel (reports 2026.0.2, AVX-512 + AMX path) |
| C4A `c4a-standard-16` | Google Axion (Neoverse V2) | 16 | 2 MiB per core / 80 MiB | `neon_fcma` | `fftw-3.3.11-neon` |

`N` is `main` 4ebcee5. It differs from the 2025d51 timed in PLAN "Status" only in the
sanitizer build flags and documents, so the local runs below are its numbers too. fbench's
smoke (every backend's self-test and the smoke tier with the guard) passed on all three
machines. Each run reached the 40-round cap with no round under load:

| Machine | Run | Rounds | Band (A/A) | FFTW ÷ FFTN, all 61 cases (weighted) | MKL ÷ FFTN |
|---|---|---|---|---|---|
| C4D (Zen 5) | 20261116-115045-ed43 | 40 | ±1.24 % | 1.050 [1.047, 1.053] | — |
| C4 (Granite Rapids) | 20261116-115322-2631 | 40 | ±1.00 % | 1.006 [1.004, 1.008] | 0.997 [0.996, 0.999] |
| C4A (Axion) | 20261116-114945-a77e | 40 | ±1.00 % | 1.340 [1.338, 1.343] | — |

(The stores also hold one-case runs 20261116-114107-1473, -114233-2397 and -114336-9464 that
only built the arms; they are not results.)

FFTN ÷ FFTW (geometric mean of each case's median time ratio, the convention of PLAN
"Status"), today's `main` and in brackets `main` before the push (ad6acbe), with the local
machines' 2026-11-16 figures for comparison:

| Group (cases) | Zen 5 | Granite Rapids | Axion | EPYC 7763 / Xeon 6240 / Mac (local) |
|---|---|---|---|---|
| powers of two ≤ 64 (3) | 0.87 (1.33) | 0.92 (1.41) | 1.02 (1.35) | 0.94 / 0.83 / 0.94 |
| 128–4096 (5) | 0.96 (1.12) | 1.04 (1.15) | 0.81 (0.83) | 1.04 / 0.94 / 0.84 |
| ≥ 16384 (5) | 1.12 (1.15) | 1.04 (1.07) | 0.75 (0.76) | 1.10 / 1.03 / 1.19 |
| mixed radix (7) | 0.79 (1.26) | 1.04 (1.28) | 0.62 (0.70) | 0.96 / 0.93 / 0.76 |
| primes (6) | 0.42 (0.78) | 0.43 (0.93) | 0.53 (1.10) | 0.62 / 0.36 / 0.48 |
| prime products (2) | 0.31 (0.66) | 0.35 (0.78) | 0.59 (0.91) | 0.45 / 0.36 / 0.47 |
| batches (5) | 0.87 (1.08) | 0.88 (1.07) | 0.83 (0.86) | 1.03 / 0.85 / 0.91 |
| in place (2) | 1.03 (1.06) | 1.26 (1.29) | 0.78 (0.78) | 1.08 / 0.99 / 0.93 |
| r2c and c2r (11) | 0.96 (1.03) | 1.07 (1.13) | 0.83 (0.84) | 1.13 / 1.04 / 1.00 |
| **one-dimensional double (46)** | **0.80 (1.05)** | **0.88 (1.12)** | **0.74 (0.86)** | 0.94 / 0.81 / 0.83 |
| planes and volumes (15) | 1.63 (1.91) | 1.48 (1.69) | 0.78 (0.83) | 1.63 / 1.27 / 1.23 |

Today's FFTN against the pre-push one over the 46 cases: Zen 5 0.76, Granite Rapids 0.78,
Axion 0.85 (local 0.78 / 0.77 / 0.85). On Granite Rapids against MKL: one-dimensional double
0.85 (1.09 before the push); by group 0.76, 0.92, 0.92, 1.02, 0.45, 0.47, 0.98, 1.32, 0.99
in the order above; planes and volumes 1.65 (Xeon 6240 against MKL: 0.85 and 1.68).

Worst ten one-dimensional cases, FFTN ÷ FFTW (local EPYC / Xeon or Mac in brackets):

- **Zen 5:** c2c 65536 1.50 (1.29 / 1.09), 65536 in place 1.42 (1.17 / 1.16), r2c 2²⁰ 1.39
  (1.44 / 1.19), c2r 4096 1.37 (1.12 / 0.97), 2²² 1.18 (1.08 / 0.86), 8 × 4096 1.17
  (0.96 / 1.00), 100000 1.17 (0.96 / 1.99), 4096 1.14 (0.99 / 0.89), c2r 2²⁰ 1.11
  (1.33 / 1.09), 46080 1.10 (1.08 / 1.29).
- **Granite Rapids:** 100000 2.16 (0.96 / 1.99), 46080 1.60 (1.08 / 1.29), c2r 65536 1.51
  (1.29 / 1.42), 65536 in place 1.50 (1.17 / 1.16), 65536 1.40 (1.29 / 1.09), r2c 65536
  1.34 (1.16 / 1.40), 262144 1.31 (1.14 / 1.24), r2c 65536 in place 1.13 (1.17 / 1.27),
  1024 forward and backward 1.13 (1.23 / 0.99; 1.20 / 1.00). Against MKL the worst are 65536
  in place 1.63, 100000 1.52, 46080 1.47 and the real 65536 rows 1.26–1.32.
- **Axion:** 8 1.24 (Mac 0.96), 4093 1.16 (1.41), r2c 65536 in place 1.12 (1.27), 65537
  1.10 (0.89), r2c 65536 1.10 (1.21), 64 × 256 1.04 (1.06), 128 × 4096 1.00 (0.90), 64 0.99
  (0.92), 16384 0.92 (1.33), 128 0.91 (0.76).

What the ratios hide on Axion: per point, FFTN is 1.4 times slower than on the Mac at 2²⁰
(9.3 against 6.6 ns) but FFTW is 2.6 times slower (14.3 against 5.6 ns); at 8 points FFTN
takes 0.97 ns per point against the Mac's 0.45 while FFTW takes 0.79 against 0.47.

## Verdict

Nothing is badly off, and the push carries over: one-dimensional double FFTN ÷ FFTW is 0.80
on Zen 5, 0.88 on Granite Rapids (0.85 of MKL) and 0.74 on Axion, inside or below the local
range (0.81–0.94), with the same pre-push gain (today's FFTN 0.76–0.85 of ad6acbe's time).
Against X21 on the same families before the push (FFTW ÷ FFTN weighted over all 61 cases
1.257 on Zen 5, 1.266 on Emerald Rapids, 0.757 on Axion), FFTN is now level with FFTW over
the whole tier on Zen 5 (FFTW 1.050 of FFTN's time) and Granite Rapids (1.006), and level
with MKL there (0.997).

Divergences by the rule written before running:

1. **Axion against the Mac, four groups in FFTN's favour:** long powers of two 0.75 (Mac
   1.19), in place 0.78 (0.93), r2c and c2r 0.83 (1.00), planes and volumes 0.78 (1.23).
   The absolute times say most of this is FFTW, not FFTN: FFTW on Linux NEON is far slower
   on long rows than on the Mac, while FFTN loses much less. So lead (a)'s long-row gap is
   mostly a gap to FFTW on Apple silicon; on a Neoverse FFTN is already 0.65–0.92 of FFTW
   at ≥ 16384. Judging long rows on the Mac alone overstates the problem on AArch64 Linux.
2. **Axion, 8 points (1.24; Mac 0.96):** the fixed cost per call is twice the Mac's per
   point while FFTW's is 1.7 times. GCC 14 against Apple Clang may be part of it (the Mac
   builds with Clang); worth a look on Graviton or Axion with Clang before any per-core rule.
3. **Granite Rapids, in place (1.26 against 0.99–1.08):** c2c 65536 in place 1.50 (out of
   place 1.40). With the Xeon's mixed rows (100000 2.16, 46080 1.60, worse than the Xeon
   6240's 1.99 and 1.29) this is lead (c): X29's chain rule keys on a Zen-like L3 and leaves
   Intel alone, and the newer Intel core pays more for it. 65536 rows lose on all three x86
   machines (1.09–1.50).

Zen 5 diverges in no group. Its worst cases are the EPYC 7763's (long powers of two, the
real 2²⁰ rows, 4096 points) at 0.1–0.25 worse (65536 1.50 against 1.29; c2r 4096 1.37
against 1.12; 4096 1.14 against 0.99, X30b's range), and it does not share the EPYC's
65537 loss (0.61 against 1.63: that one is Zen 3's, lead (e)). Its planes and volumes are as
bad as the EPYC's (64³ 5.5, 256² 4.6), a phase 4 input already noted in X21.

For lead (d): within one instruction set, cores diverge on the cases already open (long
rows, long mixed rows, in place 65536), by up to 0.3–0.6 in a ratio, not in kind; the one
new pattern is that the Mac's long-row deficit does not appear on a Neoverse. Rules by core
on x86 would first serve lead (c) (Intel's long mixed rows), which a rule on the cache
sizes the machine profile already reads may cover without a core field. No library change; LESSONS 2.9 added. Cost USD 0.75 of the
USD 20 cap ([ledger](cloud-ledger-2026-11.md)).
