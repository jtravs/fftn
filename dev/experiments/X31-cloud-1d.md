# X31: one-dimensional double on recent server chips (Zen 5, Granite Rapids, Axion)

- **Status:** running
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
  outside the range of the same group on the local machines of its architecture (x86: EPYC
  0.94-group figures and Xeon; AArch64: the Mac), or a case among a machine's worst ten that
  is above 1.2 there and below 1.0 on every local machine of its architecture, is a
  divergence and is listed for lead (d).

## Results

Pending.

## Verdict

Pending.
