# X84: float's standing against FFTW's float build

- **Status:** running
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

**Amendment 1** (2026-11-26 15:10, before the EPYC's deciding re-run). The EPYC's near-miss
re-run (run 20261126-150451-f8fa, Slurm 520153) took six rounds in 18 s and its intervals span
0.33–0.97 of FFTW ÷ FFTN for every case: too few rounds to decide anything. It is void as
deciding; the deciding re-run repeats it with `--min-rounds 24`. The Mac's and the Xeon's
re-runs (12 rounds each, intervals of a few per cent) stand.

## Results

(to come)

## Verdict

(to come)
