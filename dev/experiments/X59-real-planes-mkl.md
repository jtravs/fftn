# X59: threaded real planes on the Xeon 6240 against oneMKL

- **Status:** running
- **Owner:** X59 agent, 2026-11-19 (night round 2)
- **Phase / task:** 4.1 (threaded planes)

## Question

X52 left oneMKL's real planes at 8 threads 2.5–3.9 times faster than FFTN's 8 lanes on the Xeon
6240. Where does FFTN's time go at 1, 2, 4 and 8 lanes, is oneMKL faster alone or does FFTN scale
worse, and which change (if any) to the splitting, the column blocks, the placement of work or
the real-plane program closes the gap without losing on the EPYC 7763 and the M5 Max?

## Hypothesis

None is held yet: the diagnosis below comes first and chooses the arms. Candidates from X43, X52
and LESSONS 5.10–5.13: how real rows are split and scheduled, column blocks for teams, keeping a
member's rows and columns in its L2, fused post-processing, the real-plane program itself.

## Change under test

Arms are written here, each with its decision rule, after the diagnosis and before its judging
run.

## Method

- Cases: the standard catalogue's real planes (`^(r2c|c2r)-[0-9]+x[0-9]+-b1-(op|ip)$`).
- Machines: Xeon 6240 (GCC 14, `avx512`, oneMKL 2026.1 with its GNU OpenMP layer) for the
  diagnosis and screens; judging through `fbench_remote.py submit --serial`; EPYC 7763 (GCC 16,
  `avx2`) and M5 Max (Homebrew Clang, `neon_fcma`) for the collateral checks.
- Diagnosis harness (not in the repository; a private build of the library with per-member,
  per-phase timestamps): untraced time per transform, hardware counters over all threads, and
  per phase the span, the gap to the phase before, the work summed over members and the members
  that took part.

## Results

## Verdict
