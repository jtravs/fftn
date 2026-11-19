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

- **Arm T** (`loop::joins`, X41's T1 on the current work graph): the narrower last block of a
  lanes column phase joins the whole blocks' phase instead of running alone afterwards. Every
  real plane of even length has such a block (n/2 + 1 columns is odd). X41 rejected T1 against
  a 0.95 bar on the EPYC (0.957–0.968 over the planes it changes); the diagnosis below finds this
  phase at 7–12 % of the 8-lane time of real 256²–512² on the Xeon. One-thread programs are
  unchanged (the work graph is read by teams only); bits unchanged (the same lines, in any
  order). **Decision rule, written before its judging runs**, over the standard catalogue's
  planes and volumes (`^(c2c|r2c|c2r)-[0-9]+x[0-9]+(x[0-9]+)?-b[0-9]+-(op|ip)$`), T against
  its base at 8 lanes through `--serial` (GCC; Xeon and EPYC). Xeon: adopt T if (1) the geometric
  mean of T ÷ base over the real planes of one transform with at least 32768 complex points (the
  ones a team splits there and T changes) is at most 0.97, and (2) no case of the set is slower
  than the base by more than the larger of 2 % and the run's A/A band (a case just past it on a
  loaded machine is re-run once before it decides; a case whose work graph T does not change
  does not decide, AGENTS.md section 6). EPYC: the same with (1) at most 1.00 over the real
  planes it splits. Mac, 6 lanes, Homebrew Clang: the same with (1) at most 1.00 over the real
  planes that split; a machine that fails keeps the old graph (the flag is set per profile).

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
