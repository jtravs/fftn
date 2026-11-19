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

- **Arm R** (on top of T): a split task's phase of lines is claimed in runs of its fair share
  (chunks / cap, at least one) instead of X52's guided runs (a share of what is left, at most
  left / (2 × cap)). A member's lines are then adjacent and a phase has about cap runs. The
  diagnosis shows why it matters: neighbouring columns of a real plane share cache lines in most
  rows (a row of n/2 + 1 complex values is an odd multiple of 16 bytes), so each boundary between
  two members' runs bounces lines; FFTN's team takes 9000–11 000 loads of lines modified in
  another core's cache per real 512² plane where oneMKL's static split takes 340 (r2c) to 4700
  (c2r), and 20 000 against 1000 at r2c 2048 × 36. Stage phases keep the guided runs. Bits
  unchanged (only who runs which chunk). **Decision rule, written before its judging runs,**
  R against T at 8 lanes through `--serial` over the same case set. Xeon: adopt R if (1) the
  geometric mean of R ÷ T over the real planes that split (as for T) is at most 0.97, (2) the
  geometric mean over the other planes and volumes that split is at most 1.00, and (3) no case
  of the set is slower than T by more than the larger of 2 % and the run's A/A band (re-run once
  as for T; a case whose schedule R cannot change does not decide). EPYC and Mac (6 lanes): (2)
  and (3) over every plane and volume that splits. A machine that fails keeps the guided runs
  (gated by profile).

- **Arm R2** (on top of T, replacing R after R's run on the Mac, below): fair runs only for a
  phase whose neighbouring lines write the same cache lines: a column phase whose block is not
  a whole number of cache lines wide or whose rows are not a whole number of cache lines long,
  which every real plane is and which columns alone through the chain are (`loop::shares_lines`,
  `graph_phase::fair_runs`, `chunk_run`). Rows and c2c blocks on whole lines keep X52's guided
  runs. Bits unchanged. **Decision rule, written before its judging runs,** R2 against T at 8
  lanes through `--serial` (GCC) and at 6 lanes on the Mac (Homebrew Clang), over the same case
  set; the cases R2 can change are those a team splits with such a phase. Xeon: adopt R2 if (1) the
  geometric mean of R2 ÷ T over the real planes of one transform with at least 32768 complex
  points is at most 0.97, and (2) no case it can change is slower than T by more than the larger
  of 2 % and the run's A/A band (a case just past it on a loaded machine is re-run once). EPYC and
  Mac: (1) at most 1.00 over the cases it can change, and (2). A machine that fails keeps the
  guided runs (by profile).

- **Arm R3** (on top of T, replacing R2 after R2's run on the Mac, below): R2 for planes only;
  volumes keep the guided runs. R2's rule, unchanged, R3 against T, over the same case set (the
  cases it can change are planes).

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

### Diagnosis (Xeon 6240, `main` 0508037)

Baseline, fbench run 20261119-003606-6cac (Slurm 16555; FFTN at 1, 2, 4, 8 lanes, oneMKL
serial and at 2, 4, 8 threads, 28 real planes; 8 rounds, all loaded, 17–32 other cores busy,
band ±11 %, so this run ranks cases only): FFTN's 8 lanes ÷ oneMKL's 8 threads 1.1–2.0 from
256² up (c2r 256² 1.98, c2r 36 × 2048 1.76, c2r 1024² 1.58, r2c 2048² 1.10). X52's "2.5–3.9" was
measured before its arm S; since S the planes below 32768 complex points are not split.

**Placement.** Slurm's allocations on this machine bind nothing: `-c 8 --hint=nomultithread`
leaves a job free on all 72 CPUs (`Cpus_allowed_list: 0-71`), so every "8 lanes" run here, X43's
and X52's included, is 8 unpinned threads on a two-socket machine with other users' work on it,
not 8 cores of one socket. The diagnosis therefore runs both ways.

Harness runs (Slurm 16557 and 16563; three and four repeats, each on the idlest socket, others'
work on the machine throughout, so times of the large planes, which depend on the shared L3 and
memory, move by up to 2 between repeats; medians over repeats). Pinned: one thread per core on
one socket (FFTN by the harness, oneMKL by `OMP_PLACES`).

| | one thread ÷ oneMKL | 8 lanes ÷ oneMKL | FFTN speed-up | oneMKL speed-up | unpinned 8 ÷ oneMKL |
|---|---|---|---|---|---|
| r2c 128² | 1.33 | 2.17 | 0.73 (not split) | 1.18 | 0.98 |
| c2r 128² | 1.32 | 3.52 | 1.00 (not split) | 2.68 | 1.80 |
| r2c 256² | 1.09 | 1.23 | 2.35 | 2.65 | 0.79 |
| c2r 256² | 1.51 | 1.64 | 3.33 | 3.61 | 1.76 |
| r2c 512² | 1.08 | 1.36 | 3.25 | 4.09 | 1.64 |
| c2r 512² | 1.41 | 1.69 | 4.48 | 5.37 | 1.90 |
| r2c 1024² | 1.40 | 1.62 | 6.92 | 7.99 | 0.96 |
| c2r 1024² | 1.66 | 1.89 | 7.96 | 9.05 | 1.50 |
| r2c 2048² | 1.09 | 1.09 | 7.48 | 7.48 | 1.00 |
| c2r 2048² | 1.23 | 1.40 | 6.19 | 7.05 | 1.31 |
| r2c 2048 × 36 | 0.91 | 1.80 | 1.37 | 2.71 | 1.27 |
| c2r 2048 × 36 | 1.07 | 0.97 | 2.37 | 2.15 | 0.92 |
| r2c 36 × 2048 | 0.93 | 1.11 | 2.43 | 2.89 | 1.04 |
| c2r 36 × 2048 | 1.18 | 1.52 | 2.80 | 3.60 | 1.58 |

Both effects are real: **oneMKL is faster alone** (c2r 1.2–1.7, r2c about 1.1, 1024² 1.4), and
**FFTN scales somewhat worse** (a further 1.0–1.3 at 8 lanes, 1.8 at r2c 2048 × 36).

Where a team's time goes (pinned, 8 lanes; phase = the rows, the whole column blocks, the
narrower last block):

- **The engine is not the cost.** First chunk 1–3 µs after the call, phase hand-offs 0.2–1 µs,
  claims and bookkeeping about 3 µs per member per transform at 256².
- **The narrower last column block runs alone**: 8 µs of r2c 256²'s 67 µs (12 %), 14.5 of 196 µs
  at 512² (7 %), 44 of 1124 µs at 1024² (4 %), while seven members wait.
- **The work grows when shared**: summed over members, the column phase of r2c 256² takes 4.1
  times its one-thread time, 512² 2.5, 36 × 2048 4.2, 2048 × 36 6.9; the rows 1.2–1.7. At 1024²
  and above, where one thread already misses L2, about 1.0–1.3.
- **Cross-core traffic** explains the growth. Loads that find their line modified in another
  core's cache, per transform at 8 lanes, FFTN against oneMKL: r2c 256² 1007 / 59, r2c 512²
  11 804 / 1940, c2r 512² 9792 / 4968, r2c 2048 × 36 19 944 / 964, c2r 2048 × 36 8856 / 657. A
  real row is n/2 + 1 complex values long, an odd multiple of 16 bytes, so neighbouring columns
  share cache lines in three rows of four, and at 2048 × 36 (columns through the chain, one column
  per chunk) in every row. FFTN's guided runs give a phase of few chunks about one run per chunk,
  so nearly every block borders another member's; a static split as oneMKL's has seven such
  borders.
- **c2r carries a third array.** Out of place FFTN keeps the input and puts the half spectrum in
  a workspace block (in + block + out, 1.5 MiB at 256² against a 1 MiB L2): 15 100 L2 demand
  misses per transform at one thread against r2c's 1500 and oneMKL's c2r 6800. oneMKL keeps
  its input too (checked: no element changes), so it is not a contract difference.
- **Unpinned**, FFTN loses more than oneMKL (8 lanes pinned ÷ unpinned at c2r 512²: FFTN 0.51,
  oneMKL 0.64; 4–11 context switches per transform at 512²–1024² for FFTN's waiting members,
  none for libgomp's spinning ones).

### Arms T and R on the Mac (6 lanes)

Run 20261119-015223-410e (M5 Max, Homebrew Clang, `neon_fcma`; A = `main` 0508037, T 79669a6, R
f14f6c1; 8 rounds, all loaded with about 6 other cores busy, band ±1.4 %), the 132 planes and
volumes. T ÷ A: 0.996 over the 44 real planes that split, 0.991 over the 37 other planes and
volumes that split (c2c 100² 0.94); worst real plane r2c 2048 × 36 out of place 1.021. R ÷ T:
0.978 over the real planes (2048 × 36 0.82–0.88, 1024² and 512² 0.90–0.92, but 256² 1.07–1.12),
**1.053 over the other planes and volumes** (c2c 480 × 640 1.19–1.20, c2c 256² 1.17–1.18, 48³
1.19–1.20): fair runs of whole phases cost balance where neighbouring lines do not share cache
lines, on a machine with other work on it. **R fails its rule on the Mac**; hence arm R2.

Run 20261119-020343-b186 (R2 8080398 against T; 10 rounds, all loaded, band ±1.2 %): R2 ÷ T
0.968 over the 44 real planes that split (r2c 2048 × 36 0.84–0.89, c2r 0.90–0.94, 1024² 0.88–0.92,
512² 0.91–0.92), 1.009 over the 37 other planes and volumes, but real 48³ 1.20–1.22 (out of place,
the columns of a volume through the chain, 29 chunks a phase). **R2 fails its rule (2) on the
Mac**; R3 keeps the volumes on the guided runs.

## Verdict
