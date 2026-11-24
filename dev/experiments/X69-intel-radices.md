# X69: radices for 1024, 160 and 240 on recent Intel (the 2 MiB-L2 key under `avx512`)

- **Status:** planned (decision rules complete; the cloud run decides)
- **Owner:** x69 session, 2026-11-24
- **Phase / task:** 1 (left from the exit: Emerald Rapids' failing rows); follows X64 and X57

## Question

On the 2 MiB-L2 profile under `avx512` in double (Sapphire, Emerald and Granite Rapids), should
one-axis c2c rows of 1024, 160 and 240 points take two-pass radices (32 × 32, 8 × 20, 12 × 20)
while the plan's rows stay in the L2, instead of the rules' three passes (8 × 8 × 16, 4 × 4 × 10,
4 × 4 × 15)? These are the rows that still fail Emerald Rapids' exit check against FFTW and
oneMKL (X64: 1024 one row 1.17–1.21 of FFTW, 1024 in place 3–16 rows 1.15–1.25 of oneMKL, 160
1.14 of FFTW).

## Hypothesis

X64's diagnosis (A3 on Sapphire and Granite Rapids, not judged; runs 20261124-110848-9267,
-110857-ce33, -110906-aff2 on Sapphire, 20261124-123201-fbc0, -123210-005a, -123219-b8af on
Granite; bands ±1.00–2.01 %): forced 32 × 32 took 0.73–0.82 of the rules' time for 1024 at 1, 3
and 16 rows, out of place and in place, both directions; at 256 rows (8 MiB of rows and output)
0.99 on both, at 1024 rows 1.12 on Sapphire and 0.99 on Granite. 8 × 20 took 0.84 and 0.80 of
4 × 4 × 10's time for one row of 160, 12 × 20 0.81 and 0.79 of 4 × 4 × 15's for 240. As on the
EPYC (X57), a two-pass plan pays while the batch's data stay in a near cache and stops paying
once every one of its 20–32 streams misses at once; here the near cache is the 2 MiB private L2
(16 rows of 1024 out of place, 512 KiB, win; 256 rows, 8 MiB, tie). The same holds for 160 and
240 by the same mechanism, so their batches take the same limit.

## Change under test

X57's two-pass fit overrides, keyed to the 2 MiB-L2 profile:

- `planner/rules.cpp`: a fit table for the 2 MiB-L2 key, `avx512` double only: 160 = 8 × 20,
  240 = 12 × 20, 1024 = 32 × 32. None of the three has a plain override on `avx512` double, so
  beyond the limit `fitting_radices` gives the rules' sequence, the plan of today.
- `machine_profile::fit_large_l2` (new), `fit_row_bytes` and `fit_in_place_bytes`, set in
  `set_grid_sizes`' `large_l2` branch only: the plan takes that key's fit table instead of the
  target's (X57's `avx2` table). Limits: **rows and output at most the L2 (2 MiB: 64 rows of
  1024, 409 of 160, 273 of 240), in place rows at most half the L2 (1 MiB: the same row counts).**
  The limit lies between the last measured win (512 KiB) and the first tie (8 MiB); in place it
  counts the same rows as out of place because X57's in-place crossing on the EPYC fell at about
  the same row count as the out-of-place one (1024–1100 rows against 768–1024).
- Why this mechanism: the choice must see the batch (a plain override cannot, X54), and it must
  reach only the 2 MiB-L2 profile. Zen 4 and Zen 5 run `avx512` and already get `fit_row_bytes`
  from the large-L3 branch (for X57's `avx2` table), and so does Ice Lake-SP (Google Cloud N2's
  large L3 per CPU): an `avx512` table under `target_profile::fit_overrides` would reach them
  unmeasured. A machine-profile flag that selects the key's table keeps them out: their flag is
  false and their `avx512` fit table stays empty. The Xeon 6240's class (1 MiB L2, small L3) has
  `fit_row_bytes` = 0 and does not change; nor do the EPYC class, Apple and every `avx2` plan
  (Intel's client cores with a 2 MiB L2 have no `avx512`, and the key's table is empty there).

So by the mechanism the change can move only one-axis c2c double rows of 160, 240 and 1024 points,
contiguous on both sides, on x86-64 with a private L2 of 2 MiB or more under `avx512`, within the
limit. Planes, volumes, real rows, float and other lengths keep their plans.

Arms (one binary, campaign branch `exp/cloud-x69`: this change plus campaign-only fbench options
registered as `FFTN_X_X69` in `dev/KNOBS.toml`, not for merging):

- **O** = `FIT_ROW_BYTES=0`: the fit off, the plans before this change. Listed first, with the
  A/A control O~.
- **N** = the change as it is.
- **W** (informative, decides nothing) = `FIT_ROW_BYTES=1073741824+FIT_IN_PLACE_BYTES=1073741824`:
  every row of the three lengths on the two-pass radices, to see where the limit lies.
- **F** = FFTW 3.3.11 (MEASURE, `PLAN_NOTE=1`) and **M** = oneMKL: reported, not judged.

## Method

- Machines: Google Cloud Spot VMs, one per family, `--threads-per-core=1`, GCC 14.2 Release,
  `avx512`, double, one thread (taskset on one CPU), hot cases, statistic median: Sapphire Rapids
  `c3-standard-8`, Granite Rapids `c4-standard-16 --min-cpu-platform="Intel Granite Rapids"`,
  Emerald Rapids `c4-standard-16 --min-cpu-platform="Intel Emerald Rapids"`. Each job checks the
  CPU model first and stops on the wrong family. Zen 4 (c3d) and Zen 5 (c4d) are not run: the key
  does not reach them (their `fit_large_l2` is false; the profile tests hold it).
- **Run J** (judging), catalogue `x69`, arms O, N, W, F, M, 10–20 rounds. Cases (44, double):

  | group | cases | N's plan (expected) |
  |---|---|---|
  | 1024 out of place, forward and backward | 1, 3, 16, 32, 64 rows | 32 × 32 (moved) |
  | | 128, 256, 1024 rows | 8 × 8 × 16 (framing) |
  | 1024 in place, forward and backward | 1, 3, 16, 32, 64 rows | 32 × 32 (moved) |
  | | 128, 256 rows | 8 × 8 × 16 (framing) |
  | 160 | 1, 16, 256 rows forward, 1 row backward, 1 and 16 rows in place | 8 × 20 (moved) |
  | | 1024 rows (5 MiB) | 4 × 4 × 10 (framing) |
  | 240 | 1, 16, 256 rows forward, 1 row backward, 1 and 16 rows in place | 12 × 20 (moved) |
  | | 1024 rows (7.5 MiB) | 4 × 4 × 15 (framing) |

  32 and 64 rows of 1024 (1–2 MiB), and 160 and 240 batched, are held out: X64 measured none of
  them. The run's plan notes are the record: **the moved cases are those whose notes differ
  between N and O**; a case whose notes agree runs the same plan in the same binary and does not
  decide, whatever this table says.
- **Run C** (notes check), the standard catalogue's tracking tier, arms O and N, one round, no
  control: outside one-axis c2c rows of 160, 240 and 1024, the notes of O and N must agree on
  every case.
- Metric: N ÷ O per moved case, the median over valid rounds with its 95 % interval; over the
  moved cases, the per-round geometric mean, its median and 95 % interval (`screen_score.py`, as
  X64). The band is run J's A/A control (O against O~). The scripts:
  `screen_score.py RUN.jsonl N O --report REPORT --rule adopt|reach`.

### Decision rules (written before any judging run)

- **Adopt** (Sapphire and Granite Rapids, X64's adopt rule): N is adopted on the 2 MiB-L2 key if,
  on each of the two, over run J's moved cases the geometric mean of N ÷ O is below 1 − band with
  its interval's upper end below 1, and no moved case is slower than 1 + max(band, 3 %) (its
  whole interval above).
- **Reach** (Emerald Rapids, X64's reach rule): Emerald keeps N if no moved case is slower than
  1 + max(band, 3 %) (whole interval) and the moved cases' geometric mean is at most 1 + band.
- **Moves nothing else**: on each VM, run C shows no case outside the one-axis c2c rows of 160,
  240 and 1024 whose notes differ between O and N, and run J's framing cases (past the limit)
  keep O's plan in N. A failure here is a bug to fix before adoption, not a verdict.
- **Near misses** (AGENTS section 6, 2026-11-23): a criterion missed by less than the run's band
  gets one re-run of run J (on the same VM if it is still up, else on a new VM of the family),
  and the re-run decides.
- **Narrowing** (AGENTS section 6, 2026-11-23): if moved cases lose, N may be narrowed before
  adoption, without a new experiment: a length whose moved cases lose leaves the table; cases near
  the limit that lose (32 or 64 rows of 1024, 256 rows of 160 or 240) lower the limit to the
  largest measured footprint whose moved cases pass the rule among themselves (out of place: 512
  KiB, 1 MiB, 1.25 MiB, 1.875 MiB; in place: 256 KiB, 512 KiB). A deciding run on the final code
  is then declared in this file before it starts, and must show every excluded case on O's plan
  and no case the narrowed change still moves slower than 1 + band.
- **Families**: the three share one key. If Emerald fails the reach rule after any re-run, N is
  not adopted on the key (a split needs a signal the profile does not have today), and the
  maintainer decides. If Sapphire or Granite fails the adopt rule, N is not adopted.
- W ÷ O on the framing cases (128–1024 rows of 1024, 1024 rows of 160 and 240) places the limit
  for a follow-up; it decides nothing here. F and M give FFTN ÷ FFTW and ÷ oneMKL on the moved
  cases before (O) and after (N), for the exit-check rows.
- Adopting means `set_grid_sizes` keeps the three fields in its `large_l2` branch as committed on
  the implementation branch: N's plans in run J.

### Not measured

Threaded plans of these rows take the same radices (the choice sees the whole batch's bytes,
not a member's share); float keeps its plans. The Xeon 6240 is not run: its profile has no
`fit_row_bytes` and its plans cannot move (a plan diff on the Xeon checks this before the run).

## Results

## Verdict
