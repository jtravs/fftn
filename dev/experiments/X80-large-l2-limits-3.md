# X80: no in-place limit, and a higher out-of-place one, for the 2 MiB-L2 key's two-pass rows

- **Status:** planned (decision rules below; the judging run follows their merge)
- **Owner:** x80 session, 2026-11-25
- **Phase / task:** 1 (follows X69 and X70)

## Question

X70 gave rows of 160, 240 and 1024 the two-pass radices on the 2 MiB-L2 key while rows and
output take at most four times the L2 (in place: rows within twice the L2; 1024 out of place:
within the L2). Should the in-place limit go, and the out-of-place limit of 160 and 240 rise to
64 times the L2? Rows of 1024 out of place keep the L2.

## Hypothesis

X70's informative arm W gave every row of the three lengths the two-pass radices (runs
20261124-220554-0d7c Sapphire, -220134-10a7 Granite, -222653-aa74 Emerald; bands ±1.00 %,
±1.16 %, ±1.39 %). W ÷ O on the cases past X70's limits, where O = N:

| Case | Rows (and output) | Sapphire | Granite | Emerald |
|---|---|---|---|---|
| 1024 in place, 512 rows | 8 MiB | 0.883–0.885 | 0.908 | 0.914–0.918 |
| 1024 in place, 1024 rows | 16 MiB | 0.891–0.894 | 0.912–0.913 | 0.919–0.922 |
| 1024 in place, 2048 rows | 32 MiB | 0.890–0.894 | 0.915–0.934 | 0.912–0.914 |
| 160 in place, 4096 and 16384 rows | 10 and 40 MiB | 0.874, 0.891 | 0.884, 0.880 | 0.891, 0.888 |
| 240 in place, 4096 and 16384 rows | 15 and 60 MiB | 0.823, 0.903 | 0.838, 0.841 | 0.870, 0.874 |
| 160 in place, 65536 rows | 160 MiB | 0.911 | 0.938 | 0.939 |
| 240 in place, 65536 rows | 240 MiB | 0.903 | 0.892 | 0.944 |
| 160 out of place, 4096 and 16384 rows | 20 and 80 MiB | 0.881, 0.892 | 0.895, 0.870 | 0.880, 0.889 |
| 240 out of place, 4096 and 16384 rows | 30 and 120 MiB | 0.849, 0.944 | 0.859, 0.873 | 0.847, 0.874 |
| 160 out of place, 65536 rows | 320 MiB | 0.892 | 1.021 | 0.853 |
| 240 out of place, 65536 rows | 480 MiB | 0.945 | 1.027 | 0.999 |
| 1024 out of place, 128 and 256 rows | 4 and 8 MiB | 0.974–0.975 | 0.993–1.002 | 0.943–0.954 |

Every interval of the in-place and of the 4096- and 16384-row cases lies below 1.

- **In place** the two-pass radices won at every size tried, on all three families: rows of 1024
  to 32 MiB (0.88–0.93), of 160 and 240 to 240 MiB (0.82–0.94), far past each VM's L2 and, at
  65536 rows, at sizes only DRAM or a large shared L3 holds. Past 8 MiB the gain shrinks little
  (1024: 0.88–0.92 at 8 MiB, 0.89–0.93 at 32 MiB; 160 and 240: 0.82–0.89 at 10–15 MiB,
  0.89–0.94 at 160–240 MiB) and nowhere nears 1. One explanation: a batch too large for the cache is
  read and written once per pass, so two passes move two thirds of three passes' data. Nothing
  measured supports any limit, so there is none: in place, every batch of the three lengths takes
  the two-pass radices on the key. The run tests batches twice as large as X70's largest (rows of
  1024 to 512 MiB, of 160 and 240 to 320 and 480 MiB).
- **160 and 240 out of place** won to 16384 rows (80 and 120 MiB, 0.85–0.94) and tied or lost at
  65536 rows on Granite (1.02–1.03) and for 240 on Emerald (1.00). The crossing lies between 120
  and 320 MiB on Granite. The limit is the smallest power-of-two multiple of the L2 holding the
  largest win measured (120 MiB): rows and output within **64 times the L2** (128 MiB on a 2 MiB
  L2: 26214 rows of 160, 17476 rows of 240). Held-out sizes past it (160 and 180 MiB, and X70's
  320 and 480 MiB) place the crossing with arm W.
- **1024 out of place** keeps X70's limit, the L2: past it the arm gained little (0.94–0.98) or
  tied (Granite), and lost at 32 MiB on Sapphire in X69 (1.29–1.36).

Why out of place stops gaining past about 128 MiB on Granite while in place does not is
unexplained; the change does not depend on the reason.

## Change under test

New limits for the key, in `set_grid_sizes`' `large_l2` branch, with one new profile field.
Everything else on X69's and X70's path stays as it is: X57's fit mechanism, the key's `avx512`
double table (160 = 8 × 20, 240 = 12 × 20, 1024 = 32 × 32), `machine_profile::fit_large_l2` and
`fit_long_row_bytes` (the L2).

Today the in-place limit is the smaller of `fit_row_bytes` and `fit_in_place_bytes`, and
`fit_row_bytes` is also the out-of-place limit, so in place cannot go past the out-of-place
limit. The change:

- `fit_out_of_place_bytes` (new; default no limit): out of place, the rows and output must also
  take at most this many bytes (the limit is the smaller of it, `fit_row_bytes` and, for rows
  longer than a page, `fit_long_row_bytes`). It mirrors `fit_in_place_bytes`. Every other profile
  leaves it at its default, so their limits are unchanged (X57's on the EPYC class, none
  elsewhere).
- On the key: `fit_row_bytes` no limit (the largest `std::size_t`; it still turns the fit rows
  on), `fit_in_place_bytes` no limit (its default), `fit_out_of_place_bytes` **64 times the L2**,
  `fit_long_row_bytes` the L2 as before.

Limits on a 2 MiB L2, in rows (X70's, on `main`, in brackets):

| Length | Out of place | In place |
|---|---|---|
| 1024 | 64 (64) | none (256) |
| 160 | 26214 (1638) | none (1638) |
| 240 | 17476 (1092) | none (1092) |

**Reach** is X69's and X70's exactly: only `set_grid_sizes`' `large_l2` branch sets these values
(x86-64 with a private L2 of 2 MiB or more: Sapphire, Emerald and Granite Rapids). Zen 4 and
Zen 5 (their `avx512` plans have no fit table without the key), Ice Lake-SP VMs, the Xeon 6240's
class, the EPYC class and Apple keep their fields, and the new field's default changes no limit;
the profile tests show it with the recorded topologies.

So by the mechanism the change can move only one-axis c2c double rows of 160 and 240 (in place
past X70's limit; out of place between X70's limit and 64 L2) and of 1024 in place past 256
rows, contiguous on both sides, on the key under `avx512`. Rows of 1024 out of place, planes,
volumes, real rows, float and other lengths keep their plans.

Arms (one binary, campaign branch `exp/cloud-x80`: this change plus the campaign-only fbench
options `FIT_IN_PLACE_BYTES`, `FIT_OUT_OF_PLACE_BYTES` and `FIT_LONG_ROW_BYTES`, registered as
`FFTN_X_X80` in `dev/KNOBS.toml`, not for merging):

- **O** = `FIT_ROW_BYTES=8388608+FIT_IN_PLACE_BYTES=4194304`: X70's limits on a 2 MiB L2 (the
  out-of-place limit, 128 MiB, and the long-row limit, 2 MiB, then change nothing), so X70's
  plans, those of `main`. Listed first, with the A/A control O~. Each job checks that the VM's L2
  is 2 MiB, where these byte values are X70's.
- **N** = the change as it is.
- **W** (informative, decides nothing) = `FIT_OUT_OF_PLACE_BYTES=never+FIT_LONG_ROW_BYTES=never`:
  every row of the three lengths on the two-pass radices, to place the out-of-place crossing past
  the new limit.
- **F** = FFTW 3.3.11 (MEASURE, `PLAN_NOTE=1`) and **M** = oneMKL: reported, not judged.

## Method

- Machines: as X69 and X70, Google Cloud Spot VMs, one per family, `--threads-per-core=1`,
  GCC 14.2 Release, `avx512`, double, one thread (taskset on one CPU), hot cases, statistic
  median: Sapphire Rapids `c3-standard-8`, Granite Rapids `c4-standard-16 --min-cpu-platform="Intel
  Granite Rapids"`, Emerald Rapids `c4-standard-16 --min-cpu-platform="Intel Emerald Rapids"`.
  Each job checks the CPU model and the 2 MiB-L2 key first and stops otherwise. Zen 4 and Zen 5
  are not run: the key does not reach them.
- **Run J** (judging), catalogue `x80` (`bench/catalogues/x80.toml`, committed with this file),
  arms O, N, W, F, M, 12–24 rounds. Cases (27, double):

  | group | cases | N's plan (expected) | O's plan |
  |---|---|---|---|
  | 1024 in place, forward | 512, 2048, 8192, 32768 rows (8–512 MiB) | 32 × 32 (moved) | 8 × 8 × 16 |
  | 1024 in place, backward | 2048 rows | 32 × 32 (moved) | 8 × 8 × 16 |
  | 1024 out of place, forward | 128 rows | 8 × 8 × 16 (framing) | the same |
  | 160 out of place, forward | 2048, 4096, 16384, 24576 rows (10–120 MiB) | 8 × 20 (moved) | 4 × 4 × 10 |
  | | 32768, 65536 rows (160, 320 MiB) | 4 × 4 × 10 (held out) | the same |
  | 160 out of place, backward | 16384 rows | 8 × 20 (moved) | 4 × 4 × 10 |
  | 240 out of place, forward | 2048, 4096, 16384 rows (15–120 MiB) | 12 × 20 (moved) | 4 × 4 × 15 |
  | | 24576, 65536 rows (180, 480 MiB) | 4 × 4 × 15 (held out) | the same |
  | 240 out of place, backward | 16384 rows | 12 × 20 (moved) | 4 × 4 × 15 |
  | 160 in place, forward | 4096, 16384, 65536, 131072 rows (10–320 MiB) | 8 × 20 (moved) | 4 × 4 × 10 |
  | 240 in place, forward | 4096, 16384, 65536, 131072 rows (15–480 MiB) | 12 × 20 (moved) | 4 × 4 × 15 |

  22 cases are expected to move. The held-out cases lie past the new out-of-place limit (N runs
  O's plan there, and W ÷ O places the crossing). In place there is no limit to hold out past; the
  largest in-place cases are twice X70's largest. The run's plan notes are the record: **the
  moved cases are those whose notes differ between N and O**. A case whose notes agree runs the
  same plan in the same binary and does not decide, whatever this table says.
- **Run C** (notes check), the standard catalogue's tracking tier, arms O and N, one round, no
  control.
- Metric: N ÷ O per moved case, the median over valid rounds with its 95 % interval; over the
  moved cases, the per-round geometric mean, its median and 95 % interval (`screen_score.py`, as
  X64, X69 and X70). The band is run J's A/A control (O against O~).

### Decision rules (written before any judging run)

- **Adopt** (Sapphire and Granite Rapids, X70's adopt rule): N is adopted on the 2 MiB-L2 key if,
  on each of the two, over run J's moved cases the geometric mean of N ÷ O is below 1 − band with
  its interval's upper end below 1, and no moved case is slower than 1 + max(band, 3 %) (its
  whole interval above).
- **Reach** (Emerald Rapids, X70's reach rule): Emerald keeps N if no moved case is slower than
  1 + max(band, 3 %) (whole interval) and the moved cases' geometric mean is at most 1 + band.
- **Moves nothing else**: on each VM, run C shows no case whose notes differ between O and N
  other than one-axis c2c rows of 160 and 240 and in-place rows of 1024. In run J the held-out
  cases and the row of 1024 out of place keep O's plan in N. A failure here is a bug to fix
  before adoption, not a verdict.
- **Near misses** (AGENTS section 6, 2026-11-23): a criterion missed by less than the run's band
  gets one re-run of run J (on the same VM if it is still up, else on a new VM of the family),
  and the re-run decides.
- **Narrowing** (AGENTS section 6, 2026-11-23): if moved cases lose, N may be narrowed before
  adoption without a new experiment. Each limit goes down to the largest measured footprint at
  or below which no moved case it governs is slower than 1 + max(band, 3 %) on any of the three
  VMs, or back to X70's value:
  - `fit_in_place_bytes` (in place, all three lengths; now no limit): 480 MiB (240 × 131072),
    320 MiB (160 × 131072), 240 MiB (240 × 65536), 160 MiB (160 × 65536), 128 MiB (1024 × 8192),
    60 MiB (240 × 16384), 40 MiB (160 × 16384), 32 MiB (1024 × 2048), 15 MiB (240 × 4096),
    10 MiB (160 × 4096), 8 MiB (1024 × 512), or X70's 4 MiB;
  - `fit_out_of_place_bytes` (out of place, 160 and 240; now 128 MiB): 120 MiB (160 × 24576,
    240 × 16384), 80 MiB (160 × 16384), 30 MiB (240 × 4096), 20 MiB (160 × 4096), 15 MiB
    (240 × 2048), 10 MiB (160 × 2048), or X70's 8 MiB.

  A deciding run on the final code is then declared in this file before it starts. It must show
  every excluded case on O's plan and no case the narrowed change still moves slower than
  1 + band.
- **Families**: the three share one key. If Emerald fails the reach rule after any re-run, N is
  not adopted on the key and the maintainer decides. If Sapphire or Granite fails the adopt rule,
  N is not adopted.
- W ÷ O on the held-out cases (160 × 32768 and 65536, 240 × 24576 and 65536 out of place) places
  the out-of-place crossing for a follow-up; it decides nothing here. F and M give FFTN ÷ FFTW
  and ÷ oneMKL on the moved cases before (O) and after (N).
- Adopting means `set_grid_sizes` keeps the four fields in its `large_l2` branch as committed on
  the implementation branch: N's plans in run J.

### Not measured

Threaded plans of these rows take the same radices: the choice sees the whole batch's bytes, not
a member's share. Float keeps its plans (the key's table is double only). In place past 512 MiB
of rows is not measured; the hypothesis's reason (two passes over memory instead of three) holds
at any size. The Xeon 6240 is not run: its profile has no `fit_row_bytes`, so its plans cannot
move. A plan-notes check (the Xeon's own profile, and Granite Rapids' recorded topology) is made
before the run.

## Results

Pending.

## Verdict

Pending.
