# X70: higher limits for X69's two-pass rows (the 2 MiB-L2 key under `avx512`)

- **Status:** planned
- **Owner:** x70 session, 2026-11-24
- **Phase / task:** 1 (follows X69)

## Question

X69 gave rows of 160, 240 and 1024 points two-pass radices on the 2 MiB-L2 key while rows and
output fit the L2 (in place: half of it). Should rows of 1024 in place, and rows of 160 and 240
in either placement, keep those radices up to larger batches? Rows of 1024 out of place keep
X69's limit.

## Hypothesis

X69's informative arm W gave every row of the three lengths the two-pass radices (runs
20261124-163547-4c44 Sapphire, -162647-bc6f Granite, -164415-35a1 Emerald; bands ±1.00 %).
W ÷ O on the cases past X69's limit:

| Case | Rows (and output) | Sapphire | Granite | Emerald |
|---|---|---|---|---|
| 1024 in place, 128 rows | 2 MiB | 0.875–0.883 | 0.816–0.823 | 0.866–0.874 |
| 1024 in place, 256 rows | 4 MiB | 0.885–0.887 | 0.918–0.923 | 0.940–0.942 |
| 160, 1024 rows out of place | 5 MiB | 0.876 | 0.900 | 0.882 |
| 240, 1024 rows out of place | 7.5 MiB | 0.828 | 0.872 | 0.856 |
| 1024 out of place, 128 and 256 rows | 4 and 8 MiB | 0.954–0.968 | 0.998–1.006 | 0.971–0.989 |
| 1024 out of place, 1024 rows | 32 MiB | 1.289–1.360 | 1.001–1.002 | 0.983–0.994 |

Within X69's limit the same rows took 0.78–0.84 (160 and 240 at 256 rows).

- **1024 in place.** The gain shrinks as the rows grow: Granite 0.82 at 2 MiB and 0.92 at 4 MiB,
  Emerald 0.87 and 0.94. Sapphire stays at 0.88. Extrapolated, Granite and Emerald reach a tie
  near 8 MiB (512 rows). The limit is the largest win measured: rows within 4 MiB, twice the L2
  (256 rows).
- **160 and 240.** The gain also shrinks with the footprint: on Granite, 160 went from 0.78 at
  1.25 MiB to 0.90 at 5 MiB, and 240 from 0.79 at 1.9 MiB to 0.87 at 7.5 MiB. Nothing supports
  dropping the limit: at sizes that only DRAM holds, both plans may be limited by bandwidth.
  The limit is the smallest multiple of the L2 that holds both measured wins, 5 and 7.5 MiB:
  rows and output within 8 MiB, four times the L2 (1638 rows of 160, 1092 rows of 240).
  In place, the rows within half of it (4 MiB, the same row counts), as X69 counted.
- **1024 out of place** keeps the L2. Past it the arm tied (Granite and Emerald) or lost (1024
  rows on Sapphire).

Why 1024 out of place stops gaining at the L2 while in place and the short rows do not is
unexplained. One fact separates them: an out-of-place row of 1024 (16 KiB on each side) spans
four pages on each side, a row of 160 or 240 (2.5 and 3.75 KiB) at most two. The change uses
that as the dividing line (below). It is a working hypothesis, not an established cause.

## Change under test

New limits in `set_grid_sizes`' `large_l2` branch. Everything else on X69's path stays as it
is: X57's fit mechanism, the key's `avx512` double table (160 = 8 × 20, 240 = 12 × 20, 1024 =
32 × 32) and `machine_profile::fit_large_l2`.

- `fit_row_bytes` (out of place: rows and output): from the L2 to **four times the L2** (8 MiB).
- `fit_in_place_bytes` (in place: rows; the limit is the smaller of it and `fit_row_bytes`, as
  before): from half the L2 to **twice the L2** (4 MiB).
- `fit_long_row_bytes` (new; default no limit): out of place, a plan whose rows are longer than a
  4 KiB page (one side) fits only while its rows and output also take at most this. On the key:
  **the L2** (2 MiB). Only rows of 1024 are that long in the key's table, so 1024 out of place
  keeps X69's limit exactly. Every other profile leaves the field at its default, so X57's
  limits on the EPYC class (rows of 768 and 1024 are longer than a page) are unchanged.

Limits on a 2 MiB L2, in rows (X69's in brackets):

| Length | Out of place | In place |
|---|---|---|
| 1024 | 64 (64) | 256 (64) |
| 160 | 1638 (409) | 1638 (409) |
| 240 | 1092 (273) | 1092 (273) |

This is the smallest mechanism found that gives three limits on four cells (two length classes
by two placements): the two existing fields change value and one field with one condition is
added. A limit inside the fit table (per length and placement) would change the type shared with
X57's table.

**Reach** is X69's exactly: only `set_grid_sizes`' `large_l2` branch sets the three values (x86-64
with a private L2 of 2 MiB or more: Sapphire, Emerald and Granite Rapids). Zen 4 and Zen 5 (their
`avx512` plans have no fit table without the key), Ice Lake-SP VMs, the Xeon 6240's class, the
EPYC class and Apple keep their fields; the profile tests show it with the recorded topologies.

So by the mechanism the change can move only one-axis c2c double rows of 160 and 240 (either
placement) and of 1024 in place, contiguous on both sides, on the key under `avx512`, between
X69's limits and the new ones. Rows of 1024 out of place, planes, volumes, real rows, float and
other lengths keep their plans.

Arms (one binary, campaign branch `exp/cloud-x70`: this change plus campaign-only fbench options
`FIT_IN_PLACE_BYTES` and `FIT_LONG_ROW_BYTES`, registered as `FFTN_X_X70` in `dev/KNOBS.toml`, not
for merging):

- **O** = `FIT_ROW_BYTES=2097152+FIT_IN_PLACE_BYTES=1048576`: X69's limits on a 2 MiB L2 (the long-row
  limit, the L2, then changes nothing), so X69's plans, those of `main`. Listed first, with the
  A/A control O~. Each job checks that the VM's L2 is 2 MiB, where these byte values are X69's.
- **N** = the change as it is.
- **W** (informative, decides nothing) =
  `FIT_ROW_BYTES=1073741824+FIT_IN_PLACE_BYTES=1073741824+FIT_LONG_ROW_BYTES=1073741824`: every
  row of the three lengths on the two-pass radices, to find the crossings past the new limits.
- **F** = FFTW 3.3.11 (MEASURE, `PLAN_NOTE=1`) and **M** = oneMKL: reported, not judged.

## Method

- Machines: as X69, Google Cloud Spot VMs, one per family, `--threads-per-core=1`, GCC 14.2
  Release, `avx512`, double, one thread (taskset on one CPU), hot cases, statistic median:
  Sapphire Rapids `c3-standard-8`, Granite Rapids `c4-standard-16 --min-cpu-platform="Intel
  Granite Rapids"`, Emerald Rapids `c4-standard-16 --min-cpu-platform="Intel Emerald Rapids"`.
  Each job checks the CPU model and the 2 MiB-L2 key first and stops otherwise. Zen 4 and Zen 5
  are not run: the key does not reach them.
- **Run J** (judging), catalogue `x70` (`bench/catalogues/x70.toml`, committed with this file),
  arms O, N, W, F, M, 12–24 rounds. Cases (38, double):

  | group | cases | N's plan (expected) | O's plan |
  |---|---|---|---|
  | 1024 in place, forward and backward | 128, 192, 256 rows | 32 × 32 (moved) | 8 × 8 × 16 |
  | | 512, 1024, 2048 rows | 8 × 8 × 16 (held out) | the same |
  | 1024 out of place, forward | 128, 256 rows | 8 × 8 × 16 (framing) | the same |
  | 160 out of place, forward | 512, 1024, 1536 rows | 8 × 20 (moved) | 4 × 4 × 10 |
  | | 4096, 16384, 65536 rows | 4 × 4 × 10 (held out) | the same |
  | 160 out of place, backward | 1024 rows | 8 × 20 (moved) | 4 × 4 × 10 |
  | 160 in place, forward | 512, 1024, 1536 rows | 8 × 20 (moved) | 4 × 4 × 10 |
  | | 4096, 16384, 65536 rows | 4 × 4 × 10 (held out) | the same |
  | 240 out of place, forward | 512, 1024 rows | 12 × 20 (moved) | 4 × 4 × 15 |
  | | 4096, 16384, 65536 rows | 4 × 4 × 15 (held out) | the same |
  | 240 out of place, backward | 1024 rows | 12 × 20 (moved) | 4 × 4 × 15 |
  | 240 in place, forward | 512, 1024 rows | 12 × 20 (moved) | 4 × 4 × 15 |
  | | 4096, 16384, 65536 rows | 4 × 4 × 15 (held out) | the same |

  18 cases are expected to move. The held-out cases lie past the new limits (8 to 320 MiB of rows
  and output; 65536 rows are in DRAM on every VM): N runs O's plan there, and W ÷ O finds the
  crossing. The run's plan notes are the record: **the moved cases are those whose notes differ
  between N and O**. A case whose notes agree runs the same plan in the same binary and does not
  decide, whatever this table says.
- **Run C** (notes check), the standard catalogue's tracking tier, arms O and N, one round, no
  control.
- Metric: N ÷ O per moved case, the median over valid rounds with its 95 % interval; over the
  moved cases, the per-round geometric mean, its median and 95 % interval (`screen_score.py`, as
  X64 and X69). The band is run J's A/A control (O against O~).

### Decision rules (written before any judging run)

- **Adopt** (Sapphire and Granite Rapids, X69's adopt rule): N is adopted on the 2 MiB-L2 key if,
  on each of the two, over run J's moved cases the geometric mean of N ÷ O is below 1 − band with
  its interval's upper end below 1, and no moved case is slower than 1 + max(band, 3 %) (its
  whole interval above).
- **Reach** (Emerald Rapids, X69's reach rule): Emerald keeps N if no moved case is slower than
  1 + max(band, 3 %) (whole interval) and the moved cases' geometric mean is at most 1 + band.
- **Moves nothing else**: on each VM, run C shows no case whose notes differ between O and N
  other than one-axis c2c rows of 160 and 240 and in-place rows of 1024. In run J the held-out
  cases and the rows of 1024 out of place keep O's plan in N. A failure here is a bug to fix
  before adoption, not a verdict.
- **Near misses** (AGENTS section 6, 2026-11-23): a criterion missed by less than the run's band
  gets one re-run of run J (on the same VM if it is still up, else on a new VM of the family),
  and the re-run decides.
- **Narrowing** (AGENTS section 6, 2026-11-23): if moved cases lose, N may be narrowed before
  adoption without a new experiment. Each limit goes down to the largest measured footprint at
  which every moved case it governs passes the rule among themselves, or back to X69's value:
  - `fit_in_place_bytes` (in place, all three lengths): 3.75 MiB (240 × 1024, 160 × 1536),
    3 MiB (1024 × 192), 2.5 MiB (160 × 1024), 2 MiB (1024 × 128), 1.875 MiB (240 × 512),
    1.25 MiB (160 × 512), or X69's 1 MiB;
  - `fit_row_bytes` (out of place, 160 and 240): 7.5 MiB (160 × 1536, 240 × 1024), 5 MiB (160 ×
    1024), 3.75 MiB (240 × 512), 2.5 MiB (160 × 512), or X69's 2 MiB. The in-place limit stays the
    smaller of the two fields.

  A deciding run on the final code is then declared in this file before it starts. It must show
  every excluded case on O's plan and no case the narrowed change still moves slower than
  1 + band.
- **Families**: the three share one key. If Emerald fails the reach rule after any re-run, N is
  not adopted on the key and the maintainer decides. If Sapphire or Granite fails the adopt rule,
  N is not adopted.
- W ÷ O on the held-out cases (512–2048 rows of 1024 in place; 4096–65536 rows of 160 and 240)
  places the crossings for a follow-up; it decides nothing here. F and M give FFTN ÷ FFTW and
  ÷ oneMKL on the moved cases before (O) and after (N).
- Adopting means `set_grid_sizes` keeps the three fields in its `large_l2` branch as committed on
  the implementation branch: N's plans in run J.

### Not measured

Threaded plans of these rows take the same radices: the choice sees the whole batch's bytes, not
a member's share. Float keeps its plans (the key's table is double only). The Xeon 6240 is not
run: its profile has no `fit_row_bytes`, so its plans cannot move. A plan-notes check on the
Xeon (its own profile, and Granite Rapids' recorded topology) is made before the run.

## Results

## Verdict
