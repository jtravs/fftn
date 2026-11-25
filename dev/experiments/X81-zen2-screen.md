# X81: Zen 2 screen of X72 and X73 (cloud screen of 2026-11-25)

- **Status:** done: Zen 2 keeps all three rules (X72 arm A KEEP, arm B KEEP, X73 TIE); no code
  change
- **Owner:** lead session, 2026-11-25 (prepared by a sub-agent)
- **Phase / task:** 1 and 3 (the reach rule, AGENTS section 6: every family a rule reaches by
  cache key is screened)

## Question

Zen 2 matches Zen 3's key (`zen3_like`: x86-64, a private L2 under 1 MiB, at least 2 MiB of L3
per CPU) and the `avx2` double target on AMD's cores, so it took three rules measured only on the
EPYC 7763 (Zen 3): X72 arm A (real rows' half length on the depth-first chain,
`machine_profile::depth_real_max_bytes`), X72 arm B (AMD `avx2` double overrides for 3072, 3200,
4608 and 5120) and X73 (anchored whole transforms in threaded batches,
`machine_profile::anchored_runs`). Does Zen 2 keep each of them, or does it lose and need its own
key?

## Hypothesis

Zen 2 has Zen 3's L1 and L2 (32 KiB and 512 KiB per core), so X72's depth-first range (L2/4 to 4
L2 of half length) and its radix overrides (rows between L1 and L2/4) should win there as on Zen
3: X64's screen of Zen 2 kept X57's, X62's and X63's rules, which ride on the same key. Zen 2's L3
is 16 MiB per core complex of 4 cores (Zen 3: 32 MiB per 8), so a batch leaves a complex's L3
sooner, and anchoring a member to its rows (X73) should gain at least as much as on Zen 3 when a
team spans two complexes.

## Change under test

Campaign branch `exp/cloud-screen-1125` (cece1c5: `main` a696bfd plus the campaign-only fbench
option `X72_AMD_RADICES`, registered as `FFTN_X_SCREEN1125` in `dev/KNOBS.toml`, not for
merging). Every arm of a run is that one binary; an arm turns one rule off through an option:

- **N** = `fftn:cece1c5`, `main`'s plans (every rule on);
- **A0** = N + `DEPTH_REAL_MAX_BYTES=1`: X72 arm A off (real rows' half lengths take the
  breadth-first chain or the half-length grid, as before X72);
- **B0** = N + `X72_AMD_RADICES=0`: X72 arm B off (the four lengths take the costs' sequences, as
  before X72: `build_settings::x72_amd_radices`, `without_x72_overrides`);
- **A** = `fftn_omp:cece1c5+THREADS=8`, anchored runs on (`main`), and **M** = A +
  `ANCHORED_RUNS=0`, X73 off (runs claimed in arrival order).

X72 arm A is judged with arm B on (N against A0) and arm B with arm A on (N against B0), as each
stands on `main`.

## Method

- Machine: Google Cloud Spot VM `n2d-standard-16 --threads-per-core=1 --min-cpu-platform="AMD
  Rome"` (EPYC 7B12, Zen 2: 8 cores, two core complexes of 4 with 16 MiB of L3 each), Ubuntu 24.04,
  GCC 14.2 Release, `avx2`, double. The job checks the CPU model first and stops on another
  family. If the VM does not report Zen 2's complex (16 MiB of L3 shared by 4 CPUs), FFTN reads a
  corrected recording of the VM's topology (`FFTN_TOPOLOGY`) so that every arm plans as a
  bare-metal Zen 2 would (X64's method); the job records which. fbench, hot cases, statistic
  median, the A/A control of each run's first arm, the load of every round in the store.
- **Run Z0** (plan check, one round, no timing claims): arms N, A0, B0 on the standard
  catalogue's tracking tier. Neither rule moves a case there on Zen 3 (X72's run P).
- **Run Z1** (X72 A and B): catalogue `x72` without its five Bluestein primes (83 cases, none
  of which either arm can move by its mechanism), one thread pinned to one CPU, arms N (with the
  A/A control N~), A0, B0; 8–16 rounds within an 11-minute budget.
  - X72 arm A can move even real rows of one axis out of place, contiguous, in double, whose half
    length the c2c depth-first rule takes (half lengths of 8192–131072 points; on Zen 3, 26 of the
    catalogue's cases: real rows of 18432–245760, the batches 20480 × 21, × 3 and 40960 × 6).
  - X72 arm B can move every `avx2` double chain of 3072, 3200, 4608 or 5120 points (on Zen 3, 58
    cases: the c2c rows of those lengths, the real rows of 6144, 6400, 9216 and 10240, the
    depth-first rows with those sub-rows and the two planes).
  - Zen 2's L1 and L2 equal Zen 3's, so the same cases should move; the notes decide.
  - Plan check before these rules (no timing claims): `main`'s campaign build cece1c5 on an EPYC
    7763 (DMOG, Slurm 518448 and 518451, GCC 16, `avx2`) planning for the topology the Zen 2 VM
    recorded in X64's screen (`FFTN_TOPOLOGY`), one round. On `x72` (run
    20261125-093348-1bc1) A0 differs from N in exactly arm A's 26 cases above and B0 in exactly
    arm B's 58; B0's notes equal those of 0a41f58 (X72 arm A alone, before arm B) in all 88
    cases, so B0 runs the plans arm B replaced. On the tracking tier (run
    20261125-093410-b16a) no note differs. Every output was checked against the reference.
- **Run Z2** (X73): the client catalogue's tracking tier, its batches only (33 shapes with more
  than one transform; private, copied to the VM, never committed), 8 threads (the VM's width:
  two cache groups of 4), `fftn_omp`, threads unpinned as the client application runs them; arms
  M (with M~) and A; 10–20 rounds within a 12-minute budget.
- **Run Z3** (X73): the standard catalogue's five tracking batches
  (`c2c-(8-b4096|64-b256|128-b4096|1024-b256|100-b1024)-op`), as Z2, 12–30 rounds within 4
  minutes.
- **Moved cases.** X72 A: the cases of Z1 whose plan notes differ between N and A0; X72 B:
  between N and B0. A case whose notes agree runs the same plan in the same binary and does not
  decide. X73 changes no plan, only which member runs which whole transforms, so every case of
  Z2 and Z3 (all batches of at least two transforms) is moved.
- Metric: the rule's arm ÷ the arm without it (N ÷ A0, N ÷ B0, A ÷ M) per moved case, the median
  over valid rounds with its 95 % interval; over a run's moved cases the geometric mean with its
  interval. The band is the run's A/A control.

### Decision rules (written before any judging run)

Per rule (X72 A in Z1, X72 B in Z1, X73 in Z2 and Z3), with ratio = the rule on ÷ the rule off:

- **KEEP** if the moved cases' geometric mean is below 1 − band and no moved case is slower than
  1 + max(band, 3 %) (its whole interval above).
- **OFF** (Zen 2 loses) if the moved cases' geometric mean is above 1 + band, or a moved case is
  slower than 1 + max(band, 3 %) (whole interval above). The rule is then keyed off Zen 2. Zen 2
  is told apart from Zen 3 in the topology by the L3 of its cache group: 16 MiB per core complex
  on Zen 2, 32 MiB on Zen 3 (`zen3_like` with less than 32 MiB of L3 per group). For X72 arm B,
  which is keyed by vendor and target rather than by topology, the same signal selects the AMD
  table without the four overrides.
- Otherwise **TIE**: the rule stays on Zen 2, as "does not hurt".
- X73 is judged in Z2 and Z3, each against its own band: KEEP if both runs give KEEP, OFF if
  either gives OFF, otherwise TIE.
- **Moves nothing else** (not a verdict): Z0 shows no note differing between N and A0 or between
  N and B0; in Z1, notes differ only within each arm's family above. A failure is a bug to fix
  before any verdict.
- **Near misses** (AGENTS section 6, 2026-11-23): a criterion missed by less than the run's band
  gets one `--serial` re-run of that run, which decides. KEEP and TIE both leave the rule on, so a
  KEEP criterion missed into TIE changes nothing and is reported, not re-run; an OFF verdict
  whose every failed criterion fails by less than the band (the mean within 1 + 2 band, each slow
  case's interval starting within one band of its limit) is re-run.
- **Narrowing** (AGENTS section 6, 2026-11-23): if moved cases lose, X72 arm A may be narrowed by
  `depth_real_max_bytes` or to one kind (r2c or c2r), arm B by dropping lengths from its table,
  and X73 to whole transforms of at least 64 KiB (one side), each on Zen 2 only, without a new
  experiment. A run on the final code is then declared here before it starts, and must show the
  excluded cases on their old plans (X73: arrival-order runs) and no case the rule still moves
  slower than 1 + band.

Reach of the verdicts: other Zen 2 parts by the same signal (a `zen3_like` topology with 16 MiB
of L3 per complex), within `avx2` and 16 registers. Zen 3 keeps every rule whatever Zen 2 shows.

### Not measured

Single precision (X72 moves double only; X73 is precision-blind, judged in double), Zen 2 at 16
or more threads (the VM has 8 cores), pinned threads. FFTW and oneMKL are not run: the rules
compare FFTN's arms only.

- Budget: about USD 0.4 for this VM, within the round's USD 1.5 (the campaign stood at 15.55 of
  20 before it); rentals in [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md).

## Results

**When the rules were fixed.** The rules above were committed and pushed in c2f11e4
(2026-11-25 09:39:49 UTC; PR 120 opened 09:40:27) before either VM was created (09:43:39 and
09:43:52). PR 120 was **not yet merged** when the VMs started: the lead started them without
waiting for its ASan CI check, expected to take about 80 minutes (it took 13 on that
documents-only change). PR 120 merged at 09:53:29 with the rules' text unchanged from c2f11e4,
before this VM's first timed run (Z1 started 09:54:26) and before any run was scored.

Google Cloud Spot VM `fftn-s1125-n2d` (ID 3913590906290901796, n2d-standard-16 with one
thread per core, us-central1-a, 09:43:39–10:19:18 UTC): AMD EPYC 7B12 (Zen 2, family 23 model
49), 8 CPUs, two core complexes of 4 with 16 MiB of L3 each, 512 KiB of L2 and 32 KiB of L1
per core. The VM reported the complexes itself (`zen3_like`, 16 MiB of L3 per 4 CPUs), so
every arm planned from its own topology, no `FFTN_TOPOLOGY` recording. Ubuntu 24.04, Linux
7.0.0-1017-gcp, GCC 14.2.0 Release, `avx2`, double, campaign binary cece1c5, hot cases,
statistic median. Scores re-computed from the run stores by the screen's scorer (`--rule keep`;
`--all-moved` for X73) and checked against `fbench show` of the same stores: per moved case the
ratio rule-on ÷ rule-off with its 95 % interval; over the moved cases the per-round geometric
mean, its median and 95 % interval. "Slower" counts moved cases whose whole interval lies above
1 + max(band, 3 %).

| Run | ID | Threads | Rounds valid | Band (A/A) | Load average median, max | Rounds under load |
|---|---|---|---:|---:|---|---:|
| Z0 notes | 20261125-095359-d617 | 1, pinned | 1 | (no timing) | 3.76, 4.33 | 1 of 1 |
| Z1 x72 | 20261125-095426-d1f0 | 1, pinned | 16 | ±1.20 % (N~) | 1.06, 3.19 | 0 of 16 |
| Z2 client batches | 20261125-100331-a9cd | 8, unpinned | 15 | ±8.02 % (M~) | 2.49, 3.52 | 0 of 15 |
| Z3 standard batches | 20261125-101611-2ba1 | 8, unpinned | 30 | ±5.30 % (M~) | 2.11, 2.75 | 0 of 30 |

**Plan checks.** Z0 (the standard catalogue's tracking tier, 61 cases): no note differs between
N and A0 or between N and B0. Z1: A0 differs from N in exactly arm A's 26 expected cases and B0
from N in exactly arm B's 58; no other note differs. So the rules move on Zen 2 exactly what
they move on Zen 3 with Zen 2's topology (the plan check before the rules).

**X72 arm A** (N ÷ A0, Z1; 26 moved: even real rows out of place on the half-length depth-first
chain against the breadth-first chain or the half-length grid):

| Moved cases | Count | Ratio range | Worst case [95 %] |
|---|---:|---|---|
| r2c rows of 18432–245760 | 11 | 0.656–0.874 | 18432: 0.874 [0.854, 0.895] |
| c2r rows of 18432–245760 | 11 | 0.658–0.914 | 18432: 0.914 [0.901, 0.919] |
| r2c 20480 × 21, 40960 × 6 | 2 | 0.698–0.829 | 20480 × 21: 0.829 [0.790, 0.862] |
| c2r 20480 × 3, 40960 × 6 | 2 | 0.695–0.924 | 20480 × 3: 0.924 [0.868, 0.948] |
| **all moved** | **26** | **geomean 0.745 [0.738, 0.751]** | slower: 0; every case faster beyond the band |

**X72 arm B** (N ÷ B0, Z1; 58 moved: the four AMD `avx2` double overrides against the costs'
sequences):

| Moved cases | Count | Ratio range | Worst case [95 %] |
|---|---:|---|---|
| c2c 3072 (1, 4, 64 rows both directions; in place 1, 16) | 8 | 0.849–0.914 | 64 rows: 0.914 [0.896, 0.930] |
| c2c 3200 (the same) | 8 | 0.828–0.902 | 64 rows bwd: 0.902 [0.894, 0.919] |
| c2c 4608 (the same) | 8 | 0.899–0.974 | 64 rows bwd: 0.974 [0.928, 0.983] |
| c2c 5120 (the same) | 8 | 0.896–0.959 | 4 rows: 0.959 [0.938, 0.968] |
| c2c 12288, 18432, 20480, 8 × 3072, 3200 × 8 | 5 | 0.899–0.951 | 20480: 0.951 [0.940, 0.969] |
| real 6144 × 48, × 66 (r2c, c2r) | 4 | 0.895–0.938 | r2c × 66: 0.938 [0.918, 0.949] |
| real 6400 (× 1, r2c × 7), 9216 | 5 | 0.865–0.991 | c2r 6400: 0.991 [0.959, 1.003] |
| real 10240 (× 1, r2c × 28, c2r × 4) | 4 | 0.921–0.977 | c2r × 4: 0.977 [0.961, 0.989] |
| real 24576, 36864, 40960 (× 1, × 6) | 8 | 0.923–0.971 | c2r 40960 × 6: 0.971 [0.925, 0.998] |
| **all moved** | **58** | **geomean 0.914 [0.904, 0.922]** | slower: 0; four within the band (c2r 6400, 9216, 10240 × 4, 40960 × 6), none above 1.004 at its interval's top |

**X73** (A ÷ M, every case moved: anchored whole transforms against arrival-order runs):

| Run | Moved cases | Count | Ratio range | Worst case [95 %] |
|---|---|---:|---|---|
| Z2 | c2c rows of 384 × 7372 and × 7373 (in and out of place, both directions) | 8 | 0.766–0.984 | in place bwd × 7372: 0.984 [0.835, 1.027] |
| Z2 | c2c 750 × 4469 (the same four) | 4 | 0.787–1.004 | in place bwd: 1.004 [0.895, 1.027] |
| Z2 | c2c 1728 × 128, 864 × 384, 10240 × 3 (bwd) | 3 | 0.907–0.978 | 10240 × 3: 0.978 [0.971, 1.032] |
| Z2 | c2c planes 192² × 384, 384² × 750 (in place, both directions) | 4 | 0.910–0.973 | 384² × 750 bwd: 0.973 [0.894, 1.007] |
| Z2 | real batches (864–40960 points, 3–384 rows) | 14 | 0.854–1.014 | c2r 40960 × 6: 1.014 [0.921, 1.116] |
| Z2 | **all 33** | 33 | **geomean 0.926 [0.875, 0.958]** | slower than 1.080: 0 |
| Z3 | c2c 8 × 4096, 64 × 256, 128 × 4096, 1024 × 256, 100 × 1024 (out of place) | 5 | 0.874–0.943 | 64 × 256: 0.943 [0.891, 0.978] |
| Z3 | **all 5** | 5 | **geomean 0.906 [0.876, 0.939]** | slower than 1.053: 0 |

Z2's band is wide (±8.02 %: eight unpinned threads on a VM); the moved cases' geometric mean
lies below 1 by more than half of it, and no case's interval starts above 1.

## Verdict

- **X72 arm A: KEEP** (Z1, 0.745 [0.738, 0.751] over the 26 real rows it moves against a
  band of ±1.20 %, none slower). Zen 2 gains as Zen 3 did (X72: 0.761).
- **X72 arm B: KEEP** (Z1, 0.914 [0.904, 0.922] over 58, none slower; X72 on Zen 3: 0.906).
- **X73: TIE, stays on.** Z2: 0.926 [0.875, 0.958] against a band of ±8.02 %, a mean that misses
  KEEP's 1 − band (0.920) by 0.006, with no case slower: TIE. Z3: 0.906 [0.876, 0.939] against
  ±5.30 %, none slower: KEEP. By the rule (KEEP only if both give KEEP, OFF if either gives OFF)
  X73 is TIE, and the rule stays on Zen 2 as "does not hurt". The KEEP criterion missed into TIE
  changes nothing, so by the near-miss clause it is reported and not re-run.
- **Moves nothing else**: Z0 and Z1's notes as expected (above). No near-miss re-run and no
  narrowing were used.
- **No code change.** Zen 2 keeps every rule it reaches by Zen 3's key: X72's
  `depth_real_max_bytes`, X72's AMD `avx2` double overrides and X73's `anchored_runs`. No key for
  Zen 2 (16 MiB of L3 per complex) is added.
- **Reach**: by the reach rule above, the verdicts extend to other Zen 2 parts with the same
  signal (a `zen3_like` topology with 16 MiB of L3 per core complex), within `avx2` and 16
  registers. Not measured: single precision, 16 or more threads, pinned threads (above).
- Spend: USD 0.25 for this VM (35.7 minutes at USD 0.417/h with disk and IP); the screen's two
  VMs USD 0.40 ([cloud-ledger-2026-11.md](cloud-ledger-2026-11.md)).
