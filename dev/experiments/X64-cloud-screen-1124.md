# X64: cloud screen of 2026-11-24 (recent Intel against oneMKL; the families rules reach)

- **Status:** done: X50 adopted on the 2 MiB-L2 key; every reach screen kept except X63's Xeon
  key on Zen 4 (own key: Intel's cores only, PR 97); Emerald's exit check passes
- **Owner:** lead session, 2026-11-24
- **Phase / task:** 1 (left from the exit: recent Intel against oneMKL) and the reach rule
  (AGENTS section 6: every family a rule reaches by cache key is screened)

## Question

Do the rules that reach unmeasured families by cache key hold there (else the family gets its own
key), and do X50's depth-first convolutions win on recent Intel (Sapphire, Emerald and Granite
Rapids), where they are off today?

## Hypothesis

The reach screens hold where the family's caches match the measured machine's (Zen 2 by its CCX,
Ice Lake-SP and Zen 4 parts by the Xeon 6240's L2 key, Emerald Rapids by the 2 MiB key). X50's
convolutions win on the 2 MiB-L2 Intel parts as on the Xeon 6240 (131071 ÷ oneMKL 1.34 → 0.84
there), because their L2 is larger still.

## Change under test

Campaign branch `exp/cloud-screen-1124` (c22a9f8: `main` d4b169f plus campaign-only fbench options
registered as `FFTN_X_SCREEN1124`, not for merging). Every arm of a run is the same binary; an arm
turns one rule off or on through an option:

- X57 off: `FIT_ROW_BYTES=0`; X62's batch leaf of 64 off: `BATCH_LEAVES=never`; X62's Bluestein
  cap off: `BLUESTEIN_POW2_PART_BYTES=<L1>` + `BLUESTEIN_WHOLE_LANES=0`;
- X63: `C2R_PACKED_BYTES=<bytes>|never`; X58's leaf of 12: `BATCH_LEAF_12=0|1`;
- X50 on: `DEPTH_CONVOLUTION_MAX_BYTES=33554432` (the Xeon 6240's limit).

## Method

- Machines (Google Cloud Spot VMs, GCC 14.2, one thread per core except Axion):
  Sapphire Rapids `c3-standard-8`, Granite Rapids `c4-standard-16`, Emerald Rapids
  `c4-standard-16`, Zen 2 `n2d-standard-16` (Rome), Ice Lake-SP `n2-standard-16`, Zen 4
  `c3d-standard-16`. Each job checks the CPU model first and stops on the wrong family. Where a
  VM does not report Zen 2's CCX (16 MiB of L3 per 4 CPUs), FFTN reads a corrected topology
  recording so that every arm plans as bare-metal Zen 2 would. Ice Lake and Zen 4 VMs report a
  large L3 per CPU, so both of X63's arms set its threshold explicitly (the VM's L2 against the
  block).
- Runs and cases:
  - Sapphire, Granite: A1 Bluestein rows (4093 … 999983, 65537, 1018): N, D (X50 on), FFTW,
    oneMKL. A2 (diagnosis) 160, 240, 384, the 1024 family, 32 × 16/256, 1000 in place: N, FFTW,
    oneMKL. A3 (diagnosis) forced radices of 160, 240, 1024 against N.
  - Emerald: E1 12-point batches: N, L (X58's leaf of 12). E2 = A1. E3 the standard catalogue's
    c2c rows without 2²³ and 2²⁴ (FFTW's MEASURE planning there takes over 100 minutes): N, FFTW,
    oneMKL.
  - Zen 2: Z1 the axes catalogue, one round, plan notes only (X61 must stay off); Z2 X57's 768
    and 1024 family; Z3 X62's catalogue without 32, 2048 and the short rows; Z4 X63's moved
    planes and controls, double and float, one thread and 8 lanes.
  - Ice Lake-SP, Zen 4: X63 and X63f catalogues, double and float, one thread and 8 lanes.
- Metric: arm ÷ arm per case with the run's A/A band; only the moved cases (plan notes differ
  between the arms) decide.
- **Decision rules, written before running:**
  - **Reach** (Z2, Z3 for each of X62's two rules, Z4 ×4, E1, Ice Lake ×4, Zen 4 ×4): the family
    keeps the rule if no moved case is slower than 1 + max(band, 3 %) (whole interval) and the
    moved cases' geometric mean is at most 1 + band. Otherwise the family gets its own key: its
    old plans (the rule's field off for that class), or a limit that narrows the rule under the
    narrowing clause, with a deciding re-run declared first. X63: a family keeps it only if all
    four of its runs (one thread and 8 lanes, double and float) do; one that keeps it in double
    only gets double only. X61: kept off if Z1 counts no note "adjacent transforms".
  - **X50 on the 2 MiB-L2 key** (A1 on Sapphire and Granite, E2 on Emerald; D against N): adopt
    if on Sapphire and Granite Rapids the moved cases' geometric mean is below 1 − band with its
    interval below 1 and no moved case is slower than 1 + max(band, 3 %), and Emerald does not
    lose by the reach rule. Adopting sets `depth_convolution_max_bytes` = 32 MiB for that key.
    131071 and 999983 against oneMKL are reported, not judged.
  - **Emerald exit check** (E3): reported like X56's cloud cores, ÷ FFTW and ÷ oneMKL, tail rule
    9 of the 191 rows. A failure in cases a Granite-measured rule moves becomes a follow-up.
  - A2, A3 decide nothing: FFTW's plans and the best forced radix sequences of 160, 240 and 1024
    on recent Intel, for a follow-up.
  - A criterion missed by less than the run's band gets one re-run on a new VM, which decides
    (AGENTS section 6).
- Budget: USD 5 (campaign 12.19 of 20 before this screen); rentals in
  [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md).

## Results

Google Cloud Spot VMs in us-central1-a, 2026-11-24, GCC 14.2 Release, campaign binary c22a9f8,
one thread per core, hot cases, statistic median. Each CPU model checked before the jobs (the
job stops on the wrong family). Scores by the screen's scorer over the moved cases only (plan
notes differ between the arms): the per-round geometric mean, its median and 95 % interval.
"Slower" counts moved cases whose whole interval lies above 1 + max(band, 3 %).

**X50 on the 2 MiB-L2 key** (D, X50 on, against N, `main`'s plans; 9 of 14 cases moved: 4093 …
999983 and 65537; 1018 and the short rows do not reach the limit):

| Family | VM | Run | Band | Moved geomean [95 %] | Slower | Rule | Verdict |
|---|---|---|---:|---|---:|---|---|
| Sapphire Rapids | c3-standard-8 | 20261124-110011-c061 | ±1.57 % | 0.588 [0.578, 0.590] | 0 | adopt | adopt |
| Granite Rapids | c4-standard-16 | 20261124-122525-2325 | ±1.00 % | 0.564 [0.563, 0.565] | 0 | adopt | adopt |
| Emerald Rapids | c4-standard-16 | 20261124-105320-d4a5 | ±7.69 % | 0.615 [0.584, 0.630] | 0 | reach | keep |

The two rows the rule reports without judging, FFTN with X50 (D) against FFTW (MEASURE) and
oneMKL on the same VM:

| Family | 131071 ÷ FFTW | 131071 ÷ oneMKL | 999983 ÷ FFTW | 999983 ÷ oneMKL | N ÷ oneMKL (131071, 999983) |
|---|---:|---:|---:|---:|---|
| Sapphire Rapids | 0.70 | 0.76 | 0.57 | 0.87 | 1.25, 1.68 |
| Granite Rapids | 0.72 | 0.74 | 0.71 | 0.93 | 1.37, 1.31 |
| Emerald Rapids | 0.70 | 0.75 | 0.63 | 0.84 | 1.21, 1.19 |

Emerald's band is wide (the VM's control arm, ±7.69 %), but the moved cases' interval lies far
below it.

**Reach screens** (rule: no moved case slower, moved geomean at most 1 + band):

| Family | Rule | Run | Band | Moved | Moved geomean [95 %] | Slower | Verdict |
|---|---|---|---:|---:|---|---:|---|
| Zen 2 (n2d, Rome) | X57 fit rows | 20261124-105410-445c | ±1.65 % | 21 of 29 | 0.920 [0.917, 0.926] | 0 | keep |
| Zen 2 | X62 batch leaf of 64 | 20261124-105615-71b8 | ±1.31 % | 27 of 53 | 0.819 [0.815, 0.824] | 0 | keep |
| Zen 2 | X62 Bluestein cap C | 20261124-105615-71b8 | ±1.31 % | 12 of 53 | 0.931 [0.924, 0.939] | 0 | keep |
| Zen 2 | X63 double, 1 thread | 20261124-110251-626f | ±2.49 % | 11 of 15 | 0.877 [0.864, 0.884] | 0 | keep |
| Zen 2 | X63 double, 8 lanes | 20261124-110801-e1f6 | ±2.49 % | 11 of 15 | 0.901 [0.892, 0.907] | 0 | keep |
| Zen 2 | X63 float, 1 thread | 20261124-111234-04ed | ±2.75 % | 14 of 17 | 0.662 [0.650, 0.669] | 0 | keep |
| Zen 2 | X63 float, 8 lanes | 20261124-112055-29d2 | ±2.83 % | 14 of 17 | 0.775 [0.761, 0.782] | 0 | keep |
| Ice Lake-SP (n2) | X63 double, 1 thread | 20261124-105432-c6cd | ±6.49 % | 30 of 36 | 0.737 [0.717, 0.744] | 0 | keep |
| Ice Lake-SP | X63 double, 8 lanes | 20261124-105935-ef22 | ±2.76 % | 30 of 36 | 0.768 [0.766, 0.773] | 0 | keep |
| Ice Lake-SP | X63 float, 1 thread | 20261124-110650-ca40 | ±2.49 % | 32 of 41 | 0.624 [0.618, 0.641] | 0 | keep |
| Ice Lake-SP | X63 float, 8 lanes | 20261124-112129-831a | ±1.80 % | 32 of 41 | 0.617 [0.612, 0.622] | 0 | keep |
| Zen 4 (c3d, EPYC 9B14) | X63 double, 1 thread | 20261124-105335-27eb | ±1.39 % | 31 of 36 | 0.902 [0.899, 0.905] | 3 | own key |
| Zen 4 | X63 double, 8 lanes | 20261124-110003-6ca4 | ±1.98 % | 31 of 36 | 0.900 [0.894, 0.907] | 8 | own key |
| Zen 4 | X63 float, 1 thread | 20261124-110552-66bf | ±1.00 % | 34 of 41 | 0.833 [0.832, 0.833] | 2 | own key |
| Zen 4 | X63 float, 8 lanes | 20261124-111724-956d | ±1.92 % | 34 of 41 | 0.813 [0.795, 0.819] | 1 | own key |
| Emerald Rapids | X58 leaf of 12 | 20261124-105251-1bf0 | ±3.72 % | 10 of 12 | 0.874 [0.837, 0.897] | 0 | keep |

X61 on Zen 2 (Z1, run 20261124-105358-ea6d, the axes catalogue): no plan note "adjacent
transforms", so the rule stays off there as intended. Zen 2's VM did not report its CCX; every arm
read the corrected topology recording (the job's `--fix-zen2`).

Zen 4's losses are mid-size planes with X63's threshold at the VM's L2 (1 MiB): 512 × 1024 1.127
and 512² 1.109 at one thread, 600 × 800 1.286 and 720² 1.246 at 8 lanes, although the moved
cases' mean gained. The rule's key was the Xeon 6240's (a private L2 under 2 MiB without Zen 3's
large L3 per core), which Zen 4 parts and VMs match.

**Emerald exit check** (E3, run 20261124-105849-3908, band ±3.71 %, 191 c2c rows of the standard
catalogue without 2²³ and 2²⁴; X56's tail rule: at most 9 rows beyond noise):

| Against | Geomean | Rows above 1.05 | Beyond noise | Those rows (point, interval's lower end) |
|---|---:|---:|---:|---|
| FFTW | 0.71 | 14 | 5 | 1024 b1 bwd 1.21 (1.15), 1024 b1 in place 1.18 (1.11), 131071 1.17 (1.14), 1024 b1 1.17 (1.14), 160 1.14 (1.13) |
| oneMKL | 0.69 | 13 | 6 | 999983 1.27 (1.21), 1024 b3 in place 1.25 (1.22), 131071 1.22 (1.18), 1024 b16 1.17 (1.14), 1024 b1 in place 1.16 (1.12), 1024 b3 1.15 (1.11) |

131071 and 999983 are the convolutions X50 now moves on this key (with X50 on Emerald, 131071 ÷
oneMKL 0.75 and 999983 0.84).

**Diagnosis** (A2, A3 on Sapphire and Granite; decide nothing). `main`'s plans for these rows on
the 2 MiB-L2 key under `avx512`: 1024 = 8 × 8 × 16, split from pass 2. Forced radices against the
rules' plans (runs 20261124-110848-9267, -110857-ce33, -110906-aff2 on Sapphire,
20261124-123201-fbc0, -123210-005a, -123219-b8af on Granite; bands ±1.00-2.01 %):

- 1024 as 32 × 32: 0.73-0.82 of the rules' time for 1-16 rows, out of place and in place, both
  directions (Sapphire 0.730-0.819, Granite 0.743-0.820); 256 and 1024 rows tie or lose (0.99,
  1.12 on Sapphire at 1024 rows).
- 160 as 8 × 20: 0.84 (Sapphire), 0.80 (Granite). 240 as 12 × 20: 0.81, 0.79.
- FFTN's time over the libraries' (A2, runs 20261124-110801-0f5b on Sapphire, 20261124-123103-0fc2
  on Granite): ÷ FFTW, 160 1.06 and 1.23, 240 1.10 and 1.17, 1024 one row 1.18 and 1.12, but
  32-point batches 0.67-0.83 and 256-1024 rows of 1024 0.84-0.87; ÷ oneMKL, 160-384 0.84-0.94,
  1024 one row 0.92-0.96, 1024 in place 3-16 rows 1.06-1.21.

## Verdict

- **X50 adopted on the 2 MiB-L2 key** (`machine_profile::depth_convolution_max_bytes` = 32 MiB in
  `profile_for`'s large-L2 branch, with the depth-first chain): it passes the adopt rule on
  Sapphire (0.588) and Granite Rapids (0.564), and Emerald keeps it by the reach rule (0.615);
  none of 27 moved cases slower. 131071 against oneMKL goes from 1.21-1.37 to 0.74-0.76, 999983
  from 1.19-1.68 to 0.84-0.93.
- **Reach**: Zen 2 keeps X57, X62's two rules and X63 in all four runs; Ice Lake-SP keeps X63's
  Xeon key in all four; Emerald keeps X58. X61 stays off on Zen 2.
- **X63's Xeon key on Zen 4: own key.** All four runs lose (1-8 moved cases slower beyond 3 %).
  The rule now applies to Intel's cores only (PR 97, `machine_profile::vendor`); AMD's cores with
  `avx512` and unknown vendors keep the block. A narrower threshold for Zen 4 is unmeasured.
- **Emerald exit check passes** (5 and 6 rows beyond noise against the bound of 9). Its failing
  rows other than X50's convolutions (the 1024 family and 160) are the follow-up below.
- **Follow-up** (not judged here): on recent Intel the rules' radices for 1024 (8 × 8 × 16), 160
  and 240 lose to 32 × 32, 8 × 20 and 12 × 20 by 16-27 %; an override on the 2 MiB-L2 key
  needs its own experiment with a deciding run and rules naming the 1024 family, 160 and 240.
- Spend: USD 2.44 for the eight VMs, including the client catalogue check's Axion VM and a first
  Granite VM whose package install hung for 55 minutes (deleted by ID, no results); campaign USD
  14.63 of 20 ([cloud-ledger-2026-11.md](cloud-ledger-2026-11.md)).
