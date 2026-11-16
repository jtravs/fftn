# X35: a cache-geometry key for Granite Rapids' plans

- **Status:** done (adopted on x86-64 as arm B9: a private L2 of 2 MiB or more gets grids for
  long rows; Granite Rapids 0.59–1.007 of `main` on the 30 changed cases; Axion lost and keeps
  `main`'s plans; see Verdict)
- **Owner:** x35 session, 2026-11-16
- **Phase / task:** phase 1 one-dimensional double, follows [X31b](X31b-cloud-followup.md)
  (PLAN leads (a), (c), (d)); the alternative is the measured-choice planner (task 2.4)

## Question

X29's chains for rows with a power-of-two part ≤ 1024 points, X32's long power-of-two
chains and X33's half-length chains for long real rows apply where a private L2 has at least
2 MiB of L3 per CPU behind it. A Granite Rapids VM (Google Cloud C4, Xeon 6985P-C) reports one
504 MiB L3 for its 8 CPUs, so it takes them, and loses (X31b): X32's chains run 1.06–1.69 of
the grid's time, X29's chains 1.5 times a two-phase grid's, while X34's two-phase rule loses
where `main` makes three phases. Is there a key in the cache geometry the topology reports
(not the vendor or model) that separates Granite Rapids from Zen 3 and Zen 5 for these rules,
leaves the Mac, the EPYC 7763, Zen 5 and the Cascade Lake Xeon with their plans, and gives
Granite Rapids the plans X31b measured as better?

## Hypothesis

The candidate keys, as each machine reports them (sysfs of the fixtures, X31 and X31b):

| machine | L2 (private) | L3 instance, shared by | L3 per CPU | L3 per physical core |
|---|---|---|---|---|
| EPYC 7763 (Zen 3, DMOG node, SMT off) | 512 KiB | 32 MiB, 8 CPUs | 4 MiB | 4 MiB |
| Zen 5 (C4D VM, 8 cores, SMT off) | 1 MiB | 32 MiB, 8 CPUs | 4 MiB | 4 MiB |
| Xeon Gold 6240 (Cascade Lake, bare metal) | 1 MiB | 24.75 MiB, 36 CPUs (18 cores) | 0.69 MiB | 1.4 MiB |
| Granite Rapids (C4 VM, 8 cores, SMT off) | 2 MiB | 504 MiB, 8 CPUs | 63 MiB | about 4 MiB (128 cores; not visible in the VM) |
| Axion (C4A VM, Neoverse V2, 16 cores) | 2 MiB | 80 MiB, 16 CPUs | 5 MiB | not visible in the VM |
| M5 Max (Apple) | 16 MiB shared by 6 | — | — | — |

- **L3 per physical core** does not separate them: Granite Rapids has about 4 MiB per core
  (504 MiB over 128 cores), the same as Zen 3 and Zen 5's CCX, and a VM cannot see the
  socket's cores anyway (its L3 lists only the VM's CPUs).
- **L3 per CPU** (today's key) depends on the VM's size: a 48-core C4 would report 10.5 MiB,
  bare metal with SMT (128 cores, 256 CPUs) 1.97 MiB, on either side of the 2 MiB threshold.
- **The ratio** of L3 per CPU to L2 inherits that dependence.
- **The private L2's size** does not depend on the VM or on SMT, and separates the two
  families as they ship: Intel's server cores since Sapphire Rapids have 2 MiB (Sapphire,
  Emerald, Granite Rapids), Zen 2–5 have 512 KiB–1 MiB, Cascade Lake 1 MiB. It is also a
  plausible cause: a grid's phases work within L2, and twice the L2 makes the grid's
  sub-rows fit where a chain must stream each pass from L3, which on a mesh is far.

Expected: with the key "private L2 of at least 2 MiB", Granite Rapids gets its own profile
and the plans X31b measured as faster; the Mac, the EPYC, Zen 5 and the Cascade Lake Xeon
keep theirs exactly. Axion (Neoverse V2, also 2 MiB of private L2) moves with it; it was
measured only before X32 (X31) and is checked here.

## Change under test

One commit on `main`: a third branch of `set_grid_sizes` (`src/threading/schedule.cpp`) for a
private L2 of at least 2 MiB, ahead of the L3-per-CPU branches, which no longer apply to it.
Its fields, each from X31b's measurements on Granite Rapids:

1. **Long powers of two as grids:** `pow2_chain_bytes` 0 (and X32's radix sequences off),
   so c2c powers of two and the half-length transforms of real rows are grids beyond
   `grid_min_bytes`, as before X32 and X33 (X31b: the grid 0.59–0.94 of X32's chain).
2. **No X29 chains:** `chain_part_bytes` 0.
3. **Two phases where X29 chained:** a new field `grid_two_part_bytes` (fbench
   `GRID_TWO_PART_BYTES`): a row whose length is not a power of two and whose power-of-two
   part is at most this many bytes of complex values stays in two grid phases at any size.
   16 KiB (1024 complex doubles, X29's `chain_part_bytes`). X31b: 100000 0.66, 10⁶ 0.59,
   2 · 10⁶ 0.69 of the chain as two phases; three phases 0.89 and 1.04 at 10⁶ and 2 · 10⁶.
4. **Three phases elsewhere:** `grid_two_odd_part` 0 (X34's two phases lost 1.03–1.46 on the
   rows with an odd part ≥ 15 and a power-of-two part > 1024).
5. **The grid from a quarter of L2, three phases from half of it** (512 KiB and 1 MiB, the
   Xeon 6240's byte thresholds): X31b's `Y` gained 0.81–0.96 on eleven rows of 36864–98304
   points, 65536 0.875, and 131072 as three phases 0.721.

Bits change for every row whose plan changes; both routes are covered by the chain and grid
suites against the derived tolerances.

Arms (fbench, Release builds of git revisions):

- `N` = `main` (with the identical-arm control `N~`);
- `B` = this branch;
- `B3` = `B` with `GRID_THREE_BYTES=2097152` (three phases only beyond L2, as `main`): the
  one threshold of point 5 that X31b did not separate (rows of 1–2 MiB whose odd part is
  ≥ 15 and power-of-two part > 1024, such as 71680 and 86016, were two-phase in `Y`);
- `F` = FFTW 3.3.11 with its SIMD and FMA codelets (MEASURE, fresh wisdom); on Granite
  Rapids `M` = the oneMKL wheel.

Added after runs 1–4, because B and B3 each failed rule 1 on one or two cases (see Results):
one further arm, judged by rule 1 in runs of its own.

- `B9` = `B` with `GRID_MIN_BYTES=1048576` (the grid beyond L2/2, as `main`) and
  `GRID_TWO_PART_BYTES=32768` (two phases for power-of-two parts up to 2048 points), with
  three phases beyond L2/2 as in `B`. This is the profile committed in be32eb8.

## Method

- **Unit tests first** (no machine needed): the profile from each machine's recorded
  topology. New fixtures: `linux-gcp-c4-granite-rapids.sysfs` and
  `linux-gcp-c4a-axion.sysfs` (recorded on the VMs of this experiment) and
  `linux-gcp-c4d-zen5.sysfs` (rebuilt from X31b's recorded cache lines, 8 CPUs). The test
  checks that every field the planner reads is the same as `main` gives on the M5 Max, the
  EPYC 7763, Zen 5 and the Xeon 6240, and has the new values on Granite Rapids and Axion.
  So no Zen 5 run is needed.
- **Machines:** Google Cloud Spot VMs in us-central1, set up as X31b:
  `c4-standard-16` with minimum platform Granite Rapids and `--threads-per-core=1` (8 cores),
  and `c4a-standard-16` (Axion, 16 cores); Ubuntu 24.04, GCC 14, FFTN targets `avx512` and
  `neon_fcma`; one thread pinned with `taskset -c 2`; hot mode, median statistic.
- **Runs** per VM:
  1. the tracking tier (61 cases): `N B F (M)`, 8–40 rounds;
  2. X29's c2c rows as in X31b (49 cases, 9216 to 4 · 10⁶ points, the long powers of two
     among them) and the real rows of 46080 and 2 · 10⁶: `N B B3 F (M)`, 8–30 rounds;
  3. X33's long real rows (r2c and c2r, 32768–2²⁰, in and out of place, 20 cases):
     `N B F (M)`, 8–30 rounds.
- **Metric:** B ÷ N per case, each arm's median over rounds of its process medians, with
  fbench's paired 95 % interval; the weighted geomean over the cases whose plan differs (by
  fbench's plan notes); FFTN ÷ FFTW (and ÷ MKL) before and after.
- **Decision rule, written before running.** Release toolchain: GCC on Linux.
  1. B is adopted on a machine when, over the cases of runs 1–3 whose plan it changes, the
     weighted geomean B ÷ N is below 1 − band with its interval below 1, no such case is
     slower than 1 + max(band, 3 %), and no tracking case is slower than 1 + max(band, 3 %)
     (a case just past the limit is re-run once before it decides).
  2. Point 5's three-phase threshold: B3 replaces B's L2/2 if, over the cases whose plan
     differs between them, B3 ÷ B has a weighted geomean below 1 with its interval below 1,
     or if B fails rule 1 only on such cases and B3 passes it.
  3. Granite Rapids decides whether the key is adopted at all. If Axion fails rule 1 while
     Granite Rapids passes, the branch applies to x86 only (the profile is chosen per
     target) and Axion keeps `main`'s plans, with a follow-up; if Granite Rapids fails, no
     key is adopted and the write-up recommends the measured choice (task 2.4) with the
     numbers.
- **Budget:** at most USD 6 (campaign cap USD 20, USD 1.87 spent), ledger in
  [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md), section X35.

## Results

### Setup and runs

- **VMs.** The ledger is in [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md), section X35.
  All were Google Cloud Spot VMs in us-central1-a: Ubuntu 24.04, kernel 7.0.0-1017-gcp,
  GCC 14.2.0, transparent huge pages `madvise`. The sources came from a bundle of `main`
  2eea943 (`N`) and this branch at 1407184 (`B`, `B3`, `B9`).
  - `fftn-x35-c4` was the first Granite Rapids VM. Google preempted it at 17:25 UTC, before
    any timing.
  - `fftn-x35-c4b` ran every Granite Rapids result below: `c4-standard-16`, minimum platform
    Granite Rapids (Xeon 6985P-C), 8 cores, SMT off.
  - `fftn-x35-c4a` ran Axion's runs 1 and 2: `c4a-standard-16`, Neoverse V2, 16 cores. Its
    run 3 was stopped after one round, once Axion had failed.
- **The topology each VM reports** is in the new fixtures, recorded on the VMs as sysfs
  snapshots that the library's reader loads:
  - `linux-gcp-c4-granite-rapids.sysfs`: L1d 48 KiB, private L2 2 MiB, L3 504 MiB shared by
    CPUs 0–7;
  - `linux-gcp-c4a-axion.sysfs`: L1d 64 KiB, private L2 2 MiB, L3 80 MiB shared by CPUs 0–15;
  - Zen 5's (`linux-gcp-c4d-zen5.sysfs`) is rebuilt from X31b's recorded cache lines.
- **Runs.** fbench's smoke passed on both VMs. Every round was valid, and at most one round
  per run was under load:

| VM | Run | Cases | Arms | Rounds | Band (A/A) |
|---|---|---|---|---|---|
| Granite Rapids | 20261116-180301-b204 | tracking (61) | N B F M | 40 | ±1.00 % |
| Granite Rapids | 20261116-182942-d769 | x29 c2c rows, real 46080 and 2 · 10⁶ (53) | N B B3 F M | 27 | ±1.43 % |
| Granite Rapids | 20261116-191034-0705 | x33 real rows (20) | N B F M | 30 | ±1.00 % |
| Granite Rapids | 20261116-192111-655a | real 262144 (4) | N B B3 F M | 20 | ±1.00 % |
| Granite Rapids | 20261116-192413-6457 | tracking (61) | N B9 F M | 21 | ±1.21 % |
| Granite Rapids | 20261116-193919-29c3 | x29 rows that B9 changes or nearly changes (22) | N B9 F M | 11 | ±1.40 % |
| Granite Rapids | 20261116-195152-5453 | x33 real rows 65536–2²⁰ (16) | N B9 F M | 24 | ±1.00 % |
| Axion | 20261116-171150-dd31 | tracking (61) | N B F | 40 | ±1.00 % |
| Axion | 20261116-173427-0f7e | as run 2 (53) | N B B3 F | 30 | ±1.00 % |

The stores also hold runs that are not results:

- build-only runs: Granite Rapids -175710-4484, Axion -170857-1ab0;
- an aborted B4 run with no results, -192258-9d97;
- Axion's stopped run 3, -180728-8f79.

The runs are in the main checkout's `bench-out/runs`.

### Unit tests

`Schedule.ProfilesOfRecordedMachines` checks the 13 fields the planner reads, from each
recorded topology. I also built it with `main`'s `set_grid_sizes` in place of this
branch's. Then it passes for the M5 Max, the EPYC 7763, Zen 5 and the Xeon 6240, and fails
only for:

- Granite Rapids;
- Axion, before the x86 gate;
- a synthetic bare-metal Granite Rapids, whose L3 is shared by 256 CPUs (1.97 MiB each).
  `main` gives it the Xeon's X34 profile.

So the plans of those four machines do not change. With SMT or in a larger VM, Granite Rapids
keeps the new profile, because the key ignores L3.

### Granite Rapids, arm B (the profile as first planned)

B changed 9 tracking cases, all of them faster:

- geomean 0.754 over those cases, with per-round geomeans of 0.73–0.80;
- weighted geomean of the whole tier 0.959 [0.957, 0.961];
- no other case beyond ±1.5 %.

In run 2 it changed 26 of 53 cases (geomean 0.826), and in run 3 12 of 20 (geomean 0.797).
Two cases failed rule 1:

- **c2c 71680** (2¹¹ · 35, 1.1 MiB): 1.100 [1.091, 1.116]. B makes it three phases from L2/2;
  `main` has two.
- **c2r 131072 out of place**: 1.163 [1.158, 1.176]. Its half-length 65536 (1 MiB) becomes a
  two-phase grid from L2/4. The in-place c2r row and both r2c rows of 131072 gained
  (0.93–0.98).

### B3 (three phases only beyond L2)

B3's plan differs from B's on six rows. Over them, B3 ÷ B has a geomean of 1.050:

- 131072 goes from 0.739 to 0.895 of `main`;
- 98304 goes from 0.925 to 1.011;
- 86016 goes from 0.933 to 1.005;
- 71680 is back at 0.995.

So the first clause of rule 2 fails. Its second clause (B fails only on such cases and B3
passes rule 1) failed too. B3 makes the half-length 131072 of c2r 262144 a two-phase grid,
and run 4 put that row out of place at 1.146 [1.131, 1.172], against 0.876 with B's three
phases. By the rules written beforehand, neither B nor B3 can be adopted.

The failures share a pattern. On this machine, a two-phase grid of 1–2 MiB for the
half-length transform of an out-of-place c2r row loses (1.15–1.16), where three phases or a
chain win. Three phases lose for 71680. Hence one more arm, B9:

- the grid only beyond L2/2, so that c2r 131072 keeps `main`'s chain;
- three phases beyond L2/2, as in B (131072 0.74, c2r 262144 0.88);
- two phases for power-of-two parts up to 2048 points, so that 71680 keeps two phases.

The last point also moves 256000 (2¹¹ · 125) to two phases. X31b measured that plan at 0.985
of three.

### Granite Rapids, arm B9 (adopted)

| Run | Changed cases | Geomean B9 ÷ N | Per-round geomeans | Slowest changed case |
|---|---|---|---|---|
| tracking (-192413-6457) | 6 of 61 | 0.711 | — | 2²² 0.960 |
| x29 rows (-193919-29c3) | 16 of 22 | 0.790 | 0.77–0.81 | 256000 1.007 [0.960, 1.017] |
| x33 real rows (-195152-5453) | 8 of 16 | 0.708 | 0.70–0.72 | c2r 262144 op 0.879 |

The whole tracking tier runs at 0.967 [0.964, 0.969] of `main`. Apart from the six changed
cases, no case is beyond ±3 %. The 30 changed cases, B9 ÷ N, with FFTN ÷ FFTW and ÷ MKL
before → after:

| case | B9 ÷ N | ÷ FFTW | ÷ MKL |
|---|---|---|---|
| c2c 131072 | 0.741 | 1.75 → 1.29 | 1.18 → 0.87 |
| c2c 262144 | 0.719–0.740 | 1.71 → 1.26 | 1.07 → 0.79 |
| c2c 524288 | 0.589 | | |
| c2c 2²⁰ | 0.617–0.638 | 1.14 → 0.70 | 1.03 → 0.64 |
| c2c 2²¹, 2²² | 0.853, 0.960–0.983 | | |
| c2c 100000 | 0.670–0.677 | 2.03 → 1.36 | 1.43 → 0.97 |
| c2c 10⁶, 2 · 10⁶ | 0.588, 0.868 | 10⁶: 1.10 → 0.65 | 10⁶: 1.56 → 0.92 |
| c2c 73728, 81920, 86016, 98304 (three phases) | 0.979, 0.989, 0.934, 0.916 | | |
| c2c 256000 (two phases) | 1.007 | | |
| r2c 262144 op / ip | 0.728 / 0.715 | | |
| c2r 262144 op / ip | 0.879 / 0.794 | | |
| r2c 2²⁰ op / ip | 0.589–0.593 / 0.593 | op: 1.67 → 0.99 | |
| c2r 2²⁰ op / ip | 0.738–0.739 / 0.673 | | |
| r2c, c2r 2 · 10⁶ | 0.615, 0.775 | | |

Against the competitors:

- **Long powers of two, 65536 to 2²²** (run -193919-29c3): FFTN ÷ FFTW goes from 1.26 to
  0.98, ÷ MKL from 1.06 to 0.82.
- **The eight changed real rows:** ÷ FFTW from 1.59 to 1.13, ÷ MKL from 0.84 to 0.59.
- **The tracking tier's 46 one-dimensional double cases:** ÷ FFTW from 0.87 to 0.83 (X31:
  0.88), ÷ MKL from 0.85 to 0.81. By group: long powers of two 1.10 → 0.94, r2c and c2r
  1.11 → 1.02, mixed radix 1.01 → 0.95.
- **The worst one-dimensional cases against FFTW** are now 46080 1.62, 65536 in place 1.47,
  r2c and c2r 65536 1.38–1.41, and 100000 1.36. 262144 is down from 1.72 to 1.23.

Still open, and not changed by B9: rows of 512 KiB–1 MiB, such as 46080 and 65536. B's grid
from L2/4 gained 0.79–0.97 on those c2c rows but lost on the c2r row.

### Axion: X35's profile loses

B, before the x86 gate, against `main`, which gives Axion X29's and X32's chains:

- **tracking tier:** 9 changed cases, geomean **1.154**. 46080 1.455, 100000 1.769,
  262144 1.221, 2²⁰ 1.089, r2c 2²⁰ 1.102. Only 2²² (0.948) and 65536 in place (0.963)
  gained.
- **run 2:** 26 changed cases, geomean **1.276**. Every grid of 36864–131072 points was
  slower (1.06–1.76), 10⁶ ran at 1.971, 2 · 10⁶ at 1.599, and real 2 · 10⁶ at 1.87–1.91.
- **one-dimensional double FFTN ÷ FFTW:** 0.73 with `main`, 0.75 with B.

So Axion, with the same 2 MiB private L2, wants Zen's chains, not Granite Rapids' grids.

## Verdict

- **The key: a private L2 of at least 2 MiB, on x86-64.**
  - It separates Granite Rapids (2 MiB) from Zen 3 and Zen 5 (0.5 and 1 MiB) and from
    Cascade Lake (1 MiB). Cascade Lake keeps X34's profile through its small L3 per CPU.
  - It does not depend on the VM's size or on SMT, as L3 per CPU does.
  - L3 per physical core cannot separate these machines (about 4 MiB on all three), and a VM
    does not show it.
  - On x86 it rests on one sample per side, as X31b warned. Sapphire and Emerald Rapids
    (also 2 MiB of L2) take the new profile unmeasured.
- **The key does not carry across architectures.** Axion's Neoverse V2 has the same 2 MiB
  private L2, and with the profile it lost 1.15–1.28, up to 1.97 on single rows. By rule 3
  the branch applies to x86-64 only, and Axion keeps `main`'s plans (9e78980;
  `set_grid_sizes` now takes the architecture). The L2 size stands for the core behind it
  only within one family.
- **Granite Rapids' plans: arm B9, adopted.**
  - The profile first planned (B) and its three-phase variant (B3) each failed rule 1 on one
    or two cases: 71680 at 1.10, and c2r 131072 and 262144 out of place at 1.16 and 1.15.
  - B9 was designed after those runs, from their results, and judged by rule 1 in runs of
    its own. It passed: 30 changed cases at 0.59–1.007 of `main`'s time (geomeans 0.71,
    0.79 and 0.71 over the three runs), and no tracking case slower than 1.03.
  - B9's plans on Granite Rapids: no X29, X32 or X33 chains, so grids beyond L2/2; three
    phases beyond L2/2; two phases where the power-of-two part is at most 2048 complex
    doubles (`grid_two_part_bytes` 32 KiB); X34's odd-part rule off.
  - The effect: long powers of two run at 0.59–0.98 of `main` (262144 goes from 1.71 to
    1.26 of FFTW), 100000 at 0.67 (2.03 → 1.36 of FFTW, 0.97 of MKL), and long real rows at
    0.59–0.88. One-dimensional double FFTN ÷ FFTW goes from 0.87 to 0.83 (÷ MKL 0.85 →
    0.81).
- **Unchanged:** the M5 Max, the EPYC 7763, Zen 5 and the Xeon 6240 (the profile test
  against `main`'s code), and Axion (the x86 gate).
- **What a measured choice would still add (task 2.4).** The failures of 71680 and of the
  c2r rows show that on Granite Rapids the crossovers between chain, two phases and three
  phases depend on the factorisation, not on a cache size. B9 leaves the 512 KiB–1 MiB rows
  to `main`'s chains, although grids there gained 0.79–0.97 on c2c rows (46080 is still
  1.62 of FFTW). A planner that times the routes once per machine and length class would
  take both gains. It would also cover untested cores (Sapphire and Emerald Rapids,
  Neoverse V3, Zen 6) without another key.
- **Cost:** USD 1.64 (ledger), within the USD 6 budget. The campaign has spent USD 3.51 of
  its USD 20 cap. Every resource was deleted by its recorded ID, and the teardown was
  verified by listing on 2026-11-16 at 20:02 UTC.
