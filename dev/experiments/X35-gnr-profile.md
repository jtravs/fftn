# X35: a cache-geometry key for Granite Rapids' plans

- **Status:** running
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

State note (2026-11-16 17:57 UTC, work in progress): code arm 1407184 and unit tests done
(PR 25). Cloud: Axion VM `fftn-x35-c4a` (909358881553418641) running, run 1 done, run 2
running; the first Granite Rapids VM `fftn-x35-c4` (4451270156883495296) was preempted by
Google at 17:25:47 before any timing; its replacement `fftn-x35-c4b` (3593494037444099488)
was created at 17:54 and runs the whole job. Both have `--max-run-duration=3h` (DELETE) and a
Mac-side watchdog. Network `fftn-x35-net`, subnet `fftn-x35-sub`, firewall `fftn-x35-ssh`
alive. Next: collect, delete, analyse, write up.

## Verdict

(pending)
