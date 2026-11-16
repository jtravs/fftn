# X31b: X32 and X34 on Zen 5 and Granite Rapids

- **Status:** done (measurement only, no library change. Zen 5: X32 holds, long powers of
  two 0.58–0.86 of the grid's time, and X34 changes no plan. Granite Rapids: X32's chains
  lose, 0.59–0.94 as grids, and neither the Xeon's plan nor X29's chains fit it; see Verdict)
- **Owner:** x31b-cloud session, 2026-11-16
- **Phase / task:** phase 1 one-dimensional double, a follow-up to [X31](X31-cloud-1d.md)
  for [X32](X32-long-pow2.md) and [X34](X34-xeon-mixed.md) (PLAN leads (a), (c), (d))

## Question

X32 (long power-of-two c2c rows as chains of FFTL.jl's radices where the caches look like
Zen's) and X34 (two grid phases for rows with an odd part ≥ 15 where L3 per CPU is small)
were measured on the Mac, the EPYC 7763 and the Xeon 6240 only. On the two recent x86 cloud
cores of X31:

1. **Zen 5:** does X32's rule hold? (X31: c2c 65536 at 1.50 of FFTW.) Does X34 stay off?
2. **Granite Rapids:** X34's rule does not reach it, because its VM reports a large L3 per
   CPU and so gets X29's chains (X31: 100000 at 2.16 of FFTW, 46080 at 1.60). Does it want
   the Xeon's plan, X29's chains or something else? Does X32 change its long powers of two,
   and if so, for better or worse? Can a cache-geometry key, read from the topology the VM
   reports, separate it from Zen, or does it need a measured choice?

## Hypothesis

From the brief: Zen 5 behaves like the EPYC (X32's chains win, X34 off); Granite Rapids
behaves like the Xeon (it wants two phases at 100000 and the grid for long powers of two).

## Change under test

None (a measurement). Arms, all from fbench's Release builds of git revisions:

- `N` = `main` 5b02f07, after X32 and X34 (with fbench's identical-arm control `N~`).
- `P` = 6a74375, `main` before X32 (and X34).
- `F` = FFTW 3.3.11 with its SIMD and FMA codelets (MEASURE, fresh wisdom). On Granite Rapids
  also `M` = the oneMKL wheel (it reports 2026.0.2, AVX-512 and AMX path).
- Granite Rapids only, the profile options of `N`:
  - `X` = `CHAIN_PART_BYTES=1+GRID_TWO_ODD_PART=15`: X29's chains off, X34's two phases on.
    That is what the Xeon's branch of `set_grid_sizes` gives at this machine's own cache
    sizes. In the second and third runs `X` also has `POW2_CHAIN_BYTES=1`, which turns
    X32's chains off.
  - `Y` = `X` plus `GRID_MIN_BYTES=524288+GRID_THREE_BYTES=1048576`, the Xeon 6240's grid
    thresholds (L2 1 MiB).
  - `Z` = `CHAIN_PART_BYTES=1` alone: X29's chains off without X34.

  fbench refuses `CHAIN_PART_BYTES=0`, the spelling X34's write-up suggests: size fields
  take a whole number from 1, or `never`. A value of 1 turns X29's chains off, because a
  row's power-of-two part is at least one complex value (16 bytes).

## Method

- **Machines.** The VMs were Google Cloud Spot instances in us-central1-a, set up as in X31:
  - `c4d-standard-16` (AMD EPYC 9B45, Zen 5) and `c4-standard-16` with minimum platform
    Granite Rapids (Intel Xeon 6985P-C), each with `--threads-per-core=1`, so 8 whole cores;
  - Ubuntu 24.04, kernel 7.0.0-1017-gcp, GCC 14.2.0, FFTN target `avx512`;
  - transparent huge pages `madvise` (defrag `madvise`).
- **The topology each VM reports** (sysfs, CPU 2), which is what
  `default_machine_profile` reads:

  | VM | L1d | L2 | L3 | L3 shared by | L3 per CPU | X29/X32 branch | X34 branch |
  |---|---|---|---|---|---|---|---|
  | Zen 5 | 48 KiB, 12-way | 1 MiB, 16-way, private | 32 MiB, 16-way | CPUs 0–7 | 4 MiB | on | off |
  | Granite Rapids | 48 KiB, 12-way | 2 MiB, 16-way, private | 504 MiB, 16-way | CPUs 0–7 | 63 MiB | on | off |

  Both VMs take the large-L3 branch: X29's chains (to half of L3) and X32's power-of-two
  chains (to twice L3) apply on both, and X34 is off on both. So on Granite Rapids X32
  *does* change plans: every power-of-two row beyond its `grid_min_bytes` (1 MiB, half of
  L2) becomes a radix-8 chain.
- **Runs.** One thread, pinned with `taskset -c 2`; hot mode, median statistic. fbench's
  smoke (every backend, the guard) passed on both VMs.
  - Run 1, the standard tracking tier (61 cases), 8–40 rounds within 40 minutes.
  - Run 2, X29's catalogue, the c2c rows from 9216 to 4 · 10⁶ points (49 cases: X29's
    odd-part group, its long rows and its half-length rows without 2048–8192), 8–30 rounds.
  - Run 3, Granite Rapids only: seven rows, to split `X` into its parts.
- **Metric.** B ÷ A per case, each arm's median over rounds of its process medians, with
  fbench's paired 95 % interval and verdict. Group geomeans are as in X31; "÷ FFTW" is
  FFTN's time over FFTW's.

## Results

Every round of every run was valid, and no round was under load (except one of the 40 in
Zen 5's run 1):

| VM | Run | Arms | Rounds | Band (A/A) |
|---|---|---|---|---|
| Zen 5 | 20261116-151405-7d87 (tracking) | N P F | 40 | ±1.31 % |
| Zen 5 | 20261116-153437-b48a (x29 c2c) | N P F | 30 | ±1.34 % |
| Granite Rapids | 20261116-151931-7123 (tracking) | N X P F M | 40 | ±1.10 % |
| Granite Rapids | 20261116-155129-87a5 (x29 c2c) | N X Y P F M | 30 | ±1.77 % |
| Granite Rapids | 20261116-163226-6a91 (7 rows) | N Z X F M | 24 | ±1.33 % |

The stores also hold runs that are not results:

- one-case build runs (20261116-150809-4fba, -151037-1245, -151921-21fb);
- Granite Rapids' first tracking run (-151624-06d4) and first x29 run (-151835-8746), which
  stopped at round 0 on `CHAIN_PART_BYTES=0`.

### Zen 5: X32 holds, X34 is off

Long powers of two, `N` ÷ `P` (the chain over the pre-X32 grid; 16384 and 32768 are chains
in both arms with the same radices), and FFTN ÷ FFTW before and after (run 2; run 1 agrees
within 0.03):

| length | N ÷ P | 95 % interval | worst round | ÷ FFTW, P → N |
|---|---|---|---|---|
| 65536 | **0.576** | [0.568, 0.598] | 0.63 | 1.59 → 0.92 |
| 131072 | 0.859 | [0.806, 0.872] | 0.96 | 1.04 → 0.89 |
| 262144 | 0.709 | [0.696, 0.727] | 0.79 | 0.99 → 0.70 |
| 524288 | 0.695 | [0.685, 0.706] | 0.77 | 0.78 → 0.54 |
| 2²⁰ | 0.743 | [0.735, 0.758] | 0.80 | 1.01 → 0.75 |
| 2²¹ | 0.825 | [0.808, 0.845] | 0.91 | 0.88 → 0.73 |
| 2²² | 0.726 | [0.722, 0.741] | 0.79 | 1.19 → 0.86 |
| 65536 in place (run 1) | 0.597 | [0.587, 0.606] | — | 1.50 → 0.89 |

Geomean over the seven out-of-place rows is 0.73; every round of every changed row is below
0.96. In run 1 (tracking tier) the five changed rows give 0.67, and the whole tier
`N` ÷ `P` is 0.965 (weighted).

FFTN ÷ FFTW by group (run 1, `N` and in brackets `P`):

- long powers of two 0.82 (1.11);
- in place 0.83 (1.08);
- one-dimensional double 0.74 (0.78; X31 had 0.80 at 4ebcee5);
- every other group within 0.02 of `P`.

The worst one-dimensional cases against FFTW are now:

- r2c 2²⁰ 1.36, c2c 8 × 4096 1.18, 100000 1.17, 46080 1.15, 4096 1.11, 8 points 1.10;
- c2c 65536 (X31's worst, 1.50) has dropped out of the worst ten.

**X34 on Zen 5:** `N` and `P` (before X34) build the same plan, by the fbench notes, for
every row that is not a power of two: the 61 tracking cases and all 49 x29 rows. Only the
seven powers of two from 65536 up differ. Rows whose plan note is unchanged still moved, in
both directions:

- in run 2, 49152 (a 192 × 256 grid in both arms) at 1.05 of `P`, and 40960 at 0.96;
- in run 1, 4096 at 1.017, c2r 2²⁰ at 0.96 and r2c 1024² at 0.98;
- c2c 128 × 4096 (a batch) at 0.90.

These are differences between two builds of the same plans, not plan effects. Only 4096 and
49152 are on the slower side.

### Granite Rapids: X32's chains lose

`P` (pre-X32, the grid) ÷ `N` (X32's radix-8 chain), run 2. `X` without X32's chains gives
the same within 0.01, except at 2²² (0.989, no evidence):

| length | P ÷ N | 95 % interval | ÷ FFTW, N → P | ÷ MKL, N → P |
|---|---|---|---|---|
| 131072 | 0.882 | faster | 1.78 → 1.57 | 1.21 → 1.07 |
| 262144 | 0.716 | faster | 1.75 → 1.25 | 1.09 → 0.78 |
| 524288 | 0.591 | faster | 1.17 → 0.69 | 1.21 → 0.71 |
| 2²⁰ | 0.619 | faster | 1.13 → 0.70 | 1.08 → 0.67 |
| 2²¹ | 0.688 | faster | 1.14 → 0.78 | 1.06 → 0.73 |
| 2²² | 0.936 | faster | 1.18 → 1.10 | 1.01 → 0.95 |

16384, 32768 and 65536 are chains in both arms, with the rules' radices, because they are
within `grid_min_bytes`. In the tracking tier, `P` ÷ `N` is 262144 0.725, 2²⁰ 0.621 and
2²² 0.925, and the long powers of two go from 1.22 of FFTW with `N` to 1.02 with `P`.

The chains are steady, not bimodal. At 262144, `N` runs 7.37–7.50 ns per point in 39 of 40
rounds (one outlier round at 18.9, which hit every arm-`N` case in that process), against
the grid's 5.0–7.2. **X32 made Granite Rapids' long powers of two slower**: 262144 went from
X31's 1.31 of FFTW to 1.75–1.77.

### Granite Rapids: long mixed rows

Run 2, `X` and `Y` ÷ `N`. By the notes, `X` changes two kinds of rows, `Y` a third, and
`Z` (run 3) splits the first:

| rows | `N`'s plan | `X`'s plan | X ÷ N | Z ÷ N |
|---|---|---|---|---|
| 100000, 10⁶, 2 · 10⁶ (power-of-two part ≤ 1024: X29's chains) | chain | two-phase grid | **0.657, 0.591, 0.687** | 0.653 (two phases, under L2), 0.886, 1.041 (three phases) |
| odd part ≥ 15 and power-of-two part > 1024: 184320, 204800, 221184, 245760, 737280, 819200, 983040, 1024000 | three-phase grid | two-phase grid | **1.139, 1.033, 1.085, 1.051, 1.464, 1.260, 1.259, 1.180** (256000 0.985) | = N |

So on Granite Rapids:

- **Where X29 makes a chain, a two-phase grid is faster.** 100000 goes from 2.12 to 1.39
  of FFTW and from 1.49 to 0.97 of MKL; 10⁶ goes from 1.16 to 0.69 of FFTW. Three phases in
  place of the chain gain less (10⁶ 0.886) or nothing (2 · 10⁶ 1.04).
- **Where `main` already makes a three-phase grid, X34's two phases lose, by 1.03–1.46.**
  These are the very rows the Xeon gained on (X34: 737280 0.874, 819200 0.814, 983040
  0.818, 1024000 0.866). Here Granite Rapids is the EPYC (1.13–1.67 there), not the Xeon.
- **The Xeon's lower grid thresholds (`Y`) gain within L2.** `Y` ÷ `N` for the chains that
  `Y` turns into two-phase grids:
  - 46080 0.845 (1.63 → 1.37 of FFTW), 61440 0.808, 65536 0.875, 57344 0.877;
  - 55296 0.891, 49152 0.913, 51200 0.915, 36864 0.926, 98304 0.934;
  - 40960 0.961, 64000 0.963.

  At 131072, `Y` makes three phases where `X` makes two: 0.721 of `N`, against `X`'s 0.889.
  Rows below 512 KiB and the rest are unchanged (no evidence), and nothing that `Y` changes
  is slower except the two-phase rows above, which `X` shares.

Over the tracking tier (run 1), `X` changes only 100000 (0.655 [0.647, 0.660], the same as
the Xeon's 0.706) and `P` only the long powers of two; every other case shows no evidence
for either arm. One-dimensional double FFTN ÷ FFTW is 0.87 with `N`, 0.86 with `X` and 0.85
with `P` (X31: 0.88); ÷ MKL it is 0.84, 0.83 and 0.83.

## Verdict

- **Zen 5: X32's rule holds; X34 is off.** Long powers of two run at 0.58–0.86 of the
  pre-X32 grid's time (geomean 0.73, every round below 0.96), so the EPYC's 0.53–0.85
  carries over. c2c 65536 goes from 1.48–1.59 (X31: 1.50) to 0.87–0.92 of FFTW, and long
  powers of two from 1.11 to 0.82 of FFTW. X34 changes no plan, as its key intends.
- **Granite Rapids wants neither the Xeon's plan nor X29's chains as they stand.** From the
  three runs, it wants:
  1. grids, not X32's chains, for long powers of two: 0.59–0.94 of `main`'s time from
     131072 to 2²², up to 1.75 → 1.25 of FFTW;
  2. a two-phase grid, not X29's chain, for rows with a small power-of-two part beyond L2:
     100000 0.66, 10⁶ 0.59, 2 · 10⁶ 0.69;
  3. `main`'s three phases, not X34's two, for the other rows with an odd part ≥ 15: X34's
     rule would cost 1.03–1.46 there;
  4. probably a grid from 512 KiB (a quarter of its L2) rather than 1 MiB: 0.81–0.96 on
     eleven rows of 36864–98304 points, and 65536 at 0.875.

  This is a third profile. Its long powers of two follow the Xeon, and so does its
  two-phase grid where X29 chains. Its three-phase rows follow the EPYC.
- **Separating Granite Rapids from Zen.** The VM's topology does differ:
  - L2 is 2 MiB against 1 MiB;
  - L3 is 504 MiB in one instance against 32 MiB;
  - the L1i differs too.

  But the key X29, X32 and X34 use, L3 per CPU in the shared list, misreads a VM: it
  credits Granite Rapids' 8 vCPUs with the whole socket's 504 MiB (63 MiB each, against
  about 4 MiB per physical core). Two cache-size keys would separate these two machines:
  - an L2 of at least 2 MiB (Sapphire, Emerald and Granite Rapids, against Zen 3–5's
    0.5–1 MiB);
  - an L3 too large to be one CCX's (more than 128 MiB, say).

  Each would be fitted to one sample per side, though, and none would say which of the
  four points above applies. Zen parts with large L3 (X3D, 96 MiB) and future Intel parts
  are unmeasured. So the four points are crossovers between chain, two phases and three
  phases that move with the core, not with any cache size the VM reports. That is the case
  for a measured choice (planner 2.4: time the two or three routes once per machine and
  length class). A core field (CPUID vendor and family) is the cheaper stopgap if one
  Intel profile has to ship first.
- **No library change here.** X32 as merged costs Granite Rapids 0.59–0.94 on long powers
  of two (Intel cloud machines with large-L3 VMs); the decision on that belongs to the lead.
  fbench's `CHAIN_PART_BYTES=0` spelling in X34's write-up is refused; use 1.
- **Cost:** USD 1.12 (ledger below), within the USD 10 budget; the campaign has spent USD 1.87
  of its USD 20 cap.

## Ledger

In [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md), section X31b. Two Spot VMs
(`fftn-x31b-c4d` 1444292870677112145, `fftn-x31b-c4` 7219234991263622447), each deleted by
recorded name and ID once its evidence had been copied back and checked. The network
resources were deleted after matching their IDs, and teardown was verified by listing on
2026-11-16 16:44 UTC.
