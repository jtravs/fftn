# P1: how often the rules' choice is within 5 % of the measured choice

- **Status:** running (first pass)
- **Owner:** task/2.4-measure session, 2026-11-16
- **Phase / task:** 2.4 (acceptance: the model's first choice within 5 % of the measured best
  for at least 90 % of catalogue cases)

## Question

How often does `planning::rules` build a program within 5 % of the one `planning::measure`
picks, per machine, and where it misses, which rule misfires? The answer decides whether the
rules can stay the default as they are, and which thresholds should be refitted or handed to
measurement. A second question: on a machine whose cache geometry misleads the rules (Granite
Rapids in X31b), does measurement recover the right program?

## Hypothesis

The rules were fitted on the Mac, the EPYC 7763 and the Xeon 6240, so on those three the rules'
choice should be within 5 % on at least 90 % of the tracking tier; the misses should cluster
on long rows near the chain-or-grid thresholds (X29, X32, X34) where X31b showed the
crossovers move with the core. With the Zen-like profile fields Granite Rapids' VM gets
forced on the Xeon (X31b: X29's and X32's chains, X34 off), the rules lose on long rows and
measurement should find the Xeon's own grids again.

## Change under test

None to the library's default: measurement is opt-in (`options::effort`). Arms, all one
commit (task/2.4-measure):

- `N` = `fftn` (rules; with fbench's identical-arm control `N~`).
- `M` = `fftn+PLANNING=measure`; `Q` = `fftn+PLANNING=quick` (P2 uses it).
- Xeon only, the misfire profile `G` = `fftn+CHAIN_PART_BYTES=16384+CHAIN_PART_LIMIT_BYTES=264241152+POW2_CHAIN_BYTES=1056964608+POW2_RADIX8_BYTES=never+GRID_TWO_ODD_PART=never`
  (what `set_grid_sizes` gives Granite Rapids' VM: 504 MiB of L3 over 8 CPUs, so X29's
  chains to half of L3, X32's radix-8 chains to twice L3, X34 off), and `GM` = `G` with
  `PLANNING=measure`.

Every candidate measurement times is a program the rules can also build (build_settings
knobs); no new kernels.

## Method

- Cases: the standard catalogue, tracking tier (61 cases), one thread, hot, median.
- Machines: the Mac M5 Max (Clang 23, `neon_fcma`), the EPYC 7763 (DMOG, GCC 16, `avx2`),
  the Xeon 6240 (GCC 14, `avx512`); shared allocations (`-c 8`), load recorded per round.
- Metric: per case the paired ratio `N ÷ M` (rules' time over measured choice's) with fbench's
  95 % interval and the A/A band; and from M's notes, the rules' time over the chosen one's
  as measured inside planning (`rules/chosen`), and the chosen program per round.
- Decision rules, written before running:
  1. **Acceptance (task 2.4):** on each machine, the share of cases whose `N ÷ M` point
     estimate is at most 1.05 must be at least 90 %. A case above 1.05 is re-run before it
     counts against the rules.
  2. **Measurement must not lose:** a case where M is slower than N with fbench's verdict
     "slower" beyond the band (`N ÷ M` below 0.97) is a bug in the measure loop (its
     statistic, its samples or its tie rule) and is fixed before P2 sets budgets.
  3. **Misfire recovery (Xeon):** among the cases where `G` is slower than `N` by more than
     5 %, `GM` must be within 5 % of `N` in at least 90 %.
  4. Rules missing by more than 5 % on two machines or more are listed as follow-ups
     (a refit or a measured-only choice); no default changes in this experiment.

## Results

First pass, commit 9d2c802 (the Mac's second run ffb9046, which only rotates the order of
candidates in each round). One thread, hot, median, 12 rounds each; `N ÷ M` is the median
over rounds of the paired per-round ratio (analysis script in the session's notes; fbench's
own intervals and verdicts in `fbench show`). Every machine was loaded.

| Machine | Run | Load, A/A band | Rules within 5 % of M | `N ÷ M` geomean | fbench verdicts M vs N |
|---|---|---|---|---|---|
| Mac M5 Max, Clang 23, `neon_fcma` | 20261116-181442-4eed | 12 of 12 rounds loaded, ±5.66 % | 58 of 61 (95.1 %) | 1.001 | 1 faster, 60 no evidence |
| Mac, second run | 20261116-183405-3034 | 12 of 12 loaded, ±5.41 % | 49 of 61 (80.3 %) | 1.020 | 2 faster, 59 no evidence |
| EPYC 7763, GCC 16.2, `avx2` | 20261116-184705-6935 | 12 of 12 loaded (50–72 other cores), ±2.53 % | 59 of 61 (96.7 %) | 1.001 | 1 faster, 60 no evidence |
| Xeon 6240, GCC 14.4, `avx512` | 20261116-183511-33b2 | 7 of 12 loaded, ±7.67 % | 47 of 61 (77.0 %) | 1.031 | 61 no evidence |

- **The misses are partly noise.** In the second Mac run c2c 256, c2r 1000 and c2c 2²²
  read `N ÷ M` 1.06–1.09 although M chose the rules' program in all 12 rounds. On the Xeon,
  5 of the 14 misses had the rules' choice in 8–12 rounds (c2c 60, 1009, 4096, r2c 4096,
  c2c 128 × 4096). None of the misses has been re-run yet, as rule 1 requires.
- **Misses that repeat, with a consistent choice:**
  - planes: c2c 480 × 640 on the Mac, 1.20 and 1.24, fbench "faster" both times (M chose
    5 · 8 · 12 or 4 · 10 · 12 for 480 or 640);
  - r2c and c2r 1024² on the Xeon (1.19, 1.15) and c2r 480 × 640 (1.09);
  - c2r 2048 × 36 on the Mac (1.08, radices 2 · 9 for the 18-point half length);
  - c2c 64³ on the Mac (1.07, 1.11);
  - Xeon rows: r2c 1000 (1.12, 10 · 5 · 10), c2c 360 (1.09, 4 · 10 · 9), c2c 2²⁰ (1.08,
    three grid phases 64 · 128 · 128), r2c 65536 in and out of place (1.07, a two-phase grid
    128 · 256, where the rules make a chain).
- **Measurement never lost by fbench's verdict** (rule 2). Two EPYC cases read below 1 with
  wide intervals: r2c 65536, 0.90 (M chose four different programs over the rounds), and
  c2c 4096, 0.94 (a radix-4 chain in 6 rounds). Inside planning, those choices read 1.12–2.48
  times faster than the rules' choice. The planning loop times on page-aligned scratch
  buffers, while fbench offsets its buffers by random multiples of 64 bytes, so one
  suspicion is placement (4K aliasing, X32); the load is the other.
- **Misfire recovery on the Xeon** (rule 3). `G` was slower than `N` by more than 5 % on 11
  cases (the long powers of two 1.17–1.39, 100000 1.38, 46080 1.29, c2r 2²⁰ 1.22, plus c2c 97
  and 64 near the limit). `GM` came back within 5 % of `N` on 10 of them, choosing the
  Xeon's own grids. Over the tier `GM ÷ N` was 0.992 and `G ÷ N` 1.034. The miss was c2c 2²²
  (`GM ÷ N` 1.31): measurement timed only the rules' choice there, because building its
  candidates (0.5–1.7 s each at that size) spent the 10 s budget first. Commit 4333fff
  builds and times candidates in turn; the confirmation run (Slurm 16131) is pending.

### Second pass (commit 73eb719: placements per round, median statistic; see P2)

| Machine | Run | Load, A/A band | Rules within 5 % of M | `N ÷ M` geomean |
|---|---|---|---|---|
| Mac M5 Max, Clang 23 | 20261116-200843-f54c | 11 of 12 rounds loaded, ±1.93 % | 57 of 61 (93.4 %) | 1.008 |
| Xeon 6240, GCC 14.4 | 20261116-215950-1f7c (Slurm 16140/16141) | 3 of 12 loaded, ±4.10 % | 53 of 61 (86.9 %) | 1.013 |
| EPYC 7763, GCC 16.2 | 20261116-231702-f43c (Slurm 511174/511175) | 11 of 12 loaded, ±1.82 % | 58 of 61 (95.1 %) | 1.009 |

On the Mac every miss is a plane or volume, with a consistent choice: c2c 480 × 640 1.20,
c2r 2048 × 36 1.09, c2c 64³ 1.08, c2r 480 × 640 1.06. No case reads below 0.97 any more; the
first pass's lowest was 0.92.

On the Xeon the misses repeat the first pass, each with a mostly consistent choice:
- r2c 1024² 1.15;
- c2c 2²², 1.14 (a two-phase grid 2048 · 2048);
- r2c 1000, 1.13 (10 · 5 · 10);
- r2c 65536 out of and in place, 1.10 and 1.06 (a two-phase grid 128 · 256);
- r2c 2²⁰, 1.08 (three phases);
- c2c 2431, 1.07 (11 · 13 · 17);
- c2c 16384, 1.05.

Measure had no "slower" verdict there. c2r 2²⁰ read 0.83, with its choice split between two
grids, but fbench gave no evidence either way.

On the EPYC, c2c 65537 ran at 1.75 of measure's time with the rules' plan: measure chose
Bluestein over the rules' Rader stage (`rader=0`) in 10 of 12 rounds. That is PLAN lead (e)'s
"EPYC 65537 1.63 against X27's 1.09", now located. r2c 65536 is fbench "slower" for measure
(`N ÷ M` 0.85, [0.82, 0.92] as M ÷ N 1.18), against rule 2. M chose the rules' plan in 7
rounds, and in the other 5 a power-of-two chain that planning timed faster and fbench far
slower. This row's power-of-two chains are the ones X29 and X32 found bimodal by process
(radix-16 passes on transparent huge pages, LESSONS 3.14). Planning's fresh scratch and
fbench's buffers can sit on different page sizes, which offsets cannot fix.

**Misfire confirmation** (20261116-215243-7fc4, Slurm 16129/16131, commit 4333fff, Xeon,
0 of 8 rounds loaded, ±1.34 %): `G` was 1.25–1.54 of `N` on all 9 long rows. `GM` came back
within 5 % of `N` on all 9, choosing the Xeon's grids in every round. c2c 2²² came back too,
at 0.87 of `N` (a two-phase grid 2048 · 2048, where `N` runs three phases).

### The Xeon's misses, case by case

The Xeon's second pass (run 20261116-215950-1f7c) missed the acceptance (86.9 %). Each miss
below repeated in the first pass (20261116-183511-33b2) with the same choice. The rules'
plan is shown against measure's choice, with `N ÷ M` (rules' time over measure's).

| Case | Rules | Measure | `N ÷ M` | What it needs |
|---|---|---|---|---|
| r2c 1024² | the real axis's 512 as 8 · 16 · 4 | 8 · 8 · 8 (and 4 · 8 · 16) | 1.15–1.19 | planes' radices: the cost tables were fitted to rows, and in a plane these lines run many at a time (phase 4). Until then measure and wisdom |
| c2c 2²² | three grid phases 128 · 128 · 256 | two phases 2048 · 2048 | 1.14 (misfire run: 0.87 of `N`) | a refit of the three-phase threshold (X12b's "three phases beyond L2") for rows far beyond L3 on the Xeon; in X39's scope (the Xeon's long rows) |
| r2c 1000 | the half length 500 as 4 · 5 · 5 · 5 | 10 · 5 · 10 | 1.12–1.13 | an `avx512` double override for 500 (X2's tables) |
| r2c 65536, out of and in place | the half length 32768 as a chain 16 · 8 · 16 · 16 | a two-phase grid 128 · 256 | 1.06–1.10 | a radix-16 power-of-two chain at exactly `grid_min_bytes` (512 KiB), the kind X32 found page-dependent. The real rows' grid threshold, or the long-row radices, on the Xeon; X39's scope |
| c2c 2431 | 13 · 11 · 17 | 11 · 13 · 17 | 1.07 | the radix 11 and 13 columns, scaled from 10's and 12's rather than measured (DESIGN 11.2): a refit, or an override |
| c2c 16384 | 16 · 8 · 8 · 16 | 8 · 8 · 16 · 16 | 1.05 | at the limit; an override if it repeats on a quiet machine |

The other first-pass misses are noise: c2c 360, c2c 2²⁰ and r2c 2²⁰, where measure chose the
rules' program in 10 or more of 12 rounds. So the Xeon needs two threshold refits (in X39's
scope) and three table overrides; planes are measure's until phase 4.

## Verdict

Second pass, measurement only; no default changed.

1. **Acceptance:** met on the Mac (93.4 %) and the EPYC (95.1 %); not met on the Xeon
   (86.9 %), where the misses are long real rows, planes, 2²² and 2431. Re-running made them
   repeat, not vanish.
2. **Measurement must not lose: failed once.** EPYC r2c 65536 is 1.18 of the rules' time. The
   cause is the page backing of power-of-two chains, which planning's scratch does not share
   with the caller's buffers. A fix needs either candidates timed on more than one
   allocation, or chains of that kind kept out of the candidates where X32 found them
   bimodal. A follow-up.
3. **Misfire recovery: met,** 10 of 11 in the first pass and 9 of 9 after the budget fix.
4. **Follow-ups:**
   - planes' radices (Mac, Xeon);
   - long real rows on the Xeon (grids where the rules make chains);
   - EPYC 65537 (Bluestein over Rader, 1.75);
   - the page-backing hazard above.

The first pass's verdict follows.

1. **Acceptance:**
   - met on the EPYC (96.7 %);
   - not met on the Xeon (77.0 %, and about 85 % if the five cases where measurement kept
     the rules' choice are counted as noise);
   - inconclusive on the Mac (95.1 % and 80.3 % in two runs under load, with one stable
     miss).

   The misses above 1.05 are to be re-run on a quieter allocation before task 2.4's
   acceptance is judged.
2. **Measurement must not lose: held.** There is no "slower" verdict anywhere. The EPYC's
   two readings below 1 point at the planning loop's placement or sampling, which P2
   revisits.
3. **Misfire recovery: met** (10 of 11). The one miss was a budget defect, now fixed;
   confirmation pending.
4. **Follow-ups**, misses on two machines or more:
   - the radices of planes' axes (480 × 640 on the Mac and the Xeon, 1024² real planes on
     the Xeon and the Mac): the cost tables were fitted to rows, and a plane's columns run
     as strided lines (phase 4);
   - long real rows on the Xeon: a two-phase grid at 65536, three phases at 2²⁰.
