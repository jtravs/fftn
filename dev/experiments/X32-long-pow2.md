# X32: long powers of two as copy-free chains

- **Status:** done
- **Owner:** x32 session, 2026-11-16
- **Phase / task:** 1 (PLAN lead (a)); follows X12, X12b, X26, X26b, X29

## Question

Can long power-of-two rows (c2c ≥ 16384 points, double first) run faster as copy-free
Stockham chains of small radices, the way FFTL.jl runs them, than as FFTN's grid, on which
machines and up to which size; and why are power-of-two chains of 1–2 MiB bimodal from
process to process on the EPYC (X29 arm d)? The answer sets the chain-or-grid rule and the
chain's radices for powers of two.

## Hypothesis

FFTN ÷ FFTW on powers of two ≥ 16384 is 1.19 (Mac), 1.10 (EPYC), 1.03 (Xeon), and FFTN ÷
FFTL.jl 1.40–1.46 (2²⁰ 1.74–1.80; runs 20261116-103657-af06, -105105-30b3,
-110608-e327). FFTL.jl runs one thread's long row as a plain Stockham chain: radix 8 with a
4 up to 16384 points where there are 32 vector registers, radix 4 throughout (one 8 or 2 for
an odd exponent) beyond, no blocking, no copies (its `src/factor.jl`). Its pass is the same
loop as FFTN's lane pass (copies innermost, twiddles per group), so the difference is the
plan: FFTN's rules price radices from in-cache pass costs (X2) and pick 8 × 8 × 8 × 8 × 16
at 65536, whose radix-16 pass reads 16 streams a power of two apart (one L1 set), and switch
to the grid beyond L1 (Mac) or L2/2 (x86). X26 timed FFTN's chain forced to 4⁸ at 65536 on
the Mac at 0.94 of the grid's time (one process). Expected: a chain of radices ≤ 8 (4s
beyond some size) beats the grid on the Mac and the EPYC at 16384–2²² (X29's EPYC screens:
chains 0.53–0.74 of the grid's time), less so or not on the Xeon (X29: the chain lost 1.1–1.6
beyond L2/2 there, with the rules' radices).

Bimodality (X29): chains of 1–2 MiB ran 0.7 or up to 2.5 times the grid's time depending on
the process, steady at 4–16 MiB. Candidate causes: transparent huge pages (a buffer of 1–2
MiB is backed by huge pages or not depending on where its mapping falls, and with huge pages
the physical address bits that index the L2 equal the virtual ones, so legs a power of two
apart collide in L2 sets); 4 KiB aliasing or L2 set conflicts between the two Stockham
buffers (the chain ping-pongs between the destination and the workspace's scratch line,
whose offset from each other changes from process to process with fbench's random page
offsets).

## Change under test

- Step 0, diagnosis (no library change): a stand-alone program (not committed; it links the
  library and its private headers) times a forced chain at 65536 and 131072 points with the
  buffers placed in three ways (as fbench's harness does, in 2 MiB-aligned mappings with
  `MADV_HUGEPAGE`, and with `MADV_NOHUGEPAGE`) and controlled offsets between the input,
  the output and the workspace, and reports each buffer's huge-page backing
  (`/proc/self/smaps`) and Zen 3 counters (cycles, instructions, L2 read misses, dTLB
  misses); the machine's `/sys/kernel/mm/transparent_hugepage` settings are recorded.
- Step 1, screen (no library change): per machine and length (c2c, out of place, 16384 to
  2²²), the rules (A), the chain with the rules' radices (`STRATEGY=chain`), the chain with
  FFTL.jl's radices (`RADICES=...`: 4s with one 8 or 2; and 8s with 4s), FFTL.jl and FFTW.
- Arm (a), the rule: powers of two that the screen shows faster as chains are chains on
  that machine profile (the chain-or-grid rule in `plan_grid`, profile fields in
  `set_grid_sizes`), with the radices of the winning pattern for long rows (a cap on the
  radix of power-of-two chains beyond a size, in the rules planner).
- Arm (b), only if step 0 finds a cause FFTN controls (a placement of its own scratch): the
  fix (for example the scratch line offset from the destination), judged with arm (a).
- Arm (c), only if FFTN's chain with FFTL.jl's radices stays ≥ 10 % slower than FFTL.jl on
  rows the chain takes: a long-row pass with the group loop outside the copy loop
  (twiddles kept in registers across copies), in a new file under `src/passes/`.

Results change bits only for rows whose plan changes (another factorisation); both routes
are tested against the derived tolerances (the chain and grid suites force each).

## Method

- Cases: the standard catalogue's c2c powers of two, one row, out of place, 16384 to 2²²
  (`--tier full --filter '^c2c-(16384|32768|65536|131072|262144|524288|1048576|2097152|4194304)-b1-op'`)
  plus `c2c-65536-b1-ip`; collateral damage: the tracking tier.
- Machines: Apple M5 Max (`neon_fcma`, Homebrew Clang 23, NEON FFTW, libfftl 53171e8;
  shared, one fbench run of this experiment at a time); EPYC 7763 (DMOG, `avx2`, GCC 16,
  `-c 8` shared, `--serial`); Xeon Gold 6240 (`avx512`, GCC 14, `-c 8` shared,
  `--serial`). One thread, hot, double. fbench interleaved rounds with the A/A control.
- Metric: median paired ratios with 95 % intervals; weighted geomean over the cases whose
  plan an arm changes; FFTN ÷ FFTW and ÷ FFTL before and after; per-round ratios for the
  spread.
- Decision rules, written before running:
  1. The screen decides only the arms' shape: per machine, the radix pattern with the
     lowest geomean over the lengths where some chain beats the rules, and the largest
     length up to which chains win at every length.
  2. An arm is adopted on a machine-target (release toolchains: GCC on x86-64 Linux,
     Clang on macOS) when, over the cases whose plan it changes, the weighted geomean
     B ÷ A is below 1 − band with its interval below 1, no such case is slower than
     1 + max(band, 3 %), no tracking-tier case is slower than 1 + max(band, 3 %) (a case
     just past it on a loaded machine is re-run), and the slow mode X29 saw is absent: in
     at least 90 % of the rounds every changed case's per-round ratio B ÷ A is at most
     1.10. A win on one target and a loss on another is adopted per target.
  3. Arm (b) is built only on step 0's evidence and judged by rule 2 over at least 20
     process rounds on the EPYC; arm (c) only under the condition above, judged by rule 2.
  4. At most three arms; a failed approach is written up with its numbers and stops.

## Results

### Step 0: why power-of-two chains are bimodal on the EPYC

EPYC 7763 node (DMOG, kernel 5.14, `transparent_hugepage/enabled` = always, `defrag` =
madvise), the stand-alone program (GCC 16, `avx2`), Slurm jobs 510547, 510584, 510589,
510596; ns per point, one process per line, median of five 20 ms batches.

- **The slow mode is the rules' radices on huge pages.** The rules' chain of 131072 points
  (radix-16 passes) took 14.3–15.6 ns in 14 of 16 processes and 8.7–9.5 in 2 (the grid
  6.3–6.5). The split followed the buffers' huge-page backing exactly (`/proc/self/smaps`:
  the slow processes had the output and the workspace in separate mappings, each partly
  on 2 MiB pages), with twice the L2 read misses (0.19 M against 0.10 M per transform).
  Forced onto huge pages (`MADV_HUGEPAGE`) it took 21–25 ns at every buffer offset tried.
  On huge pages the L2's set index comes from the virtual address, so a radix-16 pass's 16
  streams, a power of two apart, fall into one 8-way L2 set; on 4 KiB pages the frames
  scatter them. FFTL.jl's radices (8 × 4⁷) took 4.4–4.8 ns in every placement, huge
  pages or not (65536 as 4⁸: 4.07–4.24 in 16 processes; 2²⁰: 6.5–6.6 on huge pages, 7.4–7.8
  without, the grid 9.6–11.2).
- **A second, smaller effect: 4K aliasing between the chain's two buffers.** With the
  scratch line a and the output 2048 bytes apart modulo 4 KiB, the radix-4 chain of 65536
  took 4.0–4.1 ns; with the output 64 bytes past a, 6.3 (huge pages) or 4.3–5.3 (4 KiB
  pages); ±64–128 bytes 1.1–1.3 times. Outputs that are not 64-byte aligned cost 5–8 %
  wherever they lie (2064, 2096, 1040, 3088 bytes from a: 4.3–4.5), and 15–90 % more near a
  (16 bytes past: 5.8–7.8; 48 before: 4.8–5.9). Buffers as `std::vector` and
  `fftn::workspace` allocate them (malloc'd mappings, 16 bytes into a page, so the output
  48 bytes before a) took 4.8–5.1 ns at 65536 and 5.3–5.5 at 131072, against 4.1 and 4.2–4.3
  half a page apart. The input's offset from a made no difference (−128 to 256 bytes).
  The 2²⁰ row was hardly affected (1.09 at +64 bytes).

So X29's bimodal chains came from the rules' radix-16 passes meeting transparent huge
pages, not from the chain as such: chains of radices ≤ 8 are steady. The aliasing is
FFTN's to avoid (arm b).

### Step 1: the screen

Mac (M5 Max, `neon_fcma`, Clang 23; this branch at b51dc31, one run per length, 6–10
rounds, the machine shared), ratio to the rules (the grid at every length here):

| length | rules' chain | FFTL's 4s (C4) | FFTL's 8s (C8) | FFTL.jl | FFTW | run, band |
|---|---|---|---|---|---|---|
| 16384 | 1.69 | 0.83 | **0.74** | 0.74 | 0.78 | 20261116-114247-3e44, ±1.00 % |
| 32768 | 1.74 | 0.88 | **0.76** | 0.81 | 0.83 | -114252-0279, ±1.64 % |
| 65536 | 1.62 | 0.87 | **0.87** | 0.82 | 0.84 | -114255-32e9, ±2.46 % |
| 131072 | 1.75 | 0.94 | **0.84** | 0.82 | 0.88 | -114300-4397, ±4.44 % |
| 262144 | 2.64 | **0.93** | 1.02 | 0.89 | 1.17 | -114318-0043, ±1.17 % |
| 2²⁰ | 1.50 | **0.66** | 0.91 | 0.58 | 0.79 | -114338-d2fe, ±3.19 % |
| 2²² | 1.05 | **0.57** | 0.75 | 0.65 | 0.91 | -114400-e822, ±1.58 % |

On the Mac every power of two from 16384 is faster as a chain of FFTL.jl's radices than as
the grid: radix 8 up to 131072 points (2 MiB, an eighth of L2), radix 4 beyond. FFTN's
chain is then within 0.9–1.1 of FFTL.jl's time except at 2²⁰ (1.14) and faster at 2²² (0.88),
so arm (c) (a new pass loop) is not called for on the Mac.

EPYC and Xeon: one run per machine with the profile options (arm (a)'s commit 63dfedb with
`POW2_CHAIN_BYTES=never`, `GRID_MIN_BYTES=131072` so that 16384 and 32768 count as long, and
`POW2_RADIX8_BYTES=1` (A4, radix 4) or `never` (A8, radix 8)) against `main` 29f9772 (N),
ratio to N:

| length | EPYC A4 | EPYC A8 | EPYC FFTL.jl | EPYC FFTW | Xeon A4 | Xeon A8 | Xeon FFTW |
|---|---|---|---|---|---|---|---|
| 16384 (a chain in N) | 1.01 | 1.00 | 0.89 | 1.04 | 1.22 | 1.07 | 0.82 |
| 32768 | **0.75** | 0.80 | 0.74 | 0.88 | 1.26 | 1.06 | 0.77 |
| 65536 | 0.70 | **0.65** | 0.64 | 0.79 | 1.94 | 1.50 | 1.12 |
| 131072 | **0.66** | 0.66 | 0.63 | 0.79 | 1.70 | 1.32 | 0.99 |
| 262144 | 0.62 | **0.58** | 0.57 | 0.82 | 1.79 | 1.26 | 0.90 |
| 524288 | 0.59 | **0.57** | 0.55 | 1.15 | 1.66 | 1.33 | 1.56 |
| 2²⁰ | 0.66 | **0.60** | 0.58 | 0.99 | 1.78 | 1.25 | 1.71 |
| 2²¹ | 1.01 | **0.79** | 0.96 | 1.08 | 1.67 | 1.21 | 1.24 |
| 2²² | 1.08 | **0.85** | 1.03 | 0.95 | 1.71 | 1.28 | 1.23 |
| 65536 in place | **0.71** | 0.73 | 0.67 | 0.84 | 1.84 | 1.43 | 1.04 |

Runs 20261116-120725-c151 (EPYC, GCC 16, 24 rounds all under load, band ±1.00 %; Slurm
510594, 510595) and 20261116-122055-3e32 (Xeon, GCC 14, `avx512`, 24 rounds, 9 under load,
band ±2.22 %; Slurm 16022, 16023). On the EPYC every power of two from 32768 to 2²² is
faster as a chain, radix 8 the better pattern from 65536 up, and FFTN's radix-8 chain is
within 1.00–1.08 of FFTL.jl's time to 2²⁰ and faster beyond (0.82): arm (c) is not called
for there either. Per round the radix-8 chains never exceeded 1.10 of N except in 2 of 24
rounds at 65536 in place (1.26): X29's slow mode is gone. On the Xeon every chain loses
(radix 8 1.06–1.50, radix 4 1.22–1.94): it keeps the grid, as X29 found with the rules'
radices.

### Arm (b): the scratch line half a page from the output

Commit edc1118 (the plan reserves 4 KiB more and execute places line a 2048 bytes past the
output modulo 4 KiB), against arm (a) at 63dfedb, both with the radix-4 chain options, on
the EPYC (run 20261116-123424-12d4, 30 rounds all under load, band ±1.00 %; Slurm 510614,
510615): 0.992–1.005 on every case, none resolved, and both arms' per-round spread alike
(the slowest round 1.03–1.27 of the median). fbench places its buffers at random multiples
of 64 bytes into a page, so the near-alias placements of step 0 (within ±128 bytes) come up
in about one round in sixteen, and never the malloc layout (16 bytes into a page) that cost
15–30 % in the diagnostic program.

### Confirmation, the Mac

Branch fa5a617 (arms a and b; b changes no bit and was neutral) against `main` 29f9772,
tracking tier with FFTW and FFTL.jl (run 20261116-123033-4008, 24 rounds, 20 under load, band
±3.47 %): c2c 16384 0.715, 65536 0.928, 262144 0.905, 2²⁰ 0.625, 2²² 0.565, 65536 in place
0.941; also 65537 0.936 and 10403 0.934; geomean over the six changed rows 0.76; nothing
slower (51 cases no evidence, slowest 1.011); weighted geomean of the tier 0.968. Per round,
every changed row stayed within 1.10 of `main` in 23 of 24 rounds. The other four lengths
(branch bac9e02, run 20261116-124859-39da, 16 rounds, band ±2.18 %): 32768 0.769, 131072
0.914, 524288 0.678, 2²¹ 0.660.

FFTN ÷ FFTW, `main` → branch (same runs): 16384 1.38 → 0.97, 32768 1.11 → 0.88, 65536
1.21 → 1.10, 131072 1.11 → 1.03, 262144 1.11 → 0.98, 524288 1.08 → 0.75, 2²⁰ 1.08 → 0.68,
2²¹ 1.25 → 0.84, 2²² 1.11 → 0.67 (geomean 1.16 → 0.87). FFTN ÷ FFTL.jl: 1.30 → 0.91,
1.19 → 0.94, 1.13 → 1.03, 1.18 → 1.10, 1.11 → 0.98, 1.46 → 1.01, 1.66 → 1.05, 1.56 → 1.05,
1.58 → 0.96 (geomean 1.34 → 1.00).

Float (the rule is in bytes), X26b's float rows, branch bac9e02 against `main` (run
20261116-131558-e5f1, 20 rounds, band ±2.23 %): c2c 65536 0.796, 2²⁰ 0.606, 2²² 0.644
(every round below 1.08); 100000 and r2c 2²⁰, whose plans do not change, 1.024 and 1.064,
no evidence.

### Confirmation, the EPYC

Branch fa5a617 against `main` 29f9772, tracking tier with FFTW and FFTL.jl (run
20261116-124743-7802, 30 rounds all under load, other jobs on 33–55 of the node's cores,
band ±1.00 %; Slurm 510632, 510633): c2c 65536 0.638, 262144 0.530, 2²⁰ 0.578, 2²² 0.850,
65536 in place 0.732 (geomean 0.66; every round below 1.0); 16384 (a chain already, its
radices unchanged) 0.998; weighted geomean of the tier 0.965; 55 cases no evidence; c2c 8
points 1.038 [1.026, 1.039], a leaf whose code no arm touches, just past the collateral
limit of 1.03 (re-run below). FFTN ÷ FFTW, `main` → branch: 65536 1.29 → 0.81, 262144 1.19
→ 0.63, 2²⁰ 1.01 → 0.59, 2²² 1.05 → 0.90, 65536 in place 1.24 → 0.95; ÷ FFTL.jl: 1.61 → 1.01,
1.78 → 0.94, 1.73 → 1.00, 0.97 → 0.83, 1.59 → 1.22.

Re-run of the short rows with the final branch bac9e02 (arm b reverted) against `main`
(run 20261116-133313-0fb3, 30 rounds, none under load, band ±1.00 %; Slurm 510736,
510737): c2c 8 points 1.014 [1.002, 1.034], no evidence; 16–128 points 0.989–0.999; 65536
0.627. The 8-point case passes.

The other four lengths (branch fa5a617, run 20261116-131325-9378, 20 rounds, 10 under
load, band ±1.00 %; Slurm 510637, 510638): 32768 0.728, 131072 0.724, 524288 0.601, 2²¹
0.749, every round below 0.90. Over the nine lengths 16384–2²², FFTN ÷ FFTW goes from 1.06
to 0.73 and FFTN ÷ FFTL.jl from 1.35 to 0.93 (per length ÷ FFTW: 0.98 → 0.96, 1.10 → 0.80,
1.29 → 0.81, 1.15 → 0.79, 1.19 → 0.63, 0.89 → 0.53, 1.01 → 0.59, 0.93 → 0.69, 1.05 → 0.90).

Float on the EPYC (X26b's float rows, bac9e02, run 20261116-133414-15bd, 15 rounds, none
under load, band ±1.00 %; Slurm 510743, 510744): c2c 65536 0.719, 2²⁰ 0.534, 2²² 0.636;
100000 1.002; r2c 2²⁰ 1.038 [1.019, 1.064] with the identical plan in both arms, re-run at
0.998 (double and float, run 20261116-134036-22d9, 20 quiet rounds; Slurm 510787).

### The Xeon

No plan changes there: its caches do not meet the rule's condition (0.7 MiB of L3 per
CPU), so `pow2_chain_bytes` stays 0 and no power-of-two chain exceeds `grid_min_bytes`
(schedule test on the recorded topology). Its confirmation run (Slurm 16037, 16038) was
cancelled once arm (b), the only change that reached its code, was reverted.

## Verdict

By the rules written before the runs:

- **Arm (a), long powers of two as chains of FFTL.jl's radices: adopted on Apple
  (`neon_fcma`, Clang) and on Zen-like caches (`avx2`, GCC 16), rejected on the Xeon.** Mac:
  geomean over the changed rows 0.76 (16384–2²² 0.565–0.941, each interval below 1),
  nothing slower in the tracking tier, within 1.10 per round in 23 of 24 rounds. EPYC: 0.66
  (0.53–0.85), every round below 1. Xeon (`avx512`, GCC 14): the screen's chains lost
  1.06–1.94, so its profile keeps the grid (no plan changes; the rule keys on X29's L3 per
  CPU, which the Xeon does not meet). Commits 42d0cb1, f2789ae. Real rows unchanged.
- **Arm (b), the scratch line half a page from the output: rejected** (fbench 0.992–1.005,
  no case resolved) and reverted (bac9e02). Step 0's 15–30 % for malloc'd buffers stands as
  a measurement of the diagnostic program; judging it needs fbench to place buffers as
  malloc does (16 bytes into a page), a follow-up.
- **Arm (c), a new long-row pass loop: not built.** FFTN's chains with FFTL.jl's radices run
  within 0.9–1.1 of FFTL.jl's time on both machines where they are adopted (geomean 1.00
  on the Mac), below the 10 % condition.
- **Step 0:** X29's bimodal power-of-two chains were the rules' radix-16 passes on
  transparent huge pages (LESSONS 3.12, 3.14); 4K aliasing between a chain's buffers costs
  up to 1.9 times (LESSONS 3.15).

Not measured: Zen 5 and other Zen parts (the rule applies wherever L2 is private and L3 per
CPU is at least 2 MiB), float (the rule is in bytes, so float rows of the same bytes change
too; not timed), rows beyond 2²² (the limits stop at the largest size measured), the
Xeon under `avx2`. DESIGN 8.5 and the decision log record the rule.
