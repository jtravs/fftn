# X12: Rows beyond the cache: the grid against the chain

- **Status:** running (Mac done: grid adopted; x86 pending)
- **Owner:** task 1.7 agent, 2026-11-14
- **Phase / task:** 1.7 (with the grid part of 1.6)

## Question

For c2c rows that do not fit L1, does the grid strategy (DESIGN.md 8.5,
`src/strategies/grid.hpp`) beat the chain, from which length, and with which split and
block width? The answer sets the planner's switch from chain to grid and the grid's rules.

## Hypothesis

The chain makes every pass over the whole row; beyond L1 every pass streams L2 or memory,
and for power-of-two lengths a lane pass's R legs lie n/R apart, map to the same few L1
sets and evict each other between units (the Mac's chain time jumps fivefold from 8192 to
16384 points, run 20261114-095234-2dcf: FFTN ÷ FFTW 1.18 → 2.51). A grid whose phases each
work on one block of columns gathered into tight scratch keeps each pass inside L1 and
touches memory once per phase. Expected: rows from 2^14 to 2^22 within 1.25 times FFTW's
time on the Mac (from 1.85–2.84), the chain kept at and below 8192.

## Change under test

Arm B: the planner's default after task 1.7 (a grid beyond `grid_min_bytes`, the L1 data
cache). Arm C: the same build with `STRATEGY=chain` (every row a chain, as before 1.7).
Arm A: FFTW 3.3.11 NEON build at MEASURE (the reference arm of the tracking tier). The
forms compared during development (all forced through `build_settings`, timed with a
scratch timer against FFTW in one process; indicative only, see Results): the decimation-
in-frequency grid in two phases against three phases; the split (n1 x n2 near sqrt(n), the
shorter first, against the longer first); block widths 2 to 64; the phase-1 twiddles as a
table of n values against two small tables; row prefetch distance 0, 4, 8 and 16 rows. The
DIT grid and the in-place sandwich were not built: the DIF form needs no transpose pass of
its own (the transpose folds into phase 1's store) and its in-place route is one scratch
array of n (the workspace block), so neither alternative had a reason to be tried first.

No switch is added to the library: the strategy and the split are private build settings
(`build_settings::strategy`, `grid_factors`) and the thresholds machine-profile fields
(`grid_min_bytes`, `grid_three_bytes`, `grid_block_bytes`, `grid_block`), set by fbench
options as the other profile fields are (bench/README.md, "FFTN options").

## Method

- Cases: the standard catalogue, tracking tier, c2c rows of one transform from 2^14 to
  2^22 points, double, out of place (`--filter 'c2c-(16384|65536|262144|1048576|4194304)-b1'`).
- Machine: the Mac (Apple M5 Max, macOS, Homebrew Clang, `neon_fcma`), serial, shared and
  loaded (the load line of the run is the noise context).
- Command:

  ```bash
  uv run --frozen python bench/fbench.py run A=fftw B=fftn C=fftn+STRATEGY=chain \
    --tier tracking --filter 'c2c-(16384|65536|262144|1048576|4194304)-b1'
  ```

- Metric: per case, the median paired ratio B ÷ A and C ÷ A over rounds with its 95 %
  interval, and B ÷ C; the A/A band of the run.
- Decision rule, written before running: adopt the grid as the default beyond
  `grid_min_bytes` if B ÷ C is below 1 − band for every case of 2^16 points and more and
  for 16384, and no case is slower than with the chain; the acceptance target of task 1.7
  is B ÷ A ≤ 1.25 for 2^16 to 2^22. Cases short of the target stay open here with their
  stage costs.

To run on the EPYC (DMOG) and the Xeon, where L1 is 32 KiB and L2 1 MiB (so the grid
starts at 2048 points and three phases at 2^14 by the default profile):

```bash
uv run --frozen python bench/fbench.py run A=fftw B=fftn C=fftn+STRATEGY=chain \
  --tier tracking --filter 'c2c-(1024|2048|4096|8192|16384|65536|262144|1048576|4194304)-b1'
# the threshold and the two-or-three-phase switch, on their own:
uv run --frozen python bench/fbench.py run A=fftn B=fftn+GRID_MIN_BYTES=never \
  C=fftn+GRID_THREE_BYTES=never --tier tracking --filter 'c2c-[0-9]+-b1'
```

## Results

Development timings (scratch timer, minimum of repeated runs, FFTN's plan against FFTW
MEASURE in one process on the loaded Mac; load average 5–9, run-to-run spread about 10 %):

- Every form gathers blocks into tight scratch: phases reading a block in place were not
  built, since a block's rows a power of two apart share cache sets (fusing the gather
  into the first pass where they do not is open).
- Two phases, n1 = n2 = sqrt(n), blocks of 8 columns: 16384 1.24, 65536 1.39, 262144 1.25,
  1048576 1.45, 4194304 2.11 times FFTW. Blocks must keep a block's chain (block x max Ni
  points in each of a and b) within L1: 4 columns at 2^20 (1.26), 2 at 2^22 (1.43); but
  narrow blocks make the gathers and scatters touch a line or a page per two to four
  elements, and they then dominated (2^22: 29 of 41 ms in gathers and scatters).
- Three phases with blocks of 32 columns: 2^20 1.12, 2^22 1.09 (64 x 128 x 128 and
  128 x 128 x 256), against 1.30 and 1.43 for the best two-phase forms; at 2^16 and below
  three phases lose (65536: 1.53 against 1.19). At 2^17 two phases (256 x 512, blocks of
  8) won (1.22 against 1.40); at 2^18 three (1.30–1.39 against 1.45–1.49). Hence the
  switch at an eighth of L2 (2 MiB on the M5's 16 MiB): two phases up to 2^17 points
  (double), three beyond.
- Phase order: 256 x 512 against 512 x 256 at 2^17, 1.22 against 1.60: the shorter
  factor first.
- Phase radices: 16 x 16 for 256 against planner v1's 2 x 8 x 16 (whose tables price a
  grouped first pass and tails, which a phase's lane passes do not have): 65536 1.23
  against 1.49. Hence the phases' own rule (fewest passes, then the most even radices).
- The twiddled transpose's per-unit division and its table read across rows: 78 µs of
  282 at 65536 before, gone after (tiles walked incrementally, the table in unit order).
- Phase-1 twiddles from two small tables (w^(q k) per block position, w^(c0 k) per block)
  against one table of n: no measurable time difference at 2^18–2^22 on the Mac; kept, for
  the n values (64 MiB at 2^22) the plan no longer holds. Errors rose by about 1 % (rms
  4.31e-16 to 4.36e-16 at 2^22).
- Row prefetch 8 rows ahead in gathers and scatters: 2^22 33.4–34.0 ms without, 28.5–29.3
  with (after the switch to three phases).
- After all of the above (default profile): 16384 1.26–1.40, 32768 1.14–1.18,
  65536 1.19–1.22, 131072 1.27, 262144 1.39–1.52, 524288 0.96–1.15, 1048576 1.05–1.20,
  2097152 1.10–1.21, 4194304 1.03–1.30 times FFTW (chain: 2.49, —, 2.23, —, 2.93, —,
  1.94, —, 2.04).

fbench A/B, run 20261114-110844-8b03 (Mac, source 02e9e54, 18 valid rounds, 9 of them
under load: load average median 5.7, max 7.3; A/A band ±1.00 %). Median ratio to FFTW
(A), 95 % interval:

| case | B grid ÷ A | C chain ÷ A | B ÷ C |
|---|---|---|---|
| c2c-16384-b1-op | 1.372 [1.354, 1.383] | 2.361 [2.329, 2.394] | 0.58 |
| c2c-65536-b1-op | 1.216 [1.201, 1.240] | 2.144 [2.038, 2.211] | 0.57 |
| c2c-262144-b1-op | 1.084 [1.034, 1.119] | 2.312 [2.203, 2.334] | 0.47 |
| c2c-1048576-b1-op | 1.377 [1.273, 1.456] | 1.934 [1.864, 1.961] | 0.71 |
| c2c-4194304-b1-op | 1.428 [1.222, 1.464] | 2.280 [2.039, 2.397] | 0.63 |
| c2c-65536-b1-ip | 1.077 [1.009, 1.085] | 1.913 [1.787, 1.939] | 0.56 |
| weighted geomean | 1.251 [1.184, 1.269] | 2.150 [2.060, 2.186] | 0.58 |

(B ÷ C is the quotient of the two medians, not a paired ratio; every case's intervals are
far apart.) The fbench ratios at 2^20 and 2^22 are worse than the development timings
(1.05–1.20 and 1.03–1.30), which took the minimum over repetitions on fixed buffers;
fbench takes medians over batches with fresh page offsets per round, and its wide
intervals there show the load. 2^18, which the development timings put at 1.3–1.5, came
out at 1.08.

### Task 1.7b: fusing the copies into the phases' end passes

The change: a phase's first pass reads its block's rows from the array directly and a
later phase's last pass writes them (with the row twiddles or the scale), instead of a
gather and a scatter around the chain on scratch (`phase_pass` in `src/passes/grid.inl`;
`build_settings::grid_fused`). Development timings (scratch timer, Mac at load 9–12, so
only large effects count): fusing both ends of every phase doubled the time wherever a
phase's end radix was 16 (65536 with 16 x 16 phases: 400 µs against 203), as the set
argument predicts (16 rows a power of two apart, 8 ways); with end radices of at most 8
(forced 4 x 16 x 4, 8 x 4 x 8, 4 x 4 x 4 x 4 for 256) fused matched the unfused 16 x 16
(218–220 µs against 202).

fbench, the same filter as above, A=fftw, B=fftn built from main a3903ed (copies), C=fftn
of this branch:

| case | run 20261114-131752-7e99: B | C (fuse each end of radix ≤ 8) | run 20261114-132841-80e0: B | C (fuse only phases of radices ≤ 8) |
|---|---|---|---|---|
| c2c-16384-b1-op | 1.330 | 1.455 | 1.309 | 1.282 (same program as B) |
| c2c-65536-b1-op | 1.100 | 1.089 | 1.052 | 1.039 (same program) |
| c2c-262144-b1-op | 1.318 | 1.141 | 1.029 | 0.922 [0.904, 0.944] |
| c2c-1048576-b1-op | 1.134 | 1.152 | 1.352 | 1.462 [1.435, 1.493] |
| c2c-4194304-b1-op | 1.027 | 1.126 | 1.374 | 1.360 (same program) |
| c2c-65536-b1-ip | 1.071 | 1.077 | 1.094 | 1.084 (same program) |
| band (A/A) | ±1.64 % | | ±1.00 % | |

Both runs were loaded (80 of 80 and 35 of 40 rounds; load average median 10.6 and about
10); B's own ratios moved by up to 0.3 between them at 2^18–2^22, so only differences
within a run count. Reading: a fused end with radix ≤ 8 next to an unfused one gained
nothing (2^14 and 2^22 lost 9–10 % in the first run); phases fused at both ends gained at
2^18 (three phases of 64 = 8 x 8: −13 % and −10 %) but lost at 2^20 (+8 %: there only
phase 1, 64 = 8 x 8, fuses, and it reads a 16 MiB input beyond L2 without the gather's
prefetching). Adopted: a phase fuses both ends or neither, only when all its radices are
at most 8 and only for rows up to `grid_fuse_bytes` (a quarter of L2, so on the Mac up to
2^18 points in double); everywhere else the program is main's. Net on the Mac: 2^18 from
1.03 to 0.92 times FFTW, every other case unchanged. The copies are not where the time
goes at 2^20 and 2^22: their marginal cost is the memory traffic and page walks of rows a
page or more apart, which a fused pass pays as well; the target there needs fewer passes
over memory or cheaper ones (larger codelets, task 1.6; padding the work array so phases
read it without set conflicts), not fusion.

### x86 thresholds, by reasoning (not measured)

The defaults come from the cache sizes. On an EPYC 7763 (L1d 32 KiB, L2 512 KiB) they give:
a grid from 4096 points (double; 8192 float), the chain at 2048; blocks within 1024 points
of scratch, so two-phase blocks of 16 at 4096, 8 at 2^14 and 4 at 2^16; three phases from
`grid_three_bytes` = 64 KiB, that is from 8192 points; fusion up to 128 KiB (8192 points).
On a Xeon Gold 6240 (32 KiB, 1 MiB) the same except three phases from 2^14 and fusion to
2^14. The suspect one is the three-phase switch: on the Mac three phases began to pay only
when two-phase blocks fell to 8 columns with the row near a quarter of L2, while on these
machines the eighth-of-L2 rule switches at 8192–16384 points, where two-phase blocks are
still 8 to 16 wide and the row fits L2 twice over; a third pass over memory there probably
costs more than it saves. Expect `GRID_THREE_BYTES` near 1–2 MiB (switch at 2^16–2^17)
to win on x86; L3 (32 MiB per CCX) makes beyond-L2 rows cheaper than on the Mac. To check:

```bash
uv run --frozen python bench/fbench.py run A=fftn B=fftn+GRID_THREE_BYTES=1048576 \
  C=fftn+GRID_THREE_BYTES=never D=fftn+GRID_FUSE_BYTES=1 --tier tracking \
  --filter 'c2c-(4096|8192|16384|65536|262144|1048576|4194304)-b1'
```

## Verdict

Adopted on the Mac by the rule above: the grid is faster than the chain in every case,
by 29–53 %, far outside the band. Task 1.7's target (≤ 1.25 times FFTW from 2^16 to 2^22)
is met at 2^16 (out of place and in place) and 2^18, not at 2^20 (1.38) or 2^22 (1.43);
16384 is at 1.37. Task 1.7b fused the copies where that pays (phases of radices ≤ 8, rows
within a quarter of L2): 2^18 to 0.92, the rest unchanged (section above). Open: 2^14
(1.28–1.33), 2^20 and 2^22 (1.35–1.37 in the second run), the three-phase rule's thresholds
on x86 (the commands above), and a fbench run without load.
