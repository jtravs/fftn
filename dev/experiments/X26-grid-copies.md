# X26: long rows, the grid's copies

- **Status:** done (arm B rejected; arm C adopted on `avx2`)
- **Owner:** x26 session, 2026-11-15
- **Phase / task:** 1 (PLAN lead 1); follows X25, X12, X12b, X4

## Question

Can the grid's data movement (gathers, scatters and the twiddled scatter between its phases)
be made cheap enough in instructions to close most of FFTN's gap to FFTW on long rows
(≥ 16384 points, mixed 46080 and 100000, r2c and c2r of 65536 and 2²⁰), and which change
does it: cheaper copy loops, wider blocks, or fewer separate copies? The answer changes
`block_copy`, `twiddle_scatter` and the grid's block and fusion rules.

## Hypothesis

X25 measured, on the Xeon at 65536, 46080 and 100000, `block_copy` at 40–52 % of FFTN's
instructions and about 40 % of its cycles, with FFTN at 1.55–3.5 times FFTW's
instructions, a higher IPC and fewer L2 misses than FFTW: the copies are
instruction-bound, not memory-bound. With blocks of one vector per row (4 columns on
AVX-512, the Xeon's 65536, 46080 and 100000; the Mac's 100000 as well) the copy loop
spends about 33 instructions per 64-byte row: a row loop with a multiplied address, two
one-trip prefetch loops and a one-trip inner loop around one load and one store (the
compiled loop is in this file's results). The EPYC (blocks of 32 at 65536 in three phases,
4 at 100000) and the Mac (16–32) have wider rows, so less overhead per byte, but the same
structure.

1. **Cheaper copy loops** (arm B): a row loop with incremented pointers, the rows that
   prefetch separate from the last `rows_ahead` that do not, one prefetch per cache line,
   and rows of 1, 2, 4 or 8 vectors copied without an inner loop. Expected: about 6–8
   instructions per one-vector row against 33, so FFTN's instructions at 65536–100000 on
   the Xeon down by a third or more, and time by 10–25 % on x86 where IPC is already high
   (2.6–4.8); smaller on the EPYC's wide blocks and on the Mac. Results bit-identical (the
   same loads, multiplications and stores in the same order per element).
2. **Wider blocks or fewer copies** (arm C, chosen from option-only screens on top of B):
   the x86 block rule keeps a block's two scratch lines within L1 (blocks of 4 at 256-long
   phases on AVX-512), so every gathered row is one line and the copies pay per-row costs
   on 64 bytes. Either wider blocks (scratch within L2, which is private and fast on x86;
   `GRID_BLOCK_BYTES` = 2 or 4 times L1) or phases that read and write the arrays directly
   in their end passes (fusion, `GRID_FUSE_BYTES` beyond the row with `GRID_PHASE_RADIX=8`,
   which X12b left unused on x86 because the grid starts beyond the fusion limit there)
   should cut the remaining copy work. The Mac's shapes (blocks of 16 at 65536 and 46080,
   4 at 100000) are screened the same way.

FFTL.jl (the reference that beats FFTN on these rows on the Mac and the EPYC) runs a
single row of one thread as a plain Stockham chain of radix-4 passes (radix 8 up to 16384
with 32 registers), with no blocking and no copies (`src/factor.jl`, `src/threads.jl`: the
four-step is only for several threads). Its passes keep at most 8 streams in flight, so
they do not thrash L1's sets. That is a chain-or-grid question for the planner, recorded
here as a lead, not tested by this experiment.

## Change under test

- Arm A: `main` at the branch point (37582e5).
- Arm B: A plus cheaper `block_copy` and `twiddle_scatter` (`src/passes/grid.inl`); the
  plans are A's.
- Arm C: B plus the winner of the screens below, as a rule in the cache sizes (a profile
  field in `set_grid_sizes` or a grid rule in `src/strategies/grid.cpp`), if one wins.

No switch is added: the screens use fbench's existing profile options (`GRID_BLOCK_BYTES`,
`GRID_FUSE_BYTES`, `GRID_PHASE_RADIX`; bench/README.md, "FFTN options").

## Method

- Cases: the standard catalogue's tracking tier (target cases and collateral damage).
  Targets: c2c b1 16384, 65536 (out of place and in place), 262144, 1048576, 4194304,
  46080, 100000; r2c and c2r 65536 and 1048576 (`--filter
  '(c2c-(16384|65536|262144|1048576|4194304|46080|100000)|r2c-65536|c2r-65536|r2c-1048576|c2r-1048576)-b1'`).
- Machines: the Mac (M5 Max, `neon_fcma`, Homebrew Clang, shared and loaded; one fbench run
  of this experiment at a time); the EPYC 7763 (DMOG, `avx2`, GCC 16, Slurm `-c 8`
  shared); the Xeon Gold 6240 (`avx512`, GCC 14, `-c 8` shared). One thread, hot, double
  and float as the catalogue has them. fbench A/B with interleaved rounds and the A/A
  control; `--serial` on the remote machines.
- Counters: `--pmu cycles,instructions` on the Xeon and the EPYC for the target cases, A
  against B, to confirm that the instructions drop where intended.
- Metric: median paired ratio to the previous arm per case with its 95 % interval;
  geomean over the target cases; FFTN ÷ FFTW (arm A and the adopted arm against FFTW in
  the same run).
- Decision rule, written before running, per machine and target (the release
  toolchains: GCC on x86-64 Linux, Clang on macOS; AGENTS.md section 6):
  1. An arm is adopted on a target when its geomean over the target cases, against the
     previous arm, lies below 1 − band (the run's A/A band) with its interval clear of
     it, and no tracking-tier case is slower than 1 + max(band, 3 %). A case just past
     that limit on a loaded run is re-run before it decides.
  2. Arm B is bit-identical by construction and changes no plan: adopted everywhere if
     it passes rule 1 on at least one target and loses (by rule 1's collateral limit) on
     none.
  3. Arm C is a plan rule: adopted per target where it passes rule 1 against B; where it
     does not, the old rule stays for that target (gated by the cache rule or by target).
  4. At most three arms; a failed approach is written up with its numbers and stops.

## Results

### Arm B: the copy loops' instructions go, the time does not follow on x86

Arm B's first form (8ac713f) and its second (d232d7a, which prefetches only the side of a
copy whose rows lie apart: the other is the tight scratch block) cut the instructions as
intended. Counters per transform (`--pmu cycles,instructions`, user mode, median of three
process rounds, A then B interleaved in one job; raw output in the campaign directories
`fftn-x26/pmu1` on each machine):

| case (default plan) | Xeon `avx512` instructions B ÷ A | EPYC `avx2` instructions B ÷ A | EPYC cycles B ÷ A |
|---|---|---|---|
| c2c 65536 (Xeon blocks of 4; EPYC 32) | 0.63 | 1.02 | 1.00 |
| c2c 262144 (32; 32) | 0.76 | 1.03 | 0.99 |
| c2c 2²⁰ (16; 16) | 0.73 | 0.79 | **1.39** |
| c2c 2²² (8; 8) | 0.67 | 0.74 | 0.87 |
| c2c 46080 (4; 32) | 0.63 | 1.02 | 1.01 |
| c2c 100000 (4; 4) | 0.66 | 0.73 | 0.79 |
| r2c / c2r 2²⁰ (16; 16) | 0.75 | 0.80 | **1.51–1.55** |

The Xeon's cycles were not usable (node load average 39–50 from other users; the unchanged
r2c 65536 chain moved by 35 %). On the EPYC (quiet node) the rows of 2²⁰ took up to 1.55
times the cycles with a fifth fewer instructions. Forcing the block width
(`GRID_BLOCK`) at 65536, 262144 and 2²⁰ on the EPYC isolated it: every faster loop
(compile-time row widths or not, prefetching both sides, one side, none, 16 or 32 rows
ahead, with locality hints 2 or 0) took 1.1–2.2 times main's cycles at blocks of 4–16
columns from 2¹⁸ points, and the same as main at blocks of 32 (16 registers a row, where
every loop is long). Variants, all diagnostic commits on unpushed `exp/x26-dev-*`
branches (EPYC, `avx2`, cycles ÷ main's at 262144 with blocks of 8 / 16; 2²⁰ with 8 / 16):

| variant | 262144 | 2²⁰ |
|---|---|---|
| main without the copies' prefetch (c30ca97) | **0.67–0.76 / 0.85–0.86** | **0.90–0.96 / 0.95–1.01** |
| main, prefetch with locality 2 (e63eadf) | 1.00 / 1.04 | 1.02 / 1.02 |
| main, read prefetch only, non-temporal (e235581) | 1.05 / 0.92 | 0.99 / 0.93 |
| B, second form (d232d7a) | 1.76–2.17 / 1.81–1.92 | 1.56–1.60 / 1.36–1.43 |
| B without prefetch (5916520) | 1.44–1.63 / 1.06–1.14 | 1.26–1.45 / 1.07–1.12 |
| B without compile-time widths (e69749c) | 2.43 / 1.85 | 1.61 / 1.40 |
| B with main's `twiddle_scatter` (7d8fa83; without prefetch 54b6eb4) | 1.79 / 1.91 (1.41 / 1.15) | 1.55 / 1.38 (1.41 / 1.06) |

Counters for main without prefetch against B without prefetch at 262144, blocks of 8
(Zen 3 raw events): L1 data-cache accesses 6.8 M against 10.9 M, demand fills from beyond
L2 0.53 M against 1.03 M, L2 prefetch hits 0.63 M against 1.11 M, for 22.3 M against 18.3 M
instructions. A copy-only microbenchmark of the same gathers and scatters (no lane passes
between them) timed the two loops the same. Reading: every gathered row lies a multiple of
4 KiB from the last, so all of a block's source lines fall into one or two L1 sets (8
ways); a loop that issues rows faster keeps more of them in flight, they evict the
scratch block that the lane passes then read again, and the misses come back (twice the
fills). Main's slow loop throttled this by accident, and its prefetch eight rows ahead
added eight more lines to the same sets: removing the prefetch from main's loop is what
made it faster. The x86 copies are bound by L1 set conflicts of power-of-two row strides,
not by their instructions; X25's instruction share was real but not the limit.

### The factorial: loop form against prefetch, per target

After the diagnosis the screens of wider blocks and fused phases on x86 (queued on arm B's
first form) were cancelled: the copies were not instruction-bound there, so they no longer
tested the hypothesis. Arm C became the change the diagnosis pointed to: no prefetch in
the copies. One fbench run per machine timed main (A), main without the copies' prefetch
(A0, c30ca97, the same program as arm C on `avx2`), arm B (d232d7a) and arm B without
prefetch (P0, 5916520) against FFTW; target-case geomeans against A (a filtered copy of
each run's store through `fbench show`):

| machine, target (run, rounds, A/A band) | A0 | B | P0 |
|---|---|---|---|
| EPYC `avx2`, GCC 16 (20261115-202504-2e8f, 27 rounds, node load 66 of 128 CPUs, ±1.00 %) | **0.920 [0.916, 0.924]** | 1.050 [1.045, 1.057] | 0.935 [0.932, 0.938] |
| Xeon `avx512`, GCC 14 (20261115-202444-a93c, 20 rounds, 9 loaded, ±1.95 %) | 1.043 [1.013, 1.063] | 0.988 [0.965, 1.007] | 1.032 [1.011, 1.050] |
| Mac `neon_fcma`, Clang 23 (20261115-201701-02d0, tracking tier, 80 rounds, 79 loaded, ±1.23 %) | 1.010 [1.004, 1.015] | 0.986 [0.982, 0.992] | 1.000 [0.995, 1.004] |

Per case on the EPYC, A0 ÷ A: c2c 65536 0.918, 262144 0.886, 2²⁰ 0.966, 2²² 1.013, 46080
0.932, 100000 0.875; r2c 65536 0.890, 2²⁰ 0.834; c2r 65536 0.920, 2²⁰ 0.880; 16384 (a chain,
code unchanged) 1.024. B there: 2²⁰ 1.40, r2c and c2r 2²⁰ 1.50–1.56, 100000 0.78, 2²² 0.86.
P0: 2²⁰ 1.10, c2r 2²⁰ 1.06, 100000 0.78. On the Xeon the prefetch pays (A0: 2²⁰ 1.16,
2²² 1.22, c2r 2²⁰ 1.06) and B is within the band (100000 0.95, r2c 2²⁰ 0.95, nothing slower).
On the Mac A0 and P0 lose at 2²⁰ and 2²² (1.05–1.19) and gain on the real 2²⁰ rows
(0.90–0.94); B gains 1.4 % (100000 0.94, 65536 0.98, nothing slower in the tracking tier),
as in run 20261115-192335-aecf with B's first form (0.986 [0.980, 0.995], band ±1.65 %).

FFTN ÷ FFTW on the EPYC, main → arm C (same run): c2c 65536 1.41 → 1.30, 262144 1.48 →
1.31, 2²⁰ 1.11 → 1.07, 2²² 1.13 → 1.14, 46080 1.62 → 1.51, 100000 1.94 → 1.70; r2c 65536
1.29 → 1.15, 2²⁰ 1.77 → 1.48; c2r 65536 1.37 → 1.26, 2²⁰ 1.60 → 1.41.

Mac screens on top of B (run 20261115-200931-6fcb; the filter ran on the quick tier, so
only 16384, 65536 and r2c/c2r 65536): blocks within 2 · L1 (`GRID_BLOCK_BYTES=262144`)
1.205, phases of radices ≤ 8 fused (`GRID_PHASE_RADIX=8`, `GRID_FUSE_BYTES=never`) 1.058,
fusion beyond L2/4 alone 1.001, no prefetch 1.023. None beats the default.

A check of arm C's program on the Xeon under `ISA=avx2` and a tracking-tier run of arm C
(5772cc4) on the EPYC for collateral damage are listed under the verdict.

## Verdict

- **Arm B (cheaper copy loops) rejected** by rule 1 on every target: within the band on
  the Mac (0.986, twice, against limits 0.988 and 0.984) and the Xeon (0.988), and slower
  on the EPYC (1.050; 2²⁰ and the real 2²⁰ rows 1.40–1.56). Reverted (e57e4d4, 67a2982).
  It cut the instructions as X25 asked (Xeon −24 to −37 %, EPYC −20 to −27 % where the rows
  are narrow), and the time did not follow: on x86 the copies are bound by L1 set
  conflicts of rows a multiple of 4 KiB apart, which a faster loop worsens.
- **Arm C (no prefetch in the copies) adopted on `avx2`** (5772cc4, `copies_prefetch` in
  `src/passes/grid.inl`): EPYC target geomean 0.920 [0.916, 0.924], every target case
  0.83–0.97 except 2²² (1.013, inside the collateral limit), and nothing else runs the
  changed code (only grid plans, all of them targets, on the EPYC). The `avx512` and NEON
  programs are unchanged: the prefetch pays there (Xeon 1.04, Mac 1.01 without it).
  Results are bit-identical.
- Wider blocks and fused phases were not measured on x86; on the Mac both lose.

Unverified at the time of writing: the tracking tier of arm C on the EPYC (submitted:
`fbench_remote.py submit epyc A=fftn:37582e5 C=fftn:5772cc4 W=fftw -- --tier tracking
--budget 3600`) and `ISA=avx2` on the Xeon (`... submit xeon A=fftn:37582e5+ISA=avx2
A0=fftn:c30ca97+ISA=avx2 W=fftw -- --tier tracking --filter <targets> --budget 1800`): if
Intel's AVX2 rows want the prefetch, the switch belongs to the core, not the target.

What it leaves for lead 1: the copies' real cost on x86 is L1 set conflicts, which no
loop form removes. Next: rows that are not a multiple of 4 KiB apart (a padded work
array for phases after the first, X4's open x86 half; the caller's input and output
cannot be padded, so two-phase grids need a workspace for it), or a different long-row
plan. FFTL.jl, which beats FFTN here, runs one thread's long row as a plain Stockham chain
of radix-4 (8 to 16384) passes with no copies at all: on the Mac, FFTN's chain forced to
4⁸ at 65536 took 237 µs against the grid's 252 and FFTW's 269 (one process each, loaded
machine; FFTN's default chain 8 × 8 × 8 × 8 × 16 took 396 µs). A chain-or-grid rule that
prices small-radix chains for long rows is the follow-up (X26b).
