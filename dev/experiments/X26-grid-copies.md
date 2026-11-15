# X26: long rows, the grid's copies

- **Status:** running
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

(pending)

## Verdict

(pending)
