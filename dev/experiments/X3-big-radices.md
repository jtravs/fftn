# X3: few big passes (radix 32 and 64) against radix-16 chains and grids

- **Status:** running
- **Owner:** task 1.6 session (r3264), 2026-11-14
- **Phase / task:** 1.6

## Question

Do passes of radix 32 and 64 (generated, ordered codelets, X8) beat the planner's radix ≤ 16
choice for rows of 512 to 65536 points, as chain passes or as grid phases, on any target and
precision? The answer decides whether planner v1's cost tables get radix 32 and 64 columns
(DESIGN.md 11.2) and whether the grid's phase rule may use them (DESIGN.md 8.5).

## Hypothesis

FFTW plans 1024 as 32 × 32 and 4096 as 64 × 64 on the Xeon (X18 item 4), and FFTN's radix
≤ 16 needs three passes at 512–4096 and four at 16384. Fewer passes mean fewer trips over
memory and less per-pass overhead, and the arithmetic per point and bit is nearly the same
(vocabulary operations per point per log₂ R of the twiddled codelet: 1.55 at 16, 1.62 at
32, 1.67 at 64). Against that, X8 counted stack spills on every target: radix 64 stores
67–681 values to the stack per pass, radix 32's grouped and lane passes 76–222, worst on
AVX2 (16 registers). The prototype's radix 20–32 passes lost everywhere. Expected:

- AVX-512 (32 registers of 512 bits): radix 32 passes win at 512–2048 and in grid phases;
  radix 64 at best ties (X18's expectation: 512–2048 and 16384 to about 1.0 of FFTW).
- AVX2 (16 registers): both lose, radix 64 badly.
- NEON (32 registers of 128 bits, C = 1 in double): radix 32 may win in double, where the
  32-point leaf already beat the chain by 16–22 % (X8).

## Change under test

No library default changes in this first measurement: every arm is the same build of
commit M (this branch's first commit), with internal build settings forced through the
fbench FFTN backend (bench/README.md, "FFTN options"; none of them is a switch of the
library, so `dev/KNOBS.toml` is untouched):

- A: the public plan (planner v1's chain, or the grid beyond L1).
- `cR1.R2...`: the chain forced to a radix sequence containing 32 or 64 (`RADICES`), with
  no more passes than the fewest a radix-16 chain needs: the best four by X2's cost
  estimate padded to each C of the target (bench/x3/run.py).
- From 4096 up: `chain` (the planner's radix-16 chain, `STRATEGY=chain`), `grid` (the grid
  with the radix-16 phase rule, `STRATEGY=grid`), `g64` (the grid's phase rule allowed
  radices up to 64: `GRID_PHASE_RADIX=64`, a build setting added for this experiment),
  and `gN1.N2[.N3]` (that phase rule on the two two-phase splits and the three-phase split
  with the fewest passes, `GRID_FACTORS`).

If a big radix wins, the second step is the planner change (commit T): cost-table columns
for 32 and 64 fitted with bench/x2/fit.py from an X2-style search campaign that includes
them, and/or the phase rule's largest radix, measured A/B against M.

## Method

- Cases: bench/catalogues/x3.toml: c2c rows of 512, 1024, 2048, 4096, 8192, 16384, 32768
  and 65536 points, forward, out of place, batches 1 and 64, double and float (32 cases).
- Driver: `bench/x3/run.py --isa T`, one fbench run per length (interleaved process rounds,
  6 to 30 valid rounds by fbench's stopping rule, 1200 s budget, the A/A control).
- Machines, compilers, instruction sets, one thread:
  - Xeon Gold 6240 (Cascade Lake), GCC 14.4, Slurm `-c 8` on the shared node: `avx512`
    and `avx2`;
  - EPYC 7763 (DMOG node16–21), GCC 16, Slurm `-c 8 --hint=nomultithread`: `avx2`;
  - Apple M5 Max, Homebrew Clang, the shared Mac: `neon_fcma`.
  The Xeon and the Mac are loaded by other work; every claim is judged against its run's
  A/A band.
- Metric: fbench's median paired ratio arm ÷ A per case with its 95 % interval.
- Decision rule, written before running:
  1. *Screen (this measurement).* A big radix is worth a planner change for a target and
     precision when its best big-radix arm beats A by more than 3 % with the interval clear
     of the band in at least two of the eight lengths at one batch or more, as a chain
     (lengths where A is a chain) or as a grid (lengths where A is a grid). Otherwise
     radix 32 and 64 stay out of the planner for that target and precision, and the
     result is recorded.
  2. *Adoption (commit T against M, interleaved A/B on the same machine).* Adopt the
     change for a target and precision if it improves the geometric mean by at least 2 %
     over the affected lengths (chains: the X2 catalogue's lengths that are multiples of
     32; grids: the X3 lengths from the grid threshold up) at batches 1 and 64, and no
     case is slower than the run's band plus 2 %. Elsewhere planner v1's choices stay
     exactly as they are (`tests/core/data/rule_radices.txt` changes only where 32 or 64
     is chosen).

## Results

Pending.

## Verdict

Pending.
