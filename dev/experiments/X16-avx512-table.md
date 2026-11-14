# X16: `avx512` against `avx2`, with AVX-512's own cost table

- **Status:** planned
- **Owner:** lead session, 2026-11-14 (tooling written with this file)
- **Phase / task:** 1.3 / 1.4

## Question

On AVX-512 machines, does `avx512` beat `avx2` once its radices are chosen with costs
measured on `avx512`, and does it still need an R × C transpose for leading radices below
C? The answer decides `avx512`'s cost table, whether `avx512` stays ahead of `avx2` in
`best_target`'s preference, and whether the R × C transpose (DESIGN.md 6.2) is built.

## Hypothesis

Planner v1 (DESIGN.md 11.2) runs `avx512` on AVX2's tables of the nearest C: double
(C = 4) on the AVX2 float table, float (C = 8) on it too, with C = 8 in the unit and tail
counts. Those costs were fitted on 16 registers of 256 bits (X2, EPYC 7763). With 32
registers of 512 bits the per-unit costs of large radices and the price of a tail differ,
so we expect the nearest-C choice to be off by several per cent at many lengths, most in
float, where a leading radix below 8 makes the grouped pass store through its tile and
the next pass run only tails (DESIGN.md 6.2, decision log 2026-11-14). With its own
table the planner should avoid those sequences wherever a leading radix of 8 or 16 exists;
lengths that have none (C ∤ n, or n with few factors of 2) stay tail-bound, and those are
where an R × C transpose could still matter.

## Change under test

Three arms, one build of each commit, all forced with the fbench FFTN backend's `ISA=`:

- (a) `ISA=avx2`, planner v1 as merged (commit M, the one the campaigns run from).
- (b) `ISA=avx512`, planner v1 as merged: the nearest-C table (commit M).
- (c) `ISA=avx512` with AVX-512's own fitted tables (commit T: M plus the rows
  `bench/x2/fit.py` writes, added to `measured_tables` in `src/planner/rules.cpp`; nothing
  else changes).

No switch: arm (c) is a commit; `dev/KNOBS.toml` is untouched.

## Method

- Cases: the X2 catalogue (`bench/catalogues/x2.toml`, tier full: c2c rows, forward, out
  of place, double and float, batches 1 and 64; 668 cases). Collateral: the standard
  catalogue's tracking tier.
- Machine: Xeon Gold 6240 (AVX-512, 36 cores, shared and loaded outside Slurm), profile
  `xeon` of `~/.config/fftn/remote.toml`, GCC 14, Release, one thread. The machine is
  never quiet: every comparison is one fbench run with interleaved arms, repeated process
  rounds and the A/A control, and is judged against that run's band; fbench records the
  load of every round. A cloud Zen 4 or Zen 5 (AVX-512 on 256-bit units) is optional later.
- Step 1, search campaigns from commit M (the data for the fit, and the ceiling: the
  campaign's baseline arm is planner v1 on that target, so `search.py report` gives
  time(v1) ÷ time(best sequence) per case for (b) and for (a)):

  ```bash
  R="uv run --frozen python scripts/remote/fbench_remote.py"
  $R submit xeon fftn --campaign bench/x2/search.py --time 12:00:00 -- \
      all --name xeon-avx512-smooth --lengths 2-1024 --isa avx512
  $R submit xeon fftn --campaign bench/x2/search.py --time 12:00:00 -- \
      all --name xeon-avx2-smooth --lengths 2-1024 --isa avx2
  # Later, the same with --name xeon-avx512-long / xeon-avx2-long --lengths 1025-
  # (`uv run --frozen python bench/x2/search.py remote xeon --isa avx512` prints all four.)
  $R fetch <job>    # each job: runs into bench-out/runs/, state into bench-out/campaigns/
  uv run --frozen python bench/x2/search.py report --name xeon-avx512-smooth
  ```

- Step 2, the fit (method: X2 "The fit"; tool: `bench/x2/fit.py`):

  ```bash
  uv run --frozen python bench/x2/fit.py --name xeon-avx512-smooth \
      --compare f64=avx2_float,f32=avx2_float --out bench-out/x16-avx512-table.txt
  uv run --frozen python bench/x2/fit.py --name xeon-avx2-smooth --table xeon_avx2 \
      --compare f64=avx2_double,f32=avx2_float   # does EPYC's AVX2 table carry to the Xeon?
  ```

  The output's tables and overrides go into `rules.cpp` with the rows
  `{target::avx512, 4, &avx512_double}` and `{target::avx512, 8, &avx512_float}` in
  `measured_tables`: commit T. Its reproduction rate (cases within 2 % of the measured
  winner) is reported next to the nearest-C table's on the same cases.

- Step 3, the comparison, one run of the three arms from commits M and T:

  ```bash
  $R submit xeon a=fftn:M+ISA=avx2 b=fftn:M+ISA=avx512 c=fftn:T+ISA=avx512 \
      --time 12:00:00 -- --catalogue bench/catalogues/x2.toml --tier full --budget 21600
  $R submit xeon a=fftn:M+ISA=avx2 b=fftn:M+ISA=avx512 c=fftn:T+ISA=avx512 \
      --time 06:00:00 -- --tier tracking --budget 10800
  ```

- Metric: fbench's paired ratios against arm (a), per case with 95 % intervals, and the
  geometric mean over the X2 lengths per batch (1, 64) and precision; c ÷ b is the ratio
  of the two `avx512` arms' ratios to (a) (both in the same run). Noise floor: the run's
  A/A band.
- Decision rule, written before running:
  1. Adopt (c) (AVX-512's own table) if its geometric mean over the X2 lengths beats (b)
     by at least 3 % at batch 1 and at batch 64, in each precision where it is applied,
     with the A/A band below 3 %, and no X2 or tracking case is more than 2 % slower than
     under (b) outside the band. A precision that does not clear it keeps the nearest-C
     table.
  2. `avx512` stays first in `best_target`'s preference on AVX-512 machines only if the
     adopted `avx512` arm ((c), else (b)) beats (a) in geometric mean at batches 1 and 64
     with the interval clear of the band. Otherwise `avx2` goes ahead of it and a
     follow-up asks why.
  3. Start R × C transpose work only if, under (c), the tail-bound lengths (those whose
     chosen first radix C does not divide, or where C does not divide n) still lose to (a)
     by more than 10 % in geometric mean.

- Caveat, recorded before running: the Xeon Gold 6240 is Cascade Lake, where 512-bit
  instructions lower the core clock (AVX-512 licence levels), and FFTW built with both
  `--enable-avx2` and `--enable-avx512` was often fastest on it with AVX2 codelets (the
  maintainer's earlier measurements; FFTW's MEASURE planner chooses per transform). So a
  verdict here holds for Cascade Lake only: the same three arms run on a recent Intel Xeon
  (Sapphire Rapids or newer) in the cloud, with MKL alongside, before `best_target`'s
  preference changes for every AVX-512 machine. If the winner differs by length rather than
  by machine, the choice of instruction set per plan belongs to phase 2's cost model
  (`avx2` and `avx512` priced side by side), not to a fixed preference.

## Results

Not run.

## Verdict

Pending.
