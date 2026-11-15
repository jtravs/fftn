# X16: `avx512` against `avx2`, with AVX-512's own cost table

- **Status:** done (2026-11-15): own table in float; `avx512` first in double; float's preference and tails go to phase 2
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

### Search campaign, Xeon Gold 6240, `ISA=avx512` (rule 1)

Campaign xeon-avx512-smooth (main 6abf186, GCC 14, 8 CPUs of a shared node, 7-smooth
n ≤ 1024, batches 1 and 64; the campaign's baseline arm is planner v1 with the nearest-C
AVX2 tables, i.e. arm (b)). Planner v1 against the best measured sequence, geometric means
over 133 lengths: 1.006 (b1 double), 1.021 (b1 float), 1.007 (b64 double), 1.019 (b64
float). `bench/x2/fit.py --isa avx512` fits tables that reproduce the measured best within
2 % in 91.5 % (double, C = 4) and 89.5 % (float, C = 8) of cases, against 81.0 % and 77.8 %
for the AVX2 tables; but the whole gap the fitted table could close is the 0.6–2.1 % above,
below rule 1's 3 %. Rule 1: not adopted; `avx512` keeps the nearest-C AVX2 tables. Radix
choice is therefore not a cause of FFTN's gap to FFTW on this Xeon (that is X18's question).

Rules 2 and 3 were first run on the X2 lengths 17–1024 (fbench job
fbench-20261114-130910-a3903ed6-d4db65; lesson 2.3's figures). They are decided below on the
planner refresh's builds.

### The refresh: arms (a), (b), (c) after X1, X3, X18 and X19 (rules 1–3)

Two runs on the Xeon Gold 6240 (GCC 14.4, Release, one thread, Slurm `-c 8` on the shared
node), the X2 catalogue's c2c rows:

- Run 20261115-002450-ca6f (Slurm 15612): (a) and (b) from main 6844ad0 (X1 merged), 600
  rows 17–32768, 80 rounds, band ±1.00 %, load average median 9.0. (b) ÷ (a): double 0.814
  (b1) and 0.878 (b64), float 1.026 and 1.073.
- Run 20261115-040151-b716 (Slurm 15636): (a) and (b) from fc0c27c (6844ad0 plus fit.py),
  (c) from fef4e0c (the refreshed AVX-512 tables, X2 "Refit after X1, X3, X18 and X19"),
  608 rows 17–65536, 71 rounds, band ±1.00 %, load average median 1.7; stopped after 4 h
  with 45 cases unresolved (ties).

| ÷ (a) `avx2` | double b1 | double b64 | float b1 | float b64 |
|---|---|---|---|---|
| (b) `avx512`, tables before | 0.810 | 0.877 | 1.019 | 1.073 |
| (c) `avx512`, refreshed tables | 0.804 | 0.872 | 0.975 | 1.022 |
| (c), 17–64 points | 1.162 | 1.202 | 1.640 | 1.664 |
| (c), 65–256 | 0.878 | 0.930 | 1.216 | 1.250 |
| (c), 257–1024 | 0.731 | 0.790 | 0.876 | 0.908 |
| (c), 1025–4096 | 0.644 | 0.757 | 0.641 | 0.727 |
| (c), 4097–65536 | 0.708 | 0.817 | 0.615 | 0.696 |
| (c), C does not divide n (double 56, float 78 lengths) | 1.001 | 1.046 | 1.199 | 1.225 |
| (c), C divides n (96, 74) | 0.708 | 0.783 | 0.783 | 0.845 |

What decides AVX-512 against AVX2 on this machine is whether C divides n (C = 4 in double,
8 in float), more than the length: float rows whose length is not a multiple of 8 take 1.20–
1.23 times AVX2's time, the others 0.78–0.85. At 60 points float takes 2.6 times AVX2's
time (run 15612), as on the cloud's Emerald Rapids and Zen 5 (2.9–3.1, X21): a lead for a
tail-tile or leaf problem in `avx512`, recorded here and not pursued.

## Verdict

For the Cascade Lake class only (Xeon Gold 6240); a verdict here does not carry to recent
Intel or AMD cores (X21: on Emerald Rapids `avx512` wins in double, 0.83–0.86, and slightly
in float, 0.95–0.99; on Zen 5 it wins in double, 0.87, and loses in float, 1.03–1.07).

1. **Rule 1, AVX-512's own table:** adopted in float ((c) ÷ (b) 0.957 at batch 1 and 0.953
   at batch 64, band ±1.00 %; the one case slower by more than 2 %, 10403 at 1.08, is
   overridden back to its previous plan). In double the table's geometric mean moves 0.6 %
   ((c) ÷ (b) 0.993, 0.994), below the rule's 3 %: double keeps avx2_float's costs, and its
   per-length overrides are adopted by X2's own rule (0.939 over the 14 lengths changed).
2. **Rule 2, the preference:** double: `avx512` stays first (0.804 and 0.872 of AVX2's time,
   clear of the band). Float: (c) beats (a) at batch 1 (0.975) but not at batch 64 (1.022),
   so by the rule `avx2` goes ahead of `avx512` for single precision on Cascade Lake, and
   the follow-up's answer is above: tails (lengths that 8 does not divide), not length as
   such. Not implemented here: `best_target` chooses per machine, not per precision or CPU
   generation, and a blanket `avx2` for float would give up 15–38 % on every float row that
   8 divides from 257 points up. The better rule is per plan (`avx512` where C divides n,
   else `avx2`), which is phase 2's cost model pricing both targets (X21 follow-up 1); until
   then the preference is unchanged and this verdict is the input.
3. **Rule 3, R × C transpose:** float lengths that C does not divide lose to (a) by 20–23 %
   under (c), above the rule's 10 %: work on tail-bound float on `avx512` is justified
   (an R × C transpose, or masked tails as icpx's code suggests, X22). Double: 0–5 %, no.

Lesson 2.3 in [LESSONS.md](../LESSONS.md) updated.
