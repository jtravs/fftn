# X32: long powers of two as copy-free chains

- **Status:** running
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

(pending)

## Verdict

(pending)
