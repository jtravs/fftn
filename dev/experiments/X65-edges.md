# X65: plan edges against a plain plan plus the user's own loop

- **Status:** running
- **Owner:** task S (5.1f), 2026-11-25 (overnight)
- **Phase / task:** 5.1f

## Question

Steps 5.1a and 5.1b put the edge vocabulary (DESIGN 12.1) into every strategy the rules
choose, fused into a pass only where a c2c row's chain takes a complex factor of stride 1, and
as a separate pass or phase everywhere else. Three questions decide what 5.1 still builds:

1. For each edge the rules build as a separate pass (real factors, strided factors,
   accumulation, factors on real and multi-axis plans, edges around the grid, the depth-first
   chain, the lanes and Bluestein), what does it cost against a plain plan plus the user's own
   separate loop, size by size?
2. Which of these separate passes should become fused edge variants?
3. What does an edge that forces a slower strategy cost?

Plus a regression check: plans without edges are unchanged by 5.1a and 5.1b.

Stored runs (5.1d, 5.1e) are not built yet; their part of X65 waits for them. A scale needs no
measurement: both edges' scales and `options::scale` are one product applied where the plan
already scales (5.1a), so normalisation costs nothing.

## Hypothesis

- A separate pass reads and writes the data once more. Where the data are in L1 or L2 that
  costs a few per cent of a transform of log n passes; where they are not, a separate pass is a
  whole sweep of memory, about one pass's worth (10–30 % of a long row or a large plane).
- The library's separate pass should be no slower than the user's loop, and faster where it
  runs through the workspace block while the data are still in cache (a c2c chain's output edge
  reads the block the last pass just wrote; the user's loop rereads the output).
- The fused variant (complex factor of stride 1 in a chain's first or last pass) should cost
  under 3 % over the plain plan: it is a multiply in a pass that is load-and-store bound.
- Accumulation through the user's own buffer costs a whole buffer's traffic more than the
  library's pass, so the library should win clearly there.

## Change under test

No library change. One build of this branch (the fbench options below; the library is `main`
d88a82f) runs three arms per edge configuration:

- **U** (baseline A): the plain plan and the user's own separate loop (`EDGE_ARM=loop`): the
  input's loop multiplies the input in place before the transform; the output's loop
  multiplies the output in place after it; accumulation transforms into a buffer of its own
  and adds it to the output. The loops are plain C++ on contiguous lines with explicit real
  arithmetic (vectorised by the compiler; cloned for AVX2 and AVX-512 on x86-64 Linux), shared
  over the executor's lanes in threaded arms.
- **L**: the plan with the edges (`EDGE_IN` or `EDGE_OUT`), as the rules build it.
- **P**: the plain plan, with no edge work at all: the floor that a fused variant approaches.

Factor arrays hold ones, so that fbench's check of the transform holds (a multiplication by
one costs what any other does; a run with every value 2 failed every case's check in both
arms, which shows both apply the factor). Accumulation zeroes the output before the check.

Edge configurations (one fbench run each):

| Run | Option | What it is | Cases |
|---|---|---|---|
| E1 | `EDGE_OUT=real.line` | a window on the output | all |
| E2 | `EDGE_OUT=complex.full` | a complex factor per element (a propagator) | c2c, r2c |
| E3 | `EDGE_OUT=acc` | accumulation | out of place |
| E4 | `EDGE_IN=complex.full` | a complex factor per input element (a prologue) | c2c, c2r |
| E5 | `EDGE_IN=real.line` | a window on the input | all |
| E6 | `EDGE_OUT=complex.s2` | a strided complex factor (stride 2 along the line) | c2c rows |

## Method

- **Cases:** catalogue `x65` (bench/catalogues/x65.toml): c2c rows of 64 to 2^20 points
  (batched below 16384, single above), batch leaves (16 × 1024), a mixed radix and a Bluestein
  length, real rows of 256 to 2^20 points, the client catalogue's rows (c2c 3072; r2c and c2r
  6144 × 66 and 32768; r2c 6400, 10240 and 10240 × 28), planes 64², 192² and 384² (the client's,
  in and out of place), real planes 256², volumes 64³, in double precision and a float subset.
  Tier `quick` (client shapes, long rows, planes and volumes) is the threaded subset.
- **Machines:** the Mac M5 Max (Homebrew Clang 23, neon_fcma; on AC power, one fbench at a time,
  `pmset -g batt` before and after each run); the EPYC 7763 (DMOG, GCC 16, avx2, Slurm shared
  `-c 8`); the Xeon 6240 (GCC 14.4, avx512, `--serial`, shared `-c 8`).
- **Mac timing, amended before any result was read** (the maintainer, 2026-11-25 night: Mac
  timing only when necessary, since heavy loads run the fans): the Mac's queue was stopped
  during its first run (E1, never analysed), and the Mac times only if the EPYC and the Xeon
  disagree on a Q2 verdict, as one short run of that kind's cases. Q2's "two of the three
  machines" then reads "both x86 machines, or one of them and the Mac's short run".
- **Thread modes:** one thread for E1–E6 on every machine; `THREADS=4` (a client width) for E2
  and E3 on tier `quick` on the EPYC (the Xeon's Slurm binds no CPUs).
- **fbench:** interleaved arms U, L, P and the control U~, process rounds (6 to 20), hot mode,
  median statistic. The noise band is the U~/U control's.
- **Metrics,** per case: L/U (Q1), P/U, and the edge's overhead over the plain plan
  O = L/P − 1, derived from the two paired ratios (its band the sum of theirs). Each case's
  `note` says where the edge went (fused, a separate pass or phase, and any strategy the rules
  would have chosen instead), which sorts the cases into the kinds below.
- **Kinds of separate pass** (by the note): (a) a real factor on a c2c row's chain; (b) a
  strided complex factor on a chain; (c) accumulation on a chain; (d) any factor or
  accumulation around a row's other strategy (grid, depth-first chain, lanes, Bluestein);
  (e) edge phases of real plans; (f) edge phases of multi-axis plans. The fused complex factor
  on chains (g) is the reference for what fusion achieves.

### Decision rules (written before any judging run)

**Q1, per edge configuration and machine**, over the cases whose edge is a separate pass or
phase: the separate pass **pays** if the geometric mean of L/U is below 1 − band; it is
**even** within ±band; any single case with L/U above 1.05 × (1 + band) (DESIGN 1.3's
tolerance) is a **defect**, listed with a proposed fix, since an edge must never be slower than
the user doing the work themselves.

**Q2, per kind (a)–(f):**
- a fused variant is **adopted for building** when the geometric mean of O over the kind's
  cases is at least 0.10 on at least two of the three machines, with O beyond its band on each
  case counted. If the variant is small (an edge mode of existing pass kinds, no generator
  change) it is implemented here and judged as its own arm against this branch (adopted per
  target when its geometric mean ratio is at most 0.97 beyond band over its cases with no case
  slower beyond band, and plans without edges compile to exactly `main`'s code); if large, it
  is proposed with drafted rules;
- the kind **stays a separate pass** when the geometric mean of O is below 0.05 on every
  machine;
- otherwise it is **deferred**: no variant now, re-judged on the client's own stages (5.2, 5.5).

**Q3:** every case whose note says the edge keeps the plan from the strategy the rules would
choose is listed with L/U and L/P. A case with L/U above 1 + band means the forced strategy costs
more than the user's loop; it gets a drafted rule (keep the rules' strategy with a separate
pass) for 5.1c.

**Regression check, per machine:** arms `fftn:8af3f30` (`main` before 5.1a, PR 128) and this
branch's plain plans, on catalogue `standard` tier `quick` and `x65` tier `full`, one thread.
Plans without edges are **unchanged** if every case's plan note (the per-axis lines) is
identical between the arms and no case is slower than 1.05 × (1 + band); a case beyond that
gets one `--serial` re-run that decides (AGENTS.md section 6). Known and parked: single tiny
transforms (c2c 8, 16, 64 × 1) at 1.06–1.08 on the EPYC and the Xeon since PR 128, with
unchanged plans and call chains.

## Results

Pending.

## Verdict

Pending.
