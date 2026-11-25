# X65: plan edges against a plain plan plus the user's own loop

- **Status:** done (no variant adopted; proposals P1–P5 for the maintainer)
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

### Arm V: the separate pass vectorised (declared before its run)

The EPYC's first run (E1) showed the library's separate pass slower than the user's loop on
almost every c2c row (a defect by Q1's rule). The edge pass was baseline scalar code with
run-time strides. Commit 7bbed5b specialises lines of unit stride (factor stride 0 or 1),
which the compiler vectorises, and skips a scale of 1 (bitwise the same results). It touches
only `src/core/edges.cpp`, code that plans without edges never run.

Deciding runs, on the EPYC and the Xeon, one thread, catalogue `x65` tier `full`: arms
`A=fftn:a88a6e3+EDGE` (before) and `V=fftn:7bbed5b+EDGE` (after) plus the control, for E1, E3
and E5 (the configurations whose separate passes it changes everywhere). V is adopted on a
machine when the geometric mean of V/A over the cases whose edge is a separate pass or phase
is at most 0.97 beyond band and no case is slower than 1 + band; Q1 and Q2 are then judged on
V's code, with E1–E6 re-run on V where V is adopted.

## Results

Library `main` d88a82f (this branch's library), fbench options at a88a6e3. EPYC 7763: DMOG,
GCC 16.2, avx2, Slurm shared `-c 8` (nodes loaded: load 10–48 during the runs). Xeon 6240:
GCC 14.4, avx512, Slurm shared `-c 8`, `--serial`. 20 valid rounds per run. Run IDs are the
fbench store's (`bench-out/runs/`, fetched from each machine's job directory). The Mac's E1
run (20261125-224252-d920) was stopped after four rounds and never analysed; no Mac timing
was needed (the x86 machines agree on every verdict).

Per kind (the note's placement of the edge, Method): geometric means of **L/U** (the
library's edges ÷ the plain plan plus the user's loop) and **L/P** (÷ the plain plan alone);
the user's loop itself cost P/U = 0.32–0.97, that is 3 % to twice the transform's time, and
10–33 % on most shapes. Kinds: (a) real factor, c2c chain; (b) strided complex factor, chain;
(c) accumulation, chain; (d) any edge around another row strategy; (e) real rows' phases;
(f) multi-axis phases; (g) the fused complex factor.

| Run (edge) | Machine, run ID, band | (a)/(b)/(c) | (d) | (e) | (f) | (g) fused |
|---|---|---|---|---|---|---|
| E1 out real.line | EPYC 20261125-224449-f22b, 1.9 % | a 1.29 / 1.52 | 1.04 / 1.17 | 1.05 / 1.18 | 1.06 / 1.18 | |
| | Xeon 20261125-224625-83ad, 3.0 % | a 1.34 / 1.84 | 1.08 / 1.25 | 1.18 / 1.37 | 1.06 / 1.26 | |
| E2 out complex.full | EPYC 20261125-224449-a721, 2.4 % | | 1.13 / 1.35 | 1.18 / 1.37 | 1.14 / 1.34 | 1.11 / 1.38 |
| | Xeon 20261125-225839-e471, 2.1 % | | 1.15 / 1.44 | 1.26 / 1.64 | 1.06 / 1.43 | 0.96 / 1.55 |
| E3 out acc | EPYC 20261125-225150-f7c2, 2.6 % | c 1.06 / 1.39 | 1.04 / 1.27 | 0.98 / 1.18 | 0.98 / 1.19 | |
| | Xeon 20261125-230758-0fcd, 2.2 % | c 0.82 / 1.50 | 0.95 / 1.47 | 0.84 / 1.16 | 0.83 / 1.13 | |
| E4 in complex.full | EPYC 20261125-225938-73aa, 1.0 % | | 1.12 / 1.35 | 1.18 / 1.38 | 1.07 / 1.27 | 1.10 / 1.37 |
| | Xeon 20261125-231828-98ff, 2.6 % | | 1.26 / 1.65 | 1.43 / 1.83 | 0.95 / 1.29 | 1.06 / 1.71 |
| E5 in real.line | EPYC 20261125-230623-e893, 1.7 % | a 1.23 / 1.44 | 1.11 / 1.25 | 1.07 / 1.20 | 1.04 / 1.15 | |
| | Xeon 20261125-232713-8760, 2.0 % | a 1.30 / 1.80 | 1.21 / 1.51 | 1.25 / 1.45 | 0.97 / 1.18 | |
| E6 out complex.s2 | EPYC 20261125-224512-a51f, 1.3 % (16 rounds) | b 1.41 / 1.82 | 1.11 / 1.37 | | | |
| | Xeon 20261125-233749-813a, 1.4 % | b 1.44 / 2.33 | 1.15 / 1.72 | | | |
| T2 E2 at 4 threads | EPYC 20261125-225551-bb9f, 1.4 % | | | 1.06 / 1.32 | 1.11 / 1.44 | 0.88 / 1.11 |
| T3 E3 at 4 threads | EPYC 20261125-225938-9258, 5.7 % | c 1.00 / 1.19 | | 0.96 / 1.29 | 0.85 / 1.24 | |

Where the cost lies, case by case (the stored runs have every case):

- **Small batched rows.** c2c 16 × 1024 and 64 × 64: L/P 2.0–7.8 in every configuration and on
  both machines, fused or separate (E2's fused factor: EPYC 4.5 and 2.0, Xeon 6.3 and 2.6),
  against the user's loop at P/U 0.32–0.73. c2c 256 × 64: L/P 1.3–2.1. These plans run each
  row through the program's walked form: a factor cursor (the edge driver runs only the walked
  form) or a separate edge stage per row takes away the flat, unrolled and leaf-row forms
  that plain small rows use, and batch leaves run one row at a time (64 × 64 on the EPYC,
  `describe()`: "batch leaves, one row at a time between the edge passes").
- **Long rows.** At 2^18–2^20 points the fused factor costs about what the user's loop does
  (E2 c2c 1048576: L/U 0.99 EPYC, 1.02 Xeon), and a separate real factor 1.15–1.20 of it
  (E1 EPYC). The separate pass through the workspace block costs a whole extra sweep.
- **Around other row strategies** (the Xeon's depth-first chains and grids from 16384 points):
  separate input passes cost 1.21–1.26 of the user's loop (E4, E5), output passes 1.08–1.15.
  On the Xeon, c2r rows with any input factor lose the depth-first chain (Q3) and cost 1.5–1.9
  of the user's loop.
- **Accumulation** pays on the Xeon (L/U 0.82–0.95 by kind) and is even on the EPYC: the
  user's version needs a buffer of its own, which costs a sweep.
- **Real and multi-axis phases** cost 1.04–1.26 of the user's loop with a factor, 0.83–0.98
  with accumulation.
- **Threads (EPYC, 4):** the same picture; one defect stands out, r2c 6400 × 1 with
  accumulation at L/U 2.33 (L 18.7 µs, P 8.5 µs, U 9.7 µs), where the edge phase of a single
  row seems to serialise the team.

**Arm V** (the separate pass vectorised, 7bbed5b), EPYC, A = a88a6e3, one thread:

| Run | Run ID, band | V/A over the separate-pass cases | Cases slower beyond band |
|---|---|---|---|
| E1 | 20261125-230033-bf28, 3.2 % | 0.992 | none |
| E3 | 20261125-230112-fbb7, 6.0 % | 0.994 | none |
| E5 | 20261125-230121-4172, 3.4 % | 0.996 | c2c 64 × 64, 1.167 |

GCC 16 vectorised the new loops (SSE2, versioned for aliasing), so the arithmetic was not the
cost. V misses its rule on the EPYC; its Xeon runs were cancelled (a machine that rejects is
enough for a change of baseline code that is the same on every target).

**Regression check** (plain plans, `O` = 8af3f30, `N` = a88a6e3, one thread):

| Machine | Catalogue, run ID, band | Geomean N/O | Worst case | Plans identical |
|---|---|---|---|---|
| EPYC | x65 full, 20261125-224745-75b7, 1.2 % | 1.002 | r2c 32768: 1.036 [0.986, 1.054] | 45 of 45 |
| EPYC | standard quick, 20261125-225348-3893, 1.3 % | 0.996 | c2c 16 × 1: 1.023 [1.013, 1.037] | 24 of 24 |
| Xeon | x65 full, 20261125-234152-8a1e, 1.2 % | 0.998 | c2r 64³: 1.011 [1.001, 1.037] | 45 of 45 |

## Verdict

**Status: done; no library change adopted. Arm V rejected and reverted (a5aa3d5).** The
fbench options (`EDGE_IN`, `EDGE_OUT`, `EDGE_ARM`) and catalogue `x65` stay for the re-runs.

**Q1 (the separate pass against the user's loop).** By the rule, the separate passes **lose**
in every factor configuration (E1, E2, E4, E5, E6) on both machines: geometric means of L/U
1.04–1.44 by kind, and defects (L/U above 1.05 × (1 + band)) on small batched rows in every
configuration, on c2c chains with a real or strided factor at every size, and on most real
rows with a complex factor. Accumulation is **even** on the EPYC and **pays** on the Xeon.
The defects' cause is structural (Results), so the fix proposed is P1 below, not a change to
the pass.

**Q2 (fused variants).** Every kind (a)–(f) has O = L/P − 1 of at least 0.10 on both
machines, which by the rule's letter adopts a fused variant of each for building. The data
argue against building them first: the fused complex factor (g) costs as much as the separate
passes where the cost is largest (small rows, L/P 2.0–7.4), and arm V showed the pass's own
work is not the cost. The proposed order, **the maintainer's decision** since it departs from
the rule's letter:

- **P1, batched forms for plans with edges** (prerequisite): the edge driver gets the flat,
  unrolled and leaf-row forms, moving the cursors per row; a c2c plan of rows takes its
  separate edges as an edge phase over each block of rows (as real and multi-axis plans do,
  5.1b-1) instead of a stage per row; batch leaves take edges (the fused factor in the batch
  leaf, separate work as a phase over the leaf's block). Drafted rule: re-run E1–E5 on the EPYC
  and the Xeon; adopt when the geometric mean of L/P over the c2c and real rows of fewer than
  4096 points falls to at most 0.80 of today's on both machines, no case is slower beyond band,
  and plans without edges compile to exactly `main`'s code.
- **P2, a fused real factor** (`edge_op` gains a real factor: the passes' edge loads read
  real values and widen them; c2c chains and batch leaves): after P1. Drafted rule: E1 and E5
  re-run, arm against P1; adopt per target when the geometric mean over kind (a) is at most
  0.95 beyond band and no case slower beyond band.
- **P3, accumulation fused into the last pass's stores** (an output `edge_op` that loads, adds
  and stores): after P1, with the same rule on E3's kind (c). Lowest priority: accumulation
  already matches or beats the user's loop.
- **P4, the team defect**: r2c 6400 × 1 with accumulation at 4 threads (L/U 2.33): look at how
  the edge phase of a single row is shared before any timing.

**Q3 (an edge that forces a slower strategy).** Two forced cases occurred, both costing more
than the user's loop, so each gets a drafted rule:

- c2c 64 × 64 on the EPYC: its batch leaves run one row at a time with any factor or
  accumulation (L/U 1.88–2.37, L/P 2.66–3.83). Part of P1.
- c2r rows with an input factor (complex, E4, or real, E5) out of place on the Xeon, whose
  rules choose the depth-first half-length chain (`describe()`: "a c2r row's input edge takes no
  depth-first chain"): 32768, 65536 (both precisions) and 1048576 points at L/U 1.53–1.93,
  L/P 1.67–2.19. **P5:** the input
  edge's phase writes the workspace block (as a real row's does out of place) and the
  depth-first chain's pairing reads the block. Drafted rule: E4 and E5 re-run on the Xeon; adopt
  when every such case's L/U is at most 1 + band and plans without edges are unchanged.

A real row's grid with an input edge did not occur in the catalogue's rows on these
machines, and the lanes over adjacent transforms with accumulation were not measured
(fbench's edge options refuse the interleaved layout). Both go into the re-run with P1.

**Regression.** Plans without edges are unchanged: every plan identical on both machines,
geometric means 1.002 and 0.996 (EPYC) and 0.998 (Xeon), no case beyond 1.05 × (1 + band).
The Xeon's standard-tier run was not made (the x65 run covers rows, planes and volumes of
both precisions; the known 1.06–1.08 on single tiny transforms is parked).

Lesson 4.28 in `dev/LESSONS.md`. DESIGN 12.1 and the decision log record the outcome.
