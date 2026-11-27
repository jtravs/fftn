# X66: block callbacks outside the kernels against the user's loop and the vocabulary

- **Status:** done 2026-11-27 on the Xeon 6240 and the EPYC 7763; the Mac leg is a daytime
  follow-up. The decision rules were drafted with 5.3's design (2026-11-26), before anything was
  built, and amended before the first judging run (below).
- **Owner:** X66 agent of the lead's evening session, 2026-11-26 (task 5.3g)
- **Phase / task:** 5.3g, on 5.3a–5.3c as built

## Question

What does a block callback (DESIGN 12.3) cost? It is called by FFTN's driver outside every ISA
region, on a block still in cache. X66 compares it against three references:

- the plain plan plus the user's own loop, which is what a user writes today;
- the vocabulary with a precomputed factor array;
- the plain plan alone.

The answers decide three things:

- the callback block size per profile (`machine_profile::callback_block_bytes`);
- where the modifying-edge copy goes;
- what `docs/fusion.md` (5.5) advises about callbacks against factors, and about building
  callbacks for AVX2.

## Hypothesis

- **Overhead.** A callback stage costs one indirect call per block, a few nanoseconds (X20's
  call overhead), plus the callback's own arithmetic.
  - With blocks of half an L1, an identity callback should cost under 3 % of the plain plan on
    rows of 1024 points and more, and on planes.
  - Short rows depend on the row block (5.2a, PR 138) grouping many rows per call.
- **Against the user's loop.** The user's loop is one more sweep of the output.
  - Where a batch exceeds the L2, that sweep goes to L3 or DRAM. The callback, run on a block
    the last pass has just written, should then win: K/U about 0.75–0.90.
  - Inside the L2 the two should be about even.
- **Against a precomputed factor.** A factor array costs one more read of an array as large as
  the side: 16 bytes per complex double. An `exp` per element costs about 10–20 ns in scalar
  libm, and a few ns vectorised.
  - At one thread, inside cache, the factor should win.
  - At full width, beyond cache, where bandwidth binds, computing should win.
- **The copy.** A `modify` input callback out of place needs one copy of the input.
  - Into a scratch line, the copy stays in cache.
  - Into the output (c2c, the strategy then in place), it saves the scratch's footprint but
    writes the output one more time.
  - Neither should differ by more than a few per cent inside the L2.
- **AVX2 builds.** A cubic callback should gain 1.3–2× from AVX2 and FMA over a baseline x86-64
  build, as any short pointwise loop does. The `exp` callback gains only with a vectorised math
  library.

## Change under test

There is no single library change. X66 judges the callbacks as built (5.3a–5.3c) against user
code, and compares settings inside FFTN by forcing them (`build_settings`, or knobs registered in
`dev/KNOBS.toml`). The arms, in one fbench session per machine and configuration:

- **P:** the plain plan, with no pointwise work at all. This is the floor.
- **U:** the plain plan, then the user's loop doing the callback's work.
  - The loop is written in the style of X65's U arm: plain C++ on the caller's array,
    vectorised by the compiler, and cloned for AVX2 and AVX-512 on x86-64 Linux.
  - Threaded arms share it over the executor's lanes.
- **K:** the plan with the callback (`modify`), as the rules build it, with the default block
  size. The callback is compiled in the benchmark's own unit, built as U's loop is.
- **K1–K4:** K with `callback_block_bytes` forced to L1/4, L1/2 (the proposed default), L1 and
  L2/4 of the machine's profile.
- **V** (the `exp` callback only): the plan with a complex factor edge holding the precomputed
  `exp` values.
- **Kb / Ka** (x86-64 only): K with the callback built for the baseline (no target attribute)
  and built for `avx2,fma`.
- **Copy arms** (input callbacks, c2c out of place):
  - **S:** the copy into a scratch line or the row block (the default).
  - **O:** the copy into the output, with the strategy built in place (5.3b's build setting).
  - **I** (informative): in place, with no copy.
  - **U_in:** the user's loop modifying their own input in place before a plain plan.

The callbacks, each pure per element:

- **identity:** returns at once. It measures the stage, its calls and any copy.
- **cubic:** v ← v (a + b |v|²), a Kerr-like term.
- **exp:** v ← v exp(i φ), where φ is a quadratic in the element's coordinates. This is a
  propagator computed per element; φ needs `origin`, the strides and `transform`.

## Method

- **Cases:** a catalogue `x66` (`bench/catalogues/x66.toml`), written with the fbench callback
  options (`EDGE_CALL=identity|cubic|exp`, the mode and side, `CALL_BUILD=base|avx2`) before
  any run:
  - c2c rows of 64 to 2^20 points, batched below 16384 (16 to 1024 rows) and single above;
  - batch leaves (16 × 1024) and a Bluestein length;
  - real rows of 256 to 2^20 points (the r2c output edge and the c2r input edge);
  - the client catalogue's rows (c2c 3072; r2c and c2r 6144 × 66 and 32768; r2c 6400 and
    10240 × 28);
  - planes of 192² and 384², in place, alone and with a stride-1 batch of 8 (the free-space
    layout, 5.3c);
  - volumes of 64³;
  - double precision, and a float subset.

  The output edge is primary. Input edges are run for the copy arms. Tier `quick` (the client
  shapes, long rows, planes) is the threaded subset.
- **Target set by mechanism** (AGENTS.md section 6):
  - The block-size rule judges only the cases whose callback stage the block size changes:
    lines it cuts, or short lines it groups.
  - The copy rule judges only c2c out-of-place input callbacks where the copy is not free
    (DESIGN 12.3: no gather, row block or run read).
- **Machines:**
  - the Mac M5 Max: on AC, one fbench at a time, `pmset -g batt` before and after each run, and
    only if the maintainer's current guidance on Mac timing allows it;
  - the Xeon 6240 (`--serial`, its load recorded);
  - the EPYC 7763 (DMOG, a small shared allocation, never `--exclusive`).

  GCC 16 on Linux and Clang on macOS (the release toolchains). One thread, and the client's
  widths where the client catalogue lists them.
- **Metric:** fbench's median paired ratios K/U, K/P, Ki/K2, V/K, Ka/Kb, O/S and S/U_in per case,
  with 95 % intervals and an A/A control for the band.
- **Decision rules,** written before running:
  1. **No loss against the user's loop.** Any case of the cubic or `exp` callback with K/U above
     1.05 × (1 + band) (DESIGN 1.3's tail rule) is a defect. A callback must not be slower than
     the user doing the work. Each defect is fixed within 5.3g, or recorded as a lead with its
     cause, with the maintainer's agreement. The geometric mean of K/U over the cases whose side
     exceeds the L2 is reported per machine and callback.
  2. **Stage overhead.** For the identity callback on the output edge, the geometric mean of K/P
     over rows of 1024 points and more and over the planes must be at most 1.03 beyond the band
     on each machine. If it is not, the callback stage's per-call cost is diagnosed and fixed
     before rule 3 is judged, by grouping more lines per call or building `block_info` once per
     stage instead of per call.
  3. **Block size.** Per machine and precision, the candidate with the lowest geometric mean of K
     over the cubic cases in the target set replaces the default L1/2 when two conditions hold:
     its ratio against K2 is at most 0.97 beyond the band, and no case is slower beyond the
     band. Otherwise L1/2 stays. The size becomes a profile value, which reaches other machines
     by the cache-key rule (AGENTS.md section 6).
  4. **The copy's destination.** Per target, O is adopted for c2c out of place when the
     geometric mean of O/S over the copy rule's cases is at most 0.97 beyond the band, and no
     case is slower beyond the band. Otherwise S stays and O's build setting and code are
     deleted. S/U_in is reported. If S/U_in lies above 1.05 × (1 + band) on a case, rule 1
     applies to it.
  5. **Callback against a precomputed factor** (informative). For each case and thread count,
     the sign of V/K beyond the band goes into `docs/fusion.md`'s advice (5.5): compute in a
     callback, or precompute a factor. It makes no library decision.
  6. **AVX2 builds** (informative). Ka/Kb per case goes into `docs/fusion.md`'s advice on
     building callbacks (target attributes or clones). It makes no library decision.
  7. **Regression.** Plans without callbacks are unchanged: the `describe()` and
     `workspace_bytes()` probes are identical to `main` before 5.3a's merge.

  Near misses are re-run (AGENTS.md section 6).

### Amendments before the first judging run (2026-11-26 evening)

Committed before any judging run, each with its reason. The drafted rules above stand except
where an item below makes them precise or narrows them.

1. **Machines.** The Mac leg is a daytime follow-up: the maintainer asked for a quiet Mac at
   night, so tonight's runs are on the Xeon 6240 (`--serial`, shared Slurm, its load recorded) and
   the EPYC 7763 (DMOG, shared Slurm, never `--exclusive`). The Mac's run (the same three sessions
   below, on AC, one fbench at a time, `pmset -g batt` before and after) judges the Apple key
   when it is made. A rule measured on one vendor is keyed to that vendor (the maintainer,
   2026-11-25; DESIGN 13.5): a block size adopted on the Xeon reaches Intel cores of its cache
   key only, one adopted on the EPYC AMD cores of its key only.
2. **Toolchains.** The EPYC builds with GCC 16 (the release toolchain); the Xeon with GCC 14.4,
   the newest its remote profile has (as X65). The benchmark target compiles with the
   repository's flags (`-O3 -ffp-contract=off`): the callbacks and the user's loops are built
   alike, both in the backend's unit (bench/backends/fftn.cpp).
3. **The work is the identity in exact arithmetic,** so that fbench's check of every case holds:
   cubic with a = 1, b = 0; exp multiplies by w (cos² φ + sin² φ) + i z sin φ with w = 1, z = 0,
   after computing cos φ and sin φ (φ up to about 50 radians). The constants are read at run
   time, so every element pays the arithmetic. On a real side, exp multiplies by the real part.
   V is the plan with `EDGE_OUT=complex.full`: a complex factor array as large as the side.
4. **K2 is K.** Both machines' default `callback_block_bytes` is L1/2 = 16 KiB (32 KiB L1d), so
   the arm K (the default) is K2. K1, K3 and K4 force `CALLBACK_BLOCK_BYTES` = 8192, 32768 and
   L2/4 (262144 on the Xeon, 131072 on the EPYC).
5. **Thresholds made precise** (as X65 and the client tail rule read them):
   - a case is *slower beyond the band* when its whole 95 % interval lies above 1 + band (fbench's
     "slower"), and a *defect* (rule 1) when its whole interval lies above 1.05 × (1 + band);
   - rule 2 fails when the geometric mean's point lies above 1.03 × (1 + band);
   - rules 3 and 4 adopt when the geometric mean's point is at most 0.97 × (1 − band) and no case
     is slower beyond the band;
   - the band is the run's A/A control (P against its copy), and the ratios between arms other
     than P come from `bench/x66/score.py`, which pairs any two arms round by round.
6. **Target sets by the plans' notes** (the mechanism, AGENTS section 6): each case's note carries
   the plan's `callbacks:` lines, which name the blocks and copies.
   - Rule 3: candidate Ki is judged on the cubic cases whose note differs between Ki and K (the
     cases whose blocks it changes); its geometric mean of Ki/K is over those cases.
   - Rule 4: the c2c out-of-place cases whose note differs between O and S (only the chain takes
     the copy into the output; the other strategies take the output or a gather already).
   - Rule 2: the c2c and real rows of 1024 points and more, and the planes, with the identity
     callback (Ki) against P.
7. **Sessions** (one thread, hot mode, median statistic, `--tier full`, 10 to 20 rounds), per
   machine:
   - **out:** P (baseline), U, K, Ki (identity), K1, K3, K4, Kb, Ka, all on the output edge, the
     callback cubic except Ki; the whole catalogue;
   - **exp:** P, Ue, Ke, V, Kbe, Kae (the exp callback; V the factor), output edge; the whole
     catalogue;
   - **in:** P, U_in, S (the input callback, cubic, the default copy), O (`CALLBACK_COPY=out`), on
     the catalogue's c2c cases; on its in-place planes S is the arm I (no copy).
8. **Threads.** Every rule is judged at one thread. Rule 5 is informative per thread count:
   tonight one thread on both machines, and, if time allows, the exp session's quick tier at
   `THREADS=8` on the EPYC (the Xeon's Slurm binds no CPUs, X59, so no threaded run there).
9. **Rule 7 by construction.** X66 adds no library code unless rule 3 adopts a block size (a
   profile value that only callback stages read: `callback_block_bytes` appears nowhere else in
   `src/`) or rule 4 adopts or deletes the copy into the output (`callback_copy_out`, read only for
   plans with an input callback). Plans without callbacks therefore keep their programs; an
   adopted change is checked with `describe()` and `workspace_bytes()` probes against `main`.
10. **Added after the first runs were stopped, before any was analysed for a verdict**
    (2026-11-26, 23:45, commit 77672c3f; the judging runs began at 23:51; this time read 23:55
    until a typo fix of 2026-11-27, which changed no measured time). The first sessions (fbench options at 7af57e23; EPYC runs
    20261126-233227-b1ed and the two beside it, on a node at load 88 with a band of ±15 %, and
    Xeon run 20261126-233149-5693, nine rounds) were stopped, their times kept as recorded. Their
    callback walked every block line by line, whatever the strides; on planes and volumes the
    last phase passes columns (`element_stride` the row's length, `line_stride` 1), so it
    walked memory at the row's stride, and on the Xeon's partial run K/U was 1.62–1.65 on the
    planes in place and 1.46 on the c2c volume while the identity callback cost the plain plan
    only 1.03–1.13 there. That measured fbench's callback, not FFTN's stage: a user who reads
    `block_info` walks the block in memory order. The callback now does (`CALL_LOOP=memory`, the
    default: along the lines in the inner loop where they lie closer together than the
    elements), and the out session adds an arm **Kn** (`CALL_LOOP=lines`, the line-by-line walk)
    for `docs/fusion.md`'s advice; Kn decides nothing. The EPYC sessions run on idle nodes
    (node20, node21, still shared allocations of 8 cores) instead of the loaded one. The rules
    are unchanged.

## Results

fbench options and library at 77672c3f (this branch; the library is `main`'s, acb7b5ae). Hot
mode, median statistic, one thread unless stated, 16–20 valid rounds per run; ratios from
`bench/x66/score.py` (any two arms paired round by round), bands from each run's A/A control.

- **EPYC 7763** (DMOG, GCC 16.2, `avx2`, shared allocations of 8 cores on idle nodes node20 and
  node21, load 1–3): out 20261126-235126-b498 (band ±1.44 %), exp 20261126-235126-5bf6 (±1.27 %),
  in 20261126-235126-67b4 (±1.00 %), exp at 8 threads (tier quick) 20261126-235126-aab9
  (±1.21 %).
- **Xeon 6240** (GCC 14.4, `avx512`, `--serial`, shared with other users' jobs: load 9–22, 19 of
  20 rounds above the quiet thresholds): out 20261126-235212-31ed (±2.59 %), exp 20261127-001612-2554 (±2.21 %, 10 rounds, 7 of them loaded),
  in 20261127-002403-2238 (±2.11 %, 10 rounds, quiet); rule 3's f64 re-run 20261127-021303-3fa6
  (±1.50 %, 20 rounds, 15 loaded).
- Stopped before the loop-order fix (amendment 10), kept as recorded and not analysed:
  20261126-233149-5693 (Xeon), 20261126-233227-b1ed and its two siblings (EPYC, loaded node).

**Rule 1 (K/U, cubic and exp).** Defects are cases whose whole interval lies above
1.05 × (1 + band).

| Machine | Cubic K/U: geomean (beyond L2) | Cubic defects | Exp Ke/Ue: geomean (beyond L2) | Exp defects |
|---|---|---|---|---|
| EPYC | 1.066 (1.093, 23 cases) | 256 × 256 c2c 1.090; r2c and c2r 256 × 64 1.135, 1.118; planes 192², 384² 1.155, 1.320; free-space planes 1.607, 1.399; c2c 64³ 1.097; f32 384² 1.348 | 1.040 (1.048) | planes 1.144, 1.221; c2c and r2c 64³ 1.108, 1.070; f32 rows 1024 × 64 and 65536 1.116, 1.110, f32 384² 1.208, f32 r2c 6144 × 66 1.116 |
| Xeon | 1.035 (1.062, 15 cases) | r2c and c2r 256 × 64 1.162, 1.223; planes 1.330, 1.580; free-space 384² 1.126; f32 384² 1.664 | 1.032 (1.064) | planes 192², 384² 1.096, 1.248; free-space 384² 1.188; f32 384² 1.084 |

The callback wins where the user's loop is a large share of a small transform (16 × 1024: 0.94
EPYC, 0.55 Xeon; Xeon 64 × 1024 0.90) and is within a few per cent on most rows. The
hypothesis (0.75–0.90 beyond L2) failed: the user's sweep from L3 costs little next to the
cubic's arithmetic, and the cases beyond L2 are dominated by the planes. Causes of the defects
(identity callbacks Ki/P on the same cases, and the plans' notes):

1. **Short rows in batches** (256 points; Ki/P 1.12–1.37): a plan with a callback runs rows of
   a chain row by row in the edge driver's walked form (`edge_batched_forms` reaches rows on one
   leaf only, X65's P1), where the plain plan takes its batched forms; the callback adds one
   call per row.
2. **Planes and volumes** (Ki/P 1.00–1.07 on the planes, so the stage is cheap): the output's blocks are a
   lanes column block's columns, cut at `callback_block_bytes` into 3–6 calls of 3–5 columns of
   192–384 points at the default 16 KiB, and the callback's inner run along the lines is that
   short. Rule 3's L2/4 removes most of it on the Xeon (0.61–0.77) and on the EPYC (0.78–0.95);
   see rule 3.
3. **The free-space layout** (a stride-1 batch of 8 planes): the last phase transforms each
   column alone (a chain, not the lanes), so its callback is called once per column of 192–384
   elements 8 × 192–8 × 384 elements apart, where the user's loop runs along the batch.
4. **Exp in float on rows** (EPYC; blocks are whole rows, Ki/P 0.99–1.05): fbench's callback
   and its loop both call `sincosf` per element; the callback's loop carries more state across
   that call. A property of the callback's code, not of FFTN's stage.

None is fixed in 5.3g; each is a lead for the maintainer (Verdict).

**Rule 2 (stage overhead, identity Ki/P over rows of 1024 points and more and the planes, 28
cases).** EPYC 1.011 [1.006, 1.017], Xeon 1.025 [1.017, 1.041]: both at most 1.03 × (1 + band).
**Passed**; rule 3 is judged.

**Rule 3 (block size, cubic, target set by the notes), per machine and precision as the rule
says.** Geometric mean of Ki/K over the cases each candidate changes; the bar is 0.97 × (1 −
band) with no case slower beyond the band. `callback_block_bytes` is one value per profile, so a
machine adopts a candidate only where both precisions pass (the lead's decision, 2026-11-27).

| Machine, precision (run) | K1 (8 KiB) | K3 (32 KiB) | K4 (L2/4) | Bar | Criterion |
|---|---|---|---|---|---|
| EPYC f64 (-b498) | 1.037 (26 cases; 2 slower) | 0.989 (24) | 0.979 (24; none slower) | 0.956 | missed |
| EPYC f32 (-b498) | 1.059 (3; 1 slower) | 0.944 (3) | 0.922 (3; none slower) | 0.956 | passed (informative) |
| Xeon f64 (-31ed) | 1.021 (27; 4 slower) | 0.969 (24) | 0.9458 (24; none slower) | 0.9449 | near miss (inside the ±2.59 % band) |
| Xeon f64 re-run (20261127-021303-3fa6, band ±1.50 %, 20 rounds, 15 loaded) | 1.027 (27; 7 slower) | 0.977 (24) | **0.952** (24; none slower; planes 0.66–0.77) | 0.9555 | **passed** (decides) |
| Xeon f32 (-31ed) | 1.159 (3; 1 slower) | 0.933 (3) | 0.839 (3; none slower) | 0.9449 | passed |

The first verdict pooled both precisions (Xeon 0.933, EPYC 0.972 over 27 cases); the second
reader split them as the rule says. The Xeon's f64 near miss got the one `--serial` re-run the
protocol requires (same f64 cases and the rule-3 arms P, K, K1, K3, K4 at 77672c3f, the rules
unchanged), which decides: it passed (0.952 at most 0.9555, no case slower beyond the band), so with float passing too the Xeon 6240's class keeps L2/4 (commit 94fec438). The EPYC keeps L1/2 (f64 missed; its f32 pass is recorded
as informative). K4 was measured only with the memory-order walk (`CALL_LOOP=memory`), never with
the line-by-line one (Kn), so L2/4's gain on the planes depends on a callback that walks its
block in memory order. Only callback stages read the field.

**Rule 4 (the copy's destination, c2c out of place, the 13 cases whose note differs between O
and S).** EPYC O/S 1.043 [1.032, 1.060], seven cases slower beyond the band (1.035–1.147); Xeon
1.065 [1.047, 1.082] over its 8 cases, five slower beyond the band (1.068–1.139). S stays on both; per the rule, O's build setting and code are deleted (commit
e9f5c959). **S/U_in** (the input callback with its copy against the user modifying their own
input in place): EPYC 1.037–1.327 on the out-of-place cases (geomean over the 19 c2c cases
1.109), 0.97–1.02 on the in-place planes (no copy); Xeon 0.79–1.42 (geomean 1.084): rows of
16384 points and more 1.27–1.42, 3072 1.25, float rows 1.21–1.26, batched rows of 64–4096
points 1.01–1.04, 4099 × 4 1.056, 16 × 1024 0.79, in place 0.86–1.02. Rule-1 defects by rule
4's letter (whole interval above 1.05 × (1 + band)): on the EPYC (threshold 1.0605, run
20261126-235126-67b4) c2c 64 × 1024, 256 × 256, 1024 × 64, 4096 × 16, 16 × 1024, 3072, the
single rows of 16384, 65536, 262144 and 2^20 points, 64³ and f32 1024 × 64 (12 cases; not 4099 ×
4, 1.037, nor f32 65536, nor the in-place planes); on the Xeon (threshold 1.0715, run
20261127-002403-2238) the single rows of 16384, 65536, 262144 and 2^20 points, 3072, and f32
1024 × 64 and 65536 (7 cases; not the batched rows of 64–4096 points, 4099 × 4, 16 × 1024, 64³
or the planes). Cause: the user's loop destroys its input, while the callback
must not, so the plan copies the input (one more sweep), and then runs the callback as a stage of
its own over the copy (a second sweep where the row exceeds the cache: the 2^20-point row's 1.33).

**Rule 5 (V/K, informative).** The precomputed factor won everywhere: EPYC 0.084–0.399 at one
thread (geomean 0.163), 0.111–0.695 at 8 threads (0.297, the long row at 8 threads 0.695); Xeon
0.071–0.354 (geomean 0.159). The `exp` per element with scalar libm cost 2.5–38 times the plain plan (EPYC).

**Rule 6 (Ka/Kb, informative).** Cubic: EPYC 0.901 (0.71–1.07 per case), Xeon 0.898
(0.70–0.94). Exp: EPYC 0.989 (float 0.94), Xeon 0.988 (0.94–1.03 per case).

**Kn/K (the line-by-line walk, informative).** Equal on rows; planes and volumes 1.3–5.1 (EPYC),
1.06–1.65 (Xeon).

**Threads (informative).** At 8 threads on the EPYC the library's exp callback took 0.18–0.26 of
the user's loop on single long rows (the loop cannot split one row; the callback stage's
chunks can), 0.89–0.94 on batches, 1.10–1.16 on the planes.

**Rule 7.** Plans without callbacks never read `callback_block_bytes` or
`callback_copy_out` (`src/`), so the adoption and the deletion leave their programs as they were.
Probes, as amendment 9 promised: 848 plans without callbacks (c2c rows forward and backward, in
and out of place, r2c and c2r rows, of 16 to 2^20 points in batches of 1 to 1024; c2c, r2c and
c2r planes of 16² to 1000², alone and in batches of 8; volumes of 16³ to 97³; both precisions)
built with the Xeon 6240's profile (`profile_for` of its recorded topology, Intel) at
`callback_block_bytes` 16 KiB (`main`'s value) and 256 KiB (the adopted one): `describe()` and
`workspace_bytes()` at 1 and 8 lanes identical for all 848, on x86-64 (GCC 16, the library at
94fec438, DMOG's login node) and on arm64 (Clang 23, the Mac). The probe compares the two
values within one build, which isolates the adopted change; the branch's other library change
is the deletion of `callback_copy_out`, read only by plans with an input callback.

The stored runs are in the main checkout's fbench store, `bench-out/runs/` (git-ignored), and in
each machine's job directory under the remote profiles' work directories.

## Verdict

Judged on the EPYC 7763 (AMD) and the Xeon 6240 (Intel); the Mac leg is a daytime follow-up
(amendment 1), so Apple's key keeps its defaults.

- **Rule 1: defects found, none fixed in 5.3g.** By the rule's letter the cases listed above
  (both callbacks, and the out-of-place c2c cases rule 4's S/U_in names: 12 on the EPYC, 7 on
  the Xeon) are defects. Each is
  recorded here as a lead with its cause; the rule wants the maintainer's agreement to leave them
  as leads, which this verdict asks for. Most of the slow cases are the planes, whose cause rule
  3 addressed on the Xeon.
- **Rule 2: passed** on both machines (1.011, 1.025).
- **Rule 3, per precision: L2/4 adopted on the Xeon 6240's class** (Intel's cores; commit
  94fec438), where both precisions pass: f32 0.839 (run -31ed, bar 0.9449), f64 0.9458 in run
  -31ed (a near miss of its bar 0.9449, inside the band) and 0.952 in the deciding `--serial`
  re-run 20261127-021303-3fa6 (bar 0.9555, none slower). **Half the L1 stays** on the EPYC
  7763's class (f64 0.979 against 0.956: missed; f32 0.922 passed, informative) and on every
  other machine. The gain needs a callback that walks its block in memory order (K4 was never
  timed with the line-by-line walk).
- **Rule 4: S stays** (the copy into the workspace block) on both machines (O/S 1.043 and
  1.065), and O's build setting, code, tests and fbench option were deleted (commit e9f5c959).
- **Rules 5 and 6** (informative) went into `docs/fusion.md`'s callbacks section: precompute a
  factor that does not change (V/K 0.07–0.40 at one thread, 0.11–0.70 at eight); build
  arithmetic callbacks with AVX2 and FMA (0.90 on both machines), which gains nothing for a
  callback bound by scalar `cos` and `sin`; walk blocks in memory order (the line-by-line walk
  1.06–5.1 on planes and volumes); run in place when the input may be overwritten.
- **Rule 7:** plans without callbacks keep their programs: neither change is read by them.

**Leads for the maintainer** (each would be its own experiment or design step):

1. **Batched forms for plans with callbacks.** Rows of 256 points pay 12–37 % for a callback that
   does nothing, because `edge_batched_forms` gives the flat and row-block forms only to rows on
   one leaf (X65's P1, narrowed); the flat form already runs callback stages. A measured reach
   for chains with callbacks would remove most of it (and of the 256-point K/U defects).
2. **A lanes column block passed whole.** On the EPYC the planes took 0.78–0.90 with L2/4's blocks
   (c2c 64³ 0.954), but the whole-catalogue block size missed the rule. A narrower rule, a column
   block of a multi-axis phase never cut by lines whatever `callback_block_bytes` says, names
   exactly those cases (an X66b on both machines).
3. **The free-space layout** (a stride-1 batch of planes): its last phase transforms each column
   alone, so its callback gets one column a call at a stride of 8 × 192–384 elements (K/U
   1.08–1.61). Holding the batch's adjacent transforms together there (the lanes over adjacent
   transforms) would give the callback runs along the batch; a design step for 12.5's stage.
4. **The input copy.** An out-of-place `modify` input callback copies the input, then runs as a
   stage of its own over the copy: two sweeps where the row exceeds the cache (Xeon rows of
   16384 points and more 1.27–1.42; EPYC 2^20 1.33). Fusing the copy into the callback stage
   (copy a block, call the callback on it while it is in L1) would leave one; letting the
   callback change the caller's input (a mode, an API decision) would leave none.
5. Not FFTN's: fbench's exp callback in float on whole-row blocks (EPYC 1.11, Ki/P 0.99–1.05
   there) is the callback's code around `sincosf`, which its loop carries less state across.

**Follow-up:** the Mac M5 Max leg of the three sessions, daytime, on AC with `pmset -g batt`
before and after, one fbench at a time; it judges Apple's key for rules 1–4. The lesson is in
[LESSONS](../LESSONS.md) 4.30.
