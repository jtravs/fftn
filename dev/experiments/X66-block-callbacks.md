# X66: block callbacks outside the kernels against the user's loop and the vocabulary

- **Status:** running (claimed 2026-11-26 evening). The decision rules were drafted with 5.3's design (2026-11-26), before
  anything was built. The agent who runs X66 claims it here and may adjust the rules only
  before the first judging run, stating each change and why.
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

## Results

Not run.

## Verdict

Not run.
