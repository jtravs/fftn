# X66: block callbacks outside the kernels against the user's loop and the vocabulary

- **Status:** planned. The decision rules were drafted with 5.3's design (2026-11-26), before
  anything was built. The agent who runs X66 claims it here and may adjust the rules only
  before the first judging run, stating each change and why.
- **Owner:** unclaimed (task 5.3g)
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

## Results

Not run.

## Verdict

Not run.
