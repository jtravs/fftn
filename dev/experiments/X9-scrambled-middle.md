# X9: scrambled intermediate order for vocabulary middles of row pipelines

- **Status:** done (2026-11-27), by reading alone. The premise does not hold: no pipeline pair pays
  a permutation or transpose pass for the natural order of its middle. Nothing is built (no
  scrambled arm, no `spectral_layout`), so nothing was timed.
- **Owner:** task AL (5.4), 2026-11-27
- **Phase / task:** 5.4, on 5.2a–5.2g and 5.3a–5.3h as built (main at 7b114b35)

## Question

Should a pipeline whose middle is vocabulary alone (a factor, a scale) run its middle in the
plan's own order instead of the natural one, and should `spectral_layout` expose that order to
callers who store their kernels permuted?

## Hypothesis

DESIGN 12.2's "Scrambled order" bullet, as written with 5.2's design (2026-11-25): a scrambled
middle "with in-place strategies ... removes two permutation passes; in the blocked grid, the
forward's last transpose and the inverse's first". Prior art (MKL's scrambled mode, djbfft,
PFFFT) suggested 10–25 % for L2-resident sizes and more beyond. Callbacks keep the natural order
whatever the result, because their coordinates must be affine in the block.

## Change under test

None was built. The brief for task AL made the arm conditional: first find, by reading the
pairs' programs (`describe()`, the chain builder), where a natural order costs a permutation or
transpose pass today; if no pair pays one, the premise does not hold and the verdict is written
without timing.

## Method

- **Code read** (main 7b114b35): `src/core/pipeline.cpp` (the chain pair, the real chain pair,
  the depth-first pair, the lanes pair, fan-out), `src/strategies/chain.hpp` and `chain.cpp`
  (`c2c_line`, `c2c_lanes`, `c2c_depth`, `depth_pair`, `r2c_line`, `c2r_line`),
  `src/passes/stockham.inl` and `src/passes/depth.inl` (the passes' index maps), DESIGN 8.2, 8.5
  and 12.2.
- **Programs printed:** `describe()` of representative pipelines by the rules, from a small
  program against the library (not committed), on the Xeon 6240 (Slurm job 17609, shared, two
  CPUs; GCC 14.4, Release, the fbench build cache of a997ebb8, whose library sources differ from
  7b114b35 only in `core/r2r.cpp` and `threading/schedule.*`). Each case at the processor's
  best target (`avx512`) and with `avx2` forced, in double: `<C, C, C>` 4096 × 32 and 6000 × 8;
  inverse first 4096 × 8; the valid halves of 2m = 18432 for `<C, C, C>` and `<T, C, T>`;
  `<C, C, C>` 65536 × 1; `<T, C, T>` 4096 × 16; the strided lines of 512 × 512 and 1024 × 1024
  planes. Between them they show all four pairs, the depth-first and lanes pairs by X67's rules.

## Results

**No pair has a permutation or transpose pass.** Every stage of every printed program is a
radix pass, a pairing pass, an edge or middle pass, a run's read or store stage, or a stride
copy. Pair by pair:

- **Chain pair** (`chain_builder::c2c_line` twice). A chain is a Stockham autosort in
  decimation-in-frequency form (`stockham.inl`: y[q + s(Rp + k)] = w^(pk) Σ_j x[q + s(p + mj)]
  ω^(jk)), so each pass writes its outputs where the next expects them and the last writes the
  natural order; DESIGN 8.5 says so ("there is no reordering step"). 4096 × 32 on `avx512` is six
  radix-16 passes (in → block → a → block, then block → out → a → out), the middle in the
  forward's last pass. Copies occur for two reasons only, neither the spectrum's order: a line of
  stride other than 1 on a target with C > 1 (the 1024 × 1024 plane's strided lines: `copy: in
  -> a` and `copy: a -> out`, which the lanes pair removes up to 512 points), and a single
  generic pass in place, which the rules no longer choose (X17).
- **Real chain pair** (`r2c_line`, `c2r_line`). The same chains on the half length; the pairing
  of bins k and h − k is arithmetic needed to form the spectrum, not a reordering. Its separate
  passes (the r2c pairing where the last pass is split, the middle, the valid halves' read and
  store stages) are X67's leads 1 and 5, unrelated to order.
- **Depth-first pair** (`depth_pair`; taken by X67's rule on the Xeon 6240's class at 18432 and
  65536). Each chain is decimation in time at the top (`depth.inl`): the first pass reads the
  R1 interleaved sub-rows x[R1 j + r] straight from the line, one block each, and the top pass
  writes X[k + M t] in natural order. The four-step transpose is folded into those two passes'
  addressing, so the pair is four sweeps of the line (first pass, top pass, first pass, top
  pass) with no transpose between them.
- **Lanes pair** (`c2c_lanes` twice; X67's rule for a plane's strided lines of 64–512 points).
  The chains are Stockham on a tile; the lanes' gather and scatter are the tile's layout (one line
  per vector lane), which a scrambled spectrum would not remove.
- **The blocked grid** is never a pipeline's strategy: `core/pipeline.cpp` builds only the four
  pairs above. On plans its transpose is folded into phase 1's store (DESIGN 8.5), and its
  in-place route is the workspace block, not a permutation pass.
- **Bluestein's convolution** (a plan's internal pair, whose kernel table the plan owns and could
  store in any order with no API at all) is built through `depth_pair` or the chain, so it too
  has no reordering step a scrambled table would save.

**Why prior art differs.** A scrambled mode pays where a library's natural order costs a
reordering step of its own: an in-place radix algorithm's digit reversal, or a SIMD algorithm
whose internal order needs a separate reorder pass. FFTN chose the Stockham autosort and folded
transposes precisely so that the natural order costs no pass (DESIGN 8.5). The 10–25 % estimate
measured the step FFTN does not have.

**What the natural order does still cost, and why order is not the fix.** Looking for a saving
that a scrambled middle would make shows two places where the passes on either side of the
middle could share work; in both, the natural order can share it as well:

1. **Chain and lanes pairs: the forward's last pass and the inverse's first pass act on the
   same R-tuples.** The last pass (m = 1, s = n/R) reads and writes {q + (n/R) j}, and the next
   chain's first pass (s = 1, m = n/R) reads {p + (n/R) j}: the same sets when the two radices
   are equal. One joined pass (the forward plain codelet, the middle's factor, the inverse's DIF
   codelet with its twiddles) would replace two: 6 → 5 passes at 4096 on `avx512`, 8 → 7 on
   `avx2`, and on the lanes pair the separate middle pass on the tile would go as well (7 → 5
   stages at 512). The radices must match: 4096 on `avx512` (16 × 16 × 16) and 6000
   (20 × 15 × 20) match as built; 1024 (8 × 8 × 16) and the lanes pair's 512 (4 × 8 × 16) would
   need the inverse in reversed radix order. That is a planner and twiddle-table question, not
   an order one.
2. **Depth-first pair: the forward's top pass and an inverse decimated in frequency at the top
   act on the same index sets.** The forward top pass writes {k + M t}; a top pass in DIF form
   reads {j1 + M j2}, the same set. A joined top pass, followed by the inverse's sub-rows with
   their transpose folded into the last pass's stores, makes three sweeps of the line instead of
   four. A scrambled order (the forward in DIF, the two sub-row chains back to back per block,
   the inverse in DIT) also makes three, with the same block traffic, so it gains nothing over
   the natural order.

What a non-sorting order would change is footprint, not passes: a Stockham chain of three or
more passes ping-pongs through scratch line a beside the block (the chain pair at 4096 touches
four lines: in, block, a, out), where an in-place pair in digit-reversed order would touch three.
X18 measured three lines against four on plans' chains at up to 15 % for 512–16384 points on a
32 KiB L1, so this is a real but separate question. It is not X9's single change: FFTN has no
in-place passes that leave the order scrambled, and building them means SIMD passes whose late
stages combine copies that are not adjacent in memory (the reason the chains are Stockham with
lane passes over adjacent copies, DESIGN 8.2 and 8.5).

## Verdict

**The premise does not hold; X9 is closed without a timing run.** No pair pays a permutation or
transpose pass for the natural order of its middle, so a scrambled middle has nothing to remove,
and `spectral_layout` is not built: an API whose order would equal the natural one buys nothing.
DESIGN 12.2's "Scrambled order" bullet is replaced by this finding. No library code changed;
every plan and pipeline is unchanged.

**Leads for the performance phase** (natural order; vocabulary middles only, since a middle
callback needs whole lines; each needs the maintainer's agreement to open, features first):

1. **A joined middle pass for the chain and lanes pairs:** the forward's last pass, the middle's
   factor and the inverse's first pass as one pass where their radices match (with the inverse's
   radix order reversed where they do not). Expected: one pass fewer of the pair's 2k on lines
   resident in L1 and L2 (an estimate, unmeasured). New pass kinds from existing codelets, so
   kernel work with a second reader; Bluestein's convolution would take the same pass.
2. **A joined top pass for the depth-first pair**, with the inverse decimated in frequency at
   the top (its sub-rows' last pass storing at stride R1): three sweeps of the line instead of
   four where the pair runs (the Xeon 6240's class, 288 KiB–2 MiB). Needs a DIF form of the
   depth-first chain (new top and last passes).
3. **The real pair:** X67's lead 1 (the middle fused into the c2r pairing pass) is the real
   analogue of lead 1 here and comes first.
4. **Footprint** (above): whether a pair touching three lines instead of four pays at L1-sized
   lines. Lowest priority: it needs a new family of in-place passes.

Lesson 4.35 in `dev/LESSONS.md`.
