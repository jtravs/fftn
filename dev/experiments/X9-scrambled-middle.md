# X9: scrambled intermediate order for vocabulary middles of row pipelines

- **Status:** done for 5.4 (2026-11-27), by reading alone; the depth-first question is left open.
  As stated, the premise does not hold: no pipeline pair runs a pass whose only job is the natural
  order of its middle. But a scrambled order can still save a traversal of the line through
  locality, in the depth-first pair, and it is not built under features-first. Nothing is built
  (no scrambled arm, no `spectral_layout`), so nothing was timed.
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
  65536). Each chain is decimation in time at the top (`depth.inl`). Its first pass reads the
  F1 interleaved sub-rows x[F1 j + r] straight from the line, each into its own block. Then a
  loop runs the rest of each sub-row's chain on its block (`looped(subs, ...)`,
  `core/pipeline.cpp`), and the top pass writes X[k + M t] in natural order. The four-step
  transpose is folded into the first and top passes' addressing, so no pass is a transpose. Counted
  as traversals of the whole line (each one reads every block back from L2 or L3, since the
  line is beyond L2 where the pair runs), the pair makes six: the forward first pass, the forward
  sub-row loop, the forward top pass, then the same three backward (one more loop per chain with
  two top factors). The sub-row chain's own copy back to its block (`chain_builder`'s depth
  stages, when the sub-row has two radices) is placement within one block, not order.
- **Lanes pair** (`c2c_lanes` twice; X67's rule for a plane's strided lines of 64–512 points).
  The chains are Stockham on a tile; the lanes' gather and scatter are the tile's layout (one line
  per vector lane), which a scrambled spectrum would not remove.
- **The blocked grid** is never a pipeline's strategy: `core/pipeline.cpp` builds only the four
  pairs above. On plans its transpose is folded into phase 1's store (DESIGN 8.5), and its
  in-place route is the workspace block, not a permutation pass.
- **Bluestein's convolution** (a plan's internal pair) is built through `depth_pair` or the
  chain, so it has the same structure: no reordering step on the chain, and the depth-first
  form's six traversals (below). Its kernel table belongs to the plan, which could store it in
  any order without a `spectral_layout`, so it is the cleanest first test of a scrambled
  depth-first pair.
- **Rader stages** permute their input and output (by powers of a primitive root), but those
  gathers and scatters are folded into stages the algorithm needs anyway (`add_rader` writes
  through the permutation), and they do not depend on the middle's order.

**Why prior art differs, in part.** A scrambled mode pays where a library's natural order costs
a reordering step of its own: an in-place radix algorithm's digit reversal, or a SIMD algorithm
whose internal order needs a separate reorder pass. FFTN's Stockham autosort and folded transposes
mean the natural order costs no such pass (DESIGN 8.5). But part of prior art's 10–25 % comes from
locality as well. With the order scrambled, the forward's last passes, the factor and the
inverse's first passes all act within small contiguous blocks, so they can run block by block in
cache. A natural-order Stockham chain cannot do that, because every pass's legs are n/R apart
(`stockham.inl`). The locality part is open to FFTN.

**What the natural order still costs.** Three places where the passes on either side of the
middle could share work:

1. **Chain and lanes pairs: the forward's last pass and the inverse's first pass act on the
   same R-tuples.** The last pass (m = 1, s = n/R) reads and writes {q + (n/R) j}, and the next
   chain's first pass (s = 1, m = n/R) reads {p + (n/R) j}: the same sets when the two radices
   are equal. One joined pass (the forward plain codelet, the middle's factor, the inverse's DIF
   codelet with its twiddles) would replace two: 6 → 5 passes at 4096 on `avx512`, 8 → 7 on
   `avx2`, and on the lanes pair the separate middle pass on the tile would go as well (7 → 5
   stages at 512). The radices must match: 4096 on `avx512` (16 × 16 × 16) and 6000
   (20 × 15 × 20) match as built; 1024 (8 × 8 × 16) and the lanes pair's 512 (4 × 8 × 16) would
   need the inverse in reversed radix order (a planner and twiddle-table question). With fan-out
   the inverse's first pass runs once per copy, so a joined pass would repeat the forward's last
   butterfly per copy, or not apply. Natural order serves this one as well as a scrambled order
   would.
2. **Depth-first pair, joined in natural order.** The forward top pass writes the sets
   {k + M t}, t < F1. The inverse's first pass reads sub-row r's elements p + (M/R′) i, that is
   line positions F1 p + r + (n/R′) i, where R′ is the sub-rows' first radix. These are the same
   sets when R′ = F1. One pass can then do the forward's top butterfly, the factor and the
   inverse's first butterfly, writing into scratch a (not in place): 6 → 5 traversals, and no DIF
   form needed. As built the radices differ (65536: F1 = 32 with sub-rows 16 × 8 × 16; 18432:
   F1 = 12 with sub-rows 8 × 12 × 16), so the tops or the sub-rows' radix order would have to
   change. With the inverse in DIF form at the top as well (its top pass joined to the forward's,
   its sub-rows storing at stride F1 in their last pass), the count is 4: first pass, forward
   sub-rows, joined top, inverse sub-rows. Natural order reaches 3 only with a symmetric fold that
   also folds the forward's first pass into its sub-row loop (each block's chain reading its
   sub-row straight from the line at stride F1, C sub-rows at a time). That is not built, and X46's
   separate first pass exists to avoid it.
3. **Depth-first pair, scrambled.** The forward in DIF at the top (its top pass from the line into
   the blocks), then per block in cache the forward sub-row chain, the factor (stored permuted)
   and the inverse sub-row chain, then the inverse's DIT top pass to the output: 3 traversals,
   because each block stays in cache between the two sub-row chains. That is one fewer than the
   best natural-order form built from today's passes (4), so **a scrambled order may pay here**.
   It needs a DIF top pass (from existing DIF codelets) and sub-row chains in place on a block.

Footprint is a separate cost: a Stockham chain of three or more passes ping-pongs through
scratch line a beside the block (the chain pair at 4096 touches four lines: in, block, a, out),
where an in-place pair in digit-reversed order would touch three. X18 measured three lines
against four on plans' chains at up to 15 % for 512–16384 points on a 32 KiB L1. FFTN has no
in-place passes that leave the order scrambled, and building them means SIMD passes whose late
stages combine copies that are not adjacent in memory (the reason the chains are Stockham with
lane passes over adjacent copies, DESIGN 8.2 and 8.5).

## Verdict

**For the chain, real and lanes pairs, the premise does not hold: no pass exists only to keep the
natural order, and the joined pass a scrambled order suggests (item 1) is open to the natural
order too. For the depth-first pair the question stays open.** A scrambled middle (item 3) would
make 3 traversals of the line where today's pair makes 6 and the best natural-order form from
today's passes makes 4. That gain comes from locality, not from removing a permutation. It is
not built: features come first (the maintainer, 2026-11-25), and building it means new kernel
passes. `spectral_layout` is not built either. The first test needs no API: Bluestein's
depth-first convolution, whose kernel table belongs to the plan. A pipeline would need
`spectral_layout` only if that test pays. DESIGN 12.2's "Scrambled order" bullet now records
this. No library code changed, and every plan and pipeline is unchanged.

**Leads for the performance phase** (vocabulary middles only, since a middle callback needs
whole lines; each needs the maintainer's agreement to open, features first):

1. **A joined middle pass for the chain and lanes pairs**, in natural order: the forward's last
   pass, the middle's factor and the inverse's first pass as one pass where their radices match
   (with the inverse's radix order reversed where they do not). Expected: one pass fewer of the
   pair's 2k on lines resident in L1 and L2 (an estimate, unmeasured). Not with fan-out (item 1).
   New pass kinds from existing codelets, so kernel work with a second reader; Bluestein's chain
   convolution would take the same pass.
2. **The scrambled depth-first pair** (item 3): 6 → 3 traversals where the pair runs (the Xeon
   6240's class, 288 KiB–2 MiB), first as Bluestein's depth-first convolution (the plan's own
   table, no API), then `spectral_layout` for pipelines if it pays. Against it, the natural-order
   alternatives: the joined pass with R′ = F1 (6 → 5, no DIF form) or with a DIF inverse (6 → 4).
3. **The real pair:** X67's lead 1 (the middle fused into the c2r pairing pass) is the real
   analogue of lead 1 here and comes first.
4. **Footprint** (above): whether a pair touching three lines instead of four pays at L1-sized
   lines. Lowest priority: it needs a new family of in-place passes.

Lesson 4.35 in `dev/LESSONS.md`.
