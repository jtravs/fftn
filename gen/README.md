# The codelet generator

`gen/fftn_gen` writes FFTN's codelets: straight-line DFTs of small size R on R vector
values, the arithmetic core of every pass ([DESIGN.md](../dev/DESIGN.md) section 7). Its
output is committed under `src/gen/`; users never run it. Never edit `src/gen/` by hand:
change the generator and regenerate. The `gen` stage of `scripts/check.sh` fails when the
tree and the generator disagree.

It is plain Python (3.14 or newer, in the project's `uv` environment) with one dependency,
`mpmath`, for exact verification.

## Command line

Run from the repository root:

```bash
uv run python -m gen.fftn_gen                # regenerate src/gen/
uv run python -m gen.fftn_gen --check src/gen  # exit 1 unless src/gen/ is exactly the output
uv run python -m gen.fftn_gen --table        # operation counts against the reference table
uv run python -m gen.fftn_gen --out DIR      # write somewhere else
uv run python -m unittest discover -s gen/tests -t gen   # the generator's tests
```

Generation builds and verifies every codelet before writing anything; a codelet that fails
verification raises and no file changes. `--check` is what the gate runs (about 12 s since radix
64), then the tests (about 40 s). `--table` prints the operation counts of every forward codelet next
to the reference, and exits 1 if a codelet misses the reference other than by a recorded
gap.

## Files

| File | What it does |
|---|---|
| `ir.py` | the DAG: nodes, terms with pending rotations, interning, value numbering, constant roots |
| `algorithms.py` | DFT builders (split radix, Good–Thomas, symmetric-pair primes, Cooley–Tukey) and `choose()` |
| `codelet.py` | variants (plain, DIT, DIF) and directions; `build()` runs the pipeline for one codelet |
| `realify.py` | the real-coefficient form: multiplications by i only at the outputs |
| `lower.py` | complex-register lowering: one vocabulary operation per node |
| `split.py` | split lowering: real and imaginary parts on real registers, FMA-shaped |
| `schedule.py` | emission order and the peak number of live values |
| `verify.py` | the exact check of an operation list against the DFT |
| `emit_cpp.py` | the C++ text and the metadata header |
| `catalogue.py` | the radices generated and the output files: the only list of radices in FFTN |
| `reference.py` | the reference table of operation counts and the recorded gaps |
| `__main__.py` | the command line |

Output: `src/gen/complex_codelets.inl` and `src/gen/split_codelets.inl` (the codelets of
the two lowerings, included inside each target's region) and `src/gen/codelets.hpp` (the
metadata tables, an ordinary header).

## Pipeline

For each radix R, variant and direction (`codelet.build`):

1. **Build** the DFT as a DAG (`algorithms.dft`), with the algorithm chosen by size
   (`algorithms.choose`):

   | Size | Algorithm | Cost of the internal factors |
   |---|---|---|
   | 2 | one butterfly | none |
   | 2^k ≥ 4 | conjugate-pair split radix, decimation in time | w^k and w^−k per k; k = 0 is free, k = n/8 is (1 ∓ i)·√½ |
   | odd prime | symmetric pairs x_j ± x_{p−j}, then cosine and sine sums | the mean of each cosine row is taken out first (below) |
   | p^e, p odd | Cooley–Tukey p × p^(e−1) | constant roots, reduced to the first octant |
   | coprime n1 · n2 | Good–Thomas: input j = (n2 j1 + n1 j2) mod n, output by the Chinese remainder theorem | none |

   The twiddle variants multiply the inputs (DIT) or the outputs (DIF) by twiddle inputs
   w_k, forward, or their conjugates, backward.
2. **Track rotations.** Every reference to a node is a `Term(node, rot)`: the node's value
   times (−i)^rot. Multiplying by −i, −1 or i changes `rot` only. `Graph.add` combines two
   terms into one `Add(a, b, r)` node, a + (−i)^r b, which becomes `add`, `subj`, `sub` or
   `addj` for r = 0, 1, 2, 3. A constant root of unity is reduced exactly (as a fraction of a
   turn) to a quarter turn, which is a rotation, and an angle in the first octant.
3. **Simplify while building.** Every node is interned twice: structurally (same
   children, same constant), and by value: each node is evaluated at three random inputs in
   256-bit arithmetic, and a node whose values are an existing node's times (−i)^q is that
   node rotated by q. This is common-subexpression elimination modulo commutativity,
   associativity, distributivity, sign and rotation. Constants are folded (c·(d·x) is
   (cd)·x, 1·x is x), negative constants become a rotation of 2, and a node whose values
   are zero is the zero term, which `add` and `scale` drop. Only nodes reachable from the
   outputs are lowered (dead-code elimination; pruned variants will rely on it).
4. **Real-coefficient form** (`realify.py`). The DAG is rebuilt with every node v split
   into R(v) and I(v), the combinations of the same atoms (inputs, twiddle products) by the
   real and imaginary parts of v's coefficients, so that every value is a real combination
   up to a pending rotation and each output is R + i·I, one `addj` or `subj`. An output
   pair X_k, X_{n−k} shares both parts, so a codelet multiplies by i (a register swap on
   x86) once per output pair, ⌊(R − 1)/2⌋ values, where the DAG as built did it inside
   every complex-coefficient node (radix 16: 7 against 19, radix 8: 3 against 6).
   Value numbering merges the parts that coincide up to sign, so the form never costs more
   operations (radix 16: 80 vector operations against 84, radix 9: 48 against 54; the
   unfused additions are the same, multiplies equal or fewer); a test holds that, and the
   one-swap-per-pair count, on every codelet (experiment X19).
5. **Lower** (`lower.py`) each node to one vocabulary operation. The register of a node
   holds its value times (−i)^m, with m chosen per node: an `Add` can take either operand
   first, which gives two choices. Wanted rotations are propagated backwards from the
   stores, which need m equal to the output's pending rotation, and each `Add` takes the
   choice that satisfies the nearest store. A store that still disagrees would need a
   `neg` or a multiply by ∓i; no codelet needs one, and `lower` raises `ForcedRotation`
   rather than emit an operation that no codelet has compiled. `acc ± c·x` becomes `fmadd_r(acc, x, ±c)` when the scaled node has no other user.
6. **Order** (`schedule.py`) by greedy list scheduling: the ready operation that adds the
   fewest live values goes next, ties to the one using the most recently computed value
   (depth first). Loads go just before their first use, stores just after their producer.
   Where the builders' creation order keeps fewer values live (radix 9), it is kept
   instead. A complex codelet that keeps more than 16 values live (AVX2's registers) is
   also ordered by `fitted_order`, which scores each ready operation by its operands
   outside a model register file of the 12–16 values touched most recently; that order
   is taken where `spill_moves` (Belady's furthest-next-use eviction in 16 registers)
   counts at least a quarter and 16 fewer stack moves: radix 32 (93 → 40 plain) and 64
   (292 → 146), plain and DIF (experiment X54). The peak number of live values is
   recorded.
7. **Verify** (`verify.py`) the ordered operation list, the same objects the emitter prints,
   in 256-bit arithmetic on every basis input (every column of the codelet's matrix) and
   three random inputs, each with random twiddles, against the definition of the variant
   with exactly reduced roots of unity. Any output off by more than 2^−200 relative is a
   `VerificationError`. So is any breach of the accessor contract, checked apart from the
   values (`check_shape`): every input leg loaded once, twiddle legs 1 to R − 1 once each
   (none for plain), every output stored once.
8. **Emit** (`emit_cpp.py`) one statement per operation, with constants printed in both
   precisions.

## The intermediate representation

Nodes (`ir.py`) are frozen, slotted dataclasses holding child ids, so pattern matching
reads naturally (`case Add(a, b, r):`) and hashing is shallow:

| Node | Value |
|---|---|
| `Zero()` | 0 (id 0 in every graph) |
| `Input(j)` | the codelet's input j |
| `Twiddle(k)` | the twiddle of leg k (twiddle variants) |
| `Add(a, b, r)` | value(a) + (−i)^r value(b), with a ≤ b |
| `Scale(a, key)` | c·value(a), c > 0 and c ≠ 1, c in `Graph.consts[key]` |
| `TwMul(a, w, conj)` | value(a)·value(w), or times its conjugate |

A `Term(node, rot)` is a reference with a pending rotation. Builders only ever handle
terms, through `Graph.add`, `Graph.sub`, `Graph.scale`, `Graph.tw_mul`, `Graph.root_mul`
(times exp(−2πi e/n)) and `ir.rotate`/`ir.neg`. The graph's `values` hold each node's three
sample values; `live(outputs)` lists the nodes the outputs need, in topological order.

The lowered form is a list of `lower.Op`: `kind` (a vocabulary operation, `load`,
`load_tw` or `store`), `dst` and `args` (value numbers), `const` for `scale` and `fmadd_r`,
`index` for loads and stores. Verification evaluates exactly this list and the emitter
prints exactly it.

## Calling convention

A radix's codelets are the static member templates of `cdft<R>`, defined inside each
target's region (the text uses only the vocabulary, so it is region-safe):

```cpp
cdft<R>::plain<T, Sign>(in, out);        // out_k = sum_j in_j exp(Sign 2 pi i jk / R)
cdft<R>::dit<T, Sign>(in, out, twid);    // in_j times t_j first, j >= 1
cdft<R>::dif<T, Sign>(in, out, twid);    // out_k times t_k after, k >= 1
```

`Sign` is −1 (forward) or +1 (backward); t_j is `twid.get(j)` forward and its conjugate
backward (`mul` and `mulc`), because twiddle tables hold forward roots only. `T` is
`double` or `float`. `in`, `out` and `twid` are objects of the caller's accessor types,
passed by const reference, with force-inlined members:

| Call | Meaning |
|---|---|
| `in.get(j)` | returns leg j of the input as a `cv<T>` (j = 0 .. R−1), each called once |
| `out.put(k, v)` | stores leg k of the output, a `cv<T>` (k = 0 .. R−1), each called once |
| `twid.get(k)` | returns the twiddle of leg k as a `tw<T>` (k = 1 .. R−1), each called once |

The member names are not `load` and `store` because a member would hide the vocabulary
function of the same name inside the accessor. Indices are integer literals, so after
inlining every address is a constant offset. Every output depends on every input, so the
first `put` happens after the last `get` and `in` and `out` may address the same memory
(in place). The simplest accessors are in `tests/isa/codelet_runner.inl`.

How the passes of task 1.2 use this (`src/passes/`, DESIGN.md 8.2); the chain is a
decimation-in-frequency Stockham chain, so passes call `plain` and `dif` only:

- **Lane pass** (lanes over copies q): `in.get(j)` loads stream j at the current block of
  copies, `out.put(k, v)` stores stream k; `twid.get(k)` returns the twiddle of (p, k) in
  every lane, read from a table holding one twiddle per entry with the vocabulary's
  `bcast_tw`.
- **Grouped pass** (lanes over butterflies p, the chain's first pass): `in.get(j)` loads C
  adjacent butterflies' leg j; `out.put` keeps the R outputs in a local `cv<T> v[R]`, which
  the pass transposes in C × C blocks after the codelet call (or writes through a tile when
  C does not divide R); `twid.get(k)` loads C per-lane twiddles contiguous in p (`load_tw`).
- **Leaf**: a plain codelet on one row; values are broadcast on loading and lane 0 is
  stored.
- **Edge variants** are accessors: scale or factor on load is `get` returning
  `scale(load(...), s)` or `cmul(load(...), f)` (the factor is a plain complex array, not
  twiddle storage); on store, `put` applies it before the store. The codelet text never
  changes.
- **Tails**: the same instantiation on a padded tile: the valid elements are gathered into
  a small stack array, all C lanes computed, the valid ones stored; nothing in the codelet
  depends on C.

## Metadata

`src/gen/codelets.hpp` holds `gen::codelets[]`, one `codelet_info` per radix, variant
(`kind`), direction (`sign`) and lowering (`form`), and `gen::radices[]`. Counts are per
call of the codelet, which processes C transforms at once:

| Field | Meaning |
|---|---|
| `twiddles` | twiddles read: R − 1, or 0 for plain |
| `adds` | real additions on a fused target: 2 per `add`, `sub`, `addj`, `subj` |
| `muls` | real multiplications: 2 per `scale` |
| `fmas` | real fused multiply-adds: 2 per `fmadd_r` |
| `cmuls` | complex multiplications by twiddles (`mul`, `mulc`); their cost is the target's |
| `vops` | vocabulary operations, loads and stores apart |
| `live` | peak live values, complex and twiddle registers together |

On a target without fused operations each `fmadd_r` is a multiply and an add. The
published counts that the reference table uses are unfused real arithmetic with a complex
multiply as 4 multiplies and 2 additions: adds + fmas + 2·cmuls additions and
muls + fmas + 4·cmuls multiplications (`Counts.flops_adds`, `flops_muls`).

## Operation counts

`reference.PLAIN` has one entry per radix, and each says where its numbers come from
(`reference.py` cites them exactly):

- **published**: the counts in the headers of FFTW 3.3's generated codelets (`n1_N`, and
  `t1_N` for the twiddle variants), and for powers of two the split-radix count
  4n log₂ n − 6n + 8;
- **derived**: no count is cited for the size, so the reference is the sum of its parts
  under the generator's algorithm, from the entries above (`good_thomas(n1, n2)`; at task
  1.1 this is how 6, 10 and 14 are referenced; 20 likewise, and 25 as `cooley_tukey(5, 5)`,
  the parts and 16 general complex multiplies between them: 352 / 184);
- **none**, with a reason: the radix is generated without a count check, and `--table`
  shows "none".

Every codelet must reach its reference (a twiddle variant adds R − 1 complex multiplies),
and the tests and `--table` check it. `--table` shows where the generator does better
(at task 1.1: 7, 11 and 14, fewer multiplies from the mean removal in the prime form).
Where it does worse, the gap is recorded in `reference.GAPS` with the generator's exact
counts and the reason, so any change, better or worse, fails a test and is looked at. At
task 1.1 the one gap is:

- **13**: 192 additions and 134 multiplies against 176 and 68. The symmetric-pair form
  costs about m² multiplies per part for m = 6 pairs; the reference uses Rader's algorithm
  (a length-12 cyclic convolution), outside task 1.1.

The primes 17, 19 and 23 (experiment X27) have no published count and are marked "none":
plain 320 / 242, 396 / 308 and 572 / 464 additions / multiplies in the symmetric-pair form,
with peak live values 20, 22 and 26 (they spill on AVX2's 16 registers). Radix 25
(experiment X42) peaks at 29.

## Adding a radix

1. Check which algorithm `algorithms.choose(R)` picks. Any R that factors into primes the
   builders handle works: powers of two, odd primes (symmetric pairs), odd prime powers
   (Cooley–Tukey) and coprime products of those (Good–Thomas). If none fits, add an
   algorithm first (next section).
2. Add R to `RADICES` in `catalogue.py`. The metadata fields are `std::uint8_t` and
   `std::uint16_t`: R ≤ 255.
3. Give R an entry in `reference.PLAIN` ("Operation counts" above): its best published
   count with `published(adds, muls, source)`, naming the source exactly; if there is none,
   a derived one (for 18 = 2 × 9, `good_thomas(2, 9)`); if neither makes sense,
   `Reference(None, "none: <reason>")`. Without an entry `--table` and the tests stop with
   a message saying so. Run `uv run python -m gen.fftn_gen --table`. If the generator does
   not reach the reference, improve the generator or record the gap in `reference.GAPS`
   with its counts and the reason, and say so in "Operation counts" above.
4. Regenerate (`uv run python -m gen.fftn_gen`) and run the generator's tests.
5. Build and run `fftn_isa_tests` (`tests/isa/codelet_test.cpp`): it runs every codelet in
   the metadata table on every target the machine has, so the new radix is tested without
   any change to the test. With the gate's Release preset:

   ```bash
   cmake --preset check-release
   cmake --build --preset check-release --target fftn_isa_tests
   build/check-release/tests/isa/fftn_isa_tests --gtest_filter='*codelet*'
   ctest --preset check-release -R 'isa\.'   # also the disassembly checks
   ```

   Look at `live` for the new codelets: above 16 the 16-register targets (AVX2) spill,
   above 32 every target does.
6. Commit the generator change and `src/gen/` together. Nothing else lists radices.

## Adding an algorithm

Write a builder in `algorithms.py` that maps a list of input terms to output terms using
only the `Graph` methods (never by constructing nodes), call `dft()` for sub-transforms so
that algorithms nest, and give `choose()` a rule that selects it. Use `g.root_mul` for
constant roots of unity: it reduces the angle and makes quarter turns free. The pipeline
after building needs no change: lowering, ordering, verification and emission work on any
DAG. Add a test of the choice and, if the algorithm is meant to reach a published count,
make sure the reference table has it.

## Adding a lowering

A lowering is a second path from the DAG to a list of operations and a second `.inl`. The
split lowering (task 1.5, below) is the example: its own `Op` kinds, a lowering module next
to `lower.py`, an `evaluate` branch in `verify.py`, an emitter for its text (another class
template beside `cdft`), its value in `gen::lowering` and its own metadata table. Pruned and fused-edge
variants (tasks 1.5 and 2.1) are new variants: a builder that leaves inputs as `ZERO` or
drops outputs before `lower()`, and their own reference in `verify.py`. A pruned codelet
no longer has every output depending on every input, so its schedule must keep all loads
before the first store if it may run in place.

## The split lowering

`split.py` (task 1.5, DESIGN.md 7.2 step 4, experiment X1) lowers the same DAG to the
real-register vocabulary (`real::` in `src/isa/vocabulary.inl`), for radices 2 to 16
(`catalogue.SPLIT_RADICES`), plain and DIF, forward only:

1. **Expand** every complex node into a real and an imaginary node of a `RealGraph`, which
   interns and value-numbers like `Graph` but modulo sign: a rotation by a power of −i
   exchanges and negates parts, so it costs nothing, and parts equal up to sign share a
   node. A twiddle multiply is four real products and two sums.
2. **FMA-shape**: a product (a constant times a value, or two values) whose users are all
   sums is fused into each of them (`fmadd`, `fnmadd`, `fmsub`); one used by anything else
   stays a multiply. Fusing a product shared by u sums costs u operations against 1 + u, on
   fused targets; the text is FMA-shaped, so a target without FMA would need its own. A
   rotation by a constant, c_a·x + c_b·(−i)^r·x whose scales nothing else uses, is built in
   the tangent form c·(u + t·v), t ≤ 1: one fused operation per part, and c fuses into the
   sums that use the parts (radix 16 plain: 156 → 152 operations, DIF 216 → 212; FFTW's FMA
   counts are 144 and 204). The remaining multiplies are scales shared by sums that each
   absorbed another product (k·p ± k'·q pairs after value numbering); factoring those at the
   real level is not done.
3. **Signs**: each register holds its node's value or its negation, chosen backwards from
   the stores as the complex lowering chooses rotations (a − b or b − a; a fused constant's
   sign is free). A store that still disagrees gets a `neg`; none does (a test holds it).
4. Order, verify (`verify_split`: basis inputs in both parts), emit `sdft<R>` into
   `src/gen/split_codelets.inl` and `gen::split_codelets[]` into `codelets.hpp`.

Calling convention: `sdft<R>::plain<T>(in, out)` and `sdft<R>::dif<T>(in, out, twid)`, with
`in.re(j)`, `in.im(j)`, `twid.re(k)`, `twid.im(k)` returning `rv<T>` and `out.put(k, re,
im)`; one line per lane, W lines per call. The backward transform is the forward codelet
with real and imaginary parts exchanged on loading and storing, and the same forward
twiddles (swap(z) = i·conj(z)); the passes do the exchange (`src/passes/split.inl`).
Split metadata counts real-register operations (`adds`, `muls`, `fmas`, `negs`, `vops`)
and live real values: radix 16 needs 34 to 40 registers, which spills on AVX2's 16 and
slightly on AVX-512's 32.

## Subtleties

- **Rotations are cheap only if the stores agree.** A store needs the register to hold the
  output itself, not a rotation of it. The backward propagation of wanted rotations finds
  an orientation for every `Add` with no forced rotation in every codelet so far. If a new
  algorithm needs one, `lower` raises `ForcedRotation`: improve the choice, or add the
  rotating operation to the lowering, the verifier and the emitter together, with a
  codelet that exercises it in the C++ test.
- **Value numbering may, very rarely, miss a merge; it does not merge wrongly.** Keys keep
  2^−150 of 256-bit values, so a missed merge (two equal values falling either side of a
  rounding boundary) needs a boundary within 2^−100 of a value, and costs only an
  operation. A wrong merge would need two different linear functions to agree at three
  random points to that precision, and the exact verification would catch it anyway.
- **Mean removal in the prime form.** Each cosine row sums to −1/2, so with t = Σ s_j the
  cosine sums are (x_0 − t/(2m)) + Σ_{j<m} D_kj (s_j − s_m). That needs m(m − 1) + 1
  multiplies instead of m², with the same number of additions and the same number of
  vocabulary operations. For p = 5 the two D entries are opposite, so value numbering
  shares one multiply between both rows; that is how 5 reaches 12 multiplies.
- **One shared scale is not fused.** Fusing a scale used twice into both users saves a
  vocabulary operation on fused targets but costs an extra multiply on unfused ones. The
  complex-register lowering keeps the shared multiply, so the unfused counts equal the
  algorithm's. FMA shaping proper (tangent-factored rotations) belongs to the split
  lowering.
- **A constant complex multiply** built as two `scale` and an `addj` or `subj` does not
  survive into the output: the real-coefficient form multiplies the parts R and I by real
  constants, and the conjugate pairs of split radix (k and q − k) share those products.
  The vocabulary has no multiply by a constant complex number (DESIGN.md 18.2).
- **Order.** The greedy scheduler keeps fewer values live in total than the builders' own
  creation order, and `schedule` takes the creation order on the one codelet where it is
  better (radix 9); a test compares the two, with the same load and store placement, over
  every codelet. The recursive red/blue partition of Frigo
  (PLDI 1999) was implemented first and lost to both; `schedule.py` names the commit that
  holds it and its numbers. The compiler reschedules anyway; the order is a hint that
  keeps its register allocator out of trouble. A codelet whose `live` exceeds 16 spills a
  little on AVX2's 16 registers, and above 32 on every target; `--table` shows `live` (at
  task 1.1, radices 14 to 16 were at 17 and 18, and still are in real-coefficient form;
  32 is at 35 and 64 at 67 in the fewest-live order, 37 and 71 in the fitted order that
  the generated text carries, which spills less).
- **Constants** are printed twice, each the exact value rounded once to its precision (to
  double, and to float directly: rounding the double to float would round twice), and
  `gen_k<T>` picks one. The double literal is the shortest decimal that reads back as that
  double; the float literal has nine significant digits, which always read back as that
  float. A test checks both roundings.
- **Determinism.** Random samples come from fixed seeds and every iteration that reaches the
  output is over ordered containers, so the output is identical on every run and machine.
  The gate depends on it.
