# X23: masked partial runs in the x86 vocabularies

- **Status:** running
- **Owner:** masked-tails session, 2026-11-15
- **Phase / task:** 1 (x86 kernels; follow-up of [X22](X22-icpx.md))

## Question

Do partial runs written as masked vector accesses in the `avx512` and `avx2` vocabularies
close most of icpx's float lead with GCC and Clang (X22: icpx 0.67 of GCC's time and 0.80
of Clang's on X2's float 7-smooth rows), so that they become the default on x86?

## Hypothesis

X22 traced icpx's float advantage at n = 105 (7 × 15, every unit partial) to the
element-by-element tile of `load_part` and `store_part` (`passes/access.inl`) and of
`store_tiled` (`passes/stockham.inl`): icpx turned it into masked AVX-512 loads and stores,
Clang partly, GCC not at all (scalar stores into a stack tile, then a 512-bit load of it: a
store-forwarding stall per leg). Writing the masked accesses in the vocabulary should give
GCC and Clang what icpx found by itself: about 0.7 of GCC's time and 0.85 of Clang's for
float rows whose passes are mostly partial, little in double (C = 4 and 2, short tails),
and nothing where every run is full (powers of two). Every arm moves data only; the
arithmetic and so the bits are main's.

## Where partial runs happen

A unit of a pass is partial where fewer than C butterflies (grouped pass, s = 1, m
butterflies: the last unit when C does not divide m) or copies (lane and generic passes:
the last block of each group when C does not divide s) remain; Bluestein's chirp passes
have one at the end of the data; the lanes strategy has one where C does not divide a
block's rows. Split passes have none (W divides s). The grouped pass also stores every
unit through the tile when C does not divide its radix R (`store_tiled`), and its last
unit always did. Counted from planner v1's radices (`rule_radices`, split runs as
`chain_builder::split_from` chooses them) for X2's 105 lengths from 64 to 1024, batch 1,
chain strategy:

| Target, precision (C) | lengths with a partial or tiled unit | with a partial lane pass | partial-run units | tiled-store units (R mod C ≠ 0, or the last unit) | all units |
|---|---|---|---|---|---|
| `avx512` float (8) | 91 | 61 | 427 | 325 (62 lengths with R mod 8 ≠ 0) | 2403 |
| `avx2` float (4) | 72 | 41 | 299 | 392 (42 lengths) | 5002 |
| `avx512` double (4) | 72 | 41 | 299 | 392 (42 lengths) | 4870 |
| `avx2` double (2) | 42 | 21 | 206 | 422 (22 lengths) | 9502 |

Examples (`avx512` float): 105 = 7 × 15: grouped radix 7 (m = 15: 1 of 2 units partial,
every unit tiled), last pass radix 15 with s = 7 (its only unit partial); 375 = 15 × 5 × 5:
grouped radix 15 (tiled; 1 of 4 partial), middle radix 5 with s = 15 (5 of 10 units
partial), last radix 5 with s = 75 (1 of 10); 441 = 9 × 7 × 7 likewise; powers of two
have none.

## Change under test

Arms are commits on branch `exp/x23-masked-tails`, each on the one before; every arm keeps
main's bits (data movement only), so the targets' bit agreement and the accuracy are
untouched.

- **A** (main 2784f5d): partial runs through a tile, element by element.
- **B** (9093a3e): the vocabularies of the
  two x86 targets get `load_n(p, n)` and `store_n(p, n, a)` (`vocabulary.inl`; flag
  `masked`): `avx512` with `__mmask8`/`__mmask16` masks of 2n reals
  (`vmovupd/vmovups {k}{z}`), `avx2` with `vmaskmovpd/vmaskmovps` and a mask from one
  compare. `load_part` and `store_part` use them where the run's elements are adjacent
  (step of one complex number: every grouped, lane, generic and chirp pass with C > 1, and
  the lanes strategy over adjacent rows); strided runs keep the tile. The grouped pass's
  last unit, for R a multiple of C, goes through the same transposes as a whole unit and
  stores only its butterflies. NEON and scalar unchanged (`masked = false`).
- **T** (5491584, on B): the grouped pass's
  `store_tiled` for R mod C ≠ 0 on the masked targets pads the R legs with zero registers
  to whole C × C blocks, transposes them, and stores each butterfly's R outputs as whole
  runs plus one `store_n` of R mod C. Every unit of such a pass is affected, not only the
  last; it costs C × C transposes of partly empty blocks, which may lose at small R (3, 5)
  with C = 8.
- **P** (34717f1, on T): `avx2`'s `store_n`
  without VMASKMOV (microcoded and slow as a store on AMD Zen 1-3), as one or two plain
  stores of 128 and 64 bits chosen by n. Loads stay VMASKMOV (cheap everywhere). `avx512`
  is the same as T.

No switch: the arms are commits, so `dev/KNOBS.toml` is unchanged.

### Amendment (2026-11-15, written after B's first runs and before B5's runs)

B as first written failed the rule on the Xeon `avx512` with GCC 14 (run
20261115-081316-fb77, Results): 63 cases slower than A beyond the band, up to 1.09, most
of them lengths with no partial unit at all (double 128 = 8 × 16, 512, 1024; float 320,
640, 768, 960). Every one runs a grouped pass with R a multiple of C. Two causes, found
before any run of the fixed arms (perf, disassembly and a 22-case probe, Results):

1. B's last-unit route for R a multiple of C looped `store_transposed` to the runtime
   `valid`, indexing the leg registers `v[R]` at run time; `v` is one array for both
   branches of the unit loop, so the compiler kept it in memory. Clang 23 suffered most
   (probe: B 1.16–1.46 of A at double 320, 448, 512 and float 128, 320); GCC's whole-unit
   path kept `v` in registers. Fix: constant loop bounds with `valid` tested inside.
2. With the partial units' masked code inlined in the loops over units, the compilers
   hoisted its set-up (masks, a pointer per leg) into each pass's prologue and lost
   registers in the loop over whole units. GCC 14: 4 % more instructions per transform at
   double 128 (perf), the lane pass's loop moving pointers through vector registers; Clang
   23: the grouped radix-8 float pass's set-up grew from 80 to 350 instructions, so with
   fix 1 alone (B2, 2fddd8d) float 128 was still 1.31 of A (GCC: double 448 1.09). Fix: on
   the masked targets the grouped pass's loop runs whole units only and the last unit
   follows it; the lane pass's loop skips its partial blocks and a second loop runs them
   (units are independent, so the order changes no bit). Nothing is called: the code
   stays inline, only out of the hot loops. NEON and scalar keep main's code (B4,
   3bbd3fd). B4's first runs showed one more instance of cause 1: in the grouped pass's
   loop the unit's butterfly count stayed a runtime value although every unit there is
   whole, so for R not a multiple of C `store_tiled` looped over it instead of unrolling
   (Clang 23: double 75 = 5 × 15 1.60 of A on `avx2`, 1.16 on `avx512`; T, whose padded
   store takes the constant, did not have it). Fix: the count is the constant C in that
   loop on the masked targets (B5).

- **B5** (1132477, on B4): B with all three fixes.
- **T5** (0baf6a3, on B5) and **P5** (489b268, on T5): T and P replayed on B5, otherwise
  unchanged (T's padded store of the last unit moves into the grouped pass's tail).

The decision rule applies to B5, T5 and P5 in place of B, T and P, word for word, on new
runs of A against them with the Method's cases, machines and compilers. B, T and P as first
written are judged on the runs already made with them and cannot be adopted whatever B5's
result; B4's runs are reported as evidence, not judged. Two earlier stacks were submitted
for timing and cancelled before any result:
B2–P2 (2fddd8d, ae43de2, d8fa8d8), when the probe showed fix 1 insufficient, and B3–P3
(7c3c531, 6c9a8e1, 534b9e6), which made the partial units `noinline` calls and so broke
the regions' rule that nothing in a region calls a function (`isa.asm_check` failed in
CI); they are not judged. The branch merges B2–P2, B3–P3, B4–P4 and then B5–P5, so its
head carries B5–P5.

## Method

- Cases: (1) X22's run 2 set, X2's catalogue at every 7-smooth length 64–1024, batch 1,
  out of place, float and double (210 cases):
  `--catalogue x2 --tier full --filter '^c2c-(6[4-9]|[7-9][0-9]|[1-9][0-9][0-9]|10[0-2][0-9])-b1-op'`.
  (2) Collateral: the standard catalogue's tracking tier (`--tier tracking`).
- Machines, compilers, instruction sets: Xeon Gold 6240 (Cascade Lake; Slurm `-c 8`,
  shared) with `ISA=avx512` and `ISA=avx2`, each with GCC 14.4 and Clang 23; the EPYC
  7763-class node (Zen 3) with `ISA=avx2` (its best), GCC 16. One thread. fbench
  interleaved process rounds with the A/A control, `--max-load 80 --max-busy 60` on the
  Xeon as in X22.
- Metric: per case, fbench's median paired ratio against A (and the arms' ratios to each
  other from the stored timings); per run and precision, the catalogue-weighted geometric
  mean with its bootstrap interval; the A/A band.
- Decision rule, written before measuring:
  - **B** becomes the default on a target if, on the Xeon with both GCC 14 and Clang 23,
    its float geomean over cases (1) is at most 0.97 of A with the interval clear of the
    band, its double geomean is at most 1 + band, and no case of (1) or (2) is slower than
    1 + band + 2 %. For `avx2` the EPYC must meet the last two conditions too; if only the
    EPYC fails, P decides.
  - **T** is kept on a target if its geomean over the float cases of (1) is at most 0.99
    of B's with the interval clear of the band, with both compilers on the Xeon, and no
    case of (1) or (2) is slower than B by more than band + 2 % (EPYC included for
    `avx2`); otherwise its commit is reverted for that target (or wholly).
  - **P** replaces the VMASKMOV stores of `avx2` if its geomean over the float cases of (1)
    on the EPYC is at most 0.99 of T's with the interval clear of the band, and on the
    Xeon `avx2` it is no slower than T beyond the band. If it wins on the EPYC but loses on
    the Xeon (llvm-mca predicts that: 8.0 against 4.5 cycles per partial unit on Cascade
    Lake), the store whose float geomeans multiplied over the two machines are lower is
    kept (`avx2` is the EPYC's best target and, after X16, possibly the Xeon's in float),
    and a choice per vendor becomes a follow-up: one `avx2` vocabulary serves both.
  - If B fails, the masked accesses are deleted and the tile stays.
  - Descriptive, not part of the decision: the fraction of icpx's float lead closed. X22
    measured icpx ÷ GCC 0.668 and icpx ÷ Clang 0.795 on cases (1); "most" means the
    chosen arm's float geomean against A is at most 0.834 with GCC and 0.898 with Clang
    (more than half of the gap). An icpx build of main and of the chosen arm (X22's
    campaign) would measure it directly.

### Commands (for the lead)

`xeon` is the GCC 14.4 profile;
`xeon-clang` is the same machine with `cxx` set to Clang 23 (X22's toolchain); `epyc` is
the DMOG profile (GCC 16). Each `submit` is one job; check the first job's log.

```bash
R="uv run --frozen python scripts/remote/fbench_remote.py"
X2='^c2c-(6[4-9]|[7-9][0-9]|[1-9][0-9][0-9]|10[0-2][0-9])-b1-op'
A=2784f5d; B=9093a3e; T=5491584; P=34717f1
Q="--max-load 80 --max-busy 60 --budget 10800"
for prof in xeon xeon-clang; do
  $R submit $prof A=fftn:$A+ISA=avx512 B=fftn:$B+ISA=avx512 T=fftn:$T+ISA=avx512 \
    -- --catalogue x2 --tier full --filter "$X2" $Q
  $R submit $prof A=fftn:$A+ISA=avx2 B=fftn:$B+ISA=avx2 T=fftn:$T+ISA=avx2 P=fftn:$P+ISA=avx2 \
    -- --catalogue x2 --tier full --filter "$X2" $Q
  $R submit $prof A=fftn:$A+ISA=avx512 B=fftn:$B+ISA=avx512 T=fftn:$T+ISA=avx512 \
    -- --tier tracking $Q
  $R submit $prof A=fftn:$A+ISA=avx2 B=fftn:$B+ISA=avx2 T=fftn:$T+ISA=avx2 P=fftn:$P+ISA=avx2 \
    -- --tier tracking $Q
done
$R submit epyc A=fftn:$A+ISA=avx2 B=fftn:$B+ISA=avx2 T=fftn:$T+ISA=avx2 P=fftn:$P+ISA=avx2 \
  -- --catalogue x2 --tier full --filter "$X2" --budget 10800
$R submit epyc A=fftn:$A+ISA=avx2 B=fftn:$B+ISA=avx2 T=fftn:$T+ISA=avx2 P=fftn:$P+ISA=avx2 \
  -- --tier tracking --budget 10800
```

The amendment's runs put both Xeon instruction sets in one job per compiler and case set
(the build is the same; ISA is a run option): arms `A512 B512 T512 A256 B256 T256 P256` =
A, B5, T5 with `ISA=avx512` and A, B5, T5, P5 with `ISA=avx2`, so the A/A band is A512's
(jobs `x23b5-xclang-x2`, `x23b5-xeon-x2`, `x23b5-xeon-trk`, `x23b5-xclang-trk`; at most
three of them at once on the Xeon, the others chained). The EPYC runs A, B5, T5, P5 with
`ISA=avx2` (`x23b5-epyc-x2`, `x23b5-epyc-trk`). B4's runs had the same shape (`x23b4-*`).

### Checks made before timing (Mac, M5 Max)

- Correctness: `tests/core/tails_test.cpp` forces every partial-run shape (grouped pass
  m and lane pass s leaving each remainder modulo 8, radices C does and does not divide,
  Bluestein, generic passes, the lanes strategy over 2–9 adjacent rows), double and
  float, both directions, in and out of place, every array ending exactly at a guard
  page, against the quadruple-precision reference, with avx2 and avx512 bit for bit
  where both run; the vocabulary test (`tests/isa/vocab_test.cpp`) checks `load_n` and
  `store_n` for every n from 0 to C against a canary. Every test program runs under SDE
  (Haswell, Skylake-SP) in the emulation workflow. The `avx2` target ran on the Mac
  through an x86-64 build (Apple Clang 21, `-arch x86_64`) under Rosetta 2 with
  `ROSETTA_ADVERTISE_AVX=1`, which executes AVX2 and FMA and reports them in CPUID.
  Replacing `store_n` or `load_n` by a whole-register access makes that test fault at
  the guard page, so it does catch an access past the array. `avx512` was compiled and
  disassembled only; it runs under SDE in CI and natively on the Xeon.
- Bits: on `avx2` under Rosetta, the outputs of every length 2 to 1100 (batches of 3, of 64
  and of 5 interleaved rows up to 64 points; both precisions and directions) hash equal for
  main and for T and P.
- Disassembly (Apple Clang 21, the library's `avx512` and `avx2` objects, main against T
  and P; instructions and stack references per function). `avx512` float: the radix-15
  lane pass 2464 → 1499 instructions (393 → 309 stack references), radix 7 1180 → 761,
  radix 5 876 → 552; the grouped radix-7 pass 1369 → 969, radix 9 1743 → 1240, but radix 15
  1814 → 1958 (two padded 8 × 8 transposes; 270 → 319 stack references) and radix 8 1239 →
  1242 with 200 → 286 stack references (its last unit's transposes). Clang already turned
  part of main's tiles into masked AVX-512 accesses (29 masked instructions in main's
  radix-15 lane pass, 72 in its grouped radix 8), as X22 found; GCC did not, so the gain
  there should be larger with GCC, which was not available for x86 on the Mac. `avx2`
  float: within 10 % of main's instruction counts. Every lane, generic and chirp pass keeps
  the tile for strided runs behind the runtime `step == 2` test although its contract
  makes runs contiguous when C > 1: a clean-up if an arm is adopted.
- Throughput models of one partial unit (four legs of a partial run loaded, added and
  stored; a scratch file, not committed), uiCA for Cascade Lake and llvm-mca: `avx512`
  float, 7 of 8 valid: tile 14.0 cycles, masked 5.5 (uiCA CLX; mca 12.0 and 4.0); `avx2`
  float, 3 of 4 valid: tile 9.1, VMASKMOV 8.3, pieces 9.0 (uiCA CLX; mca 8.0, 4.5, 8.0);
  on Zen 3 (mca znver3) tile 8.0, VMASKMOV 48.0 (its masked stores), pieces 8.0. Neither
  model charges the tile's store-forwarding stalls (X22's GCC code), so both understate
  the tile's cost; the Zen 3 figure is why arm P exists.

## Results

(pending: the lead's runs)

## Verdict

(pending)
