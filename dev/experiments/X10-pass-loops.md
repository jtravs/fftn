# X10: pass loops on x86: addressing, register budget, shuffles

- **Status:** running
- **Owner:** x86-perf session, 2026-11-14
- **Phase / task:** 1 (exit criteria); follows X18

## Question

The plan's X10 asks whether generated whole passes beat templated pass loops. X18 found
what such loops must fix on x86: for the same 16 × 16 plan, FFTN's AVX2 loops execute 1.2–1.3
times FFTW's instructions per butterfly pair (5386 against 2340 per 256-point transform with
the default plan), from per-unit address arithmetic, spills on 16 registers, and shuffles.
Can the templated loops reach FFTW's instruction counts, or do they need generating? The
answer decides whether X10 becomes generator work (whole passes) or stays in the templates.

## Hypothesis

From X18's hot-loop listings (Xeon, `ISA=avx2`, 16 × 16; FFTW `n2fv_16` 176 and `t2fv_16`
222 instructions per two butterflies):

1. **Per-unit address arithmetic.** The plain radix-16 lane loop spends 60 (Clang) to 72
   (GCC) of its 204–211 instructions on scalar work: a multiply per unit for each base
   pointer, the tail test, the group and block counters, and the 2R leg offsets, which do not
   fit 16 registers (GCC rebuilds them every unit, and on AVX-512 parks them in vector
   registers with 23 `vmovq` on port 0). Fixed pointer steps per unit, the partial unit out of
   the loop, and the offsets of radix ≥ 8 read from a table at each access (FFTW's scheme)
   should bring it near FFTW's 21–34.
2. **Spills.** The grouped radix-16 pass keeps all 16 outputs live until its transposes
   (GCC: 51 stores where 16 are outputs). Storing each C × C block as soon as its legs exist
   should remove most of them.
3. **Shuffles.** The radix-16 codelet uses 19 shuffles where FFTW's core uses about 7, and
   the grouped pass adds 15 twiddle swaps and 16–32 transpose shuffles. Fewer rotations by
   ±i in the generator's output, and the transpose stores split from the twiddle multiplies,
   reduce port-5 pressure on Intel.

Each is a separate change and arm; 1 is measured first.

## Change under test

Arms are commits of branch `exp/x10-pass-loops` from main 8affea0:

- A: 8affea0 (main).
- L: bdd3946, unit loops with fixed pointer steps, partial units after the loop (leg tables
  present but off).
- T: 6ba3297, L plus leg offsets from a table for radix ≥ 8 on `avx2` and `avx512`.
- Later arms (block-wise transposed stores, generator shuffles) are added here before they
  are measured.

No switch: arms are commits; `dev/KNOBS.toml` is untouched. None changes the arithmetic or
its order (only addressing, loop shape and store order), so outputs are bitwise identical to
A; the tests' cross-target comparisons and fbench's check confirm it.

## Method

- Cases: `x18.toml` (X18; c2c double rows 64–16384, 60, 120, 360, 1000; batches 128 × 4096
  and 1024 × 256), on the Xeon with `ISA=avx512` (default) and `ISA=avx2`, on the EPYC with
  `avx2`. NEON: the standard catalogue's tracking c2c rows on the Mac.
- Compilers: GCC 14 and Clang 23 on x86; Homebrew Clang 23 on the Mac.
- Metrics: (a) instructions and cycles per transform from `perf stat` on the Xeon at 256
  (16 × 16 and default), 1024 and 4096, both ISAs; (b) fbench median paired ratio arm ÷ A per
  case and geomean, against the A/A band, and FFTN ÷ FFTW before and after.
- Decision rule, written before running: adopt an arm if its geomean ÷ A is ≤ 0.98 on at
  least one x86 machine and ISA, with its interval clear of the A/A band, no case slower than
  1 + band + 2 % on any machine, ISA or compiler, and on the Mac the tracking c2c rows' geomean
  ≤ 1.01 with no case slower than 1.03. An arm that only removes instructions without a time
  gain is rejected (deleted).

## Results

To be filled in.

## Verdict

To be filled in.
