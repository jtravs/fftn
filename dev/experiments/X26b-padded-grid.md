# X26b: long rows, a padded intermediate for the grid

- **Status:** running
- **Owner:** X26b session, 2026-11-15
- **Phase / task:** 1 (PLAN lead 1); follows X26, X4 (its x86 half), X12, X12b

## Question

Does padding the grid's intermediate array, so that the rows a phase gathers are not a
multiple of 4 KiB apart, remove the L1 set conflicts X26 found in the grid's copies on x86,
and do the cheaper copy loops X26 rejected (arm B) pay once the conflicts are gone? The
answer changes the grid's workspace layout (`src/strategies/grid.cpp`), the copy loops
(`src/passes/grid.inl`) and whether the copies' prefetch can follow the core rather than
the target (`copies_prefetch`).

## Hypothesis

X26 measured on the EPYC (Zen 3, `avx2`) that a faster copy loop made rows of 2²⁰ points up
to 1.56 times slower with a fifth fewer instructions and twice the fills from beyond L2,
and that removing the copies' prefetch made them 0.92 of main's time. Its reading: every
row a block gathers lies a multiple of 4 KiB from the last, so a block's lines share one
or two of L1's 64 sets (8 ways; the same holds for L2's 64 KiB set period); more rows in
flight evict the scratch block the lane passes read next. On x86 every grid is copied (the
fusion limit, L2/4, lies below the grid's start at L2/2), and in the plans of the target
cases (EPYC: 65536 = 32 · 32 · 64 in blocks of 32, 2²⁰ = 64 · 128 · 128 in blocks of 16,
46080 = 32 · 36 · 40; Xeon `avx512`: 65536 = 256 · 256 and 46080 = 192 · 240 in two phases,
blocks of 4) the strided copies are:

| copy | array | row stride (elements) | paddable |
|---|---|---|---|
| phase 1 gather | input | n / N1 | no (the caller's) |
| phase 1 transposed write | intermediate | contiguous per block | (yes) |
| phase 2 gather | intermediate | n / N2 | **yes** |
| phase 2 scatter (three phases) | output | N1 | no |
| phase 3 gather and scatter | output | N1 N2 | no |

Only phase 2's gather reads an array FFTN owns. Padding each of its rows by p elements
(chosen so that the N2 rows of a block spread evenly over L1's sets) should cut L1 and L2
refills in phase 2 (counted with `run --pmu`), most where the stride is a power of two
times 4 KiB and the block is narrow (2²⁰, 2²², real 2²⁰, the Xeon's two-phase 65536 and
46080 whose phase 2 gathers rows exactly 4 KiB apart). Expected: 3–10 % on those rows on
x86, nothing on the Mac (X4: padding on top of the stagger was within the band there; its
L1 has a 16 KiB set period). With phase 2's conflicts gone, arm B's faster loops may then
pay in phase 2, but phase 1 and phase 3 still copy over power-of-two strides of the
caller's arrays, so B may still lose there: B is retried, not expected to win.

## Change under test

- Arm A: `main` at the branch point (0dc9391).
- Arm P (arm 1): A plus a padded intermediate. The machine profile gains
  `grid_pad_period`, the L1 set period (L1 size ÷ ways: 4 KiB on both x86 machines;
  unknown on macOS, which does not report the ways, so no padding there). The grid pads
  each of phase 2's rows of the intermediate by p complex elements, p the multiple of the
  block (0 to 8 blocks) whose N2 rows load L1's sets (64-byte lines) most evenly, smallest
  first. Phase 1's blocks then write a padded row each (three phases: nested loops, one
  per phase-2 row; two phases: a row stride of N1 + p in the twiddled transpose). A
  two-phase plan out of place, whose intermediate today is the output, moves it into the
  workspace block when p is not 0 (n + N2 p complex elements more workspace). Results are
  bit-identical (padding moves data, not arithmetic; tested). fbench's
  `GRID_PAD_PERIOD` option sets the field (`never`: no padding) for the screens.
- Arm PB (arm 1b): P plus X26's arm B copy loops (8ac713f, d232d7a: incremented pointers,
  rows of 1, 2, 4 or 8 vectors without an inner loop, a prefetch only on the side whose
  rows lie apart), keeping `copies_prefetch`. Bit-identical, no plan change.
- Arm PP (screen): P with the copies' prefetch on `avx2` as well (`copies_prefetch` true),
  to see whether the padding makes X26's per-target prefetch question moot.
- Arm 2 (only if the padding does not make it moot): the prefetch follows the core, a
  vendor and family field in the machine profile replacing `id != target::avx2`.

## Method

- Cases: `bench/catalogues/x26b.toml`: c2c 16384, 65536 (out of place and in place),
  262144, 2²⁰, 2²², 46080, 100000; r2c and c2r 65536, 2²⁰, 2²² (double); float c2c 65536,
  2²⁰, 2²², 100000 and r2c 2²⁰ as collateral. FFTW times double only (fbench). Collateral:
  the standard catalogue's tracking tier for the adopted arm.
- Machines: the EPYC 7763 (DMOG, `avx2`, GCC 16, `-c 8` shared; the main machine); the
  Xeon Gold 6240 (`avx512`, and forced `ISA=avx2`, GCC 14, `-c 8` shared); the Mac (M5
  Max, `neon_fcma`, Homebrew Clang, shared; one fbench run at a time). One thread, hot.
  fbench A/B with interleaved rounds and the A/A control, `--serial` on the remote
  machines.
- Counters: `run --pmu` on the EPYC and the Xeon, A against P, for 2²⁰, 100000 and the
  Xeon's 65536: L1 refills (Zen 3: `l1d` misses / `ls_refills_from_sys`; Intel:
  `L1-dcache-load-misses`, `l1d.replacement`), cycles and instructions, to check that the
  padding removes the refills it targets rather than only moving time.
- Metric: median paired ratio to the previous arm per case with its 95 % interval; geomean
  over the double target cases; FFTN ÷ FFTW.
- Decision rule, written before running, per machine and target (the release toolchains:
  GCC on x86-64 Linux, Clang on macOS; AGENTS.md section 6):
  1. An arm is adopted on a target when its geomean over the double target cases, against
     the previous arm, lies below 1 − band (the run's A/A band) with its interval clear of
     it, and no case (target, float collateral or tracking tier) is slower than
     1 + max(band, 3 %). A case just past that limit on a loaded run is re-run first.
  2. P is a layout rule keyed to the profile: adopted where it passes rule 1 against A
     (x86 machines, as their L1 geometry says); on a target where it does not, the
     profile turns it off (the Mac stays unpadded unless it passes there too with
     `GRID_PAD_PERIOD` forced).
  3. PB is adopted per target where it passes rule 1 against P (the loop form gated by
     target, as `copies_prefetch` is); rejected where it does not.
  4. PP: if it passes rule 1 against P on the EPYC, the prefetch returns on `avx2`
     (`copies_prefetch` removed) and arm 2 is moot; if it is within the band there and
     the Xeon under `ISA=avx2` passes rule 1 with it, PP is adopted as well (one rule for
     all cores); otherwise arm 2 is built and judged by rule 1 on the Xeon under `avx2`
     with the EPYC unchanged.
  5. At most three arms beyond A; a failed approach is written up with its numbers and
     stops.

## Results

## Verdict
