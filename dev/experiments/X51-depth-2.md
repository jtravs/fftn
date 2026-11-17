# X51: the depth-first chain's loose ends (after X46)

- **Status:** running
- **Owner:** x51 session, 2026-11-17 (overnight)
- **Phase / task:** 1 (phase 1 exit check, the tail rule); follows X46

## Question

Which of the rows X46 left open does the depth-first chain (DESIGN 8.5, `c2c_depth`) close,
and by which rule: Xeon 25000 (1.33 of FFTW: three factors of 2, so with C = 4 no top
factor and its sub-row are both multiples of C), Xeon 100000 (1.02 of FFTW), double rows
beyond 4 L2 (two tops), single precision on the Xeon and the Mac, the Mac's 24576 and 49152
(grids, 0.71–0.72 with a top of 4 in X46's screen), the EPYC's tops of 4 (46080 0.83, 20000
0.94 in X46's screen, losses elsewhere)?

## Hypothesis

- X46's diagnosis carries: wherever three lines and a first pass's twiddle table overfill
  L2, a depth-first chain (two lines, middle passes in cache, derived top twiddles) cuts the
  misses. 25000 lacks only a split the passes accept; a first pass whose last block of
  sub-rows is partial (C not dividing the top factor) gives it tops of 5 or 10 at a cost of
  the wasted lanes in one pass of five.
- Beyond 4 L2 one top leaves sub-rows beyond L2 (524288 / 16 = 512 KiB); two tops (16 × 16,
  sub-rows of 2048 points) keep them in L1 for one more pass over the row.
- Single precision halves the bytes: the byte range of the double rule maps to twice the
  points, and tops must be multiples of C = 8 on `avx512`.
- The EPYC (16 registers, 512 KiB L2) and the Mac (shared L2) need their own screens.

## Change under test

Arms are commits of branch `exp/x51-depth-2`, chosen after the screens (step 1) and written
here before their judging runs:

- **P (partial first pass).** `depth_first_pass` takes a top factor that C does not divide
  (the last block of each group of sub-rows is partial, as in the lane pass); no plan
  changes by itself; forced splits only.
- Further arms (rules) are named below after the screens.

Bits: a changed plan changes its rows' rounding; tests hold the derived tolerances in both
precisions and bitwise equality between serial and threaded runs.

## Method

- Machines: Xeon Gold 6240 (`avx512`, GCC 14.4, FFTW 3.3.11 AVX-512, oneMKL 2026.1; Slurm
  `-c 8` shared; judging through `fbench_remote.py submit --serial`), EPYC 7763 (`avx2`,
  GCC 16, DMOG `-c 8`), M5 Max (`neon_fcma`, Homebrew Clang, NEON FFTW).
- Step 1, screens (no default change): catalogue `x51` (bench/catalogues/x51.toml), the
  rules' plan against forced splits (`STRATEGY=depth`, `DEPTH_TOPS`), FFTW and MKL where
  they time the case.
- Step 2, each rule arm against the arm before, fbench interleaved rounds with the A/A
  control, on the standard catalogue's rows the rule changes (`--tier full`), then the
  standard tracking tier.
- Decision rule (X46's, unchanged), written before any judging run: an arm is adopted on a
  machine's profile (the release toolchain: GCC on x86-64, Clang on macOS) if over the cases
  whose plan it changes the weighted geomean B ÷ A is below 1 − band with its interval below
  1, no such case is slower than 1 + max(band, 3 %), and no tracking-tier case is slower than
  1 + max(band, 3 %) (a case just past the limit on a loaded run is re-run once). A profile on
  which an arm is not judged keeps its plans. At most three rule arms.

## Results

### Step 1, the screens

## Verdict
