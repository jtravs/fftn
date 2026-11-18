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
  which an arm is not judged keeps its plans. At most three rule arms per profile.

## Results

### Step 1, the screens

Builds f0fc171 / beed63c / ea6b9d9 (X46's head 80dbcdb plus this file, the catalogue and arm
P; the rules' plans are `main`'s). Ratios are forced split ÷ the rules' plan in the same run
(below 1: faster); where a forced split does not fit the row is a plain chain (not shown). F,
M: FFTW, MKL ÷ the rules' plan.

- **Xeon, 25000 and 100000** (run 20261117-235845-911c, Slurm 16428/16429, 12 quiet rounds,
  band ±2.37 %): 25000 with arm P's tops 10 × 2500 **0.776**, 5 × 5000 0.797 (FFTW 0.771,
  MKL 1.056: FFTN ÷ FFTW 1.30 → 1.01); 100000: 20 × 5000 is the rules' plan, 10 × 10000
  1.028, 5 1.27, 8 1.13 (FFTW 0.981: FFTN ÷ FFTW 1.02, at parity within the band); 20000: 10
  0.957, 5 1.10, 8 1.08.
- **Xeon, double rows beyond 4 L2 and 100000** (run 20261117-234639-ec26, Slurm 16424/16425,
  12 rounds, 6 loaded, band ±1.81 %): one top of 16 took 0.619 (524288), 0.718 (2²⁰), 0.713
  (2²¹), 0.757 (2²²) of the rules' grids; two tops (16 × 16, 16 × 32, 32 × 16, 32 × 32) and a
  top of 8 were no better (0.60–0.93); FFTW 1.27–1.54 and MKL 1.00–2.19 of the rules' time
  there. 100000: no split beat 20 × 5000 (20 × 5 1.085, 20 × 10 1.039, 4 × 10 1.134).
- **Xeon, single precision** (run 20261117-235845-911c): from L2/4 (32768 points) up one top
  of 16 (20 where 16 does not fit) took 0.795 (32768), 0.656 (46080; 32 0.622), 0.725 (65536;
  32 0.623), 0.679 (100000, top 20, a partial first-pass block on 8 lanes; 5 0.594), 0.605
  (131072), 0.648 (262144), 0.696 (524288), 0.741 (2²⁰; 32 × 16 0.651); below L2/4 every split
  lost (8192 1.16–2.0, 16384 1.14–1.36, 20000 1.49–1.54).
- **EPYC, double** (run 20261117-234716-9091, Slurm 512514/512515, 12 rounds all loaded, band
  ±1.00 %): one top of 4 took 0.755 (12288), 0.725 (24576), 0.765 (49152), 0.698 (98304),
  0.831 (46080), 0.921 (40000), 0.915 (20000), 0.958 (16384), but lost at 25000 (1.157),
  262144 (1.168), 524288 (1.150) and tied on the other powers of two (32768–131072, 1.02–1.06,
  no evidence); tops of 16 lost everywhere (0.92–1.70), two tops did not beat one. FFTW's
  times: 0.78–0.81 of the rules' at 12288–98304 (FFTN behind), 1.00–1.50 elsewhere.
- **M5 Max, double and float** (run 20261117-235532-bd1a, 8 rounds, all loaded, band
  ±11.2 %; run 20261118-000640-8f33, 16 rounds, all loaded, band ±4.85 %, the library rebuilt
  once during the run with an arm that changes no Apple plan): rows of an odd part of 3–15
  times a power of two above L1/8 (part 4096 points and more) took 0.67–0.75 with one top of 4
  from 20480 to 81920 points, 0.754 at 122880 and 0.856 at 98304 (top 8 0.760); beyond 2 MiB
  a top of 4 lost (147456 1.57, 196608 1.36, 393216 1.36); powers of two from 32768 to 262144
  gained nothing (0.89–1.11, no evidence at that band); the X46 range (8192, 16384, 12288)
  is unchanged. Float: no split cleared the band of the first run; a float screen (run
  20261118-011734-7804, 16 rounds, all loaded, band ±4.70 %) gave a top of 4 0.69 (40960,
  49152), 0.84 (98304, 122880) on mixed rows but 1.06 at 24576 and 1.03 at 196608, and
  nothing on powers of two (0.90–1.10) or 100000 (1.16): no float arm for the Apple profile
  (its rule would need a range of its own).

### Arms named after the screens (before their judging runs)

- **T (Xeon profile)**, c6d4c4a: a third top, `depth_top_last` = 10, tried after 16 and 20
  (arm P's partial first-pass block): 25000 = 10 × 2500 on `avx512`. Judged against P's
  plans (beed63c, `main`'s) on c2c 25000 out of place, in place, backward, and the tracking
  tier.
- **Generic limits** (3a6bda4, no plan change): `depth_pow2_max_bytes`, `depth_float`, tops up
  to 8 on `avx2`.
- **M (Apple profile)**, df67c89: the range from L1 to 16 L1 for rows that are not powers of
  two (powers of two stay at 2 L1), top 4, power-of-two part above L1/8 as before. Judged
  against N = 1a78533 (the Mac's plans as `main`) on the x51 catalogue's mixed rows
  (20480–122880; 24576, 49152, 98304 in place and backward, batches of 8) and the tracking
  tier.
  The standard catalogue has no row this changes (its non-power-of-two rows in range have
  power-of-two parts of 1024 points or fewer).
- **E (EPYC profile; x86-64, private L2 with a large L3)**, 83bd92f: rows that are not
  powers of two, L2/4 to 4 L2 (8192–131072 complex doubles on the EPYC 7763), power-of-two
  part above L1/4 (512 points), one top of 4 (the generic commit's 16-register limit). Judged
  against N = c3a5eba (`main`'s plans on the EPYC) on the x51 rows it changes (12288,
  20480–122880, 46080; 24576, 49152, 98304 in place and backward, batches of 8) and the
  tracking tier (which holds 46080, the one standard catalogue row it changes). Zen 5 (the
  same profile branch) takes it unmeasured; the AArch64 machines of that branch do not.
- **L (Xeon profile)**, c2b30f2: the range from 4 L2 to 64 L2 (to 2²² complex doubles),
  same tops; a third screen (run 20261118-001544-4037, Slurm 16435/16436, 12 rounds, 4
  loaded, band ±11.1 %) gave one top of 16 0.83 / 0.72 / 0.74 in place (524288, 2²⁰, 2²²),
  0.67–0.77 backward, 10⁶ 0.89 (top 20 0.82), 2²³ 0.88 and 2²⁴ 0.94 (within the band: left
  out). Judged against T on the standard catalogue's c2c rows it changes (524288, 2²⁰, 2²¹,
  2²², 10⁶ out of place, 2²⁰ in place).
- **F (Xeon profile)**, 70abbae: `depth_float` on, so single-precision rows take the chain
  by their bytes (32768 to 2²² points). The standard catalogue has no single-precision row
  in range; judged against L on the x51 float rows 32768–2²⁰.
- Xeon judging runs: arms N = beed63c (`main`'s plans), T, L, F, FFTW, MKL on the standard
  catalogue's changed rows (25000 and those of L); N, T, F on the x51 rows (25000 in place
  and backward, the float rows); then the tracking tier N against F. Each arm changes rows
  the others do not, so T ÷ N, L ÷ N and F ÷ N on its own rows are each arm against the one
  before.

### Step 2, the judging runs

**Mac, M** (Homebrew Clang 23, `neon_fcma`, NEON FFTW 3.3.11; M = df67c89 against N =
1a78533): run 20261118-002000-b403 (30 rounds, all under load, band ±2.98 %): the 19 changed
cases **0.735 [0.724, 0.751]**, every one faster, from 0.688 (20480) to 0.858 (98304 in
place); out of place 0.688–0.812, in place 0.748–0.858, backward 0.693–0.794, batches of 8
0.732–0.749. FFTW ÷ N on the same cases 0.866 (0.741–0.961): FFTN ÷ FFTW goes from about
1.04–1.35 to 0.79–0.98 on them. Tracking tier N against M, run 20261118-002935-541e (24
rounds, all under load, band ±4.40 %): 61 no evidence, weighted geomean 1.010 [0.994, 1.028];
three medians past 1 + max(band, 3 %) on rows whose plans do not change (r2c 2048 × 36 1.067,
c2c 256² 1.050, r2c 64³ 1.047), re-run once with c2c 16 and 46080, run 20261118-004154-53ed
(30 rounds, band ±2.18 %): 0.988–0.999, all no evidence. **M passes the rule.**

## Verdict
