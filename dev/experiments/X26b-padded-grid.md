# X26b: long rows, a padded intermediate for the grid

- **Status:** done (padding adopted where the L2 is shared by a cluster: the Mac; off on
  x86; arm B and arm 2 rejected)
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

Arms as commits: A 0dc9391 (main), P 93e0aed, PB 95f6109, PP 5e74297 (P with the copies'
prefetch on `avx2`), P2 b75a393 (P padding only an intermediate already in the workspace
block: three phases or in place), C2 0358493 (arm 2 on A). P2 builds the same program as
P on three-phase and in-place plans and as A on two-phase plans out of place. C2 and C3
(8984899, arm 2's second form) are on unpushed `exp/x26b-dev-*` branches; on the pull
request they are c8d3624 and e181b2c, P2 is 31bd862.

### Counters: padding does not cut L1 refills on x86

User-mode counts per transform, median of three process rounds, arms interleaved in one
Slurm job (`-c 8` shared; EPYC job 509800, Xeon job 15843; raw output in
`fftn-x26b/pmu-*` on each machine). Ratios to A:

| case (plan) | EPYC P: cycles / demand L1 fills / fills from beyond L2 | Xeon `avx512` P: cycles / L1D replacements / L2 lines in | Xeon `avx2` P |
|---|---|---|---|
| c2c 65536 (EPYC 32·32·64 /32; Xeon 256·256 /4) | 1.00 / 1.05 / 0.91 | 1.02 / 1.07 / 1.21 | 1.01 / 1.06 / 1.18 |
| c2c 262144 (64³ /32) | 1.01 / 1.07 / 0.95 | – | – |
| c2c 2²⁰ (64·128·128 /16) | 1.01 / 1.02 / 0.86 | 1.01 / 1.00 / 1.07 | 0.97 / 1.03 / 1.01 |
| c2c 2²² (128·128·256 /8) | 1.01 / 1.04 / 0.96 | 1.00 / 1.04 / 1.00 | 0.99 / 1.05 / 0.99 |
| c2c 46080 (EPYC unpadded; Xeon 192·240 /4) | – | 1.09 / 1.04 / 1.76 | 1.08 / 1.10 / 1.81 |
| c2c 100000 (unpadded: 50 ∤ 4) | 0.97 / 1.02 / 0.83 | 1.01 / 1.00 / 1.00 | 1.00 / 1.02 / 1.00 |
| r2c 65536 (EPYC 128·256 /4, two phases) | 1.05 / 1.12 / 6.1 | – | – |
| r2c / c2r 2²⁰ (64·64·128 /16) | 0.98–1.01 / 1.04–1.05 / 0.89–0.90 | 1.02 / 1.01 / 0.98 | 1.04 / 1.05 / 0.98 |

The padding removes some fills from beyond L2 on the EPYC's three-phase rows (5–17 %) but
the demand L1 fills rise 2–7 % on both machines and the cycles do not move. Where it moves
a two-phase plan's intermediate out of the output into the workspace block (EPYC r2c
65536, the Xeon's 65536 and 46080) it costs: L2 lines in +18–81 %, cycles +2–9 %. Reading:
a block's rows concentrated in one or two sets evict only those sets' lines while they
stream through; spread evenly they turn over every set of L1, the scratch block included.
The set conflicts of phase 2's gather are not what limits the x86 copies.

PB (X26's loops on P): instructions −11 to −13 % (EPYC), −27 to −39 % (Xeon `avx512`),
cycles 0.91–1.15 (EPYC: 100000 0.91, 2²² 0.94, r2c and c2r 2²⁰ 1.10–1.15), fill-buffer-full
cycles up to 8 times A's on the Xeon. PP (prefetch on `avx2`): on the EPYC cycles 1.06–1.16
(fills from beyond L2 0.2–0.65 of A's), on the Xeon under `avx2` 0.82–0.85 at 2²⁰ and 2²²
and 0.94 for r2c 2²⁰: the prefetch question is not moot, and on Intel `avx2` it pays.

### Mac: padding pays

Run 20261115-225104-be37 (x26b catalogue, 40 rounds, 23 loaded, band ±1.00 %), the Mac
forced to pad (`GRID_PAD_PERIOD`, which macOS's profile leaves unset). Double target
cases against A: P with a 4 KiB period 0.981 [0.975, 0.987], with 16 KiB 0.983 [0.977,
0.991], PB 0.978 [0.972, 0.986]. Per case (P, 4 KiB): 262144 0.934, 2²⁰ 0.942, r2c and c2r
2²⁰ and 2²² 0.95–0.98, float 2²² 0.942; the two-phase rows that moved their intermediate
into the block 1.002–1.012 (16384 1.009, 65536 1.010), nothing beyond 1.012. PB against P:
double 1.000 [0.994, 1.005] (no case beyond ±1.5 %), float 0.88–0.98 (65536 and 100000
0.88–0.89). FFTW ÷ FFTN (A) 0.83 over the double rows.

Run 20261115-231909-526b (50 rounds, 33 loaded, band ±1.00 %) confirmed it and tried P2,
which keeps two-phase plans out of place unpadded: double target cases P2 0.976 [0.971,
0.981], P 0.980 [0.973, 0.989]; P2 per case 262144 0.938, 2²⁰ 0.953, 2²² 0.960, r2c and
c2r 2²⁰ and 2²² 0.953–0.964, float 2²² 0.945, the two-phase rows 0.994–1.003. The same run
timed C2 (arm 2's first form, on the Mac the same choice as A: prefetch on): 1.005 [0.994,
1.014] overall but float 100000 1.082 [1.022, 1.100], and 1.099 [1.074, 1.105] again in
run 20261115-234843-b9c4 (band ±1.26 %): a run-time prefetch distance changed the
narrow-row copy loop's code. Arm 2's second form (C3, 8984899: main's loops instantiated
with and without the prefetch, one chosen per call) is neutral there: run
20261115-235700-0880 (40 rounds, band ±1.10 %), 1.002 [1.000, 1.008], every case within
0.990–1.013 (float 100000 1.001).

Collateral on the Mac, the standard tracking tier, A against the combined branch (P2 with
`GRID_PAD_PERIOD=4096`, and C3), run 20261116-001526-a5f8 (30 rounds, 25 loaded, band
±1.50 %): 1.000 [0.995, 1.007], 2 faster (262144 0.950, r2c 2²⁰ 0.963), none slower than
1.02.

### x86 timing on main's base (0dc9391)

EPYC 7763, GCC 16, run 20261116-010519-d13f (35 rounds, all on a node with 64 other busy
CPUs, band ±1.00 %), double target cases against A: P 0.993 [0.989, 0.997], PB 1.010
[1.005, 1.014] (against P 1.016), PP 1.065 [1.060, 1.069] (against P 1.072), C2 1.012
[1.009, 1.016]; FFTW 0.789 of A's time. P per case: 262144 0.962, in-place 65536 0.966,
r2c 2²⁰ 0.966, float 2²⁰ 0.963, but r2c 65536 1.035 and float 65536 1.035 (two-phase plans
whose intermediate moved into the block: P2 leaves them unpadded); the rest 0.989–1.004.
PB: 2²² 0.926, 100000 0.956, but 2²⁰ 1.085, r2c and c2r 2²⁰ 1.14, float 2²² 1.20. PP:
100000 1.22, float 100000 1.29. C2 (the run-time prefetch distance, prefetch off as in A):
2²² 1.05, 100000 1.05, float 100000 1.08 — the code-form cost seen on the Mac.

Xeon Gold 6240 `avx512`, GCC 14, run 20261116-010954-b495 (35 rounds, 8 loaded, band
±1.91 %): P 1.006 [0.994, 1.019] (46080 1.090, its intermediate moved into the block; the
rest within ±3 %), PB 0.998 [0.979, 1.016] (against P 0.992 [0.981, 1.004]; 46080 1.10,
100000 0.957, in-place 65536 0.951); FFTW 0.932 of A's time.

Xeon under `ISA=avx2`, run 20261116-021041-45ed (33 rounds, 15 loaded, band ±1.64 %): P
1.007 [0.997, 1.016], PB 0.999 [0.992, 1.005] (against P 0.991 [0.978, 0.999]), PP 0.951
[0.946, 0.957], C2 0.961 [0.956, 0.968]. With the prefetch (PP and C2 alike) 2²⁰ 0.86–0.88,
2²² 0.83–0.84, r2c and c2r 2²² 0.85–0.87, r2c and c2r 2²⁰ 0.93–0.96, float 2²² 0.91, but
float 100000 1.27 (both forms: main's loop with the prefetch on, and C2's), 46080 1.03–1.04,
100000 1.03, 65536 1.02. Float 100000 copies rows of four complex floats, 32 bytes, half a
line: one prefetch per row then costs more than it brings.

Final, the Mac: run 20261116-031759-7115 (40 rounds, 27 loaded, band ±1.12 %), main after
X28 (be593d1) against the pull request's head (e3d6109, the Mac's default profile, which now
pads): double target cases 0.974 [0.967, 0.978]; 262144 0.953, 2²⁰ 0.953, 2²² 0.944, r2c and
c2r 2²⁰ and 2²² 0.945–0.971, float 2²² 0.942; nothing above 1.002.

The final confirmation runs against main after X28 (be593d1) on the x86 machines were
cancelled (Slurm 510126, 15891) once arm 2 was rejected: the adopted change builds main's
plans and kernels on x86 (no padding there; the copies as in main).

## Verdict

By the rule written before the runs (double target cases, the run's A/A band):

- **P / P2, the padded intermediate: adopted on the Mac, rejected on x86.** Mac
  (`neon_fcma`, Clang): P2 0.976 [0.971, 0.981] against a band of ±1.00 % (the final
  branch against main after X28: 0.974 [0.967, 0.978], band ±1.12 %), nothing slower
  than 1.012 in the target set or the tracking tier (1.000 overall, run
  20261116-001526-a5f8). EPYC (`avx2`, GCC 16): P 0.993 [0.989, 0.997], not clear of 0.990.
  Xeon (GCC 14): `avx512` 1.006 [0.994, 1.019], `avx2` 1.007 [0.997, 1.016]. P2 (31bd862)
  pads only an intermediate already in the workspace block (moving one there cost 1–9 %).
  The profile's `grid_pad_period` is 4 KiB where the L2 is shared by a cluster and none
  where each core has its own (e4518ab), the distinction X12b keys the grid on. Counters
  say why x86 gains nothing: padding does not cut its L1 refills (they rise 2–7 %).
  Results are bit-identical.
- **PB, X26's cheaper copy loops on the padded grid: rejected everywhere** (against P: Mac
  1.000, EPYC 1.016, Xeon `avx512` 0.992 and `avx2` 0.991, none clear of its band). The
  Mac's float rows gained 0.88–0.95 with them: a float-specific follow-up, not judged here.
- **PP, the prefetch on `avx2` with the padding: rejected** (EPYC 1.072 against P): the
  padding did not make X26's prefetch question moot.
- **Arm 2, the prefetch following the core: rejected and reverted** (e3d6109). It wins
  the Xeon under `avx2` on the geomean (C2 0.961) but float 100000 is 1.27, past the
  collateral limit, and the same with main's own loop (PP), so the loss belongs to the
  prefetch, not to the code form. X26's per-target gate stays. Its first form's run-time
  prefetch distance also cost float 100000 8–10 % on the Mac and the EPYC; the second form
  (compile-time distance, C3) fixed that on the Mac (1.002) but was not timed on x86.
- Open: a prefetch rule by core and row width (no prefetch for rows narrower than a line,
  on Zen not at all) would keep the Xeon-under-`avx2` gains at 2²⁰–2²² (0.83–0.88); the
  float copy loops on the Mac (PB).

Slurm jobs: EPYC builds 509797, 509805, 510125 and counters 509799, 509800; timing 509806
(509798 replaced), 510126 and 509874 cancelled. Xeon builds 15835, 15837, 15844, 15859,
15890; counters 15842, 15843; timing 15836, 15845 (15838 replaced), 15860 and 15891
cancelled.
