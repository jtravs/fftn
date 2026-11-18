# X55: the depth-first chain on Granite Rapids (a 2 MiB private L2)

- **Status:** running
- **Owner:** x51 session, 2026-11-18
- **Phase / task:** 1 (phase 1 exit check, the tail rule on Granite Rapids); follows X46,
  X51 and X48

## Question

Should the depth-first chain's profile rule (DESIGN 8.5) reach Granite Rapids, whose private
L2 is exactly 2 MiB and so falls outside the Xeon 6240's branch (private L2 under 2 MiB,
little L3), and with what range and tops? X48 left 18 of 193 exit-check rows over 1.05 of
FFTW beyond noise there (11 against MKL), most of them the rows the chain closed on the Xeon
6240: 16384–262144, 46080, 65536 in every form, 100000, 2²².

## Hypothesis

X46's diagnosis (three lines and a first pass's twiddle table overfill L2) scales with L2:
with 2 MiB the breadth-first chain should start missing from about L2/4 (32768 complex
doubles), and a depth-first chain with one top of 16 (20, 10) should take those rows to
0.65–0.85 of their time, as on the Xeon 6240. 16384 (L2/8) may not gain (8192 did not on
the Xeon 6240, at L2/8).

## Change under test

- **Screen (no code change):** `main` 387add1's plans against forced splits
  (`STRATEGY=depth`, `DEPTH_TOPS` 16, 20, 32, 10), FFTW and MKL, on the standard catalogue's
  c2c rows of 16384 points and more.
- **G (Granite Rapids' profile, the large-L2 branch of `set_grid_sizes`):** depth-first
  fields set from the screen, written here before its judging run.

Bits: a changed plan changes its rows' rounding within the derived tolerances (tests).

## Method

- Machine: Google Cloud `c4-standard-16`, minimum platform Intel Granite Rapids, one thread
  per core (8 cores), Spot, us-central1-a, Ubuntu 24.04, GCC 14.2, `avx512`; FFTW 3.3.11
  built there with SSE2/AVX/AVX2/AVX-512/FMA (MEASURE), oneMKL from the `mkl-devel` wheel (as
  X48); fbench pinned to CPU 2, one thread, hot, median.
- Judging: G against N = 387add1 on the standard catalogue's rows G changes (`--tier full`,
  which holds 16384–262144, 46080, 100000 and 65536 in place, backward, × 3), then the
  tracking tier, with FFTW and MKL arms for the ratios.
- Decision rule (X46's), written before any run: G is adopted if over the cases whose plan
  it changes the weighted geomean G ÷ N is below 1 − band with its interval below 1, no such
  case is slower than 1 + max(band, 3 %), and no tracking-tier case is slower than 1 +
  max(band, 3 %) (a case just past the limit on a loaded run is re-run once).
- Budget: at most USD 2 (approved by the maintainer on 2026-11-18; campaign at USD 4.72 of
  20); ledger in [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md), section X55.

## Results

## Verdict
