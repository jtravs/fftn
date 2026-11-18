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

Host: Google Cloud `c4-standard-16` Spot, one thread per core (8 cores: Intel Xeon 6985P-C,
Granite Rapids; L1d 48 KiB, private L2 2 MiB, L3 504 MiB), Ubuntu 24.04, GCC 14.2 Release,
`avx512`; FFTW 3.3.11 (SSE2/AVX/AVX2/AVX-512/FMA), oneMKL 2026.0.2 (wheel). Every arm
pinned to CPU 2; no round under load in the judging runs.

**Screen** (run 20261118-084057-ff56, band ±2.02 %; forced tops against N's plans on the
standard rows of 16384 points and more): one top of 16 took 0.44–0.83 of N's time from
16384 to 2^20 (46080 0.444, 65536 0.520, 131072–262144 0.615–0.628, 524288 0.739, 2^20
0.811), tied at 2^21 (1.032) and took 0.665 at 2^22; it lost at 20000 (1.566). Tops of 32
were close to 16 (2^21 0.783); a top of 20 won at 100000 (0.606) where 16 tied (0.993).

**Arm G** (3c5b130): the Xeon 6240's depth-first fields on a private L2 of 2 MiB or more
(x86-64): from L2/4 (32768 complex doubles) to 64 L2, tops 16, 20, 10, double only.

| Run | Cases | G ÷ N | Band |
|---|---|---|---|
| 20261118-084656-17bd | 14 of 18 screen rows whose plan G changes | 0.688 [0.677, 0.701], 14 faster, none slower | ±1.78 % |
| 20261118-091854-4eef | standard tracking tier (61) | 0.940 [0.931, 0.950], 7 faster, 54 no evidence, none slower | ±1.92 % |

FFTN ÷ FFTW and ÷ oneMKL, N → G (same run, 20261118-084656-17bd):

| Case | ÷ FFTW | ÷ MKL |
|---|---|---|
| 32768 | 1.15 → 0.77 | 1.21 → 0.81 |
| 46080 | 1.65 → 0.72 | 1.46 → 0.64 |
| 65536 (out of place, in place, backward, × 3) | 1.27–1.54 → 0.70–0.81 | 1.30–1.66 → 0.67–0.82 |
| 100000 | 1.34 → 0.83 | 0.97 → 0.60 |
| 131072, 262144 | 1.26–1.27 → 0.80–0.81 | 0.77–0.86 → 0.49–0.54 |
| 524288, 2^20 | 0.65–0.68 → 0.48–0.60 | 0.45–0.69 → 0.41–0.54 |
| 2^21 | 0.70 → 0.72 (unchanged plan class, tie) | 0.66 → 0.67 |
| 2^22 | 1.07 → 0.74 | 0.92 → 0.63 |

16384, 20000 and 25000 lie below L2/4 and keep their plans (0.99–1.00).

## Verdict

**G adopted.** Its changed cases' geomean is 0.688 with the interval below 1 − band, no
changed case is slower, and no tracking-tier case is slower: X46's rule is met. Every long
row Granite Rapids failed in X48's exit check is now below 0.85 of FFTW and MKL's time,
except 999983 against MKL (X50's subject). X48's other Granite Rapids cases (160, 240,
1024) are untouched. The private-L2 bound is not "< 2 MiB" but a branch of its own: the
large-L2 profile (X35) now carries the same depth-first fields. Not measured: float on
Granite Rapids (stays off), Sapphire and Emerald Rapids (they share the branch by their
2 MiB L2), Clang. Cloud cost USD 0.40 ([ledger](cloud-ledger-2026-11.md), X55).
