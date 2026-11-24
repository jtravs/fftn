# X72: mid-length real rows on Zen 3

- **Status:** running
- **Owner:** x72 session, 2026-11-24 (overnight)
- **Phase / task:** 1 (the client catalogue check's largest gap, PLAN "Status")

## Question

The client catalogue check (2026-11-24, `main` d4b169f, run 20261124-102950-017c) found every
real row of 6144–20480 points out of place on the EPYC 7763 at one thread at 1.05–1.14 of
FFTW's time (12 shapes, 40 % of the catalogue's weight), which holds that machine's one-thread
geometric mean at 1.00 against a target of 0.95. Where does the time go, and which change keyed
to Zen 3's class wins it back?

## Diagnosis

### Plans and per-stage cost (run D1)

Run 20261124-220705-a157 (Slurm 518033, node18, `main` 1b1d129 plus this branch's catalogue and
fbench options, f39e659; GCC 16, `avx2`, double, one thread; every round overloaded: load
about 900 on 128 CPUs, band ±1.00 %), catalogue `x72`: the client catalogue's 17 real shapes of
6144–20480 points (lengths and batches as there) and the half-length c2c rows, both
directions. Arms: FFTN's rules (N), FFTW 3.3.11 MEASURE with `PLAN_NOTE=1` (F), FFTN
`PLANNING=measure` (M). Times per row in µs, FFTN ÷ FFTW, and each library's plan:

| case | N µs/row | N ÷ F | FFTN's plan | FFTW's plan |
|---|---|---|---|---|
| r2c 6144 × 48, × 66 | 9.87 | 1.04–1.05 | chain 12 × 16 × 16, pairing pass | rdft2-ct-dit/6 (hc2c 6) → 3 → 64 → 16 |
| c2r 6144 × 48, × 66 | 9.62 | 1.06–1.07 | pairing pass, chain 12 × 16 × 16 | rdft2-ct-dif/2 → 3 → 64 → 16 |
| r2c 6400 | 10.00 | 1.15 | chain 2 × 10 × 10 × 16, pairing pass | rdft2-ct-dit/20 → 10 → 20 → 16 |
| c2r 6400 | 8.77 | 0.96 | chain fused (2 first) | rdft2-ct-dif/2 → 10 → 20 → 16 |
| r2c 9216 | 14.32 | 0.99 | 4 × 8 × 12 × 12, pairing pass | /2 → 12 → 32 → 12 |
| c2r 9216 | 13.23 | 0.97 | fused | /2 → 12 → 32 → 12 |
| r2c 10240, × 28 | 17.14–18.45 | 1.09–1.11 | 4 × 8 × 10 × 16, pairing pass | /16 → 8 → 20 → 32 |
| c2r 10240, × 4 | 16.6–16.7 | 1.05 | fused | /2 → 8 → 20 → 32 |
| r2c 18432 | 37.00 | 1.16 | 4 × 12 × 12 × 16, pairing pass | /6 → 3 → 6 → 32 → 16 |
| c2r 18432 | 35.55 | 1.10 | fused | /2 → 16 → 3 → 32 → 6 |
| r2c 20480 × 21 | 45.9 | 1.16 | 4 × 10 × 16 × 16, pairing pass | /32 → 16 → 64 → 10 |
| c2r 20480 × 3 | 41.6 | 1.13 | fused | /2 → 8 → 20 → 64 |
| c2c 3072 | 7.41 | 1.06–1.07 | 12 × 16 × 16 | 3 → 64 → 16 |
| c2c 3200 | 7.88 | 1.10–1.11 | 2 × 10 × 10 × 16 | 10 → 20 → 16 (bwd 20 → 8 → 20) |
| c2c 4608 | 11.22 | 0.99–1.00 | 4 × 8 × 12 × 12 | 12 → 32 → 12 |
| c2c 5120 | 13.65 | 1.07–1.10 | 4 × 8 × 10 × 16 | 8 → 20 → 32 (bwd 16 → 20 → 16) |
| c2c 9216 | 26.35 | 0.96–0.98 | depth-first 4 × 2304 | 6 → 6 → 8 → 32 |
| c2c 10240 | 30.29 | 0.99 | depth-first 4 × 2560 | 16 → 64 → 10 |

- **18432 and 20480 are the chain's cost.** Their c2c half lengths 9216 and 10240 are
  depth-first chains (X51) and tie FFTW, but a real row's half-length transform never takes the
  depth-first chain: it is the breadth-first chain, then the pairing. r2c 18432 takes 37.0 µs
  where the half length takes 26.4 as a c2c row; r2c 20480 takes 45.9 µs per row where 10240
  takes 30.3.
- **6144–12800 trail where their half lengths trail.** c2c 3072, 3200 and 5120 lose to FFTW by
  6–11 % (the client check's gap 5 is the same rows); 4608 ties, and real rows of 9216 tie.
  The real overhead (time of the real row over its half length's c2c, minus 1) is FFTW's or
  close to it: r2c 6144 0.33 (FFTW 0.36), r2c 10240 0.25 (FFTW 0.24), r2c 6400 0.27 (FFTW
  0.23). FFTW pairs inside a radix pass (hc2c codelets); FFTN's r2c rows of these lengths pair
  in a separate pass (their half spectra exceed L1, and their chains end in a split run), c2r
  in the first pass where a radix of 4 or less can lead.
- **FFTN's own measurement finds little** (arm M): r2c 20480 × 21 0.90 (8 × 8 × 10 × 16), c2r
  6144 0.92 (2 × 8 × 12 × 16 fused), c2r 10240 0.95; the rest 0.96–1.02.

### Screens (not deciding)

To be filled from the screens' runs.

## Hypothesis

## Change under test

## Method

## Results

## Verdict
