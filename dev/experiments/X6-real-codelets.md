# X6: real-input codelets against the half-length complex route

- **Status:** planned (needs generator work); the half-length route's baseline measured
- **Owner:** task 2.1 session, 2026-11-14
- **Phase / task:** 2.1

## Question

Do generated real-input codelets (the pruned variants of section 7.3: a radix-R butterfly
whose inputs are real, or whose outputs are Hermitian) make even and odd real transforms
faster than the half-length complex transform with the pairing of bins k and h − k fused
into one of its passes, so that real overhead (below) falls under the 20 % of PLAN task
2.1 where the half-length route cannot reach it?

## Hypothesis

The half-length route spends, besides the complex transform of h = n/2, one addition, one
subtraction and one complex multiplication per pair of bins plus a halving (passes/real.inl).
On the Mac that pairing costs about as much per element as a radix-16 pass of an in-L1
chain, so a chain of two or three passes carries 15–45 % on top: the target is out of reach
for short rows whatever the fusion (measured below). A real-input first pass (r2c) or
Hermitian-output last pass (c2r) does the pairing's work inside a butterfly that also does
useful transform work, as FFTW's r2cf/hc2hc codelets do; we expect it to remove most of the
overhead for n ≤ 4096 and to give odd n a route at half the full-length complex cost
(odd n currently runs the full-length complex chain on (x, 0)).

## Change under test

Arm A: the half-length route as built in task 2.1 (fusion rule of plan_core.cpp
`fusion_of`). Arm B: chains whose first pass (r2c) or last pass (c2r) is a real-input or
Hermitian-output codelet pass, n even and odd. B needs: the generator to emit the pruned
variants with verified operation counts; pass kernels for them (lanes over butterflies,
real loads or Hermitian stores); a route in `chain_builder`. No switch exists yet.

## Method

- Cases: `bench/catalogues/x6.toml`: the standard tracking tier's real rows beside the c2c
  rows of half their length, double and float (`--catalogue x6 --tier tracking`), and odd
  lengths beside their c2c rows (tier full).
- Machines: the Mac (neon_fcma), the EPYC (avx2), the Xeon (avx512, avx2).
- Metric: real overhead = time(r2c of n) ÷ time(c2c of n/2) − 1 per case and its
  geometric mean; for odd n, time(r2c of n) ÷ time(c2c of n).
- Decision rule, written before running: adopt B for the lengths where its paired ratio
  to A is below 0.95 with the interval clear of the A/A band, if no case gets slower by
  more than 2 %; otherwise keep A and record the gap.

## Results

Baseline (arm A) only. The fusion rule itself was set from informal timings on a loaded
Mac (load average 7–9, best of 21 repetitions, QoS user-interactive; not an fbench run),
double unless noted. Real overhead with the pairing fused into the edge pass against the
same chain with a separate pairing pass:

| case | fused | separate |
|---|---|---|
| c2r 4096, first radix 8 (8 × 16 × 16) | +15–19 % | +28–32 % |
| c2r 4096, first radix 16 (16 × 16 × 8) | +36–44 % | +14–19 % |
| r2c 4096, last radix 8 (16 × 16 × 8, equal radices in both arms) | +9–14 % | +22–28 % |
| r2c 4096, last radix 16 (8 × 16 × 16) | +28–39 % | +23–30 % |
| r2c 32768, last radix 8, h beyond L1 (16 × 8 × 16 × 8) | +51 % | +10–19 % |
| float r2c 256, last radix 8, s = 16 (C = 2) | +71–76 % | +43–45 % |

So the rule (DESIGN.md 9.1): fuse into a codelet pass of radix at most 8 (4 on avx2,
unmeasured), with at least 4 registers of copies or groups per mirror half, and for r2c only
while the half spectrum fits L1; otherwise a separate pass. The chain's radices are reordered
to bring a radix within the limit to the fused end (r2c: 16 × 16 × 8 instead of
8 × 16 × 16, which costs the complex chain about 3 %).

fbench run 20261114-135716-0600 (Mac M5 Max, neon_fcma, double, hot, median; loaded: load
average 8–11, A/A band ±5.3 %; A = FFTW NEON at MEASURE, B = main a3903ed, C = task 2.1
528ed77). Real overhead of C from the same run, time(r2c n) ÷ time(c2c n/2) − 1:

| n | r2c | c2r | C ÷ FFTW r2c | C ÷ FFTW c2r | B ÷ FFTW r2c | B ÷ FFTW c2r |
|---|---|---|---|---|---|---|
| 256 | +43 % | +28 % | 0.92 | 0.82 | 1.26 | 1.17 |
| 1000 | +39 % | +26 % | 0.90 | 0.83 | 1.06 | 1.04 |
| 4096 | +41 % | +15 % | 0.96 | 0.88 | 1.31 | 1.14 |
| 65536 | +5 % | +16 % | 1.19 | 1.40 | 2.11 | 2.18 |
| 1048576 | +8 % | +2 % | 1.37 | 1.29 | 1.73 | 1.52 |

Beyond L1 the target holds (the grid's half-length transform carries the long rows; their
remaining gap to FFTW is the c2c grid's own, 1.10–1.22 here). Within L1 it does not: r2c
39–43 %, c2r 15–28 %.

## Verdict

Planned. Real-input codelets need the generator's pruned variants first (section 7.3);
until then the half-length route stands, and task 2.1's acceptance (real overhead ≤ 20 %)
is not met on the Mac for rows that fit L1 (see PLAN task 2.1).
