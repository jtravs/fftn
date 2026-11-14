# X20: the fixed cost of a call, and tiny transforms

- **Status:** running
- **Owner:** overhead session, 2026-11-14
- **Phase / task:** 1 (exit criteria), driver (DESIGN.md 8.3); lanes for batches stay task 2.2 / X7

## Question

Where does the time of a serial execute of a tiny c2c transform (n = 2–64, batch 1 and
64) go, between the entry (`execute_core`), the program's interpretation (the driver) and
the kernels, and how much of FFTN's distance to FFTW at n ≤ 64 does removing the first two
close?

## Hypothesis

X18 found about 4 ns of fixed cost per call and 9–13 ns of driver work per transform on
x86. On the Mac (run 20261114-095234-2dcf) FFTN ÷ NEON FFTW is 2.64 at 8, 2.15 at 16, 1.29
at 64 and 3.04 at 8 × 4096 rows, where the kernels are not the whole story: an 8-point leaf
is a few nanoseconds of arithmetic. Expected: per transform, the driver (`run_records`'
recursion over variant records, a 144-byte `pass_args` copy per stage, chunk ends, the
indirect call) costs several nanoseconds; a driver path that prepares the stages once per
call and only moves the slot addresses per transform removes most of it, with the same
kernels on the same arguments (bits unchanged).

## Change under test

Arm A: main (3433fc1). Arm B: the driver change(s) on branch `task/overhead`, one commit
each, timed against A and against FFTW in the same run.

## Method

- Decomposition: `bench/x20/x20_overhead.cpp` times the same work entered at four levels
  (ns per transform, median of 11–15 samples in one process): `api` (`execute_core`),
  `driver` (`run_transforms` with the slots prepared once), `kernel` (each stage's kernel
  through its pointer with `pass_args` prepared once), `once` (the loop around one kernel
  call). api − driver is the entry's cost, driver − kernel the interpretation's, kernel the
  floor of a design that calls kernels one transform at a time.
- Cases: catalogue `x20` (c2c rows 2–64 at batch 1 and 64; 8, 16 and 64 at 4096 rows),
  double and single precision, forward, out of place, `--tier full`.
- Machines: Apple M5 Max (NEON FFTW 3.3.11, which has no single precision; Clang 23);
  Xeon Gold 6240 (AVX-512, FFTW 3.3.11 with AVX-512 codelets at MEASURE; GCC 14.4; Slurm
  `-c 8`, shared). fbench interleaved rounds with the A/A control.
- Metric: FFTN ÷ FFTW per case and B ÷ A paired (fbench), weighted geomean over x20.
- Decision rule, written before running: adopt a driver change if (1) every test passes
  and serial, team and executor results stay bitwise identical (the same kernels on the
  same arguments); (2) B ÷ A over x20 is below 0.95 on the Mac with its interval clear of
  the A/A band; (3) no x20 case and no case of the quick tier is slower than A by more than
  the band on the Mac or the Xeon. A change that needs a new kernel entry or program shape
  (rows looped inside a kernel) is a separate arm, judged by the same rule.

## Results

### Where the time goes (before)

Mac, main + harness, Clang 23, ns per transform (`x20_overhead --reps 11 --ms 10`, load
about 3):

| n | b | api | driver | kernel | once |
|---|---|---|---|---|---|
| 2 | 1 | 6.8 | 5.4 | 1.7 | 1.1 |
| 8 | 1 | 7.7 | 6.9 | 3.9 | 2.5 |
| 16 | 1 | 12.0 | 11.2 | 7.2 | 6.5 |
| 64 | 1 | 48.0 | 47.0 | 39.1 | (2 stages) |
| 2 | 64 | 4.9 | 4.9 | 1.2 | 1.6 |
| 8 | 64 | 7.1 | 7.1 | 4.0 | 2.3 |
| 16 | 64 | 10.8 | 10.9 | 9.2 | 6.2 |
| 64 | 64 | 47.0 | 47.0 | 41.0 | (2 stages) |

Single precision is the same within 1 ns up to 16. FFTN ÷ FFTW before (Mac run
20261114-225941-d101, 7 rounds, band ±3.05 %): 2: 3.09 / 17.5 (b1 / b64), 4: 3.02 / 9.15,
8: 2.66 / 3.50, 16: 2.22 / 1.93, 32: 1.94 / 1.86, 64: 1.06 / 1.33; 4096 rows: 8 3.20,
16 2.06, 64 1.39.

## Verdict

Pending.
