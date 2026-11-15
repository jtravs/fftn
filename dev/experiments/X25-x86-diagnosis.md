# X25: why FFTN trails FFTW in one-dimensional double on x86

- **Status:** done (diagnosis; no library change)
- **Owner:** x25 session, 2026-11-15
- **Phase / task:** 1 (exit criteria); PLAN leads 1–5; follows X18, X10, X19, X1, X12b

## Question

Why does FFTN still trail FFTW in one-dimensional double precision on x86 (tracking tier,
FFTN ÷ FFTW: EPYC 1.21, Xeon 1.06, against the Mac's 0.97), and is each case group's gap
in the kernels' instruction stream, the memory hierarchy, or the fixed cost per call? The
answer orders and sharpens PLAN's leads 1–5 before any of them gets an experiment of its
own. A diagnosis: no library change.

## Hypothesis

From the baselines (runs 20261115-101319-f98f EPYC, 20261115-114105-ea43 Xeon) and X18,
X10, X19, X1:

- **128–4096 points** (EPYC 1.22, Xeon 1.07; Mac 0.91, so x86-specific): the instruction
  stream. On the Xeon port 5 (shuffles) is the bottleneck (X10: 73 % busy against FFTW's
  55 %, 58 % after X1's split runs); on Zen 3, with two shuffle pipes and four-wide
  dispatch, the expectation is instruction count and front-end throughput rather than one
  pipe. These rows fit in L1 or L2, so cache misses should explain nothing.
- **≤ 64 points** (EPYC 1.38, Xeon 1.51): fixed cost per call (FFTN's entry and driver
  against FFTW's plan-call path), plus leaves that use one lane per register (X18 item 3).
- **≥ 16384, 46080, 100000, r2c 65536** (EPYC 1.20–1.98, Xeon 1.03–2.06): memory
  hierarchy (more passes over L2 or memory than FFTW's plan), and for 46080 and 100000 the
  grid's phases with factors FFTN handles less well than FFTW's plan.

## Change under test

None: a diagnosis. Two small tools are added to fbench on the way, neither of which
changes what fbench measures by default:

- the FFTW backend's option `PLAN_NOTE=1` writes FFTW's own plan (`fftw_sprint_plan`, the
  public API) into the case's note, next to FFTN's radices;
- the harness option `--pmu EVENTS` (Linux) counts hardware events in user mode for the
  executes of one case (`perf_event_open`), after its timed batches, and reports them per
  call. On DMOG, which has no `perf` but allows user-mode counting
  (`perf_event_paranoid` = 2), this is how the EPYC's counters are read.

FFTW's source code is not read (whether to read GPL source is a maintainer decision,
postponed): FFTW's plans come from its public API, its time and counters from running it.

## Method

- Cases (double, one thread, hot, out of place, forward): c2c rows 8, 16, 64, 60, 120,
  256, 1024, 4096, 65536, 100000, 46080; r2c 65536.
- Machines: Xeon Gold 6240 (Cascade Lake, `avx512`; GCC 14; FFTW 3.3.11 with AVX-512 and
  MKL 2026.1), EPYC 7763 (Zen 3, `avx2`; DMOG; GCC 16; FFTW 3.3.11 with AVX2). Slurm
  `-c 8`, shared, never exclusive. FFTW at MEASURE as fbench runs it.
- Per case, FFTN and FFTW (and MKL on the Xeon):
  1. The plans: FFTN's `describe()` radices (fbench note) and FFTW's `fftw_sprint_plan`.
  2. Counters per transform: cycles, instructions, IPC; uops by port (Xeon: ports 0, 1, 5
     and the load and store ports) or by FP pipe (Zen 3: pipes 0–3); L1D refills and L2
     misses; front-end stalls (Xeon: uops not delivered; Zen 3: dispatch stalls).
  3. Static analysis where counters cannot attribute: uiCA (Cascade Lake) and llvm-mca
     (`znver3`) on FFTN's hot loops.
- Attribution per group: instructions per transform against IPC (what share of the cycle
  gap each explains); the busiest port or pipe and its occupancy against FFTW's; loads and
  stores per point; L1 and L2 misses per point; for ≤ 64 points the cycles outside the
  kernels (counters of a batch of 1 against the per-row counts at batch 64).

What each outcome would mean:

- **Kernel instruction stream** (more instructions at equal IPC, or a saturated port or
  pipe that FFTW's code does not saturate): codelet and pass changes (lead 3; X1's split
  runs extended, fewer shuffles, larger radices).
- **Memory hierarchy** (L1 or L2 misses per point well above FFTW's, IPC lower at similar
  instruction counts): plan and blocking changes (lead 1; X12 on x86, fewer passes over L2).
- **Fixed per-call cost** (cycles at batch 1 well above batch 64 per row, instructions
  outside the kernels): driver work (lead 4).

## Results

Commit a0e84a4 (main 5c40cc6 plus the two fbench tools). Xeon: GCC 14.4, `avx512`, FFTW
3.3.11 (SSE2 to AVX-512 codelets), oneMKL 2026.1, Slurm `-c 8` on the shared node (load
average 12–21, the X23 jobs running alongside). EPYC: GCC 16.2, `avx2`, FFTW 3.3.11 (SSE2,
AVX2), DMOG node16, `-c 8 --hint=nomultithread`. Counters are user mode, per execute, the
median of three process rounds (two for the TLB round); every group stayed scheduled the
whole time (running fraction 1.00). Each machine's FFTW planned once at MEASURE and every
later process imported that wisdom, so all counter groups of a round saw one FFTW plan.
The raw output (JSON lines per round, group and library) stays in the campaign
directories (`fftn-x25/out-*`, `tlb-*`, `perf-*`, `perf2-*` on each machine), not in the
tree.

Time ratios come from the baseline fbench runs; the counter runs' own timings are single
processes on a loaded machine and are not used for ratios. FFTN ÷ FFTW per case, Xeon /
EPYC (20261115-114105-ea43 / 20261115-101319-f98f): 8 1.83 / 1.41, 16 1.46 / 1.41, 64
1.32 / 1.31, 60 1.60 / 1.43, 120 1.62 / 1.26, 256 1.01 / 1.35, 1024 1.04 / 1.20, 4096
0.98 / 1.05, 65536 1.06 / 1.44, 46080 1.40 / 1.66, 100000 2.07 / 2.00, r2c 65536 1.36 /
1.29. FFTN ÷ MKL on the Xeon: 8 0.81, 16 1.50, 64 1.63, 60 1.97, 120 1.33, 256 1.13, 1024
0.99, 4096 1.09, 65536 1.14, 46080 1.04, 100000 1.31, r2c 1.26. The counters' cycle
ratios agree with these within 0.1 on the EPYC and in direction on the Xeon.

**FFTW's MEASURE plans are not stable from process to process** on the loaded Xeon. In
three processes it chose 15 × 4, 10 × 6 and 15 × 4 at 60 points (806 against 1211
instructions per call), and three different plans at 65536 (1.28–2.08 M instructions).
Counters of FFTW must come from one shared plan, as they do here, and time comparisons need
fbench's shared wisdom per run.

### The plans

| case | FFTN Xeon (`avx512`) | FFTW Xeon | FFTN EPYC (`avx2`) | FFTW EPYC |
|---|---|---|---|---|
| 8 | leaf 8 | `n2fv_8_sse2` | leaf 8 | `n2fv_8_sse2` |
| 16 | leaf 16 | `n2fv_16_avx2_128` | leaf 16 | `n2fv_16_sse2` |
| 64 | 4 × 16 | 8 × 8 (`t2fv_8` zmm, `n2fv_8` ymm) | 4 × 16 | 2 × 32 (`t2fv_2`, `n2fv_32`) |
| 60 | 4 × 15 | 15 × 4 (`t1fv_15` zmm, `n2fv_4` xmm) | 6 × 10 | 10 × 6 (`t2fv_10`, `n2fv_6`) |
| 120 | 4 × 3 × 10 | 10 × 12 (`t2fv_10` zmm, `n2fv_12` ymm) | 10 × 12 | 10 × 12 (`t2fv_10`, `n2fv_12`) |
| 256 | 16 × 16 | 32 × 8 (`t1fv_32` zmm, `n2fv_8` ymm) | 2 × 16 × 8 | 8 × 32 (`t2fv_8`, `n2fv_32`) |
| 1024 | 8 × 8 × 16, split from pass 2 | 32 × 32 (`t1fv_32` zmm, `n2fv_32` ymm) | 4 × 16 × 16 | 32 × 32 (`t2fv_32`, `n2fv_32`) |
| 4096 | 16 × 16 × 16, split from pass 2 | 64 × 64 (`t2fv_64` zmm, `n2fv_64` ymm) | 8 × 8 × 8 × 8, split from pass 2 | 8 × 64 × 8 (`t1fv_8`, `t1fv_64`, `n1fv_8`) |
| 65536 | grid 256 × 256, blocks of 4 columns | 64 × (32 × 32, buffered), tiled in-place transposes | grid 32 × 32 × 64, blocks of 32 | 8 × 8 × 8 (DIF `q1fv_8`) × 128, tiled transpose |
| 46080 | grid 192 × 240, blocks of 4 | 9 × 32 × 20 × 8 | grid 32 × 36 × 40, blocks of 32 | 16 × 15 × 32 × 6 |
| 100000 | grid 40 × 50 × 50, blocks of 4 | 20 × 25 × 10 × 20 | grid 40 × 50 × 50, blocks of 4 | 10 × 10 × 5 × 20 × 10 |
| r2c 65536 | half-length chain 32768 (16 × 8 × 16 × 16) + pairing | 2-point pairing, then 8 × 8 × 16 × 32 | half-length grid 128 × 256, blocks of 4, + pairing | 4-point pairing, then 2 × 8 × 64 × 32 |

FFTW runs single rows of 8 and 16 points with 128-bit codelets on both machines. FFTN
broadcasts the row across a 256- or 512-bit register (X18 item 3). FFTW's plans for the
long rows have three to five passes of radix 5–64, with buffered or tiled in-place copies.
FFTN's grids have two or three phases of two passes each, with a separate gather and
scatter for each phase whose radices exceed 8 (X12).

### Counters, Xeon (per transform; N = FFTN, F = FFTW, M = MKL)

Port shares are uops dispatched to the port ÷ cycles. With 512-bit instructions active,
port 1's vector unit is fused into port 0, so FP work and shuffles go to ports 0 and 5,
and ports 1 and 6 carry the scalar work.

| case | cycles N / F / M | instructions N / F / M | IPC N / F / M | port 5 busy N / F | port 0 busy N / F | L1D repl. N / F | L2 miss N / F | stall cycles N / F |
|---|---|---|---|---|---|---|---|---|
| c2c-8 | 65 / 40 / 117 | 200 / 139 / 238 | 3.08 / 3.47 / 2.03 | 46 % / 39 % | 46 % / 46 % | 0 / 0 | 0 / 0 | 4 / 0 |
| c2c-16 | 110 / 78 / 121 | 321 / 218 / 254 | 2.91 / 2.79 / 2.10 | 55 % / 27 % | 56 % / 54 % | 0 / 0 | 0 / 0 | 6 / 2 |
| c2c-64 | 277 / 231 / 186 | 852 / 622 / 467 | 3.07 / 2.69 / 2.51 | 54 % / 44 % | 53 % / 52 % | 0 / 0 | 0 / 0 | 4 / 2 |
| c2c-60 | 405 / 258 / 206 | 1012 / 806 / 514 | 2.50 / 3.12 / 2.50 | 39 % / 51 % | 37 % / 57 % | 0 / 0 | 0 / 0 | 30 / 5 |
| c2c-120 | 702 / 437 / 504 | 2038 / 1146 / 1313 | 2.90 / 2.62 / 2.61 | 49 % / 45 % | 48 % / 60 % | 0 / 0 | 0 / 0 | 32 / 13 |
| c2c-256 | 981 / 931 / 856 | 2654 / 2552 / 2183 | 2.70 / 2.74 / 2.55 | 73 % / 55 % | 62 % / 61 % | 0 / 0 | 0 / 0 | 9 / 18 |
| c2c-1024 | 5044 / 4899 / 5234 | 10.4k / 12.0k / 10.1k | 2.06 / 2.45 / 1.94 | 58 % / 44 % | 54 % / 62 % | 1497 / 760 | 0 / 0 | 503 / 157 |
| c2c-4096 | 25k / 29k / 25k | 43k / 50k / 46k | 1.68 / 1.74 / 1.89 | 57 % / 34 % | 48 % / 52 % | 8520 / 7402 | 0 / 1 | 5009 / 4420 |
| c2c-65536 | 1.27M / 1.14M / 1.07M | 3.24M / 2.08M / 0.95M | 2.56 / 1.83 / 0.89 | 48 % / 43 % | 47 % / 35 % | 0.21M / 0.12M | 55k / 75k | 0.29M / 0.36M |
| c2c-46080 | 0.77M / 0.68M / 0.73M | 2.31M / 1.10M / 1.27M | 3.00 / 1.62 / 1.74 | 54 % / 42 % | 54 % / 39 % | 0.13M / 0.11M | 25k / 43k | 95k / 0.18M |
| c2c-100000 | 2.81M / 1.33M / 2.11M | 8.24M / 2.35M / 3.28M | 2.93 / 1.77 / 1.56 | 43 % / 51 % | 44 % / 48 % | 0.16M / 0.20M | 0.13M / 85k | 0.58M / 0.29M |
| r2c-65536 | 0.61M / 0.55M / 0.51M | 0.52M / 1.18M / 0.54M | 0.84 / 2.16 / 1.07 | 31 % / 51 % | 28 % / 45 % | 0.10M / 80k | 71k / 21k | 0.35M / 96k |
| c2c-8 × 64 | 1227 / 1068 / 1527 | 3796 / 3307 / 4676 | 3.09 / 3.10 / 3.06 | 82 % / 58 % | 82 % / 54 % | 0 / 0 | 0 / 0 | 13 / 5 |
| c2c-16 × 64 | 3245 / 2631 / 1686 | 9587 / 7626 / 3341 | 2.95 / 2.90 / 1.98 | 87 % / 54 % | 87 % / 56 % | 169 / 180 | 0 / 0 | 36 / 66 |
| c2c-60 × 64 | 24k / 17k / 11k | 53k / 51k / 20k | 2.20 / 2.93 / 1.87 | 38 % / 51 % | 35 % / 55 % | 1914 / 1915 | 0 / 0 | 3583 / 694 |

| case | ports 1 + 6 (scalar) N / F / M | port 4 (store data) N / F | uops not delivered (front end) N / F | 512 / 256 / 128-bit FP ops N ; F |
|---|---|---|---|---|
| c2c-8 | 55 / 45 / 91 | 21 / 13 | 32 / 3 | 34 / 0 / 0 ; 0 / 0 / 28 |
| c2c-16 | 64 / 66 / 101 | 41 / 26 | 69 / 0 | 98 / 0 / 0 ; 0 / 0 / 88 |
| c2c-64 | 206 / 152 / 74 | 129 / 101 | 159 / 2 | 174 / 0 / 0 ; 114 / 120 / 0 |
| c2c-60 | 241 / 178 / 77 | 186 / 110 | 127 / 60 | 196 / 0 / 0 ; 162 / 0 / 120 |
| c2c-120 | 569 / 237 / 226 | 253 / 178 | 119 / 4 | 498 / 0 / 0 ; 273 / 290 / 0 |
| c2c-1024 | 1698 / 1851 / 1008 | 1174 / 2230 | 781 / 324 | 4768 / 0 / 0 ; 2888 / 3808 / 0 |
| c2c-65536 | 0.76M / 0.38M / 84k | 0.26M / 0.42M | 82k / 0.16M | 0.59M / 0 / 0 ; 0.40M / 0.24M / 0 |
| c2c-46080 | 0.54M / 0.20M / 0.22M | 0.18M / 0.13M | 63k / 23k | 0.43M / 0 / 0 ; 0.28M / 0.34M / 0 |
| c2c-100000 | 2.21M / 0.27M / 0.48M | 0.72M / 0.31M | 0.30M / 66k | 1.21M / 0 / 0 ; 1.06M / 0.33M / 0 |
| r2c-65536 | 66k / 0.25M / 56k | 49k / 0.13M | 12k / 10k | 0.27M / 0 / 0 ; 0.23M / 0.12M / 0.15M |

TLB and aliasing (a second counter round, whose FFTW plans differ, see above). FFTN's
second-level TLB hits are at or below FFTW's in every case (65536: 54k for loads and 2.4k
for stores, against 91k and 18k). Page walks are 0–76 per transform. 4K-aliased loads are
1.1–2.4 times FFTW's (65536: 65k against 60k). Store-forwarding blocks: 4 per call at 60
points (FFTW 0) and 146 at 100000 (FFTW 5).

AVX frequency licences: FFTN's 8-point leaf spends its cycles at licence 1 (512-bit
light), FFTW's SSE2 codelet at licence 0. So FFTN takes 1.62 times the cycles and also
runs at a lower clock (time ratio 1.84 in fbench). From 16 points up, FFTN and FFTW both
spend their cycles at licence 2.

Where FFTN's time goes (`perf record` on cycles, without the harness's own check;
instruction shares from `-e instructions:u` in brackets):

- 65536: `block_copy` 42 % (50 %), `twiddle_transpose` 22 % (14 %), the two radix-16 lane
  passes 33 % (34 %).
- 46080: `block_copy` 40 % (52 %), `twiddle_transpose` 19 % (13 %), lane passes 37 %.
- 100000: `block_copy` 40 % (40 %), `twiddle_transpose` 14 % (9 %), `twiddle_scatter`
  12 % (13 %), lane passes of 5, 10 and 8 at least 23 % (33 %), `run_records` 5 % (5 %).

FFTW's copies (`fftw_cpy2d_pair`, `fftw_cpy2d`) are 11–13 % of its time at the same
lengths. With blocks of four columns, every row a copy moves is one 64-byte vector.
`block_copy` then spends about 33 instructions per row (from its instruction share and the
rows moved: the row loop, two prefetches and a one-trip inner loop around one load and
one store). The hottest instruction is the store (25 % of the function's samples). The
radix-5 first-pass lane function at 100000 lists 119 vector moves against 76 arithmetic
instructions and 14 shuffles: spills on top of the work.

At 60 points the radix-4 grouped pass (15 groups, not a multiple of the 4 lanes) is 57 % of
the time, the radix-15 lane pass 26 %, the driver (`run_transforms`, `moved_in_nest`,
`execute_core`) 13 %. At 8 points the leaf is 68 %, FFTN's driver (`run_leaf_rows`,
`execute_core`, `transform_count`) 16 % and the backend's call 9 %. FFTW's codelet is 65 %
and `apply` with `fftw_execute_dft` 20 %.

### Counters, EPYC (per transform; N = FFTN, F = FFTW)

Zen 3 has four FP pipes: 0 and 1 multiply and FMA, 2 and 3 add, and shuffles run on 1
and 2. The op-cache events tried (PMCx0AA) counted zero and are dropped.

| case | cycles N / F | instructions N / F | IPC N / F | FP-pipe uops N / F | busiest FP pipe N / F | loads N / F | stores N / F | L1D fills N / F | L2 miss (data) N / F | L1 DTLB miss N / F |
|---|---|---|---|---|---|---|---|---|---|---|
| c2c-8 | 44 / 29 | 195 / 122 | 4.41 / 4.24 | 31 / 32 | 32 % / 45 % | 43 / 40 | 21 / 13 | 0 / 0 | 0 / 0 | 0 / 0 |
| c2c-16 | 70 / 50 | 315 / 216 | 4.51 / 4.31 | 87 / 90 | 47 % / 62 % | 80 / 69 | 43 / 26 | 0 / 0 | 0 / 0 | 0 / 0 |
| c2c-64 | 278 / 211 | 1151 / 839 | 4.13 / 3.97 | 351 / 344 | 42 % / 50 % | 289 / 262 | 163 / 140 | 0 / 0 | 0 / 0 | 0 / 0 |
| c2c-60 | 300 / 221 | 1266 / 821 | 4.21 / 3.72 | 372 / 393 | 36 % / 50 % | 331 / 274 | 187 / 114 | 0 / 0 | 0 / 0 | 0 / 0 |
| c2c-120 | 512 / 413 | 2120 / 1540 | 4.14 / 3.73 | 820 / 846 | 46 % / 61 % | 601 / 503 | 311 / 220 | 0 / 0 | 0 / 0 | 0 / 0 |
| c2c-256 | 1142 / 874 | 5306 / 3319 | 4.65 / 3.80 | 2019 / 1905 | 56 % / 68 % | 1337 / 1122 | 562 / 506 | 0 / 0 | 0 / 0 | 0 / 0 |
| c2c-1024 | 5752 / 4769 | 21.2k / 15.3k | 3.69 / 3.21 | 9830 / 9435 | 57 % / 61 % | 5538 / 5833 | 2110 / 2340 | 1828 / 1200 | 0 / 0 | 0 / 0 |
| c2c-4096 | 36k / 32k | 98k / 96k | 2.73 / 2.99 | 56k / 54k | 48 % / 50 % | 26k / 34k | 12k / 11k | 13k / 9438 | 22 / 0 | 77 / 16 |
| c2c-65536 | 1.34M / 0.93M | 5.40M / 2.78M | 4.04 / 2.98 | 1.36M / 1.23M | 36 % / 39 % | 1.13M / 0.95M | 0.49M / 0.38M | 0.37M / 0.18M | 22k / 4.8k | 7.0k / 1.7k |
| c2c-46080 | 0.92M / 0.54M | 3.86M / 1.46M | 4.19 / 2.69 | 1.00M / 0.85M | 37 % / 46 % | 0.82M / 0.57M | 0.35M / 0.17M | 0.24M / 0.12M | 16k / 34k | 2.5k / 54k |
| c2c-100000 | 2.58M / 1.26M | 12.31M / 3.65M | 4.77 / 2.89 | 2.46M / 2.13M | 30 % / 51 % | 2.29M / 1.41M | 1.01M / 0.39M | 0.18M / 0.32M | 35k / 87k | 73k / 150k |
| r2c-65536 | 0.64M / 0.48M | 2.67M / 1.36M | 4.15 / 2.85 | 0.73M / 0.65M | 39 % / 42 % | 0.51M / 0.40M | 0.21M / 0.20M | 0.12M / 0.12M | 2.8k / 28k | 19k / 4.2k |
| c2c-8 × 64 | 850 / 781 | 3723 / 3305 | 4.38 / 4.23 | 1985 / 1536 | 95 % / 54 % | 610 / 988 | 525 / 520 | 0 / 0 | 0 / 0 | 0 / 0 |
| c2c-16 × 64 | 2708 / 1919 | 9891 / 7625 | 3.65 / 3.97 | 5581 / 3915 | 80 % / 58 % | 2997 / 2428 | 1382 / 1192 | 156 / 154 | 0 / 0 | 0 / 0 |
| c2c-60 × 64 | 17k / 15k | 69k / 52k | 4.13 / 3.52 | 24k / 25k | 41 % / 49 % | 17k / 17k | 9107 / 7337 | 1920 / 1916 | 0 / 0 | 0 / 0 |

Loads that cross a cache line: FFTN 755 per 1024-point transform, 44 at 60, 27k at 65536
and 50k at 100000; FFTW 0–36 everywhere.

DMOG has no `perf`, so there is no per-function profile for the EPYC. As a proxy, FFTN
with `ISA=avx2` on the Xeon (`perf record`) spends 31–36 % of its long-row cycles in
`block_copy` and 15–22 % in `twiddle_transpose`. That covers 65536 and 46080 with the
Xeon's grid shapes (blocks of 4) and 100000 with the same grid as the EPYC. At 60 points
(6 × 10, the EPYC's plan) the driver functions take 15 % of the cycles.

### Fixed cost per call against per-row cost (cycles)

The per-row cost comes from the batch of 64 (cycles ÷ 64); the fixed cost is one call
minus one row. At 64 and 60 points the batch of 64 leaves L1 (64 KiB in and out) and FFTW
plans the batch differently, so the split is only approximate there.

| n | Xeon N: call / row / fixed | Xeon F | Xeon M | EPYC N | EPYC F |
|---|---|---|---|---|---|
| 8 | 65 / 19 / 46 | 40 / 17 / 23 | 117 / 24 / 93 | 44 / 13 / 31 | 29 / 12 / 17 |
| 16 | 110 / 51 / 60 | 78 / 41 / 37 | 121 / 26 / 94 | 70 / 42 / 28 | 50 / 30 / 20 |
| 64 | 277 / 242 / 35 | 231 / 268 / −37 | 186 / 140 / 46 | 278 / 246 / 32 | 211 / 190 / 21 |
| 60 | 405 / 377 / 28 | 258 / 270 / −12 | 206 / 166 / 40 | 300 / 263 / 37 | 221 / 230 / −9 |

## Verdict

Diagnosis done; no library change. The memory hierarchy is the main cause only for the
Xeon's r2c 65536, and in part for the EPYC's 4096. In every other case FFTN's IPC is at or
above FFTW's, and on the Xeon its TLB walks and L2 misses are at or below FFTW's. The gap
comes from FFTN executing more instructions (EPYC: 1.4–3.4 times FFTW's for the same FP
work, except at 4096), from saturating Intel's port 5, and at 8–16 points from a fixed cost
per call. Per group:

| group | Xeon (`avx512`) | EPYC (`avx2`) |
|---|---|---|
| ≤ 16 points (8: 1.83 / 1.41; 16: 1.46 / 1.41) | **fixed per-call cost**: 23 of the 25 cycles of the gap at 8 points, 22 of 32 at 16 (FFTN 46–60 cycles per call against FFTW's 23–37). In a batch the cost per row is close to FFTW's (19 against 17 cycles at 8). Also a frequency licence: FFTN's 512-bit leaf runs at licence 1 against FFTW's 128-bit codelet at 0 (time ratio 1.84, cycle ratio 1.62 at 8). Front end: 32–69 uops not delivered per call (FFTW 0–3) | **fixed per-call cost** at 8 (14 of 15 cycles); at 16 split between 8 cycles fixed and 12 in the kernel |
| 64, 60, 120 (Xeon 1.32–1.62, EPYC 1.26–1.43) | **kernel instruction stream**: 1.26–1.78 times FFTW's instructions. At 60 the radix-4 grouped pass with 15 groups takes 57 % of the time, with 1.7 times the stores, store-forwarding blocks and 3.7 times the resource stalls. At 120 FFTN runs three passes against two (2038 against 1146 instructions). Fixed cost 28–35 cycles (FFTW about 0) | **kernel instruction stream** at 64 and 120: 1.37–1.38 times the instructions, at 120 with the same 10 × 12 plan as FFTW. At 60, 37 of the 79 cycles are **fixed per-call cost** (the two-pass chain's driver, about 15 % of the time) and the kernel takes the rest |
| 128–4096 (Xeon 0.98–1.04, EPYC 1.05–1.35) | at par. What remains is **port 5** (shuffles): 57–73 % busy against FFTW's 34–55 %, with 0.85–1.04 times its instructions. L1 replacements are twice FFTW's at 1024, but they barely stall | **kernel instruction stream** at 256–1024: 1.38–1.60 times FFTW's instructions for the same FP-pipe work (2019 against 1905 uops at 256), at IPC 3.7–4.7, near Zen 3's dispatch limit. No FP pipe is saturated (busiest 56–57 %, FFTW 61–68 %). The excess is integer and move instructions (address arithmetic, X18 item 2), plus 755 line-crossing loads at 1024. At 4096 the instruction counts are equal and the gap (1.05–1.11) is **memory hierarchy**: four passes against FFTW's three, 1.4 times the L1 fills |
| ≥ 16384, 46080, 100000 (Xeon 1.06–2.07, EPYC 1.44–2.00) | **kernel instruction stream of the grid's copies**: 1.55 / 2.09 / 3.50 times FFTW's instructions (65536 / 46080 / 100000) at a higher IPC (2.6–3.0 against 1.6–1.8). `block_copy` alone is 40–52 % of FFTN's instructions and 40–42 % of its cycles; `twiddle_transpose` and `twiddle_scatter` add 19–26 %. The scalar ports 1 + 6 carry 2–8 times FFTW's uops. Not the TLB, not aliasing and not L2: FFTN has fewer L2 misses than FFTW at 65536 and 46080 | the same: 1.94 / 2.65 / 3.38 times FFTW's instructions for 1.11–1.18 times its FP-pipe uops, at IPC 4.0–4.8 (FFTW 2.7–3.0). L1 fills are twice FFTW's at 65536 and 46080, and data L2 misses 4.6 times at 65536 (22k), but with IPC at 4 the copies' instructions set the time |
| r2c 65536 (1.36 / 1.29) | **memory hierarchy**: the half-length chain of 32768 (four passes plus the pairing) issues 0.44 times FFTW's instructions but runs at IPC 0.84 against 2.16, with 3.4 times the L2 misses and 3.7 times the stall cycles | **kernel instruction stream** (here a grid of 128 × 256): 1.97 times the instructions at IPC 4.15, and a tenth of FFTW's L2 misses |

Against MKL on the Xeon the gap is larger. At 60–120 points and on the long c2c rows, MKL
executes 0.29–0.64 times FFTN's instructions; at 65536 it runs 0.95 M against 3.24 M, at
IPC 0.89, memory-bound and still the fastest.

What it points to, by PLAN lead:

1. **Long rows.** The grid's data movement, not its memory traffic:
   - (a) `block_copy` for rows of one or two vectors (blocks of 4 columns on AVX-512): no
     inner loop, no per-row prefetch for a single line, several rows per iteration. The
     target is a few instructions per 64-byte row against 33.
   - (b) Wider blocks where the budget allows (the `b × longest ≤ budget` rule gives 4
     columns at 256-long phases), or a copy that moves C × C tiles, as FFTW's tiled
     transposes do.
   - (c) Fewer separate gathers and scatters: fuse them into phases with radix-16 passes
     where X12's set conflicts allow. Re-measure this on x86, where a separate copy now
     costs 40 % of the instructions.
   - (d) The radix-5 lane pass's spills and `twiddle_scatter` at 100000.

   Expected: about half of FFTN's instructions at 46080–100000 can go. IPC is already high
   on both machines, so that is most of the gap.
2. **Primes and factors above 16.** Not measured here (no case has such a factor). The
   radix-5 passes at 100000 spill, and at 46080 the radix-15 and radix-12 passes take 16 %
   of the time. No change to the lead.
3. **x86 kernels 128–4096.** On the EPYC the problem is the AVX2 pass loops' instruction
   count, which at IPC 4–4.7 can no longer hide behind the FP pipes. Address arithmetic
   and moves add 0.4–0.6 times FFTW's whole instruction count on top of the same FP work.
   This is X18's recommendation 1 and X10's whole-pass question, now with a clear target:
   3319 instructions at 256, 15.3 k at 1024. X10's leg tables cut instructions by 5–15 %
   and gained about 1 % on the EPYC; the cut needs to be 30–40 %. The line-crossing loads
   at 1024 are a smaller item. At 4096 four AVX2 passes against FFTW's three cost L1
   traffic; X18 found three radix-16 passes losing to register pressure on AVX2, so this
   follows the instruction work rather than the planner. On the Xeon this group is at par,
   and port 5 stays the limit (X1's split runs extended to the first pass).
4. **Single short rows.** Two different costs:
   - At 8–16 points, the fixed cost per call. FFTW runs single rows of 8 and 16 in 128-bit
     codelets (SSE2, AVX2-128) and pays 8–23 cycles less per call. A single-row
     leaf in 128-bit code (one complex per register, the NEON layout) would also keep the
     Xeon at frequency licence 0.
   - At 60–120 points, the kernels on the Xeon: the radix-4 grouped pass's partial unit at
     15 groups, and three passes at 120 where FFTW uses two. On the EPYC also the two-pass
     chain's driver (37 cycles per call, about 15 %): extend X20's flat path for leaves to
     short chains.
5. **Real transforms.** On the EPYC the cost is the same grid copies as lead 1, so fix
   those first. On the Xeon the r2c chain of 32768 is memory-bound. The chain-or-grid
   rule (X12b: chain up to L2/2) should count an r2c's whole footprint: the real input and
   the half-length output, besides the chain's lines. That would put the 65536-point r2c on
   the grid there. X6's real codelets cut instructions, which the Xeon case does not need.

New, for lead 6 (batches of short rows): FFTN's batched leaves saturate one unit where
FFTW's are balanced: Xeon ports 0 and 5 at 82–87 % for 8 and 16 points × 64, EPYC FP pipes
2 and 3 at 80–95 %, with 1.29–1.43 times FFTW's FP-pipe uops. The leaf's extra adds and
shuffles limit the batch, not its memory.

Lessons added to `dev/LESSONS.md`: 1.9 (user-mode counters without `perf`; FFTW's plans
vary by process), 3.9 (a copy pass over one-vector rows is instruction-bound), and the
frequency licence of tiny 512-bit transforms, under 2.3.
