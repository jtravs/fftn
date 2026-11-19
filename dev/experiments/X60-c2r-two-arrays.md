# X60: out-of-place c2r without a third array

- **Status:** paused (2026-11-19): diagnosis done, change built and screened, no judging run;
  the change is a lead, not adopted
- **Owner:** X60 agent, 2026-11-19 (night round 2)
- **Phase / task:** 4.1 (planes), task 2.1's c2r

## Question

FFTN's out-of-place c2r of two or more axes keeps the half spectrum of one transform in the
workspace block, a third array beside the input and the output (X59: twice oneMKL's L2 misses
at one thread on the Xeon 6240). Does building the intermediate in the output instead make c2r
rows, planes and volumes faster? At which sizes, in place and out of place, at one thread and
in a team?

## Hypothesis

docs/conventions.md promises that an out-of-place plan never writes its input, c2r included.
It does not promise where the intermediate lives. The c2r of a row reads only n reals of its
half spectrum: X_1 .. X_h-1 and the real parts of X_0 and X_h (h = n/2; real.inl ignores their
imaginary parts). So after the complex axes each output row's n reals can hold its row packed:
X_0 with Re X_h in its imaginary place, then X_1 .. X_h-1. The row's c2r can then run in place
there. Input and output make 2 arrays where there were 3. That is one third less footprint,
so more planes fit L2 or L3. Beyond the caches it is one stream in six less: the column phase
writes the output where it wrote the block, and the row phase reads and writes the output where
it read the block and wrote the output. In place there is no third array, so nothing changes
there. The gain should appear where in + block + out leaves a cache level that in + out fits.

## Diagnosis: where the third array costs (one thread, before any change)

Harness (not in the repository). It builds one FFTN plan through the public API, pins it to the
idlest CPU and reads user-mode hardware counters around the executions only (two repeats,
medians).

- **Xeon 6240** (GCC 14, `avx512`; Slurm 16658). The machine was loaded by 25–33 other busy
  cores, so its cycle counts move by up to 30 % between runs. Only ratios beyond that, and the
  cache counters, rank cases.
- **EPYC 7763** (GCC 16, `avx2`; Slurm 513309). A shared node at load 113.

The rows and planes below are c2r out of place (op) against c2r in place (ip, no third array)
and r2c out of place. In the miss columns the three numbers are op / ip / r2c.

| case | Xeon op ÷ ip | Xeon L2 demand misses | Xeon LLC misses | EPYC op ÷ ip | EPYC L2 misses | EPYC fills from DRAM |
|---|---|---|---|---|---|---|
| 64² | 1.35 | 0 / 0 / 0 | 0 | 1.19 | 0 | 0 |
| 128² | 1.23 | 0 / 0 / 0 | 0 | 1.23 | 790 / 290 / 150 | 0 |
| 256² | **1.62** | 14 800 / 930 / 1 900 | 2 / 1 / 1 | 1.10 | 21 400 / 8 000 / 10 200 | 1 / 0 / 0 |
| 512² | 1.44 | 94 500 / 32 900 / 41 900 | 390 / 430 / 30 | 1.12 | 70 300 / 33 800 / 43 300 | 130 / 6 / 36 |
| 1024² | **2.03** | 382 000 / 181 000 / 222 000 | **370 000** / 52 000 / 143 000 | 1.38 | 197 000 / 66 000 / 136 000 | **116 000** / 3 700 / 51 000 |
| 2048² | 1.29 | 2.10 M / 1.79 M / 1.71 M | 2.19 M / 1.28 M / 1.78 M | 1.19 | 2.69 M / 2.38 M / 2.65 M | 870 000 / 276 000 / 626 000 |

- **Planes.** The block costs most where in + block + out passes a cache level that in + out
  fits. On the Xeon this happens twice. At 256², 1.55 MB passes its 1 MiB L2 (16 times the
  in-place L2 misses). At 1024², 25 MB passes its 24.75 MiB L3: seven times the LLC misses, 2.0
  times the time. On the EPYC, 1024² (25 MB against a 32 MiB L3 per CCX) takes 31 times the DRAM
  fills. Below L2 (64², 128²) op still runs 1.2–1.35 times ip, so the counters show no cache cost
  there. The workspace grows with the block: 644 KiB at 256², 8.3 MiB at 1024², 33 MiB at 2048².
- **Rows (1024 to 2²⁰).** A 1-D c2r has no third array to remove. Out of place, the chain pairs
  from the input into the output and runs its complex transform in place there. On the Xeon,
  rows from 131072 points are grids whose in-place phases keep a block in both placements, as
  r2c's three-phase grids do. Op ÷ ip of rows with identical programs moved 0.92–1.24 in these
  runs, which is the harness's and the machines' noise, not a contract cost. The X29 paired
  gather (no block for a two-phase c2r grid out of place) stays a lead for the Xeon, but no
  catalogue row on the Xeon is a two-phase c2r grid.

## Change under test

All arms are on branch `exp/x60-c2r`.

- **Arm P** (c1fe7b8). A c2r of two or more axes out of place packs its intermediate in the
  output. It needs an even real axis contiguous on the output and even output strides
  (`c2r_packed`, plan_core.cpp).
  - The complex axes run on columns 0 .. h-1 from the input into the output, read as h complex
    values per row.
  - Column h runs through the complex axes from the input into the block. The block shrinks
    from one transform's half spectrum to one value per row.
  - The real part of column h goes into the imaginary places of column 0, which column 0's own
    transform wrote first.
  - The rows then run their c2r in place on the packed rows (`pass_args::packed`, read by
    c2r_pre, c2r_first and c2r_leaf: Re X_h is at element 0's imaginary place).
  - Odd real axes and odd output strides keep the block, and in place is unchanged. The input is
    never written.
- **Arm P2** (8744868). P, with the packed columns built by the c2c columns' block rule
  (`column_block`, `column_radices`) instead of the real planes' rule. The output's rows are n
  reals, a power of two for most n, so the columns' elements lie a power of two apart there as in
  a c2c plane. The block's rows of n/2 + 1 columns lay an odd multiple of 16 bytes apart.
- **Arm P3** (ada7573). P2, with column h's transform into the block as one more line of the last
  column phase. That phase must have no narrower block of its own, and a team then runs the line
  beside the column blocks. Each row takes Re X_h from the block in its own phase
  (`c2r_nyquist`) instead of a strided pass over column 0.
- **Arm Q** (5440073). P3, packing only where one transform's half spectrum exceeds
  `machine_profile::c2r_packed_bytes`: the L2 where it is private, a quarter of a shared L2. On
  the EPYC that is 512 KiB, on the Xeon 1 MiB and on the M5 Max 4 MiB.

Bits differ from the block route in the last places, which `docs/conventions.md` allows
("between plans built with different options"). They are the same for every team size (tested).

## Method

- Cases: the standard catalogue's c2r planes and volumes
  (`^c2r-[0-9]+x[0-9]+(x[0-9]+)?-b[0-9]+-(op|ip)$`). The out-of-place ones with an even real axis
  are those the change can move.
- Machines:
  - EPYC 7763: GCC 16, `avx2`, DMOG through Slurm.
  - M5 Max: Homebrew Clang 23, `neon_fcma`, on AC power.
  - Xeon 6240: counters only. Its judging run was not reached.

## Results: screens (not judging runs)

**P against `main`, M5 Max** (run 20261119-073827-9bab, A/A band ±2.3 %). The real planes' rule
on NEON double (C = 1) ran the packed columns alone through the chain, their elements a power of
two apart. It lost 2.1–2.6 at 256²–512², 36 × 2048 and 32³–64³. Hence P2.

**P and P2 against `main`, EPYC, one thread** (run 20261119-075943-e2ec, band ±1.40 %):

| | 256² | 512² | 1024² | 2048² | 480 × 640 | 1000² | 48³ | 64³ | 128³ | 32 × 30 | 32³ | 3 × 5 × 4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 0.94 | 1.14 | **2.12** | 0.90 | 0.97 | 0.96 | 0.98 | 0.86 | 0.94 | 1.08 | 1.08 | 1.10 |
| P2 | 0.94 | 1.08 | 0.91 | 0.91 | 1.05 | **0.79** | 0.92 | 0.86 | 0.94 | 1.10 | 1.07 | 1.18 |

P's 1024² ran its columns alone through the chain into rows 8 KiB apart. Its L2 misses tripled
in the counter runs (Slurm 513313 against 513309).

**P3 against `main`** (one thread). EPYC run 20261119-081318-866a (band ±1.00 %); M5 Max run
20261119-081225-b844 (band ±1.00 %):

| | 256² | 512² | 1024² | 2048² | 480 × 640 | 1000² | 36 × 2048 | 2048 × 36 | 48³ | 64³ | 100³ | 128³ | 128² | 32³ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EPYC | 0.945 | **1.084** | 0.909 | 0.909 | **1.037** | **0.800** | 0.968 | 1.013 | 0.938 | **0.859** | 0.968 | 0.949 | 1.008 | 1.129 |
| Mac | 1.144 | 1.145 | 0.930 | 0.942 | 1.143 | 0.936 | 1.008 | 0.906 | 0.917 | 1.006 | 1.010 | **0.854** | 1.266 | 1.463 |

Planes and volumes below L2 lost on every machine: 8 × 6 1.34–1.37, 3 × 5 × 4 1.45–1.50, 32 × 30
1.01–1.12. Arm Q leaves those on the block. Over the cases Q changes, P3's ratios give these
geometric means:

- **EPYC** (twelve cases from 256² up): about 0.945, but 512² 1.084 and 480 × 640 1.037 lose.
- **M5 Max** (1000², 1024², 2048², 100³, 128³): about 0.93, the worst 100³ at 1.01.

**P3 against `main`, EPYC at 8 lanes** (run on job fbench-20261119-080927-ada75734-c9d5a6, band
±4.1 %):

- Gains: 2048² 0.755, 128³ 0.81, 1024² 0.87, 128² 0.91.
- Losses: 512² 1.24, 480 × 640 1.28, 32³ 1.46, 48³ 1.07.
- Overall: 1.042 over the 25 cases.

**Why 512² and 480 × 640 lose.** In the output their columns' elements lie 4–5 KiB apart, so the
c2c rule's leg check (`column_radices`, X40) gives them three-pass chains: 4 × 16 × 8 for 512 and
6 × 10 × 8 for 480. In the block, at an odd stride, they had two passes (16 × 32). That costs
about 10 % more instructions (counters, Slurm 513327). The two-pass chain forced into the output
is worse still on the EPYC (Slurm 513338, one thread, the node at load 37), so the leg check is
right there too:

| EPYC, ns | block (its radices) | packed (rule) | packed, block's radices |
|---|---|---|---|
| 512² | 664 000–683 000 (16 × 32) | 750 000 (4 × 16 × 8) | 888 000 (16 × 32) |
| 480 × 640 | 797 000 (4 × 10 × 12) | 842 000 (6 × 10 × 8) | 861 000 (15 × 32) |
| 32³ | 70 700 (32) | 86 800 (4 × 8) | 132 000 (32) |
| 256² | 158 000 (8 × 32) | 156 000 (4 × 8 × 8) | 204 000 (8 × 32) |

On the M5 Max the crowded two-pass chain
forced into the output took twice the time (local runs). There, even with the block route's own
radices (8 × 8 × 8 at 512²), the packed columns were 13–15 % slower at 256²–512² whatever the
block width (4–32 columns). The packed route's power-of-two output strides cost more inside the
caches than the third array does.

## Verdict (provisional: no judging run)

- **The third array is real but only costs beyond a cache.** Out of place, c2r planes run 1.1–2.0
  times their in-place time on the Xeon and the EPYC at one thread. The biggest costs are the
  Xeon's 256² (L2) and 1024² (L3), and the EPYC's 1024² (L3). Rows have none.
- **Packing the intermediate in the output wins large planes and volumes and loses the middle.**
  Gains: EPYC 1000² 0.80, 64³ 0.86, 1024²/2048² 0.91; M5 Max 1000²–2048² 0.93, 128³ 0.85; EPYC at
  8 lanes 2048² 0.76. Losses where the output's power-of-two row stride forces longer column
  chains or set conflicts: EPYC 512² 1.08 (8 lanes 1.24), M5 Max 128²–640 1.14–1.27, and small
  planes and volumes everywhere.
- **Not adopted.** Arm Q (pack only beyond L2) would pass a "no changed case slower" rule on the
  M5 Max by its screen but not on the EPYC (512², 480 × 640). The Xeon was not timed. The code
  stays on the branch as the lead; `main` keeps the block.
- **Next steps** (the handoff in the pull request):
  1. A gate that keeps 512², 480 × 640 and 32³ on the block. The leg check cannot be relaxed
     (above). Candidates: pack only where in + block + out passes the last-level cache share, or
     where the packed columns need no more passes than the block's. Each must be measured against
     256², 36 × 2048, 48³ and 64³, which gain with it.
  2. Write the rule over the cases the final arm changes, then judge one thread on the EPYC, the
     Xeon (`--serial`) and the M5 Max, the tracking tier, and 8 lanes on the EPYC.
  3. The X29 paired gather for the Xeon's two-phase c2r grids (no catalogue row).
- Library size, tests: not measured for a verdict. The branch's tests
  (`Columns.C2rPackedMatchesTheBlock` and `C2rPacksBeyondTheProfilesBytes`, team cases for a
  packed c2r volume and a block c2r plane) pass on the Mac (Homebrew Clang 23: neon, neon_fcma,
  scalar).
