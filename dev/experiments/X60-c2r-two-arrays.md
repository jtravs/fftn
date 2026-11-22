# X60: out-of-place c2r without a third array

- **Status:** done (2026-11-22): not adopted on any machine: the Xeon 6240 class won at one thread (0.82) and lost on volumes at 8 lanes; Zen 3 missed its 8-lane bar (0.970); Apple lost on wide planes. The code is removed
- **Owner:** X60 agent, 2026-11-19 (night round 2); X60 judge agent, 2026-11-22 (wave 1)
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

- **Arm PA** (2ad53aa, screen only). Q on `main` 021c117 (merged at 2dececc) with
  `c2r_packed_bytes` 0: packed wherever the layout allows.
- **Arm R** (285b062). Q with the threshold set per class from the PA screen (below), and the
  block wherever the class was not measured (`c2r_packed_bytes` SIZE_MAX by default):
  - the Xeon 6240's class (x86-64, a private L2 under 2 MiB, under 2 MiB of L3 per CPU): the
    L2 (1 MiB), and only for plans on `avx512` (`c2r_packed_avx512_only`: an `avx2` machine of
    that cache class keeps the block, AGENTS section 6);
  - Zen 3's class (the EPYC 7763: a private L2 under 1 MiB, a large L3 per CPU): two ninths of
    the CCX's L3, 7.1 MiB, that is input, block and output (three half spectra) past two thirds
    of the L3;
  - Apple's class (a cluster-shared L2): a quarter of the L2, 4 MiB, as Q;
  - Zen 4 and Zen 5, Granite and Sapphire Rapids, other AArch64 and unknown topologies: the
    block.

Bits differ from the block route in the last places, which `docs/conventions.md` allows
("between plans built with different options"). They are the same for every team size (tested).

## Method

- Cases: the standard catalogue's c2r planes and volumes
  (`^c2r-[0-9]+x[0-9]+(x[0-9]+)?-b[0-9]+-(op|ip)$`). The out-of-place ones with an even real axis
  are those the change can move.
- Machines:
  - EPYC 7763: GCC 16, `avx2`, DMOG through Slurm.
  - M5 Max: Homebrew Clang 23, `neon_fcma`, on AC power.
  - Xeon 6240: GCC 14.4, `avx512`, through Slurm (`--serial`): counters on 2026-11-19, the
    screen and judging runs on 2026-11-22.

### The gate screen (2026-11-22)

Arm PA against `main` 021c117, one thread unless noted, catalogue `bench/catalogues/x60.toml`
(the standard c2r planes and volumes out of place plus sizes between them; its header lists
each case's half spectrum in MiB).

- EPYC 7763, run 20261122-175616-a479: GCC 16, `avx2`, a shared node with 87 other busy cores
  (every round loaded), band ±1.24 %.
- EPYC at 8 lanes, run 20261122-175616-dd47: 92 other busy cores, band ±4.08 %.
- Xeon 6240, run 20261122-175805-b758: GCC 14.4, `avx512`, no round loaded, band ±2.29 %.
- M5 Max, run 20261122-175334-69d1: Homebrew Clang 23, `neon_fcma`, AC power, every round
  loaded by other agents' builds (load 13–21), band ±3.76 %.

PA ÷ `main` by half spectrum (MiB):

| case | MiB | EPYC | EPYC 8 lanes | Xeon | Mac |
|---|---|---|---|---|---|
| 128² | 0.13 | 0.987 | 0.869 | 0.929 | **1.216** |
| 256² | 0.50 | 0.940 | **1.112** | 0.833 | **1.141** |
| 36 × 2048 | 0.56 | 0.981 | 1.064 | 0.954 | 0.982 |
| 2048 × 36 | 0.59 | 1.009 | 0.856 | **1.079** | 0.913 |
| 320² | 0.79 | 1.006 | 0.743 | 0.886 | **1.126** |
| 48³ | 0.84 | 0.920 | 0.961 | 0.926 | 0.908 |
| 384² | 1.13 | **1.064** | 0.911 | 0.769 | **1.215** |
| 512² | 2.01 | **1.097** | **1.307** | 0.968 | **1.140** |
| 64³ | 2.06 | 0.866 | 1.144 | 0.825 | 1.007 |
| 480 × 640 | 2.35 | **1.027** | 1.071 | 0.894 | **1.105** |
| 640² | 3.13 | 1.041 | **1.189** | 0.822 | **1.119** |
| 600 × 800 | 3.67 | 0.866 | 1.042 | 0.800 | 0.984 |
| 512 × 1024 | 4.01 | **1.102** | **1.170** | 0.970 | 1.064 |
| 1024 × 512 | 4.02 | 0.961 | **1.181** | 0.754 | 0.908 |
| 768² | 4.51 | 1.002 | 1.082 | 0.816 | 0.941 |
| 864² | 5.71 | 0.838 | **1.139** | 0.835 | 0.896 |
| 96³ | 6.89 | **1.043** | 0.931 | 0.782 | 1.021 |
| 1000² | 7.64 | 0.775 | **1.112** | 0.686 | 0.891 |
| 100³ | 7.78 | 0.947 | 0.822 | 0.903 | 0.999 |
| 1024² | 8.02 | 0.928 | **1.124** | 0.739 | 0.860 |
| 128³ | 16.1 | 0.940 | 0.935 | 0.870 | 0.844 |
| 2048² | 32.0 | 0.911 | 0.763 | 0.898 | 0.996 |

Bold: slower beyond the run's band. Below 128² and 48³ PA lost or tied everywhere (16³ 1.07–1.23,
32³ 1.01–1.47; 32², 64², 96², 100² 0.91–1.04).

- **The Xeon gains almost everywhere** (geometric mean 0.883 over the 28 cases): 0.69–0.97 from
  384² up, 0.83 at 256². Its losses are within its L2: 2048 × 36 1.08 and 16³ 1.09. Its L2 (1 MiB)
  separates them, as Q had it; 256², 320², 48³ and 36 × 2048 gain too but lie on the losers' side.
- **The EPYC's losses reach 6.9 MiB** (512 × 1024 1.10, 96³ 1.04) while 600 × 800, 864², 64³, 48³
  and 256² gain. No threshold below 7 MiB separates them; from 7.6 MiB (1000²) up every case
  gains at one thread (0.78–0.95). 7.1 MiB, three half spectra at two thirds of the CCX's
  32 MiB L3, is the threshold. At 8 lanes on a node this loaded the run disagrees with the
  quieter P3 run of 2026-11-19 (1000² 1.11 against 0.96, 1024² 1.12 against 0.87); the judging
  run decides there.
- **The M5 Max keeps Q's 4 MiB**: from 768² up 0.84–1.02, below it 1.11–1.22 except 600 × 800,
  2048 × 36 and 48³.
- **Neither candidate of the handoff separates the EPYC by itself.** The LLC share by core (4 MiB)
  keeps 512 × 1024 and 96³, which lose. The pass count does not separate either: 256² gains
  (0.94) with three packed passes against the block's two, and 480 × 640 loses with three
  against three (the radix table under P3 below).

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

## Decision rule (written 2026-11-22, before the judging runs)

Arms: M = `main` (021c117, the merge base of arm R's branch), R = arm R (285b062), each run with
fbench's identical-arm control of M. Release toolchains: GCC on Linux (16 on the EPYC, 14.4 on
the Xeon), Homebrew Clang 23 on the Mac.

- **Judged cases, per target: only those R moves**, the cases of `bench/catalogues/x60.toml` whose
  half spectrum exceeds the target's threshold (every other case keeps M's plan):
  - Xeon 6240, one thread (`avx512`, threshold 1 MiB): 384², 512², 480 × 640, 640², 600 × 800,
    512 × 1024, 1024 × 512, 768², 864², 1000², 1024², 2048², 64³, 96³, 100³, 128³ (16 cases).
  - EPYC 7763 (`avx2`, 7.1 MiB), one thread and 8 lanes separately: 1000², 1024², 2048², 100³,
    128³ (5 cases).
  - M5 Max (`neon_fcma`, 4 MiB), one thread: 512 × 1024, 1024 × 512, 768², 864², 96³, 1000²,
    1024², 2048², 100³, 128³ (10 cases).
- **Adopt on a target** if, in each of its judged runs, (1) the weighted geometric mean of R ÷ M
  over its judged cases is below 0.97, and (2) no judged case is slower beyond noise: none with
  its whole 95 % interval above 1.02 × (1 + band). A case a hair past (2) on a loaded machine is
  re-run before it decides.
- **Collateral check:** the tracking tier (standard catalogue) R against M on each machine, one
  thread. A case slower beyond 1.05 × (1 + band) outside the judged set blocks adoption unless
  its plan is unchanged (describe() identical), in which case it is recorded as placement.
- **Per target.** A target that fails keeps the block (its class's value goes back to SIZE_MAX);
  a target that passes keeps R's threshold. If the EPYC passes one thread and fails 8 lanes it
  keeps the block (a team must not lose what one thread gains).

### Arm R2, Apple only (rule written 2026-11-22 after arm R's Mac judging run, before R2's)

Arm R failed its rule on the M5 Max (512 × 1024 1.068, below). Arm R2 (6e16a77) changes only
Apple's threshold, to a third of the cluster's L2 (5.3 MiB on the M5 Max). The threshold was
chosen on arm R's judged cases, so R2 is judged also on held-out sizes that no screen timed
(`bench/catalogues/x60b.toml`).

- Arms: M = `main` 021c117, R2 = 6e16a77, with the identical-arm control of M; the M5 Max, one
  thread, Homebrew Clang 23, AC power.
- Judged cases (half spectrum above 5.3 MiB): 864², 96³, 1000², 1024², 2048², 100³, 128³ of
  `x60.toml` and 640 × 1280, 1280 × 640, 960², 1200², 1536², 112³ of `x60b.toml` (13 cases), in one
  run over both catalogues' cases or two runs, each held to the rule.
- The criteria (1) and (2) above, over these cases. The other cases of both catalogues keep M's
  plan and are the collateral check with the tracking tier of arm R (Apple's change does not
  touch other classes).

### The Xeon 6240 at 8 lanes (rule written 2026-11-22 after the second reader, before the run)

Arm R's rule judged the Xeon at one thread only, but a plan runs serially or in a team as its
caller chooses, so the Xeon class's threshold also reaches team executions.

- Arms: M = `main` 021c117 and R = 285b062 (the Xeon class's threshold and code path are those
  of the final commit), each at 8 threads (`THREADS=8`), with the identical-arm control of M;
  GCC 14.4, `avx512`; `fbench_remote.py`'s job files in the `--serial` chain.
- Pinning (the Xeon's Slurm binds no CPUs, X59): fbench runs under `numactl -m` and `taskset`
  on eight idlest whole cores of the idlest socket and that socket's memory, one CPU of each
  core, chosen once at the job's start, as X59's confined runs.
- Judged cases: the Xeon's 16 of the one-thread rule (`x60.toml`, half spectrum above 1 MiB).
- Criteria (1) and (2) of the rule above over those cases. A pass keeps the Xeon class packed.
  A fail gives the Xeon class the block for every plan: a plan does not know whether a team will
  run it (the team is chosen at execution), so team plans cannot be told apart in the planner.

## Results: judging runs (2026-11-22)

R ÷ M (M = `main` 021c117) on the judged cases; band from fbench's identical-arm control of M.

| run | machine, threads | load | band | judged cases | geometric mean | worst judged case |
|---|---|---|---|---|---|---|
| 20261122-181505-34cf | Xeon 6240, 1 | 6 of 20 rounds loaded | ±2.51 % | 16 | **0.824** | 512 × 1024 0.977 |
| 20261122-181447-e6b7 | EPYC 7763, 1 | 95 other busy cores, every round | ±1.00 % | 5 | **0.911** | 100³ 0.960 |
| 20261122-181447-1a49 | EPYC 7763, 8 lanes | 95 other busy cores, every round | ±4.38 % | 5 | 0.970 | 1024² 1.130 [1.006, 1.233] |
| 20261122-184054-01f4 | EPYC 7763, 8 lanes (re-run) | 103 other busy cores, every round | ±4.14 % | 5 | **0.945** | 1000² 1.054 [0.973, 1.140] |
| 20261122-181143-ccdb | M5 Max, 1 (arm R) | every round loaded | ±1.26 % | 10 | 0.936 | **512 × 1024 1.068 [1.060, 1.079]** |
| 20261122-182925-c4a6 | M5 Max, 1 (arm R2, x60) | every round loaded | ±1.00 % | 7 | 0.926 | 96³ 1.023 [0.997, 1.039] |
| 20261122-183315-ed15 | M5 Max, 1 (arm R2, x60b) | 16 of 24 loaded | ±1.00 % | 6 | 0.906 | **640 × 1280 1.142 [1.131, 1.150]** |

- **Xeon 6240** (GCC 14.4, `avx512`): 384² 0.76, 480 × 640 0.88, 640² 0.84, 600 × 800 0.78,
  1024 × 512 0.73, 768² 0.81, 864² 0.79, 1000² 0.71, 1024² 0.76, 2048² 0.88, 64³ 0.84, 96³ 0.78,
  100³ 0.87, 128³ 0.87; 512² 0.955 and 512 × 1024 0.977 without evidence. The cases R leaves on
  the block tied (0.96–1.01; 16³ 0.94, the same plan). Passes.
- **EPYC 7763, one thread** (GCC 16, `avx2`): 1000² 0.80, 1024² 0.93, 2048² 0.92, 100³ 0.96,
  128³ 0.95, all beyond noise. The 23 cases left on the block: no evidence. Passes.
- **EPYC 7763, 8 lanes**: the first run, on a node with 95 other busy cores in every round, gave
  2048² 0.83 and 100³ 0.85 but 1000² 1.12 and 1024² 1.13 (both without evidence against a ±4.4 %
  band): 0.9702, which fails criterion (1). A re-run of the judged cases alone gave 0.945 (30
  rounds; 2048² 0.83 and 100³ 0.89 beyond noise, 128³ 0.93, 1000² 1.05 and 1024² 1.04 without
  evidence), but the rule's re-run clause covers a single case a hair past criterion (2), not the
  geometric mean (AGENTS section 6 says "a case" too), so the re-run does not decide, and 1000²
  and 1024² were above 1 in both runs. **Fails by the rule's letter.** (On a quieter node on
  2026-11-19, P3 at load 6.6, 1000² and 1024² took 0.96 and 0.87; on these nodes, with 95–103
  other busy cores, the team shared its CCX's L3 with other jobs.)
- **Xeon 6240, 8 lanes** (confined to node 1, CPUs 23, 26–29, 31, 32, 35; Slurm 16804): run
  20261122-212441-d091, 20 rounds, 12 loaded, band ±2.17 %. Geometric mean 0.901 over the 16
  judged cases (planes 0.70–1.01: 1024² 0.70, 1000² 0.76, 1024 × 512 0.77), but the volumes went
  the other way: 64³ 0.955, 96³ 1.068 [1.044, 1.104], 100³ 1.087 and 128³ 1.050 without evidence.
  96³'s interval clears the limit of 1.02 × 1.0217 = 1.042 by 0.002: a case a hair past
  criterion (2) on a loaded machine, so by the rule it is re-run before it decides (the same 16
  cases, same pinning).
- **Xeon 6240, 8 lanes, re-run** (node 1, CPUs 19, 22, 24, 26, 31, 32, 34, 35; Slurm 16807): run
  20261122-213005-de32, 30 rounds, 13 loaded, band ±1.56 %. Geometric mean 0.896 (the 12 planes
  0.850: 1024² 0.71, 1000² 0.73, 1024 × 512 0.78, the worst 512 × 1024 1.010), but the volumes
  lost again: 64³ 0.956, 96³ 1.061 [1.021, 1.076], 100³ 1.093 [1.023, 1.118], **128³ 1.081
  [1.056, 1.098]**, whose interval clears the limit of 1.02 × 1.0156 = 1.036. **Fails criterion
  (2).** At 8 lanes the Xeon class's teams run volumes plane at a time (X43b,
  `team_volume_planes`), and the packed volumes lose there in both runs (geometric mean of the
  four 1.046 in the re-run).
- **M5 Max** (Homebrew Clang 23, `neon_fcma`, AC power throughout, no sleep): arm R failed on
  512 × 1024 (1.068 against a limit of 1.033). Arm R2 (a third of the L2) passed the screen's
  cases but failed on a held-out one, 640 × 1280 at 1.142, while its transpose 1280 × 640 took
  0.884. On the M5 Max the packed route loses on wide planes (rows of 1024–1280 reals over
  512–640 columns) at 4–6.3 MiB, which no byte threshold separates from their transposes and
  from 768²–1536². Apple keeps the block.
- **Held-out sizes with arm R** (`x60b.toml`, outside the rule, a check that R's thresholds were
  not fitted to the screen's cases): Xeon run 20261122-183600-7a3a (band ±4.56 %, 7 of 13 rounds
  loaded) all nine faster, 0.78–0.91 (640 × 1280 0.794: the Xeon has no wide-plane loss); EPYC
  run 20261122-183610-eea5 (band ±1.00 %) 1200² 0.934, 1536² 0.918, 112³ 0.976 packed, the six on
  the block 0.99–1.00.
- **Collateral, tracking tier, arm R ÷ M** (61 cases): Xeon run 20261122-182127-3103 (±2.23 %)
  2 faster (c2r 1024² 0.71 and 480 × 640 0.87, both moved), 59 no evidence; EPYC run
  20261122-181447-13f4 (±1.25 %) 2 faster (c2r 1024² 0.93; c2c 120 0.93, the same plan), 59 no
  evidence; Mac run 20261122-181642-dec5 (±1.06 %) 2 faster, 59 no evidence. Nothing slower.

## Verdict (2026-11-22, after the second reader)

- **Not adopted on the Xeon 6240's class.** One thread passed (0.824 over 16 cases, every
  held-out size faster, no collateral damage), but 8 lanes failed criterion (2) on 128³ (1.081
  in the re-run that 96³'s hair-past 1.068 called for), with 96³ and 100³ above 1 in both runs.
  A plan does not know whether a team will run it, so the planner cannot keep the block for team
  executions alone, and by the rule written before the run the class keeps the block for every
  plan. Open follow-up: the packed route for planes only on this class (8 lanes: the 12 planes
  0.85, none slower beyond noise, in both runs; one thread 0.71–0.98), as a new arm with its own
  rule judged at one thread and 8 lanes.
- **Not adopted on Zen 3's class (EPYC 7763) by the rule's letter**: one thread gained (0.911
  over the 5 cases past two thirds of the CCX's L3, held-out 1200², 1536², 112³ 0.92–0.98), but
  the first 8-lane run gave 0.9702 against the bar of 0.97, with 1000² and 1024² above 1 in both
  8-lane runs. The class keeps the block. Open follow-up: a gate that packs only plans that build
  no team program, or by the team's share of the L3, as a new arm with its own rule (the
  maintainer may instead override this verdict on the re-run's 0.945).
- **Rejected on Apple**: the M5 Max keeps the block (wide planes 1.07–1.14 at every threshold
  tried).
- **Reach, for a follow-up keyed as arm R was.** The Xeon class's key (`set_grid_sizes`: x86-64,
  a private L2 under 2 MiB, under 2 MiB of L3 per CPU) also takes Ice Lake-SP, Zen 4c (1 MiB of
  L2, small L3 per core), Zen 4 VMs that report a small L3 per CPU, and Intel client parts with
  AVX-512 (Ice Lake and Tiger Lake laptops, Rocket Lake), none of them measured. The reach rule
  allows that (the same `avx512` target and register file; `c2r_packed_avx512_only` kept `avx2`
  machines of that cache class, such as Alder Lake and Zen 2, on the block), but a follow-up
  adopted on that key must say so and name these families for the next phase exit's screen.
- **Why the turn differs by machine.** The Xeon's small L3 per core (1.4 MiB) and 1 MiB L2 make
  the third array cost from L2 up (its c2r out of place ran 1.6–2.0 times in place), so the
  packed route's power-of-two output strides cost less than the block. The EPYC's 32 MiB CCX L3
  holds the three arrays to two thirds of it, and inside that the strides cost more (512 × 1024
  1.10, 96³ 1.04 in the screen). On the M5 Max a 16 MiB shared L2 holds them, and wide planes
  lose to the strides even beyond it.
- **Code.** Removed (AGENTS section 6: a path that wins nowhere by its rule is deleted):
  `src/`, `tests/`, `docs/conventions.md` and DESIGN 9.1 are `main`'s again; the branch's commits
  up to a983841 keep the packed route for a follow-up.
- **The X29 paired gather** for the Xeon's two-phase c2r grids stays a lead (no catalogue row).
- Tests, on the branch with the code (a983841): `Columns.C2rPackedMatchesTheBlock` (every
  runnable target, both precisions, against the block route to within twice the tolerance),
  `Columns.C2rPackedStridedLayouts` (strided, negative and reordered inputs, padded, negative and
  transposed outputs, odd batch distances; it failed with `in.stride[r - 1]` dropped from column
  h's start or the rows' block step zeroed, both checked by mutation),
  `C2rPacksBeyondTheProfilesBytes` (thresholds, the avx512-only flag on every target), team cases (packed volume, packed strided
  volume, block plane, packed plane: bitwise equal for every team size). Mac Homebrew Clang 23
  Release, all 2565 tests; Mac GCC 16 build with `-Werror`. The second reader's stress test
  (about 6,900 random planes and volumes against the block route, teams bitwise) passed.
