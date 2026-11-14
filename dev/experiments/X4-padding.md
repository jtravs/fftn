# X4: Padding the grid's work arrays

- **Status:** done for the grid on the Mac (stagger adopted, padding rejected); x86 and the edge-pass half of X4 open
- **Owner:** grid agent, 2026-11-14
- **Phase / task:** 1 (X4, applied to the grid of task 1.7)

## Question

Do padded work arrays, whose rows are not a power of two apart, remove the cache-set
conflicts and memory costs that keep the grid's long rows (2^20, 2^22) and radix-16 end
passes slow? The answer decides whether the grid pads its intermediate array and its
scratch, and whether its phases may then read padded rows directly with radix 16 at the
ends.

## Hypothesis

DESIGN.md 10 (rules 1, 2 and 4): R streams whose spacing is a multiple of the L1 set period
conflict once they exceed the associativity; padding breaks the spacing, and staggering the
ping-pong pair keeps them from aliasing. In the grid (X12) every strided access is to rows a
power of two apart: X12 found radix 16 at a phase's ends doubling its time when the phase
read or wrote rows directly, and the gathers of phase 2 (which read the intermediate) the
costliest single stage at 2^22. The intermediate is FFTN's own array, so it can be padded;
the caller's input and output cannot. Expected: a padded intermediate speeds phase 2's
reads (fewer set conflicts, rows spread over memory channels and banks), most at 2^20 and
2^22; phase 2 may then read its rows in its first pass with any radix; staggering the
scratch lines is a small effect, since a block's scratch fits L1.

## Change under test

All arms are one build (this branch); the arms are build settings set through fbench
options (bench/README.md), so no library switch exists (no `dev/KNOBS.toml` entry):

- B: today's grid (no padding).
- C: `GRID_STAGGER=1`: scratch line b starts 7 × 64 bytes after a multiple of a's size
  (rule 4). Padding the rows inside the scratch block was not built: a block's scratch
  fits L1 and its passes' legs are 4–8 KiB apart, spread over the sets.
- D: C plus `GRID_PAD=8`: the intermediate array's rows, as phase 2 reads them, are padded
  by 8 complex values (128 bytes of doubles, one Apple cache line) per row; the
  intermediate then lives in the workspace block for two phases out of place as well
  (today's two-phase plans write it into the output).
- E: D plus `GRID_FUSE_PADDED=1`: phase 2 reads its padded rows in its first pass whatever
  its radix (radix 16 included), instead of a gather.

Padding moves data, not arithmetic: every arm gives the same bits as B (tested,
`Grid.PaddingKeepsBits`; the stagger moves only scratch).

## Method

- Cases: standard catalogue, tracking tier, the long c2c rows:
  `--filter 'c2c-(16384|46080|65536|100000|262144|1048576|4194304)-b1'` (2^14 to 2^22,
  two smooth lengths, 65536 in place).
- Machine: the Mac (M5 Max, `neon_fcma`, double), serial, shared; the load line of the run
  is the noise context.
- Command:

  ```bash
  uv run --frozen python bench/fbench.py run A=fftw B=fftn C=fftn+GRID_STAGGER=1 \
    D=fftn+GRID_STAGGER=1+GRID_PAD=8 E=fftn+GRID_STAGGER=1+GRID_PAD=8+GRID_FUSE_PADDED=1 \
    --tier tracking --filter 'c2c-(16384|46080|65536|100000|262144|1048576|4194304)-b1'
  ```

- Metric: the paired ratios C ÷ A, D ÷ A, E ÷ A against B ÷ A per case, with intervals, and
  the A/A band.
- Decision rule, written before running: an arm becomes the default (for the cases where
  it applies) if its ratio to FFTW is lower than B's by more than the band in at least one
  case of 2^18 or more and higher by no more than the band in every case; otherwise it is
  rejected and its code deleted. If D or E wins only beyond some length, it becomes the
  default beyond the smallest length where it wins, stated as a profile threshold.

## Results

Three fbench runs on the Mac, all under heavy load (load average median 12–17, every round
marked loaded), source as stated; ratios are medians over rounds with 95 % intervals.

Run 20261114-141105-e81d (source 2b27191, 40 rounds, band ±2.66 %), every arm against FFTW
(A); the arms as above (D's padding then also moved two-phase plans' intermediate into the
workspace block):

| case | B today | C stagger | D stagger + pad | E D + fused padded |
|---|---|---|---|---|
| c2c-16384-b1-op | 1.359 | 1.313 | 1.346 | 1.263 |
| c2c-65536-b1-op | 1.245 | 1.231 | 1.257 | 1.298 |
| c2c-262144-b1-op | 1.183 | 1.179 | 1.110 | 1.103 |
| c2c-1048576-b1-op | 1.161 | 1.135 | 1.098 | 1.124 |
| c2c-4194304-b1-op | 1.047 | 1.000 | 0.971 | 1.124 |
| c2c-46080-b1-op | 1.137 | 1.147 | 1.160 | 1.115 |
| c2c-100000-b1-op | 1.266 | 1.258 | 1.341 | 1.243 |
| c2c-65536-b1-ip | 1.036 | 1.007 | 1.021 | 1.003 |

Ratios to FFTW from separate arms carry FFTW's noise twice, so two paired runs against
today's grid (A = fftn) followed:

Run 20261114-142415-0075 (source 2b27191, 30 rounds, band ±1.62 %), ratio to today:

| case | pad 8 | stagger + pad 8 | pad 8 + fused padded |
|---|---|---|---|
| c2c-16384-b1-op | 1.044 [1.033, 1.081] slower | 1.000 | 0.984 |
| c2c-65536-b1-op | 1.009 | 0.983 | 1.014 |
| c2c-262144-b1-op | 0.969 | 0.953 | 0.967 |
| c2c-1048576-b1-op | 0.989 | 0.934 [0.911, 0.962] faster | 0.998 |
| c2c-4194304-b1-op | 0.982 | 0.928 [0.913, 0.949] faster | 1.049 [1.027, 1.090] slower |
| c2c-46080-b1-op | 1.011 | 1.013 | 0.968 |
| c2c-100000-b1-op | 1.036 | 1.021 | 1.020 |
| c2c-65536-b1-ip | 1.010 | 0.969 [0.909, 0.980] faster | 1.033 |

Run 20261114-143135-0921 (source 89f22d6, the padding now only where the intermediate
already is the workspace block: three phases, in place; 30 rounds, band ±4.04 %), ratio to
today:

| case | stagger | stagger + pad 8 |
|---|---|---|
| c2c-16384-b1-op | 0.958 | 0.968 |
| c2c-65536-b1-op | 0.972 | 0.976 |
| c2c-262144-b1-op | 0.990 | 0.967 |
| c2c-1048576-b1-op | 0.956 | 0.928 [0.907, 0.946] faster |
| c2c-4194304-b1-op | 0.913 [0.894, 0.937] faster | 0.918 [0.885, 0.934] faster |
| c2c-46080-b1-op | 0.972 | 0.980 |
| c2c-100000-b1-op | 1.010 | 1.007 |
| c2c-65536-b1-ip | 0.937 | 0.950 |
| weighted geomean | 0.963 [0.935, 0.985] | 0.961 [0.935, 0.977] |

Reading: the stagger is the active part. It is faster at 2^22 in both runs that had it
(−7 % and −9 %), no case is slower beyond the band, and the geometric mean is 0.96. The
padding adds nothing separable from it: alone it changed nothing at 2^18–2^22 and cost
4.4 % at 2^14 (where it moved a two-phase plan's intermediate out of the output into the
workspace); on top of the stagger it moved 2^20 from 0.956 to 0.928, inside the 4 % band.
Reading padded rows directly with radix 16 (fused padded) gained nothing and lost 4.9 % at
2^22. The likely reading (not measured directly): the set conflicts that cost were between
the ping-pong scratch lines, not between the intermediate's rows; a block's two scratch
lines were a power-of-two size apart, so every pass read and wrote the same sets. In run
20261114-142415-0075 the padding also moved two-phase plans' intermediate into the
workspace, as in run 20261114-141105-e81d.

## Verdict

By the rule: the stagger is adopted as the grid's default (faster beyond the band at 2^22
in two runs, slower nowhere), for grid plans only (the chain's scratch lines were not
measured). Padding the intermediate and the fused-padded reads are rejected and their code
deleted (in the commit after 89f22d6; git keeps it). Net on the Mac: 2^22 about 8 % faster, 2^20
about 4 % (inside the band), the rest unchanged. Open: whether the chain's ping-pong lines
gain from the same stagger (they are a power of two apart for power-of-two rows); padding
on x86 (4 KiB set period, 8-way L1, where the stream rule binds sooner); and X4's other
question, edge-pass variants chosen by the caller's base address.
