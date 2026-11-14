# X4: Padding the grid's work arrays

- **Status:** running
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

Pending the run.

## Verdict

Pending.
