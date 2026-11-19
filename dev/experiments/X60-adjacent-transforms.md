# X60: strided axes of an array as lanes over adjacent transforms

- **Status:** running
- **Owner:** task 4.3 agent, 2026-11-19
- **Phase / task:** 4 / 4.3

## Question

When the transformed axes of an array are all strided because a dimension that is not
transformed varies fastest (the columns of a matrix, the planes (y, x) of a (y, x, w) array,
any layout with a batch dimension one element apart), should the transforms run as blocks of
adjacent transforms, one per vector lane, instead of one at a time through the chain?

## Hypothesis

Such a transform reads its elements `batch` apart, a cache line or more, so each element costs
a line, while the batch's neighbours sit in the same lines. The planes column phase already
exploits this for columns (X40: 1.44–2.07 → 0.88–0.94 of FFTW): a run of C adjacent lines is
one vector access, the whole block lands in a tile in scratch, and every line of memory is
touched once. Here every axis of the transform is such a column. Today's route takes every
transform alone through the chain (`c2c_line` with its strided first and last passes). An
uninstrumented probe on the M5 Max under load (FFTN against FFTW MEASURE, ns per point of whole
batches: columns of 64 x 1024 3.1 of FFTW's time, 64² planes in batches of 64 and 512 3.0 and
3.3, 256² x 16 2.7, middle axes 1.4–2.9) put today's route 1.4–3.9 times behind FFTW; the same
probe with the change was 0.34–1.10 of FFTW.

## Change under test

Arm B: the change: the planner (rule below), `program::lane_tail` as records, the driver's
lanes loop running the body's records, `build_work_graph` over a lanes program with loops,
describe(). Arm A: the same binary with fbench's `COLUMNS=chain` (`build_settings::
lanes_columns` false), which turns the route off. On these cases A's program is the one the
branch had before the change: all their axes are strided, so the column phases `lanes_columns`
also governs never applied to them. (A first submission compared the commits before and after
the change; its runs stopped on a harness bug in the interleaved cases with `outer` blocks,
fixed before any result was read, and one binary is the cleaner comparison. The collateral runs
keep the two commits, since they judge the code that changed.)

Rule: a c2c plan whose innermost batch dimension has at least 2 transforms one element apart on
both sides and whose every axis of more than one point is strided on both sides and fits the
lanes (`lanes_fit`: codelet radices, no Bluestein stage) runs as the lanes program of
DESIGN 9.3 with B = min over the axes of the block `column_block` gives for columns of that
axis (the transforms taking the place of the columns), the chain otherwise. Radices per axis
as `column_radices`. Not for r2c and c2r, whose halved axis needs a real lane path of its own
(left for later; they stay on the chain). Only when all the plan's transforms together exceed
the L1 data cache (`machine_profile::adjacent_min_bytes`): added after the first screen below,
which showed every case slower than the chain lies within L1. (A first form of the condition
counted one block of `outer`'s transforms and took the middle axes, whose blocks fit L1, back
to the chain, which the Mac's run of `e296850` showed: 0.81 against 0.62. All the transforms
count: a block that fits L1 still meets cold memory when many blocks follow one another.)

## Method

- Cases: the `axes` catalogue (bench/README.md "interleaved" layout), c2c groups: columns,
  columns-many, planes, planes-long-axis, backward, middle-axis, volumes, out of place and in
  place (`real` group: reported, unchanged code path, not judged).
- Machines: EPYC 7763 (DMOG, GCC 16, `avx2`) and Xeon Gold 6240 (GCC 14, `avx512`, FFTW and
  oneMKL) through `fbench_remote.py submit --serial`; M5 Max (Clang 23, `neon_fcma`) locally on
  AC power when no other fbench is running.
- Arms in one run: A, B, FFTW (MEASURE, fresh wisdom), oneMKL where there is one. One thread;
  a threaded run (8 lanes) of A and B on the EPYC and the Xeon for the check below.
- Metric: B ÷ A per case with fbench's paired interval; geometric mean of the ratios over the
  c2c cases; each machine's A/A band.
- **Decision rule (written before the judging runs):** adopt B on a machine if the geometric
  mean of B ÷ A over the c2c cases of `axes` is at most 0.85 and no such case is slower than
  1.05 of A beyond that machine's A/A band; otherwise keep the chain route on that target. At
  8 lanes no c2c case may be slower than 1.10 of A beyond the band (a block of transforms is
  one task, so a case with few transforms may lose parallelism; such a case gates the rule, see
  verdict). The ratios to FFTW and oneMKL are reported; they do not decide this experiment
  (task 4.3's acceptance is correctness, DESIGN 1.3's targets are phase 4's exit).
- Collateral: the standard catalogue's tracking tier and `x7` (the lanes strategy's rows,
  whose driver path this change touches) on the EPYC: no case outside `axes` is slower than
  1.03 beyond the band.

## Results

**Screen** (`4b02a69`, the rule without the L1 condition, so these runs are not the judgement):
B ÷ A over the 86 c2c cases: M5 Max 0.496 (run 20261119-010745-4c74, loaded, band ±1.00 %), Xeon
6240 0.387 (20261119-012036-2d43, band ±5.02 %). Every case slower than A beyond the band was
within L1: M5 Max (128 KiB) 8 x 16 and 8 x 256 1.34–1.42, 16 x 16 x 8 1.19, 16 x 16 and 16 x
256 1.09–1.15, 32 x 32 x 8 and 32 x 256 1.11–1.13, 256 x 16 1.07; Xeon (32 KiB) 8 x 16 1.20
and 16 x 16 x 8 1.13. The chain's leaves and short passes do no strided work worth avoiding
there. Hence the L1 condition.

To fill in: the judging runs.

## Verdict

Open.
