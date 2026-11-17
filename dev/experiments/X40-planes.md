# X40: where batched 2-D c2c loses, and plane programs that do not

- **Status:** running
- **Owner:** task 4.1 agent, 2026-11-17
- **Phase / task:** 4.1

## Question

Where does FFTN's time go on batched 2-D c2c planes against FFTW's and MKL's, and which
change to the plane program removes it? The answer chooses the first arms of task 4.1.

## Hypothesis

Before this experiment a plane is axis by axis, line by line (DESIGN.md 9.3 "as built" was
the chain for every line): the rows through the chain, then every column alone, its
elements `nx` apart. On targets with several complex values per register (C > 1: `avx2`,
`avx512`) a strided column is copied element by element into scratch, run there and copied
back (chain.hpp); on NEON double (C = 1) the chain's passes read and write the column at
its stride. Either way a column touches one element per cache line and the next column
reads the same lines again; when `nx` is a power of two of 4 KiB or more those lines all
fall into a few L1 and L2 sets, so the second column finds none of them (PLAN suspected
aliasing on the EPYC). Expected: the column phase is most of the time, the row phase near
the 1-D batch's speed, and FFTW runs the columns as vector loops of codelets across
neighbouring columns (rank-2 plans split into a rank-1 row plan and a vrank-1 column plan).

The fix DESIGN.md 9.3 already names: the strided axis through the lanes strategy over
blocks of neighbouring columns (C columns per vector, so no copy and no transpose), each
block's chain in a scratch tile, with the block's first pass reading and last pass writing
all B columns of one row before the next row (whole cache lines, each line read once).

## Change under test

- Arm D (diagnosis, no library change): the baseline (`main` 841c414) against FFTW with its
  printed plans (`PLAN_NOTE=1`) and MKL; and on the Mac, a scratch harness timing a plane's
  row phase (the 1-D batch of rows) and column phase (a 1-D plan of one strided axis with a
  batch of adjacent columns) apart.
- Arm L (lanes columns): in a c2c plan of two or three axes, every axis but the last whose
  neighbouring columns are adjacent on both sides (the last axis has stride 1 in the phase's
  source and destination) runs as the lanes strategy over blocks of B adjacent columns:
  `lanes_first` reads the block's B columns, the chain runs on a tile in scratch,
  `lanes_last` writes them back (in place or not). The block's units run its column groups
  innermost, so the first and last passes take each row's B columns at once. B, the tail
  (the last `nx mod B` columns, in blocks of C padded with idle lanes) and the scratch follow
  the lanes strategy's rules (`lanes_block`), with B at least a cache line's columns. The
  rows and 1-D plans are unchanged (their programs are identical; the tests' 1-D cases and
  the tracking tier's 1-D cases are the collateral check).
- Further arms only as the diagnosis and arm L's result support (block ordering or padding
  against power-of-two strides; plane-at-a-time blocking for planes beyond L2).

Bits: arm L changes the column phase's arithmetic route (each column's chain runs as a
lane of a tile instead of a copied line), so results move within the derived tolerances;
determinism across teams holds (a block is a line of the lines loop).

## Method

- Cases: the standard catalogue's c2c 2-D planes, tier full, filter
  `^c2c-[0-9]+x[0-9]+-b[0-9]+-(op|ip)$` (37 cases: the smoke planes in batches of 2, the
  planes group in and out of place, the plane batches of 16).
- Machines, compilers, instruction sets: Mac M5 Max (Apple Clang via Homebrew LLVM,
  `neon_fcma`, FFTW NEON build), EPYC 7763 (GCC 16, `avx2`), Xeon 6240 (GCC 14, `avx512`,
  with MKL). One thread. fbench A/B, interleaved rounds, A/A control; remote timing through
  `fbench_remote.py submit --serial`.
- Metric: per case the median paired ratio arm ÷ main, 95 % interval; the geometric mean
  over the cases; FFTN ÷ FFTW and ÷ MKL for the record.
- Decision rule, written before running: adopt an arm on a machine-target (release
  toolchain) if (1) the geometric mean of arm ÷ main over the 37 cases is at most 0.95, and
  (2) no case is slower than main by more than the larger of 2 % and the run's A/A band (a
  case just past it on a loaded machine is re-run before it decides), and (3) 1-D plans are
  unchanged (describe and program identical; checked by the tests and by construction). A
  win on some targets and a loss on others is adopted per target.

## Results

### Diagnosis

(pending)

## Verdict

(pending)
