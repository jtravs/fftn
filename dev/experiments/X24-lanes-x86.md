# X24: cheaper lanes on x86 (short-row batches in double)

- **Status:** running
- **Owner:** x24 session, 2026-11-15
- **Phase / task:** 2.2 (acceptance for double c2c on AVX-512), follow-up of X7 and X20

## Question

Can the lanes strategy beat the chain with X20's batched leaves in double on x86 at 8–64
points, and reach FFTW at 8–16 points? The answer sets `lanes_up_to` in double for `avx512`
and `avx2` (`src/planner/rules.cpp`, now 0) and decides whether task 2.2's acceptance can
be met for double c2c on AVX-512.

## Hypothesis

X7's amendment found the lanes 1.30 (Xeon AVX-512), 1.78 (Xeon AVX2) and 1.60 (EPYC) of
the batched-leaf chain's time at 8 points in double, yet both leave lanes idle: the chain's
leaf broadcasts each input into all C lanes and stores lane 0 (X20: the kernel is 80–95 %
of a leaf row's time on the Xeon), so the lanes do a quarter (AVX-512) or half (AVX2) of its
arithmetic per row. What X7 names as the lanes' remaining costs on x86 are:

- (a) the driver: each block copies every stage's 144-byte `pass_args` and walks the stages
  (`run_lane_blocks`), where X20 prepares the arguments once per call and runs a batch of
  leaf rows as one call;
- (b) the gather and the scatter: rows a whole row apart are not one vector access when
  C > 1, so a block goes through two extra passes of C × C transposes. FFTW instead builds
  its vectors from 128-bit inserts in the codelet's loads. One complex double is one 128-bit
  unit, so a register of C rows' element j can be built from C loads at the rows' distance
  (`vinsertf128`, `vinsertf64x4`) and stored by C 128-bit stores (`vextractf128`,
  `vextractf64x2`), with no pass of its own.

Expected: (a) small (a few nanoseconds per block of up to 64 rows: under 2 % at 8 points);
(b) large at 8–16 points, where the whole lanes body becomes one leaf pass reading and
writing the rows (three passes over the block become one), and smaller from 32 points up
as the passes between the edges dominate. With (b), the lanes at 8–16 points should be
below the chain (the leaf's arithmetic over C) and near FFTW, which does the same.

## Change under test

Three commits on `exp/x24-lanes-x86`, each an arm, all forced with fbench's
`STRATEGY=lanes` (an internal build setting, no `dev/KNOBS.toml` switch; the planner rule is
unchanged in every arm):

- **L0**: main (2784f5d), X7's lanes.
- **La**, (a) only: the lanes driver prepares the stages' arguments once per call (X20's
  `run_flat`), and a body that is one lanes leaf between the caller's arrays runs each run
  of the innermost batch loop as one call (C rows per unit, the last unit padded), entered
  from `execute_core` with the two addresses (X20's store-forwarding fix).
- **Lb**, (a) + (b): where C > 1 and the target builds a register from C separate complex
  values cheaply (`spread<T>` in the vocabulary: `avx2` and `avx512` in double; NEON in
  float, where it costs two 64-bit loads), the lanes' first pass, last pass, leaf and the
  real pairings read and write the rows themselves (`load_spread`, `store_spread`), and the
  gather and scatter remain only where a split run ends the chain (three or more passes on
  x86, from about 128 points) and for float on x86 (C × C transposes as before).

Reference arms: **C**, main with `STRATEGY=chain` (the batched-leaf chain, arm A of every
run, so every lanes arm has a paired interval against it); FFTW 3.3.11 (the machine's SIMD
build, MEASURE); oneMKL 2026.1 on the Xeon. Competitors time double only.

## Method

- Cases: `bench/catalogues/x7.toml`, tier `full` (c2c 8–256 points at batches 64 and 1024,
  8, 16, 64 and 128 at 4096; r2c and c2r 16–256 at 64 and 1024), double and float, out of
  place. Float shows (a) alone on x86 (Lb's float route there is La's).
- Machines: Xeon Gold 6240 (`avx512`, and `avx2` forced; GCC 14; Slurm `-c 8`, shared);
  EPYC 7763 (`avx2`; GCC 16; DMOG `-c 8 --hint=nomultithread`, shared). One thread,
  Release. The Mac (`neon_fcma`) only as a sanity check that nothing regressed for forced
  lanes in double (C = 1, which (b) does not touch).
- Metric: the median over rounds of the paired ratio arm ÷ C per case with its 95 %
  interval, and the run's A/A band; La ÷ L0 and Lb ÷ La from the same run's medians.
- Decision rule, written before the runs:
  1. Prerequisite: the merge gate green on the Mac and `build` + `sanitize` on the Xeon
     (both `avx2` and `avx512` run) for the Lb commit.
  2. (a) is kept if, on every machine and target, no x7 case has La ÷ L0 above 1 + band
     (medians) and the geometric mean of La ÷ L0 over the x7 cases is at most 1.00;
     otherwise it is reverted and (b) is re-timed without it.
  3. (b) is kept if, on every machine and target, no double case has Lb ÷ La above 1 + band
     (medians) and the geometric mean of Lb ÷ La over the double cases is below 0.97;
     otherwise it is reverted.
  4. Thresholds in double, X7's rule on the kept arm against C: per target, `lanes_up_to` is
     the largest catalogue length L such that no double case of length at most L (c2c,
     r2c, c2r, every batch) is slower with the lanes (interval wholly above 1 + band) and
     the geometric mean of lanes ÷ C over those cases is at most 0.97; the minimum over the
     machines sharing a target (`avx2`: the Xeon forced and the EPYC); 0 if no length
     qualifies. Float thresholds change only if rule 2 reverts (a) or a float case of La is
     slower than L0.
  5. Task 2.2's acceptance for double c2c on AVX-512 is met if the kept lanes arm's median
     time is at most FFTW's and MKL's at every c2c case of 8, 12 and 16 points.

Commands (`F="uv run --frozen python bench/fbench.py"`, remote runs through
`scripts/remote/fbench_remote.py submit <profile> <arms> -- <fbench options>`; L0 and C are main
2784f5d, La is 4fe481d and Lb abb872b; the Xeon profile is the one with MKL that
X7 used):

```bash
# Xeon, AVX-512 and AVX2 forced in one run (GCC 14)
submit xeonbase8 C=fftn:2784f5d+STRATEGY=chain L0=fftn:2784f5d+STRATEGY=lanes \
  La=fftn:4fe481d+STRATEGY=lanes Lb=fftn:abb872b+STRATEGY=lanes F=fftw:2784f5d M=mkl:2784f5d \
  C2=fftn:2784f5d+ISA=avx2+STRATEGY=chain L02=fftn:2784f5d+ISA=avx2+STRATEGY=lanes \
  La2=fftn:4fe481d+ISA=avx2+STRATEGY=lanes Lb2=fftn:abb872b+ISA=avx2+STRATEGY=lanes -- \
  --catalogue bench/catalogues/x7.toml --tier full --min-rounds 8
# EPYC, AVX2 (GCC 16)
submit epyc C=fftn:2784f5d+STRATEGY=chain L0=fftn:2784f5d+STRATEGY=lanes \
  La=fftn:4fe481d+STRATEGY=lanes Lb=fftn:abb872b+STRATEGY=lanes F=fftw:2784f5d -- \
  --catalogue bench/catalogues/x7.toml --tier full --min-rounds 8
# Mac sanity (double, forced lanes; a quick run, not a decision run)
$F run L0=fftn:2784f5d+STRATEGY=lanes Lb=fftn:abb872b+STRATEGY=lanes \
  --catalogue bench/catalogues/x7.toml --tier full --filter 'op$' --min-rounds 3 --max-rounds 3
```

## Results

Correctness before timing: the Mac's merge gate passes on abb872b (Homebrew Clang 23;
`neon_fcma`, `neon`, scalar), and so does a GCC 16 `-Werror` build. The x86 paths were
also built on the Mac for x86-64 (Apple Clang, `-arch x86_64`, FFTW oracle off) and the
whole suite run under Rosetta with `ROSETTA_ADVERTISE_AVX=1`, which runs `avx2` (not
`avx512`): 1096 tests pass. Breaking `store_spread` on purpose on `avx2` and on NEON made
the lanes tests fail, so both new paths run. `avx512` and GCC on x86 still need the Xeon.

Mac sanity, run 20261115-072430-b7db (M5 Max, `neon_fcma`, forced lanes, c2c 8, 16 and
64 points at batches 64 and 1024; 3 rounds, all under load, band ±3.55 %; not a decision
run): La ÷ L0 0.95–1.05, geometric mean 0.985, no verdicts; Lb ÷ L0 in double 0.92–1.00
(C = 1, which (b) leaves alone; one case faster, 16 × 1024 at 0.92) and in float
0.56–0.73 (C = 2, where the gather and scatter go). Nothing got slower.

The Xeon and EPYC runs are pending (commands above).

## Verdict

Pending.
