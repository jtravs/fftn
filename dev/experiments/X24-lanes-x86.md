# X24: cheaper lanes on x86 (short-row batches in double)

- **Status:** done (both changes rejected; `lanes_up_to` in double stays 0 on x86)
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
the lanes tests fail, so both new paths run. `avx512` and GCC on x86 still need the Xeon. Rule 1 was then
taken from CI instead of the Xeon (the maintainer's change of gate on 2026-11-15): the
SDE emulation workflow (`avx2` and `avx512`) passed on 7507971, main + (b) without (a)
(GitHub run 43700769747), and GCC 14 and 16 built every arm for the timing runs (Release,
tests off). No native sanitizer run was made on x86.

Mac sanity, run 20261115-072430-b7db (M5 Max, `neon_fcma`, forced lanes, c2c 8, 16 and
64 points at batches 64 and 1024; 3 rounds, all under load, band ±3.55 %; not a decision
run): La ÷ L0 0.95–1.05, geometric mean 0.985, no verdicts; Lb ÷ L0 in double 0.92–1.00
(C = 1, which (b) leaves alone; one case faster, 16 × 1024 at 0.92) and in float
0.56–0.73 (C = 2, where the gather and scatter go). Nothing got slower.

Ratios below are fbench's paired medians against the chain C (median over rounds of each
round's arm ÷ C); La ÷ L0 and Lb ÷ La are, as the rule says, ratios of those medians from
the same run (two paired comparisons, not one), so each is also given as the direct paired
ratio with its interval where it decides something. The analysis is a short script over
the runs' records; the numbers can be read with `fbench show <run>`.

### EPYC 7763 (`avx2`, GCC 16, DMOG, `-c 8`, shared)

Run 20261115-082420-9707 (Slurm 509331; arms C, L0, La, Lb, F; 88 cases, 80 rounds, none
under load; band ±1.00 %).

- **Rule 2, (a):** La ÷ L0 geometric mean 0.984 over the 88 cases (double 0.983, float
  0.986), but four cases are above 1 + band: c2c-16-b64 float 1.027 (direct paired 1.024
  [1.015, 1.042]), r2c-16-b64 1.020 (1.013 [1.005, 1.023]), c2c-8-b64 1.019 (1.020
  [1.017, 1.024]) and c2r-16-b64 float 1.014 (1.010 [1.003, 1.016]). The failures are
  0.4–1.7 % beyond the band, but three of the four direct intervals clear it, so they are
  not a quirk of dividing two medians: all four are batch 64 at 8–16 points, the smallest
  calls (one block or two), where a fixed cost per call shows most; the other 84 cases are
  0.96–1.01 (the gains are largest for r2c and c2r, 0.96–0.98). **(a) is reverted**, and
  (b) is re-timed without it.
- **Rule 3 with (a), for the record:** Lb ÷ La in double geometric mean 0.873, max 1.008
  (c2c-128-b4096): it would have passed.
- **Against C:** L0 1.08–1.79 at c2c 8–256 and 0.78–1.07 for r2c and c2r; La about the
  same; Lb 1.09–1.28 at c2c 8 and 12 points, 0.86–0.96 at 16, 1.07–1.36 from 24 up, and
  0.72–0.95 for every r2c and c2r case. Lb ÷ F 1.16–1.25 at c2c 8–16.

The re-timing commands (Lc = arm b', a7a094a; the Xeon run also replaced the planned one,
so it carries every arm):

```bash
submit epyc C=fftn:2784f5d+STRATEGY=chain L0=fftn:2784f5d+STRATEGY=lanes \
  Lc=fftn:a7a094a+STRATEGY=lanes Lb=fftn:abb872b+STRATEGY=lanes F=fftw:2784f5d -- \
  --catalogue bench/catalogues/x7.toml --tier full --filter 'op$' --min-rounds 8
submit xeon C=fftn:2784f5d+STRATEGY=chain L0=fftn:2784f5d+STRATEGY=lanes \
  La=fftn:4fe481d+STRATEGY=lanes Lb=fftn:abb872b+STRATEGY=lanes Lc=fftn:a7a094a+STRATEGY=lanes \
  F=fftw:2784f5d M=mkl:2784f5d C2=fftn:2784f5d+ISA=avx2+STRATEGY=chain \
  L02=fftn:2784f5d+ISA=avx2+STRATEGY=lanes La2=fftn:4fe481d+ISA=avx2+STRATEGY=lanes \
  Lb2=fftn:abb872b+ISA=avx2+STRATEGY=lanes Lc2=fftn:a7a094a+ISA=avx2+STRATEGY=lanes -- \
  --catalogue bench/catalogues/x7.toml --tier full --min-rounds 8
```

Re-timed without (a): run 20261115-102607-e2c1 (Slurm 509340; arms C, L0, Lb' = a7a094a
(main + (b), named Lc in the run), Lb, F; the 44 double cases, 80 rounds, 2 under load;
band ±1.00 %).

- **Rule 3, (b) alone:** Lb' ÷ L0 geometric mean 0.880 over the double cases, max 1.006
  (c2c-128-b4096, inside the band). **(b) is kept on the EPYC.** Without (a), (b) is a
  little slower than with it where the body is a single leaf (c2c 8–16 points: Lb' ÷ C
  1.12–1.37 at 8, 1.10–1.20 at 12, 0.90–1.05 at 16, against Lb's 1.08–1.28, 1.03–1.09,
  0.86–0.97): (a)'s one call per run is worth 3–8 % there, but its cost elsewhere fails
  rule 2.
- **Rule 4:** no length qualifies: c2c-8 is slower than the chain at every batch (Lb' ÷ C
  1.12–1.37), so `avx2`'s threshold in double is 0 whatever the Xeon shows. The geometric
  mean over the cases up to 64 points is 0.997, and c2c is slower at every length but 16
  (r2c and c2r are 0.73–0.97 everywhere).

### Xeon Gold 6240 (`avx512`, and `avx2` forced; GCC 14; `-c 8`, shared)

Run 20261115-105357-ff53 (Slurm 15648, record x24c-xeon; one run with every arm: C, L0, La,
Lb, Lb' (named Lc), F, M under `avx512` and C2, L02, La2, Lb2, Lc2 under `avx2`; 88 cases,
72 rounds). The planned job (15641) could not have built its MKL arm, because the profile
had no MKL root; this run replaced it. Its rounds overlapped other Xeon jobs throughout: 68
of 72 rounds counted as under load (other processes median 10, max 18 cores on 72 CPUs).
The band was 1.5–2 % in the first 30 rounds and settled at ±1.00 % (the floor) by the end.

- **Rule 2, (a):** fails under both targets. `avx512`: La ÷ L0 geometric mean 0.983, six
  cases above 1 + band, worst c2c-8-b64 1.070 (direct paired 1.074 [1.056, 1.086]), then
  float c2c-8, r2c-16, c2r-16, c2c-12 and c2c-16 at batch 64, 1.012–1.025. `avx2`: 0.987,
  three cases, c2c-8-b64 1.029 (1.033 [1.024, 1.038]), float c2c-16-b64 1.023 and c2c-8-b64
  1.011. All batch 64 at 8–16 points, as on the EPYC.
- **Rule 3, (b) without (a):** passes under `avx2` (Lb2' ÷ L02 0.797, max 0.979) and
  **fails under `avx512`**: the geometric mean is 0.957, below 0.97, but 20 double cases
  are above 1 + band, all c2c from 24 points up (1.02–1.17: c2c-128-b1024 1.171,
  c2c-128-b4096 1.169, c2c-32-b64 1.143, c2c-64-b64 1.123, c2c-24-b64 1.095) and c2r 16–256
  (1.01–1.06). (b) wins at 8–16 points (Lb' ÷ L0 0.73–0.87 at c2c 8 and 16, 0.95–1.00
  at 12; ratios of the medians against C) and in r2c,
  where each row is a leaf or a first and last pass. It loses where the body has passes
  between the edges: there (b) replaces one gather pass of C × C transposes with first and
  last passes that build each 512-bit register from four 128-bit loads (`vinsertf64x4`
  over two `vinsertf128`) and take it apart with four 128-bit stores. Presumably that costs
  more per element on Cascade Lake than the transposes it removes, but no profile was
  taken. With (a) the picture is the same (Lb ÷ La: 21 cases above the band, max 1.21).
- **Rule 4:** 0 under both targets with every arm. `avx512`: L0 ÷ C 1.23–1.45 at c2c 8,
  1.14–1.18 at 12, 1.19–1.44 at 16; Lb' 1.07–1.12, 1.08–1.18 and 0.90–1.05; even Lb (a + b)
  is slower at c2c-8-b64 (1.044) and c2c-16-b4096 (1.034). `avx2` forced: L02 ÷ C2 1.76–1.83
  at c2c 8; Lb2' 1.19–1.27.
- **Rule 5 (task 2.2, double c2c on AVX-512):** not met. With the kept arm (main's lanes,
  L0), L0 ÷ F is 1.41–1.52 at 8 points, 1.31–1.34 at 12 and 1.43–1.53 at 16, and L0 ÷ M
  1.00–1.41, 1.27–1.33 and 1.49–2.34. Neither change would have met it: the best arm, Lb,
  is 1.05–1.20 of FFTW's time at 8–16 points, and 1.09–1.76 of MKL's at 16 (MKL takes
  0.52 of the chain's time at 16 × 64 and 16 × 1024). The chain itself (which the rules
  choose) is 1.01–1.26 of FFTW's time and 0.80–1.94 of MKL's at those cases.

Outside the rules, for the follow-ups: under `avx512` main's lanes already beat the chain
on r2c and c2r of 16–64 points (L0 ÷ C 0.56–0.96 at every batch) and on c2c-24 (0.45, a
chain weakness at 24 points X7 also saw), but the rule's threshold is one length for every
kind, and c2c-8 decides it.

## Verdict

Neither change is kept, and no default changes.

- **(a) rejected (rule 2)** on all three machine-target pairs. Its geometric mean is
  0.98–0.99 everywhere, but at batch 64 of 8–16 points it is 1–7 % slower than X7's
  driver. The worst case is c2c-8-b64 under `avx512`, 1.070. Reverted (710e004).
- **(b) rejected (rule 3)**, re-timed without (a) as the rule asks (arm b', a7a094a: the
  cherry-pick of abb872b onto 2784f5d, kept on a local branch only). It passes on the
  EPYC (0.880) and on the Xeon forced to `avx2` (0.797), but under `avx512` it is
  1.02–1.17 slower at c2c 24–256 points and c2r. Reverted (6276b8b). A version for `avx2`
  only would pass rule 3 but would change nothing by default, because rule 4 still gives 0
  there.
- **Rule 4:** `lanes_up_to` stays 0 in double for `avx512` and `avx2`
  (`src/planner/rules.cpp`, the comment now cites X24). No length qualifies with any arm:
  c2c-8 is slower than the batched-leaf chain on every machine (best arm Lb ÷ C: 1.04 at
  8 × 64 on the Xeon's AVX-512, 1.08–1.28 on the EPYC). Float is unchanged: with (a)
  reverted and (b) never touching float on x86, the float lanes are X7's code.
- **Rule 5:** task 2.2's acceptance for double c2c on AVX-512 is **not met**. Main's lanes
  are 1.31–1.53 of FFTW's time at 8–16 points, and the chain the rules choose is 1.01–1.26.
  The best arm tried here (a + b) would still be 1.05–1.20.

What X24 teaches: at C = 2–4 in double, cross-row lanes cannot beat a batch of leaf rows
called once (X20) at 8 points. The leaf kernel's idle lanes cost less than the lanes'
extra loads, stores and shuffles, even with no transpose pass at all. And building wide
registers from 128-bit pieces pays on AVX2 (two pieces) but not on AVX-512 (four pieces
and a 512-bit insert) once there are inner passes to amortise a transpose over.

Follow-ups, not started: a threshold per transform kind (under `avx512` main's lanes
already take r2c and c2r of 16–64 points to 0.56–0.96 of the chain's time, and c2c-24 to
0.45); the chain's own c2c 8–16 points on AVX-512 against FFTW and MKL (the open part of
task 2.2); and the NEON float spread route, which the Mac sanity run showed at 0.56–0.73
of L0 (only a rule change could use it, since the lanes run in float only on x86).
