# X61: strided axes of an array as lanes over adjacent transforms

- **Status:** done (adopted, restricted with a team running the chain, on Apple's and the Xeon 6240's classes; off elsewhere)
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
count: a block that fits L1 still meets cold memory when many blocks follow one another. The
final condition, from the Xeon's screen, keeps the chain within L1 only where C = 1 or every
axis is of a leaf length of 16 points or fewer: on the Xeon the route gains within L1 too at
longer axes, 32 x 16 0.23, 64 x 16 0.40, 256 x 16 0.51.) An arm `Z` (`ADJACENT_MIN_BYTES=0`,
the route wherever it fits) stays in the judging runs to show what the condition costs.

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

Intermediate runs at `e296850` (the condition counted one block; Mac run
20261119-022026-b8fd: 0.492, its middle axes 0.81) and `a927f5d` (all the transforms, L1 for
every target; Mac middle axes alone, run 20261119-025917-b00a: 0.613; Xeon 20261119-024858-37a3:
0.425, no case slower, but its in-L1 columns back on the chain at 2–7 times FFTW's and
oneMKL's time) led to the final condition.

**Judging runs** at `b95d5fd` (the final rule), arms A (`COLUMNS=chain`), B, Z
(`ADJACENT_MIN_BYTES=0`), FFTW MEASURE, oneMKL on the Xeon; one thread; release toolchains (GCC
16 on the EPYC, GCC 14 on the Xeon, Homebrew Clang on the Mac). Geometric means over the 86 c2c
cases (group: columns 37, planes 33, volumes 8, middle axes 8):

| Machine (run, A/A band) | B ÷ A | columns | planes | volumes | middle | B ÷ FFTW | B ÷ oneMKL | worst B ÷ A |
|---|---|---|---|---|---|---|---|---|
| M5 Max `neon_fcma` (20261119-034655-4b0d, ±1.00 %, loaded) | 0.484 | 0.567 | 0.393 | 0.423 | 0.621 | 0.861 | – | 1.01 |
| EPYC 7763 `avx2` (20261119-035613-7c79, ±1.17 %) | 0.287 | 0.373 | 0.208 | 0.240 | 0.388 | 0.804 | – | 1.00 |
| Xeon 6240 `avx512` (20261119-040543-716a, ±4.33 %, loaded) | 0.367 | 0.438 | 0.325 | 0.288 | 0.346 | 0.760 | 0.968 | 1.01 |

No c2c case is slower than A beyond its band on any machine. The real cases are unchanged (B ÷ A
1.00, 0.998–1.000). Against FFTW the losses that remain are small interleaved planes and volumes
(Mac 32³ x 8 1.74–1.88, 16³ x 32 1.57–1.68, 128² x 8 1.38–1.45; EPYC 16² x 8 1.29–1.72, 64 x 16
1.38–1.48; Xeon 16² x 8 1.25–1.54, 16³ x 8 1.13–1.43) and the real transforms along a strided
halved axis (1.17–1.49), which keep the chain. oneMKL stays far ahead on short columns within L1
(Xeon 16 x 16 2.5–2.9, 8 x 16 1.7–1.9) and on middle axes (1.28). Arm Z (the route wherever it
fits) against B: on the EPYC the leaf-length cases within L1 would gain (B ÷ Z 1.20–1.58 on 16
x 16, 8 x 16, 8 x 256, 16² x 8), on the Xeon the comparison is within its band for most and the
screen's 8 x 16 lost 1.20 with the route, on the Mac Z ÷ B is 1.0 but for 32² x 8 out of place
(0.72) and 32 x 256 (0.87).

Collateral (EPYC, the commits before and after the route): the standard catalogue's tracking
tier, 61 cases, no evidence of change (geomean 0.996; run 20261119-020114-b31d, ±1.77 %); `x7`
(the lanes over rows, whose driver path changed) 1.004 with 87 of 88 cases without evidence and
float 8 x 64 at 1.045 (run 20261119-031709-6855, ±1.00 %, after the driver kept stages inline;
1.037 before it). Its plan is unchanged: every arm's note in both runs reads "axis 0 (n = 8): on
the lanes' tile, radices 8", the lanes over rows that `lanes_up_to` gives float on `avx2`. The
interleaved route does not apply to contiguous rows. What changed on its path is the driver's
lanes loop: `lane_tail` holds records, and stages are found by variant. So this is code placement
and a few instructions per block in the lanes driver, not a change of plan; it is recorded as
that, not as a verdict on the route.

**Threaded (8 lanes)**, A and B with `THREADS=8`, FFTW at 8 threads. A block of adjacent
transforms is one task, which a team cannot split:
- `b95d5fd`: the rule failed on planes and volumes of 8–64 transforms. EPYC up to 1.74 of A
  (20261119-044638-2ed8, ±1.67 %); Xeon up to 2.4, with 1024 x 256 at 1.2–1.6
  (20261119-045607-bb58, ±3.68 %).
- `365a96a`: a team ran the chain below 128 transforms, with the lanes on the chain's radices.
  The M5 Max's one thread lost 1.15 on 256 x 64 (run 20261119-053243-2402), and the Xeon's
  1024 x 256 was untouched.
- `60a110c`: on the x86 profiles (`adjacent_team_chain`) every team ran the chain, the lanes on
  the chain's radices. 8 lanes passed: EPYC B ÷ A 0.927, worst 1.01 (20261119-060449-88fb);
  Xeon 0.854, worst 1.04 within ±4.61 % (20261119-055855-0364). One thread on the EPYC lost
  0.29 → 0.40, with 1024 x 16 at 1.25 and 256 x 16 at 1.21 (20261119-064525-3299). The Xeon's
  one thread was fine: 0.362 (20261119-063911-f5cc).
- `c33d576`: a team ran the chain only where the radices agree, and the lanes elsewhere. The rule
  failed at 8 lanes again: EPYC up to 1.52 (20261119-073845-683d), Xeon up to 2.1
  (20261119-074020-cde5).
- `fbd01d8` (final): on the x86 profiles the route is taken only where `column_radices` are the
  chain's on every axis, and a team runs the chain there with the same bits; elsewhere the plan
  is the chain. Judging runs x61-epyc-1t, x61-xeon-1t, x61-epyc-8t, x61-xeon-8t:

  | Run (A/A band) | B ÷ A | worst B ÷ A |
  |---|---|---|
  | EPYC, one thread (20261119-082911-ac82, ±1.41 %) | 0.946 | 1.01 |
  | Xeon, one thread (20261119-082934-65c7, ±1.92 %) | 0.774 | 1.02 |
  | EPYC, 8 lanes (20261119-090928-c329, ±2.56 %) | 0.966 | 1.03 |
  | Xeon, 8 lanes (20261119-090955-8ab7, ±2.65 %) | 0.918 | 1.02 |

  Few axes' `column_radices` match the chain's, so most x86 plans are the chain again. Against
  FFTW the x86 one-thread results go back to 2.6 (EPYC) and 1.5 (Xeon) of its time.

FFTN's threaded interleaved planes and volumes stay 2–7 times FFTW's threaded time in both arms
(the chain route with few transforms), a gap this experiment does not address.

## Verdict

Adopted on the M5 Max: 0.48 of the chain's time at one thread, no case slower; the x86
condition does not apply there, and its 8 lanes were not measured.

On the EPYC 7763 and the Xeon 6240 the unrestricted route met the one-thread rule (0.29, 0.37)
but failed the 8-lane clause (up to 1.7 and 2.4). The restricted form is never slower at either
thread count, but it keeps only part of the gain: 0.95 and 0.77 at one thread.

After the second reader (AGENTS section 6: unmeasured families keep their old plans), the route
is off unless a profile is a measured class that passed. Apple's class takes the restricted form
too: its 8 lanes were never measured, and the unrestricted route lost at 8 lanes on both x86
machines. M5 Max at `9da996f`: B ÷ A 0.745 over the 86 c2c cases, no case slower (worst 1.01),
÷ FFTW 1.35 (run 20261119-112501-c289, ±1.00 %), against 0.484 unrestricted.

Per target, as AGENTS section 6 asks:
- The Xeon 6240 passes (0.774 ≤ 0.85; 8 lanes 0.918) and keeps the restricted route.
- The EPYC 7763 misses the bar (0.946), so the route is off on its class (x86-64, a private L2
  under 1 MiB with a large L3: `machine_profile::adjacent_lanes`, keyed like X59's `zen3_like`).
  The EPYC therefore runs the old route on these shapes.

Letting a team share a block's lines would bring back the one-thread 0.29 and 0.37 without the
8-lane loss, and would let the EPYC take the route. That follow-up is the main open item. DESIGN 9.3 describes the route; LESSONS
3.29 the lesson. Follow-ups:
- let a team share a block's lines (work-graph phases for lanes programs), so that teams can run
  the lanes and x86 can keep column_radices everywhere;
- a per-target condition within L1 (the EPYC's leaf-length cases, arm Z);
- a real lane path for a strided halved axis;
- small interleaved planes against FFTW;
- oneMKL's short columns;
- the x7 float case.
