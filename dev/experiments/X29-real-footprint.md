# X29: real rows in the chain-or-grid rule, and the grid's pairing

- **Status:** done
- **Owner:** x29 session, 2026-11-15
- **Phase / task:** 1–2 (PLAN lead 5; follows task 2.1, X6, X12b, X25)

## Question

Should a real row (r2c, c2r) switch from the chain to the grid at a different size than the
c2c row of its half length, and is the separate pairing pass after (r2c) or before (c2r) the
grid worth fusing into the grid's last (first) phase? The answer sets the real rows'
chain-or-grid rule (`plan_grid` in `src/core/plan_core.cpp`) and decides whether lead 5
needs grid work.

## Hypothesis

FFTN's real rows lose most at the long lengths (FFTN ÷ FFTW, tracking tier: r2c and c2r of
65536 and 2²⁰ 1.41–1.62 on the EPYC, 1.36–1.46 on the Xeon, 1.30–1.39 on the Mac; runs
20261115-101319-f98f, -114105-ea43, -094953-38c5). X25 found r2c 65536 on the Xeon bound by
memory: its half-length chain of 32768 points (512 KiB, exactly the Xeon's `grid_min_bytes`
of L2/2, so a chain) runs at IPC 0.84 with 3.4 times FFTW's L2 misses. The rule sizes the
half-length complex transform as if it were a c2c row and ignores what the real row adds:
the pairing pass reads both ends of the half spectrum (two streams) after the chain has
left it in L2 or beyond, and the in-place real grid (and every c2r grid, which runs in place
on the output) carries the workspace block as well. Expected: on the x86 machines, whose
private L2 lets a chain stream well up to L2/2 (X12b), real rows near that limit (r2c and
c2r 32768–65536) are faster as grids, i.e. the switch should come at a smaller half length
for real rows; on the Mac (grid beyond L1 already) nothing moves. The pairing pass at
2²⁰ is one more pass over memory on top of the grid's three, so fusing it into the grid's
last phase could save up to a quarter of the time there; X6 measured the Mac's real
overhead beyond L1 at 2–16 %, which bounds what fusion can gain on the Mac.

## Change under test

Arm (a), the rule: a real row is a grid when k × (its half-length complex bytes) exceed
`grid_min_bytes` (main: k = 1), with k from the screen below; nothing else changes (the
grid's split, the pairing's route and fusion limit stay). Arm (b), only if the screen says
it can pay (decision rule 3): the pairing fused into the r2c grid's last phase and the
c2r grid's first phase (blocks of columns l and L − l together), no separate pass.

Results change bits for the real rows whose plan changes (a different factorisation); both
routes are tested against the derived tolerances (the grid and chain suites force each).

## Method

- Cases: `bench/catalogues/x29.toml`: r2c and c2r rows of 4096, 8192, 16384, 32768, 65536,
  131072, 2²⁰, 1000 and 46080 points, in and out of place, and the c2c rows of half those
  lengths (out of place), one transform, double, hot. Collateral: the standard tracking tier.
- Machines: Xeon Gold 6240 (`avx512`, GCC 14.4, FFTW 3.3.11 with AVX-512, Slurm `-c 8`
  shared); EPYC 7763 (`avx2`, GCC 16.2, FFTW 3.3.11 with AVX2, DMOG `-c 8` shared); Apple
  M5 Max (`neon_fcma`, Homebrew Clang, the NEON FFTW, shared). fbench interleaved rounds
  with the A/A control, `--serial` on the Slurm machines.
- Step 1, screen: arms A = `fftn` (main, the rules), C = `fftn+STRATEGY=chain`, G =
  `fftn+STRATEGY=grid`, F = `fftw`. Per real case, C ÷ G says which side of the switch the
  row belongs on; time(real n) ÷ time(c2c n/2) − 1 under G is the real overhead of the
  grid route (the pairing pass and the reals' reading).
- Step 2, arm (a): B = the rule's commit against A = main, x29 catalogue and the tracking
  tier, on each machine.
- Metric: median paired ratios with 95 % intervals; weighted geomeans over the cases whose
  plan an arm changes; the A/A band.
- Decision rules, written before running:
  1. k is the value of {1, 2, 4} (the switch at L2/2, L2/4 and L2/8 of half-length bytes on
     x86) whose implied choice of C or G per real case has the lowest geomean ÷ A in the
     screen, per machine; if the best k differs between machines with a private L2 and the
     Mac, k is set per that profile property (as X12b's thresholds are).
  2. Arm (a) is adopted on a machine-target if over the real cases whose plan it changes the
     weighted geomean B ÷ A is below 1 − band with its interval below 1, no such case is
     slower than 1 + max(band, 3 %), and no tracking case is slower than 1 + max(band, 3 %)
     (re-run once if a case is just past it on a loaded machine). A win on one target and a
     loss on another is adopted per target; release toolchains decide (GCC on x86 Linux,
     Clang on macOS).
  3. Arm (b) is built only if the grid route's real overhead (step 1) is at least 15 % at
     65536–2²⁰ on at least two machines, so that fusion could plausibly gain 7 % or more;
     otherwise the file records why not. If built, it is judged by rule 2 over the grid's
     real rows.

### Arms added after the screen (written before their confirmation runs)

The screen's chain arm beat the grid far outside the real rows' question: on the Mac for
every 46080-point row (0.62–0.67 of the grid's time), and on the EPYC for every long row,
c2c included (0.53–0.74, to 2¹⁹ points). The rule's thresholds were set on powers of two
(X12, X12b) and never tried chains beyond L2 on x86. Two arms follow, each a rule:

- Arm (c), shared L2 (Apple): a row whose length is not a power of two stays a chain when
  the power-of-two part of its length, in bytes, is at most L1/8, or the row is at most
  2 · L1 (`chain_part_bytes`, `chain_mixed_bytes`; fbench `CHAIN_PART_BYTES`,
  `CHAIN_MIXED_BYTES`). From a Mac screen of c2c rows 2^k × odd (3 to 3125) and large rows.
- Arm (d), private L2 (x86): `grid_min_bytes` raised from L2/2 to the value the long-row
  screens (c2c and real rows to 4 Mi points, chain against grid, EPYC and Xeon) support.
- Decision rule for (c) and (d): rule 2 above over the rows whose plan the arm changes, on
  the x29 catalogue and the standard tracking tier, per machine-target.

## Results

Ratios are medians of paired per-round ratios; "C ÷ A" is the chain forced against the
rules (A, main), "G ÷ A" the grid forced. FFTN ÷ FFTW is A's time over FFTW's.

### Step 1, the screen

**Mac** (run 20261115-191654-64aa; 80 rounds, 78 loaded; band ±1.56 %). The rules' choice
is right for every power-of-two real row: the chain at 4096–16384 (the grid 1.07–1.77
there), the grid from 32768 (the chain 1.34–1.70). The exception is 46080: the chain took
0.61–0.67 of the grid's time (c2c 23040 likewise 0.58), see arm (c). Real overhead of the
grid route at 32768–2²⁰ (time(real n) ÷ time(c2c n/2) − 1): r2c 0.07–0.13, c2r in place
0.10–0.12 (2²⁰ −0.07), c2r out of place 0.22–0.26 in two phases and 0.06 at 2²⁰.

**EPYC** (run 20261115-201639-d66e; 38 rounds, 29 under load, other jobs on 66 of the
node's cores; band ±1.00 %). The chain beat the grid on **every** long row: c2c 32768
0.72, 65536 0.72, 2¹⁹ 0.53, 23040 0.64; r2c and c2r 65536 0.73–0.74, 131072 0.65–0.70,
2²⁰ 0.53–0.58, 46080 0.67. The grid's own times match the baseline run's (c2c 65536 6.7
against 6.9 ns per point), so this is not the load. With the chain FFTN ÷ FFTW is
0.58–1.10 on these rows instead of 1.07–1.68. The real rows at 32768 (half-length
256 KiB, the EPYC's `grid_min_bytes`) are already chains, and the chain is as fast as the
grid there (0.98–1.00). Real overhead with the rules' route at 65536–2²⁰: r2c 0.10–0.19,
c2r out of place 0.16–0.24.

**Xeon** (run 20261115-220233-2922; 53 rounds, 8 under load; band ±1.22 %; MKL arm as
well). Unlike the EPYC, the Xeon's thresholds are about right: beyond L2/2 the chain lost
(c2c 65536 1.23, 2¹⁹ 1.17; real 131072 1.09–1.22 except c2r out of place 0.98; 2²⁰
1.12–1.57), and at 32768 (512 KiB, a chain) the grid was 0.96. Arm (a)'s k = 2 would make
the real rows of 65536 (half-length 512 KiB) grids: r2c 0.941 out of place and 0.975 in
place, c2r 1.153 out of place and 1.000 in place, geomean 1.015: no gain. X25's r2c 65536
gap (FFTN ÷ FFTW 1.39 here) is not the threshold: the grid closes 6 % of it. Real overhead
with the rules' route at 65536–2²⁰: r2c 0.14–0.26, c2r in place 0.10–0.27, out of place
0.24–0.42 (131072: 0.42), the largest of the three machines.

So the hypothesis of arm (a) is refuted on all three machines: real rows do not need the
grid earlier than c2c rows of their half length (Xeon: no gain; EPYC: every row needs it
far later; Mac: the rules are right for powers of two). Arm (a) was not built; the screen
led to arms (c) and (d). Decision rule 3 holds for the c2r pairing (overhead out of place
0.16–0.42 on all three machines) and not for r2c on the Mac (0.07–0.13).

### Arm (b): the c2r pairing in phase 1's gather

Commit 4817ed2 (every c2r grid), then 560eeba (out of place in two phases only). Mac,
run 20261115-195518-1b08 (80 rounds, 77 loaded; band ±1.05 %), 4817ed2 ÷ main: c2r out of
place 0.921–0.944 at 32768, 65536, 131072 and 46080 (all two-phase grids); in place
1.045–1.056; 2²⁰ (three phases) 0.999 out of place, 1.022 in place; r2c and chains
unchanged (0.98–1.01). Where the workspace block stays (in place, three phases) the
gather's second read of the input costs more than the separate in-place pass, so 560eeba
keeps the pass there. The r2c pairing is not fused (DESIGN 9.1: it needs both mirror
blocks' final values in one iteration of the last phase).

### Arm (c): lengths that are not powers of two on a shared L2

Mac screens with A, C and FFTW: c2c rows 2^k × odd (run 20261115-202349-6ad1, 80 rounds,
band ±1.00 %) and large rows (20261115-204941-894b, band ±1.00 %). By the power-of-two
part p of the length: p ≤ 1024 points, the chain took 0.49–0.72 of the grid's time in all
21 rows tried, from 144 KiB to 62.5 MiB (4 Mi points), and beat FFTW (0.55–0.91); p =
2048–8192 was mixed (0.60–1.71); p ≥ 16384 the chain lost (1.06–1.78) except 327680
(0.88). Every row up to 2 · L1 (256 KiB) whose length is not a power of two won
as a chain (p up to 4096; 0.56–0.63), against 1.70 for the power of two 16384 there. Float (run
20261115-215105-598b, band ±1.00 %): with the rule in bytes (p ≤ 2048 in float) every
changed row won (0.53–0.84; geomean of the arm 0.80 of main).

Arm commit 9752672 (with b2) against main, standard tracking tier (run
20261115-210824-1a6a, 80 rounds, 55 loaded, band ±1.00 %): c2c 46080 0.630, 100000 0.600,
c2r 65536 0.925; no other case outside ±2 %; geomean 0.983. FFTN ÷ FFTW at 46080 and
100000 goes from 1.20–1.23 to 0.74–0.76. Re-run against main at be593d1 (after X26, X23b
and X28; branch cbe9386, run 20261116-012450-ab38, 80 rounds, 71 loaded, band ±1.08 %):
46080 0.643, 100000 0.617, c2r 65536 0.937, r2c 2048 × 36 0.970; slowest case 1.014
(c2c 17, no evidence); geomean 0.984. Adopted on the Mac by rule 2.

### Arms (d) and (d2): chains on x86 where each CPU has 2 MiB of L3

EPYC screens with A, C and FFTW: the screen above and c2c rows 2^k × odd (run
20261115-221128-8d85; 23 rounds, every one with the node full of other jobs, 93 of its
cores; band ±1.09 %): the chain took 0.51–0.81 of the grid's time on all 31 rows from
360 KiB to 16 MiB that the grid took, powers of two included (32768 0.73, 65536 0.72, 2¹⁹
0.57, 983040 0.72, 1024000 0.76; 100000 0.51). The rows that are no grid (odd lengths) are
unchanged (0.99). Xeon: the screen above (the chain lost beyond L2/2), so the rule keys on
the L3 per CPU (EPYC 4 MiB, 32 MiB for 8 CPUs; Xeon 0.69 MiB, 24.75 MiB for 36), with the
switch at half of L3, the largest row measured faster as a chain (16 MiB). That arm is
commit 6aba8b1.

Confirmation against main be593d1 (branch cbe9386 = arm d with b2 and c; run
20261116-040205-7105; 23 rounds, all with 64–96 of the node's cores busy; band ±1.07 %).
Weighted geomean B ÷ A 0.952, but the rule fails: the chains of the power-of-two rows of
1–2 MiB were **bimodal**, 0.7 of the grid's time in some processes and up to 2.5 times in
others (c2c 65536: 4.6–18.4 ns per point against the grid's 5.9–8.1; median ratio 1.46;
c2c 131072 1.10; r2c and c2r 131072 1.61–1.84; c2r 262144 1.69, while the same plan in arm
C took 0.92). The screens had not shown the slow mode at all (65536: 0.72 twice). Likely
cause: leg distances that are multiples of the L2's way size (64 KiB) collide in the
physically indexed L2 when the pages lie contiguous, which depends on the process. Larger
powers of two were steady (c2c 2¹⁹ 0.67, 2²⁰ 0.88; real 2²⁰ 0.64–0.69, 2²¹ 0.79–0.81), as
were all lengths whose power-of-two part is at most 1024 points: c2c 23040 0.75, 46080
0.74, 100000 0.64, 10⁶ 0.69; r2c and c2r 46080 0.75–0.79, 2·10⁶ 0.71–0.85 (largest
process-to-process spread of B within ±8 %).

Arm (d2), commit a3e6186: `grid_min_bytes` stays L2/2 on x86; with 2 MiB of L3 per CPU,
lengths whose power-of-two part is at most 16 KiB (1024 points in double) stay chains up to
half of L3 (`chain_part_bytes`, `chain_part_limit_bytes`). In the confirmation run its
plans are arm B's for exactly the steady rows above and arm A's for the rest. Tracking
tier on the EPYC, branch 5ef9932 against main 63943cc (run 20261116-055358-7335; 63
rounds, all with 64–97 other cores busy; band ±1.00 %): c2c 46080 0.750, 100000 0.656;
FFTN ÷ FFTW there 1.52 → 1.14 and 1.52 → 1.00; geomean 0.989; and **c2r 65536 out of
place 1.111**: arm b2's paired gather (65536 is a two-phase grid on the EPYC). Commit
8ba17c8 therefore keeps the paired gather to shared-L2 machines (`c2r_paired_gather`),
which restores main's plan there; nothing else outside ±3 % (largest 1.010, c2c 256, no
evidence).

Xeon: arms c and d2 change no plan there (0.7 MiB of L3 per CPU), and after 8ba17c8 b2
none either, so no Xeon confirmation run was needed (job 15959, cancelled).

## Verdict

- **Arm (a), the real footprint in the rule: rejected.** No machine gains from switching
  real rows to the grid earlier than c2c rows of their half length (Xeon: k = 2 geomean
  1.015; EPYC and Mac: the grid comes too early already, or the rule is right). X25's
  r2c 65536 gap on the Xeon is not the threshold. Not built.
- **Arm (b2), the c2r pairing in phase 1's gather: adopted on Apple** for c2r rows out of
  place in two-phase grids (commits 560eeba, 8ba17c8; Mac 0.92–0.94; the separate route
  kept in place and in three phases, where the fused gather lost 2–6 %). On the EPYC it
  lost (c2r 65536 1.11), so private-L2 machines keep the separate pass. The r2c pairing
  stays a separate pass (DESIGN 9.1 says why).
- **Arm (c), Apple: lengths that are not powers of two stay chains: adopted** (9752672;
  Mac tracking 0.984 of main at 07c9e8c, 46080 0.65, 100000 0.62, nothing slower).
- **Arm (d), x86 with 2 MiB of L3 per CPU, every row a chain to half of L3: rejected**
  by rule 2 (power-of-two chains of 1–2 MiB bimodal, up to 1.84 of main).
- **Arm (d2), the same for lengths with a power-of-two part ≤ 1024 points: adopted on
  the EPYC by its confirmation-run evidence** (a3e6186; 0.64–0.85 on the rows it changes);
  the Xeon (0.7 MiB of L3 per CPU) unchanged.

DESIGN 8.5, 9.1 and the decision log record the rules; LESSONS 3.11–3.13 the lessons.
Follow-ups for lead 1: why power-of-two chains are bimodal on Zen 3 at 1–2 MiB (huge pages?)
and whether rows of 4–16 MiB, which were steady (0.64–0.88), can take chains; whether
mixed lengths with a power-of-two part of 2048–8192 points can (mixed on the Mac, good on
the EPYC in the screens); fusing the r2c pairing into the grid's last phase (block pairs).
