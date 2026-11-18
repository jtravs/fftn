# FFTN implementation and experiment plan

This is the live plan for building FFTN as designed in [DESIGN.md](DESIGN.md), under the
rules in [AGENTS.md](../AGENTS.md). Read "Status" first. Update "Status" in every working
session; update a phase when its tasks or findings change.

## Status

**As of 2026-11-16, midday.** Leads 1–5 of the one-dimensional double push have each had
their experiments (X23b, X26–X30, X26b, X28b, X29 merged; X30b in flight) and the new
one-dimensional double numbers are below; this section is also the handoff. Lessons
that generalise are in [LESSONS.md](LESSONS.md), decisions in DESIGN.md section 19, open
questions in DESIGN.md section 18.2.

### Where FFTN stands

Tracking tier, one thread, FFTN's time ÷ FFTW's (geometric mean; below 1 is faster), all
of `main` as of 2026-11-15 morning: Mac M5 Max 1.06 (FFTN 0.56 of Accelerate's time on the
cases Accelerate supports), Xeon 6240 1.15 (MKL: FFTN 1.27 of its time), EPYC 7763 1.33. Runs
20261115-094953-38c5 (Mac), 20261115-114105-ea43 (Xeon, MKL), 20261115-101319-f98f (EPYC).

**After the 1-D double push (2026-11-16, `main` 2025d51; each run times yesterday's
`main` ad6acbe in the same run):** FFTN's time ÷ FFTW's over the 46 one-dimensional double
cases, today (yesterday): Mac 0.83 (0.98), EPYC 0.94 (1.21), Xeon 0.81 (1.05); Xeon
against MKL 0.85 (1.11). Today's FFTN against yesterday's: 0.85, 0.78, 0.77. Against
FFTL.jl: Mac 0.73 (0.87), EPYC 0.63 (0.81). By group, Mac / EPYC / Xeon (÷ FFTW):
powers of two ≤ 64 0.94 / 0.94 / 0.83; 128–4096 0.84 / 1.04 / 0.94; ≥ 16384 1.19 / 1.10 /
1.03 (FFTL ahead: FFTN 1.46 / 1.40 of its time); mixed radix 0.76 / 0.96 / 0.93 (Xeon
100000 1.99, 46080 1.29: X29's chains do not apply there); primes 0.48 / 0.62 / 0.36
(4093 1.41 / 1.32; EPYC 65537 1.63 in this run against X27's 1.09); prime products 0.47 /
0.45 / 0.36; batches 0.91 / 1.03 / 0.85; in place 0.93 / 1.08 / 0.99; r2c and c2r 1.00 /
1.13 / 1.04 (65536 and 2²⁰: Mac 1.21–1.27, EPYC 1.17–1.44, Xeon 1.19–1.42). Runs
20261116-103657-af06 (Mac, FFTL; 8 of 16 rounds under load), 20261116-105105-30b3 (EPYC,
FFTL; a shared node, every round under load, band ±1.25 %), 20261116-110608-e327 (Xeon,
MKL; band ±2.17 %).
Cloud (X31, Spot VMs, GCC 14.2, 40 rounds, none under load; `main` 4ebcee5, library as
2025d51): FFTN ÷ FFTW over the 46 cases, today (before the push): Zen 5 EPYC 9B45 0.80
(1.05), Granite Rapids Xeon 6985P-C 0.88 (1.12; MKL 0.85, was 1.09), Axion 0.74 (0.86).
Worst: Zen 5 65536 1.50; Granite Rapids 100000 2.16, 46080 1.60, in place 1.26 (lead (c)
is worse there); Axion 8 points 1.24. Axion's long rows (0.75 of FFTW) show that most of the
Mac's long-row gap is FFTW being fast on Apple silicon (LESSONS 2.9). Zen 5 does not share
Zen 3's 65537 loss (0.61). Planes and volumes: Zen 5 1.63, Granite Rapids 1.48, Axion 0.78.
**Phase 1 exit check (2026-11-18, `main` 8c18dd2: after X42, X46, X47, the `avx512` branch
padding and task 4.2; standard catalogue, the 193 c2c rows with every reference; tail rule
beyond noise, DESIGN 1.3):** Mac ÷ FFTW 0.76, 2 cases beyond noise (1.0 %; 7 above a plain
1.05): 2 × 3 in place 1.37, 32 × 4096 1.10; ÷ FFTL 0.53, 4 (2.1 %): 2²⁴ 1.69, 2²³ 1.58,
1024 × 1024 1.20, 512 × 1024 1.16. EPYC ÷ FFTW 0.80, **12 (6.2 %, the one local failure)**:
short backward pairs (64 × 2 1.31, 15 × 2 1.22), 1024 (× 3 1.22, single 1.21, backward 1.20,
× 16 1.13), 64 × 16 1.14, 2048 1.11, 1018 1.10, 768 and 100 1.09, 512 1.07; ÷ FFTL 0.46, 1
(16807 1.08). Xeon ÷ FFTW 0.70, 7 (3.6 %): 131071 1.29, 25000 1.25, 160 1.16, 15 × 2 backward
1.15, 240 1.13, 2 × 3 in place 1.11, 4 1.10; ÷ MKL 0.70, 2 (1.0 %): 999983 1.41, 131071 1.32.
X46 shows on the Xeon (FFTN 0.93 of its 2026-11-17 time over the set; 16384–262144 and 65536
in every form 0.56–0.69, 32768 1.22 → 0.82 of FFTW, 65536 ÷ MKL 1.17 → 0.69) and the Mac
(8192 1.10 → 0.85, 16384 1.05 → 0.87; 2²³ 1.12 → 0.93); X42 on 3125 and 15625 everywhere.
The Mac's 48, 96, 180 and 500 lost ratio to FFTW with FFTN's time unchanged (FFTW's MEASURE
plans). Runs 20261118-004301-b779 (Mac, 44 of 46 rounds under load, band ±1.60 %),
20261118-015016-b9d2 (EPYC, every round under load, ±1.36 %), 20261118-020838-b37b (Xeon,
14 of 20 under load, ±2.48 %). X49 (pull request 63) found fbench's FFTN backend added about
16–20 instructions per call, so the tiny cases overstate FFTN; they were re-timed on `main`
4d75c48 (rows of 2–16 points and their batches, 15 × 2 and 64 × 2 backward; each case judged
against its own run's band): Mac ÷ FFTW 0.75, 3 (64 × 2 backward 1.07 joins), ÷ FFTL
unchanged (run 20261118-015403-cf4b, ±1.20 %); the EPYC and Xeon re-times are queued (they
cannot change a verdict: the EPYC's two tiny cases would leave 10, 5.2 %).
Cloud (X48, the same rows, `main` 8c18dd2): ÷ FFTW Zen 5 0.69 with 6 beyond noise (3.1 %, was
11), Axion 0.71 with 9 (4.7 %, rows of 2–8 points, 20, 65537), Granite Rapids 0.76 with 18
(9.3 %; ÷ MKL 0.73, 11, 5.7 %: 46080, 65536 in every form, 100000, 131071–262144, 32768, 160,
240, 1024; against MKL also 999983): Granite Rapids' 2 MiB L2 does not take X46's rule.

**Phase 1 exit check again (2026-11-17, `main` 509d123, before X42; the tail rule now counts
cases beyond noise, DESIGN 1.3):** Mac ÷ FFTW 0.75, 3 cases beyond noise (1.6 %; 15 above a
plain 1.05): 2 × 3 in place 1.37, 32 1.20, 2²³ 1.13; ÷ FFTL 0.53, 3 (1.6 %). EPYC ÷ FFTW 0.80,
13 (6.7 %; 17 plain): 512–2048 (1.07–1.22, seven cases), short backward pairs (64 × 2 1.30,
15 × 2 1.20), 64 × 16 1.14, 1018 1.09, 80 and 100 1.08; ÷ FFTL 0.47, 3 (3125, 15625, 16807;
X42 takes the first two). Xeon ÷ FFTW 0.73, 10 (5.2 %; 21 plain): 32768, 100000, 25000
(1.34–1.36), 16384 1.19, 131071 1.16, 160, 240 and 384 single rows (1.14–1.33, being
re-timed against 304a8c5: the run was loaded, band ±4.68 %), 3 × 3 in place, 4; ÷ MKL 0.74,
3 (1.6 %; 30 plain): 999983 1.43, 131071 1.33, 16384 1.23. Runs 20261117-143114-4e3e (Mac,
band ±2.28 %), 20261117-143500-bc4b (EPYC, ±1.00 %), 20261117-144116-aa55 (Xeon).

Tiny calls (X49, 2026-11-18): until X49 fbench's FFTN backend spent about 20 instructions a
call of its own (a function-local static), so the exit checks above overstate FFTN's time on
single rows of 2–8 points and tiny batches on every machine. With the backend corrected:
Mac 2–8 points 0.77–0.81 of FFTW, EPYC tiny cases 0.68–0.86 and Xeon 0.89–1.02 of FFTW's
cycles. Still over FFTW: the Mac's 2 × 3 in place (1.3, store-to-load forwarding through
paired accesses, LESSONS 2.12), 15 × 2 and 64 × 2 on x86 (a two-row lane leaf, not built).

EPYC mid rows (X54, 2026-11-18): radix-32/64 codelets ordered for 16 registers (bit-identical;
GCC's radix-64 unit spills 40 % less) take plans with a radix-32/64 pass to 0.975 and make
two-pass plans win: 768 = 12 × 64 ÷ FFTW 1.08 → 0.91, `avx2` float 512–4096 0.79–0.95 of
their time. 1024 = 16 × 64 would take 1024, 1024 × 3, 1024 × 16 from 1.12–1.22 to 1.01–1.03
of FFTW but loses 25 % at 1024 × 1024, so it has no override: a batch- or footprint-aware
radix choice is the next step. 2048 (1.12) and 64 × 16 (1.14) are not moved.

**Phase 1 exit check (2026-11-16, `main` 304a8c5, standard catalogue, the 201 c2c rows,
193 with every reference; DESIGN 1.3 asks for the geometric mean and at most 5 % of cases
slower than 1.05):** Mac ÷ FFTW 0.78 (target 0.85), 20 cases over 1.05 (10 %); ÷ FFTL 0.50
(0.95), 18 (9 %). EPYC ÷ FFTW 0.84 (0.90), 40 (21 %); ÷ FFTL 0.35, 8 (4 %). Xeon ÷ FFTW
0.76 (0.85), 30 (16 %); ÷ MKL 0.78 (interim 1.10, end 1.00), 33 (17 %). Every mean target
is met; the tail rule is not, on any machine against FFTW or MKL. The failing cases, by
cause: (1) primes outside Rader's range or with a large factor in p − 1 (131071 1.33–2.61,
999983 1.19–1.99, 4093 1.33–1.42, 1018 = 2 · 509 1.36–1.46, EPYC 65537 2.51); (2) single
short rows and tiny batches (Xeon 2–30 points 1.12–1.33, Mac 20 points 1.78, 2 × 3 in place
1.18–1.44, EPYC 15 × 2 and 64 × 2 backward 1.27–1.31); (3) batches of 16 points on x86
(EPYC 1.12–1.26, Xeon ÷ MKL 1.28–1.72); (4) the EPYC's 240–2048 (1.10–1.23; X38, pull request
27: a radix-20 codelet and re-fitted `avx2` double sequences take 80–20000 to 0.82–0.96 of
main, 240, 2000, 3000, 3600 under FFTW; 512–2048 stay 1.13–1.22); (5) the Xeon's
long rows (16384–262144 1.22–1.48; 20000, 25000, 46080, 100000 1.31–1.50); (6) in-place
65536 (Xeon 1.14–1.16). Runs 20261116-160815-6deb (Mac, 10 of 12 rounds under load),
20261116-161226-416e (EPYC, every round under load, band ±1.85 %), 20261116-161123-f848
(Xeon, band ±2.03 %). The old targets (X14, task 1.3, 2026-11-17): `x86_v2` ÷ FFTW built
with SSE2 only 0.957 (EPYC) and 0.984 (Xeon); `avx` ÷ FFTW with SSE2 and AVX 0.897 and 0.943;
plain `neon` ÷ FFTW's NEON build 0.886 (Mac); all at or below FFTW, the tail rule failed by
each (29–46 % of cases above 1.05) with the causes above. Runs 20261117-004757-3a1d,
20261116-234204-8175 (EPYC), 20261116-232600-139a, 20261117-002151-a927 (Xeon),
20261116-232302-42c6 (Mac).
Cloud exit check (X45, `main` 509d123, the same 193 rows): ÷ FFTW Zen 5 0.70 with 15 cases
over 1.05 (7.8 %), Granite Rapids 0.77 with 25 (13.0 %; ÷ MKL 0.75, 22, 11.4 %), Axion 0.71
with 9 (4.7 %, the tail rule met at its edge); runs 20261117-144317-7119, -144359-0a8f, -144321-6196. Beyond noise (DESIGN 1.3 since 2026-11-17): Zen 5 11 (5.7 %: 2²³, 2²⁴, 100000, 2048, 4096
and rows of 2–6 points), Granite Rapids 22 (11.4 %; ÷ MKL 17, 8.8 %: 32768–262144, 46080 and
100000 at 1.2–1.6, with FFTW fast on that VM, LESSONS 1.12; 131071, 999983), Axion 9 (4.7 %:
rows of 2–8 points 1.2–1.47, the fixed cost of a call).
FFTL.jl updated to libfftl 1ae621e (upstream main of 2026-11-16, 111 commits after 53171e8;
same build: apple-m1, x86-64-v3), same case set, `main` e8c2639, the old libfftl as a third
arm: FFTN ÷ FFTL 0.55 on the Mac (old libfftl 0.50 in the same run), 9 cases over 1.05
(5 %); EPYC 0.49 (0.35), 12 (6 %). FFTL itself takes 0.90 (Mac) and 0.72 (EPYC) of its
old time, mostly fixed cost per call: the 3-row smoke rows 0.84 / 0.52, short-row batches
0.65 / 0.31. By group, Mac / EPYC: powers of two 0.59 / 0.51; smooth 0.60 / 0.52; primes
0.46 / 0.45; batches of long rows 1.02 / 0.91; short-row batches 0.76 / 0.62. FFTL ahead:
Mac 999983 2.79, 131071 2.58, 2²³ and 2²⁴ 1.89, 1024 × 1024 1.20, 4093 1.10, 512 × 1024
1.09; EPYC 131071 1.96, 65537 1.85, 4093 1.31, 15625 1.19, 1018 1.18, in-place 65536
1.14–1.15, 3125 1.11, 16807 1.10. Runs 20261116-191543-f558 (Mac, 45 of 50 rounds under
load), 20261116-221443-d672 (EPYC, 20 of 30).
The numbers before the push, kept for reference:
One-dimensional double only (46 cases; the planes and volumes excluded): Mac 0.97, EPYC
1.21. By group, Mac / EPYC: powers of two ≤ 64 1.14 / 1.38; 128–4096 0.91 / 1.22; ≥ 16384
1.15 / 1.20; mixed radix 0.95 / 1.29 (46080 and 100000: 1.18–1.23 / 1.67–1.98); primes
0.92 / 1.20 (17: 1.57 / 1.95; 4093: 1.24 / 1.55; 65537 EPYC 1.65; 97–1009 win on both);
batches 0.94 / 1.12; r2c and c2r 1.00 / 1.25 (65536 and 2²⁰: 1.30–1.39 / 1.41–1.62).
Against FFTL.jl (libfftl 53171e8, Mac run 20261115-170717-c72c, 1-D double): FFTN 0.86
of its time overall; FFTL ahead only on rows ≥ 16384 (1.35) and 2431 (2.57); FFTN far ahead
on single rows ≤ 64 (FFTL's fixed cost per call: 0.07–0.51) and batches (0.70).
EPYC (libfftl x86-64-v3, run 20261115-171147-bc20, no round under load): FFTN 0.81 of
FFTL's time overall (FFTL 1.49 of FFTW's); the same pattern: FFTL ahead on rows ≥ 16384
(1.51; 262144 2.04, 100000 2.00, r2c and c2r 2²⁰ 2.0–2.2) and 2431 (2.93), FFTN far ahead
on rows ≤ 64 (0.12), batches (0.40) and 128–4096 (0.75).
Planes (task 4.1, X40 first pass, pull request 43): the strided axes of c2c planes and
volumes run as lanes over blocks of adjacent columns, and a loop of leaves is one call. The
37 c2c planes of the standard catalogue, FFTN ÷ FFTW before → after: Mac 1.44 → 0.88, EPYC
2.07 → 0.94, Xeon 1.49 → 0.80; ÷ MKL 1.87 → 1.00 (1024² Mac 2.1 → 1.0, EPYC 4.0 → 0.96).
Still over FFTW or MKL: tiny planes in batches of 2 (8 × 6, 5 × 7: 1.13–1.7 of FFTW), EPYC
32 × 32 (1.35–1.42), Mac 2048 × 36 (1.12–1.16), the Xeon against MKL from 16² to 1000²
(1.08–1.38). Runs 20261117-052127-ef6a (Mac), 20261117-051803-7358 (EPYC),
20261117-051743-62dd (Xeon). Volumes (64³ 3.1 / 10.9 before) take the same column phase;
task 4.2 measures them.
Planes, second part (X41, pull request 51): multi-axis programs run from their unrolled
kernel calls (small planes 0.87–0.94 of their time), real planes' complex axes through the
column phase on x86 (0.83–0.89 over the 71 real planes), Apple's long columns in narrower
blocks (2048 × 36 0.90). The 108 c2c and real planes, FFTN ÷ FFTW after (c2c / r2c / c2r):
Mac 0.85 / 0.95 / 0.95, EPYC 0.81 / 1.02 / 1.01 (runs 20261117-115356-776c,
20261117-120955-e917; Xeon in the experiment file). Still over FFTW: EPYC 32² (1.13–1.36: its
`avx2` column phase costs 1.9 times its rows), Mac 32²–64² (1.02–1.09), real planes of odd
or tiny rows; the Xeon trails MKL on c2c planes from 16² and far behind it on real planes;
threaded (8 lanes, EPYC run 20261117-122327-970c) the largest planes scale 2.3–4.7 times
against FFTW's 4–6 (c2c 1000²–2048² 1.11–1.39 of FFTW's threads, real 1000²–2048² 1.3–2.45);
on the Xeon (run 20261117-131055-a1d0) FFTN's 8 lanes take 0.81 of FFTW's threaded time but
1.16 of oneMKL's (512²–1024² 1.2–1.45).
Threaded planes and volumes (X43, X43b, task 4.2, pull request 55): a plan may hold a team
program that every team of two or more runs, the same bits with narrower column blocks and
a volume's inner axes plane at a time. EPYC at 8 lanes: c2c planes and volumes 0.887 of their
time with blocks of 8 columns (1000² 0.66, 128² 0.75), 2048 × 36 0.72 with line blocks,
volumes 0.775 plane at a time; FFTN ÷ FFTW's better plan, c2c planes 0.953 → 0.852. The Xeon
gained nothing from the blocks (0.997, 1 MiB L2) but 0.771 from plane at a time (VTX). One thread keeps every program: plane at a time gained 0.979 (EPYC),
0.934 with 97³ 1.048 (Xeon), 0.962 (Mac). Volumes against the competitors' better plans (38
cases): one thread Mac 0.80, EPYC 0.85, Xeon 0.73 of FFTW and 1.12 of oneMKL; threaded Mac
0.76, EPYC 0.90, Xeon 0.82 of FFTW, Xeon 1.27 of oneMKL. Not met case by case: 16 × 256 × 256
(a 16-point leaf whose legs share an L1 set), 512 × 64 × 64 and 64 × 64 × 512 against FFTW;
the EPYC's 8 lanes on 17 of 38 volumes (128³ 1.33, 256³ 1.10–1.12, 256 × 256 × 16 1.06–1.20);
oneMKL on the Xeon nearly everywhere. Runs in the experiment files.
After X52 (pull requests 64 and 70): leaf columns whose legs crowd an L1 set run as two lane
passes (Mac 16 × 256 × 256 ÷ FFTW 1.41 → 0.78; EPYC 16³ 0.66 of its time, no volume over FFTW
at one thread), not on `avx512`; a team claims a split task's chunks in runs (8 lanes Xeon
0.91, EPYC 0.90, Mac 0.86–0.94; Xeon volumes ÷ oneMKL at 8 lanes 1.15 → 1.06); planes are split
by their complex points where L2 is private and from 32768 points on a mesh (the Xeon's
threaded planes of 9216–16384 points, up to 2.9 times their one-thread time, now 0.35–0.67 of
it; EPYC real 96²–100² 0.72–0.90); the EPYC's one-thread column blocks widen to 1 KiB where
their tiles fit (0.933). Open: oneMKL's threaded real planes (2.5–3.9 times faster on the
Xeon), EPYC 2048 × 36 at 8 lanes (1.35 of FFTW), the Mac's 512 × 64 × 64, 64 × 64 × 512, 128³.

Planes, one thread (X44, pull request 54): columns of 32 run as one radix-32 lane pass in
double on x86 (the row's pair leaf had left them two passes): EPYC 32² ÷ FFTW 1.39 → 1.00,
32² × 16 1.31 → 0.97; Xeon 32² ÷ MKL 1.37 → 1.07. Running the column edge passes'
runs in an inner loop was rejected (EPYC 0.94 of the time but a tiny volume 1.03, Xeon
0.962, Mac 0.997). Runs 20261117-153628-5fcb and
20261117-164127-b5e3 (EPYC), 20261117-165710-e283 (Xeon, MKL). Left: the Mac's 32²–64²
(FFTW's radix-64 codelet; the generator, X42); the Xeon's c2c planes from 64² against MKL
(1.07–1.39 case by case, 0.95 geometric mean) and real planes against MKL (their real rows
trail MKL's already); the column edge kernels' per-call set-up (counters: X44).

### Tasks

As of 2026-11-17 midday.

Done: phase 0 (0.1–0.10, 0.5b, B1, B2, X15); phase 1 tasks 1.1, 1.2, 1.5 (X1), 1.6 (X3),
1.7 and 1.7b (X12, X12b), 1.3 (`x86_v2` and `avx`, X14; merging as pull request 48), with
X8, X10, X18–X39 (the 1-D double push and the tail round); phase 2 tasks 2.1, 2.3 and 2.3b
(X17), 2.4 (`planning::measure`, wisdom, P1 acceptance met on all three machines; the cost
model P3 rejected on a first pass), 2.5 (`next_fast_length`, `relative_cost`; API agreed by
the maintainer, `length_kind` in pull request 50); phase 3 tasks 3.1–3.3 with T1 and T5; phase 4 task 4.1's first part (X40:
one-thread c2c planes 0.80–0.94 of FFTW). CI: a short required set on pull requests (no
up-to-date rule), the full per-change set on main after each merge, weekly extended jobs.

Open, by phase:

- 0: the Xeon client-catalogue baseline.
- 1: the exit: every mean target met; the tail rule (beyond noise) met at 8c18dd2 on the
  Mac, the Xeon (FFTW 3.6 %, MKL 1.0 %), Zen 5 and Axion and against FFTL, not on the EPYC
  against FFTW (6.2 %: short backward pairs, 512–2048, 64 × 16, 1018, 100) nor on Granite
  Rapids (FFTW 9.3 %, MKL 5.7 %: its long rows, without X46's rule; see the exit check of
  2026-11-18 above and X48); X2 on the Mac above 1024; X4 on x86 and for edge passes; X5
  (twiddle modes); the night round of 2026-11-17: X50 (Xeon primes against MKL: 131071,
  999983), X51 (X46's loose ends; done but Granite Rapids' profile, below), X52 (volumes); float
  radix-25 overrides.
- 2: 2.2's acceptance in double c2c on AVX-512 (re-check after X37's batches of 16);
  X6 (real-input codelets), X13; the cost model's second pass and `fbench calibrate`.
- 3: T2, T3, T4, T6; 3.5 (Kokkos executor); 3.6 to be checked against its acceptance;
  threaded planes (4.1: large planes scale worse than FFTW's and oneMKL's threads, X41).
- 4: 4.1's rest after X41, X43 and X44 (the Xeon's c2c planes from 64² and real planes against
  MKL; the Mac's 32²–64², which needs a faster radix-64 codelet; the threaded gap to oneMKL on
  the Xeon); 4.2 (volumes, X43b, pull request 55: built, scratch bound tested; acceptance not
  met case by case: the 16-point outer leaf, 512 × 64 × 64 and 64 × 64 × 512 against FFTW,
  oneMKL on the Xeon); 4.3 (axis subsets, strided arrays, real multi-dimensional).
- 5–7: not started (5: fusion and pipelines; 6: float tuning, convenience layer, DCT/DST,
  C API, cloud tuning, docs, packaging, first release; 7: on demand, FFTW3 shim, SVE,
  Python).

X46 (x46 session, 2026-11-17, pull request 59): a depth-first chain (DESIGN 8.5) for the
Xeon's rows of L2/4 to 4 L2 and the Mac's 8192–16384; see the experiment index.

X51 (x51 session, 2026-11-18, pull request 66): X46's loose ends, adopted per profile (see the
experiment index): Xeon 25000 (1.35 → 1.00 of FFTW), rows to 2²² and float; Mac and EPYC
rows of a small odd part times a large power of two (0.73 of their time). Left: 100000 at
FFTW's time (no split beat 20 × 5000), Granite Rapids (the cloud run was not made: its
profile keeps its plans), float on the Mac (mixed rows gain, 24576 and 196608 do not) and the
EPYC, 2²³–2²⁴ on the Xeon (float above 2²⁰), Zen 4 and Zen 5 (rule E is keyed to an L2
under 1 MiB).

X50 (x50 session, 2026-11-18, pull request 65): Bluestein convolutions of one copy on the
depth-first chain with fused chirp passes (DESIGN 8.5); Xeon 131071 ÷ MKL 1.34 → 0.84,
999983 1.44 → 1.05. Left: float, several copies, the EPYC, 999983 against MKL.

In flight (lead session, 2026-11-17): task 1.3
(`x86_v2` and `avx` targets, X14); task 4.1's next part (X41, task/4.1-planes-2, pull request 51, awaiting merge); a tidy fix. Task 2.4's acceptance is met (P1:
rules within 5 % of measure on 93–100 % of cases on all three machines); its cost model (P3)
was rejected on the first pass, so the long-row thresholds stay. Then the exit check again. Done, each merged with its
pull request:
- X42 (pull request 52): a radix-25 pass codelet with
  `avx2` and `avx512` double overrides (EPYC 500–25000 0.87–0.89 of main, FFTN ÷ FFTW 3125
  0.77 → 0.65; Xeon 100–25000 0.84–0.86, 3125 0.81 → 0.53, 25000 1.23 → 1.18); two-level
  radix-25/32/64 codelets for 16 registers rejected (no gain with GCC: the two-pass plans at
  512–2048 lose on IPC, not on spills); EPYC 512–2048 stay 1.13–1.22 of FFTW.
- Task 2.5 (pull requests 41, 50): `next_fast_length(n, kind)` and `relative_cost<T>(n, isa)`
  in `fftn/lengths.hpp`, a frozen reference table, the fast set to 2²⁰ pinned; API agreed
  by the maintainer (2026-11-17), with `length_kind` for the proposed `bool`; P4: picks 0.11–0.96 of the awkward lengths
  they replace; within 1.3 of the power of two above on 96 % (Mac), 78 % (x86).
- Task 4.1, second part (X41, pull request 51, not merged yet): unrolled kernel calls for
  multi-axis programs (S1, all targets), real planes' columns through the lanes on x86 (R3),
  narrower column blocks on Apple for columns four to a page (L2); T1 (the narrower block's
  phase joined) rejected at 0.96; fbench `THREADS` for FFTW and `mkl_omp`. Left: threaded
  large planes, EPYC 32², the Xeon against MKL, real planes against MKL.
- Task 4.1, first part (X40, pull request 43): columns of planes and volumes in blocks of
  adjacent columns, column radices counted against the L1's ways, leaf loops as one call: 37
  c2c planes ÷ FFTW Mac 1.44 → 0.88, EPYC 2.07 → 0.94, Xeon 1.49 → 0.80 (÷ MKL 1.87 → 1.00);
  64³ 0.09–0.46 of main. Left: small planes in batches, EPYC 32², columns beyond L2, Xeon ÷
  MKL at 16²–1000², real planes' complex axes, threaded planes; volumes' pencil blocks (4.2).
- X39 (pull request 31): the Xeon's long rows: block widths, grids for powers of two beyond
  L2/4, radix-20 passes in `avx512` double sequences and grid phases: Xeon ÷ FFTW 5000 1.04 →
  0.89, 10000 0.96 → 0.84, 32768 1.51 → 1.29, 2²² 0.86 → 0.75, 25000 1.50 → 1.34, 100000
  1.36 → 1.24; left: 16384 (needs a depth-first chain), 20000/25000/100000 (a radix-25
  codelet). The `avx512` overrides also reach Zen 5 and Granite Rapids, unmeasured.
- X39 (causes (5) and (6), pull request 31): on the Xeon profile (x86-64, private L2, small L3
  per CPU) three-phase grids in blocks of at most 16, two-phase blocks within twice L1 and c2c
  powers of two as grids beyond L2/4; FFTN ÷ FFTW 32768 1.51 → 1.29, 131072 1.20 → 1.09,
  46080 1.33 → 1.26, 65536 in place 1.13 → 1.09 (run 20261116-220608-84fe); radix-64 grid
  phases rejected. Arm D (after X38): radix-20 first passes in `avx512` double (300–25000),
  radix-20 grid phases, r2c 65536 as a grid and c2c powers of two ≥ 2²⁰ out of place in two
  phases (P1's Xeon misses): 0.905 over 26 changed cases (run 20261117-033154-673d); 5000
  1.04 → 0.89 of FFTW, 10000 0.96 → 0.84, 25000 1.50 → 1.34, 100000 1.36 → 1.24. Still open:
  25000, 20000, 100000 (1.24–1.34; a radix-25 pass), 16384 (1.23; a depth-first chain).
- X37 (pull request 28): pair leaves for 18–30 points on every target (`avx512` borrowing
  `avx2`'s), pair leaves for 8–16 on x86, a lighter leaf path on AArch64, fbench's single
  library call: Xeon 18–30 points 1.24–1.32 → 0.64–0.71 of FFTW, batches of 16 ÷ MKL 1.57–1.71
  → 0.94–1.01, Mac 20 points 1.55 → 1.01; exit cases over 1.05: Xeon 30 → 23, EPYC 36 → 30,
  Mac 24 → 19. Left: EPYC 15 × 2 and 64 × 2 backward, 64 × 16; Mac 2 × 3 in place.
- X38 (pull request 27): a radix-20 codelet and re-fitted `avx2` double sequences (EPYC 0.868
  on nine lengths; exit cases over 1.05: 32 → 27); 512–2048 left (radix-32/64 codelets spill).
- X36b (pull request 36): a Bluestein stage of C copies covers the whole line (EPYC 1018 1.53 →
  1.12 of FFTW).
- X37 (causes 2 and 3 of the exit check, pull request 28): pair leaves where none applied
  (`avx2` 18, 22, 26, taken also by `avx512` plans for 18–30; NEON 18–30 points, C = 1) and
  for 8–16 points on x86 (whole registers instead of the 128-bit leaf): Xeon 18–30 points
  0.40–0.54 of their time, 16 × 16–256 0.59 (÷ MKL 1.57–1.71 → 0.94–1.01); EPYC 18–26
  0.49–0.54, 16 × 16–4096 ÷ FFTW 1.16–1.26 → 0.89–0.93; Mac 20 points 1.55 → 1.01 of FFTW.
  fbench's backend now calls what `plan::execute` calls (2–6 points 0.80–0.96). A lighter
  leaf path (byte distances, the general path out of line) only on AArch64 (Mac 2–6 points
  0.83–0.89; on the EPYC it lost). Left: EPYC 15 × 2 and 64 × 2 backward and 64 × 16 (FFTW
  runs two rows per `ymm` with whole codelets), Xeon 2 and 4 points (1.10), Mac 2 × 3 in
  place (1.39, a latency chain between executions). Found on the way: `main` 1fe3fe7 runs
  single rows of 2–8 points 1.14–1.16 times as long as e8c2639 on the EPYC (X37's run
  20261117-014755-1442; the execute path did not change: probably code placement).
- X36 (pull request 30): Bluestein convolutions take the long-row radices beyond a profile
  threshold and a capped power-of-two part; small copy counts become one stage; split layout
  for single convolutions on x86; the EPYC's 65537 was a radix-16 Rader convolution: exit-
  check primes now below FFTW on the Mac and the EPYC (131071 0.87–0.88, 999983 0.75–0.86,
  4093 0.90–0.94, 65537 0.80); still over: EPYC 1018 (1.51), Xeon 131071 (1.28) and 999983
  (1.10).
- X35 (cloud, USD 1.64; campaign USD 3.51 of 20; pull request 25): on x86-64 a private L2 of
  ≥ 2 MiB (Granite Rapids) gets its own profile, whatever its L3 per CPU: no X29/X32/X33
  chains, grids and three phases beyond L2/2, two phases for power-of-two parts ≤ 2048
  (`grid_two_part_bytes`). Granite Rapids: 30 changed cases 0.59–1.007 of main (262144
  1.71 → 1.26 of FFTW, 100000 2.03 → 1.36, 0.97 of MKL); 1-D double ÷ FFTW 0.87 → 0.83.
  Axion (Neoverse V2, also 2 MiB of L2) lost with it and keeps main's plans; the Mac, EPYC,
  Zen 5 and Xeon 6240 unchanged. The 512 KiB–1 MiB rows (46080 1.62 of FFTW) still want a
  measured choice (task 2.4).
- Task 2.4, first milestone (pull request 26): `planning::rules`/`quick`/`measure`, the
  budgets, `fftn::wisdom` (keyed by a build identifier), candidates from the rules'
  thresholds; the API agreed by the maintainer; P1 first pass: rules within 5 % of measure
  on 96.7 % (EPYC), 93.4 % (Mac, second pass), about 85 % (Xeon, re-runs queued); measure
  never lost. Also the extended-CI fix (pull requests 23, 29).
- X38 (cause (4), pull request 27, not merged: the verdict is the lead's): FFTW runs two
  passes at 160–2048 with radix 20, 32 or 64 and radix-20 passes above; FFTN had no codelet
  of 20. Arm R (a radix-20 codelet, a pass radix only, and `avx2` double overrides from X2's
  search driver re-run on the EPYC with radix 20/32/64 candidates) takes the nine changed
  lengths to 0.868 of main (240 0.82, 2000 0.82, 3600 0.81, 384 0.85), EPYC exit cases over
  1.05 of FFTW 32 → 27 of 193; tracking 1.000; Xeon `avx2` 1.001; library +5.6 %. Rule (b)
  fails by the letter on two cases whose plans do not change (60 × 3 1.036, 323 1.049 on the
  re-run; code placement likely). 1024, 768, 2048, 512 unchanged: FFTN's best sequences are
  the planner's; its radix-32/64 codelets spill (35 and 67 live values), so FFTW's two-pass
  plans lose in FFTN (a generator follow-up).
- X31b (cloud, USD 1.12; campaign USD 1.87 of 20): X32 holds on Zen 5 (long powers of two
  0.73 of the grid; 65536 1.50 → 0.92 of FFTW). Granite Rapids reports a large shared L3,
  gets the Zen profile and loses: X32's chains 1.06–1.69 of the grid (262144 1.31 → 1.75
  of FFTW); where X29 makes chains, two-phase grids win (100000 0.66: 1.39 of FFTW, 0.97 of
  MKL); where main makes three phases, X34's two lose (1.03–1.46). A third profile; a
  measured choice (task 2.4) is the lasting answer.
- X33 (lead (b), pull request 19): a real row's half-length power of two follows X32's chain
  rule on Apple and Zen (EPYC long real rows 0.58–0.76 of main, FFTN ÷ FFTW 1.14–1.41 →
  0.82–0.93, ÷ FFTL at 2²⁰ 1.76 → 1.03; Mac 0.69–0.97); the r2c pairing in the grid's last
  phase on `avx512` and NEON (Xeon r2c 2¹⁸–2²⁰ 0.95–0.96; none on `avx2`). The pairing pass
  was cheap: removing it saves 0–5 % (LESSONS 3.17).
- X34 (lead (c), pull request 21): rows with an odd part ≥ 15 stay two-phase grids where a
  private L2 has < 2 MiB of L3 per CPU behind it (`grid_two_odd_part`): Xeon 100000 2.03 →
  1.44 of FFTW (MKL 1.34 → 0.95), 10⁶ 1.01 → 0.73, mixed rows 0.861 of main; chains rejected
  there (1.05–1.45); Granite Rapids not covered (its VM's L3 gives it X29's chains).
- X32 (lead (a), pull request 22): long power-of-two c2c rows stay chains
  with FFTL.jl's radix sequences on Apple and on Zen-like caches (Mac long powers of two
  0.57–0.94 of main, FFTN ÷ FFTW 1.16 → 0.87, ÷ FFTL.jl 1.34 → 1.00; EPYC 0.53–0.85, tracking
  0.965); the Xeon keeps the grid (chains 1.06–1.50). X29's bimodal chains were the rules'
  radix-16 passes on transparent huge pages. A scratch placement against 4K aliasing
  (15–30 % for malloc'd buffers in a diagnostic) was neutral in fbench and not adopted.
- X31 (cloud, USD 0.75): the push holds on Zen 5, Granite Rapids and Axion (above).
- X30b (lead 3): plan tables on 64-byte boundaries (EPYC 0.986 double, 0.971 float on its
  cases; nothing slower anywhere); compact (wr, wi) twiddles rejected again (single cases
  up to 1.04–1.07, apparently the FMADDSUB arithmetic, not the storage); three passes at
  4096 rejected (1.30–1.88).
- The GCC 14/15 ASan crash in `avx512` code on the Xeon: a GCC code-generation bug on
  ASan's fake stack (16.2 and Clang correct); sanitizer builds with GCC before 16 leave out
  use-after-return instrumentation (pull request 18, LESSONS 6.4).
- X28b (lead 4): pair leaves on x86 (two passes in registers), unit-stride leaves, 128-bit
  leaves of 17–23: single c2c rows FFTN ÷ FFTW Xeon `avx512` 0.98 → 0.65, EPYC 0.94 →
  0.73, Mac 0.80 → 0.76; tracking tier 0.95–0.99, nothing slower; library +2 %.
- X29 (lead 5): Apple keeps lengths that are not powers of two as chains (Mac 46080 and
  100000 0.62–0.65 of main, 0.74–0.76 of FFTW); Zen-like L3s keep rows whose power-of-two
  part is ≤ 1024 points as chains (EPYC 46080 0.75, 100000 0.66: 1.52 → 1.14 and 1.00 of
  FFTW); the c2r pairing in the grid's first gather on Apple (0.92–0.94); the real
  footprint in the rule and all-chains up to half of L3 rejected (power-of-two chains of
  1–2 MiB bimodal on the EPYC).
- X30 (lead 3): two-pass `avx2` double plans at 256 and 512 (radix 32): EPYC 0.975 of main
  on its cases, 256 0.905; twiddle storage (wr, wi) and 1024 = 16 × 64 rejected.
- X28 (lead 4): 128-bit leaves on x86, a one-call entry, a flat path, kernels taking their
  addresses as arguments: single rows 0.79–0.88 of main; tracking tier 0.95–0.96 on all
  three machines, nothing slower.
- X27 (lead 2): codelets of 17, 19, 23 and a Rader stage: those cases 0.24–0.31 of main;
  library text +25 % (B3, for the maintainer).
- X26 and X26b (lead 1): no copy prefetch on `avx2` (EPYC long rows 0.920); a padded grid
  intermediate where the L2 is shared by a cluster (Mac long rows 0.976), nothing on x86;
  cheaper copy loops and the prefetch by core rejected.
- X23 amendment 4: masked partial runs for GCC builds (`FFTN_MASKED_TAILS`, `AUTO`: on
  with GCC): float 0.53 of main on the Xeon's `avx512`, 0.72 on the EPYC, double
  0.78–0.89; two whole-unit cases just past the limit kept as recorded exceptions (Xeon
  `avx512` float 256 1.047, EPYC double 72 1.033).
- X25 (the x86 gap's causes) is done; its verdicts are folded into the leads below.

For the maintainer (decisions taken by the lead overnight, easy to revert): X26's prefetch gate by target (Intel cores without
AVX-512 lose up to 1.20 at 2²² until a rule by core and row width exists); X28's 128-bit
leaves kept on `avx2` though they gain only at 8–16 points there; X30's 512 override
kept although the Xeon under `avx2` loses r2c 1024² (1.057 on re-run; EPYC 1.014): with X26's
prefetch, a second case for rules by core on `avx2`.

### Leads, one-dimensional double first

**Next, ranked from the 2026-11-16 runs (one-dimensional double still first):**
(a) long powers of two ≥ 16384: X32 makes them chains of FFTL.jl's radices on the Mac and
the EPYC (FFTN ÷ FFTL.jl about 1.0 there now). Left: the Xeon (1.03 of FFTW; chains lose
there, so the grid's copies again, X26), Zen 5 unmeasured (the rule applies there), and
fbench placing buffers as malloc does (16 bytes into a page) to judge X32's arm (b).
(b) Long real rows (r2c and c2r of 65536 and 2²⁰, 1.17–1.44): X33 makes their half-length
powers of two chains by X32's rule (EPYC 0.58–0.76 of main, FFTN ÷ FFTW 0.82–0.93; Mac
0.69–0.97) and pairs r2c bins in the grid's last phase on `avx512` (Xeon 2¹⁸–2²⁰ 0.95–0.96).
Left: the Xeon (65536 1.35–1.45 of FFTW: its half-length chain of 32768 is memory-bound,
X25; 2²⁰ 1.13–1.46), and FFTW's real plans at 2²⁰, cheaper than its own c2c 2¹⁹.
(c) The Xeon's long mixed rows (100000 1.99, 46080 1.29): X34 (pull request 21) keeps
rows with an odd part ≥ 15 in two grid phases where L3 per CPU is small (Xeon 100000 2.03 →
1.44 of FFTW, 0.95 of MKL; chains lose there). Left: 100000 still 1.44 (FFTW's
recursive four-pass plan); 46080 (1.30, a two-phase grid within L2); Granite Rapids in an
8-vCPU VM reports 63 MiB of L3 per CPU, so X29's chains apply there (100000 2.16 as a
chain): a key that covers it needs a core field or a measured choice. (d) Rules by core on `avx2` (X26's prefetch, X30's 512
override: Intel cores without AVX-512 want other choices than Zen): deferred by the
maintainer (2026-11-16) until a wider set of machines shows how far cores of one ISA
diverge; server chips and the Mac come first, fastest everywhere is the end goal. (e) Primes outside
Rader's range (4093 1.32–1.41) and EPYC 65537 (1.63 here against X27's 1.09: check).
(f) Short-row leftovers (Xeon 30 points 1.24; EPYC 60 and 120 1.05–1.09; batches of 16
1.08–1.26 on x86; Mac 20 points 1.72). X30b covers x86 1024–4096 (EPYC 1024 1.20–1.23).

**Leads 1–5 below were the maintainer's choice of 2026-11-15; each now notes its
experiments.** Each needs an experiment file (AGENTS.md section 6) before it
runs; the numbers above are the baseline to beat. The rest (6–9) follow. X25's verdict
(counters on both x86 machines) says which change each lead needs first.

1. Long rows (≥ 16384, both machines; mixed 46080 and 100000 worst on the EPYC at 1.67–1.98;
   r2c and c2r of 65536 and 2²⁰ at 1.3–1.6): X12 on x86 (the grid's phases and blocks
   there), the real pairing fused into grid phases, radix 32 phases from 32768 (X3
   follow-up). X25: on both x86 machines it is the grid's copy loops (`block_copy` 40–52 %
   of FFTN's instructions on the Xeon). X26 (done): cutting those instructions did not cut
   the time (rejected; EPYC up to 1.56 slower), because the x86 copies are bound by L1 set
   conflicts: a block's rows lie a multiple of 4 KiB apart. Dropping the copies' prefetch
   on `avx2` was adopted (EPYC target rows 0.920). The effect belongs to the core, not the
   target: the Xeon forced to `avx2` loses without the prefetch (1.04, 2²² 1.20), so Intel
   cores without AVX-512 lose too; a core field in the machine profile (CPUID vendor and
   family) would let it follow the core. X26b (done): padding the intermediate so phase
   2's rows are not a 4 KiB multiple apart cuts no L1 refills on x86 and gains nothing
   there (Xeon 1.006, EPYC 0.993); it gains on the Mac (0.976, adopted there). The
   prefetch by core gains the Xeon under `avx2` at 2²⁰–2²² (0.83–0.88) but costs float
   100000 1.27 (32-byte rows): a rule by core and row width is open. Next: long rows as
   chains of small radices, the way FFTL.jl runs them (radix-4 Stockham passes, no
   copies; on the Mac FFTN's chain forced to 4⁸ at 65536 already matched the grid), or
   fewer strided copies of the caller's arrays (phase 1's gather and phase 3's copies
   stay power-of-two strided whatever the intermediate's layout).
   FFTL.jl is the reference to study here: on the Mac it beats both FFTN (FFTN
   1.35 of its time at ≥ 16384; 2²⁰ 1.60, 100000 1.53, 46080 1.45) and FFTW (0.86).
2. Prime lengths and prime factors above 16: done by X27 (codelets of 17, 19, 23; Rader
   for primes with a 13-smooth p − 1 in each profile's range). Left: 4093 (p − 1 has 31;
   EPYC 1.75, Mac 1.32 of FFTW) and other primes whose p − 1 is not smooth (Bluestein);
   65537 on NEON (Bluestein: Rader's random gather and scatter lost there); 128-bit leaves
   for 17–23 on x86; the build-size question of 25 % more code (B3).
3. The x86 kernels at 128–4096 (EPYC 1.05–1.35): a split-aware planner (costs of split
   passes; a first radix that lets a run start: AVX2's 1024 = 4 × 16 × 16 has none), split
   runs in Bluestein convolutions and grid phases, a split grouped first pass with an 8 × 8
   real transpose (port 5 is 58 % busy on `avx512`, FFTW 41–44 %). X25: the Xeon is at
   par here (port 5 57–73 % busy against FFTW's 34–55 %); the EPYC executes 1.38–1.60 times
   FFTW's instructions at 256–1024 for the same floating-point work, so the AVX2 pass loops
   need 30–40 % fewer instructions; at 4096 FFTN runs four passes to FFTW's three.
   X30 (done): most of that excess was a third pass; `avx2` double now plans 256 = 8 × 32
   and 512 = 16 × 32 (EPYC 0.905 and 0.946 of their time, batches 0.87–0.92). Two-pass
   1024 (16 × 64) loses in batches. At 1024–4096 the EPYC is not at its dispatch limit
   (IPC 2–2.5 by X30's counters) and refills L1 1.4–3.8 times as often as FFTW: twiddles
   stored (wr, wi) (half the bytes, same bits) won 0.95–0.97 there and 0.94 in float but
   lost up to 1.08 on short rows, so next is that storage per table (large tables only, an
   aligned load for the imaginary parts) and 64-byte aligned twiddle tables (0.97–0.98 of
   the cycles alone).
   X30b (done): plan tables now start on a 64-byte boundary (x30 double 0.986 EPYC, 0.989
   Xeon `avx512`, 0.996 `avx2`; float 0.97–0.98; Mac neutral). The compact storage per
   table kind won the EPYC's x30 cases (double 0.986, float 0.962 of that) but was rejected
   on single cases its FMADDSUB form shares (float 256 × 64 1.04, Xeon c2c 16 1.05); no gain
   in double on `avx512`. Three-pass `avx2` 4096 loses 1.3–1.9. Next: the compact storage
   with the old arithmetic (an XOR for −wi on large tables only), or the lane loops' spills.
4. Single short rows: X28 done (128-bit x86 leaves, a lighter entry, passes given their
   addresses: single c2c rows 8–120 at 0.79–0.88 of their time on every machine; FFTN ÷
   FFTW Xeon 1.62 → 1.27, EPYC 1.37 → 1.12, Mac 1.14 → 1.00). Left: 60 and 120 points
   (Xeon 1.50, EPYC 1.10–1.18): the partial-unit radix-4 pass (X23b) and, on the Xeon,
   three passes against FFTW's two; 24 points on the Xeon 2.0 (plan 4 × 6); leaves of
   8–16 points 1.07–1.19 (a unit-stride leaf without the set-up for any stride). X28b done
   (pair leaves in registers for short composites on x86, unit-stride leaves, 128-bit leaves
   of 17–23): single c2c rows FFTN ÷ FFTW Xeon `avx512` 0.98 → 0.65, EPYC 0.94 → 0.73, Mac
   0.80 → 0.76. Left: 30 points on `avx512` (1.24; C does not divide it), EPYC 60 and 120
   (1.05–1.09; too big for AVX2's registers as pair leaves), batches of 16 (1.08–1.26), the
   Mac's 20 points (1.72; a NEON leaf of 20).
5. Real transforms on x86 (EPYC 1.25): X6 real-input codelets. X29: the real footprint
   does not move the chain-or-grid switch (Xeon r2c 65536 as a grid: 6 % of its 1.39);
   the c2r grid now pairs in phase 1's gather out of place on Apple (Mac 0.92–0.94; EPYC 1.11); the real
   overhead of the grid route stays 0.14–0.42 on the Xeon (r2c pairing not fused: it needs
   both mirror blocks of the last phase). X29's chain-or-grid findings belong to lead 1:
   lengths with a small power-of-two part as chains (Mac 46080 and 100000 0.62–0.65 of
   main, EPYC 0.64–0.79), and power-of-two chains on the EPYC beyond L2/2, fast but bimodal
   at 1–2 MiB (follow-up: why, and whether 4–16 MiB rows can take them: 0.64–0.88).
6. Double c2c batches of 8–16 points on AVX-512 (task 2.2): the chain against FFTW and MKL;
   a lanes threshold per transform kind (r2c and c2r of 16–64 points under AVX-512 already
   take 0.56–0.96 of the chain's time). X25: batched leaves saturate one unit (Xeon ports
   0 and 5 at 82–87 %, EPYC FP pipes 2 and 3 up to 95 %).
7. Float: 60 points on `avx512` 2.6 times `avx2` (Xeon and cloud); `best_target` per
   precision and length; X23's masked partial runs, on for GCC builds since
   pull request 8 (`FFTN_MASKED_TAILS`, amendment 4: float 0.53 of main on `avx512`, 0.77
   on `avx2`, 0.72 on the EPYC), off for Clang until Clang 23's loss on odd double lengths (`avx2` 75:
   1.42) is found.
8. Planes and volumes (phase 4): the EPYC's 64³ c2c (10.9 times FFTW).
9. Threading: a dispatch cost that grows with the team's width (DESIGN 18.2 item 22).

Decisions the maintainer has not yet taken are listed in DESIGN.md section 18.2.

### Working notes for whoever resumes

- CI on a pull request is the gate (`ci.yml`: a short required set on pull requests, the
  full per-change set on each push to main; `extended.yml` weekly and on demand;
  `emulation.yml` for SDE and QEMU, weekly and on demand). Check main's run after a merge
  and fix a break there first. Build and test what you touch locally; do not spend
  local time on full gates.
- Remote timing: `uv run --frozen python scripts/remote/fbench_remote.py submit
  <xeon|epyc|xeon-clang> ARMS -- fbench-args` (profiles in `~/.config/fftn/remote.toml`,
  outside the repository; the Xeon profile sets the MKL root). `fetch` runs from the main
  checkout. Each arm is built once per machine into a shared cache by a short,
  wider build job before the timing job (`bench/README.md`, "Remote runs"); `--serial`
  runs timing jobs one at a time, `prune` clears old cache entries. Small shared allocations (`-c 8`), never exclusive; at most three of one
  agent's jobs on the Xeon at once; fbench's load-tolerant rounds, adding rounds rather
  than waiting for quiet. Remote gates go to DMOG first (`--mem=96G`; 160G and 4 hours for
  sanitizer builds; LSan off there, ptrace is forbidden on its nodes).
- On the Mac compare against the NEON build of FFTW, and build at most 6 jobs in parallel
  (the Mac is shared).
- Every experiment's verdict goes into its file, the index below and LESSONS.md.
- The arm branches that the X10, X15 and X24 files cite (`exp/x10-pass-loops`,
  `exp/x15-arm-*`, `exp/x24-lb-only`) exist only in the maintainer's local clone; they
  are not pushed.

Baseline runs (fbench run IDs): EPYC tracking 20261113-230721-7c95 (PATIENT timeout,
re-run with MEASURE); Xeon (GCC 14) tracking 20261114-004852-f498; Mac client catalogue
20261114-030832-198b; FFTL.jl (libfftl 53171e8) tracking 20261114-095234-2dcf.

## How this plan works

- **Phases** group work with a shared exit criterion. Phases 1–2 (kernels), 3 (threading)
  and the fbench work are separate tracks after phase 0 and can run in parallel: the
  threading engine is built and tested on the walking skeleton's programs (its chain
  passes already divide into chunks). Phase 3's performance exit is measured once the
  phase 2 kernels exist.
- **Tasks** have an ID, a short description and an acceptance test. Claim a task by adding
  `(claimed: name, date)` to its line.
- **Experiments** are listed in the index below and each gets a file under
  `dev/experiments/`, named `<ID>-<slug>.md` (template:
  [experiments/TEMPLATE.md](experiments/TEMPLATE.md)), before
  it is run. An experiment ends with a verdict and a design change or a deletion.
- **Exit criteria** are measured by fbench on the named machines. A phase is not finished
  because its tasks are ticked; it is finished when its criteria are met or the plan is
  changed, with reasons, in the decision log of DESIGN.md.
- Later phases are deliberately less detailed. Refine a phase when the one before it is
  half done.

## Phases

| Phase | Outcome | Depends on |
|---|---|---|
| 0 | Foundations: build, gate, CI, platform layer, ISA mechanism, tests, a correct but slow end-to-end library, fbench v0 | — |
| 1 | Generator and single-thread complex 1-D kernels at target speed | 0 |
| 2 | Real transforms, batches, any length, planner with cost model and measurement | 1 |
| 3 | Threading: engine, adapters, topology, scheduler | 0 (parallel with 1–2) |
| 4 | Planes, volumes, strided axes, in place | 2, 3 |
| 5 | Fusion: edge operations, convolution, transform pairs, callbacks | 4 |
| 6 | General coverage and release preparation | 5 |
| 7 | On demand: FFTW shim, SVE, Python, more axes | — |

---

### Phase 0: foundations

Goal: everything needed to develop safely and measure honestly, plus a walking skeleton:
the public API computing correct transforms of any length through deliberately simple
kernels. Speed is not a goal here.

| ID | Task | Acceptance |
|---|---|---|
| 0.1 | Repository skeleton: CMake (3.25+), presets `check-release`, `check-tidy`, `check-sanitize`, `check-tsan`, `check-coverage`; static library target with one translation unit per instruction set; `tools/coverage_gate.py`; a `uv` project (`pyproject.toml`, lock file, Python 3.14+) for all Python tooling; the Doxygen linter configuration and the example-snippet check | `scripts/check.sh` runs every stage on the Mac and on Linux GCC 14 |
| 0.2 | Platform layer: CPU feature detection (x86 CPUID, AArch64 Linux `getauxval`, macOS `sysctlbyname`, Windows), cache geometry and topology detection with overrides, aligned memory, timers | unit tests with synthetic topologies; correct output recorded for the Mac, the Xeon and the EPYC cluster |
| 0.3 | ISA regions and vocabularies: `FFTN_ISA_BEGIN/END`, per-target namespaces, vocabularies for `scalar`, `neon_fcma`, `avx2` (others in phase 1); dispatch at plan construction | all-targets link test; disassembly test per target; experiments B1, B2 |
| 0.4 | Test infrastructure: quadruple-precision reference DFT (Boost.Multiprecision, test-only), analytic oracle suites, derived tolerances, canary-page helper, allocation hook, FFTW as a test-only dependency; suites parameterised on precision (`double` and `float`) from the start | suites run against a naive DFT implementation and catch seeded faults (wrong sign, off-by-one bin, 1 ulp twiddle error at large n) |
| 0.5 | Walking skeleton: `shape`, `plan`, `workspace`, `describe()`; program and stage driver with chunked stages; a chain built from templated radix-2/3/4/5 butterflies plus the generic prime pass on the scalar vocabulary; c2c, r2c, c2r rows and in-place batched 2-D planes, any length, serial; every public declaration documented in its header; first `docs/conventions.md` and one compiled example | all oracles pass for every n ≤ 512 and sampled larger n; contracts tested; coverage threshold recorded in the gate; `apidoc` stage green |
| 0.5b | A first `fftn/kokkos.hpp`: View-based construction and the Kokkos-FFT-shaped plan class over the skeleton (serial execution only) | a small Kokkos program using rows and planes compiles and passes; gives the API review real call sites |
| 0.6 | fbench v0: driver with process rounds, paired bootstrap intervals and the identical-arm control; backends for FFTN and FFTW; the standard catalogue; quiet-machine check; JSON-lines store; remote runner for the Slurm machines | A/A reports "no evidence" on all cases; a seeded 5 % slowdown is detected; a perturbed output is flagged |
| 0.7 | fbench backends for FFTL.jl, MKL, pocketfft, KFR, Accelerate; client catalogue re-captured with directions, placement, planes and real weights | each backend passes the correctness guard on its platforms |
| 0.8 | CI: Linux x86-64 (GCC 14, current GCC, Clang), macOS arm64, Linux AArch64, Windows clang; gate stages split across jobs; SDE and QEMU runs for instruction sets the runners lack | green on an empty change |
| 0.9 | API review: the headers of 0.5 reviewed by the maintainer against DESIGN.md section 5 | decisions recorded in the decision log |

Exit: gate and CI green on all four platforms; fbench A/B trusted (its self-tests pass on
the Mac, the Xeon and the EPYC cluster); baseline numbers stored for the backends of 0.6
and 0.7. The remaining competitors of DESIGN.md section 16.2 are added as phases need
them, all before the phase 2 exit.

---

### Phase 1: generator and complex 1-D kernels

Goal: single-thread c2c rows at target speed on Apple M-series, Zen 3 and the Xeon.

| ID | Task | Acceptance |
|---|---|---|
| 1.1 | Generator core: DAG with pending rotations, split-radix, Good–Thomas, symmetric primes, simplification, value-numbering CSE, depth-first ordering, high-precision verification, C++ emission in complex-register form, metadata table; radices 2–16; `gen/README.md` (pipeline, intermediate representation, how to add a radix or lowering, command line, subtleties) | operation counts match the reference table; `gen` gate stage live; a second reader can add a radix from the README alone |
| 1.2 | Pass templates (lane, grouped, leaf) with tails and with the edge-variant hooks in place from the start (scale and factor on load and store, as the first two edge operations); twiddle tables (full mode) with exact generation; chain strategy; rules planner v0 (FFTL's rules as the starting point) | all tests pass with generated kernels on `scalar` and `neon_fcma`, in `double` and `float`; a fused scale costs no extra pass |
| 1.3 | Vocabularies `avx512`, `neon`, `x86_v2`, `avx`; in-register transposes per target; non-FMA operation order from the generator for targets without FMA; experiment X14 | tests pass under every target, natively or emulated, including whole-machine emulation of Nehalem, Sandy Bridge and Neoverse N1 |
| 1.4 | Experiments X2 (radix sets and order), X4 (padding and addressing), X8 (whole-transform leaves) | defaults set from results |
| 1.5 | Split lowering with FMA shaping; `BL` layout and conversion edge variants; experiment X1 | layout default per target decided: split runs by default on `avx2` and `avx512`, interleaved on NEON and scalar (X1, merged 2026-11-14) |
| 1.6 | Radix 32 and 64 codelets; grid strategy, few-big-passes regime; twiddle modes; experiments X3, X5 | rows of 16–64 thousand points on the Xeon at or below FFTW (X3 done: radix 32/64 priced for AVX-512 float only; X5 open) |
| 1.7 | Grid strategy, blocked regime for rows beyond L2; experiment X12 | long rows (2¹⁷–2²⁴) at or below the best competitor |

Exit (fbench standard catalogue, c2c rows, one thread): the DESIGN.md section 1.3 targets
against FFTW and FFTL.jl on M-series, Zen 3 and Xeon; within 1.10 of MKL on the Xeon;
the `x86_v2`, `avx` and plain `neon` targets, forced on the same machines, at or below
FFTW restricted to the same instruction sets. The MKL figure is interim; the end target
is 1.00. Per-length choices at this stage come from the offline searches of X2 encoded as
rules; the cost model arrives in phase 2. Cloud checks (Google Cloud or AWS) on a Zen 4 or
5, on a recent Intel Xeon (Sapphire Rapids or newer, with MKL as a competitor: its AVX-512
behaves differently from the Cascade Lake Xeon's) and on an AArch64 Linux machine
(Neoverse) confirm that nothing is badly off there; tuning them is phase 2 and 6 work.

---

### Phase 2: real transforms, batches, any length, planner

| ID | Task | Acceptance |
|---|---|---|
| 2.1 | r2c with the post fused into the last pass; c2r through the DIT chain with the pre fused; eligibility by live-value count; experiment X6 (real-input codelets) | real overhead ≤ 20 % of the half-length complex time on all three machines |
| 2.2 | Lanes strategy for row batches (tile gather and scatter) and strided lines; lane blocks sized by cache; experiment X7 | awkward short rows on AVX-512 at or below FFTW and MKL (built 2026-11-15; X7 after X20: lanes only in float on x86, 0.32–0.55 of the chain at 8–16 points; double c2c on AVX-512 still 1.1–1.3 of FFTW with the chain, not met; strided axes of planes are phase 4's) |
| 2.3 | Codelets 7, 11, 13 and composites; vectorised generic prime pass; Bluestein pass with smooth-part split | every n ≤ 4096 correct and priced; no FFTW fallback anywhere |
| 2.4 | Cost model, machine profiles and `fbench calibrate`; `planning::measure` with a generous budget (DESIGN 11.2: real-size timing, seconds per shape if needed, a quick level; never PATIENT-like search); wisdom export and import; experiments P1, P2; brought forward on 2026-11-16 ahead of more threshold rules (claimed: task/2.4-measure, 2026-11-16: `planning::measure`, wisdom, P1, P2 first pass) | the model's first choice is within 5 % of the measured best for ≥ 90 % of catalogue cases |
| 2.5 | `next_fast_length` and published per-length cost (done: `fftn/lengths.hpp`, P4; pull requests 41 and 50) | documented and tested |

Exit: all 1-D row shapes (c2c, r2c, c2r; 1 to thousands of rows; smooth, prime and
awkward lengths) meet the section 1.3 targets on one thread on M-series, Zen 3, the Xeon,
one Zen 4 or 5 and one AArch64 Linux machine; default plan construction under 1 ms for
smooth lengths.

---

### Phase 3: threading

Can start after 0.5: the engine works on the walking skeleton's programs, including
splitting a row across a chain's chunked stages.

| ID | Task | Acceptance |
|---|---|---|
| 3.1 | Work graph from program loops; `workspace` synchronisation state; the claiming engine; `completion::all` and `own`; `rows_of` | bitwise equality with serial for every team size on synthetic topologies |
| 3.2 | Scheduler as a pure function; simulation tests with adversarial interleavings | pathological cases from the prototype's log reproduced in simulation and handled |
| 3.3 | `fftn/omp.hpp` (team and executor), `fftn::thread_pool`, the custom-executor concept and C contract | TSan clean; runtime matrix (libgomp, libomp) green |
| 3.4 | Experiments T1 (dispatch and wait costs per machine), T2 (claiming against static barriers), T5 (split thresholds), T4 (run-to-run swings: first touch, NUMA balancing, false sharing) | machine profiles carry threading constants; default engine decided |
| 3.5 | `fftn/kokkos.hpp`: executor, View-based plans, Kokkos-FFT-shaped plan; experiment T3 | tests on Kokkos Serial, OpenMP and Threads |
| 3.6 | Intra-row splitting with the phase 1–2 kernels: chunked chain stages and the blocked grid (needs 1.7) | single long rows scale within a cache group on Zen 3 and the Xeon |

Exit: threaded targets of section 1.3 on Zen 3 (8 and 64 threads), Xeon (18) and M-series
(performance cores), measured with the phase 2 kernels; no fbench threads-catalogue case
slower than serial; never slower than a team of serial plans by more than 2 %.

---

### Phase 4: planes, volumes, strided axes

| ID | Task | Acceptance |
|---|---|---|
| 4.1 | Multi-axis shapes; plane-at-a-time programs; in place for all strategies | batched 2-D c2c at or below FFTW and MKL, one thread and threaded |
| 4.2 | Volumes with pencil blocks; per-thread scratch of plane or pencil-block size | 3-D c2c at or below FFTW and MKL; scratch bound tested |
| 4.3 | Arbitrary axis subsets and orders of strided arrays; real multi-dimensional transforms | property tests against the reference |

Exit: fbench planes and volumes catalogues meet the one-thread and threaded targets.

---

### Phase 5: fusion

| ID | Task | Acceptance |
|---|---|---|
| 5.1 | Edge operations: scale, window, factor, valid, embed, accumulate | each matches "transform then separate pass" within the derived tolerance (fusing a multiply into an FMA changes last bits), and is faster |
| 5.2 | `convolution` pipeline: fused, pruned, several kernels, one forward to several inverses | at or below 0.8 of three separate calls for L2-resident sizes |
| 5.3 | `spectral_pair` and in-cache callbacks, including multi-dimensional pipelines with pinned axis order, prologue and epilogue | a volume pipeline sweeps memory three times (verified by counters) |
| 5.4 | Experiment X9: scrambled intermediate order and `spectral_layout` | adopted or rejected with numbers |
| 5.5 | Owner-computes row sharing with application kernels | example and test |

Exit: pipelines meet their acceptance numbers, and the client catalogue meets the
DESIGN.md section 1.3 row for client catalogues.

---

### Phase 6: general coverage and release preparation

Float on every target and tuned where wanted (it is correct and tested from phase 1 on);
convenience layer; DCT/DST I–IV and DHT as composed plans; split-complex
layout; C API; Windows tuning; AArch64 Linux tuning on Neoverse N1, V1 and V2 (cloud);
Zen 4/5 and recent Intel (cloud); the documents of DESIGN.md section 17 complete and
reviewed as a whole (each phase adds its own part as it lands); the per-release public benchmark
report (one command, fixed cases and machines); packaging (CMake package,
pkg-config, a generated amalgamation if wanted); an informational library size report, never a gate
(the maintainer, 2026-11-17: speed takes precedence, and the present size, X27's +25 %
included, is accepted): CI prints the library's text size per target on each pull request,
experiment files record size changes, and build options (`FFTN_OLD_X86_TARGETS`,
instantiating only the kernels that win) remain the levers if size ever matters; first
tagged release.

### Phase 7: on demand

FFTW3-compatible shim; SVE (experiment X11); Python binding; more than three transformed
axes; Rader; SME on Apple M4 and later for batches of short transforms (speculative).

---

## Experiment index

Each row becomes `dev/experiments/<ID>-<slug>.md` before it runs. Status: planned,
running, done (with verdict).

| ID | Question | Phase | Status |
|---|---|---|---|
| [B1](experiments/B1-isa-regions-gcc-linux.md) | Do ISA attribute regions hold on Linux x86-64 with GCC 14 and 15 (inlining, by-value vector ABI at -O0, two-TU link, disassembly)? | 0 | done: regions hold; GCC needs `-Werror=psabi` |
| [B2](experiments/B2-isa-regions-windows-clang.md) | The same on Windows with clang on the MSVC ABI (x64 and Arm64); how to detect FCMA and pointer authentication on Windows on Arm | 0 | done: regions hold; FCMA and PAuth read from the registry (task 0.8) |
| B3 | Library build time and size as codelets and targets grow; which codelets are worth instantiating per target? | 1 | planned |
| [X1](experiments/X1-split-lowering.md) | Complex-register against split arithmetic, per target (FCMA included) and pass position (first, middle, last; short and long rows) | 1 | done: split runs on avx2 and avx512 (0.87–0.93 where a run exists), interleaved on NEON |
| [X2](experiments/X2-radix-sets.md) | Best radix sets and orders per target by exhaustive search over the standard lengths; how far are simple rules from the optimum? | 1 | running: planner v1 adopted (EPYC); own NEON tables; refreshed on every machine after X1/X3/X18/X19 (AVX-512 own tables, float 0.957 of before on the Xeon; AVX2 and NEON overrides, 0.94–0.95 on changed EPYC lengths, 0.98 on the Mac's); Mac above 1024 pending |
| [X3](experiments/X3-big-radices.md) | Few big passes (radix 16–64, in the output buffer) against chains for rows of 8–64 thousand points; spill counts per codelet and target. The prototype's radix 20–32 attempts lost everywhere: find out whether generated, ordered codelets change that | 1 | done: adopted for AVX-512 float only (planner table with radix 32/64 columns: Xeon float chains 0.977 of before, 256–1024 0.76–0.94); rejected for AVX-512 double, AVX2, NEON and as grid phases up to 16384; radix 32 phases from 32768 a follow-up after X12b |
| [X4](experiments/X4-padding.md) | Padding: incremental addressing cost; pad always, never, or only where the stream rule binds. Also: do edge passes gain from a variant chosen by the caller's base address (cache-line splits on wide registers)? | 1 | running: for the grid on the Mac, staggered scratch lines adopted (2^22 −8 %), padding the intermediate rejected; the chain's lines, x86 and edge-pass variants open |
| X5 | Twiddle modes (full, shared, derived) by length and target | 1 | planned |
| [X6](experiments/X6-real-codelets.md) | Real-input codelets against half-length complex with fused post/pre, even and odd n | 2 | planned (needs the generator's pruned variants); the half-length route's fusion rule measured on the Mac (task 2.1) |
| [X7](experiments/X7-lanes.md) | Lanes against in-row kernels for batches of short rows, by length, target and row count | 2 | done: re-decided after X20's batched leaves: lanes up to `lanes_up_to` in float on x86 only (AVX-512 100, AVX2 32; 0.32–0.55 of the chain at 8–16 points); 0 in double and on NEON |
| [X8](experiments/X8-leaves.md) | Whole-transform leaves for n ≤ 64–256 against two- and three-stage chains | 1 | done: leaf limit 32 on NEON in double (16–22 % faster), 16 elsewhere; no leaf of 64 |
| X9 | Scrambled intermediate order for transform pairs | 5 | planned |
| [X10](experiments/X10-pass-loops.md) | Generated whole passes against templated pass loops; on x86 first: addressing, register budget and shuffles of the pass loops (after X18) | 1 | done: rejected (loop addressing cuts instructions, not time; the cost is port-5 shuffles, X19 next) |
| X11 | SVE at 128 and 256 bits on Graviton 3/4 against NEON | 7 | planned |
| [X12](experiments/X12-beyond-l2.md) | Beyond-L2 forms: DIF grid, DIT grid, in-place sandwich; block sizes by cache level; the in-place route (scratch of N, permutation pass, or palindromic swap) | 1 | running: DIF grid in two or three phases adopted for rows beyond L1 on the Mac (0.47–0.71 of the chain's time, run 20261114-110844-8b03); copies fused into end passes for small-radix phases within L2/4 (2^18: 1.03 → 0.92, run 20261114-132841-80e0); x86 runs pending |
| [X12b](experiments/X12b-grid-thresholds.md) | The grid's thresholds (chain or grid, two or three phases) as rules of L1 and L2 that suit the Xeon, the EPYC and the Mac | 1 | done: with a private L2 (x86) the grid starts beyond L2/2 and three phases beyond L2; the Mac (cluster-shared L2) unchanged; x86 0.91–0.96 of the old default over the X12b cases, 4096 points on the Xeon 1.42 → 1.05 times FFTW |
| [X14](experiments/X14-old-x86.md) | Is a separate `avx` target (256 bit, no FMA, no AVX2) worth keeping over `x86_v2` on Sandy Bridge-class machines? Do the old targets meet the exit against FFTW restricted to their instruction sets? Operation order without FMA | 1 | done: `avx` kept (0.68–0.74 of `x86_v2`); both at or below the restricted FFTW, tails fail; no split runs without FMA; leaf of 32 for `x86_v2` in double |
| X13 | Next-row prefetch and non-temporal stores per machine for out-of-cache batches | 2 | planned |
| [X15](experiments/X15-compiler-vectorised-scalar.md) | What does a current compiler get from the scalar pass loops (`omp simd`, the scalar text in the `avx2` region, `-march=native` as a ceiling), against a hand vocabulary? | 0 | done: omp simd rejected (compiler-dependent); old targets need hand vocabularies |
| [X16](experiments/X16-avx512-table.md) | `avx512` against `avx2` on AVX-512 machines: does `avx512` need its own cost table, or an R × C transpose for leading radices below C? | 1 | done (Cascade Lake only): own table in float (0.955 of before); `avx512` first in double (0.80–0.87 of `avx2`); float loses at batch 64 (1.02) because of tails (lengths 8 does not divide 1.20–1.23, the rest 0.78–0.85): per-plan choice and tail work go to phase 2 |
| [X17](experiments/X17-bluestein-threshold.md) | Above which prime does Bluestein beat the generic prime pass, per target? | 2 | done: Bluestein from 17 up (threshold 16) on the Mac, the Xeon (avx512, avx2) and the EPYC (avx2), double and float |
| [X18](experiments/X18-x86-kernels.md) | Why does FFTN trail FFTW on x86 (Xeon, EPYC), and what closes the gap? | 1 | done: driver divisions and a four-line chain fixed (Xeon 1.50 → 1.25 of FFTW, EPYC 1.37 → 1.29); next X10 addressing, radix 32/64, then X1 |
| [X19](experiments/X19-generator-rotations.md) | Do codelets with fewer multiplications by ±i (the real-coefficient form: one register swap per output pair, radix 16 7 against 19) make x86 faster without slowing NEON? | 1 | done: adopted (x86 0.976 of main: Xeon avx512 0.980, avx2 0.976, EPYC 0.971; Mac 0.991); port-5 uops −4 to −16 % |
| [X20](experiments/X20-call-overhead.md) | Where does a tiny serial execute's time go (entry, driver, kernel), and what does a leaner driver close at n ≤ 64? | 1 | done: adopted (flat programs, leaves called with addresses, a batch of leaf rows in one call): Mac 0.45 of main over the x20 cases, 8 × 4096 3.08 → 0.97 of FFTW; Xeon 0.57 (one case slower at that commit, gone after two entry fixes per the harness; fbench confirmation pending) |
| [X21](experiments/X21-phase1-cloud.md) | Phase 1 exit cloud checks: is anything badly off on a recent Intel Xeon (with MKL), Zen 5 and a Neoverse, and does `avx512` over `avx2` hold off Cascade Lake? | 1 | done: nothing badly off (FFTN ÷ FFTW over the tracking c2c rows: Emerald Rapids 1.17, Zen 5 1.12, Axion 0.81; FFTN ÷ MKL 1.19); avx512 wins in double everywhere, in float not on Zen 5, and loses 3× at 60 points in float; USD 1.36 |
| [X22](experiments/X22-icpx.md) | Intel's icpx: does FFTN build and pass with it, which floating-point flags keep the numerical contract, and is it faster than GCC and Clang on the Xeon? | 1 | done: supported with `-fno-fast-math` (set by CMake); not recommended (Clang's equal in double, 3 % behind on tracking rows); float 0.80 of Clang on 7-smooth rows from masked partial runs, a portable vocabulary follow-up |
| [X23](experiments/X23-masked-tails.md) | Do masked partial runs in the `avx512` and `avx2` vocabularies close most of icpx's float lead with GCC and Clang (X22), and become the default on x86? | 1 | done: rejected for a Clang-only loss, then amendment 4 restores it for GCC builds (`FFTN_MASKED_TAILS`, `AUTO` on with GCC): float 0.53 of main on the Xeon's `avx512`, 0.77 on `avx2`, 0.72 on the EPYC; the rule missed by one whole-unit case on two machine-targets (Xeon `avx512` float 256 1.047, EPYC double 72 1.033), both kept as recorded exceptions and the option adopted (lead's decision, pull request 8). Follow-up: Clang's odd-length loss |
| [X24](experiments/X24-lanes-x86.md) | Cheaper lanes on x86: with X20's per-call preparation in the lanes driver and first and last passes that build registers from 128-bit loads of each row (no gather or scatter), do the lanes beat the batched-leaf chain in double at 8–64 points and reach FFTW at 8–16? | 2 | done: both changes rejected (per-call driver: 1–7 % slower at 8–16 × 64; per-row edge passes: up to 1.17 on AVX-512 at 24–256 points). `lanes_up_to` in double stays 0 on x86; 2.2's double c2c acceptance on AVX-512 not met (lanes 1.31–1.53 of FFTW, chain 1.01–1.26) |
| [X25](experiments/X25-x86-diagnosis.md) | Why does FFTN trail FFTW in 1-D double on x86 (EPYC 1.21, Xeon 1.06): kernel instruction stream, memory hierarchy or fixed cost per call, per case group? Counters and FFTW's plans, no library change | 1 | done: long rows (both machines) and EPYC 256–1024 are the instruction stream (the grid's `block_copy` 40–52 % of FFTN's instructions; AVX2 loops 1.4–1.6 times FFTW's instructions for the same FP work, at IPC up to 4.7); ≤ 16 points the fixed cost per call (FFTW uses 128-bit codelets); memory only for Xeon r2c 65536 (chain of 32768) and EPYC 4096; Xeon 128–4096 at par, port 5 the limit |
| [X26](experiments/X26-grid-copies.md) | Long rows: do cheaper copy loops in the grid (rows of one vector without an inner loop), wider blocks or fused phases close the gap to FFTW at ≥ 16384, 46080, 100000 and r2c/c2r 65536 and 2²⁰? | 1 | done: cheaper copy loops rejected (Mac 0.986 and Xeon 0.988, within the band; EPYC 1.05, 2²⁰ rows 1.40–1.56: the x86 copies are bound by L1 set conflicts of rows 4 KiB-multiples apart, not instructions); no prefetch in the copies adopted on `avx2` (EPYC target rows 0.920: 65536 1.41 → 1.30 of FFTW, 100000 1.94 → 1.70, r2c 2²⁰ 1.77 → 1.48); wider blocks and fused phases lose on the Mac |
| [X26b](experiments/X26b-padded-grid.md) | Long rows on x86: does a padded intermediate (phase 2's rows not a 4 KiB multiple apart) remove the grid's L1 set conflicts, and do X26's cheaper copy loops then pay? Can the copies' prefetch follow the core? | 1 | done: padding adopted where the L2 is shared by a cluster (Mac long rows 0.976 of main); on x86 it cuts no L1 refills and gains nothing (Xeon 1.006, EPYC 0.993), off; X26's copy loops rejected again; the prefetch by core rejected (Xeon `avx2` 2²² 0.84 but float 100000 1.27: needs a row-width condition) |
| [X27](experiments/X27-prime-codelets.md) | Generated codelets of radix 17, 19 and 23 (leaf and pass) against Bluestein for prime factors above 16, and their build size (B3); Rader against Bluestein where p − 1 is smooth | 1 | done: codelets adopted everywhere (targets 0.24–0.31 of the Bluestein route; 17 now 0.28–0.47 of FFTW; library +25 %); Rader adopted per target (0.4–0.8 of Bluestein from 97 up; from 16C for C ≥ 4, below 2¹⁶ on NEON) |
| [X28](experiments/X28-short-rows.md) | Single short rows (≤ 120 points, batch 1): what is left after X20, and do a lighter entry, 128-bit leaves on x86 or a flat path for two-pass chains close it? (PLAN lead 4) | 1 | done: all adopted (128-bit x86 leaves; one-call entry; passes take their addresses as arguments, no per-stage argument copy): single rows 0.79 (Xeon), 0.82 (EPYC), 0.82–0.88 (Mac) of `main`; tracking Mac 0.953, EPYC 0.964 |
| [X28b](experiments/X28b-short-leaves.md) | Single short rows after X28: do two-pass leaves held in registers (24, 48, 60 points and other short composites), leaves with constant offsets for rows of stride 1, and 128-bit leaves for 17, 19, 23 on x86 close what is left? (PLAN lead 4) | 1 | done: all adopted (pair leaves on x86: 20–120 points 0.42–0.69 of the chain on the Xeon's `avx512`, 0.51–0.76 on the EPYC; unit-stride leaves everywhere: 8–23 points 0.75–0.92; 128-bit 17–23 leaves 0.85–0.99); FFTN ÷ FFTW over single c2c rows Xeon 0.98 → 0.65, EPYC 0.94 → 0.73, Mac 0.80 → 0.76 |
| [X30](experiments/X30-x86-pass-loops.md) | x86 chains at 128–4096 (lead 3): two-pass `avx2` double sequences with radix 32/64 (P), twiddles stored (wr, wi) with duplicating loads (T), 1024 = 16 × 64 (Q) | 1 | done: P adopted (EPYC x30 double 0.975: 256 0.905, 512 0.946, batches of 64 0.87–0.92, r2c/c2r 512–1024 0.90–0.95; nothing slower; Xeon `avx2` tracking 0.999, r2c 1024² 1.034 [0.997, 1.061] re-run pending); T rejected by single cases (geomeans 0.97–0.99, float 0.94, but short rows and small grouped passes up to 1.08–1.11); Q rejected (1024 batches 1.09) |
| [X29](experiments/X29-real-footprint.md) | Real rows in the chain-or-grid rule: should r2c and c2r switch to the grid at another size than the c2c row of half their length, and is the grid's separate pairing pass worth fusing into its end phases? | 1–2 | done: real footprint rejected (no gain on any machine); c2r pairing in the grid's first gather adopted on Apple out of place in two phases (Mac 0.92–0.94; EPYC 1.11, not adopted); lengths with a power-of-two part ≤ 1024 points stay chains on Apple at any size (46080 0.65, 100000 0.62) and on Zen to half of L3 (0.64–0.85); power-of-two chains on the EPYC bimodal at 1–2 MiB, not adopted |
| [X33](experiments/X33-long-real.md) | Long real rows (r2c and c2r of 65536 and 2²⁰, lead (b)): where the time goes, and does a pairing without a memory sweep of its own (fused into the grid's last phase with block pairs, or into the chain's last pass) close the gap? | 1–2 | done: real rows' half-length powers of two as chains (X32's rule) adopted on Apple and Zen (EPYC 0.58–0.76, ÷ FFTW 1.14–1.41 → 0.82–0.93; Mac 0.69–0.97); r2c pairing in the grid's last phase adopted on `avx512` and NEON (Xeon 0.95–0.96), not on `avx2` (EPYC no gain); fused pass saved only 0–5 %: the pass was cheap |
| [X30b](experiments/X30b-twiddle-tables.md) | x86 chains at 1024–4096 (lead 3, after X30): 64-byte aligned plan tables (L), the half-size (wr, wi) twiddle storage only for passes with large tables (C), three-pass `avx2` double 4096 (R) | 1 | done: L adopted (aligned plan tables; EPYC x30 double 0.986, float 0.971; Xeon `avx512` 0.989, `avx2` 0.996; Mac 0.999); C rejected (EPYC x30 0.986 / 0.962 of L but float 256 × 64 1.04 in every variant, Xeon c2c 16 1.05, `avx512` double 1.001); R rejected (4096 three-pass 1.30–1.88) |
| [X31](experiments/X31-cloud-1d.md) | One-dimensional double after the push on recent server chips (Zen 5, Granite Rapids with MKL, Axion): where does FFTN stand against FFTW, and does any core diverge from the EPYC, Xeon and Mac enough for per-core rules (lead (d))? | 1 | done: nothing badly off; FFTN ÷ FFTW over the 46 one-dimensional double cases Zen 5 0.80, Granite Rapids 0.88 (MKL 0.85), Axion 0.74 (before the push 1.05, 1.12, 0.86); Axion beats the Mac's ratios on long rows (0.75 against 1.19), in place, real rows and planes, mostly because FFTW is slower there; Axion 8 points 1.24; Granite Rapids repeats the Xeon's long mixed rows worse (100000 2.16, 46080 1.60) and in place 65536 1.50; Zen 5 diverges in no group; USD 0.75 |
| [X31b](experiments/X31b-cloud-followup.md) | X32 and X34 on Zen 5 and Granite Rapids (cloud): does X32's power-of-two chain rule hold on Zen 5, and does Granite Rapids want the Xeon's plan, X29's chains or neither? Can the topology a VM reports separate the two? (leads (a), (c), (d)) | 1 | done, measurement only. Zen 5: X32 holds (long powers of two 0.58–0.86 of the grid; 65536 1.50 → 0.92 of FFTW); X34 changes no plan. Granite Rapids takes X32's chains (its VM reports 63 MiB of L3 per CPU), and they lose: the grid is 0.59–0.94 of their time. It wants two phases where X29 chains (100000 0.66, 10⁶ 0.59) but three phases where X34 would give two (X34's rule 1.03–1.46), and probably a grid from 512 KiB (0.81–0.96). That is a third profile; the VM's cache sizes separate it from Zen only on one sample per side, so the case is for a measured choice (planner 2.4) or a core field. USD 1.12 |
| [X32](experiments/X32-long-pow2.md) | Long powers of two (c2c ≥ 16384, lead (a)): copy-free chains of small radices (FFTL.jl's plan) against the grid, per machine; why power-of-two chains of 1–2 MiB are bimodal on the EPYC | 1 | done: chains of FFTL.jl's radices adopted on Apple (0.57–0.94 of main; ÷ FFTW 1.16 → 0.87) and Zen-like caches (EPYC 0.53–0.85); Xeon keeps the grid (chains 1.06–1.50); bimodality = radix-16 passes on transparent huge pages; scratch placement against 4K aliasing neutral in fbench, rejected |
| [X34](experiments/X34-xeon-mixed.md) | The Xeon's long rows of mixed length (100000 1.99, 46080 1.29 of FFTW): why, and does a chain rule keyed on L2, wider grid blocks for lengths with a large odd part, or other grid factors close it without a vendor switch? (lead (c)) | 1 | done: chains lose on the Xeon (1.05–1.45); wider blocks superseded; two phases for rows with an odd part ≥ 15 where L3 per CPU is small adopted (Xeon 0.86 over the 14 changed rows, 100000 0.71: 2.03 → 1.44 of FFTW, 0.95 of MKL); the EPYC (two phases 1.13–1.67) and the Mac unchanged |
| [X35](experiments/X35-gnr-profile.md) | A cache-geometry key that separates Granite Rapids from Zen 3/Zen 5 for X29's, X32's and X33's chain rules (its VM reports 63 MiB of L3 per CPU), with Granite Rapids' plans from X31b (leads (a), (c), (d)) | 1 | done: key = a private L2 of ≥ 2 MiB on x86-64 (VM- and SMT-independent; L3 per core cannot separate them). Granite Rapids (arm B9): grids and three phases beyond L2/2, two phases for power-of-two parts ≤ 2048, no chains: 30 changed cases 0.59–1.007 of main (262144 1.71 → 1.26 of FFTW, 100000 2.03 → 1.36, long real rows 0.59–0.88); 1-D double ÷ FFTW 0.87 → 0.83, ÷ MKL 0.85 → 0.81. The first profile and a variant each lost on 1–2 rows (71680 1.10, c2r out of place 1.15–1.16). Axion (same 2 MiB L2) lost 1.15–1.28 and keeps main's plans (x86 gate). Mac, EPYC, Zen 5, Xeon unchanged (profile tests). USD 1.64 |
| [X36](experiments/X36-large-primes.md) | Large primes and prime factors outside Rader's range (exit check cause 1: 131071, 999983, 4093, 1018, EPYC 65537): the convolution's radices and length, chirp passes that absorb the zero padding, few-copy stages, split convolutions | 1 | done: long-row radices for long convolutions and a power-of-two limit on Bluestein lengths per machine (Mac changed cases 0.49, EPYC 0.56; ÷ FFTW Mac 131071 2.61 → 0.88, 999983 1.99 → 0.75, 4093 1.42 → 0.90; EPYC 131071 1.64 → 0.87, 4093 1.33 → 0.94, 65537 0.80: its old Rader convolution took radix-16 passes); few copies as one stage on C > 2 (Xeon 1018 0.68); split single-copy convolutions on x86 (EPYC 0.56–0.92); half-form chirp passes rejected; Xeon 1018 ÷ MKL 1.37 → 0.90; left: EPYC 1018 1.51, Xeon 131071 1.28 and 999983 1.10 |
| [X37](experiments/X37-short-batches.md) | The exit check's short rows and small batches (causes 2 and 3): fixed cost or kernel, and do pair leaves for 18–30 points (x86 where C does not divide, NEON) and for batches of 8–16 points on x86 close them? | 1 | done: pair leaves adopted for 18, 22, 26, 30 on x86 (`avx512` takes `avx2`'s) and 18–30 on NEON (0.47–0.72 of their time), and for 8–16 points on x86 (Xeon 16 × 16–256 0.59–0.66, ÷ MKL 1.57–1.71 → 0.94–1.01; EPYC 0.85); fbench now calls `execute_all`; a lighter leaf path adopted on AArch64 only (2–6 points 0.83–0.89 on the Mac; lost on the EPYC). Exit-check cases over 1.05: Xeon 30 → 23 (MKL 32 → 27), EPYC 36 → 30, Mac 24 → 19 |
| [X38](experiments/X38-epyc-mid.md) | The EPYC's double rows of 80–20000 points (exit check cause 4; 240, 384, 640, 1024 1.19–1.23 of FFTW): do sequences re-fitted on Zen 3 with radix 32/64 (two passes where FFTW runs two), split-aware first passes or cheaper twiddle arithmetic close them? | 1 | done: R adopted by the lead with two recorded exceptions: a radix-20 codelet with re-fitted `avx2` double sequences, 0.868 of main on its nine lengths (240, 2000, 3000, 3600 now under FFTW; exit cases over 1.05 of FFTW 32 → 27 of 193), tracking 1.000; two unchanged-plan cases past rule (b) after the re-run (60 × 3 1.036, 323 1.049: plans unchanged, placement likely) kept as exceptions; library +5.6 %; 512–2048 unchanged (FFTN's radix-32/64 codelets spill) |
| [X36b](experiments/X36b-prime-tails.md) | What X36 left of the prime tail: EPYC 1018 (1.51 of FFTW; copies of a Bluestein stage one at a time or one stage over the line) and the Xeon's 131071 and 999983 (1.28, 1.10; a long convolution as a grid) | 1 | done: s = C copies as one stage while the line is not long (EPYC 1018 0.73: 1.53 → 1.12 of FFTW; Xeon 2036 0.81); the Xeon's grid convolution needs a second movable workspace slot (screen: 999983 would reach about 0.86 of FFTW, 131071 stays 1.24, its gap is cause 5) |
| [X39](experiments/X39-xeon-long.md) | The Xeon's long rows (exit-check causes 5 and 6: 16384–262144, 20000, 25000, 46080, 100000 1.22–1.50 of FFTW; in place 65536 1.16): why, and which plan rule closes them on `avx512` without changing other machines? | 1 | done: on the Xeon profile (x86-64) three-phase blocks ≤ 16 adopted (0.935 over 4 rows; 131072 1.20 → 1.09 of FFTW), two-phase blocks within 2 · L1 and c2c powers of two as grids beyond L2/4 adopted (0.958 over 7; 32768 1.51 → 1.29, 46080 1.33 → 1.26); radix-64 grid phases rejected (0.985); 16384–25000, 100000 unchanged: they need radix-20/25 codelets or a depth-first chain; measure finds 5000–25000's better chain orders; arm D (after X38): radix-20 first passes in `avx512` double sequences 300–25000, radix-20 grid phases, r2c 65536 as a grid, c2c powers of two ≥ 2²⁰ out of place in two phases (P1's misses): 0.905 over 26 changed cases (5000 1.04 → 0.89 of FFTW, 25000 1.50 → 1.34, 2²² 0.86 → 0.75) |
| [X40](experiments/X40-planes.md) | Batched 2-D c2c planes (task 4.1; 1024² 2.1 / 4.1 of FFTW on the Mac / EPYC): where does the time go against FFTW's plans and MKL (rows or strided columns, passes over memory, copies, power-of-two set conflicts), and do lanes over blocks of adjacent columns close it? | 4 | done, first pass: the columns were the gap (every column alone, each line re-read per column; a last pass of radix 16 whose legs shared an L1 set); lanes over blocks of adjacent columns and one call per loop of leaves adopted: planes 0.613 / 0.453 / 0.537 of main (Mac / EPYC / Xeon), ÷ FFTW 0.88 / 0.94 / 0.80, ÷ MKL 1.00; tiny planes, EPYC 32², long columns and the Xeon against MKL left |
| [X41](experiments/X41-planes-2.md) | The rest of task 4.1 after X40: real planes' complex axes through the column phase, small planes in batches, columns beyond L2, threaded planes against FFTW's and MKL's threads, the Xeon against MKL | 4 | done: unrolled calls (small planes 0.87–0.94), real planes' columns on x86 (0.83–0.89), Apple's long columns (0.90) adopted; T1 rejected; threaded large planes, EPYC 32² and the Xeon against MKL left |
| [X42](experiments/X42-codelets.md) | Do a radix-25 pass codelet and two-level radix-25/32/64 codelets that fit 16 registers make FFTW's plan shapes pay in FFTN (25000, 20000, 5000, 15625 on `avx512` and `avx2`; two-pass 512–2048 on `avx2`)? | 1 | done: radix 25 (x86 and scalar rows only) with `avx2` double overrides (500–25000: 0.874 then 0.891 of the arm before on the EPYC) and `avx512` (100–25000: about 0.86 then 0.842 on the Xeon; 3125 0.81 → 0.53 of FFTW); two-level codelets rejected (1.000 with GCC: spills were not the cost; LESSONS 4.17); no radix-32/64 re-fit passed; library +6.1 % x86, +2.0 % Mac |
| [X44](experiments/X44-small-planes.md) | The one-thread plane cases task 4.1 still misses after X41 (EPYC `avx2` 32², Mac 32²–64², the Xeon's c2c planes against MKL from 16², real planes against MKL): where the time goes (rows, columns, copies, fixed cost; counters on x86) and which column radices, column kernels or real-plane path close it | 4 | done: columns of 32 as one lane pass adopted on x86 (EPYC 32² 0.72, ÷ FFTW 1.39 → 1.00; Xeon 0.77–0.81, ÷ MKL 1.37 → 1.07); nested column runs rejected (EPYC 0.94 but 4 × 6 × 8 1.03; Xeon 0.962; Mac 0.997); the Mac's 32²–64² gap is the radix-64 codelet (X42); the Xeon's large c2c and real planes against MKL left |
| [X43](experiments/X43-threaded-columns.md) | Threaded planes (X41: the largest scale 2.3–4.7 times on 8 cores against FFTW's 4–6; oneMKL's threads ahead from 16² to 1024² on the Xeon): are the column phase's chunks too coarse (a block of B columns counted as one line of n points), and are its blocks too wide for a member's L2? | 4 | done: a team program with narrower column blocks (EPYC 0.887 at 8 lanes; Xeon rejected, 0.997) and line blocks for chain-only columns (EPYC 2048 × 36 0.72); coarse chunks ruled out |
| [X43b](experiments/X43b-volumes.md) | Volumes (task 4.2): the two inner axes plane at a time and the outer axis over blocks of pencils, against `main`'s three sweeps axis by axis; FFTN against FFTW and oneMKL, one thread and threaded | 4 | done for one thread (plane at a time rejected: EPYC 0.979, Xeon 97³ 1.048, Mac 0.962) and adopted for teams: EPYC 0.775 (VT), Xeon 0.771 (VTX) |
| [X45](experiments/X45-cloud-exit.md) | The phase 1 exit check on recent cloud cores (Zen 5, Granite Rapids with MKL, Axion) for `main` 509d123, and how far `main` moved since X31, X31b and X35 | 1 | done: every mean target met (÷ FFTW 0.70 / 0.77 / 0.71; Granite Rapids ÷ MKL 0.75); tail rule met on Axion only (4.7 %), Zen 5 7.8 %, Granite Rapids 13.0 % (MKL 11.4 %): Granite Rapids' long rows and in-place 65536, short single rows everywhere; Zen 5's shared rows 0.90 of X31's time |
| [X46](experiments/X46-depth-first.md) | Does a depth-first chain (each sub-row finished in L1/L2 between a first pass and an in-place decimation-in-time top pass) close the rows between L2/4 and a few times L2 (Xeon 16384, 32768, 65536 ÷ MKL, 20000–100000; Mac 8192–65536)? | 1 | done: adopted on the Xeon profile (L2/4 to 4 L2, tops of 16 or 20, derived top-pass twiddles: 12 changed rows 0.674 of their time, FFTN ÷ FFTW 1.06–1.34 → 0.70–0.86, ÷ MKL 0.58–0.85; stored twiddles 0.86, deleted) and on Apple's (L1 to 2 L1, top 4: 8192 0.73, 16384 0.84); EPYC unchanged; 25000 has no split on `avx512` |
| [X47](experiments/X47-epyc-mid-2.md) | The EPYC's double rows of 512–2048 points (1.09–1.22 of FFTW after X38 and X42): counters first (FP-register-file stalls, L1 refills, integer instructions), then two-line chains of two passes (A) and lane loops by group (B) | 1 | done: A (two-pass chains run the last pass in place on the destination: two lines) 512 0.95, 256–640 two-pass rows 0.95–0.98 on the EPYC, tracking 0.997–0.999 on three machines, but 0.987 over its 42 target cases against the rule's 0.97: adopted by the lead's decision (the named 512 cases 0.965; nothing slower); B (lane loops by group) rejected: GCC spilled the leg offsets; 768–2048 not closed |
| [X52](experiments/X52-volumes-2.md) | Volumes and threaded planes against MKL and FFTW after X43, X43b and X44 (Mac 16 × 256 × 256 1.4–1.6 of FFTW, 512 × 64 × 64 and 64 × 64 × 512 1.1–1.2; EPYC 8 lanes 17 of 38 volumes over FFTW, 2048 × 36, 2048²; Xeon ÷ oneMKL 1.15 one thread, 1.24 at 8 lanes): leaf columns whose legs share an L1 set, column blocks, how a team claims a split task's chunks | 4 | done: leaf columns as two passes (Mac, EPYC; 16 × 256 × 256 0.55, EPYC 16³ 0.66); chunk runs of at most left / (2 cap) (8 lanes 0.90–0.91, Mac 0.86–0.94); planes split by complex points and from 32768 points on a mesh (Xeon small planes 0.35–0.67, EPYC real 0.72–0.90; not the Mac); one-thread wide column blocks on the EPYC (0.933); W1, W3 rejected |
| [X48](experiments/X48-exit-3.md) | The phase 1 exit check on the cloud cores again (Zen 5, Granite Rapids with MKL, Axion) for `main` 8c18dd2, after X42, X46, X47 and task 4.2 | 1 | done: every mean target met (÷ FFTW 0.69 / 0.76 / 0.71; Granite Rapids ÷ MKL 0.73); tail rule beyond noise met on Zen 5 (3.1 %, was 5.7 %) and Axion (4.7 %), not on Granite Rapids (÷ FFTW 9.3 %, ÷ MKL 5.7 %: long rows, in-place 65536, 160–1024, primes against MKL) |
| [X49](experiments/X49-tiny-calls.md) | The fixed cost of tiny calls: single rows of 2–8 points, 64 × 2 and 15 × 2 backward, 2 × 3 and 3 × 3 in place (1.1–1.5 of FFTW on five machines): fbench's own backend cost, store-to-load forwarding through paired accesses on Apple, or kernels? | 1 | done: mostly fbench's backend (arm 0 kept: tiny rows Mac 0.77–0.81 of FFTW, EPYC 0.68–0.86 and Xeon 0.89–1.02 of FFTW's cycles); unpaired NEON (U, U2) rejected (0.974 over its targets; 2 × 3 in place 0.85); 15 × 2 and 64 × 2 on x86 are kernels (two-row lane leaf, follow-up) |
| [X51](experiments/X51-depth-2.md) | The depth-first chain's loose ends after X46: Xeon 25000 (no split with C = 4) and 100000, double rows beyond 4 L2 (two tops), single precision on the Xeon and the Mac, the Mac's 24576 and 49152, the EPYC's tops of 4 | 1 | done: P (a partial first-pass block) and T (third top 10: Xeon 25000 0.75, 1.35 → 1.00 of FFTW), L (Xeon to 64 L2: 524288–2²² 0.68–0.75), F (Xeon float: 0.687), M (Apple mixed rows to 16 L1: 0.735), E (EPYC mixed rows, top 4: 0.730) adopted; 100000 unchanged (no split wins); Granite Rapids not measured (cloud run not made) |
| [X50](experiments/X50-primes.md) | The Xeon's large primes against MKL (999983 1.43, 131071 1.33): a Bluestein stage whose convolution runs as two depth-first chains (X46), a rule on the convolution's length; EPYC 65537's Rader/Bluestein bimodality (P1b) | 1 | done: Bluestein convolutions of one copy on two depth-first chains with fused chirp passes, where a row of M would be depth-first (Xeon, Apple) and at any M beyond on the Xeon; Xeon 8179–999983 0.56–0.73 of main (÷ MKL 131071 1.34 → 0.84, 999983 1.44 → 1.05), Mac 4813–5119 0.86; EPYC 65537's flip not reproduced (Rader 1.5× faster in 24 processes), no Rader change; library +6.2 % Mac, +3.5 % x86 |
| [X54](experiments/X54-radix64-avx2.md) | Can the radix-32 and radix-64 codelets be ordered so that GCC spills far less of them on `avx2` (a list scheduler against a model register file; bits identical), and does that let two-pass plans of 768, 1024, 2048 win at both batches on the EPYC so the planner takes them? | 1 | done: A (fitted order, bit-identical), B (768) and C adopted by the maintainer's decision against rule (b)'s letter (80 at 1.035 against 1.034, byte-identical code at a moved address): radix-32/64 plans 0.975 on the EPYC, tracking EPYC 0.994, Xeon 0.999, Mac 0.999; B (two-pass `avx2` double overrides) adopted for 768 (0.84; ÷ FFTW 1.08 → 0.91), 1024 dropped by rule (c) (0.84–0.95 at batches 1–256, 1.25 at 1024 × 1024); C (`avx2` float 512, 640, 768, 2048, 4096) adopted (0.898); 2048 double and 64 × 16 not moved |
| [P1](experiments/P1-rules-vs-measure.md) | How often is the rules' choice within 5 % of `planning::measure`'s per machine (task 2.4's acceptance: 90 % of catalogue cases), which rules miss, and does measurement recover a misfiring profile (Granite Rapids' fields forced on the Xeon)? | 2 | first pass done: rules within 5 % of measure on 96.7 % of the tracking tier on the EPYC, 77 % on the Xeon (about 85 % without noise), Mac inconclusive under load; measure never slower by fbench; Granite Rapids' fields forced on the Xeon recovered in 10 of 11 cases (the miss a budget defect, fixed); planes' radices the repeat miss. Second pass (73eb719, placements per round, median): Mac 93.4 %, EPYC 95.1 %, Xeon 86.9 % (long real rows, planes, 2²², 2431 repeat); misfire 9 of 9 after the budget fix; EPYC r2c 65536 measure 1.18 slower (power-of-two chains bimodal by page backing); EPYC 65537 Bluestein 1.75 faster than the rules' Rader |
| [P2](experiments/P2-measure-budget.md) | Measurement budget: how many candidates and how long before the choice stops improving; the quick and full efforts' budgets | 2 | first pass done: Q within 0.999–1.014 of M; no winner above rank 7; plan construction dominates large transforms (2²² 12.5 s for M); choices unstable under load (43–66 % of cases one program over 12 rounds), so budgets wait for a sampling revisit; indicated quick 6, measure 8 |
| [P1b](experiments/P1b-rader-65537.md) | Rader or Bluestein at 65537 on x86: measure chose Bluestein on the EPYC (rules 1.75 of its time); should the x86 Rader range stop below 2¹⁶ as NEON's does? | 2 | done: rules unchanged. EPYC Bluestein 1.49 (double) and 1.18 (float) slower than Rader at 65537 in this run, the opposite of P1's 1.75: bimodal by process (Rader's gathers and the page size); a page-robust Rader stage is the follow-up |
| [P3](experiments/P3-cost-model.md) | A calibrated cost model for long rows (DESIGN 11.2's additive model with a memory factor per cache level, fitted per machine by `fbench calibrate`): does it choose chain, two or three grid phases within 5 % of the fastest on 90 % of long rows, and can it replace X12b/X29/X32/X33/X34's thresholds? | 2 | done, first pass: rejected (model within 5 % on 100/95 % on the EPYC but 72/55 % Mac, 72/80 % Xeon, worse than the rules there; calibrated thresholds also fail on Mac and Xeon); thresholds stay; misfires left to `planning::measure` and wisdom; a denser catalogue and a set-conflict term for a second pass |
| [P4](experiments/P4-next-fast-length.md) | Are `next_fast_length`'s lengths fast on the Mac, the EPYC and the Xeon (within 1.3 of the power of two above, faster than the awkward length), and do composites with 11–23 belong in the fast set? (task 2.5) | 2 | done: every pick faster than its awkward length (0.11–0.96); composites with 11, 17, 19 or 23 removed by the rule; within 1.3 of the power of two above on 96 % (Mac) but 78 % (EPYC, Xeon), the misses short x86 lengths below 300; rule 1's target lowered to 75 % by the maintainer (2026-11-17) |
| [T1](experiments/T1-dispatch-wait-costs.md) | Dispatch, wake and wait costs per machine and runtime (libgomp, libomp, Kokkos, pool) | 3 | done: hand-off cost and claim runs adopted on Zen 3; dispatch and work costs wait for a per-lane dispatch term |
| T2 | Claiming engine against precomputed schedule with spin barriers; keep one or both? | 3 | planned |
| T3 | Kokkos: cost of one `parallel_for` per transform; `team_barrier` behaviour | 3 | planned |
| T4 | Run-to-run swings with several concurrent teams: first touch, NUMA balancing, false sharing of synchronisation words | 3 | planned |
| [T5](experiments/T5-split-thresholds.md) | Thresholds for splitting a row, by cache-group type | 3 | done: 6144 points (Zen 3 CCX), 8192 (Apple cluster, mesh); 1024 points per member |
| T6 | Hybrid processors: performance cores only, or all | 3 | planned |

## Machines

| Machine | Use | Access |
|---|---|---|
| Apple M5 Max | NEON+FCMA development and timing when quiet | local |
| Xeon Gold 6240 (AVX-512, 2 × 18 cores) | AVX-512 and Intel timing; MKL; native AVX-512 tests | Slurm, shared: small jobs (`-c 8`), never exclusive |
| EPYC 7763 cluster (Zen 3, AVX2, 2 × 64 cores) | AVX2 timing; threading at 64; remote gates (GCC 14–16, Clang 23) | Slurm (DMOG), shared: `-c 8`–16, whole nodes only for threading runs |
| Cloud: Zen 4/5, recent Intel, Neoverse N1/V1/V2 | checks per phase exit; phase 6 tuning | rented per experiment, with a ledger |
| Old x86-64 (v2 level, or AVX without AVX2) | best effort | no hardware: the `x86_v2` and `avx` targets are timed forced on the Xeon and Zen 3, and verified by emulating Nehalem and Sandy Bridge |
