# FFTN implementation and experiment plan

This is the live plan for building FFTN as designed in [DESIGN.md](DESIGN.md), under the
rules in [AGENTS.md](../AGENTS.md). Read "Status" first. Update "Status" in every working
session; update a phase when its tasks or findings change.

## Status

**As of 2026-11-27 night (lead).** Phase 1's exit is met; phases 2–4 are built, with their exits not
yet checked; phase 5's features are built and its experiments are under way; phase 6's features
are built except the release steps (rows below). **Priority (the maintainer, 2026-11-25): features first, phase 5 and then phase 6, then
refine the remaining performance.** The performance follow-ups are parked in "Tasks" below until
then. This section is also the
handoff. Lessons that generalise are in [LESSONS.md](LESSONS.md), decisions in DESIGN.md section
19, open questions in DESIGN.md section 18.2.

| Phase | State |
|---|---|
| 0 | Done. |
| 1 | Exit met (X56, 2026-11-19): the Mac M5 Max, the Xeon 6240, the EPYC 7763 and, on Spot VMs, Zen 5, Axion, Granite Rapids, Sapphire Rapids and Ice Lake; Emerald Rapids since (X64). Recent Intel's 160, 240 and 1024 improved since (X69, X70, X80). Open but optional: X2 above 1024 on the Mac, X4, X5. |
| 2 | Built: real transforms, any length, `planning::measure`, wisdom, `next_fast_length`. Open: 2.2's double c2c batch acceptance on AVX-512; X6; X13; the cost model's second pass. Exit not checked. The client catalogue's real-row gap mostly closed (X72, X75, X78; Ice Lake X82, Zen 2 X81). |
| 3 | Built: the engine, scheduler, OpenMP adapter, `thread_pool`, Kokkos (provisional API). Scheduling improved: anchored whole transforms (X73 Zen 3, X74 Apple, X79 the Xeon 6240's class), the team state on pages of its own on the Xeon 6240's class (X79, after X76's diagnosis), real rows split by complex points there (X83). Open: T2, T4, T6; 3.6 against its acceptance. Exit not checked. |
| 4 | 4.3 done. 4.1 and 4.2 built; acceptance not met case by case (large threaded planes against oneMKL, some volumes). |
| 5 | Started 2026-11-25. Built, all new API provisional with a second read: 5.1 (5.1a–5.1e; X65 judged); 5.2a (batched forms, X65's P1), 5.2b (the pipeline API and the `<C, C, C>` chain pair), 5.2c (`<T, C, T>`, `convolution_rows`), 5.2d (stored runs on pipelines; the depth-first pair, forced only, PR 154), 5.2e (fan-out, PR 162), 5.2f (the lanes pair, forced only, PR 169); 5.3a–5.3c (block callbacks on rows, every row strategy, multi-axis plans; PRs 141, 152, 153, 158; beside stored runs on planes and volumes, PR 176), 5.3d (prologue, middle and epilogue callbacks on pipelines, PR 165), 5.3e (inverse-first spectral pairs and `<C, T, C>`, PR 166), 5.3f (`group`, PR 170); 5.5 (owner-computes, `docs/fusion.md`, fbench pipeline arms, PR 168). The maintainer accepted 18.2's "Pipelines" and "Callbacks" proposals as built on 2026-11-26 (PR 171; `array_of` open). Experiments: X66 (5.3g) judged on the Xeon 6240 and the EPYC 7763 (PR 175: L2/4 callback blocks on the Xeon 6240's class; the rule-1 leads await the maintainer; the Mac leg is a daytime follow-up); X67 (5.2g, the pair rules and pipeline planning) running; next X68 (5.3h) and X9 (5.4). |
| 6 | Started 2026-11-26; steps 6.1–6.20 (phase 6 table), API proposals in DESIGN 5.4, 5.7–5.9, questions in DESIGN 18.2, "Phase 6". Built 2026-11-26/27, new API provisional with a second read: 6.1 (packaging, PR 147), 6.2 (size report, PR 146), 6.3 (r2r kind and DHT, PR 148), 6.4 (DCT/DST II and III, PR 151), 6.5 (DCT/DST IV and I, PR 155; an independent reference for every r2r kind, PR 176), 6.6 (`fftn/easy.hpp`, PR 149), 6.7 (scipy-style r2r, PRs 159 and 161; Kokkos-FFT's one-shot functions, PR 160), 6.8 (the C API for plans, PR 157), 6.9 (the C API for fusion, PR 174), 6.10 (split complex, the conversion form, PR 173; in C, PR 177 in review), 6.11 (float coverage, PR 163; X84, PR 164). Settled by the maintainer 2026-11-26 (PR 171): split complex's fused form (6.13) moves to phase 7; phase 6's cloud budget is USD 40, spent once the features are built. Left: 6.12 (r2r speed), 6.14–6.16 (machines, cloud), 6.17 (documents), 6.18–6.20 (release). |
| 7 | Not started. |

**Client catalogue** (the check of 2026-11-24, `main` d4b169f): weighted geometric mean of FFTN ÷
FFTW at one thread / at the client's per-shape widths, against targets of 0.95 / 0.90. Since then
only experiment runs have touched it; the catalogue has not been re-measured as a whole.

| Machine | 2026-11-24 | Since |
|---|---|---|
| Mac M5 Max | 0.85 / 0.80 | anchoring at 6 threads (X74): the catalogue 0.996 of its time |
| Xeon 6240 | 0.88 / 0.81 | one thread 0.896 → 0.863 of FFTW after X75 (informative run); X78's fused c2r rows, X79's anchoring (catalogue 0.942 of its time at 16 threads, 0.961 at 8) and X83's splits since |
| EPYC 7763 | 1.00 / 0.905 | one thread 0.990 → 0.933 of FFTW after X72, so it meets 0.95; X73's anchoring at 16 threads 0.786 of its time |

The tail rule for client catalogues changed on 2026-11-25 (DESIGN 1.3, decision log): a shape
fails when the whole interval of its ratio lies above 1.05 × (1 + the run's band), as in the
standard catalogues, no longer at a plain 1.02. Under 1.02 no machine met it on 2026-11-24; the
re-check under the new rule is open. Planning time, the client's criterion (4), under 10 ms on
every shape: worst Mac 2.7 ms and Xeon 5.7 ms (X71), EPYC 4.39 ms after X72 (PR 110). The
AArch64 Linux leg of the check is pending.

The cloud campaign stands at USD 16.54 of 20 (after X80; ledger `experiments/cloud-ledger-2026-11.md`).

Adoptions under the 2026-11-23 clauses, open to the maintainer's veto: X75 arm B narrowed to half
lengths above L2/8; X78's B4n defined from a diagnosis after B's judging runs (amendment 1).

### Recent experiments (newest last)

EPYC tail (X62, 2026-11-22, branch `exp/x62-epyc-tail`, PR 90): re-timed on `main` 021c117, 15 ×
2 backward is not a loss (0.89 of FFTW) and 64 × 2 backward 1.29, 64 × 16 1.18, 1018 1.13, 2048
1.11, 72/100/120 1.08 are. Adopted (the batch leaves by the maintainer's decision, 2026-11-23): `avx2` double batch leaves of 32 and 64 move
each element of their two rows when the codelet asks for it, with one inlined codelet copy (a
second copy for an odd batch's last row thrashed the op cache), and the rules take the leaf of 64
from 2 to 1024 rows, odd counts from 5: 64-point batches 0.85 of the chain's time, 64 × 2
backward ÷ FFTW 0.98, 64 × 16 0.92. Zen 3's Bluestein convolutions keep their power-of-two part
under half the L1, a stage over C copies keeping the whole line: 1013–1021 and 1018 0.93 of their
time (1018 ÷ FFTW 0.98). Left: 2048 (kernel cost), 1024 backward, 72/100/120 (placement); the EPYC
had 4 cases beyond noise against FFTW in the judging run, where `main` had 9.

Client catalogue check (2026-11-24, `main` d4b169f, branch `dev/client-check`): the client
application's 52 tracking shapes with their measured weights, FFTN ÷ FFTW (MEASURE, its own
threads), weighted geometric mean at one thread / at the client's per-shape widths on its full
budget: Mac M5 Max 0.85 / 0.80 (6 cores), Xeon 6240 0.88 / 0.81 (8) and 0.74 (16), EPYC 7763
1.00 / 0.905 (8) and 0.82 (16), against targets of 0.95 / 0.90. Every shape at the full width:
0.66, 0.64 / 0.48, 0.65 / 0.55. No machine met the tail limit as it then read (no shape above
1.02; since 2026-11-25 DESIGN 1.3 judges client shapes at 1.05 beyond noise, not yet re-scored):
real rows of 6144–20480 points on Zen 3 (1.05–1.14 at one thread; 40 % of the weight), real rows
of 32768–40960 points everywhere (1.19–1.44), c2c rows of 3072–5120 on Zen 3 (1.06–1.12),
out-of-place c2c row batches at 16 threads on Zen 3 (1.27–1.49), 16384-point rows at 4–8 threads
on the Xeon (1.07–1.16), in-place 384² planes at 8 threads on Zen 3 (1.08–1.11). Plans take 1–2
ms per shape in the median, but real rows of 18432 points and more take 10–29 ms on x86, mostly
building twiddle tables. Runs 20261124-102656-dadb, -110613-be23 (Mac), -103457-da59,
-104940-3dec, -112531-9236 (Xeon), -102950-017c, -103014-bb67, -103024-3307 (EPYC). Next, by
weight × excess: real rows on Zen 3, long real rows, wide-team row batches, twiddle generation.
The AArch64 Linux leg is pending (lead).

Planning time (X71, 2026-11-24, branch `exp/x71-twiddles`): twiddle tables by octant symmetry,
root tables shared across lengths n 2^k and the grid's roots through a `root_cache`, every
table entry unchanged bit for bit. Plan construction over the client catalogue 0.22 (Mac,
EPYC) and 0.23 (Xeon) of its time; worst shape 2.7 / 5.1 / 5.7 ms (was 9.3 / 20.9 / 26.8): the
client's criterion (4), planning under 10 ms, now holds on all three machines. Runs
20261124-221912-bb81, -222350-028f (Mac), -222427-ac05, -222427-e7b3 (EPYC), -222921-127d, -224111-b0f2 (Xeon).

X72 (2026-11-24 night, branch `exp/x72-zen3-real-rows`): the client check's largest gap, real rows of 6144–20480 points on Zen 3. Arm A: on Zen 3's class a real row's half-length transform takes the depth-first chain where a c2c row of that length would (0.761 over the 26 real rows it moves, 18432–245760); arm B: `avx2` double overrides on AMD for 3072, 3200, 4608 and 5120 (0.906 over 58 cases). Both adopted by their rules (run 20261124-223906-cac2, EPYC 7763, band ±1.00 %), no clause used. FFTN ÷ FFTW on the client's shapes in that run: 6144–9216 0.90–0.95, 10240 1.00–1.04, 18432 1.00–1.02, 20480 0.99 (r2c) and 1.05 (c2r × 3), 40960 × 6 0.88–0.99, c2c 3072–5120 0.88–0.98. Still above 1.02 (the client limit then): r2c 10240, c2r 10240 × 4, c2r 20480 × 3 (the c2r rows lost their fused pairing). The client catalogue's one-thread score on the EPYC (run 20261124-223708-ded8): 0.990 → 0.933 of FFTW (target 0.95), shapes above 1.02 24 → 7; planning time of the rows arm A moves +5–26 % (40960 × 6 20.8 → 26.2 ms). Zen 2 is reached by both keys unmeasured.

X74 (2026-11-25 night, branch `exp/x74-anchored-other`): X73's anchored whole transforms on the other measured classes. Apple (M5 Max, 6 threads): adopted, the five standard batches 0.87–0.97 of their time, the client catalogue 0.996 (one near-miss re-run, which passed; runs 20261124-233956-4feb, -233921-232b). The Xeon 6240's class: not adopted, by one case: at 16 threads the c2c batch 864 × 384 backward took 1.16 in the deciding re-run (bimodal between processes). Everything else on the Xeon gained: 8 threads, catalogue 0.951 and standard batches 0.695 with nothing slower; 16 threads 0.949 and 0.581 (8 × 4096 from 2.54 to 0.97 of FFTW); over both sockets (informative) 0.880, the out-of-place batches 0.67–0.80. The batch was diagnosed by X76 and fixed by X79, which then adopted anchoring on the class.

X75 (2026-11-25 night, branch `exp/x75-xeon-real-rows`): the client check's second gap, long real rows on the Xeon 6240. Arm A: on the Xeon 6240's class a real row's half-length transform takes the c2c depth-first chain over its range, L2/4 to 64 L2 (0.748 over the 50 real rows it moves, 32768–2^23 points and float 65536–2^20, none slower; run 20261125-023550-8fee, band ±3.87 %). Arm B: real rows alone from half lengths above L2/8 (`depth_real_min_bytes`), adopted **narrowed** under the 2026-11-23 clause: it passed its rule (0.796 over 18), but an informative re-run found float r2c 32768 (half length L2/8 itself) 1.095, so L2/8 is excluded; the declared deciding run 20261125-041502-3d25 gives 0.821 over the 14 it still moves, none slower. The Xeon's one-thread client score (run 20261125-032026-36ec, informative): 0.896 → 0.863 of FFTW; c2r 32768 1.53 → 1.07, c2r 40960 × 6 1.38 → 1.13, r2c 32768, r2c 40960 × 6 and c2r 20480 × 3 below 1.02. Ice Lake-SP keeps its plans (unmeasured). For the maintainer's veto: the narrowing.

X77 (2026-11-25 night, branch `exp/x77-real-rows-threaded`, informative): X72's and X75's real rows at the client catalogue's widths, the route against `DEPTH_REAL_MAX_BYTES=1` (the plans it replaced) and FFTW. No case is slower beyond the band. EPYC 7763 (node18 idle, bands ±1.00 %): r2c and c2r 18432 at 4 threads 0.68 / 0.80, 40960 × 6 at 6 0.66, r2c 20480 × 21 at 8 / 16 0.84 / 0.65, c2r 20480 × 3 at 4 1.015 (re-run, within the band). Xeon 6240: r2c 18432 and 32768 at 4 0.74 / 0.87, 40960 × 6 at 6 0.88–0.90, r2c 20480 × 21 at 8 / 16 0.94 / 0.88, c2r 20480 × 3 0.92, c2r 18432 1.019 and c2r 32768 1.031 (re-run; past the band, under 1.03). The c2r route pairs in a phase of its own and scales worse over a team than the breadth-first chain with its fused pairing. FFTN ÷ FFTW at those widths 0.44–0.95 on every moved row except the Xeon's r2c 40960 × 6 at 6 (1.042). Proposed: no narrowing (a plan cannot narrow by team size without breaking the serial bits); a follow-up fusing the c2r pairing into the depth-first chain's first pass, rules drafted. Also tests for X75's shipped lower limit L2/8 + 1 and single precision (the second reader's nit on PR 111). Runs 20261125-044813-2a90, -045446-be3f, -044930-3bd3, -045000-54f9, -045030-1117 (EPYC), -045511-181a, -044856-f373, -044934-2b6b, -045638-64ec (Xeon).

X76 (2026-11-25 night, branch `exp/x76-bimodal-864`, informative): X74's bimodal batch, c2c 864 × 384 backward at 16 threads on the Xeon 6240. The slow mode is the heap's: after earlier cases in an fbench process the plan's workspace is allocated from the heap right after the plan's twiddle table, and the twiddles on the team state's 4 KiB page are then loaded from other cores' modified lines on every execution (`perf c2c`); alone, the case was fast in all 144 probe processes whatever the placement, page colours, offsets or siblings. A workspace that starts a page (fbench option `WORKSPACE=page`) removes the slow mode in both arms (after one earlier case: anchored 119 → 52 µs, arrival order 104 → 70). Proposed, not adopted: the team state on pages of its own (a threading-layout change, second reader and experiment), then X74's rules for anchoring on the Xeon class again; rules drafted in the file.

X78 (2026-11-25 night, branch `exp/x78-fused-c2r`): X77's follow-up, the c2r pairing fused into the depth-first real route's first pass. Arm B (a pairing-twiddle table of h entries read at every leg, both classes) rejected by its rule: one thread EPYC 1.151, Xeon 1.064 (runs 20261125-053120-822c, -053529-aac6). Arm B4n (the twiddles factored, v_i · w_n^(leg j), on the Xeon 6240's class for half lengths from L2/4 to 16 L2: `depth_c2r_fused_min_bytes`, `_max_bytes`) adopted under amendment 1, committed before its runs: one thread 0.853 over the 23 rows it moves, none slower (run 20261125-062612-079d, band ±4.22 %, loaded); c2r 32768 at 4 threads 0.740, 40960 × 6 at 6 threads 0.886 (÷ FFTW 0.50 and 0.70). Zen 3 keeps the separate pass: every variant lost there (1.12–1.20); declared run 20261125-062442-ddf9 shows no plan moved. For the maintainer's review: B4n was defined from a diagnosis after B's judging runs (as X75's arm B); the r2c arm was not tried.

Vendor keys (2026-11-25, branch `task/vendor-keyed-intel`; the maintainer's policy: "a rule
measured on only one vendor is keyed to that vendor"): the Xeon 6240's class (X34, X39, X46, X50,
X51, X75, X78, X61's x86 half, X63), X58's Ice Lake chain and the 2 MiB-L2 key (X35, X55, X58,
X64, X69, X70) were measured on Intel's cores only and now read `machine_profile::vendor`; AMD's
cores those keys reach (Zen 4c, Zen 5c, Zen 4 VMs reporting a small L3 per CPU) and unknown
vendors keep the fields' defaults. Not a performance change: every Intel recording (Xeon 6240,
Ice Lake N2, Sapphire, Emerald, Granite Rapids) keeps its whole profile, field for field (DESIGN
13.5).

X83 (2026-11-25, branch `exp/x83-xeon-single-width`): single transforms of 4096–65536 points on the Xeon 6240's mesh at team widths 1–16. Diagnosis (run 20261125-104559-e6f5): teams of 2 lose almost everywhere below 32768 points while 4 and 8 gain from about 15625 (so no member cap per point count); X77's real rows of 18432 already run on one member (depth-first, below `split_min_points_lines`) and 32768–36864 gain from 4 (0.63–0.93); real rows of 6144–12287 on a chain lost 1.32–2.69 because the split threshold counted n real points. Adopted on the Xeon 6240 class, Intel only: such rows are split by their complex points (`split_complex_rows`); 31 moved cases at 4, 6, 8 and 16 threads 0.570, none slower beyond the band, nothing moved at one thread (runs 20261125-112330-e202, -113445-9856, -114133-f403, -114820-78dd). Left: c2c rows of 6144–16384 points (radices 8–16 lose 1.07–1.22 at 4, odd radices gain; no size threshold separates them) and r2c 16384 (1.17 at 4).

X81 and X82 (2026-11-25, cloud screen `screen-1125`, branch `exp/x81-x82-results`; rules pushed in c2f11e4 before the VMs started, PR 120 merged 09:53 UTC, after they started, without waiting for its ASan check): Google Cloud Spot, GCC 14.2. **Zen 2** (n2d-standard-16, EPYC 7B12, its own topology, `avx2`) keeps every rule Zen 3's key gives it: X72 arm A KEEP (0.745 over the 26 real rows it moves, band ±1.20 %), arm B KEEP (0.914 over 58), X73 TIE (8 unpinned threads: the client catalogue's batches 0.926 against a band of ±8.02 %, the standard batches 0.906 KEEP), none slower; no code change, and the verdicts reach other Zen 2 parts by their 16 MiB of L3 per complex (runs 20261125-095426-d1f0, -100331-a9cd, -101611-2ba1). **Ice Lake-SP** (n2-standard-16, X58's branch, `avx512`, run 20261125-095241-aafe, band ±2.73 %): X75 adopted (0.745 over 40 real rows, 24576–2^21 points) and X78 adopted (0.772 over 16 c2r rows), none slower: X58's branch now sets `depth_real_max_bytes` = 64 L2, `depth_real_min_bytes` = L2/8 + 1 and the fused c2r range L2/4 to 16 L2, the Xeon 6240's values; the rows of 2^22 and 2^23 points take the route unmeasured on Ice Lake. Plan notes moved exactly the expected cases in every run. No near-miss re-run or narrowing. Spend USD 0.40; campaign USD 15.95 of 20 before X80.

X79 (2026-11-25, branch `exp/x79-team-state-pages`, PR 115; PR 116): X76's proposal, the team state on pages of its own in the workspace. It passed its rules on the Xeon 6240 (X16: c2c 864 × 384 backward at 16 threads unimodal at its alone time, 0.739 of its median; client catalogue 0.996, X16b 0.983, X8 0.992, X8b 0.988, nothing slower) and was neutral on the M5 Max (C6r 0.999, C6b 0.998), but cost the EPYC 7763's shortest 16-thread batch (c2c 8 × 4096 1.079, E16b; two cases past the band in E16r), wherever the state's words sat in its page. Adopted on the Xeon 6240's class only, by the maintainer's narrowing ("Yes, Xeon only", 2026-11-25): `machine_profile::team_state_page` = 4096 there, every other profile keeps `main`'s layout; the Xeon's code is what its runs judged, so no new run. Anchoring on the fix (X74's rules again): not adopted by its rules, the deciding X16r2 (20261125-100316-666c) keeps the in-place 750 × 4469 pair at 1.020 (X74's 864 × 384 loss is gone: 0.740); X8r (deciding, 0.961), X8b (0.688) and X16b (0.563) pass. The maintainer then took it for the whole class, the 2 % pair accepted ("Whole class", PR 116). Slurm 17083–17092 (Xeon), 518349–518383 (DMOG); nothing left running.

X65's P1 narrowed (2026-11-26, branch `task/5.2a-narrowed`; adopted by the lead under the narrowing clause, listed for the maintainer's veto): P1, the edge driver's batched forms for plans with edges, missed its rule (EPYC 0.868 and 0.854, Xeon 0.839 against 0.80), so it was narrowed to the cases its mechanism moves, declared in X65's file before the deciding run: `machine_profile::edge_batched_forms` becomes a reach, `leaf_rows` on Zen 3's class and the Xeon 6240's (Intel's cores), where c2c rows whose line's transform is one leaf take the forms. Deciding run (E1, E2, E4, E5, one thread): such rows 0.39–0.76 of their time, geometric means EPYC 0.549 (runs 20261126-022433-ff53, -b467, -d943, -906c) and Xeon 0.584 (20261126-022459-4555, -023407-d0eb, -024130-95c7, -024809-cd64); every other case on its walked form (notes identical), none slower beyond the band; plain plans unchanged (11,508 probes on the Mac and the EPYC, the plain driver's disassembly). Excluded with their evidence: chains of 256–1024 points, real rows, planes and volumes (0.90–1.01 in P1's runs; Xeon c2c 1024 × 64 1.029), and fused rows whose chain has two passes. Other machines keep the walked form until measured. A first run of the same code was void as deciding (fbench's notes lacked the forms' lines; the backend now keeps them).

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
**Phase 1 exit check (2026-11-18 evening, `main` after X49–X57; standard catalogue, the 193 c2c
rows with every reference; tail rule beyond noise, limit 9 cases;
[X56](experiments/X56-exit-4.md)):** every mean target and the tail rule met on all three local
machines. The EPYC was timed on `main` fa155c9; the Mac and the Xeon on 71307e0, whose library
code is that of X53's merge 0508037 (X57, the only library change since, adds plans for `avx2`
double alone, so theirs are the same). EPYC ÷ FFTW 0.79, **6 (3.1 %)**: 64 × 2 backward 1.30,
64 × 16 and 15 × 2 backward 1.13, 1018 1.10, 2048 1.09, 120 1.08 (8 at the band's 1.00 % floor);
÷ FFTL 0.46, 1 (16807 1.13). Mac ÷ FFTW 0.74, 3 (1.6 %): 2 × 3 in place 1.42, 2²⁴ 1.16, 7 × 3 in
place 1.09; ÷ FFTL 0.52, 4 (2.1 %): 2²³ 1.92, 2²⁴ 1.85, 1024 × 1024 1.16, 512 × 1024 1.08. Xeon
÷ FFTW 0.66, 1 (0.5 %, 160 1.12); ÷ MKL 0.69, 2 (1.0 %: 512 × 16 1.15, 1024 × 3 in place 1.10).
Out of the lists since the check below: EPYC 1024, × 3 and × 16 (X57), 768, 512, with 1024
backward 1.08 and 100 1.07 still over 1.05 but inside this run's band; Xeon 131071, 25000, 15 × 2
backward, 240 and, ÷ MKL, 999983 (1.41 → 1.04) and 131071; Mac 32 × 4096, 64 × 2 backward. Before
X57 the EPYC had 11 (5.7 %, run 20261118-194340-02a4 on 71307e0). Its 15 × 2 backward was 0.91
there and is 1.13 here with the same plan: FFTN's tiny cases scattered from round to round in
this run.
Runs 20261119-005944-cb48 (EPYC, every round under load, band ±1.67 %), 20261118-194015-01cf
(Mac, 34 of 53 rounds under load, ±1.00 %), 20261118-194333-fbac (Xeon, 4 of 21 under load,
±1.91 %).
Cloud cores (X56, GCC 14.2, Spot VMs): Zen 5 ÷ FFTW 0.67 with 5 beyond noise, Axion 0.67 with
1, at X53's code; after X58 Ice Lake 0.67 with 0 (oneMKL 0.71, 4), Sapphire Rapids 0.70 with 7
(oneMKL 0.69, 9), Granite Rapids 0.72 with 7 (oneMKL 0.69, 4). With these **the phase 1 exit is
met on every machine checked.**

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
unchanged (run 20261118-015403-cf4b, ±1.20 %); EPYC ÷ FFTW 0.79, still 12 beyond noise
(6.2 %; 64 × 2 backward 1.29, 15 × 2 backward 1.24; run 20261118-041517-af64, ±1.00 %); Xeon
÷ FFTW 0.69, 5 (2.6 %, was 7), ÷ MKL 0.69, 2 (run 20261118-051517-636c, ±1.00 %). No verdict
changes.
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

Batch leaves (X53, 2026-11-18): the leaf of a codelet length over C rows of a batch at once,
taken by a per-length batch-count rule. Adopted: NEON float (0.61 of `main`
on the cases it changes); EPYC 15 × 2 backward 1.21 → 0.88 of FFTW (changed cases 0.81);
Xeon 15 × 2 backward 1.16 → 0.90 of FFTW (changed cases 0.69, avx512's own four rows per
register from four rows). One unchanged tiny case per x86 machine (Xeon 2 points 1.09, EPYC
4 × 4 1.24) was set aside under the code-placement rule (same plan, byte-identical code on
its path). Arm R4 then added 8, 12, 20, 25 and 32 where the screens gained (0.51-0.58 of
R on the cases it changes), once a batch's row no longer had to match a plan of that row
alone (the maintainer, 2026-11-18). Still over FFTW:
the EPYC's 64 × 2 backward (1.29; the radix-64 codelet spills over two rows), the Mac's 2 × 3
in place.

EPYC 1024 (X57, 2026-11-18): `avx2` double rows of 768 and 1024 take their two-pass plans only
while the plan's rows and output fit three quarters of the L3 (`fit_row_bytes`; in place the
rows to half the L3, 16 MiB, measured in X57's amendment 2). 1024 ÷ FFTW
1.19 → 0.99, backward 1.16 → 1.04, × 3 1.19 → 0.99, × 16 1.13 → 1.02, × 256 1.08 → 1.01; the
cases the rule moves 0.88 of their time, tracking tier 0.989, planes and volumes unchanged. The
EPYC's cases beyond noise against FFTW: 9 → 4 of 193 in the same run (limit 9); the exit check
on `main` fa155c9 found 6 and passes (X56). Left: 64 × 2 backward, 64 × 16, 1018, 2048. 72 and 120 (1.05–1.06 since 8c18dd2) are code placement: same plans and
pass code, ratios that wander from merge to merge. Aligning the `avx2` functions to 64 bytes
fixes them but moved 1024 × 3 by 4–6 % and was rejected.

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

Axis subsets (task 4.3, pull request 82, 2026-11-19): every ordered selection of one to three
dimensions of arrays of rank 1 to 4, under C, Fortran and rotated memory orders, strides of both
signs, in and out of place, c2c, r2c and c2r, against the quadruple-precision reference on every
instruction set: already correct on main through the general shape. New: `axes_c2c`,
`axes_r2c`, `axes_c2r` (provisional, awaiting the maintainer), the axes run in memory order
whatever their listing, and X61's lanes over adjacent transforms (the transformed axes all
strided, the batch's transforms one element apart). Over the 86 c2c cases of the new `axes`
catalogue (fbench's interleaved layout), one thread:
- the M5 Max: 0.48 of the old route's time, 0.86 of FFTW's;
- where the route ran unrestricted: EPYC 0.29 and Xeon 0.37, ÷ FFTW 0.80 and 0.76, ÷ oneMKL
  0.97.

A block of adjacent transforms is one task, though, and at 8 lanes the blocks lost up to 1.7
(EPYC) and 2.4 (Xeon) to the chain, whose lines a team shares. So on those x86 profiles the
route runs only where its radices are the chain's, and a team then runs the chain with the same
bits: EPYC 0.95, Xeon 0.77 of the old route at one thread, nowhere slower at either thread
count. Adopted per target in that restricted form, with a team running the chain: on the Xeon's
class (0.77) and Apple's (0.745, run 20261119-112501-c289; 0.48 unrestricted). The route is off
on the EPYC's class, whose 0.95 misses the 0.85 bar, and on every unmeasured class.

Left:
- the x86 gains back, by letting a team share a block's lines;
- real transforms whose halved axis is strided (the chain still: 1.17–1.49 of FFTW);
- small interleaved planes and volumes against FFTW;
- oneMKL's tiny columns;
- threaded interleaved planes and volumes, 2–7 times FFTW's threaded time in both routes.

Runs in X61.

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

As of 2026-11-25. Features come first (the maintainer, 2026-11-25): phase 5, then phase 6. The
acceptance gaps and performance follow-ups below wait until then unless the maintainer says
otherwise.

Done:
- Phase 0: 0.1–0.10, 0.5b, B1, B2, X15; the client-catalogue baselines on the Mac, the Xeon 6240
  and the EPYC 7763 (the check of 2026-11-24, PR 99).
- Phase 1: tasks 1.1, 1.2, 1.3 (X14), 1.5 (X1), 1.6 (X3), 1.7 and 1.7b (X12, X12b); the exit
  (X56, 2026-11-19) on eight machines and Emerald Rapids (X64); the 1-D push and the tail rounds
  (X8, X10, X18–X39, X42, X46–X51, X53–X58, X62), recent Intel's two-pass rows (X69, X70, X80).
- Phase 2: tasks 2.1, 2.3 and 2.3b (X17), 2.4 (`planning::measure`, wisdom; P1's acceptance met;
  the cost model P3 rejected on a first pass), 2.5 (`next_fast_length`, `relative_cost`, P4); the
  client's real rows (X72, X75, X77, X78, X81, X82); planning time (X71).
- Phase 3: tasks 3.1–3.3 with T1 and T5, 3.5 (Kokkos, provisional API) with T3; anchored whole
  transforms (X73, X74, X79), the team state's pages (X76, X79), single transforms on the Xeon's
  mesh (X83).
- Phase 4: 4.1 and 4.2 built (X40, X41, X43, X43b, X44, X52, X59, X63; X60 not adopted); 4.3
  (PR 82, X61; `axes_*` agreed, provisional until the fusion design has been drafted against them).
- Phase 5: the design agreed (DESIGN section 12, PR 94).
- Machine profiles: the processor's vendor (PR 97), rules measured on one vendor keyed to it
  (PR 121), topology recordings that carry it (PR 126).
- CI: a short required set on pull requests (no up-to-date rule), the full per-change set on main
  after each merge, weekly extended and emulation jobs.

Open, features first:
- 5: plan edges, then row pipelines, callbacks and spectral pairs, X9, owner-computes and the
  documents: steps 5.1–5.5 of the phase 5 table, in that order. **5.1 is in progress** (task Q,
  2026-11-25, branches `task/5.1-edges-*`; its breakdown into PR-sized steps goes into the phase 5
  table). Experiments X65–X68 are reserved.
- 6: steps 6.1–6.20 of the phase 6 table (2026-11-26). Packaging (6.1), the size report (6.2),
  the real-to-real kind (6.3) and the convenience layer (6.6) can start now; the API steps are
  built provisional on DESIGN's proposals until the maintainer answers DESIGN 18.2, "Phase 6";
  scope, spending and the release criteria wait for those answers.
- 7: on demand (FFTW3 shim, SVE, Python, more axes).

Open, acceptance and exit checks (after the features):
- 1 (optional): X2 above 1024 on the Mac; X4 on x86 and for edge passes; X5 (twiddle modes);
  float radix-25 overrides. Left beyond noise against FFTW: the EPYC's 2048, 1024 backward and
  72/100/120 (X62).
- 2: 2.2's acceptance in double c2c on AVX-512; X6 (real-input codelets), X13; the cost model's second pass and `fbench
  calibrate`; the exit check.
- 3: T2, T4, T6; 3.6 against its acceptance; the exit check.
- 4: 4.1's and 4.2's acceptance case by case (the Xeon's c2c planes from 64² and real planes
  against oneMKL, threaded real planes against oneMKL, the EPYC's 2048 × 36 at 8 lanes, the Mac's
  32²–64², 512 × 64 × 64, 64 × 64 × 512 and 128³); 4.3's leftovers (real transforms along a
  strided halved axis without lanes, more than three axes, in-place real with a strided halved
  axis); the planes and volumes exit.
- The client catalogue's AArch64 Linux leg (lead).
- CI: the Windows jobs install a pinned LLVM 23.1.5 (PR 81) because the runner image's LLVM is
  older than its Visual Studio needs.

Performance follow-ups, parked for later:
- the Xeon 6240's c2c rows of 6144–16384 points at 4 threads (radices 8–16 lose 1.07–1.22, odd
  radices gain; a radix-aware split cost, X83);
- r2c 16384 at 4–8 threads on the Xeon (1.17 at 4, X83; 1.07–1.16 in the client check);
- the EPYC's r2c 10240 and c2r 10240 × 4 (1.00–1.04 of FFTW after X72), and c2r 20480 × 3 (1.05
  at one thread in X72's run);
- the Xeon's r2c 40960 × 6 at 6 threads (1.042 of FFTW, X77);
- the Mac's real rows of 32768–40960 points (1.19–1.44 in the client check; X72 and X75 moved
  only Zen 3 and the Xeon);
- in-place 384² planes at 8 threads on Zen 3 (1.08–1.11 in the client check);
- X80's out-of-place crossing on Granite Rapids, between 180 and 320 MiB;
- why anchoring costs the in-place 750 × 4469 batch 2 % at 16 threads on the Xeon (1.020, X79);
- why a state page of its own costs Zen 3's c2c 8 × 4096 at 16 threads (1.079, X79);
- the phase 2 and phase 3 exit checks;
- a client-catalogue re-check under the 1.05 rule (DESIGN 1.3), with the Xeon and the EPYC after
  X72–X83.

Notes kept from earlier sessions:

Decided by the maintainer on 2026-11-24 (the earlier ones are in AGENTS.md section 6 and DESIGN
15.5); the decision log in DESIGN section 19 has each:

- Kokkos (task 3.5): both restrictions accepted as they are (`Kokkos::Threads` only at the
  pinned Kokkos 5.2.1; no plan on OpenMP or Threads inside a `Kokkos::Serial` kernel). Revisit
  if Kokkos gains an "inside a kernel" query.
- Task 4.3's `axes_c2c/r2c/c2r` agreed (names, the `padded` flag, batch order by input stride),
  kept provisional until the fusion design (phase 5) has been drafted against them. X61's per-
  target outcome stands; the EPYC's float 8 × 64 at 1.045 is a recorded small loss.
- Reach by cache key accepted for X58 (client Intel AVX-512 cores, Emerald Rapids) and X57
  (other `avx2` cores with a large L3); Emerald Rapids and Zen 2 go into the next cloud screen.
- X57's in-place limit (1536 rows) is to be measured on the EPYC; until then, or if it loses,
  it is capped at the 1024 rows measured. Measured (X57 amendment 2, 2026-11-24): two-pass rows
  in place lost from 18–20 MiB on the EPYC 7763 (1024 × 1280 and 1536 1.07–1.16, 768 × 2048
  1.14–1.19); capped at half the L3, 16 MiB (1024 rows of 1024, `fit_in_place_bytes`).
- X28's 128-bit leaves on `avx2` ratified. X26's prefetch gate and X30's 512 override (and X62's
  64-point batch rule) become keyed to the vendor once the machine profile reads one: Intel
  cores with `avx2` alone keep their old plans until measured. Done (task/cpu-vendor, PR 97):
  the profile reads the vendor (CPUID), and those rules and the other `avx2` rules
  measured only on the EPYC and keyed by the target alone (X42's and X54's overrides, X53's batch
  leaves) apply to AMD's cores only (DESIGN 13.5); X30's 256, a gain on the Xeon under `avx2`
  too, stays for everyone; the EPYC's plans are unchanged. Cache-keyed
  Zen 3 rules (X51 E, X59 R6, X62 C, X63) still reach any core the key matches, Intel's
  four-core client parts and small VMs included. From X64's screen, X63's `avx512` branch (the
  Xeon 6240's key) packs on Intel's cores only; AMD's Zen 4 keeps the block. Since 2026-11-25
  (task/vendor-keyed-intel) the whole Xeon 6240 class and the 2 MiB-L2 key are Intel's only.

Cloud screen X64 (2026-11-24, Spot VMs, USD 2.44; campaign 14.63 of 20): X50's depth-first
Bluestein convolutions adopted on the 2 MiB-L2 key (Sapphire Rapids 0.588 of the time over the
moved convolutions, Granite 0.564, Emerald 0.615; FFTN ÷ oneMKL 131071 0.74–0.76, 999983
0.84–0.93). Every reach screen kept the rule except X63's Xeon key on Zen 4 (now Intel only).
Emerald Rapids passes the exit check (5 rows beyond noise ÷ FFTW, 6 ÷ oneMKL, bound 9). X69
(the same day, USD 0.38; campaign 15.01 of 20) followed X64's diagnosis: on the 2 MiB-L2 key under
`avx512` in double, one-axis c2c rows of 1024, 160 and 240 take 32 × 32, 8 × 20 and 12 × 20 while
rows and output fit the L2 (0.796, 0.780 and 0.800 of the time over the 32 moved cases on
Sapphire, Granite and Emerald Rapids, none slower; ÷ FFTW 0.79–0.82, ÷ oneMKL 0.82–0.84 there).
Left: 1024 in place at 3–16 rows still 1.03–1.05 of oneMKL on Emerald. X70 (USD 0.54; campaign
15.55 of 20) raised the limits: rows and output within four times the L2 (1638 rows of 160, 1092
of 240), in place within twice the L2 (256 rows of 1024), 1024 out of place kept at the L2 (0.858,
0.860, 0.873 over the 18 moved cases, none slower). Its arm W (not judged) still gained in place
at every size tried (1024 to 2048 rows 0.88–0.93; 160 and 240 to 65536 rows 0.82–0.94) and out of
place to 16384 rows of 160 and 240 (0.85–0.94), tying or losing only at 65536 rows out of place on
Granite and Emerald (1.00–1.03): a follow-up could drop the in-place limit for these lengths.
X80 (2026-11-25, USD 0.59; campaign 16.54 of 20) did: on the key no in-place limit, and rows and
output of 160 and 240 out of place within 64 times the L2 (128 MiB), 1024 out of place still the
L2 (Sapphire 0.899, Granite 0.903, Emerald 0.894 over the 22 moved cases, to 512 MiB of rows in
place; none slower by the rules, though Granite's in-place gain fades at 320–512 MiB: 160 ×
131072 1.016, 1024 × 32768 0.996). On the moved cases FFTN ÷ FFTW 0.91–0.93 → 0.81–0.84, ÷ oneMKL
0.94–1.04 → 0.85–0.93; on Emerald three in-place batches stay at 1.03 of oneMKL. Arm W puts
Granite's out-of-place crossing between 180 and 320 MiB (a low-priority follow-up).

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
override: Intel cores without AVX-512 want other choices than Zen): keyed to the vendor since
2026-11-24 (AMD only; Intel's `avx2` keeps its old plans until measured, DESIGN 13.5); rules
of Intel's own for `avx2` wait until a wider set of machines shows how far cores of one ISA
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
| 4.3 | Arbitrary axis subsets and orders of strided arrays; real multi-dimensional transforms (done 2026-11-19, pull request 82: property tests for every selection, `axes_c2c`/`axes_r2c`/`axes_c2r` provisional, X61) | property tests against the reference |

Exit: fbench planes and volumes catalogues meet the one-thread and threaded targets.

---

### Phase 5: fusion

The design is DESIGN.md section 12, agreed by the maintainer on 2026-11-24. Steps in order, rows
first; sizes in agent sessions are estimates (15–24 in all, 4–6 more for a multi-axis pipeline).

| ID | Task | Acceptance |
|---|---|---|
| 5.1 | Plan edges (12.1): the vocabulary (scale, real and complex factors, accumulate) in every strategy the rules choose (today the factor reaches only c2c chains of one axis); stored runs in shapes (overlap check and default strides by stored extent, the optional runs argument of `axes_*`, pruning by restricted loops, then the generator's pruned codelets); `plan_edges`, `exec_args`; experiment X65. About 6–9 sessions | each operation matches "transform then separate pass" within the derived tolerance (fusing a multiply into an FMA changes last bits) and is faster where X65's rule says; canaries around stored runs; an edge that forces a slower strategy shows in `describe()` |
| 5.1a | The public types (`edge`, `factor`, `plan_edges`, `exec_args`; provisional), the plan constructor's edges and the execute twins; scales on every plan as one product; factors and accumulation on c2c rows: a complex factor of stride 1 fused into the first or last pass through an execution-time factor cursor, the rest a separate edge pass through the workspace block; such plans are chains by the rules, which `describe()` says; refusals naming the field (claimed: fusion agent, 2026-11-25) | every operation and combination on c2c rows matches "transform then separate pass" (quadruple-precision reference, derived tolerance plus a rounding per operation) on every instruction set in both precisions; factor strides of every sign and broadcasts; in place; canaries; serial, pieces, teams (split rows included) and executors bitwise alike; scales alone bitwise equal to `options::scale` |
| 5.1b | The factor and accumulation in every strategy the rules choose: the lanes (gather and scatter), the grid (its first gather and last scatter), depth-first chains (first and top passes), batch leaves, Bluestein's depth-first convolution; real transforms (real factors on the real side, any on the complex side); multi-axis plans (the input edge in the first phase, the output edge in the last, the cursor moved by the lines loops); measured planning and wisdom with edges (scratch factor arrays for the timings). In three PRs (claimed: 5.1b agent, 2026-11-25): **5.1b-1** real and multi-axis plans, every factor and accumulation an edge phase of its own over the edge side's rows, the cursor moved by its lines loops (`mark_cursors` takes those loops' steps; its batch loops are one per batch dimension by construction), the team core built with the edges (done); **5.1b-2** the row strategies (lanes, grid, depth-first, batch leaves, Bluestein's depth-first convolution, real rows' grid and depth-first chain) and the lanes over adjacent transforms, with separate passes, the fused chain only where the rules choose the chain (done); **5.1b-3** measured planning and wisdom: candidates built and timed with the edges on scratch factor arrays, wisdom keyed by the edges (done) | 5.1a's tests on each strategy, forced and by the rules, and on real and multi-axis shapes; no plan with edges is a chain unless the rules would make it one or `describe()` says why |
| 5.1c | `describe()` names the strategy the rules would have chosen when an edge forces another, and the separate passes an edge costs (task R, 2026-11-25: done) | a test per forced case |
| 5.1d | Stored runs (`side`, `stored_run`, `shape::runs`): the overlap check and default strides by stored extent, the optional runs argument of `axes_*`, zeros not read and elements outside the runs not written, pruning by restricted loops in multi-axis plans (task R, 2026-11-25: done, on the chain; in place and beside a factor refused, rules-only planning) | canaries around stored runs; property tests of runs on every axis and side against the reference |
| 5.1e | The generator's pruned codelets for the edge passes of lines with stored runs (inputs known zero not read, outputs not wanted not computed) (2026-11-26: built for c2c rows, halves of even radices, DESIGN 12.1; real rows and multi-axis lines not yet) | the generator's exact-DFT check on every pruned codelet; 5.1d's tests on them |
| 5.1f | Experiment X65 on the built edges: fused against separate passes size by size; which separate passes (real factors, strided factors, accumulation) become fused edge variants; the cost of an edge that forces a slower strategy | X65's verdict, adopted or rejected per variant |
| 5.2 | Row pipelines in convolution form (12.2): `<T, C, T>` and `<C, C, C>` forward first, the kernel factor, stored runs, fan-out, a lanes pair on strided lines; built from the Bluestein stage's convolution, with the loop over a scratch-resident line (X36b); experiment X67. About 5–7 sessions, in steps 5.2a–5.2g; the design is DESIGN 12.2, "Plan for task 5.2" | at or below 0.8 of three separate calls for L2-resident sizes |
| 5.2a | **The edge driver's batched forms and a third cursor.** Built 2026-11-26 (task T) without the third cursor and the block offset, which go to 5.2b with their first reader; the forms' reach is `machine_profile::edge_batched_forms` (`all` for tests): X65's P1 missed its rule, then was adopted narrowed, `leaf_rows` (c2c rows whose transform is one leaf) on Zen 3's and the Xeon 6240's classes, `none` elsewhere (DESIGN 12.1). No public API. The flat form and the unrolled calls (8.3) with the factor cursors moved per transform. `factor_cursors` and `edge_record::factor_step` gain the middle cursor, read from `exec_args::mid_factor`; a stage records which factor pointer takes it. Edge-driver records gain a byte offset into the workspace block. A build setting forces a plan with edges onto these forms for tests. Whether plans with edges take them by default is X65's P1, the maintainer's decision; pipelines take them from 5.2b | Plans without edges compile to exactly `main`'s code: `describe()` and `workspace_bytes()` probes, and disassembly of the plain driver functions with Clang on arm64 and x86-64. The flat and unrolled forms with cursors are bitwise equal to the walked form, with cursors moved by batch loops of strides of every sign. Plans with edges are unchanged unless P1 is adopted |
| 5.2b | **The API and the `<C, C, C>` chain pair.** Built 2026-11-26 (DESIGN 12.2, "As built"): the chain pair is two `c2c_line` chains through the workspace block; `add_convolution`'s generalisation goes to 5.2d–e with its first reader. `fftn/pipeline.hpp`, provisional and with a second reader: `pipeline_desc`, `pipeline`, `spectral_pair_rows` (forward first) and `workspace(const pipeline&, lanes)`. No callbacks or `group` (5.3), `fanout` 1 and no runs yet. Rows with a contiguous or strided line run as the chain pair over a scratch-resident line, built by `add_convolution` generalised to endpoints and the middle cursor. The prologue fuses into the forward's first pass, a complex middle factor of stride 1 into its last pass, and the epilogue into the inverse's last pass; everything else runs as edge stages on the scratch line. Scales are one product. Leaf-sized rows are a flat body of two leaves. Serial, team, executor and `execute_transforms` are built, with `transforms_of`, `describe()`, `workspace_bytes()` and `instruction_set()`, and refusals that name the field | Checked against forward plan, then the user's multiply loop, then inverse plan, computed in quadruple precision, with the tolerance derived for two transforms plus a rounding per vocabulary operation; on every instruction set, in both precisions. Lengths: 2 to the leaf limit, then codelet, mixed, generic and Bluestein lengths. Kernel strides of every sign, broadcast and per row. Also tested: canaries; serial, pieces, teams (split lines, `completion::all` and `own`) and executors bitwise alike; `describe()` names the flat form for leaf-sized rows |
| 5.2c | **`<T, C, T>` and `convolution_rows`.** Built 2026-11-26 (DESIGN 12.2, "As built"): every factor is an edge pass on the line in cache, the scales on such a pass or the c2r chain's output; plus the edge driver's row-block step for a middle pass (PR 140's latent finding). The real chain pair: the r2c pairing fused into the forward's last pass, the c2r pairing into the inverse's first, and odd n by the full-length route. The middle over the n/2 + 1 bins is an edge stage in scratch; a fused middle in the pairing passes is an X67 arm. Real prologue and epilogue factors | 5.2b's tests on real rows, even and odd, with real and complex kernels |
| 5.2d | **Stored runs and the depth-first pair.** Built 2026-11-26 (DESIGN 12.2, "As built"): runs on both chain pairs, the pruned passes on `<C, C, C>`, the depth-first pair for contiguous `<C, C, C>` lines, forced only (X67, 5.2g, decides when the rules take it; every pipeline by the rules unchanged), Bluestein through `depth_pair` (probes unchanged); runs refused in place and beside side factors. After PR 134. Runs on the line (`stored_run`, axis 0): the input's runs are read into the scratch line with zeros elsewhere, and only the output's runs are stored from it. The depth-first pair serves long lines, from `bluestein_depth` generalised: optional chirp, one input run from logical 0, one output run, a kernel table or the middle cursor. Bluestein's stage is built through the general form | 5.1d's canaries and property tests on the line's runs (valid halves, bands, runs in any order). Bluestein plans' programs and bits are unchanged (probes). The depth-first pair is tested forced and by the rules, and team-shared within a line |
| 5.2e | **Fan-out.** Built 2026-11-26 (DESIGN 12.2, "As built"): on both chain pairs and the depth-first pair (forced); a copy that writes its output itself is shared by a team, and only there does copy k's kernel apply in the inverse's first pass (fused, `<C, C, C>` with a complex kernel of stride 1; otherwise a middle pass per copy into scratch); a copy that needs a line in memory (epilogue pass, accumulation, output runs, the depth-first pair) takes the second-region path: its first stage reads the spectrum, the rest works in the block's second region, and a team runs a line's copies one after another; 18.2's proposals 5 and 6 as written. The forward spectrum stays in the workspace block. A lines loop over the copies moves the output by `fanout_out_dist` and the middle cursor by `fanout_factor_dist`. Copy k's kernel applies in the inverse's first pass, on the member's own scratch. On a depth-first pair the spectrum sits in the block's second region (5.2a's offset). Overlapping copy outputs are refused | Copies are checked against F separate inverses. Teams share the copies, bitwise alike. Canaries between copies |
| 5.2f | **The lanes pair.** Built 2026-11-26 (DESIGN 12.2, "As built (task 5.2f)"), forced only until X67 (5.2g): `<C, C, C>` rows of a batch as the lanes over rows and adjacent strided lines as lanes columns, the middle (any kernel, per line or broadcast) as a pass on the tile, the middle callback on the tile, either direction first. Lines agree bitwise with the chain pair wherever its middle is a separate pass (the scale placement and chains then match); a complex kernel of stride 1, which the chain pair fuses (forward last pass, or with a middle callback backward first pass), within the derived tolerance (FMA, no pair leaf beside a factor edge, the scale's place; DESIGN 12.2). Not taken yet (the chain pair, `describe()` says why): prologue and epilogue factors and callbacks, accumulation, runs, fan-out, the real kinds. Designed: Short rows of a batch run as the lanes over rows: one gather, the forward lanes body, the middle on the tile, the inverse lanes body, one scatter. `<C, C, C>` strided lines with an adjacent innermost batch run as the lanes over adjacent lines (9.3's block rule). The kernel is per lane or broadcast | Every row bitwise equal to the chain pair's (the lanes' contract). The lines of strided planes. Teams and executors |
| 5.2g | **Experiment X67 and the pipeline rules.** Built 2026-11-27 (task AF; PRs 172 and 179): measured planning and wisdom for pipelines (the pairs a description takes as candidates; `wisdom::add(pipeline)`, provisional); fbench's `PIPE_RUNS`, `PIPE_FANOUT`, forced pairs, interleaved lines and FFTW's `PIPELINE=separate`, catalogue `x67`. X67 judged on the Xeon 6240 and the EPYC 7763 (the Mac leg a daytime follow-up): **5.2's acceptance not met** (P/S inside L2: EPYC 1.009 double, 1.001 float; Xeon 0.765, 0.840). Adopted: the depth-first pair for contiguous `<C, C, C>` lines from 288 KiB to 2 MiB on the Xeon 6240's class (avx512) and the lanes pair for a plane's strided lines of 64–512 points on Zen 3's and the Xeon 6240's classes, both narrowed to what X67 measured; every other pipeline and every plan unchanged (probes). Losses against three separate calls (the real pair's baseline middle pass, fan-out's copies, whole lines and short-row batches against the plans' own strategies) are leads in X67's file, awaiting the maintainer's agreement (rule 2) | X67's verdict, and 5.2's acceptance |
| 5.3 | Block callbacks on edges and middles (12.3: modes, `callback_of`, per-member scratch, nested serial execution); spectral pairs (inverse first, `<C, T, C>`), `group`; the free-space stage as three objects (12.5); experiments X66 and X68. About 6–8 sessions, in steps 5.3a–5.3h; the design is DESIGN 12.3, "Plan for task 5.3". 5.3a–5.3c need only 5.2a (PR 138); 5.3d needs 5.2b–5.2c (5.2e for fan-out); 5.3e needs 5.2d | coverage, coordinate and bitwise serial/team/executor tests with pure callbacks; the free-space stage's time and DRAM traffic at 384² at full width against today's route (X68), which also decides whether a multi-axis pipeline is built |
| 5.3a | **The callback API and the callback stage on c2c rows' chains.** `fftn/callback.hpp`, provisional and with a second reader: `block_info`, `block_fn`, `callback_mode`, `callback`, `callback_of`, `along_transforms`; `edge::call`. The callback stage (baseline code, `src/core/callbacks.cpp`) with `modify`, `produce` and `consume` on c2c rows the rules make chains (other strategies: the chain, which `describe()` says, until 5.3b); the modifying-edge copy into a scratch line; the edge driver's state with the user pointers, its records with the callback geometry, and coordinates from the enclosing loops' indices; the drive policies' per-task entry with the running lane and its slot; each slot's callback area and `workspace_bytes()`; 12.3's order with factors and runs, scales formed between callbacks; refusals that name the field | Coverage: every element passed exactly once (per-element atomic counts), with and without runs. Coordinates: each block's `transform` and `origin` against the element's address. Modes against the reference in quadruple precision: `modify` (cubic, coordinate-dependent), `produce` (the input generated from coordinates), `consume` (captured blocks against the reference output). Order beside a factor and runs; scales with a callback on each side. Canaries; the input preserved out of place. `member` below the lanes, each member's scratch its own (a pattern written and checked on the next call). Serial, pieces, teams (split rows, `all` and `own`) and executors bitwise alike with pure callbacks. A nested serial plan in a callback with its workspace from `scratch`, under teams and executors (ThreadSanitizer). A `callback_of` lambda; a throwing lambda does not compile (a compile-fail test). Plans without callbacks compile to exactly `main`'s code: `describe()` and `workspace_bytes()` probes, and disassembly of the plain driver functions with Clang on arm64 and x86-64 |
| 5.3b | **Callbacks in every row strategy.** Built 2026-11-26 (DESIGN 12.3, "As built (task 5.3b)"; second read; PRs 152 and 153): every row strategy and real rows take callbacks, the lanes on the block's rows rather than the tile, batch leaves and short rows on the row blocks and the flat form where the profile's reach gives plans with edges the batched forms, a consumed output keeping a row off the grid and the depth-first chains; `callback_block_bytes`; `callback_copy_out` for the chain; measured planning with stand-ins. Designed: The lanes over rows on the tile (the gathering and scattering passes), batch leaves and short rows on 5.2a's row blocks, the grid, the depth-first chain and Bluestein's depth-first convolution (a callback stage over the line in blocks), real rows (the real side as `T`, the complex side as h + 1 bins), the lanes over adjacent transforms; `machine_profile::callback_block_bytes`; the copy into the output for c2c out of place with the strategy then in place, behind a build setting (X66's arm); measured planning with stand-ins that do nothing; wisdom keyed by the callbacks | 5.3a's tests on each strategy, forced and by the rules, on real rows even and odd, in both precisions on every instruction set; tiles never pass idle lanes; a `describe()` test per strategy (block shape, copies, sweeps beyond the cache level); no strategy forced by a callback |
| 5.3c | **Callbacks on multi-axis plans.** Built 2026-11-26 (DESIGN 12.3, "As built (task 5.3c)"; provisional, second reader pending): c2c, r2c and c2r planes and volumes, in and out of place, strided, a stride-1 batch, a volume a plane at a time, team programs; beside a factor or accumulation the callback in that side's edge phase. Callbacks beside stored runs on planes and volumes built 2026-11-27 (task AG, PR 176, in the runs' edge phase, on each run). Designed: The input callback inside the first phase's line bodies, the output callback inside the last phase's, on the block of lines a phase runs at once (lanes column blocks; the lanes over adjacent transforms with lines along transforms); team programs' narrower blocks; real planes and volumes; `consume` keeping a c2r intermediate in the workspace | 5.3a's tests on planes and volumes: c2c, r2c and c2r, in and out of place, strided, with a stride-1 batch (the free-space layout); the team program against the program bitwise with pure callbacks; `describe()` shows no edge phase of its own for a callback |
| 5.3d | **Middle callbacks and the pipelines' edge callbacks** (after 5.2b–5.2c; fan-out after 5.2e). Built 2026-11-26 (DESIGN 12.3, "As built (task 5.3d)"; provisional, second reader pending): every pair, fan-out and the depth-first pair (forced); the middle callback passed the whole line. Designed: `pipeline_desc::middle.call` (`modify`), prologue and epilogue callbacks; the middle callback first, then the middle's vocabulary in the inverse's first pass; two scale products around a middle callback; fan-out: the middle callback once per line, the epilogue per copy, `consume` receiving each line's copies in copy order; a convolution nested in a middle | Against forward plan, then the user's callback, then inverse plan, in quadruple precision with the derived tolerance; coverage per line and per copy; copy order delivered to `consume`; a nested convolution in a middle bitwise under teams and executors; pieces and `own` |
| 5.3e | **Spectral pairs, inverse first.** Built 2026-11-26 (DESIGN 12.3, "As built (task 5.3e)"; provisional, second reader pending), except the lanes pair (5.2f has not started); the depth-first pair inverse first takes read and store stages for its runs (its run passes exist in one direction each). Designed: `first = direction::backward` for `<C, C, C>` (the chain pair with its directions swapped); the `<C, T, C>` kind (5.2c's real chain pair reversed: the c2r pairing in the first chain's first pass, the r2c pairing in the second's last, odd n by the full-length route); `spectral_pair_rows` inverse first; stored runs on the half spectrum (a band); the depth-first pair in either order; the lanes pair for `<C, C, C>` | 5.2b's tests on both kinds inverse first, even and odd n, bands at the edges and in the middle of the half spectrum; forward-first pipelines' programs and bits unchanged (probes) |
| 5.3f | **`group`.** Built 2026-11-26 (DESIGN 12.3, "As built (task 5.3f)"; provisional, second reader pending): on the chain pairs of every kind, a task one group, two members loops around the middle callback; `execute_transforms` and `transforms_of` count groups (18.2 "Callbacks" 15, 16). Not built: groups with fan-out (refused, naming `desc.group`), on the lanes and depth-first pairs (the chain pair instead). "Group on plans": plans have no such field, so nothing to refuse. `pipeline::transform_count()` (provisional) gives the tasks. Follow-up: `build_work_graph(prog, n, ...)` splits by the line's n alone and ignores G, so a single group of G = 64 lines of n = 1024 is one task that a team splits only through the members loops' line phases; weigh G there. Designed: The group's batch dimension becomes a lines loop over G scratch lines in the block: the first transform per member, the middle callback once over all of them (`group`, `group_stride`), the second per member; a task is one group, teams share the per-member loops as line phases | A coupled callback (a rotation mixing two components) against the reference; coverage per group; serial, team and executor bitwise; refusals (no such batch dimension; group on plans) |
| 5.3g | **Experiment X66.** Done 2026-11-27 on the Xeon 6240 and the EPYC 7763 ([X66](experiments/X66-block-callbacks.md); the Mac leg is a daytime follow-up): fbench's `EDGE_CALL`, `CALL_SIDE`, `CALL_MODE`, `CALL_BUILD`, `CALL_LOOP` and `CALLBACK_BLOCK_BYTES` options and catalogue `x66`; `callback_block_bytes` a quarter of L2 on the Xeon 6240's class (rule 3), half the L1 elsewhere; the copy stays in the workspace block (rule 4), `callback_copy_out` deleted; `docs/fusion.md`'s advice on callbacks. Rule 1's defects (short batched rows, planes, the free-space layout, out-of-place input callbacks) are recorded as leads, which needs the maintainer's agreement. Planned: fbench's callback options (identity, cubic and `exp` per element; mode; callbacks built for the baseline and for AVX2) and catalogue `x66`; the block size per profile; the copy's destination; the defects it finds fixed, or recorded as leads with the maintainer's agreement | X66's verdict |
| 5.3h | **The free-space stage and experiment X68.** The stage as an fbench case (arms T, O, O_f and G: [X68](experiments/X68-free-space-stage.md)), catalogue `x68` at the four sizes, the DRAM traffic scripts (uncore counters on the Xeon, perf on the EPYC where allowed, a bandwidth model on the Mac) | X68's verdict: 5.3's acceptance, and whether a multi-axis pipeline is built |
| 5.4 | Experiment X9: scrambled middle and `spectral_layout` for vocabulary middles. About 1–2 sessions | adopted or rejected with numbers |
| 5.5 | Owner-computes with edges and pipelines (`pipeline::transforms_of`); `docs/fusion.md` with compiled examples, `fftn/kokkos.hpp`'s `array_of`, fbench pipeline cases with unfused baselines. About 2–3 sessions. Built 2026-11-26 (task AB; DESIGN 12.4, "As built") except `array_of`: no library change; the deal tested for plans with edges and callbacks and for every pipeline kind (`tests/api/owner_computes_test.cpp`); `examples/owner_computes.cpp` (OpenMP: a pipeline, the application's row kernel and a plan with edges in one region); `docs/fusion.md`, which 6.17 completes as the pipeline steps land (5.2f, 5.3f); fbench `PIPELINE=fused\|separate` and catalogue `pipelines`, checked against a delay kernel. **`array_of` waits for a design:** DESIGN gives it none, so it is proposed in DESIGN 18.2 ("Pipelines", 10) for the maintainer | example and test |

Experiments (rows in the index; X65–X68 were the next free IDs on 2026-11-24 and are reserved
there, but each is claimed with its file at implementation time and renumbered if another agent
took the number first):

- X65: do vocabulary edges pay, size by size, and what does an edge that forces a slower
  strategy cost?
- X66: the cost of user code outside the kernels: identity, cubic and `exp`-per-element
  callbacks against the vocabulary with a precomputed factor array; block sizes; the scratch copy
  of a modified input edge; callbacks built for the baseline and for AVX2 (decision rules
  drafted with 5.3's design: [X66](experiments/X66-block-callbacks.md)).
- X67: the row convolution against three separate calls and FFTW with a loop: rows of 2m = 18432
  to 131072, valid halves, fan-out 1, 4 and 16 (decision rules drafted with 5.2's design:
  [X67](experiments/X67-row-pipelines.md)).
- X68: the free-space stage at 128² × 864, 192² × 384, 256² × 864 and 384² × 750, three objects
  against today's route, with DRAM traffic (uncore counters on the Xeon, perf on the EPYC where
  allowed, a bandwidth model on the Mac) and the sweeps an (x, ω) slab grouping would save
  (decision rules drafted with 5.3's design: [X68](experiments/X68-free-space-stage.md)).
- X9: as above.

All on the M5 Max, the Xeon 6240 and the EPYC 7763, at one thread and at full width.

Exit: pipelines meet their acceptance numbers, and the client catalogue meets the
DESIGN.md section 1.3 row for client catalogues.

---

### Phase 6: general coverage and release preparation

The general tier of DESIGN.md section 5.6: float on every target, tuned where wanted (it is
correct and tested from phase 1 on); the convenience layer; DCT/DST I–IV and the DHT as
composed plans; the split-complex layout; the C API; Windows tuning; AArch64 Linux tuning on
Neoverse N1, V1 and V2 (cloud); Zen 4/5 and recent Intel (cloud); the documents of DESIGN.md
section 17, complete and reviewed as a whole (each phase adds its own part as it lands); the
per-release public benchmark report (one command, fixed cases and machines); packaging (CMake
package, pkg-config, an amalgamation if wanted); an informational library size report, never
a gate (the maintainer, 2026-11-17: speed takes precedence, and the present size, X27's +25 %
included, is accepted; build options such as `FFTN_OLD_X86_TARGETS`, and instantiating only
the kernels that win, remain the levers if size ever matters); the first tagged release.

Broken into steps on 2026-11-26 (features first, the maintainer's priority of 2026-11-25). The
proposed APIs are DESIGN 5.4 (convenience layer), 5.7 (real-to-real), 5.8 (split complex) and
5.9 (C API); the open points, each with a proposal, are DESIGN 18.2, "Phase 6", cited below as
Q1–Q27. Sizes in agent sessions are estimates: about 20–28 sessions for the features and documents,
plus the measurement campaigns.

**Who** decides or builds:

- **A**: agents, autonomously, under AGENTS.md (experiments by section 6, kernels with a
  second reader).
- **P**: agents build it on the DESIGN proposal, provisional in the headers and the docs and
  with a second reader, as phase 5's APIs were; the maintainer settles the API at review.
- **M**: the maintainer decides before the work starts (scope, names that freeze, money,
  release criteria); agents prepare the material.

| ID | Task | Acceptance | Who |
|---|---|---|---|
| 6.1 | **Packaging.** Install rules for the headers and `libfftn`, an exported `fftn::fftn` with `fftnConfig.cmake` and a version file, `fftn.pc` for pkg-config; test dependencies never installed (Kokkos's install rules excluded); `docs/building.md` (DESIGN 17.1) written with it. No API change. About 1 session | A CI job on Linux, macOS and Windows builds a compiled example as a separate project three ways: `find_package` against the install tree, pkg-config (not Windows), `FetchContent`; the installed tree holds no test or third-party files | A |
| 6.2 | **Size report.** `tools/size_report.py` (standard library only) reads the text size of the library's objects per instruction-set unit, and the pull-request jobs on Linux (GCC), macOS and Windows write its table to the job summary; experiment files cite it for size changes. About half a session | the table appears on a pull request; a unit test on recorded tool output; no job's status depends on it | A |
| 6.3 | **The real-to-real kind and the DHT** (DESIGN 5.7): Built 2026-11-26 (PR 148; DESIGN 5.7, "As built"; provisional, second read): the read and store stages run on the members' scratch lines, not 5.2b's block (a phase's lines run on several members at once), the store fusing the r2c's pairing with the Hartley combination; accumulation over several axes goes through the block; rules only (measured planning and wisdom left to 6.12). As planned: `r2r_kind`, `shape::kinds`, `rows_r2r`, `planes_r2r`, `volumes_r2r`, `axes_r2r`, `r2r_plan`; the read and store stages as internal edge stages over 5.2b's scratch-resident line; the DHT by an r2c of n and its store stage; several axes as one phase per axis (5.1b's machinery); edges (scale, real factor, accumulate). Uses 5.1b and 5.2b. About 2 sessions | against a quadruple-precision direct evaluation of each definition, with the tolerance derived for the underlying transform plus a rounding per stage operation, on every instruction set in both precisions, every n ≤ 512 and sampled larger, smooth, prime and Bluestein lengths; DHT twice gives n times the input; several axes against the separable reference; strides of every sign, in place, canaries; serial, pieces, teams and executors bitwise alike; refusals name the field (Q9–Q13) | P |
| 6.4 | **DCT-II/III and DST-II/III**: Built 2026-11-26 (PR 151; DESIGN 5.7, "As built (step 6.4)"; provisional, second read): the four kinds in 6.3's phase builder, any n, kinds mixed across axes; an axis of one point of `dct2`/`dst2` folds its factor 2 into the scales; the quadruple-precision reference gained every kind (`reference::r2r`). As planned: Makhoul's read stage and the twiddled store over an r2c of n (any n), the inverse route over a c2r; the sine kinds by signs and reversals in the same stages. After 6.3. About 2 sessions | 6.3's tests for each kind; `dct3` after `dct2` and `dst3` after `dst2` give 2n times the input; mixed kinds across axes | P |
| 6.5 | **DCT-IV/DST-IV and DCT-I/DST-I**: Built 2026-11-26 (PR 155; DESIGN 5.7, "As built (step 6.5)"; provisional, second read): the four kinds on 6.3's phase builder, any n (`dct1` n >= 2), mixed across axes; the odd DCT-IV by the full zero-padded c2c of 2n (no stored runs or pruned codelets: the scratch-line chain takes none; pruning left to 6.12); one-point `dct4`/`dst4` axes fold sqrt 2, `dst1` 2, into the scales; PR 151's nits folded in; 2026-11-27 (task AG, PR 176): every kind I to IV also against an independent direct sum of the definitions (reference::r2r_from_definition), n = 1 to 64 and primes to 257. As planned: the even DCT-IV by a c2c of n/2 with twiddles; the odd one by a c2c of 2n with stored runs (5.1d) on pruned codelets (5.1e); DCT-I and DST-I by the even and odd extensions; `describe()` names these routes. After 6.4. About 2 sessions | 6.3's tests for each kind; each kind twice gives N times the input; `dct1` with n = 1 refused; Q14 | P |
| 6.6 | **Convenience layer, complex and real** (DESIGN 5.4): `fftn/easy.hpp`, `context` and `thread_context()`, `fft`, `ifft`, `rfft`, `irfft` over spans (both forms), `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft2`, `irfft2`, `rfftn`, `irfftn` over views, the three norms, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`; `docs/easy.md` with a compiled example (`examples/easy.cpp`). Needs the core only, so it can run beside 6.3–6.5. About 2–3 sessions | numpy's results and shapes on recorded cases (axes defaults, negative axes, odd and even `irfft` lengths, each norm, round trips); bitwise equal to the core plan it uses; a cache hit allocates nothing (the allocation hook); two threads on their own contexts under TSan; size and axis errors name the argument (Q4–Q8) | P — built, provisional: task/6.6-easy (2026-11-26) |
| 6.7 | **Convenience layer, real-to-real and Kokkos**: Real-to-real part built 2026-11-26 (PR 159; DESIGN 5.4, "As built (step 6.7)"; provisional, second read): scipy's `int` type and norms, `norm::ortho` implying `orthogonalize=True`, one type on every axis, the weights as real edge factors kept in the cache entry along one axis and as passes over the weighted hyperplanes over several (task/6.7-ortho-weights: no array as large as the data), `dht` 1-D and scaled as a forward transform; 795 scipy results recorded by `tests/api/data/scipy_r2r_cases.py`. Kokkos part built 2026-11-26 (PR 160; DESIGN 5.4 "Kokkos" and 5.5; provisional, second read; Q3 settled by the maintainer the same day): Kokkos-FFT 2.0's one-shot functions, `hfft` and `ihfft` included, bitwise their `view_plan` on Serial and the threaded back end; the helpers left out. As planned: scipy.fft's `dct`, `idct`, `dst`, `idst`, `dctn`, `idctn`, `dstn`, `idstn` (types 1–4, norms, the orthonormal weights by a cached real edge factor) and `dht`; `fftn::kokkos::compat`'s one-shot functions as Kokkos-FFT's, over the calling thread's context. After 6.5 and 6.6. About 1–2 sessions | scipy's results on recorded cases for every type and norm; the one-shot functions against `view_plan` on the Kokkos test back ends (Q3) | P |
| 6.8 | **C API, plans** (DESIGN 5.9): Built 2026-11-26 (PR 157; DESIGN 5.9, "As built"; provisional, second read): Q17–Q22 as proposed, with the opt-in shared library `fftn_c`; every r2r kind, 6.5's included (the C tests try each and compare it bitwise with C++). As planned: `fftn/c_api.h`, `src/c_api/`; shapes of every kind built so far, options, plans, workspaces, the serial, transforms, team and executor forms, wisdom, lengths, errors; `docs/c_api.md` with a C example. The tests' build enables C. After 6.1 (the installed header); the r2r kind once 6.3 has landed. About 2 sessions | a C test compiled as C99 with the strict warnings, `-pedantic` included, gives results bitwise equal to the C++ plan of the same shape for every kind and precision; the C executor contract drives a C pool; every constructing error fills `fftn_error` with the C++ message and no exception escapes; the downstream job of 6.1 links a C program; exported symbols start `fftn_` (Q17–Q22) | P |
| 6.9 | **C API, fusion** (DESIGN 5.9): Built 2026-11-26 (PR 174; DESIGN 5.9, "As built (step 6.9)"; provisional, second read): edges, `fftn_exec_args` and the `_args` executions; C block callbacks of every mode behind one `noexcept` C++ function per element type, a callback's failure recorded through its user pointer, an exception from one ending the program (death test); `fftn_pipeline` of every kind and pair built (`<C, C, C>` either first, `<T, C, T>`, `<C, T, C>`, fan-out, runs, groups, `transform_count`); `docs/c_api.md` "Fusion" quotes `examples/c_api_fusion.c`. As planned: edges and `exec_args`, then block callbacks and pipelines in C. After 5.3; the maintainer accepted DESIGN 12's APIs as built on 2026-11-26. About 1–2 sessions | C results bitwise equal to the C++ objects for each edge operation, callback mode and pipeline kind built in C++ | P |
| 6.10 | **Split complex, conversion form** (DESIGN 5.8): Built 2026-11-26 (PR 173; DESIGN 5.8, "As built"; provisional, second read): Q2 and Q16 as proposed (the maintainer's decision of 2026-11-26: the conversion form only in phase 6); `fftn/split.hpp`; the conversion as edge passes (c2c rows) and edge phases (planes, volumes, real rows) through the workspace block, the imaginary arrays riding on the factor cursors; rows on the chain only (follow-up: the lanes over rows on the block, DESIGN 5.8 "As built"); r2c and c2r refused in place; a null imaginary array aborts naming `split_ptr::im`; `split_ptr<T>` converts to `split_ptr<const T>`; `wisdom::add` refuses split plans; no edges, stored runs, measured planning or wisdom yet. In C since 2026-11-27 (DESIGN 5.9 "Split complex", written first, provisional, 18.2 Q28–Q31 for the maintainer; `tests/api/c_split_test.*` bitwise against the C++ split plans for every kind, rows to volumes, both precisions, in the four forms): `fftn_shape::split`, `fftn_split_ptr`, `fftn_split_const_ptr`, the `_split` executions, `fftn_wisdom_add` refusing split plans. As planned: `split_complex<T>`, `split_ptr<T>`, the c2c, r2c and c2r kinds; conversion edge phases (5.1b) into and out of the interleaved form; `describe()` names the two passes. About 2 sessions | every kind and rank against the quadruple-precision reference, and bitwise equal to the interleaved plan's result where both run the same program; arrays at any relative position, strides of every sign, in place, canaries around both arrays; teams and executors bitwise alike (Q16) | P (Q2 settled 2026-11-26: the conversion form) |
| 6.11 | **Float: coverage and standing.** Coverage built 2026-11-26 (PR 163): the core suites' forcing tests (instruction sets inside each; strategies, leaves, split and interleaved layouts, pass kinds, real routes, columns, volumes, tails) are typed tests over `fftn::test::precisions` (`tests/support/precisions.hpp`), listed as `Suite/Double.Name` and `Suite/Float.Name`, as the API suites (edges, pipelines, callbacks, r2r) now are; tests that forced a path in one precision run in both; the team cases are `_f64` and `_f32`. Standing measured 2026-11-26 (X84): 0.82 (provisional), 0.80 and 0.77 of FFTW's float build on the Mac, the EPYC and the Xeon; the cases above 1.10 are listed in X84 for Q23, with 7 EPYC cases undecided. As planned: every forcing parameter of the parameterised suites (instruction set, strategy, layout, pass kind, edges, pipelines) checked to run in float, and the gaps filled; an experiment (ID at claim time) measures float's standard catalogue at one thread on the M5 Max, the Xeon 6240 and the EPYC 7763 against FFTW's float build. About 1 session plus the runs | the suite listing shows each parameter in both precisions; the verdict lists the cases above 1.10 of FFTW, which the maintainer may name for tuning (Q23) | A (tuning: M) |
| 6.12 | **Real-to-real speed.** Claimed by task AK, 2026-11-27. Infrastructure built 2026-11-27 (no judging): fbench's kind `r2r` (the harness's direct sums of FFTW's definitions and the kinds' energy identity as its check), the FFTN, FFTW (`fftw_plan_many_r2r`, guru for strided axes) and pocketfft (`dct`, `dst`, `r2r_separable_fht`; now also single precision and the interleaved layout) backends, the `r2r` catalogue (946 cases), `bench.r2r_smoke`; experiment X85 running (arms D and L built behind profile fields, off by default; F drafted). As planned: fbench backends for FFTW's r2r and pocketfft's DCT and DST, an `r2r` catalogue; an experiment on the read and store stages fused into the first and last passes against separate stages, the lanes over adjacent lines for strided r2r axes, and a cheaper DCT-I/DST-I route judged on accuracy too. After 6.5. About 2 sessions | the experiment's verdict; Q15's target, once settled | A |
| 6.13 | **Split complex, fused**: the `SP` load and store in the first and last passes (lanes, leaves, chain ends, the grid's gather and scatter); on `avx2` and `avx512` a split run storing `SP` directly; an experiment against 6.10. Kernel work: second reader. Only if the maintainer keeps it in phase 6 | 6.10's tests on each forced strategy; the experiment's verdict | Moved to phase 7 (Q2, 2026-11-26) |
| 6.14 | **AArch64 Linux on Neoverse N1, V1 and V2** (cloud Spot VMs): the standard, planes and volumes and threads catalogues against FFTW's NEON build and Arm Performance Libraries (the fbench backend added if missing); machine keys where a family needs its own; the client catalogue's AArch64 leg (pending since 2026-11-24). Spending: Q24 | DESIGN 1.3's AArch64 row (≤ 0.90 of FFTW, ≤ 1.00 of Arm PL) on each core, or the misses recorded with their experiments | A (spending: M) |
| 6.15 | **Zen 4/5 and recent Intel** (cloud): the same catalogues, with MKL on Intel; the reach-rule screens of AGENTS section 6 for every family a rule reaches; the planes and volumes gap first (X31 measured Zen 5 at 1.63 and Granite Rapids at 1.48 of FFTW, before phase 4). Spending: Q24 | DESIGN 1.3's Zen 4/5 and Intel rows, or the misses recorded | A (spending: M) |
| 6.16 | **Windows.** fbench on Windows (timers, process rounds, the backends that build there); the standard and threads catalogues on Windows x64 with clang-cl against Linux on the same instance type; Windows on Arm correctness in CI, timing when a machine exists. The machine: Q24 | Windows ÷ Linux within the run's band as a geometric mean at one thread, no case beyond 1.05 of Linux beyond noise; no threads-catalogue case slower than serial | A (machine: M) |
| 6.17 | **Documents complete** (DESIGN 17): `docs/guide.md`; the real-to-real formulas and the split layout in `docs/conventions.md`; `docs/fusion.md` (from 5.5); README from "planned" to what exists; every example compiled; read as a whole by a second reader, then the maintainer. After the features | the snippet check and the Doxygen lint pass; no public declaration undocumented; the maintainer's review | A (review: M) |
| 6.18 | **The release report** (DESIGN 16.4): `fbench report --release <version>` writes `bench/reports/<version>.md` from stored runs of a fixed `release` catalogue on fixed machines, with every competitor's build and the run IDs; the case set and machines are Q26 | the same run IDs regenerate the same file; `bench/README.md` documents the set | A (set and machines: M) |
| 6.19 | **Amalgamation**: proposed against for the first release (Q25; DESIGN 14.3) | the maintainer's decision recorded | M |
| 6.20 | **First tagged release**: the checklist from Q27; every provisional API confirmed or listed in the release notes; `CHANGELOG.md`; the version in `config.hpp` and CMake; the report of 6.18 committed; tag. Last | the maintainer's go | M |

Order and parallelism: 6.1, 6.2, 6.3, 6.6 and 6.11 can start at once; 6.3 → 6.4 → 6.5 → 6.12;
6.5 and 6.6 → 6.7; 6.1 → 6.8; 5.3 → 6.9; 6.10 needs none (Q2 settled 2026-11-26; 6.13 is phase 7's). The campaigns 6.14–6.16
follow the features under the maintainer's priority, but need no feature: they can run whenever
the maintainer releases the spending. 6.17 and 6.18 come after the features; 6.20 is last.

Phase 5 underneath: the real-to-real routes use 5.2b's scratch-line programs, 5.1b's edge
phases and cursors, and 5.1d's runs with 5.1e's pruned codelets (the odd DCT-IV); the
convenience layer's orthonormal DCT uses 5.1b's real factors; split complex's conversion form
is two edge phases (5.1b); the C API's fusion step waits for 5.3 and for the review of DESIGN
12's provisional APIs; 6.17 takes `docs/fusion.md` from 5.5.

Exit: the steps above done or moved by the maintainer's decision; DESIGN 1.3's targets on the
AArch64 Linux, Zen 4/5 and Intel rows measured, met or with their misses recorded; Windows
within the band of Linux on the same hardware; the release criteria of Q27 met and the first
version tagged.

### Phase 7: on demand

FFTW3-compatible shim; SVE (experiment X11); Python binding; more than three transformed
axes; Rader; SME on Apple M4 and later for batches of short transforms (speculative).
Proposed for here by phase 6's plan (DESIGN 18.2, "Phase 6"): real-to-real pipelines (Q13), a
Fortran module over the C API (Q22), package-manager recipes (Q25), and the fused split-complex
form (6.13, moved here by the maintainer on 2026-11-26, Q2).

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
| X9 | Scrambled intermediate order for vocabulary middles of row pipelines (task 5.4) | 5 | planned |
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
| [X59](experiments/X59-real-planes-mkl.md) | Threaded real planes on the Xeon 6240 against oneMKL (X52 left its 8 threads 2.5–3.9 times faster than FFTN's 8 lanes): where the time goes at 1–8 lanes, then the arms that follow | 4 | done: T (a column phase's narrower last block joins its phase) adopted everywhere (Xeon 0.911, EPYC 0.978, Mac 0.998 over the real planes that split); R6 (fair runs for small shares of columns whose lines share cache lines) on the EPYC (0.909, 512² 0.70) and the Mac (0.979), not the Xeon (1.017); the Xeon's gap is mostly one-thread speed (c2r 1.2–1.7) and the cost of moving planes between L2s |
| [X60](experiments/X60-c2r-two-arrays.md) | Out-of-place c2r without a third array (X59: the half spectrum in a workspace block gives twice oneMKL's L2 misses): does packing the intermediate in the output make c2r rows, planes and volumes faster, where, one thread and threaded? | 4 | done: not adopted. The packed intermediate won at one thread (Xeon 6240 0.82 over 16 cases from its L2; EPYC 7763 0.91 past two thirds of its L3) but lost at 8 lanes (Xeon volumes 1.06–1.09, EPYC 0.970 against 0.97) and on the M5 Max's wide planes (1.07–1.14); code removed, follow-ups: planes only on the Xeon class, plans without a team program on Zen 3 |
| [X51](experiments/X51-depth-2.md) | The depth-first chain's loose ends after X46: Xeon 25000 (no split with C = 4) and 100000, double rows beyond 4 L2 (two tops), single precision on the Xeon and the Mac, the Mac's 24576 and 49152, the EPYC's tops of 4 | 1 | done: P (a partial first-pass block) and T (third top 10: Xeon 25000 0.75, 1.35 → 1.00 of FFTW), L (Xeon to 64 L2: 524288–2²² 0.68–0.75), F (Xeon float: 0.687), M (Apple mixed rows to 16 L1: 0.735), E (EPYC mixed rows, top 4: 0.730) adopted; 100000 unchanged (no split wins); Granite Rapids not measured (cloud run not made) |
| [X50](experiments/X50-primes.md) | The Xeon's large primes against MKL (999983 1.43, 131071 1.33): a Bluestein stage whose convolution runs as two depth-first chains (X46), a rule on the convolution's length; EPYC 65537's Rader/Bluestein bimodality (P1b) | 1 | done: Bluestein convolutions of one copy on two depth-first chains with fused chirp passes, where a row of M would be depth-first (Xeon, Apple) and at any M beyond on the Xeon; Xeon 8179–999983 0.56–0.73 of main (÷ MKL 131071 1.34 → 0.84, 999983 1.44 → 1.05), Mac 4813–5119 0.86; EPYC 65537's flip not reproduced (Rader 1.5× faster in 24 processes), no Rader change; library +6.2 % Mac, +3.5 % x86 |
| [X54](experiments/X54-radix64-avx2.md) | Can the radix-32 and radix-64 codelets be ordered so that GCC spills far less of them on `avx2` (a list scheduler against a model register file; bits identical), and does that let two-pass plans of 768, 1024, 2048 win at both batches on the EPYC so the planner takes them? | 1 | done: A (fitted order, bit-identical), B (768) and C adopted by the maintainer's decision against rule (b)'s letter (80 at 1.035 against 1.034, byte-identical code at a moved address): radix-32/64 plans 0.975 on the EPYC, tracking EPYC 0.994, Xeon 0.999, Mac 0.999; B (two-pass `avx2` double overrides) adopted for 768 (0.84; ÷ FFTW 1.08 → 0.91), 1024 dropped by rule (c) (0.84–0.95 at batches 1–256, 1.25 at 1024 × 1024); C (`avx2` float 512, 640, 768, 2048, 4096) adopted (0.898); 2048 double and 64 × 16 not moved |
| [X53](experiments/X53-two-row-leaves.md) | Batch leaves: C short rows of a batch side by side, one row per lane, transposed in registers (15 × 2 and 64 × 2 backward at 1.15–1.3 of FFTW on x86): for which lengths and batch counts? | 1 | done: adopted on NEON float (changed cases 0.61 of `main`), avx2 (EPYC 0.81, 15 × 2 bwd 1.21 → 0.88 of FFTW) and avx512 (Xeon 0.69, 15 × 2 bwd 1.16 → 0.90); one unchanged tiny case per x86 machine past the band set aside by the code-placement rule (same plan, byte-identical code); R4 adds 8, 12, 20, 25, 32 (0.51-0.58 of R); 64 × 2 not closed (EPYC 1.29 of FFTW; 1.18 with X54's codelet, not taken) |
| [X55](experiments/X55-gnr-depth.md) | Should the depth-first rule reach Granite Rapids (a 2 MiB private L2, outside X46's branch), and with what range and tops? (X48: its long rows over 1.05 of FFTW and MKL) | 1 | done: G adopted (0.688 over the rows it changes, tracking 0.940; long rows ÷ FFTW 0.70–0.83) |
| [X61](experiments/X61-adjacent-transforms.md) | Strided axes of an array (a batch dimension one element apart, varying fastest) as lanes over blocks of adjacent transforms, instead of each transform through the chain | 4 | done: adopted, restricted to radices the chain shares with a team running the chain, on Apple's class (0.745 of the chain's time over 86 cases) and the Xeon 6240's (0.77); off on the EPYC's class (0.95 misses the bar) and unmeasured ones. Unrestricted: 0.29 / 0.37 at one thread, but up to 2.4 slower at 8 lanes. Follow-up: a team sharing a block's lines |
| [X63](experiments/X63-c2r-planes-packed.md) | X60's follow-up: the packed c2r intermediate for planes only (volumes lost at 8 lanes on the Xeon), on the Xeon 6240 class (`avx512`, from its L2) and Zen 3 (`avx2`, from half its L3): does it pass at one thread and at 8 lanes? | 4 | done: adopted on the Xeon 6240 class (0.807 at one thread, 0.824 at 8 lanes, 31 planes from 1 MiB) and Zen 3 (0.837 and 0.767, 9 planes from 16 MiB); volumes, Apple and unmeasured classes keep the block; float planes pass too (0.59–0.81) |
| [X64](experiments/X64-cloud-screen-1124.md) | Cloud screen: do the rules that reach Zen 2, Ice Lake-SP, Zen 4 and Emerald Rapids by cache key hold there, and do X50's depth-first convolutions win on recent Intel against oneMKL? | 1 | done: X50's convolutions adopted on the 2 MiB-L2 key (Sapphire 0.588, Granite 0.564, Emerald 0.615; 131071 ÷ oneMKL 1.21–1.37 → 0.74–0.76); Zen 2, Ice Lake-SP and Emerald keep the rules they reach; Zen 4 loses X63's Xeon key (Intel only, PR 97); Emerald's exit check passes |
| [X65](experiments/X65-edges.md) | Fusion: do vocabulary edges (scale, factors, accumulate, stored runs) pay against a separate pass, size by size, and what does an edge that forces a slower strategy cost? (task 5.1f; stored runs wait for 5.1d–e) | 5 | done (first pass, task S, 2026-11-25): edges as built cost more than the plain plan plus the user's own loop on small batched rows (fused or not: such plans lose the batched program forms), on c2c chains' separate passes and on the Xeon's c2r rows that lose the depth-first chain; accumulation even or better; a vectorised separate pass rejected (0.99); plans without edges unchanged. No variant adopted; proposals P1–P5 (batched forms first) wait for the maintainer; stored runs wait for 5.1d–e. P1 built and judged in task 5.2a (2026-11-26): not adopted (EPYC 0.868 and 0.854 against 0.80); then adopted narrowed to c2c rows on one leaf on the EPYC's and the Xeon's classes (deciding run: EPYC 0.549, Xeon 0.584 over the rows it moves, every other case on its walked form, none slower) |
| [X66](experiments/X66-block-callbacks.md) | Fusion: block callbacks outside the kernels (identity, cubic, exp per element) against the user's loop and the vocabulary with a precomputed factor; block sizes per profile; the modifying-edge copy (task 5.3g; rules drafted 2026-11-26) | 5 | done |
| [X67](experiments/X67-row-pipelines.md) | Fusion: row convolution pipelines against three separate calls and FFTW with a loop; which pair (chain, depth-first, lanes) by length, and which fused middles (task 5.2g; rules drafted 2026-11-25, amended before the first judging run 2026-11-26) | 5 | done (task AF, 2026-11-27): acceptance not met (P/S inside L2 EPYC 1.009/1.001, Xeon 0.765/0.840); adopted the depth-first pair on the Xeon 6240's class and the lanes pair for strided lines on Zen 3's and the Xeon's; losses recorded as leads |
| [X68](experiments/X68-free-space-stage.md) | Fusion: the free-space stage as three objects against today's route, time and DRAM traffic; does an (x, ω) slab grouping justify a multi-axis pipeline? (task 5.3h; rules drafted 2026-11-26) | 5 | planned |
| [X69](experiments/X69-intel-radices.md) | On the 2 MiB-L2 key under `avx512` (Sapphire, Emerald, Granite Rapids), one-axis c2c rows of 1024, 160 and 240 as two passes (32 × 32, 8 × 20, 12 × 20) while rows and output fit the L2 (X57's fit overrides on that key): do they pass X64's adopt rule on Sapphire and Granite and its reach rule on Emerald? | 1 | done: adopted (Sapphire 0.796, Granite 0.780, Emerald 0.800, none slower; ÷ FFTW 0.79–0.82 and ÷ oneMKL 0.82–0.84 on the moved cases) |
| [X70](experiments/X70-large-l2-limits.md) | X69's follow-up on the 2 MiB-L2 key under `avx512`: rows of 1024 in place within twice the L2 and rows of 160 and 240 within four times the L2 (in place twice), 1024 out of place keeping the L2 (`fit_long_row_bytes`): do they pass X69's adopt rule on Sapphire and Granite and its reach rule on Emerald, and where do the gains stop? | 1 | done: adopted (Sapphire 0.858, Granite 0.860, Emerald 0.873 over 18 moved cases, none slower); two-pass still won past the new limits in place (arm W) |
| [X71](experiments/X71-twiddle-generation.md) | Planning time: twiddle tables by octant symmetry, root tables shared across lengths n 2^k, the grid's roots through a cache, all bit for bit the old tables: do the client catalogue's long real rows plan in under 10 ms on the Mac, the EPYC and the Xeon? | client (4) | done: adopted (client catalogue plan time 0.22–0.23 of before; worst shape Mac 2.7, EPYC 5.1, Xeon 5.7 ms; tables unchanged bit for bit) |
| [X72](experiments/X72-zen3-real-rows.md) | Zen 3, real rows of 6144–20480 points (the client check's largest gap): the half-length transform on the depth-first chain where a c2c row of that length takes it (arm A), and three-pass `avx2` overrides for 3072, 3200, 4608, 5120 on AMD (arm B): do they pass X51's adopt rule over the cases they move? | 1 | done: both adopted (A 0.761 over 26 real rows, B 0.906 over 58 cases, none slower; the client's real rows of 6144–40960 from up to 1.44 to at most 1.05 of FFTW) |
| [X73](experiments/X73-anchored-runs.md) | Zen 3's class at 16 unpinned threads: out-of-place c2c row batches 1.27–1.49 of FFTW because rows change core complex between executions; does anchoring each member to a fixed range of its group's whole transforms (`machine_profile::anchored_runs`) close it without slowing anything at 8 threads? | 3 | done: adopted on the `zen3_like` key (16 threads: the six out-of-place batches 0.32–0.38 of their time, ÷ FFTW 0.99–1.51 → 0.34–0.45; client catalogue 0.786; 8 threads 0.838, nothing slower) |
| [X74](experiments/X74-anchored-other.md) | X73's anchored whole transforms (`machine_profile::anchored_runs`) on the Xeon 6240's class (8 and 16 threads) and Apple (6 threads): do they help, or at least not hurt, over the client catalogue's batches and the standard batches? | 3 | done: adopted on Apple (standard batches 0.929, client catalogue 0.996, after a near-miss re-run); not on the Xeon 6240's class (16 threads: c2c 864 × 384 backward 1.16 in the deciding re-run), where all else gained (8 threads: catalogue 0.951, standard batches 0.695; 16: 0.949, 0.581) |
| [X75](experiments/X75-xeon-real-rows.md) | The Xeon 6240's class, long real rows (the client check's second gap): the half-length transform on the c2c depth-first chain over its range (arm A, L2/4 to 64 L2) and from L2/8 (arm B), as X72 on Zen 3: do they pass X72's adopt rule over the cases they move? | 1 | done: A adopted (0.748 over 50 real rows, 32768–2^23), B adopted narrowed to half lengths above L2/8 (0.821 over 14, 18432–24576); the client's 32768 and 40960 × 6 from 1.29–1.53 to 0.99–1.13 of FFTW |
| [X76](experiments/X76-bimodal-864.md) | The Xeon 6240 at 16 threads: what makes c2c 864 × 384 backward bimodal between fbench processes in both claiming rules (X74's deciding case)? Placement, page colours, offsets, frequency, siblings, the plan, the heap? | 3 | done (informative): the heap; after earlier cases the workspace follows the plan's twiddle table, whose lines on the team state's page are then loaded from other cores' modified lines (`perf c2c`); a page-aligned workspace removes it in both arms (A 119 → 52 µs, M 104 → 70); fix and anchoring rules drafted |
| [X77](experiments/X77-real-rows-threaded.md) | X72's and X75's real rows (the half-length transform on the c2c depth-first chain) at the client catalogue's thread widths (4, 6, 8, 16) on the EPYC 7763 and the Xeon 6240: is the route still faster than the plans it replaced, and how do the rows stand against FFTW? Informative | 1 | done: no case slower beyond the band; r2c rows and batches keep the gain at 4–16 threads (0.65–0.94), c2r rows at 4 threads keep less (0.79–1.03; three of five lose it, at most 1.031); no narrowing proposed, a fused c2r pairing in the depth-first first pass proposed as a follow-up |
| [X78](experiments/X78-fused-c2r-pairing.md) | X77's follow-up on Zen 3's and the Xeon 6240's classes: the c2r pairing fused into the depth-first real route's first pass (no separate pass, no workspace block): faster at one thread and at the client's widths? | 1 | done: B (a twiddle table of h entries) rejected (EPYC 1.151, Xeon 1.064); B4n (twiddles factored over the legs, Xeon 6240 class, half lengths L2/4–16 L2) adopted by amendment 1 (one thread 0.853 over 23 rows, c2r 32768 at 4 threads 0.740, 40960 × 6 at 6 0.886); Zen 3 keeps the separate pass (every variant 1.12–1.20) |
| [X80](experiments/X80-large-l2-limits-3.md) | X70's follow-up on the 2 MiB-L2 key under `avx512`: no in-place limit for the key's two-pass rows (1024, 160, 240) and rows and output of 160 and 240 out of place within 64 times the L2 (`fit_out_of_place_bytes`), 1024 out of place keeping the L2: do they pass X70's adopt rule on Sapphire and Granite and its reach rule on Emerald, and where does the out-of-place gain stop? | 1 | done: adopted (Sapphire 0.899, Granite 0.903, Emerald 0.894 over 22 moved cases to 512 MiB in place, none slower); arm W puts Granite's out-of-place crossing between 180 and 320 MiB; ÷ FFTW 0.81–0.84, ÷ oneMKL 0.85–0.93 on the moved cases |
| [X79](experiments/X79-team-state-pages.md) | The team state on pages of its own inside the workspace (X76's proposal): does it remove the Xeon 6240's slow mode of c2c 864 × 384 at 16 threads at no cost elsewhere (Xeon 8 and 16, EPYC 16, Mac 6 threads)? Then X74's anchoring rules on the Xeon 6240's class again, on the fix | 3 | done: the fix adopted on the Xeon 6240's class only, by the maintainer's narrowing (`team_state_page`; its rules failed on the EPYC 7763: c2c 8 × 4096 at 16 threads 1.079); on the Xeon it removes the slow mode (864 × 384 0.739, unimodal), the Mac is neutral; anchoring on the fix failed its rules by one pair (deciding X16r2: in-place 750 × 4469 1.020; X8r, X8b, X16b pass) and was adopted on the whole class by the maintainer's decision (catalogue 0.942 at 16 threads, 0.961 at 8) |
| [X83](experiments/X83-xeon-single-width.md) | The Xeon 6240's mesh: single transforms of 4096–65536 points gain little from a team of 4 (X77: single real rows of 18432–32768 points 0.76–1.01 of their one-thread time). Where does a team of 2 or 4 stop paying, and which scheduling rule (split threshold or member cap per point count) uses fewer members there? | 3 | done: adopted on the Xeon 6240 class (Intel): real rows on a chain split by their complex points (`split_complex_rows`); 31 moved cases 0.570 at 4–16 threads, none slower; teams of 2 lose below 32768 points, so no member cap; c2c rows of 6144–16384 left (radix-dependent) |
| [X81](experiments/X81-zen2-screen.md) | Zen 2 (Google Cloud n2d, Rome), which Zen 3's key reaches unmeasured: does it keep X72's real rows on the depth-first chain (arm A), X72's AMD `avx2` overrides for 3072, 3200, 4608 and 5120 (arm B) and X73's anchored runs at 8 threads, or lose and get its own key (16 MiB of L3 per complex)? | 1, 3 | done: Zen 2 keeps all three (X72 A KEEP 0.745 over 26, B KEEP 0.914 over 58, X73 TIE 0.926 at a ±8.02 % band and KEEP 0.906 on the standard batches); no code change |
| [X82](experiments/X82-icelake-real-rows.md) | Ice Lake-SP as Google Cloud's N2 reports it (X58's branch: large L3 per CPU), which has the Xeon 6240's c2c depth-first chain but not its real-row rules: should it take X75 (real rows' half length on the chain, above L2/8 to 64 L2) and X78 (the fused c2r pairing, L2/4 to 16 L2)? | 1 | done: both adopted on X58's branch (X75 0.745 over 40 real rows, X78 0.772 over 16 c2r rows, none slower); 2^22 and 2^23 reached unmeasured |
| [X84](experiments/X84-float-standing.md) | Float's standing (step 6.11): FFTN ÷ FFTW's single-precision build over the standard catalogue in float at one thread on the M5 Max, the EPYC 7763 and the Xeon 6240; which cases are above 1.10, for the maintainer to name for tuning (Q23)? | 6 | done, with caveats: FFTN ÷ FFTW 0.819 (Mac, provisional: net battery discharge), 0.796 (EPYC), 0.774 (Xeon, concurrent builds); c2c rows 0.69–0.70; 54, 80 (plus 7 undecided: the deciding re-run ran under load) and 94 cases above 1.10, chiefly tiny real rows and planes, real volumes, and on x86 short c2c batches and smooth mid rows; follow-ups: a quiet EPYC deciding run, a Mac run with a flat battery; tuning waits for Q23 |
| [X85](experiments/X85-r2r-speed.md) | Real-to-real speed (step 6.12): the folded DCT-I and DST-I of odd n (a real DFT of n ∓ 1, judged on accuracy too), the strided axes of planes and volumes as lanes over adjacent lines, the read and store stages fused into the passes (drafted); the standing against FFTW and pocketfft (Q15) | 6 | running: rules committed; standing (EPYC, quick tier) FFTN ÷ FFTW 0.784, DCT-III/DST-III and planes behind |
| [X56](experiments/X56-exit-4.md) | The phase 1 exit check on `main` after X49–X57: EPYC at fa155c9, Mac and Xeon at X53's code (71307e0), then the cloud cores | 1 | done: the phase 1 exit met on all nine machines (local: EPYC ÷ FFTW 6 of 193; cloud: Zen 5 5, Axion 1, Granite Rapids 7, Sapphire Rapids 7, Ice Lake 0 after X58; oneMKL at most 9) |
| [X57](experiments/X57-epyc-1024.md) | The EPYC's last exit-check cases: are 72 and 120's 5–6 % since 8c18dd2 a regression, and can 1024's radix choice see the plan's data? | 1 | done: 72 and 120 are code placement; B adopted (768 and 1024 two-pass rows while rows and output fit 3/4 of L3: moved cases 0.88, tracking 0.989, EPYC beyond noise 9 → 4 of 193); C (64-byte `avx2` functions) rejected (0.999, 1024 × 3 1.04–1.06); amendment 2: B's in-place limit measured, lost from 18 MiB, capped at half the L3 (16 MiB, 1024 rows of 1024) |
| [X58](experiments/X58-intel-keys.md) | Recent Intel keys from X56's screen: the depth-first chain on Ice Lake VMs (private L2 between 1 and 2 MiB), and no batch leaf of 12 on a private L2 of 2 MiB or more | 1 | done: both adopted. Ice Lake's changed rows 0.596 of their time (long rows ÷ FFTW 1.13–1.53 → 0.64–0.79), tracking 0.929; 12 × 64 and 12 × 1024 0.867 (Granite Rapids) and 0.892 (Sapphire Rapids) |
| [X62](experiments/X62-epyc-tail.md) | The EPYC's remaining exit cases after X56 (64 × 2 backward, 64 × 16, 1018, 2048, 120, 15 × 2 backward): which are real, and what moves them? | 1 | done: K3 (avx2 double batch leaves of 32/64 moving each element, one codelet copy) with L (the leaf of 64 from 2 to 1024 rows, odd from 5) adopted (maintainer's decision, 2026-11-23: K3's rule missed 64 × 3, which L's limits exclude): 64-point batches 0.85 of the chain, 64 × 2 backward ÷ FFTW 1.29 → 0.98, 64 × 16 1.16 → 0.92; C (Zen 3's Bluestein limit at half the L1, whole lines over C copies) adopted: 1018 1.05 → 0.98; 2048 kernel cost, 72/100/120 placement, 15 × 2 backward not a loss; EPYC beyond noise 9 → 4 in one run |
| [P1](experiments/P1-rules-vs-measure.md) | How often is the rules' choice within 5 % of `planning::measure`'s per machine (task 2.4's acceptance: 90 % of catalogue cases), which rules miss, and does measurement recover a misfiring profile (Granite Rapids' fields forced on the Xeon)? | 2 | first pass done: rules within 5 % of measure on 96.7 % of the tracking tier on the EPYC, 77 % on the Xeon (about 85 % without noise), Mac inconclusive under load; measure never slower by fbench; Granite Rapids' fields forced on the Xeon recovered in 10 of 11 cases (the miss a budget defect, fixed); planes' radices the repeat miss. Second pass (73eb719, placements per round, median): Mac 93.4 %, EPYC 95.1 %, Xeon 86.9 % (long real rows, planes, 2²², 2431 repeat); misfire 9 of 9 after the budget fix; EPYC r2c 65536 measure 1.18 slower (power-of-two chains bimodal by page backing); EPYC 65537 Bluestein 1.75 faster than the rules' Rader |
| [P2](experiments/P2-measure-budget.md) | Measurement budget: how many candidates and how long before the choice stops improving; the quick and full efforts' budgets | 2 | first pass done: Q within 0.999–1.014 of M; no winner above rank 7; plan construction dominates large transforms (2²² 12.5 s for M); choices unstable under load (43–66 % of cases one program over 12 rounds), so budgets wait for a sampling revisit; indicated quick 6, measure 8 |
| [P1b](experiments/P1b-rader-65537.md) | Rader or Bluestein at 65537 on x86: measure chose Bluestein on the EPYC (rules 1.75 of its time); should the x86 Rader range stop below 2¹⁶ as NEON's does? | 2 | done: rules unchanged. EPYC Bluestein 1.49 (double) and 1.18 (float) slower than Rader at 65537 in this run, the opposite of P1's 1.75: bimodal by process (Rader's gathers and the page size); a page-robust Rader stage is the follow-up |
| [P3](experiments/P3-cost-model.md) | A calibrated cost model for long rows (DESIGN 11.2's additive model with a memory factor per cache level, fitted per machine by `fbench calibrate`): does it choose chain, two or three grid phases within 5 % of the fastest on 90 % of long rows, and can it replace X12b/X29/X32/X33/X34's thresholds? | 2 | done, first pass: rejected (model within 5 % on 100/95 % on the EPYC but 72/55 % Mac, 72/80 % Xeon, worse than the rules there; calibrated thresholds also fail on Mac and Xeon); thresholds stay; misfires left to `planning::measure` and wisdom; a denser catalogue and a set-conflict term for a second pass |
| [P4](experiments/P4-next-fast-length.md) | Are `next_fast_length`'s lengths fast on the Mac, the EPYC and the Xeon (within 1.3 of the power of two above, faster than the awkward length), and do composites with 11–23 belong in the fast set? (task 2.5) | 2 | done: every pick faster than its awkward length (0.11–0.96); composites with 11, 17, 19 or 23 removed by the rule; within 1.3 of the power of two above on 96 % (Mac) but 78 % (EPYC, Xeon), the misses short x86 lengths below 300; rule 1's target lowered to 75 % by the maintainer (2026-11-17) |
| [T1](experiments/T1-dispatch-wait-costs.md) | Dispatch, wake and wait costs per machine and runtime (libgomp, libomp, Kokkos, pool) | 3 | done: hand-off cost and claim runs adopted on Zen 3; dispatch and work costs wait for a per-lane dispatch term |
| T2 | Claiming engine against precomputed schedule with spin barriers; keep one or both? | 3 | planned |
| [T3](experiments/T3-kokkos-dispatch.md) | Kokkos: cost of one `parallel_for` per transform; `team_barrier` behaviour | 3 | done: the dispatch costs 0.2–0.45 µs over the OpenMP region it wraps and no transform of 100 µs or more gains by bypassing Kokkos (EPYC, Xeon); team kernels match the executor; `completion::own` with `team_barrier` loses |
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
