# X64: cloud screen of 2026-11-24 (recent Intel against oneMKL; the families rules reach)

- **Status:** running
- **Owner:** lead session, 2026-11-24
- **Phase / task:** 1 (left from the exit: recent Intel against oneMKL) and the reach rule
  (AGENTS section 6: every family a rule reaches by cache key is screened)

## Question

Do the rules that reach unmeasured families by cache key hold there (else the family gets its own
key), and do X50's depth-first convolutions win on recent Intel (Sapphire, Emerald and Granite
Rapids), where they are off today?

## Hypothesis

The reach screens hold where the family's caches match the measured machine's (Zen 2 by its CCX,
Ice Lake-SP and Zen 4 parts by the Xeon 6240's L2 key, Emerald Rapids by the 2 MiB key). X50's
convolutions win on the 2 MiB-L2 Intel parts as on the Xeon 6240 (131071 ÷ oneMKL 1.34 → 0.84
there), because their L2 is larger still.

## Change under test

Campaign branch `exp/cloud-screen-1124` (c22a9f8: `main` d4b169f plus campaign-only fbench options
registered as `FFTN_X_SCREEN1124`, not for merging). Every arm of a run is the same binary; an arm
turns one rule off or on through an option:

- X57 off: `FIT_ROW_BYTES=0`; X62's batch leaf of 64 off: `BATCH_LEAVES=never`; X62's Bluestein
  cap off: `BLUESTEIN_POW2_PART_BYTES=<L1>` + `BLUESTEIN_WHOLE_LANES=0`;
- X63: `C2R_PACKED_BYTES=<bytes>|never`; X58's leaf of 12: `BATCH_LEAF_12=0|1`;
- X50 on: `DEPTH_CONVOLUTION_MAX_BYTES=33554432` (the Xeon 6240's limit).

## Method

- Machines (Google Cloud Spot VMs, GCC 14.2, one thread per core except Axion):
  Sapphire Rapids `c3-standard-8`, Granite Rapids `c4-standard-16`, Emerald Rapids
  `c4-standard-16`, Zen 2 `n2d-standard-16` (Rome), Ice Lake-SP `n2-standard-16`, Zen 4
  `c3d-standard-16`. Each job checks the CPU model first and stops on the wrong family. Where a
  VM does not report Zen 2's CCX (16 MiB of L3 per 4 CPUs), FFTN reads a corrected topology
  recording so that every arm plans as bare-metal Zen 2 would. Ice Lake and Zen 4 VMs report a
  large L3 per CPU, so both of X63's arms set its threshold explicitly (the VM's L2 against the
  block).
- Runs and cases:
  - Sapphire, Granite: A1 Bluestein rows (4093 … 999983, 65537, 1018): N, D (X50 on), FFTW,
    oneMKL. A2 (diagnosis) 160, 240, 384, the 1024 family, 32 × 16/256, 1000 in place: N, FFTW,
    oneMKL. A3 (diagnosis) forced radices of 160, 240, 1024 against N.
  - Emerald: E1 12-point batches: N, L (X58's leaf of 12). E2 = A1. E3 the standard catalogue's
    c2c rows without 2²³ and 2²⁴ (FFTW's MEASURE planning there takes over 100 minutes): N, FFTW,
    oneMKL.
  - Zen 2: Z1 the axes catalogue, one round, plan notes only (X61 must stay off); Z2 X57's 768
    and 1024 family; Z3 X62's catalogue without 32, 2048 and the short rows; Z4 X63's moved
    planes and controls, double and float, one thread and 8 lanes.
  - Ice Lake-SP, Zen 4: X63 and X63f catalogues, double and float, one thread and 8 lanes.
- Metric: arm ÷ arm per case with the run's A/A band; only the moved cases (plan notes differ
  between the arms) decide.
- **Decision rules, written before running:**
  - **Reach** (Z2, Z3 for each of X62's two rules, Z4 ×4, E1, Ice Lake ×4, Zen 4 ×4): the family
    keeps the rule if no moved case is slower than 1 + max(band, 3 %) (whole interval) and the
    moved cases' geometric mean is at most 1 + band. Otherwise the family gets its own key: its
    old plans (the rule's field off for that class), or a limit that narrows the rule under the
    narrowing clause, with a deciding re-run declared first. X63: a family keeps it only if all
    four of its runs (one thread and 8 lanes, double and float) do; one that keeps it in double
    only gets double only. X61: kept off if Z1 counts no note "adjacent transforms".
  - **X50 on the 2 MiB-L2 key** (A1 on Sapphire and Granite, E2 on Emerald; D against N): adopt
    if on Sapphire and Granite Rapids the moved cases' geometric mean is below 1 − band with its
    interval below 1 and no moved case is slower than 1 + max(band, 3 %), and Emerald does not
    lose by the reach rule. Adopting sets `depth_convolution_max_bytes` = 32 MiB for that key.
    131071 and 999983 against oneMKL are reported, not judged.
  - **Emerald exit check** (E3): reported like X56's cloud cores, ÷ FFTW and ÷ oneMKL, tail rule
    9 of the 191 rows. A failure in cases a Granite-measured rule moves becomes a follow-up.
  - A2, A3 decide nothing: FFTW's plans and the best forced radix sequences of 160, 240 and 1024
    on recent Intel, for a follow-up.
  - A criterion missed by less than the run's band gets one re-run on a new VM, which decides
    (AGENTS section 6).
- Budget: USD 5 (campaign 12.19 of 20 before this screen); rentals in
  [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md).

## Results

(to come)

## Verdict

(to come)
