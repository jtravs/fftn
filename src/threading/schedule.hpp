// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The scheduler (DESIGN.md 13.4): a pure function from (work graph, lane map, machine
// profile) to a schedule, plain data that the claiming engine (engine.hpp) follows. It
// decides where whole transforms go and which transforms are split, never what a chunk
// computes, so no schedule can change a result.
//
// A schedule is computed by every member at the start of every team execution: it is a few
// dozen arithmetic operations, allocates nothing, and every member gets the same one.
#pragma once

#include "fftn/isa.hpp"
#include "fftn/team.hpp"
#include "isa/arch.hpp"
#include "platform/cpu_features.hpp"
#include "threading/work_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace fftn::inline v0::detail {

struct topology;   // platform/topology.hpp
struct core_class; // platform/topology.hpp

// The constants the rules use, per machine. Experiments T1 (fixed costs) and T5 (split
// thresholds) calibrated them on a Zen 3 CCX (EPYC 7763), an Apple performance cluster (M5
// Max) and an 18-core Xeon mesh (task 3.4); each field cites its evidence. The defaults are
// the CCX's; profile_for sets the other kinds' values. Nothing in the rules is a literal.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding): fields grouped by meaning
struct machine_profile {
  // Who made the processor (DESIGN.md 13.5). The planner's rules for `avx2` that were measured
  // only on AMD's cores (the EPYC 7763) apply only where it is amd (planner/rules.hpp,
  // profile_of; core/plan_core.cpp, copies_prefetch): an Intel core with `avx2` but no
  // AVX-512 keeps the plans it had before them until it is measured (the maintainer, 2026-11-24;
  // the Xeon 6240 forced to `avx2` lost with two of them, X26 and X30's 512). Unknown, the default
  // here (unlike the other fields, not the CCX's), keeps those plans too. Two cache keys read
  // it as well (set_grid_sizes; the maintainer, 2026-11-25: a rule measured on one vendor only
  // is keyed to that vendor): the Xeon 6240's class and the 2 MiB-L2 key, measured on Intel's
  // cores only, apply to intel alone (X64: Zen 4, which the Xeon key reaches, lost with X63's
  // packed c2r planes); other vendors, unknown included, keep those fields' defaults.
  // default_machine_profile reads it from CPUID (detect_cpu_features); profile_for takes it as an
  // argument.
  cpu_vendor vendor = cpu_vendor::unknown;
  // The cost model: nanoseconds per point through a pass (work_graph.hpp). T1 measured 1.0
  // on the CCX (AVX2), 1.23 on the Xeon and 0.57 on the M5, but with that figure and a
  // dispatch cost that ignores the team's width the scheduler shared small batches over a
  // 64-thread OpenMP region (c2c 256 x 8: 3.1 times slower) and split r2c rows of 4096
  // points (2.4): T1's adoption run 20261114-181442-11ed kept 0.5 until rule 1 and the
  // split rule know the executor's width.
  double ns_per_work = 0.5;
  // Rule 1: run inline when the whole job takes less than inline_multiple x dispatch_ns.
  // T1 measured a team call at 2-6 us within a cache group but 26 us (pool) to 35 us
  // (OpenMP executor) at 64 lanes of the EPYC; the full-width value inlined rows up to
  // 10240 points that 8 members run 1.2-2.1 times faster (run 20261114-181442-11ed, 1.15 over
  // the catalogue), so the cost stays at a group's until it has a per-lane term.
  double dispatch_ns = 2000;
  double inline_multiple = 3;
  // Rules 2 and 3: the cost of one phase boundary inside a cache group (a claim, a done
  // counter and a wait on it). T1, 1 us chunks among a group's members: 1141 ns on the
  // CCX (8 members), 461 on the M5 (6), 2846 on the Xeon (18); adopted on the CCX (run
  // 20261114-181442-11ed: 0.989, nothing slower beyond the band); profile_for keeps the
  // provisional 700 on the other kinds, where T1's adoption runs failed for other fields.
  double sync_ns = 1100;
  // Rule 3: at least this many points per member of a split transform (T5, S2, run
  // 20261114-182645-10c9: 512 and 1024 within the band over the rows worth splitting,
  // 2048 1.03, 4096 1.14; the largest within the band).
  std::size_t min_points_per_member = 1024;
  // Rule 3: split a transform only from this size, by cache-group type (T5, S1): 6144 on a
  // CCX (runs 20261114-175237-ee55 at 8 lanes and 20261114-180642-f917 at 64: a lone
  // 4096-point row split lost 1.33 against whole behind a 64-thread OpenMP executor, an r2c
  // of 4096 points 1.5 at 8 lanes), 8192 on an Apple performance cluster (profile_for; run
  // 20261114-174818-ee78: 4096 lost 1.17, 8192 gained 0.84). On a mesh-connected group of
  // more than mesh_group_lanes members, S1 on the Xeon (run 20261114-180925-7c39) put the
  // crossover at 20480 (r2c and c2r of 16384 points lost 1.11-1.17), but that threshold
  // lost 1.05 over the catalogue (run 20261114-182732-001f: planes and few rows of 16384
  // points 1.5-2.4 times slower unsplit), so the mesh keeps 8192.
  std::size_t split_min_points = 6144;
  std::size_t split_min_points_mesh = 8192;
  // A plane or volume (a work graph of line phases, whose phases move the transform between
  // the members' caches) is split only from this many points too (0: no further limit). X52
  // arm S2: on the Xeon 6240 (a mesh-connected L3) at 8 lanes c2c planes of 9216-10000 points
  // took 1.4-2.0 times their one-thread time split, 16384 about the same, 32768 0.6.
  // profile_for sets 32768 where the cache group is a mesh.
  std::size_t split_min_points_lines = 0;
  // A real plane or volume is split by its complex points (core/plan_core.cpp; X52 arm S1).
  // set_grid_sizes sets it where each core has its own L2: on the M5 Max, whose cluster
  // shares L2, the real 96^2 and 100^2 planes it would leave whole took 1.3 times as long.
  bool split_complex_points = false;
  int mesh_group_lanes = 8;
  // Rule 2: leftover transforms after the whole ones are split only from this size (T5,
  // S3, run 20261114-182042-1f0c: at 8 lanes leftovers of 4096 points gained 0.85-0.98 and
  // of 2048 nothing; at 64, run 20261114-182347-49bf, nothing either way; the crossover is
  // within one catalogue step of the prototype's 8192, which T5's rule keeps).
  std::size_t remainder_split_min_points = 8192;
  // The logical CPUs the process may use (0: unknown). A team with more members is
  // oversubscribed: no transform is split, since a preempted member would stall its
  // partners for a whole time slice, and waiting members do not spin.
  int cpus = 0;
  // Waiting (wait.hpp): spin this long, then yield this long, then block. T1's rule (spin
  // about as long as a blocked waiter takes to wake: 6-9 us on the CCX, 7 on the M5, 10-13
  // on the Xeon) gave 9 and 22.5 us on the CCX, which made batches of 8192-16384-point rows
  // 1.05-1.11 times slower (run 20261114-181442-11ed): the study's 20 and 50 us stay.
  double spin_ns = 20000;
  double yield_ns = 50000;
  // Whole transforms are claimed in runs of (left / (claim_divisor x members)), at least
  // one: long runs while much is left (contiguous rows, few atomic operations), single
  // transforms at the end (balance). 1 on the CCX (T1, run 20261114-181442-11ed: 0.972,
  // batches of 1024 short rows 0.70-0.83, nothing slower beyond the band; 4 lost 1.20-1.39
  // there); profile_for keeps 2 on the other kinds.
  int claim_divisor = 1;
  // Whole transforms anchored to members (engine.hpp, whole_tasks; experiment X73): a group's
  // whole transforms are cut into one contiguous range per member dealt, in order of rank,
  // and each member claims its own range first, in runs of half what is left of it, before
  // taking runs of the others'. A member then runs the same transforms in every execution,
  // whoever arrives first, so a batch beyond the caches stays where the previous execution
  // left it (the member's L2 and, with unpinned threads that the team's cache groups do not
  // match, its core complex's L3). Off: runs from the group's one counter, in arrival order.
  bool anchored_runs = false;
  // The planner's (not the scheduler's), experiment X12: a c2c row whose bytes exceed
  // grid_min_bytes is a grid rather than a chain (strategies/grid.hpp), in three phases
  // rather than two beyond grid_three_bytes; the grid widens its column blocks while one
  // block's scratch (lines a and b) stays within grid_block_bytes. grid_block forces the
  // block width (0: by that rule). default_machine_profile sets the sizes from the caches
  // of the fastest cores (set_grid_sizes, experiment X12b): grid_block_bytes is the L1 data
  // cache; with an L2 of each core's own (x86) the grid starts beyond half of L2 and three
  // phases beyond L2, with an L2 shared by a cluster (Apple) beyond L1 and an eighth of L2.
  // grid_block_bytes (the L1 data cache) also bounds the r2c pairing's fusion
  // (core/plan_core.cpp, fusion_of; task 2.1 measured fusion losing beyond L1).
  std::size_t grid_min_bytes = std::size_t{64} * 1024;
  std::size_t grid_three_bytes = std::size_t{2} * 1024 * 1024;
  // Up to grid_fuse_bytes per row, a grid phase whose radices are all small reads and
  // writes the arrays' rows in its end passes instead of through a gather and a scatter;
  // beyond it the arrays leave L2 and the copies, which prefetch rows ahead, won (X12, run
  // 20261114-132841-80e0: 2^20 1.46 against 1.35 times FFTW). default_machine_profile sets
  // it to a quarter of L2.
  std::size_t grid_fuse_bytes = std::size_t{4} * 1024 * 1024;
  // A row whose length n has an odd part (n over its largest power-of-two factor) of at least
  // grid_two_odd_part stays in two grid phases at any size (0: off; experiment X34). On the
  // Xeon 6240 (private 1 MiB L2, 0.7 MiB of L3 per CPU) two phases took 0.70-1.01 of the
  // three phases' cycles on the 15 such rows measured, 71680 to 4 x 10^6 points, while rows
  // nearer a power of two (odd part 3-9: 294912, 327680, 393216) lost up to 1.13 as two
  // phases. set_grid_sizes sets it where a private L2 has less than 2 MiB of L3 per CPU
  // behind it, the machines where X29's chains for such rows do not apply.
  std::size_t grid_two_odd_part = 0;
  // A row whose length n is not a power of two and whose power-of-two part, in bytes of
  // complex values, is at most grid_two_part_bytes stays in two grid phases at any size (0:
  // off; experiment X35). On a Granite Rapids VM two phases took 0.58-0.83 of the time of
  // X29's chains at 100000, 10^6 and 2 x 10^6 points, and 71680 (power-of-two part 2048)
  // lost 1.10 in three, while rows with a larger power-of-two part gained in three phases
  // (two lost up to 1.46). set_grid_sizes sets it to 32 KiB on x86-64 where the private L2
  // has 2 MiB or more.
  std::size_t grid_two_part_bytes = 0;
  std::size_t grid_block_bytes = std::size_t{64} * 1024;
  std::size_t grid_block = 0;
  // A c2c plan whose transforms lie one element apart stays on the chain when all its
  // transforms together fit this many bytes (the L1 data cache; core/plan_core.cpp, experiment
  // X61) and either a register holds one complex value or every axis has a leaf length. There
  // a strided line costs no more than a contiguous one and the chain's own passes win (8 points
  // x 16 transforms: 1.42 of the chain on the M5 Max, 1.20 on the Xeon 6240), while with wider
  // registers longer axes gain from the lanes within L1 too (Xeon: 32 x 16 0.23 of the chain).
  // All the transforms count: blocks that each fit L1 but together do not still gain (32 x 128
  // x 16 blocks: 0.62 of the chain's time over the middle axes on the M5 Max).
  std::size_t adjacent_min_bytes = std::size_t{64} * 1024;
  // A team runs such a plan's transforms through the chain, sharing their lines, so the lanes
  // over adjacent transforms are taken only where their radices (column_radices) are the
  // chain's on every axis, and the bits agree (X61). Elsewhere the plan is the chain: giving the
  // lanes the chain's radices took the EPYC's one thread from 0.29 to 0.40 of the chain's time
  // (1024 x 16 1.25), and letting a team run the lanes where the radices differ lost up to 2.1
  // at 8 lanes on the Xeon. A block of transforms is one task: at 8 lanes the EPYC 7763 ran
  // planes and volumes of 8-64 transforms up to 1.74 times slower than the chain, and the Xeon 6240
  // up to 2.4 and 1024 x 256 still 1.2-1.6. Where it was measured: x86-64 with a private L2 below 2
  // MiB.
  bool adjacent_team_chain = false;
  // Whether a plan may take the lanes over adjacent transforms at all (X61): only on the classes
  // measured where it passed, the Xeon 6240's and Apple's (set_grid_sizes). Off on the
  // EPYC 7763's class, where under adjacent_team_chain the route kept 0.946 of the chain's
  // time, short of X61's 0.85 bar, and everywhere unmeasured.
  bool adjacent_lanes = false;
  // The widest block of a three-phase grid (0: max_grid_block, strategies/grid.hpp;
  // experiment X39). Three phases may fill twice grid_block_bytes (X12), which gives blocks of
  // 32 columns to phases of 64; on the Xeon 6240 blocks of 16 took 0.91-0.96 of their time
  // at 73728-262144 points (counters, every three-phase row with a 64-long phase) and
  // changed nothing where the budget already gives 16 or fewer (327680, 2^20).
  // set_grid_sizes sets 16 where a private L2 has less than 2 MiB of L3 per CPU behind it.
  std::size_t grid_three_block = 0;
  // The largest radix of the grid's phase rule (0: default_grid_phase_radix, 16;
  // build_settings::grid_phase_radix overrides it). Phase lengths are 13-smooth, so 20 adds
  // only X38's radix-20 pass: on the Xeon 6240 phases of 200 as 10 x 20 took 0.92 (100000)
  // and 0.83 (40000) of the cycles of 5 x 5 x 8 (experiment X39, arm D; radix 32 and 64 lost
  // there, arm B). set_grid_sizes sets 20 where a private L2 has less than 2 MiB of L3 per CPU
  // behind it, on x86-64.
  std::size_t grid_phase_radix = 0;
  // The block budget of a two-phase grid (0: grid_block_bytes; experiment X39). On the Xeon
  // 6240 blocks of 8 columns for 192-256-long phases (twice L1) took 0.86-0.98 of the cycles
  // of blocks of 4 at 32768-65536 points. set_grid_sizes sets twice L1 where a private L2 has
  // less than 2 MiB of L3 per CPU behind it.
  std::size_t grid_two_block_bytes = 0;
  // A c2c row (not the half-length transform of a real row) of a power-of-two length is a
  // grid beyond pow2_grid_bytes, below grid_min_bytes (0: grid_min_bytes alone; experiment
  // X39). On the Xeon 6240 the chain of 32768 points (half of L2) took 1.09-1.16 times the
  // two-phase grid's cycles, while 16384 lost as a grid (1.02-1.69 in every split) and so did
  // c2r 65536 (1.13; r2c 65536 gained, 0.88). set_grid_sizes sets a quarter of L2 there.
  std::size_t pow2_grid_bytes = 0;
  // A c2c row out of place whose length is a power of two stays in two grid phases beyond
  // grid_two_pow2_bytes (0: off; experiment X39). On the Xeon 6240 two phases took 0.88-0.92
  // of the three phases' time at 2^20-2^23 points out of place, but lost in place (2^20 1.10)
  // and for c2r 2^20 (1.23). set_grid_sizes sets it on the Xeon profile.
  std::size_t grid_two_pow2_bytes = 0;
  // A c2c row of one axis of [depth_min_bytes, depth_max_bytes] whose power-of-two part has
  // more than depth_part_bytes is a depth-first chain in double (strategies/chain.hpp,
  // c2c_depth; core/plan_core.cpp, depth_tops_of, which also names the limits below), with one
  // top factor, depth_top, else depth_top_alt, where its split fits; depth_max_bytes 0: never
  // (experiment X46). On the Xeon 6240's counters screen such chains with tops of 16 or 20
  // took 0.63-0.77 of the chains' and grids' time from 16384 to 262144 points (L2/4 to 4 L2);
  // 8192 points (L2/8) gained nothing. On the M5 Max tops of 4 took 0.64-0.96 from 3072 to
  // 16384 points where the power-of-two part had 2048 points or more, and lost 1.05-1.09
  // where it had 16-128 (the cluster L2's conflicts are a power-of-two problem, X29) and at
  // 32768 (1.06, against X32's chain). set_grid_sizes sets L2/4 to 4 L2 and tops 16, 20 where
  // a private L2 has less than 2 MiB of L3 per CPU behind it, on x86-64; L1 to 2 L1, top 4
  // and a power-of-two part above L1/8 where a cluster shares L2.
  std::size_t depth_min_bytes = 0;
  std::size_t depth_max_bytes = 0;
  std::size_t depth_part_bytes = 0;
  std::size_t depth_top = 0;
  std::size_t depth_top_alt = 0;
  // A third top, tried after depth_top and depth_top_alt (0: none). X51, Xeon 6240: 25000 (three
  // factors of 2, so on `avx512`'s 4 lanes neither 16 nor 20 leaves a sub-row C divides) as
  // 10 x 2500 took 0.78 of its chain's time; set_grid_sizes sets 10 with tops 16 and 20.
  std::size_t depth_top_last = 0;
  // Powers of two take the depth-first chain only up to depth_pow2_max_bytes (X51: on the
  // M5 Max X32's chains stay faster from 2 L1 up, where rows of a small odd part times a
  // large power of two gain to 16 L1; on the EPYC 7763 they do not gain at all).
  std::size_t depth_pow2_max_bytes = std::numeric_limits<std::size_t>::max();
  // Rows in single precision take the depth-first chain by their bytes, as in double, but
  // only up to depth_float_max_bytes (0: never; X51: measured on the Xeon 6240 to 2^20
  // complex floats, 8 MiB).
  std::size_t depth_float_max_bytes = 0;
  // Whether the rules may take the batch leaf of 12 points (`avx512`, double): false on a
  // private L2 of 2 MiB or more, where it lost (X58: Sapphire Rapids 12 x 64 1.17 of the
  // chain's time; it won on the Xeon 6240 and Ice Lake).
  bool batch_leaf_12 = true;
  // A Bluestein stage's convolution of one copy runs on two depth-first chains where a row
  // of its length would be one in double, by the fields above, but with this upper limit in
  // place of depth_max_bytes (0: never; core/plan_core.cpp, bluestein_tops_of; experiment
  // X50): the convolutions measured, so that a later widening of the rows' range does not
  // carry to convolutions unmeasured. A convolution is never a grid, so beyond the rows' range
  // the alternative is the breadth-first chain: on the Xeon 6240 a top of 16 took 0.56 of its
  // time at 2073600 points and 0.62 at 2^21 (32 MiB; two tops of 16 0.62 and 0.64).
  // set_grid_sizes sets 32 MiB on the Xeon profile, the longest convolution measured, and
  // 2 L1 (X46's rows' range, M of 8192-16384 points on the M5 Max) on the Apple profile.
  std::size_t depth_convolution_max_bytes = 0;
  // An even real row of one axis out of place, contiguous on both sides, runs its half-length
  // complex transform as a depth-first chain where a c2c row of that length would be one in
  // double, by the fields above, up to this many bytes of the half length (0: never;
  // core/plan_core.cpp, real_depth_tops_of; experiment X72), with the separate pairing pass
  // after it (r2c) or before it (c2r, into scratch). set_grid_sizes sets the rows' whole range
  // on Zen 3's class (X72), on the Xeon 6240's (X75) and on X58's Ice Lake branch (X82).
  std::size_t depth_real_max_bytes = 0;
  // Such a real row's half length takes the chain from the smaller of depth_min_bytes and this
  // (the largest value: no lower limit of its own; experiment X75): on the Xeon 6240 real rows
  // whose half length is above L2/8 gained (18432-24576 points, 0.73-0.95 of the time) where
  // c2c rows of that length had not (X46: 8192 points, L2/8, gained nothing).
  std::size_t depth_real_min_bytes = std::numeric_limits<std::size_t>::max();
  // A c2r row on that chain pairs its bins in the chain's first pass, from the input, instead of
  // a separate pass into the workspace block, where its half length's bytes lie from
  // depth_c2r_fused_min_bytes to depth_c2r_fused_max_bytes (0: never; core/plan_core.cpp,
  // chain_builder::c2r_depth; experiment X78). set_grid_sizes sets L2/4 to 16 L2 on the Xeon
  // 6240's class and on X58's Ice Lake branch (X82); Zen 3's keeps the separate pass, which the
  // fused first pass lost to at every length there (1.08-1.30 of its time).
  std::size_t depth_c2r_fused_min_bytes = 0;
  std::size_t depth_c2r_fused_max_bytes = 0;
  // The L1 data cache's line in bytes: a lanes column phase's block holds at least a line of
  // adjacent columns (core/plan_core.cpp, column_block; experiment X40). set_grid_sizes reads
  // it from the topology (128 on Apple's cores).
  std::size_t line_bytes = 64;
  // The memory page (16 KiB on Apple's processors): a lanes column phase chooses its block
  // by whether a column's elements lie a page or more apart (core/plan_core.cpp,
  // column_block; X40). profile_for reads it from the topology.
  std::size_t page_bytes = 4096;
  // When not 0, the one-thread program's c2c lanes column phases widen their blocks to this
  // many bytes of columns where both tiles stay within column_tile_bytes; teams keep the
  // narrower blocks in a team program (core/plan_core.cpp, column_block; X52 arm W4b).
  // set_grid_sizes sets 1 KiB on x86-64 where a private L2 has a large L3 behind it (the EPYC
  // 7763: 0.933 of the time over the planes and volumes it changes); the Xeon 6240 gained
  // 0.95 but lost 1.05 at 30 x 40 x 50.
  std::size_t column_wide_bytes = 0;
  // A c2c plan of one axis of contiguous rows whose rows and output (in place: rows) together
  // take at most this many bytes runs the two-pass rows of the profile's fit_overrides; beyond
  // it, the three-pass plan (planner/rules.hpp, fitting_radices; 0: the rules as without this
  // field).
  // Experiment X57 on the EPYC 7763 (`avx2` double, 32 MiB L3 per CCX): 1024 = 16 x 64 took
  // 0.84-0.96 of 4 x 16 x 16's time from 1 to 768 rows (24 MiB) and 1.04-1.30 from 1024 rows
  // (32 and 64 MiB); at 768 x 2048 (48 MiB) 4 x 12 x 16 took 0.75 of 12 x 64's.
  // set_grid_sizes sets three quarters of the L3 where a private L2 has a large L3 behind it
  // (x86-64, the EPYC class).
  std::size_t fit_row_bytes = 0;
  // In place, the rows must also take at most this many bytes (the limit is the smaller of the
  // two; core/plan_core.cpp). X57's amendment 2 on the EPYC 7763: in place 16 x 64 took 0.97
  // (forward) and 1.00 (backward) of 4 x 16 x 16's time at 1100 rows of 1024 points (17.2 MiB),
  // 1.07-1.16 at 1280 and 1536 rows (20 and 24 MiB); 12 x 64 1.00 and 1.05 of 4 x 12 x 16's at
  // 1536 rows of 768 (18 MiB), 1.14-1.19 at 2048 (24 MiB). set_grid_sizes sets half the L3 on
  // the EPYC class, 16 MiB there: 1024 rows of 1024, the footprint X57's runs 1 and 2 measured
  // (the maintainer's decision; 1100 rows gained too little to keep).
  std::size_t fit_in_place_bytes = std::numeric_limits<std::size_t>::max();
  // The two limits above apply to the 2 MiB-L2 key's fit table (planner/rules.hpp,
  // large_l2_fit_overrides; core/plan_core.cpp) instead of the target's (X57's `avx2` table).
  // Experiment X69: on Sapphire and Granite Rapids under `avx512` in double, 1024 = 32 x 32 took
  // 0.73-0.82 of 8 x 8 x 16's time at 1-16 rows and tied at 256 rows (8 MiB of rows and output);
  // 160 = 8 x 20 and 240 = 12 x 20 0.79-0.84 of the rules' time. set_grid_sizes sets it where
  // x86-64 has a private L2 of 2 MiB or more, with fit_row_bytes the L2 and fit_in_place_bytes
  // half of it. Zen 4 and Zen 5 (`avx512`, fit_row_bytes from the large-L3 branch) keep it false,
  // so the key's `avx512` table does not reach them. Experiment X70 raised the key's limits to
  // four times the L2 (fit_row_bytes) and twice the L2 (fit_in_place_bytes), with 1024 out of
  // place kept at the L2 by fit_long_row_bytes (0.858-0.873 over the cases it moves). Experiment
  // X80: on the key fit_row_bytes and fit_in_place_bytes no limit, out of place
  // fit_out_of_place_bytes 64 times the L2 (0.894-0.903 over the cases it moves).
  bool fit_large_l2 = false;
  // Out of place, a plan whose rows are longer than a 4 KiB page (one side) fits only while its
  // rows and output also take at most this many bytes (the limit is the smaller of it and
  // fit_row_bytes; core/plan_core.cpp). Experiment X70: on the 2 MiB-L2 key rows
  // of 160 and 240 (2.5 and 3.75 KiB) kept gaining past the L2 out of place (X69's arm W: 0.83-0.90
  // at 5 and 7.5 MiB), rows of 1024 (16 KiB) did not (0.95-1.01 at 4 and 8 MiB, 1.29-1.36 at
  // 32 MiB on Sapphire Rapids). set_grid_sizes sets the L2 on that key; elsewhere no limit, so
  // X57's EPYC limits are unchanged.
  std::size_t fit_long_row_bytes = std::numeric_limits<std::size_t>::max();
  // Out of place, the rows and output must also take at most this many bytes (the limit is the
  // smaller of it, fit_row_bytes and, for rows longer than a page, fit_long_row_bytes;
  // core/plan_core.cpp): fit_in_place_bytes' counterpart, so that in place may go further than
  // out of place. Experiment X80: on the 2 MiB-L2 key X70's arm W
  // gained in place at every size tried (rows of 1024 to 32 MiB 0.88-0.93, of 160 and 240 to
  // 240 MiB 0.82-0.94), out of place for 160 and 240 to 120 MiB of rows and output (0.85-0.94)
  // and not at 320-480 MiB on Granite Rapids (1.02-1.03). set_grid_sizes sets 64 times the L2 on
  // that key (with fit_row_bytes and fit_in_place_bytes no limit; judged on Sapphire, Granite
  // and Emerald Rapids, 0.894-0.903 over the 22 cases it moves); elsewhere no limit, so X57's
  // limits are unchanged.
  std::size_t fit_out_of_place_bytes = std::numeric_limits<std::size_t>::max();
  // A lanes column phase whose column elements lie less than a page apart keeps its block's
  // two scratch tiles within this (core/plan_core.cpp, column_block; X40). set_grid_sizes
  // sets a quarter of L2.
  std::size_t column_tile_bytes = std::size_t{256} * 1024;
  // When not 0, the budget instead for columns whose elements lie at least four to a page
  // (X41: twice L1 where a cluster shares L2).
  std::size_t column_near_tile_bytes = 0;
  // When not 0, a c2c plan of several axes also builds a team program (plan_core::team),
  // whose lanes column blocks hold at most this many bytes of columns, for executions by
  // teams of two or more (X43 arm B1). set_grid_sizes sets 128 on x86-64 where a private L2 has
  // 2 MiB or more of L3 per CPU behind it (the EPYC 7763).
  std::size_t column_team_bytes = 0;
  // A team program (plan_core::team) runs a c2c volume's two inner axes plane at a time
  // (core/plan_core.cpp, add_volume_planes; X43b arms VT and VTX): plane at a time changes no
  // bits, and teams gained where one thread did not. set_grid_sizes sets it on x86-64 where
  // each core has its own L2 of less than 2 MiB.
  bool team_volume_planes = false;
  // Column phases of planes whose neighbouring lines write the same cache lines claim small
  // shares in fair runs (work_graph.hpp, chunk_run; core/plan_core.cpp, lines_share_cache_lines;
  // X59 arm R6). set_grid_sizes sets it on x86-64 where a private L2 under 1 MiB has a large L3
  // behind it (Zen 3; Zen 4 and Zen 5 are not measured) (EPYC 7763 at 8 lanes: 0.909 of the guided
  // runs' time over the real planes that split, 512^2 0.70) and where a cluster shares L2 (M5 Max
  // at 6 lanes: 0.979); the Xeon 6240 lost (1.017, c2r 1000^2 1.21), and the profiles like it keep
  // the guided runs.
  bool fair_line_runs = false;
  // The L1 data cache's associativity (8 where the topology does not say: macOS): with
  // grid_block_bytes it gives the set period against which a column phase's last pass
  // counts its legs (core/plan_core.cpp, column_radices; X40).
  std::size_t l1_ways = 8;
  // The set period (bytes) the grid pads the rows of its intermediate array against, so that
  // the rows phase 2 gathers spread over the sets (grid_pad, experiment X26b); SIZE_MAX: no
  // padding. set_grid_sizes sets 4 KiB where the L2 is shared by a cluster (Apple: long rows
  // 0.976 of the unpadded time) and none where each core has its own (x86: the padding cut
  // no L1 refills there and gained nothing on the Xeon and the EPYC).
  std::size_t grid_pad_period = std::numeric_limits<std::size_t>::max();
  // Exceptions to grid_min_bytes for rows whose length n is not a power of two (experiment
  // X29): such a row stays a chain at any size when the power-of-two part of n, in bytes of
  // complex values, is at most chain_part_bytes, and up to chain_mixed_bytes whatever that
  // part. The chain lost beyond L1 on Apple's cluster L2 because the legs of a power-of-two
  // row's passes map to the same L1 sets; an odd factor in every leg distance spreads them.
  // 0 turns either off. set_grid_sizes sets them where the L2 is shared (an eighth of L1,
  // the way size of an 8-way L1, and twice L1) and chain_part_bytes where a private L2 has
  // 2 MiB of L3 per CPU behind it (16 KiB).
  std::size_t chain_part_bytes = 0;
  std::size_t chain_mixed_bytes = 0;
  // The largest row (bytes) chain_part_bytes keeps a chain: half of L3 where a private L2
  // has a large L3 behind it (x86 Zen), any size where the L2 is shared (X29).
  std::size_t chain_part_limit_bytes = std::numeric_limits<std::size_t>::max();
  // Long power-of-two rows (experiment X32): a c2c row of a power-of-two length (and the
  // half-length transform of a real row, X33) stays a chain beyond grid_min_bytes up to
  // pow2_chain_bytes (0: never, and the rules' radices
  // everywhere), and a power-of-two chain of more than grid_min_bytes takes FFTL.jl's
  // radices instead of the rules' (priced in cache, X2): 8s (4s for the spare factors of
  // two) up to pow2_radix8_bytes, 4s (one 8 for an odd exponent) beyond. The rules'
  // radix-16 passes read 16 streams a power of two apart; where the row's pages are huge
  // pages they share one L2 set, and such a chain ran at up to 5 times a radix-8 chain's
  // time (X29's bimodal chains on the EPYC). set_grid_sizes sets them from X32's screens.
  std::size_t pow2_chain_bytes = 0;
  std::size_t pow2_radix8_bytes = 0;
  // The largest power-of-two part (bytes of complex values) of a Bluestein stage's
  // convolution length (experiment X36, planner/rules.hpp bluestein_length). Where the L2
  // is shared (Apple), chains of a power of two of 8192 points and more ran 1.4-1.7 times
  // as long per point as chains of 7-smooth lengths just above, so set_grid_sizes sets a
  // quarter of L1 there, and the L1's size on Zen-like caches (the EPYC 7763's 4093-32749:
  // 0.36-0.69 of the time); on the Xeon 6240 a limit gained nothing. Zen 3's class (X62)
  // takes half the L1: 1013-1021 took 0.93 of their time with M = 2160 for M = 2048.
  std::size_t bluestein_pow2_part_bytes = std::numeric_limits<std::size_t>::max();
  // A Bluestein stage over as many copies of a prime as a register holds (s = C) runs as one
  // stage over the whole line, while the line is not long (long_convolution_bytes), whatever
  // the two convolutions' lengths; false: only where the whole line's convolution is no
  // longer than the copies' together (strategies/chain.cpp, X36b). On s = C the copies' chains
  // are lane passes without a grouped first pass (X36b: 16.3 against 10.0 ns per point on the
  // EPYC 7763), so a somewhat longer whole convolution still wins: X62 sets it on Zen 3's
  // class, where half the L1 as the power-of-two limit takes 1018's whole convolution to 2160
  // points, longer than two of 509's 1024. Both precisions (the field is the machine's): in
  // float, 2036 = 4 x 509 (s = C = 4) took 0.92 of its time with M = 4320, the one float line
  // measured.
  bool bluestein_whole_lanes = false;
  // A Bluestein or Rader convolution of a power-of-two length of more than this many bytes
  // takes X32's long-row radices (experiment X36; a convolution is a chain, never a grid):
  // grid_min_bytes where X32's long chains apply (Apple, Zen), four times the L2 on other
  // private L2s (the Xeon 6240 lost with them at 1 MiB and won from 8 MiB; Granite Rapids
  // takes the same rule, unmeasured).
  std::size_t long_convolution_bytes = std::size_t{8} * 1024 * 1024;
  // Whether a c2r row out of place in a two-phase grid pairs its bins in phase 1's gather
  // rather than in a separate pass (X29): set where the L2 is shared (Apple: 0.92-0.94 of
  // the separate route); on the EPYC it took 1.11.
  bool c2r_paired_gather = false;
  // A c2r plane out of place builds its intermediate packed in the output (core/plan_core.cpp,
  // c2r_packed) where one transform's half spectrum exceeds this many bytes, and keeps it in the
  // workspace block otherwise (X60, X63). The packed route wins where the block pushes input,
  // block and output past a cache level and loses inside the caches, where the output's
  // power-of-two row strides cost more than the block's odd ones. Set only for the classes
  // where X63 passed (set_grid_sizes); the block everywhere else.
  std::size_t c2r_packed_bytes = std::numeric_limits<std::size_t>::max();
  // The one instruction set whose plans c2r_packed_bytes applies to, the one it was measured on
  // (plans for another target keep the block); fftn::isa::best applies it to every target.
  fftn::isa c2r_packed_isa = fftn::isa::best;
};

// The profile of a machine of topology t (its CPU count aside): the defaults above, which
// are a Zen 3 CCX's, with the grid's sizes from t's caches, and an Apple cluster's or a
// mesh's values where T1 and T5 measured them. A pure function of the topology, so that
// the fixtures test it on any machine.
// x86_64 says whether the machine is an x86-64 one (the build's architecture by default):
// X35's rule for a private L2 of 2 MiB or more was adopted there only. vendor: the
// processor's maker, the profile's vendor (unknown by default: a recorded topology does not
// say), which set_grid_sizes reads for the Intel-only keys (the Xeon 6240's class, 2 MiB of L2).
[[nodiscard]] machine_profile profile_for(const topology& t, bool x86_64 = FFTN_ARCH_X86_64 != 0,
                                          cpu_vendor vendor = cpu_vendor::unknown) noexcept;

// The profile of the running machine: profile_for its topology, with its CPU count and
// its processor's vendor.
[[nodiscard]] machine_profile default_machine_profile() noexcept;

// Sets the grid's size fields from the caches of one core class, the fastest (sizes of 0:
// unknown, which keeps the fields' defaults), and p.vendor (the Xeon 6240's class and the
// 2 MiB-L2 key are Intel's only; DESIGN.md 13.5).
void set_grid_sizes(machine_profile& p, const core_class& c,
                    bool x86_64 = FFTN_ARCH_X86_64 != 0) noexcept;

// Which lanes (team members) share a cache: contiguous runs of group_size ranks (fftn::team).
struct lane_map {
  int lanes = 1;
  int group_size = 0; // 0, or lanes or more: one group
  // The threads of the runtime the team runs on, when it has more than the team (0: as
  // many as lanes). An executor dispatches only the lanes dealt work, but all of its
  // threads compete for the CPUs (a pool's idle threads linger, an OpenMP region keeps its
  // width): oversubscription is judged by them.
  int threads = 0;

  [[nodiscard]] int per_group() const noexcept {
    return group_size <= 0 || group_size > lanes ? lanes : group_size;
  }
  [[nodiscard]] int groups() const noexcept { return (lanes + per_group() - 1) / per_group(); }
  [[nodiscard]] int group_of(int lane) const noexcept { return lane / per_group(); }
  [[nodiscard]] int base(int group) const noexcept { return group * per_group(); }
  [[nodiscard]] int size_of(int group) const noexcept {
    const int rest = lanes - base(group);
    return rest < per_group() ? rest : per_group();
  }
};

struct schedule {
  std::size_t tasks = 0;
  // Rule 1: the first member to arrive takes every task in one claim.
  bool inline_all = false;
  bool oversubscribed = false;
  // Rules 4 and 5: tasks are dealt to groups [0, groups), in proportion to their members:
  // as many groups as the batch has a group's worth of tasks for, so a batch smaller than
  // a group stays in one (full-width teams lost at 64 lanes: 1.19 against 0.83 at 8).
  int groups = 1;
  int lanes_dealt = 1; // members in those groups
  // The members an executor dispatches (rule 4): lanes_dealt, or, when no group splits a
  // task, only as many as there are tasks (whole tasks give a member each at most). A team
  // of that many members gets the same split decisions, none: each group's share then
  // equals its members.
  int dispatch = 1;
  // Whole tasks are counted, claimed and finished in units of `grain` consecutive tasks, so
  // that a batch of any size fits the 24-bit counters (team_state.hpp); 1 unless the batch
  // has millions of transforms.
  std::size_t grain = 1;
  // What share_of needs to decide splits per group.
  bool splittable = false;
  double task_ns = 0;
  std::size_t phases = 0;
  std::uint32_t widest = 1;
  std::size_t points = 1;
  bool lines = false; // the graph has a phase of lines (a plane or volume)
  machine_profile profile;
};

// One group's tasks: [first, first + count) in batch order; the first `whole` run whole
// (claimed in runs), the other `split` are split among at most `cap` of its members each.
struct group_share {
  std::size_t first = 0;
  std::size_t count = 0;
  std::size_t whole = 0;
  std::size_t split = 0;
  int cap = 0;
};

// The counters of a team execution have 24 bits (team_state.hpp). A split task's chunks and
// a group's whole units stay below this; teams of more lanes run inline.
inline constexpr std::uint64_t max_counted = std::uint64_t{1} << 22U;

// Rule 1 for a whole job, whatever the team: it takes less than inline_multiple dispatch
// costs. Plans decide it once at construction (plan_core::runs_inline), so that the executor
// overload of a small job touches neither the executor nor the scheduler.
[[nodiscard]] bool runs_inline(const work_graph& g, const machine_profile& profile) noexcept;

[[nodiscard]] schedule make_schedule(const work_graph& g, const lane_map& lanes,
                                     const machine_profile& profile) noexcept;

[[nodiscard]] group_share share_of(const schedule& s, const lane_map& lanes, int group) noexcept;

// The units a group's whole tasks count as.
[[nodiscard]] std::size_t whole_units(const schedule& s, const group_share& share) noexcept;

// What the execution's done counter reaches when every task is done: 1 inline, else every
// group's whole units and split tasks.
[[nodiscard]] std::uint32_t done_target(const schedule& s, const lane_map& lanes) noexcept;

// completion::own: the static deal of `tasks` over `lanes` (fftn::plan::transforms_of).
// With at least as many tasks as lanes, contiguous ranges differing by at most one; with
// fewer, the first `tasks` lanes one each, so that a small batch stays on few members.
[[nodiscard]] transform_range owned_tasks(std::size_t tasks, int lanes, int lane) noexcept;

} // namespace fftn::inline v0::detail
