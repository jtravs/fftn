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

#include "fftn/team.hpp"
#include "isa/arch.hpp"
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
struct machine_profile {
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
  // The L1 data cache's line in bytes: a lanes column phase's block holds at least a line of
  // adjacent columns (core/plan_core.cpp, column_block; experiment X40). set_grid_sizes reads
  // it from the topology (128 on Apple's cores).
  std::size_t line_bytes = 64;
  // The memory page (16 KiB on Apple's processors): a lanes column phase chooses its block
  // by whether a column's elements lie a page or more apart (core/plan_core.cpp,
  // column_block; X40). profile_for reads it from the topology.
  std::size_t page_bytes = 4096;
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
  // 0.36-0.69 of the time); on the Xeon 6240 a limit gained nothing.
  std::size_t bluestein_pow2_part_bytes = std::numeric_limits<std::size_t>::max();
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
};

// The profile of a machine of topology t (its CPU count aside): the defaults above, which
// are a Zen 3 CCX's, with the grid's sizes from t's caches, and an Apple cluster's or a
// mesh's values where T1 and T5 measured them. A pure function of the topology, so that
// the fixtures test it on any machine.
// x86_64 says whether the machine is an x86-64 one (the build's architecture by default):
// X35's rule for a private L2 of 2 MiB or more was adopted there only.
[[nodiscard]] machine_profile profile_for(const topology& t,
                                          bool x86_64 = FFTN_ARCH_X86_64 != 0) noexcept;

// The profile of the running machine: profile_for its topology, with its CPU count.
[[nodiscard]] machine_profile default_machine_profile() noexcept;

// Sets the grid's size fields from the caches of one core class, the fastest (sizes of 0:
// unknown, which keeps the fields' defaults).
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
