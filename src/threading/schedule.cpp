// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "threading/schedule.hpp"

#include "core/program.hpp"
#include "fftn/isa.hpp"
#include "fftn/team.hpp"
#include "platform/cpu_features.hpp"
#include "platform/topology.hpp"
#include "threading/cache_groups.hpp"
#include "threading/work_graph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>

namespace fftn::inline v0::detail {

machine_profile profile_for(const topology& t, bool x86_64, cpu_vendor vendor) noexcept {
  machine_profile p;
  p.vendor = vendor;
  if (!t.classes.empty()) {
    set_grid_sizes(p, t.classes.front(), x86_64);
  }
  if (t.page_size != 0) {
    p.page_bytes = t.page_size;
  }
  // The defaults are a Zen 3 CCX's. Two other kinds of cache group were measured (T1, T5):
  // an Apple processor (macOS's reader, which numbers CPUs itself, topology.hpp, and an L2
  // shared by a cluster of cores rather than each core's own: an Intel Mac has per-core L2s
  // and takes the defaults), whose performance cluster runs a short row so fast that
  // splitting pays only from longer rows; a last-level group of more than mesh_group_lanes
  // cores is a mesh (the Xeon's 18 cores). On neither did T1's measured costs pass the
  // adoption test, so both keep the provisional synchronisation cost and claim runs.
  const bool cluster_l2 = !t.classes.empty() && t.classes.front().l2.shared_by >
                                                    std::max(t.classes.front().l1d.shared_by, 1U);
  const bool apple = !t.os_cpu_ids && cluster_l2;
  std::size_t cpus_per_core = t.cores.empty() ? 1 : t.cores.front().cpus.size();
  for (const cpu_group& core : t.cores) {
    cpus_per_core = std::min(cpus_per_core, core.cpus.size());
  }
  const auto group_cpus = static_cast<std::size_t>(cpus_per_cache_group(t));
  const bool mesh = group_cpus > static_cast<std::size_t>(p.mesh_group_lanes) *
                                     std::max<std::size_t>(cpus_per_core, 1);
  if (apple) {
    p.split_min_points = 8192;
  }
  if (apple || mesh) {
    p.sync_ns = 700;
    p.claim_divisor = 2;
  }
  if (mesh) {
    p.split_min_points_lines = 32768;
  }
  return p;
}

machine_profile default_machine_profile() noexcept {
  machine_profile p;
  const cpu_vendor vendor = system_cpu_vendor();
  try {
    p = profile_for(system_topology(), FFTN_ARCH_X86_64 != 0, vendor);
  } catch (...) {
    // system_topology() throws on an unreadable FFTN_TOPOLOGY recording; this function
    // is noexcept, so the defaults stand (as cache_group_size does).
    p = machine_profile{};
    p.vendor = vendor;
  }
  const unsigned cpus = std::thread::hardware_concurrency();
  p.cpus = static_cast<int>(std::min<unsigned>(cpus, std::numeric_limits<int>::max()));
  return p;
}

// Experiment X12b. Where each core has its own L2 (x86: shared only by a core's SMT
// threads, like its L1), a chain streams from it well: the grid paid only beyond half of
// L2, and three phases only beyond L2 (Xeon Gold 6240, EPYC 7763). An L2 shared by a
// cluster (Apple) is far: there the chain lost twofold once the row left L1, and three
// phases paid from an eighth of L2 (X12).
void set_grid_sizes(machine_profile& p, const core_class& c, bool x86_64) noexcept {
  const bool private_l2 = c.l2.shared_by != 0 && c.l2.shared_by <= std::max(c.l1d.shared_by, 1U);
  // A rule measured on one vendor's processors only is keyed to that vendor (the maintainer,
  // 2026-11-25; DESIGN.md 13.5): a cache key cannot tell an Intel core from an AMD one with the
  // same caches (Zen 4c and Zen 5c, 1 MiB of L2 and of L3 per CPU with SMT, match the Xeon 6240's
  // key). The Xeon 6240's class (X34, X39, X46, X50, X51, X58, X61, X63, X75, X78) and the
  // 2 MiB-L2 key (X35, X55, X58, X64, X69, X70) were measured on Intel's cores only; every other
  // vendor, unknown included, keeps those fields' defaults there. Likewise Zen 3's class (below,
  // zen3_like: X51, X59, X60, X62, X63, X65, X67, X72, X73, X85) was measured on AMD's cores only
  // (the EPYC 7763; Zen 2 in X64's and X81's screens), and an Intel core with a private L2 under
  // 1 MiB and 2 MiB of L3 per CPU (a client part, a small VM) matches its caches: the maintainer
  // keyed it to AMD on 2026-11-27.
  const bool intel = p.vendor == cpu_vendor::intel;
  const bool amd = p.vendor == cpu_vendor::amd;
  if (c.l1d.size != 0) {
    p.grid_min_bytes = c.l1d.size;
    p.grid_block_bytes = c.l1d.size;
    p.adjacent_min_bytes = c.l1d.size;
    p.callback_block_bytes = static_cast<std::uint32_t>(
        std::min<std::size_t>(c.l1d.size / 2, std::numeric_limits<std::uint32_t>::max()));
  }
  if (c.l1d.line_size != 0) {
    p.line_bytes = c.l1d.line_size;
  }
  if (c.l1d.ways != 0) {
    p.l1_ways = c.l1d.ways;
  }
  if (c.l2.size != 0) {
    if (private_l2) {
      p.grid_min_bytes = std::max(p.grid_min_bytes, c.l2.size / 2);
    }
    p.grid_three_bytes = private_l2 ? c.l2.size : c.l2.size / 8;
    // X26b: padding the intermediate paid on the Mac (0.976 over the long rows) and nothing
    // on the Xeon and the EPYC, whose L1 refills it did not cut.
    p.grid_pad_period = private_l2 ? std::numeric_limits<std::size_t>::max() : 4096;
    p.grid_fuse_bytes = c.l2.size / 4;
    p.column_tile_bytes = c.l2.size / 4;
    p.split_complex_points = private_l2;
    // X41 arm L2: where a cluster shares L2 (Apple), a block of columns whose elements lie at
    // least four to a page keeps its tiles within twice L1 (M5 Max 2048 x 36: blocks of 4
    // columns, 256 KiB of tiles, 0.90 of the time of X40's 16, 1 MiB), as the chain lost
    // there once a row left L1 (X12b); 1000 x 1000, almost a page per element, lost 1.03
    // with that budget (arm L1).
    if (!private_l2 && c.l1d.size != 0) {
      p.column_near_tile_bytes = 2 * c.l1d.size;
    }
  }
  // Experiment X29: with a private L2 and an L3 of at least 2 MiB per CPU sharing it (the
  // EPYC 7763's CCX: 32 MiB for 8 cores), a row whose length has a power-of-two part of at
  // most 1024 points took 0.64-0.85 of the grid's time as a chain, steadily, to 2 Mi points
  // (46080, 100000, 10^6 and the real rows of 46080 and 2 x 10^6), so it stays a chain to
  // half of L3. Powers of two do not: their chains of 1-2 MiB ran 1.1-1.8 times the grid's
  // time in some processes and 0.7 in others (leg distances a multiple of the L2's way,
  // conflicting when the pages lie contiguous), and the Xeon 6240 (0.7 MiB of L3 per CPU)
  // lost with chains beyond L2/2 at every length.
  const bool large_l3 = c.l3.size != 0 && c.l3.shared_by != 0 &&
                        c.l3.size / c.l3.shared_by >= std::size_t{2} * 1024 * 1024;
  // Experiment X35: on x86-64 a private L2 of 2 MiB or more (Granite Rapids; Intel's server
  // cores since Sapphire Rapids, against Zen 2-5's 0.5-1 MiB) is its own case, whatever the
  // L3 per CPU, which depends on the VM's size (a Granite Rapids VM of 8 cores reports 63 MiB
  // per CPU). On that VM, against X29's, X32's and X33's chains: grids beyond L2/2 (long
  // powers of two 0.61-0.97 of the chains' time, long real rows 0.59-0.88), three phases
  // beyond L2/2 (131072 0.74 against 0.90 in two), but two phases where the power-of-two
  // part is at most 2048 complex doubles (100000 0.67, 10^6 0.58; 71680 lost 1.10 in three;
  // X34's odd-part rule lost 1.03-1.46). Not on AArch64: an Axion VM (Neoverse V2, 2 MiB of
  // private L2) lost 1.28 over the rows this profile changed, and keeps X29's and X32's chains.
  // Measured on Intel's cores only (Granite, Sapphire and Emerald Rapids): the branch below reads
  // the vendor, and another vendor's 2 MiB L2 keeps the defaults (no large-L3 or Xeon rules).
  const bool large_l2 = x86_64 && private_l2 && c.l2.size >= std::size_t{2} * 1024 * 1024;
  // X43 arms B1 and B2, on a private L2 with a large L3 behind it (the EPYC 7763; Zen-like): at
  // 8 lanes the team program's blocks of 8 columns (128 bytes of complex doubles) took 0.887 of
  // the time of X40's 16 over the c2c planes and volumes they change (1000^2 0.66, 128^2 0.75:
  // 16 columns of 1000-2048 fill or overfill each member's 512 KiB L2, and twice the blocks
  // share out better), and blocks of a line where one thread runs columns alone 0.72 at
  // 2048 x 36; one thread lost up to 1.42 with the narrow blocks. On the Xeon 6240 (1 MiB L2,
  // little L3) they took 0.997: 512^2-2048^2 lost 1.05-1.15, so it builds no team blocks.
  if (x86_64 && private_l2 && large_l3 && !large_l2) {
    p.column_team_bytes = 128;
  }
  // X43b arms VT and VTX: teams run volumes plane at a time (8 lanes: EPYC 0.775, Xeon 6240
  // 0.771), where one thread gained too little (0.979, 0.934 with 97^3 slower, 1.048).
  if (x86_64 && private_l2 && !large_l2) {
    p.team_volume_planes = true;
  }
  // Zen 3's class: x86-64 with a private L2 under 1 MiB and a large L3 (the EPYC 7763's CCX; Zen
  // 2's too), on AMD's cores only (the maintainer, 2026-11-27: every rule keyed to it was measured
  // on AMD's cores alone). Another vendor's cores with those caches, unknown included, keep the
  // large-L3 branch's fields (X29, X32, X36, X43, X57) and the defaults of the class's own.
  // X59 arm R6 (machine_profile::fair_line_runs), only where it was measured: Zen 3's class, as
  // X51's arm E, and a cluster-shared L2 (Apple). Zen 4 and Zen 5 (1 MiB of L2, `avx512`) share
  // the large-L3 branch, unmeasured, and keep the guided runs; so do the Xeon 6240's profile,
  // where it lost, and Granite Rapids.
  const bool zen3_like = amd && x86_64 && private_l2 && large_l3 && c.l2.size != 0 &&
                         c.l2.size < std::size_t{1024} * 1024;
  // X61 (machine_profile::adjacent_lanes), only where it was measured and passed: the Xeon
  // 6240's class (x86-64, a private L2 below 2 MiB without Zen 3's large L3 per core) and Apple
  // (a cluster-shared L2), both with a team running the chain, which Apple's unmeasured 8 lanes
  // need as much as the x86 machines did. Off on the EPYC 7763's class (0.946 against the bar of
  // 0.85), on Granite and Sapphire Rapids, other AArch64 and unknown topologies, and on x86 cores
  // of other vendors than Intel (the x86 half was measured on the Xeon 6240 alone).
  const bool apple_like = !x86_64 && c.l2.shared_by != 0 && !private_l2;
  if ((intel && x86_64 && private_l2 && !large_l2 && !large_l3) || apple_like) {
    p.adjacent_lanes = true;
    p.adjacent_team_chain = true;
  }
  if (zen3_like || (c.l2.shared_by != 0 && !private_l2)) {
    p.fair_line_runs = true;
  }
  // X73 (machine_profile::anchored_runs), where it was measured: Zen 3's class (AMD's cores; 16
  // unpinned threads: out-of-place batches 0.32-0.38 of their time; Zen 2 kept it, X81), X74, Apple
  // (6 threads on the M5 Max: the standard batches 0.87-0.97, the client catalogue 0.996, none
  // slower) and, X79, the Xeon 6240's class (X61's key, Intel's cores only) once the team state had
  // pages of its own: X74's one loss, c2c 864 x 384 at 16 threads, was the state's page shared with
  // the plan's twiddles (X76).
  const bool xeon6240_like = intel && x86_64 && private_l2 && !large_l2 && !large_l3;
  if (zen3_like || apple_like || xeon6240_like) {
    p.anchored_runs = true;
  }
  // X65's P1, narrowed (machine_profile::edge_batched_forms): plans of c2c rows on one leaf with
  // factor edges run in the edge driver's batched forms, where it was measured: Zen 3's class
  // (EPYC 7763, `avx2`; AMD's cores only, by the maintainer's policy that a rule measured on one
  // vendor is keyed to it) and the Xeon 6240's (`avx512`, Intel's cores only).
  if (zen3_like || xeon6240_like) {
    p.edge_batched_forms = edge_forms_reach::leaf_rows;
  }
  // Experiment X67 (machine_profile::pipeline_depth_min_kib): <C, C, C> pipelines' lines on the
  // depth-first pair on the Xeon 6240's class, Intel's cores only, on avx512 (the target
  // measured), from 288 KiB of the line to 2 MiB in double (18432 to 131072 points: 0.64-0.81
  // of the chain pair's time; float from 36864 to 131072 points, where 32768 to 131072 took
  // 0.66-0.94). Without fan-out (core/pipeline.cpp, depth_measured).
  // The EPYC 7763's class keeps the chain pair (the pair took 1.0-3.4 of its time there).
  if (xeon6240_like) {
    p.pipeline_depth_min_kib = 288;
    p.pipeline_depth_max_kib = 2048;
    p.pipeline_depth_isa = fftn::isa::avx512;
  }
  // Experiment X67 (machine_profile::pipeline_lanes_adjacent_max): a plane's strided lines on the
  // lanes pair from 64 to 512 points, where it was measured: Zen 3's class, AMD's cores only, and
  // the Xeon 6240's (the lines of 64 x 512 and 512 x 512 planes took 0.46-0.96 of the chain pair's
  // time on the EPYC 7763, float most, and 0.58-0.72 on the Xeon).
  // Each on processors whose best target is the one measured (avx2 on Zen 3, avx512 on the Xeon):
  // a key of the machine, so that both targets of one machine take the rule alike, and an Intel
  // core of the Xeon's cache class without avx512 keeps the chain pair (AGENTS.md section 6).
  if (zen3_like) {
    p.pipeline_lanes_adjacent_max = 512;
    p.pipeline_lanes_isa = fftn::isa::avx2;
  }
  if (xeon6240_like) {
    p.pipeline_lanes_adjacent_max = 512;
    p.pipeline_lanes_isa = fftn::isa::avx512;
  }
  // X66's rule 3 (machine_profile::callback_block_bytes): on the Xeon 6240's class (Intel's cores
  // only) blocks of a quarter of L2 passed in both precisions over the cubic callback cases whose
  // blocks they change, none slower: double 0.952 on the deciding re-run (run
  // 20261127-021303-3fa6), float 0.839. A plane's lanes column block (16 columns of 192-384
  // points) reaches the callback in one call instead of 3-6 (0.61-0.77 on the planes). On the
  // EPYC 7763 double took 0.979, short of the rule's bar, so Zen 3's class and every other
  // machine keep half the L1.
  if (xeon6240_like && c.l2.size != 0) {
    p.callback_block_bytes = static_cast<std::uint32_t>(
        std::min<std::size_t>(c.l2.size / 4, std::numeric_limits<std::uint32_t>::max()));
  }
  // X85's arm D (machine_profile::r2r_folded_max): odd DCT-I (from 33 points, core/r2r.cpp) and
  // DST-I up to 129 points take the folded route on Zen 3's class (AMD's cores; the EPYC 7763,
  // measured on `avx2`), a rule keyed by the machine (r2r_folded_isa: its best target), so that
  // every target of such a machine takes it alike. The route's odd outputs are running sums, its
  // error grows like sqrt n: rule D1's rms sweep stayed inside half the r2r tests' bound to 257
  // points, but the tests' largest error of the kind twice failed at 251 on avx2
  // (tests/api/r2r_test.cpp, DST-I) and from 261 on the baseline targets, so the limit is 129,
  // where that ratio is about three quarters of its 251 value. Over the 35 cases it moves the
  // route took 0.794 of the extensions' time, none slower (the deciding run
  // 20261127-083621-be45).
  if (zen3_like) {
    p.r2r_folded_max = 129;
    p.r2r_folded_isa = fftn::isa::avx2;
  }
  if (x86_64 && private_l2 && large_l3 && !large_l2) {
    p.column_wide_bytes = 1024;
    p.fit_row_bytes = c.l3.size / 4 * 3; // X57: three quarters of the CCX's L3
    // X57 amendment 2: in place, half the L3 (1024 rows of 1024 points on the 32 MiB L3).
    p.fit_in_place_bytes = c.l3.size / 2;
  }
  // Experiment X58, an Ice Lake VM (Google Cloud N2: private L2 1.25 MiB, 6.9 MiB of L3 per
  // CPU, so the large-L3 branch below): the depth-first chain as on the Xeon 6240. One top of
  // 16 took 0.50-0.68 of the time from 32768 to 2^22 points, 20 took 0.49 at 100000. Zen 4
  // and Zen 5 (exactly 1 MiB of L2) stay out: unmeasured there.
  if (intel && x86_64 && private_l2 && !large_l2 && c.l1d.size != 0 &&
      c.l2.size > std::size_t{1024} * 1024) {
    p.depth_min_bytes = c.l2.size / 4;
    p.depth_max_bytes = 64 * c.l2.size;
    p.depth_top = 16;
    p.depth_top_alt = 20;
    p.depth_top_last = 10;
    // Experiment X82 (cloud screen of 2026-11-25, the N2 VM): the Xeon 6240's real rows on that
    // chain. X75: an even real row out of place runs its half-length transform there from half
    // lengths above L2/8 to the chain's 64 L2 (0.745 over the 40 rows it moved, 24576-2^21
    // points, none slower; run 20261125-095241-aafe). X78: c2r rows pair in the chain's first
    // pass from L2/4 to 16 L2 (0.772 over the 16 it moved). Rows of 2^22 and 2^23 points are
    // reached unmeasured on Ice Lake (X75 timed them on the Xeon 6240: 0.66-0.89).
    p.depth_real_max_bytes = p.depth_max_bytes;
    p.depth_real_min_bytes = c.l2.size / 8 + 1;
    p.depth_c2r_fused_min_bytes = c.l2.size / 4;
    p.depth_c2r_fused_max_bytes = 16 * c.l2.size;
  }
  if (large_l2 && intel) {
    p.batch_leaf_12 = false; // X58
    p.grid_three_bytes = c.l2.size / 2;
    p.grid_two_part_bytes = std::size_t{32} * 1024;
    // Experiment X55, Granite Rapids VM (Xeon 6985P-C, 2 MiB private L2): X46's and X51's
    // depth-first chain as on the Xeon 6240, from L2/4 to 64 L2 in double. One top of 16
    // took 0.44-0.81 of the grids' time in the screen (46080 0.44, 65536 0.52, 131072-2^20
    // 0.62-0.81, 2^22 0.67; run 20261118-084057-ff56); 2^21 tied (1.03).
    if (c.l1d.size != 0) {
      p.depth_min_bytes = c.l2.size / 4;
      p.depth_max_bytes = 64 * c.l2.size;
      p.depth_top = 16;
      p.depth_top_alt = 20;
      p.depth_top_last = 10;
      // Experiment X64 (cloud screen of 2026-11-24): X50's Bluestein convolutions on that chain
      // up to the Xeon 6240's limit (2^21 complex doubles, 32 MiB). Over the nine convolutions
      // they move (4093-999983), 0.588 of the time on Sapphire Rapids (c3-standard-8), 0.564 on
      // Granite Rapids and 0.615 on Emerald Rapids (c4-standard-16), none slower; FFTN's time
      // over oneMKL's there: 131071 0.74-0.76, 999983 0.84-0.93.
      p.depth_convolution_max_bytes = std::size_t{32} << 20U;
    }
    // Experiment X69: the key's fit table (`avx512` double: 1024 = 32 x 32, 160 = 8 x 20,
    // 240 = 12 x 20) while rows and output take at most the L2 (64 rows of 1024: 16 rows,
    // 512 KiB, won 0.73-0.82 and 256 rows, 8 MiB, tied on Sapphire and Granite Rapids), in
    // place rows at most half the L2 (the same rows: X57's in-place crossing on the EPYC fell
    // at about the out-of-place row count). Judged on Sapphire, Granite and Emerald Rapids:
    // 0.796, 0.780 and 0.800 of the time over the 32 cases it moves (1-64 rows of 1024, 1-256
    // of 160 and 240), none slower.
    // Experiment X70: X69's arm W kept gaining past those
    // limits for 1024 in place (2 and 4 MiB of rows: 0.82-0.94) and for 160 and 240 (5 and
    // 7.5 MiB of rows and output: 0.83-0.90), shrinking with size; 1024 out of place tied past
    // the L2 and lost at 32 MiB on Sapphire Rapids (1.29-1.36). So rows and output within four
    // times the L2 (1638 rows of 160, 1092 of 240), in place rows within twice the L2 (256 rows
    // of 1024, the largest win measured), and rows longer than a page (1024) out of place
    // within the L2 (X69's 64 rows). Judged on Sapphire, Granite and Emerald Rapids: 0.858,
    // 0.860 and 0.873 of the time over the 18 cases it moves, none slower.
    // Experiment X80: X70's arm W kept gaining past those limits, in place at every size tried
    // (rows of 1024 to 32 MiB 0.88-0.93, of 160 and 240 to 240 MiB, in DRAM, 0.82-0.94: two
    // passes over memory instead of three) and out of place for 160 and 240 to 120 MiB of rows
    // and output (0.85-0.94), but not at 320 and 480 MiB on Granite Rapids (1.02-1.03). So in
    // place no limit (fit_row_bytes no limit either; it turns the fit rows on), out of place rows
    // and output within 64 times the L2 (26214 rows of 160, 17476 of 240), and rows longer than a
    // page (1024) out of place within the L2 as before. Judged on Sapphire, Granite and Emerald
    // Rapids: 0.899, 0.903 and 0.894 of the time over the 22 cases it moves (to 512 MiB in
    // place), none slower.
    p.fit_large_l2 = true;
    p.fit_row_bytes = std::numeric_limits<std::size_t>::max();
    p.fit_in_place_bytes = std::numeric_limits<std::size_t>::max();
    p.fit_out_of_place_bytes = 64 * c.l2.size;
    p.fit_long_row_bytes = c.l2.size;
  } else if (private_l2 && large_l3 && !large_l2) {
    p.chain_part_bytes = std::size_t{16} * 1024;
    p.chain_part_limit_bytes = c.l3.size / 2;
    // Experiment X32, EPYC 7763: power-of-two rows as chains of FFTL.jl's radix-8 sequence
    // took 0.57-0.85 of the grid's time from 32768 to 2^22 points (64 MiB, twice L3, the
    // largest measured), steadily: X29's bimodal chains were the rules' radix-16 passes on
    // huge pages. Radix 8 beat radix 4 from 65536 points up.
    p.pow2_chain_bytes = 2 * c.l3.size;
    p.pow2_radix8_bytes = std::numeric_limits<std::size_t>::max();
    // Experiment X36, EPYC 7763 (run 20261116-193025-47d4): Bluestein lengths with a
    // power-of-two part of at most 4096 double points took 0.36-0.69 of the time of powers
    // of two at 4093-32749 (two scratch lines a power of two apart); 552960 (2^12 x 135) lost
    // to 2^19 (1.17), so the limit is the L1's size. Experiment X62, on Zen 3's class only
    // (zen3_like; Zen 4 and 5 and the Ice Lake VMs share this branch, unmeasured): half the
    // L1, and a stage over C copies over the whole line: 1013-1021 and 1018 took 0.93 of
    // their time with M = 2160 for 2048, 131071 0.98 (268800 for 276480). AMD's cores only:
    // another vendor's cores with Zen 3's caches take X36's L1.
    p.bluestein_pow2_part_bytes = zen3_like ? c.l1d.size / 2 : c.l1d.size;
    p.bluestein_whole_lanes = zen3_like;
    // Experiment X51 (arm E), EPYC 7763: depth-first chains with a top of 4 (16 registers:
    // tops of 16 lost, X46) for rows that are not powers of two, from L2/4 to 4 L2, whose
    // power-of-two part is above L1/4: rows of a small odd part times 2^k took 0.63-0.82 of
    // the rules' time from 12288 to 163840 points in the screen (FFTW 0.70-0.84), 46080 0.83;
    // powers of two tied or lost (X32's chains), and 20000-100000 with parts of 8-64 points
    // gained little or lost (25000 1.16). Only where it was measured: x86-64 with a private
    // L2 under 1 MiB (Zen 3's class, AMD's cores only); Zen 4 and Zen 5 (1 MiB, `avx512`), the
    // Axion VM (AArch64) and other vendors' x86 cores share this profile branch, unmeasured, and
    // keep their plans.
    if (zen3_like && c.l1d.size != 0) {
      p.depth_min_bytes = c.l2.size / 4;
      p.depth_max_bytes = 4 * c.l2.size;
      p.depth_pow2_max_bytes = 0;
      p.depth_part_bytes = c.l1d.size / 4;
      p.depth_top = 4;
      // Experiment X72: an even real row out of place runs its half-length transform on the
      // same depth-first chain, over the rows' whole range (half lengths of 8192-131072
      // points), with the pairing as a separate pass: the 26 real rows it moved (18432-245760,
      // r2c and c2r) took 0.761 of the time of the breadth-first chains and half-length grids
      // (0.65-0.94; run 20261124-223906-cac2).
      p.depth_real_max_bytes = p.depth_max_bytes;
    }
  }
  // Experiment X34: where those chains do not apply (Xeon 6240, 0.7 MiB of L3 per CPU), a
  // row whose length has an odd part of at least 15 stays in two grid phases beyond L2
  // (15 rows of 71680 to 4 x 10^6 points: 0.70-1.01 of the three phases' cycles, 100000 0.73).
  // Experiment X39, the same machines on x86-64 (measured on the Xeon 6240 only): three-phase
  // grids in blocks of at most 16 columns, grid phases with radix-20 passes, two-phase blocks
  // within twice L1, c2c and r2c powers of two as grids beyond a quarter of L2 (32768 complex
  // points), and c2c powers of two out of place in two phases beyond half of L3 (2^20 points).
  // Every field of this branch was measured on the Xeon 6240 alone, so it is Intel's only: AMD's
  // cores the key reaches (Zen 4c, Zen 5c; Zen 4 VMs reporting a small L3 per CPU) keep the
  // defaults.
  if (intel && private_l2 && !large_l2 && !large_l3) {
    p.grid_two_odd_part = 15;
    if (x86_64) {
      // X79, narrowed to this class by the maintainer (2026-11-25): the team state on 4 KiB
      // pages of its own. X16: c2c 864 x 384 backward at 16 threads unimodal at its alone time
      // (0.739 of its median), the client catalogue 0.996 and the standard batches 0.983 (8
      // threads 0.992, 0.988), nothing slower.
      p.team_state_page = 4096;
      p.grid_three_block = 16;
      p.grid_phase_radix = 20;
      // Experiment X83 (machine_profile::split_complex_rows): real rows on a chain's stages are
      // split by their complex points, so those of 6144-12286 points stay whole in a group of
      // up to 8 members (8192-16382 in a mesh group): teams of 4 and 8 had run them 1.32-2.69
      // times slower than one thread (6400-10240 points, run 20261125-104559-e6f5).
      p.split_complex_rows = true;
      if (c.l1d.size != 0 && c.l2.size != 0) {
        p.grid_two_block_bytes = 2 * c.l1d.size;
        p.pow2_grid_bytes = c.l2.size / 4;
        p.grid_two_pow2_bytes = c.l3.size / 2;
        // Experiment X46: the depth-first chain from L2/4 to 4 L2 (16384-262144 points);
        // X51: a top of 10 where neither 16 nor 20 fits (25000), and the range to 64 L2
        // (2^22 points: one top of 16 took 0.62-0.76 of the grids' time from 524288 to 2^22
        // points, 0.72-0.83 in place; 2^23 and 2^24 gained too little to tell, 0.88-0.94).
        p.depth_min_bytes = c.l2.size / 4;
        p.depth_max_bytes = 64 * c.l2.size;
        p.depth_top = 16;
        p.depth_top_alt = 20;
        // Experiment X50: Bluestein convolutions on the depth-first chain beyond, up to the
        // longest convolution measured (2^21 complex doubles, 32 MiB); longer is unmeasured.
        p.depth_convolution_max_bytes = std::size_t{32} << 20U;
        p.depth_top_last = 10;
        // X51 (arm F): single precision too, by its bytes, to 8 L2 (32768-2^20 points, the
        // measured range): tops of 16 (20 at 100000) took 0.61-0.80 of the rules' time
        // there; every split lost below L2/4 (8192-20000 points, 1.14-2.0).
        p.depth_float_max_bytes = 8 * c.l2.size;
        // Experiment X75: an even real row out of place runs its half-length transform on
        // the same chain over the rows' whole range (half lengths of 16384-2^22 complex
        // doubles), as X72 on Zen 3: every real row it moved gained, 32768-2^23 points, r2c
        // and c2r, double and single (0.59-0.89 of the time in the diagnosis, run
        // 20261125-004940-024f; 2^21-2^23 0.66-0.89, run 20261125-010730-90ff).
        p.depth_real_max_bytes = p.depth_max_bytes;
        // X75 arm B: real rows whose half length is above L2/8 (18432-24576 points took
        // 0.73-0.95 of the time, single precision 40960 and 49152 0.61-0.95; runs
        // 20261125-023550-8fee, 20261125-040403-2740). Narrowed to exclude L2/8 itself, where
        // 16384 tied and single precision r2c 32768 lost (1.095 in run 20261125-040403-2740).
        p.depth_real_min_bytes = c.l2.size / 8 + 1;
        // Experiment X78: c2r rows on that chain pair their bins in its first pass, with the
        // pairing twiddles factored over the pass's legs, from L2/4 to 16 L2 of half length
        // (0.69-0.97 of the separate pass's time at one thread, run 20261125-060137-dd4d).
        // Half lengths of 32 and 64 MiB lost (1.27-1.30), and single-precision ones below L2/4
        // (1.06-1.07).
        p.depth_c2r_fused_min_bytes = c.l2.size / 4;
        p.depth_c2r_fused_max_bytes = 16 * c.l2.size;
      }
    }
  }
  // Experiment X29, M5 Max: with the grid from L1, rows whose length has a power-of-two part
  // of at most 1024 points took 0.49-0.72 of the grid's time as chains (to 4 Mi points),
  // and every other length that is not a power of two 0.56-0.63 up to twice L1.
  if (c.l1d.size != 0 && c.l2.size != 0 && !private_l2) {
    p.chain_part_bytes = c.l1d.size / 8;
    p.chain_mixed_bytes = 2 * c.l1d.size;
    // Experiment X32, M5 Max: power-of-two rows as chains of FFTL.jl's radices took
    // 0.57-0.93 of the grid's time at every length from 16384 to 2^22 points (64 MiB, four
    // times L2, the largest measured); radix 8 won up to 131072 points (2 MiB, L2/8),
    // radix 4 from 262144.
    p.pow2_chain_bytes = 4 * c.l2.size;
    p.pow2_radix8_bytes = c.l2.size / 8;
    // Experiment X36, M5 Max: the same conflicts in a Bluestein stage's convolution.
    p.bluestein_pow2_part_bytes = c.l1d.size / 4;
    // Experiment X46 (arm C2), M5 Max: depth-first chains with a top of 4 from L1 to 2 L1
    // where the length's power-of-two part is above L1/8 (8192-16384 complex doubles; from
    // L1/2, batches of 256-1024 rows of 4096 lost 1.03-1.04, arm C). X51 (arm M): rows of a
    // small odd part times a power of two above L1/8 to 16 L1 (20480-122880 points took
    // 0.67-0.86 of their grids' time in the screen); powers of two stay at 2 L1 (32768-262144
    // gained nothing against X32's chains).
    p.depth_min_bytes = c.l1d.size;
    p.depth_max_bytes = 16 * c.l1d.size;
    p.depth_pow2_max_bytes = 2 * c.l1d.size;
    p.depth_part_bytes = c.l1d.size / 8;
    p.depth_top = 4;
    // Experiment X50: Bluestein convolutions on the depth-first chain over the same range
    // (primes 4813-5119, M 10240, 0.86 of their time on the M5 Max).
    p.depth_convolution_max_bytes = 2 * c.l1d.size;
  }
  p.c2r_paired_gather = c.l2.size != 0 && !private_l2;
  // Experiments X60 and X63 (machine_profile::c2r_packed_bytes): a c2r plane's intermediate
  // packed in the output, on the classes and targets measured; elsewhere the block. Volumes keep
  // the block everywhere (X60: the Xeon 6240's teams lost on them, 96^3-128^3 1.05-1.09 at 8
  // lanes). The Xeon 6240's class (avx512): from its L2 (X60, planes from 384^2 0.70-1.01 at
  // one thread and at 8 lanes). Zen 3's (avx2): where one half spectrum passes half the CCX's
  // L3 (X60's 8 lanes lost at 8 MiB, 1000^2 and 1024^2 1.04-1.13, and gained at 32 MiB, 2048^2
  // 0.76-0.83; X63 judged 1536^2-4096 x 1024 from 16 MiB). The threshold follows the CCX's L3,
  // which the three arrays and a team share, so Zen 2 (16 MiB per CCX, the same key) packs from
  // 8 MiB (kept by X64's screen of Zen 2). Apple keeps the block: wide planes lost at every
  // threshold tried (M5 Max 512 x 1024 1.07 at 4 MiB, 640 x 1280 1.14 at 6.3 MiB). The Xeon 6240's
  // key holds for Intel's cores only (X64, the cloud screen of 2026-11-24): it also reaches Zen 4
  // (Zen 4c with SMT, Zen 4 VMs reporting a small L3 per CPU), and on an EPYC 9B14 VM with the
  // threshold at 1 MiB the packed planes lost at one thread (512 x 1024 1.127, 512^2 1.109) and at
  // 8 lanes (600 x 800 1.286, 720^2 1.246), while Ice Lake-SP kept the gain (0.62-0.77). AMD's
  // cores with avx512, and unknown ones, keep the block. Zen 3's branch was measured on AMD's
  // cores only (the EPYC 7763; Zen 2 in X64's screen), so it is AMD's (zen3_like).
  if (intel && c.l2.size != 0 && x86_64 && private_l2 && !large_l2 && !large_l3) {
    p.c2r_packed_bytes = c.l2.size;
    p.c2r_packed_isa = fftn::isa::avx512;
  } else if (zen3_like) {
    p.c2r_packed_bytes = c.l3.size / 2;
    p.c2r_packed_isa = fftn::isa::avx2;
  }
  // Experiment X36: long convolutions (machine_profile::long_convolution_bytes).
  if (p.pow2_chain_bytes != 0) {
    p.long_convolution_bytes = p.grid_min_bytes;
  } else if (c.l2.size != 0) {
    p.long_convolution_bytes = 4 * c.l2.size;
  }
}

bool runs_inline(const work_graph& g, const machine_profile& profile) noexcept {
  return static_cast<double>(g.tasks) * (g.work * profile.ns_per_work) <
         profile.inline_multiple * profile.dispatch_ns;
}

schedule make_schedule(const work_graph& g, const lane_map& lanes,
                       const machine_profile& profile) noexcept {
  schedule s;
  s.profile = profile;
  s.tasks = g.tasks;
  s.task_ns = g.work * profile.ns_per_work;
  s.phases = g.phases.size();
  s.widest = g.widest;
  s.points = g.points;
  s.lines = std::ranges::any_of(g.phases, [](const graph_phase& ph) { return ph.nest != 0; });
  s.splittable = g.chunks > 1 && g.chunks < max_counted;
  s.oversubscribed = profile.cpus > 0 && std::max(lanes.lanes, lanes.threads) > profile.cpus;
  s.grain = g.tasks < max_counted ? 1 : (g.tasks + max_counted - 1) / max_counted;
  // Rule 1, and teams too large for the counters.
  if (lanes.lanes <= 1 || std::cmp_greater_equal(lanes.lanes, max_counted) ||
      runs_inline(g, profile)) {
    s.inline_all = true;
    s.lanes_dealt = lanes.size_of(0);
    s.dispatch = 1;
    return s;
  }
  // Rules 4 and 5: as many groups as have a group's worth of tasks, at least one; a batch
  // smaller than a group stays in one.
  const auto per_group = static_cast<std::size_t>(lanes.per_group());
  const auto groups = static_cast<std::size_t>(lanes.groups());
  s.groups = static_cast<int>(std::min(groups, (g.tasks + per_group - 1) / per_group));
  s.lanes_dealt = std::min(lanes.lanes, s.groups * lanes.per_group());
  s.dispatch = s.lanes_dealt;
  if (std::cmp_less(g.tasks, s.lanes_dealt)) {
    bool splits = false;
    for (int h = 0; h < s.groups && !splits; ++h) {
      splits = share_of(s, lanes, h).split != 0;
    }
    if (!splits) {
      s.dispatch = static_cast<int>(g.tasks);
    }
  }
  return s;
}

group_share share_of(const schedule& s, const lane_map& lanes, int group) noexcept {
  group_share share;
  if (group >= s.groups) {
    return share;
  }
  // Contiguous ranges in proportion to the groups' members, in exact integers.
  const auto dealt = static_cast<std::size_t>(s.lanes_dealt);
  const auto from = static_cast<std::size_t>(lanes.base(group));
  const auto to =
      static_cast<std::size_t>(std::min(lanes.base(group) + lanes.size_of(group), s.lanes_dealt));
  share.first = chunk_start(s.tasks, dealt, from);
  share.count = chunk_start(s.tasks, dealt, to) - share.first;
  share.whole = share.count;
  if (s.inline_all || s.oversubscribed || !s.splittable || share.count == 0) {
    return share;
  }
  // Rules 2 and 3.
  const auto members = static_cast<std::size_t>(lanes.size_of(group));
  const machine_profile& p = s.profile;
  const bool mesh = lanes.size_of(group) > p.mesh_group_lanes;
  if (s.points < (mesh ? p.split_min_points_mesh : p.split_min_points) ||
      (s.lines && s.points < p.split_min_points_lines)) {
    return share;
  }
  std::size_t split = 0;
  if (share.count >= members) {
    split = share.count % members;
    if (split == 0 || s.points < p.remainder_split_min_points) {
      return share;
    }
  } else {
    split = share.count;
  }
  const std::size_t by_points = s.points / std::max<std::size_t>(p.min_points_per_member, 1);
  const std::size_t cap = std::min({members / split, by_points, std::size_t{s.widest}});
  if (cap < 2) {
    return share;
  }
  const double team_ns =
      s.task_ns / static_cast<double>(cap) + static_cast<double>(s.phases) * p.sync_ns;
  if (team_ns >= s.task_ns) {
    return share;
  }
  share.whole = share.count - split;
  share.split = split;
  share.cap = static_cast<int>(cap);
  return share;
}

std::size_t whole_units(const schedule& s, const group_share& share) noexcept {
  return (share.whole + s.grain - 1) / s.grain;
}

std::uint32_t done_target(const schedule& s, const lane_map& lanes) noexcept {
  if (s.inline_all) {
    return 1;
  }
  std::size_t units = 0;
  for (int g = 0; g < s.groups; ++g) {
    const group_share share = share_of(s, lanes, g);
    units += whole_units(s, share) + share.split;
  }
  return static_cast<std::uint32_t>(units);
}

transform_range owned_tasks(std::size_t tasks, int lanes, int lane) noexcept {
  const auto l = static_cast<std::size_t>(lanes);
  const auto r = static_cast<std::size_t>(lane);
  if (tasks < l) {
    return r < tasks ? transform_range{.first = r, .count = 1}
                     : transform_range{.first = tasks, .count = 0};
  }
  const std::size_t first = chunk_start(tasks, l, r);
  return {.first = first, .count = chunk_start(tasks, l, r + 1) - first};
}

} // namespace fftn::inline v0::detail
