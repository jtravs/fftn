// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The target's pass entry points (pass_kernels in passes/pass_args.hpp), instantiated
// inside the region for every codelet radix of src/gen/codelets.hpp, both directions and
// every edge variant. Which rows exist comes from the generated table, so a radix added to
// the generator gets its passes here without any change. The table is constant data, built
// by the compiler (no function computes it at run time): the helpers that choose an entry
// are consteval, so no code is emitted for them (and the coverage gate does not count
// lines that only the compiler runs). No include guard.
//
// A target is compiled as several units (DESIGN.md 6.4): each row unit instantiates some
// rows of the codelet table, codelet_row<T, I>::passes (isa/rows.inl), and the target's main
// unit instantiates the rest of the table and points at those rows (isa/table.inl).

namespace {

inline constexpr edge_op nop = edge_op::none;
inline constexpr edge_op scl = edge_op::scale;
inline constexpr edge_op fac = edge_op::factor;

// A split lane pass (split.inl), or null where radix R has no split codelet.
template <class T, std::size_t R, int Sign, bool Twiddled, bool InIL, edge_op OutOp>
consteval pass_fn split_entry() {
  if constexpr (has_split_codelet(R)) {
    return &split_pass<T, R, Sign, Twiddled, InIL, OutOp>;
  } else {
    return nullptr;
  }
}

// Whether the grid's phases and the real pairing fused into a chain's edge pass take radix
// R: every radix up to 16 and the powers of two above. The primes 17, 19 and 23 (experiment
// X27) are neither: the grid runs only lengths without a prime factor above 13
// (strategies/grid.cpp) and the pairing fuses only into passes of radix 16 or less
// (chain_builder::route), so their entries stay null. Leaving them out keeps the library
// smaller (B3) and avoided a Clang 23 crash in its SLP vectorizer on c2r_first<float, 17>
// for the x86-64 scalar target.
template <std::size_t R> inline constexpr bool phase_radix = R <= dense_codelet_radix || R % 2 == 0;

template <class T, std::size_t R, int Sign, bool Twiddled, edge_op InOp, int Out>
consteval pass_fn phase_entry() {
  if constexpr (phase_radix<R>) {
    return &phase_pass<T, R, Sign, Twiddled, InOp, Out>;
  } else {
    return nullptr;
  }
}

// Whether the depth-first chain's passes (depth.inl, experiment X46) take radix R: as its
// first pass, the first radix of a sub-row's chain; as its top pass, a top factor. The primes
// 17, 19 and 23 and radix 64 are neither: their passes would add code that no rule selects
// (X3: radix 64 spills). Radix 25 is a first pass only: GCC 14.4 crashed (an internal
// compiler error in its value-range pass) compiling the top pass of radix 25, the first use
// of that codelet's decimation-in-time form.
template <std::size_t R>
inline constexpr bool depth_radix = R <= dense_codelet_radix || R == 20 || R == 25 || R == 32;
template <std::size_t R> inline constexpr bool depth_top_radix = depth_radix<R> && R != 25;

template <class T, std::size_t R, int Sign> consteval pass_fn depth_first_entry() {
  if constexpr (depth_radix<R>) {
    return &depth_first_pass<T, R, Sign>;
  } else {
    return nullptr;
  }
}

template <class T, std::size_t R, int Sign, edge_op OutOp> consteval pass_fn depth_top_entry() {
  if constexpr (depth_top_radix<R>) {
    return &depth_top_pass<T, R, Sign, OutOp>;
  } else {
    return nullptr;
  }
}

// A Bluestein convolution's forward first pass and backward top pass with the stage's chirp
// passes fused in (experiment X50 arm C).
template <class T, std::size_t R> consteval pass_fn depth_first_chirp_entry() {
  if constexpr (depth_radix<R>) {
    return &depth_first_pass<T, R, -1, first_input::chirp>;
  } else {
    return nullptr;
  }
}

// A c2r row's half-length chain's first pass with the pairing fused in (experiment X78).
template <class T, std::size_t R> consteval pass_fn depth_first_c2r_entry() {
  if constexpr (depth_radix<R>) {
    return &depth_first_pass<T, R, +1, first_input::c2r>;
  } else {
    return nullptr;
  }
}

template <class T, std::size_t R, edge_op OutOp> consteval pass_fn depth_top_chirp_entry() {
  if constexpr (depth_top_radix<R>) {
    return &depth_top_pass<T, R, +1, OutOp, true, true>;
  } else {
    return nullptr;
  }
}

// The depth-first pair's forward first pass reading one input run from logical 0 and backward
// top pass writing one output run (DESIGN.md 12.2, task 5.2d).
template <class T, std::size_t R> consteval pass_fn depth_first_head_entry() {
  if constexpr (depth_radix<R>) {
    return &depth_first_pass<T, R, -1, first_input::head>;
  } else {
    return nullptr;
  }
}

template <class T, std::size_t R, edge_op OutOp> consteval pass_fn depth_top_part_entry() {
  if constexpr (depth_top_radix<R>) {
    return &depth_top_pass<T, R, +1, OutOp, true, false>;
  } else {
    return nullptr;
  }
}

// The pruned edge passes (task 5.1e), or null where radix R has no pruned codelet.
template <std::size_t R> inline constexpr bool pruned_radix = has_pruned_codelet(R);

template <class T, std::size_t R, int Sign> consteval pass_fn grouped_head_entry() {
  if constexpr (pruned_radix<R>) {
    return &grouped_pass<T, R, Sign, nop, true>;
  } else {
    return nullptr;
  }
}

template <class T, std::size_t R, int Sign, edge_op OutOp> consteval pass_fn last_head_entry() {
  if constexpr (pruned_radix<R>) {
    return &lane_pass<T, R, Sign, false, OutOp, factor_sharing::none, true>;
  } else {
    return nullptr;
  }
}

template <class T, std::size_t R, bool Scaled> consteval pass_fn r2c_last_entry() {
  if constexpr (phase_radix<R>) {
    return &r2c_last<T, R, Scaled>;
  } else {
    return nullptr;
  }
}

template <class T, std::size_t R> consteval pass_fn c2r_first_entry() {
  if constexpr (phase_radix<R>) {
    return &c2r_first<T, R>;
  } else {
    return nullptr;
  }
}

// A leaf for lines of stride 1 (unit_leaf.inl), or null for radices above 32, whose leaves
// no plan takes by default (rules.cpp, leaf_limit_of: 32 at most), and on the x86 targets,
// whose plans take the narrow leaves (isa/narrow_avx2.cpp) up to 23 and no leaf of 32 by
// default: 3.5 % of the x86 library for nothing (X28b).
template <class T, std::size_t R, int Sign, edge_op InOp, edge_op OutOp>
consteval leaf_fn unit_leaf_entry() {
  if constexpr (R <= 32 && id != target::avx2 && id != target::avx512) {
    return &unit_leaf<T, R, Sign, InOp, OutOp>;
  } else {
    return nullptr;
  }
}

// Whether the target has passes of radix R at all. Radix 25 (experiment X42) is a pass
// radix of the x86 sequences only (and of the scalar target, which plans with avx2's
// tables): on NEON its passes lost to radix-5 passes (3125, 25000 on the Mac), so no plan
// takes it there and its rows stay empty, about 5 % of a target's code. An empty row has no
// lane pass, which the chain's checks of forced radices, the lanes strategy and the
// Bluestein convolutions read as "no codelet".
template <std::size_t R>
inline constexpr bool radix_on_target = R != 25 || (id != target::neon && id != target::neon_fcma);

// The passes of the codelet in row I of gen::radices. A plan has one scale, on one edge
// (core/plan_core.cpp), so no pass scales both: that entry stays null rather than
// instantiating a variant nothing selects.
template <class T, std::size_t I> struct radix_entries {
  static constexpr std::size_t R = gen::radices[I];
  static constexpr radix_passes value{
      .grouped =
          {
              {
                  &grouped_pass<T, R, -1, nop>,
                  &grouped_pass<T, R, -1, scl>,
                  &grouped_pass<T, R, -1, fac>,
              },
              {
                  &grouped_pass<T, R, +1, nop>,
                  &grouped_pass<T, R, +1, scl>,
                  &grouped_pass<T, R, +1, fac>,
              },
          },
      .lane =
          {
              &lane_pass<T, R, -1, true, nop>,
              &lane_pass<T, R, +1, true, nop>,
          },
      .lane_shared_in = &lane_pass<T, R, +1, true, nop, factor_sharing::input>,
      .last_shared_out = &lane_pass<T, R, -1, false, nop, factor_sharing::output>,
      .last =
          {
              {
                  &lane_pass<T, R, -1, false, nop>,
                  &lane_pass<T, R, -1, false, scl>,
                  &lane_pass<T, R, -1, false, fac>,
              },
              {
                  &lane_pass<T, R, +1, false, nop>,
                  &lane_pass<T, R, +1, false, scl>,
                  &lane_pass<T, R, +1, false, fac>,
              },
          },
      .grouped_head =
          {
              grouped_head_entry<T, R, -1>(),
              grouped_head_entry<T, R, +1>(),
          },
      .last_head =
          {
              {last_head_entry<T, R, -1, nop>(), last_head_entry<T, R, -1, scl>()},
              {last_head_entry<T, R, +1, nop>(), last_head_entry<T, R, +1, scl>()},
          },
      .leaf =
          {
              {
                  {
                      &leaf_lines<T, R, -1, nop, nop>,
                      &leaf_lines<T, R, -1, nop, scl>,
                      &leaf_lines<T, R, -1, nop, fac>,
                  },
                  {
                      &leaf_lines<T, R, -1, scl, nop>,
                      nullptr,
                      &leaf_lines<T, R, -1, scl, fac>,
                  },
                  {
                      &leaf_lines<T, R, -1, fac, nop>,
                      &leaf_lines<T, R, -1, fac, scl>,
                      &leaf_lines<T, R, -1, fac, fac>,
                  },
              },
              {
                  {
                      &leaf_lines<T, R, +1, nop, nop>,
                      &leaf_lines<T, R, +1, nop, scl>,
                      &leaf_lines<T, R, +1, nop, fac>,
                  },
                  {
                      &leaf_lines<T, R, +1, scl, nop>,
                      nullptr,
                      &leaf_lines<T, R, +1, scl, fac>,
                  },
                  {
                      &leaf_lines<T, R, +1, fac, nop>,
                      &leaf_lines<T, R, +1, fac, scl>,
                      &leaf_lines<T, R, +1, fac, fac>,
                  },
              },
          },
      .leaf_unit =
          {
              {
                  {unit_leaf_entry<T, R, -1, nop, nop>(), unit_leaf_entry<T, R, -1, nop, scl>()},
                  {unit_leaf_entry<T, R, -1, scl, nop>(), nullptr},
              },
              {
                  {unit_leaf_entry<T, R, +1, nop, nop>(), unit_leaf_entry<T, R, +1, nop, scl>()},
                  {unit_leaf_entry<T, R, +1, scl, nop>(), nullptr},
              },
          },
      .split_mid =
          {
              {
                  split_entry<T, R, -1, true, false, nop>(),
                  split_entry<T, R, -1, true, true, nop>(),
              },
              {
                  split_entry<T, R, +1, true, false, nop>(),
                  split_entry<T, R, +1, true, true, nop>(),
              },
          },
      .split_last =
          {
              {
                  split_entry<T, R, -1, false, false, nop>(),
                  split_entry<T, R, -1, false, false, scl>(),
                  split_entry<T, R, -1, false, false, fac>(),
              },
              {
                  split_entry<T, R, +1, false, false, nop>(),
                  split_entry<T, R, +1, false, false, scl>(),
                  split_entry<T, R, +1, false, false, fac>(),
              },
          },
      .phase =
          {
              {
                  phase_entry<T, R, -1, true, nop, 0>(),
                  phase_entry<T, R, -1, true, scl, 0>(),
                  phase_entry<T, R, -1, false, nop, 0>(),
                  phase_entry<T, R, -1, false, scl, 0>(),
                  phase_entry<T, R, -1, false, nop, 1>(),
                  phase_entry<T, R, -1, false, nop, 2>(),
              },
              {
                  phase_entry<T, R, +1, true, nop, 0>(),
                  phase_entry<T, R, +1, true, scl, 0>(),
                  phase_entry<T, R, +1, false, nop, 0>(),
                  phase_entry<T, R, +1, false, scl, 0>(),
                  phase_entry<T, R, +1, false, nop, 1>(),
                  phase_entry<T, R, +1, false, nop, 2>(),
              },
          },
      .r2c_last =
          {
              r2c_last_entry<T, R, false>(),
              r2c_last_entry<T, R, true>(),
          },
      .c2r_first = c2r_first_entry<T, R>(),
      .r2c_leaf =
          {
              &r2c_leaf<T, R, false>,
              &r2c_leaf<T, R, true>,
          },
      .c2r_leaf =
          {
              &c2r_leaf<T, R, nop>,
              &c2r_leaf<T, R, scl>,
          },
      .lanes_first =
          {
              &lanes_first<T, R, -1>,
              &lanes_first<T, R, +1>,
          },
      .lanes_last =
          {
              {&lanes_last<T, R, -1, nop>, &lanes_last<T, R, -1, scl>},
              {&lanes_last<T, R, +1, nop>, &lanes_last<T, R, +1, scl>},
          },
      .lanes_leaf =
          {
              {&lanes_leaf<T, R, -1, nop>, &lanes_leaf<T, R, -1, scl>},
              {&lanes_leaf<T, R, +1, nop>, &lanes_leaf<T, R, +1, scl>},
          },
      .depth_first = {depth_first_entry<T, R, -1>(), depth_first_entry<T, R, +1>()},
      .depth_top =
          {
              {
                  depth_top_entry<T, R, -1, nop>(),
                  depth_top_entry<T, R, -1, scl>(),
                  depth_top_entry<T, R, -1, fac>(),
              },
              {
                  depth_top_entry<T, R, +1, nop>(),
                  depth_top_entry<T, R, +1, scl>(),
                  depth_top_entry<T, R, +1, fac>(),
              },
          },
      .depth_first_chirp = depth_first_chirp_entry<T, R>(),
      .depth_top_chirp = {depth_top_chirp_entry<T, R, nop>(), depth_top_chirp_entry<T, R, scl>()},
      .depth_first_c2r = depth_first_c2r_entry<T, R>(),
      .depth_first_head = depth_first_head_entry<T, R>(),
      .depth_top_part = {depth_top_part_entry<T, R, nop>(), depth_top_part_entry<T, R, scl>()},
  };
};

} // namespace

// Row I's entries on this target: radix_entries<T, I>::value, or none (radix_on_target).
// noexcept: it initialises a constinit table (rows.inl), which nothing can throw from.
template <class T, std::size_t I> consteval radix_passes row_entries() noexcept {
  if constexpr (radix_on_target<gen::radices[I]>) {
    return radix_entries<T, I>::value;
  } else {
    return radix_passes{};
  }
}

// Row I of the codelet table for precision T: row_entries<T, I>(), defined by exactly
// one of the target's row units (isa/rows.inl) and read through a pointer by the target's
// table. The only names of the region besides the table with external linkage: data, so no
// unit's copy of a function can stand in for another's. Each target's are distinct
// symbols, in its own namespace.
// NOLINTNEXTLINE(misc-use-internal-linkage): defined in the row units (isa/rows.inl)
template <class T, std::size_t I> struct codelet_row {
  static const radix_passes passes;
};

namespace {

static_assert(gen::radix_count == 22, "pass_table lists the rows of gen::radices");

template <class T>
inline constexpr pass_kernels<T> pass_table{
    .codelet =
        {
            &codelet_row<T, 0>::passes,  &codelet_row<T, 1>::passes,  &codelet_row<T, 2>::passes,
            &codelet_row<T, 3>::passes,  &codelet_row<T, 4>::passes,  &codelet_row<T, 5>::passes,
            &codelet_row<T, 6>::passes,  &codelet_row<T, 7>::passes,  &codelet_row<T, 8>::passes,
            &codelet_row<T, 9>::passes,  &codelet_row<T, 10>::passes, &codelet_row<T, 11>::passes,
            &codelet_row<T, 12>::passes, &codelet_row<T, 13>::passes, &codelet_row<T, 14>::passes,
            &codelet_row<T, 15>::passes, &codelet_row<T, 16>::passes, &codelet_row<T, 17>::passes,
            &codelet_row<T, 18>::passes, &codelet_row<T, 19>::passes, &codelet_row<T, 20>::passes,
            &codelet_row<T, 21>::passes,
        },
    .generic =
        {
            {
                {
                    &generic_pass<T, -1, nop, nop>,
                    &generic_pass<T, -1, nop, scl>,
                    &generic_pass<T, -1, nop, fac>,
                },
                {
                    &generic_pass<T, -1, scl, nop>,
                    nullptr,
                    &generic_pass<T, -1, scl, fac>,
                },
                {
                    &generic_pass<T, -1, fac, nop>,
                    &generic_pass<T, -1, fac, scl>,
                    &generic_pass<T, -1, fac, fac>,
                },
            },
            {
                {
                    &generic_pass<T, +1, nop, nop>,
                    &generic_pass<T, +1, nop, scl>,
                    &generic_pass<T, +1, nop, fac>,
                },
                {
                    &generic_pass<T, +1, scl, nop>,
                    nullptr,
                    &generic_pass<T, +1, scl, fac>,
                },
                {
                    &generic_pass<T, +1, fac, nop>,
                    &generic_pass<T, +1, fac, scl>,
                    &generic_pass<T, +1, fac, fac>,
                },
            },
        },
    .copy =
        {
            {
                &copy_pass<T, nop, nop>,
                &copy_pass<T, nop, scl>,
                &copy_pass<T, nop, fac>,
            },
            {
                &copy_pass<T, scl, nop>,
                nullptr,
                &copy_pass<T, scl, fac>,
            },
            {
                &copy_pass<T, fac, nop>,
                &copy_pass<T, fac, scl>,
                &copy_pass<T, fac, fac>,
            },
        },
    .bluestein_in =
        {
            {
                &bluestein_in<T, nop, false>,
                &bluestein_in<T, scl, false>,
                &bluestein_in<T, fac, false>,
            },
            {
                &bluestein_in<T, nop, true>,
                &bluestein_in<T, scl, true>,
                &bluestein_in<T, fac, true>,
            },
        },
    .bluestein_out =
        {
            {
                &bluestein_out<T, nop, false>,
                &bluestein_out<T, scl, false>,
                &bluestein_out<T, fac, false>,
            },
            {
                &bluestein_out<T, nop, true>,
                &bluestein_out<T, scl, true>,
                &bluestein_out<T, fac, true>,
            },
        },
    .rader_in =
        {
            {
                &rader_in<T, nop, false>,
                &rader_in<T, scl, false>,
                &rader_in<T, fac, false>,
            },
            {
                &rader_in<T, nop, true>,
                &rader_in<T, scl, true>,
                &rader_in<T, fac, true>,
            },
        },
    .rader_out =
        {
            {
                &rader_out<T, nop, false>,
                &rader_out<T, scl, false>,
                &rader_out<T, fac, false>,
            },
            {
                &rader_out<T, nop, true>,
                &rader_out<T, scl, true>,
                &rader_out<T, fac, true>,
            },
        },
    .r2c_post =
        {
            &r2c_post<T, false>,
            &r2c_post<T, true>,
        },
    .r2c_pack = &r2c_pack<T>,
    .r2c_store =
        {
            &r2c_store<T, false>,
            &r2c_store<T, true>,
        },
    .c2r_pre =
        {
            &c2r_pre<T, false>,
            &c2r_pre<T, true>,
        },
    .c2r_expand = &c2r_expand<T>,
    .c2r_store =
        {
            &c2r_store<T, false>,
            &c2r_store<T, true>,
        },
    .c2r_nyquist = &c2r_nyquist<T>,
    .r2c_pack_pairs = &r2c_pack_pairs<T>,
    .r2c_post_strided =
        {
            &r2c_post_strided<T, false>,
            &r2c_post_strided<T, true>,
        },
    .c2r_pre_strided = &c2r_pre_strided<T>,
    .c2r_unpack_pairs =
        {
            &c2r_unpack_pairs<T, false>,
            &c2r_unpack_pairs<T, true>,
        },
    .block_copy =
        {
            {
                &block_copy<T, nop, false>,
                &block_copy<T, scl, false>,
            },
            {
                &block_copy<T, nop, true>,
                &block_copy<T, scl, true>,
            },
        },
    .twiddle_transpose =
        {
            &twiddle_transpose<T, -1>,
            &twiddle_transpose<T, +1>,
        },
    .twiddle_scatter =
        {
            {
                &twiddle_scatter<T, -1, false>,
                &twiddle_scatter<T, +1, false>,
            },
            {
                &twiddle_scatter<T, -1, true>,
                &twiddle_scatter<T, +1, true>,
            },
        },
    .c2r_pair_gather = &c2r_pair_gather<T>,
    .r2c_pair_gather = &r2c_pair_gather<T>,
    .r2c_pair_scatter =
        {
            &r2c_pair_scatter<T, false>,
            &r2c_pair_scatter<T, true>,
        },
    .r2c_pair_column =
        {
            &r2c_pair_column<T, false>,
            &r2c_pair_column<T, true>,
        },
    .lanes_gather = &lanes_gather<T>,
    .lanes_scatter = &lanes_scatter<T>,
    .lanes_r2c_pair =
        {
            &lanes_r2c_pair<T, false>,
            &lanes_r2c_pair<T, true>,
        },
    .lanes_c2r_pair = &lanes_c2r_pair<T>,
};

} // namespace
