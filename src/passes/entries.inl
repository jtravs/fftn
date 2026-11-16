// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The target's pass entry points (pass_kernels in passes/pass_args.hpp), instantiated
// inside the region for every codelet radix of src/gen/codelets.hpp, both directions and
// every edge variant. Which rows exist comes from the generated table, so a radix added to
// the generator gets its passes here without any change. The table is constant data, built
// by the compiler (no function computes it at run time). No include guard.
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
constexpr pass_fn split_entry() {
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
constexpr pass_fn phase_entry() {
  if constexpr (phase_radix<R>) {
    return &phase_pass<T, R, Sign, Twiddled, InOp, Out>;
  } else {
    return nullptr;
  }
}

template <class T, std::size_t R, bool Scaled> constexpr pass_fn r2c_last_entry() {
  if constexpr (phase_radix<R>) {
    return &r2c_last<T, R, Scaled>;
  } else {
    return nullptr;
  }
}

template <class T, std::size_t R> constexpr pass_fn c2r_first_entry() {
  if constexpr (phase_radix<R>) {
    return &c2r_first<T, R>;
  } else {
    return nullptr;
  }
}

// A leaf for lines of stride 1 (unit_leaf.inl), or null for radices above 32, whose leaves
// no plan takes by default (rules.cpp, leaf_limit_of: 32 at most).
template <class T, std::size_t R, int Sign, edge_op InOp, edge_op OutOp>
constexpr leaf_fn unit_leaf_entry() {
  if constexpr (R <= 32) {
    return &unit_leaf<T, R, Sign, InOp, OutOp>;
  } else {
    return nullptr;
  }
}

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
  };
};

} // namespace

// Row I of the codelet table for precision T: radix_entries<T, I>::value, defined by exactly
// one of the target's row units (isa/rows.inl) and read through a pointer by the target's
// table. The only names of the region besides the table with external linkage: data, so no
// unit's copy of a function can stand in for another's. Each target's are distinct
// symbols, in its own namespace.
template <class T, std::size_t I> struct codelet_row {
  static const radix_passes passes;
};

namespace {

static_assert(gen::radix_count == 20, "pass_table lists the rows of gen::radices");

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
            &codelet_row<T, 18>::passes, &codelet_row<T, 19>::passes,
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
            &block_copy<T, nop>,
            &block_copy<T, scl>,
        },
    .twiddle_transpose =
        {
            &twiddle_transpose<T, -1>,
            &twiddle_transpose<T, +1>,
        },
    .twiddle_scatter =
        {
            &twiddle_scatter<T, -1>,
            &twiddle_scatter<T, +1>,
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
