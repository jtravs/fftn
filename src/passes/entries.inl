// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The target's pass entry points (pass_kernels in passes/pass_args.hpp), instantiated
// inside the region for every codelet radix of src/gen/codelets.hpp, both directions and
// every edge variant. Which rows exist comes from the generated table, so a radix added to
// the generator gets its passes here without any change. The table is constant data, built
// by the compiler (no function computes it at run time). No include guard.

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

// The lanes passes that reach rows one load or store per row (lanes.inl, Spread), or null
// where the vocabulary has no spread<T>.
template <class T, std::size_t R, int Sign> constexpr pass_fn lanes_first_spread() {
  if constexpr (spread<T>) {
    return &lanes_first<T, R, Sign, true>;
  } else {
    return nullptr;
  }
}

template <class T, std::size_t R, int Sign, edge_op OutOp> constexpr pass_fn lanes_last_spread() {
  if constexpr (spread<T>) {
    return &lanes_last<T, R, Sign, OutOp, true>;
  } else {
    return nullptr;
  }
}

template <class T, std::size_t R, int Sign, edge_op OutOp> constexpr pass_fn lanes_leaf_spread() {
  if constexpr (spread<T>) {
    return &lanes_leaf<T, R, Sign, OutOp, true>;
  } else {
    return nullptr;
  }
}

template <class T, bool Scaled> constexpr pass_fn lanes_r2c_pair_spread() {
  if constexpr (spread<T>) {
    return &lanes_r2c_pair<T, Scaled, true>;
  } else {
    return nullptr;
  }
}

template <class T> constexpr pass_fn lanes_c2r_pair_spread() {
  if constexpr (spread<T>) {
    return &lanes_c2r_pair<T, true>;
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
                  &phase_pass<T, R, -1, true, nop, 0>,
                  &phase_pass<T, R, -1, true, scl, 0>,
                  &phase_pass<T, R, -1, false, nop, 0>,
                  &phase_pass<T, R, -1, false, scl, 0>,
                  &phase_pass<T, R, -1, false, nop, 1>,
                  &phase_pass<T, R, -1, false, nop, 2>,
              },
              {
                  &phase_pass<T, R, +1, true, nop, 0>,
                  &phase_pass<T, R, +1, true, scl, 0>,
                  &phase_pass<T, R, +1, false, nop, 0>,
                  &phase_pass<T, R, +1, false, scl, 0>,
                  &phase_pass<T, R, +1, false, nop, 1>,
                  &phase_pass<T, R, +1, false, nop, 2>,
              },
          },
      .r2c_last =
          {
              &r2c_last<T, R, false>,
              &r2c_last<T, R, true>,
          },
      .c2r_first = &c2r_first<T, R>,
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
              {&lanes_first<T, R, -1, false>, lanes_first_spread<T, R, -1>()},
              {&lanes_first<T, R, +1, false>, lanes_first_spread<T, R, +1>()},
          },
      .lanes_last =
          {
              {
                  {&lanes_last<T, R, -1, nop, false>, lanes_last_spread<T, R, -1, nop>()},
                  {&lanes_last<T, R, -1, scl, false>, lanes_last_spread<T, R, -1, scl>()},
              },
              {
                  {&lanes_last<T, R, +1, nop, false>, lanes_last_spread<T, R, +1, nop>()},
                  {&lanes_last<T, R, +1, scl, false>, lanes_last_spread<T, R, +1, scl>()},
              },
          },
      .lanes_leaf =
          {
              {
                  {&lanes_leaf<T, R, -1, nop, false>, lanes_leaf_spread<T, R, -1, nop>()},
                  {&lanes_leaf<T, R, -1, scl, false>, lanes_leaf_spread<T, R, -1, scl>()},
              },
              {
                  {&lanes_leaf<T, R, +1, nop, false>, lanes_leaf_spread<T, R, +1, nop>()},
                  {&lanes_leaf<T, R, +1, scl, false>, lanes_leaf_spread<T, R, +1, scl>()},
              },
          },
  };
};

static_assert(gen::radix_count == 17, "pass_table lists the rows of gen::radices");

template <class T>
inline constexpr pass_kernels<T> pass_table{
    .codelet =
        {
            radix_entries<T, 0>::value,
            radix_entries<T, 1>::value,
            radix_entries<T, 2>::value,
            radix_entries<T, 3>::value,
            radix_entries<T, 4>::value,
            radix_entries<T, 5>::value,
            radix_entries<T, 6>::value,
            radix_entries<T, 7>::value,
            radix_entries<T, 8>::value,
            radix_entries<T, 9>::value,
            radix_entries<T, 10>::value,
            radix_entries<T, 11>::value,
            radix_entries<T, 12>::value,
            radix_entries<T, 13>::value,
            radix_entries<T, 14>::value,
            radix_entries<T, 15>::value,
            radix_entries<T, 16>::value,
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
            {&lanes_r2c_pair<T, false, false>, lanes_r2c_pair_spread<T, false>()},
            {&lanes_r2c_pair<T, true, false>, lanes_r2c_pair_spread<T, true>()},
        },
    .lanes_c2r_pair = {&lanes_c2r_pair<T, false>, lanes_c2r_pair_spread<T>()},
};

} // namespace
