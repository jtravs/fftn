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
                      &leaf_pass<T, R, -1, nop, nop>,
                      &leaf_pass<T, R, -1, nop, scl>,
                      &leaf_pass<T, R, -1, nop, fac>,
                  },
                  {
                      &leaf_pass<T, R, -1, scl, nop>,
                      nullptr,
                      &leaf_pass<T, R, -1, scl, fac>,
                  },
                  {
                      &leaf_pass<T, R, -1, fac, nop>,
                      &leaf_pass<T, R, -1, fac, scl>,
                      &leaf_pass<T, R, -1, fac, fac>,
                  },
              },
              {
                  {
                      &leaf_pass<T, R, +1, nop, nop>,
                      &leaf_pass<T, R, +1, nop, scl>,
                      &leaf_pass<T, R, +1, nop, fac>,
                  },
                  {
                      &leaf_pass<T, R, +1, scl, nop>,
                      nullptr,
                      &leaf_pass<T, R, +1, scl, fac>,
                  },
                  {
                      &leaf_pass<T, R, +1, fac, nop>,
                      &leaf_pass<T, R, +1, fac, scl>,
                      &leaf_pass<T, R, +1, fac, fac>,
                  },
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
            &bluestein_in<T, nop>,
            &bluestein_in<T, scl>,
            &bluestein_in<T, fac>,
        },
    .bluestein_out =
        {
            &bluestein_out<T, nop>,
            &bluestein_out<T, scl>,
            &bluestein_out<T, fac>,
        },
    .r2c_pack_pairs = &r2c_pack_pairs<T>,
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
    .c2r_pre = &c2r_pre<T>,
    .c2r_unpack_pairs =
        {
            &c2r_unpack_pairs<T, false>,
            &c2r_unpack_pairs<T, true>,
        },
    .c2r_expand = &c2r_expand<T>,
    .c2r_store =
        {
            &c2r_store<T, false>,
            &c2r_store<T, true>,
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
};

} // namespace
