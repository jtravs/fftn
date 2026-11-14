// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The target's pass entry points (pass_kernels in passes/pass_args.hpp), instantiated
// inside the region for every codelet radix of src/gen/codelets.hpp, both directions and
// every edge variant. The radices come from the generated table, so a radix added to the
// generator gets its passes here without any change. No include guard.

namespace {

// NOLINTNEXTLINE(modernize-avoid-c-arrays): no standard-library templates in a region
inline constexpr edge_op edge_ops[edge_op_count] = {edge_op::none, edge_op::scale, edge_op::factor};

template <int Sign> inline constexpr std::size_t direction_index = Sign < 0 ? 0 : 1;

// A plan has one scale, on one edge (core/plan_core.cpp), so no pass scales both: that
// entry stays null rather than instantiating a variant nothing selects.
template <std::size_t In, std::size_t Out>
inline constexpr bool possible_edges =
    edge_ops[In] != edge_op::scale || edge_ops[Out] != edge_op::scale;

template <class T, std::size_t R, int Sign, std::size_t In, std::size_t Out>
constexpr void fill_leaves(pass_kernels<T>& k) noexcept {
  if constexpr (possible_edges<In, Out>) {
    k.leaf[R][direction_index<Sign>][In][Out] = &leaf_pass<T, R, Sign, edge_ops[In], edge_ops[Out]>;
  }
  if constexpr (Out + 1 < edge_op_count) {
    fill_leaves<T, R, Sign, In, Out + 1>(k);
  } else if constexpr (In + 1 < edge_op_count) {
    fill_leaves<T, R, Sign, In + 1, 0>(k);
  }
}

template <class T, std::size_t R, int Sign, std::size_t E>
constexpr void fill_edges(pass_kernels<T>& k) noexcept {
  constexpr std::size_t d = direction_index<Sign>;
  k.grouped[R][d][E] = &grouped_pass<T, R, Sign, edge_ops[E]>;
  k.last[R][d][E] = &lane_pass<T, R, Sign, false, edge_ops[E]>;
  if constexpr (E + 1 < edge_op_count) {
    fill_edges<T, R, Sign, E + 1>(k);
  }
}

template <class T, std::size_t R, int Sign> constexpr void fill_radix(pass_kernels<T>& k) noexcept {
  k.lane[R][direction_index<Sign>] = &lane_pass<T, R, Sign, true, edge_op::none>;
  fill_edges<T, R, Sign, 0>(k);
  fill_leaves<T, R, Sign, 0, 0>(k);
}

template <class T, std::size_t I> constexpr void fill_radices(pass_kernels<T>& k) noexcept {
  constexpr std::size_t r = gen::radices[I];
  static_assert(r <= max_codelet_radix);
  fill_radix<T, r, -1>(k);
  fill_radix<T, r, +1>(k);
  if constexpr (I + 1 < gen::radix_count) {
    fill_radices<T, I + 1>(k);
  }
}

template <class T, int Sign, std::size_t In, std::size_t Out>
constexpr void fill_generic(pass_kernels<T>& k) noexcept {
  if constexpr (possible_edges<In, Out>) {
    k.generic[direction_index<Sign>][In][Out] = &generic_pass<T, Sign, edge_ops[In], edge_ops[Out]>;
    if constexpr (Sign < 0) {
      k.copy[In][Out] = &copy_pass<T, edge_ops[In], edge_ops[Out]>;
    }
  }
  if constexpr (Out + 1 < edge_op_count) {
    fill_generic<T, Sign, In, Out + 1>(k);
  } else if constexpr (In + 1 < edge_op_count) {
    fill_generic<T, Sign, In + 1, 0>(k);
  }
}

template <class T> constexpr pass_kernels<T> make_pass_kernels() noexcept {
  pass_kernels<T> k{};
  fill_radices<T, 0>(k);
  fill_generic<T, -1, 0, 0>(k);
  fill_generic<T, +1, 0, 0>(k);
  k.r2c_pack_pairs = &r2c_pack_pairs<T>;
  k.r2c_post[0] = &r2c_post<T, false>;
  k.r2c_post[1] = &r2c_post<T, true>;
  k.r2c_pack = &r2c_pack<T>;
  k.r2c_store[0] = &r2c_store<T, false>;
  k.r2c_store[1] = &r2c_store<T, true>;
  k.c2r_pre = &c2r_pre<T>;
  k.c2r_unpack_pairs[0] = &c2r_unpack_pairs<T, false>;
  k.c2r_unpack_pairs[1] = &c2r_unpack_pairs<T, true>;
  k.c2r_expand = &c2r_expand<T>;
  k.c2r_store[0] = &c2r_store<T, false>;
  k.c2r_store[1] = &c2r_store<T, true>;
  return k;
}

} // namespace
