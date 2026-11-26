// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Runner text for the generated codelets (codelet_table.hpp): included inside each target's
// region after its vocabulary and src/gen/complex_codelets.inl, under the rules of the
// library's kernel text (src/isa/kernels.inl). No include guard. The accessors are the
// simplest a pass can give a codelet (gen/README.md, "Calling convention").

namespace {

template <class T> struct runner_in {
  const T* p;
  std::size_t stride; // reals between legs
  [[nodiscard]] FFTN_FORCE_INLINE cv<T> get(std::size_t j) const { return load(p + stride * j); }
};

template <class T> struct runner_out {
  T* p;
  std::size_t stride;
  FFTN_FORCE_INLINE void put(std::size_t k, cv<T> v) const { store(p + stride * k, v); }
};

template <class T> struct runner_twiddles {
  const T* p;
  [[nodiscard]] FFTN_FORCE_INLINE tw<T> get(std::size_t k) const {
    return load_tw(p + (k - 1) * C<T> * tw_reals<T>);
  }
};

template <class T, std::size_t I>
void run_codelet(const T* x, std::size_t xs, T* y, std::size_t ys, const T* w) {
  constexpr gen::codelet_info info = gen::codelets[I];
  using codelet = cdft<info.radix>;
  const runner_in<T> in{x, 2 * xs};
  const runner_out<T> out{y, 2 * ys};
  const runner_twiddles<T> twid{w};
  if constexpr (info.kind == gen::variant::plain) {
    codelet::template plain<T, info.sign>(in, out);
  } else if constexpr (info.kind == gen::variant::dit) {
    codelet::template dit<T, info.sign>(in, out, twid);
  } else {
    codelet::template dif<T, info.sign>(in, out, twid);
  }
}

template <class T, std::size_t I>
void run_pruned(const T* x, std::size_t xs, T* y, std::size_t ys, const T* w) {
  constexpr gen::codelet_info info = gen::pruned_codelets[I];
  using codelet = cdft<info.radix>;
  const runner_in<T> in{x, 2 * xs};
  const runner_out<T> out{y, 2 * ys};
  const runner_twiddles<T> twid{w};
  if constexpr (info.kind == gen::variant::dif_head) {
    codelet::template dif_head<T, info.sign>(in, out, twid);
  } else {
    static_assert(info.kind == gen::variant::plain_head);
    codelet::template plain_head<T, info.sign>(in, out);
  }
}

template <class T, std::size_t I> constexpr void fill_runs(codelet_runs<T>& runs) noexcept {
  runs.run[I] = &run_codelet<T, I>;
  if constexpr (I + 1 < gen::codelet_count) {
    fill_runs<T, I + 1>(runs);
  }
}

template <class T, std::size_t I> constexpr void fill_pruned(codelet_runs<T>& runs) noexcept {
  runs.pruned[I] = &run_pruned<T, I>;
  if constexpr (I + 1 < gen::pruned_codelet_count) {
    fill_pruned<T, I + 1>(runs);
  }
}

// The entry points of every codelet in precision T; the unit's table holds both precisions.
template <class T> constexpr codelet_runs<T> make_runs() noexcept {
  codelet_runs<T> runs{};
  fill_runs<T, 0>(runs);
  fill_pruned<T, 0>(runs);
  return runs;
}

} // namespace
