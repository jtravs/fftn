// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Kernel text, written once and included inside every target's region after its
// vocabulary (DESIGN.md 6.4). No include guard. It uses only the vocabulary
// (vocabulary.inl) and plain scalar C++: no standard-library templates, no lambdas. Entry
// points take only pointers and integers; helpers that handle vectors are force-inlined.
// What is here converts twiddles into the target's storage when a plan builds its tables;
// the passes are in src/passes/.

namespace {

// Writes count interleaved twiddles w as this target's twiddle storage (tw_reals<T> reals
// each). Run once, when a plan's tables are built. Needs count % C == 0.
template <class T> void pack_twiddles(const T* w, std::size_t count, T* out) {
  for (std::size_t k = 0; k < count; k += C<T>) {
    pack_tw(out + tw_reals<T> * k, w + 2 * k);
  }
}

// Writes count interleaved twiddles w as this target's broadcast storage (tw_reals<T>
// reals each, read by bcast_tw). Run once, when a plan's tables are built.
template <class T> void pack_bcast_twiddles(const T* w, std::size_t count, T* out) {
  for (std::size_t k = 0; k < count; ++k) {
    pack_tw1(out + tw_reals<T> * k, w + 2 * k);
  }
}

// The same into the compact storage (2 reals per twiddle, load_twc and bcast_twc) where the
// target has it (compact_tw), else null: a plan then keeps the pre-arranged storage.
template <class T> void pack_compact_twiddles(const T* w, std::size_t count, T* out) {
  for (std::size_t k = 0; k < count; k += C<T>) {
    pack_twc(out + 2 * k, w + 2 * k);
  }
}
template <class T> void pack_compact_bcast_twiddles(const T* w, std::size_t count, T* out) {
  for (std::size_t k = 0; k < count; ++k) {
    pack_twc1(out + 2 * k, w + 2 * k);
  }
}
template <class T> constexpr void (*compact_packer(bool broadcast))(const T*, std::size_t, T*) {
  if constexpr (compact_tw) {
    return broadcast ? &pack_compact_bcast_twiddles<T> : &pack_compact_twiddles<T>;
  } else {
    return nullptr;
  }
}

} // namespace
