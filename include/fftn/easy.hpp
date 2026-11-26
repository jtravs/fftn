// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// The convenience layer: numpy.fft's functions and scipy.fft's real-to-real ones (`dct`,
/// `dst` and their inverses and N-D forms) in one call each, plus the DHT, over spans and
/// strided views, with the plans and scratch they need kept in a per-thread cache (`context`).
///
/// Provisional (steps 6.6 and 6.7, dev/DESIGN.md 5.4): the names, the semantics and the cache
/// follow the design's proposals, which await the maintainer's review. docs/easy.md says how
/// the pieces fit together.
#pragma once

#include "fftn/config.hpp"
#include "fftn/executor.hpp"
#include "fftn/shape.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <version>

#ifdef __cpp_lib_mdspan
#include <mdspan>
#endif

namespace fftn::inline v0::easy {

/// Where a transform pair puts its factor of 1/N (N the number of points transformed), as
/// numpy.fft's and scipy.fft's `norm` argument.
///
/// The forward transforms are `fft`, `rfft`, `dct`, `dst`, `dht` and their 2-D and N-D forms;
/// the inverse ones are `ifft`, `irfft`, `idct`, `idst` and theirs. Whatever the norm, a
/// forward transform followed by its inverse returns the input (within rounding). For the
/// real-to-real kinds N is scipy's "logical" size, the product over the transformed axes of
/// 2(n - 1) for type 1 of the DCT, 2(n + 1) for type 1 of the DST, 2n for types 2 to 4 and n
/// for the DHT; and `ortho` also applies scipy's orthogonalising weights (`context::dct`).
enum class norm : unsigned char {
  /// The forward transform unscaled, the inverse scaled by 1/N: numpy's default, and the
  /// core's convention (fftn::plan) with the inverse normalised.
  backward,
  /// Both scaled by 1/sqrt(N): the transforms are unitary.
  ortho,
  /// The forward transform scaled by 1/N, the inverse unscaled.
  forward,
};

/// A borrowed strided array of `Rank` dimensions: what the multi-dimensional functions read
/// and write.
///
/// `extents` are the dimensions' lengths, slowest first as in a C array; `strides` are the
/// distances between consecutive elements along each dimension, in elements, and may be
/// negative. All strides 0 (the default) means tight C order (the last dimension contiguous).
/// Otherwise every stride is taken as given, and a stride of 0 on a dimension longer than 1
/// is refused by the functions that use the view. `data` addresses the element whose indices
/// are all 0, which need not be the lowest address when strides are negative.
///
/// A view owns nothing; the array must outlive its use.
template <class T, std::size_t Rank> struct view {
  static_assert(Rank >= 1, "fftn::easy::view: an array has at least one dimension");

  /// The element whose indices are all 0.
  T* data = nullptr;
  /// The length of each dimension, slowest first.
  std::array<std::size_t, Rank> extents{};
  /// The distance between consecutive elements along each dimension, in elements; all 0 for
  /// tight C order.
  std::array<std::ptrdiff_t, Rank> strides{};

  /// An empty view: no data and every extent 0.
  constexpr view() noexcept = default;

  /// A view of the array at `d` with extents `e` and strides `s` (all 0 by default: tight C
  /// order).
  constexpr view(T* d, const std::array<std::size_t, Rank>& e,
                 const std::array<std::ptrdiff_t, Rank>& s = {}) noexcept
      : data(d), extents(e), strides(s) {}

  /// The same array viewed with a more qualified element type: a view of `T` converts to a
  /// view of `const T`.
  template <class U>
    requires(!std::is_const_v<U> && std::is_same_v<T, const U>)
  // NOLINTNEXTLINE(misc-explicit-constructor): implicit, as T* converts to const T*
  constexpr view(const view<U, Rank>& other) noexcept
      : data(other.data), extents(other.extents), strides(other.strides) {}

#ifdef __cpp_lib_mdspan
  /// The array of a `std::mdspan` with the default accessor and a strided layout
  /// (`std::layout_right`, `std::layout_left` or `std::layout_stride`), where the standard
  /// library has `std::mdspan` (C++23). Its strides are copied as given. The functions
  /// below deduce their element types from views, so an mdspan is passed through this
  /// conversion explicitly: `fftn::easy::fftn(fftn::easy::view(in), fftn::easy::view(out))`.
  template <class U, class E, class L>
    requires(E::rank() == Rank &&
             (std::is_same_v<U, T> || (!std::is_const_v<U> && std::is_same_v<T, const U>)) &&
             L::template mapping<E>::is_always_strided())
  // NOLINTNEXTLINE(misc-explicit-constructor): implicit, so an mdspan passes as a view
  constexpr view(const std::mdspan<U, E, L, std::default_accessor<U>>& m) noexcept
      : data(m.data_handle()) {
    for (std::size_t r = 0; r < Rank; ++r) {
      extents[r] = static_cast<std::size_t>(m.extent(r));
      strides[r] = static_cast<std::ptrdiff_t>(m.stride(r));
    }
  }
#endif
};

#ifdef __cpp_lib_mdspan
/// Deduces a view's element type and rank from a `std::mdspan`.
template <class U, class E, class L>
view(const std::mdspan<U, E, L, std::default_accessor<U>>&) -> view<U, E::rank()>;
#endif

/// The dimensions a multi-dimensional function transforms or shifts, as numpy's `axes`:
/// indices into the array's dimensions, a negative one counting from the end (-1 is the
/// last). Empty means the function's default.
///
/// Built implicitly from a braced list (`{0, -1}`) or from any contiguous range of `int`
/// (`std::array`, `std::vector`, `std::span`), and copied, so it holds no reference to them.
///
/// Guarantees: holds up to `capacity` indices; a longer list is remembered as too long and
/// refused, naming `axes`, by the function it is passed to.
class axis_list {
public:
  /// The most indices a list holds.
  static constexpr std::size_t capacity = 8;

  /// The empty list: the function's default axes.
  constexpr axis_list() noexcept = default;

  /// The indices of a braced list, such as `{0, -1}`.
  // NOLINTNEXTLINE(misc-explicit-constructor): implicit, so a braced list passes as axes
  constexpr axis_list(std::initializer_list<int> list) noexcept {
    assign(list.begin(), list.size());
  }

  /// The indices of a contiguous range of `int`.
  template <class R>
    requires(std::is_convertible_v<R &&, std::span<const int>> &&
             !std::is_same_v<std::remove_cvref_t<R>, axis_list>)
  // NOLINTNEXTLINE(misc-explicit-constructor): implicit, so a range passes as axes
  constexpr axis_list(R&& range) noexcept {
    const std::span<const int> s = std::forward<R>(range);
    assign(s.data(), s.size());
  }

  /// How many indices were given (more than `capacity` when the list was too long).
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }

  /// The indices held: all of them, or the first `capacity` of a list that was too long.
  [[nodiscard]] constexpr std::span<const int> indices() const noexcept {
    return {axes_.data(), std::min(size_, capacity)};
  }

private:
  constexpr void assign(const int* p, std::size_t n) noexcept {
    size_ = n;
    std::copy_n(p, std::min(n, capacity), axes_.begin());
  }

  std::array<int, capacity> axes_{};
  std::size_t size_ = 0;
};

namespace detail {

enum class easy_kind : unsigned char { c2c, r2c, c2r, r2r };

// The family of a real-to-real call: scipy's dct and idct, dst and idst, or the DHT.
enum class easy_r2r : unsigned char { none, cosine, sine, hartley };

// One call of the layer, as the compiled part (src/easy/easy.cpp) receives it: the arrays'
// descriptions borrowed from the caller's arguments for the duration of the call.
struct easy_call {
  const char* function = ""; // for messages: "fft2"
  easy_kind kind = easy_kind::c2c;
  bool single = false; // float rather than double
  direction dir = direction::forward;
  norm scaling = norm::backward;
  std::span<const std::size_t> in_extents;
  std::span<const std::ptrdiff_t> in_strides;
  std::span<const std::size_t> out_extents;
  std::span<const std::ptrdiff_t> out_strides;
  axis_list axes;
  std::size_t default_axes = 0; // when axes is empty: 0 for all, else the last k
  bool spans = false;           // the one-dimensional forms: messages say size()
  const void* in = nullptr;
  void* out = nullptr;
  easy_r2r family = easy_r2r::none; // r2r: which transform
  int type = 0;                     // r2r cosine and sine: scipy's type, 1 to 4
};

template <class T> struct is_precision : std::false_type {};
template <> struct is_precision<double> : std::true_type {};
template <> struct is_precision<float> : std::true_type {};

template <class T> struct complex_of_precision : std::false_type {};
template <class T> struct complex_of_precision<std::complex<T>> : is_precision<T> {};

// Input and output element types of each kind's views: the input may be const.
template <class In, class Out>
concept c2c_elements =
    complex_of_precision<Out>::value && std::is_same_v<std::remove_const_t<In>, Out>;
template <class In, class Out>
concept r2c_elements = is_precision<std::remove_const_t<In>>::value &&
                       std::is_same_v<Out, std::complex<std::remove_const_t<In>>>;
template <class In, class Out>
concept c2r_elements =
    is_precision<Out>::value && std::is_same_v<std::remove_const_t<In>, std::complex<Out>>;
template <class In, class Out>
concept r2r_elements = is_precision<Out>::value && std::is_same_v<std::remove_const_t<In>, Out>;

template <class T> constexpr bool is_single() {
  if constexpr (complex_of_precision<std::remove_const_t<T>>::value) {
    return std::is_same_v<typename std::remove_const_t<T>::value_type, float>;
  } else {
    return std::is_same_v<std::remove_const_t<T>, float>;
  }
}

template <class In, class Out, std::size_t R>
easy_call call_of(const char* function, easy_kind kind, direction dir, norm scaling,
                  const view<In, R>& in, const view<Out, R>& out, const axis_list& axes,
                  std::size_t default_axes, easy_r2r family = easy_r2r::none, int type = 0) {
  return {
      .function = function,
      .kind = kind,
      .single = is_single<Out>(),
      .dir = dir,
      .scaling = scaling,
      .in_extents = in.extents,
      .in_strides = in.strides,
      .out_extents = out.extents,
      .out_strides = out.strides,
      .axes = axes,
      .default_axes = default_axes,
      .spans = false,
      .in = in.data,
      .out = out.data,
      .family = family,
      .type = type,
  };
}

// The strides a view's elements are at: as given, or tight C order when all are 0.
template <class T, std::size_t R> std::array<std::ptrdiff_t, R> strides_of(const view<T, R>& v) {
  if (std::ranges::any_of(v.strides, [](std::ptrdiff_t s) { return s != 0; })) {
    return v.strides;
  }
  std::array<std::ptrdiff_t, R> s{};
  std::ptrdiff_t span = 1;
  for (std::size_t i = R; i > 0; --i) {
    s[i - 1] = span;
    span *= static_cast<std::ptrdiff_t>(v.extents[i - 1]);
  }
  return s;
}

// Strides given with a 0 on a dimension that steps would make every element of it one.
template <class T, std::size_t R>
void check_strides(const char* function, const char* side, const view<T, R>& v) {
  if (std::ranges::none_of(v.strides, [](std::ptrdiff_t s) { return s != 0; })) {
    return;
  }
  for (std::size_t d = 0; d < R; ++d) {
    if (v.strides[d] == 0 && v.extents[d] > 1) {
      throw std::invalid_argument(std::string("fftn::easy::") + function + ": " + side +
                                  ".strides[" + std::to_string(d) +
                                  "] is 0 among nonzero strides; give every stride, or all 0 "
                                  "for tight C order");
    }
  }
}

// The dimensions listed by `axes` (all when it is empty), validated: true in `chosen` for
// each.
template <std::size_t R>
std::array<bool, R> chosen_axes(const char* function, const axis_list& axes) {
  std::array<bool, R> chosen{};
  if (axes.size() == 0) {
    chosen.fill(true);
    return chosen;
  }
  const auto fail = [function](const std::string& what) {
    throw std::invalid_argument(std::string("fftn::easy::") + function + ": axes " + what);
  };
  if (axes.size() > axis_list::capacity) {
    fail("lists " + std::to_string(axes.size()) + " dimensions; an axis_list holds at most " +
         std::to_string(axis_list::capacity));
  }
  for (const int a : axes.indices()) {
    const auto r = static_cast<long long>(R);
    if (a < -r || a >= r) {
      fail("lists " + std::to_string(a) + ", outside an array of " + std::to_string(R) +
           " dimensions");
    }
    const auto d = static_cast<std::size_t>(a < 0 ? a + r : a);
    if (chosen[d]) {
      fail("lists dimension " + std::to_string(d) + " twice");
    }
    chosen[d] = true;
  }
  return chosen;
}

template <class In, class Out, std::size_t R>
void shift(const char* function, const view<In, R>& in, const view<Out, R>& out,
           const axis_list& axes, bool inverse) {
  if (in.extents != out.extents) {
    throw std::invalid_argument(std::string("fftn::easy::") + function +
                                ": out.extents differ from in.extents");
  }
  const std::array<bool, R> chosen = chosen_axes<R>(function, axes);
  std::size_t total = 1;
  for (const std::size_t e : in.extents) {
    total *= e;
  }
  if (total == 0) {
    return;
  }
  check_strides(function, "in", in);
  check_strides(function, "out", out);
  const std::array<std::ptrdiff_t, R> is = strides_of(in);
  const std::array<std::ptrdiff_t, R> os = strides_of(out);
  // Element i goes to (i + k) mod n along each shifted dimension: k = n/2 for fftshift and
  // n - n/2 for ifftshift (numpy's roll by n//2 and by -(n//2)).
  std::array<std::size_t, R> k{};
  for (std::size_t d = 0; d < R; ++d) {
    const std::size_t n = in.extents[d];
    if (chosen[d]) {
      k[d] = inverse ? n - n / 2 : n / 2;
    }
  }
  const bool same =
      static_cast<const void*>(in.data) == static_cast<const void*>(out.data) && is == os;
  if (!same) {
    std::array<std::size_t, R> i{};
    for (std::size_t done = 0; done < total; ++done) {
      std::ptrdiff_t src = 0;
      std::ptrdiff_t dst = 0;
      for (std::size_t d = 0; d < R; ++d) {
        std::size_t j = i[d] + k[d];
        j = j >= in.extents[d] ? j - in.extents[d] : j;
        src += static_cast<std::ptrdiff_t>(i[d]) * is[d];
        dst += static_cast<std::ptrdiff_t>(j) * os[d];
      }
      out.data[dst] = in.data[src];
      for (std::size_t d = R; d > 0; --d) {
        if (++i[d - 1] < in.extents[d - 1]) {
          break;
        }
        i[d - 1] = 0;
      }
    }
    return;
  }
  // In place: each shifted dimension in turn, every line along it rotated by three reversals.
  for (std::size_t a = 0; a < R; ++a) {
    const std::size_t n = out.extents[a];
    if (k[a] == 0 || k[a] == n) {
      continue;
    }
    const std::ptrdiff_t step = os[a];
    const auto reverse = [&](Out* base, std::size_t first, std::size_t last) {
      while (first + 1 < last) {
        --last;
        std::swap(base[static_cast<std::ptrdiff_t>(first) * step],
                  base[static_cast<std::ptrdiff_t>(last) * step]);
        ++first;
      }
    };
    std::array<std::size_t, R> i{};
    for (std::size_t line = 0; line < total / n; ++line) {
      std::ptrdiff_t at = 0;
      for (std::size_t d = 0; d < R; ++d) {
        at += static_cast<std::ptrdiff_t>(i[d]) * os[d];
      }
      Out* base = out.data + at;
      reverse(base, 0, n);
      reverse(base, 0, k[a]);
      reverse(base, k[a], n);
      for (std::size_t d = R; d > 0; --d) {
        if (d - 1 == a) {
          continue;
        }
        if (++i[d - 1] < out.extents[d - 1]) {
          break;
        }
        i[d - 1] = 0;
      }
    }
  }
}

template <class T>
void frequencies(const char* function, std::size_t n, std::span<T> out, double d, bool half) {
  const std::size_t want = half ? n / 2 + 1 : n;
  if (n == 0) {
    throw std::invalid_argument(std::string("fftn::easy::") + function + ": n is 0");
  }
  if (out.size() != want) {
    throw std::invalid_argument(std::string("fftn::easy::") + function + ": out has " +
                                std::to_string(out.size()) + " elements; it needs " +
                                std::to_string(want));
  }
  // numpy's arithmetic: the integer frequency times 1 / (n d), in double.
  const double unit = 1.0 / (static_cast<double>(n) * d);
  for (std::size_t i = 0; i < want; ++i) {
    const bool negative = !half && i >= (n + 1) / 2;
    const double k = negative ? -static_cast<double>(n - i) : static_cast<double>(i);
    out[i] = static_cast<T>(k * unit);
  }
}

} // namespace detail

/// Cached plans and scratch for the convenience functions, and the functions themselves as
/// members.
///
/// Each call finds the core plan (fftn::plan) for its kind, precision, array layout
/// (extents and strides on both sides, in place or not) and norm among the context's cached
/// plans, or builds one with `planning::rules` and caches it, then executes it with the
/// context's one workspace, grown on a miss to what the plan needs. At most `max_plans()`
/// plans are kept (64 unless set); when a new plan would exceed that, the least recently
/// used one is dropped. The memory a context holds is therefore its plans (each a plan's
/// tables and, for a real-to-real call with `norm::ortho` and weights, the arrays of its
/// weights) and one workspace of the largest size any of its calls has needed since
/// construction or `clear()`.
///
/// The free functions below use the calling thread's own context (`thread_context()`).
///
/// Preconditions: one thread at a time uses a context (the free functions satisfy this by
/// using each thread's own). A call must not re-enter the context it runs on, as a task on
/// the context's executor calling the free functions on the calling thread would.
///
/// Guarantees: a call that finds its plan allocates nothing (unless an executor's
/// concurrency has grown, which grows the workspace), and computes bit for bit what the core
/// plan of that shape and `options::scale` computes; a real-to-real call with orthogonalising
/// weights is the core plan with those weights as real factors on its edges (fftn/edges.hpp),
/// and their arrays are cached with the plan. Errors are `std::invalid_argument` naming the
/// argument, thrown before anything is written; `std::bad_alloc` when memory runs out, after
/// which the context is unchanged. Provisional (steps 6.6 and 6.7).
class context {
public:
  /// The default number of plans a context keeps.
  static constexpr std::size_t default_max_plans = 64;

  /// A context with no plans that executes serially on the calling thread. Allocates
  /// nothing until its first call.
  context() noexcept;

  /// A context that executes its plans on executor `ex` (plan::execute's executor
  /// overload), sizing its workspace for `ex.concurrency()` lanes.
  ///
  /// Preconditions: the executor outlives the context.
  explicit context(executor_ref ex) noexcept;

  /// Takes over `other`'s plans and workspace; `other` is left as if newly constructed with
  /// the same executor.
  context(context&& other) noexcept;

  /// Releases this context's plans and takes over `other`'s; `other` is left as if newly
  /// constructed with the same executor.
  context& operator=(context&& other) noexcept;

  /// Not copyable: plans are cheap to share, but a workspace is not.
  context(const context&) = delete;

  /// Not copyable.
  context& operator=(const context&) = delete;

  /// Releases the plans and the workspace.
  ~context();

  /// Keeps at most `n` plans from now on, dropping the least recently used ones beyond it.
  ///
  /// Throws `std::invalid_argument` naming `max_plans` when `n` is 0.
  void set_max_plans(std::size_t n);

  /// The most plans the context keeps.
  [[nodiscard]] std::size_t max_plans() const noexcept;

  /// The number of plans the context holds now.
  [[nodiscard]] std::size_t cached_plans() const noexcept;

  /// Drops every plan and the workspace; the next call of each shape builds its plan again.
  void clear() noexcept;

  /// The forward transform of `in` into `out` (numpy.fft.fft): out[k] = sum_j in[j]
  /// exp(-2 pi i jk / n), n = in.size(), scaled as `nm` says. `in` and `out` may be the same
  /// memory (in place).
  ///
  /// Throws `std::invalid_argument` when `in` is empty (naming `in`) or `out.size()` differs
  /// from `in.size()` (naming `out`). Preconditions: otherwise `in` and `out` do not overlap.
  void fft(std::span<const std::complex<double>> in, std::span<std::complex<double>> out,
           norm nm = norm::backward) {
    run_1d("fft", detail::easy_kind::c2c, false, direction::forward, nm, in.data(), in.size(),
           out.data(), out.size());
  }

  /// `fft` in float.
  void fft(std::span<const std::complex<float>> in, std::span<std::complex<float>> out,
           norm nm = norm::backward) {
    run_1d("fft", detail::easy_kind::c2c, true, direction::forward, nm, in.data(), in.size(),
           out.data(), out.size());
  }

  /// The inverse transform of `in` into `out` (numpy.fft.ifft): exponent +2 pi i jk / n,
  /// scaled as `nm` says (by 1/n by default). Arguments, exceptions and preconditions as for
  /// `fft`.
  void ifft(std::span<const std::complex<double>> in, std::span<std::complex<double>> out,
            norm nm = norm::backward) {
    run_1d("ifft", detail::easy_kind::c2c, false, direction::backward, nm, in.data(), in.size(),
           out.data(), out.size());
  }

  /// `ifft` in float.
  void ifft(std::span<const std::complex<float>> in, std::span<std::complex<float>> out,
            norm nm = norm::backward) {
    run_1d("ifft", detail::easy_kind::c2c, true, direction::backward, nm, in.data(), in.size(),
           out.data(), out.size());
  }

  /// The forward transform of real `in` (numpy.fft.rfft): the n/2 + 1 values `fft` gives at
  /// k = 0, ..., n/2, n = in.size(), scaled as `nm` says.
  ///
  /// Throws `std::invalid_argument` when `in` is empty (naming `in`) or `out.size()` is not
  /// n/2 + 1 (naming `out`). Preconditions: otherwise `in` and `out` do not overlap.
  void rfft(std::span<const double> in, std::span<std::complex<double>> out,
            norm nm = norm::backward) {
    run_1d("rfft", detail::easy_kind::r2c, false, direction::forward, nm, in.data(), in.size(),
           out.data(), out.size());
  }

  /// `rfft` in float.
  void rfft(std::span<const float> in, std::span<std::complex<float>> out,
            norm nm = norm::backward) {
    run_1d("rfft", detail::easy_kind::r2c, true, direction::forward, nm, in.data(), in.size(),
           out.data(), out.size());
  }

  /// The inverse of `rfft` (numpy.fft.irfft with n = out.size()): the real signal of n
  /// points whose spectrum's first n/2 + 1 values are `in`, scaled as `nm` says (by 1/n by
  /// default). The imaginary parts of in[0] and, for even n, of in[n/2] are ignored, as
  /// numpy ignores them.
  ///
  /// Throws `std::invalid_argument` when `out` is empty (naming `out`) or `in.size()` is not
  /// out.size()/2 + 1 (naming `in`). Preconditions: otherwise `in` and `out` do not overlap.
  void irfft(std::span<const std::complex<double>> in, std::span<double> out,
             norm nm = norm::backward) {
    run_1d("irfft", detail::easy_kind::c2r, false, direction::backward, nm, in.data(), in.size(),
           out.data(), out.size());
  }

  /// `irfft` in float.
  void irfft(std::span<const std::complex<float>> in, std::span<float> out,
             norm nm = norm::backward) {
    run_1d("irfft", detail::easy_kind::c2r, true, direction::backward, nm, in.data(), in.size(),
           out.data(), out.size());
  }

  /// `fft` into a new vector of in.size() values, which this form allocates on every call.
  [[nodiscard]] std::vector<std::complex<double>> fft(std::span<const std::complex<double>> in,
                                                      norm nm = norm::backward) {
    std::vector<std::complex<double>> out(in.size());
    fft(in, out, nm);
    return out;
  }

  /// `fft` into a new vector, in float.
  [[nodiscard]] std::vector<std::complex<float>> fft(std::span<const std::complex<float>> in,
                                                     norm nm = norm::backward) {
    std::vector<std::complex<float>> out(in.size());
    fft(in, out, nm);
    return out;
  }

  /// `ifft` into a new vector of in.size() values, which this form allocates on every call.
  [[nodiscard]] std::vector<std::complex<double>> ifft(std::span<const std::complex<double>> in,
                                                       norm nm = norm::backward) {
    std::vector<std::complex<double>> out(in.size());
    ifft(in, out, nm);
    return out;
  }

  /// `ifft` into a new vector, in float.
  [[nodiscard]] std::vector<std::complex<float>> ifft(std::span<const std::complex<float>> in,
                                                      norm nm = norm::backward) {
    std::vector<std::complex<float>> out(in.size());
    ifft(in, out, nm);
    return out;
  }

  /// `rfft` into a new vector of in.size()/2 + 1 values, which this form allocates on every
  /// call.
  [[nodiscard]] std::vector<std::complex<double>> rfft(std::span<const double> in,
                                                       norm nm = norm::backward) {
    std::vector<std::complex<double>> out(in.size() / 2 + 1);
    rfft(in, out, nm);
    return out;
  }

  /// `rfft` into a new vector, in float.
  [[nodiscard]] std::vector<std::complex<float>> rfft(std::span<const float> in,
                                                      norm nm = norm::backward) {
    std::vector<std::complex<float>> out(in.size() / 2 + 1);
    rfft(in, out, nm);
    return out;
  }

  /// `irfft` into a new vector of `n` points, which this form allocates on every call; `n`
  /// 0 (the default) means 2(in.size() - 1), numpy's default. Throws as `irfft` does, and
  /// when `in` is empty (naming `in`).
  [[nodiscard]] std::vector<double> irfft(std::span<const std::complex<double>> in,
                                          std::size_t n = 0, norm nm = norm::backward) {
    std::vector<double> out(irfft_length(in.size(), n));
    irfft(in, out, nm);
    return out;
  }

  /// `irfft` into a new vector, in float.
  [[nodiscard]] std::vector<float> irfft(std::span<const std::complex<float>> in, std::size_t n = 0,
                                         norm nm = norm::backward) {
    std::vector<float> out(irfft_length(in.size(), n));
    irfft(in, out, nm);
    return out;
  }

  /// The forward transform along dimensions `axes` of array `in` into `out`
  /// (numpy.fft.fftn), the other dimensions looped over: all dimensions by default (at most
  /// three). `out` has `in`'s extents; it may be the same memory with the same strides (in
  /// place). Elements are `std::complex<double>` or `std::complex<float>`, the input's
  /// possibly const.
  ///
  /// Throws `std::invalid_argument` naming the argument: `axes` with an index outside the
  /// array, an index twice, none or more than three dimensions to transform; `out.extents`
  /// unlike `in.extents`; a transformed dimension of extent 0; a stride of 0 among nonzero
  /// strides on a dimension longer than 1; `out` over `in`'s memory with other strides. A
  /// dimension of extent 0 that is not transformed makes the call do nothing. Preconditions:
  /// otherwise `in` and `out` do not overlap, and `out`'s elements do not overlap each other.
  template <class In, class Out, std::size_t R>
    requires detail::c2c_elements<In, Out>
  void fftn(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
    run(detail::call_of("fftn", detail::easy_kind::c2c, direction::forward, nm, in, out, axes, 0));
  }

  /// The inverse transform along dimensions `axes` (numpy.fft.ifftn), scaled as `nm` says (by
  /// 1/N by default, N the product of the transformed extents). As `fftn` otherwise.
  template <class In, class Out, std::size_t R>
    requires detail::c2c_elements<In, Out>
  void ifftn(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
    run(detail::call_of("ifftn", detail::easy_kind::c2c, direction::backward, nm, in, out, axes,
                        0));
  }

  /// `fftn` along the last two dimensions by default (numpy.fft.fft2).
  template <class In, class Out, std::size_t R>
    requires detail::c2c_elements<In, Out>
  void fft2(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
    run(detail::call_of("fft2", detail::easy_kind::c2c, direction::forward, nm, in, out, axes, 2));
  }

  /// `ifftn` along the last two dimensions by default (numpy.fft.ifft2).
  template <class In, class Out, std::size_t R>
    requires detail::c2c_elements<In, Out>
  void ifft2(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
    run(detail::call_of("ifft2", detail::easy_kind::c2c, direction::backward, nm, in, out, axes,
                        2));
  }

  /// The forward transform of real array `in` along dimensions `axes` (numpy.fft.rfftn): the
  /// last dimension listed is halved, `out` having n/2 + 1 elements along it and `in`'s
  /// extents elsewhere. All dimensions by default (at most three). Elements are `double` or
  /// `float` in, `std::complex` of the same out; the input's possibly const. No in-place form.
  ///
  /// Throws `std::invalid_argument` as `fftn` does, naming `out.extents` when they are not
  /// those, and naming `out` when it is the same memory as `in`.
  template <class In, class Out, std::size_t R>
    requires detail::r2c_elements<In, Out>
  void rfftn(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
    run(detail::call_of("rfftn", detail::easy_kind::r2c, direction::forward, nm, in, out, axes, 0));
  }

  /// The inverse of `rfftn` (numpy.fft.irfftn with s from `out`'s extents): `out` holds the
  /// real array, `in` the half spectrum with out.extents/2 + 1 elements along the last
  /// dimension listed and `out`'s extents elsewhere. As numpy does, the inverse transform
  /// runs along the other dimensions first and the halved one last, ignoring the imaginary
  /// parts its real transform ignores. Scaled as `nm` says. `in` is preserved.
  ///
  /// Throws `std::invalid_argument` as `rfftn` does, naming `in.extents` when they are not
  /// those.
  template <class In, class Out, std::size_t R>
    requires detail::c2r_elements<In, Out>
  void irfftn(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
    run(detail::call_of("irfftn", detail::easy_kind::c2r, direction::backward, nm, in, out, axes,
                        0));
  }

  /// `rfftn` along the last two dimensions by default (numpy.fft.rfft2).
  template <class In, class Out, std::size_t R>
    requires detail::r2c_elements<In, Out>
  void rfft2(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
    run(detail::call_of("rfft2", detail::easy_kind::r2c, direction::forward, nm, in, out, axes, 2));
  }

  /// `irfftn` along the last two dimensions by default (numpy.fft.irfft2).
  template <class In, class Out, std::size_t R>
    requires detail::c2r_elements<In, Out>
  void irfft2(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
    run(detail::call_of("irfft2", detail::easy_kind::c2r, direction::backward, nm, in, out, axes,
                        2));
  }

  /// The discrete cosine transform of type `type` of `in` into `out` (scipy.fft.dct, whose
  /// `type` is 1 to 4), n = in.size() points, scaled as `nm` says. With `norm::backward` it is
  /// FFTW's unnormalised REDFT00, REDFT10, REDFT01 or REDFT11 (`fftn::r2r_kind::dct1` to
  /// `dct4`, docs/conventions.md), as scipy's. `norm::ortho` also applies scipy's
  /// orthogonalising weights (its default `orthogonalize=True`): type 1 multiplies in[0] and
  /// in[n - 1] by sqrt 2 and divides out[0] and out[n - 1] by it, type 2 divides out[0] by
  /// sqrt 2, type 3 multiplies in[0] by sqrt 2, and type 4 has none. `in` and `out` may be the
  /// same memory (in place).
  ///
  /// Throws `std::invalid_argument` when `type` is not 1 to 4 (naming `type`), `in` is empty or,
  /// for type 1, of one point (naming `in`), or `out.size()` differs from `in.size()` (naming
  /// `out`). Preconditions: otherwise `in` and `out` do not overlap.
  void dct(std::span<const double> in, std::span<double> out, int type = 2,
           norm nm = norm::backward) {
    run_1d("dct", detail::easy_kind::r2r, false, direction::forward, nm, in.data(), in.size(),
           out.data(), out.size(), detail::easy_r2r::cosine, type);
  }

  /// `dct` in float.
  void dct(std::span<const float> in, std::span<float> out, int type = 2,
           norm nm = norm::backward) {
    run_1d("dct", detail::easy_kind::r2r, true, direction::forward, nm, in.data(), in.size(),
           out.data(), out.size(), detail::easy_r2r::cosine, type);
  }

  /// The inverse of `dct` of type `type` (scipy.fft.idct): the DCT of the inverse type (types
  /// 1 and 4 are their own, 2 and 3 each other's) with that type's weights under `norm::ortho`,
  /// scaled as `nm` says (by 1/N by default, N = 2(n - 1) for type 1 and 2n otherwise), so that
  /// `idct` of `dct` with the same type and norm returns the input. Arguments, exceptions and
  /// preconditions as for `dct`.
  void idct(std::span<const double> in, std::span<double> out, int type = 2,
            norm nm = norm::backward) {
    run_1d("idct", detail::easy_kind::r2r, false, direction::backward, nm, in.data(), in.size(),
           out.data(), out.size(), detail::easy_r2r::cosine, type);
  }

  /// `idct` in float.
  void idct(std::span<const float> in, std::span<float> out, int type = 2,
            norm nm = norm::backward) {
    run_1d("idct", detail::easy_kind::r2r, true, direction::backward, nm, in.data(), in.size(),
           out.data(), out.size(), detail::easy_r2r::cosine, type);
  }

  /// The discrete sine transform of type `type` of `in` into `out` (scipy.fft.dst), n =
  /// in.size() points, scaled as `nm` says: FFTW's RODFT00, RODFT10, RODFT01 or RODFT11
  /// (`fftn::r2r_kind::dst1` to `dst4`) with `norm::backward`. `norm::ortho` also applies
  /// scipy's orthogonalising weights: type 2 divides out[n - 1] by sqrt 2, type 3 multiplies
  /// in[n - 1] by sqrt 2, and types 1 and 4 have none. `in` and `out` may be the same memory
  /// (in place).
  ///
  /// Throws `std::invalid_argument` when `type` is not 1 to 4 (naming `type`), `in` is empty
  /// (naming `in`), or `out.size()` differs from `in.size()` (naming `out`). Preconditions:
  /// otherwise `in` and `out` do not overlap.
  void dst(std::span<const double> in, std::span<double> out, int type = 2,
           norm nm = norm::backward) {
    run_1d("dst", detail::easy_kind::r2r, false, direction::forward, nm, in.data(), in.size(),
           out.data(), out.size(), detail::easy_r2r::sine, type);
  }

  /// `dst` in float.
  void dst(std::span<const float> in, std::span<float> out, int type = 2,
           norm nm = norm::backward) {
    run_1d("dst", detail::easy_kind::r2r, true, direction::forward, nm, in.data(), in.size(),
           out.data(), out.size(), detail::easy_r2r::sine, type);
  }

  /// The inverse of `dst` of type `type` (scipy.fft.idst): the DST of the inverse type with
  /// that type's weights under `norm::ortho`, scaled as `nm` says (by 1/N by default, N =
  /// 2(n + 1) for type 1 and 2n otherwise). Arguments, exceptions and preconditions as for
  /// `dst`.
  void idst(std::span<const double> in, std::span<double> out, int type = 2,
            norm nm = norm::backward) {
    run_1d("idst", detail::easy_kind::r2r, false, direction::backward, nm, in.data(), in.size(),
           out.data(), out.size(), detail::easy_r2r::sine, type);
  }

  /// `idst` in float.
  void idst(std::span<const float> in, std::span<float> out, int type = 2,
            norm nm = norm::backward) {
    run_1d("idst", detail::easy_kind::r2r, true, direction::backward, nm, in.data(), in.size(),
           out.data(), out.size(), detail::easy_r2r::sine, type);
  }

  /// The discrete Hartley transform of `in` into `out`: out[k] = sum_j in[j] cas(2 pi jk / n)
  /// with cas = cos + sin and n = in.size() (`fftn::r2r_kind::dht`), scaled as `nm` says as a
  /// forward transform: unscaled by default, by 1/sqrt(n) with `norm::ortho` and by 1/n with
  /// `norm::forward`. The DHT is its own inverse up to the factor n, so `dht` with
  /// `norm::forward` undoes `dht` with `norm::backward`, and with `norm::ortho` it undoes
  /// itself. (scipy.fft has no DHT.) `in` and `out` may be the same memory (in place).
  ///
  /// Throws `std::invalid_argument` when `in` is empty (naming `in`) or `out.size()` differs
  /// from `in.size()` (naming `out`). Preconditions: otherwise `in` and `out` do not overlap.
  void dht(std::span<const double> in, std::span<double> out, norm nm = norm::backward) {
    run_1d("dht", detail::easy_kind::r2r, false, direction::forward, nm, in.data(), in.size(),
           out.data(), out.size(), detail::easy_r2r::hartley);
  }

  /// `dht` in float.
  void dht(std::span<const float> in, std::span<float> out, norm nm = norm::backward) {
    run_1d("dht", detail::easy_kind::r2r, true, direction::forward, nm, in.data(), in.size(),
           out.data(), out.size(), detail::easy_r2r::hartley);
  }

  /// `dct` into a new vector of in.size() values, which this form allocates on every call.
  [[nodiscard]] std::vector<double> dct(std::span<const double> in, int type = 2,
                                        norm nm = norm::backward) {
    std::vector<double> out(in.size());
    dct(in, out, type, nm);
    return out;
  }

  /// `dct` into a new vector, in float.
  [[nodiscard]] std::vector<float> dct(std::span<const float> in, int type = 2,
                                       norm nm = norm::backward) {
    std::vector<float> out(in.size());
    dct(in, out, type, nm);
    return out;
  }

  /// `idct` into a new vector of in.size() values, which this form allocates on every call.
  [[nodiscard]] std::vector<double> idct(std::span<const double> in, int type = 2,
                                         norm nm = norm::backward) {
    std::vector<double> out(in.size());
    idct(in, out, type, nm);
    return out;
  }

  /// `idct` into a new vector, in float.
  [[nodiscard]] std::vector<float> idct(std::span<const float> in, int type = 2,
                                        norm nm = norm::backward) {
    std::vector<float> out(in.size());
    idct(in, out, type, nm);
    return out;
  }

  /// `dst` into a new vector of in.size() values, which this form allocates on every call.
  [[nodiscard]] std::vector<double> dst(std::span<const double> in, int type = 2,
                                        norm nm = norm::backward) {
    std::vector<double> out(in.size());
    dst(in, out, type, nm);
    return out;
  }

  /// `dst` into a new vector, in float.
  [[nodiscard]] std::vector<float> dst(std::span<const float> in, int type = 2,
                                       norm nm = norm::backward) {
    std::vector<float> out(in.size());
    dst(in, out, type, nm);
    return out;
  }

  /// `idst` into a new vector of in.size() values, which this form allocates on every call.
  [[nodiscard]] std::vector<double> idst(std::span<const double> in, int type = 2,
                                         norm nm = norm::backward) {
    std::vector<double> out(in.size());
    idst(in, out, type, nm);
    return out;
  }

  /// `idst` into a new vector, in float.
  [[nodiscard]] std::vector<float> idst(std::span<const float> in, int type = 2,
                                        norm nm = norm::backward) {
    std::vector<float> out(in.size());
    idst(in, out, type, nm);
    return out;
  }

  /// `dht` into a new vector of in.size() values, which this form allocates on every call.
  [[nodiscard]] std::vector<double> dht(std::span<const double> in, norm nm = norm::backward) {
    std::vector<double> out(in.size());
    dht(in, out, nm);
    return out;
  }

  /// `dht` into a new vector, in float.
  [[nodiscard]] std::vector<float> dht(std::span<const float> in, norm nm = norm::backward) {
    std::vector<float> out(in.size());
    dht(in, out, nm);
    return out;
  }

  /// The DCT of type `type` along dimensions `axes` of real array `in` into `out`
  /// (scipy.fft.dctn), the other dimensions looped over: all dimensions by default (at most
  /// three). It is the separable product of `dct` along each listed dimension, with the same
  /// type along each: N is the product of each dimension's logical size, and under
  /// `norm::ortho` each dimension's weights apply. `out` has `in`'s extents; it may be the same
  /// memory with the same strides (in place). Elements are `double` or `float`, the input's
  /// possibly const.
  ///
  /// Throws `std::invalid_argument` naming the argument: `type` when it is not 1 to 4; `axes`
  /// with an index outside the array, an index twice, none or more than three dimensions to
  /// transform; `out.extents` unlike `in.extents`; a transformed dimension of extent 0, or of
  /// extent 1 for type 1; a stride of 0 among nonzero strides on a dimension longer than 1;
  /// `out` over `in`'s memory with other strides. A dimension of extent 0 that is not
  /// transformed makes the call do nothing. Preconditions: otherwise `in` and `out` do not
  /// overlap, and `out`'s elements do not overlap each other.
  template <class In, class Out, std::size_t R>
    requires detail::r2r_elements<In, Out>
  void dctn(view<In, R> in, view<Out, R> out, int type = 2, axis_list axes = {},
            norm nm = norm::backward) {
    run(detail::call_of("dctn", detail::easy_kind::r2r, direction::forward, nm, in, out, axes, 0,
                        detail::easy_r2r::cosine, type));
  }

  /// The inverse of `dctn` (scipy.fft.idctn): `idct` of type `type` along each listed
  /// dimension, scaled as `nm` says (by 1/N by default). As `dctn` otherwise.
  template <class In, class Out, std::size_t R>
    requires detail::r2r_elements<In, Out>
  void idctn(view<In, R> in, view<Out, R> out, int type = 2, axis_list axes = {},
             norm nm = norm::backward) {
    run(detail::call_of("idctn", detail::easy_kind::r2r, direction::backward, nm, in, out, axes, 0,
                        detail::easy_r2r::cosine, type));
  }

  /// The DST of type `type` along dimensions `axes` (scipy.fft.dstn): `dst` along each listed
  /// dimension, the weights of `dst` under `norm::ortho`. As `dctn` otherwise (a type 1 axis
  /// of one point is allowed).
  template <class In, class Out, std::size_t R>
    requires detail::r2r_elements<In, Out>
  void dstn(view<In, R> in, view<Out, R> out, int type = 2, axis_list axes = {},
            norm nm = norm::backward) {
    run(detail::call_of("dstn", detail::easy_kind::r2r, direction::forward, nm, in, out, axes, 0,
                        detail::easy_r2r::sine, type));
  }

  /// The inverse of `dstn` (scipy.fft.idstn): `idst` of type `type` along each listed
  /// dimension, scaled as `nm` says (by 1/N by default). As `dstn` otherwise.
  template <class In, class Out, std::size_t R>
    requires detail::r2r_elements<In, Out>
  void idstn(view<In, R> in, view<Out, R> out, int type = 2, axis_list axes = {},
             norm nm = norm::backward) {
    run(detail::call_of("idstn", detail::easy_kind::r2r, direction::backward, nm, in, out, axes, 0,
                        detail::easy_r2r::sine, type));
  }

private:
  struct state;

  void run_1d(const char* function, detail::easy_kind kind, bool single, direction dir, norm nm,
              const void* in, std::size_t in_size, void* out, std::size_t out_size,
              detail::easy_r2r family = detail::easy_r2r::none, int type = 0);
  static std::size_t irfft_length(std::size_t in_size, std::size_t n);
  void run(const detail::easy_call& call);
  state& own();

  std::unique_ptr<state> state_;
  std::optional<executor_ref> executor_;
};

/// The calling thread's own context, used by the free functions: serial, created on the
/// thread's first use and destroyed when the thread exits (a `thread_local`). It is the only
/// state FFTN keeps behind the caller's back (docs/easy.md); `clear()` releases its memory,
/// and `set_max_plans` bounds it.
///
/// Preconditions: not called once the thread's `thread_local` objects are being destroyed
/// (from another `thread_local`'s destructor, say): the behaviour is then undefined.
///
/// Guarantees: each thread gets a different context, so the free functions take no lock and
/// share nothing between threads.
[[nodiscard]] context& thread_context() noexcept;

/// `context::fft` on the calling thread's context.
inline void fft(std::span<const std::complex<double>> in, std::span<std::complex<double>> out,
                norm nm = norm::backward) {
  thread_context().fft(in, out, nm);
}

/// `context::fft` in float on the calling thread's context.
inline void fft(std::span<const std::complex<float>> in, std::span<std::complex<float>> out,
                norm nm = norm::backward) {
  thread_context().fft(in, out, nm);
}

/// `context::ifft` on the calling thread's context.
inline void ifft(std::span<const std::complex<double>> in, std::span<std::complex<double>> out,
                 norm nm = norm::backward) {
  thread_context().ifft(in, out, nm);
}

/// `context::ifft` in float on the calling thread's context.
inline void ifft(std::span<const std::complex<float>> in, std::span<std::complex<float>> out,
                 norm nm = norm::backward) {
  thread_context().ifft(in, out, nm);
}

/// `context::rfft` on the calling thread's context.
inline void rfft(std::span<const double> in, std::span<std::complex<double>> out,
                 norm nm = norm::backward) {
  thread_context().rfft(in, out, nm);
}

/// `context::rfft` in float on the calling thread's context.
inline void rfft(std::span<const float> in, std::span<std::complex<float>> out,
                 norm nm = norm::backward) {
  thread_context().rfft(in, out, nm);
}

/// `context::irfft` on the calling thread's context.
inline void irfft(std::span<const std::complex<double>> in, std::span<double> out,
                  norm nm = norm::backward) {
  thread_context().irfft(in, out, nm);
}

/// `context::irfft` in float on the calling thread's context.
inline void irfft(std::span<const std::complex<float>> in, std::span<float> out,
                  norm nm = norm::backward) {
  thread_context().irfft(in, out, nm);
}

/// `context::fft` into a new vector (allocated on every call) on the calling thread's context.
[[nodiscard]] inline std::vector<std::complex<double>> fft(std::span<const std::complex<double>> in,
                                                           norm nm = norm::backward) {
  return thread_context().fft(in, nm);
}

/// `context::fft` into a new vector, in float, on the calling thread's context.
[[nodiscard]] inline std::vector<std::complex<float>> fft(std::span<const std::complex<float>> in,
                                                          norm nm = norm::backward) {
  return thread_context().fft(in, nm);
}

/// `context::ifft` into a new vector on the calling thread's context.
[[nodiscard]] inline std::vector<std::complex<double>>
ifft(std::span<const std::complex<double>> in, norm nm = norm::backward) {
  return thread_context().ifft(in, nm);
}

/// `context::ifft` into a new vector, in float, on the calling thread's context.
[[nodiscard]] inline std::vector<std::complex<float>> ifft(std::span<const std::complex<float>> in,
                                                           norm nm = norm::backward) {
  return thread_context().ifft(in, nm);
}

/// `context::rfft` into a new vector on the calling thread's context.
[[nodiscard]] inline std::vector<std::complex<double>> rfft(std::span<const double> in,
                                                            norm nm = norm::backward) {
  return thread_context().rfft(in, nm);
}

/// `context::rfft` into a new vector, in float, on the calling thread's context.
[[nodiscard]] inline std::vector<std::complex<float>> rfft(std::span<const float> in,
                                                           norm nm = norm::backward) {
  return thread_context().rfft(in, nm);
}

/// `context::irfft` into a new vector of `n` points (0: 2(in.size() - 1)) on the calling
/// thread's context.
[[nodiscard]] inline std::vector<double> irfft(std::span<const std::complex<double>> in,
                                               std::size_t n = 0, norm nm = norm::backward) {
  return thread_context().irfft(in, n, nm);
}

/// `context::irfft` into a new vector, in float, on the calling thread's context.
[[nodiscard]] inline std::vector<float> irfft(std::span<const std::complex<float>> in,
                                              std::size_t n = 0, norm nm = norm::backward) {
  return thread_context().irfft(in, n, nm);
}

/// `context::fftn` on the calling thread's context.
template <class In, class Out, std::size_t R>
  requires detail::c2c_elements<In, Out>
void fftn(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
  thread_context().fftn(in, out, axes, nm);
}

/// `context::ifftn` on the calling thread's context.
template <class In, class Out, std::size_t R>
  requires detail::c2c_elements<In, Out>
void ifftn(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
  thread_context().ifftn(in, out, axes, nm);
}

/// `context::fft2` on the calling thread's context.
template <class In, class Out, std::size_t R>
  requires detail::c2c_elements<In, Out>
void fft2(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
  thread_context().fft2(in, out, axes, nm);
}

/// `context::ifft2` on the calling thread's context.
template <class In, class Out, std::size_t R>
  requires detail::c2c_elements<In, Out>
void ifft2(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
  thread_context().ifft2(in, out, axes, nm);
}

/// `context::rfftn` on the calling thread's context.
template <class In, class Out, std::size_t R>
  requires detail::r2c_elements<In, Out>
void rfftn(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
  thread_context().rfftn(in, out, axes, nm);
}

/// `context::irfftn` on the calling thread's context.
template <class In, class Out, std::size_t R>
  requires detail::c2r_elements<In, Out>
void irfftn(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
  thread_context().irfftn(in, out, axes, nm);
}

/// `context::rfft2` on the calling thread's context.
template <class In, class Out, std::size_t R>
  requires detail::r2c_elements<In, Out>
void rfft2(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
  thread_context().rfft2(in, out, axes, nm);
}

/// `context::irfft2` on the calling thread's context.
template <class In, class Out, std::size_t R>
  requires detail::c2r_elements<In, Out>
void irfft2(view<In, R> in, view<Out, R> out, axis_list axes = {}, norm nm = norm::backward) {
  thread_context().irfft2(in, out, axes, nm);
}

/// `context::dct` on the calling thread's context.
inline void dct(std::span<const double> in, std::span<double> out, int type = 2,
                norm nm = norm::backward) {
  thread_context().dct(in, out, type, nm);
}

/// `context::dct` in float on the calling thread's context.
inline void dct(std::span<const float> in, std::span<float> out, int type = 2,
                norm nm = norm::backward) {
  thread_context().dct(in, out, type, nm);
}

/// `context::idct` on the calling thread's context.
inline void idct(std::span<const double> in, std::span<double> out, int type = 2,
                 norm nm = norm::backward) {
  thread_context().idct(in, out, type, nm);
}

/// `context::idct` in float on the calling thread's context.
inline void idct(std::span<const float> in, std::span<float> out, int type = 2,
                 norm nm = norm::backward) {
  thread_context().idct(in, out, type, nm);
}

/// `context::dst` on the calling thread's context.
inline void dst(std::span<const double> in, std::span<double> out, int type = 2,
                norm nm = norm::backward) {
  thread_context().dst(in, out, type, nm);
}

/// `context::dst` in float on the calling thread's context.
inline void dst(std::span<const float> in, std::span<float> out, int type = 2,
                norm nm = norm::backward) {
  thread_context().dst(in, out, type, nm);
}

/// `context::idst` on the calling thread's context.
inline void idst(std::span<const double> in, std::span<double> out, int type = 2,
                 norm nm = norm::backward) {
  thread_context().idst(in, out, type, nm);
}

/// `context::idst` in float on the calling thread's context.
inline void idst(std::span<const float> in, std::span<float> out, int type = 2,
                 norm nm = norm::backward) {
  thread_context().idst(in, out, type, nm);
}

/// `context::dht` on the calling thread's context.
inline void dht(std::span<const double> in, std::span<double> out, norm nm = norm::backward) {
  thread_context().dht(in, out, nm);
}

/// `context::dht` in float on the calling thread's context.
inline void dht(std::span<const float> in, std::span<float> out, norm nm = norm::backward) {
  thread_context().dht(in, out, nm);
}

/// `context::dct` into a new vector on the calling thread's context.
[[nodiscard]] inline std::vector<double> dct(std::span<const double> in, int type = 2,
                                             norm nm = norm::backward) {
  return thread_context().dct(in, type, nm);
}

/// `context::dct` into a new vector, in float, on the calling thread's context.
[[nodiscard]] inline std::vector<float> dct(std::span<const float> in, int type = 2,
                                            norm nm = norm::backward) {
  return thread_context().dct(in, type, nm);
}

/// `context::idct` into a new vector on the calling thread's context.
[[nodiscard]] inline std::vector<double> idct(std::span<const double> in, int type = 2,
                                              norm nm = norm::backward) {
  return thread_context().idct(in, type, nm);
}

/// `context::idct` into a new vector, in float, on the calling thread's context.
[[nodiscard]] inline std::vector<float> idct(std::span<const float> in, int type = 2,
                                             norm nm = norm::backward) {
  return thread_context().idct(in, type, nm);
}

/// `context::dst` into a new vector on the calling thread's context.
[[nodiscard]] inline std::vector<double> dst(std::span<const double> in, int type = 2,
                                             norm nm = norm::backward) {
  return thread_context().dst(in, type, nm);
}

/// `context::dst` into a new vector, in float, on the calling thread's context.
[[nodiscard]] inline std::vector<float> dst(std::span<const float> in, int type = 2,
                                            norm nm = norm::backward) {
  return thread_context().dst(in, type, nm);
}

/// `context::idst` into a new vector on the calling thread's context.
[[nodiscard]] inline std::vector<double> idst(std::span<const double> in, int type = 2,
                                              norm nm = norm::backward) {
  return thread_context().idst(in, type, nm);
}

/// `context::idst` into a new vector, in float, on the calling thread's context.
[[nodiscard]] inline std::vector<float> idst(std::span<const float> in, int type = 2,
                                             norm nm = norm::backward) {
  return thread_context().idst(in, type, nm);
}

/// `context::dht` into a new vector on the calling thread's context.
[[nodiscard]] inline std::vector<double> dht(std::span<const double> in, norm nm = norm::backward) {
  return thread_context().dht(in, nm);
}

/// `context::dht` into a new vector, in float, on the calling thread's context.
[[nodiscard]] inline std::vector<float> dht(std::span<const float> in, norm nm = norm::backward) {
  return thread_context().dht(in, nm);
}

/// `context::dctn` on the calling thread's context.
template <class In, class Out, std::size_t R>
  requires detail::r2r_elements<In, Out>
void dctn(view<In, R> in, view<Out, R> out, int type = 2, axis_list axes = {},
          norm nm = norm::backward) {
  thread_context().dctn(in, out, type, axes, nm);
}

/// `context::idctn` on the calling thread's context.
template <class In, class Out, std::size_t R>
  requires detail::r2r_elements<In, Out>
void idctn(view<In, R> in, view<Out, R> out, int type = 2, axis_list axes = {},
           norm nm = norm::backward) {
  thread_context().idctn(in, out, type, axes, nm);
}

/// `context::dstn` on the calling thread's context.
template <class In, class Out, std::size_t R>
  requires detail::r2r_elements<In, Out>
void dstn(view<In, R> in, view<Out, R> out, int type = 2, axis_list axes = {},
          norm nm = norm::backward) {
  thread_context().dstn(in, out, type, axes, nm);
}

/// `context::idstn` on the calling thread's context.
template <class In, class Out, std::size_t R>
  requires detail::r2r_elements<In, Out>
void idstn(view<In, R> in, view<Out, R> out, int type = 2, axis_list axes = {},
           norm nm = norm::backward) {
  thread_context().idstn(in, out, type, axes, nm);
}

/// The sample frequencies of an `fft` of `n` points spaced `d` apart (numpy.fft.fftfreq),
/// into `out`: k / (n d) for k = 0, ..., (n - 1)/2, then the negative ones from -(n/2) to -1,
/// computed in double as numpy does (the integer k times 1 / (n d)) and rounded once to the
/// output's type.
///
/// Throws `std::invalid_argument` when `n` is 0 (naming `n`) or `out.size()` is not `n`
/// (naming `out`).
inline void fftfreq(std::size_t n, std::span<double> out, double d = 1.0) {
  detail::frequencies<double>("fftfreq", n, out, d, false);
}

/// `fftfreq` in float.
inline void fftfreq(std::size_t n, std::span<float> out, double d = 1.0) {
  detail::frequencies<float>("fftfreq", n, out, d, false);
}

/// The sample frequencies of an `rfft` of `n` points spaced `d` apart
/// (numpy.fft.rfftfreq), into `out`: k / (n d) for k = 0, ..., n/2, computed as `fftfreq`'s.
///
/// Throws `std::invalid_argument` when `n` is 0 (naming `n`) or `out.size()` is not n/2 + 1
/// (naming `out`).
inline void rfftfreq(std::size_t n, std::span<double> out, double d = 1.0) {
  detail::frequencies<double>("rfftfreq", n, out, d, true);
}

/// `rfftfreq` in float.
inline void rfftfreq(std::size_t n, std::span<float> out, double d = 1.0) {
  detail::frequencies<float>("rfftfreq", n, out, d, true);
}

/// Moves the zero-frequency element to the middle of each dimension listed in `axes` (all by
/// default), as numpy.fft.fftshift: element i of a dimension of extent n goes to
/// (i + n/2) mod n. `out` has `in`'s extents; it may be the same memory with the same
/// strides (in place). Any element type that can be copied (in place: swapped).
///
/// Throws `std::invalid_argument` when `out.extents` differ from `in.extents` (naming
/// `out.extents`), `axes` is not valid (naming `axes`; an `axis_list` may list up to eight
/// dimensions here), or a view gives a stride of 0 among nonzero strides on a dimension longer
/// than 1 (naming `in.strides` or `out.strides`); an empty array is left alone. Preconditions:
/// otherwise `in` and `out` do not overlap. Never allocates unless it throws.
template <class In, class Out, std::size_t R>
  requires std::is_same_v<std::remove_const_t<In>, Out>
void fftshift(view<In, R> in, view<Out, R> out, axis_list axes = {}) {
  detail::shift("fftshift", in, out, axes, false);
}

/// The inverse of `fftshift` (numpy.fft.ifftshift): element i of a dimension of extent n goes
/// to (i + n - n/2) mod n, which differs from `fftshift` for odd n. As `fftshift` otherwise.
template <class In, class Out, std::size_t R>
  requires std::is_same_v<std::remove_const_t<In>, Out>
void ifftshift(view<In, R> in, view<Out, R> out, axis_list axes = {}) {
  detail::shift("ifftshift", in, out, axes, true);
}

} // namespace fftn::inline v0::easy
