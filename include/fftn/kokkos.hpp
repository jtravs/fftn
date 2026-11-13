// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// The Kokkos adapter: FFTN transforms of host-accessible `Kokkos::View`s.
///
/// Optional and header-only. `fftn.hpp` does not include it, and it compiles only in a
/// translation unit that can include `Kokkos_Core.hpp`: FFTN itself never depends on Kokkos.
/// It offers two levels:
///
/// - View-based construction: `shape_of` and `make_plan` turn a pair of Views and the axes to
///   transform into an ordinary `fftn::shape` or `fftn::plan`, and `data_of` gives the
///   pointers to execute it with. Views of `LayoutRight`, `LayoutLeft` and `LayoutStride`
///   are read through their extents and strides.
/// - `fftn::kokkos::view_plan`, a plan class with the constructor and execute shape of
///   Kokkos-FFT 2.0's `KokkosFFT::Plan`, and the free `execute(plan, in, out, norm)`.
///   `fftn::kokkos::compat` spells these as Kokkos-FFT does (`Plan`, `Direction`,
///   `Normalization`, `axis_type`, `shape_type`, `execute`), so code written for Kokkos-FFT
///   changes only its include and its namespace.
///
/// Execution is serial, on the calling thread. How this differs from Kokkos-FFT, and the
/// rules for each argument, are in docs/kokkos.md.
#pragma once

#include "fftn/config.hpp"
#include "fftn/options.hpp"
#include "fftn/plan.hpp"
#include "fftn/shape.hpp"
#include "fftn/workspace.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace fftn::inline v0::kokkos {

/// When and how a transform's output is scaled, as Kokkos-FFT's `Normalization`.
///
/// N is the number of points of one transform (the product of the transformed lengths; for
/// real transforms the real side's lengths).
enum class normalization : std::int8_t {
  forward,  ///< forward transforms are multiplied by 1/N, backward ones are not scaled
  backward, ///< backward transforms are multiplied by 1/N, forward ones are not scaled
  ortho,    ///< every transform is multiplied by 1/sqrt(N)
  none,     ///< nothing is scaled, as FFTN's core plans
};

/// The axes to transform, as Kokkos-FFT's `axis_type`: indices into a View's dimensions,
/// negative values counting from the end (-1 is the last dimension).
template <std::size_t Dim> using axis_type = std::array<int, Dim>;

namespace compat {

/// `KokkosFFT::Direction`, declared as Kokkos-FFT 2.0 declares it: the same names, values
/// and underlying type, so code that converts a direction to an integer sees Kokkos-FFT's
/// 0 and 1 (FFTN's own `fftn::direction` is the exponent's sign, -1 and +1). `compat::Plan`
/// takes it and converts it to `fftn::direction`.
// NOLINTNEXTLINE(performance-enum-size): int, as Kokkos-FFT's, so conversions agree with it
enum class Direction : int {
  forward = 0,  ///< the forward transform, exponent -2 pi i jk / n
  backward = 1, ///< the backward transform, exponent +2 pi i jk / n
};

} // namespace compat

namespace detail {

template <class Value> struct element_of {
  using type = Value;
};
template <class T> struct element_of<Kokkos::complex<T>> {
  using type = std::complex<T>;
};
template <class T> struct element_of<std::complex<T>> {
  using type = std::complex<T>;
};

template <class Value> struct real_of {
  using type = Value;
};
template <class T> struct real_of<Kokkos::complex<T>> {
  using type = T;
};
template <class T> struct real_of<std::complex<T>> {
  using type = T;
};

template <class Value> struct is_element : std::false_type {};
template <> struct is_element<float> : std::true_type {};
template <> struct is_element<double> : std::true_type {};
template <class T> struct is_element<std::complex<T>> : is_element<T> {};
template <class T> struct is_element<Kokkos::complex<T>> : is_element<T> {};

[[noreturn]] inline void reject(const std::string& argument, const std::string& why) {
  throw std::invalid_argument("fftn::kokkos: " + argument + " " + why);
}

// Kokkos-FFT's direction as FFTN's. A value that is neither enumerator is refused, as
// FFTN's plans refuse an invalid fftn::direction.
inline direction direction_of(compat::Direction dir) {
  switch (dir) {
  case compat::Direction::forward:
    return direction::forward;
  case compat::Direction::backward:
    return direction::backward;
  }
  reject("dir", "is neither Direction::forward nor Direction::backward");
}

// Kokkos reports strides as unsigned values; FFTN's are signed. A View's strides address its
// own allocation, so they fit.
inline std::ptrdiff_t signed_stride(std::size_t stride) {
  return static_cast<std::ptrdiff_t>(stride);
}

inline std::string dim_name(const char* view, std::size_t d, const char* what) {
  return std::string(view) + "." + what + "(" + std::to_string(d) + ")";
}

template <class View> constexpr bool host_accessible_view() {
  return Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace,
                                    typename View::memory_space>::accessible;
}

// The view's memory as bytes [first, last), for the overlap test.
template <class View> std::pair<const std::byte*, const std::byte*> byte_range(const View& v) {
  const auto* first = reinterpret_cast<const std::byte*>(v.data());
  return {first, first + v.span() * sizeof(typename View::value_type)};
}

template <class InView, class OutView>
bool same_first_element(const InView& in, const OutView& out) {
  return static_cast<const void*>(in.data()) == static_cast<const void*>(out.data());
}

template <class InView, class OutView> bool overlapping(const InView& in, const OutView& out) {
  const auto [in_first, in_last] = byte_range(in);
  const auto [out_first, out_last] = byte_range(out);
  const std::less<> before;
  return before(in_first, out_last) && before(out_first, in_last);
}

// The label of the fence execute makes. Kokkos takes fence labels as std::string, and this
// one is longer than any small-string buffer: built once, at the first plan's construction,
// so that execute never allocates (Kokkos itself allocates nothing for a fence on a host
// back end unless a Kokkos Tools library is loaded).
inline const std::string& fence_label() {
  static const std::string label("fftn::kokkos::view_plan::execute: fence before the transform");
  return label;
}

// Clears a plan's busy flag when an execution ends, however it ends.
class busy_guard {
public:
  explicit busy_guard(std::atomic<bool>& busy) noexcept : busy_(busy) {}
  busy_guard(const busy_guard&) = delete;
  busy_guard& operator=(const busy_guard&) = delete;
  busy_guard(busy_guard&&) = delete;
  busy_guard& operator=(busy_guard&&) = delete;
  ~busy_guard() { busy_.store(false, std::memory_order_release); }

private:
  std::atomic<bool>& busy_;
};

// The plan of a shape from shape_of. The shape's own checks come first, phrased in terms of
// the Views; the plan's remaining ones (overlapping output, sizes that overflow) are passed on
// with the Views named.
template <class In, class Out> fftn::plan<In, Out> plan_of(const shape<In, Out>& s, bool in_place) {
  try {
    return fftn::plan<In, Out>(s, options{.in_place = in_place});
  } catch (const std::invalid_argument& e) {
    reject("in, out", std::string("describe a layout FFTN refuses: ") + e.what());
  }
}

} // namespace detail

/// The FFTN element type of a View's values: `std::complex<T>` for `Kokkos::complex<T>` and
/// `std::complex<T>` values, `T` for real ones (`T` = `float` or `double`). The two complex
/// types have the same layout, two `T` values, so FFTN reads and writes either in place
/// (`data_of` checks the size and alignment).
template <class View>
using element_t = detail::element_of<typename View::non_const_value_type>::type;

/// A pointer to the first element of `v`, as FFTN's execute functions take it: `const` when
/// the View's values are `const`.
///
/// Guarantees: the address is `v.data()`; only the pointer type changes (a
/// `Kokkos::complex<T>*` becomes a `std::complex<T>*`, which has the same layout).
template <class View> [[nodiscard]] auto data_of(const View& v) noexcept {
  using value = View::value_type;
  static_assert(detail::is_element<typename View::non_const_value_type>::value,
                "fftn::kokkos: View values must be float, double, or Kokkos::complex or "
                "std::complex of either");
  static_assert(sizeof(value) == sizeof(element_t<View>) &&
                    alignof(value) >= alignof(element_t<View>),
                "fftn::kokkos: a complex value must be laid out as two real values");
  using element =
      std::conditional_t<std::is_const_v<value>, const element_t<View>, element_t<View>>;
  if constexpr (std::is_same_v<value, element>) {
    return v.data();
  } else {
    // Kokkos::complex<T> and std::complex<T> are both two T values (static_assert in
    // shape_of); the library reads them through its own pointer type in another
    // translation unit, as Kokkos-FFT does with FFTW's complex type.
    return reinterpret_cast<element*>(v.data());
  }
}

/// The FFTN shape that transforms the dimensions `axes` of `in` into `out`.
///
/// `axes` lists the transformed dimensions in FFTN's order (`shape::axes`): for real
/// transforms the last one is halved, as in Kokkos-FFT. Negative entries count from the end.
/// Every other dimension is a batch dimension; batch dimensions that nest (one spans the
/// other exactly, in both Views) are merged, and dimensions of extent 1 are dropped.
/// Strides and distances are the Views' own, so any layout works.
///
/// Preconditions: the Views are of equal rank, at least `Dim`; their memory is accessible
/// from the host (checked at compile time); their values are `float`, `double` or a
/// complex of either, the pair forming one of FFTN's kinds (c2c, r2c, c2r) in one precision.
///
/// Throws `std::invalid_argument` naming the argument (`in`, `out`, `axes`) when an axis is
/// out of range or repeated; an extent is 0; a stride is 0; the extents do not match the
/// kind (equal everywhere, except along the halved axis of a real transform, where the
/// complex side has n/2 + 1 for the real side's n); `in` and `out` overlap without starting
/// at the same address; or, when they do start
/// at the same address (in place), their strides disagree: equal for c2c, and FFTW's padded
/// layout for real kinds (the halved axis contiguous on both sides, every other real stride
/// twice the complex one).
///
/// Guarantees: when `in` and `out` start at the same address the shape is for a plan built
/// with `options::in_place`; nothing is read or written through the Views.
template <class InView, class OutView, std::size_t Dim>
[[nodiscard]] shape<element_t<InView>, element_t<OutView>>
shape_of(const InView& in, const OutView& out, direction dir, const axis_type<Dim>& axes) {
  static_assert(Kokkos::is_view_v<InView> && Kokkos::is_view_v<OutView>,
                "fftn::kokkos: in and out must be Kokkos::Views");
  static_assert(detail::host_accessible_view<InView>() && detail::host_accessible_view<OutView>(),
                "fftn::kokkos: the Views' memory must be accessible from the host");
  using in_value = InView::non_const_value_type;
  using out_value = OutView::non_const_value_type;
  static_assert(detail::is_element<in_value>::value && detail::is_element<out_value>::value,
                "fftn::kokkos: View values must be float, double, or Kokkos::complex or "
                "std::complex of either");
  static_assert(sizeof(in_value) == sizeof(element_t<InView>) &&
                    sizeof(out_value) == sizeof(element_t<OutView>),
                "fftn::kokkos: a complex value must be two real values");
  static_assert(!std::is_const_v<typename OutView::value_type>,
                "fftn::kokkos: out must be writable");
  constexpr std::size_t rank = InView::rank();
  static_assert(OutView::rank() == rank, "fftn::kokkos: in and out must have the same rank");
  static_assert(Dim >= 1 && Dim <= 3 && Dim <= rank,
                "fftn::kokkos: one to three axes, at most the Views' rank");
  using In = element_t<InView>;
  using Out = element_t<OutView>;
  constexpr bool r2c =
      !std::is_same_v<In, std::complex<double>> && !std::is_same_v<In, std::complex<float>>;
  constexpr bool c2r =
      !std::is_same_v<Out, std::complex<double>> && !std::is_same_v<Out, std::complex<float>>;

  // The transformed dimensions, non-negative.
  std::array<std::size_t, Dim> dims{};
  std::array<bool, rank> transformed{};
  for (std::size_t i = 0; i < Dim; ++i) {
    const int a = axes[i];
    const auto r = static_cast<int>(rank);
    if (a < -r || a >= r) {
      detail::reject("axes[" + std::to_string(i) + "]",
                     "= " + std::to_string(a) + " is outside the Views' rank " + std::to_string(r));
    }
    dims[i] = static_cast<std::size_t>(a < 0 ? a + r : a);
    if (transformed[dims[i]]) {
      detail::reject("axes", "lists dimension " + std::to_string(dims[i]) + " twice");
    }
    transformed[dims[i]] = true;
  }
  const std::size_t halved = dims[Dim - 1];

  std::array<std::size_t, rank> extent{};
  std::array<std::ptrdiff_t, rank> in_stride{};
  std::array<std::ptrdiff_t, rank> out_stride{};
  for (std::size_t d = 0; d < rank; ++d) {
    const std::size_t ni = in.extent(d);
    const std::size_t no = out.extent(d);
    if (ni == 0) {
      detail::reject(detail::dim_name("in", d, "extent"), "is 0");
    }
    if (no == 0) {
      detail::reject(detail::dim_name("out", d, "extent"), "is 0");
    }
    extent[d] = ni;
    std::size_t expected = ni;
    if ((r2c || c2r) && d == halved) {
      extent[d] = r2c ? ni : no;
      expected = r2c ? ni / 2 + 1 : no;
      if (c2r && ni != no / 2 + 1) {
        detail::reject(detail::dim_name("in", d, "extent"),
                       "= " + std::to_string(ni) + " must be n/2 + 1 for out's n = " +
                           std::to_string(no) + " along the halved axis");
      }
    }
    if (no != expected) {
      detail::reject(
          detail::dim_name("out", d, "extent"),
          "= " + std::to_string(no) + " must be " + std::to_string(expected) +
              (d == halved && r2c ? " (n/2 + 1 along the halved axis)" : " (in's extent)"));
    }
    in_stride[d] = detail::signed_stride(in.stride(d));
    out_stride[d] = detail::signed_stride(out.stride(d));
    // A dimension of extent 1 never steps, so its stride does not matter (Kokkos may report
    // 0 for it in a LayoutStride View).
    if (extent[d] == 1 && ni == 1 && no == 1) {
      continue;
    }
    if (in_stride[d] == 0) {
      detail::reject(detail::dim_name("in", d, "stride"), "is 0 (broadcast Views are refused)");
    }
    if (out_stride[d] == 0) {
      detail::reject(detail::dim_name("out", d, "stride"), "is 0 (broadcast Views are refused)");
    }
  }

  if (detail::same_first_element(in, out)) {
    for (std::size_t d = 0; d < rank; ++d) {
      if (in.extent(d) == 1 && out.extent(d) == 1) {
        continue;
      }
      if constexpr (!r2c && !c2r) {
        if (in_stride[d] != out_stride[d]) {
          detail::reject(detail::dim_name("out", d, "stride"),
                         "must equal in's: in and out start at the same address, so the "
                         "transform is in place");
        }
      } else {
        const char* real = r2c ? "in" : "out";
        const std::ptrdiff_t rs = r2c ? in_stride[d] : out_stride[d];
        const std::ptrdiff_t cs = r2c ? out_stride[d] : in_stride[d];
        if (d == halved ? rs != 1 || cs != 1 : rs != 2 * cs) {
          detail::reject(detail::dim_name(real, d, "stride"),
                         d == halved ? "must be 1 on both sides along the halved axis of an "
                                       "in-place real transform"
                                     : "must be twice the complex side's in an in-place real "
                                       "transform (FFTW's padded layout)");
        }
      }
    }
  } else if (detail::overlapping(in, out)) {
    detail::reject("out", "overlaps in without starting at the same address (in place needs "
                          "the same first element)");
  }

  shape<In, Out> s;
  s.dir = dir;
  for (std::size_t i = 0; i < Dim; ++i) {
    s.axes.push_back(axis{
        .n = extent[dims[i]],
        .in_stride = in_stride[dims[i]],
        .out_stride = out_stride[dims[i]],
    });
  }

  // Batch dimensions, widest input stride first; then merge the ones that nest.
  std::array<batch, rank> loops{};
  std::size_t count = 0;
  for (std::size_t d = 0; d < rank; ++d) {
    if (!transformed[d] && extent[d] > 1) {
      loops[count++] =
          batch{.count = extent[d], .in_dist = in_stride[d], .out_dist = out_stride[d]};
    }
  }
  std::stable_sort(loops.begin(), loops.begin() + static_cast<std::ptrdiff_t>(count),
                   [](const batch& a, const batch& b) { return a.in_dist > b.in_dist; });
  std::size_t merged = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const batch& inner = loops[i];
    if (merged > 0) {
      batch& outer = loops[merged - 1];
      const auto c = static_cast<std::ptrdiff_t>(inner.count);
      if (outer.in_dist == inner.in_dist * c && outer.out_dist == inner.out_dist * c) {
        outer = batch{
            .count = outer.count * inner.count,
            .in_dist = inner.in_dist,
            .out_dist = inner.out_dist,
        };
        continue;
      }
    }
    loops[merged++] = inner;
  }
  for (std::size_t i = 0; i < merged; ++i) {
    s.batches.push_back(loops[i]);
  }
  return s;
}

/// An FFTN plan that transforms the dimensions `axes` of `in` into `out`: the plan of
/// `shape_of(in, out, dir, axes)`, built in place when `in` and `out` start at the same
/// address. Execute it with `data_of(in)` and `data_of(out)` (or any Views of the same
/// layout) and a workspace of its own per concurrent execution.
///
/// Preconditions: as for `shape_of`.
///
/// Throws `std::invalid_argument` naming the argument when the Views are refused by
/// `shape_of` or their layout by the plan (for example output elements that overlap), and
/// `std::bad_alloc` when memory runs out.
///
/// Guarantees: as for `fftn::plan`. The result is the same plan, with the same results bit
/// for bit, as one built from the equivalent shape by hand.
template <class InView, class OutView, std::size_t Dim>
[[nodiscard]] fftn::plan<element_t<InView>, element_t<OutView>>
make_plan(const InView& in, const OutView& out, direction dir, const axis_type<Dim>& axes) {
  return detail::plan_of(shape_of(in, out, dir, axes), detail::same_first_element(in, out));
}

/// A plan with the constructor and execute shape of Kokkos-FFT 2.0's `KokkosFFT::Plan`, over
/// FFTN.
///
/// It transforms the dimensions `axes` of Views typed `InView` into Views typed `OutView`,
/// with the rules of `shape_of`, and owns everything an execution needs: the FFTN plan and
/// a workspace, both made at construction.
///
/// Differences from Kokkos-FFT (docs/kokkos.md has the full list):
/// - Execution is serial on the calling thread, whatever `ExecutionSpace` is.
/// - One plan runs one execution at a time: its executions share its workspace, and a
///   second concurrent execution throws `std::logic_error`. Concurrent transforms need one
///   plan each, or an `fftn::plan` from `make_plan` with a workspace per execution.
/// - Any layout is accepted (`LayoutStride` too), and in-place real transforms take the real
///   side's padded layout from its View's strides, which must show it.
/// - Cropping and padding (`n`, `s`, or extents that disagree) are refused.
/// - Errors are `std::invalid_argument` naming the argument; execution checks the Views it
///   is given and throws on a mismatch, and otherwise allocates nothing.
///
/// Guarantees: the results are those of the `fftn::plan` it holds, bit for bit, then scaled
/// as the execute call's `normalization` asks. Not copyable or movable, as Kokkos-FFT's.
template <class ExecutionSpace, class InView, class OutView, std::size_t Dim = 1> class view_plan {
  static_assert(Kokkos::is_execution_space_v<ExecutionSpace>,
                "fftn::kokkos::view_plan: ExecutionSpace must be a Kokkos execution space");
  static_assert(Kokkos::SpaceAccessibility<ExecutionSpace, Kokkos::HostSpace>::accessible,
                "fftn::kokkos::view_plan: ExecutionSpace must run on the host");
  static constexpr std::size_t rank = InView::rank();
  using in_element = element_t<InView>;
  using out_element = element_t<OutView>;

public:
  /// A plan of a one-dimensional transform along dimension `axis` of the Views (negative
  /// counts from the end), as Kokkos-FFT's one-axis constructor.
  ///
  /// `n`, Kokkos-FFT's output length, may be given only as the length the Views already
  /// have (FFTN does not crop or pad). Otherwise as the constructor taking `axes`.
  explicit view_plan(const ExecutionSpace& exec_space, const InView& in, const OutView& out,
                     direction dir, int axis, std::optional<std::size_t> n = std::nullopt)
    requires(Dim == 1)
      : view_plan(exec_space, in, out, dir, axis_type<1>{axis},
                  std::array<std::size_t, 1>{n.value_or(0)}) {}

  /// A plan of a transform over the dimensions `axes` of the Views (negative entries count
  /// from the end), as Kokkos-FFT's constructor; for real transforms the last axis listed is
  /// halved.
  ///
  /// `s`, Kokkos-FFT's transform lengths (`compat::shape_type`), may hold 0 (the Views'
  /// length) or the length the Views already have along each axis (the real side's length
  /// for real transforms).
  ///
  /// Throws `std::invalid_argument` naming the argument: `dir` for a real-to-complex plan
  /// that is not forward or a complex-to-real plan that is not backward (as Kokkos-FFT);
  /// `s` (or `n`) for a length the Views do not have; and everything `make_plan` refuses.
  /// Throws `std::bad_alloc` when memory runs out. Never touches the Views' values.
  explicit view_plan(const ExecutionSpace& exec_space, const InView& in, const OutView& out,
                     direction dir, const axis_type<Dim>& axes,
                     const std::array<std::size_t, Dim>& s = {})
      : view_plan(exec_space, in, out, checked_shape(in, out, dir, axes, s)) {}

  /// The one-axis constructor with Kokkos-FFT's `compat::Direction`, the spelling
  /// `compat::Plan` call sites use: as the one taking `direction`, with `dir` converted.
  ///
  /// Throws `std::invalid_argument` naming `dir` when it is neither enumerator, and
  /// otherwise as the constructor taking `direction`.
  explicit view_plan(const ExecutionSpace& exec_space, const InView& in, const OutView& out,
                     compat::Direction dir, int axis, std::optional<std::size_t> n = std::nullopt)
    requires(Dim == 1)
      : view_plan(exec_space, in, out, detail::direction_of(dir), axis, n) {}

  /// The multi-axis constructor with Kokkos-FFT's `compat::Direction`, the spelling
  /// `compat::Plan` call sites use: as the one taking `direction`, with `dir` converted.
  ///
  /// Throws `std::invalid_argument` naming `dir` when it is neither enumerator, and
  /// otherwise as the constructor taking `direction`.
  explicit view_plan(const ExecutionSpace& exec_space, const InView& in, const OutView& out,
                     compat::Direction dir, const axis_type<Dim>& axes,
                     const std::array<std::size_t, Dim>& s = {})
      : view_plan(exec_space, in, out, detail::direction_of(dir), axes, s) {}

  /// Not copyable, as Kokkos-FFT's plan.
  view_plan(const view_plan&) = delete;
  /// Not copyable, as Kokkos-FFT's plan.
  view_plan& operator=(const view_plan&) = delete;
  /// Not movable, as Kokkos-FFT's plan.
  view_plan(view_plan&&) = delete;
  /// Not movable, as Kokkos-FFT's plan.
  view_plan& operator=(view_plan&&) = delete;
  /// Releases the FFTN plan and the workspace.
  ~view_plan() = default;

  /// Transforms `in` into `out`, then scales the output as `norm` says (Kokkos-FFT's
  /// default: backward transforms by 1/N).
  ///
  /// First fences the plan's execution-space instance, so that work queued on it before the
  /// call is complete; then transforms on the calling thread and returns when `out` holds
  /// the result. Scaling multiplies each real and imaginary part of the described output by
  /// a coefficient computed in `double` and rounded once to the precision of the data.
  ///
  /// Preconditions: not called inside a Kokkos kernel (it fences); a transform inside a
  /// kernel needs a core plan from `make_plan` and a workspace per thread.
  ///
  /// Throws `std::invalid_argument` naming `in` or `out` when their extents or strides
  /// differ from those the plan was built with, when they start at the same address and the
  /// plan is not in place or the reverse, or when an out-of-place plan is given overlapping
  /// Views. Throws `std::logic_error` when another execution of this plan is running: one plan
  /// runs one execution at a time, since its executions share its workspace. Allocates
  /// nothing unless it throws (the fence's label is built at construction; a loaded Kokkos
  /// Tools library may allocate in the fence).
  ///
  /// Guarantees: as for `fftn::plan::execute` (the input of an out-of-place plan is
  /// preserved; nothing outside the described output elements is written).
  void execute(const InView& in, const OutView& out,
               normalization norm = normalization::backward) const {
    check("in", in, in_extent_, in_stride_);
    check("out", out, out_extent_, out_stride_);
    if (detail::same_first_element(in, out) != in_place_) {
      detail::reject("in, out", in_place_ ? "must start at the same address: the plan is in place"
                                          : "must not start at the same address: the plan is "
                                            "out of place");
    }
    if (!in_place_ && detail::overlapping(in, out)) {
      detail::reject("out", "overlaps in, and the plan is out of place");
    }
    if (busy_.exchange(true, std::memory_order_acquire)) {
      throw std::logic_error("fftn::kokkos: view_plan::execute was called while another execution "
                             "of the same plan was running; one plan runs one execution at a "
                             "time (use a plan per thread, or make_plan and a workspace each)");
    }
    const detail::busy_guard release(busy_);
    exec_space_.fence(detail::fence_label());
    workspace scratch(workspace_.memory());
    plan_.execute(data_of(in), data_of(out), scratch);
    scale(out, norm);
  }

  /// The FFTN plan this plan executes.
  [[nodiscard]] const fftn::plan<in_element, out_element>& fftn_plan() const noexcept {
    return plan_;
  }

  /// The FFTN plan in human-readable form (`fftn::plan::describe`).
  ///
  /// Throws `std::bad_alloc` when memory runs out.
  [[nodiscard]] std::string describe() const { return plan_.describe(); }

private:
  view_plan(ExecutionSpace exec_space, const InView& in, const OutView& out,
            const shape<in_element, out_element>& s)
      : exec_space_(std::move(exec_space)),
        plan_(detail::plan_of(s, detail::same_first_element(in, out))),
        workspace_(plan_.workspace_bytes()), dir_(s.dir),
        in_place_(detail::same_first_element(in, out)) {
    static_cast<void>(detail::fence_label());
    for (const axis& a : s.axes) {
      points_ *= a.n;
    }
    for (std::size_t d = 0; d < rank; ++d) {
      in_extent_[d] = in.extent(d);
      out_extent_[d] = out.extent(d);
      in_stride_[d] = in.stride(d);
      out_stride_[d] = out.stride(d);
    }
  }

  static shape<in_element, out_element> checked_shape(const InView& in, const OutView& out,
                                                      direction dir, const axis_type<Dim>& axes,
                                                      const std::array<std::size_t, Dim>& s) {
    if (std::is_floating_point_v<in_element> && dir != direction::forward) {
      detail::reject("dir", "must be forward for a real-to-complex plan");
    }
    if (std::is_floating_point_v<out_element> && dir != direction::backward) {
      detail::reject("dir", "must be backward for a complex-to-real plan");
    }
    auto result = shape_of(in, out, dir, axes);
    for (std::size_t i = 0; i < Dim; ++i) {
      if (s[i] != 0 && s[i] != result.axes[i].n) {
        detail::reject((Dim == 1 ? std::string("n") : "s[" + std::to_string(i) + "]"),
                       "= " + std::to_string(s[i]) + " differs from the Views' length " +
                           std::to_string(result.axes[i].n) + ": FFTN does not crop or pad");
      }
    }
    return result;
  }

  template <class View>
  static void check(const char* name, const View& v, const std::array<std::size_t, rank>& extent,
                    const std::array<std::size_t, rank>& stride) {
    for (std::size_t d = 0; d < rank; ++d) {
      if (v.extent(d) != extent[d]) {
        detail::reject(detail::dim_name(name, d, "extent"), "= " + std::to_string(v.extent(d)) +
                                                                " differs from the plan's " +
                                                                std::to_string(extent[d]));
      }
      if (extent[d] > 1 && v.stride(d) != stride[d]) {
        detail::reject(detail::dim_name(name, d, "stride"), "= " + std::to_string(v.stride(d)) +
                                                                " differs from the plan's " +
                                                                std::to_string(stride[d]));
      }
    }
  }

  void scale(const OutView& out, normalization norm) const {
    const bool forward = dir_ == direction::forward;
    double coefficient = 1;
    if ((norm == normalization::forward && forward) ||
        (norm == normalization::backward && !forward)) {
      coefficient = 1.0 / static_cast<double>(points_);
    } else if (norm == normalization::ortho) {
      coefficient = 1.0 / std::sqrt(static_cast<double>(points_));
    } else {
      return;
    }
    using value = OutView::non_const_value_type;
    using T = detail::real_of<value>::type;
    const auto c = static_cast<T>(coefficient);
    // Every described element once, in any layout: an odometer over the dimensions.
    std::array<std::size_t, rank> index{};
    std::ptrdiff_t offset = 0;
    value* const p = out.data();
    for (std::size_t k = 0, total = out.size(); k < total; ++k) {
      value& x = p[offset];
      if constexpr (std::is_floating_point_v<value>) {
        x *= c;
      } else {
        x.real(x.real() * c);
        x.imag(x.imag() * c);
      }
      for (std::size_t d = rank; d-- > 0;) {
        const auto step = static_cast<std::ptrdiff_t>(out_stride_[d]);
        if (++index[d] < out_extent_[d]) {
          offset += step;
          break;
        }
        offset -= step * static_cast<std::ptrdiff_t>(out_extent_[d] - 1);
        index[d] = 0;
      }
    }
  }

  ExecutionSpace exec_space_;
  fftn::plan<in_element, out_element> plan_;
  // The scratch of the plan's executions, lent to each in turn through a borrowing
  // workspace; busy_ makes them take turns (execute is const, as Kokkos-FFT's).
  workspace workspace_;
  mutable std::atomic<bool> busy_{false};
  direction dir_;
  bool in_place_;
  std::size_t points_ = 1;
  std::array<std::size_t, rank> in_extent_{};
  std::array<std::size_t, rank> out_extent_{};
  std::array<std::size_t, rank> in_stride_{};
  std::array<std::size_t, rank> out_stride_{};
};

/// Executes `p` on `in` and `out` with normalisation `norm`, as Kokkos-FFT's free
/// `KokkosFFT::execute`: the same as `p.execute(in, out, norm)`.
///
/// Preconditions, exceptions and guarantees: those of `view_plan::execute`.
template <class ExecutionSpace, class InView, class OutView, std::size_t Dim>
void execute(const view_plan<ExecutionSpace, InView, OutView, Dim>& p,
             const std::type_identity_t<InView>& in, const std::type_identity_t<OutView>& out,
             normalization norm = normalization::backward) {
  p.execute(in, out, norm);
}

/// Kokkos-FFT's names for this adapter, so that code written against Kokkos-FFT 2.0 needs
/// only a new include and namespace: `KokkosFFT::Plan` becomes `fftn::kokkos::compat::Plan`,
/// and so on. The names break FFTN's snake_case rule on purpose; they are aliases of the
/// adapter's own, except `Direction` (declared above), whose values are Kokkos-FFT's.
namespace compat {

/// `KokkosFFT::Plan`: the adapter's `view_plan`.
template <class ExecutionSpace, class InViewType, class OutViewType, std::size_t Dim = 1>
using Plan = kokkos::view_plan<ExecutionSpace, InViewType, OutViewType, Dim>;

/// `KokkosFFT::Normalization`: the adapter's `normalization`.
using Normalization = kokkos::normalization;

/// `KokkosFFT::axis_type`: the adapter's `axis_type`.
template <std::size_t Dim> using axis_type = kokkos::axis_type<Dim>;

/// `KokkosFFT::shape_type`: the lengths of the transformed axes, which the plan
/// constructors take as `s`. FFTN accepts only the lengths the Views already have (0
/// standing for them): it does not crop or pad.
template <std::size_t Dim> using shape_type = std::array<std::size_t, Dim>;

/// `KokkosFFT::execute`: the adapter's `execute`.
using kokkos::execute;

} // namespace compat

} // namespace fftn::inline v0::kokkos
