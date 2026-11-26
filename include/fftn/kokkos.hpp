// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// The Kokkos adapter: FFTN transforms of host-accessible `Kokkos::View`s.
///
/// Optional and header-only. `fftn.hpp` does not include it, and it compiles only in a
/// translation unit that can include `Kokkos_Core.hpp`: FFTN itself never depends on Kokkos.
/// It offers three levels:
///
/// - View-based construction: `shape_of` and `make_plan` turn a pair of Views and the axes to
///   transform into an ordinary `fftn::shape` or `fftn::plan`, and `data_of` gives the
///   pointers to execute it with. Views of `LayoutRight`, `LayoutLeft` and `LayoutStride`
///   are read through their extents and strides.
/// - Threads: `fftn::kokkos::executor` runs a plan's work on an execution-space instance, one
///   `Kokkos::parallel_for` per transform, and `fftn::kokkos::team_of` makes the members of a
///   `Kokkos::TeamPolicy` kernel an `fftn::team`.
/// - `fftn::kokkos::view_plan`, a plan class with the constructor and execute shape of
///   Kokkos-FFT 2.0's `KokkosFFT::Plan`, threaded on its instance, and the free
///   `execute(plan, in, out, norm)`. `fftn::kokkos::compat` spells these as Kokkos-FFT does
///   (`Plan`, `Direction`, `Normalization`, `axis_type`, `shape_type`, `execute`), so code
///   written for Kokkos-FFT changes only its include and its namespace. It also has
///   Kokkos-FFT's one-shot functions (`fft`, `rfft2`, `irfftn` and the rest), which keep their
///   plans in the calling thread's convenience context (fftn/easy.hpp).
///
/// `executor`, `team_of`, the threaded `view_plan` and the one-shot functions are provisional:
/// their shape awaits the maintainer's review (dev/DESIGN.md 5.5). How the adapter differs from
/// Kokkos-FFT, and the rules for each argument, are in docs/kokkos.md.
#pragma once

#include "fftn/config.hpp"
#include "fftn/easy.hpp"
#include "fftn/executor.hpp"
#include "fftn/options.hpp"
#include "fftn/plan.hpp"
#include "fftn/shape.hpp"
#include "fftn/team.hpp"
#include "fftn/workspace.hpp"

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_OPENMP
#include <omp.h>
#endif

/// @def FFTN_KOKKOS_THREADS_SUPPORTED
/// 1 when plans and executors may run on `Kokkos::Threads`, else 0. Telling whether a thread is
/// inside one of its kernels takes Kokkos' internal names, and a wrong answer hangs or aborts,
/// so that back end is accepted only with the Kokkos release those names were checked against,
/// the one the tests build (cmake/FftnTestDeps.cmake), 5.2.1. With any other release a plan or
/// executor on it does not compile.
#if defined(KOKKOS_ENABLE_THREADS) && KOKKOS_VERSION == 50201
#define FFTN_KOKKOS_THREADS_SUPPORTED 1
#else
#define FFTN_KOKKOS_THREADS_SUPPORTED 0
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
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

/// `KokkosFFT::Normalization`, declared as Kokkos-FFT 2.0 declares it: the same names,
/// values and underlying type. The plan's `execute` takes it and converts it to the
/// adapter's `normalization`, whose enumerators mean the same.
// NOLINTNEXTLINE(performance-enum-size): int, as Kokkos-FFT's, so conversions agree with it
enum class Normalization : int {
  forward = 0,  ///< forward transforms are multiplied by 1/N, backward ones are not scaled
  backward = 1, ///< backward transforms are multiplied by 1/N, forward ones are not scaled
  ortho = 2,    ///< every transform is multiplied by 1/sqrt(N)
  none = 3,     ///< nothing is scaled
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

// Kokkos-FFT's normalisation as the adapter's. A value that is neither enumerator is
// refused rather than read as one of them.
inline normalization normalization_of(compat::Normalization norm) {
  switch (norm) {
  case compat::Normalization::forward:
    return normalization::forward;
  case compat::Normalization::backward:
    return normalization::backward;
  case compat::Normalization::ortho:
    return normalization::ortho;
  case compat::Normalization::none:
    return normalization::none;
  }
  reject("norm", "is none of Normalization's enumerators");
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

// The name of the Kokkos Tools region around a plan's execution; built as fence_label is.
inline const std::string& region_label() {
  static const std::string label("fftn::kokkos::view_plan::execute");
  return label;
}

// The label of the executor's parallel_for and of the fence after it. Kokkos::parallel_for
// copies its label into a std::string on every call, so this one must fit the small-string
// buffer of every standard library (15 characters in libstdc++ and Microsoft's, 22 in libc++),
// or each dispatch would allocate.
inline const std::string& bulk_label() {
  static const std::string label("fftn::bulk");
  return label;
}

template <class ExecutionSpace>
inline constexpr bool is_serial_space =
#ifdef KOKKOS_ENABLE_SERIAL
    std::is_same_v<ExecutionSpace, Kokkos::Serial>;
#else
    false;
#endif

// The back ends the adapter can run on: those whose kernels it can reason about. Another host
// back end (HPX) would pass every other check, and then fence or dispatch from inside its own
// kernels.
template <class ExecutionSpace> constexpr bool supported_space() {
#ifdef KOKKOS_ENABLE_OPENMP
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::OpenMP>) {
    return true;
  }
#endif
#if FFTN_KOKKOS_THREADS_SUPPORTED
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Threads>) {
    return true;
  }
#endif
  return is_serial_space<ExecutionSpace>;
}

// Whether the calling thread must not dispatch, fence or open a Kokkos Tools region: it is inside
// a kernel of the build's threaded host back end (or, on Threads, is not the one thread that may
// dispatch). Every enabled back end is asked, whichever the plan runs on, because a plan on
// Serial can be called from a kernel of the other one. Kokkos 5 has no query for this
// (ExecutionSpace::in_parallel is gone), and getting it wrong is fatal: a nested parallel_for
// aborts on OpenMP and throws on Threads, and a fence in a kernel waits for that kernel.
//
// A kernel of Kokkos::Serial cannot be detected. That is why nothing here fences, dispatches or
// opens a region for a plan on Serial, and why a plan on a threaded back end has the
// precondition of not being called from a Serial kernel.
[[nodiscard]] inline bool in_host_kernel() noexcept {
#ifdef KOKKOS_ENABLE_OPENMP
  // The level, not omp_in_parallel(): Kokkos also refuses to dispatch from inside an inactive
  // region (one of a single thread), which omp_in_parallel() does not report.
  if (omp_get_level() > 0) {
    return true;
  }
#endif
#if FFTN_KOKKOS_THREADS_SUPPORTED
  using internal = Kokkos::Impl::ThreadsInternal;
  if (!internal::is_process()) {
    return true;
  }
  // The dispatching thread is the pool's last rank, and its state is Active only while it runs
  // its share of a kernel. No public interface says so: see FFTN_KOKKOS_THREADS_SUPPORTED.
  internal* const self = internal::get_thread(Kokkos::Threads::impl_thread_pool_size(0) - 1);
  return self->state().load(std::memory_order_relaxed) == Kokkos::Impl::ThreadState::Active;
#else
  return false;
#endif
}

// What parallel_for runs for the executor: the functor of one bulk call.
struct bulk_call {
  task_function task;
  void* arg;
  void operator()(int i) const { task(arg, i); }
};

// The Kokkos Tools region around a plan's execution. Only outside every kernel: unless the tool
// opts out, Kokkos fences every execution space before it calls a region callback, which inside
// a kernel waits for that kernel for ever; a tool that opts out would have several threads
// pushing and popping one stack at once.
class region_guard {
public:
  explicit region_guard(bool active, const std::string& label = region_label()) : active_(active) {
    if (active_) {
      Kokkos::Profiling::pushRegion(label);
    }
  }
  region_guard(const region_guard&) = delete;
  region_guard& operator=(const region_guard&) = delete;
  region_guard(region_guard&&) = delete;
  region_guard& operator=(region_guard&&) = delete;
  ~region_guard() {
    if (active_) {
      Kokkos::Profiling::popRegion();
    }
  }

private:
  bool active_;
};

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

// Kokkos-FFT's normalisation of an output (view_plan::execute and the one-shot functions): every
// real and imaginary part of the described elements multiplied by the coefficient computed in
// double and rounded once to the data's precision, shared among the executor's lanes from 2^16
// elements (experiment T3), cut along the output's longest dimension. Built from the output
// View's extents and strides, so any layout works.
template <class Value, std::size_t Rank> class output_scaler {
public:
  using real = real_of<Value>::type;

  output_scaler() = default;

  template <class View> explicit output_scaler(const View& out) {
    for (std::size_t d = 0; d < Rank; ++d) {
      extent_[d] = out.extent(d);
      stride_[d] = out.stride(d);
      if (extent_[d] > extent_[split_]) {
        split_ = d;
      }
    }
  }

  // Scales the output at `first` as `norm` asks of a transform in direction `dir` of `points`
  // points, on `ex`'s lanes.
  template <class Executor>
  void scale(Value* first, normalization norm, direction dir, std::size_t points,
             const Executor& ex) const {
    const bool forward = dir == direction::forward;
    double coefficient = 1;
    if ((norm == normalization::forward && forward) ||
        (norm == normalization::backward && !forward)) {
      coefficient = 1.0 / static_cast<double>(points);
    } else if (norm == normalization::ortho) {
      coefficient = 1.0 / std::sqrt(static_cast<double>(points));
    } else {
      return;
    }
    const auto c = static_cast<real>(coefficient);
    std::size_t size = 1;
    for (const std::size_t e : extent_) {
      size *= e;
    }
    // Shared only when the multiplications outweigh a dispatch (experiment T3).
    int parts = 1;
    if (size >= share_elements) {
      const auto most =
          std::min<std::size_t>(extent_[split_], static_cast<std::size_t>(ex.concurrency()));
      parts = static_cast<int>(most);
    }
    if (parts <= 1) {
      scale_box(first, extent_, c);
      return;
    }
    job j{.self = this, .first = first, .coefficient = c, .parts = parts};
    ex.bulk(parts, &scale_part, &j);
  }

private:
  // One scaling of an output: its elements, cut along dimension split_ into `parts` ranges.
  struct job {
    const output_scaler* self;
    Value* first;
    real coefficient;
    int parts;
  };

  // Multiplies every element of the box of `extent` elements that starts at `first`, walking
  // it with an odometer over the dimensions.
  void scale_box(Value* first, const std::array<std::size_t, Rank>& extent, real c) const noexcept {
    std::size_t total = 1;
    for (const std::size_t e : extent) {
      total *= e;
    }
    std::array<std::size_t, Rank> index{};
    std::ptrdiff_t offset = 0;
    for (std::size_t k = 0; k < total; ++k) {
      Value& x = first[offset];
      if constexpr (std::is_floating_point_v<Value>) {
        x *= c;
      } else {
        x.real(x.real() * c);
        x.imag(x.imag() * c);
      }
      for (std::size_t d = Rank; d-- > 0;) {
        const auto step = static_cast<std::ptrdiff_t>(stride_[d]);
        if (++index[d] < extent[d]) {
          offset += step;
          break;
        }
        offset -= step * static_cast<std::ptrdiff_t>(extent[d] - 1);
        index[d] = 0;
      }
    }
  }

  static void scale_part(void* arg, int part) {
    const auto& j = *static_cast<const job*>(arg);
    const output_scaler& self = *j.self;
    const std::size_t length = self.extent_[self.split_];
    const auto parts = static_cast<std::size_t>(j.parts);
    const auto at = static_cast<std::size_t>(part);
    const std::size_t begin = length * at / parts;
    std::array<std::size_t, Rank> extent = self.extent_;
    extent[self.split_] = length * (at + 1) / parts - begin;
    const auto start = static_cast<std::ptrdiff_t>(begin * self.stride_[self.split_]);
    self.scale_box(j.first + start, extent, j.coefficient);
  }

  static constexpr std::size_t share_elements = std::size_t{1} << 16U;

  std::array<std::size_t, Rank> extent_{};
  std::array<std::size_t, Rank> stride_{};
  // The output dimension that scaling cuts among threads: the longest.
  std::size_t split_ = 0;
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
/// Every other dimension is a batch dimension; dimensions of extent 1 are dropped, the rest
/// are listed largest input stride first (ties in the Views' dimension order), and
/// neighbours that nest (one spans the other exactly, in both Views) are merged. That order
/// fixes how transforms are numbered, which `fftn::plan::execute_transforms` uses on a plan
/// from `make_plan`. Strides and distances are the Views' own, so any layout works.
///
/// Preconditions: the Views are of equal rank, at least `Dim`; their memory is accessible
/// from the host (checked at compile time); their values are `float`, `double` or a
/// complex of either, the pair forming one of FFTN's kinds (c2c, r2c, c2r) in one precision.
///
/// Throws `std::invalid_argument` naming the argument (`in`, `out`, `axes`) when an axis is
/// out of range or repeated; an extent is 0; a stride is 0; the extents do not match the
/// kind (equal everywhere, except along the halved axis of a real transform, where the
/// complex side has n/2 + 1 for the real side's n); `in` and `out` overlap without starting
/// at the same address; or, when they do start at the same address (in place), their
/// strides disagree: equal for c2c, and FFTW's padded layout for real kinds (the halved axis
/// contiguous on both sides, every other real stride twice the complex one).
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

  // One axis per dimension of the Views, in their own strides; the shape function does the rest
  // (the batch dimensions by stride, widest first, merged where they nest).
  std::array<axis, rank> array{};
  for (std::size_t d = 0; d < rank; ++d) {
    array[d] = axis{.n = extent[d], .in_stride = in_stride[d], .out_stride = out_stride[d]};
  }
  const std::size_t last_n = extent[halved];
  return fftn::detail::shape_along_axes<In, Out>(array, std::span<const std::size_t>(dims), dir,
                                                 r2c || !c2r ? last_n : last_n / 2 + 1,
                                                 r2c ? last_n / 2 + 1 : last_n);
}

/// An FFTN plan that transforms the dimensions `axes` of `in` into `out`: the plan of
/// `shape_of(in, out, dir, axes)`, built in place when `in` and `out` start at the same
/// address. Execute it with `data_of(in)` and `data_of(out)` (or any Views of the same
/// layout) and a workspace of its own per concurrent execution. Its transforms are numbered
/// over the batch dimensions in `shape_of`'s order (largest input stride first).
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

/// An executor (`fftn::executor`) that runs FFTN's tasks on a host execution-space instance,
/// for the executor overload of `plan::execute`. Provisional (dev/DESIGN.md 5.5).
///
/// A transform that is worth sharing costs one `Kokkos::parallel_for` on the instance, labelled
/// `fftn::bulk`, with one index per lane that has work, followed by a fence of the instance. A
/// job too small to share, and every job at width 1 (so every job on `Kokkos::Serial`), runs on
/// the calling thread without one (`plan::execute` decides that).
///
/// `ExecutionSpace` is `Kokkos::Serial`, `Kokkos::OpenMP` or `Kokkos::Threads`, the last only
/// with Kokkos 5.2.1, whose internals it relies on; anything else does not compile.
///
/// Inside a kernel of the OpenMP or Threads back end `concurrency()` is 1, so an execution there
/// runs serially on the calling thread instead of nesting; on `Kokkos::Threads` the same holds
/// on every thread but the one that may dispatch. Such an execution needs a workspace of its own
/// per thread.
///
/// Preconditions: Kokkos is initialised and not finalised while the executor is used. Construct
/// it outside kernels: inside one, the width it reads may be that of the enclosing region. An
/// executor on OpenMP or Threads is not used from inside a `Kokkos::Serial` kernel, which it
/// cannot detect: it would dispatch from there, and with a Kokkos Tools library loaded the global
/// fence before that dispatch never returns.
///
/// Guarantees: results are bitwise those of serial execution. FFTN allocates nothing and sets
/// no runtime option. A dispatch allocates nothing in Kokkos on the Serial and OpenMP back ends
/// unless a Kokkos Tools library is loaded; on `Kokkos::Threads` every `parallel_for` allocates
/// short-lived strings inside Kokkos. Dispatches take the instance's mutex on OpenMP, so those
/// from several host threads take turns. Copyable; holds the instance, its width and its
/// grouping, fixed at construction. Constructing the process's first executor may allocate, to
/// detect the machine's topology.
template <class ExecutionSpace = Kokkos::DefaultHostExecutionSpace> class executor {
  static_assert(Kokkos::is_execution_space_v<ExecutionSpace>,
                "fftn::kokkos::executor: ExecutionSpace must be a Kokkos execution space");
  static_assert(detail::supported_space<ExecutionSpace>(),
                "fftn::kokkos::executor: ExecutionSpace must be Kokkos::Serial, Kokkos::OpenMP or "
                "Kokkos::Threads (Threads needs Kokkos 5.2.1)");

public:
  /// An executor on `exec`, of `exec.concurrency()` lanes.
  explicit executor(const ExecutionSpace& exec = ExecutionSpace())
      : exec_(exec), width_(std::max(exec.concurrency(), 1)),
        group_size_(cache_group_size(width_)) {
    static_cast<void>(detail::bulk_label());
  }

  /// The most tasks it runs at once: the instance's width, or 1 when the calling thread cannot
  /// dispatch to the instance (inside a kernel).
  [[nodiscard]] int concurrency() const noexcept { return detail::in_host_kernel() ? 1 : width_; }

  /// The instance's width as read at construction, wherever it is called from: the lanes to
  /// size a workspace for (`plan::workspace_bytes`).
  [[nodiscard]] int width() const noexcept { return width_; }

  /// Lanes per cache group: `fftn::cache_group_size` of the width.
  [[nodiscard]] int group_size() const noexcept { return group_size_; }

  /// The execution-space instance it dispatches to.
  [[nodiscard]] const ExecutionSpace& space() const noexcept { return exec_; }

  /// Runs `task(arg, i)` once for every `i` in `[0, n)` and returns when all have returned: in
  /// one `parallel_for` on the instance, or on the calling thread for a single task, at width
  /// 1 and inside a kernel.
  ///
  /// Preconditions: `task` does not throw (an exception cannot leave a kernel).
  void bulk(int n, task_function task, void* arg) const {
    if (n <= 0) {
      return;
    }
    if (n == 1 || width_ == 1 || detail::in_host_kernel()) {
      for (int i = 0; i < n; ++i) {
        task(arg, i);
      }
      return;
    }
    using policy = Kokkos::RangePolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Static>,
                                       Kokkos::IndexType<int>>;
    // Chunks of one: with n up to the width, no thread takes two tasks while another idles.
    Kokkos::parallel_for(detail::bulk_label(), policy(exec_, 0, n).set_chunk_size(1),
                         detail::bulk_call{.task = task, .arg = arg});
    // Kokkos lets parallel_for return before the work is done; bulk may not. The host back
    // ends tested here have finished by now, and this costs them next to nothing.
    exec_.fence(detail::bulk_label());
  }

private:
  ExecutionSpace exec_;
  int width_;
  int group_size_;
};

/// The calling member of a `Kokkos::TeamPolicy` kernel as a member of an `fftn::team`: rank
/// `member.team_rank()`, size `member.team_size()`, `fftn::cache_group_size(size)` members per
/// cache group. Provisional (dev/DESIGN.md 5.5).
///
/// For the team overload of `plan::execute`, called by every member of the team from the
/// kernel's body with one workspace per team, of `plan.workspace_bytes(team_size)` bytes; a
/// kernel with several teams uses one workspace for each league rank. With `completion::own`,
/// `member.team_barrier()` is the synchronisation that the team overload then requires.
/// Not from inside `single`, `TeamThreadRange` or `TeamVectorRange`, where the members do not
/// each make the call once.
///
/// Guarantees: results are bitwise those of serial execution; allocates nothing once the
/// machine's topology is known (the first `fftn::cache_group_size` in the process detects it).
template <class TeamMember> [[nodiscard]] team team_of(const TeamMember& member) noexcept {
  const int size = member.team_size();
  return {.rank = member.team_rank(), .size = size, .group_size = cache_group_size(size)};
}

namespace detail {

// shape_of with Kokkos-FFT's further checks of a plan's arguments: real plans' directions (r2c
// forward, c2r backward) and the lengths `s` (0, or the Views' own: FFTN does not crop or pad).
template <class InView, class OutView, std::size_t Dim>
shape<element_t<InView>, element_t<OutView>>
checked_shape(const InView& in, const OutView& out, direction dir, const axis_type<Dim>& axes,
              const std::array<std::size_t, Dim>& s) {
  if (std::is_floating_point_v<element_t<InView>> && dir != direction::forward) {
    reject("dir", "must be forward for a real-to-complex plan");
  }
  if (std::is_floating_point_v<element_t<OutView>> && dir != direction::backward) {
    reject("dir", "must be backward for a complex-to-real plan");
  }
  auto result = shape_of(in, out, dir, axes);
  for (std::size_t i = 0; i < Dim; ++i) {
    if (s[i] != 0 && s[i] != result.axes[i].n) {
      reject((Dim == 1 ? std::string("n") : "s[" + std::to_string(i) + "]"),
             "= " + std::to_string(s[i]) + " differs from the Views' length " +
                 std::to_string(result.axes[i].n) + ": FFTN does not crop or pad");
    }
  }
  return result;
}

} // namespace detail

/// A plan with the constructor and execute shape of Kokkos-FFT 2.0's `KokkosFFT::Plan`, over
/// FFTN.
///
/// It transforms the dimensions `axes` of Views typed `InView` into Views typed `OutView`,
/// with the rules of `shape_of`, and owns everything an execution needs: the FFTN plan, an
/// `executor` on the plan's instance and a workspace for the instance's width, all made at
/// construction.
///
/// Differences from Kokkos-FFT (docs/kokkos.md has the full list):
/// - Execution is threaded through Kokkos on the plan's instance (`executor`), with results
///   bitwise equal to serial execution; a Kokkos-FFT host plan threads FFTW instead.
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
  static_assert(detail::supported_space<ExecutionSpace>(),
                "fftn::kokkos::view_plan: ExecutionSpace must be Kokkos::Serial, Kokkos::OpenMP "
                "or Kokkos::Threads (Threads needs Kokkos 5.2.1)");
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
      : view_plan(exec_space, in, out, detail::checked_shape(in, out, dir, axes, s)) {}

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
  /// On OpenMP and Threads it first fences the plan's execution-space instance, so that work
  /// queued on it before the call is complete, then transforms on the instance, in a Kokkos
  /// Tools region `fftn::kokkos::view_plan::execute`. A plan on `Kokkos::Serial` transforms on
  /// the calling thread with neither fence nor region: its kernels have finished when the thread
  /// that dispatched them gets here, and it may be inside one, which it could not tell. It
  /// returns when `out` holds the result. Scaling multiplies each real and imaginary part of the
  /// described output by a coefficient computed in `double` and rounded once to the precision
  /// of the data, and is shared among the instance's threads when the output is large.
  ///
  /// Inside a kernel of the OpenMP or Threads back end, on any plan, it neither fences, nor
  /// dispatches, nor opens a region: it runs serially on the calling thread. So does a plan on
  /// `Kokkos::Serial` inside any kernel. Executions on several threads at once still need one
  /// plan each.
  ///
  /// Preconditions: a plan on OpenMP or Threads is not executed from inside a `Kokkos::Serial`
  /// kernel, which it cannot detect (use a plan on `Kokkos::Serial` there).
  ///
  /// Throws `std::invalid_argument` naming `in` or `out` when their extents or strides
  /// differ from those the plan was built with, when they start at the same address and the
  /// plan is not in place or the reverse, or when an out-of-place plan is given overlapping
  /// Views. Throws `std::logic_error` when another execution of this plan is running: one plan
  /// runs one execution at a time, since its executions share its workspace. Inside a kernel of
  /// the OpenMP or Threads back end these exceptions cannot be caught outside the kernel: one
  /// that leaves the kernel's body ends the program (`std::terminate`).
  ///
  /// FFTN allocates nothing unless it throws (the labels are built at construction). Kokkos
  /// allocates nothing either on Serial and OpenMP, unless a Kokkos Tools library is loaded; on
  /// `Kokkos::Threads` it allocates short-lived strings in every `parallel_for`, so an execution
  /// that dispatches allocates through Kokkos.
  ///
  /// Guarantees: as for `fftn::plan::execute` (the input of an out-of-place plan is
  /// preserved; nothing outside the described output elements is written), with results
  /// bitwise equal to those of serial execution for every instance width.
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
    // A plan on Serial may be inside a Serial kernel, which nothing can detect, and a fence
    // there would wait for the caller's own kernel (it takes the mutex the kernel holds).
    const bool on_instance = !detail::is_serial_space<ExecutionSpace> && !detail::in_host_kernel();
    const detail::region_guard region(on_instance);
    if (on_instance) {
      executor_.space().fence(detail::fence_label());
    }
    plan_.execute(data_of(in), data_of(out), executor_, workspace_);
    scaler_.scale(out.data(), norm, dir_, points_, executor_);
  }

  /// `execute` with Kokkos-FFT's `compat::Normalization`, the spelling `compat::Plan` call
  /// sites use: as the one taking `normalization`, with `norm` converted.
  ///
  /// Throws `std::invalid_argument` naming `norm` when it is none of the enumerators, and
  /// otherwise as `execute` taking `normalization`.
  void execute(const InView& in, const OutView& out, compat::Normalization norm) const {
    execute(in, out, detail::normalization_of(norm));
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
  view_plan(const ExecutionSpace& exec_space, const InView& in, const OutView& out,
            const shape<in_element, out_element>& s)
      : executor_(exec_space), plan_(detail::plan_of(s, detail::same_first_element(in, out))),
        workspace_(plan_.workspace_bytes(executor_.width())), dir_(s.dir),
        in_place_(detail::same_first_element(in, out)), scaler_(out) {
    static_cast<void>(detail::fence_label());
    static_cast<void>(detail::region_label());
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

  executor<ExecutionSpace> executor_;
  fftn::plan<in_element, out_element> plan_;
  // The scratch of the plan's executions, for the instance's width; busy_ makes them take
  // turns (execute is const, as Kokkos-FFT's). One object throughout, so that the team state
  // it holds stays valid from one execution to the next.
  mutable workspace workspace_;
  mutable std::atomic<bool> busy_{false};
  direction dir_;
  bool in_place_;
  detail::output_scaler<typename OutView::non_const_value_type, rank> scaler_;
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

/// `execute(p, in, out, norm)` with Kokkos-FFT's `compat::Normalization`: the same as
/// `p.execute(in, out, norm)`.
///
/// Preconditions, exceptions and guarantees: those of `view_plan::execute`.
template <class ExecutionSpace, class InView, class OutView, std::size_t Dim>
void execute(const view_plan<ExecutionSpace, InView, OutView, Dim>& p,
             const std::type_identity_t<InView>& in, const std::type_identity_t<OutView>& out,
             compat::Normalization norm) {
  p.execute(in, out, norm);
}

namespace detail {

// Kokkos-FFT's hfft and ihfft run the other real transform with the normalisation's direction
// swapped.
inline normalization swapped(normalization norm) noexcept {
  switch (norm) {
  case normalization::forward:
    return normalization::backward;
  case normalization::backward:
    return normalization::forward;
  default:
    return norm;
  }
}

// The fence a one-shot function makes before its transform; built once, as fence_label is.
inline const std::string& one_shot_fence_label() {
  static const std::string label("fftn::kokkos::compat: fence before the transform");
  return label;
}

// A Kokkos View type of `Rank` dimensions of `Value`, C-ordered in host memory.
template <class Value, std::size_t Rank> struct host_view_of {
  using type = Kokkos::View<typename host_view_of<Value, Rank - 1>::type::data_type*,
                            Kokkos::LayoutRight, Kokkos::HostSpace>;
};
template <class Value> struct host_view_of<Value, 1> {
  using type = Kokkos::View<Value*, Kokkos::LayoutRight, Kokkos::HostSpace>;
};

// A C-ordered complex copy of `in`, conjugated when `conjugate` is set: the temporary through
// which Kokkos-FFT's fft of a real View, and its hfft, run. It allocates on every call, as
// Kokkos-FFT's temporaries do.
template <class InView, std::size_t... I>
auto complex_copy(const InView& in, bool conjugate, std::index_sequence<I...> /*dims*/) {
  using real = real_of<typename InView::non_const_value_type>::type;
  constexpr std::size_t rank = InView::rank();
  using copy_view = host_view_of<Kokkos::complex<real>, rank>::type;
  const copy_view copy(Kokkos::view_alloc(std::string("fftn::kokkos::compat: complex copy"),
                                          Kokkos::WithoutInitializing),
                       in.extent(I)...);
  const auto* from = data_of(in);
  auto* to = data_of(copy);
  std::size_t total = 1;
  for (std::size_t d = 0; d < rank; ++d) {
    total *= in.extent(d);
  }
  std::array<std::size_t, rank> index{};
  for (std::size_t k = 0; k < total; ++k) {
    std::ptrdiff_t at = 0;
    for (std::size_t d = 0; d < rank; ++d) {
      at += static_cast<std::ptrdiff_t>(index[d] * in.stride(d));
    }
    std::complex<real> z(from[at]);
    to[k] = conjugate ? std::conj(z) : z;
    for (std::size_t d = rank; d-- > 0;) {
      if (++index[d] < in.extent(d)) {
        break;
      }
      index[d] = 0;
    }
  }
  return copy;
}

// Negates the imaginary part of every element of a complex View: ihfft's last step.
template <class View> void conjugate_in_place(const View& v) {
  constexpr std::size_t rank = View::rank();
  auto* data = data_of(v);
  std::size_t total = 1;
  for (std::size_t d = 0; d < rank; ++d) {
    total *= v.extent(d);
  }
  std::array<std::size_t, rank> index{};
  for (std::size_t k = 0; k < total; ++k) {
    std::ptrdiff_t at = 0;
    for (std::size_t d = 0; d < rank; ++d) {
      at += static_cast<std::ptrdiff_t>(index[d] * v.stride(d));
    }
    data[at] = std::conj(data[at]);
    for (std::size_t d = rank; d-- > 0;) {
      if (++index[d] < v.extent(d)) {
        break;
      }
      index[d] = 0;
    }
  }
}

template <class In, class Out> constexpr easy::detail::easy_kind easy_kind_of() {
  if constexpr (std::is_floating_point_v<In>) {
    return easy::detail::easy_kind::r2c;
  } else if constexpr (std::is_floating_point_v<Out>) {
    return easy::detail::easy_kind::c2r;
  } else {
    return easy::detail::easy_kind::c2c;
  }
}

// A one-shot transform (the functions of compat below): what
// view_plan(exec, in, out, dir, axes, s).execute(in, out, norm) computes, bit for bit, with the
// plan and the workspace of the calling thread's convenience context (fftn::easy::thread_context,
// fftn/easy.hpp) instead of a plan's own, so a repeated call finds its plan there. The fence,
// the region (named `label`), the executor on `exec` and the normalisation are view_plan's.
template <class ExecutionSpace, class InView, class OutView, std::size_t Dim>
void one_shot(const std::string& label, const ExecutionSpace& exec, const InView& in,
              const OutView& out, direction dir, const axis_type<Dim>& axes,
              const std::array<std::size_t, Dim>& s, normalization norm) {
  static_assert(Kokkos::is_execution_space_v<ExecutionSpace>,
                "fftn::kokkos::compat: ExecutionSpace must be a Kokkos execution space");
  static_assert(supported_space<ExecutionSpace>(),
                "fftn::kokkos::compat: ExecutionSpace must be Kokkos::Serial, Kokkos::OpenMP or "
                "Kokkos::Threads (Threads needs Kokkos 5.2.1)");
  using In = element_t<InView>;
  using Out = element_t<OutView>;
  const shape<In, Out> sh = checked_shape(in, out, dir, axes, s);
  const bool in_place = same_first_element(in, out);
  const executor<ExecutionSpace> ex(exec);
  // As view_plan::execute: no fence or region on Serial, nor inside a kernel.
  const bool on_instance = !is_serial_space<ExecutionSpace> && !in_host_kernel();
  const region_guard region(on_instance, label);
  if (on_instance) {
    exec.fence(one_shot_fence_label());
  }
  const executor_ref ref(ex);
  try {
    easy::detail::run_shape(easy::thread_context(),
                            {.kind = easy_kind_of<In, Out>(),
                             .single = std::is_same_v<typename real_of<In>::type, float>,
                             .shape = &sh,
                             .in_place = in_place,
                             .in = data_of(in),
                             .out = data_of(out),
                             .ex = &ref});
  } catch (const std::invalid_argument& e) {
    reject("in, out", std::string("describe a layout FFTN refuses: ") + e.what());
  }
  std::size_t points = 1;
  for (const axis& a : sh.axes) {
    points *= a.n;
  }
  const output_scaler<typename OutView::non_const_value_type, OutView::rank()> scaler(out);
  scaler.scale(out.data(), norm, dir, points, ex);
}

// The last Dim dimensions, Kokkos-FFT's default axes of fftn and the rest: -Dim, ..., -1.
template <std::size_t Dim> constexpr axis_type<Dim> last_axes() {
  axis_type<Dim> a{};
  for (std::size_t i = 0; i < Dim; ++i) {
    a[i] = static_cast<int>(i) - static_cast<int>(Dim);
  }
  return a;
}

template <class View> constexpr bool real_view = std::is_floating_point_v<element_t<View>>;

} // namespace detail

/// Kokkos-FFT's names for this adapter, so that code written against Kokkos-FFT 2.0 needs
/// only a new include and namespace: `KokkosFFT::Plan` becomes `fftn::kokkos::compat::Plan`,
/// and so on. The names break FFTN's snake_case rule on purpose; they are aliases of the
/// adapter's own, except `Direction` and `Normalization` (declared above), which are
/// Kokkos-FFT's own enumerations.
namespace compat {

/// `KokkosFFT::Plan`: the adapter's `view_plan`.
template <class ExecutionSpace, class InViewType, class OutViewType, std::size_t Dim = 1>
using Plan = kokkos::view_plan<ExecutionSpace, InViewType, OutViewType, Dim>;

/// `KokkosFFT::axis_type`: the adapter's `axis_type`.
template <std::size_t Dim> using axis_type = kokkos::axis_type<Dim>;

/// `KokkosFFT::shape_type`: the lengths of the transformed axes, which the plan
/// constructors take as `s`. FFTN accepts only the lengths the Views already have (0
/// standing for them): it does not crop or pad.
template <std::size_t Dim> using shape_type = std::array<std::size_t, Dim>;

/// `KokkosFFT::execute`: the adapter's `execute`.
using kokkos::execute;

// The one-shot functions (step 6.7, provisional): Kokkos-FFT 2.0's `KokkosFFT::fft` and the rest,
// with its signatures. Each computes what a plan of the same Views computes (`Plan` built for the
// call, then `execute` with `norm`), bit for bit, but keeps the plan and its workspace in the
// calling thread's convenience context (`fftn::easy::thread_context()`, fftn/easy.hpp), so a
// call that repeats a layout builds nothing; the first call of a layout allocates. Each runs on
// `exec_space` as `Plan::execute` does (the fence, the Kokkos Tools region, named after the
// function, and the threaded execution), and refuses what `Plan` refuses, with the same messages.
// Differences from Kokkos-FFT are view_plan's (docs/kokkos.md), and `hfft` leaves its input
// unchanged (Kokkos-FFT conjugates a non-const input in place).

/// `KokkosFFT::fft`: the forward complex transform along dimension `axis` of `in` into `out`,
/// normalised as `norm` says. A real `in` is first copied into a complex temporary (allocated on
/// every call, as Kokkos-FFT does). `n` may only be the length the Views have.
///
/// Throws `std::invalid_argument` as `Plan`'s constructor and `execute` do. Provisional.
template <class ExecutionSpace, class InViewType, class OutViewType>
void fft(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
         Normalization norm = Normalization::backward, int axis = -1,
         std::optional<std::size_t> n = std::nullopt) {
  static_assert(!detail::real_view<OutViewType>, "fftn::kokkos::compat::fft: out must be complex");
  static const std::string label("fftn::kokkos::compat::fft");
  if constexpr (detail::real_view<InViewType>) {
    const auto copy =
        detail::complex_copy(in, false, std::make_index_sequence<InViewType::rank()>{});
    detail::one_shot(label, exec_space, copy, out, direction::forward, axis_type<1>{axis},
                     std::array<std::size_t, 1>{n.value_or(0)}, detail::normalization_of(norm));
  } else {
    detail::one_shot(label, exec_space, in, out, direction::forward, axis_type<1>{axis},
                     std::array<std::size_t, 1>{n.value_or(0)}, detail::normalization_of(norm));
  }
}

/// `KokkosFFT::ifft`: the backward complex transform along `axis`. As `fft` otherwise; the
/// input is complex.
template <class ExecutionSpace, class InViewType, class OutViewType>
void ifft(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
          Normalization norm = Normalization::backward, int axis = -1,
          std::optional<std::size_t> n = std::nullopt) {
  static_assert(!detail::real_view<InViewType> && !detail::real_view<OutViewType>,
                "fftn::kokkos::compat::ifft: in and out must be complex");
  static const std::string label("fftn::kokkos::compat::ifft");
  detail::one_shot(label, exec_space, in, out, direction::backward, axis_type<1>{axis},
                   std::array<std::size_t, 1>{n.value_or(0)}, detail::normalization_of(norm));
}

/// `KokkosFFT::rfft`: the real-to-complex transform along `axis` (the output has n/2 + 1
/// values along it). As `fft` otherwise.
template <class ExecutionSpace, class InViewType, class OutViewType>
void rfft(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
          Normalization norm = Normalization::backward, int axis = -1,
          std::optional<std::size_t> n = std::nullopt) {
  static_assert(detail::real_view<InViewType> && !detail::real_view<OutViewType>,
                "fftn::kokkos::compat::rfft: in must be real and out complex");
  static const std::string label("fftn::kokkos::compat::rfft");
  detail::one_shot(label, exec_space, in, out, direction::forward, axis_type<1>{axis},
                   std::array<std::size_t, 1>{n.value_or(0)}, detail::normalization_of(norm));
}

/// `KokkosFFT::irfft`: the complex-to-real transform along `axis`, its length the real
/// output's. As `fft` otherwise.
template <class ExecutionSpace, class InViewType, class OutViewType>
void irfft(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
           Normalization norm = Normalization::backward, int axis = -1,
           std::optional<std::size_t> n = std::nullopt) {
  static_assert(!detail::real_view<InViewType> && detail::real_view<OutViewType>,
                "fftn::kokkos::compat::irfft: in must be complex and out real");
  static const std::string label("fftn::kokkos::compat::irfft");
  detail::one_shot(label, exec_space, in, out, direction::backward, axis_type<1>{axis},
                   std::array<std::size_t, 1>{n.value_or(0)}, detail::normalization_of(norm));
}

/// `KokkosFFT::hfft`: the transform of a Hermitian-symmetric signal given by its first half
/// (complex `in`, or real), real `out`: `irfft` of the conjugated input with the
/// normalisation's direction swapped, as Kokkos-FFT computes it. The conjugate is a temporary
/// (allocated on every call); `in` is not modified.
template <class ExecutionSpace, class InViewType, class OutViewType>
void hfft(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
          Normalization norm = Normalization::backward, int axis = -1,
          std::optional<std::size_t> n = std::nullopt) {
  static_assert(detail::real_view<OutViewType>, "fftn::kokkos::compat::hfft: out must be real");
  static const std::string label("fftn::kokkos::compat::hfft");
  const auto copy = detail::complex_copy(in, true, std::make_index_sequence<InViewType::rank()>{});
  detail::one_shot(label, exec_space, copy, out, direction::backward, axis_type<1>{axis},
                   std::array<std::size_t, 1>{n.value_or(0)},
                   detail::swapped(detail::normalization_of(norm)));
}

/// `KokkosFFT::ihfft`: the inverse of `hfft`, real `in`, complex `out`: `rfft` with the
/// normalisation's direction swapped, then the output conjugated in place, as Kokkos-FFT.
template <class ExecutionSpace, class InViewType, class OutViewType>
void ihfft(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
           Normalization norm = Normalization::backward, int axis = -1,
           std::optional<std::size_t> n = std::nullopt) {
  static_assert(detail::real_view<InViewType> && !detail::real_view<OutViewType>,
                "fftn::kokkos::compat::ihfft: in must be real and out complex");
  static const std::string label("fftn::kokkos::compat::ihfft");
  detail::one_shot(label, exec_space, in, out, direction::forward, axis_type<1>{axis},
                   std::array<std::size_t, 1>{n.value_or(0)},
                   detail::swapped(detail::normalization_of(norm)));
  detail::conjugate_in_place(out);
}

/// `KokkosFFT::fft2`: `fftn` along `axes` (the last two dimensions by default), with Kokkos-FFT's
/// argument order (norm before axes). A real `in` is copied into a complex temporary.
template <class ExecutionSpace, class InViewType, class OutViewType>
void fft2(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
          Normalization norm = Normalization::backward, axis_type<2> axes = {-2, -1},
          shape_type<2> s = {}) {
  static_assert(!detail::real_view<OutViewType>, "fftn::kokkos::compat::fft2: out must be complex");
  static const std::string label("fftn::kokkos::compat::fft2");
  if constexpr (detail::real_view<InViewType>) {
    const auto copy =
        detail::complex_copy(in, false, std::make_index_sequence<InViewType::rank()>{});
    detail::one_shot(label, exec_space, copy, out, direction::forward, axes, s,
                     detail::normalization_of(norm));
  } else {
    detail::one_shot(label, exec_space, in, out, direction::forward, axes, s,
                     detail::normalization_of(norm));
  }
}

/// `KokkosFFT::ifft2`: `ifftn` along `axes` (the last two by default).
template <class ExecutionSpace, class InViewType, class OutViewType>
void ifft2(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
           Normalization norm = Normalization::backward, axis_type<2> axes = {-2, -1},
           shape_type<2> s = {}) {
  static_assert(!detail::real_view<InViewType> && !detail::real_view<OutViewType>,
                "fftn::kokkos::compat::ifft2: in and out must be complex");
  static const std::string label("fftn::kokkos::compat::ifft2");
  detail::one_shot(label, exec_space, in, out, direction::backward, axes, s,
                   detail::normalization_of(norm));
}

/// `KokkosFFT::rfft2`: `rfftn` along `axes` (the last two by default), the last listed halved.
template <class ExecutionSpace, class InViewType, class OutViewType>
void rfft2(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
           Normalization norm = Normalization::backward, axis_type<2> axes = {-2, -1},
           shape_type<2> s = {}) {
  static_assert(detail::real_view<InViewType> && !detail::real_view<OutViewType>,
                "fftn::kokkos::compat::rfft2: in must be real and out complex");
  static const std::string label("fftn::kokkos::compat::rfft2");
  detail::one_shot(label, exec_space, in, out, direction::forward, axes, s,
                   detail::normalization_of(norm));
}

/// `KokkosFFT::irfft2`: `irfftn` along `axes` (the last two by default).
template <class ExecutionSpace, class InViewType, class OutViewType>
void irfft2(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
            Normalization norm = Normalization::backward, axis_type<2> axes = {-2, -1},
            shape_type<2> s = {}) {
  static_assert(!detail::real_view<InViewType> && detail::real_view<OutViewType>,
                "fftn::kokkos::compat::irfft2: in must be complex and out real");
  static const std::string label("fftn::kokkos::compat::irfft2");
  detail::one_shot(label, exec_space, in, out, direction::backward, axes, s,
                   detail::normalization_of(norm));
}

/// `KokkosFFT::fftn`: the forward complex transform along `axes` (by default the last `DIM`
/// dimensions, `DIM` the Views' rank), with Kokkos-FFT's argument order (axes before norm). A
/// real `in` is copied into a complex temporary. As `fft` otherwise.
template <class ExecutionSpace, class InViewType, class OutViewType,
          std::size_t DIM = InViewType::rank()>
void fftn(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
          axis_type<DIM> axes = detail::last_axes<DIM>(),
          Normalization norm = Normalization::backward, shape_type<DIM> s = {}) {
  static_assert(!detail::real_view<OutViewType>, "fftn::kokkos::compat::fftn: out must be complex");
  static const std::string label("fftn::kokkos::compat::fftn");
  if constexpr (detail::real_view<InViewType>) {
    const auto copy =
        detail::complex_copy(in, false, std::make_index_sequence<InViewType::rank()>{});
    detail::one_shot(label, exec_space, copy, out, direction::forward, axes, s,
                     detail::normalization_of(norm));
  } else {
    detail::one_shot(label, exec_space, in, out, direction::forward, axes, s,
                     detail::normalization_of(norm));
  }
}

/// `KokkosFFT::ifftn`: the backward complex transform along `axes`. As `fftn` otherwise.
template <class ExecutionSpace, class InViewType, class OutViewType,
          std::size_t DIM = InViewType::rank()>
void ifftn(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
           axis_type<DIM> axes = detail::last_axes<DIM>(),
           Normalization norm = Normalization::backward, shape_type<DIM> s = {}) {
  static_assert(!detail::real_view<InViewType> && !detail::real_view<OutViewType>,
                "fftn::kokkos::compat::ifftn: in and out must be complex");
  static const std::string label("fftn::kokkos::compat::ifftn");
  detail::one_shot(label, exec_space, in, out, direction::backward, axes, s,
                   detail::normalization_of(norm));
}

/// `KokkosFFT::rfftn`: the real-to-complex transform along `axes`, the last listed halved.
template <class ExecutionSpace, class InViewType, class OutViewType,
          std::size_t DIM = InViewType::rank()>
void rfftn(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
           axis_type<DIM> axes = detail::last_axes<DIM>(),
           Normalization norm = Normalization::backward, shape_type<DIM> s = {}) {
  static_assert(detail::real_view<InViewType> && !detail::real_view<OutViewType>,
                "fftn::kokkos::compat::rfftn: in must be real and out complex");
  static const std::string label("fftn::kokkos::compat::rfftn");
  detail::one_shot(label, exec_space, in, out, direction::forward, axes, s,
                   detail::normalization_of(norm));
}

/// `KokkosFFT::irfftn`: the complex-to-real transform along `axes`, the last listed halved.
template <class ExecutionSpace, class InViewType, class OutViewType,
          std::size_t DIM = InViewType::rank()>
void irfftn(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out,
            axis_type<DIM> axes = detail::last_axes<DIM>(),
            Normalization norm = Normalization::backward, shape_type<DIM> s = {}) {
  static_assert(!detail::real_view<InViewType> && detail::real_view<OutViewType>,
                "fftn::kokkos::compat::irfftn: in must be complex and out real");
  static const std::string label("fftn::kokkos::compat::irfftn");
  detail::one_shot(label, exec_space, in, out, direction::backward, axes, s,
                   detail::normalization_of(norm));
}

} // namespace compat

} // namespace fftn::inline v0::kokkos
