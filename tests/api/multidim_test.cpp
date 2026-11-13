// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Transforms of rank 2 and 3 beyond the c2c planes of the oracle suites: real planes and
// volumes (r2c, c2r) and complex volumes against the separable quadruple-precision
// reference, with the default layouts and with strided ones (gaps, padding, reversed axes
// and distances), in and out of place. Strided runs go through the harness, which places
// each array against a guard page, checks that no byte outside the described elements is
// written and, out of place, that the input is unchanged.

#include "reference/reference_dft.hpp"
#include "support/canary.hpp"
#include "support/exact.hpp"
#include "support/harness.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"
#include "support/transform_under_test.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using cd = std::complex<double>;
using dims_t = std::vector<std::size_t>;
using strides_t = std::vector<std::ptrdiff_t>;

std::size_t product(const dims_t& d, std::size_t from = 0) {
  std::size_t p = 1;
  for (std::size_t i = from; i < d.size(); ++i) {
    p *= d[i];
  }
  return p;
}

// The unnormalised DFT of one C-ordered array, axis by axis, rounded to double after each
// axis.
std::vector<cd> dft(int sign, const dims_t& dims, std::vector<cd> x) {
  for (std::size_t a = 0; a < dims.size(); ++a) {
    const std::size_t n = dims[a];
    const std::size_t inner = product(dims, a + 1);
    const std::size_t outer = x.size() / (n * inner);
    std::vector<cd> line(n);
    for (std::size_t o = 0; o < outer; ++o) {
      for (std::size_t i = 0; i < inner; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          line[j] = x[(o * n + j) * inner + i];
        }
        const auto y = fftn::test::reference::c2c(sign, line);
        for (std::size_t k = 0; k < n; ++k) {
          x[(o * n + k) * inner + i] = {y[k].re.hi, y[k].im.hi};
        }
      }
    }
  }
  return x;
}

// The tolerance of a transform of these dimensions against dft() (tolerance.hpp):
// - the transform under test: derived_tolerance for all its points, since the errors of
//   the axes add in quadrature and log2 of the point count is the sum over the axes;
// - the expected values: rounding to double after each axis is an input rounding of the
//   remaining axes' exact DFT, at most input_rounding<double>() per axis;
// - for c2r only, the input: the spectrum is that of real data rounded to T, an input
//   rounding of the transform under test (input_rounding<T>()). The other kinds' inputs
//   are exact in T and in double.
template <class T>
fftn::test::tolerance tolerance_for(const dims_t& dims, bool input_rounded_to_t) {
  fftn::test::tolerance t =
      fftn::test::derived_tolerance<T>(product(dims), fftn::test::stable_fft_bound) +
      fftn::test::input_rounding<double>() * static_cast<double>(dims.size());
  if (input_rounded_to_t) {
    t = t + fftn::test::input_rounding<T>();
  }
  return t;
}

template <class T> std::vector<fftn::test::exact_complex> expected(std::span<const cd> v) {
  std::vector<fftn::test::exact_complex> e;
  e.reserve(v.size());
  for (const cd& z : v) {
    e.push_back(fftn::test::exact_from(z));
  }
  return e;
}

// Checks one result; the worst error/tolerance ratio of the test is printed at its end.
class checker {
public:
  checker() = default;
  checker(const checker&) = delete;
  checker& operator=(const checker&) = delete;
  checker(checker&&) = delete;
  checker& operator=(checker&&) = delete;
  ~checker() {
    std::cout << "[ worst    ] error/tolerance " << o_.worst_ratio() << " (" << worst_at_ << ")\n";
  }

  template <class T>
  void within(std::span<const std::complex<T>> got, std::span<const cd> want,
              const fftn::test::tolerance& tol, const std::string& what) {
    fftn::test::outcome o;
    o.within(fftn::test::compare<T>(got, expected<T>(want)), tol, what);
    EXPECT_TRUE(o.passed()) << o.detail();
    if (o.worst_ratio() > o_.worst_ratio()) {
      worst_at_ = what;
    }
    o_.merge(o);
  }

private:
  fftn::test::outcome o_;
  std::string worst_at_ = "-";
};

std::string describe(const dims_t& dims, const char* what) {
  std::string s = what;
  s += ", ";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    s += (i == 0 ? "" : " x ") + std::to_string(dims[i]);
  }
  return s;
}

constexpr std::size_t count = 2;

std::vector<fftn::axis> axes_of(const dims_t& dims) {
  std::vector<fftn::axis> a;
  for (const std::size_t n : dims) {
    a.push_back({.n = n});
  }
  return a;
}

const std::vector<dims_t>& shapes() {
  static const std::vector<dims_t> v{
      {8, 6}, {5, 7},    {16, 16},  {32, 30},  {1, 9},    {9, 1},
      {6, 1}, {4, 6, 8}, {3, 5, 4}, {2, 3, 5}, {1, 4, 1}, {7, 1, 3},
  };
  return v;
}

// The extents of the complex side of a real transform: the last axis halved.
dims_t halved(dims_t dims) {
  dims.back() = dims.back() / 2 + 1;
  return dims;
}

// Logical inputs and expected outputs of `count` transforms, transform after transform,
// each in C order.
struct real_case {
  std::vector<cd> reals;    // the real data, as complex values with zero imaginary parts
  std::vector<cd> spectrum; // its half spectrum, n/2 + 1 values along the last axis
};

template <class T> real_case real_data(const dims_t& dims, std::uint64_t seed) {
  const std::size_t n = dims.back();
  const std::size_t h = n / 2 + 1;
  const std::size_t points = product(dims);
  const std::size_t lines = points / n;
  fftn::test::input_source src(seed);
  const auto x = src.reals<T>(points * count);
  real_case c;
  c.reals.assign(x.begin(), x.end());
  for (std::size_t t = 0; t < count; ++t) {
    const std::vector<cd> one(c.reals.begin() + static_cast<std::ptrdiff_t>(t * points),
                              c.reals.begin() + static_cast<std::ptrdiff_t>((t + 1) * points));
    const auto y = dft(-1, dims, one);
    for (std::size_t l = 0; l < lines; ++l) {
      c.spectrum.insert(c.spectrum.end(), y.begin() + static_cast<std::ptrdiff_t>(l * n),
                        y.begin() + static_cast<std::ptrdiff_t>(l * n + h));
    }
  }
  return c;
}

// The real parts of v, in T: the real data of a real_case, exact in T.
template <class T> std::vector<T> real_parts(const std::vector<cd>& v) {
  std::vector<T> r;
  r.reserve(v.size());
  for (const cd& z : v) {
    r.push_back(static_cast<T>(z.real()));
  }
  return r;
}

// v times `factor`: what c2r returns from the spectrum of real data (no normalisation).
std::vector<cd> scaled(const std::vector<cd>& v, std::size_t factor) {
  std::vector<cd> r;
  r.reserve(v.size());
  for (const cd& z : v) {
    r.push_back(z * static_cast<double>(factor));
  }
  return r;
}

template <class T> class MultiDim : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(MultiDim, precisions);

// ---- Default layouts.

TYPED_TEST(MultiDim, ComplexVolumesAndPlanes) {
  using T = TypeParam;
  using c = std::complex<T>;
  checker check;
  for (const dims_t& dims : shapes()) {
    const std::size_t points = product(dims);
    fftn::test::input_source src(points);
    const auto x = src.complexes<T>(points * count);
    for (const fftn::direction dir : {fftn::direction::forward, fftn::direction::backward}) {
      std::vector<cd> want;
      for (std::size_t t = 0; t < count; ++t) {
        const std::vector<cd> one(x.begin() + static_cast<std::ptrdiff_t>(t * points),
                                  x.begin() + static_cast<std::ptrdiff_t>((t + 1) * points));
        const auto y = dft(static_cast<int>(dir), dims, one);
        want.insert(want.end(), y.begin(), y.end());
      }
      const fftn::shape<c, c> s{
          .dir = dir,
          .axes = axes_of(dims),
          .batches = {{.count = count}},
      };
      for (const bool in_place : {false, true}) {
        const fftn::c2c_plan<T> p(s, {.in_place = in_place});
        fftn::workspace ws(p);
        std::vector<c> y(x.size());
        if (in_place) {
          y = x;
          p.execute(y.data(), y.data(), ws);
        } else {
          p.execute(x.data(), y.data(), ws);
        }
        check.within<T>(y, want, tolerance_for<T>(dims, false),
                        describe(dims, in_place ? "c2c in place" : "c2c"));
      }
    }
  }
}

TYPED_TEST(MultiDim, RealToComplex) {
  using T = TypeParam;
  using c = std::complex<T>;
  checker check;
  for (const dims_t& dims : shapes()) {
    const std::size_t n = dims.back();
    const std::size_t h = n / 2 + 1;
    const std::size_t lines = product(dims) / n;
    const real_case data = real_data<T>(dims, product(dims) + 1);
    const std::vector<T> x = real_parts<T>(data.reals);
    const fftn::shape<T, c> s{.axes = axes_of(dims), .batches = {{.count = count}}};
    const auto tol = tolerance_for<T>(dims, false);
    {
      const fftn::r2c_plan<T> p(s);
      fftn::workspace ws(p);
      std::vector<c> y(lines * h * count);
      p.execute(x.data(), y.data(), ws);
      check.within<T>(y, data.spectrum, tol, describe(dims, "r2c"));
    }
    {
      // In place: each real row padded to the 2h reals of its complex row.
      const fftn::r2c_plan<T> p(s, {.in_place = true});
      fftn::workspace ws(p);
      std::vector<c> y(lines * h * count);
      auto* real = reinterpret_cast<T*>(y.data());
      for (std::size_t l = 0; l < lines * count; ++l) {
        std::memcpy(real + l * 2 * h, x.data() + l * n, n * sizeof(T));
      }
      p.execute(real, y.data(), ws);
      check.within<T>(y, data.spectrum, tol, describe(dims, "r2c in place"));
    }
  }
}

TYPED_TEST(MultiDim, ComplexToReal) {
  using T = TypeParam;
  using c = std::complex<T>;
  checker check;
  for (const dims_t& dims : shapes()) {
    const std::size_t n = dims.back();
    const std::size_t h = n / 2 + 1;
    const std::size_t points = product(dims);
    const std::size_t lines = points / n;
    // A consistent half spectrum: that of real data, so c2r returns points * x.
    const real_case data = real_data<T>(dims, points + 2);
    std::vector<c> spectrum(data.spectrum.begin(), data.spectrum.end());
    const std::vector<cd> want = scaled(data.reals, points);
    const fftn::shape<c, T> s{.axes = axes_of(dims), .batches = {{.count = count}}};
    const auto tol = tolerance_for<T>(dims, true);
    const auto as_complex = [](const std::vector<T>& v) {
      return std::vector<c>(v.begin(), v.end());
    };
    {
      const fftn::c2r_plan<T> p(s);
      fftn::workspace ws(p);
      const std::vector<c> before = spectrum;
      std::vector<T> y(points * count);
      p.execute(spectrum.data(), y.data(), ws);
      EXPECT_TRUE(std::ranges::equal(std::as_bytes(std::span<const c>(before)),
                                     std::as_bytes(std::span<const c>(spectrum))))
          << "c2r out of place changed its input";
      check.within<T>(as_complex(y), want, tol, describe(dims, "c2r"));
    }
    {
      const fftn::c2r_plan<T> p(s, {.in_place = true});
      fftn::workspace ws(p);
      std::vector<c> buffer = spectrum;
      auto* real = reinterpret_cast<T*>(buffer.data());
      p.execute(buffer.data(), real, ws);
      std::vector<T> y;
      for (std::size_t l = 0; l < lines * count; ++l) {
        y.insert(y.end(), real + l * 2 * h, real + l * 2 * h + n);
      }
      check.within<T>(as_complex(y), want, tol, describe(dims, "c2r in place"));
    }
  }
}

// ---- Strided layouts, through the harness.

// One side of a layout: a stride per axis and the distance between transforms.
struct side_layout {
  strides_t stride;
  std::ptrdiff_t dist = 0;
};

// Axes nested in C order with `pad` unused elements after each axis's reach, the last axis
// at stride `last`. signs[d] < 0 reverses axis d; signs.back() reverses the transforms.
side_layout nested(const dims_t& extent, std::ptrdiff_t last, std::ptrdiff_t pad,
                   const std::vector<int>& signs) {
  side_layout s;
  s.stride.assign(extent.size(), 0);
  std::ptrdiff_t reach = 0; // of the axes inside, in elements
  for (std::size_t i = extent.size(); i > 0; --i) {
    const std::size_t d = i - 1;
    const std::ptrdiff_t last_magnitude = last < 0 ? -last : last;
    const std::ptrdiff_t magnitude = d + 1 == extent.size() ? last_magnitude : reach + 1 + pad;
    s.stride[d] = signs[d] < 0 ? -magnitude : magnitude;
    reach += static_cast<std::ptrdiff_t>(extent[d] - 1) * magnitude;
  }
  s.dist = (signs.back() < 0 ? -1 : 1) * (reach + 1 + pad);
  return s;
}

// The side of an in-place real transform's real data that shares `complex`'s memory:
// FFTW's padded layout, the last axis contiguous and every other stride and the distance
// twice the complex side's.
side_layout real_twin(const side_layout& complex) {
  side_layout s = complex;
  for (std::size_t d = 0; d + 1 < s.stride.size(); ++d) {
    s.stride[d] *= 2;
  }
  s.dist *= 2;
  return s;
}

// Element offsets of a side in logical order: transform after transform, C order within.
std::vector<std::ptrdiff_t> offsets_of(const dims_t& extent, const side_layout& s) {
  std::vector<std::ptrdiff_t> v;
  const std::size_t points = product(extent);
  v.reserve(points * count);
  for (std::size_t t = 0; t < count; ++t) {
    for (std::size_t i = 0; i < points; ++i) {
      std::ptrdiff_t off = static_cast<std::ptrdiff_t>(t) * s.dist;
      std::size_t rest = i;
      for (std::size_t a = extent.size(); a > 0; --a) {
        off += static_cast<std::ptrdiff_t>(rest % extent[a - 1]) * s.stride[a - 1];
        rest /= extent[a - 1];
      }
      v.push_back(off);
    }
  }
  return v;
}

template <class In, class Out>
fftn::shape<In, Out> shape_of(const dims_t& dims, const side_layout& in, const side_layout& out,
                              fftn::direction dir = fftn::direction::forward) {
  fftn::shape<In, Out> s;
  s.dir = dir;
  for (std::size_t d = 0; d < dims.size(); ++d) {
    s.axes.push_back({.n = dims[d], .in_stride = in.stride[d], .out_stride = out.stride[d]});
  }
  s.batches.push_back({.count = count, .in_dist = in.dist, .out_dist = out.dist});
  return s;
}

// Runs the plan of `s` on `input` laid out as `in`, gathering the outputs laid out as
// `out`; contract violations fail the test.
template <class In, class Out>
std::vector<Out> run(const fftn::shape<In, Out>& s, bool in_place, const dims_t& in_extent,
                     const side_layout& in, const dims_t& out_extent, const side_layout& out,
                     std::span<const In> input, const fftn::test::run_options& options,
                     const std::string& what) {
  const fftn::plan<In, Out> p(s, {.in_place = in_place});
  fftn::workspace ws(p);
  const auto call = [&p, &ws](const In* x, Out* y) { p.execute(x, y, ws); };
  const auto r = fftn::test::execute<In, Out>(
      call, offsets_of(in_extent, in), offsets_of(out_extent, out),
      in_place ? fftn::test::placement::in_place : fftn::test::placement::out_of_place, input,
      options);
  for (const std::string& v : r.violations) {
    ADD_FAILURE() << what << ": " << v;
  }
  return r.out;
}

// The strided layouts of a real transform: (real side, complex side, in place).
struct real_layout {
  side_layout real;
  side_layout complex;
  bool in_place = false;
  fftn::test::run_options options;
};

// Offsets of element 0 from a 64-byte boundary are given in units of T, so that they suit
// the alignment of both sides.
template <class T> std::vector<real_layout> real_layouts(const dims_t& dims) {
  const std::size_t r = dims.size();
  const dims_t cplx = halved(dims);
  const std::vector<int> forward(r + 1, 1);
  const std::vector<int> backward(r + 1, -1);
  std::vector<int> mixed(r + 1, 1);
  for (std::size_t d = 0; d < r + 1; d += 2) {
    mixed[d] = -1; // the slowest axis, every other one after it, and maybe the transforms
  }
  // In place the last axis must be contiguous on both sides (docs/conventions.md).
  std::vector<int> mixed_in_place = mixed;
  std::vector<int> backward_in_place = backward;
  mixed_in_place[r - 1] = 1;
  backward_in_place[r - 1] = 1;
  using fftn::test::guard_side;
  const auto off = [](std::size_t real_units, std::size_t complex_units, guard_side g) {
    return fftn::test::run_options{
        .in_offset = real_units * sizeof(T),
        .out_offset = complex_units * sizeof(T),
        .guard = g,
    };
  };
  const side_layout padded_in_place = nested(cplx, 1, 1, mixed_in_place);
  const side_layout reversed_in_place = nested(cplx, 1, 0, backward_in_place);
  return {
      // Gaps on the real side, everything reversed on the complex side.
      {
          .real = nested(dims, 2, 1, forward),
          .complex = nested(cplx, -1, 3, backward),
          .in_place = false,
          .options = off(1, 2, guard_side::above),
      },
      // The real side reversed, gaps and mixed directions on the complex side.
      {
          .real = nested(dims, -1, 0, backward),
          .complex = nested(cplx, 3, 2, mixed),
          .in_place = false,
          .options = off(3, 0, guard_side::below),
      },
      // In place: padded rows and mixed directions; then every axis but the last reversed.
      // Element 0 is shared, so both sides have the same offset.
      {
          .real = real_twin(padded_in_place),
          .complex = padded_in_place,
          .in_place = true,
          .options = off(2, 2, guard_side::below),
      },
      {
          .real = real_twin(reversed_in_place),
          .complex = reversed_in_place,
          .in_place = true,
          .options = off(4, 4, guard_side::above),
      },
  };
}

TYPED_TEST(MultiDim, StridedRealTransforms) {
  using T = TypeParam;
  using c = std::complex<T>;
  checker check;
  for (const dims_t& dims : shapes()) {
    const dims_t cplx = halved(dims);
    const std::size_t points = product(dims);
    const real_case data = real_data<T>(dims, points + 3);
    const std::vector<T> x = real_parts<T>(data.reals);
    const std::vector<c> spectrum(data.spectrum.begin(), data.spectrum.end());
    const std::vector<cd> want_reals = scaled(data.reals, points);
    const auto layouts = real_layouts<T>(dims);
    for (std::size_t l = 0; l < layouts.size(); ++l) {
      const real_layout& lay = layouts[l];
      const std::string tag = " layout " + std::to_string(l);
      const auto y =
          run<T, c>(shape_of<T, c>(dims, lay.real, lay.complex), lay.in_place, dims, lay.real, cplx,
                    lay.complex, x, lay.options, describe(dims, "r2c") + tag);
      check.within<T>(y, data.spectrum, tolerance_for<T>(dims, false), describe(dims, "r2c") + tag);
      const fftn::test::run_options swapped{
          .in_offset = lay.options.out_offset,
          .out_offset = lay.options.in_offset,
          .guard = lay.options.guard,
      };
      const auto z =
          run<c, T>(shape_of<c, T>(dims, lay.complex, lay.real), lay.in_place, cplx, lay.complex,
                    dims, lay.real, spectrum, swapped, describe(dims, "c2r") + tag);
      const std::vector<c> zc(z.begin(), z.end());
      check.within<T>(zc, want_reals, tolerance_for<T>(dims, true), describe(dims, "c2r") + tag);
    }
  }
}

TYPED_TEST(MultiDim, StridedComplexTransforms) {
  using T = TypeParam;
  using c = std::complex<T>;
  checker check;
  for (const dims_t& dims : shapes()) {
    const std::size_t r = dims.size();
    const std::size_t points = product(dims);
    fftn::test::input_source src(points + 4);
    const auto x = src.complexes<T>(points * count);
    const std::vector<int> backward(r + 1, -1);
    std::vector<int> mixed(r + 1, 1);
    for (std::size_t d = 1; d < r + 1; d += 2) {
      mixed[d] = -1;
    }
    const side_layout shared = nested(dims, -1, 2, mixed);
    const std::vector<std::pair<side_layout, side_layout>> layouts{
        {nested(dims, 2, 1, mixed), nested(dims, -1, 3, backward)},
        {shared, shared}, // in place
    };
    for (const fftn::direction dir : {fftn::direction::forward, fftn::direction::backward}) {
      std::vector<cd> want;
      for (std::size_t t = 0; t < count; ++t) {
        const std::vector<cd> one(x.begin() + static_cast<std::ptrdiff_t>(t * points),
                                  x.begin() + static_cast<std::ptrdiff_t>((t + 1) * points));
        const auto y = dft(static_cast<int>(dir), dims, one);
        want.insert(want.end(), y.begin(), y.end());
      }
      for (std::size_t l = 0; l < layouts.size(); ++l) {
        const auto& [in, out] = layouts[l];
        const bool in_place = l == 1;
        const std::string what = describe(dims, in_place ? "c2c in place" : "c2c");
        const auto y = run<c, c>(shape_of<c, c>(dims, in, out, dir), in_place, dims, in, dims, out,
                                 x, {.in_offset = 8, .out_offset = 8}, what);
        check.within<T>(y, want, tolerance_for<T>(dims, false), what);
      }
    }
  }
}

} // namespace
