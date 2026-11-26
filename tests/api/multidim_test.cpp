// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Transforms of rank 2 and 3 beyond the c2c planes of the oracle suites: real planes and
// volumes (r2c, c2r) and complex volumes against the separable quadruple-precision
// reference, with the default layouts and with strided ones (gaps, padding, reversed axes
// and distances), in and out of place. Strided runs go through the harness, which places
// each array against a guard page, checks that no byte outside the described elements is
// written and, out of place, that the input is unchanged.

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "reference/reference_dft.hpp"
#include "support/canary.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/harness.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/precisions.hpp"
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
#include <stdexcept>
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
      fftn::test::derived_tolerance<T>(product(dims), fftn::test::fftn_bound(product(dims))) +
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

// The last four are volumes with odd and prime extents, narrower last blocks of columns and
// pencils, and an axis of 2 (task 4.2). They run as the profile orders them (axis by axis in
// one thread); tests/core/volumes_test.cpp holds plane at a time byte for byte equal to that.
const std::vector<dims_t>& shapes() {
  static const std::vector<dims_t> v{
      {8, 6},     {5, 7},       {16, 16},     {32, 30},    {1, 9},    {9, 1},
      {6, 1},     {4, 6, 8},    {3, 5, 4},    {2, 3, 5},   {1, 4, 1}, {7, 1, 3},
      {9, 7, 11}, {17, 15, 37}, {12, 10, 64}, {31, 2, 19},
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

template <class T>
real_case real_data(const dims_t& dims, std::uint64_t seed, std::size_t transforms = count) {
  const std::size_t n = dims.back();
  const std::size_t h = n / 2 + 1;
  const std::size_t points = product(dims);
  const std::size_t lines = points / n;
  fftn::test::input_source src(seed);
  const auto x = src.reals<T>(points * transforms);
  real_case c;
  c.reals.assign(x.begin(), x.end());
  for (std::size_t t = 0; t < transforms; ++t) {
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
TYPED_TEST_SUITE(MultiDim, fftn::test::precisions, fftn::test::precision_names);

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

// ---- The shape functions of real planes and volumes.

// planes_r2c, planes_c2r, volumes_r2c and volumes_c2r build the shapes the general form
// builds (tight C-ordered defaults, one batch dimension of `count`, c2r backward), and their
// plans compute the transform: r2c and c2r against the reference, out of place and in place
// (FFTW's padded rows), and a round trip r2c then c2r scaled by 1/points, which gives the
// input back; odd and even last axes, three transforms.
template <class T> fftn::shape<T, std::complex<T>> real_shape(const dims_t& d, std::size_t k) {
  return d.size() == 2 ? fftn::planes_r2c<T>(d[0], d[1], k)
                       : fftn::volumes_r2c<T>(d[0], d[1], d[2], k);
}
template <class T> fftn::shape<std::complex<T>, T> complex_shape(const dims_t& d, std::size_t k) {
  return d.size() == 2 ? fftn::planes_c2r<T>(d[0], d[1], k)
                       : fftn::volumes_c2r<T>(d[0], d[1], d[2], k);
}

template <class In, class Out>
void expect_tight(const fftn::shape<In, Out>& s, const dims_t& d, std::size_t k) {
  ASSERT_EQ(s.axes.size(), d.size());
  for (std::size_t a = 0; a < d.size(); ++a) {
    EXPECT_EQ(s.axes[a].n, d[a]);
    EXPECT_EQ(s.axes[a].in_stride, 0);
    EXPECT_EQ(s.axes[a].out_stride, 0);
  }
  ASSERT_EQ(s.batches.size(), 1U);
  EXPECT_EQ(s.batches[0].count, k);
  EXPECT_EQ(s.batches[0].in_dist, 0);
  EXPECT_EQ(s.batches[0].out_dist, 0);
}

// The n reals of each of `lines` rows that lie `row` reals apart, made complex.
template <class T>
std::vector<std::complex<T>> rows_of(const T* v, std::size_t lines, std::size_t row,
                                     std::size_t n) {
  std::vector<std::complex<T>> out;
  for (std::size_t l = 0; l < lines; ++l) {
    out.insert(out.end(), v + l * row, v + l * row + n);
  }
  return out;
}

TYPED_TEST(MultiDim, RealPlaneAndVolumeShapeFunctions) {
  using T = TypeParam;
  using c = std::complex<T>;
  constexpr std::size_t transforms = 3;
  checker check;
  const std::vector<dims_t> cases = {{6, 9}, {5, 7}, {8, 10}, {3, 5, 7}, {4, 6, 9}, {7, 3, 4}};
  for (const dims_t& dims : cases) {
    const std::size_t n = dims.back();
    const std::size_t h = n / 2 + 1;
    const std::size_t points = product(dims);
    const std::size_t lines = points / n * transforms;
    const auto r2c = real_shape<T>(dims, transforms);
    const auto c2r = complex_shape<T>(dims, transforms);
    expect_tight(r2c, dims, transforms);
    expect_tight(c2r, dims, transforms);
    EXPECT_EQ(c2r.dir, fftn::direction::backward);
    const real_case data = real_data<T>(dims, points + 7, transforms);
    const std::vector<T> x = real_parts<T>(data.reals);
    for (const bool in_place : {false, true}) {
      const std::string where = in_place ? " in place" : "";
      const fftn::r2c_plan<T> forward(r2c, {.in_place = in_place});
      const fftn::c2r_plan<T> unscaled(c2r, {.in_place = in_place});
      const fftn::c2r_plan<T> backward(
          c2r, {.in_place = in_place, .scale = 1.0 / static_cast<double>(points)});
      fftn::workspace ws(std::max(
          {forward.workspace_bytes(), unscaled.workspace_bytes(), backward.workspace_bytes()}));
      // In place each real row is padded to the 2h reals of its complex row.
      const std::size_t row = in_place ? 2 * h : n;
      std::vector<c> y(lines * h);
      std::vector<T> separate(lines * row);
      T* real = in_place ? reinterpret_cast<T*>(y.data()) : separate.data();
      for (std::size_t l = 0; l < lines; ++l) {
        std::memcpy(real + l * row, x.data() + l * n, n * sizeof(T));
      }
      forward.execute(real, y.data(), ws);
      check.within<T>(y, data.spectrum, tolerance_for<T>(dims, false),
                      describe(dims, ("r2c shape function" + where).c_str()));
      // c2r unscaled, from a copy of the reference spectrum: points times the data.
      std::vector<c> spectrum(data.spectrum.begin(), data.spectrum.end());
      std::vector<T> z(lines * row);
      T* z_real = in_place ? reinterpret_cast<T*>(spectrum.data()) : z.data();
      unscaled.execute(spectrum.data(), z_real, ws);
      check.within<T>(rows_of(z_real, lines, row, n), scaled(data.reals, points),
                      tolerance_for<T>(dims, true),
                      describe(dims, ("c2r shape function" + where).c_str()));
      // The round trip: the scaled c2r of the r2c's output gives x back.
      backward.execute(y.data(), real, ws);
      check.within<T>(rows_of(real, lines, row, n), data.reals,
                      tolerance_for<T>(dims, false) + tolerance_for<T>(dims, true),
                      describe(dims, ("round trip" + where).c_str()));
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
std::vector<Out> run_at(const fftn::shape<In, Out>& s, bool in_place,
                        const std::vector<std::ptrdiff_t>& in_offsets,
                        const std::vector<std::ptrdiff_t>& out_offsets, std::span<const In> input,
                        const fftn::test::run_options& options, const std::string& what,
                        fftn::isa target = fftn::isa::best) {
  const fftn::plan<In, Out> p(s, {.in_place = in_place, .force_isa = target});
  fftn::workspace ws(p);
  const auto call = [&p, &ws](const In* x, Out* y) { p.execute(x, y, ws); };
  const auto r = fftn::test::execute<In, Out>(call, in_offsets, out_offsets,
                                              in_place ? fftn::test::placement::in_place
                                                       : fftn::test::placement::out_of_place,
                                              input, options);
  for (const std::string& v : r.violations) {
    ADD_FAILURE() << what << ": " << v;
  }
  return r.out;
}

template <class In, class Out>
std::vector<Out> run(const fftn::shape<In, Out>& s, bool in_place, const dims_t& in_extent,
                     const side_layout& in, const dims_t& out_extent, const side_layout& out,
                     std::span<const In> input, const fftn::test::run_options& options,
                     const std::string& what) {
  return run_at(s, in_place, offsets_of(in_extent, in), offsets_of(out_extent, out), input, options,
                what);
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

// ---- Three and four batch dimensions, nested in memory in any order, through the harness.

// A stride per axis and a distance per batch dimension.
struct batched_layout {
  strides_t stride;
  strides_t dist;
};

// Dimensions are numbered axes first, then batch dimensions, and nested in memory in `order`
// (innermost first): the innermost has stride `first`, and each other one spans the reach of
// those inside it plus `pad` unused elements. sign[i] < 0 reverses dimension i. Batch
// dimensions may lie inside axes, so the batch loops do not nest in their own order.
batched_layout nest(const dims_t& axes, const dims_t& counts, const std::vector<std::size_t>& order,
                    std::ptrdiff_t first, std::ptrdiff_t pad, const std::vector<int>& sign) {
  const std::size_t r = axes.size();
  strides_t step(r + counts.size(), 0);
  std::ptrdiff_t reach = 0;
  for (std::size_t k = 0; k < order.size(); ++k) {
    const std::size_t i = order[k];
    const std::size_t extent = i < r ? axes[i] : counts[i - r];
    const std::ptrdiff_t magnitude = k == 0 ? first : reach + 1 + pad;
    step[i] = sign[i] < 0 ? -magnitude : magnitude;
    reach += static_cast<std::ptrdiff_t>(extent - 1) * magnitude;
  }
  const auto split = step.begin() + static_cast<std::ptrdiff_t>(r);
  return {.stride = strides_t(step.begin(), split), .dist = strides_t(split, step.end())};
}

// The real side of an in-place real transform over `complex`'s memory (FFTW's padded
// layout): the last axis as is, every other stride and every distance doubled.
batched_layout real_twin(batched_layout complex) {
  for (std::size_t d = 0; d + 1 < complex.stride.size(); ++d) {
    complex.stride[d] *= 2;
  }
  for (std::ptrdiff_t& d : complex.dist) {
    d *= 2;
  }
  return complex;
}

// Element offsets in logical order: transforms numbered with the last batch dimension
// fastest, C order within each.
std::vector<std::ptrdiff_t> offsets_of(const dims_t& extent, const dims_t& counts,
                                       const batched_layout& s) {
  const std::size_t points = product(extent);
  const std::size_t transforms = product(counts);
  std::vector<std::ptrdiff_t> v;
  v.reserve(points * transforms);
  for (std::size_t t = 0; t < transforms; ++t) {
    std::ptrdiff_t base = 0;
    std::size_t rest = t;
    for (std::size_t b = counts.size(); b > 0; --b) {
      base += static_cast<std::ptrdiff_t>(rest % counts[b - 1]) * s.dist[b - 1];
      rest /= counts[b - 1];
    }
    for (std::size_t i = 0; i < points; ++i) {
      std::ptrdiff_t off = base;
      std::size_t at = i;
      for (std::size_t a = extent.size(); a > 0; --a) {
        off += static_cast<std::ptrdiff_t>(at % extent[a - 1]) * s.stride[a - 1];
        at /= extent[a - 1];
      }
      v.push_back(off);
    }
  }
  return v;
}

template <class In, class Out>
fftn::shape<In, Out> batched_shape(const dims_t& axes, const dims_t& counts,
                                   const batched_layout& in, const batched_layout& out,
                                   fftn::direction dir = fftn::direction::forward) {
  fftn::shape<In, Out> s;
  s.dir = dir;
  for (std::size_t d = 0; d < axes.size(); ++d) {
    s.axes.push_back({.n = axes[d], .in_stride = in.stride[d], .out_stride = out.stride[d]});
  }
  for (std::size_t b = 0; b < counts.size(); ++b) {
    s.batches.push_back({.count = counts[b], .in_dist = in.dist[b], .out_dist = out.dist[b]});
  }
  return s;
}

struct batched_case {
  dims_t axes;
  dims_t counts;                  // three or four batch dimensions
  std::vector<std::size_t> order; // memory nesting, innermost first (nest)
};

// Axes and batch dimensions interleaved in memory; a batch dimension of count 1 among them.
const std::vector<batched_case>& batched_cases() {
  static const std::vector<batched_case> v{
      {.axes = {6}, .counts = {2, 3, 2}, .order = {2, 0, 3, 1}},
      {.axes = {8}, .counts = {3, 2, 2}, .order = {0, 2, 3, 1}},
      {.axes = {4, 5}, .counts = {2, 1, 3, 2}, .order = {1, 4, 0, 5, 2, 3}},
      {.axes = {3, 7}, .counts = {2, 2, 1, 3}, .order = {1, 3, 0, 5, 4, 2}},
  };
  return v;
}

std::string describe(const batched_case& k, const char* what) {
  std::string s = describe(k.axes, what) + ", batches";
  for (const std::size_t c : k.counts) {
    s += " " + std::to_string(c);
  }
  return s;
}

// Alternating signs over the dimensions, starting with `first`.
std::vector<int> alternating(std::size_t dims, int first) {
  std::vector<int> v(dims);
  for (std::size_t i = 0; i < dims; ++i) {
    v[i] = i % 2 == 0 ? first : -first;
  }
  return v;
}

// `order` with the last axis moved innermost, as in-place real transforms need.
std::vector<std::size_t> last_axis_innermost(std::vector<std::size_t> order, std::size_t r) {
  std::erase(order, r - 1);
  order.insert(order.begin(), r - 1);
  return order;
}

// One case through every complex layout: strided with gaps out of place (the output nested the
// other way round and reversed), and in place.
template <class T>
void check_batched_complex(checker& check, const batched_case& k,
                           fftn::isa target = fftn::isa::best) {
  using c = std::complex<T>;
  const std::size_t dims = k.axes.size() + k.counts.size();
  const std::size_t points = product(k.axes);
  const std::size_t transforms = product(k.counts);
  fftn::test::input_source src(points + transforms);
  const auto x = src.complexes<T>(points * transforms);
  const std::vector<std::size_t> reversed(k.order.rbegin(), k.order.rend());
  const batched_layout in = nest(k.axes, k.counts, k.order, 2, 1, alternating(dims, 1));
  const batched_layout out = nest(k.axes, k.counts, reversed, 1, 3, std::vector<int>(dims, -1));
  const batched_layout shared = nest(k.axes, k.counts, k.order, 1, 2, alternating(dims, -1));
  const batched_layout tight = nest(k.axes, k.counts, k.order, 1, 0, std::vector<int>(dims, 1));
  for (const fftn::direction dir : {fftn::direction::forward, fftn::direction::backward}) {
    std::vector<cd> want;
    for (std::size_t t = 0; t < transforms; ++t) {
      const std::vector<cd> one(x.begin() + static_cast<std::ptrdiff_t>(t * points),
                                x.begin() + static_cast<std::ptrdiff_t>((t + 1) * points));
      const auto y = dft(static_cast<int>(dir), k.axes, one);
      want.insert(want.end(), y.begin(), y.end());
    }
    const auto tol = tolerance_for<T>(k.axes, false);
    const std::string what = describe(k, "c2c");
    const auto y = run_at<c, c>(batched_shape<c, c>(k.axes, k.counts, in, out, dir), false,
                                offsets_of(k.axes, k.counts, in), offsets_of(k.axes, k.counts, out),
                                x, {.in_offset = 8, .out_offset = 24}, what, target);
    check.within<T>(y, want, tol, what);
    const std::string what_in_place = describe(k, "c2c in place");
    const auto offsets = offsets_of(k.axes, k.counts, shared);
    const auto z = run_at<c, c>(
        batched_shape<c, c>(k.axes, k.counts, shared, shared, dir), true, offsets, offsets, x,
        {.in_offset = 16, .out_offset = 16, .guard = fftn::test::guard_side::below}, what_in_place,
        target);
    check.within<T>(z, want, tol, what_in_place);
    // Tight on both sides, in and out of place: where the innermost dimension is a batch
    // dimension one element apart, the transforms run as blocks of adjacent lanes.
    const std::string what_tight = describe(k, "c2c tight");
    const auto tight_offsets = offsets_of(k.axes, k.counts, tight);
    for (const bool same : {false, true}) {
      const auto sh = batched_shape<c, c>(k.axes, k.counts, tight, tight, dir);
      const auto t =
          run_at<c, c>(sh, same, tight_offsets, tight_offsets, x, {}, what_tight, target);
      check.within<T>(t, want, tol, what_tight);
      // The same through a profile that takes the lanes over adjacent transforms wherever they
      // fit (X61), so that the route meets the reference on every machine.
      fftn::detail::build_settings route;
      route.profile.adjacent_lanes = true;
      route.profile.adjacent_team_chain = false;
      route.profile.adjacent_min_bytes = 0;
      const auto core = fftn::detail::make_plan_core(
          fftn::detail::view_of(sh), {.in_place = same, .force_isa = target}, route);
      const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
      const auto call = [&core, &ws](const c* from, c* to) {
        fftn::detail::execute_core(*core, 0, fftn::detail::transform_count(*core), from, to, ws);
      };
      const auto r = fftn::test::execute<c, c>(call, tight_offsets, tight_offsets,
                                               same ? fftn::test::placement::in_place
                                                    : fftn::test::placement::out_of_place,
                                               std::span<const c>(x), {});
      for (const std::string& v : r.violations) {
        ADD_FAILURE() << what_tight << " (route): " << v;
      }
      check.within<T>(r.out, want, tol, what_tight + " (route)");
    }
  }
}

// One case through every real layout: r2c and c2r, out of place (strided, the complex side
// nested the other way round and reversed) and in place (FFTW's padded layout, the halved axis
// innermost).
template <class T>
void check_batched_real(checker& check, const batched_case& k, fftn::isa target = fftn::isa::best) {
  using c = std::complex<T>;
  const std::size_t r = k.axes.size();
  const std::size_t dims = r + k.counts.size();
  const dims_t cplx = halved(k.axes);
  const std::size_t points = product(k.axes);
  const std::size_t transforms = product(k.counts);
  const real_case data = real_data<T>(k.axes, points + transforms + 5, transforms);
  const std::vector<T> x = real_parts<T>(data.reals);
  const std::vector<c> spectrum(data.spectrum.begin(), data.spectrum.end());
  const std::vector<cd> want_reals = scaled(data.reals, points);
  const auto as_complex = [](const std::vector<T>& v) {
    return std::vector<c>(v.begin(), v.end());
  };
  const std::vector<std::size_t> reversed(k.order.rbegin(), k.order.rend());
  const batched_layout complex_in_place =
      nest(cplx, k.counts, last_axis_innermost(k.order, r), 1, 1, [&] {
        auto s = alternating(dims, -1);
        s[r - 1] = 1; // in place the last axis is contiguous on both sides
        return s;
      }());
  struct layout {
    batched_layout real;
    batched_layout complex;
    bool in_place;
    fftn::test::run_options options; // offsets in units of T, so either side may take them
  };
  const std::vector<layout> layouts{
      {
          .real = nest(k.axes, k.counts, k.order, 2, 1, alternating(dims, -1)),
          .complex = nest(cplx, k.counts, reversed, 1, 3, std::vector<int>(dims, -1)),
          .in_place = false,
          .options = {.in_offset = sizeof(T), .out_offset = 2 * sizeof(T)},
      },
      {
          .real = real_twin(complex_in_place),
          .complex = complex_in_place,
          .in_place = true,
          .options =
              {
                  .in_offset = 3 * sizeof(T),
                  .out_offset = 3 * sizeof(T),
                  .guard = fftn::test::guard_side::below,
              },
      },
  };
  for (const layout& lay : layouts) {
    const std::string r2c_what = describe(k, lay.in_place ? "r2c in place" : "r2c");
    const auto y =
        run_at<T, c>(batched_shape<T, c>(k.axes, k.counts, lay.real, lay.complex), lay.in_place,
                     offsets_of(k.axes, k.counts, lay.real),
                     offsets_of(cplx, k.counts, lay.complex), x, lay.options, r2c_what, target);
    check.within<T>(y, data.spectrum, tolerance_for<T>(k.axes, false), r2c_what);
    const std::string c2r_what = describe(k, lay.in_place ? "c2r in place" : "c2r");
    const fftn::test::run_options swapped{
        .in_offset = lay.options.out_offset,
        .out_offset = lay.options.in_offset,
        .guard = lay.options.guard,
    };
    const auto z = run_at<c, T>(
        batched_shape<c, T>(k.axes, k.counts, lay.complex, lay.real, fftn::direction::backward),
        lay.in_place, offsets_of(cplx, k.counts, lay.complex),
        offsets_of(k.axes, k.counts, lay.real), spectrum, swapped, c2r_what, target);
    check.within<T>(as_complex(z), want_reals, tolerance_for<T>(k.axes, true), c2r_what);
  }
}

TYPED_TEST(MultiDim, ManyBatchDimensionsComplex) {
  checker check;
  for (const batched_case& k : batched_cases()) {
    check_batched_complex<TypeParam>(check, k);
  }
}

TYPED_TEST(MultiDim, ManyBatchDimensionsReal) {
  checker check;
  for (const batched_case& k : batched_cases()) {
    check_batched_real<TypeParam>(check, k);
  }
}

// ---- Every subset and order of the axes of a strided array (task 4.3).

// The ordered selections of one to three of the dimensions of an array of rank `rank`.
std::vector<std::vector<std::size_t>> axis_selections(std::size_t rank) {
  std::vector<std::vector<std::size_t>> all;
  std::vector<std::size_t> now;
  const auto extend = [&](const auto& self) -> void {
    if (!now.empty()) {
      all.push_back(now);
    }
    if (now.size() == 3) {
      return;
    }
    for (std::size_t d = 0; d < rank; ++d) {
      if (std::ranges::find(now, d) == now.end()) {
        now.push_back(d);
        self(self);
        now.pop_back();
      }
    }
  };
  extend(extend);
  return all;
}

// How the array's dimensions nest in memory, innermost first: C order, the reverse (Fortran
// order) and C order rotated, so that the fastest dimension is neither end's.
std::vector<std::vector<std::size_t>> memory_orders(std::size_t rank) {
  std::vector<std::size_t> c_order(rank);
  for (std::size_t d = 0; d < rank; ++d) {
    c_order[d] = rank - 1 - d;
  }
  std::vector<std::size_t> rotated = c_order;
  std::ranges::rotate(rotated, rotated.begin() + 1);
  std::vector<std::vector<std::size_t>> v{c_order};
  for (const auto& o : {std::vector<std::size_t>(c_order.rbegin(), c_order.rend()), rotated}) {
    if (std::ranges::find(v, o) == v.end()) {
      v.push_back(o);
    }
  }
  return v;
}

// The case of transforming the dimensions `selected` of an array of extents `array` whose
// dimensions nest as `memory`: the other dimensions are batch dimensions, listed in the
// array's order or (`reverse_batches`) the opposite one, so that neither the batch order nor
// the axes' has to follow the memory's.
batched_case selection_case(const dims_t& array, const std::vector<std::size_t>& selected,
                            const std::vector<std::size_t>& memory, bool reverse_batches) {
  batched_case k;
  std::vector<std::size_t> rest;
  for (std::size_t d = 0; d < array.size(); ++d) {
    if (std::ranges::find(selected, d) == selected.end()) {
      rest.push_back(d);
    }
  }
  if (reverse_batches) {
    std::ranges::reverse(rest);
  }
  std::vector<std::size_t> position(array.size());
  for (std::size_t i = 0; i < selected.size(); ++i) {
    k.axes.push_back(array[selected[i]]);
    position[selected[i]] = i;
  }
  for (std::size_t j = 0; j < rest.size(); ++j) {
    k.counts.push_back(array[rest[j]]);
    position[rest[j]] = selected.size() + j;
  }
  for (const std::size_t d : memory) {
    k.order.push_back(position[d]);
  }
  return k;
}

// Arrays of rank 1 to 4; extents odd and even (a halved axis of either parity), of 1, and
// long enough for the strategies of strided columns (blocks of lanes, X40).
const std::vector<dims_t>& axis_arrays() {
  static const std::vector<dims_t> v{
      {7}, {4, 5}, {6, 3}, {16, 12}, {3, 4, 5}, {5, 1, 6}, {8, 9, 16}, {2, 3, 4, 5},
  };
  return v;
}

// Runs every selection of every array under every instruction set the machine has.
template <class Check> void for_each_selection(const Check& run) {
  for (const fftn::isa target : fftn::test::runnable_isas()) {
    SCOPED_TRACE(fftn::isa_name(target));
    for (const dims_t& array : axis_arrays()) {
      for (const auto& selected : axis_selections(array.size())) {
        for (const auto& memory : memory_orders(array.size())) {
          for (const bool reverse_batches : {false, true}) {
            std::string trace = "array";
            for (const std::size_t n : array) {
              trace += " " + std::to_string(n);
            }
            trace += ", axes";
            for (const std::size_t d : selected) {
              trace += " " + std::to_string(d);
            }
            trace += ", memory (innermost first)";
            for (const std::size_t d : memory) {
              trace += " " + std::to_string(d);
            }
            SCOPED_TRACE(trace + (reverse_batches ? ", batches reversed" : ""));
            run(selection_case(array, selected, memory, reverse_batches), target);
          }
        }
      }
    }
  }
}

TYPED_TEST(MultiDim, EveryAxisSelectionComplex) {
  checker check;
  for_each_selection([&](const batched_case& k, fftn::isa target) {
    check_batched_complex<TypeParam>(check, k, target);
  });
}

TYPED_TEST(MultiDim, EveryAxisSelectionReal) {
  checker check;
  for_each_selection([&](const batched_case& k, fftn::isa target) {
    check_batched_real<TypeParam>(check, k, target);
  });
}

// ---- The shape functions for chosen axes of an array (axes_c2c, axes_r2c, axes_c2r).

// One side of the layout a shape states explicitly.
template <class In, class Out> batched_layout layout_of(const fftn::shape<In, Out>& s, bool input) {
  batched_layout l;
  for (const fftn::axis& a : s.axes) {
    l.stride.push_back(input ? a.in_stride : a.out_stride);
  }
  for (const fftn::batch& b : s.batches) {
    l.dist.push_back(input ? b.in_dist : b.out_dist);
  }
  return l;
}

template <class In, class Out> dims_t counts_of(const fftn::shape<In, Out>& s) {
  dims_t c;
  for (const fftn::batch& b : s.batches) {
    c.push_back(b.count);
  }
  return c;
}

// The array as the shape functions take it: tight C order, or strided with gaps, nested in
// another order on each side and with strides of both signs.
struct array_layouts {
  std::vector<fftn::axis> tight;
  std::vector<fftn::axis> strided;
};

array_layouts layouts_of(const dims_t& array) {
  const std::size_t r = array.size();
  std::vector<std::size_t> order(r);
  for (std::size_t d = 0; d < r; ++d) {
    order[d] = r - 1 - d;
  }
  const std::vector<std::size_t> reversed(order.rbegin(), order.rend());
  const batched_layout in = nest(array, {}, order, 2, 1, alternating(r, 1));
  const batched_layout out = nest(array, {}, reversed, 1, 3, alternating(r, -1));
  array_layouts l;
  for (std::size_t d = 0; d < r; ++d) {
    l.tight.push_back({.n = array[d]});
    l.strided.push_back({.n = array[d], .in_stride = in.stride[d], .out_stride = out.stride[d]});
  }
  return l;
}

// Whether `offsets` fill [0, elements) exactly: a tight layout.
bool fills_tightly(std::vector<std::ptrdiff_t> offsets, std::size_t elements) {
  std::ranges::sort(offsets);
  for (std::size_t i = 0; i < elements; ++i) {
    if (i >= offsets.size() || std::cmp_not_equal(offsets[i], i)) {
      return false;
    }
  }
  return offsets.size() == elements;
}

TYPED_TEST(MultiDim, AxesShapeFunctionComplex) {
  using T = TypeParam;
  using c = std::complex<T>;
  checker check;
  for (const dims_t& array : axis_arrays()) {
    const array_layouts layouts = layouts_of(array);
    for (const auto& selected : axis_selections(array.size())) {
      for (const bool strided : {false, true}) {
        const auto& given = strided ? layouts.strided : layouts.tight;
        for (const fftn::direction dir : {fftn::direction::forward, fftn::direction::backward}) {
          const auto s = fftn::axes_c2c<T>(given, selected, dir);
          ASSERT_EQ(s.axes.size(), selected.size());
          dims_t extents;
          for (std::size_t i = 0; i < selected.size(); ++i) {
            EXPECT_EQ(s.axes[i].n, array[selected[i]]);
            extents.push_back(s.axes[i].n);
          }
          const dims_t counts = counts_of(s);
          const std::size_t points = product(extents);
          const std::size_t transforms = product(counts);
          EXPECT_EQ(points * transforms, product(array));
          EXPECT_EQ(s.dir, dir);
          const batched_layout in = layout_of(s, true);
          const batched_layout out = layout_of(s, false);
          const auto in_offsets = offsets_of(extents, counts, in);
          const auto out_offsets = offsets_of(extents, counts, out);
          if (!strided) {
            EXPECT_TRUE(fills_tightly(in_offsets, product(array)));
            EXPECT_TRUE(fills_tightly(out_offsets, product(array)));
          }
          fftn::test::input_source src(points + transforms + 9);
          const auto x = src.complexes<T>(points * transforms);
          std::vector<cd> want;
          for (std::size_t t = 0; t < transforms; ++t) {
            const std::vector<cd> one(x.begin() + static_cast<std::ptrdiff_t>(t * points),
                                      x.begin() + static_cast<std::ptrdiff_t>((t + 1) * points));
            const auto y = dft(static_cast<int>(dir), extents, one);
            want.insert(want.end(), y.begin(), y.end());
          }
          const auto y = run_at<c, c>(s, false, in_offsets, out_offsets, x, {}, "axes_c2c");
          check.within<T>(y, want, tolerance_for<T>(extents, false), describe(extents, "axes_c2c"));
          if (!strided) {
            const auto z =
                run_at<c, c>(s, true, in_offsets, out_offsets, x, {}, "axes_c2c in place");
            check.within<T>(z, want, tolerance_for<T>(extents, false),
                            describe(extents, "axes_c2c in place"));
          }
        }
      }
    }
  }
}

TYPED_TEST(MultiDim, AxesShapeFunctionReal) {
  using T = TypeParam;
  using c = std::complex<T>;
  checker check;
  const auto as_complex = [](const std::vector<T>& v) {
    return std::vector<c>(v.begin(), v.end());
  };
  for (const dims_t& array : axis_arrays()) {
    const array_layouts layouts = layouts_of(array);
    for (const auto& selected : axis_selections(array.size())) {
      // In place the halved dimension must be the fastest, which only the array's last is
      // by default.
      const bool can_be_in_place = selected.back() + 1 == array.size();
      for (const int variant : {0, 1, 2}) {
        const bool strided = variant == 1;
        const bool in_place = variant == 2;
        if (in_place && !can_be_in_place) {
          continue;
        }
        const auto& given = strided ? layouts.strided : layouts.tight;
        const auto forward = fftn::axes_r2c<T>(given, selected, in_place);
        const auto backward = fftn::axes_c2r<T>(given, selected, in_place);
        EXPECT_EQ(backward.dir, fftn::direction::backward);
        dims_t extents;
        for (const fftn::axis& a : forward.axes) {
          extents.push_back(a.n);
        }
        const dims_t cplx = halved(extents);
        const dims_t counts = counts_of(forward);
        const std::size_t transforms = product(counts);
        const std::size_t points = product(extents);
        EXPECT_EQ(points * transforms, product(array));
        const real_case data = real_data<T>(extents, points + transforms + 11, transforms);
        const std::vector<T> x = real_parts<T>(data.reals);
        const std::vector<c> spectrum(data.spectrum.begin(), data.spectrum.end());
        const std::string name = in_place ? "axes_r2c in place" : "axes_r2c";
        const auto y =
            run_at<T, c>(forward, in_place, offsets_of(extents, counts, layout_of(forward, true)),
                         offsets_of(cplx, counts, layout_of(forward, false)), x, {}, name);
        check.within<T>(y, data.spectrum, tolerance_for<T>(extents, false),
                        describe(extents, name.c_str()));
        const auto z = run_at<c, T>(
            backward, in_place, offsets_of(cplx, counts, layout_of(backward, true)),
            offsets_of(extents, counts, layout_of(backward, false)), spectrum, {}, name);
        check.within<T>(as_complex(z), scaled(data.reals, points), tolerance_for<T>(extents, true),
                        describe(extents, name.c_str()));
      }
    }
  }
}

// The axes run in memory order however the shape lists them (DESIGN.md 9.3), so two listings
// of one transform are the same program and give the same bits.
TYPED_TEST(MultiDim, AxisOrderOfAShapeChangesNoBit) {
  using T = TypeParam;
  using c = std::complex<T>;
  const dims_t array{16, 9, 8};
  const std::vector<fftn::axis> given{{.n = 16}, {.n = 9}, {.n = 8}};
  fftn::test::input_source src(41);
  const auto x = src.complexes<T>(product(array));
  const auto run = [&](const std::vector<std::size_t>& selected) {
    const auto s = fftn::axes_c2c<T>(given, selected, fftn::direction::forward);
    const fftn::c2c_plan<T> p(s);
    fftn::workspace ws(p);
    std::vector<c> y(x.size());
    p.execute(x.data(), y.data(), ws);
    return y;
  };
  for (const auto& [a, b] :
       std::vector<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>>{
           {{0, 1, 2}, {2, 1, 0}},
           {{0, 2}, {2, 0}},
           {{1, 2}, {2, 1}},
           {{0, 1}, {1, 0}},
       }) {
    const auto ya = run(a);
    const auto yb = run(b);
    EXPECT_TRUE(std::ranges::equal(std::as_bytes(std::span<const c>(ya)),
                                   std::as_bytes(std::span<const c>(yb))))
        << "axes " << a.size();
  }
}

// Neighbours whose merged span would overflow ptrdiff_t stay two batch dimensions (the product
// is never formed), and a dimension of length 0 is refused by name.
TEST(AxesShape, MergeNeverOverflows) {
  const auto pow2 = [](unsigned k) { return static_cast<std::ptrdiff_t>(std::size_t{1} << k); };
  const std::vector<fftn::axis> array{
      {.n = 2, .in_stride = pow2(50), .out_stride = pow2(50)},
      {.n = std::size_t{1} << 24U, .in_stride = pow2(40), .out_stride = pow2(40)},
      {.n = 2},
  };
  const std::vector<std::size_t> last{2};
  const auto s = fftn::axes_c2c<double>(array, last, fftn::direction::forward);
  ASSERT_EQ(s.batches.size(), 2U);
  EXPECT_EQ(s.batches[0].count, 2U);
  EXPECT_EQ(s.batches[1].count, std::size_t{1} << 24U);
  const std::vector<fftn::axis> empty{{.n = 4}, {.n = 0}};
  const std::vector<std::size_t> first{0};
  try {
    static_cast<void>(fftn::axes_c2c<double>(empty, first, fftn::direction::forward));
    ADD_FAILURE() << "a dimension of length 0 was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("dimension 1 has length 0"), std::string::npos)
        << e.what();
  }
}

// A real shape built without `padded` and planned in place is refused with a hint at `padded`.
TEST(AxesShape, UnpaddedRealShapeInPlaceSaysSo) {
  const std::vector<fftn::axis> array{{.n = 6}, {.n = 8}};
  const std::vector<std::size_t> both{0, 1};
  try {
    const fftn::r2c_plan<double> p(fftn::axes_r2c<double>(array, both), {.in_place = true});
    ADD_FAILURE() << "an unpadded in-place real shape was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("padded = true"), std::string::npos) << e.what();
  }
  const fftn::r2c_plan<double> ok(fftn::axes_r2c<double>(array, both, true), {.in_place = true});
  EXPECT_FALSE(ok.describe().empty());
}

TEST(AxesShape, RefusesBadSelections) {
  const std::vector<fftn::axis> array{{.n = 4}, {.n = 5}, {.n = 6}};
  const auto with = [&](std::vector<std::size_t> selected) {
    return fftn::axes_c2c<double>(array, selected, fftn::direction::forward);
  };
  EXPECT_THROW(static_cast<void>(with({3})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(with({1, 1})), std::invalid_argument);
  EXPECT_EQ(with({2, 0}).axes[0].n, 6U);
}

TEST(AxesShape, BatchesAreOrderedByStrideAndMerged) {
  // Dimension {1} of a tight 4 x 5 x 6 x 7: batch dimensions 4 (widest stride) and 6 x 7, which
  // nest exactly and merge into one of 42 transforms one element apart.
  const std::vector<fftn::axis> array{{.n = 4}, {.n = 5}, {.n = 6}, {.n = 7}};
  const std::vector<std::size_t> middle{1};
  const auto s = fftn::axes_c2c<double>(array, middle, fftn::direction::forward);
  ASSERT_EQ(s.axes.size(), 1U);
  EXPECT_EQ(s.axes[0].in_stride, 42);
  ASSERT_EQ(s.batches.size(), 2U);
  EXPECT_EQ(s.batches[0].count, 4U);
  EXPECT_EQ(s.batches[0].in_dist, 210);
  EXPECT_EQ(s.batches[1].count, 42U);
  EXPECT_EQ(s.batches[1].in_dist, 1);
}

} // namespace
