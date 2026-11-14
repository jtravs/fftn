// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "check.hpp"

#include "backend.hpp"
#include "dd.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fbench {
namespace {

std::uint64_t splitmix(std::uint64_t& state) {
  std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31U);
}

// Uniform in [-1, 1) with 53 random bits.
double uniform(std::uint64_t& state) {
  return static_cast<double>(splitmix(state) >> 11U) * 0x1.0p-52 - 1.0;
}

std::size_t idx(std::int64_t i) { return static_cast<std::size_t>(i); }

// Weight of half-spectrum column k in the full spectrum of a real signal of length n:
// columns 0 and n/2 stand for themselves, the others for a conjugate pair too.
double half_weight(std::int64_t k, std::int64_t n) { return (k == 0 || 2 * k == n) ? 1.0 : 2.0; }

side make_side(bool complex, int rank, const std::array<std::int64_t, 3>& dims,
               std::int64_t last_logical, std::int64_t last_phys, std::int64_t batch,
               std::size_t real_bytes) {
  side s;
  s.complex = complex;
  s.rank = rank;
  s.shape = dims;
  s.shape.at(idx(rank - 1)) = last_logical;
  s.last_phys = last_phys;
  std::int64_t outer = 1;
  for (int d = 0; d + 1 < rank; ++d) {
    outer *= dims.at(idx(d));
  }
  s.dist = outer * last_phys;
  s.count = outer * last_logical;
  s.bytes = static_cast<std::size_t>(batch * s.dist) * (complex ? 2 * real_bytes : real_bytes);
  return s;
}

// exp(sign 2 pi i m / n) for 0 <= m < n from two tables of about sqrt(n) entries each, so
// that long transforms need no table of n double-double entries.
class root_table {
public:
  root_table(std::int64_t n, int sign) : n_(n) {
    constexpr std::int64_t whole = 4096;
    step_ =
        n <= whole ? n : static_cast<std::int64_t>(std::ceil(std::sqrt(static_cast<double>(n))));
    for (std::int64_t r = 0; r < step_; ++r) {
      low_.push_back(unit_root(r, n, sign));
    }
    for (std::int64_t q = 0; q * step_ < n; ++q) {
      high_.push_back(unit_root(q * step_, n, sign));
    }
  }

  [[nodiscard]] ddc operator()(std::int64_t m) const {
    if (step_ == n_) {
      return low_[idx(m)];
    }
    return high_[idx(m / step_)] * low_[idx(m % step_)];
  }

private:
  std::int64_t n_;
  std::int64_t step_ = 1;
  std::vector<ddc> low_;
  std::vector<ddc> high_;
};

struct value {
  double re = 0.0;
  double im = 0.0;

  constexpr value() = default;
  constexpr value(double r, double i) : re(r), im(i) {}
};

value load(const side& s, const double* data, std::int64_t b, std::int64_t flat) {
  const std::int64_t e = s.offset(b, flat);
  if (s.complex) {
    return {data[idx(2 * e)], data[idx(2 * e + 1)]};
  }
  return {data[idx(e)], 0.0};
}

// The exact DFT of the pristine input of transform b at output flat index j.
ddc reference(const layout& l, const std::vector<root_table>& roots, const double* x,
              std::int64_t b, std::int64_t j) {
  const int last = l.rank - 1;
  const std::int64_t n_last = l.dims.at(idx(last));
  const std::int64_t in_last = l.in.shape.at(idx(last));
  std::array<std::int64_t, 3> out_index{};
  std::int64_t rest = j;
  for (int d = last; d >= 0; --d) {
    const std::int64_t extent = l.out.shape.at(idx(d));
    out_index.at(idx(d)) = rest % extent;
    rest /= extent;
  }
  std::array<std::int64_t, 3> k{}; // input index in the outer dimensions
  std::array<std::int64_t, 3> m{}; // (out_index[d] * k[d]) mod dims[d]
  const std::int64_t outer = l.in.count / in_last;
  const ddc one{dd{1.0, 0.0}, dd{}};
  ddc sum{};
  for (std::int64_t o = 0; o < outer; ++o) {
    ddc outer_root = one;
    for (int d = 0; d < last; ++d) {
      outer_root = outer_root * roots[idx(d)](m.at(idx(d)));
    }
    std::int64_t ml = 0;
    const std::int64_t step = out_index.at(idx(last));
    for (std::int64_t kl = 0; kl < in_last; ++kl) {
      ddc w = roots[idx(last)](ml);
      if (last > 0) {
        w = w * outer_root;
      }
      value v = load(l.in, x, b, o * in_last + kl);
      if (l.kind == fbench_c2r) {
        const double weight = half_weight(kl, n_last);
        v.re *= weight;
        v.im *= weight;
      }
      sum = mac(sum, w, v.re, v.im);
      ml += step;
      if (ml >= n_last) {
        ml -= n_last;
      }
    }
    for (int d = last - 1; d >= 0; --d) {
      const std::int64_t n = l.dims.at(idx(d));
      ++k.at(idx(d));
      m.at(idx(d)) += out_index.at(idx(d));
      if (m.at(idx(d)) >= n) {
        m.at(idx(d)) -= n;
      }
      if (k.at(idx(d)) < n) {
        break;
      }
      k.at(idx(d)) = 0;
      m.at(idx(d)) = 0;
    }
  }
  return sum;
}

std::vector<std::int64_t> pick_samples(std::int64_t count, std::int64_t wanted,
                                       std::uint64_t& rng) {
  std::vector<std::int64_t> chosen;
  if (wanted >= count) {
    for (std::int64_t i = 0; i < count; ++i) {
      chosen.push_back(i);
    }
    return chosen;
  }
  // The ends and the middle always, where indexing mistakes show first.
  std::unordered_set<std::int64_t> set{
      0,
      count - 1,
      count / 2,
      std::min<std::int64_t>(1, count - 1),
  };
  while (std::cmp_less(set.size(), wanted)) {
    set.insert(static_cast<std::int64_t>(splitmix(rng) % static_cast<std::uint64_t>(count)));
  }
  chosen.assign(set.begin(), set.end());
  std::ranges::sort(chosen);
  return chosen;
}

// Energy of one transform, with half-spectrum weights where the side is a half spectrum.
double energy(const layout& l, const side& s, const double* data, std::int64_t b,
              bool half_spectrum) {
  const std::int64_t n_last = l.dims.at(idx(l.rank - 1));
  const std::int64_t s_last = s.shape.at(idx(l.rank - 1));
  dd sum{};
  for (std::int64_t i = 0; i < s.count; ++i) {
    const value v = load(s, data, b, i);
    const double weight = half_spectrum ? half_weight(i % s_last, n_last) : 1.0;
    sum = sum + dd{weight * (v.re * v.re + v.im * v.im), 0.0};
  }
  return sum.hi;
}

} // namespace

fbench_case layout::to_case() const {
  return fbench_case{
      .kind = kind,
      .rank = rank,
      .dims = dims.data(),
      .batch = batch,
      .sign = sign,
      .in_place = in_place ? 1 : 0,
      .in_last = in.last_phys,
      .out_last = out.last_phys,
      .in_dist = in.dist,
      .out_dist = out.dist,
      .precision = precision,
  };
}

layout make_layout(std::int32_t kind, int rank, const std::array<std::int64_t, 3>& dims,
                   std::int64_t batch, int sign, bool in_place, std::int32_t precision) {
  if (rank < 1 || rank > 3) {
    throw std::invalid_argument("rank must be 1, 2 or 3");
  }
  if (batch < 1) {
    throw std::invalid_argument("batch must be at least 1");
  }
  if (precision != fbench_f64 && precision != fbench_f32) {
    throw std::invalid_argument("unknown precision");
  }
  layout l;
  l.precision = precision;
  l.real_bytes = precision == fbench_f32 ? sizeof(float) : sizeof(double);
  l.kind = kind;
  l.rank = rank;
  l.batch = batch;
  l.in_place = in_place;
  for (int d = 0; d < 3; ++d) {
    const std::int64_t n = d < rank ? dims.at(idx(d)) : 1;
    // FFTW's dimensions are int; FFTN's are wider, but no case needs more.
    if (n < 1 || n > std::numeric_limits<std::int32_t>::max()) {
      throw std::invalid_argument("every length must be in [1, 2^31)");
    }
    l.dims.at(idx(d)) = n;
    l.n_total *= n;
  }
  const std::int64_t n_last = l.dims.at(idx(rank - 1));
  const std::int64_t half = n_last / 2 + 1;
  const std::int64_t padded = in_place ? 2 * half : n_last;
  if (kind == fbench_c2c) {
    if (sign != -1 && sign != 1) {
      throw std::invalid_argument("sign must be -1 or +1");
    }
    l.sign = sign;
    l.in = make_side(true, rank, l.dims, n_last, n_last, batch, l.real_bytes);
    l.out = l.in;
  } else if (kind == fbench_r2c) {
    l.sign = -1;
    l.in = make_side(false, rank, l.dims, n_last, padded, batch, l.real_bytes);
    l.out = make_side(true, rank, l.dims, half, half, batch, l.real_bytes);
  } else if (kind == fbench_c2r) {
    l.sign = 1;
    l.in = make_side(true, rank, l.dims, half, half, batch, l.real_bytes);
    l.out = make_side(false, rank, l.dims, n_last, padded, batch, l.real_bytes);
  } else {
    throw std::invalid_argument("unknown kind");
  }
  if (in_place) {
    l.buffer_bytes_in = std::max(l.in.bytes, l.out.bytes);
  } else {
    l.buffer_bytes_in = l.in.bytes;
    l.buffer_bytes_out = l.out.bytes;
  }
  return l;
}

void fill_input(const layout& l, void* in, std::uint64_t seed) {
  auto* x = static_cast<double*>(in);
  std::fill_n(x, l.reals_in(), 0.0);
  std::uint64_t rng = seed;
  for (std::int64_t b = 0; b < l.batch; ++b) {
    for (std::int64_t i = 0; i < l.in.count; ++i) {
      const std::int64_t e = l.in.offset(b, i);
      if (l.in.complex) {
        x[idx(2 * e)] = uniform(rng);
        x[idx(2 * e + 1)] = uniform(rng);
      } else {
        x[idx(e)] = uniform(rng);
      }
    }
  }
  if (l.kind != fbench_c2r) {
    return;
  }
  // Hermitian consistency of the columns k = 0 and k = n/2 of the half spectrum: entry K'
  // must be the conjugate of entry -K' (indices mod the lengths), and real when K' = -K'.
  const int last = l.rank - 1;
  const std::int64_t n_last = l.dims.at(idx(last));
  const std::int64_t half = l.in.shape.at(idx(last));
  const std::int64_t outer = l.in.count / half;
  std::vector<std::int64_t> columns{0};
  if (n_last % 2 == 0 && n_last > 0) {
    columns.push_back(n_last / 2);
  }
  for (std::int64_t b = 0; b < l.batch; ++b) {
    for (std::int64_t o = 0; o < outer; ++o) {
      std::int64_t rest = o;
      std::int64_t neg = 0;
      std::int64_t scale = 1;
      for (int d = last - 1; d >= 0; --d) {
        const std::int64_t n = l.dims.at(idx(d));
        const std::int64_t k = rest % n;
        rest /= n;
        neg += ((n - k) % n) * scale;
        scale *= n;
      }
      for (const std::int64_t c : columns) {
        const std::int64_t e = l.in.offset(b, o * half + c);
        if (neg == o) {
          x[idx(2 * e + 1)] = 0.0;
        } else if (neg < o) {
          const std::int64_t f = l.in.offset(b, neg * half + c);
          x[idx(2 * e)] = x[idx(2 * f)];
          x[idx(2 * e + 1)] = -x[idx(2 * f + 1)];
        }
      }
    }
  }
}

void perturb_output(const layout& l, void* out, double eps, std::uint64_t seed) {
  auto* y = static_cast<double*>(out);
  std::uint64_t rng = seed;
  for (std::int64_t b = 0; b < l.batch; ++b) {
    for (std::int64_t i = 0; i < l.out.count; ++i) {
      const std::int64_t e = l.out.offset(b, i);
      const double factor = 1.0 + ((splitmix(rng) & 1U) != 0U ? eps : -eps);
      if (l.out.complex) {
        y[idx(2 * e)] *= factor;
        y[idx(2 * e + 1)] *= factor;
      } else {
        y[idx(e)] *= factor;
      }
    }
  }
}

check_result check_output(const layout& l, const void* in, const void* out, std::int64_t budget,
                          std::uint64_t seed) {
  const auto* x = static_cast<const double*>(in);
  const auto* y = static_cast<const double*>(out);
  check_result r;
  const double eps = l.precision == fbench_f32 ? std::numeric_limits<float>::epsilon()
                                               : std::numeric_limits<double>::epsilon();
  r.tol = 10.0 * eps * (1.0 + std::log2(static_cast<double>(l.n_total)));
  std::vector<root_table> roots;
  roots.reserve(static_cast<std::size_t>(l.rank));
  for (int d = 0; d < l.rank; ++d) {
    roots.emplace_back(l.dims.at(idx(d)), l.sign);
  }
  std::vector<std::int64_t> rows{0};
  if (l.batch > 1) {
    rows.push_back(l.batch - 1);
  }
  r.rows = static_cast<std::int64_t>(rows.size());
  const std::int64_t per_sample = std::max<std::int64_t>(1, l.in.count);
  const std::int64_t wanted = std::clamp<std::int64_t>(
      budget / (per_sample * r.rows), std::min<std::int64_t>(4, l.out.count), l.out.count);
  std::uint64_t rng = seed ^ 0x5eedULL;
  double diff2 = 0.0;
  double ref2 = 0.0;
  double diff_max = 0.0;
  std::int64_t compared = 0;
  const bool real_out = l.kind == fbench_c2r;
  for (const std::int64_t b : rows) {
    const std::vector<std::int64_t> samples = pick_samples(l.out.count, wanted, rng);
    r.samples = static_cast<std::int64_t>(samples.size());
    for (const std::int64_t j : samples) {
      const ddc ref = reference(l, roots, x, b, j);
      const value got = load(l.out, y, b, j);
      const double dre = (got.re - ref.re.hi) - ref.re.lo;
      const double dim = real_out ? 0.0 : (got.im - ref.im.hi) - ref.im.lo;
      const double rre = ref.re.hi;
      const double rim = real_out ? 0.0 : ref.im.hi;
      const double d2 = dre * dre + dim * dim;
      diff2 += d2;
      ref2 += rre * rre + rim * rim;
      diff_max = std::max(diff_max, std::sqrt(d2));
      if (std::isnan(d2)) {
        diff_max = std::numeric_limits<double>::infinity();
      }
      ++compared;
    }
    const double e_in = energy(l, l.in, x, b, l.kind == fbench_c2r);
    const double e_out = energy(l, l.out, y, b, l.kind == fbench_r2c);
    const double expected = static_cast<double>(l.n_total) * e_in;
    const double p = std::abs(e_out - expected) / expected;
    r.parseval = std::isnan(p) ? std::numeric_limits<double>::infinity() : std::max(r.parseval, p);
  }
  r.err_rms = std::sqrt(diff2 / ref2);
  r.err_max = diff_max / std::sqrt(ref2 / static_cast<double>(compared));
  r.pass = r.err_rms <= r.tol && r.err_max <= 8.0 * r.tol && r.parseval <= r.tol;
  return r;
}

namespace {

// Writes an impulse (c2c, r2c) or the half spectrum of one (c2r) at multi-index p into
// every transform, and the exact transform computed in double into `expect`.
void impulse_case(const layout& l, const std::array<std::int64_t, 3>& p, double* x,
                  double* expect) {
  std::memset(x, 0, l.buffer_bytes_in);
  std::memset(expect, 0, std::max(l.out.bytes, l.buffer_bytes_in));
  const auto spectrum = [&](const side& s, std::int64_t flat, int sign) {
    double phase = 0.0;
    std::int64_t rest = flat;
    for (int d = l.rank - 1; d >= 0; --d) {
      const std::int64_t k = rest % s.shape.at(idx(d));
      rest /= s.shape.at(idx(d));
      phase += static_cast<double>((k * p.at(idx(d))) % l.dims.at(idx(d))) /
               static_cast<double>(l.dims.at(idx(d)));
    }
    const double angle = 2.0 * std::numbers::pi * static_cast<double>(sign) * phase;
    return value{std::cos(angle), std::sin(angle)};
  };
  std::int64_t p_flat = 0;
  for (int d = 0; d < l.rank; ++d) {
    p_flat = p_flat * l.dims.at(idx(d)) + p.at(idx(d));
  }
  for (std::int64_t b = 0; b < l.batch; ++b) {
    if (l.kind == fbench_c2r) {
      for (std::int64_t i = 0; i < l.in.count; ++i) {
        const value v = spectrum(l.in, i, -1);
        const std::int64_t e = l.in.offset(b, i);
        x[idx(2 * e)] = v.re;
        x[idx(2 * e + 1)] = v.im;
      }
      expect[idx(l.out.offset(b, p_flat))] = static_cast<double>(l.n_total);
      continue;
    }
    const std::int64_t e = l.in.offset(b, p_flat);
    x[idx(l.in.complex ? 2 * e : e)] = 1.0;
    for (std::int64_t i = 0; i < l.out.count; ++i) {
      const value v = spectrum(l.out, i, l.sign);
      const std::int64_t f = l.out.offset(b, i);
      expect[idx(2 * f)] = v.re;
      expect[idx(2 * f + 1)] = v.im;
    }
  }
}

std::string check_impulses() {
  struct shape {
    std::int32_t kind;
    int rank;
    std::array<std::int64_t, 3> dims;
    std::array<std::int64_t, 3> at;
  };
  const std::array<shape, 6> shapes{
      shape{.kind = fbench_c2c, .rank = 1, .dims = {12, 1, 1}, .at = {5, 0, 0}},
      shape{.kind = fbench_c2c, .rank = 2, .dims = {4, 6, 1}, .at = {3, 1, 0}},
      shape{.kind = fbench_r2c, .rank = 1, .dims = {9, 1, 1}, .at = {2, 0, 0}},
      shape{.kind = fbench_r2c, .rank = 2, .dims = {5, 8, 1}, .at = {1, 3, 0}},
      shape{.kind = fbench_c2r, .rank = 2, .dims = {6, 7, 1}, .at = {4, 2, 0}},
      shape{.kind = fbench_c2c, .rank = 3, .dims = {3, 4, 5}, .at = {2, 1, 3}},
  };
  constexpr std::int64_t big_budget = std::int64_t{1024} * 1024;
  for (const shape& s : shapes) {
    const layout l = make_layout(s.kind, s.rank, s.dims, 3, -1, false, fbench_f64);
    std::vector<double> x(l.reals_in());
    std::vector<double> y(std::max(l.out.bytes, l.buffer_bytes_in) / sizeof(double));
    impulse_case(l, s.at, x.data(), y.data());
    const check_result good = check_output(l, x.data(), y.data(), big_budget, 1);
    if (!good.pass || good.samples != l.out.count) {
      return "the check rejects an exact transform (kind " + std::to_string(s.kind) + ", rank " +
             std::to_string(s.rank) + ")";
    }
    // One wrong value in the last transform must fail.
    y[idx(l.out.complex ? 2 * l.out.offset(2, 1) : l.out.offset(2, 1))] += 1e-9;
    if (check_output(l, x.data(), y.data(), big_budget, 1).pass) {
      return "the check accepts a wrong transform (kind " + std::to_string(s.kind) + ")";
    }
  }
  return {};
}

} // namespace

std::string selftest() {
  // Unit roots: |w| = 1, w^a w^b = w^(a+b) and known values, to double-double accuracy.
  constexpr double close = 1e-30;
  const std::array<std::int64_t, 6> lengths{6, 8, 12, 97, 1000, std::int64_t{16} * 1024 * 1024};
  for (const std::int64_t n : lengths) {
    for (std::int64_t a = 0; a < n; a += std::max<std::int64_t>(1, n / 37)) {
      const ddc w = unit_root(a, n, -1);
      const dd norm = w.re * w.re + w.im * w.im - dd{1.0, 0.0};
      const std::int64_t b = (3 * a + 1) % n;
      const ddc product = w * unit_root(b, n, -1);
      const ddc direct = unit_root((a + b) % n, n, -1);
      const dd dre = product.re - direct.re;
      const dd dim = product.im - direct.im;
      if (std::abs(norm.hi) > close || std::abs(dre.hi) > close || std::abs(dim.hi) > close) {
        return "unit_root(" + std::to_string(a) + ", " + std::to_string(n) + ") is inaccurate";
      }
    }
  }
  const ddc sixth = unit_root(1, 6, 1);
  const ddc eighth = unit_root(1, 8, 1);
  if (sixth.re.hi != 0.5 || std::abs(sixth.re.lo) > close ||
      std::abs((eighth.re - eighth.im).hi) > close) {
    return "unit_root misses cos(pi/3) = 1/2 or cos(pi/4) = sin(pi/4)";
  }
  return check_impulses();
}

} // namespace fbench
