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

// `interleave` above 1: the fbench_case::interleave layout, `batch` is `interleave`, and
// `blocks` of it lie one after another.
side make_side(bool complex, int rank, const std::array<std::int64_t, 3>& dims,
               std::int64_t last_logical, std::int64_t last_phys, std::int64_t batch,
               std::size_t real_bytes, std::int64_t interleave = 1, std::int64_t blocks = 1) {
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
  const std::int64_t span = outer * last_phys; // of one transform
  s.count = outer * last_logical;
  const std::size_t unit = complex ? 2 * real_bytes : real_bytes;
  if (interleave > 1) {
    s.stride = interleave;
    s.dist = 1;
    s.block = interleave;
    s.block_dist = interleave * span;
    s.bytes = static_cast<std::size_t>(blocks * s.block_dist) * unit;
  } else {
    s.dist = span;
    s.block = batch;
    s.bytes = static_cast<std::size_t>(batch * s.dist) * unit;
  }
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

// ---- Real-to-real (task 6.12) ---------------------------------------------------------------

// One axis of a real-to-real case: its kind's kernel K(k, j), the weight of x_j in output k,
// from FFTW's definitions (DESIGN.md 5.7):
//   dct1  w_j cos(2 pi j k / 2(n-1)),             w = 1 at j = 0 and n-1, else 2
//   dct2  2 cos(2 pi (2j+1) k / 4n)
//   dct3  w_j cos(2 pi j (2k+1) / 4n),            w = 1 at j = 0, else 2
//   dct4  2 cos(2 pi (2j+1)(2k+1) / 8n)
//   dst1  2 sin(2 pi (j+1)(k+1) / 2(n+1))
//   dst2  2 sin(2 pi (2j+1)(k+1) / 4n)
//   dst3  w_j sin(2 pi (j+1)(2k+1) / 4n),         w = 1 at j = n-1, else 2
//   dst4  2 sin(2 pi (2j+1)(2k+1) / 8n)
//   dht   cos + sin(2 pi j k / n)
// The angle's numerator is reduced modulo its denominator M exactly (two residues below
// M <= 8n, n <= 2^27, so their product fits 64 bits), and the root comes from a double-double
// table, so no rounding of the angle enters.
class r2r_axis {
public:
  r2r_axis(std::int32_t kind, std::int64_t n)
      : kind_(kind), n_(n), m_(period(kind, n)), roots_(m_, 1) {}

  [[nodiscard]] dd kernel(std::int64_t k, std::int64_t j) const {
    std::int64_t p = j;
    std::int64_t q = k;
    double w = 2.0;
    switch (kind_) {
    case fbench_dct1:
      w = (j == 0 || j == n_ - 1) ? 1.0 : 2.0;
      break;
    case fbench_dct2:
      p = 2 * j + 1;
      break;
    case fbench_dct3:
      q = 2 * k + 1;
      w = j == 0 ? 1.0 : 2.0;
      break;
    case fbench_dct4:
    case fbench_dst4:
      p = 2 * j + 1;
      q = 2 * k + 1;
      break;
    case fbench_dst1:
      p = j + 1;
      q = k + 1;
      break;
    case fbench_dst2:
      p = 2 * j + 1;
      q = k + 1;
      break;
    case fbench_dst3:
      p = j + 1;
      q = 2 * k + 1;
      w = j == n_ - 1 ? 1.0 : 2.0;
      break;
    default: // dht
      w = 1.0;
      break;
    }
    const ddc root = roots_(((p % m_) * (q % m_)) % m_);
    dd v = root.re;
    if (kind_ == fbench_dht) {
      v = root.re + root.im;
    } else if (kind_ >= fbench_dst1) {
      v = root.im;
    }
    return v * w;
  }

  // The weight of index i in the energy identity sum_k u_k Y_k^2 = N sum_j v_j x_j^2 (`output`:
  // u, else v): 1/2 where the orthonormal form scales an end by sqrt 2, else 1.
  [[nodiscard]] double energy_weight(std::int64_t i, bool output) const {
    switch (kind_) {
    case fbench_dct1:
      return (i == 0 || i == n_ - 1) ? 0.5 : 1.0;
    case fbench_dct2:
      return output && i == 0 ? 0.5 : 1.0;
    case fbench_dct3:
      return !output && i == 0 ? 0.5 : 1.0;
    case fbench_dst2:
      return output && i == n_ - 1 ? 0.5 : 1.0;
    case fbench_dst3:
      return !output && i == n_ - 1 ? 0.5 : 1.0;
    default:
      return 1.0;
    }
  }

  // N: the kind's logical length, by which a transform and its inverse multiply.
  [[nodiscard]] std::int64_t logical() const {
    switch (kind_) {
    case fbench_dct1:
      return 2 * (n_ - 1);
    case fbench_dst1:
      return 2 * (n_ + 1);
    case fbench_dht:
      return n_;
    default:
      return 2 * n_;
    }
  }

private:
  static std::int64_t period(std::int32_t kind, std::int64_t n) {
    switch (kind) {
    case fbench_dct1:
      return 2 * (n - 1);
    case fbench_dst1:
      return 2 * (n + 1);
    case fbench_dct4:
    case fbench_dst4:
      return 8 * n;
    case fbench_dht:
      return n;
    default:
      return 4 * n;
    }
  }

  std::int32_t kind_;
  std::int64_t n_;
  std::int64_t m_;
  root_table roots_;
};

// The multi-index (slowest first) of row-major flat index `flat` in the case's shape.
std::array<std::int64_t, 3> unflatten(const layout& l, std::int64_t flat) {
  std::array<std::int64_t, 3> at{};
  for (int d = l.rank - 1; d >= 0; --d) {
    at.at(idx(d)) = flat % l.dims.at(idx(d));
    flat /= l.dims.at(idx(d));
  }
  return at;
}

// sum_i u(i) v_i^2 of transform b over the side, u the product of the axes' energy weights.
double r2r_energy(const layout& l, const std::vector<r2r_axis>& axes, const side& s,
                  const double* data, std::int64_t b, bool output) {
  dd sum{};
  for (std::int64_t i = 0; i < s.count; ++i) {
    const std::array<std::int64_t, 3> at = unflatten(l, i);
    double u = 1.0;
    for (int d = 0; d < l.rank; ++d) {
      u *= axes[idx(d)].energy_weight(at.at(idx(d)), output);
    }
    const double v = data[idx(s.offset(b, i))];
    sum = sum + two_prod(u * v, v);
  }
  return sum.hi;
}

check_result check_r2r(const layout& l, const double* x, const double* y, std::int64_t budget,
                       std::uint64_t seed) {
  check_result r;
  const double eps = l.precision == fbench_f32 ? std::numeric_limits<float>::epsilon()
                                               : std::numeric_limits<double>::epsilon();
  std::vector<r2r_axis> axes;
  axes.reserve(idx(l.rank));
  double logical = 1.0;
  for (int d = 0; d < l.rank; ++d) {
    axes.emplace_back(l.r2r_kinds.at(idx(d)), l.dims.at(idx(d)));
    logical *= static_cast<double>(axes.back().logical());
  }
  r.tol = 10.0 * eps * (1.0 + std::log2(std::max(2.0, logical)));
  std::vector<std::int64_t> rows{0};
  if (l.batch > 1) {
    rows.push_back(l.batch - 1);
  }
  r.rows = static_cast<std::int64_t>(rows.size());
  const std::int64_t count = l.out.count;
  const std::int64_t wanted =
      std::clamp<std::int64_t>(budget / (std::max<std::int64_t>(1, count) * r.rows),
                               std::min<std::int64_t>(4, count), count);
  std::uint64_t rng = seed ^ 0x5eedULL;
  const int last = l.rank - 1;
  const std::int64_t n_last = l.dims.at(idx(last));
  const std::int64_t outer = count / n_last;
  double diff2 = 0.0;
  double ref2 = 0.0;
  double diff_max = 0.0;
  std::int64_t compared = 0;
  std::array<std::vector<dd>, 3> kernel;
  for (const std::int64_t b : rows) {
    const std::vector<std::int64_t> samples = pick_samples(count, wanted, rng);
    r.samples = static_cast<std::int64_t>(samples.size());
    for (const std::int64_t j : samples) {
      // The kernel of each axis at this output's index, then the separable sum.
      const std::array<std::int64_t, 3> k = unflatten(l, j);
      for (int d = 0; d < l.rank; ++d) {
        std::vector<dd>& row = kernel.at(idx(d));
        row.resize(idx(l.dims.at(idx(d))));
        for (std::int64_t i = 0; i < l.dims.at(idx(d)); ++i) {
          row[idx(i)] = axes[idx(d)].kernel(k.at(idx(d)), i);
        }
      }
      dd sum{};
      for (std::int64_t o = 0; o < outer; ++o) {
        dd outer_weight{1.0, 0.0};
        std::int64_t rest = o;
        for (int d = last - 1; d >= 0; --d) {
          const std::int64_t i = rest % l.dims.at(idx(d));
          rest /= l.dims.at(idx(d));
          outer_weight = outer_weight * kernel.at(idx(d))[idx(i)];
        }
        dd line{};
        const std::vector<dd>& row = kernel.at(idx(last));
        for (std::int64_t i = 0; i < n_last; ++i) {
          line = line + row[idx(i)] * x[idx(l.in.offset(b, o * n_last + i))];
        }
        sum = sum + (last > 0 ? line * outer_weight : line);
      }
      const double got = y[idx(l.out.offset(b, j))];
      const double d = (got - sum.hi) - sum.lo;
      diff2 += d * d;
      ref2 += sum.hi * sum.hi;
      diff_max =
          std::isnan(d) ? std::numeric_limits<double>::infinity() : std::max(diff_max, std::abs(d));
      ++compared;
    }
    const double e_in = r2r_energy(l, axes, l.in, x, b, false);
    const double e_out = r2r_energy(l, axes, l.out, y, b, true);
    const double expected = logical * e_in;
    const double p = std::abs(e_out - expected) / expected;
    r.parseval = std::isnan(p) ? std::numeric_limits<double>::infinity() : std::max(r.parseval, p);
  }
  r.err_rms = std::sqrt(diff2 / ref2);
  r.err_max = diff_max / std::sqrt(ref2 / static_cast<double>(compared));
  r.pass = r.err_rms <= r.tol && r.err_max <= 8.0 * r.tol && r.parseval <= r.tol;
  return r;
}

} // namespace

fbench_case layout::to_case() const {
  return fbench_case{
      .kind = kind,
      .rank = rank,
      .dims = dims.data(),
      .batch = interleave > 1 ? interleave : batch, // the transforms of one block
      .sign = sign,
      .in_place = in_place ? 1 : 0,
      .in_last = in.last_phys,
      .out_last = out.last_phys,
      .in_dist = in.dist,
      .out_dist = out.dist,
      .precision = precision,
      .interleave = interleave,
      .outer = outer,
      .outer_in_dist = in.block_dist,
      .outer_out_dist = out.block_dist,
      .r2r_kinds = kind == fbench_r2r ? r2r_kinds.data() : nullptr,
  };
}

layout make_layout(std::int32_t kind, int rank, const std::array<std::int64_t, 3>& dims,
                   std::int64_t batch, int sign, bool in_place, std::int32_t precision,
                   bool interleaved, std::int64_t outer,
                   const std::array<std::int32_t, 3>& r2r_kinds) {
  if (rank < 1 || rank > 3) {
    throw std::invalid_argument("rank must be 1, 2 or 3");
  }
  if (batch < 1) {
    throw std::invalid_argument("batch must be at least 1");
  }
  if (outer < 1 || (!interleaved && outer != 1)) {
    throw std::invalid_argument("outer needs the interleaved layout and must be at least 1");
  }
  if (interleaved && (batch < 2 || (in_place && kind != fbench_c2c && kind != fbench_r2r))) {
    throw std::invalid_argument("the interleaved layout needs batch >= 2 and not in-place r2c "
                                "or c2r");
  }
  if (precision != fbench_f64 && precision != fbench_f32) {
    throw std::invalid_argument("unknown precision");
  }
  layout l;
  l.precision = precision;
  l.real_bytes = precision == fbench_f32 ? sizeof(float) : sizeof(double);
  l.kind = kind;
  l.rank = rank;
  const std::int64_t interleave = interleaved ? batch : 1;
  l.batch = batch * outer;
  l.interleave = interleave;
  l.outer = outer;
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
    l.in = make_side(true, rank, l.dims, n_last, n_last, batch, l.real_bytes, interleave, outer);
    l.out = l.in;
  } else if (kind == fbench_r2c) {
    l.sign = -1;
    l.in = make_side(false, rank, l.dims, n_last, padded, batch, l.real_bytes, interleave, outer);
    l.out = make_side(true, rank, l.dims, half, half, batch, l.real_bytes, interleave, outer);
  } else if (kind == fbench_c2r) {
    l.sign = 1;
    l.in = make_side(true, rank, l.dims, half, half, batch, l.real_bytes, interleave, outer);
    l.out = make_side(false, rank, l.dims, n_last, padded, batch, l.real_bytes, interleave, outer);
  } else if (kind == fbench_r2r) {
    for (int d = 0; d < rank; ++d) {
      const std::int32_t k = r2r_kinds.at(idx(d));
      // The reference reduces its angles with products of two residues below 8n: n <= 2^27
      // keeps them in 64 bits.
      if (k < fbench_dct1 || k > fbench_dht || l.dims.at(idx(d)) > (std::int64_t{1} << 27) ||
          (k == fbench_dct1 && l.dims.at(idx(d)) < 2)) {
        throw std::invalid_argument("r2r needs a kind per axis, lengths up to 2^27 and dct1 "
                                    "from n = 2");
      }
      l.r2r_kinds.at(idx(d)) = k;
    }
    l.sign = -1;
    l.in = make_side(false, rank, l.dims, n_last, n_last, batch, l.real_bytes, interleave, outer);
    l.out = l.in;
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
  if (l.kind == fbench_r2r) {
    return check_r2r(l, static_cast<const double*>(in), static_cast<const double*>(out), budget,
                     seed);
  }
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

check_result check_delay(const layout& l, const void* in, const void* out, std::int64_t fanout,
                         bool valid) {
  const auto* x = static_cast<const double*>(in);
  const auto* y = static_cast<const double*>(out);
  check_result r;
  const bool real = l.kind == fbench_r2c;
  const std::int64_t n = l.dims.at(0);
  if (l.rank != 1 || (real && l.interleave != 1) || l.in_place || (l.kind != fbench_c2c && !real) ||
      fanout < 1 || l.batch % fanout != 0 || (valid && n % 4 != 0)) {
    return r; // not a layout the delayed check knows: never a pass
  }
  const double eps = l.precision == fbench_f32 ? std::numeric_limits<float>::epsilon()
                                               : std::numeric_limits<double>::epsilon();
  // Two transforms and a multiplication: twice the transform's bound.
  r.tol = 20.0 * eps * (1.0 + std::log2(static_cast<double>(l.n_total)));
  const std::int64_t m = n / 2;
  const std::int64_t count = valid ? m : n; // output values per row
  // An r2c case's output rows hold n reals from the start of each row of n/2 + 1 complex values
  // (FFTW's padded in-place layout of the real side).
  const auto out_value = [&](std::int64_t b, std::int64_t j) -> value {
    if (real) {
      return {y[idx(2 * l.out.offset(b, 0) + j)], 0.0};
    }
    return load(l.out, y, b, j);
  };
  // Logical position i of row b's line: the input row's value, or zero beyond the valid half.
  const auto line_value = [&](std::int64_t b, std::int64_t i) -> value {
    if (valid && i >= m) {
      return {0.0, 0.0};
    }
    return load(l.in, x, b, i);
  };
  std::vector<std::int64_t> rows{0};
  for (const std::int64_t b : {fanout - 1, l.batch - 1}) {
    if (b != rows.back()) {
      rows.push_back(b);
    }
  }
  r.rows = static_cast<std::int64_t>(rows.size());
  r.samples = count;
  double diff2 = 0.0;
  double ref2 = 0.0;
  double diff_max = 0.0;
  for (const std::int64_t b : rows) {
    const std::int64_t line = b / fanout * fanout;
    double e_in = 0.0;
    double e_out = 0.0;
    for (std::int64_t j = 0; j < count; ++j) {
      const std::int64_t logical = valid ? m / 2 + j : j;
      const value want = line_value(line, (logical + n - 1) % n);
      const value got = out_value(b, j);
      const double dre = got.re - want.re;
      const double dim = got.im - want.im;
      const double d2 = dre * dre + dim * dim;
      diff2 += d2;
      ref2 += want.re * want.re + want.im * want.im;
      diff_max = std::isnan(d2) ? std::numeric_limits<double>::infinity()
                                : std::max(diff_max, std::sqrt(d2));
      e_in += want.re * want.re + want.im * want.im;
      e_out += got.re * got.re + got.im * got.im;
    }
    const double p = std::abs(e_out - e_in) / e_in;
    r.parseval = std::isnan(p) ? std::numeric_limits<double>::infinity() : std::max(r.parseval, p);
  }
  r.err_rms = std::sqrt(diff2 / ref2);
  r.err_max = diff_max / std::sqrt(ref2 / static_cast<double>(r.rows * count));
  r.pass = r.err_rms <= r.tol && r.err_max <= 8.0 * r.tol && r.parseval <= r.tol;
  return r;
}

check_result check_identity(const layout& l, const void* in, const void* out) {
  const auto* x = static_cast<const double*>(in);
  const auto* y = static_cast<const double*>(out);
  check_result r;
  if (l.kind != fbench_c2c) {
    return r; // not a kind the identity check knows: never a pass
  }
  const double eps = l.precision == fbench_f32 ? std::numeric_limits<float>::epsilon()
                                               : std::numeric_limits<double>::epsilon();
  r.tol = 40.0 * eps * (1.0 + std::log2(static_cast<double>(l.n_total)));
  r.rows = l.batch;
  r.samples = l.n_total;
  double diff2 = 0.0;
  double ref2 = 0.0;
  double diff_max = 0.0;
  for (std::int64_t b = 0; b < l.batch; ++b) {
    double e_in = 0.0;
    double e_out = 0.0;
    for (std::int64_t j = 0; j < l.n_total; ++j) {
      const value want = load(l.in, x, b, j);
      const value got = load(l.out, y, b, j);
      const double dre = got.re - want.re;
      const double dim = got.im - want.im;
      const double d2 = dre * dre + dim * dim;
      diff2 += d2;
      ref2 += want.re * want.re + want.im * want.im;
      diff_max = std::isnan(d2) ? std::numeric_limits<double>::infinity()
                                : std::max(diff_max, std::sqrt(d2));
      e_in += want.re * want.re + want.im * want.im;
      e_out += got.re * got.re + got.im * got.im;
    }
    const double p = std::abs(e_out - e_in) / e_in;
    r.parseval = std::isnan(p) ? std::numeric_limits<double>::infinity() : std::max(r.parseval, p);
  }
  r.err_rms = std::sqrt(diff2 / ref2);
  r.err_max = diff_max / std::sqrt(ref2 / static_cast<double>(l.batch * l.n_total));
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

// The delayed check (check_delay): an exact delay passes, one wrong value, no delay and a delay
// of two fail, for complex rows and for real rows in an r2c case's output.
std::string check_delays() {
  for (const std::int32_t kind : {fbench_c2c, fbench_r2c}) {
    const layout l = make_layout(kind, 1, {12, 1, 1}, 3, -1, false, fbench_f64);
    std::vector<double> x(l.reals_in());
    fill_input(l, x.data(), 7);
    const auto write = [&](std::int64_t delay) {
      std::vector<double> y(l.reals_out(), 0.0);
      for (std::int64_t b = 0; b < l.batch; ++b) {
        for (std::int64_t j = 0; j < 12; ++j) {
          const value v = load(l.in, x.data(), b, (j + 12 - delay) % 12);
          if (kind == fbench_r2c) {
            y[idx(2 * l.out.offset(b, 0) + j)] = v.re;
          } else {
            y[idx(2 * l.out.offset(b, j))] = v.re;
            y[idx(2 * l.out.offset(b, j) + 1)] = v.im;
          }
        }
      }
      return y;
    };
    std::vector<double> y = write(1);
    if (!check_delay(l, x.data(), y.data()).pass) {
      return "the delayed check rejects an exact delay (kind " + std::to_string(kind) + ")";
    }
    y[idx(kind == fbench_r2c ? 2 * l.out.offset(2, 0) + 5 : 2 * l.out.offset(2, 5))] += 1e-9;
    if (check_delay(l, x.data(), y.data()).pass || check_delay(l, x.data(), write(0).data()).pass ||
        check_delay(l, x.data(), write(2).data()).pass) {
      return "the delayed check accepts a wrong output (kind " + std::to_string(kind) + ")";
    }
    // The valid halves of two copies per line (X67): 4 rows of 12, lines from rows 0 and 2.
    const layout v = make_layout(kind, 1, {12, 1, 1}, 4, -1, false, fbench_f64);
    std::vector<double> xv(v.reals_in());
    fill_input(v, xv.data(), 9);
    // Copy b's output from its line's input row (or its own row), delayed by `delay`.
    const auto halves = [&](std::int64_t delay, bool own_row) {
      std::vector<double> out(v.reals_out(), 0.0);
      for (std::int64_t b = 0; b < v.batch; ++b) {
        const std::int64_t line = own_row ? b : b / 2 * 2;
        for (std::int64_t j = 0; j < 6; ++j) {
          const std::int64_t i = 3 + j - delay;
          const value w = i < 6 ? load(v.in, xv.data(), line, i) : value{0.0, 0.0};
          if (kind == fbench_r2c) {
            out[idx(2 * v.out.offset(b, 0) + j)] = w.re;
          } else {
            out[idx(2 * v.out.offset(b, j))] = w.re;
            out[idx(2 * v.out.offset(b, j) + 1)] = w.im;
          }
        }
      }
      return out;
    };
    if (!check_delay(v, xv.data(), halves(1, false).data(), 2, true).pass) {
      return "the delayed check rejects exact valid halves of copies (kind " +
             std::to_string(kind) + ")";
    }
    if (check_delay(v, xv.data(), halves(1, true).data(), 2, true).pass ||
        check_delay(v, xv.data(), halves(0, false).data(), 2, true).pass ||
        check_delay(v, xv.data(), halves(1, false).data(), 1, false).pass) {
      return "the delayed check accepts wrong valid halves of copies (kind " +
             std::to_string(kind) + ")";
    }
  }
  // Interleaved c2c lines (a plane's strided axis): 3 lines of 12, one element apart, in 2 blocks.
  const layout il = make_layout(fbench_c2c, 1, {12, 1, 1}, 3, -1, false, fbench_f64, true, 2);
  std::vector<double> xi(il.reals_in());
  fill_input(il, xi.data(), 11);
  const auto shifted = [&](std::int64_t delay) {
    std::vector<double> out(il.reals_out(), 0.0);
    for (std::int64_t b = 0; b < il.batch; ++b) {
      for (std::int64_t j = 0; j < 12; ++j) {
        const value w = load(il.in, xi.data(), b, (j + 12 - delay) % 12);
        out[idx(2 * il.out.offset(b, j))] = w.re;
        out[idx(2 * il.out.offset(b, j) + 1)] = w.im;
      }
    }
    return out;
  };
  if (!check_delay(il, xi.data(), shifted(1).data()).pass) {
    return "the delayed check rejects an exact delay of interleaved lines";
  }
  if (check_delay(il, xi.data(), shifted(0).data()).pass ||
      check_delay(il, xi.data(), shifted(2).data()).pass) {
    return "the delayed check accepts a wrong delay of interleaved lines";
  }
  return {};
}

// Y_k of an impulse at j on one axis of n, written from FFTW's definitions as DESIGN.md 5.7
// tabulates them (in double, with std::cos and std::sin), independently of r2r_axis's reduced
// integer angles.
double r2r_impulse(std::int32_t kind, std::int64_t n, std::int64_t j, std::int64_t k) {
  const double pi = std::numbers::pi;
  const auto dj = static_cast<double>(j);
  const auto dk = static_cast<double>(k);
  const auto dn = static_cast<double>(n);
  switch (kind) {
  case fbench_dct1:
    if (j == 0) {
      return 1.0;
    }
    if (j == n - 1) {
      return k % 2 == 0 ? 1.0 : -1.0;
    }
    return 2.0 * std::cos(pi * dj * dk / (dn - 1.0));
  case fbench_dct2:
    return 2.0 * std::cos(pi * (dj + 0.5) * dk / dn);
  case fbench_dct3:
    return j == 0 ? 1.0 : 2.0 * std::cos(pi * dj * (dk + 0.5) / dn);
  case fbench_dct4:
    return 2.0 * std::cos(pi * (dj + 0.5) * (dk + 0.5) / dn);
  case fbench_dst1:
    return 2.0 * std::sin(pi * (dj + 1.0) * (dk + 1.0) / (dn + 1.0));
  case fbench_dst2:
    return 2.0 * std::sin(pi * (dj + 0.5) * (dk + 1.0) / dn);
  case fbench_dst3:
    if (j == n - 1) {
      return k % 2 == 0 ? 1.0 : -1.0;
    }
    return 2.0 * std::sin(pi * (dj + 1.0) * (dk + 0.5) / dn);
  case fbench_dst4:
    return 2.0 * std::sin(pi * (dj + 0.5) * (dk + 0.5) / dn);
  default: {
    const double a = 2.0 * pi * dj * dk / dn;
    return std::cos(a) + std::sin(a);
  }
  }
}

// The real-to-real check: an impulse's exact transform passes for every kind (rows of odd and
// even length, a plane of mixed kinds, the interleaved layout in place), and one wrong value in
// the last transform fails.
std::string check_r2r_impulses() {
  struct shape {
    int rank;
    std::array<std::int64_t, 3> dims;
    std::array<std::int32_t, 3> kinds;
    std::array<std::int64_t, 3> at;
    bool interleaved;
  };
  std::vector<shape> shapes;
  for (std::int32_t k = fbench_dct1; k <= fbench_dht; ++k) {
    shapes.push_back({1, {7, 1, 1}, {k, 0, 0}, {3, 0, 0}, false});
    shapes.push_back({1, {6, 1, 1}, {k, 0, 0}, {0, 0, 0}, false}); // j = 0: the weights' ends
    shapes.push_back({1, {8, 1, 1}, {k, 0, 0}, {6, 0, 0}, k % 2 == 0});
  }
  shapes.push_back({2, {5, 6, 1}, {fbench_dct2, fbench_dst3, 0}, {1, 5, 0}, false});
  shapes.push_back({3, {3, 4, 5}, {fbench_dct1, fbench_dht, fbench_dst4}, {2, 1, 3}, true});
  constexpr std::int64_t big_budget = std::int64_t{1024} * 1024;
  for (const shape& s : shapes) {
    const layout l = make_layout(fbench_r2r, s.rank, s.dims, 3, -1, s.interleaved, fbench_f64,
                                 s.interleaved, 1, s.kinds);
    std::vector<double> x(l.reals_in(), 0.0);
    std::vector<double> y(l.reals_out(), 0.0);
    std::int64_t p_flat = 0;
    for (int d = 0; d < s.rank; ++d) {
      p_flat = p_flat * s.dims.at(idx(d)) + s.at.at(idx(d));
    }
    for (std::int64_t b = 0; b < l.batch; ++b) {
      x[idx(l.in.offset(b, p_flat))] = 1.0;
      for (std::int64_t i = 0; i < l.out.count; ++i) {
        double v = 1.0;
        std::int64_t rest = i;
        for (int d = s.rank - 1; d >= 0; --d) {
          const std::int64_t n = s.dims.at(idx(d));
          v *= r2r_impulse(s.kinds.at(idx(d)), n, s.at.at(idx(d)), rest % n);
          rest /= n;
        }
        y[idx(l.out.offset(b, i))] = v;
      }
    }
    const std::string what = "r2r kind " + std::to_string(s.kinds[0]) + ", rank " +
                             std::to_string(s.rank) + ", n " + std::to_string(s.dims[0]);
    const check_result good = check_output(l, x.data(), y.data(), big_budget, 1);
    if (!good.pass || good.samples != l.out.count) {
      return "the check rejects an exact transform (" + what + ")";
    }
    y[idx(l.out.offset(2, 1))] += 1e-9;
    if (check_output(l, x.data(), y.data(), big_budget, 1).pass) {
      return "the check accepts a wrong transform (" + what + ")";
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
  if (std::string failed = check_impulses(); !failed.empty()) {
    return failed;
  }
  if (std::string failed = check_delays(); !failed.empty()) {
    return failed;
  }
  // The identity check: the input itself passes, in place and out of place; one wrong value fails.
  for (const bool in_place : {true, false}) {
    const layout l = make_layout(fbench_c2c, 3, {4, 3, 5}, 2, -1, in_place, fbench_f64);
    std::vector<double> x(l.reals_in());
    fill_input(l, x.data(), 11);
    std::vector<double> y(l.reals_out(), 0.0);
    for (std::int64_t b = 0; b < l.batch; ++b) {
      for (std::int64_t j = 0; j < l.n_total; ++j) {
        const value v = load(l.in, x.data(), b, j);
        y[idx(2 * l.out.offset(b, j))] = v.re;
        y[idx(2 * l.out.offset(b, j) + 1)] = v.im;
      }
    }
    if (!check_identity(l, x.data(), y.data()).pass) {
      return "the identity check rejects the input itself";
    }
    y[idx(2 * l.out.offset(1, 7))] += 1e-9;
    if (check_identity(l, x.data(), y.data()).pass) {
      return "the identity check accepts a wrong output";
    }
  }
  return check_r2r_impulses();
}

} // namespace fbench
