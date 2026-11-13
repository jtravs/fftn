// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "reference/reference_dft.hpp"

#include "support/exact.hpp"

// Boost.Multiprecision's umbrella header for cpp_bin_float is its documented entry point;
// include-cleaner would instead name the internal headers it happens to use.
#include <boost/multiprecision/cpp_bin_float.hpp> // NOLINT(misc-include-cleaner): as above

#include <complex>
#include <cstddef>
#include <map>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fftn::test::reference {
namespace {

// Everything this file takes from Boost: the type, pi, sin and cos (Multiprecision's own
// implementations in its standalone mode; no Boost.Math).
// NOLINTBEGIN(misc-include-cleaner): provided through the umbrella header, as above
using quad = boost::multiprecision::cpp_bin_float_quad;

const quad& half_pi() {
  static const quad value =
      quad(boost::multiprecision::default_ops::get_constant_pi<quad::backend_type>()) / 2;
  return value;
}

quad cosine(const quad& x) { return boost::multiprecision::cos(x); }
quad sine(const quad& x) { return boost::multiprecision::sin(x); }
// NOLINTEND(misc-include-cleaner)

struct qcomplex {
  quad re;
  quad im;
};

exact_real to_exact(const quad& v) {
  const auto hi = static_cast<double>(v);
  const auto lo = static_cast<double>(v - quad(hi));
  return {.hi = hi, .lo = lo};
}

exact_complex to_exact(const qcomplex& v) { return {.re = to_exact(v.re), .im = to_exact(v.im)}; }

void check_length(std::size_t n) {
  if (n == 0 || n > (std::size_t{1} << 32U)) {
    throw std::invalid_argument("reference: length 0 or above 2^32");
  }
}

// exp(sign * 2 pi i * m / n). The angle 2 pi m / n is split in integers into a quadrant
// q pi / 2 and a remainder (pi / 2) r / n, and the remainder is folded to at most pi / 4,
// so sin and cos only ever see a small argument and the symmetries hold exactly.
struct octant {
  std::size_t q = 0; // quadrant
  std::size_t r = 0; // the angle within it is (pi / 2) r / n, with r <= n / 2
  bool fold = false; // the angle was folded: cos and sin swap
};

octant split(std::size_t m, std::size_t n) {
  const std::size_t four_m = 4 * (m % n);
  octant o{.q = four_m / n, .r = four_m % n, .fold = false};
  if (2 * o.r > n) {
    o.r = n - o.r;
    o.fold = true;
  }
  return o;
}

quad angle(std::size_t r, std::size_t n) { return half_pi() * quad(r) / quad(n); }

// cos and sin of the folded angle, placed in the right quadrant with the right sign.
qcomplex place(const quad& cos_phi, const quad& sin_phi, octant o, int sign) {
  const quad& c = o.fold ? sin_phi : cos_phi;
  const quad& s = o.fold ? cos_phi : sin_phi;
  qcomplex w;
  switch (o.q) {
  case 0:
    w = {.re = c, .im = s};
    break;
  case 1:
    w = {.re = -s, .im = c};
    break;
  case 2:
    w = {.re = -c, .im = -s};
    break;
  default:
    w = {.re = s, .im = -c};
    break;
  }
  if (sign < 0) {
    w.im = -w.im;
  }
  return w;
}

qcomplex root(int sign, std::size_t m, std::size_t n) {
  check_length(n);
  const octant o = split(m, n);
  const quad phi = angle(o.r, n);
  return place(cosine(phi), sine(phi), o, sign);
}

// Twiddle tables are cached per (sign, n) for the life of the process: the suites ask for
// the same lengths again and again, and sin and cos are the costly part. A table needs
// them only for the n/2 + 1 folded angles, and the table of the other sign is its
// conjugate. Entries are bitwise those of root().
const std::vector<qcomplex>& table(int sign, std::size_t n) {
  check_length(n);
  static std::mutex lock;
  static std::map<std::pair<int, std::size_t>, std::vector<qcomplex>> cache;
  const std::scoped_lock guard(lock);
  if (const auto it = cache.find({sign, n}); it != cache.end()) {
    return it->second;
  }
  std::vector<qcomplex> w;
  w.reserve(n);
  if (const auto other = cache.find({-sign, n}); other != cache.end()) {
    for (const qcomplex& v : other->second) {
      w.push_back({.re = v.re, .im = -v.im});
    }
  } else {
    std::vector<quad> c;
    std::vector<quad> s;
    c.reserve(n / 2 + 1);
    s.reserve(n / 2 + 1);
    for (std::size_t r = 0; r <= n / 2; ++r) {
      const quad phi = angle(r, n);
      c.push_back(cosine(phi));
      s.push_back(sine(phi));
    }
    for (std::size_t m = 0; m < n; ++m) {
      const octant o = split(m, n);
      w.push_back(place(c[o.r], s[o.r], o, sign));
    }
  }
  return cache.emplace(std::pair{sign, n}, std::move(w)).first->second;
}

int checked_sign(int sign) {
  if (sign != -1 && sign != 1) {
    throw std::invalid_argument("reference: sign must be -1 or +1");
  }
  return sign;
}

// The nonzero entries of x, converted exactly; zeros are skipped, which makes impulse-like
// inputs cheap.
struct sparse_complex {
  std::vector<std::size_t> index;
  std::vector<qcomplex> value;
};

sparse_complex nonzeros(std::span<const std::complex<double>> x) {
  sparse_complex s;
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (x[j] != std::complex<double>{}) {
      s.index.push_back(j);
      s.value.push_back({.re = quad(x[j].real()), .im = quad(x[j].imag())});
    }
  }
  return s;
}

// sum_j x_j w[(k j) mod n] for one bin k.
qcomplex dft_bin(const sparse_complex& x, const std::vector<qcomplex>& w, std::size_t k) {
  const std::size_t n = w.size();
  const std::size_t step = k % n;
  qcomplex acc{.re = quad(0), .im = quad(0)};
  for (std::size_t i = 0; i < x.index.size(); ++i) {
    // Exact argument reduction: the index product is reduced modulo n in integers (it cannot
    // overflow, since n is at most 2^32).
    const std::size_t idx = (step * x.index[i]) % n;
    const qcomplex& t = w[idx];
    const qcomplex& v = x.value[i];
    acc.re += v.re * t.re - v.im * t.im;
    acc.im += v.re * t.im + v.im * t.re;
  }
  return acc;
}

std::vector<std::size_t> all_indices(std::size_t n) {
  std::vector<std::size_t> v(n);
  std::iota(v.begin(), v.end(), std::size_t{0});
  return v;
}

} // namespace

exact_complex unit_root(int sign, std::size_t m, std::size_t n) {
  return to_exact(root(checked_sign(sign), m, n));
}

const std::vector<exact_complex>& unit_roots(int sign, std::size_t n) {
  static std::mutex lock;
  static std::map<std::pair<int, std::size_t>, std::vector<exact_complex>> cache;
  const auto& w = table(checked_sign(sign), n);
  const std::scoped_lock guard(lock);
  auto [it, inserted] = cache.try_emplace({sign, n});
  if (inserted) {
    it->second.reserve(n);
    for (const qcomplex& v : w) {
      it->second.push_back(to_exact(v));
    }
  }
  return it->second;
}

std::vector<exact_complex> c2c(int sign, std::span<const std::complex<double>> x,
                               std::span<const std::size_t> bins) {
  std::vector<exact_complex> out;
  if (x.empty()) {
    return out;
  }
  const auto& w = table(checked_sign(sign), x.size());
  const auto sx = nonzeros(x);
  out.reserve(bins.size());
  for (const std::size_t k : bins) {
    out.push_back(to_exact(dft_bin(sx, w, k)));
  }
  return out;
}

std::vector<exact_complex> c2c(int sign, std::span<const std::complex<double>> x) {
  return c2c(sign, x, all_indices(x.size()));
}

std::vector<exact_complex> r2c(std::span<const double> x, std::span<const std::size_t> bins) {
  for (const std::size_t k : bins) {
    if (k > x.size() / 2) {
      throw std::invalid_argument("reference::r2c: bin above n/2");
    }
  }
  const std::vector<std::complex<double>> z(x.begin(), x.end());
  return c2c(-1, z, bins);
}

std::vector<exact_complex> r2c(std::span<const double> x) {
  return r2c(x, all_indices(x.size() / 2 + 1));
}

// x_j = Re X_0 + [n even] Re X_{n/2} (-1)^j + 2 sum_{0 < k < n/2} Re(X_k exp(+2 pi i j k / n)).
// Only real parts of bins 0 and n/2 enter, which is the convention that c2r ignores their
// imaginary parts.
std::vector<exact_real> c2r(std::size_t n, std::span<const std::complex<double>> X,
                            std::span<const std::size_t> indices) {
  if (X.size() != n / 2 + 1) {
    throw std::invalid_argument("reference::c2r: needs n/2 + 1 values");
  }
  std::vector<exact_real> out;
  if (n == 0) {
    return out;
  }
  const auto& w = table(+1, n);
  const std::size_t paired_end = (n + 1) / 2; // bins 1 .. paired_end - 1 have a mirror
  out.reserve(indices.size());
  for (const std::size_t j : indices) {
    quad acc = quad(X[0].real());
    if (n % 2 == 0) {
      const quad nyquist = quad(X[n / 2].real());
      acc += (j % 2 == 0) ? nyquist : quad(-nyquist);
    }
    quad pairs = 0;
    for (std::size_t k = 1; k < paired_end; ++k) {
      const qcomplex& t = w[(k * (j % n)) % n];
      pairs += quad(X[k].real()) * t.re - quad(X[k].imag()) * t.im;
    }
    acc += 2 * pairs;
    out.push_back(to_exact(acc));
  }
  return out;
}

std::vector<exact_real> c2r(std::size_t n, std::span<const std::complex<double>> X) {
  return c2r(n, X, all_indices(n));
}

std::vector<exact_complex> c2c_plane(int sign, std::size_t ny, std::size_t nx,
                                     std::span<const std::complex<double>> x,
                                     std::span<const std::pair<std::size_t, std::size_t>> points) {
  if (x.size() != ny * nx) {
    throw std::invalid_argument("reference::c2c_plane: size is not ny * nx");
  }
  std::vector<exact_complex> out;
  if (x.empty()) {
    return out;
  }
  const auto& wx = table(checked_sign(sign), nx);
  const auto& wy = table(sign, ny);
  std::vector<sparse_complex> rows;
  rows.reserve(ny);
  for (std::size_t y = 0; y < ny; ++y) {
    rows.push_back(nonzeros(x.subspan(y * nx, nx)));
  }
  // The transform along x at bin kx, for every row, is shared by all points with that kx.
  std::map<std::size_t, sparse_complex> columns;
  for (const auto& [ky, kx] : points) {
    static_cast<void>(ky);
    if (columns.contains(kx)) {
      continue;
    }
    sparse_complex col;
    for (std::size_t y = 0; y < ny; ++y) {
      const qcomplex v = dft_bin(rows[y], wx, kx);
      if (v.re != 0 || v.im != 0) {
        col.index.push_back(y);
        col.value.push_back(v);
      }
    }
    columns.emplace(kx, std::move(col));
  }
  out.reserve(points.size());
  for (const auto& [ky, kx] : points) {
    out.push_back(to_exact(dft_bin(columns.at(kx), wy, ky)));
  }
  return out;
}

std::vector<exact_complex> c2c_plane(int sign, std::size_t ny, std::size_t nx,
                                     std::span<const std::complex<double>> x) {
  std::vector<std::pair<std::size_t, std::size_t>> points;
  points.reserve(ny * nx);
  for (std::size_t ky = 0; ky < ny; ++ky) {
    for (std::size_t kx = 0; kx < nx; ++kx) {
      points.emplace_back(ky, kx);
    }
  }
  return c2c_plane(sign, ny, nx, x, points);
}

} // namespace fftn::test::reference
