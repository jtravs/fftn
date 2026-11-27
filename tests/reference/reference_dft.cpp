// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "reference/reference_dft.hpp"

#include "support/exact.hpp"

// Boost.Multiprecision's umbrella header for cpp_bin_float is its documented entry point;
// include-cleaner would instead name the internal headers it happens to use.
#include <boost/multiprecision/cpp_bin_float.hpp> // NOLINT(misc-include-cleaner): as above

#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
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

// Results are remembered for the life of the process, by the bits of the input and the list
// of outputs asked for: the suites that force every instruction set, both precisions' plans
// of one input, both directions and every edge variant ask for the same transform again and
// again (the same rows on each target, with scales and output factors applied afterwards),
// and at O(n^2) in 113-bit arithmetic those repeats were most of the sanitizer job's time.
// A stored result is the one the first evaluation gave, so nothing changes but the time. Of
// a full complex transform only the forward one is stored: bin k of the backward one is bin
// (n - k) mod n of the forward one, bit for bit, since both sum the same twiddle (w+[m] is
// the conjugate of w-[m] in the tables, and w-[n - m] is that conjugate exactly) in the same
// order (ReferenceDft.BothSignsShareOneEvaluation holds this). Each store is bounded and
// starts afresh when full.
namespace {

template <class V> class store {
public:
  explicit store(std::size_t limit_bytes) : limit_(limit_bytes) {}

  std::optional<std::vector<V>> find(const std::vector<std::uint64_t>& key) {
    const std::scoped_lock guard(lock_);
    if (const auto it = map_.find(key); it != map_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  // The caller returns its result too, so the store keeps a copy, made only when it is kept.
  void put(std::vector<std::uint64_t> key, const std::vector<V>& value) {
    const std::size_t size = key.size() * sizeof(std::uint64_t) + value.size() * sizeof(V);
    const std::scoped_lock guard(lock_);
    if (bytes_ + size > limit_) {
      map_.clear();
      bytes_ = 0;
    }
    if (size <= limit_ && map_.try_emplace(std::move(key), value).second) {
      bytes_ += size;
    }
  }

private:
  std::size_t limit_;
  std::mutex lock_;
  std::map<std::vector<std::uint64_t>, std::vector<V>> map_;
  std::size_t bytes_ = 0;
};

store<exact_complex>& complex_store() {
  static store<exact_complex> s(std::size_t{192} << 20U);
  return s;
}

store<exact_real>& real_store() {
  static store<exact_real> s(std::size_t{64} << 20U);
  return s;
}

// What a result depends on: which function (kind) and its integer argument (the sign, or
// c2r's length), the input's bits and the outputs asked for (none: all of them).
std::vector<std::uint64_t> key_of(std::uint64_t kind, std::uint64_t arg,
                                  std::span<const std::complex<double>> x,
                                  std::span<const std::size_t> outputs) {
  std::vector<std::uint64_t> key{kind, arg, x.size(), outputs.size()};
  key.reserve(key.size() + 2 * x.size() + outputs.size());
  for (const std::complex<double>& v : x) {
    key.push_back(std::bit_cast<std::uint64_t>(v.real()));
    key.push_back(std::bit_cast<std::uint64_t>(v.imag()));
  }
  key.insert(key.end(), outputs.begin(), outputs.end());
  return key;
}

constexpr std::uint64_t full_kind = 1;
constexpr std::uint64_t bins_kind = 2;
constexpr std::uint64_t c2r_kind = 3;

std::uint64_t sign_arg(int sign) { return sign < 0 ? 0 : 1; }

std::vector<exact_complex> evaluate_c2c(int sign, std::span<const std::complex<double>> x,
                                        std::span<const std::size_t> bins) {
  const auto& w = table(sign, x.size());
  const auto sx = nonzeros(x);
  std::vector<exact_complex> out;
  out.reserve(bins.size());
  for (const std::size_t k : bins) {
    out.push_back(to_exact(dft_bin(sx, w, k)));
  }
  return out;
}

// x_j = Re X_0 + [n even] Re X_{n/2} (-1)^j + 2 sum_{0 < k < n/2} Re(X_k exp(+2 pi i j k / n)).
// Only real parts of bins 0 and n/2 enter, which is the convention that c2r ignores their
// imaginary parts.
std::vector<exact_real> evaluate_c2r(std::size_t n, std::span<const std::complex<double>> X,
                                     std::span<const std::size_t> indices) {
  const auto& w = table(+1, n);
  const std::size_t paired_end = (n + 1) / 2; // bins 1 .. paired_end - 1 have a mirror
  std::vector<exact_real> out;
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

std::vector<exact_complex> with_sign(int sign, std::vector<exact_complex> forward) {
  if (sign < 0) {
    return forward;
  }
  const std::size_t n = forward.size();
  std::vector<exact_complex> out(n);
  for (std::size_t k = 0; k < n; ++k) {
    out[k] = forward[(n - k) % n];
  }
  return out;
}

} // namespace

std::vector<exact_complex> c2c(int sign, std::span<const std::complex<double>> x,
                               std::span<const std::size_t> bins) {
  if (x.empty()) {
    return {};
  }
  const int s = checked_sign(sign);
  std::vector<std::uint64_t> key = key_of(bins_kind, sign_arg(s), x, bins);
  if (auto found = complex_store().find(key)) {
    return std::move(*found);
  }
  std::vector<exact_complex> out = evaluate_c2c(s, x, bins);
  complex_store().put(std::move(key), out);
  return out;
}

std::vector<exact_complex> c2c(int sign, std::span<const std::complex<double>> x) {
  if (x.empty()) {
    return {};
  }
  const int s = checked_sign(sign);
  std::vector<std::uint64_t> key = key_of(full_kind, sign_arg(-1), x, {});
  if (auto found = complex_store().find(key)) {
    return with_sign(s, std::move(*found));
  }
  std::vector<exact_complex> forward = evaluate_c2c(-1, x, all_indices(x.size()));
  complex_store().put(std::move(key), forward);
  return with_sign(s, std::move(forward));
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

std::vector<exact_real> c2r(std::size_t n, std::span<const std::complex<double>> X,
                            std::span<const std::size_t> indices) {
  if (X.size() != n / 2 + 1) {
    throw std::invalid_argument("reference::c2r: needs n/2 + 1 values");
  }
  if (n == 0) {
    return {};
  }
  std::vector<std::uint64_t> key = key_of(c2r_kind, n, X, indices);
  if (auto found = real_store().find(key)) {
    return std::move(*found);
  }
  std::vector<exact_real> out = evaluate_c2r(n, X, indices);
  real_store().put(std::move(key), out);
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

namespace {

// The period of the angles of a kind's kernel on a line of n: every angle is 2 pi m / period
// for an integer m, so the kernel is read from the table of that period, with the index
// product reduced modulo it in integers.
std::size_t period(r2r_type t, std::size_t n) {
  switch (t) {
  case r2r_type::dct1:
    if (n < 2) {
      throw std::invalid_argument("reference::r2r: dct1 needs n >= 2");
    }
    return 2 * (n - 1);
  case r2r_type::dst1:
    return 2 * (n + 1);
  case r2r_type::dct4:
  case r2r_type::dst4:
    return 8 * n;
  case r2r_type::dht:
    return n;
  default:
    break;
  }
  return 4 * n;
}

// The coefficient of input j in output k of a line of n (DESIGN.md 5.7's table), from the table
// w of exp(-2 pi i m / period): cos is Re w, sin is -Im w.
quad coefficient(r2r_type t, std::size_t n, const std::vector<qcomplex>& w, std::size_t j,
                 std::size_t k) {
  const std::size_t p = w.size();
  const auto c = [&](std::size_t m) { return w[m % p].re; };
  const auto s = [&](std::size_t m) { return -w[m % p].im; };
  const quad two(2);
  switch (t) {
  case r2r_type::dct1:
    return (j == 0 || j == n - 1 ? quad(1) : two) * c(j * k);
  case r2r_type::dct2:
    return two * c(((2 * j) + 1) * k);
  case r2r_type::dct3:
    return (j == 0 ? quad(1) : two) * c(j * ((2 * k) + 1));
  case r2r_type::dct4:
    return two * c(((2 * j) + 1) * ((2 * k) + 1));
  case r2r_type::dst1:
    return two * s((j + 1) * (k + 1));
  case r2r_type::dst2:
    return two * s(((2 * j) + 1) * (k + 1));
  case r2r_type::dst3:
    return (j == n - 1 ? quad(1) : two) * s((j + 1) * ((2 * k) + 1));
  case r2r_type::dst4:
    return two * s(((2 * j) + 1) * ((2 * k) + 1));
  case r2r_type::dht:
    break;
  }
  return c(j * k) + s(j * k);
}

void check_type(r2r_type t) {
  if (static_cast<unsigned char>(t) > static_cast<unsigned char>(r2r_type::dht)) {
    throw std::invalid_argument("reference::r2r: not a kind");
  }
}

} // namespace

std::vector<exact_real> r2r(std::span<const r2r_type> kinds, std::span<const std::size_t> extents,
                            std::span<const double> x) {
  std::size_t total = 1;
  for (const std::size_t n : extents) {
    check_length(n);
    total *= n;
  }
  if (extents.empty() || x.size() != total || kinds.size() != extents.size()) {
    throw std::invalid_argument(
        "reference::r2r: size is not the product of the extents, or not one kind per axis");
  }
  std::vector<quad> v(x.begin(), x.end());
  std::vector<quad> line;
  std::vector<quad> done;
  // Axis d's lines: `inner` elements apart along it, `outer` blocks of n * inner.
  std::size_t inner = total;
  for (std::size_t d = 0; d < extents.size(); ++d) {
    const std::size_t n = extents[d];
    const r2r_type t = kinds[d];
    check_type(t);
    inner /= n;
    const std::size_t outer = total / (n * inner);
    const std::vector<qcomplex>& w = table(-1, period(t, n));
    line.resize(n);
    done.resize(n);
    for (std::size_t o = 0; o < outer; ++o) {
      for (std::size_t i = 0; i < inner; ++i) {
        const std::size_t base = o * n * inner + i;
        for (std::size_t j = 0; j < n; ++j) {
          line[j] = v[base + j * inner];
        }
        for (std::size_t k = 0; k < n; ++k) {
          quad acc = 0;
          for (std::size_t j = 0; j < n; ++j) {
            if (line[j] != 0) {
              acc += line[j] * coefficient(t, n, w, j, k);
            }
          }
          done[k] = acc;
        }
        for (std::size_t k = 0; k < n; ++k) {
          v[base + k * inner] = done[k];
        }
      }
    }
  }
  std::vector<exact_real> out;
  out.reserve(total);
  for (const quad& q : v) {
    out.push_back(to_exact(q));
  }
  return out;
}

std::vector<exact_real> r2r(r2r_type kind, std::span<const double> x,
                            std::span<const std::size_t> outputs) {
  const std::size_t n = x.size();
  check_length(n);
  check_type(kind);
  const std::vector<qcomplex>& w = table(-1, period(kind, n));
  std::vector<exact_real> out;
  out.reserve(outputs.size());
  for (const std::size_t k : outputs) {
    if (k >= n) {
      throw std::invalid_argument("reference::r2r: output index out of range");
    }
    quad acc = 0;
    for (std::size_t j = 0; j < n; ++j) {
      if (x[j] != 0) {
        acc += quad(x[j]) * coefficient(kind, n, w, j, k);
      }
    }
    out.push_back(to_exact(acc));
  }
  return out;
}

std::vector<exact_real> r2r_from_definition(r2r_type kind, std::span<const double> x) {
  const std::size_t n = x.size();
  check_length(n);
  check_type(kind);
  if (kind == r2r_type::dct1 && n < 2) {
    throw std::invalid_argument("reference::r2r_from_definition: dct1 needs n >= 2");
  }
  // The angles are pi m / d, d the definition's denominator.
  std::size_t d = 2 * n;
  switch (kind) {
  case r2r_type::dct1:
    d = n - 1;
    break;
  case r2r_type::dst1:
    d = n + 1;
    break;
  case r2r_type::dct4:
  case r2r_type::dst4:
    d = 4 * n;
    break;
  case r2r_type::dht:
    d = n;
    break;
  default:
    break;
  }
  const quad pi = 2 * half_pi();
  std::vector<quad> cos_of(2 * d);
  std::vector<quad> sin_of(2 * d);
  for (std::size_t m = 0; m < 2 * d; ++m) {
    const quad a = pi * quad(m) / quad(d);
    cos_of[m] = cosine(a);
    sin_of[m] = sine(a);
  }
  const auto c = [&](std::size_t m) { return cos_of[m % (2 * d)]; };
  const auto s = [&](std::size_t m) { return sin_of[m % (2 * d)]; };
  const auto sign = [](std::size_t k) { return k % 2 == 0 ? quad(1) : quad(-1); };
  std::vector<exact_real> out;
  out.reserve(n);
  for (std::size_t k = 0; k < n; ++k) {
    quad sum = 0;
    quad outside = 0;
    switch (kind) {
    case r2r_type::dct1: // x_0 + (-1)^k x_(n-1) + 2 sum_(j=1..n-2) x_j cos(pi j k / (n - 1))
      outside = quad(x[0]) + (sign(k) * quad(x[n - 1]));
      for (std::size_t j = 1; j + 1 < n; ++j) {
        sum += quad(x[j]) * c(j * k);
      }
      break;
    case r2r_type::dct2: // 2 sum x_j cos(pi (j + 1/2) k / n)
      for (std::size_t j = 0; j < n; ++j) {
        sum += quad(x[j]) * c(((2 * j) + 1) * k);
      }
      break;
    case r2r_type::dct3: // x_0 + 2 sum_(j=1..n-1) x_j cos(pi j (k + 1/2) / n)
      outside = quad(x[0]);
      for (std::size_t j = 1; j < n; ++j) {
        sum += quad(x[j]) * c(j * ((2 * k) + 1));
      }
      break;
    case r2r_type::dct4: // 2 sum x_j cos(pi (j + 1/2)(k + 1/2) / n)
      for (std::size_t j = 0; j < n; ++j) {
        sum += quad(x[j]) * c(((2 * j) + 1) * ((2 * k) + 1));
      }
      break;
    case r2r_type::dst1: // 2 sum x_j sin(pi (j + 1)(k + 1) / (n + 1))
      for (std::size_t j = 0; j < n; ++j) {
        sum += quad(x[j]) * s((j + 1) * (k + 1));
      }
      break;
    case r2r_type::dst2: // 2 sum x_j sin(pi (j + 1/2)(k + 1) / n)
      for (std::size_t j = 0; j < n; ++j) {
        sum += quad(x[j]) * s(((2 * j) + 1) * (k + 1));
      }
      break;
    case r2r_type::dst3: // (-1)^k x_(n-1) + 2 sum_(j=0..n-2) x_j sin(pi (j + 1)(k + 1/2) / n)
      outside = sign(k) * quad(x[n - 1]);
      for (std::size_t j = 0; j + 1 < n; ++j) {
        sum += quad(x[j]) * s((j + 1) * ((2 * k) + 1));
      }
      break;
    case r2r_type::dst4: // 2 sum x_j sin(pi (j + 1/2)(k + 1/2) / n)
      for (std::size_t j = 0; j < n; ++j) {
        sum += quad(x[j]) * s(((2 * j) + 1) * ((2 * k) + 1));
      }
      break;
    case r2r_type::dht: // sum x_j (cos + sin)(2 pi j k / n), the sum alone (no factor 2)
      for (std::size_t j = 0; j < n; ++j) {
        sum += quad(x[j]) * (c(2 * j * k) + s(2 * j * k));
      }
      out.push_back(to_exact(sum));
      continue;
    }
    out.push_back(to_exact(outside + (2 * sum)));
  }
  return out;
}

std::vector<exact_real> dht(std::span<const std::size_t> extents, std::span<const double> x) {
  const std::vector<r2r_type> kinds(extents.size(), r2r_type::dht);
  return r2r(kinds, extents, x);
}

} // namespace fftn::test::reference
