// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// A naive DFT in the precision under test, as a subject for the suites, and seeded-fault
// variants of it that the suites must catch (dev/PLAN.md task 0.4).
//
// The correct version evaluates X_k = sum_j x_j w[(k j) mod n] with a table of correctly
// rounded twiddles (from the reference) and pairwise summation, so its error grows like
// that of an FFT (eps * sqrt(log2 n) RMS) rather than like recursive summation's
// eps * sqrt(n), and the FFT tolerances apply to it. Real transforms go through the complex
// one: r2c keeps bins 0..n/2 of the transform of the real row; c2r extends the half
// spectrum to a Hermitian one (taking only the real parts of bins 0 and n/2) and keeps the
// real part of the backward transform. Planes transform rows, then columns.
//
// Each fault changes one thing:
//   wrong_sign          the exponent has the opposite sign
//   bin_shift           output bin k holds X_{k+1}: an off-by-one in the output index
//   scaled              the output is divided by n (an unwanted normalisation)
//   dropped_element     the last input element of each row is ignored
//   twiddle_base_ulp    the twiddle table is generated from the base twiddle exp(+-2 pi i/n)
//                       with its real part one ulp too large, each w[m] being the exact m-th
//                       power of that base (as a generator using powers or a recurrence
//                       would produce): the error of w[m] grows like m ulps, so it shows
//                       from a few dozen points on (tolerance.hpp)
//   twiddle_entry_ulp   one twiddle table entry, near n/8, is one ulp further from its
//                       exact value; only an entry-by-entry check of the table sees this
//   c2r_uses_dc_imag    c2r adds the imaginary part of bin 0 to every output instead of
//                       ignoring it
//   touches_gaps        every output element written is followed by a write of zero to the
//                       next element in memory, which for strided output is a gap
//   destroys_input      an out-of-place call overwrites its input with zeros

#pragma once

#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/transform_under_test.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fftn::test {

enum class fault : std::uint8_t {
  none,
  wrong_sign,
  bin_shift,
  scaled,
  dropped_element,
  twiddle_base_ulp,
  twiddle_entry_ulp,
  c2r_uses_dc_imag,
  touches_gaps,
  destroys_input,
};

inline constexpr std::array all_faults{
    fault::wrong_sign,       fault::bin_shift,        fault::scaled,
    fault::dropped_element,  fault::twiddle_base_ulp, fault::twiddle_entry_ulp,
    fault::c2r_uses_dc_imag, fault::touches_gaps,     fault::destroys_input,
};

inline std::string_view fault_name(fault f) {
  switch (f) {
  case fault::none:
    return "none";
  case fault::wrong_sign:
    return "wrong_sign";
  case fault::bin_shift:
    return "bin_shift";
  case fault::scaled:
    return "scaled";
  case fault::dropped_element:
    return "dropped_element";
  case fault::twiddle_base_ulp:
    return "twiddle_base_ulp";
  case fault::twiddle_entry_ulp:
    return "twiddle_entry_ulp";
  case fault::c2r_uses_dc_imag:
    return "c2r_uses_dc_imag";
  case fault::touches_gaps:
    return "touches_gaps";
  case fault::destroys_input:
    return "destroys_input";
  }
  return "?";
}

template <class T> class naive_dft {
public:
  using cplx = std::complex<T>;

  explicit naive_dft(fault f = fault::none) : fault_(f) {}

  // The table w[m] = exp(sign * 2 pi i m / n), m < n, as this subject uses it.
  [[nodiscard]] std::vector<cplx> twiddles(direction d, std::size_t n) const {
    const int sign = fault_ == fault::wrong_sign ? -sign_of(d) : sign_of(d);
    const auto& exact = reference::unit_roots(sign, n);
    std::vector<cplx> w;
    w.reserve(n);
    for (const exact_complex& v : exact) {
      w.push_back(round_to<T>(v));
    }
    if (fault_ == fault::twiddle_base_ulp && n > 1) {
      // base' = base + delta with delta one ulp of base's real part, and base'^m =
      // base^m (1 + delta conj(base))^m = base^m (1 + m eta) to first order, since
      // m |eta| is far below 1 for any n tested here.
      const std::complex<double> base(exact[1].re.hi, exact[1].im.hi);
      const T re = w[1].real();
      const double delta =
          static_cast<double>(std::nextafter(re, std::numeric_limits<T>::infinity())) -
          static_cast<double>(re);
      const std::complex<double> eta = delta * std::conj(base);
      for (std::size_t m = 0; m < n; ++m) {
        const std::complex<double> exact_m(exact[m].re.hi, exact[m].im.hi);
        const std::complex<double> v = exact_m + exact_m * (static_cast<double>(m) * eta);
        w[m] = {static_cast<T>(v.real()), static_cast<T>(v.imag())};
      }
    }
    if (fault_ == fault::twiddle_entry_ulp) {
      if (const auto m = perturbed_entry(sign, n)) {
        // Away from the exact value: the entry's error becomes more than one ulp.
        const exact_complex& v = exact[*m];
        T re = w[*m].real();
        T im = w[*m].imag();
        if (std::abs(error_of(re, v.re)) > 0.1 * ulp_of<T>(v.re.hi)) {
          re = std::nextafter(re, error_of(re, v.re) > 0 ? std::numeric_limits<T>::infinity()
                                                         : -std::numeric_limits<T>::infinity());
        } else {
          im = std::nextafter(im, error_of(im, v.im) > 0 ? std::numeric_limits<T>::infinity()
                                                         : -std::numeric_limits<T>::infinity());
        }
        w[*m] = {re, im};
      }
    }
    return w;
  }

  // The entry that twiddle_entry_ulp perturbs: the first from n/8 on with a component
  // whose rounding error is more than a tenth of an ulp, so that the fault leaves it at
  // least 1.1 ulps away. None when the whole table is exact (n = 1, 2, 4).
  static std::optional<std::size_t> perturbed_entry(int sign, std::size_t n) {
    const auto& exact = reference::unit_roots(sign, n);
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t m = (n / 8 + i) % n;
      const exact_complex& v = exact[m];
      const cplx r = round_to<T>(v);
      if (std::abs(error_of(r.real(), v.re)) > 0.1 * ulp_of<T>(v.re.hi) ||
          std::abs(error_of(r.imag(), v.im)) > 0.1 * ulp_of<T>(v.im.hi)) {
        return m;
      }
    }
    return std::nullopt;
  }

  // One transform of a contiguous row: out[k] = sum_j in[j] w[(k j) mod n]. `out` may be
  // `in`.
  void transform(const std::vector<cplx>& w, const cplx* in, cplx* out) const {
    const std::size_t n = w.size();
    std::vector<cplx> x(in, in + n);
    if (fault_ == fault::dropped_element && n > 0) {
      x[n - 1] = cplx{};
    }
    std::vector<cplx> result(n);
    std::vector<cplx> terms(n);
    for (std::size_t k = 0; k < n; ++k) {
      std::size_t idx = 0;
      for (std::size_t j = 0; j < n; ++j) {
        terms[j] = multiply(x[j], w[idx]);
        idx += k;
        if (idx >= n) {
          idx -= n;
        }
      }
      result[k] = pairwise_sum(terms);
    }
    for (std::size_t k = 0; k < n; ++k) {
      cplx v = fault_ == fault::bin_shift ? result[(k + 1) % n] : result[k];
      if (fault_ == fault::scaled) {
        v /= static_cast<T>(n);
      }
      out[k] = v;
    }
  }

  [[nodiscard]] subject<T> as_subject() const {
    subject<T> s;
    s.name = fault_ == fault::none ? "naive" : "naive_" + std::string(fault_name(fault_));
    const auto self = std::make_shared<const naive_dft>(*this);
    s.c2c = [self](direction d, const rows& r) -> c2c_call<T> {
      return [self, r, w = self->twiddles(d, r.n)](const cplx* in, cplx* out) {
        self->c2c_rows(w, r, in, out);
      };
    };
    s.r2c = [self](const rows& r) -> r2c_call<T> {
      return [self, r, w = self->twiddles(direction::forward, r.n)](const T* in, cplx* out) {
        self->r2c_rows(w, r, in, out);
      };
    };
    s.c2r = [self](const rows& r) -> c2r_call<T> {
      return [self, r, w = self->twiddles(direction::backward, r.n)](const cplx* in, T* out) {
        self->c2r_rows(w, r, in, out);
      };
    };
    s.c2c_planes = [self](direction d, const planes& p) -> c2c_call<T> {
      return [self, p, wy = self->twiddles(d, p.ny), wx = self->twiddles(d, p.nx)](
                 const cplx* in, cplx* out) { self->c2c_planes(wy, wx, p, in, out); };
    };
    return s;
  }

private:
  static cplx multiply(cplx a, cplx b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
  }

  // Pairwise (cascade) summation: each level adds neighbours, so a sum of n terms passes
  // through log2 n roundings rather than n.
  static cplx pairwise_sum(std::vector<cplx>& v) {
    std::size_t len = v.size();
    if (len == 0) {
      return {};
    }
    while (len > 1) {
      const std::size_t half = len / 2;
      for (std::size_t i = 0; i < half; ++i) {
        v[i] = v[2 * i] + v[2 * i + 1];
      }
      if (len % 2 != 0) {
        v[half] = v[len - 1];
      }
      len = half + len % 2;
    }
    return v[0];
  }

  template <class E> static E& at(E* base, std::ptrdiff_t offset) { return *(base + offset); }
  template <class E> static const E& at(const E* base, std::ptrdiff_t offset) {
    return *(base + offset);
  }

  // Writes one output element; with touches_gaps, also the element after it in memory when
  // that is a gap inside the row (strided output, not the row's last element).
  template <class E>
  void write(E* out, std::ptrdiff_t offset, std::ptrdiff_t stride, bool last, E v) const {
    at(out, offset) = v;
    if (fault_ == fault::touches_gaps && !last && (stride > 1 || stride < -1)) {
      at(out, offset + (stride > 0 ? 1 : -1)) = E{};
    }
  }

  template <class E> void destroy(const E* in, std::size_t len, const rows& r) const {
    if (fault_ != fault::destroys_input || r.place == placement::in_place) {
      return;
    }
    auto* writable = const_cast<E*>(in);
    for (std::size_t b = 0; b < r.count; ++b) {
      for (std::size_t j = 0; j < len; ++j) {
        at(writable, index(b, j, r.in_stride, r.in_dist)) = E{};
      }
    }
  }

  static std::ptrdiff_t index(std::size_t b, std::size_t j, std::ptrdiff_t stride,
                              std::ptrdiff_t dist) {
    return static_cast<std::ptrdiff_t>(b) * dist + static_cast<std::ptrdiff_t>(j) * stride;
  }

  void c2c_rows(const std::vector<cplx>& w, const rows& r, const cplx* in, cplx* out) const {
    std::vector<cplx> row(r.n);
    for (std::size_t b = 0; b < r.count; ++b) {
      for (std::size_t j = 0; j < r.n; ++j) {
        row[j] = at(in, index(b, j, r.in_stride, r.in_dist));
      }
      transform(w, row.data(), row.data());
      for (std::size_t k = 0; k < r.n; ++k) {
        write(out, index(b, k, r.out_stride, r.out_dist), r.out_stride, k + 1 == r.n, row[k]);
      }
    }
    destroy(in, r.n, r);
  }

  void r2c_rows(const std::vector<cplx>& w, const rows& r, const T* in, cplx* out) const {
    std::vector<cplx> row(r.n);
    for (std::size_t b = 0; b < r.count; ++b) {
      for (std::size_t j = 0; j < r.n; ++j) {
        row[j] = at(in, index(b, j, r.in_stride, r.in_dist));
      }
      transform(w, row.data(), row.data());
      for (std::size_t k = 0; k <= r.n / 2; ++k) {
        write(out, index(b, k, r.out_stride, r.out_dist), r.out_stride, k == r.n / 2, row[k]);
      }
    }
    destroy(in, r.n, r);
  }

  void c2r_rows(const std::vector<cplx>& w, const rows& r, const cplx* in, T* out) const {
    const std::size_t n = r.n;
    std::vector<cplx> row(n);
    for (std::size_t b = 0; b < r.count; ++b) {
      const cplx dc = at(in, index(b, 0, r.in_stride, r.in_dist));
      row[0] = {dc.real(), T(0)};
      for (std::size_t k = 1; k < (n + 1) / 2; ++k) {
        const cplx v = at(in, index(b, k, r.in_stride, r.in_dist));
        row[k] = v;
        row[n - k] = std::conj(v);
      }
      if (n % 2 == 0 && n > 0) {
        row[n / 2] = {at(in, index(b, n / 2, r.in_stride, r.in_dist)).real(), T(0)};
      }
      transform(w, row.data(), row.data());
      for (std::size_t j = 0; j < n; ++j) {
        T v = row[j].real();
        if (fault_ == fault::c2r_uses_dc_imag) {
          v += dc.imag();
        }
        write(out, index(b, j, r.out_stride, r.out_dist), r.out_stride, j + 1 == n, v);
      }
    }
    destroy(in, n / 2 + 1, r);
  }

  void c2c_planes(const std::vector<cplx>& wy, const std::vector<cplx>& wx, const planes& p,
                  const cplx* in, cplx* out) const {
    std::vector<cplx> plane(p.ny * p.nx);
    std::vector<cplx> column(p.ny);
    for (std::size_t b = 0; b < p.count; ++b) {
      const auto base = static_cast<std::ptrdiff_t>(b);
      for (std::size_t y = 0; y < p.ny; ++y) {
        for (std::size_t x = 0; x < p.nx; ++x) {
          plane[y * p.nx + x] =
              at(in, base * p.in_dist + static_cast<std::ptrdiff_t>(y) * p.in_y_stride +
                         static_cast<std::ptrdiff_t>(x) * p.in_x_stride);
        }
        transform(wx, plane.data() + y * p.nx, plane.data() + y * p.nx);
      }
      for (std::size_t x = 0; x < p.nx; ++x) {
        for (std::size_t y = 0; y < p.ny; ++y) {
          column[y] = plane[y * p.nx + x];
        }
        transform(wy, column.data(), column.data());
        for (std::size_t y = 0; y < p.ny; ++y) {
          plane[y * p.nx + x] = column[y];
        }
      }
      for (std::size_t y = 0; y < p.ny; ++y) {
        for (std::size_t x = 0; x < p.nx; ++x) {
          write(out,
                base * p.out_dist + static_cast<std::ptrdiff_t>(y) * p.out_y_stride +
                    static_cast<std::ptrdiff_t>(x) * p.out_x_stride,
                p.out_x_stride, x + 1 == p.nx, plane[y * p.nx + x]);
        }
      }
    }
    if (fault_ == fault::destroys_input && p.place == placement::out_of_place) {
      auto* writable = const_cast<cplx*>(in);
      for (std::size_t b = 0; b < p.count; ++b) {
        for (std::size_t y = 0; y < p.ny; ++y) {
          for (std::size_t x = 0; x < p.nx; ++x) {
            at(writable, static_cast<std::ptrdiff_t>(b) * p.in_dist +
                             static_cast<std::ptrdiff_t>(y) * p.in_y_stride +
                             static_cast<std::ptrdiff_t>(x) * p.in_x_stride) = cplx{};
          }
        }
      }
    }
  }

  fault fault_;
};

} // namespace fftn::test
