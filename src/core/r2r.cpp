// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "core/r2r.hpp"

#include "core/driver.hpp"
#include "core/edge_driver.hpp"
#include "core/edges.hpp"
#include "core/pipeline.hpp"
#include "core/plan_core.hpp"
#include "core/program.hpp"
#include "core/twiddles.hpp"
#include "core/wisdom_entry.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/edges.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "isa/kernel_table.hpp"
#include "passes/pass_args.hpp"
#include "planner/rules.hpp"
#include "strategies/chain.hpp"
#include "threading/schedule.hpp"
#include "threading/work_graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

[[noreturn]] void reject(const std::string& argument, const std::string& why) {
  throw std::invalid_argument("fftn::plan: " + argument + " " + why);
}

std::string axis_name(std::size_t d) { return "shape.axes[" + std::to_string(d) + "]"; }

const char* kind_text(r2r_kind k) {
  switch (k) {
  case r2r_kind::dct1:
    return "dct1";
  case r2r_kind::dct2:
    return "dct2";
  case r2r_kind::dct3:
    return "dct3";
  case r2r_kind::dct4:
    return "dct4";
  case r2r_kind::dst1:
    return "dst1";
  case r2r_kind::dst2:
    return "dst2";
  case r2r_kind::dst3:
    return "dst3";
  case r2r_kind::dst4:
    return "dst4";
  case r2r_kind::dht:
    break;
  }
  return "dht";
}

// The read stage of a DHT line of n reals (core/r2r.hpp): units [first, last) of the packed
// complex line in scratch (dst, stride 1), read at src (pass_args::src_stride reals apart) and,
// with Factor, each real times the input edge's factor (in_factor, src_qstride elements apart along
// the line). Even: unit j is x_2j + i x_2j+1 (h = n/2 units); otherwise x_j + 0i (n units).
// Baseline code, outside every ISA region, like the edge passes (core/edges.cpp); step 6.12 is to
// fuse it into the chain's first pass.
template <class T, bool Factor, bool Even>
void dht_read(const pass_args* a, const void* src, void* dst, std::size_t first, std::size_t last) {
  const auto* x = static_cast<const T*>(src);
  auto* z = static_cast<T*>(dst);
  const std::ptrdiff_t s = a->src_stride;
  const T* f = nullptr;
  if constexpr (Factor) {
    f = static_cast<const T*>(a->in_factor);
  }
  const std::ptrdiff_t fs = a->src_qstride;
  for (std::size_t k = first; k < last; ++k) {
    const auto i = static_cast<std::ptrdiff_t>(k);
    const std::ptrdiff_t j = Even ? 2 * i : i;
    T re = x[j * s];
    T im{};
    if constexpr (Even) {
      im = x[(j + 1) * s];
    }
    if constexpr (Factor) {
      re = re * f[j * fs];
      if constexpr (Even) {
        im = im * f[(j + 1) * fs];
      }
    }
    z[2 * i] = re;
    z[2 * i + 1] = im;
  }
}

// One output of the store stage: v times the output edge's factor (Factor: out_factor, src_qstride
// elements apart along the line) and the scales' product (Scaled), stored or, with Accumulate,
// added to the output's value.
template <class T, bool Factor, bool Scaled, bool Accumulate>
[[gnu::always_inline]] inline void put(const pass_args* a, T* y, std::size_t k, T v) {
  const auto i = static_cast<std::ptrdiff_t>(k);
  if constexpr (Factor) {
    v = v * static_cast<const T*>(a->out_factor)[i * a->src_qstride];
  }
  if constexpr (Scaled) {
    v = v * static_cast<T>(a->scale);
  }
  T* yi = y + i * a->dst_stride;
  if constexpr (Accumulate) {
    v = *yi + v;
  }
  *yi = v;
}

// The store stage of a DHT line of even n = 2h from Z (src, stride 1), the forward complex
// transform of h of the packed line: bin k of the line's real DFT is X_k = E_k + w^k O_k with
// E_k = (Z_k + conj Z_(h-k)) / 2 and O_k = -i (Z_k - conj Z_(h-k)) / 2 (indices modulo h; E and O
// are the transforms of the even and the odd reals) and w = exp(-2 pi i / n), w^k at
// pass_args::twiddles (k = 0 .. h, interleaved), and H_k = Re X_k - Im X_k,
// H_(n-k) = Re X_k + Im X_k. Unit k (0 .. h) writes H_k and, for 0 < k < h, H_(n-k), at dst
// (dst_stride reals apart): every output once. Written out rather than with std::complex's
// operator*, whose NaN recovery (C Annex G) is not the passes' arithmetic.
template <class T, bool Factor, bool Scaled, bool Accumulate>
void dht_store_even(const pass_args* a, const void* src, void* dst, std::size_t first,
                    std::size_t last) {
  const auto* z = static_cast<const T*>(src);
  auto* y = static_cast<T*>(dst);
  const auto* w = static_cast<const T*>(a->twiddles);
  const std::size_t n = a->n;
  const std::size_t h = n / 2;
  const T half = T(0.5);
  for (std::size_t k = first; k < last; ++k) {
    const std::size_t p = k == h ? 0 : k;
    const std::size_t q = k == 0 ? 0 : h - k;
    const T zr = z[2 * p];
    const T zi = z[2 * p + 1];
    const T cr = z[2 * q];
    const T ci = -z[2 * q + 1];
    const T er = (zr + cr) * half;
    const T ei = (zi + ci) * half;
    // O = -i (Z_k - conj Z_(h-k)) / 2.
    const T o_re = (zi - ci) * half;
    const T o_im = (cr - zr) * half;
    const T wr = w[2 * k];
    const T wi = w[2 * k + 1];
    const T xr = er + (wr * o_re - wi * o_im);
    const T xi = ei + (wr * o_im + wi * o_re);
    put<T, Factor, Scaled, Accumulate>(a, y, k, xr - xi);
    if (k != 0 && k != h) {
      put<T, Factor, Scaled, Accumulate>(a, y, n - k, xr + xi);
    }
  }
}

// The store stage of a DHT line of odd n from Z (src, stride 1), the forward complex transform
// of n of x + 0i, the line's real DFT whole: unit k writes H_k = Re Z_k - Im Z_k.
template <class T, bool Factor, bool Scaled, bool Accumulate>
void dht_store_odd(const pass_args* a, const void* src, void* dst, std::size_t first,
                   std::size_t last) {
  const auto* z = static_cast<const T*>(src);
  auto* y = static_cast<T*>(dst);
  for (std::size_t k = first; k < last; ++k) {
    put<T, Factor, Scaled, Accumulate>(a, y, k, z[2 * k] - z[2 * k + 1]);
  }
}

// The kinds' routes (DESIGN.md 5.7): the DHT's, Makhoul's for DCT-II and DST-II over the forward
// transform, and its inverse for DCT-III and DST-III over the backward one.
enum class route : unsigned char { dht, dct2, dct3, dst2, dst3 };

route route_of(r2r_kind k) {
  switch (k) {
  case r2r_kind::dct2:
    return route::dct2;
  case r2r_kind::dct3:
    return route::dct3;
  case r2r_kind::dst2:
    return route::dst2;
  case r2r_kind::dst3:
    return route::dst3;
  default:
    break;
  }
  return route::dht;
}

// Makhoul's order (J. Makhoul, "A fast cosine transform in one and multiple dimensions", IEEE
// Trans. ASSP 28, 1980): v_m = x_(2m) while 2m < n, then v_m = x_(2n-1-2m), the even-indexed
// reals ascending and the odd-indexed descending. The index in the line of v_m.
constexpr std::size_t makhoul(std::size_t m, std::size_t n) {
  return 2 * m < n ? 2 * m : (2 * n) - 1 - (2 * m);
}

// Real p of a line (src_stride reals apart), with Factor times the input edge's factor (f,
// src_qstride elements apart along the line).
template <class T, bool Factor>
[[gnu::always_inline]] inline T line_real(const pass_args* a, const T* x, const T* f,
                                          std::size_t p) {
  const auto i = static_cast<std::ptrdiff_t>(p);
  T v = x[i * a->src_stride];
  if constexpr (Factor) {
    v = v * f[i * a->src_qstride];
  }
  return v;
}

// The read stage of a DCT-II line (Sine: DST-II) of n reals: units [first, last) of v in
// Makhoul's order packed into scratch as the r2c packs a line (n even: h = n/2 units
// v_2j + i v_2j+1; odd: n units v_j + 0i). DST-II reads (-1)^p x_p, since
// dst2(x)_k = dct2((-1)^j x_j)_(n-1-k); the negation is exact.
template <class T, bool Sine, bool Factor, bool Even>
void dct2_read(const pass_args* a, const void* src, void* dst, std::size_t first,
               std::size_t last) {
  const auto* x = static_cast<const T*>(src);
  auto* z = static_cast<T*>(dst);
  const T* f = nullptr;
  if constexpr (Factor) {
    f = static_cast<const T*>(a->in_factor);
  }
  const std::size_t n = a->n;
  const auto v = [&](std::size_t m) {
    const std::size_t p = makhoul(m, n);
    const T r = line_real<T, Factor>(a, x, f, p);
    return Sine && (p % 2) != 0 ? -r : r;
  };
  for (std::size_t k = first; k < last; ++k) {
    if constexpr (Even) {
      z[2 * k] = v(2 * k);
      z[2 * k + 1] = v((2 * k) + 1);
    } else {
      z[2 * k] = v(k);
      z[2 * k + 1] = T{};
    }
  }
}

// W_k = conj(t_k) (a - i b) with t_k = exp(-i pi k / (2n)) = tr + i ti: the spectrum the
// DCT-III's inverse route feeds its c2r, from a = x_k and b = x_(n-k).
template <class T> [[gnu::always_inline]] inline void untwist(T a, T b, T tr, T ti, T& wr, T& wi) {
  wr = a * tr - b * ti;
  wi = -(a * ti + b * tr);
}

// The read stage of a DCT-III line (Sine: DST-III) of n reals. The DCT-III of x is the c2r of n
// of the Hermitian W_k = conj(t_k) (x_k - i x_(n-k)) (x_n = 0), read back in Makhoul's order
// (the store stage). n even: unit k (0 .. h-1, h = n/2) writes the c2r's packed bin
// Z_k = (W_k + conj W_(h-k)) + i conj(w^k) (W_k - conj W_(h-k)), w = exp(-2 pi i / n), whose
// backward transform of h is v_2j + i v_2j+1 with no halving: the factor 2 of the r2c's pairing
// inverted is the c2r's own. n odd: unit k (0 .. (n-1)/2) writes W_k and its conjugate at n - k.
// Twiddles (pass_args::twiddles): n even, per k = 0 .. h, w^k then t_k; odd, t_k. DST-III reads
// the line reversed, since dst3(x)_k = (-1)^k dct3(x reversed)_k.
template <class T, bool Sine, bool Factor, bool Even>
void dct3_read(const pass_args* a, const void* src, void* dst, std::size_t first,
               std::size_t last) {
  const auto* x = static_cast<const T*>(src);
  auto* z = static_cast<T*>(dst);
  const auto* w = static_cast<const T*>(a->twiddles);
  const T* f = nullptr;
  if constexpr (Factor) {
    f = static_cast<const T*>(a->in_factor);
  }
  const std::size_t n = a->n;
  const auto in = [&](std::size_t j) {
    return j == n ? T{} : line_real<T, Factor>(a, x, f, Sine ? n - 1 - j : j);
  };
  for (std::size_t k = first; k < last; ++k) {
    if constexpr (Even) {
      const std::size_t h = n / 2;
      const std::size_t q = h - k;
      T pr{};
      T pi{};
      T qr{};
      T qi{};
      untwist(in(k), in(n - k), w[4 * k + 2], w[4 * k + 3], pr, pi);
      untwist(in(q), in(n - q), w[4 * q + 2], w[4 * q + 3], qr, qi);
      const T ar = pr + qr;
      const T ai = pi - qi;
      const T br = pr - qr;
      const T bi = pi + qi;
      // C = conj(w^k) B, then Z = A + i C.
      const T wr = w[4 * k];
      const T wi = w[4 * k + 1];
      const T cr = wr * br + wi * bi;
      const T ci = wr * bi - wi * br;
      z[2 * k] = ar - ci;
      z[2 * k + 1] = ai + cr;
    } else {
      T pr{};
      T pi{};
      untwist(in(k), in(n - k), w[2 * k], w[2 * k + 1], pr, pi);
      z[2 * k] = pr;
      z[2 * k + 1] = pi;
      if (k != 0) {
        z[2 * (n - k)] = pr;
        z[2 * (n - k) + 1] = -pi;
      }
    }
  }
}

// The store stage of a DCT-II line (Sine: DST-II) of n reals from Z (src, stride 1), the
// forward complex transform of the read stage's packed line: Y_k = 2 Re(t_k V_k) and
// Y_(n-k) = -2 Im(t_k V_k), V the real DFT of v and t_k = exp(-i pi k / (2n)), one complex
// multiply per pair. n even: unit k (0 .. h) forms 2 V_k by the r2c's pairing of bins k and
// h - k without its halving, (Z_k + conj Z_(h-k)) - i w^k (Z_k - conj Z_(h-k)); odd: unit k
// (0 .. (n-1)/2) takes V_k = Z_k. Twiddles as dct3_read's. DST-II writes Y_k at n - 1 - k.
template <class T, bool Sine, bool Factor, bool Scaled, bool Accumulate, bool Even>
void dct2_store(const pass_args* a, const void* src, void* dst, std::size_t first,
                std::size_t last) {
  const auto* z = static_cast<const T*>(src);
  auto* y = static_cast<T*>(dst);
  const auto* w = static_cast<const T*>(a->twiddles);
  const std::size_t n = a->n;
  const std::size_t h = n / 2;
  const auto out = [&](std::size_t k, T v) {
    put<T, Factor, Scaled, Accumulate>(a, y, Sine ? n - 1 - k : k, v);
  };
  for (std::size_t k = first; k < last; ++k) {
    T pr{};
    T pi{};
    if constexpr (Even) {
      const std::size_t p = k == h ? 0 : k;
      const std::size_t q = k == 0 ? 0 : h - k;
      const T zr = z[2 * p];
      const T zi = z[2 * p + 1];
      const T cr = z[2 * q];
      const T ci = -z[2 * q + 1];
      // 2E = Z_p + conj Z_q, 2O = -i (Z_p - conj Z_q), 2V = 2E + w^k 2O.
      const T er = zr + cr;
      const T ei = zi + ci;
      const T o_re = zi - ci;
      const T o_im = cr - zr;
      const T wr = w[4 * k];
      const T wi = w[4 * k + 1];
      const T vr = er + (wr * o_re - wi * o_im);
      const T vi = ei + (wr * o_im + wi * o_re);
      const T tr = w[4 * k + 2];
      const T ti = w[4 * k + 3];
      pr = tr * vr - ti * vi;
      pi = tr * vi + ti * vr;
    } else {
      const T zr = z[2 * k];
      const T zi = z[2 * k + 1];
      const T tr = w[2 * k];
      const T ti = w[2 * k + 1];
      pr = T(2) * (tr * zr - ti * zi);
      pi = T(2) * (tr * zi + ti * zr);
    }
    out(k, pr);
    if (k != 0 && (!Even || k != h)) {
      out(n - k, -pi);
    }
  }
}

// The store stage of a DCT-III line (Sine: DST-III) of n reals from the backward complex
// transform of the read stage's line: v in Makhoul's order, written back to the line's order.
// n even: unit j (0 .. h-1) holds v_2j + i v_2j+1; odd: unit j (0 .. n-1) holds v_j in its real
// part. DST-III negates the odd-indexed outputs ((-1)^k), exactly.
template <class T, bool Sine, bool Factor, bool Scaled, bool Accumulate, bool Even>
void dct3_store(const pass_args* a, const void* src, void* dst, std::size_t first,
                std::size_t last) {
  const auto* z = static_cast<const T*>(src);
  auto* y = static_cast<T*>(dst);
  const std::size_t n = a->n;
  const auto out = [&](std::size_t m, T v) {
    const std::size_t p = makhoul(m, n);
    put<T, Factor, Scaled, Accumulate>(a, y, p, Sine && (p % 2) != 0 ? -v : v);
  };
  for (std::size_t j = first; j < last; ++j) {
    if constexpr (Even) {
      out(2 * j, z[2 * j]);
      out((2 * j) + 1, z[2 * j + 1]);
    } else {
      out(j, z[2 * j]);
    }
  }
}

template <class T, route R, bool Factor, bool Even> pass_fn read_of() {
  if constexpr (R == route::dht) {
    return &dht_read<T, Factor, Even>;
  } else if constexpr (R == route::dct2 || R == route::dst2) {
    return &dct2_read<T, R == route::dst2, Factor, Even>;
  } else {
    return &dct3_read<T, R == route::dst3, Factor, Even>;
  }
}

template <class T, route R> pass_fn read_kernel_of(bool factor, bool even) {
  if (even) {
    return factor ? read_of<T, R, true, true>() : read_of<T, R, false, true>();
  }
  return factor ? read_of<T, R, true, false>() : read_of<T, R, false, false>();
}

template <class T> pass_fn read_kernel(route r, bool factor, bool even) {
  switch (r) {
  case route::dct2:
    return read_kernel_of<T, route::dct2>(factor, even);
  case route::dct3:
    return read_kernel_of<T, route::dct3>(factor, even);
  case route::dst2:
    return read_kernel_of<T, route::dst2>(factor, even);
  case route::dst3:
    return read_kernel_of<T, route::dst3>(factor, even);
  case route::dht:
    break;
  }
  return read_kernel_of<T, route::dht>(factor, even);
}

template <class T, route R, bool Factor, bool Scaled, bool Accumulate, bool Even>
pass_fn store_of() {
  if constexpr (R == route::dht) {
    return Even ? &dht_store_even<T, Factor, Scaled, Accumulate>
                : &dht_store_odd<T, Factor, Scaled, Accumulate>;
  } else if constexpr (R == route::dct2 || R == route::dst2) {
    return &dct2_store<T, R == route::dst2, Factor, Scaled, Accumulate, Even>;
  } else {
    return &dct3_store<T, R == route::dst3, Factor, Scaled, Accumulate, Even>;
  }
}

template <class T, route R, bool Factor, bool Scaled>
pass_fn store_kernel_of(bool accumulate, bool even) {
  if (accumulate) {
    return even ? store_of<T, R, Factor, Scaled, true, true>()
                : store_of<T, R, Factor, Scaled, true, false>();
  }
  return even ? store_of<T, R, Factor, Scaled, false, true>()
              : store_of<T, R, Factor, Scaled, false, false>();
}

template <class T, route R>
pass_fn store_kernel_of(bool factor, bool scaled, bool accumulate, bool even) {
  if (factor) {
    return scaled ? store_kernel_of<T, R, true, true>(accumulate, even)
                  : store_kernel_of<T, R, true, false>(accumulate, even);
  }
  return scaled ? store_kernel_of<T, R, false, true>(accumulate, even)
                : store_kernel_of<T, R, false, false>(accumulate, even);
}

template <class T>
pass_fn store_kernel(route r, bool factor, bool scaled, bool accumulate, bool even) {
  switch (r) {
  case route::dct2:
    return store_kernel_of<T, route::dct2>(factor, scaled, accumulate, even);
  case route::dct3:
    return store_kernel_of<T, route::dct3>(factor, scaled, accumulate, even);
  case route::dst2:
    return store_kernel_of<T, route::dst2>(factor, scaled, accumulate, even);
  case route::dst3:
    return store_kernel_of<T, route::dst3>(factor, scaled, accumulate, even);
  case route::dht:
    break;
  }
  return store_kernel_of<T, route::dht>(factor, scaled, accumulate, even);
}

// The twiddles of a DCT or DST line of n (dct3_read, dct2_store): n even, per k = 0 .. n/2 the
// pairing's w^k = exp(-2 pi i k / n) then t_k = exp(-i pi k / (2n)); odd, t_k for
// k = 0 .. (n-1)/2. Each correctly rounded (core/twiddles.hpp).
template <class T> std::vector<T> dct_twiddles(std::size_t n) {
  std::vector<T> table;
  if (n % 2 == 0) {
    const std::size_t h = n / 2;
    const std::vector<std::complex<T>> w = unit_roots<T>(-1, n);
    table.reserve(4 * (h + 1));
    for (std::size_t k = 0; k <= h; ++k) {
      const std::complex<T> t = unit_root<T>(-1, k, 4 * n);
      table.insert(table.end(), {w[k].real(), w[k].imag(), t.real(), t.imag()});
    }
  } else {
    table.reserve(n + 1);
    for (std::size_t k = 0; 2 * k < n; ++k) {
      const std::complex<T> t = unit_root<T>(-1, k, 4 * n);
      table.insert(table.end(), {t.real(), t.imag()});
    }
  }
  return table;
}

std::uint32_t chunks_of(std::size_t units, std::size_t chunk_points) {
  return static_cast<std::uint32_t>(
      std::clamp<std::size_t>(units / std::max<std::size_t>(chunk_points, 1), 1,
                              std::numeric_limits<std::uint32_t>::max()));
}

// The edges of a real-to-real plan as it applies them: the input's real factor in the first
// phase's read stage, the output's real factor and accumulation in the last phase's store
// stage, and every scale as one product in that store stage (the transform is linear, so where
// the scale applies changes only the rounding).
struct r2r_edges {
  edge_side in;
  edge_side out;
  double scale = 1.0;
};

r2r_edges resolve(const shape_view& s, const options& opts, const edges_view* e) {
  check_edge_scale(opts.scale, s.prec, "options.scale");
  r2r_edges r;
  r.scale = opts.scale;
  if (e == nullptr) {
    return r;
  }
  // side_of (core/edges.cpp) refuses a complex factor on either side: both are real.
  r.in = resolve_edge_side(s, e->in, false, "edges.in");
  r.out = resolve_edge_side(s, e->out, true, "edges.out");
  if (opts.scale != 1.0 && e->out.scale != 1.0) {
    reject("edges.out.scale", "and options.scale are both given; options.scale is a shortcut for "
                              "edges.out.scale, so give one of them");
  }
  r.scale = opts.scale * e->in.scale * e->out.scale;
  if (!scale_fits_precision(r.scale, s.prec)) {
    reject("edges.in.scale", "times edges.out.scale and options.scale is not finite or outside "
                             "the range of the plan's precision");
  }
  if (r.out.accumulate && opts.in_place) {
    reject("edges.out.accumulate", "is refused on an in-place plan, whose output's prior values "
                                   "are its input (the maintainer, 2026-11-25)");
  }
  return r;
}

// A phase's view of one slot: where it is and its strides per axis, in reals.
struct view {
  slot where = slot::in;
  std::vector<std::ptrdiff_t> stride;
};

// The bytes one step along axis e moves the slots of src and dst (a loop's step: in, out,
// block, twiddles).
std::array<std::ptrdiff_t, moving_slots> step_along(std::size_t e, const view& src, const view& dst,
                                                    std::ptrdiff_t real_bytes) {
  std::array<std::ptrdiff_t, moving_slots> step{};
  for (const view* v : {&src, &dst}) {
    const std::ptrdiff_t bytes = v->stride[e] * real_bytes;
    switch (v->where) {
    case slot::in:
      step[0] = bytes;
      break;
    case slot::out:
      step[1] = bytes;
      break;
    case slot::block:
      step[2] = bytes;
      break;
    case slot::a:
    case slot::b:
      break;
    }
  }
  return step;
}

template <class T> plan_tables<T>& tables_of(plan_core& c) {
  if constexpr (std::is_same_v<T, double>) {
    return c.tables_f64;
  } else {
    return c.tables_f32;
  }
}

const char* place_text(slot src, slot dst) {
  if (src == slot::in) {
    return dst == slot::out ? "from the input into the output" : "from the input into the block";
  }
  if (dst != slot::out) {
    return "in place in the block";
  }
  return src == slot::block ? "from the block into the output" : "in place on the output";
}

template <class T>
void build(plan_core& c, const shape_view& s, const r2r_layout& l, const r2r_edges& e,
           const build_settings& settings) {
  const target_profile profile =
      plan_profile(c, of<T>(*find_kernels(c.isa_target)).complex_lanes, settings);
  chain_builder<T> chain = plan_chain<T>(c, profile, settings);
  program& p = c.prog;
  const auto rb = static_cast<std::ptrdiff_t>(sizeof(T));
  const std::size_t r = s.axes.size();
  p.transforms = 1;
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    p.records.emplace_back(loop{
        .kind = loop_kind::batch,
        .count = s.batches[b].count,
        .step = {l.in_dist[b] * rb, l.out_dist[b] * rb, 0},
    });
    p.transforms *= s.batches[b].count;
  }
  p.batch_loops = s.batches.size();

  // The lines loops of a phase run widest stride first (the order a C array lists its
  // dimensions in); the phases run narrowest first, so that the contiguous axis of a C-ordered
  // array goes first, from the input. Axes of one point need no phase of their own; a transform
  // of one point is one phase over its last axis, the edges' copy.
  std::vector<std::size_t> order(r);
  std::iota(order.begin(), order.end(), std::size_t{0});
  const auto magnitude = [](std::ptrdiff_t v) { return v < 0 ? -v : v; };
  std::ranges::stable_sort(order, [&](std::size_t x, std::size_t y) {
    return std::tuple(magnitude(l.out_stride[x]), magnitude(l.in_stride[x])) >
           std::tuple(magnitude(l.out_stride[y]), magnitude(l.in_stride[y]));
  });
  std::vector<std::size_t> phases;
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    if (s.axes[*it].n > 1) {
      phases.push_back(*it);
    }
  }
  if (phases.empty()) {
    phases.push_back(r - 1);
  }
  // An axis of one point without a phase is the identity for the DHT, DCT-III and DST-III, and
  // multiplies by 2 for DCT-II and DST-II (Y_0 = 2 x_0): an exact factor, which joins the scales'
  // product in the last store stage.
  double gain = 1.0;
  for (std::size_t d = 0; d < r; ++d) {
    if (s.axes[d].n == 1 && std::ranges::find(phases, d) == phases.end()) {
      const route k = route_of(s.kinds[d]);
      gain *= k == route::dct2 || k == route::dst2 ? 2.0 : 1.0;
    }
  }
  if (!scale_fits_precision(e.scale * gain, s.prec)) {
    reject("options.scale", "times the factor 2 of each DCT-II or DST-II axis of one point is "
                            "outside the range of the plan's precision");
  }

  const view in_view{.where = slot::in, .stride = l.in_stride};
  const view out_view{.where = slot::out, .stride = l.out_stride};
  // With accumulation over several phases, the intermediate is a whole transform of reals in the
  // block, C-ordered by the shape's axes; otherwise the output.
  const bool through_block = e.out.accumulate && phases.size() > 1;
  std::vector<std::ptrdiff_t> block_stride(r);
  std::size_t transform_reals = 1;
  for (std::size_t i = r; i > 0; --i) {
    block_stride[i - 1] = static_cast<std::ptrdiff_t>(transform_reals);
    transform_reals *= s.axes[i - 1].n;
  }
  const view mid_view =
      through_block ? view{.where = slot::block, .stride = std::move(block_stride)} : out_view;
  const bool in_factor = e.in.kind != factor_kind::none;
  const bool out_factor = e.out.kind != factor_kind::none;

  std::vector<std::pair<std::size_t, factor_steps>> line_loops;
  std::size_t scratch = 1;
  const std::size_t chunk = settings.chunk_points;
  for (std::size_t i = 0; i < phases.size(); ++i) {
    const std::size_t d = phases[i];
    const bool first = i == 0;
    const bool last = i + 1 == phases.size();
    const view& src = first ? in_view : mid_view;
    const view& dst = last ? out_view : mid_view;
    std::vector<std::size_t> open;
    for (const std::size_t ax : order) {
      if (ax == d || s.axes[ax].n == 1) {
        continue;
      }
      factor_steps f{};
      if (first && in_factor) {
        f[0] = e.in.axis_strides[ax] * static_cast<std::ptrdiff_t>(e.in.factor_bytes);
      }
      if (last && out_factor) {
        f[1] = e.out.axis_strides[ax] * static_cast<std::ptrdiff_t>(e.out.factor_bytes);
      }
      open.push_back(p.records.size());
      line_loops.emplace_back(p.records.size(), f);
      p.records.emplace_back(loop{
          .kind = loop_kind::lines,
          .count = s.axes[ax].n,
          .step = step_along(ax, src, dst, rb),
      });
    }
    const std::size_t n = s.axes[d].n;
    const r2r_kind kind = s.kinds[d];
    const route rt = route_of(kind);
    const bool dht = rt == route::dht;
    const bool second = rt == route::dct2 || rt == route::dst2;
    const bool sine = rt == route::dst2 || rt == route::dst3;
    const std::string kname = kind_text(kind);
    const bool even = n % 2 == 0;
    const std::size_t m = even ? n / 2 : n;
    const std::size_t half_units = (n + 1) / 2;
    const bool read_factor = first && in_factor;
    const bool store_factor = last && out_factor;
    // DCT-III and DST-III: Makhoul's route inverted, over the backward transform.
    const int sign = dht || second ? -1 : 1;
    const T* dct_table = nullptr;
    if (!dht) {
      const std::vector<T> twiddles = dct_twiddles<T>(n);
      auto& table = tables_of<T>(c).emplace_back(twiddles.begin(), twiddles.end());
      dct_table = table.data();
    }

    std::vector<stage> line;
    stage rd;
    rd.run = read_kernel<T>(rt, read_factor, even);
    rd.args.src_stride = src.stride[d];
    rd.args.src_qstride = read_factor ? e.in.axis_strides[d] : 0;
    rd.args.n = n;
    rd.src = src.where;
    rd.dst = slot::a;
    rd.units = dht || second || even ? m : half_units;
    rd.chunks = chunks_of(rd.units, chunk);
    rd.moving_factors = read_factor ? moving_in_factor : 0;
    const std::string packing =
        even ? " as " + std::to_string(m) + " complex pairs" : " as complex values";
    if (dht) {
      rd.name = "r2r read: dht line packed into scratch" + packing;
    } else if (second) {
      rd.name = "r2r read: " + kname + " line" + (sine ? " with alternating signs" : "") +
                " in Makhoul's order into scratch" + packing;
    } else {
      rd.args.twiddles = dct_table;
      rd.name = "r2r read: " + kname + " line" + (sine ? " reversed" : "") +
                " twiddled into its Hermitian spectrum" +
                (even ? ", paired into the c2r's " + std::to_string(m) + " complex values"
                      : " of " + std::to_string(n) + " complex values") +
                " in scratch";
    }
    if (read_factor) {
      rd.name += ", times the input edge's real factor";
    }
    line.push_back(std::move(rd));

    std::size_t need = 0;
    const slot z = chain.scratch_transform(m, sign, need, line);
    scratch = std::max(scratch, need);

    const double scale = last ? e.scale * gain : 1.0;
    const bool scaled = scale != 1.0;
    const bool accumulate = last && e.out.accumulate;
    stage st;
    st.run = store_kernel<T>(rt, store_factor, scaled, accumulate, even);
    st.args.dst_stride = dst.stride[d];
    st.args.src_qstride = store_factor ? e.out.axis_strides[d] : 0;
    st.args.n = n;
    st.args.scale = scaled ? scale : 1.0;
    if (dht && even) {
      const std::vector<std::complex<T>> w = unit_roots<T>(-1, n);
      auto& table = tables_of<T>(c).emplace_back(2 * (m + 1));
      for (std::size_t k = 0; k <= m; ++k) {
        table[2 * k] = w[k].real();
        table[2 * k + 1] = w[k].imag();
      }
      st.args.twiddles = table.data();
    } else if (second) {
      st.args.twiddles = dct_table;
    }
    st.src = z;
    st.dst = dst.where;
    if (dht) {
      st.units = even ? m + 1 : n;
    } else if (second) {
      st.units = even ? m + 1 : half_units;
    } else {
      st.units = m;
    }
    st.chunks = chunks_of(st.units, chunk);
    st.moving_factors = store_factor ? moving_out_factor : 0;
    std::string store_what;
    if (dht) {
      store_what = even ? "the r2c pairing and the Hartley combination" : "the Hartley combination";
    } else if (second) {
      store_what = std::string(even ? "the r2c pairing and the " : "the ") + kname + " twiddle" +
                   (sine ? ", reversed" : "");
    } else {
      store_what = std::string(even ? "the c2r's unpacking" : "the real parts") +
                   " from Makhoul's order" + (sine ? ", with alternating signs" : "");
    }
    st.name = "r2r store: " + store_what +
              (store_factor ? ", times the output edge's real factor" : "") +
              (scaled ? ", scaled" : "") + (accumulate ? ", accumulated" : "");
    line.push_back(std::move(st));
    for (stage& x : line) {
      p.records.emplace_back(std::move(x));
    }
    for (const std::size_t at : open) {
      std::get<loop>(p.records[at]).body_end = static_cast<std::uint32_t>(p.records.size());
    }
    std::string route_text;
    if (dht) {
      route_text = even ? " (the r2c pairing fused with the Hartley combination)"
                        : " (the Hartley combination)";
    } else if (second) {
      route_text = std::string(" (") + (even ? "the r2c pairing fused with " : "") +
                   "Makhoul's twiddle" + (sine ? ", the sine kind by signs and reversal" : "") +
                   ")";
    } else {
      route_text = std::string(" (") + (even ? "the c2r's unpacking" : "the real parts") +
                   " back from Makhoul's order" +
                   (sine ? ", the sine kind by reversal and signs" : "") + ")";
    }
    c.axis_notes.push_back("phase " + std::to_string(i + 1) + ": " + axis_name(d) +
                           " (n = " + std::to_string(n) + "), " + kname + " " +
                           place_text(src.where, dst.where) + ": the read stage into scratch" +
                           (dht      ? std::string()
                            : second ? " (Makhoul's order)"
                                     : " (Makhoul's twiddle inverted)") +
                           ", the " + (sign < 0 ? "forward" : "backward") + " complex chain of " +
                           std::to_string(m) + (even ? "" : " (n odd)") + " there, radices " +
                           (m == 1 ? std::string("none") : chain.radix_text(m)) +
                           ", then the store stage" + route_text);
  }
  for (std::size_t b = 0; b < p.batch_loops; ++b) {
    std::get<loop>(p.records[b]).body_end = static_cast<std::uint32_t>(p.records.size());
  }

  // The edge driver's records (core/edge_driver.hpp): a stage that reads a factor keeps its
  // kernel there and carries guards; the loops move the cursors (the batch loops by the
  // factors' batch strides, the first phase's lines loops by the input factor's axis strides and
  // the last phase's by the output factor's).
  c.edge_records.assign(p.records.size(), edge_record{});
  for (std::size_t i = 0; i < p.records.size(); ++i) {
    stage* st = std::get_if<stage>(&p.records[i]);
    if (st != nullptr && st->moving_factors != 0) {
      c.edge_records[i].run = st->run;
      st->run = &factor_guard_pass;
    }
  }
  for (std::size_t b = 0; b < p.batch_loops; ++b) {
    c.edge_records[b].factor_step = {
        in_factor ? e.in.batch_steps.at(b) : 0,
        out_factor ? e.out.batch_steps.at(b) : 0,
        0,
    };
  }
  for (const auto& [at, f] : line_loops) {
    c.edge_records[at].factor_step = f;
  }
  const std::uint8_t none = 0;
  c.factor_args = static_cast<std::uint8_t>((in_factor ? moving_in_factor : none) |
                                            (out_factor ? moving_out_factor : none));
  c.needs_args = c.factor_args != 0;
  if (!c.needs_args) {
    c.edge_records.clear();
  }

  c.block_elements = through_block ? (transform_reals + 1) / 2 : 0;
  lay_out_workspace(c, scratch, 2 * sizeof(T), false, 0); // no callbacks on r2r (refused)
}

std::string edges_text(const r2r_edges& e, bool through_block) {
  std::string t;
  const auto add = [&t](const std::string& line) { t += (t.empty() ? "" : "\n") + line; };
  if (e.in.kind != factor_kind::none) {
    add("input edge: times a real factor in the first phase's read stage");
  }
  if (e.out.kind != factor_kind::none || e.out.accumulate) {
    std::string what = e.out.kind != factor_kind::none ? "times a real factor" : "";
    if (e.out.accumulate) {
      what += what.empty() ? "accumulated" : ", accumulated";
    }
    add("output edge: " + what + " in the last phase's store stage");
  }
  if (e.scale != 1.0) {
    add("scales: one product, applied once in the last phase's store stage");
  }
  if (through_block) {
    add("accumulation over several axes: the phases before the last go through the workspace "
        "block, so that the output's prior values last until the last phase adds to them");
  }
  if (!t.empty()) {
    add("the edges cost no separate pass");
  }
  return t;
}

} // namespace

std::shared_ptr<const plan_core> make_r2r_core(const shape_view& shape, const options& opts,
                                               const edges_view* edges,
                                               const build_settings& settings) {
  if (shape.kinds.size() != shape.axes.size()) {
    reject("shape.kinds", "holds " + std::to_string(shape.kinds.size()) + " kinds for " +
                              std::to_string(shape.axes.size()) +
                              " axes; a real-to-real shape needs one kind per axis");
  }
  if (!shape.runs.empty()) {
    reject("shape.runs", "are given: real-to-real plans take no stored runs (DESIGN.md 5.7)");
  }
  if (!std::isfinite(opts.measure_seconds) || opts.measure_seconds < 0) {
    reject("options.measure_seconds", "must be finite and not negative");
  }
  const r2r_layout layout = resolve_r2r_layout(shape, opts.in_place);
  for (std::size_t d = 0; d < shape.kinds.size(); ++d) {
    const r2r_kind k = shape.kinds[d];
    const auto field = [d] { return "shape.kinds[" + std::to_string(d) + "]"; };
    if (static_cast<unsigned char>(k) > static_cast<unsigned char>(r2r_kind::dht)) {
      reject(field(), "is not an r2r_kind");
    }
    if (k == r2r_kind::dct1 && shape.axes[d].n == 1) {
      reject(field(), "is r2r_kind::dct1 on an axis of one point (" + axis_name(d) +
                          ".n = 1); DCT-I needs n >= 2");
    }
    if (k == r2r_kind::dct1 || k == r2r_kind::dct4 || k == r2r_kind::dst1 || k == r2r_kind::dst4) {
      reject(field(), std::string("is r2r_kind::") + kind_text(k) +
                          ", which is not built yet (task 6.5); built are dct2, dct3, dst2, "
                          "dst3 and dht");
    }
  }
  const r2r_edges e = resolve(shape, opts, edges);

  auto c = std::make_shared<plan_core>();
  c->kind = transform_kind::r2r;
  c->prec = shape.prec;
  c->dir = direction::forward;
  c->in_place = opts.in_place;
  for (const axis& a : shape.axes) {
    c->lengths.push_back(a.n);
  }
  for (const batch& b : shape.batches) {
    c->counts.push_back(b.count);
  }
  c->in_lo = layout.in_lo;
  c->in_hi = layout.in_hi;
  c->out_lo = layout.out_lo;
  c->out_hi = layout.out_hi;
  c->isa_target = plan_target_of(opts);
  if (shape.prec == precision::f64) {
    build<double>(*c, shape, layout, e, settings);
  } else {
    build<float>(*c, shape, layout, e, settings);
  }
  mark_flat(c->prog, settings.unroll_calls);
  if (c->needs_args) {
    c->edge_batched = make_edge_forms(c->prog, c->edge_records, settings.unroll_calls, 0, 0, 0);
  }
  const std::size_t points =
      std::accumulate(c->lengths.begin(), c->lengths.end(), std::size_t{1}, std::multiplies<>());
  c->graph = build_work_graph(c->prog, points, settings.chunk_points);
  c->profile = settings.profile;
  c->runs_inline = runs_inline(c->graph, c->profile);
  c->built = strategy_choice::chain;
  c->strategy_name = "the real-to-real phases";

  // describe().
  std::string kinds;
  std::string lengths;
  for (std::size_t d = 0; d < shape.axes.size(); ++d) {
    kinds += (d == 0 ? "" : " x ") + std::string(kind_text(shape.kinds[d]));
    lengths += (d == 0 ? "" : " x ") + std::to_string(shape.axes[d].n);
  }
  const std::size_t transforms = c->prog.transforms;
  c->pipeline_title = "fftn plan: r2r " + kinds + ", " +
                      (shape.prec == precision::f64 ? "double" : "float") + ", n = " + lengths +
                      ", " + std::to_string(transforms) +
                      (transforms == 1 ? " transform" : " transforms") +
                      (opts.in_place ? ", in place" : ", out of place");
  const bool backward = std::ranges::any_of(
      shape.kinds, [](r2r_kind k) { return k == r2r_kind::dct3 || k == r2r_kind::dst3; });
  c->strategy_note = std::string("real-to-real (DESIGN.md 5.7): each axis a phase over its lines, "
                                 "each line packed into scratch by a read stage, through the ") +
                     (backward ? "forward or backward" : "forward") +
                     " complex chain there and out by a store stage, one line at a time";
  c->edges_note = edges_text(e, c->block_elements != 0);
  if (opts.effort != planning::rules || opts.use != nullptr) {
    c->planning_note = "by the rules: measured planning and wisdom take no real-to-real plans yet "
                       "(task 6.12)";
  }
  const bool any_edge = edges != nullptr && (e.in.kind != factor_kind::none ||
                                             e.out.kind != factor_kind::none || e.out.accumulate);
  // Wisdom keys the plan as plan_core.cpp keys others: scales alone as one product in
  // options::scale, otherwise the edges' key, which carries every scale.
  options keyed = opts;
  keyed.scale = e.scale;
  c->shape_key =
      any_edge ? shape_key(shape, opts) + edges_key(*edges, opts.scale) : shape_key(shape, keyed);
  return c;
}

} // namespace fftn::inline v0::detail
