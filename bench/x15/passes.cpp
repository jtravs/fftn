// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Experiment X15, pass level (dev/experiments/X15-compiler-vectorised-scalar.md): times
// radix4_pass (src/isa/kernels.inl), the one pass every target has, in several builds of
// the same kernel text, in one process. One process is one round; bench/x15/x15_passes.py
// runs the rounds and takes the medians. Never run by the gate except `--check`.
//
//   x15-passes --describe            the build and the variants this machine can run
//   x15-passes --check               scalar-text variants equal "scalar" bit for bit
//   x15-passes --l1-bytes N --l2-bytes N [--m 16,32,...,4096] [--precisions f64,f32]
//              [--levels l1,l2,hot] [--variants a,b,...] [--order 2,0,1,...]
//              [--batch-time 0.002] [--batches 7] [--warmup 0.02] [--seed 1]
//
// Variants (those the processor runs; --describe lists them):
//
//   scalar, scalar~      scalar text in a plain region (arm A); "scalar~" is the same
//                        function timed separately, the identical-arm control
//   scalar_simd          the same with omp simd (arm B)
//   avx2_text[_simd]     scalar text in an avx2 region, without and with omp simd (arm C)
//   avx2, neon_fcma      the library's hand-vocabulary kernels (arm D)
//   lib_scalar           the library's own scalar table as this build made it: plain in
//                        arm A, omp simd in arm B, -march/-mcpu=native in arm E
//   lib_avx2_scalar_text the library's arm C table, when the build has it
//
// The copies of the scalar text (variant.cpp) are compiled with the harness's flags only,
// never the library's experimental ones, so in an arm E build lib_scalar is E and the
// copies are still A, B and C.
//
// Residency, for each m (a set is one input, output and twiddle table, sized with four
// reals per twiddle for every variant so all variants get the same sets):
//
//   l1    one set, reused; only where the set fills at most half of L1
//   l2    k >= 2 sets in rotation, k S >= 4 L1 (so each call finds its set outside L1),
//         only where k S fills at most half of L2
//   hot   one set, reused, every m (wherever the cache hierarchy keeps it)
//
// One JSON line per variant, precision, m and level: ns per call (median and minimum over
// batches, and every batch) and the median ns per butterfly (m butterflies per call).

#include "fbench_build_info.hpp"
#include "harness/json.hpp"
#include "isa/arch.hpp"
#include "isa/kernel_table.hpp"
#include "isa/target.hpp"
#include "platform/cpu_features.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifdef __APPLE__
#include <pthread/qos.h>
#include <sys/qos.h>
#endif

namespace fftn::inline v0::detail {
namespace x15_scalar {
extern const kernel_table table;
}
namespace x15_scalar_simd {
extern const kernel_table table;
}
#if FFTN_ARCH_X86_64
namespace x15_avx2_text {
extern const kernel_table table;
}
namespace x15_avx2_text_simd {
extern const kernel_table table;
}
#endif
} // namespace fftn::inline v0::detail

namespace {

namespace fd = fftn::detail;
using steady = std::chrono::steady_clock;

// How a variant's output must relate to "scalar"'s.
enum class expect : unsigned char {
  same_bits, // scalar text whose order of operations is unchanged
  near,      // a hand vocabulary: other roundings, same transform
  report,    // arm E: same bits unless the compiler changed them; reported, not failed
};

struct variant {
  std::string name;
  const fd::kernel_table* table;
  expect relation;
};

#if defined(FFTN_X_NATIVE) && FFTN_X_NATIVE
constexpr expect lib_scalar_relation = expect::report;
#else
constexpr expect lib_scalar_relation = expect::same_bits;
#endif

std::vector<variant> available_variants() {
  [[maybe_unused]] const fd::cpu_features features = fd::detect_cpu_features();
  std::vector<variant> v{
      {.name = "scalar", .table = &fd::x15_scalar::table, .relation = expect::same_bits},
      {.name = "scalar~", .table = &fd::x15_scalar::table, .relation = expect::same_bits},
      {.name = "scalar_simd", .table = &fd::x15_scalar_simd::table, .relation = expect::same_bits},
  };
#if FFTN_ARCH_X86_64
  if (fd::runs_on(fd::target::avx2, features)) {
    v.push_back(
        {.name = "avx2_text", .table = &fd::x15_avx2_text::table, .relation = expect::same_bits});
    v.push_back({.name = "avx2_text_simd",
                 .table = &fd::x15_avx2_text_simd::table,
                 .relation = expect::same_bits});
    v.push_back({.name = "avx2", .table = &fd::avx2::table, .relation = expect::near});
  }
#endif
#if FFTN_ARCH_AARCH64
  if (fd::runs_on(fd::target::neon_fcma, features)) {
    v.push_back({.name = "neon_fcma", .table = &fd::neon_fcma::table, .relation = expect::near});
  }
#endif
  v.push_back({.name = "lib_scalar", .table = &fd::scalar::table, .relation = lib_scalar_relation});
#if FFTN_ARCH_X86_64 && defined(FFTN_X_AVX2_SCALAR_TEXT) && FFTN_X_AVX2_SCALAR_TEXT
  if (fd::runs_on(fd::target::avx2, features)) {
    v.push_back({.name = "lib_avx2_scalar_text",
                 .table = &fd::avx2::scalar_text::table,
                 .relation = expect::same_bits});
  }
#endif
  return v;
}

struct settings {
  std::vector<std::size_t> ms{16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  std::vector<std::string> precisions{"f64", "f32"};
  std::vector<std::string> levels{"l1", "l2", "hot"};
  std::vector<std::string> variants; // empty: all available
  std::vector<std::size_t> order;    // a permutation of the selected variants
  std::size_t l1_bytes = 0;
  std::size_t l2_bytes = 0;
  double batch_time = 0.002;
  int batches = 7;
  double warmup = 0.02;
  std::uint64_t seed = 1;
};

// A buffer of T aligned to 64 bytes.
template <class T> class buffer {
public:
  explicit buffer(std::size_t count) : storage_(count + 64 / sizeof(T)) {
    auto address = reinterpret_cast<std::uintptr_t>(storage_.data());
    offset_ = ((64 - address % 64) % 64) / sizeof(T);
  }
  T* data() { return storage_.data() + offset_; }
  [[nodiscard]] const T* data() const { return storage_.data() + offset_; }

private:
  std::vector<T> storage_;
  std::size_t offset_ = 0;
};

// The forward twiddles exp(-2 pi i j p / 4m) as three rows j = 1, 2, 3 of m, interleaved.
template <class T> std::vector<T> plain_twiddles(std::size_t m) {
  std::vector<T> w(6 * m);
  const double n = 4.0 * static_cast<double>(m);
  for (std::size_t j = 1; j < 4; ++j) {
    for (std::size_t p = 0; p < m; ++p) {
      const double angle = -2.0 * std::numbers::pi * static_cast<double>(j * p) / n;
      w[2 * ((j - 1) * m + p)] = static_cast<T>(std::cos(angle));
      w[2 * ((j - 1) * m + p) + 1] = static_cast<T>(std::sin(angle));
    }
  }
  return w;
}

template <class T> std::vector<T> random_reals(std::size_t count, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<T> v(count);
  for (T& x : v) {
    x = static_cast<T>(u(rng));
  }
  return v;
}

// Reals in one set, with four reals per twiddle (the most any variant stores).
template <class T> std::size_t set_bytes(std::size_t m) {
  return (8 * m + 8 * m + 12 * m) * sizeof(T);
}

// The sets a variant runs on for one case.
template <class T> struct case_sets {
  std::vector<buffer<T>> x;
  std::vector<buffer<T>> y;
  std::vector<buffer<T>> tw;
};

template <class T>
case_sets<T> make_sets(const fd::kernels_of<T>& k, std::size_t m, std::size_t count,
                       std::uint64_t seed) {
  case_sets<T> s;
  const std::vector<T> input = random_reals<T>(8 * m, seed);
  const std::vector<T> w = plain_twiddles<T>(m);
  for (std::size_t i = 0; i < count; ++i) {
    s.x.emplace_back(8 * m);
    std::ranges::copy(input, s.x.back().data());
    s.y.emplace_back(8 * m);
    std::fill_n(s.y.back().data(), 8 * m, T(0));
    s.tw.emplace_back(3 * m * k.tw_reals);
    k.pack_twiddles(w.data(), 3 * m, s.tw.back().data());
  }
  return s;
}

template <class T>
double run_calls(const fd::kernels_of<T>& k, case_sets<T>& s, std::size_t m, std::int64_t calls) {
  // Through a volatile copy, so the call cannot be resolved at compile time.
  auto* volatile pass = k.radix4_pass;
  const std::size_t sets = s.x.size();
  const steady::time_point t0 = steady::now();
  for (std::int64_t c = 0; c < calls; ++c) {
    const auto i = static_cast<std::size_t>(c) % sets;
    pass(s.x[i].data(), s.y[i].data(), s.tw[i].data(), m, -1);
  }
  return std::chrono::duration<double>(steady::now() - t0).count();
}

double median(std::vector<double> v) {
  std::ranges::sort(v);
  const std::size_t mid = v.size() / 2;
  return v.size() % 2 == 1 ? v[mid] : 0.5 * (v[mid - 1] + v[mid]);
}

// The number of sets for a level, or 0 when the level does not apply to this m.
std::size_t sets_for(const std::string& level, std::size_t bytes, const settings& st) {
  if (level == "hot") {
    return 1;
  }
  if (level == "l1") {
    return 2 * bytes <= st.l1_bytes ? 1 : 0;
  }
  const std::size_t k = std::max<std::size_t>(2, (4 * st.l1_bytes + bytes - 1) / bytes);
  return 2 * k * bytes <= st.l2_bytes ? k : 0;
}

template <class T>
std::string measure(const variant& v, const char* precision, std::size_t m,
                    const std::string& level, std::size_t sets, const settings& st) {
  const fd::kernels_of<T>& k = fd::of<T>(*v.table);
  case_sets<T> s = make_sets<T>(k, m, sets, st.seed + m);
  std::int64_t calls = 1;
  double elapsed = run_calls(k, s, m, calls);
  while (elapsed < 0.5 * st.batch_time) {
    calls *= 2;
    elapsed = run_calls(k, s, m, calls);
  }
  calls = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(static_cast<double>(calls) *
                                                                        st.batch_time / elapsed)));
  const steady::time_point warm_start = steady::now();
  while (std::chrono::duration<double>(steady::now() - warm_start).count() < st.warmup) {
    run_calls(k, s, m, calls);
  }
  std::vector<double> ns;
  ns.reserve(static_cast<std::size_t>(st.batches));
  for (int b = 0; b < st.batches; ++b) {
    ns.push_back(run_calls(k, s, m, calls) / static_cast<double>(calls) * 1e9);
  }
  const double med = median(ns);
  return fbench::json()
      .str("type", "timing")
      .str("variant", v.name)
      .str("precision", precision)
      .integer("m", static_cast<std::int64_t>(m))
      .str("level", level)
      .integer("sets", static_cast<std::int64_t>(sets))
      .integer("set_bytes", static_cast<std::int64_t>(set_bytes<T>(m)))
      .integer("calls", calls)
      .numbers("batches_ns", ns)
      .num("median_ns", med)
      .num("min_ns", *std::ranges::min_element(ns))
      .num("ns_per_butterfly", med / static_cast<double>(m))
      .object();
}

template <class T>
void time_precision(const std::vector<variant>& chosen, const char* precision, const settings& st) {
  for (const std::size_t m : st.ms) {
    for (const std::string& level : st.levels) {
      const std::size_t sets = sets_for(level, set_bytes<T>(m), st);
      if (sets == 0) {
        continue;
      }
      for (const std::size_t i : st.order) {
        std::printf("%s\n", measure<T>(chosen[i], precision, m, level, sets, st).c_str());
        std::fflush(stdout);
      }
    }
  }
}

// ---- --check -------------------------------------------------------------------------------

template <class T> std::vector<T> one_pass(const variant& v, std::size_t m, int sign) {
  const fd::kernels_of<T>& k = fd::of<T>(*v.table);
  const std::vector<T> x = random_reals<T>(8 * m, 7 + m);
  const std::vector<T> w = plain_twiddles<T>(m);
  std::vector<T> tw(3 * m * k.tw_reals);
  k.pack_twiddles(w.data(), 3 * m, tw.data());
  std::vector<T> y(8 * m);
  k.radix4_pass(x.data(), y.data(), tw.data(), m, sign);
  return y;
}

template <class T> int check_precision(const std::vector<variant>& all, const char* precision) {
  int failures = 0;
  for (std::size_t m = 4; m <= 4096; m *= 2) {
    for (const int sign : {-1, +1}) {
      const std::vector<T> ref = one_pass<T>(all.front(), m, sign);
      double scale = 0.0;
      for (const T r : ref) {
        scale = std::max(scale, std::abs(static_cast<double>(r)));
      }
      for (const variant& v : all) {
        const std::vector<T> y = one_pass<T>(v, m, sign);
        if (std::memcmp(y.data(), ref.data(), y.size() * sizeof(T)) == 0) {
          continue;
        }
        double diff = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i) {
          diff = std::max(diff, std::abs(static_cast<double>(y[i]) - static_cast<double>(ref[i])));
        }
        const double relative = diff / scale;
        // One radix-4 butterfly and one complex product: a few roundings.
        const double bound = 16.0 * std::numeric_limits<T>::epsilon();
        const bool fail =
            v.relation == expect::same_bits || (v.relation == expect::near && relative > bound);
        if (fail || v.relation == expect::report) {
          std::printf("%s: %s, %s, m = %zu, sign %+d: max |diff| / max |ref| = %.3g\n",
                      fail ? "FAIL" : "note", v.name.c_str(), precision, m, sign, relative);
        }
        failures += fail ? 1 : 0;
      }
    }
  }
  return failures;
}

// ---- Command line --------------------------------------------------------------------------

std::vector<std::string> split(std::string_view text) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    if (end > start) {
      out.emplace_back(text.substr(start, end - start));
    }
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }
  return out;
}

std::size_t to_size(const std::string& text) {
  return static_cast<std::size_t>(std::strtoull(text.c_str(), nullptr, 10));
}

std::string describe(const std::vector<variant>& all) {
  std::string names = "[";
  for (const variant& v : all) {
    names += (names.size() > 1 ? "," : "") + fbench::json_quote(v.name);
  }
  names += ']';
  return fbench::json()
      .str("type", "describe")
      .str("compiler", fbench::build::compiler)
      .str("build_type", fbench::build::build_type)
      .str("flags", fbench::build::flags)
      .str("commit", fbench::build::commit)
      .str("x15_switches", FFTN_X15_LIBRARY_SWITCHES)
      .boolean("x86_64", FFTN_ARCH_X86_64 != 0)
      .raw("variants", names)
      .object();
}

int usage(const char* why) {
  std::fprintf(stderr, "x15-passes: %s (usage at the top of bench/x15/passes.cpp)\n", why);
  return 2;
}

} // namespace

namespace {

int run(int argc, char** argv) {
#ifdef __APPLE__
  // As fbench's harness: keeps the process on the performance cores.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  const std::vector<variant> all = available_variants();
  settings st;
  const std::span<char*> args(argv, static_cast<std::size_t>(argc));
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view a = args[i];
    if (a == "--describe") {
      std::printf("%s\n", describe(all).c_str());
      return 0;
    }
    if (a == "--check") {
      const int failures = check_precision<double>(all, "f64") + check_precision<float>(all, "f32");
      std::printf("x15-passes --check: %d failure%s over %zu variants\n", failures,
                  failures == 1 ? "" : "s", all.size());
      return failures == 0 ? 0 : 1;
    }
    if (i + 1 >= args.size()) {
      return usage("an option is missing its value");
    }
    const std::string value = args[++i];
    if (a == "--m") {
      st.ms.clear();
      for (const std::string& s : split(value)) {
        st.ms.push_back(to_size(s));
      }
    } else if (a == "--precisions") {
      st.precisions = split(value);
    } else if (a == "--levels") {
      st.levels = split(value);
    } else if (a == "--variants") {
      st.variants = split(value);
    } else if (a == "--order") {
      for (const std::string& s : split(value)) {
        st.order.push_back(to_size(s));
      }
    } else if (a == "--l1-bytes") {
      st.l1_bytes = to_size(value);
    } else if (a == "--l2-bytes") {
      st.l2_bytes = to_size(value);
    } else if (a == "--batch-time") {
      st.batch_time = std::strtod(value.c_str(), nullptr);
    } else if (a == "--batches") {
      st.batches = static_cast<int>(std::clamp(std::strtol(value.c_str(), nullptr, 10), 1L, 1000L));
    } else if (a == "--warmup") {
      st.warmup = std::strtod(value.c_str(), nullptr);
    } else if (a == "--seed") {
      st.seed = std::strtoull(value.c_str(), nullptr, 10);
    } else {
      return usage("unknown option");
    }
  }
  if (st.l1_bytes == 0 || st.l2_bytes == 0) {
    return usage("--l1-bytes and --l2-bytes are required for timing");
  }
  std::vector<variant> chosen;
  if (st.variants.empty()) {
    chosen = all;
  } else {
    for (const std::string& name : st.variants) {
      const auto it = std::ranges::find(all, name, &variant::name);
      if (it == all.end()) {
        return usage("a variant this machine cannot run, or no such variant");
      }
      chosen.push_back(*it);
    }
  }
  if (st.order.empty()) {
    for (std::size_t i = 0; i < chosen.size(); ++i) {
      st.order.push_back(i);
    }
  }
  std::vector<std::size_t> sorted = st.order;
  std::ranges::sort(sorted);
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    if (sorted.size() != chosen.size() || sorted[i] != i) {
      return usage("--order must be a permutation of the chosen variants' indices");
    }
  }
  std::printf("%s\n", describe(chosen).c_str());
  // A quarter second of work first, untimed, so the first case does not pay for the core
  // leaving a low-power state.
  {
    const fd::kernels_of<double>& k = fd::of<double>(*chosen.front().table);
    case_sets<double> s = make_sets<double>(k, 256, 1, st.seed);
    const steady::time_point start = steady::now();
    while (std::chrono::duration<double>(steady::now() - start).count() < 0.25) {
      run_calls(k, s, 256, 64);
    }
  }
  for (const std::string& p : st.precisions) {
    if (p == "f64") {
      time_precision<double>(chosen, "f64", st);
    } else if (p == "f32") {
      time_precision<float>(chosen, "f32", st);
    } else {
      return usage("precisions are f64 and f32");
    }
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "x15-passes: %s\n", e.what());
    return 1;
  }
}
