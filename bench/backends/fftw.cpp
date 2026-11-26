// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for FFTW 3, built only when FFTN_BENCH_FFTW_ROOT names an installation:
// double precision, and single precision as well when the installation has libfftw3f
// (FBENCH_FFTW_FLOAT). FFTW is GPL: this executable is for our own measurements and is never
// distributed (AGENTS.md section 10).
//
// Options (space-separated KEY=VALUE, from the arm's +OPTION=value list):
//   PLANNER=ESTIMATE|MEASURE|PATIENT|EXHAUSTIVE   planning effort (default MEASURE)
//   WISDOM_IN=<file>    import wisdom from the file when it exists
//   WISDOM_OUT=<file>   export the session's wisdom to the file at the end
//   PLAN_NOTE=1         write the plan FFTW chose (fftw_sprint_plan) into the case's note
//   THREADS=N           plan with N threads (fftw_plan_with_nthreads, FFTW's POSIX-threads
//                       library, so no OpenMP runtime joins the process); default 1, which
//                       never initialises FFTW's threads; N > 1 needs libfftw3_threads in
//                       the installation (FBENCH_FFTW_THREADS), and is double precision only
// Wisdom is otherwise fresh in every process: stale accumulated wisdom slowed FFTW by
// 1.3x in the prototype (DESIGN.md finding E10). The driver keeps one wisdom file per run
// and planner setting, so a campaign starts fresh and its rounds share their plans. Single
// precision is a separate library with wisdom of its own, kept in <file>.f32.

#include "backends/common.hpp"
#include "harness/backend.hpp"
#include "harness/json.hpp"

#include <fftw3.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace {

struct session {
  unsigned flags = FFTW_MEASURE;
  std::string planner = "MEASURE";
  std::string wisdom_in;
  std::string wisdom_out;
  bool plan_note = false;
  int threads = 1;
  std::string description;
};

session& state() {
  static session s;
  return s;
}

// fftw_plan or fftwf_plan, by `single`.
struct plan_record {
  void* plan = nullptr;
  std::int32_t kind = fbench_c2c;
  bool single = false;
};

void say(char* message, std::size_t size, const std::string& text) {
  std::snprintf(message, size, "%s", text.c_str());
}

// FFTW appends its SIMD extensions to the version string ("fftw-3.3.11-sse2-avx2"); a build
// without any is scalar.
std::string simd_of(std::string_view version) {
  const std::size_t dash = version.find('-', version.find('-') + 1);
  return dash == std::string_view::npos ? "scalar" : std::string(version.substr(dash + 1));
}

// The file that defines FFTW's code in this process; empty if FFTW is linked statically.
// The address of a symbol as seen from the executable is no use on Linux: a function's
// address is the executable's canonical PLT entry, and a data symbol is copy-relocated
// into the executable, so dladdr on either names the executable itself. dlsym with
// RTLD_NEXT searches only the objects loaded after this one, so it finds the real
// definition (on macOS too).
std::string loaded_from(const char* symbol) {
#ifdef _WIN32
  (void)symbol;
  return "";
#else
  const void* const address = dlsym(RTLD_NEXT, symbol);
  Dl_info info{};
  if (address == nullptr || dladdr(address, &info) == 0 || info.dli_fname == nullptr) {
    return "";
  }
  return info.dli_fname;
#endif
}

// One precision's FFTW functions under common names: fftw_* (double) and fftwf_* (float)
// are two libraries with the same API.
template <class R> struct api;

template <> struct api<double> {
  using plan = fftw_plan;
  using complex = fftw_complex;
  using iodim64 = fftw_iodim64;
  static constexpr auto plan_many_dft = &fftw_plan_many_dft;
  static constexpr auto plan_many_dft_r2c = &fftw_plan_many_dft_r2c;
  static constexpr auto plan_many_dft_c2r = &fftw_plan_many_dft_c2r;
  static constexpr auto plan_guru64_dft = &fftw_plan_guru64_dft;
  static constexpr auto plan_guru64_dft_r2c = &fftw_plan_guru64_dft_r2c;
  static constexpr auto plan_guru64_dft_c2r = &fftw_plan_guru64_dft_c2r;
  static constexpr auto execute_dft = &fftw_execute_dft;
  static constexpr auto execute_dft_r2c = &fftw_execute_dft_r2c;
  static constexpr auto execute_dft_c2r = &fftw_execute_dft_c2r;
  static constexpr auto destroy_plan = &fftw_destroy_plan;
  static constexpr auto sprint_plan = &fftw_sprint_plan;
  static constexpr auto forget_wisdom = &fftw_forget_wisdom;
  static constexpr auto import_wisdom_from_file = &fftw_import_wisdom_from_file;
  static constexpr auto export_wisdom_to_file = &fftw_export_wisdom_to_file;
  static constexpr auto cleanup = &fftw_cleanup;
};

#ifdef FBENCH_FFTW_FLOAT
template <> struct api<float> {
  using plan = fftwf_plan;
  using complex = fftwf_complex;
  using iodim64 = fftwf_iodim64;
  static constexpr auto plan_many_dft = &fftwf_plan_many_dft;
  static constexpr auto plan_many_dft_r2c = &fftwf_plan_many_dft_r2c;
  static constexpr auto plan_many_dft_c2r = &fftwf_plan_many_dft_c2r;
  static constexpr auto plan_guru64_dft = &fftwf_plan_guru64_dft;
  static constexpr auto plan_guru64_dft_r2c = &fftwf_plan_guru64_dft_r2c;
  static constexpr auto plan_guru64_dft_c2r = &fftwf_plan_guru64_dft_c2r;
  static constexpr auto execute_dft = &fftwf_execute_dft;
  static constexpr auto execute_dft_r2c = &fftwf_execute_dft_r2c;
  static constexpr auto execute_dft_c2r = &fftwf_execute_dft_c2r;
  static constexpr auto destroy_plan = &fftwf_destroy_plan;
  static constexpr auto sprint_plan = &fftwf_sprint_plan;
  static constexpr auto forget_wisdom = &fftwf_forget_wisdom;
  static constexpr auto import_wisdom_from_file = &fftwf_import_wisdom_from_file;
  static constexpr auto export_wisdom_to_file = &fftwf_export_wisdom_to_file;
  static constexpr auto cleanup = &fftwf_cleanup;
};
#endif

// The wisdom file of a precision: single precision appends ".f32".
template <class R> std::string wisdom_file(const std::string& file) {
  return file.empty() || sizeof(R) == sizeof(double) ? file : file + ".f32";
}

// Forgets the process's wisdom and imports the file's when it exists; false when the file
// exists but does not import.
template <class R> bool import_wisdom(const std::string& file) {
  api<R>::forget_wisdom();
  if (file.empty()) {
    return true;
  }
  std::FILE* const f = std::fopen(file.c_str(), "r");
  if (f == nullptr) {
    return true;
  }
  const int ok = api<R>::import_wisdom_from_file(f);
  std::fclose(f);
  return ok != 0;
}

template <class R> void export_wisdom(const std::string& file) {
  if (file.empty()) {
    return;
  }
  if (std::FILE* f = std::fopen(file.c_str(), "w"); f != nullptr) {
    api<R>::export_wisdom_to_file(f);
    std::fclose(f);
  }
}

// The plan in FFTW's own notation, one line (experiment X25): which codelets, radices,
// buffering and vector loops it chose. fftw_sprint_plan's string is malloc'ed.
template <class R> std::string plan_line(typename api<R>::plan p) {
  char* const text = api<R>::sprint_plan(p);
  const std::string_view plan_text = text != nullptr ? text : "";
  std::string line;
  for (const char ch : plan_text) {
    const bool blank = ch == '\n' || ch == ' ';
    if (!blank || (!line.empty() && line.back() != ' ')) {
      line.push_back(blank ? ' ' : ch);
    }
  }
  std::free(text); // FFTW's API allocates the string with malloc
  return line;
}

template <class R>
typename api<R>::plan make_plan(const fbench_case* c, void* in, void* out, unsigned flags) {
  using A = api<R>;
  using C = typename A::complex;
  // FFTW's guru interface takes int; the harness keeps every length below 2^31, and the
  // embeddings and distances of the catalogue's cases fit too.
  std::array<int, 3> n{};
  std::array<int, 3> in_embed{};
  std::array<int, 3> out_embed{};
  for (int d = 0; d < c->rank; ++d) {
    const auto i = static_cast<std::size_t>(d);
    n.at(i) = static_cast<int>(c->dims[i]);
    in_embed.at(i) = n.at(i);
    out_embed.at(i) = n.at(i);
  }
  const auto last = static_cast<std::size_t>(c->rank - 1);
  in_embed.at(last) = static_cast<int>(c->in_last);
  out_embed.at(last) = static_cast<int>(c->out_last);
  const auto batch = static_cast<int>(c->batch);
  const auto in_dist = static_cast<int>(c->in_dist);
  const auto out_dist = static_cast<int>(c->out_dist);
  // No FFTW_UNALIGNED: the harness plans on buffers with the alignment of every later
  // execute, and FFTW_UNALIGNED silently drops FFTW's aligned SIMD codelets (finding E10).
  if (c->interleave > 1) {
    // The interleaved layout (fbench_case::interleave) through the guru interface: strided
    // axes, the transforms one element apart.
    const auto in_strides = fbench::backends::element_strides(*c, c->in_last);
    const auto out_strides = fbench::backends::element_strides(*c, c->out_last);
    std::array<typename A::iodim64, 3> axes{};
    for (int d = 0; d < c->rank; ++d) {
      const auto i = static_cast<std::size_t>(d);
      axes.at(i) = {
          .n = c->dims[i],
          .is = in_strides[i] * c->interleave,
          .os = out_strides[i] * c->interleave,
      };
    }
    std::array<typename A::iodim64, 2> loops{};
    int loop_rank = 0;
    if (c->outer > 1) {
      loops.at(0) = {.n = c->outer, .is = c->outer_in_dist, .os = c->outer_out_dist};
      loop_rank = 1;
    }
    loops.at(static_cast<std::size_t>(loop_rank)) = {
        .n = c->batch,
        .is = c->in_dist,
        .os = c->out_dist,
    };
    ++loop_rank;
    if (c->kind == fbench_c2c) {
      return A::plan_guru64_dft(c->rank, axes.data(), loop_rank, loops.data(), static_cast<C*>(in),
                                static_cast<C*>(out), c->sign, flags);
    }
    if (c->kind == fbench_r2c) {
      return A::plan_guru64_dft_r2c(c->rank, axes.data(), loop_rank, loops.data(),
                                    static_cast<R*>(in), static_cast<C*>(out), flags);
    }
    return A::plan_guru64_dft_c2r(c->rank, axes.data(), loop_rank, loops.data(),
                                  static_cast<C*>(in), static_cast<R*>(out), flags);
  }
  if (c->kind == fbench_c2c) {
    return A::plan_many_dft(c->rank, n.data(), batch, static_cast<C*>(in), in_embed.data(), 1,
                            in_dist, static_cast<C*>(out), out_embed.data(), 1, out_dist, c->sign,
                            flags);
  }
  if (c->kind == fbench_r2c) {
    return A::plan_many_dft_r2c(c->rank, n.data(), batch, static_cast<R*>(in), in_embed.data(), 1,
                                in_dist, static_cast<C*>(out), out_embed.data(), 1, out_dist,
                                flags);
  }
  return A::plan_many_dft_c2r(c->rank, n.data(), batch, static_cast<C*>(in), in_embed.data(), 1,
                              in_dist, static_cast<R*>(out), out_embed.data(), 1, out_dist, flags);
}

template <class R>
std::int32_t create(const fbench_case* c, void* in, void* out, void** plan, char* message,
                    std::size_t message_size) {
  const typename api<R>::plan p = make_plan<R>(c, in, out, state().flags);
  if (p == nullptr) {
    say(message, message_size, "FFTW returned no plan");
    return fbench_error;
  }
  if (state().plan_note) {
    say(message, message_size, plan_line<R>(p));
  }
  auto record = std::make_unique<plan_record>();
  record->plan = static_cast<void*>(p);
  record->kind = c->kind;
  record->single = sizeof(R) == sizeof(float);
  *plan = record.release();
  return fbench_ok;
}

template <class R> void execute(const plan_record& r, void* in, void* out) {
  using A = api<R>;
  using C = typename A::complex;
  const auto p = static_cast<typename A::plan>(r.plan);
  if (r.kind == fbench_c2c) {
    A::execute_dft(p, static_cast<C*>(in), static_cast<C*>(out));
  } else if (r.kind == fbench_r2c) {
    A::execute_dft_r2c(p, static_cast<R*>(in), static_cast<C*>(out));
  } else {
    A::execute_dft_c2r(p, static_cast<C*>(in), static_cast<R*>(out));
  }
}

} // namespace

extern "C" {

std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size) {
  session& s = state();
  const std::string_view text = options;
  std::size_t at = 0;
  while (at < text.size()) {
    const std::size_t end = std::min(text.find(' ', at), text.size());
    const std::string_view token = text.substr(at, end - at);
    at = end + 1;
    if (token.empty()) {
      continue;
    }
    const std::size_t eq = token.find('=');
    const std::string_view key = token.substr(0, eq);
    const std::string_view val = eq == std::string_view::npos ? "" : token.substr(eq + 1);
    if (key == "PLANNER") {
      using entry = std::pair<std::string_view, unsigned>;
      constexpr std::array planners{
          entry{"ESTIMATE", FFTW_ESTIMATE},
          entry{"MEASURE", FFTW_MEASURE},
          entry{"PATIENT", FFTW_PATIENT},
          entry{"EXHAUSTIVE", FFTW_EXHAUSTIVE},
      };
      bool found = false;
      for (const auto& [name, flag] : planners) {
        if (val == name) {
          s.flags = flag;
          s.planner = name;
          found = true;
        }
      }
      if (!found) {
        say(message, message_size, "PLANNER is ESTIMATE, MEASURE, PATIENT or EXHAUSTIVE");
        return fbench_error;
      }
    } else if (key == "WISDOM_IN") {
      s.wisdom_in = val;
    } else if (key == "WISDOM_OUT") {
      s.wisdom_out = val;
    } else if (key == "PLAN_NOTE") {
      if (val != "0" && val != "1") {
        say(message, message_size, "PLAN_NOTE is 0 or 1");
        return fbench_error;
      }
      s.plan_note = val == "1";
    } else if (key == "THREADS") {
      const std::string v(val);
      char* stop = nullptr;
      const long t = std::strtol(v.c_str(), &stop, 10);
      if (v.empty() || stop != v.c_str() + v.size() || t < 1 || t > 4096) {
        say(message, message_size, "THREADS is a whole number from 1 to 4096");
        return fbench_error;
      }
      s.threads = static_cast<int>(t);
    } else {
      say(message, message_size, "unknown FFTW option " + std::string(key));
      return fbench_error;
    }
  }
#ifdef FBENCH_FFTW_THREADS
  if (s.threads > 1) {
    if (fftw_init_threads() == 0) {
      say(message, message_size, "fftw_init_threads failed");
      return fbench_error;
    }
    fftw_plan_with_nthreads(s.threads);
  }
#else
  if (s.threads > 1) {
    say(message, message_size,
        "THREADS > 1 needs libfftw3_threads, which " FBENCH_FFTW_ROOT " does not have");
    return fbench_error;
  }
#endif
  if (!import_wisdom<double>(s.wisdom_in)) {
    say(message, message_size, "cannot import wisdom from " + s.wisdom_in);
    return fbench_error;
  }
  const std::string version = fftw_version;
  fbench::json description;
  description.str("library", "fftw")
      .str("version", version)
      .str("isa", simd_of(version))
      .raw("supports", R"(["c2c","r2c","c2r"])");
#ifdef FBENCH_FFTW_FLOAT
  // The threaded planner is initialised for the double library only.
  if (s.threads == 1) {
    if (!import_wisdom<float>(wisdom_file<float>(s.wisdom_in))) {
      say(message, message_size, "cannot import wisdom from " + wisdom_file<float>(s.wisdom_in));
      return fbench_error;
    }
    description.raw("precisions", R"(["f64","f32"])")
        .str("isa_f32", simd_of(fftwf_version))
        .str("loaded_from_f32", loaded_from("fftwf_execute"));
  }
#endif
  s.description = description.str("planner", s.planner)
                      .integer("threads", s.threads)
                      .str("cc", fftw_cc)
                      .str("codelet_optim", fftw_codelet_optim)
                      .str("loaded_from", loaded_from("fftw_execute"))
                      .str("expected_root", FBENCH_FFTW_ROOT)
                      .inner();
  return fbench_ok;
}

void fbench_end() {
  const session& s = state();
  export_wisdom<double>(s.wisdom_out);
#ifdef FBENCH_FFTW_FLOAT
  if (s.threads == 1) {
    export_wisdom<float>(wisdom_file<float>(s.wisdom_out));
    api<float>::cleanup();
  }
#endif
#ifdef FBENCH_FFTW_THREADS
  if (s.threads > 1) {
    fftw_cleanup_threads();
    return;
  }
#endif
  api<double>::cleanup();
}

const char* fbench_describe() { return state().description.c_str(); }

std::int32_t fbench_create(const fbench_case* c, void* in, void* out, void** plan, char* message,
                           std::size_t message_size) {
  if (c->precision == fbench_f32) {
#ifdef FBENCH_FFTW_FLOAT
    return create<float>(c, in, out, plan, message, message_size);
#else
    say(message, message_size, "this installation has no libfftw3f");
    return fbench_unsupported;
#endif
  }
  return create<double>(c, in, out, plan, message, message_size);
}

void fbench_execute(const void* plan, void* in, void* out) {
  const auto* r = static_cast<const plan_record*>(plan);
#ifdef FBENCH_FFTW_FLOAT
  if (r->single) {
    execute<float>(*r, in, out);
    return;
  }
#endif
  execute<double>(*r, in, out);
}

void fbench_destroy(void* plan) {
  const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan));
#ifdef FBENCH_FFTW_FLOAT
  if (r->single) {
    api<float>::destroy_plan(static_cast<fftwf_plan>(r->plan));
    return;
  }
#endif
  api<double>::destroy_plan(static_cast<fftw_plan>(r->plan));
}
}
