// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for FFTW 3 (double precision), built only when FFTN_BENCH_FFTW_ROOT names
// an installation. FFTW is GPL: this executable is for our own measurements and is never
// distributed (AGENTS.md section 10).
//
// Options (space-separated KEY=VALUE, from the arm's +OPTION=value list):
//   PLANNER=ESTIMATE|MEASURE|PATIENT|EXHAUSTIVE   planning effort (default MEASURE)
//   WISDOM_IN=<file>    import wisdom from the file when it exists
//   WISDOM_OUT=<file>   export the session's wisdom to the file at the end
//   PLAN_NOTE=1         write the plan FFTW chose (fftw_sprint_plan) into the case's note
// Wisdom is otherwise fresh in every process: stale accumulated wisdom slowed FFTW by
// 1.3x in the prototype (DESIGN.md finding E10). The driver keeps one wisdom file per run
// and planner setting, so a campaign starts fresh and its rounds share their plans.

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
  std::string description;
};

session& state() {
  static session s;
  return s;
}

struct plan_record {
  fftw_plan plan = nullptr;
  std::int32_t kind = fbench_c2c;
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
std::string loaded_from() {
#ifdef _WIN32
  return "";
#else
  const void* const symbol = dlsym(RTLD_NEXT, "fftw_execute");
  Dl_info info{};
  if (symbol == nullptr || dladdr(symbol, &info) == 0 || info.dli_fname == nullptr) {
    return "";
  }
  return info.dli_fname;
#endif
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
    } else {
      say(message, message_size, "unknown FFTW option " + std::string(key));
      return fbench_error;
    }
  }
  fftw_forget_wisdom();
  if (!s.wisdom_in.empty()) {
    if (std::FILE* f = std::fopen(s.wisdom_in.c_str(), "r"); f != nullptr) {
      const int ok = fftw_import_wisdom_from_file(f);
      std::fclose(f);
      if (ok == 0) {
        say(message, message_size, "cannot import wisdom from " + s.wisdom_in);
        return fbench_error;
      }
    }
  }
  const std::string version = fftw_version;
  s.description = fbench::json()
                      .str("library", "fftw")
                      .str("version", version)
                      .str("isa", simd_of(version))
                      .str("precision", "double")
                      .raw("supports", R"(["c2c","r2c","c2r"])")
                      .str("planner", s.planner)
                      .str("cc", fftw_cc)
                      .str("codelet_optim", fftw_codelet_optim)
                      .str("loaded_from", loaded_from())
                      .str("expected_root", FBENCH_FFTW_ROOT)
                      .inner();
  return fbench_ok;
}

void fbench_end() {
  const session& s = state();
  if (!s.wisdom_out.empty()) {
    if (std::FILE* f = std::fopen(s.wisdom_out.c_str(), "w"); f != nullptr) {
      fftw_export_wisdom_to_file(f);
      std::fclose(f);
    }
  }
  fftw_cleanup();
}

const char* fbench_describe() { return state().description.c_str(); }

std::int32_t fbench_create(const fbench_case* c, void* in, void* out, void** plan, char* message,
                           std::size_t message_size) {
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
  const unsigned flags = state().flags;
  // No FFTW_UNALIGNED: the harness plans on buffers with the alignment of every later
  // execute, and FFTW_UNALIGNED silently drops FFTW's aligned SIMD codelets (finding E10).
  fftw_plan p = nullptr;
  if (c->kind == fbench_c2c) {
    p = fftw_plan_many_dft(c->rank, n.data(), batch, static_cast<fftw_complex*>(in),
                           in_embed.data(), 1, in_dist, static_cast<fftw_complex*>(out),
                           out_embed.data(), 1, out_dist, c->sign, flags);
  } else if (c->kind == fbench_r2c) {
    p = fftw_plan_many_dft_r2c(c->rank, n.data(), batch, static_cast<double*>(in), in_embed.data(),
                               1, in_dist, static_cast<fftw_complex*>(out), out_embed.data(), 1,
                               out_dist, flags);
  } else {
    p = fftw_plan_many_dft_c2r(c->rank, n.data(), batch, static_cast<fftw_complex*>(in),
                               in_embed.data(), 1, in_dist, static_cast<double*>(out),
                               out_embed.data(), 1, out_dist, flags);
  }
  if (p == nullptr) {
    say(message, message_size, "FFTW returned no plan");
    return fbench_error;
  }
  if (state().plan_note) {
    // The plan in FFTW's own notation, one line (experiment X25): which codelets, radices,
    // buffering and vector loops it chose. fftw_sprint_plan's string is malloc'ed.
    char* const text = fftw_sprint_plan(p);
    const std::string_view plan_text = text != nullptr ? text : "";
    std::string line;
    for (const char ch : plan_text) {
      const bool blank = ch == '\n' || ch == ' ';
      if (!blank || (!line.empty() && line.back() != ' ')) {
        line.push_back(blank ? ' ' : ch);
      }
    }
    std::free(text); // NOLINT(cppcoreguidelines-no-malloc): FFTW's API allocates with malloc
    say(message, message_size, line);
  }
  auto record = std::make_unique<plan_record>();
  record->plan = p;
  record->kind = c->kind;
  *plan = record.release();
  return fbench_ok;
}

void fbench_execute(const void* plan, void* in, void* out) {
  const auto* r = static_cast<const plan_record*>(plan);
  if (r->kind == fbench_c2c) {
    fftw_execute_dft(r->plan, static_cast<fftw_complex*>(in), static_cast<fftw_complex*>(out));
  } else if (r->kind == fbench_r2c) {
    fftw_execute_dft_r2c(r->plan, static_cast<double*>(in), static_cast<fftw_complex*>(out));
  } else {
    fftw_execute_dft_c2r(r->plan, static_cast<fftw_complex*>(in), static_cast<double*>(out));
  }
}

void fbench_destroy(void* plan) {
  const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan));
  fftw_destroy_plan(r->plan);
}
}
