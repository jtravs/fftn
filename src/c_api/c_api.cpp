// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The C API (fftn/c_api.h, step 6.8; DESIGN.md 5.9). The plan handle holds the same
// type-erased core as plan<In, Out>, and the C shape converts to the core's shape_view, so a C
// plan runs exactly the C++ plan's program. No exception leaves this file: every function that
// can fail catches everything and reports it in an fftn_error or a status.

#include "fftn/c_api.h"

#include "core/wisdom_access.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/executor.h"
#include "fftn/executor.hpp"
#include "fftn/fftn.hpp"
#include "fftn/isa.hpp"
#include "fftn/lengths.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "fftn/team.hpp"
#include "fftn/wisdom.hpp"
#include "fftn/workspace.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// The handles. They live in the global namespace because the C header declares them there.
struct fftn_plan {
  std::shared_ptr<const fftn::detail::plan_core> core;
  // The handle's references (fftn_plan_ref); the core is shared with nothing else.
  std::atomic<std::size_t> references{1};
};

struct fftn_workspace {
  explicit fftn_workspace(std::size_t bytes) : ws(bytes) {}
  explicit fftn_workspace(std::span<std::byte> memory) noexcept : ws(memory) {}
  fftn::workspace ws;
};

struct fftn_wisdom {
  fftn_wisdom() = default;
  explicit fftn_wisdom(fftn::wisdom from) noexcept : w(std::move(from)) {}
  fftn::wisdom w;
};

namespace {

namespace detail = fftn::detail;

// The C enumerations mirror the C++ ones value for value, so they convert with a cast.
static_assert(FFTN_DCT1 == static_cast<int>(fftn::r2r_kind::dct1));
static_assert(FFTN_DCT4 == static_cast<int>(fftn::r2r_kind::dct4));
static_assert(FFTN_DST1 == static_cast<int>(fftn::r2r_kind::dst1));
static_assert(FFTN_DST4 == static_cast<int>(fftn::r2r_kind::dst4));
static_assert(FFTN_DHT == static_cast<int>(fftn::r2r_kind::dht));
static_assert(FFTN_ISA_SCALAR == static_cast<int>(fftn::isa::scalar));
static_assert(FFTN_ISA_X86_V2 == static_cast<int>(fftn::isa::x86_v2));
static_assert(FFTN_ISA_AVX == static_cast<int>(fftn::isa::avx));
static_assert(FFTN_ISA_AVX2 == static_cast<int>(fftn::isa::avx2));
static_assert(FFTN_ISA_AVX512 == static_cast<int>(fftn::isa::avx512));
static_assert(FFTN_ISA_NEON == static_cast<int>(fftn::isa::neon));
static_assert(FFTN_ISA_NEON_FCMA == static_cast<int>(fftn::isa::neon_fcma));
static_assert(FFTN_ISA_BEST == static_cast<int>(fftn::isa::best));
static_assert(FFTN_PLANNING_RULES == static_cast<int>(fftn::planning::rules));
static_assert(FFTN_PLANNING_QUICK == static_cast<int>(fftn::planning::quick));
static_assert(FFTN_PLANNING_MEASURE == static_cast<int>(fftn::planning::measure));
static_assert(FFTN_FORWARD == static_cast<int>(fftn::direction::forward));
static_assert(FFTN_BACKWARD == static_cast<int>(fftn::direction::backward));

// The value a C caller stored in an enumeration-typed field or argument, read as the int it
// is in C: a value outside the enumeration, which the C side may hold, is never loaded as the
// C++ enumeration type (UBSan's enum check would rightly flag it).
template <class E> int raw_value(const E& e) noexcept {
  static_assert(sizeof(E) == sizeof(int), "C enumerations are ints");
  int v = 0;
  std::memcpy(&v, &e, sizeof v);
  return v;
}

void set_error(fftn_error* error, int code, std::string_view message) noexcept {
  if (error == nullptr) {
    return;
  }
  error->code = code;
  const std::size_t n = std::min(message.size(), sizeof(error->message) - 1);
  if (n != 0) { // an empty view's data() may be null, which memcpy may not be given
    std::memcpy(static_cast<char*>(error->message), message.data(), n);
  }
  error->message[n] = '\0';
}

// The status of the exception being handled, with its message in `error`. Called only from
// a catch block. `file` says whether a std::runtime_error is a file that could not be read or
// written (wisdom's load and save) rather than a failure inside FFTN.
int report(fftn_error* error, bool file = false) noexcept {
  int code = FFTN_ERROR_INTERNAL;
  try {
    throw;
  } catch (const std::invalid_argument& e) {
    code = FFTN_ERROR_INVALID_ARGUMENT;
    set_error(error, code, e.what());
  } catch (const std::bad_alloc& e) {
    code = FFTN_ERROR_OUT_OF_MEMORY;
    set_error(error, code, e.what());
  } catch (const std::runtime_error& e) {
    code = file ? FFTN_ERROR_FILE : FFTN_ERROR_INTERNAL;
    set_error(error, code, e.what());
  } catch (const std::exception& e) {
    set_error(error, code, e.what());
  } catch (...) {
    set_error(error, code, "fftn: an exception that is not a std::exception");
  }
  return code;
}

[[noreturn]] void refuse(const std::string& where, const std::string& why) {
  throw std::invalid_argument(where + " " + why);
}

// A C array of `count` elements: NULL only when it is empty.
template <class T> std::span<const T> array_of(const T* data, std::size_t count, const char* name) {
  if (data == nullptr && count != 0) {
    refuse(std::string("fftn_plan_create: shape.") + name, "is NULL but its count is not 0");
  }
  return data == nullptr ? std::span<const T>{} : std::span<const T>(data, count);
}

// The C shape as the plan core's shape_view, with the arrays it borrows converted into
// vectors this object owns (the C structs' enumerations do not have the C++ types' sizes).
struct converted_shape {
  std::vector<fftn::axis> axes;
  std::vector<fftn::batch> batches;
  std::vector<fftn::stored_run> runs;
  std::vector<fftn::r2r_kind> kinds;
  detail::shape_view view;

  explicit converted_shape(const fftn_shape* s) {
    const std::string where = "fftn_plan_create: shape";
    if (s == nullptr) {
      refuse(where, "is NULL");
    }
    const int kind = raw_value(s->kind);
    switch (kind) {
    case FFTN_C2C:
      view.kind = detail::transform_kind::c2c;
      if (s->direction != FFTN_FORWARD && s->direction != FFTN_BACKWARD) {
        refuse(where + ".direction", "is " + std::to_string(s->direction) +
                                         ", neither FFTN_FORWARD (-1) nor FFTN_BACKWARD (+1)");
      }
      view.dir = static_cast<fftn::direction>(s->direction);
      break;
    case FFTN_R2C:
      view.kind = detail::transform_kind::r2c;
      view.dir = fftn::direction::forward;
      break;
    case FFTN_C2R:
      view.kind = detail::transform_kind::c2r;
      view.dir = fftn::direction::backward;
      break;
    case FFTN_R2R:
      view.kind = detail::transform_kind::r2r;
      view.dir = fftn::direction::forward;
      break;
    default:
      refuse(where + ".kind", "is " + std::to_string(kind) + ", not an fftn_kind");
    }
    const int precision = raw_value(s->precision);
    switch (precision) {
    case FFTN_DOUBLE:
      view.prec = detail::precision::f64;
      break;
    case FFTN_FLOAT:
      view.prec = detail::precision::f32;
      break;
    default:
      refuse(where + ".precision", "is " + std::to_string(precision) + ", not an fftn_precision");
    }
    for (const fftn_axis& a : array_of(s->axes, s->rank, "axes")) {
      axes.push_back({.n = a.n, .in_stride = a.in_stride, .out_stride = a.out_stride});
    }
    for (const fftn_batch& b : array_of(s->batches, s->batch_rank, "batches")) {
      batches.push_back({.count = b.count, .in_dist = b.in_dist, .out_dist = b.out_dist});
    }
    for (const fftn_stored_run& r : array_of(s->runs, s->run_count, "runs")) {
      const int side = raw_value(r.where);
      if (side != FFTN_IN && side != FFTN_OUT) {
        refuse(where + ".runs[" + std::to_string(runs.size()) + "].where",
               "is " + std::to_string(side) + ", neither FFTN_IN nor FFTN_OUT");
      }
      runs.push_back({
          .axis = r.axis,
          .where = side == FFTN_IN ? fftn::side::in : fftn::side::out,
          .logical = r.logical,
          .stored = r.stored,
          .count = r.count,
      });
    }
    if (kind == FFTN_R2R) {
      if (s->r2r_kinds == nullptr) {
        refuse(where + ".r2r_kinds", "is NULL; an FFTN_R2R shape needs one kind per axis");
      }
      for (const fftn_r2r_kind& k : std::span(s->r2r_kinds, s->rank)) {
        const int v = raw_value(k);
        if (v < FFTN_DCT1 || v > FFTN_DHT) {
          refuse(where + ".r2r_kinds[" + std::to_string(kinds.size()) + "]",
                 "is " + std::to_string(v) + ", not an fftn_r2r_kind");
        }
        kinds.push_back(static_cast<fftn::r2r_kind>(v));
      }
    } else if (s->r2r_kinds != nullptr) {
      refuse(where + ".r2r_kinds", "is not NULL; only an FFTN_R2R shape has kinds");
    }
    view.axes = axes;
    view.batches = batches;
    view.runs = runs;
    view.kinds = kinds;
  }
};

fftn::options options_of(const fftn_options* o) {
  if (o == nullptr) {
    return {};
  }
  const std::string where = "fftn_plan_create: options.";
  if (o->force_isa < FFTN_ISA_SCALAR || o->force_isa > FFTN_ISA_BEST) {
    refuse(where + "force_isa", "is " + std::to_string(o->force_isa) + ", not an fftn_isa");
  }
  if (o->effort < FFTN_PLANNING_RULES || o->effort > FFTN_PLANNING_MEASURE) {
    refuse(where + "effort", "is " + std::to_string(o->effort) + ", not an fftn_planning");
  }
  return {
      .in_place = o->in_place != 0,
      .scale = o->scale,
      .force_isa = static_cast<fftn::isa>(o->force_isa),
      .effort = static_cast<fftn::planning>(o->effort),
      .measure_seconds = o->measure_seconds,
      .measure_scratch_bytes = o->measure_scratch_bytes,
      .use = o->use == nullptr ? nullptr : &o->use->w,
  };
}

// snprintf's contract for a string the library made.
std::size_t copy_out(const std::string& text, char* buffer, std::size_t size) noexcept {
  if (size != 0) {
    const std::size_t n = std::min(text.size(), size - 1);
    std::memcpy(buffer, text.data(), n);
    buffer[n] = '\0';
  }
  return text.size();
}

fftn::team team_of(fftn_team t) noexcept {
  return {.rank = t.rank, .size = t.size, .group_size = t.group_size};
}

// Takes ownership from a handle the C caller passed back.
template <class T> void destroy(T* handle) noexcept { const std::unique_ptr<T> owned(handle); }

} // namespace

extern "C" {

fftn_options fftn_options_default(void) {
  const fftn::options d{};
  return {
      .in_place = d.in_place ? 1 : 0,
      .scale = d.scale,
      .force_isa = static_cast<int>(d.force_isa),
      .effort = static_cast<int>(d.effort),
      .measure_seconds = d.measure_seconds,
      .measure_scratch_bytes = d.measure_scratch_bytes,
      .use = nullptr,
  };
}

fftn_plan* fftn_plan_create(const fftn_shape* shape, const fftn_options* options,
                            fftn_error* error) {
  try {
    const converted_shape s(shape);
    auto plan = std::make_unique<fftn_plan>();
    plan->core = detail::make_plan_core(s.view, options_of(options));
    set_error(error, FFTN_OK, {});
    return plan.release();
  } catch (...) {
    report(error);
    return nullptr;
  }
}

fftn_plan* fftn_plan_ref(fftn_plan* plan) {
  if (plan != nullptr) {
    plan->references.fetch_add(1, std::memory_order_relaxed);
  }
  return plan;
}

void fftn_plan_release(fftn_plan* plan) {
  // The last release frees the plan: every earlier release happens before it.
  if (plan != nullptr && plan->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    destroy(plan);
  }
}

size_t fftn_plan_workspace_bytes(const fftn_plan* plan, int lanes) {
  return detail::workspace_bytes(*plan->core, lanes);
}

size_t fftn_plan_describe(const fftn_plan* plan, char* buffer, size_t size) {
  try {
    return copy_out(detail::describe(*plan->core), buffer, size);
  } catch (...) {
    return copy_out({}, buffer, size);
  }
}

int fftn_plan_instruction_set(const fftn_plan* plan) {
  return static_cast<int>(detail::instruction_set(*plan->core));
}

fftn_workspace* fftn_workspace_create(size_t bytes, fftn_error* error) {
  try {
    auto ws = std::make_unique<fftn_workspace>(bytes);
    set_error(error, FFTN_OK, {});
    return ws.release();
  } catch (...) {
    report(error);
    return nullptr;
  }
}

fftn_workspace* fftn_workspace_borrow(void* memory, size_t bytes, fftn_error* error) {
  try {
    if (memory == nullptr && bytes != 0) {
      refuse("fftn_workspace_borrow: memory", "is NULL but bytes is " + std::to_string(bytes));
    }
    const std::span<std::byte> span = memory == nullptr
                                          ? std::span<std::byte>{}
                                          : std::span(static_cast<std::byte*>(memory), bytes);
    auto ws = std::make_unique<fftn_workspace>(span);
    set_error(error, FFTN_OK, {});
    return ws.release();
  } catch (...) {
    report(error);
    return nullptr;
  }
}

void fftn_workspace_destroy(fftn_workspace* workspace) { destroy(workspace); }

// The workspace arguments stay non-const, as the header declares them (DESIGN.md 5.9): an
// execution writes the workspace's memory and its team word.
// NOLINTBEGIN(misc-const-correctness)
void fftn_execute(const fftn_plan* plan, const void* in, void* out, fftn_workspace* workspace) {
  detail::execute_all(*plan->core, in, out, workspace->ws);
}

void fftn_execute_transforms(const fftn_plan* plan, size_t first, size_t count, const void* in,
                             void* out, fftn_workspace* workspace) {
  detail::execute_core(*plan->core, first, count, in, out, workspace->ws);
}

void fftn_execute_team(const fftn_plan* plan, const void* in, void* out, fftn_team team,
                       fftn_workspace* workspace, fftn_completion done) {
  detail::execute_team(*plan->core, in, out, team_of(team), workspace->ws,
                       done == FFTN_COMPLETION_OWN ? fftn::completion::own : fftn::completion::all);
}

fftn_transform_range fftn_transforms_of(const fftn_plan* plan, fftn_team team) {
  const fftn::transform_range r = detail::transforms_of(*plan->core, team_of(team));
  return {.first = r.first, .count = r.count};
}

int fftn_execute_on(const fftn_plan* plan, const void* in, void* out, const fftn_executor* executor,
                    fftn_workspace* workspace) {
  if (executor == nullptr || executor->concurrency == nullptr || executor->bulk == nullptr ||
      workspace == nullptr) {
    return FFTN_ERROR_INVALID_ARGUMENT;
  }
  try {
    detail::execute_on(*plan->core, in, out, fftn::executor_ref(*executor), workspace->ws);
    return FFTN_OK;
  } catch (const std::bad_alloc&) {
    return FFTN_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return FFTN_ERROR_EXECUTOR;
  }
}
// NOLINTEND(misc-const-correctness)

fftn_wisdom* fftn_wisdom_create(fftn_error* error) {
  try {
    auto w = std::make_unique<fftn_wisdom>();
    set_error(error, FFTN_OK, {});
    return w.release();
  } catch (...) {
    report(error);
    return nullptr;
  }
}

fftn_wisdom* fftn_wisdom_from_text(const char* text, fftn_error* error) {
  try {
    if (text == nullptr) {
      refuse("fftn_wisdom_from_text: text", "is NULL");
    }
    auto w = std::make_unique<fftn_wisdom>(fftn::wisdom(std::string_view(text)));
    set_error(error, FFTN_OK, {});
    return w.release();
  } catch (...) {
    report(error);
    return nullptr;
  }
}

fftn_wisdom* fftn_wisdom_load(const char* path, fftn_error* error) {
  try {
    if (path == nullptr) {
      refuse("fftn_wisdom_load: path", "is NULL");
    }
    auto w = std::make_unique<fftn_wisdom>(fftn::wisdom::load(path));
    set_error(error, FFTN_OK, {});
    return w.release();
  } catch (...) {
    report(error, true);
    return nullptr;
  }
}

int fftn_wisdom_save(const fftn_wisdom* wisdom, const char* path, fftn_error* error) {
  try {
    if (path == nullptr) {
      refuse("fftn_wisdom_save: path", "is NULL");
    }
    wisdom->w.save(path);
    set_error(error, FFTN_OK, {});
    return FFTN_OK;
  } catch (...) {
    return report(error, true);
  }
}

size_t fftn_wisdom_text(const fftn_wisdom* wisdom, char* buffer, size_t size) {
  try {
    return copy_out(wisdom->w.text(), buffer, size);
  } catch (...) {
    return copy_out({}, buffer, size);
  }
}

int fftn_wisdom_add(fftn_wisdom* wisdom, const fftn_plan* plan, fftn_error* error) {
  try {
    detail::wisdom_access::add(wisdom->w, *plan->core);
    set_error(error, FFTN_OK, {});
    return FFTN_OK;
  } catch (...) {
    return report(error);
  }
}

size_t fftn_wisdom_size(const fftn_wisdom* wisdom) { return wisdom->w.size(); }

void fftn_wisdom_destroy(fftn_wisdom* wisdom) { destroy(wisdom); }

size_t fftn_next_fast_length(size_t n, fftn_length_kind kind, fftn_error* error) {
  try {
    const int k = raw_value(kind);
    if (k != FFTN_LENGTH_COMPLEX && k != FFTN_LENGTH_REAL) {
      refuse("fftn_next_fast_length: kind",
             "is " + std::to_string(k) + ", not an fftn_length_kind");
    }
    const std::size_t m = fftn::next_fast_length(n, static_cast<fftn::length_kind>(k));
    set_error(error, FFTN_OK, {});
    return m;
  } catch (...) {
    report(error);
    return 0;
  }
}

double fftn_relative_cost(size_t n, fftn_precision precision, int target, fftn_error* error) {
  try {
    const int p = raw_value(precision);
    if (p != FFTN_DOUBLE && p != FFTN_FLOAT) {
      refuse("fftn_relative_cost: precision",
             "is " + std::to_string(p) + ", not an fftn_precision");
    }
    if (target < FFTN_ISA_SCALAR || target > FFTN_ISA_BEST) {
      refuse("fftn_relative_cost: target", "is " + std::to_string(target) + ", not an fftn_isa");
    }
    const auto isa = static_cast<fftn::isa>(target);
    const double cost =
        p == FFTN_DOUBLE ? fftn::relative_cost<double>(n, isa) : fftn::relative_cost<float>(n, isa);
    set_error(error, FFTN_OK, {});
    return cost;
  } catch (...) {
    report(error);
    return 0;
  }
}

const char* fftn_isa_name(int target) {
  return target < FFTN_ISA_SCALAR || target > FFTN_ISA_BEST
             ? "unknown"
             : fftn::isa_name(static_cast<fftn::isa>(target));
}

const char* fftn_version(void) { return fftn::version(); }

} // extern "C"
