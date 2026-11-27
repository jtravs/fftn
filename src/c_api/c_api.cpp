// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The C API (fftn/c_api.h, step 6.8; DESIGN.md 5.9). The plan handle holds the same
// type-erased core as plan<In, Out>, and the C shape converts to the core's shape_view, so a C
// plan runs exactly the C++ plan's program. No exception leaves this file: every function that
// can fail catches everything and reports it in an fftn_error or a status.

#include "fftn/c_api.h"

#include "core/edges.hpp"
#include "core/wisdom_access.hpp"
#include "fftn/callback.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/edges.hpp"
#include "fftn/executor.h"
#include "fftn/executor.hpp"
#include "fftn/fftn.hpp"
#include "fftn/isa.hpp"
#include "fftn/lengths.hpp"
#include "fftn/options.hpp"
#include "fftn/pipeline.hpp"
#include "fftn/shape.hpp"
#include "fftn/team.hpp"
#include "fftn/wisdom.hpp"
#include "fftn/workspace.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <complex>
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
  // The C callbacks of the input edge, the middle (none for a plan) and the output edge. The
  // core holds call_c's instantiations in their place, and each execution hands it a thunk
  // with the C function and the caller's user pointer (c_exec_args).
  std::array<fftn_block_fn, 3> calls{};
  // The shape's kind and whether its complex sides are split (fftn_shape::layout): the _split
  // executions pass the imaginary arrays of the complex sides only, and every execution checks
  // that its form matches the plan's layout (DESIGN.md 18.2, Q31).
  fftn::detail::transform_kind kind = fftn::detail::transform_kind::c2c;
  bool split = false;
};

// A pipeline handle: as a plan's, around a pipeline's core.
struct fftn_pipeline {
  std::shared_ptr<const fftn::detail::plan_core> core;
  std::atomic<std::size_t> references{1};
  std::array<fftn_block_fn, 3> calls{};
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
static_assert(FFTN_DCT2 == static_cast<int>(fftn::r2r_kind::dct2));
static_assert(FFTN_DCT3 == static_cast<int>(fftn::r2r_kind::dct3));
static_assert(FFTN_DCT4 == static_cast<int>(fftn::r2r_kind::dct4));
static_assert(FFTN_DST1 == static_cast<int>(fftn::r2r_kind::dst1));
static_assert(FFTN_DST2 == static_cast<int>(fftn::r2r_kind::dst2));
static_assert(FFTN_DST3 == static_cast<int>(fftn::r2r_kind::dst3));
static_assert(FFTN_DST4 == static_cast<int>(fftn::r2r_kind::dst4));
static_assert(FFTN_DHT == static_cast<int>(fftn::r2r_kind::dht));
static_assert(FFTN_LENGTH_COMPLEX == static_cast<int>(fftn::length_kind::complex));
static_assert(FFTN_LENGTH_REAL == static_cast<int>(fftn::length_kind::real));
static_assert(FFTN_COMPLETION_ALL == static_cast<int>(fftn::completion::all));
static_assert(FFTN_COMPLETION_OWN == static_cast<int>(fftn::completion::own));
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

// The value a C caller stored in an enumeration-typed field or argument, read as an int (the
// enumeration's underlying type is the compiler's choice, int-sized here, unsigned int with
// GCC and Clang when no value is negative): a value outside the enumeration, which the C side
// may hold, is never loaded as the C++ enumeration type (UBSan's enum check would rightly flag
// it).
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
template <class T>
std::span<const T> array_of(const T* data, std::size_t count, const std::string& where) {
  if (data == nullptr && count != 0) {
    refuse(where, "is NULL but its count is not 0");
  }
  return data == nullptr ? std::span<const T>{} : std::span<const T>(data, count);
}

// The C stored runs as C++ ones; `where` names the array.
std::vector<fftn::stored_run> runs_of(const fftn_stored_run* data, std::size_t count,
                                      const std::string& where) {
  std::vector<fftn::stored_run> runs;
  for (const fftn_stored_run& r : array_of(data, count, where)) {
    const int side = raw_value(r.where);
    if (side != FFTN_IN && side != FFTN_OUT) {
      refuse(where + "[" + std::to_string(runs.size()) + "].where",
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
  return runs;
}

// The C shape as the plan core's shape_view, with the arrays it borrows converted into
// vectors this object owns (the C structs' enumerations do not have the C++ types' sizes).
struct converted_shape {
  std::vector<fftn::axis> axes;
  std::vector<fftn::batch> batches;
  std::vector<fftn::stored_run> runs;
  std::vector<fftn::r2r_kind> kinds;
  detail::shape_view view;

  // `function` names the C function in messages.
  converted_shape(const fftn_shape* s, const std::string& function) {
    const std::string where = function + ": shape";
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
    // The spans are named: built in the range statement from a temporary name, they would look
    // dangling to GCC's -Wdangling-reference.
    const std::span<const fftn_axis> c_axes = array_of(s->axes, s->rank, where + ".axes");
    for (const fftn_axis& a : c_axes) {
      axes.push_back({.n = a.n, .in_stride = a.in_stride, .out_stride = a.out_stride});
    }
    const std::span<const fftn_batch> c_batches =
        array_of(s->batches, s->batch_rank, where + ".batches");
    for (const fftn_batch& b : c_batches) {
      batches.push_back({.count = b.count, .in_dist = b.in_dist, .out_dist = b.out_dist});
    }
    runs = runs_of(s->runs, s->run_count, where + ".runs");
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
    const int layout = raw_value(s->layout);
    if (layout != FFTN_INTERLEAVED && layout != FFTN_SPLIT) {
      refuse(where + ".layout", "is " + std::to_string(layout) + ", not an fftn_layout");
    }
    if (layout == FFTN_SPLIT && kind == FFTN_R2R) {
      refuse(where + ".layout", "is FFTN_SPLIT; an FFTN_R2R shape has no complex side to split");
    }
    view.split = layout == FFTN_SPLIT;
    view.axes = axes;
    view.batches = batches;
    view.runs = runs;
    view.kinds = kinds;
  }
};

fftn::options options_of(const fftn_options* o, const std::string& function) {
  if (o == nullptr) {
    return {};
  }
  const std::string where = function + ": options.";
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

// fftn_plan_ref and fftn_plan_release for either handle.
template <class H> H* add_reference(H* h) noexcept {
  if (h != nullptr) {
    h->references.fetch_add(1, std::memory_order_relaxed);
  }
  return h;
}

template <class H> void drop_reference(H* h) noexcept {
  // The last release frees the handle: every earlier release happens before it.
  if (h != nullptr && h->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    destroy(h);
  }
}

// ---- Fusion (step 6.9) ----

static_assert(FFTN_FACTOR_NONE == static_cast<int>(fftn::factor_kind::none));
static_assert(FFTN_FACTOR_REAL == static_cast<int>(fftn::factor_kind::real));
static_assert(FFTN_FACTOR_COMPLEX == static_cast<int>(fftn::factor_kind::complex));
static_assert(FFTN_CALLBACK_MODIFY == static_cast<int>(fftn::callback_mode::modify));
static_assert(FFTN_CALLBACK_PRODUCE == static_cast<int>(fftn::callback_mode::produce));
static_assert(FFTN_CALLBACK_CONSUME == static_cast<int>(fftn::callback_mode::consume));
static_assert(FFTN_ALONG_TRANSFORMS == fftn::along_transforms);
static_assert(FFTN_NO_GROUP == fftn::no_group);

// What the core passes a C callback as its user pointer: the C function and the caller's own
// pointer, built by each execution on its caller's stack (c_exec_args). A team member's
// callbacks receive the pointers of its own call (edge_state_of copies them per member), and an
// executor's tasks run within the call, so the thunk outlives every call that reads it.
struct c_thunk {
  fftn_block_fn fn = nullptr;
  void* user = nullptr;
};

// The block_fn<E> the core calls in place of a C callback on elements of type E: it converts
// the block's place into the C struct and calls the C function. One instantiation per element
// type, so that the core calls it through its own type (core/callbacks.cpp, invoke_as). It is
// noexcept: an exception thrown by a "C" callback written in C++ ends the program here
// (std::terminate) and never reaches the caller of the execution (c_api.h, fftn_block_fn).
template <class E> void call_c(void* user, const fftn::block_info& info, E* data) noexcept {
  const auto* thunk = static_cast<const c_thunk*>(user);
  const fftn_block_info c{
      .transform = info.transform,
      .rank = info.origin.size(),
      .origin = info.origin.data(),
      .element_axis = info.element_axis,
      .line_axis = info.line_axis,
      .length = info.length,
      .lines = info.lines,
      .element_stride = info.element_stride,
      .line_stride = info.line_stride,
      .group = info.group,
      .group_stride = info.group_stride,
      .copy = info.copy,
      .member = info.member,
      .scratch = info.scratch.empty() ? nullptr : info.scratch.data(),
      .scratch_bytes = info.scratch.size(),
  };
  thunk->fn(thunk->user, &c, data);
}

template <class E> detail::erased_block_fn erased_call_c() noexcept {
  // converted back to block_fn<E> before any call, as fftn::detail::view_of converts an edge's
  return reinterpret_cast<detail::erased_block_fn>(&call_c<E>);
}

// The call_c for a side of complex or real elements in precision `p`.
detail::erased_block_fn call_c_for(bool complex, detail::precision p) noexcept {
  if (p == detail::precision::f64) {
    return complex ? erased_call_c<std::complex<double>>() : erased_call_c<double>();
  }
  return complex ? erased_call_c<std::complex<float>>() : erased_call_c<float>();
}

// A C factor's kind and strides into `view`; `where` names the factor.
void convert_factor(const fftn_factor& f, const std::string& where, detail::edge_view& view) {
  const int kind = raw_value(f.kind);
  if (kind < FFTN_FACTOR_NONE || kind > FFTN_FACTOR_COMPLEX) {
    refuse(where + ".kind", "is " + std::to_string(kind) + ", not an fftn_factor_kind");
  }
  view.kind = static_cast<fftn::factor_kind>(kind);
  view.axis_strides = array_of(f.axis_strides, f.axis_stride_count, where + ".axis_strides");
  view.batch_strides = array_of(f.batch_strides, f.batch_stride_count, where + ".batch_strides");
}

// A C callback into `view`, whose side holds complex or real elements in precision `p`; the C
// function goes to `keep`, the handle's.
void convert_callback(const fftn_callback& c, const std::string& where, bool complex,
                      detail::precision p, detail::edge_view& view, fftn_block_fn& keep) {
  const int mode = raw_value(c.mode);
  if (mode < FFTN_CALLBACK_MODIFY || mode > FFTN_CALLBACK_CONSUME) {
    refuse(where + ".mode", "is " + std::to_string(mode) + ", not an fftn_callback_mode");
  }
  // Without a function the core sees none, and judges the mode and scratch as the C++
  // constructor does (refused unless they are the defaults).
  view.call = c.fn == nullptr ? nullptr : call_c_for(complex, p);
  view.call_mode = static_cast<fftn::callback_mode>(mode);
  view.call_scratch = c.scratch_bytes;
  keep = c.fn;
}

// A C edge as the core's view; the strides stay borrowed from the C arrays.
detail::edge_view edge_view_of(const fftn_edge& e, const std::string& where, bool complex,
                               detail::precision p, fftn_block_fn& keep) {
  detail::edge_view v;
  v.scale = e.scale;
  convert_factor(e.mul, where + ".mul", v);
  v.accumulate = e.accumulate != 0;
  convert_callback(e.call, where + ".call", complex, p, v, keep);
  return v;
}

// The exec_args a core receives for a C execution: the factor arrays as given, and for each
// edge with a C callback a thunk in place of the user pointer. It points into itself, so it is
// built where the execution's call is and never copied or moved.
struct c_exec_args {
  std::array<c_thunk, 3> thunks{};
  fftn::exec_args args{};

  c_exec_args(const std::array<fftn_block_fn, 3>& calls, const fftn_exec_args* a) noexcept {
    const fftn_exec_args given = a == nullptr ? fftn_exec_args{} : *a;
    args.in_factor = given.in_factor;
    args.mid_factor = given.mid_factor;
    args.out_factor = given.out_factor;
    const std::array<void*, 3> users = {given.in_user, given.mid_user, given.out_user};
    const std::array<void**, 3> slots = {&args.in_user, &args.mid_user, &args.out_user};
    for (std::size_t i = 0; i < 3; ++i) {
      thunks.at(i) = {.fn = calls.at(i), .user = users.at(i)};
      *slots.at(i) = calls.at(i) != nullptr ? &thunks.at(i) : users.at(i);
    }
  }
  c_exec_args(const c_exec_args&) = delete;
  c_exec_args& operator=(const c_exec_args&) = delete;
  c_exec_args(c_exec_args&&) = delete;
  c_exec_args& operator=(c_exec_args&&) = delete;
  ~c_exec_args() = default;
};

// Whether a plan's input and output hold complex values.
bool complex_input(detail::transform_kind k) noexcept {
  return k == detail::transform_kind::c2c || k == detail::transform_kind::c2r;
}
bool complex_output(detail::transform_kind k) noexcept {
  return k == detail::transform_kind::c2c || k == detail::transform_kind::r2c;
}

// ---- Split complex ----

// Ends the program when an execution's form does not match the plan's layout (c_api.h,
// fftn_plan_create; DESIGN.md 18.2, Q31), as contract_failure does for a plan with factors
// executed without its arrays. C cannot refuse it at compile time as C++ does, and the serial,
// pieces and team executions cannot return an error (the executor forms could; Q31 keeps them
// alike): a split plan through an interleaved form would run without its imaginary arrays, an
// interleaved plan through a _split form would read the real parts as interleaved values.
void check_layout(const fftn_plan* plan, bool split_form) noexcept {
  if (plan->split == split_form) {
    return;
  }
  detail::contract_failure(
      split_form ? "fftn: a plan of interleaved data was executed through a _split form of the C "
                   "API (fftn_execute_split and the others take split-complex plans only)"
                 : "fftn: a split-complex plan was executed through a C execution without the "
                   "_split suffix (use fftn_execute_split and the others)");
}

// What a split C execution hands the core, as plan::execute's split overloads do in C++
// (detail::slot_array, detail::split_args): the real parts (or a real side's array) through the
// slots, and each complex side's imaginary array through its factor pointer. A real side's `im`
// is never read.
struct c_split_args {
  const void* in = nullptr;
  void* out = nullptr;
  fftn::exec_args args{};

  c_split_args(const fftn_plan* plan, fftn_split_const_ptr i, fftn_split_ptr o) noexcept
      : in(i.re), out(o.re) {
    args.in_factor = complex_input(plan->kind) ? i.im : nullptr;
    args.out_factor = complex_output(plan->kind) ? o.im : nullptr;
  }
};

// A C pipeline description as the core's view, with the arrays it borrows converted.
struct converted_pipeline {
  std::vector<fftn::batch> batches;
  std::vector<fftn::stored_run> runs;
  std::array<fftn_block_fn, 3> calls{};
  detail::pipeline_view view;

  explicit converted_pipeline(const fftn_pipeline_desc* d) {
    const std::string where = "fftn_pipeline_create: desc";
    if (d == nullptr) {
      refuse(where, "is NULL");
    }
    // Whether the input and output, and the middle, hold complex values.
    bool complex_ends = true;
    bool complex_middle = true;
    const int kind = raw_value(d->kind);
    switch (kind) {
    case FFTN_PIPELINE_CCC:
      view.kind = detail::pipeline_kind::complex_pair;
      if (d->first != FFTN_FORWARD && d->first != FFTN_BACKWARD) {
        refuse(where + ".first", "is " + std::to_string(d->first) +
                                     ", neither FFTN_FORWARD (-1) nor FFTN_BACKWARD (+1)");
      }
      view.first = static_cast<fftn::direction>(d->first);
      break;
    case FFTN_PIPELINE_TCT:
      view.kind = detail::pipeline_kind::real_convolution;
      complex_ends = false;
      break;
    case FFTN_PIPELINE_CTC:
      view.kind = detail::pipeline_kind::real_middle;
      complex_middle = false;
      break;
    default:
      refuse(where + ".kind", "is " + std::to_string(kind) + ", not an fftn_pipeline_kind");
    }
    const int precision = raw_value(d->precision);
    if (precision != FFTN_DOUBLE && precision != FFTN_FLOAT) {
      refuse(where + ".precision", "is " + std::to_string(precision) + ", not an fftn_precision");
    }
    view.prec = precision == FFTN_DOUBLE ? detail::precision::f64 : detail::precision::f32;
    view.line = {.n = d->line.n, .in_stride = d->line.in_stride, .out_stride = d->line.out_stride};
    const std::span<const fftn_batch> c_batches =
        array_of(d->batches, d->batch_rank, where + ".batches");
    for (const fftn_batch& b : c_batches) {
      batches.push_back({.count = b.count, .in_dist = b.in_dist, .out_dist = b.out_dist});
    }
    runs = runs_of(d->runs, d->run_count, where + ".runs");
    view.batches = batches;
    view.runs = runs;
    view.prologue =
        edge_view_of(d->prologue, where + ".prologue", complex_ends, view.prec, calls[0]);
    view.middle.scale = d->middle.scale;
    convert_factor(d->middle.mul, where + ".middle.mul", view.middle);
    convert_callback(d->middle.call, where + ".middle.call", complex_middle, view.prec, view.middle,
                     calls[1]);
    view.epilogue =
        edge_view_of(d->epilogue, where + ".epilogue", complex_ends, view.prec, calls[2]);
    view.fanout = d->fanout;
    view.fanout_out_dist = d->fanout_out_dist;
    view.fanout_factor_dist = d->fanout_factor_dist;
    view.group = d->group;
  }
};

// fftn_execute_on's checks and statuses, for a plan or a pipeline, with the C arguments.
template <class H>
int execute_on_c(const H* h, const void* in, void* out, const fftn_executor* executor,
                 fftn_workspace* workspace, const fftn_exec_args* args) noexcept {
  if (executor == nullptr || executor->concurrency == nullptr || executor->bulk == nullptr ||
      workspace == nullptr) {
    return FFTN_ERROR_INVALID_ARGUMENT;
  }
  try {
    const c_exec_args a(h->calls, args);
    detail::execute_on(*h->core, in, out, fftn::executor_ref(*executor), workspace->ws, a.args);
    return FFTN_OK;
  } catch (const std::bad_alloc&) {
    return FFTN_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return FFTN_ERROR_EXECUTOR;
  }
}

} // namespace

extern "C" {

fftn_options fftn_options_default(void) noexcept {
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
                            fftn_error* error) noexcept {
  try {
    const converted_shape s(shape, "fftn_plan_create");
    auto plan = std::make_unique<fftn_plan>();
    plan->core = detail::make_plan_core(s.view, options_of(options, "fftn_plan_create"));
    plan->kind = s.view.kind;
    plan->split = s.view.split;
    set_error(error, FFTN_OK, {});
    return plan.release();
  } catch (...) {
    report(error);
    return nullptr;
  }
}

fftn_plan* fftn_plan_ref(fftn_plan* plan) noexcept { return add_reference(plan); }

void fftn_plan_release(fftn_plan* plan) noexcept { drop_reference(plan); }

size_t fftn_plan_workspace_bytes(const fftn_plan* plan, int lanes) noexcept {
  return detail::workspace_bytes(*plan->core, lanes);
}

size_t fftn_plan_describe(const fftn_plan* plan, char* buffer, size_t size) noexcept {
  try {
    return copy_out(detail::describe(*plan->core), buffer, size);
  } catch (...) {
    return copy_out({}, buffer, size);
  }
}

int fftn_plan_instruction_set(const fftn_plan* plan) noexcept {
  return static_cast<int>(detail::instruction_set(*plan->core));
}

fftn_workspace* fftn_workspace_create(size_t bytes, fftn_error* error) noexcept {
  try {
    auto ws = std::make_unique<fftn_workspace>(bytes);
    set_error(error, FFTN_OK, {});
    return ws.release();
  } catch (...) {
    report(error);
    return nullptr;
  }
}

fftn_workspace* fftn_workspace_borrow(void* memory, size_t bytes, fftn_error* error) noexcept {
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

void fftn_workspace_destroy(fftn_workspace* workspace) noexcept { destroy(workspace); }

// The workspace arguments stay non-const, as the header declares them (DESIGN.md 5.9): an
// execution writes the workspace's memory and its team word.
// NOLINTBEGIN(misc-const-correctness)
void fftn_execute(const fftn_plan* plan, const void* in, void* out,
                  fftn_workspace* workspace) noexcept {
  check_layout(plan, false);
  detail::execute_all(*plan->core, in, out, workspace->ws);
}

void fftn_execute_transforms(const fftn_plan* plan, size_t first, size_t count, const void* in,
                             void* out, fftn_workspace* workspace) noexcept {
  check_layout(plan, false);
  detail::execute_core(*plan->core, first, count, in, out, workspace->ws);
}

void fftn_execute_team(const fftn_plan* plan, const void* in, void* out, fftn_team team,
                       fftn_workspace* workspace, fftn_completion done) noexcept {
  check_layout(plan, false);
  detail::execute_team(*plan->core, in, out, team_of(team), workspace->ws,
                       raw_value(done) == FFTN_COMPLETION_OWN ? fftn::completion::own
                                                              : fftn::completion::all);
}

fftn_transform_range fftn_transforms_of(const fftn_plan* plan, fftn_team team) noexcept {
  const fftn::transform_range r = detail::transforms_of(*plan->core, team_of(team));
  return {.first = r.first, .count = r.count};
}

int fftn_execute_on(const fftn_plan* plan, const void* in, void* out, const fftn_executor* executor,
                    fftn_workspace* workspace) noexcept {
  if (executor == nullptr || executor->concurrency == nullptr || executor->bulk == nullptr ||
      workspace == nullptr) {
    return FFTN_ERROR_INVALID_ARGUMENT;
  }
  check_layout(plan, false);
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

fftn_wisdom* fftn_wisdom_create(fftn_error* error) noexcept {
  try {
    auto w = std::make_unique<fftn_wisdom>();
    set_error(error, FFTN_OK, {});
    return w.release();
  } catch (...) {
    report(error);
    return nullptr;
  }
}

fftn_wisdom* fftn_wisdom_from_text(const char* text, fftn_error* error) noexcept {
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

fftn_wisdom* fftn_wisdom_load(const char* path, fftn_error* error) noexcept {
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

int fftn_wisdom_save(const fftn_wisdom* wisdom, const char* path, fftn_error* error) noexcept {
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

size_t fftn_wisdom_text(const fftn_wisdom* wisdom, char* buffer, size_t size) noexcept {
  try {
    return copy_out(wisdom->w.text(), buffer, size);
  } catch (...) {
    return copy_out({}, buffer, size);
  }
}

int fftn_wisdom_add(fftn_wisdom* wisdom, const fftn_plan* plan, fftn_error* error) noexcept {
  try {
    if (plan->split) {
      refuse("fftn_wisdom_add: plan", "is a split-complex plan, whose programs wisdom does not "
                                      "record (C++ offers no wisdom::add for it)");
    }
    detail::wisdom_access::add(wisdom->w, *plan->core);
    set_error(error, FFTN_OK, {});
    return FFTN_OK;
  } catch (...) {
    return report(error);
  }
}

size_t fftn_wisdom_size(const fftn_wisdom* wisdom) noexcept { return wisdom->w.size(); }

void fftn_wisdom_destroy(fftn_wisdom* wisdom) noexcept { destroy(wisdom); }

size_t fftn_next_fast_length(size_t n, fftn_length_kind kind, fftn_error* error) noexcept {
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

double fftn_relative_cost(size_t n, fftn_precision precision, int target,
                          fftn_error* error) noexcept {
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

const char* fftn_isa_name(int target) noexcept {
  return target < FFTN_ISA_SCALAR || target > FFTN_ISA_BEST
             ? "unknown"
             : fftn::isa_name(static_cast<fftn::isa>(target));
}

const char* fftn_version(void) noexcept { return fftn::version(); }

// ---- Fusion (step 6.9) ----

fftn_edge fftn_edge_default(void) noexcept {
  const fftn::edge<double> d{};
  fftn_edge e{};
  e.scale = d.scale;
  e.accumulate = d.accumulate ? 1 : 0;
  return e;
}

fftn_plan* fftn_plan_create_edges(const fftn_shape* shape, const fftn_options* options,
                                  const fftn_plan_edges* edges, fftn_error* error) noexcept {
  try {
    const std::string function = "fftn_plan_create_edges";
    const converted_shape s(shape, function);
    const fftn::options o = options_of(options, function);
    auto plan = std::make_unique<fftn_plan>();
    if (edges == nullptr) {
      plan->core = detail::make_plan_core(s.view, o);
    } else {
      const detail::edges_view e{
          .in = edge_view_of(edges->in, function + ": edges.in", complex_input(s.view.kind),
                             s.view.prec, plan->calls[0]),
          .out = edge_view_of(edges->out, function + ": edges.out", complex_output(s.view.kind),
                              s.view.prec, plan->calls[2]),
      };
      plan->core = detail::make_plan_core_with_edges(s.view, o, e);
    }
    plan->kind = s.view.kind;
    plan->split = s.view.split;
    set_error(error, FFTN_OK, {});
    return plan.release();
  } catch (...) {
    report(error);
    return nullptr;
  }
}

// NOLINTBEGIN(misc-const-correctness): as for the executions above
void fftn_execute_args(const fftn_plan* plan, const void* in, void* out, fftn_workspace* workspace,
                       const fftn_exec_args* args) noexcept {
  check_layout(plan, false);
  const c_exec_args a(plan->calls, args);
  detail::execute_all(*plan->core, in, out, workspace->ws, a.args);
}

void fftn_execute_transforms_args(const fftn_plan* plan, size_t first, size_t count, const void* in,
                                  void* out, fftn_workspace* workspace,
                                  const fftn_exec_args* args) noexcept {
  check_layout(plan, false);
  const c_exec_args a(plan->calls, args);
  detail::execute_core(*plan->core, first, count, in, out, workspace->ws, a.args);
}

void fftn_execute_team_args(const fftn_plan* plan, const void* in, void* out, fftn_team team,
                            fftn_workspace* workspace, fftn_completion done,
                            const fftn_exec_args* args) noexcept {
  check_layout(plan, false);
  const c_exec_args a(plan->calls, args);
  detail::execute_team(*plan->core, in, out, team_of(team), workspace->ws,
                       raw_value(done) == FFTN_COMPLETION_OWN ? fftn::completion::own
                                                              : fftn::completion::all,
                       a.args);
}

int fftn_execute_on_args(const fftn_plan* plan, const void* in, void* out,
                         const fftn_executor* executor, fftn_workspace* workspace,
                         const fftn_exec_args* args) noexcept {
  if (executor != nullptr && executor->concurrency != nullptr && executor->bulk != nullptr &&
      workspace != nullptr) { // otherwise refused by execute_on_c without executing
    check_layout(plan, false);
  }
  return execute_on_c(plan, in, out, executor, workspace, args);
}

fftn_pipeline_desc fftn_pipeline_desc_default(void) noexcept {
  const fftn::pipeline_desc<std::complex<double>, std::complex<double>, std::complex<double>> d{};
  fftn_pipeline_desc c{};
  c.kind = FFTN_PIPELINE_CCC;
  c.precision = FFTN_DOUBLE;
  c.first = static_cast<int>(d.first);
  c.prologue = fftn_edge_default();
  c.middle.scale = d.middle.scale;
  c.epilogue = fftn_edge_default();
  c.fanout = d.fanout;
  c.group = d.group;
  return c;
}

fftn_pipeline* fftn_pipeline_create(const fftn_pipeline_desc* desc, const fftn_options* options,
                                    fftn_error* error) noexcept {
  try {
    const converted_pipeline d(desc);
    auto pipeline = std::make_unique<fftn_pipeline>();
    pipeline->core =
        detail::make_pipeline_core(d.view, options_of(options, "fftn_pipeline_create"));
    pipeline->calls = d.calls;
    set_error(error, FFTN_OK, {});
    return pipeline.release();
  } catch (...) {
    report(error);
    return nullptr;
  }
}

fftn_pipeline* fftn_pipeline_ref(fftn_pipeline* pipeline) noexcept {
  return add_reference(pipeline);
}

void fftn_pipeline_release(fftn_pipeline* pipeline) noexcept { drop_reference(pipeline); }

size_t fftn_pipeline_workspace_bytes(const fftn_pipeline* pipeline, int lanes) noexcept {
  return detail::workspace_bytes(*pipeline->core, lanes);
}

size_t fftn_pipeline_describe(const fftn_pipeline* pipeline, char* buffer, size_t size) noexcept {
  try {
    return copy_out(detail::describe(*pipeline->core), buffer, size);
  } catch (...) {
    return copy_out({}, buffer, size);
  }
}

int fftn_pipeline_instruction_set(const fftn_pipeline* pipeline) noexcept {
  return static_cast<int>(detail::instruction_set(*pipeline->core));
}

size_t fftn_pipeline_transform_count(const fftn_pipeline* pipeline) noexcept {
  return detail::transform_count(*pipeline->core);
}

void fftn_pipeline_execute(const fftn_pipeline* pipeline, const void* in, void* out,
                           fftn_workspace* workspace, const fftn_exec_args* args) noexcept {
  const c_exec_args a(pipeline->calls, args);
  detail::execute_all(*pipeline->core, in, out, workspace->ws, a.args);
}

void fftn_pipeline_execute_transforms(const fftn_pipeline* pipeline, size_t first, size_t count,
                                      const void* in, void* out, fftn_workspace* workspace,
                                      const fftn_exec_args* args) noexcept {
  const c_exec_args a(pipeline->calls, args);
  detail::execute_core(*pipeline->core, first, count, in, out, workspace->ws, a.args);
}

void fftn_pipeline_execute_team(const fftn_pipeline* pipeline, const void* in, void* out,
                                fftn_team team, fftn_workspace* workspace, fftn_completion done,
                                const fftn_exec_args* args) noexcept {
  const c_exec_args a(pipeline->calls, args);
  detail::execute_team(*pipeline->core, in, out, team_of(team), workspace->ws,
                       raw_value(done) == FFTN_COMPLETION_OWN ? fftn::completion::own
                                                              : fftn::completion::all,
                       a.args);
}

fftn_transform_range fftn_pipeline_transforms_of(const fftn_pipeline* pipeline,
                                                 fftn_team team) noexcept {
  const fftn::transform_range r = detail::transforms_of(*pipeline->core, team_of(team));
  return {.first = r.first, .count = r.count};
}

int fftn_pipeline_execute_on(const fftn_pipeline* pipeline, const void* in, void* out,
                             const fftn_executor* executor, fftn_workspace* workspace,
                             const fftn_exec_args* args) noexcept {
  return execute_on_c(pipeline, in, out, executor, workspace, args);
}

// ---- Split complex ----

void fftn_execute_split(const fftn_plan* plan, fftn_split_const_ptr in, fftn_split_ptr out,
                        fftn_workspace* workspace) noexcept {
  check_layout(plan, true);
  const c_split_args a(plan, in, out);
  detail::execute_all(*plan->core, a.in, a.out, workspace->ws, a.args);
}

void fftn_execute_transforms_split(const fftn_plan* plan, size_t first, size_t count,
                                   fftn_split_const_ptr in, fftn_split_ptr out,
                                   fftn_workspace* workspace) noexcept {
  check_layout(plan, true);
  const c_split_args a(plan, in, out);
  detail::execute_core(*plan->core, first, count, a.in, a.out, workspace->ws, a.args);
}

void fftn_execute_team_split(const fftn_plan* plan, fftn_split_const_ptr in, fftn_split_ptr out,
                             fftn_team team, fftn_workspace* workspace,
                             fftn_completion done) noexcept {
  check_layout(plan, true);
  const c_split_args a(plan, in, out);
  detail::execute_team(*plan->core, a.in, a.out, team_of(team), workspace->ws,
                       raw_value(done) == FFTN_COMPLETION_OWN ? fftn::completion::own
                                                              : fftn::completion::all,
                       a.args);
}

int fftn_execute_on_split(const fftn_plan* plan, fftn_split_const_ptr in, fftn_split_ptr out,
                          const fftn_executor* executor, fftn_workspace* workspace) noexcept {
  if (executor == nullptr || executor->concurrency == nullptr || executor->bulk == nullptr ||
      workspace == nullptr) {
    return FFTN_ERROR_INVALID_ARGUMENT;
  }
  check_layout(plan, true);
  try {
    const c_split_args a(plan, in, out);
    detail::execute_on(*plan->core, a.in, a.out, fftn::executor_ref(*executor), workspace->ws,
                       a.args);
    return FFTN_OK;
  } catch (const std::bad_alloc&) {
    return FFTN_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return FFTN_ERROR_EXECUTOR;
  }
}
// NOLINTEND(misc-const-correctness)

} // extern "C"
