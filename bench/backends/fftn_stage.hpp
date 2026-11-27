// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The free-space stage as an fbench case (task 5.3h, experiment X68,
// dev/experiments/X68-free-space-stage.md; DESIGN.md 12.5): the FFTN backend's STAGE option.
//
// A case is a field of `batch` components (1, or 2 coupled ones), each stored as (y, x, w): c2c,
// rank 3, dims ny x nx x nw, in place, double precision, w the stored frequencies, contiguous. The
// stage takes the field through
//
//     the propagator (exp(i kz dz) from coordinates) -> inverse (y, x) planes -> along w: to the
//     time line, the nonlinear term (cubic; coupled across components), back -> forward (y, x)
//     planes -> the finish (a pointwise linear update with the stage's normalisation)
//
// and writes it back over the field. The time line is complex (an envelope: nt = nw points, every
// bin stored) unless STAGE_TIME names the case's stored count: then it is real, of nt points, with
// the stored bins a band [first, first + nw) of its nt/2 + 1 (stored runs). The arithmetic's
// constants are read at run time and make the stage the identity in exact arithmetic, every element
// still paying its work (as EDGE_CALL's, X66): fbench checks the output against the input
// ("output":"identity", harness/check.hpp).
//
// The arms (STAGE=):
//   T   today's route: plain plans and the application's loops in the client application's order
//       (a scratch volume in plane-major order, rows in node chunks, transposing crossings);
//   O   the three objects of DESIGN.md 12.5: inverse planes with a prologue callback, a spectral
//       pair on the w pencils with a middle callback, forward planes with an epilogue callback;
//   Of  O with the propagator as a complex factor array (the vocabulary) instead of the callback;
//   G   the (x, w) slab grouping emulated with the same objects: the inverse y plan, then per y
//       slab, run whole by the member that owns it, the x plan, the pencils' pair and the x plan
//       back, then the forward y plan;
//   C   one in-place sweep of the field (a copy loop multiplying by one): the bandwidth reference
//       of the traffic model, not a stage.
#pragma once

#include "harness/backend.hpp"

#include <fftn/executor.hpp>
#include <fftn/isa.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fbench::stage {

enum class arm_kind : unsigned char { today, objects, objects_factor, slabs, copy };

// A real time line for cases of `stored` frequencies: nt points, the band from bin `first`.
struct time_line {
  std::size_t stored = 0;
  std::size_t nt = 0;
  std::size_t first = 0;
};

struct config {
  arm_kind arm = arm_kind::objects;
  std::vector<time_line> real_lines;      // STAGE_TIME
  fftn::isa isa = fftn::isa::best;        // ISA
  int threads = 1;                        // THREADS
  std::optional<fftn::executor_ref> exec; // the session's executor; none: serial
  // STAGE_SELFCHECK (a test): before the arm is made, it and T run the stage with work that is
  // not the identity (coordinates and order matter) on one field, and must agree to 1e-12.
  bool selfcheck = false;
  bool nonlinear = false; // that work (internal)
  // STAGE_PROFILE (a diagnosis): the case's note starts with each step's time, from three runs.
  bool profile = false;
};

class arm {
public:
  arm() = default;
  arm(const arm&) = delete;
  arm& operator=(const arm&) = delete;
  arm(arm&&) = delete;
  arm& operator=(arm&&) = delete;
  virtual ~arm() = default;
  // Runs the stage on the field in place.
  virtual void run(void* field) const = 0;
  // The case's note: the objects' describe lines and the route's sweeps.
  [[nodiscard]] virtual std::string note() const = 0;
};

// T, O, Of, G or C.
std::optional<arm_kind> parse_arm(std::string_view text);
// STORED.NT.FIRST[,STORED.NT.FIRST...]; nullopt when malformed (FIRST + STORED must fit in the
// half spectrum's interior: 1 <= FIRST and FIRST + STORED <= NT/2, NT even).
std::optional<std::vector<time_line>> parse_time(std::string_view text);
// The arm of a case, or nullptr with `why` when the case is not a stage (not c2c rank 3 in place
// in double, more than 2 components).
std::unique_ptr<arm> make(const fbench_case& c, const config& cfg, std::string& why);

} // namespace fbench::stage
