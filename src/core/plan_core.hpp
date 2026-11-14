// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The type-erased plan behind plan<In, Out> (include/fftn/detail/core.hpp): the validated
// layout, the program, the tables it points into, and the workspace layout.
#pragma once

#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/options.hpp"
#include "isa/target.hpp"
#include "threading/schedule.hpp"
#include "threading/work_graph.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace fftn::inline v0::detail {

// Construction knobs for tests; plans built through the public API use the defaults.
struct build_settings {
  // About this many points of work per chunk of a stage (program.hpp) or of a run of
  // lines (work_graph.hpp). Experiment T5 sets the value; here it only has to give long
  // rows several chunks.
  std::size_t chunk_points = 2048;
  // The machine the plan's team executions are scheduled for (schedule.hpp).
  machine_profile profile = default_machine_profile();
};

class plan_core {
public:
  plan_core() = default;
  // Stages point into the tables, so a core never moves or copies.
  plan_core(const plan_core&) = delete;
  plan_core& operator=(const plan_core&) = delete;
  plan_core(plan_core&&) = delete;
  plan_core& operator=(plan_core&&) = delete;
  ~plan_core() = default;

  transform_kind kind = transform_kind::c2c;
  precision prec = precision::f64;
  direction dir = direction::forward;
  bool in_place = false;
  std::vector<std::size_t> lengths; // per axis
  std::vector<std::size_t> counts;  // per batch dimension
  target isa_target = target::scalar;

  program prog;
  work_graph graph;        // the program's shareable work (DESIGN.md 13.2)
  machine_profile profile; // what the scheduler assumes of the machine
  std::vector<std::vector<double>> tables_f64;
  std::vector<std::vector<float>> tables_f32;
  std::vector<std::string> axis_notes; // one line per axis for describe()

  // Workspace: scratch lines a and b of scratch_line complex values each, then the block,
  // each starting on a 64-byte boundary of the aligned workspace; bytes includes the slack
  // for aligning an arbitrary workspace address.
  std::size_t scratch_line = 0;
  std::size_t block_elements = 0;
  std::size_t offset_b = 0;
  std::size_t offset_block = 0;
  std::size_t bytes = 0;
};

[[nodiscard]] std::shared_ptr<const plan_core>
make_plan_core(const shape_view& shape, const options& opts, const build_settings& settings);

} // namespace fftn::inline v0::detail
