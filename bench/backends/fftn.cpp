// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for FFTN.
//
// FFTN has no transform API yet (task 0.5), so every case is reported as unsupported and
// the backend only identifies the library. Wiring the walking skeleton in takes:
//   - fbench_create: build an FFTN shape from the case (kind, rank, dims, batch, sign,
//     placement, the embeddings in_last/out_last and the distances in_dist/out_dist), plan
//     it, allocate its workspace, and keep both in a heap record returned through `plan`;
//   - fbench_execute: execute the plan on (in, out) with the record's workspace;
//   - fbench_destroy: release the record;
//   - fbench_describe: list the kinds the plan handles in "supports" (from then on smoke
//     fails on any unsupported case of those kinds);
//   - fbench_begin: accept FFTN plan options (the arm's +OPTION=value list) and reject
//     unknown ones; fbench_describe: add the instruction set the plan selected and the
//     plan's own description.

#include "harness/backend.hpp"
#include "harness/json.hpp"

#include <fftn/fftn.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace {

std::string& description() {
  static std::string text;
  return text;
}

} // namespace

extern "C" {

std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size) {
  if (!std::string_view(options).empty()) {
    std::snprintf(message, message_size, "FFTN has no plan options yet: %s", options);
    return fbench_error;
  }
  description() = fbench::json()
                      .str("library", "fftn")
                      .str("version", fftn::version())
                      .str("isa", "none")
                      .str("precision", "double")
                      .raw("supports", "[]")
                      .inner();
  return fbench_ok;
}

void fbench_end() {}

const char* fbench_describe() { return description().c_str(); }

std::int32_t fbench_create(const fbench_case* /*c*/, void* /*in*/, void* /*out*/, void** plan,
                           char* message, std::size_t message_size) {
  *plan = nullptr;
  std::snprintf(message, message_size, "FFTN has no transform API yet (task 0.5)");
  return fbench_unsupported;
}

void fbench_execute(const void* /*plan*/, void* /*in*/, void* /*out*/) {}

void fbench_destroy(void* /*plan*/) {}
}
