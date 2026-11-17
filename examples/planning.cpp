// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Measured planning and wisdom: the example quoted in docs/planning.md. A plan is measured
// once, its program recorded as wisdom and written out as text; a second plan built from
// that text runs the same program without measuring. Exits with 0 when the two plans'
// results are bitwise identical.

#include <fftn/fftn.hpp>

#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

bool same_results() {
  // snippet-begin: measure
  const auto shape = fftn::rows_c2c<double>(4096, 16, fftn::direction::forward);

  // Time the candidate programs once (a fraction of a second here) and keep the fastest.
  const fftn::c2c_plan<double> measured(shape, {.effort = fftn::planning::measure});

  // Record the choice; text() is what save() writes to a file.
  fftn::wisdom w;
  w.add(measured);
  const std::string text = w.text();

  // Later, perhaps in another run on the same machine: the same program, without timing.
  const fftn::wisdom loaded(text);
  const fftn::c2c_plan<double> again(shape, {.use = &loaded});
  // snippet-end: measure

  std::vector<std::complex<double>> x(std::size_t{4096} * 16);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = {static_cast<double>(i % 7), static_cast<double>(i % 5)};
  }
  std::vector<std::complex<double>> y1(x.size());
  std::vector<std::complex<double>> y2(x.size());
  fftn::workspace ws(measured.workspace_bytes());
  measured.execute(x.data(), y1.data(), ws);
  fftn::workspace ws2(again.workspace_bytes());
  again.execute(x.data(), y2.data(), ws2);
  if (again.describe().find("from wisdom") == std::string::npos) {
    std::fputs(again.describe().c_str(), stderr);
    return false;
  }
  return std::memcmp(y1.data(), y2.data(), y1.size() * sizeof(y1[0])) == 0;
}

} // namespace

int main() {
  try {
    if (!same_results()) {
      std::fputs("planning example: the plan from wisdom differs from the measured plan\n", stderr);
      return 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "planning example: %s\n", e.what());
    return 1;
  }
  return 0;
}
