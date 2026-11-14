// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// A team of the caller's own std::threads executing one plan together, then each thread
// working on the rows it owns: the example quoted in docs/threading.md. Exits with 0 when
// the team's results are bit for bit those of serial execution.

#include <fftn/fftn.hpp>

#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <numbers>
#include <thread>
#include <vector>

namespace {

bool team_matches_serial() {
  const std::size_t n = 4096;
  const std::size_t count = 6;
  std::vector<std::complex<double>> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::polar(1.0, 2 * std::numbers::pi * static_cast<double>(i % 97) / 97.0);
  }

  // snippet-begin: team
  const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
  const int members = 4;

  // One workspace for the whole team: scratch for each member and the team's shared state.
  fftn::workspace ws(plan, members);

  std::vector<std::complex<double>> y(n * count);
  std::vector<std::thread> team;
  team.reserve(members);
  for (int rank = 0; rank < members; ++rank) {
    team.emplace_back([&, rank] {
      const fftn::team me{.rank = rank, .size = members};

      // Every member calls execute; it returns when the whole output is written.
      plan.execute(x.data(), y.data(), me, ws);

      // With completion::own a member returns as soon as its own rows are written, and
      // can go straight on to work on them: the rows of transforms_of(me).
      plan.execute(x.data(), y.data(), me, ws, fftn::completion::own);
      const fftn::transform_range mine = plan.transforms_of(me);
      for (std::size_t i = mine.first * n; i < (mine.first + mine.count) * n; ++i) {
        y[i] /= static_cast<double>(n);
      }
    });
  }
  for (std::thread& t : team) {
    t.join();
  }
  // snippet-end: team

  // The same transform, serially: the team's result is identical bit for bit.
  std::vector<std::complex<double>> serial(n * count);
  fftn::workspace serial_ws(plan);
  plan.execute(x.data(), serial.data(), serial_ws);
  for (std::size_t i = 0; i < serial.size(); ++i) {
    if (y[i] != serial[i] / static_cast<double>(n)) {
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  try {
    const bool same = team_matches_serial();
    std::printf("team result %s serial\n", same ? "equals" : "differs from");
    return same ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}
