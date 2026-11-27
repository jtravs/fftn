// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "fftn/wisdom.hpp"

#include "core/build_id.hpp"
#include "core/candidates.hpp"
#include "core/edges.hpp"
#include "core/plan_core.hpp"
#include "core/runs.hpp"
#include "core/wisdom_access.hpp"
#include "core/wisdom_entry.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "fftn/isa.hpp"
#include "fftn/shape.hpp"
#include "platform/cpu_model.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fftn::inline v0 {
namespace detail {

namespace {

std::vector<std::string_view> split_tabs(std::string_view line) {
  std::vector<std::string_view> f;
  std::size_t at = 0;
  while (true) {
    const std::size_t end = line.find('\t', at);
    f.push_back(line.substr(at, end == std::string_view::npos ? std::string_view::npos : end - at));
    if (end == std::string_view::npos) {
      return f;
    }
    at = end + 1;
  }
}

std::string hex16(std::uint64_t v) {
  std::string s(16, '0');
  for (std::size_t i = 16; i-- > 0;) {
    s[i] = "0123456789abcdef"[v & 15U];
    v >>= 4U;
  }
  return s;
}

std::optional<std::uint64_t> parse_hex16(std::string_view s) {
  if (s.size() != 16) {
    return std::nullopt;
  }
  std::uint64_t v = 0;
  for (const char ch : s) {
    unsigned d = 0;
    if (ch >= '0' && ch <= '9') {
      d = static_cast<unsigned>(ch - '0');
    } else if (ch >= 'a' && ch <= 'f') {
      d = static_cast<unsigned>(ch - 'a' + 10);
    } else {
      return std::nullopt;
    }
    v = (v << 4U) | d;
  }
  return v;
}

bool blank_or_comment(std::string_view line) {
  const std::size_t first = line.find_first_not_of(" \t\r");
  return first == std::string_view::npos || line[first] == '#';
}

std::string_view without_cr(std::string_view line) {
  return !line.empty() && line.back() == '\r' ? line.substr(0, line.size() - 1) : line;
}

// Where a line applies: its first five fields (format to shape key), or the whole line when
// it has fewer (another format's), so that adding the same foreign line twice keeps one.
std::string place_of_line(std::string_view line) {
  std::size_t at = 0;
  for (int field = 0; field < 5; ++field) {
    at = line.find('\t', at);
    if (at == std::string_view::npos) {
      return std::string(line);
    }
    ++at;
  }
  return std::string(line.substr(0, at - 1));
}

constexpr std::string_view header =
    "# fftn wisdom: one entry per line, tab-separated: format, build (version+identifier), "
    "processor, instruction set, shape, program, check";

} // namespace

std::string wisdom_build() { return std::string(fftn::version()) + "+" + build_id(); }

const std::string& wisdom_cpu() {
  // Read once: the processor model does not change while the process runs.
  static const std::string model = cpu_model();
  return model;
}

std::string wisdom_place(std::string_view isa, std::string_view key) {
  return std::string(wisdom_format) + '\t' + wisdom_build() + '\t' + wisdom_cpu() + '\t' +
         std::string(isa) + '\t' + std::string(key);
}

std::optional<wisdom_record> parse_record(std::string_view line) {
  const std::vector<std::string_view> f = split_tabs(without_cr(line));
  if (f.size() != 7 || f[0] != wisdom_format || f[1].empty() || f[2].empty() || f[3].empty() ||
      f[4].empty()) {
    return std::nullopt;
  }
  // Only the canonical text of a choice: "grid=3.5" or "radices=04.4" would name a program
  // other than the one written.
  const std::optional<plan_choice> choice = parse_choice(f[5]);
  if (!choice || choice_text(*choice) != f[5]) {
    return std::nullopt;
  }
  const std::optional<std::uint64_t> check = parse_hex16(f[6]);
  if (!check) {
    return std::nullopt;
  }
  return wisdom_record{
      .build = std::string(f[1]),
      .cpu = std::string(f[2]),
      .isa = std::string(f[3]),
      .key = std::string(f[4]),
      .program = std::string(f[5]),
      .check = *check,
  };
}

std::string record_text(const wisdom_record& r) {
  return std::string(wisdom_format) + '\t' + r.build + '\t' + r.cpu + '\t' + r.isa + '\t' + r.key +
         '\t' + r.program + '\t' + hex16(r.check);
}

wisdom_record record_of(const plan_core& c) {
  return {
      .build = wisdom_build(),
      .cpu = wisdom_cpu(),
      .isa = isa_name(instruction_set(c)),
      .key = c.shape_key,
      .program = c.program_text,
      .check = program_fingerprint(c),
  };
}

std::optional<wisdom_record> find_entry(const wisdom& w, const std::string& place) {
  const std::string* line = wisdom_access::find(w, place);
  return line == nullptr ? std::nullopt : parse_record(*line);
}

namespace {

const char* kind_word(transform_kind k) {
  switch (k) {
  case transform_kind::c2c:
    break;
  case transform_kind::r2c:
    return "r2c";
  case transform_kind::c2r:
    return "c2r";
  case transform_kind::r2r:
    return "r2r";
  }
  return "c2c";
}

} // namespace

std::string shape_key(const shape_view& s, const options& opts) {
  std::ostringstream o;
  o << kind_word(s.kind) << (s.prec == precision::f64 ? " double" : " float");
  if (s.kind == transform_kind::c2c) {
    o << (s.dir == direction::forward ? " forward" : " backward");
  }
  // The scale changes the results (and an edge operation of the program), so it is part of
  // the key, written exactly.
  std::array<char, 64> scale{};
  std::snprintf(scale.data(), scale.size(), "%a", opts.scale);
  o << (opts.in_place ? " in-place" : " out-of-place") << " scale " << scale.data() << " axes";
  for (const axis& a : s.axes) {
    o << ' ' << a.n << '/' << a.in_stride << '/' << a.out_stride;
  }
  o << " batches";
  for (const batch& b : s.batches) {
    o << ' ' << b.count << '/' << b.in_dist << '/' << b.out_dist;
  }
  o << runs_key(s);
  // A real-to-real shape's kinds (r2r_kind's values), one per axis; other kinds have none.
  if (s.kind == transform_kind::r2r) {
    o << " kinds";
    for (const r2r_kind k : s.kinds) {
      o << ' ' << static_cast<unsigned>(k);
    }
  }
  return o.str();
}

std::string pipeline_key(const pipeline_view& d, const options& opts) {
  const std::array<axis, 1> line{d.line};
  shape_view s;
  s.kind = transform_kind::c2c;
  s.prec = d.prec;
  s.dir = d.first;
  s.axes = line;
  s.batches = d.batches;
  s.runs = d.runs;
  std::ostringstream o;
  o << "pipeline "
    << (d.kind == pipeline_kind::complex_pair       ? "c2c-c2c"
        : d.kind == pipeline_kind::real_convolution ? "r2c-c2r"
                                                    : "c2r-r2c")
    << ' ' << shape_key(s, opts) << " fanout " << d.fanout << '/' << d.fanout_out_dist << '/'
    << d.fanout_factor_dist << " group "
    << (d.group == static_cast<std::size_t>(-1) ? std::string("none") : std::to_string(d.group))
    << " prologue-epilogue" << edges_key({.in = d.prologue, .out = d.epilogue}, opts.scale)
    << " middle" << edges_key({.in = d.middle, .out = {}}, 1.0);
  // The three scales always: edges_key writes nothing for an edge with only a scale, and the
  // scales change the results (shape_key holds options::scale).
  std::array<char, 64> scale{};
  for (const double v : {d.prologue.scale, d.middle.scale, d.epilogue.scale}) {
    std::snprintf(scale.data(), scale.size(), "%a", v);
    o << " scale " << scale.data();
  }
  return o.str();
}

} // namespace detail

wisdom::wisdom(std::string_view text) {
  const std::string build = detail::wisdom_build();
  std::size_t at = 0;
  std::size_t number = 0;
  while (at < text.size()) {
    const std::size_t end = std::min(text.find('\n', at), text.size());
    const std::string_view line = detail::without_cr(text.substr(at, end - at));
    at = end + 1;
    ++number;
    if (detail::blank_or_comment(line)) {
      continue;
    }
    const std::vector<std::string_view> f = detail::split_tabs(line);
    const bool ours = f.size() >= 2 && f[0] == detail::wisdom_format && f[1] == build;
    if (!line.starts_with("fftn-wisdom-") || (ours && !detail::parse_record(line))) {
      throw std::invalid_argument("fftn::wisdom: line " + std::to_string(number) +
                                  " is not a wisdom entry");
    }
    add_entry(std::string(line));
  }
}

wisdom wisdom::load(const std::filesystem::path& path) {
  const std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("fftn::wisdom::load: cannot read " + path.string());
  }
  std::ostringstream text;
  text << in.rdbuf();
  if (in.bad()) {
    throw std::runtime_error("fftn::wisdom::load: cannot read " + path.string());
  }
  try {
    return wisdom(text.str());
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(path.string() + ": " + e.what());
  }
}

void wisdom::save(const std::filesystem::path& path) const {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << text();
  out.flush();
  if (!out) {
    throw std::runtime_error("fftn::wisdom::save: cannot write " + path.string());
  }
}

std::string wisdom::text() const {
  std::string t(detail::header);
  t += '\n';
  for (const std::string& e : entries_) {
    t += e;
    t += '\n';
  }
  return t;
}

void wisdom::add(const wisdom& other) {
  for (const std::string& e : other.entries_) {
    add_entry(e);
  }
}

std::string wisdom::entry_of(const detail::plan_core& core) {
  return detail::record_text(detail::record_of(core));
}

void wisdom::add_entry(std::string entry) {
  std::string place = detail::place_of_line(entry);
  if (const auto it = index_.find(place); it != index_.end()) {
    entries_[it->second] = std::move(entry);
    return;
  }
  index_.emplace(std::move(place), entries_.size());
  entries_.push_back(std::move(entry));
}

} // namespace fftn::inline v0
