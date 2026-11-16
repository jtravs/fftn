// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "fftn/wisdom.hpp"

#include "core/candidates.hpp"
#include "core/plan_core.hpp"
#include "core/wisdom_entry.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "fftn/isa.hpp"
#include "platform/cpu_model.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fftn::inline v0 {
namespace detail {

struct wisdom_access {
  static const std::vector<std::string>& entries(const wisdom& w) { return w.entries_; }
};

namespace {

// The processor model, read once: it does not change while the process runs.
const std::string& this_cpu() {
  static const std::string model = cpu_model();
  return model;
}

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

bool same_place(const wisdom_record& a, const wisdom_record& b) {
  return a.version == b.version && a.cpu == b.cpu && a.isa == b.isa && a.key == b.key;
}

bool blank_or_comment(std::string_view line) {
  const std::size_t first = line.find_first_not_of(" \t\r");
  return first == std::string_view::npos || line[first] == '#';
}

constexpr std::string_view header =
    "# fftn wisdom: one entry per line, tab-separated: format, library version, processor, "
    "instruction set, shape, program, check";

} // namespace

std::optional<wisdom_record> parse_record(std::string_view line) {
  if (!line.empty() && line.back() == '\r') {
    line.remove_suffix(1);
  }
  const std::vector<std::string_view> f = split_tabs(line);
  if (f.size() != 7 || f[0] != wisdom_format || f[1].empty() || f[2].empty() || f[3].empty() ||
      f[4].empty() || !parse_choice(f[5])) {
    return std::nullopt;
  }
  const std::optional<std::uint64_t> check = parse_hex16(f[6]);
  if (!check) {
    return std::nullopt;
  }
  return wisdom_record{.version = std::string(f[1]),
                       .cpu = std::string(f[2]),
                       .isa = std::string(f[3]),
                       .key = std::string(f[4]),
                       .program = std::string(f[5]),
                       .check = *check};
}

std::string record_text(const wisdom_record& r) {
  return std::string(wisdom_format) + '\t' + r.version + '\t' + r.cpu + '\t' + r.isa + '\t' +
         r.key + '\t' + r.program + '\t' + hex16(r.check);
}

wisdom_record record_of(const plan_core& c) {
  return {.version = fftn::version(),
          .cpu = this_cpu(),
          .isa = isa_name(instruction_set(c)),
          .key = c.shape_key,
          .program = c.program_text,
          .check = program_fingerprint(c)};
}

std::optional<wisdom_record> find_entry(const wisdom& w, const plan_core& c) {
  const wisdom_record here = record_of(c);
  for (const std::string& line : wisdom_access::entries(w)) {
    std::optional<wisdom_record> r = parse_record(line);
    if (r && same_place(*r, here)) {
      return r;
    }
  }
  return std::nullopt;
}

std::string shape_key(const shape_view& s, const options& opts) {
  std::ostringstream o;
  o << (s.kind == transform_kind::c2c   ? "c2c"
        : s.kind == transform_kind::r2c ? "r2c"
                                        : "c2r")
    << (s.prec == precision::f64 ? " double" : " float");
  if (s.kind == transform_kind::c2c) {
    o << (s.dir == direction::forward ? " forward" : " backward");
  }
  o << (opts.in_place ? " in-place" : " out-of-place") << " axes";
  for (const axis& a : s.axes) {
    o << ' ' << a.n << '/' << a.in_stride << '/' << a.out_stride;
  }
  o << " batches";
  for (const batch& b : s.batches) {
    o << ' ' << b.count << '/' << b.in_dist << '/' << b.out_dist;
  }
  return o.str();
}

} // namespace detail

wisdom::wisdom(std::string_view text) {
  std::size_t at = 0;
  std::size_t number = 0;
  while (at < text.size()) {
    const std::size_t end = std::min(text.find('\n', at), text.size());
    const std::string_view line = text.substr(at, end - at);
    at = end + 1;
    ++number;
    if (detail::blank_or_comment(line)) {
      continue;
    }
    const std::optional<detail::wisdom_record> r = detail::parse_record(line);
    if (!r) {
      throw std::invalid_argument("fftn::wisdom: line " + std::to_string(number) +
                                  " is not a wisdom entry");
    }
    add_entry(detail::record_text(*r));
  }
}

wisdom wisdom::load(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
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
  const std::optional<detail::wisdom_record> r = detail::parse_record(entry);
  for (std::string& e : entries_) {
    const std::optional<detail::wisdom_record> old = detail::parse_record(e);
    if (r && old && detail::same_place(*old, *r)) {
      e = std::move(entry);
      return;
    }
  }
  entries_.push_back(std::move(entry));
}

} // namespace fftn::inline v0
