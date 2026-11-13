// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// A minimal JSON object writer for the harness's output lines and the backends' describe.
#pragma once

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

namespace fbench {

inline std::string json_quote(std::string_view s) {
  std::string out(1, '"');
  for (const char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20U) {
      std::array<char, 8> esc{};
      std::snprintf(esc.data(), esc.size(), R"(\u%04x)", static_cast<unsigned>(c));
      out += esc.data();
    } else {
      out += c;
    }
  }
  out += '"';
  return out;
}

// Shortest round-trip form; non-finite values become null.
inline std::string json_number(double v) {
  if (!std::isfinite(v)) {
    return "null";
  }
  std::array<char, 32> text{};
  const auto res = std::to_chars(text.data(), text.data() + text.size(), v);
  return {text.data(), res.ptr};
}

class json {
public:
  json& str(std::string_view key, std::string_view v) { return raw(key, json_quote(v)); }
  json& num(std::string_view key, double v) { return raw(key, json_number(v)); }
  json& integer(std::string_view key, std::int64_t v) { return raw(key, std::to_string(v)); }
  json& boolean(std::string_view key, bool v) { return raw(key, v ? "true" : "false"); }

  json& numbers(std::string_view key, std::span<const double> values) {
    std::string list(1, '[');
    for (const double v : values) {
      list += (list.size() > 1 ? "," : "") + json_number(v);
    }
    list += ']';
    return raw(key, list);
  }

  // A value that is already JSON text.
  json& raw(std::string_view key, std::string_view text) {
    separate();
    text_ += json_quote(key);
    text_ += ':';
    text_ += text;
    return *this;
  }

  // Members (`"a":1,"b":2`) that are already JSON text.
  json& members(std::string_view text) {
    if (!text.empty()) {
      separate();
      text_ += text;
    }
    return *this;
  }

  // The members without braces, for a backend's describe.
  [[nodiscard]] std::string inner() const { return text_; }
  [[nodiscard]] std::string object() const { return '{' + text_ + '}'; }

private:
  void separate() {
    if (!text_.empty()) {
      text_ += ',';
    }
  }

  std::string text_;
};

} // namespace fbench
