// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "core/edge_driver.hpp"

#include "core/driver.hpp"
#include "core/edges.hpp"
#include "core/program.hpp"
#include "fftn/edges.hpp"
#include "passes/pass_args.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

constexpr const char* missing_array = "fftn: a plan or pipeline with factors was executed "
                                      "without the factor arrays it needs (exec_args)";

const void* source_of(slot s, const slot_addresses& at) noexcept {
  switch (s) {
  case slot::in:
    return at.in;
  case slot::out:
    return at.out;
  case slot::a:
    return at.a;
  case slot::b:
    return at.b;
  case slot::block:
    break;
  }
  return at.block;
}

// The input is never a destination (program.hpp).
void* destination_of(slot s, const slot_addresses& at) noexcept {
  switch (s) {
  case slot::out:
  case slot::in:
    return at.out;
  case slot::a:
    return at.a;
  case slot::b:
    return at.b;
  case slot::block:
    break;
  }
  return at.block;
}

// driver.cpp's moved(): a slot a loop does not move may be null, and takes no offset.
slot_addresses moved(const slot_addresses& at, const std::array<std::ptrdiff_t, moving_slots>& step,
                     std::size_t times) noexcept {
  const auto k = static_cast<std::ptrdiff_t>(times);
  slot_addresses next = at;
  if (step[0] != 0) {
    next.in += step[0] * k;
  }
  if (step[1] != 0) {
    next.out += step[1] * k;
  }
  if (step[2] != 0) {
    next.block += step[2] * k;
  }
  next.tw += step[3] * k;
  return next;
}

factor_cursors moved(factor_cursors f, const factor_steps& step, std::size_t times) noexcept {
  const auto k = static_cast<std::ptrdiff_t>(times);
  f.in += static_cast<std::uintptr_t>(step[0] * k);
  f.out += static_cast<std::uintptr_t>(step[1] * k);
  f.mid += static_cast<std::uintptr_t>(step[2] * k);
  return f;
}

factor_cursors moved_by(factor_cursors f, const factor_steps& bytes) noexcept {
  f.in += static_cast<std::uintptr_t>(bytes[0]);
  f.out += static_cast<std::uintptr_t>(bytes[1]);
  f.mid += static_cast<std::uintptr_t>(bytes[2]);
  return f;
}

// Sets the factor pointers of `a` that read cursors (`moving`, stage::moving_factors) from f:
// in_factor from the input's cursor or the middle's, out_factor from the output's or the
// middle's.
void set_cursors(pass_args& a, std::uint8_t moving, factor_cursors f) noexcept {
  if ((moving & moving_in_factor) != 0) {
    a.in_factor = reinterpret_cast<const void*>(f.in);
  } else if ((moving & moving_mid_in_factor) != 0) {
    a.in_factor = reinterpret_cast<const void*>(f.mid);
  }
  if ((moving & moving_out_factor) != 0) {
    a.out_factor = reinterpret_cast<const void*>(f.out);
  } else if ((moving & moving_mid_out_factor) != 0) {
    a.out_factor = reinterpret_cast<const void*>(f.mid);
  }
}

// The bytes a loop moves slot s per iteration (0 for the scratch lines, which never move).
std::ptrdiff_t step_of(const loop& l, slot s) noexcept {
  switch (s) {
  case slot::in:
    return l.step[0];
  case slot::out:
    return l.step[1];
  case slot::block:
    return l.step[2];
  case slot::a:
  case slot::b:
    break;
  }
  return 0;
}

// Whether a stage reading the cursors `moving` reads the same factors on every iteration of a
// loop that moves them by `step`.
bool cursors_fixed(std::uint8_t moving, const factor_steps& step) noexcept {
  return ((moving & moving_in_factor) == 0 || step[0] == 0) &&
         ((moving & moving_out_factor) == 0 || step[1] == 0) &&
         ((moving & moving_mid_factor) == 0 || step[2] == 0);
}

// The bytes the factor of an edge pass reading the cursor `moving` (one of them: the input's, the
// output's or the middle's) moves per iteration of a loop moving the cursors by `step`.
std::ptrdiff_t block_factor_step(std::uint8_t moving, const factor_steps& step) noexcept {
  if ((moving & moving_out_factor) != 0) {
    return step[1];
  }
  if ((moving & moving_mid_factor) != 0) {
    return step[2];
  }
  return step[0];
}

// `a` with the cursors in the factors a stage reads (`moving`), copied into `local`; `a` itself
// where it reads none.
const pass_args* with_cursors(const pass_args& a, std::uint8_t moving, factor_cursors f,
                              pass_args& local) noexcept {
  if (moving == 0) {
    return &a;
  }
  local = a;
  set_cursors(local, moving, f);
  return &local;
}

// Chunks [first, last) of stage st (record e), as driver.cpp's run_stage_chunks runs them, with
// the cursors in the arguments of a stage that reads them and its kernels from e.
void run_stage(const stage& st, const edge_record& e, const slot_addresses& at, factor_cursors f,
               std::size_t first, std::size_t last) noexcept {
  if (st.moving_factors == 0) {
    run_stage_chunks(st, at, first, last);
    return;
  }
  pass_args args = st.args;
  if (st.moving_twiddles) {
    args.twiddles = static_cast<const std::byte*>(args.twiddles) + at.tw;
  }
  set_cursors(args, st.moving_factors, f);
  const void* src = source_of(st.src, at);
  void* dst = destination_of(st.dst, at);
  if (e.leaf != nullptr) {
    if (first == 0 && last != 0) {
      e.leaf(&args, src, dst, 0, 0, 1);
    }
    return;
  }
  const std::size_t begin = chunk_start(st.units, st.chunks, first);
  const std::size_t end = chunk_start(st.units, st.chunks, last);
  if (begin < end) {
    e.run(&args, src, dst, begin, end);
  }
}

void run_records(const program& p, const std::vector<edge_record>& e, std::size_t begin,
                 std::size_t end, const slot_addresses& at, factor_cursors f) noexcept {
  for (std::size_t i = begin; i < end; ++i) {
    const record& r = p.records[i];
    if (const stage* st = std::get_if<stage>(&r)) {
      run_stage(*st, e[i], at, f, 0, st->chunks);
      continue;
    }
    // Every line on its own: a loop of one leaf gives each line the bits of the one call
    // driver.cpp makes of it (experiment X40).
    const loop& l = *std::get_if<loop>(&r);
    const slot_addresses from = moved(at, l.start, 1);
    for (std::size_t c = 0; c < l.count; ++c) {
      run_records(p, e, i + 1, l.body_end, moved(from, l.step, c), moved(f, e[i].factor_step, c));
    }
    i = l.body_end - 1;
  }
}

// Iteration `index` of the loops [begin, begin + depth), the last varying fastest.
std::pair<slot_addresses, factor_cursors> moved_in_nest(const program& p,
                                                        const std::vector<edge_record>& e,
                                                        std::size_t begin, std::size_t depth,
                                                        std::size_t index, const slot_addresses& at,
                                                        factor_cursors f) noexcept {
  slot_addresses here = at;
  std::size_t rest = index;
  for (std::size_t i = begin + depth; i > begin && rest != 0; --i) {
    const loop& l = *std::get_if<loop>(&p.records[i - 1]);
    here = moved(here, l.step, rest % l.count);
    f = moved(f, e[i - 1].factor_step, rest % l.count);
    rest /= l.count;
  }
  return {here, f};
}

// A stage of a lanes block (driver.cpp's run_lane_stage), with the cursors in the arguments of
// a stage that reads them and its kernel from e.
void run_lane_stage(const stage& st, const edge_record& e, const slot_addresses& at,
                    factor_cursors f, std::size_t rows) noexcept {
  pass_args args = st.args;
  args.rows = static_cast<std::uint32_t>(rows);
  pass_fn run = st.run;
  if (st.moving_factors != 0) {
    set_cursors(args, st.moving_factors, f);
    run = e.run;
  }
  run(&args, source_of(st.src, at), destination_of(st.dst, at), 0, st.units);
}

// Records [begin, end) of a lanes program's body or tail for a block of `rows` rows (driver.cpp's
// run_lane_records), the loops moving the cursors too.
void run_lane_records(const std::vector<record>& records, const std::vector<edge_record>& e,
                      std::size_t begin, std::size_t end, const slot_addresses& at,
                      factor_cursors f, std::size_t rows) noexcept {
  for (std::size_t i = begin; i < end; ++i) {
    const record& r = records[i];
    if (const stage* st = std::get_if<stage>(&r)) {
      run_lane_stage(*st, e[i], at, f, rows);
      continue;
    }
    const loop& l = *std::get_if<loop>(&r);
    const slot_addresses from = moved(at, l.start, 1);
    for (std::size_t c = 0; c < l.count; ++c) {
      run_lane_records(records, e, i + 1, l.body_end, moved(from, l.step, c),
                       moved(f, e[i].factor_step, c), rows);
    }
    i = l.body_end - 1;
  }
}

// Transforms [first, first + count) of a lanes program (driver.cpp's run_lane_blocks): the
// cursors move with the blocks by the innermost batch loop's factor steps.
void run_lane_blocks(const program& p, const std::vector<edge_record>& e,
                     const std::vector<edge_record>& tail, std::size_t first, std::size_t count,
                     const slot_addresses& base, factor_cursors f) noexcept {
  const std::size_t inner_index = p.batch_loops - 1;
  const loop& inner = *std::get_if<loop>(&p.records[inner_index]);
  const std::size_t end = first + count;
  std::size_t t = first;
  while (t < end) {
    slot_addresses at = transform_addresses(p, t, base);
    factor_cursors cursors = edge_transform_cursors(p, e, t, f);
    const std::size_t in_loop = inner.count - (t < inner.count ? t : t % inner.count);
    std::size_t run = end - t < in_loop ? end - t : in_loop;
    t += run;
    while (run != 0) {
      const bool whole = run >= p.lanes;
      std::size_t rows = p.lanes;
      if (!whole) {
        rows = run < p.lane_unit ? run : p.lane_unit;
      }
      if (whole) {
        run_lane_records(p.records, e, p.batch_loops, p.records.size(), at, cursors, rows);
      } else {
        run_lane_records(p.lane_tail, tail, 0, p.lane_tail.size(), at, cursors, rows);
      }
      run -= rows;
      if (run != 0) {
        at = moved(at, inner.step, rows);
        cursors = moved(cursors, e[inner_index].factor_step, rows);
      }
    }
  }
}

// The arguments a flat body's stages run with over one execution: a stage that reads cursors
// from a copy of its arguments made once, whose factor pointers are set for each transform (a
// copy per call cost 3-5 ns per stage on x86, X28); the others from the plan's own. The copies
// live in raw storage, constructed for those stages only, so that an execution of one short row
// pays for no others.
class flat_args {
public:
  explicit flat_args(const edge_forms& forms) noexcept : forms_(forms) {
    for (std::size_t k = 0; k < forms.flat.size(); ++k) {
      if (forms.flat[k].moving != 0) {
        copy_[k] =
            std::construct_at(reinterpret_cast<pass_args*>(storage_.data() + k * sizeof(pass_args)),
                              forms.flat[k].s.args);
      }
    }
  }
  // Stage k's arguments with the cursors f.
  [[nodiscard]] const pass_args* of(std::size_t k, factor_cursors f) noexcept {
    pass_args* a = copy_[k];
    if (a == nullptr) {
      return &forms_.flat[k].s.args;
    }
    set_cursors(*a, forms_.flat[k].moving, f);
    return a;
  }

private:
  const edge_forms& forms_;
  std::array<pass_args*, flat_stages_max> copy_{};
  alignas(pass_args) std::array<std::byte, sizeof(pass_args) * flat_stages_max> storage_;
};

// Stage k of a flat body on the slots `at`, whole (driver.cpp's run_whole), with the cursors f:
// the arguments and units the walked form gives it (chunks 0 to `chunks`).
void run_flat_stage(const edge_forms& forms, std::size_t k, flat_args& args,
                    const slot_addresses& at, factor_cursors f) noexcept {
  const edge_flat_stage& x = forms.flat[k];
  const pass_args* a = args.of(k, f);
  if (x.s.leaf != nullptr) {
    x.s.leaf(a, source_of(x.s.src, at), destination_of(x.s.dst, at), 0, 0, 1);
    return;
  }
  if (x.s.units != 0) {
    x.s.run(a, source_of(x.s.src, at), destination_of(x.s.dst, at), 0, x.s.units);
  }
}

// Transforms [first, first + count) of a flat program (driver.cpp's run_flat and run_leaf_rows),
// the cursors moved per transform. count is at least 1.
void run_edge_flat(const program& p, const std::vector<edge_record>& e, const edge_forms& forms,
                   std::size_t first, std::size_t count, const slot_addresses& base,
                   factor_cursors f) noexcept {
  const loop* only = p.batch_loops == 1 ? std::get_if<loop>(p.records.data()) : nullptr;
  slot_addresses at = transform_addresses(p, first, base);
  factor_cursors cursors = edge_transform_cursors(p, e, first, f);
  if (forms.leaf_rows && only != nullptr && cursors_fixed(forms.flat[0].moving, e[0].factor_step)) {
    // One call of the leaf over the rows, as run_leaf_rows makes it, every row reading the same
    // factors (the leaf computes each row as it would alone, X40).
    const edge_flat_stage& x = forms.flat[0];
    pass_args local;
    x.s.leaf(with_cursors(x.s.args, x.moving, cursors, local), source_of(x.s.src, at),
             destination_of(x.s.dst, at), step_of(*only, x.s.src), step_of(*only, x.s.dst), count);
    return;
  }
  flat_args args(forms);
  for (std::size_t t = first; t < first + count; ++t) {
    if (t != first) {
      if (only != nullptr) {
        at = moved(at, only->step, 1);
        cursors = moved(cursors, e[0].factor_step, 1);
      } else {
        at = transform_addresses(p, t, base);
        cursors = edge_transform_cursors(p, e, t, f);
      }
    }
    for (std::size_t k = 0; k < forms.flat.size(); ++k) {
      run_flat_stage(forms, k, args, at, cursors);
    }
  }
}

// Transforms [first, first + count) of a flat program of one batch loop in blocks of
// forms.row_block rows: each stage over the block's rows before the next, row j's line of the
// workspace block j block_row_bytes past its start; a block_call stage in one call over the rows
// (an edge pass with pass_args::rows, a leaf over `rows` lines), the others row by row. Each row
// meets the kernels and arguments of the flat form in the same order, so the bits are the same.
void run_edge_row_blocks(const program& p, const std::vector<edge_record>& e,
                         const edge_forms& forms, std::size_t first, std::size_t count,
                         const slot_addresses& base, factor_cursors f) noexcept {
  const loop& l = *std::get_if<loop>(p.records.data());
  const factor_steps& fs = e[0].factor_step;
  const std::size_t end = first + count;
  slot_addresses at = transform_addresses(p, first, base);
  factor_cursors cursors = edge_transform_cursors(p, e, first, f);
  std::size_t t = first;
  const std::size_t stages = forms.flat.size();
  flat_args args(forms);
  while (true) {
    const std::size_t rows = end - t < forms.row_block ? end - t : forms.row_block;
    for (std::size_t k = 0; k < stages; ++k) {
      const edge_flat_stage& x = forms.flat[k];
      if (x.block_call) {
        if (x.s.leaf != nullptr) {
          x.s.leaf(args.of(k, cursors), source_of(x.s.src, at), destination_of(x.s.dst, at),
                   x.src_dist, x.dst_dist, rows);
        } else {
          // The block's rows, in a copy: the last block may be shorter.
          pass_args block_args = *args.of(k, cursors);
          block_args.rows = static_cast<std::uint32_t>(rows);
          x.s.run(&block_args, source_of(x.s.src, at), destination_of(x.s.dst, at), 0, x.s.units);
        }
        continue;
      }
      // The stages up to the next block call, row by row: they may pass a row through the
      // scratch lines, which hold one line.
      std::size_t segment_end = k + 1;
      while (segment_end < stages && !forms.flat[segment_end].block_call) {
        ++segment_end;
      }
      slot_addresses row = at;
      factor_cursors row_cursors = cursors;
      for (std::size_t j = 0; j < rows; ++j) {
        if (j != 0) {
          row = moved(row, l.step, 1);
          row.block += forms.block_row_bytes;
          row_cursors = moved(row_cursors, fs, 1);
        }
        for (std::size_t m = k; m < segment_end; ++m) {
          run_flat_stage(forms, m, args, row, row_cursors);
        }
      }
      k = segment_end - 1;
    }
    t += rows;
    if (t == end) {
      return;
    }
    // Moved only to a block that follows, so no address beyond the batch is formed.
    at = moved(at, l.step, rows);
    cursors = moved(cursors, fs, rows);
  }
}

// One transform of an unrolled program (driver.cpp's run_calls), with its cursors f.
void run_edge_calls(const edge_forms& forms, const slot_addresses& at, factor_cursors f) noexcept {
  // Indexed by slot (program.hpp); the input is never a destination.
  const std::array<const std::byte*, 5> sources = {at.in, at.out, at.a, at.b, at.block};
  const std::array<std::byte*, 5> destinations = {at.out, at.out, at.a, at.b, at.block};
  for (const edge_call& x : forms.calls) {
    const call& c = x.c;
    const std::byte* src = sources[static_cast<std::size_t>(c.src)];
    std::byte* dst = destinations[static_cast<std::size_t>(c.dst)];
    // A slot that does not move has offset 0 and may be null (no block without a workspace).
    if (c.src_off != 0) {
      src += c.src_off;
    }
    if (c.dst_off != 0) {
      dst += c.dst_off;
    }
    pass_args local;
    const pass_args* a =
        with_cursors(forms.call_args[c.args], x.moving, moved_by(f, x.factor_off), local);
    if (c.leaf != nullptr) {
      c.leaf(a, src, dst, c.src_step, c.dst_step, c.units);
    } else {
      c.run(a, src, dst, 0, c.units);
    }
  }
}

// The bytes the slots in, out and block have moved (moved()'s step order).
using offsets = std::array<std::ptrdiff_t, 3>;

std::ptrdiff_t offset_of(slot s, const offsets& off) noexcept {
  switch (s) {
  case slot::in:
    return off[0];
  case slot::out:
    return off[1];
  case slot::block:
    return off[2];
  case slot::a:
  case slot::b:
    break;
  }
  return 0;
}

// driver.cpp's unroll_records with the cursors: a loop of one leaf is one call over its lines
// unless the leaf reads cursors the loop moves, when each line is a call of its own (a leaf
// reads one set of factors per call). False where the program has moving twiddles or more
// than calls_max calls.
bool unroll_edge_records(const program& p, const std::vector<edge_record>& e, std::size_t begin,
                         std::size_t end, const offsets& off, const factor_steps& factor_off,
                         const std::vector<std::size_t>& arg_of, std::vector<edge_call>& out) {
  for (std::size_t i = begin; i < end; ++i) {
    const record& r = p.records[i];
    if (const stage* st = std::get_if<stage>(&r)) {
      if (st->moving_twiddles || out.size() == calls_max) {
        return false;
      }
      if (st->leaf == nullptr && st->units == 0) {
        continue; // run_whole runs nothing
      }
      const bool reads = st->moving_factors != 0;
      out.push_back({
          .c =
              {
                  .run = reads ? e[i].run : st->run,
                  .leaf = reads ? e[i].leaf : st->leaf,
                  .units = st->leaf != nullptr ? 1 : st->units,
                  .src_off = offset_of(st->src, off),
                  .dst_off = offset_of(st->dst, off),
                  .args = static_cast<std::uint32_t>(arg_of[i]),
                  .src = st->src,
                  .dst = st->dst,
              },
          .factor_off = factor_off,
          .moving = st->moving_factors,
      });
      continue;
    }
    const loop& l = *std::get_if<loop>(&r);
    const offsets from = {off[0] + l.start[0], off[1] + l.start[1], off[2] + l.start[2]};
    if (l.body_end == i + 2) {
      const stage* only = std::get_if<stage>(&p.records[i + 1]);
      if (only != nullptr && only->leaf != nullptr && step_of(l, only->src) != 0 &&
          step_of(l, only->dst) != 0 && cursors_fixed(only->moving_factors, e[i].factor_step)) {
        if (out.size() == calls_max) {
          return false;
        }
        const bool reads = only->moving_factors != 0;
        out.push_back({
            .c =
                {
                    .leaf = reads ? e[i + 1].leaf : only->leaf,
                    .units = l.count,
                    .src_off = offset_of(only->src, from),
                    .dst_off = offset_of(only->dst, from),
                    .src_step = step_of(l, only->src),
                    .dst_step = step_of(l, only->dst),
                    .args = static_cast<std::uint32_t>(arg_of[i + 1]),
                    .src = only->src,
                    .dst = only->dst,
                },
            .factor_off = factor_off,
            .moving = only->moving_factors,
        });
        i = l.body_end - 1;
        continue;
      }
    }
    if (l.step[3] != 0) {
      return false;
    }
    for (std::size_t c = 0; c < l.count; ++c) {
      const auto k = static_cast<std::ptrdiff_t>(c);
      if (!unroll_edge_records(
              p, e, i + 1, l.body_end,
              {from[0] + l.step[0] * k, from[1] + l.step[1] * k, from[2] + l.step[2] * k},
              {factor_off[0] + e[i].factor_step[0] * k, factor_off[1] + e[i].factor_step[1] * k,
               factor_off[2] + e[i].factor_step[2] * k},
              arg_of, out)) {
        return false;
      }
    }
    i = l.body_end - 1;
  }
  return true;
}

} // namespace

edge_forms make_edge_forms(const program& p, const std::vector<edge_record>& e, bool unroll,
                           std::size_t row_block, std::ptrdiff_t block_row_bytes,
                           std::ptrdiff_t element_bytes) {
  edge_forms forms;
  if (p.lanes != 0) {
    return forms; // already in blocks of transforms (run_lane_blocks)
  }
  if (!p.flat) {
    if (!unroll) {
      return forms;
    }
    std::vector<std::size_t> arg_of(p.records.size(), 0);
    std::vector<pass_args> args;
    for (std::size_t i = p.batch_loops; i < p.records.size(); ++i) {
      if (const stage* st = std::get_if<stage>(&p.records[i])) {
        arg_of[i] = args.size();
        args.push_back(st->args);
      }
    }
    std::vector<edge_call> calls;
    if (unroll_edge_records(p, e, p.batch_loops, p.records.size(), {}, {}, arg_of, calls) &&
        !calls.empty()) {
      forms.calls = std::move(calls);
      forms.call_args = std::move(args);
    }
    return forms;
  }
  for (std::size_t i = p.batch_loops; i < p.records.size(); ++i) {
    const stage& st = *std::get_if<stage>(&p.records[i]);
    const bool reads = st.moving_factors != 0;
    forms.flat.push_back({
        .s =
            {
                .run = reads ? e[i].run : st.run,
                .leaf = reads ? e[i].leaf : st.leaf,
                .units = st.units,
                .src = st.src,
                .dst = st.dst,
                .args = st.args,
            },
        .moving = st.moving_factors,
    });
  }
  forms.leaf_rows = p.leaf_rows;
  if (row_block < 2 || p.batch_loops != 1 || block_row_bytes <= 0 || element_bytes <= 0) {
    return forms;
  }
  const loop& l = *std::get_if<loop>(p.records.data());
  if (l.step[2] != 0 || l.count < 2) {
    return forms;
  }
  forms.row_block = row_block;
  forms.block_row_bytes = block_row_bytes;
  // The bytes between rows' lines in a slot; the scratch lines hold one line, used row by row.
  const auto dist = [&](slot s, std::ptrdiff_t& d) {
    if (s == slot::a || s == slot::b) {
      return false;
    }
    d = s == slot::block ? block_row_bytes : step_of(l, s);
    return true;
  };
  for (std::size_t k = 0; k < forms.flat.size(); ++k) {
    edge_flat_stage& x = forms.flat[k];
    const stage& st = *std::get_if<stage>(&p.records[p.batch_loops + k]);
    std::ptrdiff_t src_dist = 0;
    std::ptrdiff_t dst_dist = 0;
    if (!dist(st.src, src_dist) || !dist(st.dst, dst_dist)) {
      continue;
    }
    if (st.leaf != nullptr) {
      if (cursors_fixed(st.moving_factors, e[0].factor_step)) {
        x.block_call = true;
        x.src_dist = src_dist;
        x.dst_dist = dst_dist;
      }
      continue;
    }
    // A separate edge pass (edge_pass_stage, or a pipeline's middle pass): its rows src_dist and
    // dst_dist elements apart, its factor the batch loop's step of the cursor it reads apart.
    const bool edge_pass = st.name.starts_with("edge in") || st.name.starts_with("edge out") ||
                           st.name.starts_with("edge middle");
    if (!edge_pass || st.args.rows != 0 || st.args.src_dist != 0 || st.args.dst_dist != 0 ||
        src_dist % element_bytes != 0 || dst_dist % element_bytes != 0) {
      continue;
    }
    x.block_call = true;
    x.s.args.src_dist = src_dist / element_bytes;
    x.s.args.dst_dist = dst_dist / element_bytes;
    x.s.args.dst_qstride = block_factor_step(st.moving_factors, e[0].factor_step);
  }
  return forms;
}

void factor_guard_pass(const pass_args* /*args*/, const void* /*src*/, void* /*dst*/,
                       std::size_t /*first*/, std::size_t /*last*/) {
  contract_failure(missing_array);
}

void factor_guard_leaf(const pass_args* /*args*/, const void* /*src*/, void* /*dst*/,
                       std::ptrdiff_t /*src_dist*/, std::ptrdiff_t /*dst_dist*/,
                       std::size_t /*count*/) {
  contract_failure(missing_array);
}

factor_cursors cursors_of(const exec_args& args, std::uint8_t needed) noexcept {
  const bool in = (needed & moving_in_factor) != 0;
  const bool out = (needed & moving_out_factor) != 0;
  const bool mid = (needed & moving_mid_factor) != 0;
  if ((in && args.in_factor == nullptr) || (out && args.out_factor == nullptr) ||
      (mid && args.mid_factor == nullptr)) {
    contract_failure(missing_array);
  }
  return {
      .in = in ? reinterpret_cast<std::uintptr_t>(args.in_factor) : 0,
      .out = out ? reinterpret_cast<std::uintptr_t>(args.out_factor) : 0,
      .mid = mid ? reinterpret_cast<std::uintptr_t>(args.mid_factor) : 0,
  };
}

factor_cursors edge_transform_cursors(const program& p, const std::vector<edge_record>& e,
                                      std::size_t t, factor_cursors f) noexcept {
  std::size_t rest = t;
  for (std::size_t i = p.batch_loops; i > 0 && rest != 0; --i) {
    const loop& l = *std::get_if<loop>(&p.records[i - 1]);
    f = moved(f, e[i - 1].factor_step, rest % l.count);
    rest /= l.count;
  }
  return f;
}

void run_edge_transforms(const program& p, const std::vector<edge_record>& e,
                         const std::vector<edge_record>& tail, const edge_forms& forms,
                         std::size_t first, std::size_t count, const slot_addresses& base,
                         factor_cursors f) noexcept {
  if (p.lanes != 0) {
    if (count != 0) {
      run_lane_blocks(p, e, tail, first, count, base, f);
    }
    return;
  }
  if (count == 0) {
    return; // forms no address past the batch (the forms below would)
  }
  if (forms.row_block != 0) {
    run_edge_row_blocks(p, e, forms, first, count, base, f);
    return;
  }
  if (!forms.flat.empty()) {
    run_edge_flat(p, e, forms, first, count, base, f);
    return;
  }
  if (!forms.calls.empty()) {
    for (std::size_t t = first; t < first + count; ++t) {
      run_edge_calls(forms, transform_addresses(p, t, base), edge_transform_cursors(p, e, t, f));
    }
    return;
  }
  for (std::size_t t = first; t < first + count; ++t) {
    run_records(p, e, p.batch_loops, p.records.size(), transform_addresses(p, t, base),
                edge_transform_cursors(p, e, t, f));
  }
}

void run_edge_stage_chunks(const program& p, const std::vector<edge_record>& e, std::size_t index,
                           const slot_addresses& at, factor_cursors f, std::size_t first,
                           std::size_t last) noexcept {
  run_stage(*std::get_if<stage>(&p.records[index]), e[index], at, f, first, last);
}

void run_edge_lines(const program& p, const std::vector<edge_record>& e, std::size_t begin,
                    std::size_t nest, std::size_t body_end, std::size_t first, std::size_t last,
                    const slot_addresses& at, factor_cursors f) noexcept {
  slot_addresses from = at;
  for (std::size_t i = begin; i < begin + nest; ++i) {
    from = moved(from, std::get_if<loop>(&p.records[i])->start, 1);
  }
  for (std::size_t line = first; line < last; ++line) {
    const auto [here, cursors] = moved_in_nest(p, e, begin, nest, line, from, f);
    run_records(p, e, begin + nest, body_end, here, cursors);
  }
}

} // namespace fftn::inline v0::detail
