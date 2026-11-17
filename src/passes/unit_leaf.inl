// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Leaves for lines of stride 1 on both sides (experiment X28b): stockham.inl's leaf_lines
// with the leg distance a constant. Kernel text in the manner of stockham.inl, included
// after it. No include guard.
//
// leaf_lines works for any stride, so it computes the R leg offsets of each side from the
// strides at run time and keeps them in registers; at 8 to 16 points that set-up was most
// of what separated a single row from FFTW's (X28: 6-9 cycles per call on the Xeon 6240
// and the EPYC 7763). Here every offset is a constant in the addressing. Each lane does the
// same arithmetic as in leaf_lines, so the bits are the same.

namespace {

// As leaf_lines, for src_stride = dst_stride = 1. Reads scale.
template <class T, std::size_t R, int Sign, edge_op InOp, edge_op OutOp>
void unit_leaf(const pass_args* a, const void* src, void* dst, std::ptrdiff_t src_dist,
               std::ptrdiff_t dst_dist, std::size_t count) {
  static_assert(InOp != edge_op::factor && OutOp != edge_op::factor,
                "the factor edges keep leaf_lines");
  one_in<T, InOp> in{
      .p = nullptr,
      .leg = 2,
      .op = make_edge<T, InOp>(a, nullptr, 0, 1),
  };
  one_out<T, OutOp> out{
      .p = nullptr,
      .leg = 2,
      .op = make_edge<T, OutOp>(a, nullptr, 0, 1),
  };
  for (std::size_t u = 0; u < count; ++u) {
    in.p = line_at<T>(src, u, src_dist);
    out.p = line_at<T>(dst, u, dst_dist);
    cdft<R>::template plain<T, Sign>(in, out);
  }
}

} // namespace
