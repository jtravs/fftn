# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""What is generated: the radices, variants and directions, and the output files.

This is the only list of radices in FFTN: the C++ side reads src/gen/codelets.hpp, which
is written from it (DESIGN.md 7.2, step 7). Adding a radix: gen/README.md.
"""

from __future__ import annotations

from .codelet import SIGNS, VARIANTS, Codelet, build
from .emit_cpp import codelet_inl, metadata_hpp

# Every size 2 to 16, then the powers of two that whole-transform leaves (experiment X8)
# and few-big-pass chains (task 1.6, experiment X3) use.
RADICES = (*range(2, 17), 32, 64)


def all_codelets() -> list[Codelet]:
    return [build(r, v, s) for r in RADICES for v in VARIANTS for s in SIGNS]


def generate(codelets: list[Codelet] | None = None) -> dict[str, str]:
    """File name (under src/gen/) to contents."""
    if codelets is None:
        codelets = all_codelets()
    return {
        "complex_codelets.inl": codelet_inl(codelets),
        "codelets.hpp": metadata_hpp(codelets),
    }
