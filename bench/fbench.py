#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""fbench: FFTN's benchmark rig (DESIGN.md 16; usage in bench/README.md).

uv run --frozen python bench/fbench.py run A=fftw B=fftw+PLANNER=PATIENT --tier quick
uv run --frozen python bench/fbench.py smoke --build build/check-release --strict
uv run --frozen python bench/fbench.py list --tier tracking
uv run --frozen python bench/fbench.py show latest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fbenchlib.cli import main

if __name__ == "__main__":
    sys.exit(main())
