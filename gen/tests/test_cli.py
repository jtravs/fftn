# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The command line: writing, checking (the gen gate stage) and C++ literals."""

from __future__ import annotations

import contextlib
import io
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mpmath
from fftn_gen import __main__ as cli
from fftn_gen.catalogue import generate
from fftn_gen.codelet import build
from fftn_gen.emit_cpp import literal_double, literal_float

from .test_codelets import codelets

_FILES = {"complex_codelets.inl": "a\n", "codelets.hpp": "b\n"}


class CommandLine(unittest.TestCase):
    def run_cli(self, *args: str) -> int:
        with (
            mock.patch.object(cli, "generate", return_value=dict(_FILES)),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return cli.main(list(args))

    def test_write_then_check_passes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_cli("--out", d), 0)
            self.assertEqual(self.run_cli("--check", d), 0)

    def test_check_fails_on_a_changed_missing_or_extra_file(self) -> None:
        for change in ("edit", "remove", "extra"):
            with self.subTest(change), tempfile.TemporaryDirectory() as d:
                self.run_cli("--out", d)
                path = Path(d)
                match change:
                    case "edit":
                        (path / "codelets.hpp").write_text("c\n")
                    case "remove":
                        (path / "codelets.hpp").unlink()
                    case "extra":
                        (path / "stale.inl").write_text("x\n")
                self.assertEqual(self.run_cli("--check", d), 1)

    def test_table_without_a_reference_entry_fails_with_a_message(self) -> None:
        err = io.StringIO()
        with (
            mock.patch.object(cli, "all_codelets", return_value=[build(18, "plain", -1)]),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(err),
        ):
            self.assertEqual(cli.main(["--table"]), 1)
        self.assertIn("radix 18 has no entry", err.getvalue())

    def test_the_committed_tree_matches_the_generator(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for name, text in generate(list(codelets())).items():
            self.assertEqual((root / "src" / "gen" / name).read_text(encoding="utf-8"), text, name)


class Literals(unittest.TestCase):
    def test_each_precision_gets_its_own_correct_rounding(self) -> None:
        c = mpmath.sqrt(mpmath.mpf(1) / 2)
        self.assertEqual(float(literal_double(c)), float(c))
        f = literal_float(c)
        self.assertTrue(f.endswith("F"))
        with mpmath.workprec(24):
            nearest = float(+c)
        self.assertEqual(struct.unpack("f", struct.pack("f", float(f[:-1])))[0], nearest)

    def test_every_codelet_function_checks_its_sign(self) -> None:
        text = generate(list(codelets()))["complex_codelets.inl"]
        functions = text.count("static FFTN_FORCE_INLINE void")
        self.assertEqual(functions, 3 * len({c.radix for c in codelets()}))
        self.assertEqual(text.count("static_assert(Sign == -1 || Sign == 1"), functions)

    def test_float_literal_of_an_integer_is_valid_cpp(self) -> None:
        self.assertEqual(literal_float(mpmath.mpf(2)), "2.0F")

    def test_output_does_not_depend_on_string_hashing(self) -> None:
        # Two processes with different hash seeds (set and dict order of str keys would
        # differ) must write byte-identical files. Run side by side to save time.
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as d:
            runs = []
            for seed in ("0", "4242"):
                out = Path(d) / seed
                env = {**os.environ, "PYTHONHASHSEED": seed}
                runs.append(
                    subprocess.Popen(
                        [sys.executable, "-m", "gen.fftn_gen", "--out", str(out)],
                        cwd=root,
                        env=env,
                        stdout=subprocess.DEVNULL,
                    )
                )
            self.assertEqual([r.wait() for r in runs], [0, 0])
            for name in generate(list(codelets())):
                a = (Path(d) / "0" / name).read_bytes()
                self.assertEqual(a, (Path(d) / "4242" / name).read_bytes(), name)

    def test_codelets_are_deterministic(self) -> None:
        from fftn_gen.catalogue import all_codelets

        again = {(c.radix, c.variant, c.sign): c for c in all_codelets()}
        for c in codelets():
            other = again[(c.radix, c.variant, c.sign)]
            self.assertEqual(
                [(o.kind, o.args, o.index) for o in c.ops],
                [(o.kind, o.args, o.index) for o in other.ops],
            )


if __name__ == "__main__":
    unittest.main()
