# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0

import math
import os
import tempfile
import unittest
from pathlib import Path

from fbenchlib import arms, catalogue


class Standard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = catalogue.load("standard")

    def test_ids_unique_and_tiers(self):
        ids = [c.id for c in self.cat.cases]
        self.assertEqual(len(ids), len(set(ids)))
        quick = {c.id for c in self.cat.select("quick")}
        tracking = {c.id for c in self.cat.select("tracking")}
        self.assertTrue(quick <= tracking)
        self.assertTrue(30 <= len(tracking) <= 70)
        self.assertEqual(len(self.cat.select("full")), len(self.cat.cases))

    def test_covers_design_16_1(self):
        cases = self.cat.cases
        pow2 = {c.dims[0] for c in cases if c.kind == "c2c" and len(c.dims) == 1}
        self.assertTrue({2**k for k in range(1, 25)} <= pow2)
        self.assertTrue(any(c.batch >= 1000 for c in cases))
        self.assertEqual({c.kind for c in cases}, {"c2c", "r2c", "c2r"})
        self.assertEqual({c.place for c in cases}, {"in", "out"})
        self.assertTrue(any(len(c.dims) == 2 for c in cases))
        self.assertIn(999983, pow2 | {c.dims[0] for c in cases if len(c.dims) == 1})

    def test_smoke_covers_every_kind_placement_and_rank(self):
        smoke = self.cat.select("smoke")
        combos = {(c.kind, c.place, len(c.dims)) for c in smoke}
        for kind in ("c2c", "r2c", "c2r"):
            for place in ("in", "out"):
                for rank in (1, 2):
                    self.assertIn((kind, place, rank), combos)
        self.assertTrue(any(c.sign > 0 and c.kind == "c2c" for c in smoke))

    def test_case_line_and_flops(self):
        c = catalogue.Case("r2c", (64, 48), 4, "in", -1)
        self.assertEqual(c.id, "r2c-64x48-b4-ip")
        self.assertEqual(
            c.line(), "id=r2c-64x48-b4-ip kind=r2c dims=64x48 batch=4 place=in sign=-1"
        )
        n = 64 * 48
        self.assertAlmostEqual(c.flops, 4 * 2.5 * n * math.log2(n))
        d = catalogue.Case("c2c", (8,), 1, "out", 1)
        self.assertEqual(d.id, "c2c-8-b1-op-bwd")
        self.assertAlmostEqual(d.flops, 5 * 8 * 3)

    def test_interleaved_case(self):
        c = catalogue.Case("c2c", (32,), 8, "out", -1, layout="interleaved", outer=16)
        self.assertEqual(c.id, "c2c-32-b8-op-il-o16")
        self.assertEqual(
            c.line(),
            "id=c2c-32-b8-op-il-o16 kind=c2c dims=32 batch=8 place=out sign=-1 "
            "layout=interleaved outer=16",
        )
        self.assertAlmostEqual(c.flops, 8 * 16 * 5 * 32 * 5)
        cases = catalogue.load("axes").cases
        self.assertTrue(all(c.layout == "interleaved" for c in cases))
        self.assertTrue(any(c.outer > 1 for c in cases))

    def test_single_precision_case(self):
        c = catalogue.Case("c2c", (8,), 1, "out", 1, precision="f32")
        self.assertEqual(c.id, "c2c-8-b1-op-bwd-f32")
        self.assertEqual(
            c.line(), "id=c2c-8-b1-op-bwd-f32 kind=c2c dims=8 batch=1 place=out sign=1 prec=f32"
        )
        smoke = self.cat.select("smoke")
        self.assertEqual({c.kind for c in smoke if c.precision == "f32"}, {"c2c", "r2c", "c2r"})


class Threads(unittest.TestCase):
    """The threads catalogue: a group per scheduling rule (DESIGN.md 13.4)."""

    @classmethod
    def setUpClass(cls):
        cls.cat = catalogue.load("threads")

    def test_tiers(self):
        quick = {c.id for c in self.cat.select("quick")}
        tracking = {c.id for c in self.cat.select("tracking")}
        self.assertTrue(tracking <= quick)
        self.assertTrue(30 <= len(tracking) <= 70)
        smoke = self.cat.select("smoke")
        self.assertEqual({c.kind for c in smoke}, {"c2c", "r2c", "c2r"})
        self.assertTrue(any(c.batch > 1 for c in smoke))

    def test_rules_have_cases(self):
        quick = self.cat.select("quick")
        groups = {c.group for c in quick}
        for g in ("rows-long", "rows-few", "rows-remainder", "rows-many", "small", "planes"):
            self.assertIn(g, groups)
        # A case listed by several groups keeps its first group's name (smoke's, here).
        long_rows = {
            c.dims[0] for c in quick if c.kind == "c2c" and len(c.dims) == 1 and c.batch == 1
        }
        self.assertTrue({2**k for k in range(10, 21)} <= long_rows)
        # Leftover rows: one more than 8, 16, 32 and 64 lanes.
        leftover = {c.batch for c in quick if c.group == "rows-remainder"}
        self.assertEqual(leftover, {9, 17, 33, 65})

    def test_threaded_arms_parse(self):
        a = arms.parse("Y=fftn_omp+THREADS=16+SPLIT_MIN_POINTS=never+SYNC_NS=0")
        self.assertEqual(a.library, "fftn_omp")
        self.assertEqual(
            a.backend_options(), {"THREADS": "16", "SPLIT_MIN_POINTS": "never", "SYNC_NS": "0"}
        )


class Format(unittest.TestCase):
    def load(self, text: str) -> catalogue.Catalogue:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text(text)
            return catalogue.load(str(path))

    def test_product_merge_and_errors(self):
        cat = self.load(
            '[[group]]\nkind = ["c2c", "r2c"]\ndims = [8, "4x6"]\nbatch = [1, 2]\n'
            '[[group]]\ndims = [8]\ntiers = ["quick"]\n'
        )
        self.assertEqual(len(cat.cases), 8)
        self.assertEqual([c.id for c in cat.select("quick")], ["c2c-8-b1-op"])
        with self.assertRaises(ValueError):
            self.load("[[group]]\ndims = [8]\nsize = 3\n")
        with self.assertRaises(ValueError):
            self.load('[[group]]\ndims = ["8by8"]\n')
        with self.assertRaises(ValueError):
            self.load('[[group]]\nkind = "r2c"\ndims = [8]\ndirection = "bwd"\n')
        with self.assertRaises(ValueError):
            self.load("[[group]]\ndims = [8]\n[[group]]\ndims = [8]\nweight = 2\n")
        with self.assertRaises(ValueError):
            self.load('[[group]]\ndims = [8]\ntiers = ["nightly"]\n')
        with self.assertRaises(ValueError):
            self.load('[[group]]\ndims = [8]\nprecision = "f16"\n')
        both = self.load('[[group]]\ndims = [8]\nprecision = ["f64", "f32"]\n')
        self.assertEqual([c.id for c in both.cases], ["c2c-8-b1-op", "c2c-8-b1-op-f32"])


class ArmGrammar(unittest.TestCase):
    def test_parse(self):
        a = arms.parse("B=fftw:build/x+PLANNER=PATIENT+DELAY=0.05")
        self.assertEqual((a.label, a.library, a.build), ("B", "fftw", "build/x"))
        self.assertEqual(a.backend_options(), {"PLANNER": "PATIENT"})
        self.assertEqual(a.harness_args(), ["--delay", "0.05"])
        self.assertEqual(a.spec, "B=fftw:build/x+PLANNER=PATIENT+DELAY=0.05")
        b = arms.parse("fftn:abc123")
        self.assertEqual((b.label, b.library, b.build, b.options), ("fftn", "fftn", "abc123", ()))
        c = arms.parse("A=fftw+WISDOM=none")
        self.assertEqual(c.backend_options(), {})
        self.assertEqual(c.option("WISDOM"), "none")
        ctl = a.as_control()
        self.assertTrue(ctl.control)
        self.assertEqual(ctl.label, "B~")

    def test_rejects(self):
        for bad in ("", "A=", "A=FFTW", "A=fftw+delay", "A=fftw+DELAY=x", "A=fftw+X=a b"):
            with self.assertRaises(ValueError, msg=bad):
                arms.parse(bad)

    def test_test_only_options_are_not_library_switches(self):
        # Experimental library switches are FFTN_X_*, registered in dev/KNOBS.toml; the
        # harness's test options must never look like one.
        for name in arms.HARNESS_OPTIONS | arms.DRIVER_OPTIONS:
            self.assertFalse(name.startswith("FFTN_X_"))

    def test_identical_configuration(self):
        desc = {"library": "fftn", "harness": {"commit": "abc", "flags": "-O3"}}
        a = arms.Binary(Path("/x/bench/fbench-fftn"), describe=dict(desc))
        b = arms.Binary(Path("/y/bench/fbench-fftn"), describe=dict(desc))
        arm = arms.parse("A=fftn")
        # Two build directories of one commit and configuration are one arm.
        self.assertEqual(arms.configuration(a, arm), arms.configuration(b, arm))
        self.assertNotEqual(
            arms.configuration(a, arm), arms.configuration(a, arms.parse("A=fftn+DELAY=0.1"))
        )
        other = arms.Binary(b.path, describe={**desc, "harness": {"commit": "def"}})
        self.assertNotEqual(arms.configuration(a, arm), arms.configuration(other, arm))

    def test_fftn_isa_arms(self):
        # One build, two instruction sets (the FFTN backend's ISA option): the option reaches
        # the backend, and the two arms are different configurations.
        a = arms.parse("A=fftn+ISA=avx2")
        b = arms.parse("B=fftn+ISA=avx512")
        self.assertEqual(b.backend_options(), {"ISA": "avx512"})
        binary = arms.Binary(Path("/x/bench/fbench-fftn"), describe={"library": "fftn"})
        self.assertNotEqual(arms.configuration(binary, a), arms.configuration(binary, b))

    def test_refusals(self):
        ok = {"loaded_from": "/opt/fftw/lib/libfftw3.so", "expected_root": "/opt/fftw"}
        self.assertIsNone(arms.library_problem(ok))
        wrong = {"loaded_from": "/usr/lib/libfftw3.so", "expected_root": "/opt/fftw"}
        self.assertIn("built against", arms.library_problem(wrong))
        prefix = {"loaded_from": "/opt/fftw-old/lib/libfftw3.so", "expected_root": "/opt/fftw"}
        self.assertIsNotNone(arms.library_problem(prefix))
        self.assertIsNone(arms.library_problem({}))
        # On Linux a symbol's address seen from the executable names the executable.
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "fbench-fftw"
            exe.touch()
            itself = {"loaded_from": str(exe), "expected_root": tmp}
            self.assertIn("executable itself", arms.library_problem(itself, exe))
        release = {"harness": {"build_type": "Release", "sanitize": ""}}
        self.assertIsNone(arms.timing_problem(release))
        asan = {"harness": {"build_type": "RelWithDebInfo", "sanitize": "address"}}
        self.assertIn("sanitizers", arms.timing_problem(asan))
        debug = {"harness": {"build_type": "Debug", "sanitize": ""}}
        self.assertIn("Debug", arms.timing_problem(debug))

    def test_library_through_a_symlinked_root(self):
        # An installation whose lib directory links into a build tree is the installation;
        # another copy of the library is not, and neither is a ".." walked through the link.
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / "build" / "lib"
            build.mkdir(parents=True)
            (build / "libx.so").touch()
            (build.parent / "libx.so").touch()
            root = Path(tmp) / "prefix"
            root.mkdir()
            (root / "lib").symlink_to(build)
            other = Path(tmp) / "elsewhere" / "lib"
            other.mkdir(parents=True)
            (other / "libx.so").touch()

            def problem(loaded: Path) -> str | None:
                info = {"loaded_from": str(loaded), "expected_root": str(root)}
                return arms.library_problem(info)

            self.assertIsNone(problem(root / "lib" / "libx.so"))
            self.assertIn("built against", problem(other / "libx.so"))
            through_link = problem(root / "lib" / ".." / "libx.so")
            if os.name == "nt":
                # Windows removes ".." before it follows links: that path is prefix\libx.so,
                # inside the installation, and what the loader would have opened.
                self.assertIsNone(through_link)
            else:
                self.assertIn("built against", through_link)


if __name__ == "__main__":
    unittest.main()
