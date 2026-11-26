# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""size_report's reading of the size tools' listings of a static library."""

from __future__ import annotations

import unittest

from size_report import member_name, parse, table

# Apple's size (Xcode 26) over libfftn.a: six named columns, the name unlabelled.
DARWIN = """\
__TEXT\t__DATA\t__OBJC\tothers\tdec\thex
44\t0\t0\t32\t76\t4c\t/b/check-release/src/libfftn.a(version.cpp.o)
1897\t168\t0\t256\t2321\t911\t/b/check-release/src/libfftn.a(target.cpp.o)
554888\t21016\t0\t80128\t656032\ta02a0\t/b/check-release/src/libfftn.a(scalar.cpp.o)
700000\t0\t0\t1000\t701000\tab248\t/b/check-release/src/libfftn.a(rows1_scalar.cpp.o)
623708\t0\t0\t1000\t624708\t98844\t/b/check-release/src/libfftn.a(neon.cpp.o)
300000\t0\t0\t1000\t301000\t497c8\t/b/check-release/src/libfftn.a(rows8_neon.cpp.o)
90000\t0\t0\t100\t90100\t15ff4\t/b/check-release/src/libfftn.a(pairs_neon.cpp.o)
80000\t0\t0\t100\t80100\t138e4\t/b/check-release/src/libfftn.a(batch_large_neon_fcma.cpp.o)
0\t0\t0\t0\t0\t0\t/b/check-release/src/libfftn.a(avx512.cpp.o)
16\t0\t0\t0\t16\t10\t/b/check-release/src/libfftn.a(rows2_avx.cpp.o)
"""

# GNU size (binutils 2.44) over libfftn.a: the name is labelled and carries the archive.
GNU = """\
   text\t   data\t    bss\t    dec\t    hex\tfilename
     52\t      0\t      0\t     52\t     34\tversion.cpp.o (ex build/src/libfftn.a)
 912345\t   1200\t      0\t 913545\t  df089\tavx2.cpp.o (ex build/src/libfftn.a)
 400000\t      0\t      0\t 400000\t  61a80\trows3_avx2.cpp.o (ex build/src/libfftn.a)
  20000\t      0\t      0\t  20000\t   4e20\tnarrow_avx2.cpp.o (ex build/src/libfftn.a)
 500000\t      0\t      0\t 500000\t  7a120\tavx.cpp.o (ex build/src/libfftn.a)
   9000\t     16\t      8\t   9024\t   2340\tplan_core.cpp.o (ex build/src/libfftn.a)
      0\t      0\t      0\t      0\t      0\tneon_fcma.cpp.o (ex build/src/libfftn.a)
"""

# llvm-size over fftn.lib (COFF), with Windows paths and .obj objects.
COFF = """\
   text    data     bss     dec     hex filename
  70000       0       0   70000   11170 avx512.cpp.obj (ex build\\src\\fftn.lib)
  30000       0       0   30000    7530 batch_avx512.cpp.obj (ex build\\src\\fftn.lib)
   5000       0       0    5000    1388 rules.cpp.obj (ex build\\src\\fftn.lib)
"""


class Members(unittest.TestCase):
    def test_every_archive_form_gives_the_object_name(self) -> None:
        self.assertEqual(member_name("/b/src/libfftn.a(rows1_avx2.cpp.o)"), "rows1_avx2.cpp.o")
        self.assertEqual(member_name("avx2.cpp.o (ex build/src/libfftn.a)"), "avx2.cpp.o")
        self.assertEqual(member_name("isa\\avx.cpp.obj (ex b\\fftn.lib)"), "avx.cpp.obj")
        self.assertEqual(member_name("build/src/CMakeFiles/fftn.dir/isa/neon.cpp.o"), "neon.cpp.o")


class Grouping(unittest.TestCase):
    def test_darwin_units_go_to_their_target_and_kind(self) -> None:
        r = parse(DARWIN)
        self.assertEqual(r.column, "__TEXT")
        self.assertEqual(r.targets["scalar"], {"main": 554888, "rows": 700000, "leaves": 0})
        self.assertEqual(r.targets["neon"], {"main": 623708, "rows": 300000, "leaves": 90000})
        self.assertEqual(r.targets["neon_fcma"]["leaves"], 80000)
        self.assertEqual(r.common, 44 + 1897)  # version and target.cpp: not a kernel unit
        self.assertEqual(r.total(), 44 + 1897 + 554888 + 700000 + 623708 + 300000 + 90000 + 80016)

    def test_gnu_size_names_with_the_archive(self) -> None:
        r = parse(GNU)
        self.assertEqual(r.column, "text")
        # avx and avx2 are told apart; the narrow leaves belong to avx2's region.
        self.assertEqual(r.targets["avx2"], {"main": 912345, "rows": 400000, "leaves": 20000})
        self.assertEqual(r.targets["avx"]["main"], 500000)
        self.assertEqual(r.common, 52 + 9000)

    def test_coff_objects(self) -> None:
        r = parse(COFF)
        self.assertEqual(r.targets["avx512"], {"main": 70000, "rows": 0, "leaves": 30000})
        self.assertEqual(r.common, 5000)

    def test_empty_units_of_the_other_architecture_are_one_row(self) -> None:
        rows = table(parse(DARWIN))
        self.assertFalse(any("`avx512`" in row or "`avx`" in row for row in rows))
        self.assertIn("| other architecture (empty units) | | | | 16 |", rows[-2])
        self.assertTrue(rows[-1].startswith("| **library** | | | | **2,350,553** |"))

    def test_output_without_a_header_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            parse("12 0 0 12 c a.o\n")
        with self.assertRaises(ValueError):
            parse("")


if __name__ == "__main__":
    unittest.main()
