# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from fbenchlib import store


class Scrub(unittest.TestCase):
    def test_prefixes_in_nested_records(self):
        prefixes = [("/home/me/fftn", "<base>"), ("/home/me", "~"), ("/opt/fftw", "<fftw-root>")]
        prefixes.sort(key=lambda p: -len(p[0]))
        record = {
            "binary": "/home/me/fftn/build/dev/bench/fbench-fftw",
            "argv": ["run", "A=fftn:/home/me/fftn/build/x", "--out-dir", "/home/me/elsewhere"],
            "describe": {
                "loaded_from": "/opt/fftw/lib/libfftw3.so.3",
                "expected_root": "/opt/fftw",
            },
            "other": "/home/mex/kept",
            "n": 3,
        }
        out = store.scrub(record, prefixes)
        self.assertEqual(out["binary"], "<base>/build/dev/bench/fbench-fftw")
        self.assertEqual(out["argv"][1], "A=fftn:<base>/build/x")
        self.assertEqual(out["argv"][3], "~/elsewhere")
        self.assertEqual(out["describe"]["loaded_from"], "<fftw-root>/lib/libfftw3.so.3")
        self.assertEqual(out["describe"]["expected_root"], "<fftw-root>")
        self.assertEqual(out["other"], "/home/mex/kept")  # a prefix only at a path boundary
        self.assertEqual(out["n"], 3)

    def test_store_holds_no_absolute_home_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bench-out"
            st = store.RunStore(out_dir, "r1")
            st.add_prefix("/opt/fftw", "<fftw-root>")
            st.append(
                {
                    "type": "run",
                    "binary": str(out_dir.parent / "build" / "fbench-fftw"),
                    "home": str(Path.home() / "notes"),
                    "root": "/opt/fftw/lib",
                }
            )
            rec = json.loads(st.path.read_text())
            # The separator after the token is the platform's (a backslash on Windows).
            self.assertEqual(rec["binary"], str(Path("<base>", "build", "fbench-fftw")))
            self.assertEqual(rec["home"], str(Path("~", "notes")))
            self.assertEqual(rec["root"], "<fftw-root>/lib")
            self.assertNotIn(str(Path.home()), st.path.read_text())


if __name__ == "__main__":
    unittest.main()
