# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Writes scipy_r2r_cases.txt: scipy.fft's real-to-real results on small cases, recorded for
tests/api/easy_r2r_test.cpp (step 6.7, the convenience layer's dct, dst and the rest).

Run offline, by hand, in a scratch environment with numpy and scipy; neither is a dependency of
FFTN or of its tools, and the build never runs this script:

    uv venv /tmp/scipy-env && VIRTUAL_ENV=/tmp/scipy-env uv pip install numpy scipy
    /tmp/scipy-env/bin/python tests/api/data/scipy_r2r_cases.py > tests/api/data/scipy_r2r_cases.txt

The DHT, which scipy.fft lacks, is recorded from numpy.fft.fft as Re X - Im X.

Format: comment lines starting with #, then one record per case, each on three lines:

    case FUNCTION TYPE NORM RANK EXTENTS... NAXES AXES...
    in VALUES...
    out VALUES...

FUNCTION is dct, idct, dst, idst, dctn, idctn, dstn, idstn or dht; TYPE is scipy's type (0 for
the DHT); NORM is backward, ortho or forward; the arrays are C-ordered; NAXES is 0 for the
function's default axes (all) and otherwise followed by the axes as passed to scipy (negative
ones count from the end). Values are Python's shortest round-trip decimal form of doubles. The
inputs are multiples of 1/256 in [-1, 1), exact in float and in double.
"""

import sys

import numpy as np
import scipy
import scipy.fft as sf

NORMS = ("backward", "ortho", "forward")
LENGTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17)
# (extents, axes): the default (all), one axis, reordered and negative axes, and an array of
# rank 4 with two transformed axes.
SHAPES = (
    ((3, 4), None),
    ((2, 3, 5), None),
    ((4, 5), (0,)),
    ((2, 3, 4), (-1, 0)),
    ((2, 3, 2, 3), (1, 3)),
)
ONE_D = {"dct": sf.dct, "idct": sf.idct, "dst": sf.dst, "idst": sf.idst}
N_D = {"dctn": sf.dctn, "idctn": sf.idctn, "dstn": sf.dstn, "idstn": sf.idstn}

rng = np.random.default_rng(20261126)


def values(shape):
    return rng.integers(-256, 256, size=shape).astype(np.float64) / 256.0


def line(tag, a):
    return tag + " " + " ".join(repr(float(v)) for v in np.ravel(a))


def record(out, function, type_, norm, x, axes, y):
    head = ["case", function, str(type_), norm, str(x.ndim)]
    head += [str(e) for e in x.shape]
    head += ["0"] if axes is None else [str(len(axes))] + [str(a) for a in axes]
    out.append(" ".join(head))
    out.append(line("in", x))
    out.append(line("out", y))


def main():
    out = [
        "# SPDX-FileCopyrightText: 2026 John C. Travers and contributors",
        "# SPDX-License-Identifier: Apache-2.0",
        # The results are scipy's, so its version is recorded; numpy only draws the inputs, and a
        # numpy whose random stream differs gives other inputs, equally valid.
        f"# scipy {scipy.__version__}: written by scipy_r2r_cases.py; do not edit",
    ]
    for n in LENGTHS:
        x = values((n,))
        for name, f in ONE_D.items():
            for type_ in (1, 2, 3, 4):
                if type_ == 1 and name.endswith("ct") and n == 1:
                    continue  # scipy's DCT-I needs two points, as FFTN's
                for norm in NORMS:
                    record(out, name, type_, norm, x, None, f(x, type=type_, norm=norm))
        spectrum = np.fft.fft(x)
        hartley = spectrum.real - spectrum.imag
        for norm in NORMS:
            scale = {"backward": 1.0, "ortho": 1.0 / np.sqrt(n), "forward": 1.0 / n}[norm]
            record(out, "dht", 0, norm, x, None, hartley * scale)
    for extents, axes in SHAPES:
        x = values(extents)
        for name, f in N_D.items():
            for type_ in (1, 2, 3, 4):
                for norm in NORMS:
                    record(out, name, type_, norm, x, axes, f(x, type=type_, axes=axes, norm=norm))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
