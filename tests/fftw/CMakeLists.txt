# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# FFTW as an independent implementation (DESIGN.md 15.1, oracle 3): every subject but FFTW
# compared with FFTW, output by output. The cases skip when the build has no FFTW
# (FFTN_TEST_WITH_FFTW=OFF).

fftn_add_test(fftn_fftw_tests
  SOURCES fftw_suite_test.cpp bluestein_test.cpp
  LIBRARIES fftn_test_support)
