// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The direct DFT in quadruple precision (DESIGN.md 15.1, oracle 2).
//
// Every sum is evaluated in Boost.Multiprecision's 113-bit binary float. Twiddles are
// exp(sign * 2 pi i * m / n) with the index product k * j reduced modulo n in integers
// before any multiplication, and the angle reduced to the first octant, so their error is a
// few units of 2^-113. Inputs are taken as doubles (a float converts exactly) and outputs
// are exact_real values (hi + lo), to be rounded once to the precision under test with
// round_to<T>() or compared directly.
//
// Conventions are FFTW's (DESIGN.md 5.1): sign -1 is forward, nothing is normalised, r2c
// gives bins 0..n/2 and c2r reads n/2 + 1 values, ignoring the imaginary parts of bin 0 and,
// for even n, of bin n/2.
//
// Cost: about 0.14 us per complex multiply-add, so a full 1024-point transform takes about
// 0.15 s (measured on an Apple M5 Max). For large n, evaluate only some outputs with the
// overloads that take a list of bins or indices.
//
// Results are remembered for the life of the process (in bounded stores), by the input's
// bits and the outputs asked for: a repeated call returns at once, bit for bit what a new
// evaluation would give. All n outputs of a complex transform in one direction give those
// of the other direction too.

#pragma once

#include "support/exact.hpp"

#include <complex>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace fftn::test::reference {

// exp(sign * 2 pi i * m / n) for any m (reduced modulo n first); sign is -1 or +1.
exact_complex unit_root(int sign, std::size_t m, std::size_t n);

// All n roots exp(sign * 2 pi i * m / n), m = 0..n-1. Tables are cached per (sign, n) for
// the life of the process (thread-safe), since the analytic checks look them up often.
const std::vector<exact_complex>& unit_roots(int sign, std::size_t n);

// The DFT of x (length n) with exponent sign: all n outputs, or the listed bins.
std::vector<exact_complex> c2c(int sign, std::span<const std::complex<double>> x);
std::vector<exact_complex> c2c(int sign, std::span<const std::complex<double>> x,
                               std::span<const std::size_t> bins);

// The forward DFT of real x: bins 0..n/2, or the listed bins (each at most n/2).
std::vector<exact_complex> r2c(std::span<const double> x);
std::vector<exact_complex> r2c(std::span<const double> x, std::span<const std::size_t> bins);

// The backward real DFT of length n from the n/2 + 1 values X: all n outputs, or the listed
// indices.
std::vector<exact_real> c2r(std::size_t n, std::span<const std::complex<double>> X);
std::vector<exact_real> c2r(std::size_t n, std::span<const std::complex<double>> X,
                            std::span<const std::size_t> indices);

// The 2-D DFT of one ny x nx plane stored row-major (x along rows): all outputs, row-major,
// or the listed (ky, kx) points.
std::vector<exact_complex> c2c_plane(int sign, std::size_t ny, std::size_t nx,
                                     std::span<const std::complex<double>> x);
std::vector<exact_complex> c2c_plane(int sign, std::size_t ny, std::size_t nx,
                                     std::span<const std::complex<double>> x,
                                     std::span<const std::pair<std::size_t, std::size_t>> points);

// The discrete Hartley transform (DESIGN.md 5.7) of a C-ordered array of reals of the given
// extents (one or more): the separable product of the 1-D transforms
// H_k = sum_j x_j cas(2 pi jk / n), cas = cos + sin, along every axis, evaluated in 113-bit
// arithmetic throughout (the intermediate axes are not rounded). Not remembered: callers that
// repeat an input keep the result.
std::vector<exact_real> dht(std::span<const std::size_t> extents, std::span<const double> x);

} // namespace fftn::test::reference
