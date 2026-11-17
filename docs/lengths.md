# Fast lengths

FFTN transforms every length, but not every length equally fast. A power of two is fast, and
so are most lengths whose prime factors are small. A large prime is computed through Rader's
or Bluestein's algorithm, several times slower per point. An application that may pad its
data, a simulation grid for example, can choose a length FFTN computes fast. The
declarations, preconditions and guarantees are in `include/fftn/lengths.hpp`; this page says
what the two queries mean.

```cpp file=examples/lengths.cpp region=lengths
// The grid needs at least 1001 points: pad it to the next fast length.
const std::size_t n = fftn::next_fast_length(1001); // 1008 = 2^4 * 3^2 * 7
const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(n, 1, fftn::direction::forward));

// Real transforms: an even length whose half is fast.
const std::size_t r = fftn::next_fast_length(1001, fftn::length_kind::real);

// This machine's estimate, per point, relative to the power of two at or above the length.
const double prime = fftn::relative_cost<double>(4093);  // a large prime: several times
const double padded = fftn::relative_cost<double>(4096); // a power of two: 1
```

## `next_fast_length`

`next_fast_length(n)` is the smallest length at or above n that FFTN computes fast. Fast
means that the planner's reference cost table prices a row of the length, per point, within
1.25 times a row of the power of two at or above it.
- "Fast" is a property of the reference table, not a measured guarantee on every machine.
  Experiment P4 timed the lengths it picks: every pick was faster than the length it
  replaced, and on the two x86 machines 78 % of them ran within 1.3 times the power of two
  above, but some did not (140, the pick for 131, ran at 1.76 times 256's time per point on
  one Xeon).
- The reference table is a frozen copy, fixed for each library version whatever machine the
  program runs on and apart from the tables the planner tunes, so a grid chosen with it is
  the same everywhere.
- Every fast length is a product of 2, 3, 5, 7 and 13, or one of the primes 11, 17, 19 and
  23 alone, which FFTN computes in a single codelet. Not every such product is fast: 27, 39
  and 45 are not, for example. Composites with a factor 11, 17, 19 or 23 are never fast:
  measured, they were slower than the table believes (P4).
- The result is never above the next power of two, and is usually within a few per cent of
  n: up to 200000, the largest step is from 41 to 48.
- n may be at most 2^40; above it, `next_fast_length` throws `std::invalid_argument`. A call
  costs microseconds, under a millisecond up to 2^40.

`next_fast_length(n, fftn::length_kind::real)` is for real transforms. The result is even and its half is fast,
since a real transform of an even length runs a complex transform of half that length; for n
of 0, 1 or 2 it is 2.
- In two or three dimensions only the halved axis, the last, takes `length_kind::real`. The other axes
  are complex transforms of their full length and take `next_fast_length(n)`.
- scipy's `next_fast_len(target, real=True)` means something else: it returns a 5-smooth
  length, odd ones included, since its real transforms take odd lengths directly. FFTN
  halves only even real lengths, so its real lengths are even.

## `relative_cost`

`relative_cost<T>(n)` is this machine's estimate of a complex row of n points of `T`, per
point, relative to the power of two at or above n. It uses the cost tables of the
instruction set a plan would use here (or of another instruction set, if one is named). It
is about 1 for lengths as fast as a power of two, and several for large primes.

It is the planner's estimate for rows in cache, not a measurement. It does not price memory
beyond the first-level cache, nor the long-row choice between a chain and a blocked grid.
n may be at most 2^40, as for `next_fast_length`.
Wider vector registers prefer lengths with a larger power-of-two factor: with eight complex
floats per register (AVX-512 in float), lengths that 8 does not divide cost more, and
`relative_cost` shows it where `next_fast_length`, fixed across machines, does not.
