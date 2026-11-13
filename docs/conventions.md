# FFTN conventions

What FFTN computes and how it reads and writes memory, stated exactly. The declarations and
their preconditions are documented in the public headers (`include/fftn/`); this page
does not repeat them. FFTN follows FFTW's conventions throughout.

## Example

```cpp file=examples/rows_c2c.cpp region=rows
const std::size_t n = 12;
const std::size_t count = 3;

// Plans are built once and reused: three rows of 12 points, one after another.
const fftn::c2c_plan<double> forward(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
const fftn::c2c_plan<double> backward(fftn::rows_c2c<double>(n, count, fftn::direction::backward),
                                      {.in_place = true});

// Scratch for one execution at a time, allocated here and never during execute.
fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

// Row r holds the tone exp(+2 pi i (r + 1) j / n).
std::vector<std::complex<double>> x(n * count);
for (std::size_t r = 0; r < count; ++r) {
  for (std::size_t j = 0; j < n; ++j) {
    const double phase = 2 * std::numbers::pi * double((r + 1) * j) / double(n);
    x[r * n + j] = std::polar(1.0, phase);
  }
}

// Forward, out of place: row r of the spectrum is n at bin r + 1 and 0 elsewhere.
std::vector<std::complex<double>> y(n * count);
forward.execute(x.data(), y.data(), ws);

// Backward, in place (the plan says so): FFTN does not normalise, so this gives n * x.
backward.execute(y.data(), y.data(), ws);
```

## What is computed

For one transform of logical lengths n₀ × … × n_{r−1} (r = 1, 2 or 3 axes, n₀ slowest):

- **c2c**, `shape<std::complex<T>, std::complex<T>>`:
  Y[k] = Σ_j X[j] · exp(s · 2πi · Σ_d j_d k_d / n_d), with s = −1 for
  `direction::forward` and s = +1 for `direction::backward`.
- **r2c**, `shape<T, std::complex<T>>`: the forward c2c transform of the real input, of
  which only the outputs with k_{r−1} ≤ n_{r−1}/2 are written: n_{r−1}/2 + 1 complex
  values along the last axis (integer division). The others follow from Hermitian
  symmetry. The imaginary parts of the bins that must be real are 0 to rounding.
- **c2r**, `shape<std::complex<T>, T>`: the backward transform of the Hermitian array
  whose stored half is the input, real by construction. For rows, the imaginary parts of
  bin 0 and, for even n, of bin n/2 are ignored. For more axes, the backward c2c
  transforms of the other axes are applied first and the rows of the last axis are then
  read as above.
- Nothing is normalised: backward after forward multiplies by n₀ · … · n_{r−1}.
- Lengths are any integer of at least 1. Prime factors above 5 currently go through a
  direct O(p²) pass, which `plan::describe()` reports; it is slow for large primes.

## Shapes and layouts

A shape lists its transformed axes (`shape::axes`, slowest first) and its batch
dimensions (`shape::batches`, slowest first; none means one transform). Transforms are
numbered in batch order with the last batch dimension fastest; `execute_rows` uses these
numbers.

- **Units.** Strides and distances count elements of each side's own type: real values
  on the real side of r2c and c2r, complex values on the complex side. They may be
  negative. The pointers passed to `execute` address element 0 of transform 0, which with
  negative strides is not the lowest address.
- **Defaults.** A stride or distance of 0 means the tight C-ordered default, worked out
  for each zero independently of the other values: the last axis has stride 1; each
  other axis spans the axes after it; the innermost batch dimension is one whole array
  apart; each outer batch dimension spans the inner ones. Along the last axis an array
  spans n values, except the complex side of real transforms (n/2 + 1) and the real side
  of in-place real transforms (2(n/2 + 1), see below).
- **Real transforms out of place.** The real side has n values per row and the complex
  side n/2 + 1, each tight by default.
- **No overlap.** No two output elements may share an address. Plan construction checks
  a sufficient condition: ordered by absolute stride, each dimension's stride (or
  distance) exceeds the reach of all smaller ones, which every nested layout (rows,
  columns, padded rows, planes, reversed strides) satisfies. Inputs may overlap each
  other.

## In place

A plan built with `options::in_place` is executed with the same address as input and
output, and overwrites its input. The layouts must agree, which construction checks:

- c2c: every input stride and distance equals the output's.
- r2c and c2r (FFTW's padded layout): the last axis is contiguous on both sides, and every
  other stride and every distance on the real side is twice the complex side's. A real
  row then occupies the first n of the 2(n/2 + 1) reals of its complex row; the last
  reals of the row are padding, overwritten by r2c and ignored by c2r. With default
  strides and distances this is what you get.

Out of place, input and output must not overlap at all.

## What is preserved

- The input of an out-of-place plan is never written, for every kind (c2r included: a
  c2r of two or more axes keeps its intermediate in the workspace instead).
- No memory outside the described output elements is written: not the gaps between
  strided elements or rows, not the padding of out-of-place rows.
- An in-place plan overwrites the elements it describes, padding included.

## Alignment

Arrays may have any alignment the element type allows (that of `T`, also for
`std::complex<T>` data). Workspaces over caller memory may have any alignment at all.
Alignment never changes results: every offset gives the same bits.

## Workspaces and execution

Each execution needs a `workspace` of at least `plan::workspace_bytes()` bytes, used by
no other execution meanwhile. Plans hold no scratch, so one plan may be executed from
several threads at once with a workspace each. Execution never allocates, locks, throws
or does I/O; everything is decided and allocated when the plan is constructed.

## Accuracy

With N the number of points in one transform, L = max(1, log₂ N) and ε the machine
epsilon of `T`, every transform the test suites run satisfies, against a
quadruple-precision reference, a relative RMS error ‖Y − X‖₂ / ‖X‖₂ ≤ 3ε√L and a largest
error ‖Y − X‖∞ / ‖X‖∞ ≤ 3εL, for every length up to 512, sampled lengths up to 2¹⁴ (primes
among them) and both precisions. Measured errors sit 5 or more times below these bounds.
Twiddle factors are correctly rounded.

## Determinism

For a given plan, results are bitwise identical:

- across repeated executions, copies of the plan and concurrent executions;
- for every transform of a batch, whatever its position, through `execute` or through
  `execute_rows` with any `first`;
- at any alignment of the arrays and the workspace.

Results may differ in the last bits between instruction sets, between plans built with
different options (in place or not), and between library versions.
