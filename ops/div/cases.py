"""The cases with which the **Div** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled pair of operands per case.  Each case goes through a
                     one-node Div model (opset 14) and through the implementation, and
                     the two results are compared bit for bit, the sign of a zero
                     included, because the document specifies it.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the domain of the integer
                     types, the exact arithmetic of the float section, the worked
                     examples of every section of the document, the document's declared
                     types against the ONNX schema, and the real section, which has no
                     ONNX type and is checked against exact rational arithmetic instead.
  * ``COVERAGE``  -- the four section anchors of the document, each with the number of
                     cases that exercise it (the real section is exercised by the exact
                     rational cases of the real sweep: a real number has no ONNX type and
                     cannot go through a one-node model).

Two measurements of 2026-11-22 are respected here: the runtime's signed 32- and 64-bit
Div kernel dies with SIGFPE on the pair (INT_MIN, -1), so that pair never reaches ONNX
Runtime -- not through a case, not through a sweep array, and not through ``impl``,
which in the Lean conformance replay *is* ONNX Runtime; its value is stated from exact
arithmetic in the int sweep's summary.  And no negative Python literal is handed to an
unsigned ``dtype=`` constructor: every unsigned array is built from values that are
already non-negative.

The real section's cases are decided by the document's own broadcasting rule, computed
independently with ``np.broadcast_shapes`` over the two operand shapes; the shape a case
reports as documented is that shape, never the shape an implementation happened to
return and never a shape guessed from the layout of the document's matrix notation.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Div"
OPSET = 14
IR_VERSION = 10

NODE_OUTPUTS = ("out",)

ALL_FLOAT = [np.float16, np.float32, np.float64]
SIGNED_INT = [np.int8, np.int16, np.int32, np.int64]
UNSIGNED_INT = [np.uint8, np.uint16, np.uint32, np.uint64]
ALL_INT = SIGNED_INT + UNSIGNED_INT
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}

# The signed widths whose Div kernel lets the process die on (INT_MIN, -1).
ORT_SIGFPE_DTYPES = (np.dtype(np.int32), np.dtype(np.int64))


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One Div node over two arrays of the same type, in a single runtime call."""
    proto = helper.np_dtype_to_tensor_dtype(a.dtype)
    vi_a = helper.make_tensor_value_info("A", proto, list(a.shape))
    vi_b = helper.make_tensor_value_info("B", proto, list(b.shape))
    vi_c = helper.make_tensor_value_info("C", proto, None)
    node = helper.make_node(OP, ["A", "B"], ["C"], name="div0")
    graph = helper.make_graph([node], "div_graph", [vi_a, vi_b], [vi_c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    (c,) = sess.run(["C"], {"A": a, "B": b})
    return c


def _bits_equal(want: np.ndarray, got: np.ndarray) -> bool:
    """Bit-for-bit equality, NaN equal to NaN, of two arrays of the same type."""
    if want.shape != got.shape or want.dtype != got.dtype:
        return False
    if want.dtype.kind == "f":
        uint = UINT_OF_WIDTH[want.dtype.itemsize]
        bw = np.ascontiguousarray(want).view(uint)
        bg = np.ascontiguousarray(got).view(uint)
        both_nan = np.isnan(want) & np.isnan(got)
        return bool(np.all((bw == bg) | both_nan))
    return bool(np.all(want == got))


def _diff_indices(want: np.ndarray, got: np.ndarray, limit=5):
    """Flat indices where two arrays differ (NaN equal to NaN); None on shape/type."""
    if want.shape != got.shape or want.dtype != got.dtype:
        return None
    if want.dtype.kind == "f":
        uint = UINT_OF_WIDTH[want.dtype.itemsize]
        bw = np.ascontiguousarray(want).view(uint)
        bg = np.ascontiguousarray(got).view(uint)
        both_nan = np.isnan(want) & np.isnan(got)
        mask = ~((bw == bg) | both_nan)
    else:
        mask = ~(want == got)
    return [int(i) for i in np.nonzero(mask.ravel())[0][:limit]]


# ---------------------------------------------------------------------------
# exact arithmetic: the reference the document itself states
# ---------------------------------------------------------------------------
#
# The float formula of the document is round(A / B): the exact quotient in the reals,
# rounded to the nearest value of the type with roundTiesToEven and an unbounded
# exponent range (overflow gives an infinity, underflow a zero, both of the sign of the
# exact quotient).  It is computed here from the operands' exact values -- Fraction of
# the float, never of the decimal literal -- independently of the runtime and of the
# platform's library.


def _pow2(e: int) -> Fraction:
    return Fraction(2) ** e


def _round_half_even(x: Fraction) -> int:
    """roundTiesToEven of a non-negative rational to the nearest integer."""
    q = x.numerator // x.denominator
    r = x - q
    half = Fraction(1, 2)
    if r > half:
        return q + 1
    if r < half:
        return q
    return q if q % 2 == 0 else q + 1


def _frac_round_to_type(value: Fraction, dtype):
    """Round an exact rational to the nearest value of `dtype`, ties to even.

    The exponent range is unbounded, as the document states: a magnitude whose nearest
    value exceeds the largest finite value of the type gives an infinity of its sign.
    """
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    p = f.nmant + 1                       # significand bits, implicit bit included
    emin = f.minexp                       # exponent of the smallest normal
    max_finite = Fraction(float(f.max))
    if value == 0:
        return dt.type(0.0)
    sign = -1.0 if value < 0 else 1.0
    a = abs(value)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while _pow2(e) > a:
        e -= 1
    while _pow2(e + 1) <= a:
        e += 1
    if e < emin:
        e = emin                          # the subnormal range: uniform spacing
    spacing = _pow2(e - p + 1)
    n = _round_half_even(a / spacing)
    result = n * spacing
    if result > max_finite:
        return dt.type(sign * float("inf"))
    return dt.type(sign * float(result))


def _exact_float_div(x, y, dtype):
    """C[i] of **Div** (float, float): the document's case analysis, in exact terms."""
    dt = np.dtype(dtype)
    fx, fy = float(x), float(y)
    nan = dt.type(float("nan"))
    inf = float("inf")
    if np.isnan(fx) or np.isnan(fy):
        return nan                                   # the first case: propagation
    if np.isinf(fx) and np.isinf(fy):
        return nan                                   # the invalid operation, inf/inf
    if fx == 0.0 and fy == 0.0:
        return nan                                   # the invalid operation, 0/0
    sign = -1.0 if (bool(np.signbit(fx)) != bool(np.signbit(fy))) else 1.0
    if fy == 0.0:
        return dt.type(sign * inf)                   # division by zero
    if np.isinf(fx):
        return dt.type(sign * inf)                   # an infinity over a finite non-null
    if fx == 0.0 or np.isinf(fy):
        return dt.type(sign * 0.0)                   # a null quotient, signed
    return _frac_round_to_type(Fraction(fx) / Fraction(fy), dt)


def _double_rounded_float16(x, y):
    """The same quotient computed in float32 and narrowed, or None where it does not
    apply.  A kernel that divides float16 operands in float32 rounds twice; on the rare
    pair whose exact quotient sits a hair from a float16 midpoint that differs from the
    single rounding the document states.  It is recorded, not required."""
    fx, fy = float(x), float(y)
    if (np.isnan(fx) or np.isnan(fy) or np.isinf(fx) or np.isinf(fy) or fy == 0.0):
        return None
    with np.errstate(all="ignore"):
        q = np.float32(fx) / np.float32(fy)
    return np.float16(q)


# ---------------------------------------------------------------------------
# the integer case analysis of the document, transcribed literally
# ---------------------------------------------------------------------------


def _trunc_div(x: int, y: int) -> int:
    """trunc(x / y): the integer part, rounded toward zero, of two integers."""
    q = abs(x) // abs(y)
    return q if ((x < 0) == (y < 0)) else -q


def doc_int_div(x: int, y: int, n: int) -> int:
    """The case analysis of **Div** (int, int), in exact arithmetic."""
    q = _trunc_div(x, y)
    lo, hi = -(2 ** (n - 1)), 2 ** (n - 1) - 1
    if q > hi:
        return q - 2 ** n
    if q < lo:
        return q + 2 ** n
    return q


def doc_uint_div(x: int, y: int) -> int:
    """The formula of **Div** (uint, uint): the floor, which is the truncation."""
    return x // y


def _doc_int_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dt = a.dtype
    n = dt.itemsize * 8
    signed = dt.kind == "i"
    xs = a.ravel().tolist()
    ys = b.ravel().tolist()
    if signed:
        vals = [doc_int_div(int(x), int(y), n) for x, y in zip(xs, ys)]
    else:
        vals = [doc_uint_div(int(x), int(y)) for x, y in zip(xs, ys)]
    return np.array(vals, dtype=dt).reshape(a.shape)


def _rand_values(rng, dt, n):
    """n random values of `dt`, drawn uniformly over the whole bit patterns of the type."""
    bits = dt.itemsize * 8
    hi = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    lo = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    raw = (hi << np.uint64(32)) | lo
    if bits < 64:
        raw = raw & np.uint64((1 << bits) - 1)
    return raw.astype(np.dtype("u%d" % dt.itemsize)).view(dt)


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------


def _float_cases():
    f16 = np.float16
    f32 = np.float32
    f64 = np.float64
    inf = float("inf")
    nan = float("nan")
    f16max = float(np.finfo(np.float16).max)
    f16tiny = float(np.finfo(np.float16).tiny)
    f16sub = float(np.nextafter(np.float16(0.0), np.float16(1.0)))
    f32max = float(np.finfo(np.float32).max)
    f32tiny = float(np.finfo(np.float32).tiny)
    f32sub = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    f64max = float(np.finfo(np.float64).max)
    f64sub = float(np.nextafter(np.float64(0.0), np.float64(1.0)))

    out = []
    out.append(("float16 Example 1 (every quotient representable)",
                [np.array([6.0, -9.0, 1.0], dtype=f16),
                 np.array([3.0, 4.0, 8.0], dtype=f16)]))
    out.append(("float32 Example 1",
                [np.array([6.0, -9.0, 1.0], dtype=f32),
                 np.array([3.0, 4.0, 8.0], dtype=f32)]))
    out.append(("double Example 3 (three rounded quotients)",
                [np.array([1.0, 10.0, -10.0], dtype=f64),
                 np.array([3.0, 3.0, 3.0], dtype=f64)]))
    out.append(("float32 Example 2 (the special values, as documented)",
                [np.array([1.0, -1.0, inf, inf, 0.0, -0.0, 0.0, nan], dtype=f32),
                 np.array([0.0, 0.0, inf, 4.0, 4.0, 4.0, 0.0, 1.0], dtype=f32)]))
    out.append(("float16 Example 2 (the special values, as documented)",
                [np.array([1.0, -1.0, inf, inf, 0.0, -0.0, 0.0, nan], dtype=f16),
                 np.array([0.0, 0.0, inf, 4.0, 4.0, 4.0, 0.0, 1.0], dtype=f16)]))
    out.append(("double Example 2 (the special values, as documented)",
                [np.array([1.0, -1.0, inf, inf, 0.0, -0.0, 0.0, nan], dtype=f64),
                 np.array([0.0, 0.0, inf, 4.0, 4.0, 4.0, 0.0, 1.0], dtype=f64)]))
    out.append(("a NaN operand in every position (float32)",
                [np.array([nan, 1.0, nan, -inf, 0.0], dtype=f32),
                 np.array([1.0, nan, 0.0, inf, nan], dtype=f32)]))
    out.append(("signed null quotients, a zero over a finite non-null (float32)",
                [np.array([0.0, -0.0, 0.0, -0.0], dtype=f32),
                 np.array([2.0, 2.0, -2.0, -2.0], dtype=f32)]))
    out.append(("signed null quotients, a finite dividend over an infinity (float32)",
                [np.array([1.0, -1.0, 0.0, -0.0], dtype=f32),
                 np.array([inf, -inf, inf, -inf], dtype=f32)]))
    out.append(("an infinity over a finite non-null divisor (float32)",
                [np.array([inf, -inf, inf, -inf], dtype=f32),
                 np.array([2.0, 2.0, -2.0, -2.0], dtype=f32)]))
    out.append(("division by zero, both signs of the divisor (float32)",
                [np.array([1.0, -1.0, 1.0, -1.0], dtype=f32),
                 np.array([0.0, 0.0, -0.0, -0.0], dtype=f32)]))
    out.append(("overflow to an infinity (float32)",
                [np.array([f32max, -f32max, f32max], dtype=f32),
                 np.array([f32sub, f32sub, 0.5], dtype=f32)]))
    out.append(("underflow to a signed zero and to a subnormal (float32)",
                [np.array([f32sub, -f32sub, 3.0 * f32sub, f32tiny], dtype=f32),
                 np.array([2.0, 2.0, 2.0, 2.0], dtype=f32)]))
    out.append(("overflow and underflow in double",
                [np.array([f64max, -f64max, f64sub, -f64sub], dtype=f64),
                 np.array([f64sub, 0.5, 2.0, 2.0], dtype=f64)]))
    out.append(("float16 range limits",
                [np.array([f16max, -f16max, f16sub, -f16sub, f16tiny], dtype=f16),
                 np.array([f16sub, 2.0, 2.0, 2.0, 2.0], dtype=f16)]))
    out.append(("quotients rounded to the type (float32)",
                [np.array([1.0, 1.0, 7.0, -7.0], dtype=f32),
                 np.array([3.0, 10.0, 2.0, 2.0], dtype=f32)]))
    out.append(("broadcast a rank-1 divisor over a 2x2 dividend (float32)",
                [np.array([[4.0, 9.0], [16.0, 25.0]], dtype=f32),
                 np.array([2.0, 5.0], dtype=f32)]))
    out.append(("rank-0 dividend broadcast over a rank-1 divisor (float32)",
                [np.array(6.0, dtype=f32),
                 np.array([2.0, 3.0, 4.0], dtype=f32)]))
    out.append(("rank-0 divisor broadcast over a rank-2 dividend (double)",
                [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=f64),
                 np.array(2.0, dtype=f64)]))
    out.append(("two rank-0 operands (float32)",
                [np.array(-7.0, dtype=f32), np.array(2.0, dtype=f32)]))
    out.append(("a zero-sized dimension over a dimension of size 1 (float32)",
                [np.zeros((0, 3), dtype=f32), np.ones((3,), dtype=f32)]))
    out.append(("two equal zero-sized dimensions (double)",
                [np.zeros((0, 2), dtype=f64), np.zeros((0, 2), dtype=f64)]))
    return out


def _int_cases():
    i8, i16, i32, i64 = np.int8, np.int16, np.int32, np.int64
    i8min, i8max = int(np.iinfo(i8).min), int(np.iinfo(i8).max)
    i32min, i32max = int(np.iinfo(i32).min), int(np.iinfo(i32).max)
    i64min, i64max = int(np.iinfo(i64).min), int(np.iinfo(i64).max)

    out = []
    out.append(("int Example 1 (int8: truncation toward zero and the wrap)",
                [np.array([7, -7, 7, -7, -128], dtype=i8),
                 np.array([2, 2, -2, -2, -1], dtype=i8)]))
    out.append(("int16 truncation toward zero, not the floor",
                [np.array([7, -7, 7, -7, 1, -1], dtype=i16),
                 np.array([2, 2, -2, -2, -2, 2], dtype=i16)]))
    out.append(("int8 extremes, the wrap of MIN over -1 included",
                [np.array([i8min, i8max, i8min + 1, i8max, 0], dtype=i8),
                 np.array([-1, -1, -1, 1, 1], dtype=i8)]))
    out.append(("int16 extremes, the wrap of MIN over -1 included",
                [np.array([int(np.iinfo(i16).min), int(np.iinfo(i16).max)], dtype=i16),
                 np.array([-1, -1], dtype=i16)]))
    out.append(("int32 boundary values, MIN over -1 excluded",
                [np.array([i32min, i32max, i32max, i32min + 1], dtype=i32),
                 np.array([1, 1, -1, -1], dtype=i32)]))
    out.append(("int64 boundary values, MIN over -1 excluded",
                [np.array([i64min, i64max, i64max, i64min + 1], dtype=i64),
                 np.array([1, 1, -1, -1], dtype=i64)]))
    out.append(("int32 broadcast: a rank-1 divisor over a 2x2 dividend",
                [np.array([[7, -7], [-9, 9]], dtype=i32),
                 np.array([2, -2], dtype=i32)]))
    out.append(("int8 rank-0 divisor broadcast over a 2x2 dividend",
                [np.array([[7, -7], [9, -9]], dtype=i8), np.array(-2, dtype=i8)]))
    out.append(("int16 a zero-sized dividend over a size-1 divisor",
                [np.zeros((0,), dtype=i16), np.array([2], dtype=i16)]))
    out.append(("int8 a null dividend over a non-null divisor",
                [np.zeros((3,), dtype=i8), np.array([2, -3, 4], dtype=i8)]))
    out.append(("int64 truncation of large quotients",
                [np.array([10 ** 18, -10 ** 18, 10 ** 18 - 1], dtype=i64),
                 np.array([3, 3, 7], dtype=i64)]))
    return out


def _uint_cases():
    u8, u16, u32, u64 = np.uint8, np.uint16, np.uint32, np.uint64

    out = []
    out.append(("uint Example 1 (uint8)",
                [np.array([7, 200, 255, 6], dtype=u8),
                 np.array([2, 3, 255, 6], dtype=u8)]))
    out.append(("uint8 quotients of every boundary",
                [np.array([0, 1, 127, 128, 255, 255], dtype=u8),
                 np.array([1, 2, 2, 255, 1, 255], dtype=u8)]))
    out.append(("uint16 extremes",
                [np.array([0, 1, 65535, 65534], dtype=u16),
                 np.array([1, 65535, 65535, 2], dtype=u16)]))
    out.append(("uint32 extremes",
                [np.array([0, 1, 4294967295, 4294967294], dtype=u32),
                 np.array([1, 4294967295, 4294967295, 2], dtype=u32)]))
    out.append(("uint64 extremes",
                [np.array([0, 1, 2 ** 64 - 1, 2 ** 64 - 2], dtype=u64),
                 np.array([1, 2 ** 64 - 1, 2 ** 64 - 1, 2], dtype=u64)]))
    out.append(("uint8 broadcast: a rank-1 divisor over a 2x2 dividend",
                [np.array([[7, 200], [255, 6]], dtype=u8),
                 np.array([2, 3], dtype=u8)]))
    out.append(("uint16 rank-0 dividend broadcast over a rank-1 divisor",
                [np.array(255, dtype=u16), np.array([1, 3, 5], dtype=u16)]))
    out.append(("uint32 a zero-sized dimension over a dimension of size 1",
                [np.zeros((0, 3), dtype=u32), np.ones((1, 3), dtype=u32)]))
    return out


# ---------------------------------------------------------------------------
# the real section: exact rationals, no ONNX type
# ---------------------------------------------------------------------------


def _real_examples():
    """The real section's cases: exact rationals, decided by the document's rule.

    Each entry is (label, A, B, documented C or None).  The documented result follows
    the document's broadcasting rule -- ``C`` has the shape of ``A`` and ``B`` broadcast
    to a common shape -- and is written with exactly that shape: a rank-0 operand
    broadcast over a rank-1 operand gives a rank-1 result, never a matrix row.
    """
    F = Fraction
    return [
        ("real Example 1",
         [F(7), F(-9), F(1)],
         [F(2), F(4), F(3)],
         [F(7, 2), F(-9, 4), F(1, 3)]),
        ("real Example 2",
         [[F(4), F(9)], [F(16), F(25)]],
         [F(2), F(5)],
         [[F(2), F(9, 5)], [F(8), F(5)]]),
        ("real: 1/3 is exact, not approximated",
         [F(1)],
         [F(3)],
         [F(1, 3)]),
        ("real: a null dividend over a non-null divisor (rank-0 dividend)",
         F(0),
         [F(2), F(-3), F(5, 2)],
         [F(0), F(0), F(0)]),
        ("real: rank-0 divisor broadcast over a 2x2 dividend",
         [[F(1), F(2)], [F(3), F(4)]],
         F(2),
         [[F(1, 2), F(1)], [F(3, 2), F(2)]]),
        ("real: a rank-1 dividend over a column divisor of shape (2, 1)",
         [F(3), F(5)],
         [[F(2)], [F(4)]],
         [[F(3, 2), F(5, 2)], [F(3, 4), F(5, 4)]]),
    ]


_FLOAT_CASES = _float_cases()
_INT_CASES = _int_cases()
_UINT_CASES = _uint_cases()
_REAL_EXAMPLES = _real_examples()


def cases():
    for item in _FLOAT_CASES:
        yield item
    for item in _INT_CASES:
        yield item
    for item in _UINT_CASES:
        yield item


COVERAGE = {
    "real": len(_REAL_EXAMPLES),
    "float": len(_FLOAT_CASES),
    "int": len(_INT_CASES),
    "uint": len(_UINT_CASES),
}


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_int_domain(impl):
    """The document's integer case analysis against ONNX Runtime and the implementation.

    The two 8-bit types are covered exhaustively over every non-null divisor; the wider
    types use their boundary values and a random sample.  The pair (INT_MIN, -1) of
    int32 and int64 is excluded from every array sent to the runtime and from every
    call of `impl`, which is ONNX Runtime itself in the Lean conformance replay: the
    runtime's signed 32- and 64-bit Div kernel dies with SIGFPE on it.  The document's
    value for the pair is stated in the summary, from exact arithmetic.  A null divisor
    is outside the document's decided cases and appears in no array.
    """
    rng = np.random.default_rng(20261003)
    rows, failures = [], []
    for dtype in ALL_INT:
        dt = np.dtype(dtype)
        info = np.iinfo(dt)
        exhaustive = dt.itemsize == 1
        if exhaustive:
            xs = list(range(int(info.min), int(info.max) + 1))
            ys = [v for v in xs if v != 0]
            pairs = [(x, y) for x in xs for y in ys]
        else:
            if dt.kind == "i":
                edges = [int(info.min), int(info.min) + 1, -3, -2, -1, 0, 1, 2, 3,
                         int(info.max) - 1, int(info.max)]
            else:
                edges = [0, 1, 2, 3, int(info.max) - 1, int(info.max)]
            pairs = [(x, y) for x in edges for y in edges if y != 0]
            ra = _rand_values(rng, dt, 200).tolist()
            rb = _rand_values(rng, dt, 200).tolist()
            pairs += [(x, y) for x, y in zip(ra, rb) if y != 0]
        note = ""
        if dt in ORT_SIGFPE_DTYPES:
            m = int(info.min)
            pairs = [(x, y) for x, y in pairs if not (x == m and y == -1)]
            note = ("; (%d, -1) not sent to the runtime (SIGFPE); the document's value "
                    "is %d, by exact arithmetic"
                    % (m, doc_int_div(m, -1, dt.itemsize * 8)))

        aa = np.array([p[0] for p in pairs], dtype=dt)
        bb = np.array([p[1] for p in pairs], dtype=dt)

        want = _doc_int_array(aa, bb)          # the document, literally
        try:
            ref = run_ort_batch(aa, bb)        # ONNX Runtime
        except Exception as exc:
            failures.append({"type": dt.name, "who": "ONNX Runtime refused the batch",
                             "detail": "%s: %s" % (type(exc).__name__, exc)})
            rows.append("%s: refused by the runtime" % dt.name)
            continue
        got = np.asarray(impl.div(aa, bb))     # the implementation

        bad = _diff_indices(want, ref, limit=5)
        if bad is None:
            failures.append({"type": dt.name, "who": "the document vs ONNX Runtime",
                             "detail": "shape %s or type %s differs from the document's "
                                       "%s / %s" % (ref.shape, ref.dtype,
                                                    want.shape, want.dtype)})
        elif bad:
            failures.append({
                "type": dt.name, "who": "the document vs ONNX Runtime",
                "detail": "; ".join("doc: %d / %d = %d, ort: %d"
                                    % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                       int(want.ravel()[i]), int(ref.ravel()[i]))
                                    for i in bad)})
        bad = _diff_indices(ref, got, limit=5)
        if bad is None:
            failures.append({"type": dt.name, "who": "the implementation vs ONNX Runtime",
                             "detail": "shape %s or type %s differs from the runtime's "
                                       "%s / %s" % (got.shape, got.dtype,
                                                    ref.shape, ref.dtype)})
        elif bad:
            failures.append({
                "type": dt.name, "who": "the implementation vs ONNX Runtime",
                "detail": "; ".join("%d / %d: ort %d, impl %d"
                                    % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                       int(np.asarray(ref).ravel()[i]),
                                       int(np.asarray(got).ravel()[i])) for i in bad)})
        rows.append("%s: %s%s"
                    % (dt.name,
                       "exhaustive (65280 operand pairs)" if exhaustive
                       else "%d operand pairs" % len(aa), note))
    return {"sweep": "int domain (document vs ORT vs implementation)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def _float_probe(dt, rng):
    """The operands of the exact float sweep: the special values and a random sample."""
    f = np.finfo(dt)
    maxf = float(f.max)
    tiny = float(f.tiny)
    small = float(np.nextafter(dt.type(0.0), dt.type(1.0)))
    specials = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, 2.0, -2.0, 3.0, -3.0,
                tiny, -tiny, small, -small, maxf, -maxf, maxf / 2.0, -maxf / 2.0,
                float("inf"), float("-inf"), float("nan")]
    pairs = [(x, y) for x in specials for y in specials]

    n_rand = 200
    ra = (rng.random(n_rand) * 2.0 - 1.0) * maxf
    rb = (rng.random(n_rand) * 2.0 - 1.0) * maxf
    ra[:80] = (rng.random(80) * 2.0 - 1.0) * tiny        # the subnormal range
    rb[:80] = (rng.random(80) * 2.0 - 1.0) * tiny
    pairs += list(zip(ra.tolist(), rb.tolist()))

    a = np.array([p[0] for p in pairs], dtype=dt)
    b = np.array([p[1] for p in pairs], dtype=dt)
    return a, b


def sweep_float_exact(impl):
    """The float section's formula against exact arithmetic, ORT, and the implementation.

    Each element is round(A / B) as the document states it, computed with `Fraction`
    from the operands' exact values; the special values are decided by the document's
    own case analysis before any Fraction is formed.  A float16 departure that is the
    same quotient rounded through float32 is recorded as double rounding, not required.
    """
    rng = np.random.default_rng(20261004)
    rows, failures = [], []
    recorded = 0
    checked = 0
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        a, b = _float_probe(dt, rng)
        exact = np.array([_exact_float_div(x, y, dt)
                          for x, y in zip(a.tolist(), b.tolist())], dtype=dt)
        ref = run_ort_batch(a, b)
        got = np.asarray(impl.div(a, b))
        checked += int(a.size)

        for who, other in (("ONNX Runtime", ref), ("the implementation", got)):
            bad = _diff_indices(exact, other, limit=None)
            if bad is None:
                failures.append({"type": dt.name, "who": who,
                                 "detail": "shape %s or type %s differs from the exact "
                                           "result's %s / %s"
                                           % (other.shape, other.dtype,
                                              exact.shape, exact.dtype)})
                continue
            real_bad = []
            for i in bad:
                if dt == np.dtype(np.float16):
                    dr = _double_rounded_float16(a.ravel()[i], b.ravel()[i])
                    if dr is not None and _bits_equal(
                            np.asarray(other.ravel()[i:i + 1]).ravel(),
                            np.asarray(dr).reshape(1)):
                        recorded += 1
                        continue
                real_bad.append(i)
            if real_bad:
                failures.append({
                    "type": dt.name, "who": "%s vs the exact quotient of the document" % who,
                    "detail": "; ".join(
                        "%r / %r: exact %r, %s %r"
                        % (float(a.ravel()[i]), float(b.ravel()[i]),
                           float(exact.ravel()[i]), who, float(other.ravel()[i]))
                        for i in real_bad[:5]),
                })
        rows.append("%s: %d operand pairs" % (dt.name, a.size))
    summary = ("%d operand pairs: %s; the float16 departures that are the same quotient "
               "rounded through float32 are recorded, not required (%d recorded)"
               % (checked, ", ".join(rows), recorded))
    return {"sweep": "the float formula against exact arithmetic (Fraction)",
            "summary": summary, "failures": failures}


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact rationals.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    implementation against the exact quotient of the section's Examples and of its
    formula, in exact rational arithmetic.  The expected shape of every case is the
    document's broadcast shape of the two operand shapes, computed independently with
    ``np.broadcast_shapes``; the documented result is checked against that same shape,
    so a case whose documented layout disagrees with the broadcasting rule is reported
    here rather than passed on to the implementation.
    """
    failures = []
    examples = _REAL_EXAMPLES
    checked = 0
    for label, a, b, documented in examples:
        aa = np.array(a, dtype=object)
        bb = np.array(b, dtype=object)
        try:
            shape = tuple(int(s) for s in np.broadcast_shapes(aa.shape, bb.shape))
        except Exception as exc:
            failures.append({"who": label,
                             "detail": "the document's broadcasting rule rejects shapes "
                                       "%s and %s: %s: %s"
                                       % (aa.shape, bb.shape, type(exc).__name__, exc)})
            continue
        ba, bc = np.broadcast_arrays(aa, bb)
        want = np.empty(ba.shape, dtype=object)
        for i in np.ndindex(ba.shape):
            want[i] = ba[i] / bc[i]

        checked += 1
        try:
            got = np.asarray(impl.div(aa, bb), dtype=object)
        except Exception as exc:
            failures.append({"who": label,
                             "detail": "the implementation raised %s: %s"
                                       % (type(exc).__name__, exc)})
            continue
        if got.shape != shape:
            failures.append({"who": label,
                             "detail": "the document's broadcast shape %s, implementation "
                                       "shape %s" % (shape, got.shape)})
        else:
            bad = [i for i in np.ndindex(got.shape) if got[i] != want[i]]
            if bad:
                failures.append({
                    "who": label,
                    "detail": "; ".join("exact %s, implementation %s"
                                        % (want[i], got[i]) for i in bad[:5])})

        if documented is not None:
            doc = np.array(documented, dtype=object)
            if doc.shape != shape:
                failures.append({
                    "who": label,
                    "detail": "the documented result has shape %s, the document's "
                              "broadcast shape of %s and %s is %s"
                              % (doc.shape, aa.shape, bb.shape, shape)})
            else:
                bad_doc = [i for i in np.ndindex(doc.shape) if doc[i] != want[i]]
                if bad_doc:
                    failures.append({
                        "who": label,
                        "detail": "; ".join("documented %s, exact %s"
                                            % (doc[i], want[i]) for i in bad_doc[:5])})
    return {"sweep": "the real section (exact rational arithmetic, no ONNX Runtime)",
            "summary": "%d cases checked against Fraction arithmetic, their shapes "
                       "against the document's broadcasting rule (np.broadcast_shapes)"
                       % checked,
            "failures": failures}


def _spec_examples():
    """The worked examples of the real, float and integer sections, as documented."""
    f32 = np.float32
    inf = float("inf")
    nan = float("nan")
    return [
        ("float Example 1",
         np.array([6.0, -9.0, 1.0], dtype=f32),
         np.array([3.0, 4.0, 8.0], dtype=f32),
         np.array([2.0, -2.25, 0.125], dtype=f32)),
        ("float Example 2",
         np.array([1.0, -1.0, inf, inf, 0.0, -0.0, 0.0, nan], dtype=f32),
         np.array([0.0, 0.0, inf, 4.0, 4.0, 4.0, 0.0, 1.0], dtype=f32),
         np.array([inf, -inf, nan, inf, 0.0, -0.0, nan, nan], dtype=f32)),
        ("float Example 3",
         np.array([1.0, 10.0, -10.0], dtype=np.float64),
         np.array([3.0, 3.0, 3.0], dtype=np.float64),
         np.array([0.3333333333333333, 3.3333333333333335, -3.3333333333333335],
                  dtype=np.float64)),
        ("int Example 1",
         np.array([7, -7, 7, -7, -128], dtype=np.int8),
         np.array([2, 2, -2, -2, -1], dtype=np.int8),
         np.array([3, -3, -3, 3, -128], dtype=np.int8)),
        ("uint Example 1",
         np.array([7, 200, 255, 6], dtype=np.uint8),
         np.array([2, 3, 255, 6], dtype=np.uint8),
         np.array([3, 66, 1, 1], dtype=np.uint8)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, and as ONNX Runtime does it."""
    failures = []
    examples = _spec_examples()
    for label, a, b, documented in examples:
        ref = run_ort_batch(a, b)
        got = np.asarray(impl.div(a, b))
        if not _bits_equal(documented, ref):
            failures.append({"who": label, "detail": "documented %s, ORT %s"
                             % (documented.tolist(), ref.tolist())})
        if not _bits_equal(documented, got):
            failures.append({"who": label, "detail": "documented %s, implementation %s"
                             % (documented.tolist(), got.tolist())})
    return {"sweep": "the document's worked examples",
            "summary": "%d examples, as documented vs ORT vs implementation"
                       % len(examples),
            "failures": failures}


def sweep_schema_types(impl):
    """The document's declared type families against the ONNX schema's admitted set.

    A type the document declares that the schema does not admit makes every case of that
    type open at model resolution, and a case set that drops it would converge on the
    wrong document.  A type the schema admits that the document does not specify is
    recorded in the summary: the profile specifies a subset.
    """
    declared = ["tensor(float16)", "tensor(float)", "tensor(double)",
                "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)",
                "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)"]
    try:
        from onnx import defs
        schema = defs.get_schema(OP, OPSET)
        admitted = {}
        for constraint in schema.type_constraints:
            admitted[constraint.type_param_str] = list(constraint.allowed_type_strs)
        allowed = set(admitted.get("T", []))
    except Exception as exc:
        return {"sweep": "the document's types against the ONNX schema",
                "summary": "the schema of %s-%d could not be read (%s: %s)"
                           % (OP, OPSET, type(exc).__name__, exc),
                "failures": []}
    failures = []
    for name in declared:
        if name not in allowed:
            failures.append({"who": "the document declares %s" % name,
                             "detail": "the schema of %s-%d admits %s for T"
                                       % (OP, OPSET, sorted(allowed))})
    extra = sorted(allowed - set(declared))
    summary = ("%d declared types, every one admitted by the schema of %s-%d"
               % (len(declared), OP, OPSET))
    if extra:
        summary += ("; the schema also admits %s, which the document does not specify"
                    % ", ".join(extra))
    return {"sweep": "the document's types against the ONNX schema",
            "summary": summary, "failures": failures}


def sweeps():
    return [sweep_int_domain, sweep_float_exact, sweep_real_examples,
            sweep_spec_examples, sweep_schema_types]
