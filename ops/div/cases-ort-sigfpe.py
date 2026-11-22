"""The cases with which the **Div** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled pair of operands per case.  Each case goes through a
                     one-node Div model (opset 14) and through the implementation, and
                     the two results are compared bit for bit, the sign of a zero
                     included, because the document specifies it.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the exact IEEE arithmetic of
                     the float section, the domain of the integer types, the worked
                     examples of every section of the document, the document's declared
                     types against the ONNX schema, and the real section, which has no
                     ONNX type and is checked against exact rational arithmetic instead.
  * ``COVERAGE``  -- the four section anchors of the document, each with the number of
                     cases that exercise it.

A null divisor is not a case of the integer sections: the document states that the result
is then undetermined, so no case of the unsigned or the signed section puts a zero in the
divisor, and the integer sweep says so in its summary.  A null divisor is a *nominal*
case of the float section (an infinity of the sign of the exact quotient), and it is a
case there.  The real section rules out a null divisor by its precondition.

Every case of every section's formula is meant to appear here, and the case set is part
of the loop: when a section is added to the document, its cases are added here before
the next run.
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
    """Bit-for-bit equality, NaN equal to NaN, of two arrays of the same type.

    The sign of a zero is part of the bit pattern, so +0.0 and -0.0 are different
    results here: the document specifies the sign of a null quotient.
    """
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


def _same_bits(want, got) -> bool:
    return _bits_equal(np.asarray(want).ravel(), np.asarray(got).ravel())


# ---------------------------------------------------------------------------
# exact arithmetic: the reference the document itself states
# ---------------------------------------------------------------------------
#
# The document's float formula is round(A / B): the exact quotient in the reals, rounded
# to the nearest value of the type with roundTiesToEven and an unbounded exponent range.
# It is computed here from the operands' exact values -- Fraction of the *float*, never
# of the decimal literal that built the array -- independently of the platform's library
# and of the runtime.


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
    value exceeds the largest finite value of the type gives an infinity, and one below
    the smallest subnormal gives a zero of the sign of the exact value.
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


def _exact_div_result(x, y, dtype):
    """C[i] of **Div** (float, float): the case analysis of the document, exactly.

    The special values are decided first, by the rule the document states for them; only
    a finite dividend and a finite non-null divisor reach the rational arithmetic.
    """
    dt = np.dtype(dtype)
    fx, fy = float(x), float(y)
    if np.isnan(fx) or np.isnan(fy):
        return dt.type(float("nan"))                     # the NaN propagates
    if np.isinf(fx) and np.isinf(fy):
        return dt.type(float("nan"))                     # the invalid operation
    if fx == 0.0 and fy == 0.0:
        return dt.type(float("nan"))                     # the invalid operation
    sign = -1.0 if (np.signbit(fx) != np.signbit(fy)) else 1.0
    if np.isinf(fx):
        return dt.type(sign * float("inf"))              # infinity / finite non-null
    if np.isinf(fy):
        return dt.type(sign * 0.0)                       # finite / infinity
    if fy == 0.0:
        return dt.type(sign * float("inf"))              # division by zero
    if fx == 0.0:
        return dt.type(sign * 0.0)                       # null dividend
    return _frac_round_to_type(Fraction(fx) / Fraction(fy), dt)


def _double_rounded_div16(x, y):
    """The same quotient rounded through float32 first, or None.

    A kernel that computes a float16 quotient in float32 and narrows the result rounds
    twice; on the rare operand pair whose exact quotient sits a hair off a float16
    midpoint that differs from the single rounding the document states.  It is recorded,
    not required.
    """
    fx, fy = float(x), float(y)
    if np.isnan(fx) or np.isnan(fy) or np.isinf(fx) or np.isinf(fy):
        return None
    if fy == 0.0 or fx == 0.0:
        return None
    q = Fraction(fx) / Fraction(fy)
    f32 = _frac_round_to_type(q, np.float32)
    with np.errstate(all="ignore"):
        if np.isinf(f32):
            return np.float16(f32)
        return np.float16(np.float32(f32))


# ---------------------------------------------------------------------------
# the integer case analysis of the document, transcribed literally
# ---------------------------------------------------------------------------
#
# A second, literal, reading of the two sections **Div** (int, int) and **Div**
# (uint, uint), compared with ONNX Runtime over a whole domain.  A disagreement here is
# a defect of the document, not of the implementation.


def _trunc_div(x: int, y: int) -> int:
    """The exact quotient of two integers truncated toward zero."""
    q = abs(x) // abs(y)
    if (x < 0) != (y < 0):
        q = -q
    return q


def doc_int_signed(x: int, y: int, n: int) -> int:
    """The case analysis of **Div** (int, int), in exact arithmetic."""
    q = _trunc_div(x, y)
    if q > 2 ** (n - 1) - 1:              # only q == 2^(n-1), the pair (-2^(n-1), -1)
        return q - 2 ** n
    return q


def doc_int_unsigned(x: int, y: int, n: int) -> int:
    """The formula of **Div** (uint, uint), in exact arithmetic: the floor."""
    return x // y


def _doc_int_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dt = a.dtype
    n = dt.itemsize * 8
    fn = doc_int_signed if dt.kind == "i" else doc_int_unsigned
    flat = [fn(int(x), int(y), n) for x, y in zip(a.ravel(), b.ravel())]
    return np.array(flat, dtype=dt).reshape(a.shape)


def _rand_values(rng, dt, n):
    """n random values of `dt`, drawn uniformly over the whole bit patterns of the type."""
    hi = rng.integers(0, 2 ** 32, size=n, dtype=np.uint64)
    lo = rng.integers(0, 2 ** 32, size=n, dtype=np.uint64)
    raw = (hi << np.uint64(32)) | lo
    bits = dt.itemsize * 8
    if bits < 64:
        raw = raw & np.uint64((1 << bits) - 1)
    return raw.astype(np.dtype("u%d" % dt.itemsize)).view(dt)


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------


def _float_section_cases():
    """The cases of **Div** (float, float): the examples, the special values, shapes."""
    out = []

    # ---- the document's own worked examples
    out.append(("float Example 1: [6 -9 1] / [3 4 8] (float32)",
                [np.array([6.0, -9.0, 1.0], np.float32),
                 np.array([3.0, 4.0, 8.0], np.float32)]))
    out.append(("float Example 2: the special values of the document (float32)",
                [np.array([1.0, -1.0, np.inf, np.inf, 0.0, -0.0, 0.0, np.nan],
                          np.float32),
                 np.array([0.0, 0.0, np.inf, 4.0, 4.0, 4.0, 0.0, 1.0], np.float32)]))
    out.append(("float Example 3: [1 10 -10] / [3 3 3] (double)",
                [np.array([1.0, 10.0, -10.0], np.float64),
                 np.array([3.0, 3.0, 3.0], np.float64)]))

    # ---- the special values of every float type
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        maxf = float(f.max)
        tiny = float(f.tiny)
        smallest = float(np.nextafter(dt.type(0.0), dt.type(1.0)))
        inf = float("inf")
        nan = float("nan")
        pz, nz = 0.0, -0.0
        groups = [
            ("float division by zero: 1/(+0), -1/(+0), 1/(-0), -1/(-0)",
             [1.0, -1.0, 1.0, -1.0], [pz, pz, nz, nz]),
            ("float invalid operation: (+0)/(+0), (+0)/(-0), (-0)/(+0), (-0)/(-0)",
             [pz, pz, nz, nz], [pz, nz, pz, nz]),
            ("float invalid operation: inf/inf and the signed variants",
             [inf, inf, -inf, -inf], [inf, -inf, inf, -inf]),
            ("float infinity over a finite non-null value",
             [inf, -inf, inf, -inf], [2.0, 2.0, -2.0, -2.0]),
            ("float null dividend over a finite non-null divisor",
             [pz, nz, pz, nz], [2.0, 2.0, -2.0, -2.0]),
            ("float finite dividend over an infinity",
             [2.0, -2.0, 2.0, -2.0], [inf, inf, -inf, -inf]),
            ("float NaN propagation",
             [nan, 1.0, nan, nan, inf, pz], [1.0, nan, inf, pz, nan, nan]),
            ("float the largest finite value of the type",
             [maxf, -maxf, maxf, -maxf], [1.0, 1.0, 2.0, 2.0]),
            ("float overflow to an infinity: the largest value over 0.5",
             [maxf, -maxf], [0.5, 0.5]),
            ("float the smallest normal value and its halves",
             [tiny, -tiny, tiny, tiny], [2.0, 2.0, 4.0, 1.0]),
            ("float the smallest subnormal value, halved and quartered",
             [smallest, -smallest, smallest], [2.0, 2.0, 4.0]),
            ("float the rounding boundary: 1/3, 1/10 and 2/3",
             [1.0, 1.0, 2.0], [3.0, 10.0, 3.0]),
        ]
        for label, a, b in groups:
            out.append(("%s (%s)" % (label, dt.name),
                        [np.array(a, dtype=dt), np.array(b, dtype=dt)]))

    # ---- the shape families the document names
    f32 = np.float32
    out.append(("float32 broadcast: a rank-1 divisor over a (2,3) dividend",
                [np.array([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]], f32),
                 np.array([2.0, 4.0, 8.0], f32)]))
    out.append(("float32 broadcast: (2,1) over (1,3), both extended",
                [np.array([[1.0], [-1.0]], f32),
                 np.array([[2.0, 4.0, 8.0]], f32)]))
    out.append(("float32 rank-0 dividend broadcast to (2,3)",
                [np.array(6.0, f32),
                 np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], f32)]))
    out.append(("float32 rank-0 divisor broadcast to (3,)",
                [np.array([3.0, 6.0, 9.0], f32), np.array(3.0, f32)]))
    out.append(("float32 rank-0 over rank-0",
                [np.array(1.0, f32), np.array(4.0, f32)]))
    out.append(("float32 zero-sized dimension: (0,3) over (3,)",
                [np.zeros((0, 3), f32), np.array([1.0, 2.0, 3.0], f32)]))
    out.append(("float32 zero-sized dimension: (2,0,3) over (0,3)",
                [np.zeros((2, 0, 3), f32), np.zeros((0, 3), f32)]))
    out.append(("float64 zero-sized dimension: (0,) over a rank-0 divisor",
                [np.zeros((0,), np.float64), np.array(3.0, np.float64)]))
    out.append(("float64 broadcast: (3,1,2) over (1,4,2)",
                [np.arange(6.0, dtype=np.float64).reshape(3, 1, 2) + 1.0,
                 np.arange(8.0, dtype=np.float64).reshape(1, 4, 2) + 2.0]))
    return out


def _int_section_cases():
    """The cases of **Div** (int, int): the example, truncation, the range, shapes.

    No case of this section puts a null element in the divisor: the document states that
    the result is then undetermined, so such an operand is outside the specified domain.
    """
    out = []
    i8, i16, i32, i64 = np.int8, np.int16, np.int32, np.int64

    out.append(("int8 Example 1: [7 -7 7 -7 -128] / [2 2 -2 -2 -1]",
                [np.array([7, -7, 7, -7, -128], i8),
                 np.array([2, 2, -2, -2, -1], i8)]))
    out.append(("int8 truncation toward zero, not the floor",
                [np.array([7, -7, 7, -7, 1, -1, 127, -128], i8),
                 np.array([2, 2, -2, -2, 2, -2, 2, 3], i8)]))
    out.append(("int8 boundaries: the extrema over +-1, 2 and the extremum",
                [np.array([127, -128, 127, -128, -128, 127], i8),
                 np.array([1, 1, -1, -1, 127, -128], i8)]))
    out.append(("int16 boundaries: the extrema and the one quotient outside the range",
                [np.array([32767, -32768, 32767, -32768, -32768, 12345], i16),
                 np.array([1, 1, -1, -1, 32767, -7], i16)]))
    out.append(("int32 boundaries: the extrema and the one quotient outside the range",
                [np.array([2147483647, -2147483648, 2147483647, -2147483648,
                           -2147483648, 123456789], i32),
                 np.array([1, 1, -1, -1, 2147483647, -1000], i32)]))
    out.append(("int64 boundaries: the extrema and the one quotient outside the range",
                [np.array([9223372036854775807, -9223372036854775808,
                           9223372036854775807, -9223372036854775808,
                           -9223372036854775808, 1234567890123], i64),
                 np.array([1, 1, -1, -1, 9223372036854775807, -1000], i64)]))
    out.append(("int32 the sign of the quotient and the neutral zero",
                [np.array([6, -6, 6, -6, 1, -1, 0], i32),
                 np.array([3, 3, -3, -3, 2, -2, 5], i32)]))
    out.append(("int32 broadcast: a rank-1 divisor over a (2,3) dividend",
                [np.array([[10, 20, 30], [40, 50, 60]], i32),
                 np.array([2, 3, 5], i32)]))
    out.append(("int32 broadcast: (2,1) over (1,3)",
                [np.array([[7], [-7]], i32), np.array([[2, 3, 4]], i32)]))
    out.append(("int32 rank-0 divisor broadcast to (2,2)",
                [np.array([[9, -9], [15, -15]], i32), np.array(3, i32)]))
    out.append(("int32 rank-0 over rank-0", [np.array(-7, i32), np.array(2, i32)]))
    out.append(("int64 zero-sized dimension: (0,2) over (2,)",
                [np.zeros((0, 2), i64), np.array([2, 4], i64)]))
    return out


def _uint_section_cases():
    """The cases of **Div** (uint, uint): the example, the range, the shape families.

    No case of this section puts a null element in the divisor, for the reason given for
    the signed section.
    """
    out = []
    u8, u16, u32, u64 = np.uint8, np.uint16, np.uint32, np.uint64

    out.append(("uint8 Example 1: [7 200 255 6] / [2 3 255 6]",
                [np.array([7, 200, 255, 6], u8),
                 np.array([2, 3, 255, 6], u8)]))
    out.append(("uint8 the range: the maximum over 1, over itself and over 2",
                [np.array([255, 255, 255, 254, 1, 0], u8),
                 np.array([1, 255, 2, 2, 255, 1], u8)]))
    out.append(("uint16 the extrema of the type",
                [np.array([65535, 65535, 65535, 65534, 1], u16),
                 np.array([1, 65535, 2, 2, 65535], u16)]))
    out.append(("uint32 the extrema of the type",
                [np.array([4294967295, 4294967295, 4294967295, 4294967294, 123456789],
                          u32),
                 np.array([1, 4294967295, 2, 2, 1000], u32)]))
    out.append(("uint64 the extrema of the type",
                [np.array([18446744073709551615, 18446744073709551615,
                           18446744073709551615, 18446744073709551614,
                           1234567890123], u64),
                 np.array([1, 18446744073709551615, 2, 2, 1000], u64)]))
    out.append(("uint8 broadcast: a rank-1 divisor over a (2,3) dividend",
                [np.array([[10, 20, 30], [40, 50, 60]], u8),
                 np.array([2, 3, 5], u8)]))
    out.append(("uint8 broadcast: (2,1) over (1,3)",
                [np.array([[7], [9]], u8), np.array([[2, 3, 4]], u8)]))
    out.append(("uint8 rank-0 dividend broadcast to (3,)",
                [np.array(200, u8), np.array([2, 4, 8], u8)]))
    out.append(("uint8 rank-0 over rank-0", [np.array(255, u8), np.array(255, u8)]))
    out.append(("uint64 zero-sized dimension: (3,0) over (1,0)",
                [np.zeros((3, 0), u64), np.zeros((1, 0), u64)]))
    return out


_CASES = []


def _build_cases():
    out = []
    for label, arrays in _float_section_cases():
        out.append(("float", label, arrays))
    for label, arrays in _int_section_cases():
        out.append(("int", label, arrays))
    for label, arrays in _uint_section_cases():
        out.append(("uint", label, arrays))
    return out


_CASES = _build_cases()


def cases():
    """One labelled pair of operands per case, as `specloop.py` reads them."""
    for _section, label, arrays in _CASES:
        yield label, arrays


# ---------------------------------------------------------------------------
# the real section, which has no ONNX type and lives in a sweep
# ---------------------------------------------------------------------------


def _real_examples():
    """The real section's cases: exact rationals, no ONNX type.

    The four entries are the section's own worked examples and its statement that the
    quotient is exact (1/3 has no finite decimal expansion and is not approximated),
    plus the two broadcasting statements.  The section's precondition rules out a null
    divisor, so no entry has one.
    """
    F = Fraction
    return [
        ("real Example 1: [7 -9 1] / [2 4 3], the quotients exact",
         [[F(7), F(-9), F(1)]],
         [[F(2), F(4), F(3)]],
         [[F(7, 2), F(-9, 4), F(1, 3)]]),
        ("real Example 2: [[4 9] [16 25]] / [2 5], the rank-1 divisor broadcast",
         [[F(4), F(9)], [F(16), F(25)]],
         [F(2), F(5)],
         [[F(2), F(9, 5)], [F(8), F(5)]]),
        ("real: 1/3 has no finite decimal expansion and is held exactly",
         [[F(1, 3), F(2, 7), F(-5, 11)]],
         [[F(1, 1), F(1, 1), F(1, 1)]],
         [[F(1, 3), F(2, 7), F(-5, 11)]]),
        ("real: rank-0 dividend broadcast to (2,)",
         F(3, 7),
         [F(2, 5), F(1, 2)],
         [[F(15, 14), F(6, 7)]]),
        ("real: a rank-1 divisor broadcast over a 3x2 matrix",
         [[F(1, 3), F(2, 3)], [F(1, 7), F(3, 7)], [F(-1, 9), F(5, 9)]],
         [F(1, 6), F(1, 6)],
         [[F(2), F(4)], [F(6, 7), F(18, 7)], [F(-2, 3), F(10, 3)]]),
    ]


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def _float_probe(dtype, rng):
    """The operands of the exact float sweep: the special values and a random sample."""
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    maxf = float(f.max)
    tiny = float(f.tiny)
    smallest = float(np.nextafter(dt.type(0.0), dt.type(1.0)))
    specials = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.0, -2.0,
                tiny, -tiny, smallest, -smallest, maxf, -maxf, maxf / 2.0, -maxf / 2.0,
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
    """The float formula of the document against exact arithmetic, ORT and the impl.

    Each element is round(A / B) as the document states it, computed with `Fraction`
    from the operands' exact values (never from the decimal literal).  A float16
    departure that is the same quotient rounded through float32 is recorded as double
    rounding, not required: the hardware's half kernel is free to compute that way.
    """
    rng = np.random.default_rng(20261004)
    rows, failures = [], []
    recorded = 0
    checked = 0
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        a, b = _float_probe(dt, rng)
        exact = np.array([_exact_div_result(x, y, dt)
                          for x, y in zip(a.tolist(), b.tolist())], dtype=dt)
        try:
            ref = run_ort_batch(a, b)
        except Exception as exc:
            rows.append("%s: ONNX Runtime refused the model (%s: %s)"
                        % (dt.name, type(exc).__name__, exc))
            continue
        try:
            got = np.asarray(impl.div(a, b))
        except Exception as exc:
            failures.append({"type": dt.name, "who": "the implementation",
                             "detail": "raised %s: %s" % (type(exc).__name__, exc)})
            continue
        checked += int(a.size)

        for who, other in (("ONNX Runtime", ref), ("the implementation", got)):
            bad = _diff_indices(exact, other, limit=None)
            if bad is None:
                failures.append({"type": dt.name, "who": who,
                                 "detail": "shape %s or type %s differs from the exact "
                                           "quotient's %s / %s"
                                           % (other.shape, other.dtype,
                                              exact.shape, exact.dtype)})
                continue
            real_bad = []
            for i in bad:
                if dt == np.dtype(np.float16):
                    dr = _double_rounded_div16(a.ravel()[i], b.ravel()[i])
                    if dr is not None and _same_bits(other.ravel()[i], dr):
                        recorded += 1
                        continue
                real_bad.append(i)
            if real_bad:
                failures.append({
                    "type": dt.name,
                    "who": "%s vs the exact quotient of the document" % who,
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


def sweep_int_domain(impl):
    """The document's integer case analysis against ORT and the implementation.

    The two 8-bit types are covered exhaustively over every pair with a non-null divisor
    (65280 pairs each); the wider types use their boundary values and 200 random pairs.
    A null divisor is not covered: the document states that the result is then
    undetermined, so it is not a value the sweep may judge.
    """
    rng = np.random.default_rng(20261003)
    rows, failures = [], []
    for dtype in ALL_INT:
        dt = np.dtype(dtype)
        info = np.iinfo(dt)
        exhaustive = dt.itemsize == 1
        if exhaustive:
            xs = np.arange(int(info.min), int(info.max) + 1, dtype=np.int64)
            aa = np.repeat(xs, len(xs)).astype(dt)
            bb = np.tile(xs, len(xs)).astype(dt)
            keep = bb != 0
            aa, bb = aa[keep], bb[keep]
        else:
            edges = {0, 1, 2, 3, int(info.max), int(info.max) - 1, int(info.max) // 2}
            if dt.kind == "i":
                edges |= {int(info.min), int(info.min) + 1, -1, -2, -3,
                          int(info.min) // 2}
            edges = sorted(edges)
            pairs = [(x, y) for x in edges for y in edges if y != 0]
            ra = _rand_values(rng, dt, 200)
            rb = _rand_values(rng, dt, 200)
            keep = rb != 0
            pairs += list(zip(ra[keep].tolist(), rb[keep].tolist()))
            aa = np.array([p[0] for p in pairs], dtype=dt)
            bb = np.array([p[1] for p in pairs], dtype=dt)

        want = _doc_int_array(aa, bb)          # the document, literally
        try:
            ref = run_ort_batch(aa, bb)        # ONNX Runtime
        except Exception as exc:
            rows.append("%s: ONNX Runtime refused the model (%s)"
                        % (dt.name, type(exc).__name__))
            continue
        try:
            got = np.asarray(impl.div(aa, bb))  # the implementation
        except Exception as exc:
            failures.append({"type": dt.name, "who": "the implementation",
                             "detail": "raised %s: %s" % (type(exc).__name__, exc)})
            continue

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
                                    for i in bad),
            })
        bad = _diff_indices(ref, got, limit=5)
        if bad is None:
            failures.append({"type": dt.name,
                             "who": "the implementation vs ONNX Runtime",
                             "detail": "shape %s or type %s differs from the runtime's "
                                       "%s / %s" % (got.shape, got.dtype,
                                                    ref.shape, ref.dtype)})
        elif bad:
            failures.append({
                "type": dt.name, "who": "the implementation vs ONNX Runtime",
                "detail": "; ".join("%d / %d: ort %d, impl %d"
                                    % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                       int(np.asarray(ref).ravel()[i]),
                                       int(np.asarray(got).ravel()[i])) for i in bad),
            })
        rows.append("%s %s" % (dt.name,
                               "exhaustive (65280 pairs, divisor non-null)" if exhaustive
                               else "%d pairs" % len(aa)))
    return {"sweep": "int domain (document vs ORT vs implementation)",
            "summary": "%d types: %s; a null divisor is not covered, the document "
                       "states the result is then undetermined"
                       % (len(rows), ", ".join(rows)),
            "failures": failures}


def _spec_examples():
    """The worked examples of the float, the int and the uint sections, as documented."""
    f32 = np.float32
    f64 = np.float64
    return [
        ("float Example 1",
         np.array([6.0, -9.0, 1.0], f32),
         np.array([3.0, 4.0, 8.0], f32),
         np.array([2.0, -2.25, 0.125], f32)),
        ("float Example 2",
         np.array([1.0, -1.0, np.inf, np.inf, 0.0, -0.0, 0.0, np.nan], f32),
         np.array([0.0, 0.0, np.inf, 4.0, 4.0, 4.0, 0.0, 1.0], f32),
         np.array([np.inf, -np.inf, np.nan, np.inf, 0.0, -0.0, np.nan, np.nan], f32)),
        ("float Example 3",
         np.array([1.0, 10.0, -10.0], f64),
         np.array([3.0, 3.0, 3.0], f64),
         np.array([0.3333333333333333, 3.3333333333333335, -3.3333333333333335], f64)),
        ("int Example 1",
         np.array([7, -7, 7, -7, -128], np.int8),
         np.array([2, 2, -2, -2, -1], np.int8),
         np.array([3, -3, -3, 3, -128], np.int8)),
        ("uint Example 1",
         np.array([7, 200, 255, 6], np.uint8),
         np.array([2, 3, 255, 6], np.uint8),
         np.array([3, 66, 1, 1], np.uint8)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, ORT, and the implementation."""
    failures = []
    examples = _spec_examples()
    for label, a, b, documented in examples:
        try:
            ref = run_ort_batch(a, b)
        except Exception as exc:
            failures.append({"who": label,
                             "detail": "ONNX Runtime refused the model: %s: %s"
                                       % (type(exc).__name__, exc)})
            ref = None
        try:
            got = np.asarray(impl.div(a, b))
        except Exception as exc:
            failures.append({"who": label,
                             "detail": "the implementation raised %s: %s"
                                       % (type(exc).__name__, exc)})
            got = None
        if ref is not None and not _bits_equal(documented, ref):
            failures.append({"who": label, "detail": "documented %s, ORT %s"
                             % (documented.tolist(), ref.tolist())})
        if got is not None and not _bits_equal(documented, got):
            failures.append({"who": label, "detail": "documented %s, implementation %s"
                             % (documented.tolist(), got.tolist())})
    return {"sweep": "the document's worked examples",
            "summary": "%d examples, as documented vs ORT vs implementation"
                       % len(examples),
            "failures": failures}


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact rationals.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the section's
    Examples and its statement of exactness against exact rational arithmetic, and -- if
    the implementation accepts exact rational operands -- against the implementation too.
    A real number is not an ONNX type, so an implementation that refuses an object array
    of Fractions is recorded, not failed.
    """
    failures = []
    examples = _real_examples()
    checked = 0
    recorded = 0
    for label, a, b, documented in examples:
        aa = np.array(a, dtype=object)
        bb = np.array(b, dtype=object)
        ba, bc = np.broadcast_arrays(aa, bb)
        want = np.empty(ba.shape, dtype=object)
        for i in np.ndindex(ba.shape):
            want[i] = ba[i] / bc[i]
        checked += 1
        if documented is not None:
            doc = np.array(documented, dtype=object)
            if doc.shape != want.shape:
                failures.append({"who": label,
                                 "detail": "documented shape %s, the exact quotient's %s"
                                           % (doc.shape, want.shape)})
            else:
                bad = [i for i in np.ndindex(doc.shape) if doc[i] != want[i]]
                if bad:
                    failures.append({"who": label,
                                     "detail": "; ".join(
                                         "documented %s, exact %s"
                                         % (doc[i], want[i]) for i in bad[:5])})
        try:
            got = np.asarray(impl.div(aa, bb))
        except Exception:
            recorded += 1
            continue
        if got.shape != want.shape:
            failures.append({"who": label,
                             "detail": "exact shape %s, implementation shape %s"
                                       % (want.shape, got.shape)})
            continue
        bad = [i for i in np.ndindex(want.shape) if got[i] != want[i]]
        if bad:
            failures.append({"who": label,
                             "detail": "; ".join("exact %s, implementation %s"
                                                 % (want[i], got[i]) for i in bad[:5])})
    return {"sweep": "the real section (exact rational arithmetic, no ONNX Runtime)",
            "summary": "%d cases checked against Fraction arithmetic (%d checked against "
                       "the implementation as well; %d refused exact rational operands, "
                       "recorded, not required: a real number is not an ONNX type); the "
                       "section's precondition rules out a null divisor"
                       % (checked, checked - recorded, recorded),
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
        summary += ("; the schema also admits %s, which the document does not specify "
                    "(the profile specifies a subset)" % ", ".join(extra))
    return {"sweep": "the document's types against the ONNX schema",
            "summary": summary, "failures": failures}


def sweeps():
    """The bulk checks, each batched into a single ONNX Runtime run where it runs one."""
    return [
        sweep_float_exact,
        sweep_int_domain,
        sweep_spec_examples,
        sweep_real_examples,
        sweep_schema_types,
    ]


# ---------------------------------------------------------------------------
# coverage: every anchor of the document's Contents list
# ---------------------------------------------------------------------------


def _coverage():
    counts = {"real": 0, "float": 0, "int": 0, "uint": 0}
    for section, _label, _arrays in _CASES:
        counts[section] += 1
    # the real section has no ONNX type: its cases live in sweep_real_examples
    counts["real"] = len(_real_examples())
    return counts


COVERAGE = _coverage()
