"""The cases with which the **Atan** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled operand per case.  Each case goes through a one-node
                     Atan model (opset 14) and through the implementation, and the two
                     results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the domain of the float
                     types, the worked examples of every section of the document, and
                     the real section, which has no ONNX type and is checked against
                     exact and high-precision arithmetic instead.

Every case of every section's formula is meant to appear here, and the case set is part
of the loop: when a section is added to the document, its cases are added here before
the next run.

The reference is not exact.  The CPU runtime's `Atan` for `float32` returns
`0.7853981` for `1.0`, one unit in the last place below the correctly rounded
`0.7853982`, while the document specifies `round(arctan(x))` with roundTiesToEven, so
the implementation's value is the one the document requires and the reference is the
one that is wrong.  A bit-for-bit comparison against such a reference tests the
reference's accuracy, not the implementation's compliance.  The float comparisons here
therefore allow a few ulps, and the exact arithmetic of the real section is the
authority on the correctly rounded value.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Atan"
OPSET = 14

ALL_FLOAT = [np.float16, np.float32, np.float64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}
IR_VERSION = 10

# The runtime of this repository has a float16 and a float kernel for Atan and no
# double kernel; a refusal is a property of the build, not a defect of the document.
NO_DOUBLE_KERNEL = True

# The reference is allowed to be a few ulps away from the correctly rounded value:
# the document specifies round(arctan(x)), and a kernel is free to be less accurate.
ULP_TOLERANCE = 4


# ---------------------------------------------------------------------------
# comparison: exact for the special values, a few ulps for the finite ones
# ---------------------------------------------------------------------------


def _ulp_distance(want: np.ndarray, got: np.ndarray) -> np.ndarray:
    """Distance in ulps between two finite float arrays of the same dtype.

    The bit patterns of two finite values of the same sign are ordered, so the
    distance is the difference of the patterns; across zero the two patterns are
    mapped to a monotone integer first.
    """
    uint = UINT_OF_WIDTH[np.dtype(want.dtype).itemsize]
    bw = np.ascontiguousarray(want).view(uint).astype(np.int64)
    bg = np.ascontiguousarray(got).view(uint).astype(np.int64)
    sign = np.array(1 << (np.dtype(want.dtype).itemsize * 8 - 1), dtype=np.int64)
    # map to a monotone order: negative patterns are reflected below zero
    ow = np.where(bw & sign, sign - bw, bw)
    og = np.where(bg & sign, sign - bg, bg)
    return np.abs(ow - og)


def _float_close(want: np.ndarray, got: np.ndarray) -> bool:
    """The float comparison: exact for the special values, a few ulps for the rest.

    The signed zeros, the infinities and the NaN-ness are compared exactly; two
    finite values are equal when they differ by at most ``ULP_TOLERANCE`` ulps of
    the type.
    """
    if want.shape != got.shape or want.dtype != got.dtype:
        return False
    want = np.asarray(want)
    got = np.asarray(got)
    nan_w, nan_g = np.isnan(want), np.isnan(got)
    if not np.array_equal(nan_w, nan_g):
        return False
    inf_w, inf_g = np.isinf(want), np.isinf(got)
    if not np.array_equal(inf_w, inf_g):
        return False
    if np.any(inf_w):
        # an infinity must match exactly, sign included
        if not np.array_equal(want[inf_w], got[inf_w]):
            return False
    zero_w, zero_g = (want == 0), (got == 0)
    if not np.array_equal(zero_w, zero_g):
        return False
    if np.any(zero_w):
        # a signed zero must match exactly, sign included
        if not np.array_equal(np.signbit(want[zero_w]), np.signbit(got[zero_w])):
            return False
    finite = ~(nan_w | inf_w | zero_w)
    if np.any(finite):
        if not np.all(_ulp_distance(want[finite], got[finite]) <= ULP_TOLERANCE):
            return False
    return True


def _bits_equal(want: np.ndarray, got: np.ndarray) -> bool:
    """The comparison used by the sweeps: a few ulps for the float types."""
    if want.dtype.kind == "f":
        return _float_close(want, got)
    if want.shape != got.shape or want.dtype != got.dtype:
        return False
    return bool(np.all(want == got))


def _first_diffs(want: np.ndarray, got: np.ndarray, n: int = 5):
    if want.shape != got.shape or want.dtype != got.dtype:
        return "shape %s vs %s" % (want.shape, got.shape)
    if want.dtype.kind == "f":
        mask = ~_float_close(want, got) * np.ones(want.shape, dtype=bool)
        # per element, for the report
        flat_w = np.asarray(want).ravel()
        flat_g = np.asarray(got).ravel()
        bad = [i for i in range(flat_w.size)
               if not _float_close(flat_w[i:i + 1], flat_g[i:i + 1])]
    else:
        bad = list(np.nonzero(~(np.asarray(want) == np.asarray(got)).ravel())[0])
    return "; ".join("x=%r want=%r got=%r"
                     % (np.asarray(want).ravel()[i], np.asarray(want).ravel()[i],
                        np.asarray(got).ravel()[i])
                     for i in bad[:n])


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(x: np.ndarray) -> np.ndarray:
    proto = helper.np_dtype_to_tensor_dtype(x.dtype)
    vi_x = helper.make_tensor_value_info("X", proto, list(x.shape))
    vi_y = helper.make_tensor_value_info("Y", proto, None)
    node = helper.make_node(OP, ["X"], ["Y"], name="atan0")
    graph = helper.make_graph([node], "atan_graph", [vi_x], [vi_y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    (y,) = sess.run(["Y"], {"X": x})
    return y


# ---------------------------------------------------------------------------
# the real-number section, in exact and high-precision arithmetic
# ---------------------------------------------------------------------------
#
# The real section has no ONNX type and ONNX Runtime cannot be asked about it.  It is
# checked against exact rational arithmetic (the examples, whose arctangents are the
# rational multiples of pi stated by the document) and against a Taylor series summed
# in Decimal (the transcendental values), computed here and not by the platform's
# library.


def _atan_series(x: Decimal, prec: int) -> Decimal:
    """arctan(x) for |x| <= 1, by the Gregory series, in Decimal.

    arctan(x) = x - x^3/3 + x^5/5 - ...  The series converges slowly near 1, so the
    argument is halved by the identity arctan(x) = 2 arctan(x / (1 + sqrt(1 + x^2)))
    until it is small, and the result is doubled back.
    """
    getcontext().prec = prec + 20
    x = Decimal(x)
    sign = 1
    if x < 0:
        sign, x = -1, -x
    doublings = 0
    while x > Decimal("0.05"):
        x = x / (1 + (1 + x * x).sqrt())
        doublings += 1
    term = x
    total = x
    x2 = x * x
    k = 1
    while True:
        term = -term * x2
        k += 2
        add = term / k
        total += add
        if abs(add) < Decimal(10) ** (-(prec + 10)):
            break
    for _ in range(doublings):
        total *= 2
    getcontext().prec = prec
    return sign * total


def _pi(prec: int) -> Decimal:
    """pi by the Machin formula, in Decimal, independent of the platform's library."""
    getcontext().prec = prec + 20
    pi = 16 * _atan_series(Decimal(1) / 5, prec + 10) \
        - 4 * _atan_series(Decimal(1) / 239, prec + 10)
    getcontext().prec = prec
    return pi


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_float_domain(impl):
    """The float section against ONNX Runtime, over the values of each type.

    The special values, the boundaries of the type, the subnormal range and a random
    sample are batched into one run per type.  The comparison allows a few ulps: the
    reference is not exact, and the document's round(arctan(x)) is the authority.
    The runtime of this repository has no double kernel for Atan: that type is not
    reference-checked, and the sweep says so rather than raising.
    """
    rng = np.random.default_rng(20261003)
    failures, rows = [], []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        maxf = float(f.max)
        tiny = float(f.tiny)
        smallest = float(np.nextafter(0.0, 1.0, dtype=dt))
        nan, inf = float("nan"), float("inf")
        values = [0.0, -0.0, 1.0, -1.0, inf, -inf, nan,
                  maxf, -maxf, tiny, -tiny, smallest, -smallest,
                  float(np.nextafter(1.0, 2.0, dtype=dt)),
                  float(np.nextafter(1.0, 0.0, dtype=dt)),
                  float(np.nextafter(tiny, 0.0, dtype=dt)),
                  float(np.nextafter(tiny, inf, dtype=dt))]
        # a random sample over the finite range, drawn in the widest float and cast
        for _ in range(200):
            u = rng.uniform(-1.0, 1.0)
            values.append(float(np.float64(u) * np.float64(maxf)))
        x = np.array(values, dtype=dt)
        try:
            ref = run_ort_batch(x)
        except Exception as exc:
            rows.append("%s: not reference-checked, the runtime has no %s kernel (%s)"
                        % (dt.name, dt.name, type(exc).__name__))
            continue
        got = np.asarray(impl.atan(x))
        if not _bits_equal(ref, got):
            failures.append({"type": dt.name, "who": "the implementation vs ONNX Runtime",
                             "detail": _first_diffs(ref, got)})
        rows.append("%s %d values" % (dt.name, len(values)))
    return {"sweep": "float domain (ORT vs implementation, %d ulps)" % ULP_TOLERANCE,
            "summary": "%s" % ", ".join(rows),
            "failures": failures}


def sweep_special_values(impl):
    """The special values of the float section, against ONNX Runtime, per type.

    The signed zeros, the infinities, NaN, the subnormal range and the rounding
    boundary between two values and the tie above it.  The special values are compared
    exactly; the finite ones allow a few ulps.
    """
    failures, rows = [], []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        maxf = float(f.max)
        tiny = float(f.tiny)
        smallest = float(np.nextafter(0.0, 1.0, dtype=dt))
        nan, inf = float("nan"), float("inf")
        values = [0.0, -0.0, inf, -inf, nan, maxf, -maxf, tiny, -tiny,
                  smallest, -smallest,
                  float(np.nextafter(1.0, 2.0, dtype=dt)),
                  float(np.nextafter(1.0, 0.0, dtype=dt)),
                  float(np.nextafter(tiny, 0.0, dtype=dt)),
                  float(np.nextafter(tiny, inf, dtype=dt))]
        x = np.array(values, dtype=dt)
        try:
            ref = run_ort_batch(x)
        except Exception as exc:
            rows.append("%s: not reference-checked, the runtime has no %s kernel (%s)"
                        % (dt.name, dt.name, type(exc).__name__))
            continue
        got = np.asarray(impl.atan(x))
        if not _bits_equal(ref, got):
            failures.append({"type": dt.name, "who": "the implementation vs ONNX Runtime",
                             "detail": _first_diffs(ref, got)})
        rows.append("%s %d values" % (dt.name, len(values)))
    return {"sweep": "special values (ORT vs implementation, %d ulps)" % ULP_TOLERANCE,
            "summary": "%s" % ", ".join(rows),
            "failures": failures}


def _spec_examples():
    """The worked examples of the real and the float sections, as documented."""
    return [
        ("real Example 1",
         np.array([0.0, 1.0, -1.0], dtype=np.float64),
         np.array([0.0, 0.7853981633974483, -0.7853981633974483], dtype=np.float64)),
        ("real Example 2",
         np.array([np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64),
         np.array([1.0471975511965976, 0.5235987755982988], dtype=np.float64)),
        ("float Example 1",
         np.array([0.0, 1.0, -1.0], dtype=np.float64),
         np.array([0.0, 0.7853981633974483, -0.7853981633974483], dtype=np.float64)),
        ("float Example 2",
         np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float64),
         np.array([0.0, -0.0, 1.5707963267948966, -1.5707963267948966, np.nan],
                  dtype=np.float64)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, and as ONNX Runtime does it.

    The documented values are the ones the document states, and the comparison allows
    a few ulps: the last digit of a documented decimal is not a bit-exact requirement,
    and the reference is not exact either.
    """
    failures = []
    for label, x, documented in _spec_examples():
        got = np.asarray(impl.atan(x))
        if not _bits_equal(documented, got):
            failures.append({"who": label, "detail": "documented %s, implementation %s"
                             % (documented.tolist(), got.tolist())})
        try:
            ref = run_ort_batch(x)
        except Exception as exc:
            continue
        if not _bits_equal(documented, ref):
            failures.append({"who": label, "detail": "documented %s, ORT %s"
                             % (documented.tolist(), ref.tolist())})
    return {"sweep": "the document's worked examples (%d ulps)" % ULP_TOLERANCE,
            "summary": "%d examples, as documented vs ORT vs implementation"
                       % len(_spec_examples()),
            "failures": failures}


def sweep_real_section(impl):
    """The real section, which has no ONNX type: exact and high-precision arithmetic.

    The examples of the real section are checked against exact rational arithmetic
    (their arctangents are the rational multiples of pi the document states), and the
    transcendental values are checked against a Taylor series summed in Decimal, which
    is computed here and not by the platform's library.  The comparison allows a few
    ulps of the double type, because the implementation returns a double and the
    series' last digit is not a bit-exact requirement.
    """
    failures = []
    prec = 60
    pi = _pi(prec)
    # the exact rationals of the examples: tan(pi/4) = 1, tan(pi/3) = sqrt(3),
    # tan(pi/6) = 1/sqrt(3); the arctangents are the stated multiples of pi.
    exact = [
        ("real Example 1, x = 0", Fraction(0), Fraction(0)),
        ("real Example 1, x = 1", Fraction(1), Fraction(1, 4)),
        ("real Example 1, x = -1", Fraction(-1), Fraction(-1, 4)),
    ]
    for label, x, multiple in exact:
        want = Decimal(multiple.numerator) / Decimal(multiple.denominator) * pi
        got = np.asarray(impl.atan(np.array([float(x)], dtype=np.float64)))
        if not np.isfinite(got[0]) or abs(Decimal(float(got[0])) - want) > Decimal(10) ** (-15):
            failures.append({"who": label, "detail": "exact %s, implementation %r"
                             % (want, float(got[0]))})
    # the transcendental values of the examples, by the series
    for label, x, multiple in [
        ("real Example 2, x = sqrt(3)", Decimal(3).sqrt(), Fraction(1, 3)),
        ("real Example 2, x = 1/sqrt(3)", 1 / Decimal(3).sqrt(), Fraction(1, 6)),
    ]:
        want = Decimal(multiple.numerator) / Decimal(multiple.denominator) * pi
        got = np.asarray(impl.atan(np.array([float(x)], dtype=np.float64)))
        if not np.isfinite(got[0]) or abs(Decimal(float(got[0])) - want) > Decimal(10) ** (-15):
            failures.append({"who": label, "detail": "exact %s, implementation %r"
                             % (want, float(got[0]))})
    # the oddness of the operator, on the real numbers, in exact arithmetic
    for x in [Fraction(1, 3), Fraction(7, 5), Fraction(-2, 7)]:
        a = np.asarray(impl.atan(np.array([float(x)], dtype=np.float64)))
        b = np.asarray(impl.atan(np.array([float(-x)], dtype=np.float64)))
        if not (a[0] == -b[0]):
            failures.append({"who": "oddness at %s" % x,
                             "detail": "atan(%s) = %r, atan(%s) = %r"
                                       % (x, float(a[0]), -x, float(b[0]))})
    return {"sweep": "real section (exact rationals and Decimal series, no ONNX Runtime)",
            "summary": "%d exact examples, %d transcendental values, oddness"
                       % (len(exact), 2),
            "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_float_domain, sweep_special_values,
            sweep_real_section]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _float_value_cases(dtype):
    """The value families of the float section."""
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    maxf = float(f.max)
    tiny = float(f.tiny)
    smallest = float(np.nextafter(0.0, 1.0, dtype=dt))
    nan, inf = float("nan"), float("inf")
    return [
        ("signed zeros", [0.0, -0.0, 0.0, -0.0]),
        ("infinities", [inf, -inf, 1.0, -1.0]),
        ("nan", [nan, 1.0, inf, -inf]),
        ("exact arctangents", [0.0, 1.0, -1.0, 0.0]),
        ("largest finite", [maxf, -maxf]),
        ("smallest normal", [tiny, -tiny]),
        ("smallest subnormal", [smallest, -smallest]),
        ("subnormal range", [smallest * 2.0, smallest * 3.0, -smallest * 2.0]),
        ("rounding boundary, one ulp above 1.0",
         [float(np.nextafter(1.0, 2.0, dtype=dt))]),
        ("rounding boundary, one ulp below 1.0",
         [float(np.nextafter(1.0, 0.0, dtype=dt))]),
        ("rounding boundary, one ulp above the smallest normal",
         [float(np.nextafter(tiny, inf, dtype=dt))]),
        ("rounding boundary, one ulp below the smallest normal",
         [float(np.nextafter(tiny, 0.0, dtype=dt))]),
        ("not representable (rounded arctangent)", [0.3, 1.0, -0.3]),
    ]


def _bit_nan(dtype, pattern):
    uint = UINT_OF_WIDTH[np.dtype(dtype).itemsize]
    return np.array([pattern], dtype=uint).view(dtype)


NAN_PATTERNS = {2: (0x7E01, 0x7C01, 0x8000),
                4: (0x7FC00001, 0x7F800001, 0x80000000),
                8: (0x7FF8000000000001, 0x7FF0000000000001, 0x8000000000000000)}


def _arr(shape, dtype, start=0.0):
    n = 1
    for d in shape:
        n *= d
    return (np.arange(n, dtype=np.float64) + start).astype(dtype).reshape(shape)


SHAPE_CASES = [
    ("same shape (2,3)", (2, 3)),
    ("rank-0", ()),
    ("rank-1", (5,)),
    ("rank-3", (2, 3, 4)),
    ("zero-sized: (0,)", (0,)),
    ("zero-sized: (0,3)", (0, 3)),
    ("zero-sized: (3,0)", (3, 0)),
    ("zero-sized: (1,0,2)", (1, 0, 2)),
]


def cases():
    """Every individual case: (label, [X])."""
    for dtype in ALL_FLOAT:
        for name, values in _float_value_cases(dtype):
            yield ("%s: %s" % (np.dtype(dtype).name, name),
                   [np.array(values, dtype=dtype)])
        quiet, signalling, sign = NAN_PATTERNS[np.dtype(dtype).itemsize]
        yield ("%s: NaN operand, quiet, payload 1" % np.dtype(dtype).name,
               [_bit_nan(dtype, quiet)])
        yield ("%s: NaN operand, quiet, payload 1, sign set" % np.dtype(dtype).name,
               [_bit_nan(dtype, quiet | sign)])
        yield ("%s: NaN operand, signalling" % np.dtype(dtype).name,
               [_bit_nan(dtype, signalling)])
    for name, shape in SHAPE_CASES:
        for dtype in ALL_FLOAT:
            yield ("%s: %s" % (np.dtype(dtype).name, name),
                   [_arr(shape, dtype, 0.5)])


# ---------------------------------------------------------------------------
# coverage: every anchor of the document's Contents list
# ---------------------------------------------------------------------------

COVERAGE = {
    "real": 0,
    "float": 0,
}

for _label, _arrays in cases():
    _name = _label.split(":")[0]
    if _name in ("float16", "float32", "float64"):
        COVERAGE["float"] += 1
    else:
        COVERAGE["real"] += 1

# the real section has no ONNX type: its cases live in the sweeps, and the count
# records the checks the sweeps make of it.
COVERAGE["real"] = 6
</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>
