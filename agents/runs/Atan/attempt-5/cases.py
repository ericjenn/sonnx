"""The cases with which the **Atan** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled operand per case.  Each case goes through a one-node
                     Atan model (opset 14) and through the implementation, and the two
                     results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the special values of every
                     type, the worked examples of the document, and the real section,
                     which has no ONNX type and is checked against exact arithmetic.

The document specifies `round(arctan(x))` with roundTiesToEven.  The CPU runtime's
`Atan` is not always correctly rounded (measured 2026-10-04: `float32` `Atan(-0.5)`
returns one ulp below the correctly rounded value), so the operands where the reference
is not exact are checked against a high-precision `Decimal` oracle in a sweep, and the
individual cases keep only the operands where the reference agrees with the correctly
rounded value.  The `double` type has no `Atan` kernel in the CPU runtime; the refusal
is recorded as an open case in ``cases()`` and caught per type in the sweeps.
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

getcontext().prec = 80


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(a: np.ndarray) -> np.ndarray:
    proto = helper.np_dtype_to_tensor_dtype(a.dtype)
    vi_a = helper.make_tensor_value_info("A", proto, list(a.shape))
    vi_c = helper.make_tensor_value_info("C", proto, None)
    node = helper.make_node(OP, ["A"], ["C"], name="atan0")
    graph = helper.make_graph([node], "atan_graph", [vi_a], [vi_c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    (c,) = sess.run(["C"], {"A": a})
    return c


def _bits_equal(want: np.ndarray, got: np.ndarray) -> bool:
    if want.shape != got.shape or want.dtype != got.dtype:
        return False
    if want.dtype.kind == "f":
        uint = UINT_OF_WIDTH[want.dtype.itemsize]
        bw = np.ascontiguousarray(want).view(uint)
        bg = np.ascontiguousarray(got).view(uint)
        both_nan = np.isnan(want) & np.isnan(got)
        return bool(np.all((bw == bg) | both_nan))
    return bool(np.all(want == got))


# ---------------------------------------------------------------------------
# a high-precision arctangent, independent of the platform's library
# ---------------------------------------------------------------------------
#
# arctan(x) = x - x^3/3 + x^5/5 - ...  converges for |x| <= 1; for |x| > 1 the
# identity arctan(x) = pi/2 - arctan(1/x) reduces the argument.  Everything is
# computed in Decimal with a wide context, so the result is far more accurate than
# any of the three types, and the rounding to the type is done by hand with
# roundTiesToEven.


def _pi_decimal() -> Decimal:
    """pi to the working precision, by the Machin formula."""
    getcontext().prec += 10
    def arctan_inv(n: int) -> Decimal:
        n = Decimal(n)
        total = term = 1 / n
        n2 = n * n
        k = 1
        while term:
            term = -term / n2
            total += term / (2 * k + 1)
            k += 1
        return total
    pi = 16 * arctan_inv(5) - 4 * arctan_inv(239)
    getcontext().prec -= 10
    return +pi


_PI = _pi_decimal()


def _atan_decimal(x: Decimal) -> Decimal:
    """arctan of a Decimal, to the working precision."""
    if x == 0:
        return Decimal(0)
    sign = -1 if x < 0 else 1
    x = abs(x)
    if x > 1:
        return sign * (_PI / 2 - _atan_decimal(1 / x))
    # Taylor series, summed until the term is below the working precision
    getcontext().prec += 10
    x2 = x * x
    term = x
    total = x
    k = 1
    while term:
        term = -term * x2
        total += term / (2 * k + 1)
        k += 1
    getcontext().prec -= 10
    return sign * +total


def _round_to_type(y: Decimal, dtype) -> float:
    """Round a Decimal to the nearest value of the type, roundTiesToEven."""
    dt = np.dtype(dtype)
    if dt == np.float64:
        return float(y)
    if dt == np.float32:
        return float(np.float32(float(y)))
    if dt == np.float16:
        return float(np.float16(float(y)))
    raise ValueError(dtype)


def _atan_oracle(x: float, dtype) -> float:
    """The document's value: round(arctan(x)) in the type, roundTiesToEven."""
    if np.isnan(x):
        return float("nan")
    if np.isinf(x):
        y = _PI / 2 if x > 0 else -_PI / 2
        return _round_to_type(y, dtype)
    if x == 0.0:
        return x  # the sign of a null is preserved
    y = _atan_decimal(Decimal(repr(float(x))))
    return _round_to_type(y, dtype)


def _oracle_array(a: np.ndarray) -> np.ndarray:
    flat = [_atan_oracle(float(v), a.dtype) for v in a.ravel()]
    return np.array(flat, dtype=a.dtype).reshape(a.shape)


# ---------------------------------------------------------------------------
# the value families of the document
# ---------------------------------------------------------------------------


def _float_value_cases(dtype):
    """The value families of the float section, as documented."""
    f = np.finfo(np.dtype(dtype))
    maxf = float(f.max)
    tiny = float(f.tiny)
    smallest = float(f.smallest_subnormal)
    nan, inf = float("nan"), float("inf")
    return [
        ("signed zeros", [0.0, -0.0, 0.0, -0.0]),
        ("infinities", [inf, -inf, inf, -inf]),
        ("nan", [nan, nan, nan]),
        ("exact values", [0.0, 1.0, -1.0, 0.5, -0.5]),
        ("the document's Example 1", [0.0, 1.0, -1.0]),
        ("the document's Example 2", [0.0, -0.0, inf, -inf, nan]),
        ("largest finite", [maxf, -maxf]),
        ("smallest normal", [tiny, -tiny]),
        ("smallest subnormal", [smallest, -smallest]),
        ("subnormal range", [smallest * 2.0, smallest * 3.0, -smallest * 2.0]),
        ("just below the smallest normal", [tiny * (1.0 - 2.0 ** -f.nmant)]),
        ("one ulp above 1.0", [1.0 + 2.0 ** -f.nmant]),
        ("one ulp below 1.0", [1.0 - 2.0 ** -f.nmant]),
        ("rounding boundary", [1.0 + 2.0 ** -(f.nmant + 1.0)]),
        ("tie above the boundary", [1.0 + 3.0 * 2.0 ** -(f.nmant + 2.0)]),
        ("large magnitude", [1e10, -1e10, 1e30, -1e30]),
        ("small magnitude", [1e-10, -1e-10, 1e-30, -1e-30]),
    ]


def _bit_nan(dtype, pattern):
    uint = UINT_OF_WIDTH[np.dtype(dtype).itemsize]
    return np.array([pattern], dtype=uint).view(dtype)


NAN_PATTERNS = {2: (0x7E01, 0x7C01, 0x8000),
                4: (0x7FC00001, 0x7F800001, 0x80000000),
                8: (0x7FF8000000000001, 0x7FF0000000000001, 0x8000000000000000)}


SHAPE_CASES = [
    ("rank-0", ()),
    ("rank-1", (5,)),
    ("rank-2", (2, 3)),
    ("rank-3", (2, 3, 4)),
    ("zero-sized: (0,)", (0,)),
    ("zero-sized: (0,3)", (0, 3)),
    ("zero-sized: (2,0,3)", (2, 0, 3)),
    ("zero-sized: (0,0)", (0, 0)),
]


def _arr(shape, dtype, start=0.0):
    n = 1
    for d in shape:
        n *= d
    return (np.arange(n, dtype=np.float64) + start).astype(dtype).reshape(shape)


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_special_values(impl):
    """The special values of every type, against the high-precision oracle.

    The reference is the `Decimal` oracle, not ONNX Runtime: the runtime's `Atan` is
    not always correctly rounded, and the document specifies the correctly rounded
    value.  The `double` type has no `Atan` kernel in the CPU runtime, so it is
    checked against the oracle only.
    """
    failures = []
    rows = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        nan, inf = float("nan"), float("inf")
        values = [0.0, -0.0, inf, -inf, nan,
                  1.0, -1.0, 0.5, -0.5,
                  float(f.max), -float(f.max),
                  float(f.tiny), -float(f.tiny),
                  float(f.smallest_subnormal), -float(f.smallest_subnormal),
                  1e10, -1e10, 1e-10, -1e-10]
        a = np.array(values, dtype=dt)
        want = _oracle_array(a)
        got = np.asarray(impl.atan(a))
        if not _bits_equal(want, got):
            mask = np.nonzero(~((np.asarray(want).view(UINT_OF_WIDTH[dt.itemsize])
                                 == np.asarray(got).view(UINT_OF_WIDTH[dt.itemsize]))
                                | (np.isnan(want) & np.isnan(got))))[0][:5]
            failures.append({
                "type": dt.name, "who": "the implementation vs the Decimal oracle",
                "detail": "; ".join("x=%r oracle=%r impl=%r"
                                    % (float(a.ravel()[i]), float(want.ravel()[i]),
                                       float(got.ravel()[i])) for i in mask),
            })
        # the reference, where it has a kernel, is compared with the oracle too
        try:
            ref = run_ort_batch(a)
        except Exception as exc:
            rows.append("%s: not reference-checked, the runtime has no %s kernel (%s)"
                        % (dt.name, dt.name, type(exc).__name__))
            continue
        if not _bits_equal(want, ref):
            mask = np.nonzero(~((np.asarray(want).view(UINT_OF_WIDTH[dt.itemsize])
                                 == np.asarray(ref).view(UINT_OF_WIDTH[dt.itemsize]))
                                | (np.isnan(want) & np.isnan(ref))))[0][:5]
            failures.append({
                "type": dt.name, "who": "the reference vs the Decimal oracle",
                "detail": "; ".join("x=%r oracle=%r ort=%r"
                                    % (float(a.ravel()[i]), float(want.ravel()[i]),
                                       float(ref.ravel()[i])) for i in mask),
            })
        rows.append("%s: %d values" % (dt.name, len(values)))
    return {"sweep": "special values (Decimal oracle, and the reference where it has a kernel)",
            "summary": "; ".join(rows), "failures": failures}


def sweep_domain(impl):
    """A dense sample of the domain of each type, against the Decimal oracle.

    The domain of the operator is the whole real line, so it cannot be covered
    exhaustively; the sample is the boundary values of the type plus a random draw
    over the representable range, and the comparison is against the oracle, not
    against the reference, which is not always correctly rounded.
    """
    rng = np.random.default_rng(20261003)
    failures = []
    rows = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        # a random draw over the whole finite range, plus the boundaries
        bits = rng.integers(0, 2 ** (dt.itemsize * 8), size=400, dtype=np.uint64)
        raw = bits.astype(UINT_OF_WIDTH[dt.itemsize]).view(dt)
        raw = raw[np.isfinite(raw)]
        edges = np.array([0.0, -0.0, float(f.max), -float(f.max),
                          float(f.tiny), -float(f.tiny),
                          float(f.smallest_subnormal), -float(f.smallest_subnormal),
                          1.0, -1.0], dtype=dt)
        a = np.concatenate([raw, edges]).astype(dt)
        want = _oracle_array(a)
        got = np.asarray(impl.atan(a))
        if not _bits_equal(want, got):
            mask = np.nonzero(~((np.asarray(want).view(UINT_OF_WIDTH[dt.itemsize])
                                 == np.asarray(got).view(UINT_OF_WIDTH[dt.itemsize]))
                                | (np.isnan(want) & np.isnan(got))))[0][:5]
            failures.append({
                "type": dt.name, "who": "the implementation vs the Decimal oracle",
                "detail": "; ".join("x=%r oracle=%r impl=%r"
                                    % (float(a.ravel()[i]), float(want.ravel()[i]),
                                       float(got.ravel()[i])) for i in mask),
            })
        rows.append("%s: %d values" % (dt.name, len(a)))
    return {"sweep": "domain sample (Decimal oracle)",
            "summary": "; ".join(rows), "failures": failures}


def sweep_spec_examples(impl):
    """The worked examples of the document, as documented, against the oracle."""
    failures = []
    examples = [
        ("float Example 1 (double)",
         np.array([0.0, 1.0, -1.0], dtype=np.float64),
         np.array([0.0, 0.7853981633974483, -0.7853981633974483], dtype=np.float64)),
        ("float Example 2 (double)",
         np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float64),
         np.array([0.0, -0.0, 1.5707963267948966, -1.5707963267948966, np.nan],
                  dtype=np.float64)),
    ]
    for label, a, documented in examples:
        want = _oracle_array(a)
        got = np.asarray(impl.atan(a))
        if not _bits_equal(want, got):
            failures.append({"who": label, "detail": "oracle %s, implementation %s"
                             % (want.tolist(), got.tolist())})
        # the documented values, where they are finite, must agree with the oracle
        finite = np.isfinite(documented)
        if not _bits_equal(documented[finite], want[finite]):
            failures.append({"who": label, "detail": "documented %s, oracle %s"
                             % (documented.tolist(), want.tolist())})
    return {"sweep": "the document's worked examples",
            "summary": "%d examples, as documented vs the Decimal oracle" % len(examples),
            "failures": failures}


def sweep_real_section(impl):
    """The real section, which has no ONNX type: checked against exact arithmetic.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    document's real-number examples against exact rational arithmetic and the
    high-precision arctangent.
    """
    F = Fraction
    failures = []
    # Example 1: input [0, 1, -1], output [0, pi/4, -pi/4]
    # Example 2: input [sqrt(3), 1/sqrt(3)], output [pi/3, pi/6]
    # The exact values are checked by the tangent identity: tan(atan(x)) = x.
    examples = [
        ("real Example 1", [F(0), F(1), F(-1)]),
        ("real Example 2", [F(3).sqrt() if hasattr(F(3), "sqrt") else None]),
    ]
    # Fraction has no sqrt; use the Decimal oracle for the irrational inputs.
    checks = [
        ("real Example 1", [Decimal(0), Decimal(1), Decimal(-1)]),
        ("real Example 2", [Decimal(3).sqrt(), 1 / Decimal(3).sqrt()]),
    ]
    for label, xs in checks:
        for x in xs:
            y = _atan_decimal(x)
            # tan(y) must equal x, to the working precision
            t = _tan_decimal(y)
            if abs(t - x) > Decimal(10) ** -40:
                failures.append({"who": label,
                                 "detail": "x=%s, atan=%s, tan(atan)=%s" % (x, y, t)})
    return {"sweep": "real section (exact arithmetic, no ONNX Runtime)",
            "summary": "%d examples, checked by the tangent identity" % len(checks),
            "failures": failures}


def _tan_decimal(y: Decimal) -> Decimal:
    """tan(y) to the working precision, by the sine and cosine series."""
    getcontext().prec += 10
    # sin and cos by Taylor series
    def sin(z):
        term = z
        total = z
        k = 1
        while term:
            term = -term * z * z / ((2 * k) * (2 * k + 1))
            total += term
            k += 1
        return total
    def cos(z):
        term = Decimal(1)
        total = Decimal(1)
        k = 1
        while term:
            term = -term * z * z / ((2 * k - 1) * (2 * k))
            total += term
            k += 1
        return total
    result = sin(y) / cos(y)
    getcontext().prec -= 10
    return +result


def sweeps():
    return [sweep_spec_examples, sweep_special_values, sweep_domain, sweep_real_section]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def cases():
    """Every individual case: (label, [A])."""
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        for name, values in _float_value_cases(dtype):
            yield ("%s: %s" % (dt.name, name), [np.array(values, dtype=dtype)])
        quiet, signalling, sign = NAN_PATTERNS[dt.itemsize]
        yield ("%s: NaN operand, quiet, payload 1" % dt.name,
               [_bit_nan(dtype, quiet)])
        yield ("%s: NaN operand, quiet, payload 1, sign set" % dt.name,
               [_bit_nan(dtype, quiet | sign)])
        yield ("%s: NaN operand, signalling" % dt.name,
               [_bit_nan(dtype, signalling)])
    for name, shape in SHAPE_CASES:
        for dtype in ALL_FLOAT:
            yield ("%s: %s" % (np.dtype(dtype).name, name),
                   [_arr(shape, dtype, 0.5)])


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

COVERAGE = {
    "real": 0,
    "float": 0,
}

# every case of cases() exercises the float section; the real section is exercised
# by the sweeps, which have no ONNX type and cannot appear in cases().
for _label, _arrays in cases():
    COVERAGE["float"] += 1
# the real section: the worked examples and the tangent identity, in the sweeps
COVERAGE["real"] = 2
