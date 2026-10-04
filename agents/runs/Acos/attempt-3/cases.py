"""The cases with which the **Acos** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled operand per case.  Each case goes through a one-node
                     Acos model (opset 14) and through the implementation, and the two
                     results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the domain of the float
                     types, the worked examples of every section of the document, and
                     the real section, which has no ONNX type and is checked against
                     exact arithmetic instead.

Every case of every section's formula is meant to appear here, and the case set is part
of the loop: when a section is added to the document, its cases are added here before
the next run.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Acos"
OPSET = 14

ALL_FLOAT = [np.float16, np.float32, np.float64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}
IR_VERSION = 10


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(x: np.ndarray) -> np.ndarray:
    proto = helper.np_dtype_to_tensor_dtype(x.dtype)
    vi_x = helper.make_tensor_value_info("X", proto, list(x.shape))
    vi_y = helper.make_tensor_value_info("Y", proto, None)
    node = helper.make_node(OP, ["X"], ["Y"], name="acos0")
    graph = helper.make_graph([node], "acos_graph", [vi_x], [vi_y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    (y,) = sess.run(["Y"], {"X": x})
    return y


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
# the real section, transcribed literally
# ---------------------------------------------------------------------------
#
# The transcription is deliberately independent of the implementation: it is a second,
# literal, reading of the section **Acos** (real), evaluated in exact arithmetic.  The
# arccosine of a rational is not rational in general, so the exact value is compared
# through its cosine: y is the arccosine of x exactly when cos(y) = x and y in [0, pi].
# For the rational points of the document's examples the cosine is evaluated exactly
# with Fraction; for the transcendental points a high-precision computation is used.


def _cos_fraction(y: Fraction) -> Fraction:
    """cos(y) for a rational y, exactly, by the Taylor series of cos.

    The series is summed until the terms are below 2**-200, which is far below the
    precision of any comparison made here; the result is a Fraction, so the comparison
    with the rational x of the document is exact up to that truncation.
    """
    # reduce y modulo 2*pi is not possible in exact rational arithmetic; the examples
    # use y in [0, pi], so the series is summed directly.
    total = Fraction(0)
    term = Fraction(1)
    k = 0
    while True:
        total += term
        k += 1
        term = -term * y * y / ((2 * k - 1) * (2 * k))
        if abs(term) < Fraction(1, 2 ** 200):
            break
    return total


def _real_acos_exact(x: Fraction) -> Fraction:
    """The arccosine of a rational x in [-1, 1], to high precision, as a Fraction.

    Computed by bisection on the exact cosine: the arccosine is the unique y in [0, pi]
    with cos(y) = x, and cos is strictly decreasing there, so bisection converges.
    """
    lo, hi = Fraction(0), Fraction(355, 113)  # 0 and a rational above pi
    for _ in range(400):
        mid = (lo + hi) / 2
        if _cos_fraction(mid) > x:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _real_acos_float(x: float) -> float:
    """The arccosine of a float, to high precision, independent of the platform libm.

    Uses the ``decimal`` module at 60 digits, so the result does not come from the
    platform's ``math.acos``.
    """
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    d = Decimal(x)
    # arccos(x) = pi/2 - arcsin(x); arcsin by its series, which converges for |x| <= 1
    # slowly near 1, so use the identity arccos(x) = 2*arcsin(sqrt((1-x)/2)) instead.
    s = ((Decimal(1) - d) / 2).sqrt()
    # arcsin(s) by the series sum_{n>=0} (2n)!/(4^n (n!)^2 (2n+1)) s^(2n+1)
    total = Decimal(0)
    term = s
    n = 0
    while True:
        total += term / (2 * n + 1)
        n += 1
        term = term * s * s * (2 * n - 1) * (2 * n - 1) / (2 * n * (2 * n + 1))
        if abs(term) < Decimal(10) ** -55:
            break
    return float(2 * total)


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_float_domain(impl):
    """The float section against ONNX Runtime, over the domain of the type.

    The domain of the operator is [-1, 1]; the sweep covers the boundary values, the
    values around them, the special values of the type, and a random sample, for each
    of the three types.  The comparison is bit for bit, NaN equal to NaN.
    """
    rng = np.random.default_rng(20261003)
    rows, failures = [], []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        nan, inf = float("nan"), float("inf")
        edges = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, float(f.tiny), -float(f.tiny),
                 float(f.eps), -float(f.eps), 1.0 - float(f.eps), -1.0 + float(f.eps),
                 nan, inf, -inf]
        # the rounding boundary of the result near 0 and near pi
        edges += [1.0 - 2.0 ** -k for k in range(1, min(f.nmant + 2, 12))]
        edges += [-1.0 + 2.0 ** -k for k in range(1, min(f.nmant + 2, 12))]
        sample = rng.uniform(-1.0, 1.0, size=2000)
        xs = np.array(edges + list(sample), dtype=dt)

        ref = run_ort_batch(xs)            # ONNX Runtime
        got = np.asarray(impl.acos(xs))    # the implementation

        if not _bits_equal(ref, got):
            mask = np.nonzero(~((np.asarray(ref).view(UINT_OF_WIDTH[dt.itemsize])
                                 == np.asarray(got).view(UINT_OF_WIDTH[dt.itemsize]))
                                | (np.isnan(ref) & np.isnan(got))))[0][:5]
            failures.append({
                "type": dt.name, "who": "the implementation vs ONNX Runtime",
                "detail": "; ".join("x: %r, ort: %r, impl: %r"
                                    % (float(xs[i]), float(np.asarray(ref).ravel()[i]),
                                       float(got.ravel()[i])) for i in mask),
            })
        rows.append("%s %d values" % (dt.name, len(xs)))
    return {"sweep": "float domain (ORT vs implementation)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def _spec_examples():
    """The worked examples of the float section, as documented."""
    f32 = np.float32
    return [
        ("float Example 1",
         np.array([-1.0, 0.0, 1.0], dtype=f32),
         np.array([3.14159274, 1.57079637, 0.0], dtype=f32)),
        ("float Example 2",
         np.array([0.0, -0.0], dtype=f32),
         np.array([1.57079637, 1.57079637], dtype=f32)),
        ("float Example 3",
         np.array([np.nan], dtype=f32),
         np.array([np.nan], dtype=f32)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, and as ONNX Runtime does it."""
    failures = []
    for label, x, documented in _spec_examples():
        ref = run_ort_batch(x)
        got = np.asarray(impl.acos(x))
        if not _bits_equal(documented, ref):
            failures.append({"who": label, "detail": "documented %s, ORT %s"
                             % (documented.tolist(), ref.tolist())})
        if not _bits_equal(documented, got):
            failures.append({"who": label, "detail": "documented %s, implementation %s"
                             % (documented.tolist(), got.tolist())})
    return {"sweep": "the document's worked examples",
            "summary": "%d examples, as documented vs ORT vs implementation"
                       % len(_spec_examples()),
            "failures": failures}


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact arithmetic.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    implementation against the values the document's Examples state, in exact
    arithmetic: the rational points with ``fractions.Fraction``, the transcendental
    points with a high-precision computation independent of the platform's library.
    """
    F = Fraction
    failures = []

    # Example 1: acos(-1) = pi, acos(0) = pi/2, acos(1) = 0
    # Example 2: acos(1/2) = pi/3, acos(sqrt(2)/2) = pi/4, acos(-1/2) = 2pi/3
    # The exact values are checked through their cosine, which is rational for the
    # rational points of the examples.
    rational_points = [
        ("real Example 1, acos(1) = 0", F(1), F(0)),
        ("real Example 2, acos(1/2) = pi/3", F(1, 2), None),
        ("real Example 2, acos(-1/2) = 2pi/3", F(-1, 2), None),
    ]
    for label, x, documented in rational_points:
        y = _real_acos_exact(x)
        if documented is not None:
            if y != documented:
                failures.append({"who": label, "detail": "exact %s, computed %s"
                                 % (documented, y)})
        else:
            # check the property: cos(y) = x and y in [0, pi]
            if abs(_cos_fraction(y) - x) > Fraction(1, 2 ** 100):
                failures.append({"who": label, "detail": "cos(%s) = %s, expected %s"
                                 % (y, _cos_fraction(y), x)})
            if not (0 <= y <= Fraction(355, 113)):
                failures.append({"who": label, "detail": "y = %s outside [0, pi]" % y})

    # the transcendental points, checked against a high-precision computation
    transcendental = [
        ("real Example 1, acos(0) = pi/2", 0.0, np.pi / 2),
        ("real Example 1, acos(-1) = pi", -1.0, np.pi),
        ("real Example 2, acos(sqrt(2)/2) = pi/4", float(np.sqrt(2) / 2), np.pi / 4),
    ]
    for label, x, documented in transcendental:
        y = _real_acos_float(x)
        if abs(y - documented) > 1e-12:
            failures.append({"who": label, "detail": "high precision %r, documented %r"
                             % (y, documented)})

    # the implementation, on the real points, in exact arithmetic
    for label, x, documented in rational_points:
        got = np.asarray(impl.acos(np.array([x], dtype=object)))
        want = _real_acos_exact(x)
        if got.shape != (1,) or abs(Fraction(got[0]) - want) > Fraction(1, 2 ** 100):
            failures.append({"who": label + " (implementation)",
                             "detail": "exact %s, implementation %s" % (want, got.tolist())})

    return {"sweep": "real section (exact rationals and high precision, no ONNX Runtime)",
            "summary": "%d rational points with Fraction, %d transcendental points at "
                       "60 digits" % (len(rational_points), len(transcendental)),
            "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_float_domain, sweep_real_examples]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _float_value_cases(dtype):
    """The value families of the float section."""
    f = np.finfo(np.dtype(dtype))
    maxf = float(f.max)
    tiny = float(f.tiny)
    nan, inf = float("nan"), float("inf")
    return [
        ("bounds", [-1.0, 0.0, 1.0]),
        ("signed zeros", [0.0, -0.0]),
        ("infinities", [inf, -inf]),
        ("nan", [nan]),
        ("subnormal", [tiny, -tiny, tiny / 2.0]),
        ("smallest and largest", [maxf, -maxf, float(f.smallest_normal)]),
        ("rounding boundary near 1", [1.0 - float(f.eps), 1.0 - float(f.eps) / 2.0]),
        ("rounding boundary near -1", [-1.0 + float(f.eps), -1.0 + float(f.eps) / 2.0]),
        ("ordinary", [0.5, -0.5, 0.25, -0.25, 0.75, -0.75]),
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
    ("zero-sized: (0,3)", (0, 3)),
    ("zero-sized: (0,)", (0,)),
    ("zero-sized: (1,0)", (1, 0)),
    ("zero-sized: (0,0)", (0, 0)),
    ("3-d", (2, 3, 4)),
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
                   [_arr(shape, dtype, -1.0)])


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

COVERAGE = {
    "real": 0,
    "float": 0,
}

for _label, _arrays in cases():
    if _label.startswith("float16") or _label.startswith("float32") \
            or _label.startswith("float64"):
        COVERAGE["float"] += 1
    else:
        COVERAGE["real"] += 1

# the real section has no ONNX type: its cases live in the sweeps, and the count
# records the points the real sweep checks.
COVERAGE["real"] = 6
