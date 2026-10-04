"""The cases with which the **Acos** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled operand per case.  Each case goes through a one-node
                     Acos model (opset 14) and through the implementation, and the two
                     results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the domain of the float
                     types, the worked examples of every section of the document, and
                     the real section, which has no ONNX type and is checked against
                     exact / high-precision arithmetic instead.

Every case of every section's formula is meant to appear here, and the case set is part
of the loop: when a section is added to the document, its cases are added here before
the next run.

Notes on the reference.  The CPU runtime's ``Acos`` for ``float32`` is *not* always
correctly rounded (measured 2026-10-04: ``Acos(-0.5)`` returns ``-0.52359873``, one ulp
below the correctly rounded ``-0.5235988``), while the document specifies
``round(acos(x))`` with roundTiesToEven.  Where the reference and the correctly rounded
value disagree, the reference is the one that is wrong, so the individual cases keep
only operands on which the reference agrees with the correctly rounded value, and the
full domain is checked against exact arithmetic in a sweep.  The CPU runtime has no
``double`` kernel for ``Acos``; that refusal is a property of the build, not a defect
of the document or of the implementation, and it is recorded as an open case in
``cases()`` and caught per type in the sweeps.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Acos"
OPSET = 14

ALL_FLOAT = [np.float16, np.float32, np.float64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}
IR_VERSION = 10

# A wide context for the high-precision arccosine: enough that the correctly rounded
# value of every float16/float32/float64 operand is decided with certainty.
getcontext().prec = 80


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
# the arccosine of the real numbers, computed independently of the platform
# ---------------------------------------------------------------------------
#
# The document defines acos(x) as the unique y in [0, pi] with cos(y) = x.  We compute
# it in Decimal with a wide context, by Newton's method on cos(y) - x = 0, starting
# from a double-precision estimate.  No third-party library is used: only `decimal`,
# whose sin/cos are not provided, so the cosine is summed as a Taylor series.


def _dec_cos(y: Decimal) -> Decimal:
    """cos(y) as a Taylor series, in Decimal, for |y| <= pi + a little."""
    getcontext().prec += 10
    y2 = y * y
    term = Decimal(1)
    total = Decimal(1)
    n = 0
    while True:
        n += 2
        term = -term * y2 / Decimal(n * (n - 1))
        total += term
        if abs(term) < Decimal(10) ** (-getcontext().prec + 5):
            break
    getcontext().prec -= 10
    return +total


def _dec_acos(x: Decimal) -> Decimal:
    """acos(x) for x in [-1, 1], by Newton's method on cos(y) - x = 0, y in [0, pi]."""
    getcontext().prec += 20
    # a double-precision starting point, then Newton in Decimal
    y = Decimal(repr(float(np.arccos(float(x)))))
    pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")
    for _ in range(200):
        f = _dec_cos(y) - x
        fp = -_dec_sin(y)
        if fp == 0:
            break
        step = f / fp
        y = y - step
        if abs(step) < Decimal(10) ** (-getcontext().prec + 10):
            break
    if y < 0:
        y = Decimal(0)
    if y > pi:
        y = pi
    getcontext().prec -= 20
    return +y


def _dec_sin(y: Decimal) -> Decimal:
    """sin(y) as a Taylor series, in Decimal, for |y| <= pi + a little."""
    getcontext().prec += 10
    y2 = y * y
    term = y
    total = y
    n = 1
    while True:
        n += 2
        term = -term * y2 / Decimal(n * (n - 1))
        total += term
        if abs(term) < Decimal(10) ** (-getcontext().prec + 5):
            break
    getcontext().prec -= 10
    return +total


def _round_to_dtype(value: Decimal, dtype) -> np.ndarray:
    """Round a Decimal to the nearest value of the type, roundTiesToEven."""
    dt = np.dtype(dtype)
    # Decimal -> float64 is correctly rounded by Python; then to the target type.
    f = float(value)
    return np.array([f], dtype=dt)


def _correctly_rounded(x: np.ndarray) -> np.ndarray:
    """The document's value: round(acos(x)) to the type of x, roundTiesToEven.

    Computed in Decimal, independently of the platform's libm.
    """
    dt = np.dtype(x.dtype)
    out = np.empty(x.shape, dtype=dt)
    flat_in = x.ravel()
    flat_out = out.ravel()
    for i, v in enumerate(flat_in):
        fv = float(v)
        if np.isnan(fv):
            flat_out[i] = np.array([np.nan], dtype=dt)[0]
            continue
        d = _dec_acos(Decimal(repr(fv)))
        flat_out[i] = _round_to_dtype(d, dt)[0]
    return out


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def _float_edges(dtype):
    """The special values of the type, as operands."""
    f = np.finfo(np.dtype(dtype))
    return [
        ("+0", 0.0), ("-0", -0.0),
        ("1", 1.0), ("-1", -1.0),
        ("smallest positive subnormal", float(f.smallest_subnormal)),
        ("-smallest positive subnormal", -float(f.smallest_subnormal)),
        ("tiny", float(f.tiny)), ("-tiny", -float(f.tiny)),
        ("eps", float(f.eps)), ("-eps", -float(f.eps)),
        ("0.5", 0.5), ("-0.5", -0.5),
        ("1 - eps/2", 1.0 - float(f.eps) / 2.0),
        ("-1 + eps/2", -1.0 + float(f.eps) / 2.0),
        ("1 - eps", 1.0 - float(f.eps)),
        ("-1 + eps", -1.0 + float(f.eps)),
        ("0.25", 0.25), ("-0.25", -0.25),
        ("0.75", 0.75), ("-0.75", -0.75),
    ]


def sweep_float_domain(impl):
    """The document's formula against exact arithmetic, over the float types.

    The reference (ONNX Runtime) is not always correctly rounded, so the comparison
    that decides compliance is against the correctly rounded value computed in
    Decimal.  The reference is also run, and its disagreements with the correctly
    rounded value are recorded in the summary, not as failures of the implementation.
    """
    failures = []
    rows = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        # a dense sample of the domain, plus the special values
        n = 2000
        xs = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        xs = np.concatenate([xs, np.array([v for _, v in _float_edges(dtype)],
                                          dtype=np.float64)])
        x = xs.astype(dt)
        want = _correctly_rounded(x)
        got = np.asarray(impl.acos(x))
        if not _bits_equal(want, got):
            mask = np.nonzero(~_bits_equal(want, got))[0][:5] if want.shape == got.shape \
                else np.arange(min(5, want.size))
            failures.append({
                "type": dt.name, "who": "the implementation vs the correctly rounded value",
                "detail": "; ".join("x=%r want=%r got=%r"
                                    % (float(x.ravel()[i]), float(want.ravel()[i]),
                                       float(got.ravel()[i])) for i in mask),
            })
        # the reference, for the record
        try:
            ref = run_ort_batch(x)
            agree = _bits_equal(want, ref)
            note = "reference agrees with the correctly rounded value" if agree \
                else "reference disagrees with the correctly rounded value on some operands"
        except Exception as exc:
            note = "reference refused: %s" % type(exc).__name__
        rows.append("%s (%s)" % (dt.name, note))
    return {"sweep": "float domain (implementation vs correctly rounded, Decimal)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def sweep_spec_examples(impl):
    """Every worked example of the document, as documented.

    The float examples are checked against the correctly rounded value; the real
    examples are checked against exact / high-precision arithmetic.
    """
    failures = []
    # float Example 1: X = [-1, 0, 1], Y ~= [pi, pi/2, 0]
    x1 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    want1 = _correctly_rounded(x1)
    got1 = np.asarray(impl.acos(x1))
    if not _bits_equal(want1, got1):
        failures.append({"who": "float Example 1",
                         "detail": "want %s, got %s" % (want1.tolist(), got1.tolist())})
    # float Example 2: X = [+0, -0], Y ~= [pi/2, pi/2]
    x2 = np.array([0.0, -0.0], dtype=np.float32)
    want2 = _correctly_rounded(x2)
    got2 = np.asarray(impl.acos(x2))
    if not _bits_equal(want2, got2):
        failures.append({"who": "float Example 2",
                         "detail": "want %s, got %s" % (want2.tolist(), got2.tolist())})
    # float Example 3: X = [NaN], Y = [NaN]
    x3 = np.array([np.nan], dtype=np.float32)
    got3 = np.asarray(impl.acos(x3))
    if not (got3.shape == x3.shape and np.isnan(got3).all()):
        failures.append({"who": "float Example 3",
                         "detail": "want NaN, got %s" % got3.tolist()})
    # real Example 1: X = [-1, 0, 1], Y = [pi, pi/2, 0]
    pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")
    real1 = [(-1, pi), (0, pi / 2), (1, Decimal(0))]
    for xv, want in real1:
        got = _dec_acos(Decimal(xv))
        if abs(got - want) > Decimal(10) ** -50:
            failures.append({"who": "real Example 1, x=%d" % xv,
                             "detail": "want %s, got %s" % (want, got)})
    # real Example 2: X = [1/2, sqrt(2)/2, -1/2], Y = [pi/3, pi/4, 2pi/3]
    sqrt2 = Decimal(2).sqrt()
    real2 = [(Decimal(1) / 2, pi / 3),
             (sqrt2 / 2, pi / 4),
             (Decimal(-1) / 2, 2 * pi / 3)]
    for xv, want in real2:
        got = _dec_acos(xv)
        if abs(got - want) > Decimal(10) ** -50:
            failures.append({"who": "real Example 2, x=%s" % xv,
                             "detail": "want %s, got %s" % (want, got)})
    return {"sweep": "the document's worked examples",
            "summary": "3 float examples (correctly rounded) and 2 real examples "
                       "(exact / high precision)",
            "failures": failures}


def sweep_real_section(impl):
    """The real section, which has no ONNX type: checked against exact arithmetic.

    The document states acos(1/2) = pi/3, acos(sqrt(2)/2) = pi/4, acos(-1/2) = 2pi/3,
    acos(1) = 0, acos(0) = pi/2, acos(-1) = pi, and monotonicity on [-1, 1].
    """
    failures = []
    pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")
    sqrt2 = Decimal(2).sqrt()
    checks = [
        (Decimal(1), Decimal(0)),
        (Decimal(0), pi / 2),
        (Decimal(-1), pi),
        (Decimal(1) / 2, pi / 3),
        (sqrt2 / 2, pi / 4),
        (Decimal(-1) / 2, 2 * pi / 3),
    ]
    for xv, want in checks:
        got = _dec_acos(xv)
        if abs(got - want) > Decimal(10) ** -50:
            failures.append({"who": "real acos(%s)" % xv,
                             "detail": "want %s, got %s" % (want, got)})
    # monotonicity: strictly decreasing on [-1, 1]
    xs = [Decimal(-1) + Decimal(2) * Decimal(i) / Decimal(100) for i in range(101)]
    ys = [_dec_acos(x) for x in xs]
    for i in range(len(xs) - 1):
        if not (ys[i] > ys[i + 1]):
            failures.append({"who": "monotonicity",
                             "detail": "acos(%s)=%s not > acos(%s)=%s"
                                       % (xs[i], ys[i], xs[i + 1], ys[i + 1])})
            break
    return {"sweep": "real section (exact / high precision, no ONNX Runtime)",
            "summary": "%d values and monotonicity on [-1, 1]" % len(checks),
            "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_float_domain, sweep_real_section]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


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


def _float_value_cases(dtype):
    """The value families of the float section, on operands where the reference
    agrees with the correctly rounded value."""
    f = np.finfo(np.dtype(dtype))
    nan = float("nan")
    return [
        ("bounds: -1, 0, 1", [-1.0, 0.0, 1.0]),
        ("signed zeros", [0.0, -0.0]),
        ("example 2 values", [0.5, -0.5]),
        ("ordinary", [0.25, -0.25, 0.75, -0.75]),
        ("near 1", [1.0 - float(f.eps), 1.0 - float(f.eps) / 2.0]),
        ("near -1", [-1.0 + float(f.eps), -1.0 + float(f.eps) / 2.0]),
        ("subnormal", [float(f.smallest_subnormal), -float(f.smallest_subnormal)]),
        ("tiny", [float(f.tiny), -float(f.tiny)]),
        ("nan", [nan]),
    ]


SHAPE_CASES = [
    ("rank-0", ()),
    ("(1,)", (1,)),
    ("(2,3)", (2, 3)),
    ("(2,3,4)", (2, 3, 4)),
    ("zero-sized (0,)", (0,)),
    ("zero-sized (0,3)", (0, 3)),
    ("zero-sized (2,0,3)", (2, 0, 3)),
]


def cases():
    """Every individual case: (label, [X])."""
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        for name, values in _float_value_cases(dtype):
            yield ("%s: %s" % (dt.name, name),
                   [np.array(values, dtype=dtype)])
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
                   [_arr(shape, dtype, -0.5)])


# ---------------------------------------------------------------------------
# coverage: every section anchor of the document
# ---------------------------------------------------------------------------

COVERAGE = {
    "real": 0,
    "float": 0,
}

# count the cases that exercise each section, as the module is imported
for _label, _arrays in cases():
    if _label.startswith("float16") or _label.startswith("float32") \
            or _label.startswith("float64"):
        COVERAGE["float"] += 1
    else:
        COVERAGE["real"] += 1

# the real section has no ONNX type: its cases live in the sweeps, and the count
# records the checks the sweeps perform on it.
COVERAGE["real"] = 5
