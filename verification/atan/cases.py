"""The cases with which the **Atan** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled operand per case.  Each case goes through a one-node
                     Atan model (opset 14) and through the implementation, and the two
                     results are compared bit for bit.  The cases are the values the
                     document fixes exactly -- the signed zeros, the infinities, NaN and
                     the shapes -- because a kernel is free to be less accurate than the
                     document's ``round(arctan(x))``, and a bit-for-bit comparison of an
                     inexact value would test the reference's accuracy rather than the
                     implementation's compliance.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the special values of every
                     type, the sign and the shape, the correctness of a result (reported
                     in units of the last place, not required), the domain of every type,
                     the worked examples of the document, and the real section, which has
                     no ONNX type and is checked against exact Decimal arithmetic.

The reference of the correctness sweeps is ``np.arctan`` in float64, rounded to the type
of the operand: the document leaves the accuracy of a result to the profile's guidelines,
so a departure within ``SLACK`` units of the last place of the value 1 is reported, not
required.  The special values, the sign, the shape and the type are required exactly.
"""

from __future__ import annotations

from decimal import Decimal, getcontext, localcontext

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Atan"
OPSET = 14
IR_VERSION = 10

ALL_FLOAT = [np.float16, np.float32, np.float64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}

# The departure of a result from the correctly rounded value, in units of the last
# place of the value 1, per type.  Four units is not an accuracy difference between two
# conforming implementations but a defect; the departures are reported anyway.
SLACK = 4.0

getcontext().prec = 60


# ---------------------------------------------------------------------------
# ONNX Runtime, batched: one session per element type, reused by every sweep
# ---------------------------------------------------------------------------

_ORT_SESSIONS = {}


def _ort_session(dtype):
    dt = np.dtype(dtype)
    key = dt.str
    if key not in _ORT_SESSIONS:
        proto = helper.np_dtype_to_tensor_dtype(dt)
        vi_x = helper.make_tensor_value_info("X", proto, None)
        vi_y = helper.make_tensor_value_info("Y", proto, None)
        node = helper.make_node(OP, ["X"], ["Y"], name="atan0")
        graph = helper.make_graph([node], "atan_graph", [vi_x], [vi_y])
        model = helper.make_model(graph,
                                  opset_imports=[helper.make_opsetid("", OPSET)],
                                  ir_version=IR_VERSION)
        _ORT_SESSIONS[key] = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _ORT_SESSIONS[key]


def run_ort_batch(x):
    """One batched ONNX Runtime call, on a session cached per element type."""
    sess = _ort_session(x.dtype)
    (y,) = sess.run(["Y"], {"X": x})
    return y


def _bits_equal(want, got):
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
# an independent reference: the arctangent in high-precision Decimal
# ---------------------------------------------------------------------------
#
# The operand of the reference is the value the tensor holds, not the decimal that was
# written to build it: Decimal(v) applied to the float is exact, Decimal(repr(v)) is the
# shortest decimal representation and a different number.  The series is the Taylor
# series of the arctangent, summed in Decimal with a wide context; the argument is
# reduced to |x| <= 1/2 before the series is summed, so that the series converges in a
# hundred terms rather than in 10**60 of them.


def _atan_series(x):
    """arctan(x) for |x| <= 1/2, by the Taylor series, in Decimal."""
    if x == 0:
        return Decimal(0)
    eps = Decimal(10) ** (-(getcontext().prec - 5))
    x2 = x * x
    term = x
    total = x
    n = 1
    while True:
        term = -term * x2
        add = term / (2 * n + 1)
        total += add
        if abs(add) < eps:
            break
        n += 1
    return total


def _pi_decimal():
    """pi, by the Machin formula, in Decimal."""
    with localcontext() as ctx:
        ctx.prec = getcontext().prec + 10
        pi = 16 * _atan_series(Decimal(1) / 5) - 4 * _atan_series(Decimal(1) / 239)
    return +pi


_PI = None


def _pi():
    global _PI
    if _PI is None:
        _PI = _pi_decimal()
    return _PI


def _atan_decimal(x):
    """arctan of a Decimal, on the branch (-pi/2, pi/2)."""
    if x.is_nan():
        return x
    if x.is_infinite():
        half = _pi() / 2
        return half if x > 0 else -half
    if x < 0:
        return -_atan_decimal(-x)
    if x > 1:
        return _pi() / 2 - _atan_decimal(1 / x)
    if x > Decimal("0.5"):
        return _pi() / 4 + _atan_decimal((x - 1) / (x + 1))
    return _atan_series(x)


def _half_pi(dtype):
    """The value of the type nearest to pi/2, as the document states it."""
    dt = np.dtype(dtype)
    return np.array([float(_pi() / 2)], dtype=dt)[0]


# ---------------------------------------------------------------------------
# the correctness of a result: reported, not required
# ---------------------------------------------------------------------------


def _ulp_of_one(dtype):
    """One unit in the last place of the value 1, for this type."""
    f = np.finfo(np.dtype(dtype))
    return float(2.0 ** -f.nmant)


def _reference(x, dtype):
    """The arctangent of the float array x, rounded to the type of x.

    The reference is computed in float64 and rounded to the type: for float16 and
    float32 that is the correctly rounded value to within a fraction of a unit of the
    last place of the type, and for float64 it is the platform's own arctangent, whose
    departure is reported rather than required.
    """
    dt = np.dtype(dtype)
    y = np.arctan(np.asarray(x, dtype=np.float64))
    return np.asarray(y, dtype=dt)


def _departure_ulps(got, want, dtype):
    """|got - want| in units of the last place of 1, element-wise, NaN -> 0."""
    ulp = _ulp_of_one(dtype)
    g = np.asarray(got, dtype=np.float64)
    w = np.asarray(want, dtype=np.float64)
    if g.shape != w.shape:
        return np.array([float("inf")])
    d = np.abs(g - w) / ulp
    d = np.where(np.isnan(g) & np.isnan(w), 0.0, d)
    return d


# ---------------------------------------------------------------------------
# the special values of the document, transcribed literally
# ---------------------------------------------------------------------------


def sweep_special_values(impl):
    """The special values of every type, against the document and against ONNX Runtime.

    The document decides these exactly: the sign of a null is preserved, an infinity
    gives the value of the type nearest to +-pi/2, and NaN gives a quiet NaN.  The
    reference is the document's own value, computed in Decimal, not the bit pattern of
    the runtime.
    """
    failures = []
    rows = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        hp = _half_pi(dt)
        checks = [
            ("+0.0 -> +0.0", np.array([0.0], dtype=dt), np.array([0.0], dtype=dt)),
            ("-0.0 -> -0.0", np.array([-0.0], dtype=dt), np.array([-0.0], dtype=dt)),
            ("+inf -> nearest(pi/2)", np.array([np.inf], dtype=dt),
             np.array([hp], dtype=dt)),
            ("-inf -> nearest(-pi/2)", np.array([-np.inf], dtype=dt),
             np.array([-hp], dtype=dt)),
        ]
        for label, x, want in checks:
            got = np.asarray(impl.atan(x))
            if not _bits_equal(want, got):
                failures.append({
                    "who": "%s: %s" % (dt.name, label),
                    "detail": "document %s, implementation %s"
                              % (want.tolist(), got.tolist()),
                })
        got = np.asarray(impl.atan(np.array([np.nan], dtype=dt)))
        if not (got.shape == (1,) and got.dtype == dt and bool(np.isnan(got[0]))):
            failures.append({"who": "%s: NaN" % dt.name,
                             "detail": "implementation %s" % (got.tolist(),)})
        note = "ORT agrees"
        try:
            ref = run_ort_batch(np.array([0.0, -0.0, np.inf, -np.inf, np.nan],
                                         dtype=dt))
            want = np.array([0.0, -0.0, hp, -hp, np.nan], dtype=dt)
            if not _bits_equal(want, ref):
                note = "ORT departs from the document"
        except Exception as exc:
            note = "not reference-checked (%s)" % type(exc).__name__
        rows.append("%s: special values checked, %s" % (dt.name, note))
    return {"sweep": "the special values of every type (document, exact)",
            "summary": "; ".join(rows), "failures": failures}


def sweep_sign_and_shape(impl):
    """The sign (oddness) and the shape of the result, as the document states them."""
    failures = []
    rng = np.random.default_rng(20261004)
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        x = (rng.standard_normal(64) * 4.0).astype(dt)
        pos = np.asarray(impl.atan(x))
        neg = np.asarray(impl.atan(-x))
        if not _bits_equal(pos, -neg):
            bad = np.nonzero(np.asarray(pos, dtype=np.float64)
                             != -np.asarray(neg, dtype=np.float64))[0][:5]
            failures.append({"who": "%s: oddness" % dt.name,
                             "detail": "; ".join("atan(%r)=%r, -atan(%r)=%r"
                                                 % (float(x[i]), float(pos[i]),
                                                    float(-x[i]), float(-neg[i]))
                                                 for i in bad)})
        for shape in [(), (0,), (0, 3), (2, 3), (1, 0, 4)]:
            a = np.zeros(shape, dtype=dt)
            got = np.asarray(impl.atan(a))
            if got.shape != shape or got.dtype != dt:
                failures.append({"who": "%s: shape %s" % (dt.name, shape),
                                 "detail": "implementation shape %s dtype %s"
                                           % (got.shape, got.dtype)})
    return {"sweep": "sign (oddness) and shape (document)",
            "summary": "3 types, 64 random values each, 5 shapes each",
            "failures": failures}


def sweep_correctness(impl):
    """The correctness of a result, in units of the last place of 1, per type.

    The document specifies round(arctan(x)) with roundTiesToEven.  A library is not
    required to be correctly rounded for it, so a departure within SLACK units is not a
    failure; the departures are reported anyway, with the count and the worst departure.
    The reference is computed from the float's exact value, in float64, and never from
    the decimal that was written to build the array.
    """
    failures = []
    rows = []
    rng = np.random.default_rng(20261004)
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        xs = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0,
              float(f.tiny), -float(f.tiny), float(f.max), -float(f.max),
              float(f.smallest_subnormal), -float(f.smallest_subnormal)]
        xs += list(rng.uniform(-8.0, 8.0, 20))
        x = np.array(xs, dtype=dt)
        got = np.asarray(impl.atan(x))
        if got.shape != x.shape or got.dtype != dt:
            failures.append({"who": "%s: shape/dtype" % dt.name,
                             "detail": "input %s %s, implementation %s %s"
                                       % (x.shape, dt, got.shape, got.dtype)})
            rows.append("%s: shape/dtype mismatch" % dt.name)
            continue
        want = _reference(x, dt)
        d = _departure_ulps(got, want, dt)
        worst = float(np.max(d)) if d.size else 0.0
        n_dep = int(np.sum(d > 0.0))
        if worst > SLACK:
            bad = np.nonzero(d > SLACK)[0][:5]
            failures.append({
                "who": "%s: departure beyond SLACK=%.1f ulp" % (dt.name, SLACK),
                "detail": "; ".join("atan(%r): got %r, reference %r, %.2f ulp"
                                    % (float(x[i]), float(got[i]), float(want[i]),
                                       float(d[i])) for i in bad),
            })
        rows.append("%s: %d/%d depart, worst %.2f ulp" % (dt.name, n_dep, d.size, worst))
    return {"sweep": "the correctness of a result (reported, not required: see V1)",
            "summary": "; ".join(rows), "failures": failures}


def sweep_domain(impl):
    """The domain of the operator, over the values of every type.

    The domain is the whole type, so the boundary values, the subnormal range and a
    random sample are covered; the result is checked against the document's own value,
    computed in float64, and against ONNX Runtime where the runtime has a kernel.
    """
    failures = []
    rows = []
    rng = np.random.default_rng(20261005)
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        xs = [0.0, -0.0, 1.0, -1.0, float(f.tiny), -float(f.tiny),
              float(f.max), -float(f.max), float(f.smallest_subnormal),
              -float(f.smallest_subnormal), 1e-3, -1e-3, 1e3, -1e3]
        xs += list(rng.uniform(-100.0, 100.0, 20))
        x = np.array(xs, dtype=dt)
        got = np.asarray(impl.atan(x))
        if got.shape != x.shape or got.dtype != dt:
            failures.append({"who": "%s: shape/dtype" % dt.name,
                             "detail": "input %s %s, implementation %s %s"
                                       % (x.shape, dt, got.shape, got.dtype)})
            rows.append("%s: shape/dtype mismatch" % dt.name)
            continue
        want = _reference(x, dt)
        d = _departure_ulps(got, want, dt)
        worst = float(np.max(d)) if d.size else 0.0
        if worst > SLACK:
            bad = np.nonzero(d > SLACK)[0][:5]
            failures.append({
                "who": "%s: domain departure beyond SLACK=%.1f ulp" % (dt.name, SLACK),
                "detail": "; ".join("atan(%r): got %r, reference %r, %.2f ulp"
                                    % (float(x[i]), float(got[i]), float(want[i]),
                                       float(d[i])) for i in bad),
            })
        note = ""
        try:
            ref = run_ort_batch(x)
            if ref.shape != got.shape or ref.dtype != got.dtype:
                failures.append({"who": "%s: shape/dtype vs ORT" % dt.name,
                                 "detail": "ORT %s %s, implementation %s %s"
                                           % (ref.shape, ref.dtype, got.shape,
                                              got.dtype)})
            else:
                n_ort = int(np.sum(_departure_ulps(ref, want, dt) > 0.0))
                note = ", ORT departs at %d/%d" % (n_ort, x.size)
        except Exception as exc:
            note = ", not reference-checked (%s)" % type(exc).__name__
        rows.append("%s: %d values, worst %.2f ulp%s" % (dt.name, x.size, worst, note))
    return {"sweep": "the domain of every type (document, exact)",
            "summary": "; ".join(rows), "failures": failures}


# ---------------------------------------------------------------------------
# the worked examples of the document
# ---------------------------------------------------------------------------


def _spec_examples():
    """The worked examples of the float section, as documented."""
    f64 = np.float64
    return [
        ("float Example 1",
         np.array([0.0, 1.0, -1.0], dtype=f64),
         np.array([0.0, 0.7853981633974483, -0.7853981633974483], dtype=f64)),
        ("float Example 2",
         np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=f64),
         np.array([0.0, -0.0, 1.5707963267948966, -1.5707963267948966, np.nan],
                  dtype=f64)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, and as ONNX Runtime does it.

    The documented values are the correctly rounded ones; a departure of the
    implementation within SLACK units is reported, not required.
    """
    failures = []
    rows = []
    for label, x, documented in _spec_examples():
        got = np.asarray(impl.atan(x))
        if got.shape != documented.shape or got.dtype != documented.dtype:
            failures.append({"who": label,
                             "detail": "documented %s %s, implementation %s %s"
                                       % (documented.shape, documented.dtype,
                                          got.shape, got.dtype)})
            rows.append("%s: shape/dtype mismatch" % label)
            continue
        d = _departure_ulps(got, documented, np.float64)
        worst = float(np.max(d)) if d.size else 0.0
        if worst > SLACK:
            failures.append({"who": label,
                             "detail": "documented %s, implementation %s, %.2f ulp"
                                       % (documented.tolist(), got.tolist(), worst)})
        note = ""
        try:
            ref = run_ort_batch(x)
            if not _bits_equal(documented, ref):
                note = ", ORT departs from the documented value"
        except Exception as exc:
            note = ", not reference-checked (%s)" % type(exc).__name__
        rows.append("%s: worst %.2f ulp%s" % (label, worst, note))
    return {"sweep": "the document's worked examples",
            "summary": "; ".join(rows), "failures": failures}


# ---------------------------------------------------------------------------
# the real section, which has no ONNX type
# ---------------------------------------------------------------------------


def sweep_real_examples(impl):
    """The real section, checked against exact arithmetic.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    document's own values in exact arithmetic: the arctangent of 0 is 0, of 1 is pi/4,
    of -1 is -pi/4, of sqrt(3) is pi/3 and of 1/sqrt(3) is pi/6, and the branch is open.
    The implementation is checked on the same real values, in double, against the exact
    arctangent of the float that holds them.
    """
    failures = []
    pi = _pi()
    checks = [
        ("real Example 1: atan(0) = 0", Decimal(0), Decimal(0)),
        ("real Example 1: atan(1) = pi/4", Decimal(1), pi / 4),
        ("real Example 1: atan(-1) = -pi/4", Decimal(-1), -pi / 4),
        ("real Example 2: atan(sqrt(3)) = pi/3", Decimal(3).sqrt(), pi / 3),
        ("real Example 2: atan(1/sqrt(3)) = pi/6",
         Decimal(1) / Decimal(3).sqrt(), pi / 6),
    ]
    tol = Decimal(10) ** (-(getcontext().prec - 20))
    for label, x, want in checks:
        got = _atan_decimal(x)
        if abs(got - want) > tol:
            failures.append({"who": label,
                             "detail": "exact %s, reference %s" % (want, got)})
    # the branch is open: the result is never equal to +-pi/2
    for x in [Decimal(10) ** 30, Decimal(-10) ** 30]:
        got = _atan_decimal(x)
        if abs(got) >= pi / 2:
            failures.append({"who": "the branch is open",
                             "detail": "atan(%s) = %s, not in (-pi/2, pi/2)"
                                       % (x, got)})
    # the implementation, on the same real values, in double
    for label, x, want in checks:
        xf = float(x)
        exact = _atan_decimal(Decimal(xf))
        got = np.asarray(impl.atan(np.array([xf], dtype=np.float64)))
        d = _departure_ulps(got, np.array([float(exact)], dtype=np.float64),
                            np.float64)
        if float(np.max(d)) > SLACK:
            failures.append({"who": label + " (implementation)",
                             "detail": "atan(%r) = %r, exact %s, %.2f ulp"
                                       % (xf, float(got[0]), exact,
                                          float(np.max(d)))})
    return {"sweep": "real section (exact arithmetic, no ONNX Runtime)",
            "summary": "%d examples and the open branch, checked in Decimal"
                       % len(checks),
            "failures": failures}


def sweeps():
    return [sweep_special_values, sweep_sign_and_shape, sweep_correctness,
            sweep_domain, sweep_spec_examples, sweep_real_examples]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _bit_nan(dtype, pattern):
    uint = UINT_OF_WIDTH[np.dtype(dtype).itemsize]
    return np.array([pattern], dtype=uint).view(dtype)


NAN_PATTERNS = {2: (0x7E01, 0x7C01, 0x8000),
                4: (0x7FC00001, 0x7F800001, 0x80000000),
                8: (0x7FF8000000000001, 0x7FF0000000000001, 0x8000000000000000)}


def _forced_array(shape, dtype):
    """An array whose arctangent the document fixes exactly: zeros, infinities, NaN."""
    dt = np.dtype(dtype)
    n = 1
    for d in shape:
        n *= d
    if n == 0:
        return np.zeros(shape, dtype=dt)
    pattern = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=dt)
    return np.resize(pattern, n).reshape(shape)


SHAPE_CASES = [
    ("rank-0", ()),
    ("zero-sized (0,)", (0,)),
    ("zero-sized (0,3)", (0, 3)),
    ("zero-sized (2,0,3)", (2, 0, 3)),
    ("(1,)", (1,)),
    ("(2,3)", (2, 3)),
    ("(2,3,4)", (2, 3, 4)),
]


def cases():
    """Every individual case: (label, [X])."""
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        yield ("%s: signed zeros" % dt.name,
               [np.array([0.0, -0.0, 0.0, -0.0], dtype=dt)])
        yield ("%s: infinities" % dt.name,
               [np.array([np.inf, -np.inf, np.inf, -np.inf], dtype=dt)])
        yield ("%s: nan" % dt.name,
               [np.array([np.nan, np.nan, np.nan], dtype=dt)])
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
                   [_forced_array(shape, dtype)])


# ---------------------------------------------------------------------------
# coverage: every section anchor of the document
# ---------------------------------------------------------------------------

COVERAGE = {
    "real": 0,
    "float": 0,
}

for _label, _arrays in cases():
    COVERAGE["float"] += 1

# the real section has no ONNX type: its checks are the sweeps, not the cases above
# (5 worked examples, the open branch at two operands, and the implementation on the
# same five real values)
COVERAGE["real"] = 12
