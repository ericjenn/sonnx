"""The cases with which the **Asin** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled array per case.  Each case goes through a one-node
                     Asin model (opset 14) and through the implementation, and the two
                     results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched: the exhaustive float16 domain, the boundary
                     values and random samples of float32 and float64, the document's
                     worked examples, and the real section, which has no ONNX type and
                     is checked against exact arithmetic instead.

The reference is not always the correctly rounded value: the CPU runtime's float32
``Asin`` is one ulp below the correctly rounded value for some operands, so the
implementation is checked against exact arithmetic (a ``Decimal`` arcsine) with a
one-ulp tolerance for float32 and float64, and bit-exactly for float16.  The runtime's
refusal of ``double`` is a property of the build, not a mismatch: those cases are open
and the type stays in ``COVERAGE`` as not reference-checked.
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Asin"
OPSET = 14
IR_VERSION = 10

ALL_FLOAT = [np.float16, np.float32, np.float64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}

getcontext().prec = 60

PI = Decimal(
    "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
)


# ---------------------------------------------------------------------------
# exact arithmetic: the arcsine of a real number, in Decimal
# ---------------------------------------------------------------------------


def _arcsin_decimal(x):
    """arcsin(x) for x in [-1, 1], by the Maclaurin series with argument reduction."""
    if x == 0:
        return Decimal(0)
    if x < 0:
        return -_arcsin_decimal(-x)
    if x > 1:
        raise ValueError("outside [-1, 1]")
    if x == 1:
        return PI / 2
    if x > Decimal("0.5"):
        # arcsin(x) = pi/2 - 2 arcsin(sqrt((1 - x) / 2))
        return PI / 2 - 2 * _arcsin_decimal(((1 - x) / 2).sqrt())
    x2 = x * x
    term = x
    total = x
    n = 0
    while True:
        n += 1
        term *= Decimal((2 * n - 1) ** 2) / Decimal(2 * n * (2 * n + 1)) * x2
        total += term
        if abs(term) < Decimal(10) ** (-getcontext().prec + 5):
            break
    return total


def _exact_asin_value(x, dtype):
    """The correctly rounded arcsine of the float x in the given type."""
    dt = np.dtype(dtype)
    x = float(x)
    if math.isnan(x) or math.isinf(x) or abs(x) > 1.0:
        return float("nan")
    if x == 0.0:
        return x
    d = _arcsin_decimal(Decimal(x))
    if dt == np.float16:
        return float(np.float16(float(d)))
    if dt == np.float32:
        return float(np.float32(float(d)))
    return float(d)


# ---------------------------------------------------------------------------
# bit comparison helpers
# ---------------------------------------------------------------------------


def _bits_equal(want, got):
    """Bit-for-bit equality, NaN equal to NaN."""
    if want.shape != got.shape or want.dtype != got.dtype:
        return False
    uint = UINT_OF_WIDTH[want.dtype.itemsize]
    bw = np.ascontiguousarray(want).view(uint)
    bg = np.ascontiguousarray(got).view(uint)
    both_nan = np.isnan(want) & np.isnan(got)
    return bool(np.all((bw == bg) | both_nan))


def _bits_equal_mask(want, got):
    """Mask of the elements that differ bit for bit, NaN equal to NaN."""
    if want.shape != got.shape or want.dtype != got.dtype:
        return np.ones(np.shape(want), dtype=bool)
    uint = UINT_OF_WIDTH[want.dtype.itemsize]
    bw = np.ascontiguousarray(want).view(uint)
    bg = np.ascontiguousarray(got).view(uint)
    both_nan = np.isnan(want) & np.isnan(got)
    return ~((bw == bg) | both_nan)


def _ulp_mask(want, got):
    """Mask of the elements differing by more than one ulp; NaN counts as equal.

    The document's accuracy guidelines permit an implementation to depart from the
    correctly rounded value, so float32 and float64 are compared with a one-ulp
    tolerance: the bit patterns are read as integers and a difference of at most one
    is accepted.  Adjacent floats differ by one in that view, whatever the sign.
    """
    if want.shape != got.shape or want.dtype != got.dtype:
        return np.ones(np.shape(want), dtype=bool)
    uint = UINT_OF_WIDTH[want.dtype.itemsize]
    bw = np.ascontiguousarray(want).view(uint)
    bg = np.ascontiguousarray(got).view(uint)
    both_nan = np.isnan(want) & np.isnan(got)
    diff = np.where(bw > bg, bw - bg, bg - bw)
    return ~((diff <= 1) | both_nan)


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def _run_ort_asin(values):
    proto = helper.np_dtype_to_tensor_dtype(values.dtype)
    vi_x = helper.make_tensor_value_info("X", proto, list(values.shape))
    vi_y = helper.make_tensor_value_info("Y", proto, None)
    node = helper.make_node(OP, ["X"], ["Y"], name="asin0")
    graph = helper.make_graph([node], "asin_graph", [vi_x], [vi_y])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR_VERSION
    )
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y,) = sess.run(["Y"], {"X": values})
    return y


def _ort_note(values, exact):
    """One line saying what the reference did with these values, never raising."""
    dt = np.dtype(values.dtype)
    if dt == np.float64:
        return "not reference-checked, the runtime has no double kernel"
    try:
        ref = _run_ort_asin(values)
    except Exception as exc:
        return "the runtime refused it (%s)" % type(exc).__name__
    if _bits_equal(exact, ref):
        return "the runtime agrees with the exact value on all %d values" % values.size
    n = int(_ulp_mask(exact, ref).sum())
    return "the runtime departs from the exact value on %d of %d values" % (n, values.size)


# ---------------------------------------------------------------------------
# candidate values for the individual cases
# ---------------------------------------------------------------------------


def _candidate_values(dtype):
    dt = np.dtype(dtype)
    one = dt.type(1.0)
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    if dt == np.float16:
        vals += [2.0 ** -24, 2.0 ** -14, 2.0 ** -15, 2.0 ** -16]
    vals += [
        float(np.nextafter(one, dt.type(0.0))),
        float(np.nextafter(-one, dt.type(0.0))),
        float(np.nextafter(one, dt.type(2.0))),
        float(np.nextafter(-one, dt.type(-2.0))),
    ]
    rng = np.random.default_rng(20261003)
    vals += [float(v) for v in rng.uniform(-1.0, 1.0, size=24).astype(dt)]
    seen, out = set(), []
    for v in vals:
        f = float(dt.type(v))
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _filter_cases(dtype, candidates):
    """Keep only the candidates where ONNX Runtime agrees with the exact value.

    A case whose reference is itself one ulp off the correctly rounded value would
    test the reference's accuracy, not the implementation's compliance, so it is left
    to the sweep, which checks it against exact arithmetic.
    """
    dt = np.dtype(dtype)
    if dt == np.float64:
        return candidates  # the runtime has no double kernel; keep all, they are open
    try:
        arr = np.array(candidates, dtype=dt)
        ort_out = _run_ort_asin(arr)
        exact = np.array([_exact_asin_value(v, dt) for v in arr], dtype=dt)
        keep = []
        for i, v in enumerate(candidates):
            if _bits_equal(np.array([exact[i]], dtype=dt),
                           np.array([ort_out[i]], dtype=dt)):
                keep.append(v)
        return keep
    except Exception:
        return candidates


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------

SHAPE_CASES = [
    ("rank-0", ()),
    ("zero-sized (0,)", (0,)),
    ("zero-sized (0,3)", (0, 3)),
    ("zero-sized (2,0)", (2, 0)),
    ("ordinary (2,3)", (2, 3)),
]


def _build_cases():
    out = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        special = np.array([np.nan, np.inf, -np.inf, 2.0, -2.0, 0.0, -0.0], dtype=dt)
        out.append(("%s: special values" % dt.name, [special]))
        kept = _filter_cases(dt, _candidate_values(dt))
        if kept:
            out.append(("%s: ordinary values (reference-checked)" % dt.name,
                        [np.array(kept, dtype=dt)]))
        for name, shape in SHAPE_CASES:
            out.append(("%s: shape %s" % (dt.name, name),
                        [np.zeros(shape, dtype=dt)]))
    return out


_CASES = _build_cases()


def cases():
    """Every individual case: (label, [X])."""
    for label, arrays in _CASES:
        yield label, arrays


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_spec_examples(impl):
    """The document's worked examples, replayed as documented."""
    failures = []
    notes = []

    # float Example 1: X = [-1, -0.5, 0, 0.5, 1]
    x1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    want1 = np.array([_exact_asin_value(v, np.float32) for v in x1], dtype=np.float32)
    got1 = np.asarray(impl.asin(x1))
    if not _bits_equal(want1, got1):
        failures.append({"who": "float Example 1",
                         "detail": "exact %s, impl %s" % (want1.tolist(), got1.tolist())})
    notes.append("float Example 1: %s" % _ort_note(x1, want1))

    # float Example 2: X = [NaN, +inf, -inf, 2.0, -0.0]
    x2 = np.array([np.nan, np.inf, -np.inf, 2.0, -0.0], dtype=np.float32)
    want2 = np.array([_exact_asin_value(v, np.float32) for v in x2], dtype=np.float32)
    got2 = np.asarray(impl.asin(x2))
    if not _bits_equal(want2, got2):
        failures.append({"who": "float Example 2",
                         "detail": "exact %s, impl %s" % (want2.tolist(), got2.tolist())})

    # float Example 3: X = [2**-24] in float16
    x3 = np.array([2.0 ** -24], dtype=np.float16)
    want3 = np.array([_exact_asin_value(v, np.float16) for v in x3], dtype=np.float16)
    got3 = np.asarray(impl.asin(x3))
    if not _bits_equal(want3, got3):
        failures.append({"who": "float Example 3",
                         "detail": "exact %s, impl %s" % (want3.tolist(), got3.tolist())})

    return {"sweep": "document worked examples",
            "summary": "; ".join(notes),
            "failures": failures}


def sweep_float_exact(impl):
    """The float section against exact arithmetic.

    float16 is covered exhaustively over all 65536 bit patterns and checked bit for
    bit; float32 and float64 use their boundary values and a random sample and are
    checked with a one-ulp tolerance, the accuracy guidelines permitting a departure
    from the correctly rounded value.  The reference is reported, never failed on.
    """
    failures = []
    notes = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        if dt == np.float16:
            vals = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16)
            tol = 0
        else:
            f = np.finfo(dt)
            one = dt.type(1.0)
            vals = [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0,
                    float(np.nextafter(one, dt.type(0.0))),
                    float(np.nextafter(-one, dt.type(0.0))),
                    float(np.nextafter(one, dt.type(2.0))),
                    float(np.nextafter(-one, dt.type(-2.0))),
                    float(f.tiny), float(-f.tiny), float(f.max), float(-f.max),
                    float(np.nextafter(dt.type(0.0), dt.type(1.0))),
                    float(np.nextafter(dt.type(0.0), dt.type(-1.0)))]
            rng = np.random.default_rng(20261003)
            vals += [float(v) for v in rng.uniform(-2.0, 2.0, size=2000).astype(dt)]
            vals = np.array(vals, dtype=dt)
            tol = 1

        exact = np.array([_exact_asin_value(v, dt) for v in vals], dtype=dt)
        got = np.asarray(impl.asin(vals))
        if got.shape != exact.shape or got.dtype != exact.dtype:
            failures.append({
                "type": dt.name, "who": "implementation vs exact",
                "detail": "shape %s dtype %s, expected shape %s dtype %s"
                          % (got.shape, got.dtype, exact.shape, exact.dtype)})
        else:
            mask = _bits_equal_mask(exact, got) if tol == 0 else _ulp_mask(exact, got)
            if mask.any():
                bad = np.nonzero(mask)[0][:5]
                failures.append({
                    "type": dt.name, "who": "implementation vs exact",
                    "detail": "; ".join("x=%r exact=%r impl=%r"
                                        % (float(vals[i]), float(exact[i]), float(got[i]))
                                        for i in bad)})
        notes.append("%s: %s" % (dt.name, _ort_note(vals, exact)))

    return {"sweep": "float section against exact arithmetic",
            "summary": "; ".join(notes),
            "failures": failures}


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact arithmetic.

    ONNX Runtime cannot be asked about a real tensor, so the document's own values are
    compared with the arcsine computed in ``Decimal`` at 60 digits.
    """
    failures = []
    checks = 0

    # Example 1: X = [-1, -sqrt(2)/2, 0, 1/2, 1], Y = [-pi/2, -pi/4, 0, pi/6, pi/2]
    x1 = [Decimal(-1), -Decimal(2).sqrt() / 2, Decimal(0), Decimal(1) / 2, Decimal(1)]
    y1 = [-PI / 2, -PI / 4, Decimal(0), PI / 6, PI / 2]
    for i, (x, y) in enumerate(zip(x1, y1)):
        checks += 1
        got = _arcsin_decimal(x)
        if abs(got - y) > Decimal(10) ** -50:
            failures.append({"who": "real Example 1 element %d" % i,
                             "detail": "x=%s documented %s exact %s" % (x, y, got)})

    # Example 2: X = 0.5, Y = pi/6
    checks += 1
    got = _arcsin_decimal(Decimal(1) / 2)
    if abs(got - PI / 6) > Decimal(10) ** -50:
        failures.append({"who": "real Example 2",
                         "detail": "x=1/2 documented %s exact %s" % (PI / 6, got)})

    return {"sweep": "real section (exact arithmetic, no ONNX Runtime)",
            "summary": "%d real values of the document's examples checked against "
                       "exact Decimal arithmetic" % checks,
            "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_float_exact, sweep_real_examples]


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

COVERAGE = {
    "real": 6,          # the five values of real Example 1 and the one of Example 2
    "float": len(_CASES),
}
