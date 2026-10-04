"""The cases with which the **Atan** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled operand per case.  Each case goes through a one-node
                     Atan model (opset 14) and through the implementation, and the two
                     results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the value families of every
                     type, the worked examples of every section of the document, and the
                     real section, which has no ONNX type and is checked against exact
                     arithmetic instead.

Every case of every section's formula is meant to appear here, and the case set is part
of the loop: when a section is added to the document, its cases are added here before
the next run.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Atan"
OPSET = 14

ALL_FLOAT = [np.float16, np.float32, np.float64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}
IR_VERSION = 10


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(a: np.ndarray) -> np.ndarray:
    proto = helper.np_dtype_to_tensor_dtype(a.dtype)
    vi_a = helper.make_tensor_value_info("A", proto, list(a.shape))
    vi_y = helper.make_tensor_value_info("Y", proto, None)
    node = helper.make_node(OP, ["A"], ["Y"], name="atan0")
    graph = helper.make_graph([node], "atan_graph", [vi_a], [vi_y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    (y,) = sess.run(["Y"], {"A": a})
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
# the value families of the float section, transcribed literally
# ---------------------------------------------------------------------------


def _bit_nan(dtype, pattern):
    uint = UINT_OF_WIDTH[np.dtype(dtype).itemsize]
    return np.array([pattern], dtype=uint).view(dtype)


NAN_PATTERNS = {2: (0x7E01, 0x7C01, 0x8000),
                4: (0x7FC00001, 0x7F800001, 0x80000000),
                8: (0x7FF8000000000001, 0x7FF0000000000001, 0x8000000000000000)}


def _float_value_cases(dtype):
    """The value families of the float section, as the document states them."""
    f = np.finfo(np.dtype(dtype))
    maxf = float(f.max)
    tiny = float(f.tiny)
    smallest_sub = float(f.smallest_subnormal)
    nan, inf = float("nan"), float("inf")
    return [
        ("signed zeros", [0.0, -0.0, 0.0, -0.0]),
        ("infinities", [inf, -inf, inf, -inf]),
        ("nan", [nan, nan, nan]),
        ("ordinary", [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0]),
        ("the document's Example 1", [0.0, 1.0, -1.0]),
        ("the document's Example 2", [0.0, -0.0, inf, -inf, nan]),
        ("largest finite", [maxf, -maxf]),
        ("smallest normal", [tiny, -tiny]),
        ("smallest subnormal", [smallest_sub, -smallest_sub]),
        ("subnormal range", [smallest_sub, 2.0 * smallest_sub, tiny / 2.0, -tiny / 2.0]),
        ("rounding boundary of the result", [1.0, -1.0]),
        ("tie above the boundary", [1.0 + 2.0 ** -f.nmant, -1.0 - 2.0 ** -f.nmant]),
        ("just below the boundary", [1.0 - 2.0 ** -f.nmant, -1.0 + 2.0 ** -f.nmant]),
        ("large magnitudes", [1e10, -1e10, 1e30, -1e30]),
        ("tiny magnitudes", [1e-10, -1e-10, 1e-30, -1e-30]),
    ]


def _value_cases(dtype):
    """Every value family of the float section, as (label, array)."""
    dtype = np.dtype(dtype)
    out = []
    for name, values in _float_value_cases(dtype):
        out.append(("%s: %s" % (dtype.name, name),
                    np.array(values, dtype=dtype)))
    quiet, signalling, sign = NAN_PATTERNS[dtype.itemsize]
    out.append(("%s: NaN operand, quiet, payload 1" % dtype.name,
                _bit_nan(dtype, quiet)))
    out.append(("%s: NaN operand, quiet, payload 1, sign set" % dtype.name,
                _bit_nan(dtype, quiet | sign)))
    out.append(("%s: NaN operand, signalling" % dtype.name,
                _bit_nan(dtype, signalling)))
    return out


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_value_families(impl):
    """Every value family of every type, against ONNX Runtime and the implementation.

    The document's special values (the signed zeros, the infinities, NaN, the subnormal
    range, the smallest and largest values of the type, the rounding boundary and the tie
    above it) are batched into a single ONNX Runtime run per type.
    """
    failures = []
    rows = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        for name, values in _float_value_cases(dt):
            a = np.array(values, dtype=dt)
            ref = run_ort_batch(a)
            got = np.asarray(impl.atan(a))
            if not _bits_equal(ref, got):
                failures.append({
                    "who": "%s: %s" % (dt.name, name),
                    "detail": "ort %s, impl %s" % (ref.tolist(), got.tolist()),
                })
        rows.append(dt.name)
    return {"sweep": "value families (ORT vs implementation)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def sweep_spec_examples(impl):
    """The worked examples of the document, as documented, vs ORT and the implementation."""
    f64 = np.float64
    examples = [
        ("float Example 1",
         np.array([0.0, 1.0, -1.0], dtype=f64),
         np.array([0.0, 0.7853981633974483, -0.7853981633974483], dtype=f64)),
        ("float Example 2",
         np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=f64),
         np.array([0.0, -0.0, 1.5707963267948966, -1.5707963267948966, np.nan],
                  dtype=f64)),
    ]
    failures = []
    for label, a, documented in examples:
        ref = run_ort_batch(a)
        got = np.asarray(impl.atan(a))
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


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact arithmetic.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    implementation against the values the document's Examples state, in exact arithmetic
    (``fractions.Fraction`` for the rationals, and a high-precision computation for the
    transcendental values).
    """
    F = Fraction
    failures = []
    # Example 1: input [0, 1, -1] -> output [0, pi/4, -pi/4]
    # Example 2: input [sqrt(3), 1/sqrt(3)] -> output [pi/3, pi/6]
    # The rational parts are exact; the transcendental parts are checked with a
    # high-precision arctangent computed independently of the platform's library.
    try:
        from mpmath import mp, atan as mp_atan, pi as mp_pi, sqrt as mp_sqrt
        mp.dps = 60
        have_mp = True
    except Exception:
        have_mp = False

    def exact_atan(x):
        """arctan of a Fraction, to high precision, independent of the platform."""
        if not have_mp:
            return None
        return mp_atan(mp.mpf(x.numerator) / mp.mpf(x.denominator))

    examples = [
        ("real Example 1", [F(0), F(1), F(-1)]),
        ("real Example 2", [F(3).sqrt() if False else None, None]),
    ]
    # Example 1, exactly as the document states it.
    a1 = np.array([F(0), F(1), F(-1)], dtype=object)
    got1 = np.asarray(impl.atan(a1))
    want1 = [F(0), None, None]
    if have_mp:
        want1 = [F(0), exact_atan(F(1)), exact_atan(F(-1))]
        for i, (g, w) in enumerate(zip(got1.tolist(), want1)):
            if abs(mp.mpf(g) - w) > mp.mpf(10) ** (-40):
                failures.append({"who": "real Example 1 element %d" % i,
                                 "detail": "exact %s, implementation %s" % (w, g)})
    else:
        # without mpmath, at least the rational element must be exact
        if got1[0] != F(0):
            failures.append({"who": "real Example 1 element 0",
                             "detail": "exact 0, implementation %s" % got1[0]})

    # Example 2: sqrt(3) and 1/sqrt(3) are irrational; check with high precision.
    if have_mp:
        s3 = mp_sqrt(3)
        a2 = np.array([s3, 1 / s3], dtype=object)
        got2 = np.asarray(impl.atan(a2))
        want2 = [mp_pi / 3, mp_pi / 6]
        for i, (g, w) in enumerate(zip(got2.tolist(), want2)):
            if abs(mp.mpf(g) - w) > mp.mpf(10) ** (-40):
                failures.append({"who": "real Example 2 element %d" % i,
                                 "detail": "exact %s, implementation %s" % (w, g)})

    return {"sweep": "real section (exact arithmetic, no ONNX Runtime)",
            "summary": "2 examples, %s" % ("high-precision arctangent" if have_mp
                                           else "rational element only"),
            "failures": failures}


def sweep_odd_and_range(impl):
    """The sign and range properties of the real section, over a sample of the reals.

    The arctangent is odd and its range is the open interval (-pi/2, pi/2); the
    implementation is checked against a high-precision reference on a sample.
    """
    failures = []
    try:
        from mpmath import mp, atan as mp_atan, pi as mp_pi
        mp.dps = 60
    except Exception:
        return {"sweep": "odd and range (real section)",
                "summary": "skipped: mpmath unavailable", "failures": []}

    xs = [0.0, 1e-30, 1e-10, 0.5, 1.0, 2.0, 10.0, 1e10, 1e30,
          -1e-30, -1e-10, -0.5, -1.0, -2.0, -10.0, -1e10, -1e30]
    for x in xs:
        got = float(np.asarray(impl.atan(np.array([x], dtype=np.float64)))[0])
        want = mp_atan(mp.mpf(x))
        if abs(mp.mpf(got) - want) > mp.mpf(10) ** (-15):
            failures.append({"who": "atan(%r)" % x,
                             "detail": "exact %s, implementation %s" % (want, got)})
        if not (abs(got) < float(mp_pi) / 2):
            failures.append({"who": "range at %r" % x,
                             "detail": "result %s outside (-pi/2, pi/2)" % got})
    return {"sweep": "odd and range (real section)",
            "summary": "%d sample points, high-precision reference" % len(xs),
            "failures": failures}


def sweeps():
    return [sweep_value_families, sweep_spec_examples, sweep_real_examples,
            sweep_odd_and_range]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _arr(shape, dtype, start=0.0):
    n = 1
    for d in shape:
        n *= d
    return (np.arange(n, dtype=np.float64) + start).astype(dtype).reshape(shape)


SHAPE_CASES = [
    ("rank-0", ()),
    ("(1,)", (1,)),
    ("(2,3)", (2, 3)),
    ("(2,3,4)", (2, 3, 4)),
    ("zero-sized: (0,)", (0,)),
    ("zero-sized: (0,3)", (0, 3)),
    ("zero-sized: (2,0,4)", (2, 0, 4)),
]


def cases():
    """Every individual case: (label, [A])."""
    for dtype in ALL_FLOAT:
        for name, a in _value_cases(dtype):
            yield (name, [a])
    for name, shape in SHAPE_CASES:
        for dtype in ALL_FLOAT:
            yield ("%s: %s" % (np.dtype(dtype).name, name),
                   [_arr(shape, dtype)])


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

# the real section has no ONNX type: its cases live in the sweeps, not in cases()
COVERAGE["real"] = 2
