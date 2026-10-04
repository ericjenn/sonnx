"""The cases with which the **Neg** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled operand per case.  Each case goes through a one-node
                     Neg model (opset 14) and through the implementation, and the two
                     results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the domain of the integer
                     types, the special values of the floating-point types, the worked
                     examples of every section, and the real section, which has no ONNX
                     type and is checked against exact rational arithmetic instead.

Every case of every section's formula is meant to appear here, and the case set is part
of the loop: when a section is added to the document, its cases are added here before
the next run.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Neg"
OPSET = 14
IR_VERSION = 10

ALL_FLOAT = [np.float16, np.float32, np.float64]
ALL_INT = [np.int8, np.int16, np.int32, np.int64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}

# one quiet, one signalling, one with the sign bit set, and two payload extremes
NAN_BITS = {
    2: [0x7E00, 0x7C01, 0xFE00, 0x7FFF, 0xFC01],
    4: [0x7FC00000, 0x7F800001, 0xFFC00000, 0x7FFFFFFF, 0xFF800001],
    8: [0x7FF8000000000000, 0x7FF0000000000001, 0xFFF8000000000000,
        0x7FFFFFFFFFFFFFFF, 0xFFF0000000000001],
}


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(x: np.ndarray) -> np.ndarray:
    """One Neg node, one input, one output, run once."""
    proto = helper.np_dtype_to_tensor_dtype(x.dtype)
    vi_x = helper.make_tensor_value_info("X", proto, list(x.shape))
    vi_y = helper.make_tensor_value_info("Y", proto, None)
    node = helper.make_node(OP, ["X"], ["Y"], name="neg0")
    graph = helper.make_graph([node], "neg_graph", [vi_x], [vi_y])
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


def _first_float_diff(want, got, x):
    if want.shape != got.shape or want.dtype != got.dtype:
        return "shape %s vs %s, dtype %s vs %s" % (want.shape, got.shape,
                                                   want.dtype, got.dtype)
    uint = UINT_OF_WIDTH[want.dtype.itemsize]
    bw = np.ascontiguousarray(want).view(uint)
    bg = np.ascontiguousarray(got).view(uint)
    both_nan = np.isnan(want) & np.isnan(got)
    bad = np.nonzero(~((bw == bg) | both_nan))[0][:5]
    return "; ".join("x=%r want=%r got=%r"
                     % (x.ravel()[i], want.ravel()[i], got.ravel()[i]) for i in bad)


def _first_int_diff(want, got, x):
    if want.shape != got.shape or want.dtype != got.dtype:
        return "shape %s vs %s, dtype %s vs %s" % (want.shape, got.shape,
                                                   want.dtype, got.dtype)
    bad = np.nonzero(np.asarray(want).astype(np.int64)
                     != np.asarray(got).astype(np.int64))[0][:5]
    return "; ".join("x=%d want=%d got=%d"
                     % (int(x.ravel()[i]), int(want.ravel()[i]), int(got.ravel()[i]))
                     for i in bad)


# ---------------------------------------------------------------------------
# the case analyses of the document, transcribed literally
# ---------------------------------------------------------------------------
#
# The transcription is deliberately independent of the implementation: it is a
# second, literal, reading of the three sections, compared with ONNX Runtime over a
# whole domain.  A disagreement here is a defect of the document, not of the
# implementation.


def doc_int_neg(x: int, n: int) -> int:
    """The case analysis of **Neg** (int), in exact arithmetic.

    Y[i] = -X[i] if X[i] > -2^(n-1), and -2^(n-1) if X[i] = -2^(n-1): the minimum value
    of the type is its own negation.
    """
    lo = -(2 ** (n - 1))
    if x == lo:
        return lo
    return -x


def _doc_int_array(a: np.ndarray) -> np.ndarray:
    n = a.dtype.itemsize * 8
    flat = [doc_int_neg(int(v), n) for v in a.ravel()]
    return np.array(flat, dtype=a.dtype).reshape(a.shape)


def _doc_float_neg(a: np.ndarray) -> np.ndarray:
    """The float section: the value of the same magnitude and the opposite sign.

    The negation is exact, so the result is the operand with its sign bit flipped; a
    NaN operand gives a NaN, whose sign and payload the document leaves unspecified.
    """
    dt = a.dtype
    uint = UINT_OF_WIDTH[dt.itemsize]
    sign = uint(1) << (dt.itemsize * 8 - 1)
    b = np.ascontiguousarray(a).view(uint) ^ sign
    return b.view(dt)


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def _rand_int(rng, dtype):
    """One random value of this signed integer type, drawn over the whole domain."""
    dt = np.dtype(dtype)
    n = dt.itemsize
    if n == 8:
        hi = int(rng.integers(0, 2 ** 32, dtype=np.uint64))
        lo = int(rng.integers(0, 2 ** 32, dtype=np.uint64))
        bits = (hi << 32) | lo
    else:
        bits = int(rng.integers(0, 2 ** (8 * n), dtype=np.uint64))
    return int(np.array([bits], dtype=UINT_OF_WIDTH[n]).view(dt)[0])


def sweep_int_domain(impl):
    """The document's integer case analysis against ONNX Runtime, over a domain.

    The 8-bit type is covered exhaustively (all 256 values); the wider types use their
    boundary values and 200 random values.
    """
    rng = np.random.default_rng(20261003)
    rows, failures = [], []
    for dtype in ALL_INT:
        dt = np.dtype(dtype)
        info = np.iinfo(dt)
        exhaustive = dt.itemsize == 1
        if exhaustive:
            x = np.arange(int(info.min), int(info.max) + 1, dtype=np.int64).astype(dt)
        else:
            edges = sorted({int(info.min), int(info.min) + 1, int(info.min) + 2,
                            -1, 0, 1, int(info.max) - 1, int(info.max)})
            vals = [int(v) for v in edges]
            vals += [_rand_int(rng, dt) for _ in range(200)]
            x = np.array(vals, dtype=dt)
        want = _doc_int_array(x)          # the document, literally
        ref = run_ort_batch(x)            # ONNX Runtime
        got = np.asarray(impl.neg(x))     # the implementation
        if not _bits_equal(want, ref):
            failures.append({"who": "%s: the document vs ONNX Runtime" % dt.name,
                             "detail": _first_int_diff(want, ref, x)})
        if not _bits_equal(ref, got):
            failures.append({"who": "%s: ONNX Runtime vs the implementation" % dt.name,
                             "detail": _first_int_diff(ref, got, x)})
        rows.append("%s %s"
                    % (dt.name, "exhaustive" if exhaustive else "%d values" % x.size))
    return {"sweep": "int domain (document vs ORT vs implementation)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def _float_special_array(dtype):
    """Every special value of the type, plus the neighbours of the boundaries."""
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    uint = UINT_OF_WIDTH[dt.itemsize]
    smallest_sub = np.array([1], dtype=uint).view(dt)[0]
    largest_sub = np.array([(uint(1) << f.nmant) - 1], dtype=uint).view(dt)[0]
    one = np.array([1.0], dtype=dt)[0]
    two = np.array([2.0], dtype=dt)[0]
    zero = np.array([0.0], dtype=dt)[0]
    vals = [
        0.0, -0.0,                                   # the signed zeros
        np.inf, -np.inf,                             # the infinities
        1.0, -1.0,
        float(f.max), -float(f.max),                 # the largest finite value
        float(f.tiny), -float(f.tiny),               # the smallest normal value
        float(smallest_sub), -float(smallest_sub),   # the smallest subnormal value
        float(largest_sub), -float(largest_sub),     # the largest subnormal value
        float(np.nextafter(one, two)),               # the boundary above 1.0
        float(np.nextafter(one, zero)),              # the boundary below 1.0
        float(np.nextafter(two, one)),               # the boundary below 2.0
        float(np.nextafter(zero, one)),              # the boundary above 0.0
    ]
    arr = np.array(vals, dtype=dt)
    nan_arr = np.array(NAN_BITS[dt.itemsize], dtype=uint).view(dt)
    return np.concatenate([arr, nan_arr])


def sweep_float_specials(impl):
    """The special values of the three floating-point types, batched.

    The document's float section is a sign flip: the result has the magnitude of the
    operand and the opposite sign, and a NaN operand gives a NaN.  The expected value
    is therefore the operand with its sign bit flipped, compared with ONNX Runtime and
    with the implementation; a NaN is compared as a NaN, its payload being unspecified
    by the document.
    """
    failures, rows = [], []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        x = _float_special_array(dt)
        want = _doc_float_neg(x)
        ref = run_ort_batch(x)
        got = np.asarray(impl.neg(x))
        if not _bits_equal(want, ref):
            failures.append({"who": "%s: the document vs ONNX Runtime" % dt.name,
                             "detail": _first_float_diff(want, ref, x)})
        if not _bits_equal(want, got):
            failures.append({"who": "%s: the document vs the implementation" % dt.name,
                             "detail": _first_float_diff(want, got, x)})
        rows.append("%s %d values" % (dt.name, x.size))
    return {"sweep": "float special values (document vs ORT vs implementation)",
            "summary": "%s" % ", ".join(rows),
            "failures": failures}


def _spec_examples():
    """The worked examples of the float and the integer sections, as documented."""
    f16, f32 = np.float16, np.float32
    return [
        ("float Example 1",
         np.array([1.5, 0.0, -0.0, np.inf, -np.inf, np.nan], dtype=f32),
         np.array([-1.5, -0.0, 0.0, -np.inf, np.inf, np.nan], dtype=f32)),
        ("float Example 2 (float16)",
         np.array([65504.0, 2.0 ** -24, 1.0], dtype=f16),
         np.array([-65504.0, -(2.0 ** -24), -1.0], dtype=f16)),
        ("int Example (int8)",
         np.array([-6, 0, 100, -128], dtype=np.int8),
         np.array([6, 0, -100, -128], dtype=np.int8)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, and as ONNX Runtime does it."""
    failures = []
    for label, x, documented in _spec_examples():
        ref = run_ort_batch(x)
        got = np.asarray(impl.neg(x))
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


def _real_cases():
    """The real section, as exact rationals: (label, X, the documented Y)."""
    F = Fraction
    return [
        ("real Example (X = [-3, 0, 1/3])",
         [F(-3), F(0), F(1, 3)], [F(3), F(0), F(-1, 3)]),
        ("real rank-0 (X = 7/5)", [F(7, 5)], [F(-7, 5)]),
        ("real zero-sized (X = [])", [], []),
        ("real rationals not representable in binary",
         [F(1, 3), F(-2, 7), F(5, 11), F(-13, 17), F(0)],
         [F(-1, 3), F(2, 7), F(-5, 11), F(13, 17), F(0)]),
        ("real large rationals",
         [F(10 ** 20 + 1, 3), F(-(10 ** 20 + 1), 7)],
         [F(-(10 ** 20 + 1), 3), F(10 ** 20 + 1, 7)]),
        ("real halves (exactly representable)",
         [F(1, 2), F(-1, 2), F(3, 2)], [F(-1, 2), F(1, 2), F(-3, 2)]),
    ]


def sweep_real_exact(impl):
    """The real section, which has no ONNX type: checked against exact rationals.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    implementation against the exact negation in ``fractions.Fraction``, including the
    document's own Example (X = [-3, 0, 1/3], Y = [3, 0, -1/3]) and the rank-0 and
    zero-sized shapes.  The section has no transcendental value, so no high-precision
    library is needed beyond exact rational arithmetic.
    """
    failures = []
    checks = 0
    for label, x_list, want_list in _real_cases():
        x = np.array(x_list, dtype=object)
        want = np.array(want_list, dtype=object)
        checks += 1
        try:
            got = np.asarray(impl.neg(x))
        except Exception as exc:
            failures.append({"who": label,
                             "detail": "the implementation raised %s: %s"
                                       % (type(exc).__name__, exc)})
            continue
        if got.shape != want.shape:
            failures.append({"who": label, "detail": "shape %s, expected %s"
                             % (got.shape, want.shape)})
            continue
        bad = [i for i in range(want.size) if got.ravel()[i] != want.ravel()[i]]
        if bad:
            failures.append({"who": label, "detail": "; ".join(
                "neg(%s) = %s, exact %s"
                % (x.ravel()[i], got.ravel()[i], want.ravel()[i]) for i in bad[:5])})
    return {"sweep": "real section (exact rationals, no ONNX Runtime)",
            "summary": "%d checks against fractions.Fraction; the section has no "
                       "transcendental value, so no high-precision library is needed"
                       % checks,
            "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_float_specials, sweep_int_domain,
            sweep_real_exact]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _float_value_cases(dtype):
    """The value families of the float section, as operands of the type."""
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    uint = UINT_OF_WIDTH[dt.itemsize]
    smallest_sub = np.array([1], dtype=uint).view(dt)[0]
    largest_sub = np.array([(uint(1) << f.nmant) - 1], dtype=uint).view(dt)[0]
    one = np.array([1.0], dtype=dt)[0]
    two = np.array([2.0], dtype=dt)[0]
    zero = np.array([0.0], dtype=dt)[0]
    inf = np.array([np.inf], dtype=dt)[0]
    nan = np.array([np.nan], dtype=dt)[0]
    out = [
        ("signed zeros", [0.0, -0.0]),
        ("infinities", [inf, -inf]),
        ("NaN", [nan]),
        ("largest finite value", [float(f.max), -float(f.max)]),
        ("smallest normal value", [float(f.tiny), -float(f.tiny)]),
        ("smallest subnormal value", [float(smallest_sub), -float(smallest_sub)]),
        ("largest subnormal value", [float(largest_sub), -float(largest_sub)]),
        ("one and its neighbours",
         [1.0, float(np.nextafter(one, two)), float(np.nextafter(one, zero)), -1.0]),
        ("rounding boundary above 2.0", [float(np.nextafter(two, one)), 2.0]),
        ("ordinary", [1.5, -2.5, 0.25, -0.125]),
        ("float Example 1", [1.5, 0.0, -0.0, inf, -inf, nan]),
    ]
    if dt == np.dtype(np.float16):
        out.append(("float Example 2 (float16)", [65504.0, 2.0 ** -24, 1.0]))
    return out


def _float_nan_cases(dtype):
    """The NaN payloads of the type, as operands."""
    dt = np.dtype(dtype)
    uint = UINT_OF_WIDTH[dt.itemsize]
    return [("NaN payload %d" % i, np.array([bits], dtype=uint).view(dt))
            for i, bits in enumerate(NAN_BITS[dt.itemsize])]


def _int_value_cases(dtype):
    """The value families of the integer section, as operands of the type."""
    dt = np.dtype(dtype)
    info = np.iinfo(dt)
    lo, hi = int(info.min), int(info.max)
    out = [
        ("minimum value (its own negation)", [lo]),
        ("minimum value + 1", [lo + 1]),
        ("minus one", [-1]),
        ("zero", [0]),
        ("one", [1]),
        ("maximum value - 1", [hi - 1]),
        ("maximum value", [hi]),
        ("extremes together", [lo, lo + 1, -1, 0, 1, hi - 1, hi]),
    ]
    if dt == np.dtype(np.int8):
        out.append(("int Example (int8)", [-6, 0, 100, -128]))
    return out


SHAPES = [
    ("rank-0", ()),
    ("zero-sized (0,)", (0,)),
    ("zero-sized (0,3)", (0, 3)),
    ("zero-sized (2,0)", (2, 0)),
    ("zero-sized (0,0)", (0, 0)),
    ("vector (3,)", (3,)),
    ("matrix (2,3)", (2, 3)),
    ("rank-3 (2,3,4)", (2, 3, 4)),
    ("rank-4 (2,1,3,1)", (2, 1, 3, 1)),
]


def _arr(shape, dtype):
    n = 1
    for d in shape:
        n *= d
    return np.arange(n, dtype=np.float64).astype(dtype).reshape(shape)


def _build_cases():
    out = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        for name, values in _float_value_cases(dt):
            out.append(("float", "%s: %s" % (dt.name, name),
                        np.array(values, dtype=dt)))
        for name, array in _float_nan_cases(dt):
            out.append(("float", "%s: %s" % (dt.name, name), array))
    for dtype in ALL_INT:
        dt = np.dtype(dtype)
        for name, values in _int_value_cases(dt):
            out.append(("int", "%s: %s" % (dt.name, name),
                        np.array(values, dtype=dt)))
    for name, shape in SHAPES:
        for dtype in ALL_FLOAT + ALL_INT:
            dt = np.dtype(dtype)
            out.append(("float" if dt.kind == "f" else "int",
                        "%s: %s" % (dt.name, name), _arr(shape, dt)))
    return out


_CASES = _build_cases()


def cases():
    """Every individual case: (label, [X])."""
    for _section, label, array in _CASES:
        yield (label, [array])


# ---------------------------------------------------------------------------
# coverage: every anchor of the document's Contents list
# ---------------------------------------------------------------------------

_FLOAT_SWEEP_CHECKS = len(ALL_FLOAT) + sum(
    1 for _, x, _ in _spec_examples() if x.dtype.kind == "f")
_INT_SWEEP_CHECKS = len(ALL_INT) + sum(
    1 for _, x, _ in _spec_examples() if x.dtype.kind == "i")
_REAL_SWEEP_CHECKS = len(_real_cases())

COVERAGE = {"real": _REAL_SWEEP_CHECKS,
            "float": _FLOAT_SWEEP_CHECKS,
            "int": _INT_SWEEP_CHECKS}
for _section, _label, _array in _CASES:
    COVERAGE[_section] += 1
