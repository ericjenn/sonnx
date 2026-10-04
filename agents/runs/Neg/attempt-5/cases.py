"""The cases with which the **Neg** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled operand per case.  Each case goes through a one-node
                     Neg model (opset 14) and through the implementation, and the two
                     results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the whole domain of the
                     8-bit integer type, the special values of the three floating-point
                     types, the worked examples of every section, and the real section,
                     which has no ONNX type and is checked against exact arithmetic.

Every formula of every section is transcribed here as a literal reading of the document,
independent of the implementation, and the case set grows with the document.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
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

# The NaN payloads used as operands: a quiet NaN, a signalling NaN, and the sign bit.
NAN_PATTERNS = {2: (0x7E01, 0x7C01, 0x8000),
                4: (0x7FC00001, 0x7F800001, 0x80000000),
                8: (0x7FF8000000000001, 0x7FF0000000000001, 0x8000000000000000)}


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(x: np.ndarray) -> np.ndarray:
    """One batched Neg through ONNX Runtime, for the sweeps."""
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
    """Bit for bit for the floating-point types (NaN equal to NaN), by value for ints."""
    if want.shape != got.shape or want.dtype != got.dtype:
        return False
    if want.dtype.kind == "f":
        uint = UINT_OF_WIDTH[want.dtype.itemsize]
        bw = np.ascontiguousarray(want).view(uint)
        bg = np.ascontiguousarray(got).view(uint)
        both_nan = np.isnan(want) & np.isnan(got)
        return bool(np.all((bw == bg) | both_nan))
    return bool(np.all(want == got))


def _first_diffs(want: np.ndarray, got: np.ndarray, n: int = 4) -> str:
    """A short, readable account of the first elements that differ."""
    if want.shape != got.shape or want.dtype != got.dtype:
        return "shape %s vs %s, type %s vs %s" % (want.shape, got.shape,
                                                 want.dtype, got.dtype)
    if want.dtype.kind == "f":
        uint = UINT_OF_WIDTH[want.dtype.itemsize]
        bw = np.ascontiguousarray(want).view(uint).ravel()
        bg = np.ascontiguousarray(got).view(uint).ravel()
        both_nan = np.isnan(want).ravel() & np.isnan(got).ravel()
        idx = np.nonzero(~((bw == bg) | both_nan))[0][:n]
        return "; ".join("index %d: want %s (0x%x), got %s (0x%x)"
                         % (int(i), want.ravel()[i], int(bw[i]),
                            got.ravel()[i], int(bg[i])) for i in idx)
    idx = np.nonzero((want != got).ravel())[0][:n]
    return "; ".join("index %d: want %d, got %d"
                     % (int(i), int(want.ravel()[i]), int(got.ravel()[i]))
                     for i in idx)


def _impl_neg(impl, x):
    """Call the implementation, returning (result, error message)."""
    try:
        return np.asarray(impl.neg(x)), None
    except Exception as exc:  # a sweep must not raise: it is recorded as a failure
        return None, "%s: %s" % (type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# the document's formulas, transcribed literally
# ---------------------------------------------------------------------------
#
# The transcription is deliberately independent of the implementation: it is a second,
# literal, reading of the three sections, compared with ONNX Runtime and with the
# implementation.  A disagreement between the document and ONNX Runtime is a defect of
# the document, not of the implementation.


def doc_int_neg(x: int, n: int) -> int:
    """The case analysis of **Neg** (int), in exact arithmetic.

    Y[i] = -X[i] if -X[i] lies in [-2^(n-1), 2^(n-1)-1], and -X[i] - 2^n otherwise.
    """
    hi = 2 ** (n - 1) - 1
    y = -x
    if y > hi:
        return y - 2 ** n
    return y


def doc_float_neg_bits(x: np.ndarray) -> np.ndarray:
    """The float section, literally: the sign bit is flipped, the magnitude is kept."""
    uint = UINT_OF_WIDTH[x.dtype.itemsize]
    sign = uint(1) << (x.dtype.itemsize * 8 - 1)
    return (np.ascontiguousarray(x).view(uint) ^ sign).view(x.dtype)


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, and as ONNX Runtime does it."""
    failures = []
    checked = 0

    # the real section: X = [1/3, -2.5, 0], Y = [-1/3, 2.5, 0], in exact arithmetic
    F = Fraction
    x = np.array([F(1, 3), F(-5, 2), F(0)], dtype=object)
    documented = np.array([F(-1, 3), F(5, 2), F(0)], dtype=object)
    got, err = _impl_neg(impl, x)
    checked += 1
    if err:
        failures.append({"who": "real Example", "detail": "the implementation raised %s" % err})
    elif got.shape != documented.shape or any(got.ravel()[i] != documented.ravel()[i]
                                              for i in range(documented.size)):
        failures.append({"who": "real Example", "detail": "documented %s, implementation %s"
                         % (documented.tolist(), got.tolist())})

    # float Example 1: X = [1.5, -2.25, 0.0, -0.0], Y = [-1.5, 2.25, -0.0, 0.0]
    x = np.array([1.5, -2.25, 0.0, -0.0], dtype=np.float32)
    documented = np.array([-1.5, 2.25, -0.0, 0.0], dtype=np.float32)
    ref = run_ort_batch(x)
    got, err = _impl_neg(impl, x)
    checked += 1
    if not _bits_equal(documented, ref):
        failures.append({"who": "float Example 1", "detail": "documented %s, ORT %s"
                         % (documented.tolist(), ref.tolist())})
    if err:
        failures.append({"who": "float Example 1", "detail": "the implementation raised %s" % err})
    elif not _bits_equal(documented, got):
        failures.append({"who": "float Example 1", "detail": "documented %s, implementation %s"
                         % (documented.tolist(), got.tolist())})

    # float Example 2: X = [+inf, -inf, NaN], Y = [-inf, +inf, NaN]
    x = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)
    documented = np.array([-np.inf, np.inf, np.nan], dtype=np.float32)
    ref = run_ort_batch(x)
    got, err = _impl_neg(impl, x)
    checked += 1
    if not _bits_equal(documented, ref):
        failures.append({"who": "float Example 2", "detail": "documented %s, ORT %s"
                         % (documented.tolist(), ref.tolist())})
    if err:
        failures.append({"who": "float Example 2", "detail": "the implementation raised %s" % err})
    elif not _bits_equal(documented, got):
        failures.append({"who": "float Example 2", "detail": "documented %s, implementation %s"
                         % (documented.tolist(), got.tolist())})

    # int Example: X = [5, -5, 127, -128], Y = [-5, 5, -127, -128] for int8
    x = np.array([5, -5, 127, -128], dtype=np.int8)
    documented = np.array([-5, 5, -127, -128], dtype=np.int8)
    ref = run_ort_batch(x)
    got, err = _impl_neg(impl, x)
    checked += 1
    if not _bits_equal(documented, ref):
        failures.append({"who": "int Example", "detail": "documented %s, ORT %s"
                         % (documented.tolist(), ref.tolist())})
    if err:
        failures.append({"who": "int Example", "detail": "the implementation raised %s" % err})
    elif not _bits_equal(documented, got):
        failures.append({"who": "int Example", "detail": "documented %s, implementation %s"
                         % (documented.tolist(), got.tolist())})

    return {"sweep": "the document's worked examples",
            "summary": "%d examples, as documented vs ORT vs implementation" % checked,
            "failures": failures}


def _float_specials(dt):
    """Every special value of the type, as a bit pattern: zeros, infinities, NaNs,
    the subnormal range, the smallest normal and the largest finite value."""
    dt = np.dtype(dt)
    f = np.finfo(dt)
    uint = UINT_OF_WIDTH[dt.itemsize]
    bits = dt.itemsize * 8
    sign = uint(1) << (bits - 1)
    quiet, signalling, _ = NAN_PATTERNS[dt.itemsize]
    nmant, nexp = f.nmant, f.nexp
    largest_sub = (1 << nmant) - 1
    smallest_norm = 1 << nmant
    largest_finite = ((1 << nexp) - 2) << nmant | ((1 << nmant) - 1)
    patterns = [
        0, int(sign),                                            # +0 and -0
        int(np.array([np.inf], dtype=dt).view(uint)[0]),         # +inf
        int(np.array([-np.inf], dtype=dt).view(uint)[0]),        # -inf
        quiet, quiet | int(sign), signalling, signalling | int(sign),
        1, 2, largest_sub,                                       # the subnormal range
        smallest_norm, smallest_norm + 1,                        # the smallest normal
        largest_finite - 1, largest_finite,                      # the largest finite
        int(np.array([1.0], dtype=dt).view(uint)[0]),
        int(np.array([-1.0], dtype=dt).view(uint)[0]),
        int(np.array([0.5], dtype=dt).view(uint)[0]),
    ]
    return np.array(patterns, dtype=uint).view(dt)


def sweep_float_special(impl):
    """The special values of the three floating-point types, bit for bit.

    The document says the negation flips the sign and leaves the magnitude unchanged,
    for every value of the type including the special numbers; the sign and the payload
    of a NaN are not specified, so any NaN is a conforming result.
    """
    failures, rows = [], []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        x = _float_specials(dt)
        documented = doc_float_neg_bits(x)
        ref = run_ort_batch(x)
        got, err = _impl_neg(impl, x)
        if err:
            failures.append({"who": dt.name, "detail": "the implementation raised %s" % err})
            continue
        if not _bits_equal(documented, ref):
            failures.append({"who": "%s: the document vs ONNX Runtime" % dt.name,
                             "detail": _first_diffs(documented, ref)})
        if not _bits_equal(documented, got):
            failures.append({"who": "%s: the document vs the implementation" % dt.name,
                             "detail": _first_diffs(documented, got)})
        if not _bits_equal(ref, got):
            failures.append({"who": "%s: ONNX Runtime vs the implementation" % dt.name,
                             "detail": _first_diffs(ref, got)})
        rows.append("%s %d values" % (dt.name, x.size))
    return {"sweep": "the float special values (sign flip, bit for bit)",
            "summary": ", ".join(rows), "failures": failures}


def _rand_int(rng, dt):
    """One random value of this integer type, drawn over the whole domain."""
    n = dt.itemsize * 8
    if n <= 32:
        lo, hi = int(np.iinfo(dt).min), int(np.iinfo(dt).max)
        return int(rng.integers(lo, hi, dtype=np.int64, endpoint=True))
    u = int.from_bytes(rng.bytes(8), "little")
    return int(np.array([u], dtype=np.uint64).view(np.int64)[0])


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
        n = dt.itemsize * 8
        if dt.itemsize == 1:
            xs = np.arange(int(info.min), int(info.max) + 1, dtype=np.int64)
            how = "exhaustive, %d values" % xs.size
        else:
            edges = sorted({int(info.min), int(info.min) + 1, -1, 0, 1,
                            int(info.max) - 1, int(info.max)})
            xs = np.array(edges + [_rand_int(rng, dt) for _ in range(200)], dtype=np.int64)
            how = "%d boundary and random values" % xs.size
        a = xs.astype(dt)

        want = np.array([doc_int_neg(int(v), n) for v in a.ravel()],
                        dtype=dt).reshape(a.shape)          # the document, literally
        ref = run_ort_batch(a)                              # ONNX Runtime
        got, err = _impl_neg(impl, a)                       # the implementation
        if err:
            failures.append({"who": dt.name, "detail": "the implementation raised %s" % err})
            continue
        if not _bits_equal(want, ref):
            failures.append({"who": "%s: the document vs ONNX Runtime" % dt.name,
                             "detail": _first_diffs(want, ref)})
        if not _bits_equal(want, got):
            failures.append({"who": "%s: the document vs the implementation" % dt.name,
                             "detail": _first_diffs(want, got)})
        if not _bits_equal(ref, got):
            failures.append({"who": "%s: ONNX Runtime vs the implementation" % dt.name,
                             "detail": _first_diffs(ref, got)})
        rows.append("%s %s" % (dt.name, how))
    return {"sweep": "the integer domain (document vs ORT vs implementation)",
            "summary": "; ".join(rows), "failures": failures}


def sweep_real_exact(impl):
    """The real section, which has no ONNX type: checked against exact arithmetic.

    ONNX Runtime cannot be asked about a real tensor, so the implementation is checked
    against the exact negation of the operand: `fractions.Fraction` for the rationals of
    the document's Example, and `decimal.Decimal` at 60 digits for the transcendental
    values, whose negation is exact as well.
    """
    getcontext().prec = 60
    F = Fraction
    failures, checked = [], 0

    rationals = [
        ("the document's Example", [F(1, 3), F(-5, 2), F(0)]),
        ("signed zeros and integers", [F(0), F(1), F(-1)]),
        ("rationals", [F(7, 3), F(-22, 7), F(1, 10 ** 6), F(-10 ** 6, 3)]),
        ("a rank-0 tensor", [F(3, 4)]),
    ]
    for label, values in rationals:
        x = np.array(values, dtype=object)
        want = np.array([-v for v in values], dtype=object)
        got, err = _impl_neg(impl, x)
        if err:
            failures.append({"who": label, "detail": "the implementation raised %s" % err})
            continue
        checked += len(values)
        if got.shape != want.shape or any(got.ravel()[i] != want.ravel()[i]
                                          for i in range(want.size)):
            failures.append({"who": label, "detail": "exact %s, implementation %s"
                             % (want.tolist(), got.tolist())})

    pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798")
    e = Decimal("2.71828182845904523536028747135266249775724709369995957496696762772407663035354759457138217852516642743")
    values = [pi, e, -pi, -e]
    x = np.array(values, dtype=object)
    want = np.array([-v for v in values], dtype=object)
    got, err = _impl_neg(impl, x)
    if err:
        failures.append({"who": "pi and e at 60 digits",
                         "detail": "the implementation raised %s" % err})
    else:
        checked += len(values)
        if got.shape != want.shape or any(got.ravel()[i] != want.ravel()[i]
                                          for i in range(want.size)):
            failures.append({"who": "pi and e at 60 digits",
                             "detail": "exact %s, implementation %s"
                             % (want.tolist(), got.tolist())})

    return {"sweep": "the real section (exact rationals and high-precision decimals, "
                     "no ONNX Runtime)",
            "summary": "%d values checked against exact arithmetic" % checked,
            "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_float_special, sweep_int_domain, sweep_real_exact]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _float_cases(dtype):
    """The value families of the float section, one array per case."""
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    uint = UINT_OF_WIDTH[dt.itemsize]
    bits = dt.itemsize * 8
    sign = uint(1) << (bits - 1)
    quiet, signalling, _ = NAN_PATTERNS[dt.itemsize]
    nmant, nexp = f.nmant, f.nexp
    largest_sub = (1 << nmant) - 1
    smallest_norm = 1 << nmant
    largest_finite = ((1 << nexp) - 2) << nmant | ((1 << nmant) - 1)

    def from_bits(patterns):
        return np.array(patterns, dtype=uint).view(dt)

    yield ("signed zeros", np.array([0.0, -0.0, 0.0, -0.0], dtype=dt))
    yield ("infinities", np.array([np.inf, -np.inf, np.inf, -np.inf], dtype=dt))
    yield ("NaN, quiet", from_bits([quiet]))
    yield ("NaN, quiet with the sign bit set", from_bits([quiet | int(sign)]))
    yield ("NaN, signalling", from_bits([signalling]))
    yield ("NaN, signalling with the sign bit set", from_bits([signalling | int(sign)]))
    yield ("NaN, several payloads", from_bits([quiet, quiet + 1, quiet + 2, signalling]))
    yield ("the subnormal range", from_bits([1, 2, largest_sub - 1, largest_sub]))
    yield ("the smallest normal and its neighbours",
           from_bits([smallest_norm - 1, smallest_norm, smallest_norm + 1]))
    yield ("the largest finite value and its neighbour",
           from_bits([largest_finite - 1, largest_finite]))
    yield ("the document's Example 1", np.array([1.5, -2.25, 0.0, -0.0], dtype=dt))
    yield ("the document's Example 2", np.array([np.inf, -np.inf, np.nan], dtype=dt))
    one = dt.type(1.0)
    yield ("the values around 1.0 (Neg does not round)",
           np.array([one, np.nextafter(one, dt.type(2.0)),
                     np.nextafter(one, dt.type(0.0)),
                     dt.type(1.0 + 2.0 ** -nmant)], dtype=dt))
    yield ("ordinary values", np.array([1.0, -2.5, 0.25, -0.125, 3.0], dtype=dt))


def _int_cases(dtype):
    """The value families of the int section, one array per case."""
    dt = np.dtype(dtype)
    info = np.iinfo(dt)
    lo, hi = int(info.min), int(info.max)
    yield ("the document's Example values", np.array([5, -5, 127, -128], dtype=dt))
    yield ("the minimum and the maximum", np.array([lo, hi], dtype=dt))
    yield ("the minimum value alone", np.array([lo], dtype=dt))
    yield ("the maximum value alone", np.array([hi], dtype=dt))
    yield ("zero, one and minus one", np.array([0, 1, -1], dtype=dt))
    yield ("the neighbours of the minimum", np.array([lo, lo + 1, lo + 2], dtype=dt))
    yield ("the neighbours of the maximum", np.array([hi, hi - 1, hi - 2], dtype=dt))
    yield ("a run of small values", np.array(list(range(-8, 9)), dtype=dt))


SHAPES = [
    ("rank-0 (the tensor with no dimension)", ()),
    ("one element", (1,)),
    ("the document's Example shape", (3,)),
    ("(2, 3)", (2, 3)),
    ("(2, 3, 4)", (2, 3, 4)),
    ("a zero-sized dimension: (0,)", (0,)),
    ("a zero-sized dimension: (0, 3)", (0, 3)),
    ("a zero-sized dimension: (3, 0)", (3, 0)),
    ("a zero-sized dimension: (1, 0, 2)", (1, 0, 2)),
    ("two zero-sized dimensions: (0, 0)", (0, 0)),
    ("a zero-sized dimension in the middle: (2, 0, 3)", (2, 0, 3)),
]


def _arr(shape, dtype, start=0.0):
    n = 1
    for d in shape:
        n *= d
    return (np.arange(n, dtype=np.float64) + start).astype(dtype).reshape(shape)


def _real_cases():
    """The real section, which has no ONNX type: exact rationals as objects."""
    F = Fraction
    yield ("the document's Example", np.array([F(1, 3), F(-5, 2), F(0)], dtype=object))
    yield ("signed zeros and integers", np.array([F(0), F(1), F(-1)], dtype=object))
    yield ("rationals", np.array([F(7, 3), F(-22, 7), F(1, 10 ** 6)], dtype=object))
    yield ("a rank-0 tensor", np.array(F(3, 4), dtype=object).reshape(()))
    yield ("a zero-sized dimension", np.zeros((0,), dtype=object))


def _build_cases():
    out = []
    for dtype in ALL_FLOAT:
        for name, arr in _float_cases(dtype):
            out.append(("float", "%s: %s" % (np.dtype(dtype).name, name), [arr]))
    for dtype in ALL_INT:
        for name, arr in _int_cases(dtype):
            out.append(("int", "%s: %s" % (np.dtype(dtype).name, name), [arr]))
    for name, shape in SHAPES:
        for dtype in ALL_FLOAT:
            out.append(("float", "%s: %s" % (np.dtype(dtype).name, name),
                        [_arr(shape, dtype)]))
        for dtype in ALL_INT:
            out.append(("int", "%s: %s" % (np.dtype(dtype).name, name),
                        [_arr(shape, dtype)]))
    for name, arr in _real_cases():
        out.append(("real", "real: %s" % name, [arr]))
    return out


_CASES = _build_cases()

COVERAGE = {"real": 0, "float": 0, "int": 0}
for _section, _label, _arrays in _CASES:
    COVERAGE[_section] = COVERAGE.get(_section, 0) + 1


def cases():
    """Every individual case: (label, [X])."""
    for _section, label, arrays in _CASES:
        yield (label, arrays)
