"""The cases with which the **Abs** specification is checked.

Read by ``specloop.py``:

  * ``cases()``  -- one labelled operand per case, run through a one-node Abs model
                    (opset 14) and through the implementation, and the two results
                    compared bit for bit;
  * ``sweeps()`` -- the bulk checks: the float rule (sign bit cleared) over every
                    float16 pattern and over the boundaries and a sample of float32 /
                    float64, the integer case analyses over every 8- and 16-bit value
                    and over the boundaries of the wider types, the worked examples of
                    the four sections, the real section (exact rationals, no ONNX type)
                    and the document's type families against the ONNX schema's.

**Abs is exact** -- the document specifies a cleared sign bit, an identity on the
unsigned types and the minimum of a signed type mapped to itself -- so every check here
is a check of an exactly stated value and not of an accuracy.  The one thing the
document leaves open is the sign and payload of a NaN result, so a NaN is compared as a
NaN, never by its bits.

The real section has no ONNX type and cannot go through a one-node model; its cases live
in ``sweep_real_examples`` and are counted under the ``real`` anchor of ``COVERAGE``.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Abs"
OPSET = 14
IR_VERSION = 10

UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}
ALL_FLOAT = (np.float16, np.float32, np.float64)
ALL_INT = (np.int8, np.int16, np.int32, np.int64)
ALL_UINT = (np.uint8, np.uint16, np.uint32, np.uint64)


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_abs(a: np.ndarray) -> np.ndarray:
    """One Abs node of opset 14, one run, over the whole array."""
    a = np.asarray(a)
    proto = helper.np_dtype_to_tensor_dtype(a.dtype)
    vi_x = helper.make_tensor_value_info("X", proto, list(a.shape))
    vi_y = helper.make_tensor_value_info("Y", proto, None)
    node = helper.make_node(OP, ["X"], ["Y"], name="abs0")
    graph = helper.make_graph([node], "abs_graph", [vi_x], [vi_y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    (y,) = sess.run(["Y"], {"X": a})
    return y


# ---------------------------------------------------------------------------
# the document's rules, transcribed literally and independently
# ---------------------------------------------------------------------------


def _sign_cleared(a: np.ndarray) -> np.ndarray:
    """**Abs** (float): the operand with its sign bit cleared.

    This is the document's rule read literally off the bit pattern: NaN -> NaN,
    +-inf -> +inf, -0 -> +0, and every finite value keeps its magnitude exactly.
    """
    a = np.asarray(a)
    dt = a.dtype
    uint = UINT_OF_WIDTH[dt.itemsize]
    flat = np.reshape(np.ascontiguousarray(a), -1).view(uint)
    sign = uint(1) << (dt.itemsize * 8 - 1)
    return (flat & ~sign).view(dt).reshape(a.shape)


def _doc_int_abs(a: np.ndarray) -> np.ndarray:
    """**Abs** (int): the three-case analysis of [E_ABS_INT_FUNC_0010]."""
    a = np.asarray(a)
    dt = a.dtype
    n = dt.itemsize * 8
    lo = -(1 << (n - 1))
    out = []
    for v in a.ravel().tolist():
        v = int(v)
        if v >= 0:
            out.append(v)
        elif v > lo:
            out.append(-v)
        else:
            out.append(lo)
    return np.array(out, dtype=dt).reshape(a.shape)


def _doc_uint_abs(a: np.ndarray) -> np.ndarray:
    """**Abs** (uint): the identity of [E_ABS_UINT_FUNC_0010]."""
    return np.array(np.asarray(a), copy=True)


def _float_layout(dtype) -> tuple:
    """(exponent bits, mantissa bits) of a binary interchange format."""
    return {16: (5, 10), 32: (8, 23), 64: (11, 52)}[np.dtype(dtype).itemsize * 8]


def _float_patterns(dtype) -> dict:
    """The special bit patterns of the type, computed from its width."""
    dt = np.dtype(dtype)
    w = dt.itemsize * 8
    e, m = _float_layout(dt)
    sign = 1 << (w - 1)
    exp_all_ones = ((1 << e) - 1) << m
    max_finite = (((1 << e) - 2) << m) | ((1 << m) - 1)
    return {
        "quiet_nan": exp_all_ones | (1 << (m - 1)),
        "quiet_nan_neg": sign | exp_all_ones | (1 << (m - 1)),
        "signalling_nan": exp_all_ones | 1,
        "min_subnormal": 1,
        "min_subnormal_neg": sign | 1,
        "max_subnormal": (1 << m) - 1,
        "min_normal": 1 << m,
        "min_normal_neg": sign | (1 << m),
        "max_finite": max_finite,
        "max_finite_neg": sign | max_finite,
    }


def _bits_array(dtype, pattern) -> np.ndarray:
    """A one-element array of the given type, built from its bit pattern."""
    dt = np.dtype(dtype)
    uint = UINT_OF_WIDTH[dt.itemsize]
    return np.array([pattern], dtype=uint).view(dt)


# ---------------------------------------------------------------------------
# comparison helpers (NaN == NaN; the bits otherwise)
# ---------------------------------------------------------------------------


def _diff_mask(want, got):
    """Flat boolean mask of the elements that differ, or None if not comparable."""
    want = np.asarray(want)
    got = np.asarray(got)
    if want.shape != got.shape or want.dtype != got.dtype:
        return None
    if want.dtype.kind == "f":
        uint = UINT_OF_WIDTH[want.dtype.itemsize]
        bw = np.reshape(np.ascontiguousarray(want), -1).view(uint)
        bg = np.reshape(np.ascontiguousarray(got), -1).view(uint)
        both_nan = np.isnan(want).ravel() & np.isnan(got).ravel()
        return ~((bw == bg) | both_nan)
    return ~(np.ravel(want) == np.ravel(got))


def _same(want, got) -> bool:
    mask = _diff_mask(want, got)
    return mask is not None and not bool(np.any(mask))


def _first_diffs(want, got, limit=5) -> str:
    mask = _diff_mask(want, got)
    if mask is None:
        return ("shape %s vs %s, dtype %s vs %s"
                % (np.shape(want), np.shape(got),
                   np.asarray(want).dtype, np.asarray(got).dtype))
    idx = np.nonzero(mask)[0][:limit]
    w = np.ravel(np.asarray(want))
    g = np.ravel(np.asarray(got))
    return "; ".join("index %d: %r vs %r" % (int(i), w[i], g[i]) for i in idx)


def _random_values(rng, dtype, n):
    """Random values over the whole domain of an integer type."""
    dt = np.dtype(dtype)
    w = dt.itemsize * 8
    if w <= 32:
        if dt.kind == "i":
            return rng.integers(-(1 << (w - 1)), 1 << (w - 1), size=n,
                                dtype=np.int64).astype(dt)
        return rng.integers(0, 1 << w, size=n, dtype=np.uint64).astype(dt)
    hi = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    lo = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    return ((hi << np.uint64(32)) | lo).astype(dt)


def _random_patterns(rng, dtype, n):
    """Random bit patterns of a floating-point type, as an array of that type."""
    dt = np.dtype(dtype)
    w = dt.itemsize * 8
    uint = UINT_OF_WIDTH[dt.itemsize]
    if w == 32:
        return rng.integers(0, 1 << 32, size=n, dtype=np.uint64).astype(uint).view(dt)
    hi = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    lo = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    return ((hi << np.uint64(32)) | lo).astype(uint).view(dt)


# ---------------------------------------------------------------------------
# the real section: exact rational arithmetic, no ONNX type
# ---------------------------------------------------------------------------

_REAL_EXAMPLES = [
    ("real Example 1",
     [Fraction(7, 2), Fraction(-2, 1), Fraction(0, 1), Fraction(-29, 4)],
     [Fraction(7, 2), Fraction(2, 1), Fraction(0, 1), Fraction(29, 4)]),
    ("real, dyadic values of both signs",
     [Fraction(-1, 2), Fraction(1, 4), Fraction(-1024, 1), Fraction(-1, 1048576)],
     [Fraction(1, 2), Fraction(1, 4), Fraction(1024, 1), Fraction(1, 1048576)]),
    ("real, the exact magnitude of a negative rational",
     [Fraction(-1, 3), Fraction(-5, 7), Fraction(-22, 7)],
     [Fraction(1, 3), Fraction(5, 7), Fraction(22, 7)]),
]


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_type_families(impl):
    """The document's declared types against the ONNX schema's type constraints."""
    failures = []
    doc_types = ("tensor(float16)", "tensor(float)", "tensor(double)",
                 "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)",
                 "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)")
    try:
        import onnx
        schema = onnx.defs.get_schema(OP, OPSET)
    except Exception as exc:
        return {"sweep": "the document's type families against the ONNX schema's",
                "summary": "the schema of %s at opset %d could not be read: %s: %s"
                           % (OP, OPSET, type(exc).__name__, exc),
                "failures": []}
    allowed = set()
    params = []
    for c in schema.type_constraints:
        params.append(c.type_param_str)
        allowed.update(c.allowed_type_strs)
    missing = [t for t in doc_types if t not in allowed]
    if missing:
        failures.append({
            "who": "the document's declared types vs the schema's",
            "detail": "the document announces %s, which the schema of %s at opset %d "
                      "does not admit; it admits %s"
                      % (", ".join(missing), OP, OPSET, ", ".join(sorted(allowed))),
        })
    extra = sorted(allowed - set(doc_types))
    return {"sweep": "the document's type families against the ONNX schema's",
            "summary": "document: %d types, schema: %d over %s; admitted by the schema "
                       "and not specified here: %s"
                       % (len(doc_types), len(allowed), ", ".join(params),
                          ", ".join(extra) if extra else "none"),
            "failures": failures}


def sweep_spec_examples(impl):
    """The worked examples of the float, int and uint sections, as documented."""
    f32 = np.float32
    examples = [
        ("float Example 1",
         np.array([-1.5, 2.25, 0.0, -0.0, -np.inf, np.inf, np.nan], dtype=f32),
         np.array([1.5, 2.25, 0.0, 0.0, np.inf, np.inf, np.nan], dtype=f32)),
        ("float Example 2",
         np.array([[-3.0, -1.0], [-2.5, 4.0]], dtype=f32),
         np.array([[3.0, 1.0], [2.5, 4.0]], dtype=f32)),
        ("int Example 1 (int8)",
         np.array([-6, 100, 127, 0, -127, -128], dtype=np.int8),
         np.array([6, 100, 127, 0, 127, -128], dtype=np.int8)),
        ("uint Example 1 (uint8)",
         np.array([0, 6, 200, 255], dtype=np.uint8),
         np.array([0, 6, 200, 255], dtype=np.uint8)),
    ]
    failures = []
    for label, x, documented in examples:
        got = np.asarray(impl.abs(x))
        if not _same(documented, got):
            failures.append({"who": label + ": the implementation vs the documented "
                                            "result",
                             "detail": _first_diffs(documented, got)})
        try:
            ref = run_ort_abs(x)
        except Exception as exc:
            continue
        if not _same(documented, ref):
            failures.append({"who": label + ": the document vs ONNX Runtime",
                             "detail": _first_diffs(documented, ref)})
    return {"sweep": "the document's worked examples",
            "summary": "%d examples (float 2, int 1, uint 1), as documented vs ONNX "
                       "Runtime vs the implementation; the real section's example is "
                       "checked in the real sweep, which has no ONNX type" % len(examples),
            "failures": failures}


def sweep_float_exact(impl):
    """The float rule (sign bit cleared) over the values of each type.

    float16 is covered exhaustively -- its domain is the 65536 bit patterns; float32
    and float64 get their special patterns and a sample of 400 random patterns each.
    ONNX Runtime is asked as well where it has a kernel, but the document's rule is
    the arbiter: Abs is exact and has no accuracy to forgive.
    """
    failures, rows = [], []
    rng = np.random.default_rng(20261003)
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        w = dt.itemsize * 8
        if w == 16:
            a = np.arange(0, 1 << 16, dtype=np.uint16).view(dt)
            how = "all 65536 bit patterns"
        else:
            uint = UINT_OF_WIDTH[dt.itemsize]
            pats = set(_float_patterns(dt).values())
            pats.update({0, 1, 1 << (w - 1), (1 << (w - 1)) | 1, (1 << w) - 1})
            edge = np.array(sorted(pats), dtype=uint).view(dt)
            a = np.concatenate([edge, _random_patterns(rng, dt, 400)])
            how = "the special patterns + 400 random patterns"
        want = _sign_cleared(a)
        got = np.asarray(impl.abs(a))
        if not _same(want, got):
            failures.append({
                "who": "%s: the implementation vs the document's sign-clearing rule"
                       % dt.name,
                "detail": _first_diffs(want, got)})
        try:
            ref = run_ort_abs(a)
        except Exception as exc:
            rows.append("%s: %s, the runtime refused it (%s: %s), so the implementation "
                        "is checked against the document alone"
                        % (dt.name, how, type(exc).__name__,
                           str(exc).splitlines()[0][:80]))
            continue
        if not _same(want, ref):
            failures.append({"who": "%s: the document vs ONNX Runtime" % dt.name,
                             "detail": _first_diffs(want, ref)})
        rows.append("%s: %s" % (dt.name, how))
    return {"sweep": "the float rule (the sign bit cleared) over the types' values",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def sweep_int_exact(impl):
    """The integer case analyses over the values of each signed and unsigned type."""
    failures, rows = [], []
    rng = np.random.default_rng(20261004)
    for dtype in ALL_INT + ALL_UINT:
        dt = np.dtype(dtype)
        w = dt.itemsize * 8
        info = np.iinfo(dt)
        lo, hi = int(info.min), int(info.max)
        if w <= 16:
            start = lo if dt.kind == "i" else 0
            a = np.arange(start, hi + 1, dtype=np.int64).astype(dt)
            how = "all %d values" % a.size
        else:
            if dt.kind == "i":
                edges = [lo, lo + 1, lo + 2, -2, -1, 0, 1, 2, hi - 1, hi]
            else:
                edges = [0, 1, 2, 3, hi // 2, hi - 2, hi - 1, hi]
            a = np.concatenate([np.array(edges, dtype=dt),
                                _random_values(rng, dt, 400)])
            how = "the boundary values + 400 random values"
        want = _doc_int_abs(a) if dt.kind == "i" else _doc_uint_abs(a)
        got = np.asarray(impl.abs(a))
        if not _same(want, got):
            failures.append({
                "who": "%s: the implementation vs the document's case analysis"
                       % dt.name,
                "detail": _first_diffs(want, got)})
        try:
            ref = run_ort_abs(a)
        except Exception as exc:
            rows.append("%s: %s, the runtime refused it (%s: %s), so the implementation "
                        "is checked against the document alone"
                        % (dt.name, how, type(exc).__name__,
                           str(exc).splitlines()[0][:80]))
            continue
        if not _same(want, ref):
            failures.append({"who": "%s: the document vs ONNX Runtime" % dt.name,
                             "detail": _first_diffs(want, ref)})
        rows.append("%s: %s" % (dt.name, how))
    return {"sweep": "the integer case analyses over the types' values",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact rationals."""
    failures, checked = [], []
    for label, xs, documented in _REAL_EXAMPLES:
        exact = [abs(x) for x in xs]
        if exact != list(documented):
            failures.append({
                "who": label + ": the document's own example",
                "detail": "the absolute values in exact arithmetic are %s, the document "
                          "states %s"
                          % ([str(v) for v in exact], [str(v) for v in documented])})
            continue
        if all(Fraction(float(v)) == v for v in xs):
            arr = np.array([float(v) for v in xs], dtype=np.float64)
            want = np.array([float(v) for v in documented], dtype=np.float64)
            got = np.asarray(impl.abs(arr))
            if not _same(want, got):
                failures.append({"who": label + ": the implementation",
                                 "detail": _first_diffs(want, got)})
            checked.append(label + " (exact rationals, and float64 where representable)")
        else:
            checked.append(label + " (exact rationals only; not representable in "
                                   "float64)")
    return {"sweep": "the real section (exact rationals, no ONNX Runtime)",
            "summary": "%d examples: %s" % (len(_REAL_EXAMPLES), "; ".join(checked)),
            "failures": failures}


def sweeps():
    return [sweep_type_families, sweep_spec_examples, sweep_float_exact,
            sweep_int_exact, sweep_real_examples]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _float_cases(dtype):
    """The value families of the float section, for one type."""
    dt = np.dtype(dtype)
    nan, inf = float("nan"), float("inf")
    p = _float_patterns(dt)
    return [
        ("Example 1: the two zeros, the two infinities and a NaN",
         np.array([-1.5, 2.25, 0.0, -0.0, -inf, inf, nan], dtype=dt)),
        ("Example 2: the sign cleared, each element at its index",
         np.array([[-3.0, -1.0], [-2.5, 4.0]], dtype=dt)),
        ("the two signed zeros", np.array([-0.0, 0.0, -0.0, 0.0], dtype=dt)),
        ("the two infinities", np.array([inf, -inf, -inf, inf], dtype=dt)),
        ("the smallest subnormal and its negation",
         np.concatenate([_bits_array(dt, p["min_subnormal"]),
                         _bits_array(dt, p["min_subnormal_neg"])])),
        ("the largest subnormal", _bits_array(dt, p["max_subnormal"])),
        ("the smallest normal and its negation",
         np.concatenate([_bits_array(dt, p["min_normal"]),
                         _bits_array(dt, p["min_normal_neg"])])),
        ("the largest finite value and its negation",
         np.concatenate([_bits_array(dt, p["max_finite"]),
                         _bits_array(dt, p["max_finite_neg"])])),
        ("a quiet NaN, a signalling NaN and a NaN with the sign set",
         np.concatenate([_bits_array(dt, p["quiet_nan"]),
                         _bits_array(dt, p["signalling_nan"]),
                         _bits_array(dt, p["quiet_nan_neg"])])),
        ("ordinary positive and negative values",
         np.array([-1.5, 2.25, -0.5, 1.0, -1.0, 0.25], dtype=dt)),
        ("a rank-0 tensor", np.array(-3.5, dtype=dt)),
        ("a rank-0 tensor holding -0", np.array(-0.0, dtype=dt)),
        ("a zero-sized dimension", np.zeros((0, 3), dtype=dt)),
        ("a zero-sized last dimension", np.zeros((2, 0), dtype=dt)),
        ("a zero-sized rank-1 tensor", np.zeros((0,), dtype=dt)),
        ("a 2x3x2 tensor",
         (np.arange(12, dtype=np.float64) - 6.0).astype(dt).reshape(2, 3, 2)),
    ]


def _int_cases(dtype):
    """The value families of the signed section, for one type."""
    dt = np.dtype(dtype)
    info = np.iinfo(dt)
    lo, hi = int(info.min), int(info.max)
    out = []
    if dt.itemsize == 1:
        out.append(("Example 1: -6, 100, 127, 0, -127, -128 (as documented)",
                    np.array([-6, 100, 127, 0, -127, -128], dtype=dt)))
    out.append(("the minimum value, its neighbours, -1, 0 and the maximum",
                np.array([lo, lo + 1, lo + 2, -2, -1, 0, 1, 2, hi - 1, hi], dtype=dt)))
    out.append(("the minimum value alone", np.array([lo], dtype=dt)))
    out.append(("a rank-0 tensor holding the minimum value", np.array(lo, dtype=dt)))
    out.append(("a rank-0 tensor", np.array(-1, dtype=dt)))
    out.append(("a zero-sized rank-1 tensor", np.zeros((0,), dtype=dt)))
    out.append(("a zero-sized 2-D tensor", np.zeros((0, 4), dtype=dt)))
    out.append(("a 2x3 tensor", np.array([[-1, 2, -3], [4, lo, hi]], dtype=dt)))
    return out


def _uint_cases(dtype):
    """The value families of the unsigned section, for one type."""
    dt = np.dtype(dtype)
    info = np.iinfo(dt)
    hi = int(info.max)
    out = []
    if dt.itemsize == 1:
        out.append(("Example 1: 0, 6, 200, 255 (as documented)",
                    np.array([0, 6, 200, 255], dtype=dt)))
    out.append(("0, 1, 2 and the maximum value",
                np.array([0, 1, 2, hi // 2, hi - 1, hi], dtype=dt)))
    out.append(("the maximum value alone", np.array([hi], dtype=dt)))
    out.append(("a rank-0 tensor holding the maximum value", np.array(hi, dtype=dt)))
    out.append(("a rank-0 tensor holding 0", np.array(0, dtype=dt)))
    out.append(("a zero-sized rank-1 tensor", np.zeros((0,), dtype=dt)))
    out.append(("a zero-sized 2-D tensor", np.zeros((0, 4), dtype=dt)))
    out.append(("a 2x3 tensor", np.array([[0, 1, 2], [3, hi, hi // 2]], dtype=dt)))
    return out


_CASES = []
_COVERAGE = {"real": 0, "float": 0, "int": 0, "uint": 0}


def _build():
    for dtype in ALL_FLOAT:
        name = np.dtype(dtype).name
        for label, arr in _float_cases(dtype):
            _CASES.append(("%s: %s" % (name, label), [arr]))
            _COVERAGE["float"] += 1
    for dtype in ALL_INT:
        name = np.dtype(dtype).name
        for label, arr in _int_cases(dtype):
            _CASES.append(("%s: %s" % (name, label), [arr]))
            _COVERAGE["int"] += 1
    for dtype in ALL_UINT:
        name = np.dtype(dtype).name
        for label, arr in _uint_cases(dtype):
            _CASES.append(("%s: %s" % (name, label), [arr]))
            _COVERAGE["uint"] += 1
    # the real section has no ONNX type: its cases are the exact-rational examples of
    # sweep_real_examples, and they are counted here so the anchor is not vacuous
    _COVERAGE["real"] = len(_REAL_EXAMPLES)


_build()

COVERAGE = dict(_COVERAGE)


def cases():
    """Every individual case: one operand per case, and no attributes (Abs has none)."""
    for item in _CASES:
        yield item
