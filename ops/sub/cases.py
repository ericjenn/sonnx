"""The cases with which the **Sub** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled pair of operands per case.  Each case goes through a
                     one-node Sub model (opset 14) and through the implementation, and
                     the two results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the domain of the integer
                     types, the special values and a random sample of the float types,
                     the broadcasting families, the worked examples of every section of
                     the document, the schema's admitted types, and the real section,
                     which has no ONNX type and is checked against exact rational
                     arithmetic instead.
  * ``COVERAGE``  -- the four section anchors of the document, each with the number of
                     cases that exercise it.

The document's rule is transcribed literally (``_doc_result`` and the ``doc_int_*``
case analyses below) and is the arbiter: ONNX Runtime is asked for its answer as a
second measurement, never as the definition.  Every case of every section's formula is
meant to appear here, and the case set is part of the loop: when a section is added to
the document, its cases are added here before the next run.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Sub"
OPSET = 14
IR_VERSION = 10

NODE_OUTPUTS = ("out",)

ALL_FLOAT = [np.float16, np.float32, np.float64]
SIGNED_INT = [np.int8, np.int16, np.int32, np.int64]
UNSIGNED_INT = [np.uint8, np.uint16, np.uint32, np.uint64]
ALL_INT = SIGNED_INT + UNSIGNED_INT
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One batched ONNX Runtime run of a one-node Sub model."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    proto = helper.np_dtype_to_tensor_dtype(a.dtype)
    vi_a = helper.make_tensor_value_info("A", proto, list(a.shape))
    vi_b = helper.make_tensor_value_info("B", proto, list(b.shape))
    vi_c = helper.make_tensor_value_info("C", proto, None)
    node = helper.make_node(OP, ["A", "B"], ["C"], name="sub0")
    graph = helper.make_graph([node], "sub_graph", [vi_a, vi_b], [vi_c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    (c,) = sess.run(["C"], {"A": a, "B": b})
    return c


def _try_ort(a, b):
    """ONNX Runtime's answer, or ``(None, reason)`` when this build refuses the type.

    A type the runtime has no kernel for is not a failure of the implementation and not
    a defect of the document: the sweep says so in its summary instead of raising.
    """
    try:
        return np.asarray(run_ort_batch(a, b)), None
    except Exception as exc:  # no kernel, no shape, no session: the build's word
        return None, "%s: %s" % (type(exc).__name__, exc)


def _bits_equal(want: np.ndarray, got: np.ndarray) -> bool:
    """Bit-for-bit comparison of floating-point types, NaN equal to NaN."""
    want = np.asarray(want)
    got = np.asarray(got)
    if want.shape != got.shape or want.dtype != got.dtype:
        return False
    if want.dtype.kind == "f":
        uint = UINT_OF_WIDTH[want.dtype.itemsize]
        bw = np.ascontiguousarray(want).view(uint)
        bg = np.ascontiguousarray(got).view(uint)
        both_nan = np.isnan(want) & np.isnan(got)
        return bool(np.all((bw == bg) | both_nan))
    return bool(np.all(want == got))


def _mismatch_indices(want: np.ndarray, got: np.ndarray):
    """Flat indices where two arrays of the same type differ (NaN == NaN)."""
    want = np.asarray(want)
    got = np.asarray(got)
    if want.shape != got.shape or want.dtype != got.dtype:
        return None
    if want.dtype.kind == "f":
        uint = UINT_OF_WIDTH[want.dtype.itemsize]
        bw = np.ascontiguousarray(want).view(uint)
        bg = np.ascontiguousarray(got).view(uint)
        both_nan = np.isnan(want) & np.isnan(got)
        bad = ~((bw == bg) | both_nan)
    else:
        bad = want != got
    return np.nonzero(np.ravel(bad))[0]


def _align(x, shape):
    """``x`` reshaped to ``shape`` when it holds the same number of elements.

    A rank-0 result comes back from ONNX Runtime as a one-element array, and a
    one-element array sometimes comes back as a rank-0 result: that is a difference of
    representation, not of value, and it must not be reported as a disagreement.
    """
    arr = np.asarray(x)
    shape = tuple(shape)
    if arr.shape == shape:
        return arr
    n = 1
    for d in shape:
        n *= int(d)
    if arr.size == n:
        return arr.reshape(shape)
    return arr


# ---------------------------------------------------------------------------
# the document, transcribed literally and independently of the implementation
# ---------------------------------------------------------------------------
#
# The float section is the IEEE 754 subtraction with an unbounded exponent range and
# roundTiesToEven, which is exactly what the type's own arithmetic gives in
# round-to-nearest (the hardware's overflow-to-infinity is the document's rounding rule
# applied past the largest finite value, and the sign of a null difference is the
# IEEE 754 sign rule the document restates).  The two integer sections are the case
# analyses of the document, in exact Python integers.


def _doc_float(a, b):
    with np.errstate(all="ignore"):
        return np.subtract(np.asarray(a), np.asarray(b))


def doc_int_signed(x: int, y: int, n: int) -> int:
    """The case analysis of **Sub** (int, int), in exact arithmetic."""
    lo, hi = -(2 ** (n - 1)), 2 ** (n - 1) - 1
    d = x - y
    if d > hi:
        return d - 2 ** n
    if d < lo:
        return d + 2 ** n
    return d


def doc_int_unsigned(x: int, y: int, n: int) -> int:
    """The case analysis of **Sub** (uint, uint), in exact arithmetic."""
    d = x - y
    return d if d >= 0 else d + 2 ** n


def _doc_int_array(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    ba, bb = np.broadcast_arrays(a, b)
    dt = ba.dtype
    n = dt.itemsize * 8
    fn = doc_int_signed if dt.kind == "i" else doc_int_unsigned
    flat = [fn(int(x), int(y), n)
            for x, y in zip(ba.ravel().tolist(), bb.ravel().tolist())]
    return np.array(flat, dtype=dt).reshape(ba.shape)


def _doc_result(a, b):
    """The document's result for one pair of operands, by its own rule."""
    a = np.asarray(a)
    if a.dtype.kind == "f":
        return _doc_float(a, b)
    return _doc_int_array(a, b)


def _smallest_subnormal(dt):
    f = np.finfo(dt)
    v = getattr(f, "smallest_subnormal", None)
    if v is not None:
        return float(v)
    return float(f.tiny) * 2.0 ** (-int(f.nmant))


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def _rand_value(rng, bits, signed):
    """One random value of an integer type, drawn over the whole domain."""
    hi = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    lo = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    v = (hi << 32) | lo
    if bits < 64:
        v &= (1 << bits) - 1
    if signed and v >= (1 << (bits - 1)):
        v -= 1 << bits
    return v


def _make_int_array(values, dt):
    """Build an array of an integer type from Python ints over the whole domain."""
    return np.array([int(v) for v in values], dtype=object).astype(dt)


def sweep_int_domain(impl):
    """The document's integer case analysis over a domain, against ORT and the impl.

    The two 8-bit types are covered exhaustively (all 65536 ordered pairs each); the
    wider types use their boundary values and 200 random pairs.
    """
    rng = np.random.default_rng(20261003)
    rows, failures = [], []
    for dtype in ALL_INT:
        dt = np.dtype(dtype)
        info = np.iinfo(dt)
        bits = dt.itemsize * 8
        signed = dt.kind == "i"
        exhaustive = dt.itemsize == 1
        if exhaustive:
            xs = list(range(int(info.min), int(info.max) + 1))
            a_vals = [x for x in xs for _ in xs]
            b_vals = [y for _ in xs for y in xs]
        else:
            edges = sorted({int(info.min), int(info.min) + 1, int(info.max) - 1,
                            int(info.max), 0, 1, -1})
            edges = [v for v in edges if int(info.min) <= v <= int(info.max)]
            pairs = [(x, y) for x in edges for y in edges]
            pairs += [(_rand_value(rng, bits, signed), _rand_value(rng, bits, signed))
                      for _ in range(200)]
            a_vals = [p[0] for p in pairs]
            b_vals = [p[1] for p in pairs]
        aa = _make_int_array(a_vals, dt)
        bb = _make_int_array(b_vals, dt)

        want = _doc_int_array(aa, bb)              # the document, literally
        ref, err = _try_ort(aa, bb)                # ONNX Runtime
        if ref is not None:
            ref = _align(ref, want.shape)
            if not _bits_equal(want, ref):
                bad = _mismatch_indices(want, ref)
                bad = [] if bad is None else bad[:5]
                failures.append({
                    "type": dt.name, "who": "the document vs ONNX Runtime",
                    "detail": "; ".join("doc: %d - %d = %d, ort: %d"
                                        % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                           int(want.ravel()[i]), int(ref.ravel()[i]))
                                        for i in bad) or "shape or type differs",
                })
        try:
            got = _align(np.asarray(impl.sub(aa, bb)), want.shape)
        except Exception as exc:
            failures.append({"type": dt.name,
                             "who": "the implementation vs the document",
                             "detail": "the implementation raised %s: %s"
                                       % (type(exc).__name__, exc)})
            got = None
        if got is not None and not _bits_equal(want, got):
            bad = _mismatch_indices(want, got)
            bad = [] if bad is None else bad[:5]
            failures.append({
                "type": dt.name, "who": "the implementation vs the document",
                "detail": "; ".join("doc: %d - %d = %d, impl: %d"
                                    % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                       int(want.ravel()[i]), int(got.ravel()[i]))
                                    for i in bad) or "shape or type differs",
            })
        rows.append("%s %s%s" % (dt.name,
                                 "exhaustive (65536 pairs)" if exhaustive
                                 else "%d pairs" % len(aa),
                                 "" if ref is not None else " (not reference-checked)"))
    return {"sweep": "int domain (document vs ORT vs implementation)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def _float_specials(dt):
    """The special values of one floating-point type, as operands."""
    f = np.finfo(dt)
    maxf = float(f.max)
    tiny = float(f.tiny)
    sub = _smallest_subnormal(dt)
    return [0.0, -0.0, float("inf"), float("-inf"), float("nan"),
            maxf, -maxf, tiny, -tiny, sub, -sub,
            1.0, -1.0, 0.5, 2.0, 3.0]


def sweep_float_specials(impl):
    """Every ordered pair of the special values of each float type.

    The float section is the IEEE 754 subtraction, and Sub has no reduction and no
    window: the two implementations (the document's rule transcribed with the type's
    own arithmetic, ONNX Runtime) have nothing to disagree about here, so both are
    measured.  Signed zeros, NaNs (NaN equal to NaN), subnormals, the extremes and the
    overflow thresholds are compared bit for bit.
    """
    failures, rows = [], []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        vals = np.array(_float_specials(dt), dtype=dt)
        n = len(vals)
        a = np.repeat(vals, n)
        b = np.tile(vals, n)
        want = np.asarray(_doc_result(a, b))
        try:
            got = _align(np.asarray(impl.sub(a, b)), want.shape)
        except Exception as exc:
            failures.append({"type": dt.name, "who": "the implementation",
                             "detail": "the implementation raised %s: %s"
                                       % (type(exc).__name__, exc)})
            got = None
        if got is not None and not _bits_equal(want, got):
            bad = _mismatch_indices(want, got)
            bad = [] if bad is None else bad[:5]
            failures.append({
                "type": dt.name, "who": "the document vs the implementation",
                "detail": "; ".join("%r - %r: doc %r, impl %r"
                                    % (float(a.ravel()[i]), float(b.ravel()[i]),
                                       want.ravel()[i], got.ravel()[i])
                                    for i in bad) or "shape or type differs",
            })
        ref, err = _try_ort(a, b)
        if ref is not None:
            ref = _align(ref, want.shape)
            if not _bits_equal(want, ref):
                bad = _mismatch_indices(want, ref)
                bad = [] if bad is None else bad[:5]
                failures.append({
                    "type": dt.name, "who": "the document vs ONNX Runtime",
                    "detail": "; ".join("%r - %r: doc %r, ort %r"
                                        % (float(a.ravel()[i]), float(b.ravel()[i]),
                                           want.ravel()[i], ref.ravel()[i])
                                        for i in bad) or "shape or type differs",
                })
        rows.append("%s %d pairs%s"
                    % (dt.name, n * n,
                       "" if ref is not None
                       else " (not reference-checked: %s)" % str(err)[:80]))
    return {"sweep": "float special values",
            "summary": "signed zeros, infinities, NaN, subnormals, extremes, overflow "
                       "thresholds; %s" % ", ".join(rows),
            "failures": failures}


def sweep_float_random(impl):
    """A random sample of each float type, wide in exponent."""
    rng = np.random.default_rng(20261004)
    failures, rows = [], []
    count = 500
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        half = int(f.maxexp) // 2
        exps = rng.integers(-half, half, size=count)
        with np.errstate(all="ignore"):
            a = np.ldexp(rng.standard_normal(count), exps).astype(dt)
            b = np.ldexp(rng.standard_normal(count), exps[::-1].copy()).astype(dt)
        want = np.asarray(_doc_result(a, b))
        try:
            got = _align(np.asarray(impl.sub(a, b)), want.shape)
        except Exception as exc:
            failures.append({"type": dt.name, "who": "the implementation",
                             "detail": "the implementation raised %s: %s"
                                       % (type(exc).__name__, exc)})
            got = None
        if got is not None and not _bits_equal(want, got):
            bad = _mismatch_indices(want, got)
            bad = [] if bad is None else bad[:5]
            failures.append({
                "type": dt.name, "who": "the document vs the implementation",
                "detail": "; ".join("%r - %r: doc %r, impl %r"
                                    % (float(a.ravel()[i]), float(b.ravel()[i]),
                                       want.ravel()[i], got.ravel()[i])
                                    for i in bad) or "shape or type differs",
            })
        ref, err = _try_ort(a, b)
        if ref is not None:
            ref = _align(ref, want.shape)
            if not _bits_equal(want, ref):
                bad = _mismatch_indices(want, ref)
                bad = [] if bad is None else bad[:5]
                failures.append({
                    "type": dt.name, "who": "the document vs ONNX Runtime",
                    "detail": "; ".join("%r - %r: doc %r, ort %r"
                                        % (float(a.ravel()[i]), float(b.ravel()[i]),
                                           want.ravel()[i], ref.ravel()[i])
                                        for i in bad) or "shape or type differs",
                })
        rows.append("%s %d%s" % (dt.name, count,
                                 "" if ref is not None else " (not reference-checked)"))
    return {"sweep": "float random sample",
            "summary": "%d random values of each type, exponents over half the range: %s"
                       % (count, ", ".join(rows)),
            "failures": failures}


def _arr(shape, dtype, start=0.0):
    n = 1
    for d in shape:
        n *= int(d)
    return (np.arange(n, dtype=np.float64) + start).astype(dtype).reshape(shape)


SHAPE_SWEEP = [
    ((2, 3), (2, 3)),
    ((2, 3), (1, 3)),
    ((2, 3), (3,)),
    ((2, 3, 4), (4,)),
    ((2, 3, 4), (3, 1)),
    ((2, 1, 4), (3, 1)),
    ((5, 1, 3), (1, 4, 1)),
    ((), (2, 3)),
    ((2, 3), ()),
    ((), ()),
    ((0, 3), (1, 3)),
    ((0,), (1,)),
    ((0, 3), (3,)),
    ((1, 0), (2, 0)),
    ((0, 3), (0, 1)),
    ((0,), ()),
]


def sweep_shapes(impl):
    """The broadcasting families of the document, shape and values together."""
    failures, checked, refusals = [], 0, 0
    for sa, sb in SHAPE_SWEEP:
        for dtype in (np.float32, np.int32, np.uint8):
            dt = np.dtype(dtype)
            a = _arr(sa, dtype)
            b = _arr(sb, dtype, 1.0)
            want = np.asarray(_doc_result(a, b))
            checked += 1
            try:
                got = np.asarray(impl.sub(a, b))
            except Exception as exc:
                failures.append({"who": "%s: %s - %s" % (dt.name, sa, sb),
                                 "detail": "the implementation raised %s: %s"
                                           % (type(exc).__name__, exc)})
                continue
            if not _bits_equal(want, got):
                failures.append({
                    "who": "%s: %s - %s (the document vs the implementation)"
                           % (dt.name, sa, sb),
                    "detail": "the document gives shape %s values %s, the implementation "
                              "shape %s values %s"
                              % (list(want.shape), want.ravel().tolist(),
                                 list(got.shape), got.ravel().tolist())})
            ref, err = _try_ort(a, b)
            if ref is None:
                refusals += 1
                continue
            ref = _align(ref, want.shape)
            if not _bits_equal(want, ref):
                failures.append({
                    "who": "%s: %s - %s (the document vs ONNX Runtime)"
                           % (dt.name, sa, sb),
                    "detail": "the document gives shape %s values %s, ORT shape %s "
                              "values %s"
                              % (list(want.shape), want.ravel().tolist(),
                                 list(np.shape(ref)), np.ravel(ref).tolist())})
    return {"sweep": "broadcasting families (same shape, dim 1, smaller rank, rank-0, "
                     "zero-sized)",
            "summary": "%d shape pairs checked; the runtime refused %d of them"
                       % (checked, refusals),
            "failures": failures}


def _spec_examples():
    """The worked examples of the float, the int and the uint sections, as documented."""
    f32 = np.float32
    f64 = np.float64
    return [
        ("float Example 1",
         np.array([[3.0, 4.5], [16.0, 1.0], [25.5, 24.25]], dtype=f32),
         np.array([[3.0, 2.0], [4.0, 0.0], [5.0, 4.0]], dtype=f32),
         np.array([[0.0, 2.5], [12.0, 1.0], [20.5, 20.25]], dtype=f32)),
        ("float Example 2",
         np.array([1.0, np.inf, -0.0, 0.0], dtype=f32),
         np.array([1.0, np.inf, 0.0, -0.0], dtype=f32),
         np.array([0.0, np.nan, -0.0, 0.0], dtype=f32)),
        ("float Example 3",
         np.array([0.3, 1.0], dtype=f64),
         np.array([0.1, 1.0], dtype=f64),
         np.array([0.3 - 0.1, 0.0], dtype=f64)),
        ("int Example 1",
         np.array([6, 5, -35], dtype=np.int8),
         np.array([3, 3, 3], dtype=np.int8),
         np.array([3, 2, -38], dtype=np.int8)),
        ("int Example 2",
         np.array([127, -128, 0], dtype=np.int8),
         np.array([-1, 1, 1], dtype=np.int8),
         np.array([-128, 127, -1], dtype=np.int8)),
        ("uint Example 1",
         np.array([0, 255, 4], dtype=np.uint8),
         np.array([1, 1, 7], dtype=np.uint8),
         np.array([255, 254, 253], dtype=np.uint8)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, vs its own rule, vs ORT."""
    failures = []
    examples = _spec_examples()
    for label, a, b, documented in examples:
        want = np.asarray(_doc_result(a, b))
        doc = np.asarray(documented)
        if not _bits_equal(doc, want):
            failures.append({"who": label,
                             "detail": "documented %s, the document's rule %s"
                                       % (doc.ravel().tolist(), want.ravel().tolist())})
        ref, err = _try_ort(a, b)
        if ref is not None:
            ref = _align(ref, doc.shape)
            if not _bits_equal(doc, ref):
                failures.append({"who": label, "detail": "documented %s, ORT %s"
                                 % (doc.ravel().tolist(), np.ravel(ref).tolist())})
        try:
            got = _align(np.asarray(impl.sub(a, b)), doc.shape)
        except Exception as exc:
            failures.append({"who": label,
                             "detail": "the implementation raised %s: %s"
                                       % (type(exc).__name__, exc)})
            continue
        if not _bits_equal(doc, got):
            failures.append({"who": label, "detail": "documented %s, implementation %s"
                             % (doc.ravel().tolist(), got.ravel().tolist())})
    return {"sweep": "the document's worked examples",
            "summary": "%d examples, as documented vs the document's rule vs ORT vs "
                       "the implementation" % len(examples),
            "failures": failures}


def _real_examples():
    """The real section's cases: exact rationals, no ONNX type."""
    F = Fraction
    return [
        ("real Example 1",
         [[F("6.1"), F("9.5"), F("35.7")]],
         [[F(3), F("3.3"), F("5.1")]],
         [[F("3.1"), F("6.2"), F("30.6")]]),
        ("real Example 2",
         [[F("3.7"), F("4.4")], [F("16.2"), F("0.5")], [F("25.3"), F("24.8")]],
         [[F(1), F(2)]],
         [[F("2.7"), F("2.4")], [F("15.2"), F("-1.5")], [F("24.3"), F("22.8")]]),
        ("real Example 3 (B is a rank-0 tensor)",
         [[F(5), F("3.25")], [F(4), F("1.75")]],
         F("2.5"),
         [[F("2.5"), F("0.75")], [F("1.5"), F("-0.75")]]),
        ("real: the exactness of 1/3 - 1/6 and the cancellation of 1/10 - 1/10",
         [[F(1, 3), F(1, 10)]],
         [[F(1, 6), F(1, 10)]],
         [[F(1, 6), F(0)]]),
        ("real: a rank-0 operand broadcast over a 3x2 matrix",
         [[F(1, 3), F(2, 3)], [F(1, 7), F(3, 7)], [F(-1, 9), F(5, 9)]],
         F(1, 6),
         None),
    ]


_REAL_EXAMPLES = _real_examples()


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact rationals.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    implementation against the exact difference the real section specifies (in
    `fractions.Fraction`) and against the values the document's Examples state.
    """
    failures = []
    for label, a, b, documented in _REAL_EXAMPLES:
        aa = np.array(a, dtype=object)
        bb = np.array(b, dtype=object)
        ba, bc = np.broadcast_arrays(aa, bb)
        want = np.empty(ba.shape, dtype=object)
        for i in np.ndindex(ba.shape):
            want[i] = ba[i] - bc[i]
        try:
            got = np.asarray(impl.sub(aa, bb))
        except Exception as exc:
            failures.append({"who": label,
                             "detail": "the implementation raised %s: %s"
                                       % (type(exc).__name__, exc)})
            continue
        if got.shape != want.shape:
            failures.append({"who": label,
                             "detail": "exact shape %s, implementation shape %s"
                                       % (want.shape, got.shape)})
            continue
        bad = [i for i in np.ndindex(want.shape) if got[i] != want[i]]
        if bad:
            failures.append({"who": label,
                             "detail": "; ".join("exact %s, implementation %s"
                                                 % (want[i], got[i]) for i in bad[:5])})
        if documented is not None:
            doc = np.array(documented, dtype=object)
            if doc.shape != got.shape:
                failures.append({"who": label,
                                 "detail": "documented shape %s, implementation shape %s"
                                           % (doc.shape, got.shape)})
            else:
                off = [i for i in np.ndindex(doc.shape) if doc[i] != got[i]]
                if off:
                    failures.append({"who": label,
                                     "detail": "documented %s, implementation %s"
                                               % (doc.tolist(), got.tolist())})
    return {"sweep": "real section (exact rationals, no ONNX Runtime)",
            "summary": "%d cases, exact Fraction differences and the documented values"
                       % len(_REAL_EXAMPLES), "failures": failures}


ONNX_TYPE_OF_DTYPE = {
    "float16": "tensor(float16)",
    "float32": "tensor(float)",
    "float64": "tensor(double)",
    "bfloat16": "tensor(bfloat16)",
    "int8": "tensor(int8)",
    "int16": "tensor(int16)",
    "int32": "tensor(int32)",
    "int64": "tensor(int64)",
    "uint8": "tensor(uint8)",
    "uint16": "tensor(uint16)",
    "uint32": "tensor(uint32)",
    "uint64": "tensor(uint64)",
}


def _onnx_type_name(dtype):
    """The ONNX type string of a NumPy dtype.

    The document names its types as ONNX does (`float`, `double`, `int8`), the schema
    names them `tensor(float)`, `tensor(double)`, `tensor(int8)`: the two are compared
    through this mapping, never as NumPy dtype names.
    """
    name = np.dtype(dtype).name
    return ONNX_TYPE_OF_DTYPE.get(name, "tensor(%s)" % name)


def sweep_schema_types(impl):
    """The document's type families against the types the ONNX schema admits.

    The document announces `float16`/`float`/`double`, `int8`..`int64` and
    `uint8`..`uint64`; the schema of Sub-14 is asked for the same list, without running
    anything.  A type the document admits and the schema does not is a defect of the
    document; a type the schema admits and the document does not specify is recorded in
    the summary, because the profile specifies a subset.
    """
    documented = [_onnx_type_name(d) for d in ALL_FLOAT + ALL_INT]
    try:
        import onnx
        schema = onnx.defs.get_schema(OP, OPSET)
        allowed = set()
        for constraint in schema.type_constraints:
            allowed.update(constraint.allowed_type_strs)
    except Exception as exc:
        return {"sweep": "the document's type families vs the ONNX schema",
                "summary": "the schema of %s-%d could not be read (%s: %s)"
                           % (OP, OPSET, type(exc).__name__, exc),
                "failures": []}
    failures = []
    for name in documented:
        if name not in allowed:
            failures.append({"who": "the document declares %s" % name,
                             "detail": "the schema of %s-%d admits %s for T"
                                       % (OP, OPSET, sorted(allowed))})
    extra = sorted(allowed - set(documented))
    summary = ("%d declared types, every one admitted by the schema of %s-%d"
               % (len(documented), OP, OPSET))
    if extra:
        summary += ("; the schema also admits %s, which the document does not specify "
                    "(recorded, not required)" % ", ".join(extra))
    return {"sweep": "the document's type families vs the ONNX schema",
            "summary": summary, "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_shapes, sweep_int_domain,
            sweep_float_specials, sweep_float_random, sweep_real_examples,
            sweep_schema_types]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _case_list():
    """Every case of `cases()`, tagged with the section of the document it exercises."""
    f16, f32, f64 = np.float16, np.float32, np.float64
    items = []

    def add(section, label, a, b):
        items.append((section, label, [a, b]))

    # -- the float section: the three worked examples ------------------------
    add("float", "float Example 1: a 3x2 difference, every difference representable",
        np.array([[3.0, 4.5], [16.0, 1.0], [25.5, 24.25]], dtype=f32),
        np.array([[3.0, 2.0], [4.0, 0.0], [5.0, 4.0]], dtype=f32))
    add("float", "float Example 2: inf - inf = NaN, -0 - +0 = -0, +0 - -0 = +0",
        np.array([1.0, np.inf, -0.0, 0.0], dtype=f32),
        np.array([1.0, np.inf, 0.0, -0.0], dtype=f32))
    add("float", "float Example 3: 0.3 - 0.1 rounded in double, and 1.0 - 1.0 = +0",
        np.array([0.3, 1.0], dtype=f64),
        np.array([0.1, 1.0], dtype=f64))

    # -- the signed zeros: the four combinations, in each float type ---------
    zeros = np.array([0.0, 0.0, -0.0, -0.0])
    zeros_b = np.array([0.0, -0.0, -0.0, 0.0])
    for dt in ALL_FLOAT:
        add("float", "%s: (+0)-(+0), (+0)-(-0), (-0)-(-0), (-0)-(+0)"
                     % np.dtype(dt).name,
            zeros.astype(dt), zeros_b.astype(dt))
    add("float", "double: a null difference of equal non-zero operands is +0",
        np.array([0.5, -2.5, 1e300], dtype=f64),
        np.array([0.5, -2.5, 1e300], dtype=f64))

    # -- the infinities: the invalid operation and the four infinite results -
    add("float", "float32: inf-inf and -inf-(-inf) are the invalid operation (NaN)",
        np.array([np.inf, -np.inf], dtype=f32),
        np.array([np.inf, -np.inf], dtype=f32))
    add("float", "float32: inf-(-inf) = +inf and -inf-inf = -inf",
        np.array([np.inf, -np.inf], dtype=f32),
        np.array([-np.inf, np.inf], dtype=f32))
    add("float", "float16: an infinite operand with a finite one gives that infinity",
        np.array([np.inf, 1.0, -np.inf, -1.0], dtype=f16),
        np.array([1.0, np.inf, -1.0, -np.inf], dtype=f16))

    # -- NaN propagation -----------------------------------------------------
    add("float", "double: NaN propagates from either side, and NaN-NaN is NaN",
        np.array([np.nan, 1.0, np.nan, np.nan], dtype=f64),
        np.array([1.0, np.nan, np.nan, np.inf], dtype=f64))

    # -- the subnormal range and the smallest normal -------------------------
    for dt in ALL_FLOAT:
        sub = _smallest_subnormal(dt)
        tiny = float(np.finfo(dt).tiny)
        add("float", "%s: subnormals, the smallest normal, and sub-sub" % np.dtype(dt).name,
            np.array([sub, sub, tiny, -tiny, sub, -sub], dtype=dt),
            np.array([sub, -sub, sub, sub, tiny, tiny], dtype=dt))

    # -- the extremes and the overflow threshold -----------------------------
    for dt in ALL_FLOAT:
        maxf = float(np.finfo(dt).max)
        add("float", "%s: overflow to +-inf, max-max = +0, and 1.5*max -> inf"
                     % np.dtype(dt).name,
            np.array([maxf, -maxf, maxf, maxf / 2.0], dtype=dt),
            np.array([-maxf, maxf, maxf, -maxf], dtype=dt))

    # -- the rounding boundary and the tie above it --------------------------
    add("float", "float32: 1 - 2**-25 is a tie, rounded to 1.0 (roundTiesToEven)",
        np.array([1.0], dtype=f32), np.array([2.0 ** -25], dtype=f32))
    add("float", "float32: 1 - 3*2**-25 is a tie, rounded to 1 - 2**-23",
        np.array([1.0], dtype=f32), np.array([3.0 * 2.0 ** -25], dtype=f32))
    add("float", "float32: 2**24 + 2 - 1 is a tie, rounded to the even 2**24",
        np.array([float(2 ** 24 + 2)], dtype=f32), np.array([1.0], dtype=f32))
    add("float", "double: 1 - 2**-53 is a tie, rounded to 1.0",
        np.array([1.0], dtype=f64), np.array([2.0 ** -53], dtype=f64))
    add("float", "float32: 0.1 - 0.3 is not representable, and is rounded",
        np.array([0.1], dtype=f32), np.array([0.3], dtype=f32))

    # -- the broadcasting families of the float section ----------------------
    add("float", "broadcast (2,3) - (3,): the smaller rank is a row",
        np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=f32),
        np.array([0.5, 1.5, 2.5], dtype=f32))
    add("float", "broadcast (2,3,4) - (3,1)",
        _arr((2, 3, 4), f32, 1.0), _arr((3, 1), f32, 0.5))
    add("float", "rank-0 A broadcast to the shape of a (2,3) B",
        np.array(2.5, dtype=f32),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=f32))
    add("float", "zero-sized (0,3) - (1,3): the result is empty",
        np.zeros((0, 3), dtype=f32), np.zeros((1, 3), dtype=f32))
    add("float", "zero-sized (0,3) - (0,1): compatible, and the result is empty",
        np.zeros((0, 3), dtype=f32), np.zeros((0, 1), dtype=f32))

    # -- the int section: the two worked examples ---------------------------
    add("int", "int Example 1: every difference in the int8 range",
        np.array([6, 5, -35], dtype=np.int8),
        np.array([3, 3, 3], dtype=np.int8))
    add("int", "int Example 2: 127-(-1) = -128 and -128-1 = 127 in int8",
        np.array([127, -128, 0], dtype=np.int8),
        np.array([-1, 1, 1], dtype=np.int8))

    # -- the int section: the boundary of the type -------------------------
    add("int", "int8: min-1, max-(-1), min-min, max-max, min-max, max-min",
        np.array([-128, 127, -128, 127, -128, 127], dtype=np.int8),
        np.array([1, -1, -128, 127, 127, -128], dtype=np.int8))
    for dt in SIGNED_INT[1:]:
        info = np.iinfo(dt)
        lo, hi = int(info.min), int(info.max)
        add("int", "%s: min-1, max-(-1), min-min, max-max, min-max, max-min"
                   % np.dtype(dt).name,
            np.array([lo, hi, lo, hi, lo, hi], dtype=dt),
            np.array([1, -1, lo, hi, hi, lo], dtype=dt))
    add("int", "int32: a null difference of equal operands is 0",
        np.array([-7, 0, 2 ** 31 - 1], dtype=np.int32),
        np.array([-7, 0, 2 ** 31 - 1], dtype=np.int32))
    add("int", "int32: rank-0 A broadcast to the shape of a (3,) B",
        np.array(-10, dtype=np.int32),
        np.array([3, -40, 100], dtype=np.int32))
    add("int", "int64: zero-sized (0,) - ()",
        np.zeros((0,), dtype=np.int64), np.array(1, dtype=np.int64))

    # -- the uint section: the worked example ------------------------------
    add("uint", "uint Example 1: 0-1 = 255, 255-1 = 254, 4-7 = 253 in uint8",
        np.array([0, 255, 4], dtype=np.uint8),
        np.array([1, 1, 7], dtype=np.uint8))

    # -- the uint section: the boundary of the type ------------------------
    add("uint", "uint8: 0-1, 0-255, 255-255, 4-7, 255-1, 128-129",
        np.array([0, 0, 255, 4, 255, 128], dtype=np.uint8),
        np.array([1, 255, 255, 7, 1, 129], dtype=np.uint8))
    for dt in UNSIGNED_INT[1:]:
        info = np.iinfo(dt)
        hi = int(info.max)
        add("uint", "%s: 0-1, 0-max, max-max, max-1" % np.dtype(dt).name,
            np.array([0, 0, hi, hi], dtype=dt),
            np.array([1, hi, hi, 1], dtype=dt))
    add("uint", "uint8: rank-0 A broadcast to the shape of a (2,) B",
        np.array(10, dtype=np.uint8),
        np.array([3, 40], dtype=np.uint8))
    add("uint", "uint16: zero-sized (1,0) - (2,0) is compatible, and the result is empty",
        np.zeros((1, 0), dtype=np.uint16), np.zeros((2, 0), dtype=np.uint16))

    # -- the type consistency and the same-shape case of every family ------
    add("int", "int16: same shape, an in-range difference",
        np.array([100, -100, 0], dtype=np.int16),
        np.array([-100, 100, 0], dtype=np.int16))
    add("uint", "uint32: same shape, an in-range difference",
        np.array([100, 100, 0], dtype=np.uint32),
        np.array([40, 100, 7], dtype=np.uint32))

    return items


_CASES = _case_list()


def cases():
    for _section, label, arrays in _CASES:
        yield label, arrays


# ---------------------------------------------------------------------------
# coverage: every section anchor of the document, with the cases that exercise it
# ---------------------------------------------------------------------------

COVERAGE = {}
for _section, _label, _arrays in _CASES:
    COVERAGE[_section] = COVERAGE.get(_section, 0) + 1
COVERAGE["real"] = len(_REAL_EXAMPLES)
