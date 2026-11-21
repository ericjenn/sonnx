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

ALL_FLOAT = [np.float16, np.float32, np.float64]
ALL_INT = [np.int8, np.int16, np.int32, np.int64,
           np.uint8, np.uint16, np.uint32, np.uint64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}
IR_VERSION = 10


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

    A type the runtime has no kernel for is not a failure of the implementation and
    not a defect of the document: the sweep says so in its summary instead of raising.
    """
    try:
        return np.asarray(run_ort_batch(a, b)), None
    except Exception as exc:  # no kernel, no shape, no session: the build's word
        return None, "%s: %s" % (type(exc).__name__, exc)


def _bits_equal(want: np.ndarray, got: np.ndarray) -> bool:
    """Bit-for-bit comparison of floating-point types, NaN equal to NaN."""
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


# ---------------------------------------------------------------------------
# the document, transcribed literally and independently of the implementation
# ---------------------------------------------------------------------------
#
# The float section is the IEEE 754 subtraction with an unbounded exponent range and
# roundTiesToEven, which is exactly what the type's own arithmetic gives in
# round-to-nearest (the hardware's overflow-to-infinity is the document's rounding
# rule applied past the largest finite value).  The two integer sections are the case
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
    return np.array(list(values), dtype=object).astype(dt)


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
            edges = sorted({min(max(int(v), int(info.min)), int(info.max))
                            for v in (info.min, info.min + 1, info.max - 1,
                                      info.max, 0, 1, -1)})
            pairs = [(x, y) for x in edges for y in edges]
            pairs += [(_rand_value(rng, bits, signed), _rand_value(rng, bits, signed))
                      for _ in range(200)]
            a_vals = [p[0] for p in pairs]
            b_vals = [p[1] for p in pairs]
        aa = _make_int_array(a_vals, dt)
        bb = _make_int_array(b_vals, dt)

        want = _doc_int_array(aa, bb)              # the document, literally
        ref, err = _try_ort(aa, bb)                # ONNX Runtime
        if ref is not None and not _bits_equal(want, ref):
            bad = _mismatch_indices(want, ref)[:5]
            failures.append({
                "type": dt.name, "who": "the document vs ONNX Runtime",
                "detail": "; ".join("doc: %d - %d = %d, ort: %d"
                                    % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                       int(want.ravel()[i]), int(ref.ravel()[i]))
                                    for i in bad),
            })
        try:
            got = np.asarray(impl.sub(aa, bb))
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
                                 "exhaustive" if exhaustive else "%d pairs" % len(aa),
                                 "" if ref is not None else " (not reference-checked)"))
    return {"sweep": "int domain (document vs ORT vs implementation)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def _smallest_subnormal(dt):
    f = np.finfo(dt)
    v = getattr(f, "smallest_subnormal", None)
    if v is not None:
        return float(v)
    return float(f.tiny) * 2.0 ** (-int(f.nmant))


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
        want = _doc_result(a, b)
        try:
            got = np.asarray(impl.sub(a, b))
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
                                    % (float(a[i]), float(b[i]),
                                       want.ravel()[i], got.ravel()[i])
                                    for i in bad) or "shape or type differs",
            })
        ref, err = _try_ort(a, b)
        if ref is not None and not _bits_equal(want, ref):
            bad = _mismatch_indices(want, ref)
            bad = [] if bad is None else bad[:5]
            failures.append({
                "type": dt.name, "who": "the document vs ONNX Runtime",
                "detail": "; ".join("%r - %r: doc %r, ort %r"
                                    % (float(a[i]), float(b[i]),
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
        want = _doc_result(a, b)
        try:
            got = np.asarray(impl.sub(a, b))
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
                                    % (float(a[i]), float(b[i]),
                                       want.ravel()[i], got.ravel()[i])
                                    for i in bad) or "shape or type differs",
            })
        ref, err = _try_ort(a, b)
        if ref is not None and not _bits_equal(want, ref):
            bad = _mismatch_indices(want, ref)
            bad = [] if bad is None else bad[:5]
            failures.append({
                "type": dt.name, "who": "the document vs ONNX Runtime",
                "detail": "; ".join("%r - %r: doc %r, ort %r"
                                    % (float(a[i]), float(b[i]),
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
        n *= d
    return (np.arange(n, dtype=np.float64) + start).astype(dtype).reshape(shape)


SHAPE_SWEEP = [
    ((2, 3), (2, 3)),
    ((2, 3), (1, 3)),
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
            a = _arr(sa, dtype)
            b = _arr(sb, dtype, 1.0)
            want = _doc_result(a, b)
            checked += 1
            try:
                got = np.asarray(impl.sub(a, b))
            except Exception as exc:
                failures.append({"who": "%s: %s - %s" % (np.dtype(dtype).name, sa, sb),
                                 "detail": "the implementation raised %s: %s"
                                           % (type(exc).__name__, exc)})
                continue
            if not _bits_equal(want, got):
                failures.append({
                    "who": "%s: %s - %s" % (np.dtype(dtype).name, sa, sb),
                    "detail": "the document gives shape %s, the implementation %s"
                              % (list(want.shape), list(got.shape))})
            ref, err = _try_ort(a, b)
            if ref is None:
                refusals += 1
            elif not _bits_equal(want, ref):
                failures.append({
                    "who": "%s: %s - %s (the document vs ONNX Runtime)"
                           % (np.dtype(dtype).name, sa, sb),
                    "detail": "the document gives %s, ORT %s"
                              % (want.tolist(), ref.tolist())})
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
        want = _doc_result(a, b)
        doc = np.asarray(documented)
        if not _bits_equal(doc, want):
            failures.append({"who": label,
                             "detail": "documented %s, the document's rule %s"
                                       % (doc.tolist(), want.tolist())})
        ref, err = _try_ort(a, b)
        if ref is not None and not _bits_equal(doc, ref):
            failures.append({"who": label, "detail": "documented %s, ORT %s"
                             % (doc.tolist(), ref.tolist())})
        try:
            got = np.asarray(impl.sub(a, b))
        except Exception as exc:
            failures.append({"who": label,
                             "detail": "the implementation raised %s: %s"
                                       % (type(exc).__name__, exc)})
            continue
        if not _bits_equal(doc, got):
            failures.append({"who": label, "detail": "documented %s, implementation %s"
                             % (doc.tolist(), got.tolist())})
    return {"sweep": "the document's worked examples",
            "summary": "%d examples, as documented vs the document's rule vs ORT vs "
                       "the implementation" % len(examples),
            "failures": failures}


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact rationals.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    implementation against the exact difference the real section specifies (in
    `fractions.Fraction`) and against the values the document's Examples state.
    """
    F = Fraction
    examples = [
        ("real Example 1",
         [[F("6.1"), F("9.5"), F("35.7")]],
         [[F(3), F("3.3"), F("5.1")]],
         [[F("3.1"), F("6.2"), F("30.6")]]),
        ("real Example 2",
         [[F("3.7"), F("4.4")], [F("16.2"), F("0.5")], [F("25.3"), F("24.8")]],
         [[F(1), F(2)]],
         [[F("2.7"), F("2.4")], [F("15.2"), F("-1.5")], [F("24.3"), F("22.8")]]),
        ("real Example 3",
         [[F(5), F("3.25")], [F(4), F("1.75")]],
         [[F("2.5")]],
         [[F("2.5"), F("0.75")], [F("1.5"), F("-0.75")]]),
    ]
    failures = []
    for label, a, b, documented in examples:
        aa = np.array(a, dtype=object)
        bb = np.array(b, dtype=object)
        ba, bc = np.broadcast_arrays(aa, bb)
        want = np.array([[x - y for x, y in zip(ra, rb)]
                         for ra, rb in zip(ba.tolist(), bc.tolist())], dtype=object)
        try:
            got = np.asarray(impl.sub(aa, bb))
        except Exception as exc:
            failures.append({"who": label,
                             "detail": "the implementation raised %s: %s"
                                       % (type(exc).__name__, exc)})
            continue
        if got.shape != want.shape or any(got[i] != want[i]
                                          for i in np.ndindex(want.shape)):
            failures.append({"who": label, "detail": "exact %s, implementation %s"
                             % (want.tolist(), got.tolist())})
        doc = np.array(documented, dtype=object)
        if got.shape != doc.shape or any(got[i] != doc[i]
                                         for i in np.ndindex(doc.shape)):
            failures.append({"who": label, "detail": "documented %s, implementation %s"
                             % (doc.tolist(), got.tolist())})
    return {"sweep": "real section (exact rationals, no ONNX Runtime)",
            "summary": "%d examples, exact Fraction differences and the documented values"
                       % len(examples), "failures": failures}


def sweep_schema_types(impl):
    """The document's type families against the types the ONNX schema admits.

    The document announces float16/float/double, int8..int64 and uint8..uint64; the
    schema of Sub-14 is asked for the same list, without running anything.  A type the
    document admits and the schema does not is a defect of the document.
    """
    documented = ["tensor(%s)" % np.dtype(d).name for d in ALL_FLOAT + ALL_INT]
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
                             "detail": "the schema of %s-%d admits %s"
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
    return [sweep_spec_examples, sweep_int_domain, sweep_float_specials,
            sweep_float_random, sweep_shapes, sweep_real_examples, sweep_schema_types]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _float_cases():
    """The float section: the special values, the signed zeros, the rounding boundary."""
    out = []
    f32 = np.float32
    f16 = np.float16
    f64 = np.float64

    # Example 2 of the float section, replayed as documented
    out.append(("float Example 2: +inf - +inf = NaN, -0 - +0 = -0, +0 - -0 = +0",
                [np.array([1.0, np.inf, -0.0, 0.0], dtype=f32),
                 np.array([1.0, np.inf, 0.0, -0.0], dtype=f32)]))

    # the four signed-zero differences, each on its own
    out.append(("float: +0 - +0 = +0",
                [np.array([0.0], dtype=f32), np.array([0.0], dtype=f32)]))
    out.append(("float: +0 - -0 = +0",
                [np.array([0.0], dtype=f32), np.array([-0.0], dtype=f32)]))
    out.append(("float: -0 - -0 = +0",
                [np.array([-0.0], dtype=f32), np.array([-0.0], dtype=f32)]))
    out.append(("float: -0 - +0 = -0",
                [np.array([-0.0], dtype=f32), np.array([0.0], dtype=f32)]))

    # the infinities: the six combinations that give an infinity, and the two that give NaN
    out.append(("float: +inf - finite = +inf",
                [np.array([np.inf], dtype=f32), np.array([1.0], dtype=f32)]))
    out.append(("float: finite - -inf = +inf",
                [np.array([1.0], dtype=f32), np.array([-np.inf], dtype=f32)]))
    out.append(("float: +inf - -inf = +inf",
                [np.array([np.inf], dtype=f32), np.array([-np.inf], dtype=f32)]))
    out.append(("float: -inf - finite = -inf",
                [np.array([-np.inf], dtype=f32), np.array([1.0], dtype=f32)]))
    out.append(("float: finite - +inf = -inf",
                [np.array([1.0], dtype=f32), np.array([np.inf], dtype=f32)]))
    out.append(("float: -inf - +inf = -inf",
                [np.array([-np.inf], dtype=f32), np.array([np.inf], dtype=f32)]))
    out.append(("float: +inf - +inf = NaN",
                [np.array([np.inf], dtype=f32), np.array([np.inf], dtype=f32)]))
    out.append(("float: -inf - -inf = NaN",
                [np.array([-np.inf], dtype=f32), np.array([-np.inf], dtype=f32)]))

    # NaN propagation
    out.append(("float: NaN - 1 = NaN",
                [np.array([np.nan], dtype=f32), np.array([1.0], dtype=f32)]))
    out.append(("float: 1 - NaN = NaN",
                [np.array([1.0], dtype=f32), np.array([np.nan], dtype=f32)]))

    # the subnormal range: the smallest subnormal, and a difference that is subnormal
    sub = _smallest_subnormal(np.dtype(f32))
    out.append(("float: smallest subnormal - 0 = smallest subnormal",
                [np.array([sub], dtype=f32), np.array([0.0], dtype=f32)]))
    out.append(("float: 0 - smallest subnormal = -smallest subnormal",
                [np.array([0.0], dtype=f32), np.array([sub], dtype=f32)]))
    out.append(("float: 2*smallest subnormal - smallest subnormal = smallest subnormal",
                [np.array([2.0 * sub], dtype=f32), np.array([sub], dtype=f32)]))

    # the extremes and the overflow threshold
    maxf = float(np.finfo(f32).max)
    out.append(("float: max - (-max) = +inf (overflow)",
                [np.array([maxf], dtype=f32), np.array([-maxf], dtype=f32)]))
    out.append(("float: -max - max = -inf (overflow)",
                [np.array([-maxf], dtype=f32), np.array([maxf], dtype=f32)]))
    out.append(("float: max - max = +0",
                [np.array([maxf], dtype=f32), np.array([maxf], dtype=f32)]))

    # the rounding boundary: 1.0 - 2^-24 is the midpoint between 1.0 and 1.0 - 2^-23
    out.append(("float: 1.0 - 2^-24 (rounds to 1.0, ties to even)",
                [np.array([1.0], dtype=f32), np.array([2.0 ** -24], dtype=f32)]))
    out.append(("float: 1.0 - 3*2^-25 (rounds to 1.0 - 2^-23)",
                [np.array([1.0], dtype=f32), np.array([3.0 * 2.0 ** -25], dtype=f32)]))

    # float16: the same special values, and the smallest subnormal
    sub16 = _smallest_subnormal(np.dtype(f16))
    out.append(("float16: -0 - +0 = -0",
                [np.array([-0.0], dtype=f16), np.array([0.0], dtype=f16)]))
    out.append(("float16: +inf - +inf = NaN",
                [np.array([np.inf], dtype=f16), np.array([np.inf], dtype=f16)]))
    out.append(("float16: smallest subnormal - 0",
                [np.array([sub16], dtype=f16), np.array([0.0], dtype=f16)]))
    out.append(("float16: max - (-max) = +inf",
                [np.array([float(np.finfo(f16).max)], dtype=f16),
                 np.array([-float(np.finfo(f16).max)], dtype=f16)]))

    # float64: the same special values
    out.append(("float64: -0 - +0 = -0",
                [np.array([-0.0], dtype=f64), np.array([0.0], dtype=f64)]))
    out.append(("float64: +inf - +inf = NaN",
                [np.array([np.inf], dtype=f64), np.array([np.inf], dtype=f64)]))
    out.append(("float64: 0.3 - 0.1 (the rounded difference)",
                [np.array([0.3], dtype=f64), np.array([0.1], dtype=f64)]))

    # Example 1 of the float section
    out.append(("float Example 1",
                [np.array([[3.0, 4.5], [16.0, 1.0], [25.5, 24.25]], dtype=f32),
                 np.array([[3.0, 2.0], [4.0, 0.0], [5.0, 4.0]], dtype=f32)]))

    # Example 3 of the float section
    out.append(("float Example 3",
                [np.array([0.3, 1.0], dtype=f64),
                 np.array([0.1, 1.0], dtype=f64)]))

    return out


def _int_cases():
    """The int section: the worked examples, the overflow boundaries, the sign."""
    out = []
    i8 = np.int8
    i16 = np.int16
    i32 = np.int32
    i64 = np.int64

    out.append(("int Example 1",
                [np.array([6, 5, -35], dtype=i8),
                 np.array([3, 3, 3], dtype=i8)]))
    out.append(("int Example 2: 127 - (-1) = -128, -128 - 1 = 127, 0 - 1 = -1",
                [np.array([127, -128, 0], dtype=i8),
                 np.array([-1, 1, 1], dtype=i8)]))

    # the overflow boundaries of each signed type
    for dt in (i8, i16, i32, i64):
        info = np.iinfo(dt)
        out.append(("%s: max - (-1) = min (overflow)" % np.dtype(dt).name,
                    [np.array([info.max], dtype=dt),
                     np.array([-1], dtype=dt)]))
        out.append(("%s: min - 1 = max (overflow)" % np.dtype(dt).name,
                    [np.array([info.min], dtype=dt),
                     np.array([1], dtype=dt)]))
        out.append(("%s: min - min = 0" % np.dtype(dt).name,
                    [np.array([info.min], dtype=dt),
                     np.array([info.min], dtype=dt)]))
        out.append(("%s: max - max = 0" % np.dtype(dt).name,
                    [np.array([info.max], dtype=dt),
                     np.array([info.max], dtype=dt)]))

    return out


def _uint_cases():
    """The uint section: the worked example, the underflow boundary."""
    out = []
    u8 = np.uint8
    u16 = np.uint16
    u32 = np.uint32
    u64 = np.uint64

    out.append(("uint Example 1: 0 - 1 = 255, 255 - 1 = 254, 4 - 7 = 253",
                [np.array([0, 255, 4], dtype=u8),
                 np.array([1, 1, 7], dtype=u8)]))

    for dt in (u8, u16, u32, u64):
        info = np.iinfo(dt)
        out.append(("%s: 0 - 1 = max (underflow)" % np.dtype(dt).name,
                    [np.array([0], dtype=dt),
                     np.array([1], dtype=dt)]))
        out.append(("%s: max - 0 = max" % np.dtype(dt).name,
                    [np.array([info.max], dtype=dt),
                     np.array([0], dtype=dt)]))
        out.append(("%s: max - max = 0" % np.dtype(dt).name,
                    [np.array([info.max], dtype=dt),
                     np.array([info.max], dtype=dt)]))

    return out


def _shape_cases():
    """The shape families of the document: same shape, dim 1, smaller rank, rank-0,
    zero-sized."""
    out = []
    f32 = np.float32
    i32 = np.int32

    out.append(("shape: same shape (2, 3)",
                [np.arange(6, dtype=f32).reshape(2, 3),
                 np.arange(6, dtype=f32).reshape(2, 3) + 1.0]))
    out.append(("shape: dim 1 broadcast (2, 3) - (1, 3)",
                [np.arange(6, dtype=f32).reshape(2, 3),
                 np.arange(3, dtype=f32).reshape(1, 3)]))
    out.append(("shape: smaller rank (2, 3, 4) - (4,)",
                [np.arange(24, dtype=f32).reshape(2, 3, 4),
                 np.arange(4, dtype=f32)]))
    out.append(("shape: smaller rank (2, 3, 4) - (3, 1)",
                [np.arange(24, dtype=f32).reshape(2, 3, 4),
                 np.arange(3, dtype=f32).reshape(3, 1)]))
    out.append(("shape: rank-0 A broadcast to (2, 3)",
                [np.array(2.5, dtype=f32),
                 np.arange(6, dtype=f32).reshape(2, 3)]))
    out.append(("shape: rank-0 B broadcast to (2, 3)",
                [np.arange(6, dtype=f32).reshape(2, 3),
                 np.array(2.5, dtype=f32)]))
    out.append(("shape: rank-0 - rank-0",
                [np.array(2.5, dtype=f32), np.array(1.0, dtype=f32)]))
    out.append(("shape: zero-sized (0, 3) - (1, 3)",
                [np.zeros((0, 3), dtype=f32),
                 np.arange(3, dtype=f32).reshape(1, 3)]))
    out.append(("shape: zero-sized (0,) - (1,)",
                [np.zeros((0,), dtype=f32),
                 np.arange(1, dtype=f32)]))
    out.append(("shape: zero-sized (0, 3) - (3,)",
                [np.zeros((0, 3), dtype=f32),
                 np.arange(3, dtype=f32)]))
    out.append(("shape: zero-sized (1, 0) - (2, 0)",
                [np.zeros((1, 0), dtype=f32),
                 np.zeros((2, 0), dtype=f32)]))
    out.append(("shape: zero-sized (0,) - rank-0",
                [np.zeros((0,), dtype=f32),
                 np.array(1.0, dtype=f32)]))
    out.append(("shape: int32 (2, 3) - (3,)",
                [np.arange(6, dtype=i32).reshape(2, 3),
                 np.arange(3, dtype=i32)]))

    return out


def cases():
    """Every case of every section of the document."""
    for item in _float_cases():
        yield item
    for item in _int_cases():
        yield item
    for item in _uint_cases():
        yield item
    for item in _shape_cases():
        yield item


# ---------------------------------------------------------------------------
# coverage: every section anchor of the document, with the number of cases
# ---------------------------------------------------------------------------

COVERAGE = {
    "real": 0,
    "float": 0,
    "int": 0,
    "uint": 0,
}

for _label, _arrays in cases():
    _dt = np.asarray(_arrays[0]).dtype
    if _dt.kind == "f":
        COVERAGE["float"] += 1
    elif _dt.kind == "i":
        COVERAGE["int"] += 1
    elif _dt.kind == "u":
        COVERAGE["uint"] += 1

# the real section has no ONNX type and no case in cases(); its cases are the three
# worked examples of the real section, checked in sweep_real_examples
COVERAGE["real"] = 3
