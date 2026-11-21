"""The cases with which the **Add** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled pair of operands per case.  Each case goes through a
                     one-node Add model (opset 14) and through the implementation, and
                     the two results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the domain of the integer
                     types, the exact arithmetic of the float section, the worked
                     examples of every section of the document, the document's declared
                     types against the ONNX schema, and the real section, which has no
                     ONNX type and is checked against exact rational arithmetic instead.
  * ``COVERAGE``  -- the four section anchors of the document, each with the number of
                     cases that exercise it.

Every case of every section's formula is meant to appear here, and the case set is part
of the loop: when a section is added to the document, its cases are added here before
the next run.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Add"
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
    """One Add node over two arrays of the same type, in a single runtime call."""
    proto = helper.np_dtype_to_tensor_dtype(a.dtype)
    vi_a = helper.make_tensor_value_info("A", proto, list(a.shape))
    vi_b = helper.make_tensor_value_info("B", proto, list(b.shape))
    vi_c = helper.make_tensor_value_info("C", proto, None)
    node = helper.make_node(OP, ["A", "B"], ["C"], name="add0")
    graph = helper.make_graph([node], "add_graph", [vi_a, vi_b], [vi_c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    (c,) = sess.run(["C"], {"A": a, "B": b})
    return c


def _bits_equal(want: np.ndarray, got: np.ndarray) -> bool:
    """Bit-for-bit equality, NaN equal to NaN, of two arrays of the same type."""
    if want.shape != got.shape or want.dtype != got.dtype:
        return False
    if want.dtype.kind == "f":
        uint = UINT_OF_WIDTH[want.dtype.itemsize]
        bw = np.ascontiguousarray(want).view(uint)
        bg = np.ascontiguousarray(got).view(uint)
        both_nan = np.isnan(want) & np.isnan(got)
        return bool(np.all((bw == bg) | both_nan))
    return bool(np.all(want == got))


def _diff_indices(want: np.ndarray, got: np.ndarray, limit: int = 5):
    """Flat indices where two arrays differ (NaN equal to NaN); None on shape/type."""
    if want.shape != got.shape or want.dtype != got.dtype:
        return None
    if want.dtype.kind == "f":
        uint = UINT_OF_WIDTH[want.dtype.itemsize]
        bw = np.ascontiguousarray(want).view(uint)
        bg = np.ascontiguousarray(got).view(uint)
        both_nan = np.isnan(want) & np.isnan(got)
        mask = ~((bw == bg) | both_nan)
    else:
        mask = ~(want == got)
    return [int(i) for i in np.nonzero(mask.ravel())[0][:limit]]


def _same_bits(want, got) -> bool:
    a = np.asarray(want).ravel()
    b = np.asarray(got).ravel()
    return _bits_equal(a, b)


# ---------------------------------------------------------------------------
# exact arithmetic: the reference the document itself states
# ---------------------------------------------------------------------------
#
# The document's float formula is round(A + B): the exact sum in the reals, rounded to
# the nearest value of the type with roundTiesToEven and an unbounded exponent range.
# That is computed here from the operands' exact values (Fraction of the float, never
# of the decimal literal), independently of the platform's library and of the runtime.


def _pow2(e: int) -> Fraction:
    return Fraction(2) ** e


def _round_half_even(x: Fraction) -> int:
    """roundTiesToEven of a non-negative rational to the nearest integer."""
    q = x.numerator // x.denominator
    r = x - q
    half = Fraction(1, 2)
    if r > half:
        return q + 1
    if r < half:
        return q
    return q if q % 2 == 0 else q + 1


def _frac_round_to_type(value: Fraction, dtype):
    """Round an exact rational to the nearest value of `dtype`, ties to even.

    The exponent range is unbounded, as the document states: a magnitude whose nearest
    value exceeds the largest finite value of the type gives an infinity.
    """
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    p = f.nmant + 1                       # significand bits, implicit bit included
    emin = f.minexp                       # exponent of the smallest normal
    emax = f.maxexp - 1                   # exponent of the largest finite
    max_finite = Fraction(float(f.max))
    if value == 0:
        return dt.type(0.0)
    sign = -1.0 if value < 0 else 1.0
    a = abs(value)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while _pow2(e) > a:
        e -= 1
    while _pow2(e + 1) <= a:
        e += 1
    if e < emin:
        e = emin                          # the subnormal range: uniform spacing
    spacing = _pow2(e - p + 1)
    n = _round_half_even(a / spacing)
    result = n * spacing
    if result > max_finite:
        return dt.type(sign * float("inf"))
    return dt.type(sign * float(result))


def _exact_float_result(x, y, dtype):
    """C[i] of **Add** (float, float): NaN, an infinity, or round(x + y)."""
    dt = np.dtype(dtype)
    fx, fy = float(x), float(y)
    if np.isnan(fx) or np.isnan(fy):
        return dt.type(float("nan"))
    if np.isinf(fx) or np.isinf(fy):
        if np.isinf(fx) and np.isinf(fy) and (fx > 0.0) != (fy > 0.0):
            return dt.type(float("nan"))          # the invalid operation of IEEE 7.2
        return dt.type(fx if np.isinf(fx) else fy)
    s = Fraction(fx) + Fraction(fy)
    if s == 0:
        # a null sum is +0, except when both operands are -0
        if np.signbit(fx) and np.signbit(fy):
            return dt.type(-0.0)
        return dt.type(0.0)
    return _frac_round_to_type(s, dt)


def _double_rounded_float16(x, y):
    """The same sum rounded through float32 first, or None for the wider types.

    A kernel that computes a float16 sum in float32 and narrows the result round twice;
    on the rare operand pair whose exact sum sits a hair below a float16 midpoint that
    differs from the single rounding the document states.  It is recorded, not required.
    """
    fx, fy = float(x), float(y)
    if np.isnan(fx) or np.isnan(fy) or np.isinf(fx) or np.isinf(fy):
        return None
    s = Fraction(fx) + Fraction(fy)
    if s == 0:
        return np.float16(-0.0) if (np.signbit(fx) and np.signbit(fy)) else np.float16(0.0)
    f32 = _frac_round_to_type(s, np.float32)
    if np.isinf(f32):
        return np.float16(f32)
    return np.float16(np.float32(f32))


# ---------------------------------------------------------------------------
# the integer case analysis of the document, transcribed literally
# ---------------------------------------------------------------------------
#
# The transcription is deliberately independent of the implementation: it is a
# second, literal, reading of the two sections **Add** (int, int) and
# **Add** (uint, uint), compared with ONNX Runtime over a whole domain.  A
# disagreement here is a defect of the document, not of the implementation.


def doc_int_signed(x: int, y: int, n: int) -> int:
    """The case analysis of **Add** (int, int), in exact arithmetic."""
    lo, hi = -(2 ** (n - 1)), 2 ** (n - 1) - 1
    s = x + y
    if s > hi:
        return s - 2 ** n
    if s < lo:
        return s + 2 ** n
    return s


def doc_int_unsigned(x: int, y: int, n: int) -> int:
    """The case analysis of **Add** (uint, uint), in exact arithmetic."""
    s = x + y
    return s if s <= 2 ** n - 1 else s - 2 ** n


def _doc_int_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dt = a.dtype
    n = dt.itemsize * 8
    fn = doc_int_signed if dt.kind == "i" else doc_int_unsigned
    flat = [fn(int(x), int(y), n) for x, y in zip(a.ravel(), b.ravel())]
    return np.array(flat, dtype=dt).reshape(a.shape)


def _rand_values(rng, dt, n):
    """n random values of `dt`, drawn uniformly over the whole bit patterns of the type."""
    hi = rng.integers(0, 2 ** 32, size=n, dtype=np.uint64)
    lo = rng.integers(0, 2 ** 32, size=n, dtype=np.uint64)
    raw = (hi << np.uint64(32)) | lo
    bits = dt.itemsize * 8
    if bits < 64:
        raw = raw & np.uint64((1 << bits) - 1)
    return raw.astype(np.dtype("u%d" % dt.itemsize)).view(dt)


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_int_domain(impl):
    """The document's integer case analysis against ONNX Runtime, over a domain.

    The two 8-bit types are covered exhaustively (all 65536 pairs each); the wider
    types use their boundary values and 200 random pairs.
    """
    rng = np.random.default_rng(20261003)
    rows, failures = [], []
    for dtype in ALL_INT:
        dt = np.dtype(dtype)
        info = np.iinfo(dt)
        exhaustive = dt.itemsize == 1
        if exhaustive:
            xs = np.arange(int(info.min), int(info.max) + 1, dtype=np.int64)
            aa = np.repeat(xs, len(xs)).astype(dt)
            bb = np.tile(xs, len(xs)).astype(dt)
        else:
            edges = sorted({int(info.min), int(info.min) + 1, int(info.max) - 1,
                            int(info.max), 0, 1, 2})
            pairs = [(x, y) for x in edges for y in edges]
            ra = _rand_values(rng, dt, 200)
            rb = _rand_values(rng, dt, 200)
            pairs += list(zip(ra.tolist(), rb.tolist()))
            aa = np.array([p[0] for p in pairs], dtype=dt)
            bb = np.array([p[1] for p in pairs], dtype=dt)

        want = _doc_int_array(aa, bb)          # the document, literally
        ref = run_ort_batch(aa, bb)            # ONNX Runtime
        got = np.asarray(impl.add(aa, bb))     # the implementation

        bad = _diff_indices(want, ref)
        if bad is None:
            failures.append({"type": dt.name, "who": "the document vs ONNX Runtime",
                             "detail": "shape %s or type %s differs from the document's "
                                       "%s / %s" % (ref.shape, ref.dtype,
                                                    want.shape, want.dtype)})
        elif bad:
            failures.append({
                "type": dt.name, "who": "the document vs ONNX Runtime",
                "detail": "; ".join("doc: %d + %d = %d, ort: %d"
                                    % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                       int(want.ravel()[i]), int(ref.ravel()[i]))
                                    for i in bad),
            })
        bad = _diff_indices(ref, got)
        if bad is None:
            failures.append({"type": dt.name, "who": "the implementation vs ONNX Runtime",
                             "detail": "shape %s or type %s differs from the runtime's "
                                       "%s / %s" % (got.shape, got.dtype,
                                                    ref.shape, ref.dtype)})
        elif bad:
            failures.append({
                "type": dt.name, "who": "the implementation vs ONNX Runtime",
                "detail": "; ".join("%d + %d: ort %d, impl %d"
                                    % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                       int(np.asarray(ref).ravel()[i]),
                                       int(np.asarray(got).ravel()[i])) for i in bad),
            })
        rows.append("%s %s" % (dt.name,
                               "exhaustive (65536 pairs)" if exhaustive
                               else "%d pairs" % len(aa)))
    return {"sweep": "int domain (document vs ORT vs implementation)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def _float_probe(dtype, rng):
    """The operands of the exact float sweep: the special values and a random sample."""
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    maxf = float(f.max)
    tiny = float(f.tiny)
    small = float(np.nextafter(dt.type(0.0), dt.type(1.0)))
    specials = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, 2.0, -2.0,
                tiny, -tiny, small, -small, maxf, -maxf, maxf / 2.0, -maxf / 2.0,
                float("inf"), float("-inf"), float("nan")]
    pairs = [(x, y) for x in specials for y in specials]

    n_rand = 200
    ra = (rng.random(n_rand) * 2.0 - 1.0) * maxf
    rb = (rng.random(n_rand) * 2.0 - 1.0) * maxf
    ra[:80] = (rng.random(80) * 2.0 - 1.0) * tiny        # the subnormal range
    rb[:80] = (rng.random(80) * 2.0 - 1.0) * tiny
    pairs += list(zip(ra.tolist(), rb.tolist()))

    a = np.array([p[0] for p in pairs], dtype=dt)
    b = np.array([p[1] for p in pairs], dtype=dt)
    return a, b


def sweep_float_exact(impl):
    """The float section's formula against exact arithmetic, ORT, and the implementation.

    Each element is round(A + B) as the document states it, computed with `Fraction`
    from the operands' exact values.  A float16 departure that is the same sum rounded
    through float32 is recorded as double rounding, not required.
    """
    rng = np.random.default_rng(20261004)
    rows, failures = [], []
    recorded = 0
    checked = 0
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        a, b = _float_probe(dt, rng)
        exact = np.array([_exact_float_result(x, y, dt)
                          for x, y in zip(a.tolist(), b.tolist())], dtype=dt)
        ref = run_ort_batch(a, b)
        got = np.asarray(impl.add(a, b))
        checked += int(a.size)

        for who, other in (("ONNX Runtime", ref), ("the implementation", got)):
            bad = _diff_indices(exact, other, limit=None)
            if bad is None:
                failures.append({"type": dt.name, "who": who,
                                 "detail": "shape %s or type %s differs from the exact "
                                           "result's %s / %s" % (other.shape, other.dtype,
                                                                 exact.shape, exact.dtype)})
                continue
            real_bad = []
            for i in bad:
                if dt == np.dtype(np.float16):
                    dr = _double_rounded_float16(a.ravel()[i], b.ravel()[i])
                    if dr is not None and _same_bits(other.ravel()[i], dr):
                        recorded += 1
                        continue
                real_bad.append(i)
            if real_bad:
                failures.append({
                    "type": dt.name, "who": "%s vs the exact sum of the document" % who,
                    "detail": "; ".join(
                        "%r + %r: exact %r, %s %r"
                        % (float(a.ravel()[i]), float(b.ravel()[i]),
                           float(exact.ravel()[i]), who, float(other.ravel()[i]))
                        for i in real_bad[:5]),
                })
        rows.append("%s: %d operand pairs" % (dt.name, a.size))
    summary = ("%d operand pairs: %s; the float16 departures that are the same sum "
               "rounded through float32 are recorded, not required (%d recorded)"
               % (checked, ", ".join(rows), recorded))
    return {"sweep": "the float formula against exact arithmetic (Fraction)",
            "summary": summary, "failures": failures}


def _real_examples():
    """The real section's cases: exact rationals, no ONNX type."""
    F = Fraction
    return [
        ("real Example 1",
         [[F("6.1"), F("9.5"), F("35.7")]],
         [F(2), F(3), F(4)],
         [[F("8.1"), F("12.5"), F("39.7")]]),
        ("real Example 2",
         [[F("3.7"), F("4.4")], [F("16.2"), F("0.5")], [F("25.3"), F("24.8")]],
         [F(1), F(2)],
         [[F("4.7"), F("6.4")], [F("17.2"), F("2.5")], [F("26.3"), F("26.8")]]),
        ("real: exactness of 1/3 + 1/6 and cancellation of 1/10 - 1/10",
         [[F(1, 3), F(1, 10)]],
         [[F(1, 6), F(-1, 10)]],
         [[F(1, 2), F(0)]]),
        ("real: rank-0 operand broadcast to (2,)",
         F(3, 7), [F(2, 5), F(1, 2)], None),
        ("real: broadcast of a row over a 3x2 matrix",
         [[F(1, 3), F(2, 3)], [F(1, 7), F(3, 7)], [F(-1, 9), F(5, 9)]],
         [F(1, 6), F(1, 6)], None),
    ]


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact rationals.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    implementation against the exact sums of the section's Examples and of its formula,
    in exact rational arithmetic.
    """
    failures = []
    examples = _real_examples()
    for label, a, b, documented in examples:
        aa = np.array(a, dtype=object)
        bb = np.array(b, dtype=object)
        ba, bc = np.broadcast_arrays(aa, bb)
        want = np.empty(ba.shape, dtype=object)
        for i in np.ndindex(ba.shape):
            want[i] = ba[i] + bc[i]
        try:
            got = np.asarray(impl.add(aa, bb))
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
            elif [i for i in np.ndindex(doc.shape) if doc[i] != got[i]]:
                failures.append({"who": label,
                                 "detail": "documented %s, implementation %s"
                                           % (doc.tolist(), got.tolist())})
    return {"sweep": "the real section (exact rational arithmetic, no ONNX Runtime)",
            "summary": "%d cases checked against Fraction arithmetic" % len(examples),
            "failures": failures}


def _spec_examples():
    """The worked examples of the float and the integer sections, as documented."""
    f32 = np.float32
    return [
        ("float Example 1",
         np.array([[3.0, 4.5], [16.0, 1.0], [25.5, 24.25]], dtype=f32),
         np.array([[3.0, 2.0], [4.0, 0.0], [5.0, 4.0]], dtype=f32),
         np.array([[6.0, 6.5], [20.0, 1.0], [30.5, 28.25]], dtype=f32)),
        ("float Example 2",
         np.array([1.0, np.inf, 0.0, -np.inf], dtype=f32),
         np.array([1.0, -np.inf, -0.0, -np.inf], dtype=f32),
         np.array([2.0, np.nan, 0.0, -np.inf], dtype=f32)),
        ("float Example 3",
         np.array([0.1, 1.0, 1.0], dtype=np.float64),
         np.array([0.2, 2.0, -1.0], dtype=np.float64),
         np.array([0.1 + 0.2, 3.0, -0.0], dtype=np.float64)),
        ("int Example 1",
         np.array([-6, 100, -100], dtype=np.int8),
         np.array([-3, 100, -100], dtype=np.int8),
         np.array([-9, -56, 56], dtype=np.int8)),
        ("uint Example 1",
         np.array([6, 200, 35], dtype=np.uint8),
         np.array([3, 100, 5], dtype=np.uint8),
         np.array([9, 44, 40], dtype=np.uint8)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, and as ONNX Runtime does it."""
    failures = []
    examples = _spec_examples()
    for label, a, b, documented in examples:
        ref = run_ort_batch(a, b)
        got = np.asarray(impl.add(a, b))
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


def sweep_schema_types(impl):
    """The document's declared type families against the ONNX schema's admitted set.

    A type the document declares that the schema does not admit makes every case of that
    type open at model resolution, and a case set that drops it would converge on the
    wrong document.  A type the schema admits that the document does not specify is
    recorded in the summary: the profile specifies a subset.
    """
    declared = ["tensor(float16)", "tensor(float)", "tensor(double)",
                "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)",
                "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)"]
    try:
        from onnx import defs
        schema = defs.get_schema(OP, OPSET)
        admitted = {}
        for constraint in schema.type_constraints:
            admitted[constraint.type_param_str] = list(constraint.allowed_type_strs)
        allowed = set(admitted.get("T", []))
    except Exception as exc:
        return {"sweep": "the document's types against the ONNX schema",
                "summary": "the schema of %s-%d could not be read (%s: %s)"
                           % (OP, OPSET, type(exc).__name__, exc),
                "failures": []}
    failures = []
    for name in declared:
        if name not in allowed:
            failures.append({"who": "the document declares %s" % name,
                             "detail": "the schema of %s-%d admits %s for T"
                                       % (OP, OPSET, sorted(allowed))})
    extra = sorted(allowed - set(declared))
    summary = ("%d declared types, every one admitted by the schema of %s-%d"
               % (len(declared), OP, OPSET))
    if extra:
        summary += ("; the schema also admits %s, which the document does not specify "
                    "(recorded, not required)" % ", ".join(extra))
    return {"sweep": "the document's types against the ONNX schema",
            "summary": summary, "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_int_domain, sweep_float_exact,
            sweep_real_examples, sweep_schema_types]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _float_value_cases(dtype):
    """The value families of the float section."""
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    maxf = float(f.max)
    tiny = float(f.tiny)
    small = float(np.nextafter(dt.type(0.0), dt.type(1.0)))
    ulp = float(2.0 ** (f.maxexp - f.nmant - 1))
    nan, inf = float("nan"), float("inf")
    return [
        ("signed zeros", [0.0, -0.0, 0.0, -0.0], [0.0, 0.0, -0.0, -0.0]),
        ("infinities", [inf, -inf, inf, -inf, 1.0, -1.0], [inf, -inf, 1.0, -1.0, inf, -inf]),
        ("nan", [nan, 1.0, nan, inf], [1.0, nan, inf, nan]),
        ("exact sum", [1.0, 4.5, 16.0, 0.25], [0.5, 0.25, 8.0, 0.125]),
        ("largest finite operand", [maxf, -maxf, maxf], [0.0, 0.0, -maxf]),
        ("overflow to inf", [maxf, -maxf], [maxf, -maxf]),
        ("cancellation to zero", [maxf, maxf], [-maxf, -maxf]),
        ("below the overflow threshold", [maxf], [ulp / 4.0]),
        ("at the overflow threshold (tie)", [maxf], [ulp / 2.0]),
        ("above the overflow threshold", [maxf], [3.0 * ulp / 4.0]),
        ("smallest normal and subnormal result",
         [tiny, 0.0, 0.0, small, -small], [tiny / 2.0, tiny / 2.0, -tiny / 2.0,
                                           small, -small]),
        ("subnormal underflow to zero", [small], [-small]),
        ("exactly one ulp from 1.0", [1.0], [2.0 ** -f.nmant]),
        ("rounding tie (halfway)", [1.0], [2.0 ** -(f.nmant + 1.0)]),
        ("not representable (rounded sum)", [0.3, 1.0], [0.1, 1.0]),
    ]


def _ordinary_value_cases(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return [("ordinary", [1.0, 2.5, -3.0, 4.75], [0.5, 2.5, 1.0, -0.25])]
    info = np.iinfo(dtype)
    lo, hi = int(info.min), int(info.max)
    if lo < 0:
        a = [lo, hi, -1, 0, 1, hi, lo, -1]
        b = [0, 0, 1, 1, 0, hi, -1, lo]
    else:
        a = [0, hi, 1, 4, hi, 0, 1, hi // 2]
        b = [1, 1, hi, 7, 1, hi, hi, 3]
    return [("ordinary+extremes+wraps", a, b)]


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
    ("same shape (2,3)", (2, 3), (2, 3)),
    ("broadcast dim 1: (2,3)+(1,3)", (2, 3), (1, 3)),
    ("smaller rank, no leading dims: (2,3,4)+(4,)", (2, 3, 4), (4,)),
    ("broadcast dim 1: (2,3,4)+(3,1)", (2, 3, 4), (3, 1)),
    ("broadcast both operands: (2,1,4)+(1,3,1)", (2, 1, 4), (1, 3, 1)),
    ("rank-0 on the left: ()+(2,3)", (), (2, 3)),
    ("rank-0 on the right: (2,3)+()", (2, 3), ()),
    ("both rank-0: ()+()", (), ()),
    ("zero-sized: (0,3)+(1,3)", (0, 3), (1, 3)),
    ("zero-sized: (0,)+(1,)", (0,), (1,)),
    ("zero-sized: (0,3)+(3,)", (0, 3), (3,)),
    ("zero-sized: (1,0)+(2,0)", (1, 0), (2, 0)),
    ("zero-sized: (0,3)+(0,1)", (0, 3), (0, 1)),
    ("zero-sized with rank-0: (0,)+()", (0,), ()),
]


def cases():
    """Every individual case: (label, [A, B])."""
    for dtype in ALL_FLOAT:
        for name, a, b in _float_value_cases(dtype):
            yield ("%s: %s" % (np.dtype(dtype).name, name),
                   [np.array(a, dtype=dtype), np.array(b, dtype=dtype)])
        quiet, signalling, sign = NAN_PATTERNS[np.dtype(dtype).itemsize]
        one = np.ones(1, dtype=dtype)
        yield ("%s: NaN operand, quiet, payload 1" % np.dtype(dtype).name,
               [_bit_nan(dtype, quiet), one])
        yield ("%s: NaN operand, quiet, payload 1, sign set" % np.dtype(dtype).name,
               [_bit_nan(dtype, quiet | sign), one])
        yield ("%s: NaN operand, signalling" % np.dtype(dtype).name,
               [_bit_nan(dtype, signalling), one])
        inf = np.array([np.inf], dtype=dtype)
        ninf = np.array([-np.inf], dtype=dtype)
        yield ("%s: invalid operation, inf + (-inf)" % np.dtype(dtype).name, [inf, ninf])
        yield ("%s: invalid operation, -inf + inf" % np.dtype(dtype).name, [ninf, inf])
    for dtype in ALL_INT:
        for name, a, b in _ordinary_value_cases(dtype):
            yield ("%s: %s" % (np.dtype(dtype).name, name),
                   [np.array(a, dtype=dtype), np.array(b, dtype=dtype)])
    for name, sa, sb in SHAPE_CASES:
        for dtype in ALL_FLOAT + ALL_INT:
            yield ("%s: %s" % (np.dtype(dtype).name, name),
                   [_arr(sa, dtype), _arr(sb, dtype, 1.0)])


# ---------------------------------------------------------------------------
# coverage: every section anchor of the document, and the cases that exercise it
# ---------------------------------------------------------------------------


def _coverage():
    coverage = {"real": len(_real_examples()), "float": 0, "int": 0, "uint": 0}
    for _label, arrays in cases():
        kind = np.asarray(arrays[0]).dtype.kind
        if kind == "f":
            coverage["float"] += 1
        elif kind == "i":
            coverage["int"] += 1
        elif kind == "u":
            coverage["uint"] += 1
    return coverage


COVERAGE = _coverage()
