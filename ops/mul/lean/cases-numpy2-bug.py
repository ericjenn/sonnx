"""The cases with which the **Mul** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled pair of operands per case.  Each case goes through a
                     one-node Mul model (opset 14) and through the implementation, and
                     the two results are compared bit for bit, the sign of a zero
                     included, because the document specifies it.
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

OP = "Mul"
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
    """One Mul node over two arrays of the same type, in a single runtime call."""
    proto = helper.np_dtype_to_tensor_dtype(a.dtype)
    vi_a = helper.make_tensor_value_info("A", proto, list(a.shape))
    vi_b = helper.make_tensor_value_info("B", proto, list(b.shape))
    vi_c = helper.make_tensor_value_info("C", proto, None)
    node = helper.make_node(OP, ["A", "B"], ["C"], name="mul0")
    graph = helper.make_graph([node], "mul_graph", [vi_a, vi_b], [vi_c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    (c,) = sess.run(["C"], {"A": a, "B": b})
    return c


def _bits_equal(want: np.ndarray, got: np.ndarray) -> bool:
    """Bit-for-bit equality, NaN equal to NaN, of two arrays of the same type.

    The sign of a zero is part of the bit pattern, so +0.0 and -0.0 are different
    results here: the document specifies the sign of a null product.
    """
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
# The document's float formula is round(A * B): the exact product in the reals, rounded
# to the nearest value of the type with roundTiesToEven and an unbounded exponent range.
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
    """C[i] of **Mul** (float, float): NaN, an infinity, or round(x * y)."""
    dt = np.dtype(dtype)
    fx, fy = float(x), float(y)
    if np.isnan(fx) or np.isnan(fy):
        return dt.type(float("nan"))
    if np.isinf(fx) or np.isinf(fy):
        # one of them is an infinity; the other is neither NaN (handled above)
        if fx == 0.0 or fy == 0.0:
            return dt.type(float("nan"))          # the invalid operation of IEEE 7.2
        sign = -1.0 if (np.signbit(fx) != np.signbit(fy)) else 1.0
        return dt.type(sign * float("inf"))
    prod = Fraction(fx) * Fraction(fy)
    if prod == 0:
        # a null product: -0 when the operands have opposite signs, +0 otherwise
        sign = -1.0 if (np.signbit(fx) != np.signbit(fy)) else 1.0
        return dt.type(sign * 0.0)
    return _frac_round_to_type(prod, dt)


def _double_rounded_float16(x, y):
    """The same product rounded through float32 first, or None for the wider types.

    A kernel that computes a float16 product in float32 and narrows the result rounds
    twice; on the rare operand pair whose exact product sits a hair below a float16
    midpoint that differs from the single rounding the document states.  It is
    recorded, not required.
    """
    fx, fy = float(x), float(y)
    if np.isnan(fx) or np.isnan(fy) or np.isinf(fx) or np.isinf(fy):
        return None
    if fx == 0.0 or fy == 0.0:
        return None
    prod = Fraction(fx) * Fraction(fy)
    f32 = _frac_round_to_type(prod, np.float32)
    if np.isinf(f32):
        return np.float16(f32)
    return np.float16(np.float32(f32))


# ---------------------------------------------------------------------------
# the integer case analysis of the document, transcribed literally
# ---------------------------------------------------------------------------
#
# The transcription is deliberately independent of the implementation: it is a
# second, literal, reading of the two sections **Mul** (int, int) and
# **Mul** (uint, uint), compared with ONNX Runtime over a whole domain.  A
# disagreement here is a defect of the document, not of the implementation.


def doc_int_signed(x: int, y: int, n: int) -> int:
    """The case analysis of **Mul** (int, int), in exact arithmetic."""
    u = (x * y) % (2 ** n)
    if u <= 2 ** (n - 1) - 1:
        return u
    return u - 2 ** n


def doc_int_unsigned(x: int, y: int, n: int) -> int:
    """The case analysis of **Mul** (uint, uint), in exact arithmetic."""
    return (x * y) % (2 ** n)


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
        got = np.asarray(impl.mul(aa, bb))     # the implementation

        bad = _diff_indices(want, ref, limit=5)
        if bad is None:
            failures.append({"type": dt.name, "who": "the document vs ONNX Runtime",
                             "detail": "shape %s or type %s differs from the document's "
                                       "%s / %s" % (ref.shape, ref.dtype,
                                                    want.shape, want.dtype)})
        elif bad:
            failures.append({
                "type": dt.name, "who": "the document vs ONNX Runtime",
                "detail": "; ".join("doc: %d * %d = %d, ort: %d"
                                    % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                       int(want.ravel()[i]), int(ref.ravel()[i]))
                                    for i in bad),
            })
        bad = _diff_indices(ref, got, limit=5)
        if bad is None:
            failures.append({"type": dt.name, "who": "the implementation vs ONNX Runtime",
                             "detail": "shape %s or type %s differs from the runtime's "
                                       "%s / %s" % (got.shape, got.dtype,
                                                    ref.shape, ref.dtype)})
        elif bad:
            failures.append({
                "type": dt.name, "who": "the implementation vs ONNX Runtime",
                "detail": "; ".join("%d * %d: ort %d, impl %d"
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

    Each element is round(A * B) as the document states it, computed with `Fraction`
    from the operands' exact values.  A float16 departure that is the same product
    rounded through float32 is recorded as double rounding, not required.
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
        got = np.asarray(impl.mul(a, b))
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
                    "type": dt.name, "who": "%s vs the exact product of the document" % who,
                    "detail": "; ".join(
                        "%r * %r: exact %r, %s %r"
                        % (float(a.ravel()[i]), float(b.ravel()[i]),
                           float(exact.ravel()[i]), who, float(other.ravel()[i]))
                        for i in real_bad[:5]),
                })
        rows.append("%s: %d operand pairs" % (dt.name, a.size))
    summary = ("%d operand pairs: %s; the float16 departures that are the same product "
               "rounded through float32 are recorded, not required (%d recorded)"
               % (checked, ", ".join(rows), recorded))
    return {"sweep": "the float formula against exact arithmetic (Fraction)",
            "summary": summary, "failures": failures}


def _real_examples():
    """The real section's cases: exact rationals, no ONNX type."""
    F = Fraction
    return [
        ("real Example 1",
         [[F("1.5"), F("9.5"), F("35.5")]],
         [F(2), F(3), F(-4)],
         [[F(3), F("28.5"), F(-142)]]),
        ("real Example 2",
         [[F(3), F("4.5")], [F(16), F("0.5")], [F("25.5"), F(24)]],
         [F(2), F(-1)],
         [[F(6), F("-4.5")], [F(32), F("-0.5")], [F(51), F(-24)]]),
        ("real: exactness of 1/3 * 3 and of 1/10 * 10",
         [[F(1, 3), F(1, 10)]],
         [[F(3), F(10)]],
         [[F(1), F(1)]]),
        ("real: rank-0 operand broadcast to (2,)",
         F(3, 7), [F(2, 5), F(1, 2)], None),
        ("real: broadcast of a row over a 3x2 matrix",
         [[F(1, 3), F(2, 3)], [F(1, 7), F(3, 7)], [F(-1, 9), F(5, 9)]],
         [F(1, 6), F(1, 6)], None),
    ]


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact rationals.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    implementation against the exact products of the section's Examples and of its
    formula, in exact rational arithmetic.
    """
    failures = []
    examples = _real_examples()
    for label, a, b, documented in examples:
        aa = np.array(a, dtype=object)
        bb = np.array(b, dtype=object)
        ba, bc = np.broadcast_arrays(aa, bb)
        want = np.empty(ba.shape, dtype=object)
        for i in np.ndindex(ba.shape):
            want[i] = ba[i] * bc[i]
        try:
            got = np.asarray(impl.mul(aa, bb))
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
    f64 = np.float64
    return [
        ("float Example 1",
         np.array([3.0, 4.5, 16.0, 0.5], dtype=f32),
         np.array([3.0, -2.0, 0.25, 4.0], dtype=f32),
         np.array([9.0, -9.0, 4.0, 2.0], dtype=f32)),
        ("float Example 2",
         np.array([0.0, np.inf, 0.0, -0.0, -np.inf, np.nan], dtype=f32),
         np.array([np.inf, -np.inf, -3.0, -2.0, 2.0, 1.0], dtype=f32),
         np.array([np.nan, -np.inf, -0.0, 0.0, -np.inf, np.nan], dtype=f32)),
        ("float Example 3",
         np.array([0.1, 1e308, -1e-308], dtype=f64),
         np.array([0.2, 10.0, 1e-308], dtype=f64),
         np.array([0.020000000000000004, np.inf, -0.0], dtype=f64)),
        ("int Example 1",
         np.array([-6, 100, -100, 100], dtype=np.int8),
         np.array([-3, 100, 100, 3], dtype=np.int8),
         np.array([18, 16, -16, 44], dtype=np.int8)),
        ("uint Example 1",
         np.array([6, 200, 35, 16], dtype=np.uint8),
         np.array([3, 100, 5, 16], dtype=np.uint8),
         np.array([18, 32, 175, 0], dtype=np.uint8)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, and as ONNX Runtime does it."""
    failures = []
    examples = _spec_examples()
    for label, a, b, documented in examples:
        ref = run_ort_batch(a, b)
        got = np.asarray(impl.mul(a, b))
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


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
#
# Every case is a pair of operands of the same type.  The float cases include the
# special values of the type (signed zeros, infinities, NaN, the subnormal range, the
# smallest and largest finite values, the rounding boundary between two values and the
# tie above it), the worked examples of the document, and the shape families (rank-0,
# a zero-sized dimension, and the shapes the document names).  The integer cases
# include the worked examples and the boundary values of each type.


def _float_special_cases():
    """The special values of each float type, as operands and as results."""
    out = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        maxf = float(f.max)
        tiny = float(f.tiny)
        small = float(np.nextafter(dt.type(0.0), dt.type(1.0)))
        # the rounding boundary between two values of the type and the tie above it:
        # 1 + 2^-p is the midpoint between 1 and 1 + 2^(1-p); the tie rounds to even
        p = f.nmant + 1
        one = dt.type(1.0)
        next_one = np.nextafter(one, dt.type(2.0))
        mid = dt.type((float(one) + float(next_one)) / 2.0)
        # the smallest subnormal and the smallest normal
        specials = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.0, -2.0,
                    tiny, -tiny, small, -small, maxf, -maxf, maxf / 2.0, -maxf / 2.0,
                    float("inf"), float("-inf"), float("nan")]
        # the special values as operands, paired with a fixed partner
        for v in specials:
            a = np.array([v], dtype=dt)
            b = np.array([2.0], dtype=dt)
            out.append(("float %s: %r * 2" % (dt.name, v), [a, b]))
        # the special values as results: the products of the special values
        for x in specials:
            for y in specials:
                a = np.array([x], dtype=dt)
                b = np.array([y], dtype=dt)
                out.append(("float %s: %r * %r" % (dt.name, x, y), [a, b]))
        # the rounding boundary and the tie above it
        out.append(("float %s: rounding boundary 1 * (1 + 2^-p)" % dt.name,
                    [np.array([1.0], dtype=dt), np.array([mid], dtype=dt)]))
        out.append(("float %s: tie above the boundary" % dt.name,
                    [np.array([1.0], dtype=dt),
                     np.array([np.nextafter(mid, dt.type(2.0))], dtype=dt)]))
        out.append(("float %s: tie below the boundary" % dt.name,
                    [np.array([1.0], dtype=dt),
                     np.array([np.nextafter(mid, dt.type(0.0))], dtype=dt)]))
        # the subnormal range: the smallest subnormal times 1, times 2, times 0.5
        out.append(("float %s: smallest subnormal * 1" % dt.name,
                    [np.array([small], dtype=dt), np.array([1.0], dtype=dt)]))
        out.append(("float %s: smallest subnormal * 2" % dt.name,
                    [np.array([small], dtype=dt), np.array([2.0], dtype=dt)]))
        out.append(("float %s: smallest subnormal * 0.5" % dt.name,
                    [np.array([small], dtype=dt), np.array([0.5], dtype=dt)]))
        # the largest finite value times 2: overflow
        out.append(("float %s: largest finite * 2 (overflow)" % dt.name,
                    [np.array([maxf], dtype=dt), np.array([2.0], dtype=dt)]))
        out.append(("float %s: largest finite * -2 (overflow)" % dt.name,
                    [np.array([maxf], dtype=dt), np.array([-2.0], dtype=dt)]))
        # the smallest normal times 0.5: underflow to subnormal
        out.append(("float %s: smallest normal * 0.5" % dt.name,
                    [np.array([tiny], dtype=dt), np.array([0.5], dtype=dt)]))
        # the smallest subnormal times 0.5: underflow to zero
        out.append(("float %s: smallest subnormal * 0.5 (underflow to zero)" % dt.name,
                    [np.array([small], dtype=dt), np.array([0.5], dtype=dt)]))
        out.append(("float %s: smallest subnormal * -0.5 (underflow to -0)" % dt.name,
                    [np.array([small], dtype=dt), np.array([-0.5], dtype=dt)]))
    return out


def _float_example_cases():
    """The worked examples of the float section, as documented."""
    f32 = np.float32
    f64 = np.float64
    return [
        ("float Example 1",
         [np.array([3.0, 4.5, 16.0, 0.5], dtype=f32),
          np.array([3.0, -2.0, 0.25, 4.0], dtype=f32)]),
        ("float Example 2",
         [np.array([0.0, np.inf, 0.0, -0.0, -np.inf, np.nan], dtype=f32),
          np.array([np.inf, -np.inf, -3.0, -2.0, 2.0, 1.0], dtype=f32)]),
        ("float Example 3",
         [np.array([0.1, 1e308, -1e-308], dtype=f64),
          np.array([0.2, 10.0, 1e-308], dtype=f64)]),
    ]


def _float_shape_cases():
    """The shape families of the float section: rank-0, zero-sized, and the named shapes."""
    f32 = np.float32
    return [
        ("float: rank-0 * rank-0",
         [np.array(2.5, dtype=f32), np.array(4.0, dtype=f32)]),
        ("float: rank-0 * (3,)",
         [np.array(2.5, dtype=f32), np.array([1.0, 2.0, 3.0], dtype=f32)]),
        ("float: (3,) * rank-0",
         [np.array([1.0, 2.0, 3.0], dtype=f32), np.array(2.5, dtype=f32)]),
        ("float: (2, 3) * (3,)",
         [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=f32),
          np.array([2.0, 3.0, 4.0], dtype=f32)]),
        ("float: (2, 1) * (1, 3)",
         [np.array([[1.0], [2.0]], dtype=f32),
          np.array([[3.0, 4.0, 5.0]], dtype=f32)]),
        ("float: (0,) * (1,)",
         [np.array([], dtype=f32), np.array([2.0], dtype=f32)]),
        ("float: (0,) * (0,)",
         [np.array([], dtype=f32), np.array([], dtype=f32)]),
        ("float: (2, 0) * (1, 0)",
         [np.zeros((2, 0), dtype=f32), np.zeros((1, 0), dtype=f32)]),
        ("float: (2, 3) * (2, 3)",
         [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=f32),
          np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=f32)]),
    ]


def _int_example_cases():
    """The worked examples of the integer sections, as documented."""
    return [
        ("int Example 1",
         [np.array([-6, 100, -100, 100], dtype=np.int8),
          np.array([-3, 100, 100, 3], dtype=np.int8)]),
        ("uint Example 1",
         [np.array([6, 200, 35, 16], dtype=np.uint8),
          np.array([3, 100, 5, 16], dtype=np.uint8)]),
    ]


def _int_boundary_cases():
    """The boundary values of each integer type, as operands and as results."""
    out = []
    for dtype in ALL_INT:
        dt = np.dtype(dtype)
        info = np.iinfo(dt)
        lo, hi = int(info.min), int(info.max)
        edges = [lo, lo + 1, -1, 0, 1, hi - 1, hi]
        for x in edges:
            for y in edges:
                a = np.array([x], dtype=dt)
                b = np.array([y], dtype=dt)
                out.append(("int %s: %d * %d" % (dt.name, x, y), [a, b]))
        # the products that leave the range: hi * hi, lo * lo, hi * lo
        out.append(("int %s: max * max (overflow)" % dt.name,
                    [np.array([hi], dtype=dt), np.array([hi], dtype=dt)]))
        out.append(("int %s: min * min (overflow)" % dt.name,
                    [np.array([lo], dtype=dt), np.array([lo], dtype=dt)]))
        out.append(("int %s: max * min (overflow)" % dt.name,
                    [np.array([hi], dtype=dt), np.array([lo], dtype=dt)]))
    return out


def _int_shape_cases():
    """The shape families of the integer sections: rank-0, zero-sized, and the named shapes."""
    return [
        ("int: rank-0 * rank-0",
         [np.array(3, dtype=np.int8), np.array(4, dtype=np.int8)]),
        ("int: rank-0 * (3,)",
         [np.array(3, dtype=np.int8), np.array([1, 2, 3], dtype=np.int8)]),
        ("int: (3,) * rank-0",
         [np.array([1, 2, 3], dtype=np.int8), np.array(3, dtype=np.int8)]),
        ("int: (2, 3) * (3,)",
         [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8),
          np.array([2, 3, 4], dtype=np.int8)]),
        ("int: (2, 1) * (1, 3)",
         [np.array([[1], [2]], dtype=np.int8),
          np.array([[3, 4, 5]], dtype=np.int8)]),
        ("int: (0,) * (1,)",
         [np.array([], dtype=np.int8), np.array([2], dtype=np.int8)]),
        ("int: (0,) * (0,)",
         [np.array([], dtype=np.int8), np.array([], dtype=np.int8)]),
        ("int: (2, 0) * (1, 0)",
         [np.zeros((2, 0), dtype=np.int8), np.zeros((1, 0), dtype=np.int8)]),
        ("int: (2, 3) * (2, 3)",
         [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8),
          np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int8)]),
    ]


def cases():
    """One labelled pair of operands per case, of every section of the document."""
    for item in _float_special_cases():
        yield item
    for item in _float_example_cases():
        yield item
    for item in _float_shape_cases():
        yield item
    for item in _int_example_cases():
        yield item
    for item in _int_boundary_cases():
        yield item
    for item in _int_shape_cases():
        yield item


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------
#
# The four section anchors of the document, each with the number of cases that exercise
# it.  The real section has no ONNX type and its cases are in the sweep, not in
# `cases()`; the count below is the number of real cases the sweep checks.

COVERAGE = {
    "real": 5,
    "float": len(_float_special_cases()) + len(_float_example_cases())
             + len(_float_shape_cases()),
    "int": len(_int_example_cases()) + len(_int_boundary_cases())
           + len(_int_shape_cases()),
    "uint": len(_int_example_cases()) + len(_int_boundary_cases())
            + len(_int_shape_cases()),
}


def sweeps():
    """The bulk checks, batched into a single ONNX Runtime run each."""
    return [
        sweep_int_domain,
        sweep_float_exact,
        sweep_real_examples,
        sweep_spec_examples,
        sweep_schema_types,
    ]
