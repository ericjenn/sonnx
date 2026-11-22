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
UINT_OF_WIDTH = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


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


def _diff_indices(want: np.ndarray, got: np.ndarray, limit=None):
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
    idx = [int(i) for i in np.nonzero(mask.ravel())[0]]
    return idx if limit is None else idx[:limit]


# ---------------------------------------------------------------------------
# exact arithmetic: the reference the document itself states
# ---------------------------------------------------------------------------
#
# The document's float formula is round(A * B): the exact product in the reals, rounded
# to the nearest value of the type with roundTiesToEven and an unbounded exponent range.
# It is computed here from the operands' exact values (Fraction of the float, never of
# the decimal literal that built the array), independently of the platform and of the
# runtime.  The integer sections are the Euclidean remainder modulo 2**n.


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
    """C[i] of **Mul** (float, float): NaN, an infinity, a signed zero, or round(x*y)."""
    dt = np.dtype(dtype)
    fx, fy = float(x), float(y)
    if np.isnan(fx) or np.isnan(fy):
        return dt.type(float("nan"))
    sx, sy = bool(np.signbit(fx)), bool(np.signbit(fy))
    if np.isinf(fx) or np.isinf(fy):
        if (np.isinf(fx) and fy == 0.0) or (np.isinf(fy) and fx == 0.0):
            return dt.type(float("nan"))          # the invalid operation of IEEE 7.2
        return dt.type(float("-inf") if (sx != sy) else float("inf"))
    if fx == 0.0 or fy == 0.0:
        return dt.type(-0.0 if (sx != sy) else 0.0)
    return _frac_round_to_type(Fraction(fx) * Fraction(fy), dt)


def doc_int_signed(x: int, y: int, n: int) -> int:
    """The case analysis of **Mul** (int, int), in exact arithmetic."""
    u = (x * y) % (2 ** n)                # Euclidean remainder, taken in [0, 2**n-1]
    return u if u <= 2 ** (n - 1) - 1 else u - 2 ** n


def doc_int_unsigned(x: int, y: int, n: int) -> int:
    """The formula of **Mul** (uint, uint), in exact arithmetic."""
    return (x * y) % (2 ** n)


def _doc_int_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dt = a.dtype
    n = dt.itemsize * 8
    fn = doc_int_signed if dt.kind == "i" else doc_int_unsigned
    flat = [fn(int(x), int(y), n) for x, y in zip(a.tolist(), b.tolist())]
    return np.array(flat, dtype=dt).reshape(a.shape)


def _rand_values(rng, dt, n):
    """n random values of `dt`, drawn uniformly over the whole bit patterns of the type."""
    dt = np.dtype(dt)
    bits = dt.itemsize * 8
    if bits == 64:
        hi = rng.integers(0, 2 ** 32, size=n, dtype=np.uint64)
        lo = rng.integers(0, 2 ** 32, size=n, dtype=np.uint64)
        raw = (hi << np.uint64(32)) | lo
    else:
        raw = rng.integers(0, 1 << bits, size=n, dtype=np.uint64)
    return raw.astype(UINT_OF_WIDTH[dt.itemsize]).view(dt)


# ---------------------------------------------------------------------------
# cases()
# ---------------------------------------------------------------------------

f16 = np.float16
f32 = np.float32
f64 = np.float64


def _case_list():
    """The individual cases, tagged with the section of the document they exercise."""
    out = []

    # --- the float section -------------------------------------------------
    out.append(("float", "float Example 1 (float32): every product representable",
                [np.array([3.0, 4.5, 16.0, 0.5], dtype=f32),
                 np.array([3.0, -2.0, 0.25, 4.0], dtype=f32)]))
    out.append(("float", "float Example 2 (float32): zero by infinity, signs of zeros",
                [np.array([0.0, np.inf, 0.0, -0.0, -np.inf, np.nan], dtype=f32),
                 np.array([np.inf, -np.inf, -3.0, -2.0, 2.0, 1.0], dtype=f32)]))
    out.append(("float", "float Example 3 (double): rounding, overflow, underflow to -0",
                [np.array([0.1, 1e308, -1e-308], dtype=f64),
                 np.array([0.2, 10.0, 1e-308], dtype=f64)]))
    out.append(("float", "float16: exact products of representable values",
                [np.array([1.5, 2.0, -0.5, 0.25, 0.0, -1.0], dtype=f16),
                 np.array([2.0, -3.0, 4.0, 8.0, -0.0, -1.0], dtype=f16)]))
    out.append(("float", "float16: special values, every branch of the formula",
                [np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0], dtype=f16),
                 np.array([-0.0, -1.0, 0.0, 2.0, 3.0, np.inf, -np.inf], dtype=f16)]))
    out.append(("float", "float32: special values, extremes and the subnormal range",
                [np.array([0.0, -0.0, np.inf, -np.inf, np.nan,
                           np.finfo(f32).max, np.finfo(f32).tiny,
                           float(np.nextafter(f32(0.0), f32(1.0)))], dtype=f32),
                 np.array([-0.0, 0.0, -2.0, 2.0, 1.0,
                           -2.0, np.finfo(f32).tiny, 2.0], dtype=f32)]))
    out.append(("float", "float64: overflow of the largest finite value, both signs",
                [np.array([np.finfo(f64).max, -np.finfo(f64).max, 2.0, -2.0], dtype=f64),
                 np.array([2.0, 2.0, np.finfo(f64).max, np.finfo(f64).max], dtype=f64)]))
    out.append(("float", "float64: rounding of a product that is not representable",
                [np.array([0.1, 1.0 / 3.0, 1e-300, -1e-300], dtype=f64),
                 np.array([0.2, 3.0, 1e-300, 1e-300], dtype=f64)]))
    out.append(("float", "broadcast: (3,2) by (2,) (float32)",
                [np.array([[3.0, 4.5], [16.0, 0.5], [25.5, 24.0]], dtype=f32),
                 np.array([2.0, -1.0], dtype=f32)]))
    out.append(("float", "rank-0 operand broadcast to (3,) (float32)",
                [np.array(2.5, dtype=f32), np.array([1.0, -2.0, 4.0], dtype=f32)]))
    out.append(("float", "two rank-0 operands (float64)",
                [np.array(-1.5, dtype=f64), np.array(4.0, dtype=f64)]))
    out.append(("float", "broadcast: (2,1,3) by (1,4,1) (float32)",
                [np.arange(6, dtype=f32).reshape(2, 1, 3) + 0.5,
                 np.arange(4, dtype=f32).reshape(1, 4, 1) - 1.5]))
    out.append(("float", "zero-sized dimension (2,0) by (1,0) (float32)",
                [np.zeros((2, 0), dtype=f32), np.ones((1, 0), dtype=f32)]))
    out.append(("float", "zero-sized dimension (0,3) by (3,) (float32)",
                [np.zeros((0, 3), dtype=f32), np.array([1.0, 2.0, 3.0], dtype=f32)]))

    # --- the signed integer section ---------------------------------------
    out.append(("int", "int Example 1 (int8)",
                [np.array([-6, 100, -100, 100], dtype=np.int8),
                 np.array([-3, 100, 100, 3], dtype=np.int8)]))
    out.append(("int", "int8: every boundary pair of the 8-bit domain",
                [np.array([-128, -128, 127, 127, -1, 1, 0, -128], dtype=np.int8),
                 np.array([-128, 127, 127, -128, -1, -1, 127, 1], dtype=np.int8)]))
    out.append(("int", "int16: boundary products wrapping modulo 2**16",
                [np.array([-32768, 32767, -1, 128, -128, 256], dtype=np.int16),
                 np.array([-32768, 32767, -1, 256, 256, 256], dtype=np.int16)]))
    out.append(("int", "int32: boundary products wrapping modulo 2**32",
                [np.array([-2 ** 31, 2 ** 31 - 1, -1, 65536, -65536], dtype=np.int32),
                 np.array([-2 ** 31, 2 ** 31 - 1, -1, 65536, 65536], dtype=np.int32)]))
    out.append(("int", "int64: boundary products wrapping modulo 2**64",
                [np.array([-2 ** 63, 2 ** 63 - 1, -1, 2 ** 32, -2 ** 32],
                          dtype=np.int64),
                 np.array([-2 ** 63, 2 ** 63 - 1, -1, 2 ** 32, 2 ** 32],
                          dtype=np.int64)]))
    out.append(("int", "int8: rank-0 operand broadcast to (3,)",
                [np.array(-7, dtype=np.int8),
                 np.array([3, -4, 5], dtype=np.int8)]))
    out.append(("int", "int32: (2,2) by (2,) broadcast, and a zero-sized dimension",
                [np.array([[1, -2], [3, -4]], dtype=np.int32),
                 np.array([5, 6], dtype=np.int32)]))
    out.append(("int", "int16: zero-sized dimension (2,0) by (1,0)",
                [np.zeros((2, 0), dtype=np.int16), np.ones((1, 0), dtype=np.int16)]))

    # --- the unsigned integer section -------------------------------------
    out.append(("uint", "uint Example 1 (uint8)",
                [np.array([6, 200, 35, 16], dtype=np.uint8),
                 np.array([3, 100, 5, 16], dtype=np.uint8)]))
    out.append(("uint", "uint8: every boundary pair of the 8-bit domain",
                [np.array([0, 0, 255, 255, 16, 1, 128], dtype=np.uint8),
                 np.array([255, 0, 255, 2, 16, 255, 2], dtype=np.uint8)]))
    out.append(("uint", "uint16: boundary products wrapping modulo 2**16",
                [np.array([65535, 65535, 256, 1], dtype=np.uint16),
                 np.array([65535, 2, 256, 65535], dtype=np.uint16)]))
    out.append(("uint", "uint32: boundary products wrapping modulo 2**32",
                [np.array([2 ** 32 - 1, 2 ** 32 - 1, 65536, 1], dtype=np.uint32),
                 np.array([2 ** 32 - 1, 2, 65536, 2 ** 32 - 1], dtype=np.uint32)]))
    out.append(("uint", "uint64: boundary products wrapping modulo 2**64",
                [np.array([2 ** 64 - 1, 2 ** 64 - 1, 2 ** 32, 1], dtype=np.uint64),
                 np.array([2 ** 64 - 1, 2, 2 ** 32, 2 ** 64 - 1], dtype=np.uint64)]))
    out.append(("uint", "uint8: rank-0 operand broadcast to (4,)",
                [np.array(200, dtype=np.uint8),
                 np.array([2, 3, 4, 5], dtype=np.uint8)]))
    out.append(("uint", "uint32: zero-sized dimension (0,4) by (4,)",
                [np.zeros((0, 4), dtype=np.uint32),
                 np.array([2, 3, 4, 5], dtype=np.uint32)]))
    return out


def _real_examples():
    """The real section's examples: exact rationals, no ONNX type."""
    F = Fraction
    return [
        ("real Example 1",
         [[F(3, 2), F(19, 2), F(71, 2)]],
         [F(2), F(3), F(-4)],
         [[F(3), F(57, 2), F(-142)]]),
        ("real Example 2",
         [[F(3), F(9, 2)], [F(16), F(1, 2)], [F(51, 2), F(24)]],
         [F(2), F(-1)],
         [[F(6), F(-9, 2)], [F(32), F(-1, 2)], [F(51), F(-24)]]),
        ("real: rank-0 operand broadcast to (3,)",
         F(3, 7),
         [F(2, 5), F(1, 2), F(-4, 3)],
         [F(6, 35), F(3, 14), F(-4, 7)]),
    ]


_CASES = _case_list()
_REAL = _real_examples()


def cases():
    """One labelled pair of operands per case, in the section's own terms."""
    for _section, label, arrays in _CASES:
        yield label, arrays


COVERAGE = {
    "real": len(_REAL),
    "float": sum(1 for s, _, _ in _CASES if s == "float"),
    "int": sum(1 for s, _, _ in _CASES if s == "int"),
    "uint": sum(1 for s, _, _ in _CASES if s == "uint"),
}


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_int_domain(impl):
    """The document's integer formulas against ONNX Runtime and the implementation.

    The two 8-bit types are covered exhaustively (all 65536 pairs each); the wider
    types use their boundary values and 200 random pairs.
    """
    rng = np.random.default_rng(20261122)
    rows, failures = [], []
    for dtype in ALL_INT:
        dt = np.dtype(dtype)
        info = np.iinfo(dt)
        exhaustive = dt.itemsize == 1
        if exhaustive:
            xs = np.arange(int(info.min), int(info.max) + 1, dtype=np.int64)
            aa = np.repeat(xs, xs.size).astype(dt)
            bb = np.tile(xs, xs.size).astype(dt)
        else:
            edges = sorted({int(info.min), int(info.min) + 1,
                            int(info.max) - 1, int(info.max), 0, 1, 2})
            pairs = [(x, y) for x in edges for y in edges]
            pairs += list(zip(_rand_values(rng, dt, 200).tolist(),
                              _rand_values(rng, dt, 200).tolist()))
            aa = np.array([p[0] for p in pairs], dtype=dt)
            bb = np.array([p[1] for p in pairs], dtype=dt)

        want = _doc_int_array(aa, bb)          # the document, literally
        try:
            ref = run_ort_batch(aa, bb)        # ONNX Runtime
        except Exception as exc:
            rows.append("%s: ONNX Runtime refused (%s)" % (dt.name, type(exc).__name__))
            continue

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

        try:
            got = np.asarray(impl.mul(aa, bb))
        except Exception as exc:
            failures.append({"type": dt.name, "who": "the implementation",
                             "detail": "raised %s: %s" % (type(exc).__name__, exc)})
            rows.append("%s: %d pairs" % (dt.name, aa.size))
            continue

        bad = _diff_indices(want, got, limit=5)
        if bad is None:
            failures.append({"type": dt.name, "who": "the implementation vs the document",
                             "detail": "shape %s or type %s differs from the document's "
                                       "%s / %s" % (got.shape, got.dtype,
                                                    want.shape, want.dtype)})
        elif bad:
            failures.append({
                "type": dt.name, "who": "the implementation vs the document",
                "detail": "; ".join("%d * %d: doc %d, impl %d"
                                    % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                       int(want.ravel()[i]), int(got.ravel()[i]))
                                    for i in bad),
            })
        rows.append("%s %s" % (dt.name,
                               "exhaustive (65536 pairs)" if exhaustive
                               else "%d pairs" % aa.size))
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
    ra[:60] = (rng.random(60) * 2.0 - 1.0) * tiny        # around the normal boundary
    rb[:60] = (rng.random(60) * 2.0 - 1.0) * tiny
    pairs += list(zip(ra.tolist(), rb.tolist()))

    a = np.array([p[0] for p in pairs], dtype=dt)
    b = np.array([p[1] for p in pairs], dtype=dt)
    return a, b


def sweep_float_exact(impl):
    """The float section's formula against exact arithmetic, ORT, and the implementation.

    Each element is round(A * B) as the document states it, computed with `Fraction`
    from the operands' exact values (the float, never the decimal literal).  A float16
    departure of the runtime is recorded, not required: it is the runtime's arithmetic
    that is in doubt there, not the document.
    """
    rng = np.random.default_rng(20261122)
    rows, failures = [], []
    recorded = 0
    checked = 0
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        a, b = _float_probe(dt, rng)
        exact = np.array([_exact_float_result(x, y, dt)
                          for x, y in zip(a.tolist(), b.tolist())], dtype=dt)
        checked += int(a.size)

        try:
            ref = run_ort_batch(a, b)
        except Exception as exc:
            rows.append("%s: ONNX Runtime refused (%s)" % (dt.name, type(exc).__name__))
            ref = None
        if ref is not None:
            bad = _diff_indices(exact, ref, limit=None)
            if bad is None:
                failures.append({"type": dt.name, "who": "ONNX Runtime",
                                 "detail": "shape %s or type %s differs from the exact "
                                           "result's %s / %s"
                                           % (ref.shape, ref.dtype,
                                              exact.shape, exact.dtype)})
            elif bad and dt == np.dtype(np.float16):
                recorded += len(bad)          # recorded, not required
            elif bad:
                failures.append({
                    "type": dt.name,
                    "who": "ONNX Runtime vs the exact product of the document",
                    "detail": "; ".join(
                        "%r * %r: exact %r, ort %r"
                        % (float(a.ravel()[i]), float(b.ravel()[i]),
                           float(exact.ravel()[i]), float(ref.ravel()[i]))
                        for i in bad[:5]),
                })

        try:
            got = np.asarray(impl.mul(a, b))
        except Exception as exc:
            failures.append({"type": dt.name, "who": "the implementation",
                             "detail": "raised %s: %s" % (type(exc).__name__, exc)})
            rows.append("%s: %d operand pairs" % (dt.name, a.size))
            continue

        bad = _diff_indices(exact, got, limit=None)
        if bad is None:
            failures.append({"type": dt.name, "who": "the implementation",
                             "detail": "shape %s or type %s differs from the exact "
                                       "result's %s / %s"
                                       % (got.shape, got.dtype,
                                          exact.shape, exact.dtype)})
        elif bad:
            failures.append({
                "type": dt.name,
                "who": "the implementation vs the exact product of the document",
                "detail": "; ".join(
                    "%r * %r: exact %r, impl %r"
                    % (float(a.ravel()[i]), float(b.ravel()[i]),
                       float(exact.ravel()[i]), float(got.ravel()[i]))
                    for i in bad[:5]),
            })
        rows.append("%s: %d operand pairs" % (dt.name, a.size))
    summary = ("%d operand pairs: %s; %d float16 departures of ONNX Runtime are "
               "recorded, not required" % (checked, ", ".join(rows), recorded))
    return {"sweep": "the float formula against exact arithmetic (Fraction)",
            "summary": summary, "failures": failures}


def _spec_examples():
    """The worked examples of the float and the integer sections, as documented."""
    return [
        ("float Example 1 (float32)",
         np.array([3.0, 4.5, 16.0, 0.5], dtype=f32),
         np.array([3.0, -2.0, 0.25, 4.0], dtype=f32),
         np.array([9.0, -9.0, 4.0, 2.0], dtype=f32)),
        ("float Example 2 (float32)",
         np.array([0.0, np.inf, 0.0, -0.0, -np.inf, np.nan], dtype=f32),
         np.array([np.inf, -np.inf, -3.0, -2.0, 2.0, 1.0], dtype=f32),
         np.array([np.nan, -np.inf, -0.0, 0.0, -np.inf, np.nan], dtype=f32)),
        ("float Example 3 (double)",
         np.array([0.1, 1e308, -1e-308], dtype=f64),
         np.array([0.2, 10.0, 1e-308], dtype=f64),
         np.array([0.020000000000000004, np.inf, -0.0], dtype=f64)),
        ("int Example 1 (int8)",
         np.array([-6, 100, -100, 100], dtype=np.int8),
         np.array([-3, 100, 100, 3], dtype=np.int8),
         np.array([18, 16, -16, 44], dtype=np.int8)),
        ("uint Example 1 (uint8)",
         np.array([6, 200, 35, 16], dtype=np.uint8),
         np.array([3, 100, 5, 16], dtype=np.uint8),
         np.array([18, 32, 175, 0], dtype=np.uint8)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, ORT, and the implementation."""
    failures = []
    examples = _spec_examples()
    for label, a, b, documented in examples:
        try:
            ref = run_ort_batch(a, b)
        except Exception as exc:
            failures.append({"who": label,
                             "detail": "ONNX Runtime refused %s: %s"
                                       % (type(exc).__name__, exc)})
            ref = None
        if ref is not None and not _bits_equal(documented, ref):
            failures.append({"who": label,
                             "detail": "documented %s, ORT %s"
                                       % (documented.tolist(), ref.tolist())})
        try:
            got = np.asarray(impl.mul(a, b))
        except Exception as exc:
            failures.append({"who": label,
                             "detail": "the implementation raised %s: %s"
                                       % (type(exc).__name__, exc)})
            continue
        if not _bits_equal(documented, got):
            failures.append({"who": label,
                             "detail": "documented %s, implementation %s"
                                       % (documented.tolist(), got.tolist())})
    return {"sweep": "the document's worked examples",
            "summary": "%d examples, as documented vs ORT vs implementation"
                       % len(examples),
            "failures": failures}


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact rationals.

    ONNX Runtime cannot be asked about a real tensor.  The section's Examples and its
    broadcast rules are checked against `Fraction` arithmetic, and against the values
    the document itself writes.  An implementation that does not accept exact rational
    operands is recorded in the summary rather than failed: a real number has no ONNX
    type and the harness cannot carry one either.
    """
    failures = []
    notes = []
    examples = _real_examples()
    for label, a, b, documented in examples:
        aa = np.array(a, dtype=object)
        bb = np.array(b, dtype=object)
        ba, bc = np.broadcast_arrays(aa, bb)
        want = np.empty(ba.shape, dtype=object)
        for i in np.ndindex(ba.shape):
            want[i] = Fraction(ba[i]) * Fraction(bc[i])
        try:
            got = np.asarray(impl.mul(aa, bb))
        except Exception as exc:
            notes.append("%s: the implementation raised %s"
                         % (label, type(exc).__name__))
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
        doc = np.array(documented, dtype=object)
        if doc.shape != want.shape:
            failures.append({"who": label,
                             "detail": "documented shape %s, exact shape %s"
                                       % (doc.shape, want.shape)})
        else:
            diff = [i for i in np.ndindex(doc.shape) if Fraction(doc[i]) != want[i]]
            if diff:
                failures.append({"who": label,
                                 "detail": "; ".join("documented %s, exact %s"
                                                     % (doc[i], want[i]) for i in diff[:5])})
    summary = "%d cases checked against Fraction arithmetic" % len(examples)
    if notes:
        summary += "; " + "; ".join(notes)
    return {"sweep": "the real section (exact rational arithmetic, no ONNX Runtime)",
            "summary": summary, "failures": failures}


def sweep_schema_types(impl):
    """The document's declared type families against the ONNX schema's admitted set.

    A type the document declares that the schema does not admit makes every case of that
    type open at model resolution; a case set that dropped it would converge on the wrong
    document.  A type the schema admits that the document does not specify is recorded in
    the summary: the profile specifies a subset.
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
        summary += "; the schema also admits %s, which the document does not specify" \
                   % ", ".join(extra)
    return {"sweep": "the document's types against the ONNX schema",
            "summary": summary, "failures": failures}


def sweeps():
    """The bulk checks; each takes the implementation and returns its report."""
    return [
        sweep_int_domain,
        sweep_float_exact,
        sweep_spec_examples,
        sweep_real_examples,
        sweep_schema_types,
    ]
