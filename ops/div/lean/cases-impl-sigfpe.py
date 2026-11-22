"""The cases with which the **Div** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled pair of operands per case, run through a one-node Div
                     model (opset 14) and through the implementation, compared bit for
                     bit, the sign of a zero included (the document specifies it).
  * ``sweeps()``  -- the bulk checks, batched into single ONNX Runtime calls: the
                     integer domains of the document, the float section against exact
                     rational arithmetic, the worked examples, the document's declared
                     types against the ONNX schema, and the real section, which has no
                     ONNX type and is checked with `Fraction` alone.
  * ``COVERAGE``  -- the four section anchors of the document, each with the number of
                     cases that exercise it; the real section's cases are the ones of
                     `sweep_real_examples`, a real tensor having no ONNX type.

A null divisor of the integer sections determines no value in the document (its
"Error conditions" section says so), so no such case sits in `cases()`: it is recorded
by `sweep_int_zero_divisor`.  The pair (-2**(n-1), -1) of int32 and int64 is the
document's nominal wrap, but the C integer division overflows for it and ONNX Runtime
aborts the process with SIGFPE, so that pair is decided by exact arithmetic in
`sweep_int_out_of_range_pair` and is never sent to the runtime.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Div"
OPSET = 14
IR_VERSION = 10

NODE_OUTPUTS = ("out",)

ALL_FLOAT = [np.float16, np.float32, np.float64]
SIGNED_INT = [np.int8, np.int16, np.int32, np.int64]
UNSIGNED_INT = [np.uint8, np.uint16, np.uint32, np.uint64]
ALL_INT = SIGNED_INT + UNSIGNED_INT
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}

SLACK_ULP = 2.0


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; single cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One Div node over two arrays of the same type, in a single runtime call."""
    a = np.asarray(a)
    b = np.asarray(b)
    proto = helper.np_dtype_to_tensor_dtype(a.dtype)
    node = helper.make_node(OP, ["A", "B"], ["C"], name="div0")
    graph = helper.make_graph(
        [node], "div_graph",
        [helper.make_tensor_value_info("A", proto, list(a.shape)),
         helper.make_tensor_value_info("B", proto, list(b.shape))],
        [helper.make_tensor_value_info("C", proto, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    with np.errstate(all="ignore"):
        (c,) = sess.run(["C"], {"A": a, "B": b})
    return c


# ---------------------------------------------------------------------------
# comparison helpers: bit patterns, NaN equal to NaN
# ---------------------------------------------------------------------------


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
    flat = [int(i) for i in np.nonzero(mask.ravel())[0]]
    return flat if limit is None else flat[:limit]


def _order_key(value, dt) -> int:
    """A monotone integer key of a finite float: the difference is a distance in ulps."""
    dt = np.dtype(dt)
    u = np.dtype(UINT_OF_WIDTH[dt.itemsize])
    bits = int(np.array([value], dtype=dt).view(u)[0])
    sign = 1 << (u.itemsize * 8 - 1)
    return -(bits & (sign - 1)) if (bits & sign) else bits


# ---------------------------------------------------------------------------
# exact arithmetic: the reference the document itself states
# ---------------------------------------------------------------------------


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

    The exponent range is unbounded while deciding the rounding, as the document
    states: a magnitude whose rounded value exceeds the largest finite value of the
    type gives an infinity of its sign, and one below the smallest subnormal gives a
    zero of its sign.
    """
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    p = f.nmant + 1
    emin = f.minexp
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
        e = emin
    spacing = _pow2(e - p + 1)
    n = _round_half_even(a / spacing)
    result = n * spacing
    if result > max_finite:
        return dt.type(sign * float("inf"))
    return dt.type(sign * float(result))


def _doc_float_div(x, y, dtype):
    """C[i] of **Div** (float, float), by the case analysis of the document."""
    dt = np.dtype(dtype)
    fx, fy = float(x), float(y)
    if np.isnan(fx) or np.isnan(fy):
        return dt.type(float("nan"))
    sign = -1.0 if (np.signbit(fx) != np.signbit(fy)) else 1.0
    if np.isinf(fx) and np.isinf(fy):
        return dt.type(float("nan"))                  # the invalid operation of IEEE 7.2
    if fx == 0.0 and fy == 0.0:
        return dt.type(float("nan"))                  # the invalid operation of IEEE 7.2
    if np.isinf(fx):
        return dt.type(sign * float("inf"))
    if fy == 0.0:
        return dt.type(sign * float("inf"))
    if fx == 0.0:
        return dt.type(sign * 0.0)
    if np.isinf(fy):
        return dt.type(sign * 0.0)
    return _frac_round_to_type(Fraction(fx) / Fraction(fy), dt)


def _doc_int_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """**Div** (int, int) and **Div** (uint, uint) transcribed literally.

    trunc(a/b) toward zero, then read modulo 2**n as a value of the type; for the
    unsigned types the truncation and the floor coincide and no reduction is needed.
    """
    dt = a.dtype
    n = dt.itemsize * 8
    lo, hi = -(2 ** (n - 1)), 2 ** (n - 1) - 1
    out = []
    for x, y in zip(a.ravel().tolist(), b.ravel().tolist()):
        x, y = int(x), int(y)
        if y == 0:
            raise ValueError("null divisor: outside the documented domain")
        q = abs(x) // abs(y)
        if (x < 0) != (y < 0):
            q = -q
        if dt.kind == "i":
            if q > hi:
                q -= 2 ** n
            if q < lo:
                q += 2 ** n
        out.append(q)
    return np.array(out, dtype=dt).reshape(a.shape)


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------


def _build_cases():
    f16, f32, f64 = np.float16, np.float32, np.float64
    i8, i16, i32, i64 = np.int8, np.int16, np.int32, np.int64
    u8, u16, u32, u64 = np.uint8, np.uint16, np.uint32, np.uint64
    nan, inf = float("nan"), float("inf")
    f32max = float(np.finfo(np.float32).max)
    f64max = float(np.finfo(np.float64).max)
    i16min, i16max = int(np.iinfo(i16).min), int(np.iinfo(i16).max)
    i32min, i32max = int(np.iinfo(i32).min), int(np.iinfo(i32).max)
    i64min, i64max = int(np.iinfo(i64).min), int(np.iinfo(i64).max)
    u8max, u16max = int(np.iinfo(u8).max), int(np.iinfo(u16).max)
    u32max, u64max = int(np.iinfo(u32).max), int(np.iinfo(u64).max)

    def A(values, dtype):
        return np.array(values, dtype=dtype)

    out = []

    # ---------------- the float section ----------------
    out.append(("float", "Example 1 (float32)",
                [A([6.0, -9.0, 1.0], f32), A([3.0, 4.0, 8.0], f32)]))
    out.append(("float", "Example 1 (float64)",
                [A([6.0, -9.0, 1.0], f64), A([3.0, 4.0, 8.0], f64)]))
    out.append(("float", "Example 1 (float16, every quotient exact)",
                [A([6.0, -9.0, 1.0], f16), A([3.0, 4.0, 8.0], f16)]))
    out.append(("float", "Example 2: the special values (float32)",
                [A([1.0, -1.0, inf, inf, 0.0, -0.0, 0.0, nan], f32),
                 A([0.0, 0.0, inf, 4.0, 4.0, 4.0, 0.0, 1.0], f32)]))
    out.append(("float", "Example 2: the special values (float16)",
                [A([1.0, -1.0, inf, inf, 0.0, -0.0, 0.0, nan], f16),
                 A([0.0, 0.0, inf, 4.0, 4.0, 4.0, 0.0, 1.0], f16)]))
    out.append(("float", "Example 3: the rounded quotients (float64)",
                [A([1.0, 10.0, -10.0], f64), A([3.0, 3.0, 3.0], f64)]))
    out.append(("float", "the signed zeros of a null quotient (float32)",
                [A([0.0, -0.0, 0.0, -0.0], f32), A([2.0, 2.0, -2.0, -2.0], f32)]))
    out.append(("float", "the invalid operations inf/inf and 0/0 (float32)",
                [A([inf, -inf, inf, -inf, 0.0, -0.0, 0.0, -0.0], f32),
                 A([inf, inf, -inf, -inf, 0.0, 0.0, -0.0, -0.0], f32)]))
    out.append(("float", "a finite dividend over an infinity gives a signed zero (float32)",
                [A([1.0, -1.0, 1.0, -1.0], f32), A([inf, inf, -inf, -inf], f32)]))
    out.append(("float", "an infinity over a finite non-null value (float32)",
                [A([inf, -inf, inf, -inf], f32), A([2.0, 2.0, -2.0, -2.0], f32)]))
    out.append(("float", "division by zero: 1/(+0), -1/(+0), 1/(-0), -1/(-0) (float32)",
                [A([1.0, -1.0, 1.0, -1.0], f32), A([0.0, 0.0, -0.0, -0.0], f32)]))
    out.append(("float", "a NaN operand propagates (float32)",
                [A([nan, 1.0, nan, inf, -0.0], f32),
                 A([1.0, nan, inf, nan, nan], f32)]))
    out.append(("float", "overflow: the largest finite value over 0.5 (float32)",
                [A([f32max, -f32max, f32max], f32), A([0.5, 0.5, 2.0], f32)]))
    out.append(("float", "the rounded quotients 1/3, 10/3, -10/3 (float32)",
                [A([1.0, 10.0, -10.0], f32), A([3.0, 3.0, 3.0], f32)]))
    out.append(("float", "the rounded quotients 1/7, 22/7, -1/7 (float64)",
                [A([1.0, 22.0, -1.0], f64), A([7.0, 7.0, 7.0], f64)]))
    out.append(("float", "finite double quotients and the sign of their zeros (float64)",
                [A([1.0, -1.0, 1.0, -1.0], f64), A([8.0, 8.0, -8.0, -8.0], f64)]))
    out.append(("float", "the special values of double (float64)",
                [A([0.0, -0.0, inf, -inf, nan, f64max], f64),
                 A([-0.0, 0.0, -inf, inf, 1.0, 0.5], f64)]))
    out.append(("float", "float16 quotients that are exact (float16)",
                [A([6.0, -9.0, 1.0, 0.25, -0.25, 3.0], f16),
                 A([3.0, 4.0, 8.0, 0.5, 0.5, 0.75], f16)]))
    out.append(("float", "broadcast: a rank-1 divisor over a 2x3 dividend (float32)",
                [A([[4.0, 9.0], [16.0, 25.0]], f32), A([2.0, 5.0], f32)]))
    out.append(("float", "broadcast: (2,1) over (1,3) (float64)",
                [A([[1.0], [2.0]], f64), A([[1.0, 2.0, 4.0]], f64)]))
    out.append(("float", "rank-0 dividend over a rank-1 divisor (float32)",
                [np.array(6.0, dtype=f32), A([2.0, 3.0, 4.0], f32)]))
    out.append(("float", "a rank-1 dividend over a rank-0 divisor (float64)",
                [A([1.0, 2.0, 3.0], f64), np.array(4.0, dtype=f64)]))
    out.append(("float", "zero-sized dimensions: (0,3) over (1,3) (float32)",
                [np.zeros((0, 3), dtype=f32), A([[2.0, 4.0, 8.0]], f32)]))
    out.append(("float", "zero-sized dimensions: (2,0) over a rank-0 divisor (float64)",
                [np.zeros((2, 0), dtype=f64), np.array(2.0, dtype=f64)]))

    # ---------------- the signed integer section ----------------
    out.append(("int", "Example 1: truncation and the quotient outside the range (int8)",
                [A([7, -7, 7, -7, -128], i8), A([2, 2, -2, -2, -1], i8)]))
    out.append(("int", "truncation toward zero: 7/2, -7/2, 7/-2, -7/-2 (int16)",
                [A([7, -7, 7, -7], i16), A([2, 2, -2, -2], i16)]))
    out.append(("int", "a magnitude below the divisor gives the neutral 0 (int8)",
                [A([-1, -2, -3, 1, 2, 3], i8), A([2, 3, 4, -2, -3, -4], i8)]))
    out.append(("int", "the boundaries of int8 (int8)",
                [A([-128, -128, -127, 127, 127, 0, -1, 1], i8),
                 A([1, 3, 2, 1, -1, 5, 1, 127], i8)]))
    out.append(("int", "the boundaries of int16, the wrap of -32768/-1 included (int16)",
                [A([i16min, i16min, i16max, i16max, -1, 0], i16),
                 A([-1, 2, 2, -1, 1, 7], i16)]))
    out.append(("int", "the boundaries of int32 (int32)",
                [A([i32min, i32min, i32max, i32max, -i32max], i32),
                 A([2, 1, -2, 3, i32max], i32)]))
    out.append(("int", "the boundaries of int64 (int64)",
                [A([i64min, i64max, i64max, -1], i64),
                 A([2, 2, -1, i64max], i64)]))
    out.append(("int", "broadcast: (2,1) over (1,3) (int32)",
                [A([[7], [-7]], i32), A([[2, -2, 3]], i32)]))
    out.append(("int", "a rank-0 dividend over a rank-1 divisor (int64)",
                [np.array(100, dtype=i64), A([3, -7, 100], i64)]))
    out.append(("int", "a rank-1 dividend over a rank-0 divisor (int16)",
                [A([7, -9, 100], i16), np.array(4, dtype=i16)]))
    out.append(("int", "zero-sized dimensions: (0,4) over (1,4) (int8)",
                [np.zeros((0, 4), dtype=i8), A([[2, 3, 4, 5]], i8)]))
    out.append(("int", "zero-sized dimensions: (3,0) over a rank-0 divisor (int32)",
                [np.zeros((3, 0), dtype=i32), np.array(3, dtype=i32)]))

    # ---------------- the unsigned integer section ----------------
    out.append(("uint", "Example 1 (uint8)",
                [A([7, 200, 255, 6], u8), A([2, 3, 255, 6], u8)]))
    out.append(("uint", "the boundaries of uint8 (uint8)",
                [A([u8max, u8max, u8max - 1, 1, 0, 5], u8),
                 A([1, u8max, 2, u8max, 7, 3], u8)]))
    out.append(("uint", "the boundaries of uint16 (uint16)",
                [A([u16max, u16max, u16max - 1, 1, 0], u16),
                 A([1, u16max, 2, u16max, 1], u16)]))
    out.append(("uint", "the boundaries of uint32 (uint32)",
                [A([u32max, u32max, u32max - 1, 0], u32),
                 A([1, u32max, 2, u32max], u32)]))
    out.append(("uint", "the boundaries of uint64 (uint64)",
                [A([u64max, u64max, 0, 1], u64),
                 A([1, u64max, 1, u64max], u64)]))
    out.append(("uint", "broadcast: (2,1) over (1,3) (uint8)",
                [A([[7], [200]], u8), A([[2, 3, 255]], u8)]))
    out.append(("uint", "a rank-0 dividend over a rank-1 divisor (uint32)",
                [np.array(100, dtype=u32), A([3, 7, 100], u32)]))
    out.append(("uint", "zero-sized dimensions: (0,3) over (1,3) (uint8)",
                [np.zeros((0, 3), dtype=u8), A([[2, 3, 4]], u8)]))
    out.append(("uint", "zero-sized dimensions: (2,0) over a rank-0 divisor (uint64)",
                [np.zeros((2, 0), dtype=u64), np.array(7, dtype=u64)]))

    return out


_CASES = _build_cases()


def cases():
    """One labelled pair of operands per case, in the document's own terms."""
    for _section, label, arrays in _CASES:
        yield (label, arrays)


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def _rand_values(rng, dt, n):
    """n random values of `dt`, drawn uniformly over the bit patterns of the type."""
    hi = rng.integers(0, 2 ** 32, size=n, dtype=np.uint64)
    lo = rng.integers(0, 2 ** 32, size=n, dtype=np.uint64)
    raw = (hi << np.uint64(32)) | lo
    bits = dt.itemsize * 8
    if bits < 64:
        raw = raw & np.uint64((1 << bits) - 1)
    return raw.astype(np.dtype("u%d" % dt.itemsize)).view(dt)


def sweep_int_domain(impl):
    """The document's integer case analysis against ONNX Runtime and the implementation.

    The two 8-bit types are covered exhaustively (every non-null divisor against every
    dividend); the wider types use their boundary values and 200 random pairs.  The
    pair (min, -1) of int32 and int64 is removed from the arrays the runtime is asked
    about and decided by `sweep_int_out_of_range_pair`.
    """
    rng = np.random.default_rng(20261004)
    rows, failures = [], []
    for dtype in ALL_INT:
        dt = np.dtype(dtype)
        info = np.iinfo(dt)
        exhaustive = dt.itemsize == 1
        if exhaustive:
            xs = np.arange(int(info.min), int(info.max) + 1, dtype=np.int64)
            aa = np.repeat(xs, xs.size).astype(dt)
            bb = np.tile(xs, xs.size).astype(dt)
            keep = bb != 0
            aa, bb = aa[keep], bb[keep]
        else:
            if dt.kind == "i":
                edges = [int(info.min), int(info.min) + 1, int(info.min) + 2,
                         -3, -2, -1, 0, 1, 2, 3,
                         int(info.max) - 2, int(info.max) - 1, int(info.max)]
            else:
                edges = [0, 1, 2, 3,
                         int(info.max) - 2, int(info.max) - 1, int(info.max)]
            pairs = [(x, y) for x in edges for y in edges if y != 0]
            ra = _rand_values(rng, dt, 200)
            rb = _rand_values(rng, dt, 200)
            pairs += [(int(x), int(y)) for x, y in zip(ra.tolist(), rb.tolist())
                      if int(y) != 0]
            aa = np.array([p[0] for p in pairs], dtype=dt)
            bb = np.array([p[1] for p in pairs], dtype=dt)
        if dt.kind == "i" and dt.itemsize >= 4:
            keep = ~((aa == dt.type(int(info.min))) & (bb == dt.type(-1)))
            aa, bb = aa[keep], bb[keep]

        want = _doc_int_array(aa, bb)
        ref = run_ort_batch(aa, bb)
        with np.errstate(all="ignore"):
            got = np.asarray(impl.div(aa, bb))

        bad = _diff_indices(want, ref, limit=5)
        if bad is None:
            failures.append({"type": dt.name, "who": "the document vs ONNX Runtime",
                             "detail": "shape %s or type %s differs from the document's "
                                       "%s / %s" % (ref.shape, ref.dtype,
                                                    want.shape, want.dtype)})
        elif bad:
            failures.append({
                "type": dt.name, "who": "the document vs ONNX Runtime",
                "detail": "; ".join("doc: %d / %d = %d, ort: %d"
                                    % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                       int(want.ravel()[i]), int(ref.ravel()[i]))
                                    for i in bad),
            })
        bad = _diff_indices(want, got, limit=5)
        if bad is None:
            failures.append({"type": dt.name, "who": "the implementation vs the document",
                             "detail": "shape %s or type %s differs from the document's "
                                       "%s / %s" % (got.shape, got.dtype,
                                                    want.shape, want.dtype)})
        elif bad:
            failures.append({
                "type": dt.name, "who": "the implementation vs the document",
                "detail": "; ".join("%d / %d: document %d, implementation %d"
                                    % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                       int(want.ravel()[i]), int(got.ravel()[i]))
                                    for i in bad),
            })
        rows.append("%s %s" % (dt.name,
                               "exhaustive (%d pairs)" % aa.size if exhaustive
                               else "%d pairs" % aa.size))
    return {"sweep": "the integer domains (document vs ORT vs implementation)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def sweep_int_out_of_range_pair(impl):
    """The one quotient outside the range of the signed types, by exact arithmetic.

    For (-2**(n-1), -1) the document states the result -2**(n-1) (the value congruent
    to the exact quotient modulo 2**n).  int8 and int16 are checked among the domain
    above; int32 and int64 cannot be sent to ONNX Runtime at all, because the C integer
    division overflows for that pair and the process dies with SIGFPE, so the document
    is the only reference here.
    """
    failures = []
    rows = []
    for dt in (np.dtype(np.int32), np.dtype(np.int64)):
        info = np.iinfo(dt)
        a = np.array([int(info.min)], dtype=dt)
        b = np.array([-1], dtype=dt)
        want = int(_doc_int_array(a, b)[0])
        try:
            with np.errstate(all="ignore"):
                got = int(np.asarray(impl.div(a, b)).ravel()[0])
        except Exception as exc:
            failures.append({"who": "%s / -1 (document vs implementation)" % dt.name,
                             "detail": "the document states %d; the implementation "
                                       "raised %s: %s"
                                       % (want, type(exc).__name__, exc)})
            continue
        if got != want:
            failures.append({"who": "%s / -1 (document vs implementation)" % dt.name,
                             "detail": "the document states %d (the exact quotient "
                                       "%d read modulo 2**%d); the implementation gives "
                                       "%d" % (want, 2 ** (dt.itemsize * 8 - 1),
                                               dt.itemsize * 8, got)})
        rows.append("%s: document %d, implementation %d" % (dt.name, want, got))
    return {"sweep": "the quotient outside the range of int32 and int64 (exact "
                     "arithmetic; the runtime cannot evaluate the pair)",
            "summary": "; ".join(rows) + "; ONNX Runtime is not asked: the C division "
                       "overflows for the pair and aborts the process",
            "failures": failures}


def sweep_int_zero_divisor(impl):
    """A null divisor of the integer types: recorded, the document determines no value."""
    notes = []
    for dtype in ALL_INT:
        dt = np.dtype(dtype)
        a = np.array([0, 1, 5], dtype=dt)
        b = np.zeros(3, dtype=dt)
        try:
            with np.errstate(all="ignore"):
                r = np.asarray(impl.div(a, b))
            notes.append("%s: returned %s" % (dt.name, r.tolist()))
        except Exception as exc:
            notes.append("%s: raised %s: %s" % (dt.name, type(exc).__name__, exc))
    return {"sweep": "a null divisor of the integer types (recorded, not required)",
            "summary": "the document determines no value for a null divisor and the "
                       "runtime is not asked (the C division would abort the process): "
                       + "; ".join(notes),
            "failures": []}


def _float_probe(dt, rng):
    """The operands of the exact float sweep: the special values and a random sample."""
    f = np.finfo(np.dtype(dt))
    maxf = float(f.max)
    tiny = float(f.tiny)
    small = float(np.nextafter(dt.type(0.0), dt.type(1.0)))
    specials = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, 3.0,
                maxf, -maxf, tiny, -tiny, small, -small,
                float("inf"), float("-inf"), float("nan")]
    a_list, b_list = [], []
    for x in specials:
        for y in specials:
            a_list.append(x)
            b_list.append(y)
    n = 120
    for v in ((rng.random(n) * 2.0 - 1.0) * maxf).tolist():
        a_list.append(v)
    for v in ((rng.random(n) * 2.0 - 1.0) * maxf).tolist():
        b_list.append(v)
    m = 60
    sub_a = [small * float(k) for k in rng.integers(1, 1024, m)]
    sub_b = [small * float(k) for k in rng.integers(1, 1024, m)]
    a_list += sub_a
    b_list += sub_b
    return (np.array(a_list, dtype=dt), np.array(b_list, dtype=dt))


def _compare_to_exact(exact, other, dt):
    """(mismatches, departures): a NaN/inf/sign-of-zero difference, and ulp distances."""
    mismatch, departures = [], []
    e, o = exact.ravel(), other.ravel()
    for i in range(e.size):
        x, y = e[i], o[i]
        if np.isnan(x) and np.isnan(y):
            continue
        if np.isnan(x) or np.isnan(y):
            mismatch.append(i)
            continue
        if np.isinf(x) or np.isinf(y):
            if float(x) != float(y):
                mismatch.append(i)
            continue
        if x == 0 and y == 0:
            if np.signbit(x) != np.signbit(y):
                mismatch.append(i)
            continue
        if float(x) == float(y):
            continue
        departures.append((i, abs(_order_key(y, dt) - _order_key(x, dt))))
    return mismatch, departures


def sweep_float_exact(impl):
    """The float section's formula against exact arithmetic, ORT, and the implementation.

    Each element is the document's case analysis: NaN, an infinity, a signed zero, or
    round(A/B) with roundTiesToEven and an unbounded exponent range, computed with
    `Fraction` from the operands' exact values (never from their shortest decimal
    representation).  The runtime's departures from that value are recorded, not
    required; the implementation may depart by SLACK_ULP units of the last place and
    the departures are reported anyway.
    """
    rng = np.random.default_rng(20261003)
    rows, failures, notes = [], [], []
    checked = 0
    impl_departures = 0
    impl_worst = 0
    ref_departures = 0
    ref_worst = 0
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        a, b = _float_probe(dt, rng)
        exact = np.array([_doc_float_div(x, y, dt)
                          for x, y in zip(a.tolist(), b.tolist())], dtype=dt)
        ref = None
        try:
            ref = run_ort_batch(a, b)
        except Exception as exc:
            notes.append("%s: ONNX Runtime refused the probe (%s: %s)"
                         % (dt.name, type(exc).__name__, exc))
        with np.errstate(all="ignore"):
            got = np.asarray(impl.div(a, b))
        checked += int(a.size)

        if ref is not None and ref.shape == exact.shape and ref.dtype == exact.dtype:
            mismatch, departures = _compare_to_exact(exact, ref, dt)
            ref_departures += len(mismatch) + len(departures)
            if departures:
                ref_worst = max(ref_worst, max(d for _, d in departures))
            if mismatch:
                notes.append("%s: the runtime disagrees with the document at %d values "
                             "(the first: exact %r, ort %r), recorded, not required"
                             % (dt.name, len(mismatch),
                                float(exact.ravel()[mismatch[0]]),
                                float(ref.ravel()[mismatch[0]])))

        if got.shape != exact.shape or got.dtype != exact.dtype:
            failures.append({"who": "the implementation",
                             "type": dt.name,
                             "detail": "shape %s / type %s, the document's are %s / %s"
                                       % (got.shape, got.dtype, exact.shape,
                                          exact.dtype)})
            rows.append("%s: %d operand pairs (shape or type differs)"
                        % (dt.name, a.size))
            continue
        mismatch, departures = _compare_to_exact(exact, got, dt)
        impl_departures += len(departures)
        if departures:
            impl_worst = max(impl_worst, max(d for _, d in departures))
        too_far = [(i, d) for i, d in departures if d > SLACK_ULP]
        if mismatch:
            failures.append({
                "who": "the implementation vs the document",
                "type": dt.name,
                "detail": "; ".join(
                    "%r / %r: documented %r, implementation %r"
                    % (float(a.ravel()[i]), float(b.ravel()[i]),
                       float(exact.ravel()[i]), float(got.ravel()[i]))
                    for i in mismatch[:5]),
            })
        if too_far:
            failures.append({
                "who": "the implementation vs the document",
                "type": dt.name,
                "detail": "; ".join(
                    "%r / %r: correctly rounded %r, implementation %r, %d ulp"
                    % (float(a.ravel()[i]), float(b.ravel()[i]),
                       float(exact.ravel()[i]), float(got.ravel()[i]), d)
                    for i, d in too_far[:5]),
            })
        rows.append("%s: %d operand pairs" % (dt.name, a.size))

    summary = ("%d operand pairs over float16/float32/float64 against round(A/B) in "
               "Fraction arithmetic (the special values as operands and as results, the "
               "subnormal range and its ties, overflow and underflow); the "
               "implementation departs at %d values, worst %d ulp, slack %g; ONNX "
               "Runtime departs at %d values, worst %d ulp, recorded, not required"
               % (checked, impl_departures, impl_worst, SLACK_ULP,
                  ref_departures, ref_worst))
    if notes:
        summary += "; " + "; ".join(notes)
    return {"sweep": "the float formula against exact arithmetic (Fraction)",
            "summary": summary,
            "failures": failures}


def _spec_examples():
    """The worked examples of the float, int and uint sections, as documented."""
    f32, f64 = np.float32, np.float64
    nan, inf = float("nan"), float("inf")
    return [
        ("float Example 1",
         np.array([6.0, -9.0, 1.0], dtype=f32),
         np.array([3.0, 4.0, 8.0], dtype=f32),
         np.array([2.0, -2.25, 0.125], dtype=f32)),
        ("float Example 2",
         np.array([1.0, -1.0, inf, inf, 0.0, -0.0, 0.0, nan], dtype=f32),
         np.array([0.0, 0.0, inf, 4.0, 4.0, 4.0, 0.0, 1.0], dtype=f32),
         np.array([inf, -inf, nan, inf, 0.0, -0.0, nan, nan], dtype=f32)),
        ("float Example 3",
         np.array([1.0, 10.0, -10.0], dtype=f64),
         np.array([3.0, 3.0, 3.0], dtype=f64),
         np.array([0.3333333333333333, 3.3333333333333335, -3.3333333333333335],
                  dtype=f64)),
        ("int Example 1",
         np.array([7, -7, 7, -7, -128], dtype=np.int8),
         np.array([2, 2, -2, -2, -1], dtype=np.int8),
         np.array([3, -3, -3, 3, -128], dtype=np.int8)),
        ("uint Example 1",
         np.array([7, 200, 255, 6], dtype=np.uint8),
         np.array([2, 3, 255, 6], dtype=np.uint8),
         np.array([3, 66, 1, 1], dtype=np.uint8)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the typed sections: as documented, ORT, implementation."""
    failures = []
    examples = _spec_examples()
    for label, a, b, documented in examples:
        ref = run_ort_batch(a, b)
        with np.errstate(all="ignore"):
            got = np.asarray(impl.div(a, b))
        if not _bits_equal(documented, ref):
            failures.append({"who": "%s vs ONNX Runtime" % label,
                             "detail": "documented %s, ort %s"
                                       % (documented.tolist(), ref.tolist())})
        if not _bits_equal(documented, got):
            failures.append({"who": "%s vs the implementation" % label,
                             "detail": "documented %s, implementation %s"
                                       % (documented.tolist(), got.tolist())})
    return {"sweep": "the document's worked examples",
            "summary": "%d examples, as documented vs ONNX Runtime vs implementation"
                       % len(examples),
            "failures": failures}


def sweep_schema_types(impl):
    """The document's declared type families against the ONNX schema's admitted set."""
    declared = ["tensor(float16)", "tensor(float)", "tensor(double)",
                "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)",
                "tensor(uint8)", "tensor(uint16)", "tensor(uint32)",
                "tensor(uint64)"]
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
                    "(the profile specifies a subset)" % extra)
    return {"sweep": "the document's types against the ONNX schema",
            "summary": summary, "failures": failures}


def _real_examples():
    """The real section's cases: exact rationals, no ONNX type."""
    F = Fraction
    return [
        ("real Example 1",
         np.array([[F(7), F(-9), F(1)]], dtype=object),
         np.array([[F(2), F(4), F(3)]], dtype=object),
         np.array([[F(7, 2), F(-9, 4), F(1, 3)]], dtype=object)),
        ("real Example 2",
         np.array([[F(4), F(9)], [F(16), F(25)]], dtype=object),
         np.array([F(2), F(5)], dtype=object),
         np.array([[F(2), F(9, 5)], [F(8), F(5)]], dtype=object)),
        ("real: the quotients of 1/3 by 1/6, 1/10 by -1/10, 7 by 2",
         np.array([F(1, 3), F(1, 10), F(7)], dtype=object),
         np.array([F(1, 6), F(-1, 10), F(2)], dtype=object),
         None),
        ("real: a rank-0 dividend over a rank-1 divisor",
         np.array(F(3, 7), dtype=object),
         np.array([F(2, 5), F(1, 2)], dtype=object),
         None),
        ("real: broadcast of a row over a 3x2 matrix",
         np.array([[F(1, 3), F(2, 3)], [F(1, 7), F(3, 7)], [F(-1, 9), F(5, 9)]],
                  dtype=object),
         np.array([F(1, 6), F(1, 6)], dtype=object),
         None),
        ("real: a zero-sized dimension gives an empty result",
         np.zeros((0, 2), dtype=object),
         np.array([[F(2), F(5)]], dtype=object),
         np.zeros((0, 2), dtype=object)),
    ]


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact rationals.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    document's two Examples, its formula and its broadcasting rules against the
    quotients of `Fraction` arithmetic, and compares the implementation where it can
    evaluate a real tensor at all.
    """
    failures, notes = [], []
    checks = _real_examples()
    for label, a, b, documented in checks:
        ba, bc = np.broadcast_arrays(np.asarray(a, dtype=object),
                                     np.asarray(b, dtype=object))
        want = np.empty(ba.shape, dtype=object)
        for i in np.ndindex(ba.shape):
            want[i] = ba[i] / bc[i]
        if documented is not None:
            doc = np.asarray(documented, dtype=object)
            if doc.shape != want.shape:
                failures.append({"who": label,
                                 "detail": "documented shape %s, exact shape %s"
                                           % (doc.shape, want.shape)})
            else:
                bad = [i for i in np.ndindex(doc.shape) if doc[i] != want[i]]
                if bad:
                    failures.append({"who": label,
                                     "detail": "; ".join(
                                         "documented %s, exact %s"
                                         % (doc[i], want[i]) for i in bad[:5])})
        try:
            with np.errstate(all="ignore"):
                got = np.asarray(impl.div(a, b))
        except Exception as exc:
            notes.append("%s: the implementation cannot evaluate a real tensor (%s: %s)"
                         % (label, type(exc).__name__, exc))
            continue
        if got.dtype != object:
            notes.append("%s: the implementation returned dtype %s for a real tensor, "
                         "the exact quotient was not compared" % (label, got.dtype))
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
    summary = ("%d real cases checked against Fraction arithmetic (the two Examples, "
               "the formula, the rank-0 and zero-sized rules); no ONNX type, so no ONNX "
               "Runtime" % len(checks))
    if notes:
        summary += "; " + "; ".join(notes)
    return {"sweep": "the real section (exact rational arithmetic, no ONNX Runtime)",
            "summary": summary,
            "failures": failures}


def sweeps():
    """The bulk checks, each batched into one ONNX Runtime call where it needs one."""
    return [sweep_int_domain,
            sweep_int_out_of_range_pair,
            sweep_int_zero_divisor,
            sweep_float_exact,
            sweep_spec_examples,
            sweep_schema_types,
            sweep_real_examples]


# ---------------------------------------------------------------------------
# coverage: every section anchor of the document
# ---------------------------------------------------------------------------

COVERAGE = {anchor: 0 for anchor in ("real", "float", "int", "uint")}
for _entry in _CASES:
    COVERAGE[_entry[0]] += 1
COVERAGE["real"] = len(_real_examples())
