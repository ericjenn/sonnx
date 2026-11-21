"""The cases with which the **Mul** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled pair of operands per case.  Each case goes through a
                     one-node Mul model (opset 14) and through the implementation, and
                     the two results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the domain of the integer
                     types, the worked examples of every section of the document, and
                     the real section, which has no ONNX type and is checked against
                     exact rational arithmetic instead.

Every case of every section's formula is meant to appear here.  The two facts that make
the multiplication different from the addition, and that the cases below are built
around, are

  * the sign of a floating-point result is the exclusive or of the signs of the
    operands, so that a null product is not always +0, and the product of a zero by an
    infinity is the invalid operation;
  * the exact product of two n-bit integers is of the order of 2^(2n), so the reduction
    modulo 2^n may be applied several times, and may give a value of the other sign, or
    zero, although no operand is zero.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Mul"
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
# the integer definition of the document, transcribed literally
# ---------------------------------------------------------------------------
#
# The transcription is deliberately independent of the implementation: it is a second,
# literal, reading of the two sections **Mul** (int, int) and **Mul** (uint, uint),
# compared with ONNX Runtime over a whole domain.  A disagreement here is a defect of
# the document, not of the implementation.


def doc_int(x: int, y: int, n: int, signed: bool) -> int:
    """The value of the n-bit type congruent to the exact product, modulo 2^n.

    The document states the result as a congruence, not as a case analysis: the exact
    product may leave the range of the type by several multiples of 2^n, so a single
    conditional reduction would not do.  This is the literal reading: the non-negative
    residue of the product modulo 2^n, read as a signed value when the type is signed.
    """
    r = (x * y) % (2 ** n)
    if signed and r > 2 ** (n - 1) - 1:
        r -= 2 ** n
    return r


def _doc_int_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dt = a.dtype
    n = dt.itemsize * 8
    signed = dt.kind == "i"
    flat = [doc_int(int(x), int(y), n, signed)
            for x, y in zip(a.ravel(), b.ravel())]
    return np.array(flat, dtype=dt).reshape(a.shape)


def _rand_value(rng, info, dtype):
    """One random value of this integer type, drawn over the whole domain."""
    bits = np.dtype(dtype).itemsize * 8
    if info.min < 0:
        return int(rng.integers(info.min, info.max, dtype=np.int64, endpoint=True))
    if bits <= 32:
        return int(rng.integers(0, 2 ** bits, dtype=np.uint64))
    hi = int(rng.integers(0, 2 ** 32, dtype=np.uint64))
    lo = int(rng.integers(0, 2 ** 32, dtype=np.uint64))
    return (hi << 32) | lo


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_int_domain(impl):
    """The document's integer congruence against ONNX Runtime, over a domain.

    The two 8-bit types are covered exhaustively (all 65536 pairs each); the wider types
    use their boundary values, the values whose product is the largest and the smallest
    the type can have, and 200 random pairs.
    """
    rng = np.random.default_rng(20261003)
    rows, failures = [], []
    for dtype in ALL_INT:
        dt = np.dtype(dtype)
        info = np.iinfo(dt)
        exhaustive = dt.itemsize == 1
        if exhaustive:
            xs = np.arange(int(info.min), int(info.max) + 1, dtype=np.int64)
            a = np.repeat(xs, len(xs))
            b = np.tile(xs, len(xs))
        else:
            edges = sorted({min(max(int(v), int(info.min)), int(info.max))
                            for v in (info.min, info.min + 1, info.max - 1,
                                      info.max, 0, 1, -1)})
            pairs = [(x, y) for x in edges for y in edges]
            pairs += [(_rand_value(rng, info, dt), _rand_value(rng, info, dt))
                      for _ in range(200)]
            a = np.array([p[0] for p in pairs], dtype=dt)
            b = np.array([p[1] for p in pairs], dtype=dt)
        aa, bb = a.astype(dt), b.astype(dt)

        want = _doc_int_array(aa, bb)          # the document, literally
        ref = run_ort_batch(aa, bb)            # ONNX Runtime
        got = np.asarray(impl.mul(aa, bb))     # the implementation

        if not _bits_equal(want, ref):
            bad = np.nonzero(want.astype(np.float64)
                             != np.asarray(ref).astype(np.float64))[0][:5]
            failures.append({
                "type": dt.name, "who": "the document vs ONNX Runtime",
                "detail": "; ".join("doc: %d * %d = %d, ort: %d"
                                    % (int(aa.ravel()[i]), int(bb.ravel()[i]),
                                       int(want.ravel()[i]), int(ref.ravel()[i]))
                                    for i in bad),
            })
        if not _bits_equal(ref, got):
            mask = np.nonzero(np.asarray(ref).astype(np.float64)
                              != np.asarray(got).astype(np.float64))[0][:5]
            failures.append({
                "type": dt.name, "who": "the implementation vs ONNX Runtime",
                "detail": "; ".join("ort: %d, impl: %d"
                                    % (int(np.asarray(ref).ravel()[i]),
                                       int(got.ravel()[i])) for i in mask),
            })
        rows.append("%s %s" % (dt.name, "exhaustive" if exhaustive else "%d pairs" % len(aa)))
    return {"sweep": "int domain (document vs ORT vs implementation)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def _spec_examples():
    """The worked examples of the float, the integer and the unsigned sections."""
    f32, f64 = np.float32, np.float64
    return [
        ("float Example 1",
         np.array([[3.0, 4.5], [16.0, 1.0], [25.5, 24.25]], dtype=f32),
         np.array([[3.0, 2.0], [4.0, 0.0], [5.0, 4.0]], dtype=f32),
         np.array([[9.0, 9.0], [64.0, 0.0], [127.5, 97.0]], dtype=f32)),
        ("float Example 2",
         np.array([1.0, np.inf, -np.inf, 0.0, -0.0, -0.0], dtype=f32),
         np.array([1.0, 0.0, -2.0, -3.0, 5.0, -0.0], dtype=f32),
         np.array([1.0, np.nan, np.inf, -0.0, -0.0, 0.0], dtype=f32)),
        ("float Example 3",
         np.array([0.1, 1.0, 1.0e308], dtype=f64),
         np.array([0.2, 3.0, 1.0e308], dtype=f64),
         np.array([0.020000000000000004, 3.0, np.inf], dtype=f64)),
        ("float Example 4",
         np.array([2.2250738585072014e-308, -2.2250738585072014e-308, 0.0, -0.0],
                  dtype=f64),
         np.array([0.25, 0.25, -1.0, -0.0], dtype=f64),
         np.array([5.562684646268003e-309, -5.562684646268003e-309, -0.0, 0.0],
                  dtype=f64)),
        ("int Example 1",
         np.array([-6, 11, 100, 16], dtype=np.int8),
         np.array([-3, 13, 100, 16], dtype=np.int8),
         np.array([18, -113, 16, 0], dtype=np.int8)),
        ("uint Example 1",
         np.array([6, 200, 20, 16], dtype=np.uint8),
         np.array([3, 200, 13, 16], dtype=np.uint8),
         np.array([18, 64, 4, 0], dtype=np.uint8)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, and as ONNX Runtime does it."""
    failures = []
    for label, a, b, documented in _spec_examples():
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
                       % len(_spec_examples()),
            "failures": failures}


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact rationals."""
    F = Fraction
    examples = [
        ("real Example 1", [[F("6.1"), F("9.5"), F("35.7")]], [F(2), F(3), F(4)],
         [[F("12.2"), F("28.5"), F("142.8")]]),
        ("real Example 2",
         [[F("3.7"), F("4.4")], [F("16.2"), F("0.5")], [F("25.3"), F("24.8")]],
         [F(2), F(10)],
         [[F("7.4"), F("44.0")], [F("32.4"), F("5.0")], [F("50.6"), F("248.0")]]),
    ]
    failures = []
    for label, a, b, documented in examples:
        aa = np.array(a, dtype=object)
        bb = np.array(b, dtype=object)
        got = np.asarray(impl.mul(aa, bb))
        ra, rb = np.broadcast_arrays(aa, bb)
        want = np.array([[x * y for x, y in zip(row_a, row_b)]
                         for row_a, row_b in zip(ra.tolist(), rb.tolist())],
                        dtype=object)
        if got.shape != want.shape or any(got[i] != want[i]
                                          for i in np.ndindex(want.shape)):
            failures.append({"who": label, "detail": "exact %s, implementation %s"
                             % (want.tolist(), got.tolist())})
        if got.shape != np.array(documented, dtype=object).shape:
            failures.append({"who": label, "detail": "documented %s, implementation %s"
                             % (np.array(documented, dtype=object).tolist(), got.tolist())})
    return {"sweep": "real section (exact rationals, no ONNX Runtime)",
            "summary": "%d examples" % len(examples), "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_int_domain, sweep_real_examples]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _float_value_cases(dtype):
    """The value families of the float section."""
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    maxf = float(f.max)
    tiny = float(f.tiny)          # smallest normal: tiny / 2 and below are subnormal
    smallest = float(f.smallest_subnormal)
    step = 2.0 ** -f.nmant         # the spacing of the type just above 1
    nan, inf = float("nan"), float("inf")
    return [
        ("signed zeros", [0.0, -0.0, 0.0, -0.0, 0.0, -0.0],
         [5.0, 5.0, -5.0, -5.0, 0.0, -0.0]),
        ("infinities", [inf, -inf, inf, -inf, inf, -inf],
         [inf, inf, -inf, -inf, 2.0, -2.0]),
        ("zero times infinity (invalid)", [0.0, -0.0, inf, -inf], [inf, -inf, 0.0, -0.0]),
        ("nan", [nan, 1.0, nan, inf], [1.0, nan, inf, nan]),
        ("exact product", [1.0, 4.5, 16.0, 0.25], [0.5, 0.25, 8.0, 0.125]),
        ("overflow to inf", [maxf, -maxf, maxf, maxf], [maxf, maxf, 2.0, 1.0 + step]),
        ("just below the largest finite value", [maxf, maxf], [1.0, 1.0 - step / 2.0]),
        ("subnormal result", [tiny, tiny, -tiny, tiny], [0.5, 0.25, 0.5, 1.0]),
        ("underflow to zero", [tiny, -tiny, smallest, -tiny],
         [tiny, tiny, 0.5, smallest]),
        ("product needing rounding", [0.1, 0.3, 1.0 + step, 1.0 + step],
         [0.2, 0.1, 1.0 + step, 1.0 - step / 2.0]),
        ("exactly one step above 1.0", [1.0 + step], [1.0 + step]),
    ]


def _ordinary_value_cases(dtype):
    """The extreme products, the sign flips and the products that are zero."""
    dt = np.dtype(dtype)
    info = np.iinfo(dt)
    lo, hi = int(info.min), int(info.max)
    bits = dt.itemsize * 8
    half = 2 ** (bits // 2)        # half * half == 2^bits, a multiple of 2^bits
    over = isqrt(hi) + 1           # the smallest value whose square exceeds hi
    if lo < 0:
        a = [lo, hi, lo, hi, -1, 0, 1, over, half, -half]
        b = [lo, hi, hi, lo, -1, 1, 1, over, half, half]
    else:
        a = [lo, hi, hi, 1, 0, 1, over, half, half]
        b = [hi, hi, lo, 1, 1, 1, over, half, 0]
    return [("extremes+sign flips+zero products", a, b)]


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
        zero = np.zeros(1, dtype=dtype)
        yield ("%s: invalid operation, inf * 0" % np.dtype(dtype).name, [inf, zero])
        yield ("%s: invalid operation, 0 * -inf" % np.dtype(dtype).name,
               [zero, -inf])
    for dtype in ALL_INT:
        for name, a, b in _ordinary_value_cases(dtype):
            yield ("%s: %s" % (np.dtype(dtype).name, name),
                   [np.array(a, dtype=dtype), np.array(b, dtype=dtype)])
    for name, sa, sb in SHAPE_CASES:
        for dtype in ALL_FLOAT + ALL_INT:
            yield ("%s: %s" % (np.dtype(dtype).name, name),
                   [_arr(sa, dtype), _arr(sb, dtype, 1.0)])
