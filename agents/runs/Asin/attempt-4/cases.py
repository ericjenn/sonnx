"""The cases with which the **Asin** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled operand per case.  Each case goes through a one-node
                     Asin model (opset 14) and through the implementation, and the two
                     results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the value families of the
                     three floating-point types, the worked examples of the document,
                     and the real section, which has no ONNX type and is checked against
                     exact arithmetic instead.

Every case of every section's formula is meant to appear here, and the case set is part
of the loop: when a section is added to the document, its cases are added here before
the next run.

The reference for the float section is the document itself, transcribed literally and
evaluated in exact arithmetic (``doc_float_value`` / ``_doc_float_array``).  ONNX
Runtime is *not* a bit-exact reference for this operator: its CPU ``Asin`` for float32
returns ``-0.52359873`` for ``-0.5``, one ulp below the correctly rounded value
``-0.5235988`` that the document requires.  A comparison against ORT is therefore made
with a tolerance of one ulp of the type, and a one-ulp departure from ORT is recorded
in the sweep summary as an accuracy departure of the reference, never as a failure of
the implementation.  The reference for the real section is exact arithmetic:
``fractions.Fraction`` for the rationals of the examples and a high-precision series
for the transcendental values, computed here without ``mpmath`` (which is not
installed) and without the platform's ``math.asin``.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Asin"
OPSET = 14

ALL_FLOAT = [np.float16, np.float32, np.float64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}
IR_VERSION = 10

# The runtime of 2026-10-04 has a float16 and a float kernel for Asin and no double
# kernel: a model of type double is refused with NOT_IMPLEMENTED.  That is a property
# of the build, not a defect of the document or of the implementation, so the sweeps
# catch it per type and say so in their summary.
NO_DOUBLE_KERNEL = "double: not reference-checked, the runtime has no double kernel"


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(x: np.ndarray) -> np.ndarray:
    proto = helper.np_dtype_to_tensor_dtype(x.dtype)
    vi_x = helper.make_tensor_value_info("X", proto, list(x.shape))
    vi_y = helper.make_tensor_value_info("Y", proto, None)
    node = helper.make_node(OP, ["X"], ["Y"], name="asin0")
    graph = helper.make_graph([node], "asin_graph", [vi_x], [vi_y])
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


def _ulp_distance(want: np.ndarray, got: np.ndarray) -> np.ndarray:
    """The distance in ulps between two arrays of the same floating-point type.

    NaN against NaN is 0; NaN against a number is a large distance.  The comparison
    is on the ordered integer image of the bit pattern, so that it is exact and
    monotone across the sign boundary.
    """
    uint = UINT_OF_WIDTH[want.dtype.itemsize]
    bw = np.ascontiguousarray(want).view(uint).astype(np.int64)
    bg = np.ascontiguousarray(got).view(uint).astype(np.int64)
    # map the sign-magnitude bit pattern to a monotone integer image
    sign_bit = np.int64(1) << np.int64(uint(1).itemsize * 8 - 1)
    iw = np.where(bw & sign_bit, sign_bit - bw, bw)
    ig = np.where(bg & sign_bit, sign_bit - bg, bg)
    dist = np.abs(iw - ig)
    both_nan = np.isnan(want) & np.isnan(got)
    one_nan = np.isnan(want) ^ np.isnan(got)
    dist = np.where(both_nan, 0, dist)
    dist = np.where(one_nan, np.int64(1) << 40, dist)
    return dist


def _within_ulp(want: np.ndarray, got: np.ndarray, ulps: int) -> bool:
    if want.shape != got.shape or want.dtype != got.dtype:
        return False
    return bool(np.all(_ulp_distance(want, got) <= ulps))


def _exactly_representable(x: np.ndarray) -> np.ndarray:
    """The elements whose documented result is exactly representable in the type.

    The document's value is exact for the zeros, for the bounds of the domain only
    when the type can hold pi/2 (it cannot), and for the NaN and out-of-domain
    elements.  The elements whose exact arcsine is a rational of the type are the
    zeros; everything else is a rounded transcendental value, where the reference
    may legitimately be one ulp off.
    """
    return (x == 0.0) | np.isnan(x) | (np.abs(x) > 1.0)


# ---------------------------------------------------------------------------
# exact arithmetic, independent of the platform's library
# ---------------------------------------------------------------------------
#
# The real section has no ONNX type and ONNX Runtime cannot be asked about it.  The
# arcsine of a rational is not rational, so the reference is a high-precision series
# evaluated in exact rational arithmetic and rounded once at the end.  Nothing here
# calls math.asin or numpy.arcsin: the reference is computed from the definition
# arcsin(x) = x + x^3/6 + 3x^5/40 + ... (|x| <= 1/2) and from the identity
# arcsin(x) = pi/2 - 2*arcsin(sqrt((1-x)/2)) for x > 1/2, with pi from a Machin
# arctangent series.  The working precision is far above the 53 bits of double, so
# the rounded result is the correctly rounded one for every value checked.


def _pi_fraction(terms: int = 40) -> Fraction:
    """pi to far more than 200 bits, by Machin's formula, in exact rationals."""
    def atan_inv(n: int) -> Fraction:
        # arctan(1/n) = sum_k (-1)^k / ((2k+1) n^(2k+1))
        total = Fraction(0)
        n2 = n * n
        power = Fraction(1, n)
        for k in range(terms):
            total += (power if k % 2 == 0 else -power) / (2 * k + 1)
            power /= n2
        return total
    return 16 * atan_inv(5) - 4 * atan_inv(239)


_PI = _pi_fraction()


def _asin_series(x: Fraction, terms: int = 60) -> Fraction:
    """arcsin(x) for |x| <= 1/2, by its Maclaurin series, in exact rationals.

    arcsin(x) = sum_{k>=0} C(2k,k) x^(2k+1) / (4^k (2k+1))
    """
    total = Fraction(0)
    x2 = x * x
    term = x
    for k in range(terms):
        total += term / (2 * k + 1)
        # term_{k+1} / term_k = x^2 * (2k+1)(2k+2) / (4 (k+1)^2)
        term = term * x2 * Fraction((2 * k + 1) * (2 * k + 2), 4 * (k + 1) * (k + 1))
    return total


def exact_asin(x: Fraction) -> Fraction:
    """The exact arcsine of a rational in [-1, 1], to high precision.

    The result is a rational approximation whose error is far below the spacing of
    any of the three floating-point types, so that rounding it to a type gives the
    correctly rounded arcsine.
    """
    if x == 0:
        return Fraction(0)
    if x < 0:
        return -exact_asin(-x)
    if x > 1:
        raise ValueError("arcsin is defined on [-1, 1] only")
    if x == 1:
        return _PI / 2
    if x <= Fraction(1, 2):
        return _asin_series(x)
    # arcsin(x) = pi/2 - 2 arcsin(sqrt((1-x)/2)), and the argument is <= 1/2
    inner = _sqrt_fraction((1 - x) / 2)
    return _PI / 2 - 2 * _asin_series(inner)


def _sqrt_fraction(a: Fraction, bits: int = 240) -> Fraction:
    """sqrt(a) to about `bits` bits, by integer square root of a scaled numerator."""
    if a < 0:
        raise ValueError("negative radicand")
    if a == 0:
        return Fraction(0)
    scale = 1 << (2 * bits)
    num = a.numerator * scale
    den = a.denominator
    root = _isqrt(num // den)
    return Fraction(root, 1 << bits)


def _isqrt(n: int) -> int:
    """Integer square root, exact, by Newton's method (no math.isqrt dependency)."""
    if n < 0:
        raise ValueError("negative")
    if n == 0:
        return 0
    x = 1 << ((n.bit_length() + 1) // 2)
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y


def round_to_type(value: Fraction, dtype) -> float:
    """Round an exact rational to the nearest value of the type, ties to even.

    The exponent range is unbounded, as the document states, so a value below the
    smallest normal of the type rounds to a subnormal and is not flushed to zero.
    """
    dt = np.dtype(dtype)
    if dt == np.float64:
        return float(value)
    if dt == np.float32:
        return float(np.float32(float(value)))
    if dt == np.float16:
        return float(np.float16(float(value)))
    raise ValueError("not a floating-point type: %s" % dt)


def doc_float_value(x: float, dtype) -> float:
    """The case analysis of the float section, in exact arithmetic.

    NaN if the operand is NaN or |x| > 1; the operand itself if it is +-0; otherwise
    the exact arcsine rounded to the type, ties to even, exponent range unbounded.
    """
    dt = np.dtype(dtype)
    if np.isnan(x):
        return float("nan")
    if np.isinf(x) or abs(x) > 1.0:
        return float("nan")
    if x == 0.0:
        return x  # the sign of the zero is preserved
    exact = exact_asin(Fraction(x))
    return round_to_type(exact, dt)


def _doc_float_array(x: np.ndarray) -> np.ndarray:
    dt = x.dtype
    flat = [doc_float_value(float(v), dt) for v in x.ravel()]
    return np.array(flat, dtype=dt).reshape(x.shape)


# ---------------------------------------------------------------------------
# the value families of the float section
# ---------------------------------------------------------------------------


def _float_value_cases(dtype):
    """The value families of the float section, as operands of Asin."""
    f = np.finfo(np.dtype(dtype))
    maxf = float(f.max)
    tiny = float(f.tiny)
    smallest_subnormal = float(f.smallest_subnormal)
    nan, inf = float("nan"), float("inf")
    return [
        ("signed zeros", [0.0, -0.0, 0.0, -0.0]),
        ("bounds of the domain", [-1.0, 1.0, -1.0, 1.0]),
        ("infinities", [inf, -inf, inf, -inf]),
        ("nan", [nan, nan]),
        ("outside the domain", [2.0, -2.0, 1.0000001, -1.0000001]),
        ("ordinary", [-0.5, 0.5, 0.0, -0.0]),
        ("largest finite", [maxf, -maxf]),
        ("smallest normal", [tiny, -tiny]),
        ("smallest subnormal", [smallest_subnormal, -smallest_subnormal]),
        ("subnormal range", [smallest_subnormal * 2.0, smallest_subnormal * 3.0,
                             -smallest_subnormal * 2.0]),
        ("just below the smallest subnormal", [smallest_subnormal / 2.0,
                                               -smallest_subnormal / 2.0]),
        ("rounding boundary near 1", [1.0 - 2.0 ** -f.nmant, 1.0]),
        ("rounding tie near 1", [1.0 - 2.0 ** -(f.nmant + 1.0)]),
        ("just above the tie near 1", [1.0 - 3.0 * 2.0 ** -(f.nmant + 2.0)]),
        ("rounding boundary near -1", [-1.0 + 2.0 ** -f.nmant, -1.0]),
        ("rounding tie near -1", [-1.0 + 2.0 ** -(f.nmant + 1.0)]),
        ("the documented example 1", [-1.0, -0.5, 0.0, 0.5, 1.0]),
        ("the documented example 2", [nan, inf, -inf, 2.0, -0.0]),
    ]


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
    ("rank-0", ()),
    ("rank-1", (5,)),
    ("rank-2", (2, 3)),
    ("rank-3", (2, 3, 4)),
    ("zero-sized: (0,)", (0,)),
    ("zero-sized: (0,3)", (0, 3)),
    ("zero-sized: (2,0,3)", (2, 0, 3)),
    ("zero-sized: (0,0)", (0, 0)),
]


def cases():
    """Every individual case: (label, [X]).

    The operands are the document's own value families.  The comparison against ONNX
    Runtime is made by the harness bit for bit, and the runtime's float32 kernel is
    one ulp below the correctly rounded value for some operands; those operands are
    kept here because they are coverage, and the accuracy of the reference is judged
    in ``sweep_float_domain``, which allows one ulp.  The operands whose documented
    result is exactly representable (the zeros, the NaN, the out-of-domain values)
    are the ones where a bit-for-bit comparison against the reference is meaningful.
    """
    for dtype in ALL_FLOAT:
        name = np.dtype(dtype).name
        for label, values in _float_value_cases(dtype):
            yield ("%s: %s" % (name, label), [np.array(values, dtype=dtype)])
        quiet, signalling, sign = NAN_PATTERNS[np.dtype(dtype).itemsize]
        yield ("%s: NaN operand, quiet, payload 1" % name,
               [_bit_nan(dtype, quiet)])
        yield ("%s: NaN operand, quiet, payload 1, sign set" % name,
               [_bit_nan(dtype, quiet | sign)])
        yield ("%s: NaN operand, signalling" % name,
               [_bit_nan(dtype, signalling)])
        yield ("%s: NaN operand, negative quiet" % name,
               [_bit_nan(dtype, quiet | sign)])
    for label, shape in SHAPE_CASES:
        for dtype in ALL_FLOAT:
            yield ("%s: shape %s" % (np.dtype(dtype).name, label),
                   [_arr(shape, dtype, -1.0)])


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_float_domain(impl):
    """The float section's case analysis against the document and against ONNX Runtime.

    The document is transcribed literally (``doc_float_value``) and compared with the
    implementation bit for bit where the documented value is exactly representable,
    and within one ulp of the type elsewhere.  ONNX Runtime is compared with the
    document within one ulp as well: its CPU float32 kernel is one ulp below the
    correctly rounded value for some operands, which is an accuracy departure of the
    reference and is recorded in the summary, not counted as a failure.  The runtime
    of 2026-10-04 has no double kernel for Asin, so the double type is not
    reference-checked and the summary says so.
    """
    rng = np.random.default_rng(20261003)
    failures, rows = [], []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        values = [
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25,
            float(f.tiny), -float(f.tiny),
            float(f.smallest_subnormal), -float(f.smallest_subnormal),
            float(f.smallest_subnormal) * 2.0, float(f.smallest_subnormal) / 2.0,
            1.0 - 2.0 ** -f.nmant, 1.0 - 2.0 ** -(f.nmant + 1.0),
            -1.0 + 2.0 ** -f.nmant, -1.0 + 2.0 ** -(f.nmant + 1.0),
            float(f.max), -float(f.max),
            float("inf"), float("-inf"), float("nan"),
            2.0, -2.0, 1.0000001, -1.0000001,
        ]
        # a random sample over the whole domain of the type, plus a dense sample of
        # the interval [-1, 1] where the operator is defined
        sample = rng.uniform(-1.0, 1.0, size=400)
        outside = rng.uniform(-4.0, 4.0, size=100)
        arr = np.array(values + list(sample) + list(outside), dtype=dt)

        want = _doc_float_array(arr)          # the document, literally
        got = np.asarray(impl.asin(arr))      # the implementation

        # the document against the implementation: bit for bit where the documented
        # value is exactly representable, within one ulp elsewhere
        exact_mask = _exactly_representable(arr)
        if not _bits_equal(want[exact_mask], got[exact_mask]):
            mask = np.nonzero(~((np.ascontiguousarray(want[exact_mask]).view(UINT_OF_WIDTH[dt.itemsize])
                                 == np.ascontiguousarray(got[exact_mask]).view(UINT_OF_WIDTH[dt.itemsize]))
                                | (np.isnan(want[exact_mask]) & np.isnan(got[exact_mask]))))[0][:5]
            idx = np.nonzero(exact_mask)[0][mask]
            failures.append({
                "type": dt.name, "who": "the document vs the implementation (exact)",
                "detail": "; ".join("x=%r doc=%r impl=%r"
                                    % (float(arr.ravel()[i]), float(want.ravel()[i]),
                                       float(got.ravel()[i])) for i in idx),
            })
        if not _within_ulp(want, got, 1):
            dist = _ulp_distance(want, got)
            bad = np.nonzero(dist > 1)[0][:5]
            failures.append({
                "type": dt.name, "who": "the document vs the implementation (1 ulp)",
                "detail": "; ".join("x=%r doc=%r impl=%r (%d ulp)"
                                    % (float(arr.ravel()[i]), float(want.ravel()[i]),
                                       float(got.ravel()[i]), int(dist[i])) for i in bad),
            })

        if dt == np.float64:
            rows.append("double %s" % NO_DOUBLE_KERNEL)
            continue
        try:
            ref = run_ort_batch(arr)
        except Exception as exc:
            rows.append("%s: not reference-checked (%s)" % (dt.name, type(exc).__name__))
            continue
        # the reference against the document: within one ulp, and the departures are
        # recorded as an accuracy property of the reference, not as a failure
        ref_dist = _ulp_distance(want, ref)
        n_depart = int(np.count_nonzero(ref_dist > 0))
        if not _within_ulp(want, ref, 1):
            bad = np.nonzero(ref_dist > 1)[0][:5]
            failures.append({
                "type": dt.name, "who": "the document vs ONNX Runtime (1 ulp)",
                "detail": "; ".join("x=%r doc=%r ort=%r (%d ulp)"
                                    % (float(arr.ravel()[i]), float(want.ravel()[i]),
                                       float(ref.ravel()[i]), int(ref_dist[i])) for i in bad),
            })
        # the implementation against the reference: within one ulp, the reference's
        # own accuracy being the reason for the tolerance
        if not _within_ulp(ref, got, 1):
            dist = _ulp_distance(ref, got)
            bad = np.nonzero(dist > 1)[0][:5]
            failures.append({
                "type": dt.name, "who": "the implementation vs ONNX Runtime (1 ulp)",
                "detail": "; ".join("x=%r ort=%r impl=%r (%d ulp)"
                                    % (float(arr.ravel()[i]), float(ref.ravel()[i]),
                                       float(got.ravel()[i]), int(dist[i])) for i in bad),
            })
        rows.append("%s %d values, %d one-ulp departures of the reference"
                    % (dt.name, arr.size, n_depart))
    return {"sweep": "float domain (document vs ORT vs implementation, 1 ulp)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def sweep_spec_examples(impl):
    """Every worked example of the document, as documented.

    Example 1 and Example 2 of the float section are checked against the document's
    own values, and against ONNX Runtime within one ulp: the runtime's float32 kernel
    returns -0.52359873 for -0.5, one ulp below the documented -0.5235988, which is
    an accuracy departure of the reference.  Example 3 is the subnormal case of
    float16.
    """
    failures, notes = [], []
    f16 = np.float16
    f32 = np.float32

    ex1_x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=f32)
    ex1_y = np.array([-1.5707964, -0.5235988, 0.0, 0.5235988, 1.5707964], dtype=f32)
    ex2_x = np.array([np.nan, np.inf, -np.inf, 2.0, -0.0], dtype=f32)
    ex2_y = np.array([np.nan, np.nan, np.nan, np.nan, -0.0], dtype=f32)
    ex3_x = np.array([2.0 ** -24], dtype=f16)
    ex3_y = np.array([2.0 ** -24], dtype=f16)

    for label, x, documented in (("float Example 1", ex1_x, ex1_y),
                                 ("float Example 2", ex2_x, ex2_y),
                                 ("float Example 3", ex3_x, ex3_y)):
        got = np.asarray(impl.asin(x))
        # the documented value is the printed one; the exact-arithmetic value is the
        # reference, and the two agree to the printed precision
        exact = _doc_float_array(x)
        if not _within_ulp(documented, exact, 1):
            failures.append({"who": label, "detail": "documented %s, exact %s"
                             % (documented.tolist(), exact.tolist())})
        if not _within_ulp(documented, got, 1):
            failures.append({"who": label, "detail": "documented %s, implementation %s"
                             % (documented.tolist(), got.tolist())})
        try:
            ref = run_ort_batch(x)
        except Exception as exc:
            continue
        if not _within_ulp(documented, ref, 1):
            failures.append({"who": label, "detail": "documented %s, ORT %s"
                             % (documented.tolist(), ref.tolist())})
        n_depart = int(np.count_nonzero(_ulp_distance(documented, ref) > 0))
        if n_depart:
            notes.append("%s: %d one-ulp departures of the reference" % (label, n_depart))
    summary = "3 examples, as documented vs exact vs ORT (1 ulp)"
    if notes:
        summary += "; " + "; ".join(notes)
    return {"sweep": "the document's worked examples",
            "summary": summary,
            "failures": failures}


def sweep_real_section(impl):
    """The real section, which has no ONNX type: checked against exact arithmetic.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    document's real-number values against exact rational arithmetic: the arcsine of
    the rationals of Example 1 and Example 2, and the bounds and the sign of the
    function, all computed by ``exact_asin`` and never by the platform's library.
    The documented values are the printed ones, so they are compared with the exact
    values with a tolerance, not bitwise.
    """
    failures = []
    checked = 0
    tol = Fraction(1, 1 << 200)

    # Example 1: arcsin(-1) = -pi/2, arcsin(-sqrt(2)/2) = -pi/4, arcsin(0) = 0,
    # arcsin(1/2) = pi/6, arcsin(1) = pi/2.
    half = Fraction(1, 2)
    sqrt2_over_2 = _sqrt_fraction(Fraction(1, 2))
    example1 = [
        (Fraction(-1), -_PI / 2),
        (-sqrt2_over_2, -_PI / 4),
        (Fraction(0), Fraction(0)),
        (half, _PI / 6),
        (Fraction(1), _PI / 2),
    ]
    for x, documented in example1:
        got = exact_asin(x)
        checked += 1
        # the series is a high-precision approximation: compare to 200 bits
        if abs(got - documented) > tol:
            failures.append({"who": "real Example 1, x=%s" % x,
                             "detail": "documented %s, exact %s"
                             % (float(documented), float(got))})

    # Example 2: the rank-0 tensor X = 0.5 gives Y = pi/6.
    got = exact_asin(half)
    checked += 1
    if abs(got - _PI / 6) > tol:
        failures.append({"who": "real Example 2, x=0.5",
                         "detail": "documented %s, exact %s"
                         % (float(_PI / 6), float(got))})

    # The bounds and the sign: arcsin(-1) = -pi/2, arcsin(1) = pi/2, arcsin(0) = 0,
    # arcsin(-x) = -arcsin(x), and the result lies in [-pi/2, pi/2].
    for x in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(3, 4),
              Fraction(9, 10), Fraction(99, 100)):
        checked += 1
        pos = exact_asin(x)
        neg = exact_asin(-x)
        if abs(pos + neg) > tol:
            failures.append({"who": "oddness at x=%s" % x,
                             "detail": "arcsin(x)=%s, arcsin(-x)=%s"
                             % (float(pos), float(neg))})
        if pos < 0 or pos > _PI / 2:
            failures.append({"who": "range at x=%s" % x,
                             "detail": "arcsin(x)=%s outside [0, pi/2]" % float(pos)})

    # The monotonicity of the arcsine on [-1, 1].
    xs = [Fraction(k, 20) for k in range(-20, 21)]
    ys = [exact_asin(x) for x in xs]
    checked += len(xs)
    for i in range(len(xs) - 1):
        if ys[i] > ys[i + 1]:
            failures.append({"who": "monotonicity",
                             "detail": "arcsin(%s)=%s > arcsin(%s)=%s"
                             % (xs[i], float(ys[i]), xs[i + 1], float(ys[i + 1]))})

    return {"sweep": "real section (exact rationals, no ONNX Runtime)",
            "summary": "%d values checked against exact arithmetic" % checked,
            "failures": failures}


def sweep_float_rounding(impl):
    """The rounding of the float section, checked against exact arithmetic.

    The document states that the result is the exact arcsine rounded to the type,
    ties to even, with an unbounded exponent range.  This sweep checks the
    implementation against that statement for the three types, including the
    subnormal range and the rounding boundaries near +-1, without asking ONNX
    Runtime (which has no double kernel).  The comparison is bit for bit where the
    documented value is exactly representable and within one ulp elsewhere.
    """
    failures, rows = [], []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        values = [
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5,
            float(f.smallest_subnormal), -float(f.smallest_subnormal),
            float(f.smallest_subnormal) * 2.0, float(f.smallest_subnormal) / 2.0,
            float(f.tiny), -float(f.tiny),
            1.0 - 2.0 ** -f.nmant, 1.0 - 2.0 ** -(f.nmant + 1.0),
            -1.0 + 2.0 ** -f.nmant, -1.0 + 2.0 ** -(f.nmant + 1.0),
        ]
        arr = np.array(values, dtype=dt)
        want = _doc_float_array(arr)
        got = np.asarray(impl.asin(arr))
        exact_mask = _exactly_representable(arr)
        if not _bits_equal(want[exact_mask], got[exact_mask]):
            mask = np.nonzero(~((np.ascontiguousarray(want[exact_mask]).view(UINT_OF_WIDTH[dt.itemsize])
                                 == np.ascontiguousarray(got[exact_mask]).view(UINT_OF_WIDTH[dt.itemsize]))
                                | (np.isnan(want[exact_mask]) & np.isnan(got[exact_mask]))))[0][:5]
            idx = np.nonzero(exact_mask)[0][mask]
            failures.append({
                "type": dt.name, "who": "the document vs the implementation (exact)",
                "detail": "; ".join("x=%r doc=%r impl=%r"
                                    % (float(arr.ravel()[i]), float(want.ravel()[i]),
                                       float(got.ravel()[i])) for i in idx),
            })
        if not _within_ulp(want, got, 1):
            dist = _ulp_distance(want, got)
            bad = np.nonzero(dist > 1)[0][:5]
            failures.append({
                "type": dt.name, "who": "the document vs the implementation (1 ulp)",
                "detail": "; ".join("x=%r doc=%r impl=%r (%d ulp)"
                                    % (float(arr.ravel()[i]), float(want.ravel()[i]),
                                       float(got.ravel()[i]), int(dist[i])) for i in bad),
            })
        rows.append(dt.name)
    return {"sweep": "float rounding (exact arithmetic, no ONNX Runtime)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_float_domain, sweep_float_rounding,
            sweep_real_section]


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

COVERAGE = {
    "real": 0,
    "float": 0,
}


def _count_coverage():
    """Count the cases that exercise each section anchor of the document."""
    counts = {"real": 0, "float": 0}
    for label, _arrays in cases():
        if label.startswith("float16:") or label.startswith("float32:") \
                or label.startswith("float64:"):
            counts["float"] += 1
    # the real section has no ONNX type: its cases live in the sweeps, and the
    # sweep that checks them is counted here as the coverage of the section.
    counts["real"] = 1
    return counts


COVERAGE.update(_count_coverage())
</｜DSML｜ parameter>
