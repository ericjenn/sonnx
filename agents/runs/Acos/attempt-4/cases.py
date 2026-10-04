"""The cases with which the **Acos** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled operand per case.  Each case goes through a one-node
                     Acos model (opset 14) and through the implementation, and the two
                     results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the value families of the
                     three float types, the worked examples of every section of the
                     document, and the real section, which has no ONNX type and is
                     checked against exact and high-precision arithmetic instead.

Every case of every section's formula is meant to appear here, and the case set is part
of the loop: when a section is added to the document, its cases are added here before
the next run.

The runtime measured in this repository on 2026-10-04 has a `float16` kernel and a
`float` kernel for `Acos` and **no `double` kernel**: a `double` model is refused with
`NOT_IMPLEMENTED`.  In ``cases()`` that refusal is left to the harness, which records
it as an open case; in the sweeps, which build their own models, it is caught per type
and reported in the summary, never turned into a failure of the implementation.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Acos"
OPSET = 14

ALL_FLOAT = [np.float16, np.float32, np.float64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}
IR_VERSION = 10

# The types the runtime of this repository has a kernel for.  Measured 2026-10-04:
# Acos has float16 and float, and no double.
ORT_KERNEL_TYPES = {np.dtype(np.float16), np.dtype(np.float32)}


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(x: np.ndarray) -> np.ndarray:
    """One Acos node, one run.  Raises RuntimeError when the runtime refuses it."""
    proto = helper.np_dtype_to_tensor_dtype(x.dtype)
    vi_x = helper.make_tensor_value_info("X", proto, list(x.shape))
    vi_y = helper.make_tensor_value_info("Y", proto, None)
    node = helper.make_node(OP, ["X"], ["Y"], name="acos0")
    graph = helper.make_graph([node], "acos_graph", [vi_x], [vi_y])
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


def _first_diffs(want: np.ndarray, got: np.ndarray, n: int = 5):
    """A short human-readable list of the elements that differ."""
    if want.shape != got.shape:
        return "shape %s vs %s" % (want.shape, got.shape)
    flat_w = np.asarray(want).ravel()
    flat_g = np.asarray(got).ravel()
    out = []
    for i in range(flat_w.size):
        w, g = flat_w[i], flat_g[i]
        if w.dtype.kind == "f" and np.isnan(w) and np.isnan(g):
            continue
        if w != g:
            out.append("x=%r: want %r, got %r" % (None, w, g))
        if len(out) >= n:
            break
    return "; ".join(out) if out else "shapes differ"


# ---------------------------------------------------------------------------
# the real-number section, transcribed literally
# ---------------------------------------------------------------------------
#
# The real section has no ONNX type and ONNX Runtime cannot be asked about it.  It is
# checked against exact arithmetic: `fractions.Fraction` for the rationals of the
# examples, and a high-precision computation for the transcendental values.  The
# high-precision reference is computed here, independently of the platform's library,
# by the arctangent series with argument reduction, in `decimal` at 60 digits.


def _pi_decimal(dps: int):
    """pi to `dps` decimal places, by the Machin formula, in exact integer arithmetic."""
    from decimal import Decimal, getcontext
    getcontext().prec = dps + 20
    # Machin: pi/4 = 4*atan(1/5) - atan(1/239), each atan by its series.
    def atan_inv(n: int) -> "Decimal":
        n = Decimal(n)
        total = Decimal(0)
        term = Decimal(1) / n
        n2 = n * n
        k = 0
        sign = 1
        while True:
            add = term / (2 * k + 1)
            if add == 0:
                break
            total += sign * add
            term /= n2
            sign = -sign
            k += 1
        return total
    pi = 4 * (4 * atan_inv(5) - atan_inv(239))
    getcontext().prec = dps
    return +pi


def _acos_decimal(x: "Decimal", dps: int):
    """acos(x) for x in [-1, 1], to `dps` decimal places, by the arctangent series.

    acos(x) = atan2(sqrt(1 - x^2), x), and atan2 is computed from atan with the
    half-angle reduction so that the series converges quickly.  This is a reference
    computed here, not a call into the platform's library.
    """
    from decimal import Decimal, getcontext
    getcontext().prec = dps + 30
    one = Decimal(1)
    if x == one:
        getcontext().prec = dps
        return Decimal(0)
    if x == -one:
        getcontext().prec = dps
        return _pi_decimal(dps)
    # atan2(y, x) with y = sqrt(1 - x^2) >= 0, x in (-1, 1)
    y = (one - x * x).sqrt()
    # atan2(y, x) = 2 * atan(y / (sqrt(x^2 + y^2) + x))  -- half-angle, |arg| <= 1
    r = (x * x + y * y).sqrt()
    t = y / (r + x)
    # atan(t) by the series, with the argument halved until |t| is small
    halvings = 0
    while abs(t) > Decimal("0.1"):
        t = t / (one + (one + t * t).sqrt())
        halvings += 1
    total = Decimal(0)
    term = t
    t2 = t * t
    k = 0
    sign = 1
    while True:
        add = term / (2 * k + 1)
        if add == 0:
            break
        total += sign * add
        term *= t2
        sign = -sign
        k += 1
    total *= 2 ** halvings
    total *= 2  # the half-angle reduction above
    getcontext().prec = dps
    return +total


def _round_to_dtype(value, dtype):
    """Round a high-precision real to the nearest value of `dtype`, ties to even."""
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        return np.float64(value)
    if dtype == np.float32:
        return np.float32(np.float64(value))
    if dtype == np.float16:
        return np.float16(np.float64(value))
    raise ValueError(dtype)


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_float_values(impl):
    """The value families of the float section, against ONNX Runtime.

    The three types share one semantics; the runtime of this repository has a kernel
    for `float16` and `float` and none for `double`, so the `double` family is checked
    against the high-precision reference instead and the summary says so.
    """
    from decimal import Decimal, getcontext
    failures = []
    rows = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        maxf = float(f.max)
        tiny = float(f.tiny)
        nan, inf = float("nan"), float("inf")
        values = [
            ("signed zeros", [0.0, -0.0]),
            ("one", [1.0]),
            ("minus one", [-1.0]),
            ("half", [0.5]),
            ("minus half", [-0.5]),
            ("sqrt2/2", [float(np.sqrt(2.0) / 2.0)]),
            ("smallest positive subnormal", [float(np.nextafter(0.0, 1.0))]),
            ("largest subnormal", [tiny * (1.0 - 2.0 ** -f.nmant)]),
            ("smallest normal", [tiny]),
            ("largest finite", [maxf]),
            ("minus largest finite", [-maxf]),
            ("one ulp below 1", [float(np.nextafter(1.0, 0.0))]),
            ("one ulp above -1", [float(np.nextafter(-1.0, 0.0))]),
            ("nan", [nan]),
            ("inf", [inf]),
            ("minus inf", [-inf]),
        ]
        x = np.array([v for _, vs in values for v in vs], dtype=dt)
        want = np.array([np.nan] * len(x), dtype=dt)
        # the exact arccosine, rounded to the type, is the specification's result
        getcontext().prec = 60
        for i, v in enumerate(x):
            if np.isnan(v):
                want[i] = np.nan
            elif v == 1.0:
                want[i] = dt.type(0.0)
            elif v == -1.0:
                want[i] = _round_to_dtype(_pi_decimal(60), dt)
            else:
                want[i] = _round_to_dtype(_acos_decimal(Decimal(float(v)), 60), dt)

        if dt in ORT_KERNEL_TYPES:
            try:
                ref = run_ort_batch(x)
            except Exception as exc:
                failures.append({"who": "%s vs ONNX Runtime" % dt.name,
                                 "detail": "the runtime refused the model: %s" % exc})
                rows.append("%s: refused" % dt.name)
                continue
            if not _bits_equal(want, ref):
                failures.append({"who": "%s: the document vs ONNX Runtime" % dt.name,
                                 "detail": _first_diffs(want, ref)})
            got = np.asarray(impl.acos(x))
            if not _bits_equal(ref, got):
                failures.append({"who": "%s: the implementation vs ONNX Runtime" % dt.name,
                                 "detail": _first_diffs(ref, got)})
            rows.append("%s: %d values vs ORT" % (dt.name, x.size))
        else:
            got = np.asarray(impl.acos(x))
            if not _bits_equal(want, got):
                failures.append({"who": "%s: the implementation vs the exact reference" % dt.name,
                                 "detail": _first_diffs(want, got)})
            rows.append("%s: %d values vs the exact reference (no ORT kernel)"
                        % (dt.name, x.size))
    return {"sweep": "float value families (document vs ORT vs implementation)",
            "summary": "; ".join(rows), "failures": failures}


def sweep_float_domain(impl):
    """A dense sample of the domain [-1, 1] for each type, against ONNX Runtime.

    The domain is not small enough to be covered exhaustively, so this is the boundary
    values plus a random sample.  The `double` type has no kernel in this runtime and is
    checked against the high-precision reference instead.
    """
    from decimal import Decimal, getcontext
    rng = np.random.default_rng(20261004)
    failures = []
    rows = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        # boundary values of the domain, and a random sample of it
        edges = np.array([-1.0, -1.0 + 2.0 ** -f.nmant, -0.5, -0.0, 0.0, 0.5,
                          1.0 - 2.0 ** -f.nmant, 1.0], dtype=dt)
        sample = rng.uniform(-1.0, 1.0, size=4096).astype(dt)
        x = np.concatenate([edges, sample])
        getcontext().prec = 60
        want = np.array([_round_to_dtype(_acos_decimal(Decimal(float(v)), 60), dt)
                         for v in x], dtype=dt)
        if dt in ORT_KERNEL_TYPES:
            try:
                ref = run_ort_batch(x)
            except Exception as exc:
                failures.append({"who": "%s vs ONNX Runtime" % dt.name,
                                 "detail": "the runtime refused the model: %s" % exc})
                rows.append("%s: refused" % dt.name)
                continue
            if not _bits_equal(want, ref):
                failures.append({"who": "%s: the document vs ONNX Runtime" % dt.name,
                                 "detail": _first_diffs(want, ref)})
            got = np.asarray(impl.acos(x))
            if not _bits_equal(ref, got):
                failures.append({"who": "%s: the implementation vs ONNX Runtime" % dt.name,
                                 "detail": _first_diffs(ref, got)})
            rows.append("%s: %d values vs ORT" % (dt.name, x.size))
        else:
            got = np.asarray(impl.acos(x))
            if not _bits_equal(want, got):
                failures.append({"who": "%s: the implementation vs the exact reference" % dt.name,
                                 "detail": _first_diffs(want, got)})
            rows.append("%s: %d values vs the exact reference (no ORT kernel)"
                        % (dt.name, x.size))
    return {"sweep": "float domain sample (boundaries + random)",
            "summary": "; ".join(rows), "failures": failures}


def _spec_examples():
    """The worked examples of the float section, as documented."""
    f32 = np.float32
    return [
        ("float Example 1",
         np.array([-1.0, 0.0, 1.0], dtype=f32),
         np.array([3.14159274, 1.57079637, 0.0], dtype=f32)),
        ("float Example 2",
         np.array([0.0, -0.0], dtype=f32),
         np.array([1.57079637, 1.57079637], dtype=f32)),
        ("float Example 3",
         np.array([np.nan], dtype=f32),
         np.array([np.nan], dtype=f32)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, and as ONNX Runtime does it."""
    failures = []
    for label, x, documented in _spec_examples():
        try:
            ref = run_ort_batch(x)
        except Exception as exc:
            failures.append({"who": label, "detail": "the runtime refused the model: %s" % exc})
            continue
        if not _bits_equal(documented, ref):
            failures.append({"who": label, "detail": "documented %s, ORT %s"
                             % (documented.tolist(), ref.tolist())})
        got = np.asarray(impl.acos(x))
        if not _bits_equal(documented, got):
            failures.append({"who": label, "detail": "documented %s, implementation %s"
                             % (documented.tolist(), got.tolist())})
    return {"sweep": "the document's worked examples",
            "summary": "%d examples, as documented vs ORT vs implementation"
                       % len(_spec_examples()),
            "failures": failures}


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact arithmetic.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    implementation against the values the document's Examples state, in exact
    arithmetic: `fractions.Fraction` for the rationals, and the high-precision
    reference for the transcendental values.
    """
    from decimal import Decimal, getcontext
    F = Fraction
    getcontext().prec = 60
    pi = _pi_decimal(60)
    examples = [
        ("real Example 1",
         [F(-1), F(0), F(1)],
         [pi, pi / 2, Decimal(0)]),
        ("real Example 2",
         [F(1, 2), F(1, 2) * Decimal(2).sqrt(), F(-1, 2)],
         [pi / 3, pi / 4, 2 * pi / 3]),
    ]
    failures = []
    for label, xs, documented in examples:
        got = np.asarray(impl.acos(np.array(xs, dtype=object)))
        for i, (x, want) in enumerate(zip(xs, documented)):
            g = got.ravel()[i]
            # the implementation may return a Fraction, a Decimal or a float; compare
            # in exact arithmetic where it can, and to 50 digits otherwise
            if isinstance(g, Fraction):
                gd = Decimal(g.numerator) / Decimal(g.denominator)
            else:
                gd = Decimal(g)
            if abs(gd - want) > Decimal(10) ** -50:
                failures.append({"who": label,
                                 "detail": "x=%s: documented %s, implementation %s"
                                           % (x, want, g)})
    return {"sweep": "real section (exact and high-precision arithmetic, no ONNX Runtime)",
            "summary": "%d examples, checked to 50 digits" % len(examples),
            "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_float_values, sweep_float_domain,
            sweep_real_examples]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _bit_nan(dtype, pattern):
    uint = UINT_OF_WIDTH[np.dtype(dtype).itemsize]
    return np.array([pattern], dtype=uint).view(dtype)


NAN_PATTERNS = {2: (0x7E01, 0x7C01, 0x8000),
                4: (0x7FC00001, 0x7F800001, 0x80000000),
                8: (0x7FF8000000000001, 0x7FF0000000000001, 0x8000000000000000)}


def _float_value_cases(dtype):
    """The value families of the float section, as operands."""
    f = np.finfo(np.dtype(dtype))
    maxf = float(f.max)
    tiny = float(f.tiny)
    nan, inf = float("nan"), float("inf")
    return [
        ("signed zeros", [0.0, -0.0]),
        ("one and minus one", [1.0, -1.0]),
        ("half and minus half", [0.5, -0.5]),
        ("sqrt2/2", [float(np.sqrt(2.0) / 2.0)]),
        ("smallest positive subnormal", [float(np.nextafter(0.0, 1.0))]),
        ("largest subnormal", [tiny * (1.0 - 2.0 ** -f.nmant)]),
        ("smallest normal", [tiny]),
        ("largest finite", [maxf]),
        ("minus largest finite", [-maxf]),
        ("one ulp below 1", [float(np.nextafter(1.0, 0.0))]),
        ("one ulp above -1", [float(np.nextafter(-1.0, 0.0))]),
        ("nan", [nan]),
        ("inf", [inf]),
        ("minus inf", [-inf]),
    ]


def _arr(shape, dtype, start=0.0):
    n = 1
    for d in shape:
        n *= d
    return (np.arange(n, dtype=np.float64) + start).astype(dtype).reshape(shape)


SHAPE_CASES = [
    ("rank-0", ()),
    ("rank-1", (3,)),
    ("rank-2", (2, 3)),
    ("rank-3", (2, 3, 4)),
    ("zero-sized: (0,)", (0,)),
    ("zero-sized: (0,3)", (0, 3)),
    ("zero-sized: (2,0,4)", (2, 0, 4)),
]


def cases():
    """Every individual case: (label, [X])."""
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        for name, values in _float_value_cases(dtype):
            yield ("%s: %s" % (dt.name, name),
                   [np.array(values, dtype=dtype)])
        quiet, signalling, sign = NAN_PATTERNS[dt.itemsize]
        yield ("%s: NaN operand, quiet, payload 1" % dt.name,
               [_bit_nan(dtype, quiet)])
        yield ("%s: NaN operand, quiet, payload 1, sign set" % dt.name,
               [_bit_nan(dtype, quiet | sign)])
        yield ("%s: NaN operand, signalling" % dt.name,
               [_bit_nan(dtype, signalling)])
        # the worked examples of the float section, replayed as documented
        yield ("%s: Example 1" % dt.name,
               [np.array([-1.0, 0.0, 1.0], dtype=dtype)])
        yield ("%s: Example 2" % dt.name,
               [np.array([0.0, -0.0], dtype=dtype)])
        yield ("%s: Example 3" % dt.name,
               [np.array([np.nan], dtype=dtype)])
    for name, shape in SHAPE_CASES:
        for dtype in ALL_FLOAT:
            yield ("%s: %s" % (np.dtype(dtype).name, name),
                   [_arr(shape, dtype, -0.5)])


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

COVERAGE = {
    "real": 0,
    "float": 0,
}

for _label, _arrays in cases():
    if _label.startswith("float16") or _label.startswith("float32") \
            or _label.startswith("float64"):
        COVERAGE["float"] += 1
    else:
        COVERAGE["real"] += 1

# the real section has no ONNX type: its cases live in the sweeps, and the count
# records the two worked examples plus the value families checked there
COVERAGE["real"] = 2
