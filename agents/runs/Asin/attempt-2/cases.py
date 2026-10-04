"""The cases with which the **Asin** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled operand per case.  Each case goes through a
                     one-node Asin model (opset 14) and through the implementation,
                     and the two results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls
                     because they must be batched into a single run: the float
                     domain (boundary values and a random sample), the document's
                     worked examples, and the real section, which has no ONNX type
                     and is checked against exact arithmetic instead.

Every case of every section's formula is meant to appear here, and the case set is
part of the loop: when a section is added to the document, its cases are added here
before the next run.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Asin"
OPSET = 14
IR_VERSION = 10

ALL_FLOAT = [np.float16, np.float32, np.float64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}

# ---------------------------------------------------------------------------
# high-precision arcsine, independent of the platform's library
# ---------------------------------------------------------------------------

getcontext().prec = 60
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")


def _asin_decimal(x: Decimal) -> Decimal:
    """The exact arcsine of a Decimal in [-1, 1], to 60 digits.

    Uses the identity asin(x) = pi/2 - 2*asin(sqrt((1-x)/2)) for x > 1/2 to
    reduce the argument, and the Taylor series for the remaining small argument.
    """
    if x == 0:
        return Decimal(0)
    if x < 0:
        return -_asin_decimal(-x)
    if x > 1:
        raise ValueError("asin domain")
    if x > Decimal("0.5"):
        z = ((Decimal(1) - x) / 2).sqrt()
        return PI / 2 - 2 * _asin_decimal(z)
    x2 = x * x
    term = x
    total = term
    n = 0
    while True:
        n += 1
        term = term * x2 * Decimal((2 * n - 1) ** 2) / Decimal(2 * n * (2 * n + 1))
        total += term
        if abs(term) < Decimal(10) ** (-55):
            break
    return total


def _spec_float_asin(x, dtype):
    """The value the float section specifies for one operand, as a scalar of dtype."""
    dt = np.dtype(dtype)
    if np.isnan(x):
        return np.array([np.nan], dtype=dt)[0]
    if x == 0.0:
        return np.array([x], dtype=dt)[0]
    if abs(x) > 1.0:
        return np.array([np.nan], dtype=dt)[0]
    y = _asin_decimal(Decimal(float(x)))
    return np.array([str(y)], dtype=dt)[0]


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch_asin(x: np.ndarray) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------

NAN_PATTERNS = {2: (0x7E01, 0x7C01, 0x8000),
                4: (0x7FC00001, 0x7F800001, 0x80000000),
                8: (0x7FF8000000000001, 0x7FF0000000000001, 0x8000000000000000)}


def _bit_nan(dtype, pattern):
    uint = UINT_OF_WIDTH[np.dtype(dtype).itemsize]
    return np.array([pattern], dtype=uint).view(dtype)


def _float_special_cases(dtype):
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    tiny = float(f.tiny)
    smallest_sub = float(getattr(f, "smallest_subnormal", tiny * 2.0 ** -f.nmant))
    maxf = float(f.max)
    one = 1.0
    next_after_one = float(np.nextafter(dt.type(one), dt.type(0.0)))
    next_after_one_up = float(np.nextafter(dt.type(one), dt.type(2.0)))
    next_after_neg_one = float(np.nextafter(dt.type(-one), dt.type(0.0)))
    next_after_neg_one_down = float(np.nextafter(dt.type(-one), dt.type(-2.0)))
    return [
        ("signed zeros", [0.0, -0.0]),
        ("infinities", [np.inf, -np.inf]),
        ("nan", [np.nan]),
        ("subnormals", [smallest_sub, -smallest_sub, tiny, -tiny]),
        ("largest finite", [maxf, -maxf]),
        ("around +1", [one, next_after_one, next_after_one_up]),
        ("around -1", [-one, next_after_neg_one, next_after_neg_one_down]),
        ("outside domain", [2.0, -2.0, 1.0 + float(f.eps), -1.0 - float(f.eps)]),
        ("exact values", [0.5, -0.5, 0.25, -0.25, 0.75, -0.75]),
        ("example 1", [-1.0, -0.5, 0.0, 0.5, 1.0]),
        ("example 2", [np.nan, np.inf, -np.inf, 2.0, -0.0]),
    ]


SHAPE_CASES = [
    ("rank-0", ()),
    ("zero-sized (0,)", (0,)),
    ("zero-sized (0,3)", (0, 3)),
    ("zero-sized (2,0,3)", (2, 0, 3)),
    ("shape (2,3)", (2, 3)),
    ("shape (1,2,3)", (1, 2, 3)),
    ("shape (4,)", (4,)),
]


def _all_cases():
    out = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        for name, vals in _float_special_cases(dtype):
            out.append(("%s: %s" % (dt.name, name), [np.array(vals, dtype=dt)]))
        quiet, signalling, sign = NAN_PATTERNS[dt.itemsize]
        out.append(("%s: NaN quiet payload 1" % dt.name, [_bit_nan(dt, quiet)]))
        out.append(("%s: NaN quiet payload 1 sign set" % dt.name,
                    [_bit_nan(dt, quiet | sign)]))
        out.append(("%s: NaN signalling" % dt.name, [_bit_nan(dt, signalling)]))
        if dt == np.float16:
            out.append(("float16: example 3 subnormal",
                        [np.array([2.0 ** -24], dtype=dt)]))
        for name, shape in SHAPE_CASES:
            if shape == ():
                arr = np.array(0.5, dtype=dt)
            else:
                size = int(np.prod(shape))
                if size == 0:
                    arr = np.zeros(shape, dtype=dt)
                else:
                    arr = np.linspace(-1.0, 1.0, num=size).astype(dt).reshape(shape)
            out.append(("%s: %s" % (dt.name, name), [arr]))
    return out


_CASES = _all_cases()


def cases():
    """Every individual case: (label, [X])."""
    yield from _CASES


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_float_domain(impl):
    """The float section's formula over boundary values and a random sample.

    For each of the three types, the batch is run through ONNX Runtime and through
    the implementation, and both are compared with the value the document specifies
    (computed with a 60-digit arcsine and rounded to the type).
    """
    try:
        failures = []
        counts = []
        for dtype in ALL_FLOAT:
            dt = np.dtype(dtype)
            f = np.finfo(dt)
            tiny = float(f.tiny)
            smallest_sub = float(getattr(f, "smallest_subnormal",
                                         tiny * 2.0 ** -f.nmant))
            maxf = float(f.max)
            one = 1.0
            next_after_one = float(np.nextafter(dt.type(one), dt.type(0.0)))
            next_after_one_up = float(np.nextafter(dt.type(one), dt.type(2.0)))
            next_after_neg_one = float(np.nextafter(dt.type(-one), dt.type(0.0)))
            next_after_neg_one_down = float(np.nextafter(dt.type(-one), dt.type(-2.0)))
            vals = [
                0.0, -0.0, np.inf, -np.inf, np.nan,
                smallest_sub, -smallest_sub, tiny, -tiny, maxf, -maxf,
                one, -one, next_after_one, next_after_one_up,
                next_after_neg_one, next_after_neg_one_down,
                2.0, -2.0, 1.0 + float(f.eps), -1.0 - float(f.eps),
                0.5, -0.5, 0.25, -0.25, 0.75, -0.75,
            ]
            rng = np.random.default_rng(20261003)
            vals += rng.uniform(-1.0, 1.0, size=2000).tolist()
            vals += rng.uniform(-3.0, 3.0, size=200).tolist()
            arr = np.array(vals, dtype=dt)
            ref = run_ort_batch_asin(arr)
            got = np.asarray(impl.asin(arr))
            want = np.array([_spec_float_asin(float(x), dt) for x in arr], dtype=dt)

            if not _bits_equal(ref, got):
                uint = UINT_OF_WIDTH[dt.itemsize]
                mask = ~((np.asarray(ref).view(uint) == np.asarray(got).view(uint))
                         | (np.isnan(ref) & np.isnan(got)))
                idx = np.nonzero(mask)[0][:5]
                failures.append({
                    "type": dt.name,
                    "who": "implementation vs ONNX Runtime",
                    "detail": "; ".join("x=%r ort=%r impl=%r"
                                        % (float(arr[i]), float(ref[i]), float(got[i]))
                                        for i in idx),
                })

            spec_nan = np.isnan(want)
            if not np.all(np.isnan(ref[spec_nan])):
                failures.append({
                    "type": dt.name,
                    "who": "spec vs ONNX Runtime (NaN expected)",
                    "detail": "ORT did not return NaN for %d values"
                              % int(np.sum(~np.isnan(ref[spec_nan]))),
                })
            mask = ~spec_nan
            if not _bits_equal(want[mask], ref[mask]):
                w = want[mask]
                r = ref[mask]
                uint = UINT_OF_WIDTH[dt.itemsize]
                diff = ~((w.view(uint) == r.view(uint))
                         | (np.isnan(w) & np.isnan(r)))
                idx = np.nonzero(diff)[0][:5]
                failures.append({
                    "type": dt.name,
                    "who": "spec vs ONNX Runtime",
                    "detail": "; ".join("x=%r spec=%r ort=%r"
                                        % (float(arr[mask][i]), float(w[i]), float(r[i]))
                                        for i in idx),
                })
            counts.append("%s %d values" % (dt.name, len(arr)))
        return {
            "sweep": "float domain (boundary + random, spec vs ORT vs implementation)",
            "summary": "; ".join(counts),
            "failures": failures,
        }
    except Exception as exc:
        return {"sweep": "float domain", "summary": "ERROR",
                "failures": [{"who": "sweep", "detail": "%s: %s"
                              % (type(exc).__name__, exc)}]}


def sweep_spec_examples(impl):
    """The document's worked examples for the float section, replayed as documented."""
    try:
        failures = []
        # float Example 1
        x1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        ref1 = run_ort_batch_asin(x1)
        got1 = np.asarray(impl.asin(x1))
        want1 = np.array([_spec_float_asin(float(v), np.float32) for v in x1],
                         dtype=np.float32)
        if not _bits_equal(want1, ref1):
            failures.append({"who": "float Example 1 spec vs ORT",
                             "detail": "spec %s, ORT %s"
                                       % (want1.tolist(), ref1.tolist())})
        if not _bits_equal(want1, got1):
            failures.append({"who": "float Example 1 spec vs impl",
                             "detail": "spec %s, impl %s"
                                       % (want1.tolist(), got1.tolist())})
        # float Example 2
        x2 = np.array([np.nan, np.inf, -np.inf, 2.0, -0.0], dtype=np.float32)
        ref2 = run_ort_batch_asin(x2)
        got2 = np.asarray(impl.asin(x2))
        if not (np.isnan(ref2[0]) and np.isnan(ref2[1]) and np.isnan(ref2[2])
                and np.isnan(ref2[3]) and ref2[4] == 0.0 and np.signbit(ref2[4])):
            failures.append({"who": "float Example 2 ORT",
                             "detail": "ORT %s" % ref2.tolist()})
        if not (np.isnan(got2[0]) and np.isnan(got2[1]) and np.isnan(got2[2])
                and np.isnan(got2[3]) and got2[4] == 0.0 and np.signbit(got2[4])):
            failures.append({"who": "float Example 2 impl",
                             "detail": "impl %s" % got2.tolist()})
        # float16 Example 3
        x3 = np.array([2.0 ** -24], dtype=np.float16)
        ref3 = run_ort_batch_asin(x3)
        got3 = np.asarray(impl.asin(x3))
        want3 = np.array([2.0 ** -24], dtype=np.float16)
        if not _bits_equal(want3, ref3):
            failures.append({"who": "float16 Example 3 spec vs ORT",
                             "detail": "spec %s, ORT %s"
                                       % (want3.tolist(), ref3.tolist())})
        if not _bits_equal(want3, got3):
            failures.append({"who": "float16 Example 3 spec vs impl",
                             "detail": "spec %s, impl %s"
                                       % (want3.tolist(), got3.tolist())})
        return {
            "sweep": "the document's worked examples (float)",
            "summary": "3 examples checked against spec, ORT, implementation",
            "failures": failures,
        }
    except Exception as exc:
        return {"sweep": "float examples", "summary": "ERROR",
                "failures": [{"who": "sweep", "detail": "%s: %s"
                              % (type(exc).__name__, exc)}]}


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact arithmetic.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    document's real examples against a 60-digit arcsine computed independently of
    the platform's library.
    """
    try:
        failures = []
        sqrt2_over_2 = Decimal(2).sqrt() / 2
        inputs = [Decimal(-1), -sqrt2_over_2, Decimal(0), Decimal(1) / 2, Decimal(1)]
        expected = [-PI / 2, -PI / 4, Decimal(0), PI / 6, PI / 2]
        for i, (x, e) in enumerate(zip(inputs, expected)):
            y = _asin_decimal(x)
            if abs(y - e) > Decimal(10) ** (-50):
                failures.append({
                    "who": "real Example 1 element %d" % i,
                    "detail": "asin(%s) = %s, documented %s" % (x, y, e),
                })
        y = _asin_decimal(Decimal(1) / 2)
        if abs(y - PI / 6) > Decimal(10) ** (-50):
            failures.append({
                "who": "real Example 2",
                "detail": "asin(0.5) = %s, documented %s" % (y, PI / 6),
            })
        return {
            "sweep": "real section (exact arithmetic, no ONNX Runtime)",
            "summary": "2 examples checked against high-precision arcsine (60 digits)",
            "failures": failures,
        }
    except Exception as exc:
        return {"sweep": "real examples", "summary": "ERROR",
                "failures": [{"who": "sweep", "detail": "%s: %s"
                              % (type(exc).__name__, exc)}]}


def sweeps():
    return [sweep_float_domain, sweep_spec_examples, sweep_real_examples]


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

COVERAGE = {
    "real": 2,          # the two real examples checked in sweep_real_examples
    "float": len(_CASES),
}
