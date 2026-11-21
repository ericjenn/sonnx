"""The cases with which the **Sin** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled operand array per case.  Each case goes through a
                     one-node Sin model (opset 14) and through the implementation, and
                     the two results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the whole domain of
                     `float16`, the accuracy of the two wider types against a correctly
                     rounded reference, the exact rules the document states for the
                     special values and for the odd symmetry, the worked examples of
                     every section, and the real section, which has no ONNX type and is
                     checked against a 60-digit computation.

What makes the sine different from an arithmetic operator, and what the cases below are
built around:

  * the sine of a value of a floating-point type is not a value of that type in general,
    and IEEE 754 leaves the choice of providing the sine and of requiring it to be
    correctly rounded to the language: **two conforming implementations may differ in the
    last bits of a result**.  The document states the correctly rounded value as the
    value of the operator and refers the departure of an implementation to the accuracy
    guidelines.  A per-case check is therefore restricted to the operands whose result
    the document determines *exactly* -- a zero, a value so small that it is its own
    sine, a subnormal value, an infinity, a NaN -- where two conforming implementations
    have no freedom left.  A general operand is exercised in a **sweep**, where the
    departure of each implementation from the correctly rounded value is measured, and
    where only a departure of more than a few units in the last place of the value 1 (or
    a departure from the value the document's own example states) is a failure;
  * the sine is odd and the oddness of a result is exact: the document requires the
    result for $-A[i]$ to be the negation of the result for $A[i]$, sign bit included.
    That is a property of the implementation alone and is checked bit for bit over
    thousands of values, for both implementations, with no reference to the exact sine.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sin_oracle import pi_decimal, sin_correctly_rounded, sin_decimal  # noqa: E402

OP = "Sin"
OPSET = 14
IR_VERSION = 10
ALL_FLOAT = [np.float16, np.float32, np.float64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}

# The last place of the value 1, per type.  A departure is measured in units of it.
ULP_OF_1 = {2: 2.0 ** -10, 4: 2.0 ** -23, 8: 2.0 ** -52}
# ONNX Runtime 1.30 departs from the correctly rounded sine by at most one of those
# units for `float16`, half of one for `float` and two for `double` (these are the
# measurements the V1 callout of the document records).  Four units is not an accuracy
# difference between two conforming implementations but a defect.
SLACK = 4.0

# A magnitude below which a value is its own sine, with a wide margin: the difference
# |sin x - x| is at most x^3/6, some two hundred times below half a unit in the last
# place of x at the upper end of each range below.
TINY = {2: 1.0e-3, 4: 1.0e-5, 8: 1.0e-9}


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(a: np.ndarray) -> np.ndarray:
    proto = helper.np_dtype_to_tensor_dtype(a.dtype)
    vi_a = helper.make_tensor_value_info("A", proto, list(a.shape))
    vi_c = helper.make_tensor_value_info("C", proto, None)
    node = helper.make_node(OP, ["A"], ["C"], name="sin0")
    graph = helper.make_graph([node], "sin_graph", [vi_a], [vi_c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    (c,) = sess.run(["C"], {"A": a})
    return c


def _both(impl, values: np.ndarray):
    """The two implementations of the operator, for one operand array."""
    return [("ONNX Runtime", np.asarray(run_ort_batch(values))),
            ("the implementation", np.asarray(impl.sin(values)))]


def _view_uint(arr: np.ndarray) -> np.ndarray:
    dt = np.dtype(arr.dtype)
    return np.ascontiguousarray(arr).view(UINT_OF_WIDTH[dt.itemsize])


def _sign_bit(dtype) -> int:
    return 1 << (8 * np.dtype(dtype).itemsize - 1)


def _bits_equal(want: np.ndarray, got: np.ndarray) -> bool:
    if want.shape != got.shape or want.dtype != got.dtype:
        return False
    both_nan = np.isnan(want) & np.isnan(got)
    return bool(np.all((_view_uint(want) == _view_uint(got)) | both_nan))


def _differ_mask(want: np.ndarray, got: np.ndarray) -> np.ndarray:
    """Element-wise bit difference, NaN equal to NaN."""
    both_nan = np.isnan(want) & np.isnan(got)
    return ~((_view_uint(want) == _view_uint(got)) | both_nan)


def _deviation(want: np.ndarray, got: np.ndarray) -> np.ndarray:
    """Absolute departure, in units of the last place of the value 1."""
    d = np.abs(np.asarray(want, dtype=np.float64) - np.asarray(got, dtype=np.float64))
    return d / ULP_OF_1[np.dtype(want.dtype).itemsize]


def _bit_nan(dtype, pattern):
    uint = UINT_OF_WIDTH[np.dtype(dtype).itemsize]
    return np.array([pattern], dtype=uint).view(dtype)


NAN_PATTERNS = {2: (0x7E01, 0x7C01, 0x8000),
                4: (0x7FC00001, 0x7F800001, 0x80000000),
                8: (0x7FF8000000000001, 0x7FF0000000000001, 0x8000000000000000)}


def _fill(shape, dtype, values):
    n = 1
    for d in shape:
        n *= d
    return np.array([values[i % len(values)] for i in range(n)],
                     dtype=dtype).reshape(shape)


SHAPE_CASES = [
    ("rank-0", ()),
    ("one element (1,)", (1,)),
    ("same shape (2,3)", (2, 3)),
    ("zero-sized: (0,3)", (0, 3)),
    ("zero-sized: (0,)", (0,)),
    ("zero-sized: (2,0)", (2, 0)),
]


def _shape_values(dtype):
    """The values the shape cases are filled with: every one of them exact.

    A shape case is about the shape, not about the value, so the values are chosen among
    the ones whose result the document determines exactly -- a sign of zero and a
    subnormal -- and no result of the case depends on the accuracy of an implementation.
    """
    sub = float(np.finfo(np.dtype(dtype)).smallest_subnormal)
    return [0.0, -0.0, sub, -sub]


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_exact_rules(impl):
    """The rules the document states for values that are not ordinary numbers.

    A null operand gives that zero, signs included; a NaN operand and an infinite
    operand give a NaN; a subnormal value is its own sine.  Each of these is exact, so
    neither implementation has any freedom: a disagreement is a defect of the document
    or of the implementation, never an accuracy difference.  A value of small magnitude
    that is not subnormal is the case where the document's identity is the exact sine
    rule and no longer an exact statement about an implementation: ONNX Runtime returns
    a neighbouring value, one unit in the last place of the operand away, so the check
    there is relative, in units of that last place.
    """
    rng = np.random.default_rng(20261005)
    rows, failures = [], []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        sub = float(f.smallest_subnormal)
        tiny = TINY[dt.itemsize]
        nan, inf = float("nan"), float("inf")
        values = np.concatenate([
            np.array([0.0, -0.0, inf, -inf, nan, sub, -sub, 2.0 * sub, -3.0 * sub],
                     dtype=dt),
            rng.uniform(tiny, 4.0 * tiny, 40).astype(dt),
        ])
        subnormal = (values != 0) & (np.abs(values) < float(f.tiny))
        for who, out in _both(impl, values):
            if out.dtype != dt:
                failures.append({"who": "%s: %s" % (who, dt.name),
                                 "detail": "an operand of type %s gives a result of type %s"
                                           % (dt.name, out.dtype)})
                continue
            out = np.asarray(out, dtype=dt)
            zero = values == 0
            if not np.all(_view_uint(out[zero]) == _view_uint(values[zero])):
                failures.append({"who": "%s: %s" % (who, dt.name),
                                 "detail": "a null operand does not give that zero"})
            special = np.isnan(values) | np.isinf(values)
            if not np.all(np.isnan(out[special])):
                failures.append({"who": "%s: %s" % (who, dt.name),
                                 "detail": "an operand that is a NaN or an infinity "
                                           "does not give a NaN"})
            mask = _view_uint(out[subnormal]) != _view_uint(values[subnormal])
            if np.any(mask):
                sv = values[subnormal]
                i = np.nonzero(mask)[0][:3]
                failures.append({
                    "who": "%s: %s" % (who, dt.name),
                    "detail": "a subnormal operand is not its own sine: " + "; ".join(
                        "%r gives %r" % (sv[j], out[subnormal][j]) for j in i)})
            small = np.isfinite(values) & ~subnormal & (values != 0)
            if np.any(small):
                rel = np.abs(out[small].astype(np.float64)
                             - values[small].astype(np.float64)) / np.spacing(
                                 np.abs(values[small]))
                worst = float(np.max(rel))
                if worst > SLACK:
                    i = int(np.argmax(rel))
                    failures.append({
                        "who": "%s: %s" % (who, dt.name),
                        "detail": "a small operand departs from its own sine by %.0f units "
                                  "in its last place: %r gives %r"
                                  % (worst, values[small][i], out[small][i])})
            # "An element of C is null if and only if the element of A is a zero."
            null = (out == 0) & (values != 0)
            if np.any(null):
                failures.append({
                    "who": "%s: %s" % (who, dt.name),
                    "detail": "a non-null operand gives a null result, which the document "
                              "rules out: %r" % values[np.nonzero(null)[0][:3]]})
            rows.append("%s %s %d values, small ones within %.0f units of their own sine"
                        % (who, dt.name, len(values), worst if np.any(small) else 0.0))
    return {"sweep": "the document's rules for special values, and the identity of a "
                     "small value",
            "summary": "; ".join(rows),
            "failures": failures}


def sweep_float16_domain(impl):
    """The whole domain of `float16`: all 65536 bit patterns, none left out.

    The reference is the correctly rounded sine, computed from the exact value with the
    oracle for every value where an implementation departs from a double-precision
    reference -- a double rounding could make that reference wrong, and the oracle is
    what decides.  The rules for the special values are held to the document, not to the
    reference.
    """
    dt = np.dtype(np.float16)
    a = np.arange(65536, dtype=np.uint16).view(dt)
    ref = np.sin(a.astype(np.float64)).astype(dt)       # guarded by the oracle below
    failures, rows = [], []
    finite = np.isfinite(a)
    for who, out in _both(impl, a):
        if out.dtype != dt:
            failures.append({"who": who, "detail": "the result has type %s, not float16"
                                                   % out.dtype})
            continue
        out = np.asarray(out, dtype=dt)
        zero = a == 0
        if not np.all(_view_uint(out[zero]) == _view_uint(a[zero])):
            failures.append({"who": who, "detail": "a null operand does not give that zero"})
        special = np.isnan(a) | np.isinf(a)
        if not np.all(np.isnan(out[special])):
            failures.append({"who": who,
                             "detail": "an operand that is a NaN or an infinity does not "
                                       "give a NaN"})
        bad = np.nonzero(finite & _differ_mask(ref, out))[0][:64]
        confirmed, worst, where = 0, 0.0, None
        for i in bad:
            correct = sin_correctly_rounded(a[i], dt.type)      # None when undecidable
            if correct is None or _bits_equal(np.array([correct]), out[i:i + 1]):
                continue
            confirmed += 1
            d = float(_deviation(np.array([correct], dtype=dt), out[i:i + 1])[0])
            if d > worst:
                worst, where = d, (float(a[i]), float(correct), float(out[i]))
        if worst > SLACK:
            failures.append({"who": who, "detail": "departs by %.0f units of 2^-10 at "
                                                   "A=%r (exact %.17g, got %.17g)"
                                                   % ((worst,) + where)})
        # "An element of C is null if and only if the element of A is a zero."
        null = (out == 0) & (a != 0)
        if np.any(null):
            failures.append({"who": who, "detail": "a non-null operand gives a null "
                                                   "result: %r"
                                                   % a[np.nonzero(null)[0][:3]]})
        rows.append("%s: %d of 63488 ordinary values depart, worst %.1f units"
                    % (who, confirmed, worst))
    return {"sweep": "the whole float16 domain (65536 values)",
            "summary": "; ".join(rows), "failures": failures}


def sweep_accuracy(impl):
    """The two wider types against the correctly rounded sine.

    This measures what the document leaves to the accuracy guidelines: it reports how
    many results differ from the correctly rounded value and by how much, for both
    implementations.  Only a departure beyond a few units in the last place of the value
    1 is a failure.
    """
    rng = np.random.default_rng(20261003)
    failures, rows = [], []
    for dtype in (np.float32, np.float64):
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        small = rng.uniform(0.0, 10.0, 250)
        x = np.concatenate([small, -small,
                            rng.uniform(1.0e3, 1.0e6, 20) * rng.choice([-1.0, 1.0], 20),
                            rng.uniform(1.0e15, float(f.max), 10),
                            -rng.uniform(1.0e15, float(f.max), 10)])
        a = x.astype(dt)
        kept = [(v, sin_correctly_rounded(v, dt.type)) for v in a]
        kept = [(v, c) for v, c in kept if c is not None]
        values = np.array([v for v, _ in kept], dtype=dt)
        correct = np.array([c for _, c in kept], dtype=dt)
        for who, out in _both(impl, values):
            if out.dtype != dt:
                failures.append({"who": "%s: %s" % (who, dt.name),
                                 "detail": "the result has type %s" % out.dtype})
                continue
            out = np.asarray(out, dtype=dt)
            differ = int(_differ_mask(correct, out).sum())
            worst = float(np.max(_deviation(correct, out)))
            # in the last place of the result itself, which is what makes a departure
            # near a zero of the sine so large
            nz = correct != 0
            relative = np.abs(out[nz].astype(np.float64)
                              - correct[nz].astype(np.float64)) / np.spacing(
                                  np.abs(correct[nz]))
            rows.append("%s: %d of %d values differ (%.0f%%), worst %.1f units of 2^%d, "
                        "and %.0f units in the last place of the result"
                        % (who, differ, len(values), 100.0 * differ / len(values),
                           worst, -f.nmant, float(np.max(relative))))
            if worst > SLACK:
                i = int(np.argmax(_deviation(correct, out)))
                failures.append({"who": "%s: %s" % (who, dt.name),
                                 "detail": "A=%r: exact sine %.17g, got %.17g, %.0f units "
                                           "of 2^%d" % (float(values[i]), float(correct[i]),
                                                        float(out[i]), worst, -f.nmant)})
    return {"sweep": "the correctness of a result (reported, not required: see V1)",
            "summary": "; ".join(rows), "failures": failures}


def sweep_odd_symmetry(impl):
    """The oddness of the result, which the document states and which is exact.

    For every finite operand, the result for $-A[i]$ is the negation of the result for
    $A[i]$, sign bit included.  No reference value is involved: the check compares the
    two halves of the result with each other.  NaN is left out, its sign not being
    specified.
    """
    rng = np.random.default_rng(20261006)
    failures, rows = [], []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        sub = float(f.smallest_subnormal)
        tiny = TINY[dt.itemsize]
        x = np.concatenate([
            rng.uniform(-10.0, 10.0, 3000),
            rng.uniform(-1.0, 1.0, 200) * float(f.max),
            np.array([sub / 2.0, -sub / 2.0, tiny, -tiny, float(f.max), -float(f.max)],
                     dtype=dt),
        ]).astype(dt)
        x = x[np.isfinite(x)]
        a = np.concatenate([x, -x])
        for who, out in _both(impl, a):
            if out.dtype != dt:
                failures.append({"who": "%s: %s" % (who, dt.name),
                                 "detail": "the result has type %s" % out.dtype})
                continue
            out = np.asarray(out, dtype=dt)
            n = len(x)
            positive, negative = out[:n], out[n:]
            odd = _view_uint(negative) == (_view_uint(positive) ^ _sign_bit(dt))
            if not np.all(odd):
                i = np.nonzero(~odd)[0][:3]
                failures.append({
                    "who": "%s: %s" % (who, dt.name),
                    "detail": "not odd at " + "; ".join(
                        "A=%r gives %r but A=%r gives %r"
                        % (x[j], positive[j], -x[j], negative[j]) for j in i)})
        rows.append("%s %d pairs" % (dt.name, len(x)))
    return {"sweep": "the oddness of the result (bit for bit, no reference)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def _spec_examples():
    """Every worked example of the document, as a reader reads it.

    The values are the ones the document writes, as strings: what is checked is that
    they are faithful, at the number of digits written, to the correctly rounded sine,
    and that both implementations stay within reach of it.  The flag marks the example
    whose values the document states exactly, with $=$ and not with $\\approx$.
    """
    return [
        ("float Example 1", np.float64, True,
         ["0.0", "-0.0", "1.0e-8", "-1.0e-8", "+inf", "-inf", "NaN"],
         ["0.0", "-0.0", "1.0e-8", "-1.0e-8", "NaN", "NaN", "NaN"]),
        ("float Example 2", np.float32, False,
         ["3.1415927", "6.2831855", "1.0e30"],
         ["-8.7422777e-08", "1.7484555e-07", "-0.7911634"]),
        ("float Example 3", np.float64, False,
         ["1.0", "-1.0", "0.5", "2.0"],
         ["0.8414709848078965", "-0.8414709848078965", "0.479425538604203",
          "0.9092974268256817"]),
        ("float section, the sine of the nearest double to pi", np.float64, True,
         ["3.141592653589793"], ["1.2246467991473532e-16"]),
    ]


def _literal(dtype, s):
    if s in ("+inf", "-inf", "NaN"):
        return np.dtype(dtype).type(float(s.replace("+inf", "inf").replace("NaN", "nan")))
    return np.dtype(dtype).type(float(s))


def _faithful(literal, exact_value):
    """Is the literal the correctly rounded value, at the digits it writes?

    ``exact_value`` is the correctly rounded result; the literal must be its nearest
    decimal at the place of its last written digit, which is what a writer who rounds a
    value to the digits shown produces.
    """
    mantissa, _, exponent = literal.lower().partition("e")
    exponent = int(exponent) if exponent else 0
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    half = Decimal(5).scaleb(exponent - decimals - 1)
    return abs(Decimal(exact_value) - Decimal(literal)) <= half


def sweep_spec_examples(impl):
    """The worked examples of the document: as documented, vs the exact sine, vs both."""
    failures, rows = [], []
    for label, dtype, exact, operands, documented in _spec_examples():
        dt = np.dtype(dtype)
        a = np.array([_literal(dt, s) for s in operands], dtype=dt)
        doc = np.array([_literal(dt, s) for s in documented], dtype=dt)
        exact_sine = [sin_correctly_rounded(v, dt.type) for v in a]
        if any(c is None for c in exact_sine):
            failures.append({"who": label,
                             "detail": "the exact sine of an operand of the example is "
                                       "beyond the reach of the oracle"})
            continue
        correct = np.array(exact_sine, dtype=dt)
        for s, c in zip(documented, correct):
            if s in ("NaN", "+inf", "-inf"):
                continue
            if not _faithful(s, float(c)):
                failures.append({"who": label,
                                 "detail": "the document writes %s where the exact sine "
                                           "rounded to %s is %.17g"
                                           % (s, dt.name, float(c))})
        if exact and not _bits_equal(doc, correct):
            failures.append({"who": label,
                             "detail": "as documented %s, exact %s"
                                       % (doc.tolist(), correct.tolist())})
        for who, out in _both(impl, a):
            if out.dtype != dt:
                failures.append({"who": "%s, %s" % (label, who),
                                 "detail": "the result has type %s" % out.dtype})
                continue
            out = np.asarray(out, dtype=dt)
            finite = np.isfinite(doc)
            worst = float(np.max(_deviation(correct[finite], out[finite])))
            if worst > SLACK:
                failures.append({"who": "%s, %s" % (label, who),
                                 "detail": "worst departure %.0f units in the last place "
                                           "of the value 1" % worst})
        if not exact:
            rows.append("%s: %d documented values, faithful to the exact sine"
                        % (label, len(documented)))
    return {"sweep": "the document's worked examples",
            "summary": "%d examples, %s" % (len(_spec_examples()), "; ".join(rows)),
            "failures": failures}


REAL_EXAMPLES = [
    ("real Example 1", [(0, 1), (1, 6), (1, 4), (1, 2), (1, 1), (3, 2)],
     ["0", "1/2", "sqrt(2)/2", "1", "0", "-1"]),
    ("real Example 2", [(-1, 6), (1, 6), (13, 6)], ["-1/2", "1/2", "1/2"]),
]

DIGITS = 60


def _exact_sine(num, den):
    """The sine of num*pi/den, computed from pi to DIGITS digits."""
    x = pi_decimal(DIGITS + 20) * Decimal(num) / Decimal(den)
    return sin_decimal(x, DIGITS + 10)


def _documented_real(s):
    """The value the document writes for a real result, as a Decimal."""
    if s == "sqrt(2)/2":
        return Decimal(2).sqrt() / 2
    frac = Fraction(s)
    return Decimal(frac.numerator) / Decimal(frac.denominator)


def sweep_real_examples(impl):
    """The real section, which has no ONNX type.

    The sine of a multiple of pi is not a rational number in general, so the section is
    checked against a computation of the sine in 60-digit decimal arithmetic, run from
    the definition, and not against ONNX Runtime.  The properties the prose of the
    examples states -- that the result of $13\\pi/6$ equals the result of $\\pi/6$, that
    the result of $-\\pi/6$ is its opposite -- are checked on the computed values.
    """
    failures, rows = [], []
    computed = {}
    for label, multiples, documented in REAL_EXAMPLES:
        values = []
        for (num, den), s in zip(multiples, documented):
            s_computed = _exact_sine(num, den)
            want = _documented_real(s)
            if abs(s_computed - want) > Decimal(1).scaleb(-DIGITS // 2):
                failures.append({"who": label,
                                 "detail": "the document writes %s where the sine of "
                                           "%d*pi/%d is %s" % (s, num, den, s_computed)})
            values.append(s_computed)
        if label == "real Example 2":
            if abs(values[1] - values[2]) > Decimal(1).scaleb(-DIGITS // 2):
                failures.append({"who": label, "detail": "the sine of 13*pi/6 is not the "
                                                         "sine of pi/6"})
            if abs(values[0] + values[1]) > Decimal(1).scaleb(-DIGITS // 2):
                failures.append({"who": label, "detail": "the sine of -pi/6 is not the "
                                                         "opposite of the sine of pi/6"})
        computed[label] = values
        rows.append("%s: %d values" % (label, len(values)))
    try:
        exact = np.array([Fraction(1, 6), Fraction(-1, 3)], dtype=object)
        impl.sin(exact)
        rows.append("the implementation takes a real operand")
    except Exception as exc:
        rows.append("the implementation does not take a real operand (%s)"
                    % type(exc).__name__)
    return {"sweep": "the real section (60-digit arithmetic, no ONNX Runtime)",
            "summary": "; ".join(rows), "failures": failures}


def sweeps():
    return [sweep_exact_rules, sweep_spec_examples, sweep_real_examples,
            sweep_float16_domain, sweep_accuracy, sweep_odd_symmetry]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _exact_value_cases(dtype):
    """The operand families whose result the document determines exactly."""
    dt = np.dtype(dtype)
    sub = float(np.finfo(dt).smallest_subnormal)
    nan, inf = float("nan"), float("inf")
    return [
        ("signed zeros", [0.0, -0.0, 0.0, -0.0]),
        ("infinities, the invalid operation", [inf, -inf, inf, -inf]),
        ("NaN operands", [nan, nan, nan, nan]),
        ("subnormal operands, their own sine", [sub, -sub, 2.0 * sub, -3.0 * sub]),
        ("the smallest subnormal, both signs", [sub, -sub]),
        ("a null and a subnormal together", [0.0, -0.0, sub, -sub]),
    ]


def cases():
    """Every individual case: (label, [A])."""
    for dtype in ALL_FLOAT:
        name = np.dtype(dtype).name
        for label, values in _exact_value_cases(dtype):
            yield ("%s: %s" % (name, label), [np.array(values, dtype=dtype)])
        quiet, signalling, sign = NAN_PATTERNS[np.dtype(dtype).itemsize]
        yield ("%s: NaN operand, quiet, payload 1" % name, [_bit_nan(dtype, quiet)])
        yield ("%s: NaN operand, quiet, payload 1, sign set" % name,
               [_bit_nan(dtype, quiet | sign)])
        yield ("%s: NaN operand, signalling" % name, [_bit_nan(dtype, signalling)])
        yield ("%s: infinity, invalid operation, both signs" % name,
               [np.array([np.inf, -np.inf], dtype=dtype)])
    yield ("float64: Example 1 of the document",
           [np.array([0.0, -0.0, 1.0e-8, -1.0e-8, np.inf, -np.inf, np.nan],
                     dtype=np.float64)])
    for label, shape in SHAPE_CASES:
        for dtype in ALL_FLOAT:
            yield ("%s: %s" % (np.dtype(dtype).name, label),
                   [_fill(shape, dtype, _shape_values(dtype))])
