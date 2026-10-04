#!/usr/bin/env python3
"""Compare impl_from_spec.sub with ONNX Runtime's Sub, case by case.

Each case is a pair of NumPy arrays.  A single-node ONNX model (one Sub node,
opset 14, inputs A and B of the same element type, output C) is run with
onnxruntime, and its output is compared bit-exactly with the result of
impl_from_spec.sub on the same arrays.

  - float results are compared as raw bit patterns (view on an unsigned integer
    of the same width), with NaN == NaN (a differing NaN payload is reported
    separately, not as a mismatch);
  - integer results are compared by value.

An ONNX Runtime refusal (unsupported type, zero-sized dimension, rank-0, ...) is
caught and recorded as "ORT rejected"; it is a result, not a failure.  The exit
status is non-zero only when a real mismatch is found.

Usage:
    python test_vs_ort.py              compact table
    python test_vs_ort.py --markdown   the same table as markdown
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import impl_from_spec as spec  # noqa: E402

import onnx  # noqa: E402
from onnx import helper  # noqa: E402
import onnxruntime as ort  # noqa: E402


ALL_FLOAT = [np.float16, np.float32, np.float64]
ALL_INT = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
ALL_DTYPES = ALL_FLOAT + ALL_INT

_UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def _ordinary_value_cases(dtype):
    """Ordinary values, plus the extremes and the wrapping pairs."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return [
            ("ordinary", [1.0, 2.5, -3.0, 4.75], [0.5, 2.5, 1.0, -0.25]),
        ]
    info = np.iinfo(dtype)
    lo, hi = int(info.min), int(info.max)
    if lo < 0:  # signed: include min, max, -1, 0, 1 and both wrap directions
        a = [lo, hi, -1, 0, 1, hi, lo, -1]
        b = [0, 0, 1, 1, 0, -1, 1, -1]
    else:  # unsigned: include 0, 1, max and the wrapping pairs
        a = [0, hi, 1, 4, hi, 0, 1, hi // 2]
        b = [1, 1, hi, 7, hi, 0, hi, 3]
    return [("ordinary+extremes+wraps", a, b)]


def _float_value_cases(dtype):
    """The value families of the float section."""
    f = np.finfo(np.dtype(dtype))
    maxf = float(f.max)
    tiny = float(f.tiny)  # smallest normal; tiny/2 is subnormal
    # half an ulp of maxf, which is the width of the band just above maxf in
    # which IEEE 754 rounds to maxf instead of overflowing to infinity
    quarter_ulp = float(2.0 ** (f.maxexp - 3 - f.nmant))
    half_ulp = float(2.0 ** (f.maxexp - f.nmant - 1))
    nan, inf = float("nan"), float("inf")
    return [
        ("signed zeros", [0.0, -0.0, 0.0, -0.0], [0.0, 0.0, -0.0, -0.0]),
        ("infinities", [inf, -inf, inf, -inf, 1.0, -1.0], [inf, inf, 1.0, -1.0, -inf, inf]),
        ("nan", [nan, 1.0, nan, inf], [1.0, nan, inf, nan]),
        ("exact difference", [1.0, 4.5, 16.0, 0.25], [0.5, 0.25, 8.0, 0.125]),
        ("overflow to inf", [maxf, -maxf], [-maxf, maxf]),
        ("above max, below overflow threshold", [maxf], [-quarter_ulp]),
        ("subnormal result", [tiny, 0.0, 0.0], [tiny / 2.0, -tiny / 2.0, tiny / 2.0]),
        # exactly halfway between 1.0 and 1.0 + ulp(1.0): the specification says
        # "the nearest representable value ... according to the IEEE 754 rounding
        # rules" without naming a tie rule; both here round to even (1.0)
        ("rounding tie (halfway)", [1.0 + 2.0 ** -f.nmant, 1.0], [2.0 ** -(f.nmant + 1), 0.0]),
        ("not representable (0.3-0.1)", [0.3, 1.0], [0.1, 1.0]),
        # The overflow threshold: half_ulp is half the spacing of the type at its
        # largest finite value, so maxf - b is exactly a quarter of a ulp below the
        # midpoint between maxf and the next power of two, then the midpoint itself
        # (a tie, which roundTiesToEven breaks upwards so that the result overflows),
        # then a quarter of a ulp above it.  V1 of the amendments turns on these three.
        ("below the overflow threshold", [maxf], [-(half_ulp / 2.0)]),
        ("at the overflow threshold (tie)", [maxf], [-(half_ulp)]),
        ("above the overflow threshold", [maxf], [-(1.5 * half_ulp)]),
    ]


def _arr(shape, dtype, start=0.0):
    n = 1
    for d in shape:
        n *= d
    return (np.arange(n, dtype=np.float64) + start).astype(dtype).reshape(shape)


# (name, shape of A, shape of B, dtypes it is run with)
SHAPE_CASES = [
    ("same shape (2,3)", (2, 3), (2, 3), ALL_DTYPES),
    ("broadcast dim 1: (2,3)-(1,3)", (2, 3), (1, 3), ALL_DTYPES),
    ("smaller rank, no leading dims: (2,3,4)-(4,)", (2, 3, 4), (4,), ALL_DTYPES),
    ("broadcast dim 1: (2,3,4)-(3,1)", (2, 3, 4), (3, 1), ALL_DTYPES),
    ("rank-0 on the left: ()-(2,3)", (), (2, 3), ALL_DTYPES),
    ("rank-0 on the right: (2,3)-()", (2, 3), (), ALL_DTYPES),
    ("both rank-0: ()-()", (), (), ALL_DTYPES),
    ("zero-sized: (0,3)-(1,3)", (0, 3), (1, 3), ALL_DTYPES),
    ("zero-sized: (0,)-(1,)", (0,), (1,), ALL_DTYPES),
    ("zero-sized: (0,3)-(3,)", (0, 3), (3,), ALL_DTYPES),
    ("zero-sized: (1,0)-(2,0)", (1, 0), (2, 0), ALL_DTYPES),
    ("zero-sized: (0,3)-(0,1)", (0, 3), (0, 1), ALL_DTYPES),
    ("zero-sized with rank-0: (0,)-()", (0,), (), ALL_DTYPES),
]


def _bit_nan(dtype, pattern):
    """One NaN of this type carrying the given bit pattern."""
    uint = _UINT_OF_WIDTH[np.dtype(dtype).itemsize]
    return np.array([pattern], dtype=uint).view(dtype)


# the quiet and signalling NaN patterns, and the sign bit, of each float type
NAN_PATTERNS = {2: (0x7E01, 0x7C01, 0x8000), 4: (0x7FC00001, 0x7F800001, 0x80000000),
                8: (0x7FF8000000000001, 0x7FF0000000000001, 0x8000000000000000)}


def build_cases():
    cases = []
    for dtype in ALL_FLOAT:
        for name, a, b in _float_value_cases(dtype):
            cases.append((name, np.array(a, dtype=dtype), np.array(b, dtype=dtype), False))
        # V2 of the amendments: which NaN is the result.  A NaN operand is propagated,
        # and inf - inf is the invalid operation; the payload and the sign of the NaN
        # that comes back are not fixed by the specification, so compare() accepts any
        # quiet NaN and reports a payload difference rather than a mismatch.
        quiet, signalling, sign = NAN_PATTERNS[np.dtype(dtype).itemsize]
        one = np.ones(1, dtype=dtype)
        cases.append(("NaN operand, quiet, payload 1", _bit_nan(dtype, quiet), one, False))
        cases.append(("NaN operand, quiet, payload 1, sign set",
                      _bit_nan(dtype, quiet | sign), one, False))
        cases.append(("NaN operand, signalling", _bit_nan(dtype, signalling), one, False))
        inf = np.array([np.inf], dtype=dtype)
        ninf = np.array([-np.inf], dtype=dtype)
        cases.append(("invalid operation: inf - inf", inf, inf, False))
        cases.append(("invalid operation: -inf - (-inf)", ninf, ninf, False))
    for dtype in ALL_INT:
        for name, a, b in _ordinary_value_cases(dtype):
            cases.append((name, np.array(a, dtype=dtype), np.array(b, dtype=dtype), False))
    # float64 stands in for the real section: the real examples of ops/sub.md
    cases.append((
        "real section (float64 stand-in)",
        np.array([6.1, 9.5, 35.7], dtype=np.float64),
        np.array([3.0, 3.3, 5.1], dtype=np.float64),
        True,
    ))
    cases.append((
        "real section, rank-0 (float64 stand-in)",
        np.array(2.5, dtype=np.float64),
        np.array(1.75, dtype=np.float64),
        True,
    ))
    # the real numbers have no overflow: this exact real difference is not
    # representable in float64, so the stand-in rounds it, as does ONNX Runtime
    cases.append((
        "real section, above max below overflow threshold (float64 stand-in)",
        np.array([float(np.finfo(np.float64).max)], dtype=np.float64),
        np.array([-(2.0 ** 969)], dtype=np.float64),
        True,
    ))
    for name, sa, sb, dtypes in SHAPE_CASES:
        for dtype in dtypes:
            cases.append((name, _arr(sa, dtype), _arr(sb, dtype, 1.0), False))
    return cases


# ---------------------------------------------------------------------------
# the real section, checked in exact arithmetic and without ONNX Runtime
# ---------------------------------------------------------------------------


def check_real_examples():
    """The three worked examples of the real section, as exact rationals.

    The types of the real section are not among the types of ONNX Sub, so the
    examples are the part of the document ONNX Runtime cannot be asked about.
    They are checked against exact rational arithmetic instead: the document
    writes them as decimal numbers, and it claims the subtraction is exact.
    """
    from fractions import Fraction as F

    examples = [
        ("real Example 1", [[F("6.1"), F("9.5"), F("35.7")]],
         [F(3), F("3.3"), F("5.1")], [[F("3.1"), F("6.2"), F("30.6")]]),
        ("real Example 2",
         [[F("3.7"), F("4.4")], [F("16.2"), F("0.5")], [F("25.3"), F("24.8")]],
         [F(1), F(2)],
         [[F("2.7"), F("2.4")], [F("15.2"), F("-1.5")], [F("24.3"), F("22.8")]]),
        ("real Example 3", [[F(5), F("3.25")], [F(4), F("1.75")]], F("2.5"),
         [[F("2.5"), F("0.75")], [F("1.5"), F("-0.75")]]),
    ]
    rows = []
    for name, a, b, c in examples:
        aa, bb, cc = np.array(a, dtype=object), np.array(b, dtype=object), np.array(c, dtype=object)
        got = spec.sub(aa, bb)
        same = got.shape == cc.shape and all(got[i] == cc[i] for i in np.ndindex(cc.shape))
        rows.append((name, str(aa.shape), str(bb.shape), "object",
                     "PASS" if same else "MISMATCH", "exact rationals, no ONNX Runtime"))
        if not same:
            rows.append(("", "", "", "", "", "spec = %r" % (got,)))
    return rows


# ---------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------


def run_ort(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Build and run a one-node Sub model for this pair of arrays."""
    proto = helper.np_dtype_to_tensor_dtype(a.dtype)
    vi_a = helper.make_tensor_value_info("A", proto, list(a.shape))
    vi_b = helper.make_tensor_value_info("B", proto, list(b.shape))
    vi_c = helper.make_tensor_value_info("C", proto, None)
    node = helper.make_node("Sub", ["A", "B"], ["C"], name="sub0")
    graph = helper.make_graph([node], "sub_graph", [vi_a, vi_b], [vi_c])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=10
    )
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (c,) = sess.run(["C"], {"A": a, "B": b})
    return c


# ---------------------------------------------------------------------------
# bit-exact comparison
# ---------------------------------------------------------------------------


def compare(expected: np.ndarray, actual: np.ndarray):
    """Return (equal, nan_payload_differ) for two arrays of the same dtype."""
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False, False
    if expected.dtype.kind == "f":
        uint = _UINT_OF_WIDTH[expected.dtype.itemsize]
        be = np.ascontiguousarray(expected).view(uint)
        ba = np.ascontiguousarray(actual).view(uint)
        both_nan = np.isnan(expected) & np.isnan(actual)
        payload = bool(np.any(both_nan & (be != ba)))
        equal = (be == ba) | both_nan
        return bool(np.all(equal)), payload
    return bool(np.all(expected == actual)), False


def bits(arr: np.ndarray) -> str:
    """Bit pattern of an array, for the report."""
    if arr.dtype.kind == "f":
        return str(np.ascontiguousarray(arr).view(_UINT_OF_WIDTH[arr.dtype.itemsize]))
    return str(arr)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    markdown = "--markdown" in sys.argv
    rows = []
    n_pass = n_mismatch = n_rejected = 0
    divergences = []

    for name, a, b, is_real in build_cases():
        label = name + (" [real]" if is_real else "")
        dt = a.dtype.name
        try:
            got = run_ort(a, b)
        except Exception as exc:  # a refusal is a result
            n_rejected += 1
            msg = str(exc).strip().splitlines()[0]
            rows.append((label, str(a.shape), str(b.shape), dt, "ORT rejected", msg))
            continue

        # The real section is not one of the types of ONNX Sub, so its cases are run
        # with float64 operands and the float64 result stands for the real result.
        # sub() sends float64 to the float section, whose subtraction is the correctly
        # rounded exact difference, which is that same arithmetic.
        want = spec.sub(a, b)
        equal, payload = compare(want, got)
        if equal:
            n_pass += 1
            note = "NaN payload differs" if payload else ""
            rows.append((label, str(a.shape), str(b.shape), dt, "PASS", note))
            if payload:
                divergences.append(
                    ("NaN payload difference (not a mismatch)", label, a, b, want, got)
                )
        else:
            n_mismatch += 1
            rows.append((label, str(a.shape), str(b.shape), dt, "MISMATCH", ""))
            divergences.append(("result mismatch", label, a, b, want, got))

    out = sys.stdout
    header = ("case", "shape A", "shape B", "dtype", "verdict", "note")

    def write_table(rows):
        if markdown:
            out.write("| " + " | ".join(header) + " |\n")
            out.write("| " + " | ".join("---" for _ in header) + " |\n")
            for r in rows:
                out.write("| " + " | ".join(str(x) if x else "" for x in r) + " |\n")
        else:
            w = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
            out.write("  ".join(h.ljust(w[i]) for i, h in enumerate(header)) + "\n")
            out.write("  ".join("-" * x for x in w) + "\n")
            for r in rows:
                out.write("  ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + "\n")

    real_rows = check_real_examples()
    n_real_fail = sum(1 for r in real_rows if r[4] == "MISMATCH")
    out.write("real section, exact arithmetic (ONNX Runtime has no real type)\n\n")
    write_table(real_rows)
    out.write("\nfloat and integer sections against ONNX Runtime\n\n")
    write_table(rows)

    for kind, label, a, b, want, got in divergences:
        out.write("\n" + "=" * 70 + "\n")
        out.write(f"{kind}: {label}\n")
        out.write(f"A ({a.dtype}) = {a!r}\n")
        out.write(f"B ({b.dtype}) = {b!r}\n")
        out.write(f"spec  = {want!r}\n")
        out.write(f"ORT   = {got!r}\n")
        if a.dtype.kind == "f":
            out.write(f"spec bits = {bits(want)}\n")
            out.write(f"ORT  bits = {bits(got)}\n")

    total = n_pass + n_mismatch + n_rejected
    out.write(
        f"\nreal section: {len(real_rows)} examples, {len(real_rows) - n_real_fail} exact, "
        f"{n_real_fail} wrong\n"
    )
    out.write(
        f"summary: {total} cases run, {n_pass} passed, "
        f"{n_mismatch} mismatched, {n_rejected} refused by ONNX Runtime\n"
    )
    return 1 if n_mismatch or n_real_fail else 0


if __name__ == "__main__":
    sys.exit(main())
