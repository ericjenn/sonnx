#!/usr/bin/env python3
"""Compare impl_from_spec.add with ONNX Runtime's Add, case by case.

Each case is a pair of NumPy arrays.  A single-node ONNX model (one Add node,
opset 14, inputs A and B of the same element type, output C) is run with
onnxruntime, and its output is compared with the result of impl_from_spec.add
on the same arrays.

  - float results are compared as raw bit patterns (view on an unsigned integer
    of the same width), with NaN == NaN (a differing NaN payload is reported
    separately, not as a mismatch);
  - integer results are compared by value.

An ONNX Runtime refusal (unsupported type, ...) is caught and recorded as
"ORT rejected"; it is a result, not a failure.  The exit status is non-zero only
when a real mismatch is found.

Two independent checks of the integer sections are made as well, one per family
since the specification gives the signed and the unsigned types a section each:

  - the two case analyses of the document, **Add** (int, int) and **Add**
    (uint, uint), are transcribed literally, in exact Python integers, and
    compared with ONNX Runtime over the *whole* domain of `int8` and `uint8`
    (all 65536 pairs), and over the boundary values of the wider types;
  - the same comparison is made against the implementation of impl_from_spec.

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
# ONNX Runtime
# ---------------------------------------------------------------------------


def run_ort(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Build and run a one-node Add model for this pair of arrays."""
    proto = helper.np_dtype_to_tensor_dtype(a.dtype)
    vi_a = helper.make_tensor_value_info("A", proto, list(a.shape))
    vi_b = helper.make_tensor_value_info("B", proto, list(b.shape))
    vi_c = helper.make_tensor_value_info("C", proto, None)
    node = helper.make_node("Add", ["A", "B"], ["C"], name="add0")
    graph = helper.make_graph([node], "add_graph", [vi_a, vi_b], [vi_c])
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
# the integer case analysis of the document, transcribed literally
# ---------------------------------------------------------------------------


def doc_int_signed(x: int, y: int, n: int) -> int:
    """The case analysis of **Add** (int, int), in exact arithmetic.

    The document writes, for the element C[i] of the result of the addition
    of two values of the n-bit signed type of that section:

        the sum                    if it lies in [-2**(n-1), 2**(n-1) - 1],
        the sum + 2**n             if it is lower than -2**(n-1),
        the sum - 2**n             if it is greater than 2**(n-1) - 1.
    """
    lo, hi = -(2 ** (n - 1)), 2 ** (n - 1) - 1
    s = x + y
    if s > hi:
        return s - 2 ** n
    if s < lo:
        return s + 2 ** n
    return s


def doc_int_unsigned(x: int, y: int, n: int) -> int:
    """The case analysis of **Add** (uint, uint), in exact arithmetic.

    The document writes, for the element C[i] of the result of the addition
    of two values of the n-bit unsigned type of that section:

        the sum                    if it is at most 2**n - 1,
        the sum - 2**n             if it is greater than 2**n - 1.
    """
    s = x + y
    return s if s <= 2 ** n - 1 else s - 2 ** n


def doc_int_exact(x: int, y: int, n: int, signed: bool) -> int:
    """The case analysis of the section that covers the type in question."""
    return doc_int_signed(x, y, n) if signed else doc_int_unsigned(x, y, n)


def doc_int_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply doc_int_exact element-wise, and return an array of a's dtype."""
    n = np.dtype(a.dtype).itemsize * 8
    signed = a.dtype.kind == "i"
    flat = [doc_int_exact(int(x), int(y), n, signed)
            for x, y in zip(a.ravel(), b.ravel())]
    return np.array(flat, dtype=a.dtype).reshape(np.broadcast_shapes(a.shape, b.shape))


# ---------------------------------------------------------------------------
# a literal transcription of the document, against ONNX Runtime, over a domain
# ---------------------------------------------------------------------------


def _rand_value(rng, info, dtype):
    """One random value of this integer type, drawn over the whole domain.

    numpy cannot draw from [0, 2**64 - 1] in one call, so the 64-bit types are
    assembled from two 32-bit draws.
    """
    bits = np.dtype(dtype).itemsize * 8
    if info.min < 0:
        return int(rng.integers(info.min, info.max, dtype=np.int64, endpoint=True))
    if bits <= 32:
        return int(rng.integers(0, 2 ** bits, dtype=np.uint64))
    hi = int(rng.integers(0, 2 ** 32, dtype=np.uint64))
    lo = int(rng.integers(0, 2 ** 32, dtype=np.uint64))
    return (hi << 32) | lo


def check_int_domain(dtype, exhaustive: bool, rng):
    """Compare the document's integer case analysis with ONNX Runtime.

    ``exhaustive`` covers every pair of 8-bit values; otherwise the boundary
    values and a batch of random pairs are used.
    """
    dt = np.dtype(dtype)
    info = np.iinfo(dt)
    if exhaustive:
        xs = np.arange(info.min, info.max + 1, dtype=np.int64)
        a = np.repeat(xs, len(xs))
        b = np.tile(xs, len(xs))
    else:
        edges = sorted({min(max(int(v), int(info.min)), int(info.max)) for v in
                        (info.min, info.min + 1, info.max - 1, info.max, 0, 1, -1)})
        pairs = [(x, y) for x in edges for y in edges]
        pairs += [(_rand_value(rng, info, dtype), _rand_value(rng, info, dtype))
                  for _ in range(200)]
        # built directly in the type under test: the 64-bit unsigned values do
        # not fit an int64 intermediate
        a = np.array([p[0] for p in pairs], dtype=dt)
        b = np.array([p[1] for p in pairs], dtype=dt)
    aa, bb = a.astype(dt), b.astype(dt)

    want = doc_int_array(aa, bb)
    got = run_ort(aa, bb)
    ok_ort = bool(np.array_equal(want, got))

    impl = spec.add(aa, bb)
    ok_impl = compare(want, impl)[0]

    if not ok_ort:
        bad = np.nonzero(want.astype(np.int64) != np.asarray(got).astype(np.int64))[0][:5]
        detail = "; ".join("doc %d + %d = %d, ort %d" % (a[i], b[i], want.ravel()[i], got.ravel()[i])
                           for i in bad)
    else:
        detail = ""
    return ("%s (%s)" % (dt.name, "exhaustive" if exhaustive else "%d pairs" % len(a)),
            "PASS" if ok_ort else "MISMATCH", "PASS" if ok_impl else "MISMATCH", detail)


# ---------------------------------------------------------------------------
# the document's own examples
# ---------------------------------------------------------------------------


def check_real_examples():
    """The worked examples of the real section, as exact rationals.

    The type of the real section is not one of the types of ONNX Add, so the
    examples are the part of the document ONNX Runtime cannot be asked about.
    They are checked against exact rational arithmetic instead.
    """
    from fractions import Fraction as F

    examples = [
        ("real Example 1", [[F("6.1"), F("9.5"), F("35.7")]], [F(2), F(3), F(4)],
         [[F("8.1"), F("12.5"), F("39.7")]]),
        ("real Example 2",
         [[F("3.7"), F("4.4")], [F("16.2"), F("0.5")], [F("25.3"), F("24.8")]],
         [F(1), F(2)],
         [[F("4.7"), F("6.4")], [F("17.2"), F("2.5")], [F("26.3"), F("26.8")]]),
    ]
    rows = []
    for name, a, b, c in examples:
        aa, bb, cc = np.array(a, dtype=object), np.array(b, dtype=object), np.array(c, dtype=object)
        got = spec.add(aa, bb)
        same = got.shape == cc.shape and all(got[i] == cc[i] for i in np.ndindex(cc.shape))
        rows.append((name, str(aa.shape), str(bb.shape), "object",
                     "PASS" if same else "MISMATCH", "exact rationals, no ONNX Runtime"))
        if not same:
            rows.append(("", "", "", "", "", "spec = %r" % (got,)))
    return rows


def check_spec_examples():
    """The worked examples of the float and int sections, against ONNX Runtime."""
    rows = []
    cases = []

    f32 = np.float32
    cases.append(("float Example 1",
                  np.array([[3.0, 4.5], [16.0, 1.0], [25.5, 24.25]], dtype=f32),
                  np.array([[3.0, 2.0], [4.0, 0.0], [5.0, 4.0]], dtype=f32),
                  np.array([[6.0, 6.5], [20.0, 1.0], [30.5, 28.25]], dtype=f32)))
    cases.append(("float Example 2",
                  np.array([1.0, np.inf, 0.0, -np.inf], dtype=f32),
                  np.array([1.0, -np.inf, -0.0, -np.inf], dtype=f32),
                  np.array([2.0, np.nan, 0.0, -np.inf], dtype=f32)))
    cases.append(("float Example 3",
                  np.array([0.1, 1.0, 1.0], dtype=np.float64),
                  np.array([0.2, 2.0, -1.0], dtype=np.float64),
                  np.array([0.1 + 0.2, 3.0, 0.0], dtype=np.float64)))
    cases.append(("int Example 1",
                  np.array([6, 200, 35], dtype=np.uint8),
                  np.array([3, 100, 5], dtype=np.uint8),
                  np.array([9, 44, 40], dtype=np.uint8)))
    cases.append(("int Example 2",
                  np.array([-6, 100, -100], dtype=np.int8),
                  np.array([-3, 100, -100], dtype=np.int8),
                  np.array([-9, -56, 56], dtype=np.int8)))

    for name, a, b, documented in cases:
        got = run_ort(a, b)
        impl = spec.add(a, b)
        ok = compare(documented, got)[0] and compare(documented, impl)[0]
        rows.append((name, str(a.shape), str(b.shape), a.dtype.name,
                     "PASS" if ok else "MISMATCH",
                     "as documented" if ok else "documented %s, ort %s" % (bits(documented), bits(got))))
    return rows


# ---------------------------------------------------------------------------
# the value and shape families
# ---------------------------------------------------------------------------


def _ordinary_value_cases(dtype):
    """Ordinary values, the extremes and the wrapping pairs."""
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


def _float_value_cases(dtype):
    """The value families of the float section."""
    f = np.finfo(np.dtype(dtype))
    maxf = float(f.max)
    tiny = float(f.tiny)  # smallest normal; tiny/2 is subnormal
    # the spacing of the type at its largest finite value max: the exact sums
    # in (max, max + ulp/2) round back to max, the sum exactly at the midpoint
    # max + ulp/2 is a tie that breaks upward, and any larger sum rounds to a
    # value beyond max, which with an unbounded exponent range is the overflow
    # case of the standard
    ulp = float(2.0 ** (f.maxexp - f.nmant - 1))
    nan, inf = float("nan"), float("inf")
    return [
        ("signed zeros", [0.0, -0.0, 0.0, -0.0], [0.0, 0.0, -0.0, -0.0]),
        ("infinities", [inf, -inf, inf, -inf, 1.0, -1.0], [inf, -inf, 1.0, -1.0, inf, -inf]),
        ("nan", [nan, 1.0, nan, inf], [1.0, nan, inf, nan]),
        ("exact sum", [1.0, 4.5, 16.0, 0.25], [0.5, 0.25, 8.0, 0.125]),
        ("overflow to inf", [maxf, -maxf], [maxf, -maxf]),
        ("cancellation to zero", [maxf, maxf], [-maxf, -maxf]),
        ("below the overflow threshold", [maxf], [ulp / 4.0]),
        ("at the overflow threshold (tie)", [maxf], [ulp / 2.0]),
        ("above the overflow threshold", [maxf], [3.0 * ulp / 4.0]),
        ("subnormal result", [tiny, 0.0, 0.0], [tiny / 2.0, tiny / 2.0, -tiny / 2.0]),
        ("exactly one ulp from 1.0", [1.0], [2.0 ** -f.nmant]),
        ("rounding tie (halfway)", [1.0], [2.0 ** -(f.nmant + 1.0)]),
        ("not representable (rounded sum)", [0.3, 1.0], [0.1, 1.0]),
    ]


def _arr(shape, dtype, start=0.0):
    n = 1
    for d in shape:
        n *= d
    return (np.arange(n, dtype=np.float64) + start).astype(dtype).reshape(shape)


SHAPE_CASES = [
    ("same shape (2,3)", (2, 3), (2, 3), ALL_DTYPES),
    ("broadcast dim 1: (2,3)+(1,3)", (2, 3), (1, 3), ALL_DTYPES),
    ("smaller rank, no leading dims: (2,3,4)+(4,)", (2, 3, 4), (4,), ALL_DTYPES),
    ("broadcast dim 1: (2,3,4)+(3,1)", (2, 3, 4), (3, 1), ALL_DTYPES),
    ("rank-0 on the left: ()+(2,3)", (), (2, 3), ALL_DTYPES),
    ("rank-0 on the right: (2,3)+()", (2, 3), (), ALL_DTYPES),
    ("both rank-0: ()+()", (), (), ALL_DTYPES),
    ("zero-sized: (0,3)+(1,3)", (0, 3), (1, 3), ALL_DTYPES),
    ("zero-sized: (0,)+(1,)", (0,), (1,), ALL_DTYPES),
    ("zero-sized: (0,3)+(3,)", (0, 3), (3,), ALL_DTYPES),
    ("zero-sized: (1,0)+(2,0)", (1, 0), (2, 0), ALL_DTYPES),
    ("zero-sized: (0,3)+(0,1)", (0, 3), (0, 1), ALL_DTYPES),
    ("zero-sized with rank-0: (0,)+()", (0,), (), ALL_DTYPES),
]


def _bit_nan(dtype, pattern):
    """One NaN of this type carrying the given bit pattern."""
    uint = _UINT_OF_WIDTH[np.dtype(dtype).itemsize]
    return np.array([pattern], dtype=uint).view(dtype)


NAN_PATTERNS = {2: (0x7E01, 0x7C01, 0x8000), 4: (0x7FC00001, 0x7F800001, 0x80000000),
                8: (0x7FF8000000000001, 0x7FF0000000000001, 0x8000000000000000)}


def build_cases():
    cases = []
    for dtype in ALL_FLOAT:
        for name, a, b in _float_value_cases(dtype):
            cases.append((name, np.array(a, dtype=dtype), np.array(b, dtype=dtype), False))
        quiet, signalling, sign = NAN_PATTERNS[np.dtype(dtype).itemsize]
        one = np.ones(1, dtype=dtype)
        cases.append(("NaN operand, quiet, payload 1", _bit_nan(dtype, quiet), one, False))
        cases.append(("NaN operand, quiet, payload 1, sign set",
                      _bit_nan(dtype, quiet | sign), one, False))
        cases.append(("NaN operand, signalling", _bit_nan(dtype, signalling), one, False))
        inf = np.array([np.inf], dtype=dtype)
        ninf = np.array([-np.inf], dtype=dtype)
        cases.append(("invalid operation: inf + (-inf)", inf, ninf, False))
        cases.append(("invalid operation: -inf + inf", ninf, inf, False))
    for dtype in ALL_INT:
        for name, a, b in _ordinary_value_cases(dtype):
            cases.append((name, np.array(a, dtype=dtype), np.array(b, dtype=dtype), False))
    for name, sa, sb, dtypes in SHAPE_CASES:
        for dtype in dtypes:
            cases.append((name, _arr(sa, dtype), _arr(sb, dtype, 1.0), False))
    return cases


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    markdown = "--markdown" in sys.argv
    rows = []
    n_pass = n_mismatch = n_rejected = 0
    divergences = []

    for name, a, b, is_real in build_cases():
        label = name
        dt = a.dtype.name
        try:
            got = run_ort(a, b)
        except Exception as exc:  # a refusal is a result
            n_rejected += 1
            msg = str(exc).strip().splitlines()[0]
            rows.append((label, str(a.shape), str(b.shape), dt, "ORT rejected", msg))
            continue

        want = spec.add(a, b)
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
    header = ("case", "shape A", "shape B", "dtype", "spec vs ORT", "note")

    def write_table(rows):
        width = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
        out.write("  ".join(h.ljust(width[i]) for i, h in enumerate(header)) + "\n")
        out.write("  ".join("-" * x for x in width) + "\n")
        for r in rows:
            out.write("  ".join(str(v).ljust(width[i]) for i, v in enumerate(r)) + "\n")

    real_rows = check_real_examples()
    n_real_fail = sum(1 for r in real_rows if r[4] == "MISMATCH")
    out.write("real section, exact arithmetic (ONNX Runtime has no real type)\n\n")
    write_table(real_rows)

    out.write("\nthe worked examples of the document, against ONNX Runtime\n\n")
    example_rows = check_spec_examples()
    n_example_fail = sum(1 for r in example_rows if r[4] == "MISMATCH")
    write_table(example_rows)

    out.write("\nthe integer case analysis of the document, in exact arithmetic\n\n")
    domain_header = ("domain", "document vs ORT", "implementation vs document", "detail")
    rng = np.random.default_rng(20261003)
    domain_rows = [
        check_int_domain(np.int8, True, rng),
        check_int_domain(np.uint8, True, rng),
        check_int_domain(np.int16, False, rng),
        check_int_domain(np.int32, False, rng),
        check_int_domain(np.int64, False, rng),
        check_int_domain(np.uint16, False, rng),
        check_int_domain(np.uint32, False, rng),
        check_int_domain(np.uint64, False, rng),
    ]
    w = [max(len(str(r[i])) for r in domain_rows + [domain_header]) for i in range(4)]
    out.write("  ".join(h.ljust(w[i]) for i, h in enumerate(domain_header)) + "\n")
    out.write("  ".join("-" * x for x in w) + "\n")
    for r in domain_rows:
        out.write("  ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + "\n")
    n_domain_fail = sum(1 for r in domain_rows if r[1] == "MISMATCH" or r[2] == "MISMATCH")

    out.write("\nfloat and integer value and shape families, against ONNX Runtime\n\n")
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
        f"\nreal examples: {len(real_rows)} cases, {len(real_rows) - n_real_fail} exact, "
        f"{n_real_fail} wrong\n"
    )
    out.write(
        f"worked examples: {len(example_rows)} cases, {len(example_rows) - n_example_fail} as "
        f"documented, {n_example_fail} wrong\n"
    )
    out.write(
        f"integer domain: {len(domain_rows)} domains, {len(domain_rows) - n_domain_fail} agree\n"
    )
    out.write(
        f"family cases: {total} run, {n_pass} passed, {n_mismatch} mismatched, "
        f"{n_rejected} refused by ONNX Runtime\n"
    )
    return 1 if (n_mismatch or n_real_fail or n_example_fail or n_domain_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
