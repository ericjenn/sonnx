"""Reference implementation of the ONNX Sub operator, written from ops/sub.md alone.

The only document consulted for the semantics is ``ops/sub.md`` (revision
2026-10-03, based on ONNX opset 14).  No ONNX documentation, no ONNX Runtime
source and no other file of this repository was read for semantics.

The specification organises the operator in three sections:

``real``
    A type whose values are the mathematical real numbers.  NumPy has no such
    type.  The only way to exercise this section is to use the widest hardware
    float available, so "real" is implemented here as ``float64``: the operands
    are float64 values and their difference is computed in float64.  A float64
    stand-in cannot show anything the float section does not already show,
    except where the two sections differ, namely where the exact difference is
    outside the range of float64.

``float``
    float16 / float / double, IEEE 754 subtraction.

``int``
    the eight integer types int8..uint64, subtraction modulo 2**n.

The entry point for the ONNX operator is :func:`sub`; the real section, which
has no ONNX type, is :func:`sub_real`.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# real section
# ---------------------------------------------------------------------------


def sub_real(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sub for the mathematical real numbers, implemented as float64.

    ops/sub.md, real section: "The subtraction is exact: C[i] is the difference
    in R of the two elements, with no approximation."  The same text gives the
    broadcasting rules, the rank-0 rule and the zero-sized-dimension rule, which
    are the NumPy rules, so ``np.subtract`` with float64 operands performs the
    whole definition: broadcast, then the exact difference of the two values.

    This is the "real" section, which is not one of the types of ONNX Sub; it is
    checked by feeding float64 values and treating the float64 result as the
    representation of the real result.  ``sub`` follows the float section for
    float64 operands (the float section lists double among its types); the two
    differ only where the exact real difference is not representable in float64.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype != np.float64 or b.dtype != np.float64:
        raise TypeError("the real section is checked with float64 operands")
    return np.subtract(a, b, dtype=np.float64)


# ---------------------------------------------------------------------------
# float section
# ---------------------------------------------------------------------------


def _sub_float(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sub for float16 / float32 / float64, following E_SUB_FLOAT_FUNC_0010.

    The four cases of the definition are evaluated explicitly:

    1. NaN if either operand is NaN, or if the operands are both +inf or both
       -inf (the invalid operation of IEEE 754 section 7.2);
    2. the exact difference, when it is representable in the type of C;
    3. +-inf, when the exact difference overflows the range of the type;
    4. otherwise the exact difference rounded to the nearest representable
       value of the type of C.

    plus, for a null exact difference, "+0, except when A is -0 and B is +0, in
    which case it is -0".

    The exact difference is carried as a pair of float64 values, ``s + e``,
    produced by an error-free transformation: ``s`` is the value of the
    difference rounded to float64 and ``e`` the rounding error, and the identity
    ``s + e == a - b`` is exact for the real numbers.  This represents the exact
    difference whatever the type of the operands, so the three cases are decided
    on the exact difference itself; and the cast of ``s`` to the type of C is the
    correct rounding of that difference, because float64 is wide enough for the
    double rounding float64 -> float32/float16 to be innocuous (Figueroa's
    condition q >= 2p + 2 holds: 53 >= 50 for float32 and for float16).
    """
    dtype = a.dtype
    if b.dtype != dtype:
        raise TypeError("A, B and C have the same type (E_SUB_FLOAT_CONSTR_A_0020)")
    a, b = np.broadcast_arrays(a, b)

    a_inf = np.isinf(a)
    b_inf = np.isinf(b)

    # case 1: NaN operand, or two infinities of the same sign (invalid op)
    nan_mask = (
        np.isnan(a)
        | np.isnan(b)
        | (a_inf & b_inf & (np.signbit(a) == np.signbit(b)))
    )

    # exact difference of the two operands, as the float64 pair (s, e) with
    # s + e == a - b exactly (Ogita-Rump-Oishi branch-free TwoSum, with y = -b)
    with np.errstate(over="ignore", invalid="ignore"):
        x = a.astype(np.float64)
        y = -b.astype(np.float64)
        s = x + y                # the difference rounded to float64
        yv = s - x
        xv = s - yv
        e = (y - yv) + (x - xv)  # the rounding error: s + e is the exact difference

        # case 3: the exact difference overflows the range of the type of C,
        # i.e. it is outside [-max, +max].  Since s + e is the exact difference,
        # this is exactly: |s| > max, or |s| == max with a non-null error of the
        # sign of s (which is what happens just above +max).
        max_finite = np.float64(np.finfo(dtype).max)
        overflow = (np.abs(s) > max_finite) | (
            (np.abs(s) == max_finite)
            & (e != np.float64(0.0))
            & (np.signbit(e) == np.signbit(s))
        )

        # cases 2 and 4: the exact difference when it is representable, otherwise
        # that difference rounded to the nearest value of the type of C (NumPy's
        # cast is round-half-to-even, the IEEE 754 default rounding).
        quiet = np.where(np.isfinite(s) & ~overflow, s, np.float64(0.0))
        out = quiet.astype(dtype)
    # case 3 result: +-inf, with the sign of the exact difference
    out[overflow] = np.copysign(np.array(np.inf, dtype=dtype), s[overflow])
    # case 1 result
    out[nan_mask] = np.array(np.nan, dtype=dtype)
    return out


# ---------------------------------------------------------------------------
# int section
# ---------------------------------------------------------------------------


def _sub_int(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sub for the eight integer types, following E_SUB_INT_FUNC_0010.

    "The result is thus the exact difference modulo 2**n": the difference is
    computed exactly (in Python integers, so that no intermediate width can
    overflow), then the residue class is represented by the value that lies in
    the range of the type, exactly as the three cases of the definition say.
    """
    dtype = a.dtype
    if b.dtype != dtype:
        raise TypeError("A, B and C have the same type (E_SUB_INT_CONSTR_A_0020)")
    a, b = np.broadcast_arrays(a, b)

    info = np.iinfo(dtype)
    bits = info.bits
    lo = int(info.min)  # for unsigned types this is 0
    hi = int(info.max)
    modulus = 1 << bits

    # exact mathematical difference of the operands, as Python integers
    diff = np.frompyfunc(lambda x, y: int(x) - int(y), 2, 1)(a, b)

    # the three cases of the definition.  Only one of the two wrap cases can
    # apply: the difference of two n-bit values lies within 2**n - 1 of the
    # range of the type.
    def wrap(d):
        d = int(d)
        if d < lo:  # lower than the minimum value of the type
            return d + modulus
        if d > hi:  # greater than the maximum value of the type
            return d - modulus
        return d  # representable in the type of C

    diff = np.frompyfunc(wrap, 1, 1)(diff)
    # asarray: frompyfunc returns a scalar for a rank-0 operand
    return np.asarray(diff, dtype=dtype)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def sub(a, b) -> np.ndarray:
    """Element-wise subtraction C = A - B, dtype-preserving.

    Dispatches on the element type to the section of ops/sub.md that specifies
    it.  The type of C is the type of A and B (E_SUB_*_CONSTR_*_0020).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype != b.dtype:
        raise TypeError("A, B and C have the same type")
    kind = a.dtype.kind
    if kind == "f" and a.dtype in (np.float16, np.float32, np.float64):
        return _sub_float(a, b)
    if kind in ("i", "u"):
        return _sub_int(a, b)
    raise TypeError(f"type {a.dtype} is not one of the types of the specification")
