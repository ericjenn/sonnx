"""An implementation of the ONNX ``Sub`` operator as specified in ``ops/sub.md``.

Written from that document alone (revision 2026-10-03, based on ONNX opset 14).
It implements the three sections of the document:

  * **Sub** (real, real)                          -- E_SUB_REAL_FUNC_0010
  * **Sub** (float16, float, double)             -- E_SUB_FLOAT_FUNC_0010
  * **Sub** (int8 ... uint64)                    -- E_SUB_INT_FUNC_0010

Only ``numpy`` and the standard library are imported.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sub"]


# The document defines the operator only for a single common operand type
# ("Tensors A, B, and C have the same type", e.g. [E_SUB_FLOAT_CONSTR_A_0020]);
# the numpy dtype kinds below are the ones the document covers.
_FLOAT_KIND = "f"      # float16, float (float32), double (float64)
_INT_KINDS = "iu"      # int8, int16, int32, int64, uint8, uint16, uint32, uint64
_OBJECT_KIND = "O"     # exact Python numbers: the concrete stand-in for "real"


def _broadcast_and_type(a: np.ndarray, b: np.ndarray) -> np.dtype:
    """Return the common dtype of the result.

    "Tensors A, B, and C have the same type" is a precondition on the operands,
    so in the specified use both operands already have the same dtype.  When they
    disagree (outside the document's scope), numpy's promotion decides, and the
    wrap-around rule of the integer section is applied at the promoted width.
    """
    return np.result_type(a.dtype, b.dtype)


def _sub_real(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sub (real, real) -- E_SUB_REAL_FUNC_0010.

    "The subtraction is exact: C[i] is the difference in R of the two elements,
    with no approximation."  numpy has no machine type that *is* the mathematical
    reals; object arrays of exact Python numbers (int, Fraction, Decimal, ...)
    subtract exactly, and are the concrete stand-in for this section.
    """
    return np.subtract(a, b)


def _sub_float(a: np.ndarray, b: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Sub (float16, float, double) -- E_SUB_FLOAT_FUNC_0010.

    The case analysis of the document:

      * NaN when either broadcast operand is NaN, or when both are +inf or both
        -inf (the invalid operation of IEEE 754 section 7.2).  numpy propagates a
        NaN operand and computes inf - inf = NaN; the document adds that the NaN
        is quiet and that "its sign and its payload are not specified ... every
        quiet NaN of the type of C is a conforming result", so numpy's quiet NaN
        conforms.
      * The exact difference when it is representable in the type of C.  IEEE 754
        subtraction returns exactly the correctly rounded exact difference, so
        for a representable difference the computed value *is* that difference.
      * Otherwise round(A[i] - B[i]) -- "rounded to the nearest value of the type
        of C using the roundTiesToEven attribute of IEEE 754, the exponent range
        of the type being unbounded: a magnitude whose nearest value exceeds the
        largest finite value of the type of C gives an infinity of the sign of
        x".  That is precisely IEEE 754 round-to-nearest-ties-to-even with the
        overflow threshold at the midpoint between the largest finite value and
        the next power of two (at that midpoint the tie breaks upwards, to the
        next power of two, which exceeds the largest finite value) -- which is
        what numpy's correctly rounded subtraction does for every float type.

    So the three cases are one operation: the second and the third case are the
    single correctly rounded IEEE 754 subtraction, and the first case (NaN) is
    that same subtraction.  numpy computes float16 arithmetic with a float32
    intermediate and one final rounding to float16, whose double rounding is
    harmless here because the intermediate precision 24 is at least 2*11 + 2.

    "When the exact difference is null, the result is +0, except when A[i] is -0
    and B[i] is +0, in which case it is -0" is likewise the IEEE 754 rule that
    round-to-nearest-ties-to-even implements for x - y, and numpy follows it
    (-0 - +0 = -0, +0 - +0 = +0, -0 - -0 = +0, x - x = +0).

    Overflow and the invalid operation are nominal here ("Every condition of the
    list is part of the nominal behavior of the operator; none of them is an
    error"), so the numpy warnings they would raise are silenced.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        return np.subtract(
            a.astype(dtype, copy=False), b.astype(dtype, copy=False)
        )


def _sub_int(a: np.ndarray, b: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Sub (int8 ... uint64) -- E_SUB_INT_FUNC_0010.

    C[i] is "the exact difference modulo 2^n", where n is the number of bits of
    the type; the three cases of the definition -- representable in the type of
    C, lower than its minimum (add 2^n), greater than its maximum (subtract 2^n)
    -- are exactly the three residues of that one modular operation, which is
    two's-complement wrap-around.  Subtraction performed in the operand's own
    integer dtype is that wrap-around, and yields the signed range for the signed
    types and the unsigned range for the unsigned types:

        int8   127 - (-1)  ->  -128   (document, Example 2)
        int8  -128 -   1   ->   127   (document, Example 2)
        uint8    0 -   1   ->   255   (document, Example 3)
        uint8  255 -   1   ->   254   (document, Example 3)

    The wrap-around is nominal ("no error can occur"), so an integer overflow
    warning is silenced rather than raised.
    """
    ai = a.astype(dtype, copy=False)
    bi = b.astype(dtype, copy=False)
    with np.errstate(over="ignore"):
        return np.subtract(ai, bi)


def sub(a, b) -> np.ndarray:
    """Return C = Sub(A, B) as specified in ``ops/sub.md``.

    ``a`` and ``b`` are numpy arrays of any of the dtypes the document covers
    (float16/float/double, the eight signed and unsigned integer types, and
    object arrays of exact reals), of any shape, including rank-0 tensors and
    zero-sized dimensions.

    Broadcasting, rank-0 tensors and zero-sized dimensions need no special code:
    "Tensors of different shapes" in each section specifies the multidirectional
    (numpy-style) broadcasting rules, "Tensors with no dimension" specifies that a
    rank-0 operand broadcasts to the shape of the other operand, and "Zero-sized
    dimensions" specifies that a zero-sized dimension is compatible with a size-1
    dimension and with an equal zero-sized one, the resulting dimension being 0 --
    which is what numpy's broadcasting computes (and an incompatible shape, such
    as a size-0 against an unrelated size-5 dimension, is rejected by numpy, as
    the document's compatibility rule requires).
    """
    a = np.asarray(a)
    b = np.asarray(b)

    dtype = _broadcast_and_type(a, b)

    if dtype.kind == _FLOAT_KIND:
        result = _sub_float(a, b, dtype)
    elif dtype.kind in _INT_KINDS:
        result = _sub_int(a, b, dtype)
    elif dtype.kind == _OBJECT_KIND:
        result = _sub_real(a, b)
    else:
        raise TypeError(
            "Sub is not defined by ops/sub.md for operands of dtype kind "
            f"{dtype.kind!r}; the document covers float16, float, double, the "
            "eight signed/unsigned integer types, and (as 'real') object arrays "
            "of exact numbers"
        )

    # A numpy ufunc applied to rank-0 operands returns a bare scalar rather than
    # an array (and a bare Python object for object arrays).  "A tensor with no
    # dimension (a rank-0 tensor)" is a tensor whose shape is empty, so C is
    # restored as a 0-d array.  For every other shape this is a no-op.
    return np.asarray(result)
