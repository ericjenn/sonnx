"""
Asin operator.

This module implements the Asin operator as specified in the document
"Based on ONNX documentation Asin version 7, the definition of Asin in opset 14."

The document defines the operator for two families of types:
- real numbers, where the result is the exact real arcsine;
- floating-point types (float16, float, double), where the result is the
  arcsine rounded to the nearest value of the type using roundTiesToEven.

This implementation provides the floating-point semantics, because the real
semantics cannot be represented in finite-precision floating-point arithmetic.
The document itself states for the floating-point types: "The three types share
one semantics: the IEEE 754 standard defines the same arcsine for all of them,
and they differ only by their precision and by the range of the values they
represent."

The entry point is the function `asin`.
"""

import numpy as np


def asin(X):
    """
    Computes the arcsine of tensor X element-wise.

    Implements the floating-point semantics of the Asin operator as specified
    in the document. For any tensor index i of the result Y:

        Y[i] = NaN                     if X[i] is NaN or |X[i]| > 1
        Y[i] = X[i]                    if X[i] is ±0
        Y[i] = round(arcsin(X[i]))     otherwise

    where arcsin(x) is the unique value y such that sin(y) = x and
    y ∈ [-π/2, π/2], and round(x) is the value of x rounded to the nearest
    value of the type of X and Y using the roundTiesToEven attribute of
    IEEE 754, the exponent range of the type being unbounded.

    The document further specifies:
    - "The result never overflows."
    - "A result whose magnitude is below the smallest normal value of the type
      is a subnormal value: the rounding is the one defined above, with an
      unbounded exponent range, so that the result is the subnormal value
      nearest to the exact arcsine and is not flushed to zero."
    - "The arcsine is an odd function, so that the result has the sign of the
      element of X for every element that is not NaN."
    - "Tensors X and Y have the same type."
    - "Tensor Y has the same shape as tensor X."
    - "A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is
      empty. The result is a rank-0 tensor whose single element is the arcsine
      of the single element of X."
    - "Tensor X may have a zero-sized dimension. Tensor Y then has the same
      zero-sized dimension and is empty: the operator is applied to no element."

    Parameters
    ----------
    X : array_like
        The tensor whose arcsine is computed. The document specifies that X is
        a floating-point tensor of type float16, float, or double. The document
        does not specify behavior for other types.

    Returns
    -------
    Y : numpy.ndarray
        The element-wise arcsine of X, with the same shape and (for the
        specified floating-point types) the same dtype as X.
    """
    # DECISION: The document defines the operator for both real numbers and
    # floating-point types. The real semantics require exact real arithmetic,
    # which is not representable in floating-point. The floating-point section
    # is the one that specifies concrete types (float16, float, double) and
    # IEEE 754 rounding. We therefore implement the floating-point semantics.
    # The document says: "The three types share one semantics: the IEEE 754
    # standard defines the same arcsine for all of them, and they differ only
    # by their precision and by the range of the values they represent."

    # DECISION: The document does not specify an algorithm for computing the
    # rounded arcsine. We use numpy's arcsin. For float16 and float32, we
    # compute in float64 and then cast back to the original dtype. This
    # improves the accuracy of the rounding, because the float64 result is
    # much closer to the exact arcsine than a direct float16/float32
    # computation would be, reducing the chance of double-rounding errors.
    # The document says: "round(arcsin(X[i]))" with roundTiesToEven, and
    # "the result is that value rounded to the type of X and Y." It does not
    # prescribe an implementation.

    # DECISION: The document does not mention warnings. numpy's arcsin emits
    # a RuntimeWarning for invalid operations (|X[i]| > 1 or ±inf). The
    # document says: "Invalid operation, i.e. the arcsine of an element whose
    # magnitude is greater than 1, including ±inf ... nominal: specified by
    # E_ASIN_FLOAT_FUNC_0010, an element whose magnitude exceeds 1 is an
    # admissible operand and the result is NaN". Since the document does not
    # mention warnings, and the instruction says not to suppress warnings the
    # document does not mention, we do not suppress the warning.

    # DECISION: The document specifies floating-point types only. It does not
    # specify behavior for integer, complex, or other dtypes. We do not add
    # validation; we rely on numpy's type promotion. For non-floating-point
    # inputs, numpy will upcast to a floating-point type (e.g., int -> float64)
    # and compute the arcsine. The document says nothing about such inputs.

    # DECISION: The document says "Tensors X and Y have the same type." For
    # the specified floating-point types (float16, float32, float64), our
    # implementation preserves the dtype. For other types, the dtype may
    # change due to numpy's type promotion, as the document does not specify
    # them.

    # DECISION: The document says "A tensor with no dimension (a rank-0
    # tensor) is a tensor whose shape is empty. The result is a rank-0 tensor
    # whose single element is the arcsine of the single element of X." We
    # return a 0-d numpy array for rank-0 input, as numpy represents rank-0
    # tensors as 0-d arrays. The document does not specify the exact Python
    # type of the returned tensor.

    # DECISION: The document says "Tensor X may have a zero-sized dimension.
    # Tensor Y then has the same zero-sized dimension and is empty: the
    # operator is applied to no element." We rely on numpy to preserve the
    # shape, including zero-sized dimensions. numpy's arcsin on an empty
    # array returns an empty array of the same shape.

    # DECISION: The document says "The arcsine of -0 is -0 and the arcsine of
    # +0 is +0". numpy's arcsin preserves the sign of zero. We rely on this.

    # DECISION: The document says "The result never overflows." We do not
    # perform any overflow checks, as the result is always within [-π/2, π/2].

    # DECISION: The document says "The NaN of the first case is a quiet NaN;
    # its sign and its payload are not specified". numpy's arcsin returns a
    # quiet NaN. We do not attempt to control the sign or payload.

    X = np.asarray(X)
    dtype = X.dtype

    if dtype == np.float16 or dtype == np.float32:
        # Compute in float64 for better rounding accuracy, then cast back.
        X64 = X.astype(np.float64)
        Y64 = np.arcsin(X64)
        Y = Y64.astype(dtype)
    else:
        # For float64 and any other dtype, use numpy's arcsin directly.
        Y = np.arcsin(X)

    return Y
