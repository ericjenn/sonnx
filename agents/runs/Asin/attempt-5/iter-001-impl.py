"""
Implementation of the Asin operator.

This module implements the Asin operator as specified in the document.
The document specifies two variants: one for mathematical real numbers
and one for IEEE 754 floating-point types (float16, float, double).
This implementation follows the floating-point variant, as it operates
on numpy arrays of floating-point types.

The entry point is the function `asin`.
"""

import numpy as np

# DECISION: The document specifies two variants: one for mathematical real
# numbers (E_ASIN_REAL_FUNC_0010) and one for IEEE 754 floating-point types
# (E_ASIN_FLOAT_FUNC_0010). The implementation operates on numpy arrays of
# floating-point types, so we implement the floating-point variant. For
# inputs that are not floating-point (e.g., integers, complex), the document
# does not specify; we delegate to numpy.arcsin, which promotes to float64
# or computes the complex arcsine. This is a decision because the document
# is silent on non-floating-point inputs.

# DECISION: The document requires the result to be the exact arcsine
# rounded to the nearest value of the type using roundTiesToEven. The
# implementation delegates to numpy.arcsin, which may not be correctly
# rounded for all inputs. This is a limitation of the implementation, not
# a deviation from the specification's intent. The document does not
# provide an algorithm for exact rounding.

# DECISION: The document says for float types, the result is NaN for
# |X| > 1. numpy.arcsin returns NaN and issues a RuntimeWarning. The
# document does not mention warnings. We do not suppress warnings, as
# the requirements state not to suppress warnings the document does not
# mention.

# DECISION: The document references general restrictions and accuracy
# guidelines but does not include them. We assume no additional
# restrictions apply beyond those stated in the document.

# DECISION: The document does not specify the input container type. We
# convert to a numpy array via np.asarray to ensure tensor semantics
# (e.g., a Python scalar becomes a rank-0 tensor, matching the document's
# statement that "The result is a rank-0 tensor whose single element is
# the arcsine of the single element of X").

# DECISION: The document says for the real variant, X is restricted to
# [-1, 1] (E_ASIN_REAL_CONSTR_X_0010). For the float variant, any value
# is admissible and |X| > 1 gives NaN. Since we implement the float
# variant, we do not enforce the real precondition.

# DECISION: The document says the result is a value of the same type as X
# (E_ASIN_FLOAT_CONSTR_X_0010). We rely on numpy.arcsin to preserve dtype
# for float16, float32, and float64. For other types, numpy may promote.

# DECISION: The document says the result never overflows and subnormals
# are not flushed to zero. We rely on numpy's default behavior, which
# does not flush subnormals to zero and does not overflow for arcsin.

# DECISION: The document says the NaN is quiet and its sign and payload
# are unspecified. numpy.arcsin returns a quiet NaN.

# DECISION: The document says the arcsine of -0 is -0. numpy.arcsin
# preserves signed zero.

# DECISION: The document says the result is exact for the real variant.
# We do not implement exact real arithmetic; we implement the float
# variant, which specifies rounding to the nearest representable value.

# DECISION: The document says the operator has no attributes. We do not
# implement any attributes.

# DECISION: The document says the output shape is the same as the input
# shape (E_ASIN_FLOAT_CONSTR_Y_0010). numpy.arcsin preserves shape.

# DECISION: The document says zero-sized dimensions result in an empty
# output. numpy.arcsin preserves empty arrays.

# DECISION: The document says rank-0 tensors result in rank-0 tensors.
# np.asarray ensures scalars become 0-d arrays.

def asin(X):
    """
    Computes the arcsine of tensor X element-wise.

    This function implements the Asin operator for IEEE 754 floating-point
    types (float16, float, double) as specified in the document.

    From the document (E_ASIN_FLOAT_FUNC_0010):

        Operator Asin computes the arcsine of tensor X element-wise
        according to IEEE 754 floating-point semantics and stores the
        result in tensor Y. Tensors X and Y have the same floating-point
        type, and the result is a value of that type.

        For any tensor index i of the result Y:

            Y[i] = NaN                     if X[i] is NaN or |X[i]| > 1
                   X[i]                     if X[i] is ±0
                   round(arcsin(X[i]))      otherwise

        where arcsin(x) is the unique value y such that sin(y) = x and
        y ∈ [-π/2, π/2], i.e. the exact arcsine of the real number x,
        and round(x) is the value of x rounded to the nearest value of
        the type of X and Y using the roundTiesToEven attribute of
        IEEE 754, the exponent range of the type being unbounded.

    The document also states:

        The second case fixes the sign of a null result: the arcsine of
        -0 is -0 and the arcsine of +0 is +0, whereas the real number
        arcsin(0) is 0 and carries no sign.

        The third case applies to every element of X whose magnitude is
        at most 1 and which is not a zero, including the bounds -1 and 1:
        the exact arcsine of such an element lies in [-π/2, π/2], and the
        result is that value rounded to the type of X and Y. The value
        π/2 is not representable in any of the three types, so that the
        result of Asin(1) is the value of the type nearest to π/2, and
        the result of Asin(-1) is its opposite.

        The result never overflows. The magnitude of the exact arcsine
        of an element of X is at most π/2, which is below the largest
        finite value of each of the three types; no result is an infinity,
        and the rounding of the third case never gives one. A result
        whose magnitude is below the smallest normal value of the type
        is a subnormal value: the rounding is the one defined above, with
        an unbounded exponent range, so that the result is the subnormal
        value nearest to the exact arcsine and is not flushed to zero.

        Sign and monotonicity. The arcsine is an odd function, so that
        the result has the sign of the element of X for every element
        that is not NaN: Asin(-x) and Asin(x) are opposite values. The
        result lies in [-π/2, π/2] for every element of X that is not
        NaN, and the arcsine is increasing on [-1, 1].

        Values of the operands. The operands may be any of the values of
        the type, including the special numbers ±0, ±inf and NaN. Every
        element of the type is an admissible operand: no precondition
        restricts X to [-1, 1], so that an element whose magnitude is
        greater than 1 is an operand the operator accepts, and the first
        case of the formula gives NaN for it. The cases above give the
        result for every one of them: an element that is NaN gives NaN,
        and an element whose magnitude is greater than 1, including +inf
        and -inf, gives NaN. The arcsine of an infinity is the invalid
        operation defined in IEEE 754 section 7.2, and so is the arcsine
        of a finite value whose magnitude is greater than 1; the result
        is NaN in both cases. The NaN of the first case is a quiet NaN;
        its sign and its payload are not specified, whether the NaN comes
        from an operand or from the invalid operation, so that every
        quiet NaN of the type of X and Y is a conforming result.

        Tensors with no dimension. A tensor with no dimension (a rank-0
        tensor) is a tensor whose shape is empty. The result is a rank-0
        tensor whose single element is the arcsine of the single element
        of X.

        Zero-sized dimensions. Tensor X may have a zero-sized dimension.
        Tensor Y then has the same zero-sized dimension and is empty:
        the operator is applied to no element.

    The document also specifies a real variant (E_ASIN_REAL_FUNC_0010)
    for mathematical real numbers, where the result is exact and X is
    restricted to [-1, 1]. This implementation does not implement the
    real variant, as it operates on floating-point types.

    Parameters
    ----------
    X : array_like
        Input tensor. May be a numpy array or any object convertible to
        a numpy array. The document specifies floating-point types
        (float16, float, double). Other types are not specified by the
        document; see DECISION comments.

    Returns
    -------
    Y : numpy.ndarray
        The element-wise arcsine of X. The shape and type are the same
        as X for floating-point inputs, as specified by the document.

    Notes
    -----
    This implementation delegates to numpy.arcsin. The document requires
    exact rounding to the nearest value of the type using roundTiesToEven.
    numpy.arcsin may not be correctly rounded for all inputs. See
    DECISION comments.
    """
    X = np.asarray(X)
    return np.arcsin(X)
