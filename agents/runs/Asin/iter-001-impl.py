import numpy as np

# DECISION: The document defines Asin for real numbers and for the floating-point
# types float16, float, and double. The entry point takes one positional
# argument. We implement the operator using numpy's arcsin, which for
# floating-point inputs follows the floating-point specification. For
# non-floating-point inputs, the document does not specify the result type or
# behavior; we rely on numpy's type promotion. This is a choice not specified
# by the document.

# DECISION: The document requires the result to be the correctly rounded exact
# arcsine (roundTiesToEven, unbounded exponent). The document does not specify
# an algorithm to compute this. We use numpy's arcsin, which is the standard
# implementation available. Numpy's arcsin may not be correctly rounded in all
# cases. This is a deviation from the strict specification, but the document
# does not provide a method to achieve exact rounding.

# DECISION: The document references "General restrictions" but does not include
# them. We cannot implement them.

# DECISION: The document says for real, X elements lie in [-1, 1]. For float,
# out-of-range gives NaN. If the input is an integer array with out-of-range
# values, the document does not specify behavior. We apply numpy's arcsin,
# which gives NaN.

# DECISION: The document does not mention complex types. We do not handle them.

# DECISION: The document says the result for float is a value of the same type
# as X. Numpy's arcsin preserves dtype for float16, float32, and float64. For
# other types, it promotes. This is a choice not specified by the document.

# DECISION: The document says the result is exact for real numbers. In Python,
# we cannot represent exact real numbers. We use floating point. The
# floating-point section is the operational specification.

# DECISION: The document says the accuracy of an implementation whose result
# departs from the value stated above is analyzed in the accuracy guidelines.
# We do not have the accuracy guidelines, so we cannot implement them.

# DECISION: The document is based on ONNX documentation Asin version 7, opset
# 14. We do not have the ONNX documentation, so we rely solely on the provided
# specification.

# DECISION: The document says for float16, the result is rounded directly to
# float16. Numpy's implementation for float16 may compute in a wider type and
# then round, which can cause double rounding. The document does not specify
# an algorithm. We rely on numpy's implementation.

# DECISION: The document says the NaN of the first case is a quiet NaN; its
# sign and payload are not specified. Numpy's arcsin returns a quiet NaN. We
# rely on numpy's NaN.

# DECISION: The document says subnormal results are not flushed to zero.
# Numpy does not flush subnormals by default. We rely on numpy's behavior.

# DECISION: The document says the result never overflows. Numpy's arcsin does
# not overflow for arcsine. We rely on numpy's behavior.

# DECISION: The document says the operator has no attributes. Our function
# takes one positional argument. This matches the document.

# DECISION: The document says the input is a tensor. We accept array-like
# inputs. This is a choice not specified by the document.

# DECISION: The document says the output is a tensor. We return a numpy array.
# This is a choice not specified by the document.

# DECISION: The document says shape is preserved. Numpy preserves shape. We
# rely on numpy's behavior.

# DECISION: The document says type consistency. Numpy preserves dtype for
# float types. We rely on numpy's behavior.

# DECISION: The document says for real, the result is the unique value in
# [-pi/2, pi/2]. Numpy's arcsin returns that. We rely on numpy's behavior.

# DECISION: The document says for float, Asin(1) is nearest to pi/2. Numpy's
# arcsin returns that. We rely on numpy's behavior.

# DECISION: The document says for float, Asin(-1) is opposite. Numpy's arcsin
# returns that. We rely on numpy's behavior.

# DECISION: The document says for float, Asin(±0) is ±0. Numpy's arcsin
# preserves signed zero. We rely on numpy's behavior.

# DECISION: The document says for float, Asin(NaN) is NaN. Numpy's arcsin
# returns NaN. We rely on numpy's behavior.

# DECISION: The document says for float, Asin(±inf) is NaN. Numpy's arcsin
# returns NaN. We rely on numpy's behavior.

# DECISION: The document says for float, Asin(x) for |x|>1 is NaN. Numpy's
# arcsin returns NaN. We rely on numpy's behavior.

# DECISION: The document says for float, the result is rounded to nearest
# using roundTiesToEven. Numpy's arcsin may not use roundTiesToEven for the
# final rounding. We rely on numpy's rounding.

# DECISION: The document says the result is a value of that type. Numpy's
# arcsin preserves dtype. We rely on numpy's behavior.

# DECISION: The document says the result lies in [-pi/2, pi/2] for every
# element not NaN. Numpy's arcsin does. We rely on numpy's behavior.

# DECISION: The document says the arcsine is increasing on [-1,1]. Numpy's
# arcsin is increasing. We rely on numpy's behavior.

# DECISION: The document says rank-0 tensor result is rank-0. Numpy's arcsin
# on 0-d array returns 0-d array. We rely on numpy's behavior.

# DECISION: The document says zero-sized dimensions result is empty. Numpy's
# arcsin on empty array returns empty array. We rely on numpy's behavior.

# DECISION: The document says the operands may be any value including special
# numbers. Numpy's arcsin handles them. We rely on numpy's behavior.

# DECISION: The document says every element is admissible. Numpy's arcsin
# accepts any float. We rely on numpy's behavior.

# DECISION: The document says invalid operation gives NaN. Numpy's arcsin
# gives NaN. We rely on numpy's behavior.

# DECISION: The document says no error condition. Numpy's arcsin does not
# raise errors for these inputs. We rely on numpy's behavior.

# DECISION: The document says the operator has no attribute. Our function has
# no extra parameters. This matches the document.

# DECISION: The document says the input is X. Our function takes X. This
# matches the document.

# DECISION: The document says the output is Y. Our function returns Y. This
# matches the document.

# DECISION: The document says the constraints. We do not enforce them. This
# matches the document, which does not require validation.

# DECISION: The document says the general restrictions are applicable. We do
# not have them. We cannot implement them.


def asin(X):
    """
    Computes the arcsine of tensor X element-wise.

    This function implements the Asin operator as specified in the document.

    For the real type, the document states:

        Operator Asin computes the arcsine of tensor X element-wise and stores
        the result in tensor Y. The arcsine is the inverse of the sine: the
        result is the angle whose sine is the element of X.

        For any tensor index i of the result Y:

            Y[i] = arcsin(X[i])

        where arcsin(x) is the unique value y such that sin(y) = x and
        y in [-pi/2, pi/2].

        The result is exact: Y[i] is the real number arcsin(X[i]), with no
        approximation.

        Bounds and sign. arcsin(-1) = -pi/2, arcsin(0) = 0 and arcsin(1) = pi/2;
        the result lies in [-pi/2, pi/2] for every element of X. The arcsine is
        an odd function, so that arcsin(-x) = -arcsin(x): the result has the
        sign of the element of X, and a null element gives a null result. The
        arcsine is increasing on [-1, 1].

        Tensors with no dimension. A tensor with no dimension (a rank-0 tensor)
        is a tensor whose shape is empty. The result is a rank-0 tensor whose
        single element is the arcsine of the single element of X.

        Zero-sized dimensions. Tensor X may have a zero-sized dimension. Tensor
        Y then has the same zero-sized dimension and is empty: the operator is
        applied to no element.

    For the floating-point types (float16, float, double), the document states:

        Operator Asin computes the arcsine of tensor X element-wise according
        to IEEE 754 floating-point semantics and stores the result in tensor Y.
        Tensors X and Y have the same floating-point type, and the result is a
        value of that type.

        For any tensor index i of the result Y:

            Y[i] =
            {
                NaN                     if X[i] is NaN or |X[i]| > 1
                X[i]                    if X[i] is ±0
                round(arcsin(X[i]))     otherwise
            }

        where arcsin(x) is the unique value y such that sin(y) = x and
        y in [-pi/2, pi/2], i.e. the exact arcsine of the real number x,
        round(x) is the value of x rounded to the nearest value of the type of
        X and Y using the roundTiesToEven attribute of IEEE 754, the exponent
        range of the type being unbounded.

        The second case fixes the sign of a null result: the arcsine of -0 is
        -0 and the arcsine of +0 is +0, whereas the real number arcsin(0) is 0
        and carries no sign.

        The third case applies to every element of X whose magnitude is at most
        1 and which is not a zero, including the bounds -1 and 1: the exact
        arcsine of such an element lies in [-pi/2, pi/2], and the result is
        that value rounded to the type of X and Y. The value pi/2 is not
        representable in any of the three types, so that the result of Asin(1)
        is the value of the type nearest to pi/2, and the result of Asin(-1) is
        its opposite.

        The result never overflows. The magnitude of the exact arcsine of an
        element of X is at most pi/2, which is below the largest finite value
        of each of the three types; no result is an infinity, and the rounding
        of the third case never gives one. A result whose magnitude is below
        the smallest normal value of the type is a subnormal value: the
        rounding is the one defined above, with an unbounded exponent range, so
        that the result is the subnormal value nearest to the exact arcsine and
        is not flushed to zero.

        Sign and monotonicity. The arcsine is an odd function, so that the
        result has the sign of the element of X for every element that is not
        NaN: Asin(-x) and Asin(x) are opposite values. The result lies in
        [-pi/2, pi/2] for every element of X that is not NaN, and the arcsine
        is increasing on [-1, 1].

        Values of the operands. The operands may be any of the values of the
        type, including the special numbers ±0, ±inf and NaN. Every element of
        the type is an admissible operand: no precondition restricts X to
        [-1, 1], so that an element whose magnitude is greater than 1 is an
        operand the operator accepts, and the first case of the formula gives
        NaN for it. The cases above give the result for every one of them: an
        element that is NaN gives NaN, and an element whose magnitude is
        greater than 1, including +inf and -inf, gives NaN. The arcsine of an
        infinity is the invalid operation defined in IEEE 754 section 7.2, and
        so is the arcsine of a finite value whose magnitude is greater than 1;
        the result is NaN in both cases. The NaN of the first case is a quiet
        NaN; its sign and its payload are not specified, whether the NaN comes
        from an operand or from the invalid operation, so that every quiet NaN
        of the type of X and Y is a conforming result.

        Tensors with no dimension. A tensor with no dimension (a rank-0 tensor)
        is a tensor whose shape is empty. The result is a rank-0 tensor whose
        single element is the arcsine of the single element of X.

        Zero-sized dimensions. Tensor X may have a zero-sized dimension. Tensor
        Y then has the same zero-sized dimension and is empty: the operator is
        applied to no element.

    The document also states the constraints:

        [E_ASIN_REAL_CONSTR_X_0010] Definition domain
        Statement: Every element of X lies in [-1, 1]: for every tensor index i,
        -1 <= X[i] <= 1.

        [E_ASIN_REAL_CONSTR_Y_0010] Shape definition
        Statement: Tensor Y has the same shape as tensor X.

        [E_ASIN_FLOAT_CONSTR_X_0010] Type consistency
        Statement: Tensors X and Y have the same type.

        [E_ASIN_FLOAT_CONSTR_Y_0010] Shape definition
        Statement: Tensor Y has the same shape as tensor X.

        [E_ASIN_FLOAT_CONSTR_Y_0020] Type consistency
        Statement: see constraint E_ASIN_FLOAT_CONSTR_X_0010 on tensor X.
    """
    return np.arcsin(X)
