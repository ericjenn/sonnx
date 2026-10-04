"""
Module implementing the Asin operator.

This module implements the Asin operator as specified in the provided document.
The document defines the operator for two type categories: "real" and
"float16, float, double". The function `asin` computes the element-wise
arcsine of its input.
"""

import numpy as np


def asin(X):
    """
    Computes the element-wise arcsine of X.

    This function implements the Asin operator as specified in the provided
    document. The document defines the operator for two type categories:
    "real" and "float16, float, double".

    For type real, the document states:

        Operator Asin computes the arcsine of tensor X element-wise and stores
        the result in tensor Y. The arcsine is the inverse of the sine: each
        element of Y is the value of [-pi/2, pi/2] whose sine is the
        corresponding element of X.

        For every tensor index i of the result Y, the result Y[i] is the unique
        value y such that

            sin(y) = X[i]  and  y in [-pi/2, pi/2]

        The equation sin(y) = X[i] alone has infinitely many solutions, one in
        each interval of length 2*pi; the interval [-pi/2, pi/2] selects exactly
        one of them, and it is therefore as specifying as the equation. The
        result is the arcsine of X[i], written arcsin(X[i]).

        Domain. The equation has a solution only when X[i] lies in [-1, 1],
        the range of the sine; outside that interval no real number y satisfies
        it, and the operator is not defined. The domain of the operator is the
        set of tensors whose every element lies in [-1, 1], and it is stated as
        the precondition E_ASIN_REAL_CONSTR_X_0010 on input X.

        Bounds and sign. The result lies in [-pi/2, pi/2] for every element of
        the domain: arcsin(-1) = -pi/2, arcsin(0) = 0 and arcsin(1) = pi/2.
        The arcsine is odd, arcsin(-x) = -arcsin(x), so that the result has the
        sign of the operand and is null only for a null operand.

        Tensors with no dimension. A tensor with no dimension (a rank-0 tensor)
        is a tensor whose shape is empty. The result is a rank-0 tensor whose
        single element is the arcsine of the single element of X.

        Zero-sized dimensions. An operand may have a zero-sized dimension. The
        result has the same shape, so that the corresponding dimension of Y has
        size 0 and Y is empty.

    For types float16, float, double, the document states:

        Operator Asin computes the arcsine of tensor X element-wise according
        to IEEE 754 floating-point semantics and stores the result in tensor Y.
        Each element of Y is the arcsine of the corresponding element of X,
        rounded to the type of X and Y.

        For any tensor index i of the result Y:

            Y[i] = arcsin_(f)(X[i]) =
                NaN       if X[i] is NaN
                NaN       if X[i] is +inf or -inf, or |X[i]| > 1
                X[i]      if X[i] is +0 or -0
                round(arcsin(X[i]))  otherwise

        where arcsin_(f) is the arcsine for the floating-point type of X and Y,
        arcsin(x) is the arcsine of the real number x, and round(x) is the
        value of x rounded to the nearest value of the type of X and Y using
        the roundTiesToEven attribute of IEEE 754, the exponent range of the
        type being unbounded.

        NaN operand. An operand that is NaN is propagated: the result is NaN.
        The NaN of the result is a quiet NaN; its sign and its payload are not
        specified, so that every quiet NaN of the type of X and Y is a
        conforming result. A signaling NaN operand is a NaN and gives the same
        result.

        Operand outside the domain. An operand whose magnitude is greater than
        1, including an infinite operand, lies outside the domain of the
        arcsine: the operation is the invalid operation defined in IEEE 754
        section 7.2, and the result is NaN. That NaN is a quiet NaN, and every
        quiet NaN of the type of X and Y is a conforming result.

        Null operand. arcsin(+0) = +0 and arcsin(-0) = -0: the result is the
        operand itself, sign included.

        Operand in the domain. For 0 < |X[i]| <= 1, the result is the arcsine
        of X[i] rounded as defined above. The exact arcsine is not
        representable in general: arcsin(±1) = ±pi/2 is irrational, and so is
        arcsin(x) for most x; the result is then the nearest value of the type
        of X and Y. The result is never an infinity, since |arcsin(x)| <=
        pi/2 for every x of the domain, so that the operator never overflows.
        The result may be subnormal: for a subnormal operand, the exact arcsine
        is a value of the same magnitude, and it rounds to a subnormal value of
        the type.

        Bounds and sign. The result lies in [-pi/2, pi/2] for every operand of
        the domain, and it is NaN otherwise. The arcsine is odd, so that the
        result has the sign of the operand for every operand of the domain.

        Tensors with no dimension. A tensor with no dimension (a rank-0 tensor)
        is a tensor whose shape is empty. The result is a rank-0 tensor whose
        single element is the arcsine of the single element of X.

        Zero-sized dimensions. An operand may have a zero-sized dimension. The
        result has the same shape, so that the corresponding dimension of Y has
        size 0 and Y is empty.

    The function uses numpy.arcsin, which implements the element-wise arcsine
    for floating-point types and preserves the shape and dtype of the input for
    float16, float32, and float64.
    """

    # DECISION: The document references "General restrictions" but does not
    # include them. We do not implement any general restrictions because they
    # are not specified in the provided document.
    # Quote: "[General restrictions](./../common/general_restrictions.md) are applicable."

    # DECISION: For inputs that are not float16, float32, or float64, the
    # document's "real" section applies. The document does not specify a
    # concrete representation for "real tensor". We rely on numpy's default
    # behavior, which converts integer inputs to float64 and computes the
    # arcsine. This is a choice because the document says nothing about the
    # output type for real inputs.
    # Quote: "X: real tensor" and "Y: real tensor".

    # DECISION: The document states that for real inputs, the precondition
    # E_ASIN_REAL_CONSTR_X_0010 requires every element of X to lie in [-1, 1].
    # It does not specify an error condition for its violation. We do not add
    # validation. If the precondition is violated, numpy.arcsin returns NaN
    # and may emit a RuntimeWarning. This is a choice because the document
    # says "No error condition" and does not specify behavior outside the
    # domain for real inputs.
    # Quote: "The only condition ... is the operand outside the domain, and it
    # is ruled out by the precondition on X; ... No error condition."

    # DECISION: The document states that for float types, an operand outside
    # the domain gives NaN as a nominal result. numpy.arcsin does this and
    # emits a RuntimeWarning for invalid value. The document does not mention
    # warnings, and the requirements forbid suppressing warnings not mentioned.
    # We do not suppress the warning.
    # Quote: "Invalid operation ... nominal: specified ... the result is NaN"
    # and "No error condition."

    # DECISION: The document states that for float types, "Tensors X and Y
    # have the same type." We rely on numpy.arcsin to preserve the dtype for
    # float16, float32, and float64 inputs.
    # Quote: "Tensors X and Y have the same type."

    # DECISION: The document states that for real inputs, "The results are
    # exact: they are the values of the real numbers, not rounded values."
    # In numpy, we cannot represent exact real numbers; we use floating-point
    # arithmetic. This is a limitation of the implementation environment.
    # Quote: "The results are exact: they are the values of the real numbers,
    # not rounded values."

    # DECISION: The document does not specify behavior for complex inputs.
    # The document specifies real tensors and floating-point tensors only.
    # We do not add validation for complex inputs; numpy.arcsin will compute
    # the complex arcsine, which is outside the specification.
    # Quote: "X: real tensor" and "X: floating-point tensor".

    # DECISION: The document does not specify behavior for NaN or infinite
    # inputs in the "real" section, because NaN and infinity are not real
    # numbers. We rely on numpy.arcsin, which returns NaN for such inputs.
    # Quote: "X: real tensor" and "The equation has a solution only when X[i]
    # lies in [-1, 1]".

    return np.arcsin(X)
