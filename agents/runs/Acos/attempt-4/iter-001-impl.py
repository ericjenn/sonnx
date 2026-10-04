"""
Implementation of the ONNX Acos operator, as specified in the provided document.

The operator computes the element-wise arccosine of a tensor X.
"""

import numpy as np


def acos(X):
    """
    Compute the element-wise arccosine of X.

    Implements the Acos operator for real and floating-point tensors.

    For real tensors (E_ACOS_REAL_FUNC_0010):
        "Operator Acos computes the arccosine of tensor X element-wise and stores
        the result in tensor Y. The arccosine is the inverse of the cosine: each
        element of Y is the angle of [0, pi] whose cosine is the corresponding
        element of X."
        "For any tensor index i of the result Y: Y[i] = acos(X[i])"
        "The result is the value of that branch, not a way of computing it."
        "The arccosine is defined on [-1, 1] ... Every element of Y lies in [0, pi]."
        "Values at the bounds and monotonicity. acos(1) = 0, acos(0) = pi/2 and
        acos(-1) = pi."
        "Exactness. For real numbers the result is exact: Y[i] is the real number
        acos(X[i]), with no approximation."
        "Tensors with no dimension. A tensor with no dimension (a rank-0 tensor)
        is a tensor whose shape is empty. The result is a rank-0 tensor whose
        single element is the arccosine of the single element of X."
        "Zero-sized dimensions. An operand may have a zero-sized dimension. The
        result has the same shape as X and is then empty."

    For floating-point tensors (E_ACOS_FLOAT_FUNC_0010):
        "Operator Acos computes the arccosine of tensor X element-wise and stores
        the result in tensor Y. Each element of Y is the value of the arccosine
        of the corresponding element of X, rounded to the type of X and Y."
        "For any tensor index i of the result Y: Y[i] = round(acos(X[i]))"
        "where acos is the arccosine of the real numbers ... and round(y) is the
        value of y rounded to the nearest value of the type of X and Y using the
        roundTiesToEven attribute of IEEE 754, the exponent range of the type
        being unbounded."
        "An operand that is NaN gives a NaN result; that NaN is a quiet NaN whose
        sign and payload are not specified, so that every quiet NaN of the type
        of X and Y is a conforming result."
        "An operand that is +0 or -0 gives pi/2 rounded to the type of X and Y:
        the arccosine of a null value is pi/2 whatever the sign of the zero, so
        that +0 and -0 give the same result."
        "The result lies in [0, pi], which is within the range of the three types:
        it is never an infinity, and the rounding of the exact arccosine to the
        type of X and Y never overflows. The result is null only when the operand
        is 1, and it is then +0."
        "The result is the real arccosine rounded to the type of X and Y; it is
        exact only when the real arccosine is representable in that type."
        "Tensors with no dimension. ..."
        "Zero-sized dimensions. ..."

    Constraints:
        E_ACOS_REAL_CONSTR_X_0010 / E_ACOS_FLOAT_CONSTR_X_0010: Every element of
        X lies in [-1, 1].
        E_ACOS_FLOAT_CONSTR_X_0020: Tensors X and Y have the same type.
        E_ACOS_REAL_CONSTR_Y_0010 / E_ACOS_FLOAT_CONSTR_Y_0010: Tensor Y has the
        same shape as X.

    The document states that out-of-domain operands are ruled out by the
    precondition and that there is no error condition. This implementation does
    not validate the input; it relies on numpy's arccos, which returns NaN for
    out-of-domain values and may emit a RuntimeWarning. The document does not
    mention suppressing such warnings, so they are not suppressed.
    """
    # DECISION: The document references "[General restrictions](./../common/general_restrictions.md)"
    # but does not include their content. We implement only the restrictions
    # explicitly stated in this document; the referenced general restrictions are
    # not available and are therefore not applied.
    #
    # DECISION: The document specifies that for real numbers the result is exact
    # ("For real numbers the result is exact: Y[i] is the real number acos(X[i]),
    # with no approximation."). numpy's arccos computes a floating-point
    # approximation. We use numpy's arccos as the only available implementation
    # in this environment; the exact real result is not representable in
    # floating point for irrational values. This is a deviation from the
    # mathematical exactness stated for real tensors, forced by the choice of
    # numpy as the implementation tool.
    #
    # DECISION: The document does not specify the concrete output type for the
    # "real" section (it only says "real tensor" and "Y: real tensor"). For
    # floating-point inputs, numpy's arccos preserves the input dtype (float16,
    # float32, float64), satisfying E_ACOS_FLOAT_CONSTR_X_0020 ("Tensors X and Y
    # have the same type"). For integer or other real inputs, numpy's arccos
    # returns float64. The document does not state what type a "real tensor"
    # should have in this implementation, so we accept numpy's default.
    #
    # DECISION: The document requires, for floating-point types, that the result
    # be "rounded to the nearest value of the type of X and Y using the
    # roundTiesToEven attribute of IEEE 754". numpy's arccos is used directly;
    # we do not implement a correctly-rounded arccosine. numpy's ufunc may not
    # guarantee correct rounding for all inputs, especially for float16. The
    # document does not specify an algorithm, only the mathematical result, so
    # we rely on the platform's arccos.
    #
    # DECISION: The document says that a NaN operand gives a quiet NaN whose
    # sign and payload are not specified. numpy's arccos returns a NaN; we do
    # not attempt to control its sign or payload, which is permitted by the
    # document.
    #
    # DECISION: The document says that out-of-domain operands are ruled out by
    # the precondition and that there is no error condition. It does not specify
    # behavior when the precondition is violated. We do not add validation; we
    # let numpy's arccos return NaN and emit any warning it normally would. The
    # document does not mention suppressing warnings, so we do not suppress them.
    #
    # DECISION: The document says that for a rank-0 tensor the result is a
    # rank-0 tensor. numpy's ufunc returns a 0-d array for a 0-d array input and
    # a scalar for a scalar input. The document does not specify the Python
    # representation of a tensor, so we return numpy's result directly without
    # forcing a container.
    return np.arccos(X)
