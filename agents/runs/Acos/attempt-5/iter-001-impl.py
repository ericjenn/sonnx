"""
Module implementing the ONNX Acos operator (opset 14) as specified.

The specification defines Acos for type real and for types float16, float, double.
This module exposes the operator as the function `acos`.

The specification is quoted in the function docstring below.
"""

import numpy as np


def acos(X):
    """
    Compute the element-wise arccosine of tensor X.

    Specification (Acos version 7, based on ONNX opset 14):

    Signature:
        Y = Acos(X)
    where
        X: value whose arccosine is computed
        Y: result of the element-wise arccosine of X

    Function (real):
        [E_ACOS_REAL_FUNC_0010]
        Part 1. Operator Acos computes the arccosine of tensor X element-wise and
        stores the result in tensor Y. The arccosine is the inverse of the cosine:
        each element of Y is the angle of [0, pi] whose cosine is the corresponding
        element of X.
        Part 2. For any tensor index i of the result Y:
            Y[i] = acos(X[i])
        where acos is defined by the following property: for every x in [-1, 1],
        acos(x) is the unique value y such that
            cos(y) = x and y in [0, pi]
        The equation cos(y) = x alone has infinitely many solutions, one in each
        interval of length 2*pi; the interval [0, pi] is the branch on which it has
        exactly one, and it is therefore as specifying as the equation. The result
        is the value of that branch, not a way of computing it.
        Domain and range. The arccosine is defined on [-1, 1]: for x in that
        interval, cos takes the value x at exactly one point of [0, pi], whereas
        for x outside it, no real number has x as its cosine. The constraint
        [E_ACOS_REAL_CONSTR_X_0010] requires every element of X to lie in [-1, 1],
        so that the result is defined for every element of X. Every element of Y
        lies in [0, pi].
        Values at the bounds and monotonicity. acos(1) = 0, acos(0) = pi/2 and
        acos(-1) = pi. The arccosine is strictly decreasing on [-1, 1]: if
        x1 < x2, then acos(x1) > acos(x2).
        Exactness. For real numbers the result is exact: Y[i] is the real number
        acos(X[i]), with no approximation. For instance acos(1/2) = pi/3 and
        acos(sqrt(2)/2) = pi/4.
        Tensors with no dimension. A tensor with no dimension (a rank-0 tensor) is
        a tensor whose shape is empty. The result is a rank-0 tensor whose single
        element is the arccosine of the single element of X.
        Zero-sized dimensions. An operand may have a zero-sized dimension. The
        result has the same shape as X and is then empty.

    Function (float):
        [E_ACOS_FLOAT_FUNC_0010]
        Part 1. Operator Acos computes the arccosine of tensor X element-wise and
        stores the result in tensor Y. Each element of Y is the value of the
        arccosine of the corresponding element of X, rounded to the type of X and Y.
        Part 2. For any tensor index i of the result Y:
            Y[i] = round(acos(X[i]))
        where
        - acos is the arccosine of the real numbers, defined in the section for
          type real: for every x in [-1, 1], the unique value y such that
          cos(y) = x and y in [0, pi],
        - round(y) is the value of y rounded to the nearest value of the type of X
          and Y using the roundTiesToEven attribute of IEEE 754, the exponent range
          of the type being unbounded.
        Values of the operands. The operands may be any of the values of the type.
        An operand that is NaN gives a NaN result; that NaN is a quiet NaN whose
        sign and payload are not specified, so that every quiet NaN of the type of
        X and Y is a conforming result. An operand that is +inf or -inf lies
        outside [-1, 1] and is ruled out by the constraint
        [E_ACOS_FLOAT_CONSTR_X_0010]. An operand that is +0 or -0 gives pi/2
        rounded to the type of X and Y: the arccosine of a null value is pi/2
        whatever the sign of the zero, so that +0 and -0 give the same result.
        Values of the result. The result lies in [0, pi], which is within the range
        of the three types: it is never an infinity, and the rounding of the exact
        arccosine to the type of X and Y never overflows. The result is null only
        when the operand is 1, and it is then +0.
        Exactness. The result is the real arccosine rounded to the type of X and Y;
        it is exact only when the real arccosine is representable in that type.
        For instance acos(1) = 0 is exact, whereas acos(0) = pi/2 is representable
        in none of the three types and the result is the nearest value of the type.
        Tensors with no dimension. A tensor with no dimension (a rank-0 tensor) is
        a tensor whose shape is empty. The result is a rank-0 tensor whose single
        element is the arccosine of the single element of X.
        Zero-sized dimensions. An operand may have a zero-sized dimension. The
        result has the same shape as X and is then empty.

    Inputs:
        X: real tensor or floating-point tensor. The value whose arccosine is
        computed.
        Constraints:
        [E_ACOS_REAL_CONSTR_X_0010] Definition domain: Every element of X lies in
        [-1, 1].
        [E_ACOS_FLOAT_CONSTR_X_0010] Definition domain: Every element of X lies in
        [-1, 1].
        [E_ACOS_FLOAT_CONSTR_X_0020] Type consistency: Tensors X and Y have the
        same type.

    Outputs:
        Y: real tensor or floating-point tensor. Tensor Y is the element-wise
        arccosine of X.
        Constraints:
        [E_ACOS_REAL_CONSTR_Y_0010] Shape definition: Tensor Y has the same shape
        as X.
        [E_ACOS_FLOAT_CONSTR_Y_0010] Shape definition: Tensor Y has the same shape
        as X.
        [E_ACOS_FLOAT_CONSTR_Y_0020] Type consistency: see constraint
        [E_ACOS_FLOAT_CONSTR_X_0020] on tensor X.

    Attributes:
        Operator Acos has no attribute.

    Error conditions:
        No error condition. Invalid operation (operand outside [-1, 1]) is ruled
        out by the precondition. NaN operand is nominal: the result is a quiet NaN.
        Overflow is not applicable. Division by zero is not applicable.
    """
    # DECISION: The document specifies the operator for type real and for types
    # float16, float, double. It does not specify behavior for other dtypes
    # (e.g., integers, complex). We use numpy's np.arccos, which for integer
    # inputs promotes to float64. This is a choice not specified by the document.
    #
    # DECISION: The document requires exact real arccosine for the real type and
    # correctly rounded (roundTiesToEven) results for the float types. numpy's
    # np.arccos is used as the implementation; it may not be correctly rounded for
    # all inputs. This is a limitation of the available numpy implementation and
    # is not specified by the document.
    #
    # DECISION: The document says a rank-0 tensor is a tensor whose shape is
    # empty, and the result is a rank-0 tensor. numpy's np.arccos on a 0-d array
    # returns a numpy scalar, not an ndarray. We wrap the result with np.asarray
    # to ensure the returned value is a 0-d ndarray, satisfying the rank-0 tensor
    # requirement. The document does not specify the Python type of the returned
    # tensor, but this choice makes the output an ndarray.
    #
    # DECISION: The document says invalid operation (operand outside [-1, 1]) is
    # ruled out by the precondition and that there is no error condition. It does
    # not specify behavior when the precondition is violated. We do not add
    # validation; we call np.arccos, which for out-of-domain inputs returns NaN
    # and may emit a RuntimeWarning. We do not suppress warnings or errors, as the
    # document does not mention them.
    #
    # DECISION: The document says an operand that is NaN gives a NaN result, and
    # every quiet NaN of the type is conforming. We rely on numpy's NaN
    # propagation through np.arccos.
    #
    # DECISION: The document says +0 and -0 give the same result (pi/2 rounded).
    # We rely on numpy's np.arccos, which returns the same value for +0 and -0.
    #
    # DECISION: The document says X and Y have the same type. We rely on numpy's
    # np.arccos preserving the dtype for float16, float32, and float64 inputs.
    #
    # DECISION: The document says the result has the same shape as X, including
    # zero-sized dimensions. We rely on numpy's np.arccos preserving the shape.
    #
    # DECISION: The document references "General restrictions" but does not
    # include them. We assume they do not affect the implementation beyond what
    # is stated in the provided specification.
    #
    # DECISION: The document says the operator has no attribute. Our function
    # takes only the single operand X, as required.
    return np.asarray(np.arccos(X))
