"""
Atan operator implementation.

This module implements the ONNX Atan operator (opset 14) as specified in the
provided document. The entry point is the function `atan`.

The document contains two sections: Atan (real) and Atan (float). The real
section defines the mathematical arctangent for real numbers. The float section
specifies the concrete behavior for the types float16, float, and double.

The document states that no error conditions apply.
"""

import numpy as np


def atan(input):
    """
    Compute the element-wise arctangent of the input tensor.

    This function implements the Atan operator as specified in the document.

    The document states (Atan (real), E_ATAN_REAL_FUNC_0010):

        Operator Atan computes the arctangent of every element of the input
        tensor, element-wise, and stores the results in the output tensor. The
        arctangent is the inverse of the tangent: the result is the angle, in
        radians, whose tangent is the element.

        For every element input[i] of the input tensor, the result output[i] is
        the unique value y such that

            tan(y) = input[i] and y in (-pi/2, pi/2)

        where i is any tensor index of input and output, tan is the tangent of
        the real numbers, its argument being an angle in radians, and
        (-pi/2, pi/2) is the branch of the tangent on which it is strictly
        increasing and takes every real value exactly once.

        The branch is open: the tangent is not defined at ±pi/2, and the result
        is never equal to ±pi/2.

        Domain and range: The domain of the operator is R: the tangent takes
        every real value on the branch, so that the equation has a solution for
        every element of the input tensor. The range is the open interval
        (-pi/2, pi/2).

        Sign: The tangent is an odd function, so the arctangent is odd:
        output[i] = -output[j] whenever input[i] = -input[j]. In particular,
        the arctangent of 0 is 0.

        Shapes: The operator is unary and element-wise: the output tensor has
        the same shape as the input tensor, and each element of the output
        depends only on the element of the input at the same index; no
        broadcasting applies.

        Tensors with no dimension: A tensor with no dimension (a rank-0 tensor)
        is a tensor whose shape is empty. It has a single element, and the
        result is the arctangent of that element, a tensor with no dimension.

        Zero-sized dimensions: An input tensor may have a zero-sized dimension.
        It then has no element, and the output tensor has the same shape and is
        empty.

    The document states (Atan (float), E_ATAN_FLOAT_FUNC_0010):

        Operator Atan computes the arctangent of every element of the input
        tensor, element-wise, and stores the results in the output tensor. The
        arctangent is the inverse of the tangent: the result is the angle, in
        radians, whose tangent is the element.

        For every element input[i] of the input tensor that is finite, the
        result output[i] is the value of the type of input and output that is
        the arctangent of input[i]:

            output[i] = round(arctan(input[i]))

        where i is any tensor index of input and output, arctan(x) is the
        unique real y such that tan(y) = x and y in (-pi/2, pi/2), the
        arctangent of the real numbers, and round(y) is the value of y rounded
        to the nearest value of the type of input and output using the
        roundTiesToEven attribute of IEEE 754, the exponent range of the type
        being unbounded: a magnitude whose nearest value exceeds the largest
        finite value of the type of input and output gives an infinity of the
        sign of y, and a null y gives a null value of the same sign.

        Special values: The arctangent is defined for every value of the type,
        including the special numbers, and the formula above or the case stated
        here gives the result for each of them:

        - an element that is ±0 gives ±0: this is the value of the formula
          above, whose round gives a null result the sign of the element, so
          that the case restates the definition of round and adds nothing to
          it;
        - an element that is +∞ gives the value of the type of input and output
          nearest to pi/2, and an element that is -∞ gives the value of that
          type nearest to -pi/2: the arctangent of +∞ is pi/2 and the
          arctangent of -∞ is -pi/2, and the result is that bound rounded to
          the type by the round defined above, a finite value;
        - an element that is NaN gives a quiet NaN of the type of input and
          output: the result of the operator for such an element is a quiet
          NaN, and neither its sign nor its payload is specified, so that every
          quiet NaN of the type of input and output is a conforming result.

        Rounding: The arctangent of an element is in general not representable
        in the type of input and output; the result is then the nearest value
        of the type, as stated above. The magnitude of the result is at most
        the nearest value of the type to pi/2, which is below the largest
        finite value of every type of this section, so that no result is an
        infinity and no overflow occurs. A result whose magnitude is below the
        smallest normal value of the type is a subnormal value or a zero; this
        is the nominal behavior of the operator.

        Shapes: The operator is unary and element-wise: the output tensor has
        the same shape as the input tensor, and each element of the output
        depends only on the element of the input at the same index; no
        broadcasting applies.

        Tensors with no dimension: A tensor with no dimension (a rank-0 tensor)
        is a tensor whose shape is empty. It has a single element, and the
        result is the arctangent of that element, a tensor with no dimension.

        Zero-sized dimensions: An input tensor may have a zero-sized dimension.
        It then has no element, and the output tensor has the same shape and is
        empty.

        Type consistency: Tensors input and output have the same type.

    The document states that no error conditions apply.

    Parameters
    ----------
    input : numpy.ndarray
        The tensor whose arctangent is computed. The document specifies
        floating-point tensors of type float16, float32, or float64 (the
        document's `float16`, `float`, and `double`). The document also
        describes a real tensor, but the error conditions state that the
        operator does not apply to integers, and the float section provides the
        concrete type consistency rule.

    Returns
    -------
    numpy.ndarray
        The element-wise arctangent of the input tensor, with the same shape
        and the same type as the input.

    Notes
    -----
    DECISION: The document does not state which section (real or float) applies
    to the Python function. I implement the float section for floating-point
    inputs, preserving the input dtype, because the real section's domain is
    real numbers and the error conditions state the operator does not apply to
    integers; the float section is the concrete specification for the supported
    types.

    DECISION: The document specifies roundTiesToEven rounding. I rely on
    numpy's arctan for the computation. The document does not specify an
    algorithm, and numpy is the required library. This may not achieve correct
    rounding for all inputs, especially for float16, where numpy may compute in
    a higher precision and then round.

    DECISION: The document says a NaN input gives a quiet NaN of the type of
    input and output, with unspecified sign and payload. I return numpy's NaN
    as is; the document says every quiet NaN is a conforming result.

    DECISION: The document says ±0 gives ±0. I rely on numpy's arctan, which
    preserves the sign of zero.

    DECISION: The document says +∞ gives the value of the type nearest to pi/2,
    and -∞ gives the value nearest to -pi/2. I rely on numpy's arctan, which
    returns the correctly rounded bound for the supported types.

    DECISION: The document says the output has the same shape as the input and
    no broadcasting applies. I rely on numpy's element-wise arctan, which
    preserves shape and does not broadcast.

    DECISION: The document says tensors with no dimension and zero-sized
    dimensions are handled. I rely on numpy's shape preservation.

    DECISION: The document says tensors input and output have the same type.
    I rely on numpy's arctan preserving the dtype for floating-point inputs.
    If the input is not a floating-point numpy array, the behavior is outside
    the specified domain; the document does not specify such cases.
    """
    return np.arctan(input)
