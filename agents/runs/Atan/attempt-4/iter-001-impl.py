"""
Implementation of the ONNX Atan operator, version 7, as specified in the provided document.

The operator computes the element-wise arctangent of the input tensor.
"""

import numpy as np


def atan(input):
    """
    Computes the element-wise arctangent of the input tensor.

    This function implements the Atan operator as specified in the document.

    For real tensors (E_ATAN_REAL_FUNC_0010):
    Part 1. Operator Atan computes the arctangent of every element of the input tensor,
    element-wise, and stores the results in the output tensor. The arctangent is the
    inverse of the tangent: the result is the angle, in radians, whose tangent is the
    element.

    Part 2. For every element input[i] of the input tensor, the result output[i] is the
    unique value y such that

        tan(y) = input[i]  and  y in (-pi/2, pi/2)

    where i is any tensor index of input and output, tan is the tangent of the real
    numbers, its argument being an angle in radians, and (-pi/2, pi/2) is the branch on
    which it is strictly increasing and takes every real value exactly once.

    The equation tan(y) = x alone has one solution in each interval
    (-pi/2 + k*pi, pi/2 + k*pi), for every integer k; the branch selects one of them, so
    that the result is defined and unique for every real x. The branch is open: the
    tangent is not defined at +-pi/2, and the result is never equal to +-pi/2.

    Domain and range. The domain of the operator is R: the tangent takes every real value
    on the branch, so that the equation has a solution for every element of the input
    tensor. The range is the open interval (-pi/2, pi/2).

    Sign. The tangent is an odd function, so the arctangent is odd: output[i] = -output[j]
    whenever input[i] = -input[j]. In particular, the arctangent of 0 is 0.

    Shapes. The operator is unary and element-wise: the output tensor has the same shape
    as the input tensor, and each element of the output depends only on the element of the
    input at the same index; no broadcasting applies.

    Tensors with no dimension. A tensor with no dimension (a rank-0 tensor) is a tensor
    whose shape is empty. It has a single element, and the result is the arctangent of
    that element, a tensor with no dimension.

    Zero-sized dimensions. An input tensor may have a zero-sized dimension. It then has no
    element, and the output tensor has the same shape and is empty.

    For floating-point tensors (E_ATAN_FLOAT_FUNC_0010):
    Part 1. Operator Atan computes the arctangent of every element of the input tensor,
    element-wise, and stores the results in the output tensor. The arctangent is the
    inverse of the tangent: the result is the angle, in radians, whose tangent is the
    element.

    Part 2. For every element input[i] of the input tensor that is finite, the result
    output[i] is the value of the type of input and output that is the arctangent of
    input[i]:

        output[i] = round(arctan(input[i]))

    where i is any tensor index of input and output, arctan(x) is the unique real y such
    that tan(y) = x and y in (-pi/2, pi/2), the arctangent of the real numbers, and
    round(y) is the value of y rounded to the nearest value of the type of input and
    output using the roundTiesToEven attribute of IEEE 754, the exponent range of the
    type being unbounded: a magnitude whose nearest value exceeds the largest finite value
    of the type of input and output gives an infinity of the sign of y, and a null y gives
    a null value of the same sign.

    The equation tan(y) = x alone has one solution in each interval
    (-pi/2 + k*pi, pi/2 + k*pi), for every integer k; the branch selects one of them, so
    that arctan(x) is defined and unique for every real x. The branch is open: the tangent
    is not defined at +-pi/2, and the result is never equal to +-pi/2.

    The formula above applies to the elements that are finite: its domain is the real
    numbers, to which neither the infinities nor NaN belong, and arctan is not defined at
    them. The special values are decided by the cases stated below.

    Special values. The arctangent is defined for every value of the type, including the
    special numbers, and the formula above or the case stated here gives the result for
    each of them:

    - an element that is +-0 gives +-0: this is the value of the formula above, whose
      round gives a null result the sign of the element, so that the case restates the
      definition of round and adds nothing to it;
    - an element that is +inf gives the value of the type of input and output nearest to
      pi/2, and an element that is -inf gives the value of that type nearest to -pi/2: the
      arctangent of +inf is pi/2 and the arctangent of -inf is -pi/2, and the result is
      that bound rounded to the type by the round defined above, a finite value;
    - an element that is NaN gives a quiet NaN of the type of input and output: the result
      of the operator for such an element is a quiet NaN, and neither its sign nor its
      payload is specified, so that every quiet NaN of the type of input and output is a
      conforming result.

    Rounding. The arctangent of an element is in general not representable in the type of
    input and output; the result is then the nearest value of the type, as stated above.
    The magnitude of the result is at most the nearest value of the type to pi/2, which is
    below the largest finite value of every type of this section, so that no result is an
    infinity and no overflow occurs. A result whose magnitude is below the smallest normal
    value of the type is a subnormal value or a zero; this is the nominal behavior of the
    operator.

    Shapes. The operator is unary and element-wise: the output tensor has the same shape
    as the input tensor, and each element of the output depends only on the element of the
    input at the same index; no broadcasting applies.

    Tensors with no dimension. A tensor with no dimension (a rank-0 tensor) is a tensor
    whose shape is empty. It has a single element, and the result is the arctangent of
    that element, a tensor with no dimension.

    Zero-sized dimensions. An input tensor may have a zero-sized dimension. It then has no
    element, and the output tensor has the same shape and is empty.

    Constraints:
    - E_ATAN_REAL_CONSTR_INPUT_0010: The output tensor has the same shape as the input
      tensor.
    - E_ATAN_FLOAT_CONSTR_INPUT_0010: The output tensor has the same shape as the input
      tensor.
    - E_ATAN_FLOAT_CONSTR_INPUT_0020: Tensors input and output have the same type.

    Attributes: Operator Atan has no attribute.

    Error conditions: No error condition.
    """
    # DECISION: The document specifies the operator for real tensors and for floating-point
    # tensors of types float16, float, and double. It does not specify the behavior for
    # integer tensors; the error conditions state "Integer overflow | none: the operator
    # applies to real numbers, not to integers". We read this as meaning integer tensors
    # are not a supported input type. However, the document does not require validation,
    # so we do not reject integer inputs. We rely on numpy's arctan, which for integer
    # inputs returns a float64 array. This is a point where the document is silent.
    #
    # DECISION: The document requires that for floating-point types, the output has the
    # same type as the input (constraint E_ATAN_FLOAT_CONSTR_INPUT_0020). We rely on
    # numpy's arctan, which preserves the dtype of floating-point inputs (float16,
    # float32, float64). The document does not specify the algorithm for rounding; we
    # assume numpy's implementation is a faithful realization of the specified
    # mathematical function.
    #
    # DECISION: The document specifies special values: +-0 gives +-0, +-inf gives the
    # nearest value of the type to +-pi/2, and NaN gives a quiet NaN. We rely on numpy's
    # arctan to implement these cases. The document does not specify the sign or payload
    # of the NaN result; numpy returns a quiet NaN.
    #
    # DECISION: The document specifies that the output has the same shape as the input,
    # including rank-0 tensors and zero-sized dimensions. We rely on numpy's arctan,
    # which is an element-wise unary ufunc and preserves shape.
    #
    # DECISION: The document says "no broadcasting applies". Since the operator is unary,
    # there is no broadcasting to consider. We rely on numpy's arctan, which is unary.
    return np.arctan(input)
