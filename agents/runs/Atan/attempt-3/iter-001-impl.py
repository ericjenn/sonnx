"""
Implementation of the ONNX Atan operator, as specified in the provided document.

The document defines Atan for real tensors and for floating-point tensors of types
float16, float, and double. The entry point is `atan`.
"""

import numpy as np


def atan(input):
    """
    Computes the element-wise arctangent of the input tensor.

    Quoting the specification:

    [E_ATAN_REAL_FUNC_0010] Part 1. Operator Atan computes the arctangent of every
    element of the input tensor, element-wise, and stores the results in the output
    tensor. The arctangent is the inverse of the tangent: the result is the angle,
    in radians, whose tangent is the element.

    [E_ATAN_FLOAT_FUNC_0010] Part 1. Operator Atan computes the arctangent of every
    element of the input tensor, element-wise, and stores the results in the output
    tensor. The arctangent is the inverse of the tangent: the result is the angle,
    in radians, whose tangent is the element.

    Part 2. For every element input[i] of the input tensor that is not NaN, the
    result output[i] is the value of the type of input and output that is the
    arctangent of input[i]:

        output[i] = round(arctan(input[i]))

    where round(y) is the value of y rounded to the nearest value of the type of
    input and output using the roundTiesToEven attribute of IEEE 754, the exponent
    range of the type being unbounded.

    Special values:
    - an element that is ±0 gives ±0: arctan(+0) = +0 and arctan(-0) = -0, the
      sign of the element being preserved;
    - an element that is ±∞: the case is decided by the formula above, whose
      `round` definition gives the nearest value of the type of input and output
      to ±π/2;
    - an element that is NaN gives a quiet NaN of the type of input and output:
      the result of the operator for such an element is a quiet NaN, and neither
      its sign nor its payload is specified, so that every quiet NaN of the type
      of input and output is a conforming result.

    Shapes: The operator is unary and element-wise: the output tensor has the same
    shape as the input tensor, and each element of the output depends only on the
    element of the input at the same index; no broadcasting applies.

    Tensors with no dimension: A tensor with no dimension (a rank-0 tensor) is a
    tensor whose shape is empty. It has a single element, and the result is the
    arctangent of that element, a tensor with no dimension.

    Zero-sized dimensions: An input tensor may have a zero-sized dimension. It then
    has no element, and the output tensor has the same shape and is empty.

    Error conditions: No error condition. The arctangent is defined for every real
    number and for every value of the floating-point types, including ±∞ and NaN.

    Attributes: Operator Atan has no attribute.

    Inputs: input: real tensor / floating-point tensor. The tensor whose arctangent
    is computed.

    Outputs: output: real tensor / floating-point tensor. The element-wise
    arctangent of the input tensor.

    Constraints:
    - [E_ATAN_REAL_CONSTR_INPUT_0010] Shape definition: The output tensor has the
      same shape as the input tensor.
    - [E_ATAN_FLOAT_CONSTR_INPUT_0010] Shape definition: The output tensor has the
      same shape as the input tensor.
    - [E_ATAN_FLOAT_CONSTR_INPUT_0020] Type consistency: Tensors input and output
      have the same type.
    """
    # DECISION: The document describes two sections: "Atan (real)" and
    # "Atan (float)". The entry point is `atan`. The document does not explicitly
    # state which section the function should implement, nor how to handle inputs
    # that are not one of the floating-point types float16, float, or double.
    # The "real" section is a mathematical description for real numbers, while the
    # "float" section specifies the concrete floating-point types. We implement the
    # operator using numpy's `np.arctan`, which is element-wise, preserves the shape,
    # and for floating-point inputs preserves the dtype (float16, float32, float64).
    # For other dtypes (e.g., integers), the document says "the operator applies to
    # real numbers, not to integers" (Error conditions, Integer overflow), and does
    # not specify behavior; we rely on numpy's default promotion to float64.
    #
    # DECISION: The document specifies that the result is the arctangent rounded to
    # the nearest value of the type using roundTiesToEven. It does not specify a
    # particular algorithm. We use `np.arctan`, which is the standard element-wise
    # arctangent in numpy. For float16 and float32, numpy's implementation may
    # compute in a higher precision and round to the target type; this is consistent
    # with the specification's rounding requirement to the extent that numpy's
    # ufunc provides it.
    #
    # DECISION: The document specifies special values: ±0 -> ±0, ±∞ -> nearest
    # value to ±π/2, NaN -> quiet NaN. `np.arctan` implements these behaviors:
    # it preserves the sign of zero, returns the correctly rounded value for
    # infinity, and propagates NaN. We rely on numpy for these cases.
    #
    # DECISION: The document says "no broadcasting applies". `np.arctan` is a
    # unary element-wise ufunc and does not broadcast. We do not add any
    # broadcasting logic.
    #
    # DECISION: The document says the output has the same shape as the input.
    # `np.arctan` preserves the shape, including rank-0 and zero-sized dimensions.
    # We do not reshape or validate.
    #
    # DECISION: The document says "No error condition." We do not add any
    # validation or error handling beyond what numpy's `np.arctan` naturally does.
    return np.arctan(input)
