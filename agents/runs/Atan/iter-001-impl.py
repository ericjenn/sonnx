"""
Implementation of the ONNX Atan operator (opset 14) as specified in the provided document.

This module implements the Atan operator for floating-point tensors (float16, float, double).
The specification defines the operator in two sections: "Atan (real)" and "Atan (float)".
The float section is more specific for floating-point types and is the one implemented here.

The operator computes the element-wise arctangent of the input tensor. The result is the
angle in radians whose tangent is the input element, lying in the open interval
(-π/2, π/2). For floating-point types, the result is rounded to the nearest value of the
input type using roundTiesToEven.

The implementation uses numpy's arctan. For float16 and float32, the computation is
performed in float64 and then cast back to the original type to better approximate the
correctly rounded result. For float64, numpy's arctan is used directly.
"""

# DECISION: The document specifies the operator for floating-point types (float16, float,
# double). It does not specify behavior for other types (e.g., integers). We do not add
# validation; we apply numpy's arctan, which for integer input returns float64. This is
# based on the statement "Integer overflow: none: the operator applies to floating-point
# types, not to integers" and the absence of any specification for other types.
#
# DECISION: The document requires the result to be rounded to the nearest value of the
# type using roundTiesToEven. For float16 and float32, we compute the arctangent in
# float64 and then cast to the original type, relying on numpy's casting which uses
# roundTiesToEven. This is a choice of implementation to better approximate the correctly
# rounded result; the document does not specify the computation method.
#
# DECISION: For float64, we use numpy's `np.arctan` directly. The document requires
# correct rounding, but numpy's implementation may not be correctly rounded for all
# inputs. We rely on it as the available implementation. This is a point where the
# document's requirement may not be fully met by the implementation.
#
# DECISION: The document states that for NaN input, the result is a quiet NaN and neither
# its sign nor its payload is specified. We use numpy's arctan, which returns a quiet NaN,
# and we do not attempt to control its sign or payload.
#
# DECISION: The document states that for ±0 input, the result is ±0. We rely on numpy's
# arctan preserving the sign of zero.
#
# DECISION: The document states that for +inf and -inf, the result is the nearest value of
# the type to π/2 and -π/2 respectively. We rely on numpy's arctan returning π/2 and -π/2
# in float64, and then casting to the original type yields the nearest value.
#
# DECISION: The document has two sections: "Atan (real)" and "Atan (float)". The float
# section is more specific for floating-point types. We implement the float semantics for
# floating-point inputs, as the operator is defined for float16, float, double. The real
# section is a subset and does not add requirements for floating-point types.
#
# DECISION: The document says "Tensors input and output have the same type." We preserve
# the dtype for float16, float32, and float64. For other types, we do not enforce this
# constraint, as the operator is not specified for them.
#
# DECISION: The document says "no broadcasting applies". Our implementation is element-wise
# and does not broadcast. Numpy's arctan is element-wise.
#
# DECISION: The document says "A tensor with no dimension (a rank-0 tensor) ... has a
# single element". Numpy's arctan handles rank-0 arrays and returns a rank-0 array.
#
# DECISION: The document says "An input tensor may have a zero-sized dimension. It then has
# no element, and the output tensor has the same shape and is empty." Numpy's arctan
# handles empty arrays and returns an empty array of the same shape.
#
# DECISION: The document says "Operator Atan has no attribute." Our function takes only the
# input operand.
#
# DECISION: The document says "No error condition." We do not raise any errors.

import numpy as np

def atan(input):
    """
    Computes the element-wise arctangent of the input tensor.

    This function implements the Atan operator for floating-point tensors as specified
    in the document. The relevant part of the specification is quoted below.

    From the "Atan (float)" section, Function, E_ATAN_FLOAT_FUNC_0010:

    Part 1. Operator Atan computes the arctangent of every element of the input tensor,
    element-wise, and stores the results in the output tensor. The arctangent is the
    inverse of the tangent: the result is the angle, in radians, whose tangent is the
    element.

    Part 2. For every element input[i] of the input tensor that is finite, the result
    output[i] is the value of the type of input and output that is the arctangent of
    input[i]:

        output[i] = round(arctan(input[i]))

    where
    - i is any tensor index of input and output,
    - arctan(x) is the unique real y such that tan(y) = x and y ∈ (-π/2, π/2), the
      arctangent of the real numbers,
    - round(y) is the value of y rounded to the nearest value of the type of input and
      output using the roundTiesToEven attribute of IEEE 754, the exponent range of the
      type being unbounded: a magnitude whose nearest value exceeds the largest finite
      value of the type of input and output gives an infinity of the sign of y, and a
      null y gives a null value of the same sign.

    The equation tan(y) = x alone has one solution in each interval
    (-π/2 + kπ, π/2 + kπ), for every integer k; the branch selects one of them, so that
    arctan(x) is defined and unique for every real x. The branch is open: the tangent is
    not defined at ±π/2, and the result is never equal to ±π/2.

    The formula above applies to the elements that are finite: its domain is the real
    numbers, to which neither the infinities nor NaN belong, and arctan is not defined at
    them. The special values are decided by the cases stated below.

    Special values. The arctangent is defined for every value of the type, including the
    special numbers, and the formula above or the case stated here gives the result for
    each of them:

    - an element that is ±0 gives ±0: this is the value of the formula above, whose round
      gives a null result the sign of the element, so that the case restates the
      definition of round and adds nothing to it;
    - an element that is +∞ gives the value of the type of input and output nearest to
      π/2, and an element that is -∞ gives the value of that type nearest to -π/2: the
      arctangent of +∞ is π/2 and the arctangent of -∞ is -π/2, and the result is that
      bound rounded to the type by the round defined above, a finite value;
    - an element that is NaN gives a quiet NaN of the type of input and output: the result
      of the operator for such an element is a quiet NaN, and neither its sign nor its
      payload is specified, so that every quiet NaN of the type of input and output is a
      conforming result.

    Rounding. The arctangent of an element is in general not representable in the type of
    input and output; the result is then the nearest value of the type, as stated above.
    The magnitude of the result is at most the nearest value of the type to π/2, which is
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

    Zero-sized dimensions. An input tensor may have a zero-sized dimension. It then has
    no element, and the output tensor has the same shape and is empty.

    Parameters
    ----------
    input : array_like
        The tensor whose arctangent is computed. It should be a floating-point tensor
        of type float16, float32, or float64.

    Returns
    -------
    output : ndarray
        The element-wise arctangent of the input tensor, with the same shape and type as
        the input for the specified floating-point types.

    Notes
    -----
    The implementation uses numpy's arctan. For float16 and float32, the computation is
    performed in float64 and then cast back to the original type to better approximate
    the correctly rounded result. For float64, numpy's arctan is used directly.

    See the module-level DECISION comments for points where the specification is silent
    or where implementation choices were made.
    """
    x = np.asarray(input)
    if x.dtype == np.float16:
        return np.arctan(x.astype(np.float64)).astype(np.float16)
    elif x.dtype == np.float32:
        return np.arctan(x.astype(np.float64)).astype(np.float32)
    else:
        return np.arctan(x)
