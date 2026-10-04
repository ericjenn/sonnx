"""
Implementation of the ONNX Neg operator (opset 14) as specified.

This module implements the Neg operator for the real, float16/float/double,
and int8/int16/int32/int64 type families. The entry point is `neg`.
"""

import numpy as np

# DECISION: The document describes three type families (real, float, int) but does not
# specify how a single entry point should dispatch or whether it should handle all.
# We implement one function `neg` that applies element-wise negation to any numpy array,
# relying on numpy's type-preserving unary minus. The document says for each family:
# "Operator Neg flips the sign of every element of tensor X and stores the result in tensor Y."
# This is the common operation.

# DECISION: The document says "tensor" but does not define the concrete data structure.
# The task requires numpy. We assume numpy arrays and use `np.negative`, which operates
# element-wise and preserves shape and dtype.

# DECISION: The document says "real tensor" but does not specify which numpy dtype
# corresponds to "real". We do not convert; we apply unary minus to the input as-is.
# The document says "The negation is exact: Y[i] is the opposite of X[i] in R, with no
# approximation." We rely on the input's own type and numpy's unary minus, which is
# exact for the types numpy supports.

# DECISION: The document says the sign and payload of a NaN result are not specified.
# We do not attempt to control them; numpy's unary minus produces a NaN, which is
# conforming. The document says "the negation of a NaN is a NaN; the sign and the
# payload of that NaN are not specified, so that every quiet NaN of the type of X and Y
# is a conforming result."

# DECISION: The document states no error conditions. We perform no validation and do
# not suppress any warnings that numpy may emit for integer overflow, as the document
# does not mention warnings. The document says "No error condition."

# DECISION: The document specifies that Y has the same shape as X. We rely on numpy's
# unary minus preserving shape, including rank-0 and zero-sized dimensions. The document
# says "Tensor Y has the same shape as X." and "Tensors with no dimension... Y is the
# rank-0 tensor..." and "Zero-sized dimensions... Y is empty."

# DECISION: The document specifies that X and Y have the same type. We rely on numpy's
# unary minus preserving dtype for signed ints and floats. The document says "Tensors X
# and Y have the same type."

# DECISION: The document says the operator has no attributes. Our function takes only
# the input tensor. The document says "Operator Neg has no attribute."

# DECISION: The document references general restrictions but does not include them.
# We implement only the specific Neg semantics described here. The document says
# "General restrictions are applicable." but does not provide them.

# DECISION: The document says no specific restrictions apply. We do not restrict input
# values or shapes. The document says "No specific restrictions apply to the Neg operator."

# DECISION: The document provides examples but does not require special handling. We do
# not special-case them. The document says "The effect of the operator is illustrated on
# the following example."

# DECISION: The document is based on ONNX opset 14. We implement the semantics as
# described. The document says "Based on ONNX documentation Neg version 14."

# DECISION: The document names the input X and the output Y. We use X as the parameter
# name and return the result. The document says "Inputs: X: ..." and "Outputs: Y: ...".

# DECISION: The document specifies special floating-point values: +0 -> -0, -0 -> +0,
# +inf -> -inf, -inf -> +inf. We rely on numpy's IEEE 754 negation for these. The
# document says "the negation of +0 is -0, and the negation of -0 is +0; the negation
# of +inf is -inf, and the negation of -inf is +inf".

# DECISION: The document specifies the integer minimum value negates to itself. We rely
# on numpy's two's-complement wrap-around for signed integers, which matches the
# specified modulo 2^n reduction. The document says "The negation of the minimum value...
# The result is then -2^{n-1} itself: for int8, -(-128) gives -128."

# DECISION: The document says the negation is exact. We rely on numpy's unary minus,
# which is exact for the types. The document says "The negation is exact: Y[i] is the
# opposite of X[i] in R, with no approximation." and "The negation of IEEE 754 is exact:
# it changes the sign of the operand and leaves its magnitude unchanged, so that no
# rounding occurs and the result is always representable in the type of X and Y."

# DECISION: The document says invalid operation, overflow, underflow, and division by
# zero are nominal. We do not perform any operation that would raise these. The document
# says "Every condition of the list is part of the nominal behavior of the operator;
# none of them is an error. No error condition."


def neg(X):
    """
    Neg operator.

    This function implements the ONNX Neg operator (opset 14) as specified for the
    real, float16/float/double, and int8/int16/int32/int64 type families.

    The document states for the real type family:
    "Operator Neg flips the sign of every element of tensor X and stores the result
    in tensor Y. Each element of Y is the negation of the element of X that has the
    same index. For any tensor index i of the result Y: Y[i] = -X[i] where -X[i] is
    the negation in R of the element X[i]. The negation is exact: Y[i] is the
    opposite of X[i] in R, with no approximation."

    For the float type family:
    "Operator Neg flips the sign of every element of tensor X according to IEEE 754
    floating-point semantics and stores the result in tensor Y. Each element of Y is
    the negation of the element of X that has the same index. For any tensor index i
    of the result Y: Y[i] = -_(f) X[i] where -_(f) is the negation of the floating-point
    type of X and Y, i.e. -_(f16), -_(f32) or -_(f64) according to that type. The
    negation of IEEE 754 is exact: it changes the sign of the operand and leaves its
    magnitude unchanged, so that no rounding occurs and the result is always
    representable in the type of X and Y. In particular: the negation of +0 is -0,
    and the negation of -0 is +0; the negation of +inf is -inf, and the negation of
    -inf is +inf; the negation of a finite value is the finite value of the opposite
    sign and of the same magnitude, whether that value is normal or subnormal; the
    negation of a NaN is a NaN; the sign and the payload of that NaN are not specified,
    so that every quiet NaN of the type of X and Y is a conforming result."

    For the int type family:
    "Operator Neg flips the sign of every element of tensor X and stores the result
    in tensor Y. Each element of Y is the negation of the element of X that has the
    same index. Tensors X and Y have the same signed n-bit type, and the result is a
    value of that type. For any tensor index i of the result Y: Y[i] = -X[i] if -X[i]
    lies in [-2^{n-1}, 2^{n-1}-1]; Y[i] = -X[i] -_(in) 2^n if -X[i] is greater than
    2^{n-1}-1, where -X[i] is the exact negation in Z of the element X[i], before the
    reduction modulo 2^n, -_(in) is the subtraction of the n-bit signed type: the
    exact difference of its two operands, read as a signed n-bit value, i.e. reduced
    modulo 2^n into [-2^{n-1}, 2^{n-1}-1], and n is the number of bits of the type of
    X and Y. The result is thus the exact negation modulo 2^n, read as a signed value.
    The values of an n-bit signed type lie in [-2^{n-1}, 2^{n-1}-1], so that their exact
    negation lies in [-(2^{n-1}-1), 2^{n-1}]: the second case applies only to the
    minimum value -2^{n-1} of the type, whose negation 2^{n-1} is not a value of the
    type, and the reduction is applied at most once. The negation of the minimum value.
    The range of the type is not symmetric about zero: the negation of the minimum
    value -2^{n-1} is 2^{n-1}, which is not representable in the type. The result is
    then -2^{n-1} itself: for int8, -(-128) gives -128. Tensors X and Y have the same
    type and no wider type is used, so that the value of Y is that value of the type."

    The document also states for all type families:
    "Tensors with no dimension. A tensor with no dimension (a rank-0 tensor) is a
    tensor whose shape is empty. It has a single element, and Y is the rank-0 tensor
    whose single element is the negation of that element."
    "Zero-sized dimensions. An operand may have a zero-sized dimension. The
    corresponding dimension of Y then has size 0, and Y is empty."
    "Operator Neg has no attribute."
    "No error condition."

    The document specifies the shape constraint:
    "Tensor Y has the same shape as X."
    and the type consistency constraint:
    "Tensors X and Y have the same type."

    This implementation uses numpy's unary minus (`np.negative`), which performs
    element-wise negation, preserves shape, and preserves dtype for the supported
    types. It does not add any validation or error handling, as the document states
    no error conditions.
    """
    return np.negative(X)
