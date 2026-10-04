"""
Implementation of the ONNX Neg operator (opset 14) as specified in the provided document.

The operator negates a tensor element-wise. The document defines three type families:
- real
- float16, float, double
- int8, int16, int32, int64

The entry point is `neg(X)`, which returns the element-wise negation of X.
"""

import numpy as np


def neg(X):
    """
    Neg operator.

    Signature:
        Y = Neg(X)

    where
    - X: tensor whose elements are negated
    - Y: result of the element-wise negation of X

    Function (real):
        Operator Neg negates tensor X element-wise and stores the result in tensor Y.
        Each element of Y is the opposite of the element of X that has the same index.
        For any tensor index i of the result Y:
            Y[i] = -X[i]
        where
        - X[i] is the element of X at index i,
        - -X[i] is the negation of X[i] in R.
        The negation is exact: Y[i] is the opposite of X[i] in R, with no approximation.

    Function (float):
        Operator Neg negates tensor X element-wise according to IEEE 754 floating-point
        semantics and stores the result in tensor Y. Each element of Y is the opposite
        of the element of X that has the same index.
        For any tensor index i of the result Y:
            Y[i] = -_(f) X[i]
        where
        - X[i] is the element of X at index i,
        - -_(f) is the negation of the floating-point type of X, i.e. -_(f16), -_(f32)
          or -_(f64) according to that type: the result is the value of that type whose
          magnitude is the magnitude of X[i] and whose sign is the opposite of the sign
          of X[i].
        The negation is exact: it changes the sign of the operand and leaves its magnitude
        unchanged, so that no rounding occurs and the result is a value of the type of X
        for every operand.

    Function (int):
        Operator Neg negates tensor X element-wise and stores the result in tensor Y.
        Each element of Y is the opposite of the element of X that has the same index.
        Tensors X and Y have the same signed n-bit type, and the result is a value of
        that type.
        For any tensor index i of the result Y:
            Y[i] = -X[i] if X[i] > -2^{n-1}
            Y[i] = -2^{n-1} if X[i] = -2^{n-1}
        where
        - X[i] is the element of X at index i and -X[i] its negation in Z,
        - n is the number of bits of the type of X.
        The result is thus the value of the type that represents -X[i] modulo 2^n,
        read as a signed value.

    Shape:
        Tensor Y has the shape of X.
        A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty
        and which holds a single element. The operator applies to that element, and Y
        is a rank-0 tensor.
        An operand may have a zero-sized dimension. The corresponding dimension of Y
        then has size 0, and Y is empty.

    Type consistency:
        Tensors X and Y have the same type.

    Attributes:
        Operator Neg has no attribute.

    Error conditions:
        No error condition.
    """
    # DECISION: The document references "[General restrictions](./../common/general_restrictions.md)"
    # but that document is not included in the provided specification. The document says
    # "General restrictions are applicable." Since the referenced document is not part of
    # the provided specification, we do not implement any additional restrictions. This is
    # a point where the specification is incomplete.
    #
    # DECISION: The document does not specify the concrete tensor representation or API.
    # The task instructs to use numpy. We assume the input X is a numpy ndarray (or numpy
    # scalar) of the appropriate dtype, and we return a numpy array/scalar of the same dtype.
    # The document speaks of "tensor" and "element-wise negation" but does not name a
    # concrete tensor library.
    #
    # DECISION: The document specifies only the types real, float16, float, double, int8,
    # int16, int32, int64. It does not specify behavior for other types (e.g., unsigned
    # integers, bool, complex). We do not add validation or special handling; we simply
    # apply numpy's unary negation, which is the element-wise negation for the given input.
    # The document says nothing about other types.
    #
    # DECISION: For floating-point NaN, the document says "Its sign and its payload are
    # not specified, so that every quiet NaN of the type of X is a conforming result."
    # Numpy's unary negation may flip the sign bit and preserve the payload. We do not
    # attempt to canonicalize NaN; any quiet NaN is conforming.
    #
    # DECISION: For signed integer overflow (negation of the minimum value), the document
    # says "the result is the value of the type that represents -X[i] modulo 2^n, read as
    # a signed value." Numpy's unary negation on signed integer arrays wraps modulo 2^n
    # (two's complement). We rely on this behavior. If numpy emits a warning for overflow,
    # we do not suppress it, because the document does not mention warnings and the
    # overflow is nominal.
    #
    # DECISION: The document specifies the mathematical result but not the implementation.
    # We use numpy's unary negation, which implements the specified element-wise negation
    # for all specified types. This covers the exactness of real and floating-point
    # negation, the sign of zero, infinities, NaN, the shape preservation, the dtype
    # preservation, the rank-0 tensor case, the zero-sized dimension case, and the
    # signed integer modulo 2^n wrap. The document does not name a concrete tensor library
    # or implementation, so this is a choice.
    #
    # DECISION: The document says "No error condition." We do not add validation or error
    # handling, because the document specifies no error conditions.
    #
    # DECISION: The document says "Operator Neg has no attribute." We do not implement
    # any attributes.
    #
    # DECISION: The document says "The operator performs no division." We do not perform
    # division.
    #
    # DECISION: The document says "The operator performs none of the operations of IEEE
    # 754 section 7.2." We do not perform any such operations.

    return np.negative(X)
