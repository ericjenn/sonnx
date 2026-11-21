import numpy as np

# DECISION: The document describes tensors and numeric types abstractly and does
# not prescribe a Python representation.  This implementation represents tensors
# as NumPy arrays and relies on NumPy dtypes for the concrete real/float/int/uint
# types, as the task requires a numpy implementation.

# DECISION: The "real" section describes exact real subtraction, but the
# document's concrete real types are the floating-point types whose semantics
# are specified in the float section.  NumPy floating subtraction implements the
# latter.

def sub(A, B):
    """
    Element-wise subtraction of tensor B from tensor A.

    This implements the operator Sub:

        C[i] = A_broadcast[i] - B_broadcast[i]

    as specified in the document.  The document says:

        "Operator Sub subtracts tensor B from tensor A element-wise and stores
        the result in tensor C.  Each element of C is the difference of the
        elements of A and B that correspond to it under the broadcasting rules
        given below."

    Broadcasting follows the document's multidirectional (Numpy-style) rules:

        "Tensors A and B are broadcast to the shape of C following the
        multidirectional (Numpy-style) broadcasting rules: the shapes are
        aligned on their last dimension; a dimension that one operand does not
        have, including the leading dimensions missing from the operand with
        the smaller rank, is treated as a dimension of size 1; and a dimension
        of size 1 is extended to the size of the corresponding dimension of the
        other operand.  Tensor C has the resulting shape."

    A rank-0 tensor is broadcast to the shape of the other operand; a zero-sized
    dimension is compatible with a dimension of size 1 or with an equal
    zero-sized dimension, giving an empty result when appropriate.

    For floating-point types the document specifies IEEE 754 subtraction,
    including NaN for invalid operations, signed infinities for overflow,
    exact subnormal results, and the signed-zero rule:

        "C[i] = A_broadcast[i] -_(f) B_broadcast[i]"

    with the case analysis of E_SUB_FLOAT_FUNC_0010.  NumPy's floating-point
    subtraction implements these IEEE semantics.

    For signed integer types the document specifies the exact integer
    difference reduced modulo 2^n and read back in the same signed n-bit type:

        "The result is thus the exact difference modulo 2^n, read as a signed
        value."

    For unsigned integer types the document specifies the exact integer
    difference reduced modulo 2^n in the same unsigned n-bit type:

        "The result is the value of that type that represents the exact
        difference of the two elements modulo 2^n."

    NumPy's fixed-width integer subtraction on same-dtype operands wraps modulo
    2^n, matching these definitions.

    The document states: "Operator Sub has no attribute."  Therefore this
    function takes no keyword attribute parameters.

    Parameters
    ----------
    A : numpy.ndarray
        The value from which B is subtracted.
    B : numpy.ndarray
        The value subtracted from A.

    Returns
    -------
    numpy.ndarray
        The element-wise result C, with the broadcast shape of A and B.

    """

    # DECISION: The document states shape compatibility as a constraint
    # ("The shapes of A and B are compatible for broadcasting.") but gives no
    # error condition for incompatible shapes.  I do not add a shape check;
    # if the shapes are incompatible, NumPy raises its broadcasting error.

    # DECISION: The document states type consistency as a constraint
    # ("Tensors A, B, and C have the same type.") but gives no error condition
    # for a violation.  I do not add a type check; NumPy may promote mixed
    # dtypes if the constraint is violated.

    # DECISION: For floating-point NaN results the document says that the
    # quiet NaN's sign and payload are not specified.  NumPy supplies a
    # conforming quiet NaN.

    # DECISION: The document declares floating-point invalid operation and
    # overflow nominal and specifies their NaN or infinity results.  It says
    # nothing about runtime warnings; I leave NumPy's default warning behavior
    # untouched and neither suppress nor promote warnings to errors.

    # DECISION: The document requires integer results modulo 2^n in the same
    # fixed-width type, but does not prescribe an implementation.  NumPy's
    # subtraction on same-dtype integer arrays wraps modulo 2^n, so I rely on
    # that behavior.

    result = np.subtract(A, B)

    # Return a tensor even for rank-0 inputs: NumPy can return a scalar for
    # 0-d operands, while the document describes the output as a tensor.
    return np.asarray(result)
