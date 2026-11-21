import numpy as np


def neg(x):
    """Element-wise negation of a tensor, implementing the ONNX ``Neg`` operator.

    The real-number semantics are specified by ``E_NEG_REAL_FUNC_0010``:

        "Operator Neg negates tensor X element-wise and stores the result in
        tensor Y. Each element of Y is the opposite of the element of X that
        has the same index."

    and, for every tensor index ``i`` of ``Y``,

        "Y[i] = -X[i]"

    with the additional statement that "The negation is exact".

    The floating-point semantics are specified by ``E_NEG_FLOAT_FUNC_0010``:

        "Operator Neg negates tensor X element-wise according to IEEE 754
        floating-point semantics and stores the result in tensor Y."

    and, for every tensor index ``i`` of ``Y``,

        "Y[i] = -(f) X[i]"

    where "-(f)" is the negation of the floating-point type of ``X``: "the
    result is the value of that type whose magnitude is the magnitude of X[i]
    and whose sign is the opposite of the sign of X[i]."  In particular:
    ``+0`` becomes ``-0``, ``-0`` becomes ``+0``, ``+inf`` becomes ``-inf``,
    ``-inf`` becomes ``+inf``, and a NaN remains a NaN.  The document says of a
    NaN operand that "Its sign and its payload are not specified".

    The signed-integer semantics are specified by ``E_NEG_INT_FUNC_0010``:

        "For any tensor index i of the result Y:

            Y[i] = -X[i]      if X[i] > -2**(n-1)
            Y[i] = -2**(n-1)  if X[i] = -2**(n-1)"

    and "The result is thus the value of the type that represents -X[i]
    modulo 2**n, read as a signed value."  Thus the minimum value of the
    signed type is its own negation: for ``int8``, ``-(-128)`` is ``-128``.

    The output shape constraints are:

        "[E_NEG_REAL_CONSTR_Y_0010] Shape definition
         Statement: Tensor Y has the shape of X."

        "[E_NEG_FLOAT_CONSTR_Y_0010] Shape definition
         Statement: Tensor Y has the shape of X."

        "[E_NEG_INT_CONSTR_Y_0010] Shape definition
         Statement: Tensor Y has the shape of X."

    so the returned array has exactly the shape of the input tensor.  A rank-0
    input remains rank-0, and zero-sized dimensions remain zero-sized.

    The operator has no attributes:

        "Operator Neg has no attribute."

    Therefore this function takes exactly one positional operand and no
    keyword attribute parameters.
    """
    # DECISION: The document specifies the operator over tensors and says
    # "Tensors X and Y have the same type" and "Tensor Y has the shape of X",
    # but it does not define a Python input-conversion rule for array-like
    # objects.  We treat ``np.asarray(x)`` as the input tensor X, preserving its
    # inferred dtype and shape, and return a NumPy array of that same dtype and
    # shape.
    x = np.asarray(x)

    if x.dtype.kind == "i":
        # E_NEG_INT_FUNC_0010: "The result is thus the value of the type that
        # represents -X[i] modulo 2**n, read as a signed value."  Performing the
        # two's-complement negation in the corresponding unsigned dtype gives
        # exactly that modulo 2**n result, including the specified second case
        # for the minimum value: for int8, -(-128) is -128.
        #
        # DECISION: The document gives the integer semantics for the signed
        # types int8, int16, int32 and int64 only.  We apply the same modulo
        # two's-complement rule to any NumPy signed integer dtype, since the
        # document's integer rule is stated in terms of the signed n-bit type.
        unsigned_dtype = np.dtype(x.dtype.str.replace("i", "u"))
        u = x.view(unsigned_dtype)
        y_u = np.subtract(np.zeros_like(u), u)
        return y_u.view(x.dtype)

    # E_NEG_FLOAT_FUNC_0010: "Y[i] = -(f) X[i]" where the result has "the
    # magnitude of X[i]" and "the sign opposite of the sign of X[i]".  NumPy's
    # floating-point negation flips the sign bit, so it implements +0 -> -0,
    # -0 -> +0, +inf -> -inf, -inf -> +inf, and leaves the magnitude of finite
    # values unchanged.
    #
    # DECISION: The document says for a NaN operand that "Its sign and its
    # payload are not specified, so that every quiet NaN of the type of X is a
    # conforming result."  We therefore delegate NaN handling to NumPy's
    # negation, which returns a NaN, rather than choosing a particular sign or
    # payload.
    #
    # DECISION: The document's first section gives the general real-number
    # semantics "Y[i] = -X[i]" with exact negation.  For real dtypes not
    # explicitly listed as float16/float/double (if any are supplied), NumPy's
    # negation is used as the most direct realization of that same rule; the
    # document says nothing about any additional Python-level type dispatch.
    return np.negative(x)
