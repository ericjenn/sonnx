import numpy as np

# DECISION: The document says "General restrictions are applicable" and points
# to ../global_restrictions.md, but that file is not part of the supplied
# specification. I therefore add no restriction or validation derived from it.
#
# DECISION: The document says "Operator Mul has no attribute." Thus `mul` takes
# no keyword parameters for attributes.
#
# DECISION: The document specifies a `real` type with exact multiplication but
# gives no concrete numpy dtype for it. I dispatch on the numpy dtype: operands
# with the same integer dtype use the signed/unsigned modulo semantics; all
# other operands use numpy multiplication. For floating dtypes this is the
# IEEE 754 behavior of E_MUL_FLOAT_FUNC_0010. For a non-floating real dtype such
# as object/Decimal, np.multiply is used and the document says nothing about how
# such a dtype is represented.
#
# DECISION: The document imposes "Type consistency": "Tensors A, B and C have
# the same type." It says nothing about calls that violate this constraint. I do
# not validate it; if the two operand dtypes differ, I fall through to
# np.multiply and let numpy determine the result dtype.
#
# DECISION: The document says integer overflow is nominal and specifies the
# exact product reduced modulo 2^n. To implement that reduction without relying
# on host signed-overflow behavior, the integer helper computes the product with
# Python integers (object arrays) and applies the modulo explicitly.
#
# DECISION: The document says zero-sized dimensions are compatible with a
# dimension of size 1 and with an equal zero-sized dimension. numpy
# broadcasting already implements exactly those rules; no extra handling is
# added.


def _mul_integer(A, B, dtype):
    """
    Integer multiplication for E_MUL_INT_FUNC_0010 and E_MUL_UINT_FUNC_0010.

    Signed case, quoted from E_MUL_INT_FUNC_0010:
        "Operator Mul multiplies tensor A by tensor B element-wise and stores
        the result in output tensor C. Each element of C is the product of the
        elements of A and B that correspond to it under the broadcasting rules
        given below. Tensors A, B and C have the same signed n-bit type, and
        the result is a value of that type."

        For any tensor index i of the result C:
            C[i] = u if u <= 2^{n-1}-1
                   u - 2^n if u >= 2^{n-1}
            where u = (A_tilde[i] * B_tilde[i]) mod 2^n

        "x mod 2^n is the remainder of the Euclidean division of x by 2^n,
        taken in [0, 2^n-1]".

    Unsigned case, quoted from E_MUL_UINT_FUNC_0010:
        "C[i] = (A_tilde[i] * B_tilde[i]) mod 2^n"

        "The result is thus the unique value of the type of A and B that is
        congruent to the exact product modulo 2^n".

    The exact product is computed in Python integers so that the modulo
    reduction is applied to the mathematical product, not to an already
    overflowed fixed-width intermediate.
    """
    n = dtype.itemsize * 8
    modulus = 1 << n

    a_obj = A.astype(object)
    b_obj = B.astype(object)
    exact_product = a_obj * b_obj

    u = exact_product % modulus

    if dtype.kind == "i":
        high = 1 << (n - 1)
        u = np.where(u >= high, u - modulus, u)

    return u.astype(dtype)


def _mul_float(A, B):
    """
    Floating-point multiplication for E_MUL_FLOAT_FUNC_0010.

    Quoted from E_MUL_FLOAT_FUNC_0010:
        "Operator Mul multiplies tensor A by tensor B element-wise according to
        IEEE 754 floating-point semantics and stores the result in tensor C.
        Each element of C is the floating-point product of the elements of A and
        B that correspond to it under the broadcasting rules given below."

        For any tensor index i of the result C:
            C[i] = A_tilde[i] *_(f) B_tilde[i]
            =
            NaN
                if A_tilde[i] or B_tilde[i] is NaN, or if one of them is a zero
                and the other an infinity;
            sigma * infinity
                if neither is NaN, one of them is an infinity, and the other is
                neither a zero nor NaN;
            round(A_tilde[i] * B_tilde[i])
                otherwise.

        "round(x) is the value of x rounded to the nearest value of the type of
        A and B using the roundTiesToEven attribute of IEEE 754, the exponent
        range of the type being unbounded: a magnitude whose nearest value
        exceeds the largest finite value of the type of A and B gives an
        infinity of the sign of x."

    The numpy element-wise multiply implements the IEEE 754 operations for the
    floating dtypes in the document.
    """
    return np.multiply(A, B)


def mul(A, B):
    """
    Element-wise multiplication of two tensors.

    The document gives the signature:
        C = Mul(A, B)
    where
        A is "value by which B is multiplied",
        B is "value multiplied by A",
        C is "result of the element-wise multiplication of A and B".

    From E_MUL_REAL_FUNC_0010:
        "Operator Mul multiplies tensor A by tensor B element-wise and stores
        the result in tensor C. Each element of C is the product of the
        elements of A and B that correspond to it under the broadcasting rules
        given below."

    Broadcasting, quoted from the same function:
        "Tensors A and B are broadcast to the shape of C following the
        multidirectional (Numpy-style) broadcasting rules: the shapes are
        aligned on their last dimension; a dimension that one operand does not
        have, including the leading dimensions missing from the operand with
        the smaller rank, is treated as a dimension of size 1; and a dimension
        of size 1 is extended to the size of the corresponding dimension of the
        other operand. Tensor C has the resulting shape."

        "A tensor with no dimension (a rank-0 tensor) is a tensor whose shape
        is empty. It is broadcast to the shape of the other operand, so that
        each element of C is the product of the two operands and C has the
        shape of the other operand."

        "An operand may have a zero-sized dimension. Such a dimension is
        compatible with a dimension of size 1 of the other operand, and with an
        equal zero-sized dimension; the corresponding dimension of C then has
        size 0, and C is empty."

    Attributes, quoted from the document:
        "Operator Mul has no attribute."

    Error conditions, quoted from the document:
        "No error condition."

    The floating-point branch implements E_MUL_FLOAT_FUNC_0010; the same-type
    signed and unsigned integer branches implement E_MUL_INT_FUNC_0010 and
    E_MUL_UINT_FUNC_0010 respectively.
    """
    A = np.asarray(A)
    B = np.asarray(B)

    if A.dtype == B.dtype and A.dtype.kind in ("i", "u"):
        return _mul_integer(A, B, A.dtype)

    return _mul_float(A, B)
