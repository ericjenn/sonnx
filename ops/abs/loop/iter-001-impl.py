"""``Abs`` — element-wise absolute value of a tensor.

This module implements the specification of the ``Abs`` operator, revision
2026-10-03 (based on ONNX opset 14), for the four families of the document:

* "**Abs** operator for type real",
* "**Abs** operator for types ``float16``, ``float``, ``double``",
* "**Abs** operator for types ``int8``, ``int16``, ``int32``, ``int64``",
* "**Abs** operator for types ``uint8``, ``uint16``, ``uint32``, ``uint64``".

The entry point is :func:`abs`, which takes the single operand ``X`` and
returns the single output ``Y``.

Attributes
----------
The document states, for every one of the four families:

    "Operator **Abs** has no attribute."

so the entry point takes exactly one positional parameter and no keyword
parameter.

DECISION: the document says nothing about how a "tensor" is represented in
Python.  numpy arrays are used, and the operand is passed through
``numpy.asarray`` so that array-likes are accepted; the result is returned as
a ``numpy.ndarray`` whose dtype is the dtype of the operand (as required by
the constraints "Tensors X and Y have the same type").

DECISION: the document defines only the four families listed above (real,
float16/float/double, int8..int64, uint8..uint64) and says nothing about any
other dtype (``bool``, complex, object, ...).  For a dtype outside those
families the implementation falls back to ``numpy.abs``, which is an
undocumented extension rather than a reading of the specification.
"""

import numpy as np


def _abs_signed_int(x):
    """Absolute value of a signed ``n``-bit integer tensor, element-wise.

    Specification, ``E_ABS_INT_FUNC_0010``:

        Y[i] = X[i]                     if X[i] >= 0
        Y[i] = -X[i]                    if -2^(n-1) < X[i] < 0
        Y[i] = -2^(n-1)                 if X[i] = -2^(n-1)

    "where ``-X[i]`` is the negation in Z of ``X[i]``, before any reduction"
    and "the third case is the value ``-X[i] = 2^(n-1)`` reduced modulo ``2^n``
    to the type of ``X``, that is ``2^(n-1) - 2^n = -2^(n-1)``".

    The document says (``E_ABS_INT_FUNC_0010``): "The operator maps that value
    to itself: tensors X and Y have the same type, no wider type is used, and
    the result is the value of the type congruent to ``2^(n-1)`` modulo
    ``2^n``".  Both the second and the third case are therefore the exact
    negation reduced modulo ``2^n``, which is exactly two's-complement
    modular negation: for the bit pattern ``u`` of a negative value, the
    result reduced modulo ``2^n`` is ``(2^n - u) mod 2^n``, i.e. the
    negation in the unsigned type of the same width.

    DECISION: the document says nothing about *how* the modular reduction is
    performed.  It is performed here by reinterpreting the bits as the
    unsigned type of the same width and negating in that type, whose
    wraparound is the required reduction modulo ``2^n``.  This reproduces the
    third case without relying on the behaviour of signed overflow and
    without emitting any warning, which is required by "No error condition."
    """

    u_dtype = np.dtype("u%d" % x.dtype.itemsize)
    xc = np.ascontiguousarray(x)
    # Same bits, read as the unsigned type of the same width.
    u = xc.view(u_dtype)
    # Negation in the unsigned type: wraps modulo 2^n, silently.
    negated = np.negative(u)
    # Y[i] = -X[i] (reduced) for X[i] < 0, Y[i] = X[i] for X[i] >= 0.
    magnitude = np.where(xc < 0, negated, u).astype(u_dtype, copy=False)
    # Reinterpret the bits back as the signed type of X (= type of Y).
    return np.ascontiguousarray(magnitude).view(x.dtype)


def abs(X):
    """Element-wise absolute value.

    Implements, for every supported family, the signature of the document:

        Y = Abs(X)

    with ``X``: "tensor whose absolute value is computed" and ``Y``: "tensor
    whose elements are the absolute values of the elements of X".

    Real (``E_ABS_REAL_FUNC_0010``): "Operator **Abs** computes the absolute
    value of tensor X element-wise and stores the result in tensor Y.  For any
    tensor index i of the result Y: Y[i] = |X[i]| = X[i] if X[i] >= 0, -X[i]
    if X[i] < 0", where "|X[i]| is the absolute value in R of X[i]" and
    "-X[i] is the negation in R of X[i]".  "The result is exact".

    Floating point (``E_ABS_FLOAT_FUNC_0010``): "Operator **Abs** computes the
    absolute value of tensor X element-wise according to IEEE 754
    floating-point semantics", where ``|X[i]|`` "is the absolute value of
    X[i] in the type of X, i.e. the operand with its sign bit cleared".  A NaN
    operand gives "a quiet NaN of the type of X", ``+inf`` and ``-inf`` both
    give ``+inf``, ``+0`` and ``-0`` both give ``+0``, and "The operator
    introduces neither a rounding nor an overflow nor an underflow".

    Signed integers (``E_ABS_INT_FUNC_0010``): see :func:`_abs_signed_int`;
    in particular the minimum value ``-2^(n-1)`` is mapped to itself.

    Unsigned integers (``E_ABS_UINT_FUNC_0010``): "Operator **Abs** computes
    the absolute value of tensor X element-wise ... For any tensor index i of
    the result Y: Y[i] = X[i]", and "The operator is the identity on the
    unsigned types. ... Y has the same shape, the same type and the same
    contents as X."

    Shape: for all four families "Tensor Y has the shape of X", "A tensor with
    no dimension (a rank-0 tensor) has a single element.  The result is a
    rank-0 tensor whose single element is the absolute value of that
    element", and "An operand may have a zero-sized dimension.  The
    corresponding dimension of Y then has size 0 as well, and Y is empty".
    All of this follows from the element-wise nature of the computation and
    needs no special code: numpy preserves the shape, including rank 0 and
    zero-sized dimensions.

    Errors: for all four families the document concludes "No error condition",
    so no validation is added and no warning or error is suppressed here.

    DECISION: the document says nothing about whether the unsigned identity
    may return the operand array itself or must return a distinct array.  A
    copy is returned, so that every branch of this function returns an array
    that does not alias the operand.

    DECISION: the document says nothing about the sign or payload of a NaN
    result ("Its sign and its payload are not specified, so that every quiet
    NaN of the type of X is a conforming result").  ``numpy.abs`` clears the
    sign bit and yields a quiet NaN, which is one conforming result.

    DECISION: the document says nothing about dtypes outside the four families
    (``bool``, complex, object, ...); ``numpy.abs`` is applied to them, which
    is not specified behaviour.
    """

    x = np.asarray(X)
    kind = x.dtype.kind

    if kind == "u":
        # E_ABS_UINT_FUNC_0010: "Y[i] = X[i]" — the identity on unsigned types.
        return np.array(x, copy=True)

    if kind == "i":
        # E_ABS_INT_FUNC_0010, including the minimum value mapped to itself.
        return _abs_signed_int(x)

    # E_ABS_REAL_FUNC_0010 and E_ABS_FLOAT_FUNC_0010: the operand with its
    # sign bit cleared (for floats), the exact absolute value (no rounding,
    # no overflow, no underflow).  ``numpy.abs`` on numpy floating dtypes is
    # the sign-bit-clearing operation and preserves the dtype.
    return np.abs(x)
