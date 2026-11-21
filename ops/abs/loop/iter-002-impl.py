"""Numpy implementation of the ``Abs`` operator (ONNX opset 14).

The specification is split into four sections, one per type family, and this
module mirrors that split:

* the real numbers (``E_ABS_REAL_FUNC_0010``),
* the IEEE 754 types ``float16``, ``float`` (``float32``) and ``double``
  (``float64``) (``E_ABS_FLOAT_FUNC_0010``),
* the signed integer types ``int8``, ``int16``, ``int32`` and ``int64``
  (``E_ABS_INT_FUNC_0010``),
* the unsigned integer types ``uint8``, ``uint16``, ``uint32`` and ``uint64``
  (``E_ABS_UINT_FUNC_0010``).

The operator has no attribute ("Operator **Abs** has no attribute."), so the
entry point takes the single operand ``X`` as its only positional argument.
"""

import numpy as np

__all__ = ["abs"]

# Bit that turns a NaN of the given IEEE 754 binary format into a quiet NaN
# (see "A NaN operand gives a quiet NaN result" in [E_ABS_FLOAT_FUNC_0010]).
#   binary16: bit 9, binary32: bit 22, binary64: bit 51.
_QUIET_BIT = {
    2: 0x0200,
    4: 0x00400000,
    8: 0x0008000000000000,
}


def abs(X):
    """Return the element-wise absolute value of the tensor ``X``.

    The signature of the operator is

        Y = Abs(X)

    "Operator **Abs** computes the absolute value of tensor X element-wise and
    stores the result in tensor Y" ([E_ABS_REAL_FUNC_0010], [E_ABS_FLOAT_
    FUNC_0010], [E_ABS_INT_FUNC_0010], [E_ABS_UINT_FUNC_0010]).  The operator
    has no attribute, so this entry point accepts the operand as its only
    positional argument and no keyword parameter.

    The result always has the shape of ``X`` ("Tensor Y has the shape of X",
    [E_ABS_*_CONSTR_Y_0010]) and the same type ("Tensors X and Y have the same
    type", [E_ABS_*_CONSTR_X_0010]).  "A tensor with no dimension (a rank-0
    tensor) has a single element.  The result is a rank-0 tensor whose single
    element is the absolute value of that element."  and "An operand may have
    a zero-sized dimension.  The corresponding dimension of Y then has size 0
    as well, and Y is empty." -- both fall out of numpy without special code.

    The behaviour depends on the type of ``X``, which the document divides
    into the four families above; dispatch is on ``numpy.dtype.kind``.

    DECISION: the document describes only real, floating-point, signed integer
    and unsigned integer tensors.  It says nothing about booleans, complex
    numbers or any other element type, so those are delegated unchanged to
    ``numpy.abs`` (which, for example, returns a real floating-point tensor
    for a complex operand, a case the document does not cover).

    DECISION: the document says nothing about the byte order of an element
    type.  A non-native operand is therefore converted to native byte order
    first, so that the bit-level manipulations in the helpers below are
    correct; its kind, width, shape and values are unchanged.
    """
    x = np.asarray(X)
    if not x.dtype.isnative:
        x = x.astype(x.dtype.newbyteorder("="))
    kind = x.dtype.kind
    if kind == "f":
        return _abs_float(x)
    if kind == "i":
        return _abs_signed_int(x)
    if kind == "u":
        return _abs_unsigned_int(x)
    return _abs_other(x)


def _abs_float(x):
    """Absolute value of a ``float16`` / ``float`` / ``double`` tensor.

    Implements [E_ABS_FLOAT_FUNC_0010]:

        "Operator **Abs** computes the absolute value of tensor X element-wise
        according to IEEE 754 floating-point semantics ... |X[i]| is the
        absolute value of X[i] in the type of X, i.e. the operand with its
        sign bit cleared"

    Clearing the sign bit is exactly what ``numpy.abs`` does for floating-point
    data, and it is not an arithmetic operation, so "The absolute value of a
    finite element of X is therefore a finite value of the type of X, exact,
    and the operator introduces neither a rounding nor an overflow nor an
    underflow": ``-0`` gives ``+0`` ("the sign of a null result is positive"),
    ``+inf`` and ``-inf`` both give ``+inf``, subnormals stay subnormal, and
    ``float16``, ``float32`` and ``float64`` are all preserved exactly.

    "A NaN operand gives a quiet NaN result.  Its sign and its payload are not
    specified, so that every quiet NaN of the type of X is a conforming
    result, whether the operand NaN is itself quiet or signalling."  Clearing
    the sign bit alone would leave a *signalling* NaN signalling, so the quiet
    bit of every NaN element is additionally set below; for a NaN that is
    already quiet this changes nothing, and the sign and payload it keeps are
    among the values the document leaves unspecified.

    DECISION: for floating-point formats other than the three named by the
    document (for instance ``longdouble``) the quiet bit is not set, because
    the document does not give the layout of such a type; the sign bit is
    still cleared by ``numpy.abs``.
    """
    y = np.abs(x)
    quiet_bit = _QUIET_BIT.get(x.dtype.itemsize)
    if quiet_bit is not None:
        nan_mask = np.isnan(y)
        if nan_mask.any():
            u_dtype = np.dtype("u{}".format(x.dtype.itemsize))
            bits = y.view(u_dtype)
            y = np.where(nan_mask, bits | u_dtype.type(quiet_bit), bits)
            y = y.view(x.dtype)
    return y


def _abs_signed_int(x):
    """Absolute value of an ``int8`` / ``int16`` / ``int32`` / ``int64`` tensor.

    Implements [E_ABS_INT_FUNC_0010]:

        Y[i] = X[i]             if X[i] >= 0
        Y[i] = -X[i]            if -2**(n-1) < X[i] < 0
        Y[i] = -2**(n-1)        if X[i] == -2**(n-1)

    "The operator maps that value to itself: tensors X and Y have the same
    type, no wider type is used, and the result is the value of the type
    congruent to 2**(n-1) modulo 2**n, which is -2**(n-1)."  For ``int8``,
    "the absolute value of -128 is -128".

    The work is done on the two's-complement bit pattern, read as the unsigned
    type of the same width: for a negative element the magnitude is ``0 - u``
    modulo ``2**n`` ("the value of that type that represents the exact
    absolute value modulo 2**n"), and the resulting bit pattern is read back
    as the signed type of ``X``.  For the minimum value this yields the
    pattern of ``2**(n-1)``, whose signed reading is again the minimum, which
    is the third case above.  Non-negative elements are copied unchanged
    ("The absolute value of 0 is 0, given by the first case.").

    DECISION: the document names only ``int8``, ``int16``, ``int32`` and
    ``int64``.  A signed integer type of another width (say ``intp`` on an
    unusual platform) is not described, and is delegated to ``numpy.abs``.
    """
    itemsize = x.dtype.itemsize
    if itemsize not in (1, 2, 4, 8):
        return _abs_other(x)
    u_dtype = np.dtype("u{}".format(itemsize))
    ux = x.view(u_dtype)
    magnitude = np.where(x < 0, u_dtype.type(0) - ux, ux)
    return magnitude.view(x.dtype)


def _abs_unsigned_int(x):
    """Absolute value of a ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` tensor.

    Implements [E_ABS_UINT_FUNC_0010]:

        "Y[i] = X[i]"

    "The operator is the identity on the unsigned types.  Every value of an
    n-bit unsigned type is non-negative, so that its absolute value is the
    value itself, and no element of X is changed: Y has the same shape, the
    same type and the same contents as X.  No result can leave the range of
    the type, no rounding takes place, and no value of the type is mapped to a
    value of the opposite sign."

    DECISION: the document says that Y has "the same shape, the same type and
    the same contents as X" but says nothing about whether Y may be the very
    array X.  A fresh array is returned, so that the output never aliases the
    operand.
    """
    return x.copy()


def _abs_other(x):
    """Element types the document does not describe.

    DECISION: the document covers only real, floating-point, signed integer
    and unsigned integer tensors, and says nothing about booleans, complex
    numbers or any other element type.  Such operands are passed to
    ``numpy.abs``, which is the closest reading of "computes the absolute
    value of tensor X element-wise" available for a type the document does not
    define.
    """
    return np.abs(x)
