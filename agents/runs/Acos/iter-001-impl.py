"""Acos operator (ONNX, opset 14).

This module implements the operator ``Acos`` as specified by the document
"Acos version 7" (revision 2026-10-03, based on ONNX opset 14).  The
document specifies the operator twice: once for the type ``real`` and once
for the types ``float16``, ``float`` and ``double``.

The entry point is :func:`acos`, which takes the single operand ``X`` and
returns the result ``Y``.
"""

import numpy as np

# DECISION: The document states that "[General restrictions](./../common/
# general_restrictions.md) are applicable" and that "No specific
# restrictions apply to the Acos operator", but the text of the general
# restrictions is not reproduced in the provided specification.  Nothing
# from them is implemented here.


def acos(X):
    r"""Element-wise arccosine of the tensor ``X``.

    The document specifies, for the type ``real``:

        **Part 1.** Operator **Acos** computes the arccosine of tensor
        :math:`X` element-wise and stores the result in tensor :math:`Y`.
        The arccosine is the inverse of the cosine: each element of
        :math:`Y` is the angle of :math:`[0, \pi]` whose cosine is the
        corresponding element of :math:`X`.

        **Part 2.** For any tensor index :math:`i` of the result
        :math:`Y`:

        .. math:: Y[i] = \text{acos}(X[i])

        where :math:`\text{acos}` is defined by the following property:
        for every :math:`x` in :math:`[-1, 1]`, :math:`\text{acos}(x)` is
        the unique value :math:`y` such that :math:`\cos(y) = x` and
        :math:`y \in [0, \pi]`.

    and, for the types ``float16``, ``float`` and ``double``:

        **Part 1.** Operator **Acos** computes the arccosine of tensor
        :math:`X` element-wise and stores the result in tensor :math:`Y`.
        Each element of :math:`Y` is the value of the arccosine of the
        corresponding element of :math:`X`, rounded to the type of
        :math:`X` and :math:`Y`.

        **Part 2.** For any tensor index :math:`i` of the result
        :math:`Y`:

        .. math:: Y[i] = \text{round}(\text{acos}(X[i]))

        where :math:`\text{round}(y)` is the value of :math:`y` rounded to
        the nearest value of the type of :math:`X` and :math:`Y` using the
        roundTiesToEven attribute of IEEE 754, the exponent range of the
        type being unbounded.

    The document also states:

    * "The constraint E_ACOS_REAL_CONSTR_X_0010 requires every element of
      :math:`X` to lie in :math:`[-1, 1]`, so that the result is defined
      for every element of :math:`X`.  Every element of :math:`Y` lies in
      :math:`[0, \pi]`."
    * "An operand that is NaN gives a NaN result; that NaN is a quiet NaN
      whose sign and payload are not specified, so that every quiet NaN of
      the type of :math:`X` and :math:`Y` is a conforming result."
    * "An operand that is :math:`+0` or :math:`-0` gives :math:`\pi/2`
      rounded to the type of :math:`X` and :math:`Y`: the arccosine of a
      null value is :math:`\pi/2` whatever the sign of the zero, so that
      :math:`+0` and :math:`-0` give the same result."
    * "The result is null only when the operand is :math:`1`, and it is
      then :math:`+0`."
    * "A tensor with no dimension (a rank-0 tensor) is a tensor whose
      shape is empty.  The result is a rank-0 tensor whose single element
      is the arccosine of the single element of :math:`X`."
    * "An operand may have a zero-sized dimension.  The result has the
      same shape as :math:`X` and is then empty."
    * Constraint E_ACOS_FLOAT_CONSTR_X_0020: "Tensors :math:`X` and
      :math:`Y` have the same type."
    * Constraint E_ACOS_FLOAT_CONSTR_Y_0010: "Tensor :math:`Y` has the
      same shape as :math:`X`."

    Parameters
    ----------
    X : array_like
        The value whose arccosine is computed.  The document calls it a
        tensor; every element of it lies in :math:`[-1, 1]`.

    Returns
    -------
    numpy.ndarray
        The element-wise arccosine of ``X``, with the same shape as ``X``
        and, for the floating-point types, the same type as ``X``.
    """

    # DECISION: The document says that X is a tensor and says nothing about
    # operands that are not already arrays (Python scalars, lists, ...).
    # ``numpy.asarray`` is used to obtain a tensor from the operand; for an
    # operand that is already an array it is the identity.
    x = np.asarray(X)

    # DECISION: The document specifies the result of the operator (the real
    # arccosine, rounded to the type of X and Y) but says nothing about how
    # to compute it.  This module computes it with ``numpy.arccos``.  For
    # the ``float`` and ``double`` types numpy delegates to the platform's
    # ``acosf``/``acos``, which is not guaranteed to be correctly rounded
    # (roundTiesToEven) for every input, whereas the document requires the
    # correctly rounded result; a correctly rounded arccosine is not
    # available here.
    y = np.arccos(x)

    # DECISION: Constraint E_ACOS_FLOAT_CONSTR_X_0020 states that "Tensors X
    # and Y have the same type".  ``numpy.arccos`` preserves the dtype for
    # float16, float32 and float64, so this cast is normally a no-op; it is
    # applied only if numpy were to return a different floating-point type.
    # For float16, if numpy computes in a wider type and rounds back, the
    # result is a double rounding of the exact arccosine, a case the
    # document does not discuss.
    if np.issubdtype(x.dtype, np.floating) and y.dtype != x.dtype:
        y = y.astype(x.dtype)

    # DECISION: The document rules out operands outside [-1, 1] by the
    # preconditions E_ACOS_REAL_CONSTR_X_0010 and E_ACOS_FLOAT_CONSTR_X_0010
    # ("Every element of X lies in [-1, 1]"), so no validation is performed
    # here.  For such an operand numpy returns NaN and emits a
    # RuntimeWarning; the document does not mention that behaviour and the
    # warning is not suppressed.

    # DECISION: The document defines the operator for the type ``real`` and
    # for the types ``float16``, ``float`` and ``double``; it says nothing
    # about other dtypes.  No dtype is imposed: numpy's own type promotion
    # applies (for instance an integer operand yields a float64 result), and
    # a complex operand yields the complex arccosine, which is outside the
    # document's scope.

    # DECISION: The document says that for real numbers "the result is
    # exact".  Floating-point values cannot represent the real arccosine
    # exactly in general, so for floating-point operands the rounding rule
    # of the float section (roundTiesToEven to the type of X and Y) applies;
    # there is no separate exact code path.

    # DECISION: The document says that a NaN operand gives a quiet NaN whose
    # sign and payload are not specified, that +0 and -0 both give pi/2
    # rounded, and that the result is null only for the operand 1 (then +0).
    # ``numpy.arccos`` already has these properties, so no special handling
    # is added.

    # DECISION: The document says that the result has the same shape as X,
    # that it is a rank-0 tensor when X is, and that it is empty when X has
    # a zero-sized dimension.  ``numpy.arccos`` preserves the shape in all
    # these cases, so no special handling is added.

    # DECISION: The document says that the result of a rank-0 operand is a
    # rank-0 tensor.  ``numpy.arccos`` returns a numpy scalar for a 0-d
    # input; ``numpy.asarray`` turns it into a 0-d array, the representation
    # of a rank-0 tensor, and is the identity for every other result.
    return np.asarray(y)
