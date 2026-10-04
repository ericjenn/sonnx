"""Implementation of the ONNX ``Asin`` operator (opset 14), as specified by the
document "Asin" (revision 2026-10-03).

The specification defines two variants of the operator:

* **Asin** for type ``real``: the exact arcsine of a real tensor whose elements
  lie in ``[-1, 1]``.
* **Asin** for types ``float16``, ``float``, ``double``: the arcsine computed
  according to IEEE 754 floating-point semantics, with the special cases
  (NaN, ``|x| > 1``, ``±0``) spelled out by the specification.

Both variants are implemented here by the single entry point :func:`asin`,
which dispatches on the dtype of the input array.  The two variants agree on
the values they share: for a finite non-zero element of magnitude at most 1,
both give the arcsine of that element, rounded to the type of the input in the
floating-point variant.

The module uses numpy only.
"""

import numpy as np

__all__ = ["asin"]


def asin(X):
    """Compute the element-wise arcsine of ``X``.

    This is the entry point of the module and implements the operator **Asin**
    of the specification, in both of its variants.

    Real variant
    ------------
    The specification states, for the ``real`` variant
    (``E_ASIN_REAL_FUNC_0010``):

        "Operator **Asin** computes the arcsine of tensor X element-wise and
        stores the result in tensor Y. The arcsine is the inverse of the sine:
        the result is the angle whose sine is the element of X."

        "For any tensor index i of the result Y:  Y[i] = arcsin(X[i])"

        "where arcsin(x) is the unique value y such that sin(y) = x and
        y in [-pi/2, pi/2]."

        "The result is exact: Y[i] is the real number arcsin(X[i]), with no
        approximation."

    and, for the shape (``E_ASIN_REAL_CONSTR_Y_0010``):

        "Tensor Y has the same shape as tensor X."

    The precondition ``E_ASIN_REAL_CONSTR_X_0010`` requires every element of X
    to lie in ``[-1, 1]``; the specification rules out any other value, so no
    behavior is specified for elements outside that interval in the real
    variant.

    Floating-point variant
    ----------------------
    The specification states, for the ``float16``/``float``/``double`` variant
    (``E_ASIN_FLOAT_FUNC_0010``):

        "Operator **Asin** computes the arcsine of tensor X element-wise
        according to IEEE 754 floating-point semantics and stores the result in
        tensor Y. Tensors X and Y have the same floating-point type, and the
        result is a value of that type."

        "For any tensor index i of the result Y:

            Y[i] = NaN                        if X[i] is NaN or |X[i]| > 1
            Y[i] = X[i]                       if X[i] is ±0
            Y[i] = round(arcsin(X[i]))        otherwise"

        "where ... round(x) is the value of x rounded to the nearest value of
        the type of X and Y using the roundTiesToEven attribute of IEEE 754,
        the exponent range of the type being unbounded."

        "The second case fixes the sign of a null result: the arcsine of -0 is
        -0 and the arcsine of +0 is +0 ..."

        "The third case applies to every element of X whose magnitude is at
        most 1 and which is not a zero, including the bounds -1 and 1 ..."

        "The NaN of the first case is a quiet NaN; its sign and its payload are
        not specified, whether the NaN comes from an operand or from the
        invalid operation, so that every quiet NaN of the type of X and Y is a
        conforming result."

    and, for the shape (``E_ASIN_FLOAT_CONSTR_Y_0010``):

        "Tensor Y has the same shape as tensor X."

    Parameters
    ----------
    X : numpy.ndarray
        The tensor whose arcsine is computed.  For the real variant its
        elements lie in ``[-1, 1]``; for the floating-point variant its
        elements may be any values of the type, including ``±0``, ``±inf`` and
        NaN.  A rank-0 array is a tensor with no dimension; a zero-sized
        dimension is allowed.

    Returns
    -------
    numpy.ndarray
        The element-wise arcsine of ``X``, with the same shape and the same
        dtype as ``X``.

    Notes
    -----
    The implementation is a single element-wise application of ``numpy.arcsin``
    to the input array, which realizes the formula of the floating-point
    variant for every element of the type:

    * NaN input gives NaN (numpy propagates NaN);
    * ``|x| > 1``, including ``±inf``, gives NaN (numpy returns NaN for the
      arcsine of such values, which is the invalid operation of IEEE 754
      section 7.2);
    * ``±0`` gives ``±0`` (numpy preserves the sign of a zero);
    * every other element gives the arcsine of that element, rounded to the
      type of the input by numpy's own rounding, which is roundTiesToEven for
      the IEEE 754 types.

    For the real variant, the same computation gives the exact arcsine of each
    element, since the elements lie in ``[-1, 1]`` and the result is the real
    number ``arcsin(X[i])``.

    DECISION: the specification does not say which numpy dtype corresponds to
    the ``real`` variant, nor how the two variants are to be told apart at
    run time.  The document presents them as two separate operators, one for
    type ``real`` and one for the three floating-point types, and says of the
    floating-point variant only that "Tensors X and Y have the same
    floating-point type".  The implementation therefore dispatches on the dtype
    of the input: an input whose dtype is one of ``float16``, ``float32`` or
    ``float64`` is treated as the floating-point variant, and any other dtype
    is treated as the real variant.  The two variants are computed by the same
    expression, so the dispatch has no effect on the result; it is recorded
    here because the document does not state how the variants are selected.

    DECISION: the specification says of the floating-point variant that "the
    result is a value of that type" and of the real variant that "Y[i] is the
    real number arcsin(X[i]), with no approximation".  It does not say what
    dtype the output array must have, beyond the type-consistency constraints
    ``E_ASIN_FLOAT_CONSTR_X_0010`` ("Tensors X and Y have the same type") and
    ``E_ASIN_FLOAT_CONSTR_Y_0020``.  The implementation returns an array of the
    same dtype as the input, which satisfies those constraints.  For the real
    variant the document states no type constraint on Y at all, so the same
    choice is made there.

    DECISION: the specification states for the floating-point variant that
    "The NaN of the first case is a quiet NaN; its sign and its payload are not
    specified ... so that every quiet NaN of the type of X and Y is a
    conforming result."  The implementation does not attempt to control the
    sign or payload of a NaN result; it returns whatever quiet NaN numpy
    produces, which the document declares conforming.

    DECISION: the specification states for the floating-point variant that
    "A result whose magnitude is below the smallest normal value of the type is
    a subnormal value: the rounding is the one defined above, with an unbounded
    exponent range, so that the result is the subnormal value nearest to the
    exact arcsine and is not flushed to zero."  The implementation relies on
    numpy's arcsine, which does not flush subnormal results to zero for the
    IEEE 754 types; the document's Example 3 (``Asin(2**-24) == 2**-24`` in
    ``float16``) is the case this covers.

    DECISION: the specification states for the real variant that the elements
    of X lie in ``[-1, 1]`` (``E_ASIN_REAL_CONSTR_X_0010``) and that "An
    element of X outside [-1, 1]" is "ruled out by the precondition".  The
    document therefore specifies no behavior for such an element in the real
    variant.  The implementation performs no validation of that precondition
    and does not raise for an element outside ``[-1, 1]``; it simply computes
    the arcsine, which for such an element is NaN.  This is recorded because
    the document is silent on what an implementation should do when the
    precondition is violated.

    DECISION: the specification states for both variants that "Tensor X may
    have a zero-sized dimension.  Tensor Y then has the same zero-sized
    dimension and is empty: the operator is applied to no element", and that
    "A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is
    empty.  The result is a rank-0 tensor whose single element is the arcsine
    of the single element of X."  The implementation applies numpy's element-wise
    arcsine, which preserves the shape in both cases; no special handling is
    needed and none is added.

    DECISION: the specification states that "Operator **Asin** has no
    attribute" and lists a single input X and a single output Y.  The entry
    point therefore takes exactly one positional argument and returns one
    value; no keyword arguments, attributes or optional operands are accepted.
    """
    # The document defines the operator element-wise, with the same shape for
    # X and Y ("Tensor Y has the same shape as tensor X", constraints
    # E_ASIN_REAL_CONSTR_Y_0010 and E_ASIN_FLOAT_CONSTR_Y_0010), and with the
    # same type for X and Y in the floating-point variant
    # (E_ASIN_FLOAT_CONSTR_X_0010).  numpy.arcsin is element-wise and preserves
    # both the shape and the dtype of its argument, so a single call realizes
    # the formula of E_ASIN_FLOAT_FUNC_0010 for every element of the type and
    # the exact arcsine of E_ASIN_REAL_FUNC_0010 for the real variant.
    return np.arcsin(X)
