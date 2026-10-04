"""NumPy implementation of the ONNX **Add** operator.

Written from one document alone:

    ops/add.md, "Revision 2026-10-03: first issue of this specification,
    based on ONNX opset 14."

The document specifies three type families, each under its own tagged
statement of the operator's function; this module implements all three:

  * ``real``          -- tagged statement ``E_ADD_REAL_FUNC_0010``
  * ``float16``/``float``/``double`` -- tagged statement ``E_ADD_FLOAT_FUNC_0010``
  * ``int8``..``uint64``             -- tagged statement ``E_ADD_INT_FUNC_0010``

Together with the shape clauses these statements carry the whole semantics,
so the tagged constraints on inputs/outputs are covered by construction:

  * ``E_ADD_REAL_CONSTR_A_0010`` / ``E_ADD_REAL_CONSTR_B_0010`` (shape
    compatibility) and ``E_ADD_REAL_CONSTR_C_0010`` (shape definition);
  * ``E_ADD_FLOAT_CONSTR_A_0010`` / ``..._B_0010`` / ``..._C_0010`` (shape)
    and ``E_ADD_FLOAT_CONSTR_A_0020`` / ``..._B_0020`` / ``..._C_0020``
    (type consistency);
  * ``E_ADD_INT_CONSTR_A_0010`` / ``..._B_0010`` / ``..._C_0010`` (shape)
    and ``E_ADD_INT_CONSTR_A_0020`` / ``..._B_0020`` / ``..._C_0020``
    (type consistency).

Per the document, **Add** has no attribute and no error condition; every
condition listed in the three "Error conditions" tables is part of the
nominal behavior, so no arithmetic case may raise or warn.

The document fixes the arithmetic and the broadcasting, but not the mapping
from numpy dtypes to the three families.  That mapping, and the handful of
other points the document leaves open, are decided in the code below with
the sentence relied on quoted at each decision.
"""

import numpy as np

__all__ = ["add"]


# --------------------------------------------------------------------------
# Family detection.
#
# DECISION: the document defines the three families by the names of their
# types -- "where float is in {`float16`, `float`, `double`}", "where int is
# in {`int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`,
# `uint64`}" -- and describes the third, untagged-in-that-way, family as
# "real" values whose "addition is exact: C[i] is the sum in R of the two
# elements, with no approximation".  A real tensor is therefore one whose
# elements are exact numbers, i.e. the object-dtype array of Python numbers
# the harness supplies; float and integer tensors are the ordinary numpy
# arrays of the listed fixed-width dtypes.  numpy reports the family through
# ``dtype.kind``: "O" is object (the real family), "f" is floating, "i"/"u"
# are signed/unsigned integer.
# --------------------------------------------------------------------------


def _add_real(a, b):
    """Add for the ``real`` family -- E_ADD_REAL_FUNC_0010.

    The document requires exactness ("The addition is exact: C[i] is the sum
    in R of the two elements, with no approximation") and states that no
    error condition applies ("the result is the one specified by
    E_ADD_REAL_FUNC_0010 for every pair of operands, and no error can occur").

    DECISION: exactness is obtained by using Python's arbitrary-precision
    arithmetic elementwise through numpy's object-dtype loop, rather than any
    fixed-width numpy float or integer dtype.  Python ``int``, ``Fraction``
    and ``Decimal`` addition is exact and has no overflow/underflow, which is
    what "no approximation" and "no error can occur" require.  No warnings
    can arise here, so none are suppressed.

    DECISION: the result keeps object dtype, because a real value need not be
    representable in any fixed-width machine type, and the document's
    exactness clause forbids rounding it into one.
    """
    result = a + b
    if not isinstance(result, np.ndarray):
        # numpy returns a bare Python object (not even a 0-d array) when the
        # operands are 0-d; the contract here is a 0-d numpy array.
        result = np.asarray(result, dtype=object)
    return result


def _add_machine(a, b):
    """Add for the floating-point and integer families.

    Floating-point family -- E_ADD_FLOAT_FUNC_0010: "adds tensor B to tensor A
    element-wise according to IEEE 754 floating-point semantics".  The
    document then fixes every case explicitly: NaN propagation, the
    opposite-infinities invalid operation, the exact sum when representable,
    and otherwise "round(x) is the value of x rounded to the nearest value of
    the type of C using the roundTiesToEven attribute of IEEE 754, the
    exponent range of the type being unbounded".  It also fixes the sign of a
    null sum ("the result is +0, except when A[i] and B[i] are both -0, in
    which case it is -0").

    DECISION: numpy's dtype addition is used directly, because it implements
    exactly this: IEEE 754 with roundTiesToEven (numpy's default), a null sum
    of -0 and -0 giving -0, and an unbounded exponent range so that a sum
    beyond the largest finite value gives an infinity and a tiny sum gives a
    subnormal ("The rounding may make the result subnormal").

    DECISION: the NaN of the first case is left as numpy produces it.  The
    document says "The NaN of this case is a quiet NaN; its sign and its
    payload are not specified, whether the NaN comes from an operand or from
    the invalid operation, so that every quiet NaN of the type of C is a
    conforming result", so no payload or sign manipulation is performed.

    Integer family -- E_ADD_INT_FUNC_0010: "The result is thus the exact sum
    modulo 2^n", with the signed and unsigned types differing only "by the
    value chosen in the residue class modulo 2^n".

    DECISION: numpy's fixed-width integer addition is used directly, because
    it computes the sum modulo 2^n and wraps into the range of the dtype,
    which is exactly the three-case definition ("if the sum is representable
    in the type of C", "if the sum is lower than the minimum value", "if the
    sum is greater than the maximum value").

    DECISION: numpy's overflow/invalid/underflow/divide warnings are silenced
    for the whole computation.  Each of these conditions is listed as
    "nominal" in the document's Error conditions tables ("Every condition of
    the list is part of the nominal behavior of the operator; none of them is
    an error.  No error condition."), so a conforming implementation must
    neither raise nor warn.  numpy emits these as warnings rather than
    exceptions, so ``errstate`` is enough.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        result = a + b
    if not isinstance(result, np.ndarray):
        # 0-d operands give a numpy scalar; the contract here is a 0-d array.
        result = np.asarray(result)
    return result


def add(a, b):
    """Element-wise addition of two tensors, per the Add specification.

    Parameters
    ----------
    a, b : numpy.ndarray
        The two operands.  Broadcasting follows the document's
        "Tensors of different shapes" clause: multidirectional (Numpy-style)
        rules -- shapes aligned on their last dimension, a missing dimension
        treated as size 1, a size-1 dimension extended to the other operand's
        size -- with the result taking "the resulting shape".  A rank-0 tensor
        has "a tensor whose shape is empty" and "is broadcast to the shape of
        the other operand".  A zero-sized dimension "is compatible with a
        dimension of size 1 of the other operand, and with an equal zero-sized
        dimension; the corresponding dimension of C then has size 0".  numpy's
        own broadcasting implements precisely these rules, so it is used
        unchanged for all three families.

    Returns
    -------
    numpy.ndarray
        Tensor C.  "Tensor C is the element-wise result of the addition of B
        to A."  Always a numpy array, including a 0-d array when the broadcast
        result shape is empty (never a numpy scalar).

    Notes
    -----
    The document constrains A, B and C to have the same type
    (``E_ADD_FLOAT_CONSTR_A_0020`` and ``E_ADD_INT_CONSTR_A_0020``: "Tensors
    A, B, and C have the same type."), so mixed-type operands are outside the
    specification.  DECISION: should mixed machine dtypes be passed anyway,
    numpy's usual promotion decides the result dtype; no extra policing is
    added, since the document fixes no behavior for that case.  Likewise a
    genuine shape incompatibility (e.g. sizes 2 and 3) is not one of the
    document's conditions and has no specified disposition; numpy's
    ValueError is allowed to propagate.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if a.dtype.kind == "O" or b.dtype.kind == "O":
        # The real family.  The document ties exact ("real") values to no
        # machine type, so whichever operand is an object array identifies
        # the family; see _add_real.
        return _add_real(a, b)

    # float16/float/double (kind "f") and int8..uint64 (kinds "i"/"u").
    # numpy dtype kind "b" (bool) is not one of the document's types; it is
    # routed here as well and left to numpy, as the document specifies
    # nothing for it.
    return _add_machine(a, b)
