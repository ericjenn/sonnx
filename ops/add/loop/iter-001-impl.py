"""The ``Add`` operator, as specified by the document "Add version 14".

The specification has four sections, one per family of types, and all four
open their *Function* section with the same sentence:

    "Operator **Add** adds tensor :math:`B$` to tensor :math:`A$` element-wise
    and stores the result in tensor :math:`C$`. Each element of :math:`C$` is
    the sum of the elements of :math:`A$` and :math:`B$` that correspond to it
    under the broadcasting rules given below."

The four sections differ only in what "the sum" means:

* ``real``  -- "The addition is exact: :math:`C[i]$` is the sum in
  :math:`\\mathbb{R}$` of the two elements, with no approximation."
* ``float16``/``float``/``double`` -- "Operator **Add** adds tensor :math:`B$`
  to tensor :math:`A$` element-wise according to IEEE 754 floating-point
  semantics"; NaN propagation, ``+inf + -inf = NaN``, overflow to infinity and
  gradual underflow are all "nominal", the section listing "No error
  condition".
* ``int8``/``int16``/``int32``/``int64`` -- "the result of the addition is the
  value of that type that represents the exact sum of the two elements modulo
  :math:`2^n$`", i.e. wraparound; the overflow "is part of the nominal
  behavior of the operator; there is no error".
* ``uint8``/``uint16``/``uint32``/``uint64`` -- the same modulo :math:`2^n$`
  reduction, read as an unsigned value.

All four sections give the same shape rules:

    "Tensors :math:`A$` and :math:`B$` are broadcast to the shape of
    :math:`C$` following the multidirectional (Numpy-style) broadcasting
    rules: the shapes are aligned on their last dimension; a dimension that
    one operand does not have ... is treated as a dimension of size :math:`1$`;
    and a dimension of size :math:`1$` is extended to the size of the
    corresponding dimension of the other operand.  ...  A tensor with no
    dimension (a rank-0 tensor) ... is broadcast to the shape of the other
    operand ...  An operand may have a zero-sized dimension.  Such a dimension
    is compatible with a dimension of size :math:`1$` of the other operand, and
    with an equal zero-sized dimension; the corresponding dimension of
    :math:`C$` then has size :math:`0$`, and :math:`C$` is empty."

The operator has no attribute and a single output :math:`C$`.

DECISION: the document gives the operator no attribute ("Operator **Add** has
no attribute."), so the entry point takes exactly the two operands as
positional arguments and has no keyword parameter.
"""

import numpy as np

__all__ = ["add"]


def add(A, B):
    """Element-wise addition ``C = Add(A, B)``.

    Implements, for the type family of the operands, the *Function* section of
    the specification, quoted here for each family.

    real
        "[E_ADD_REAL_FUNC_0010] Operator **Add** adds tensor :math:`B$` to
        tensor :math:`A$` element-wise and stores the result in tensor
        :math:`C$`. Each element of :math:`C$` is the sum of the elements of
        :math:`A$` and :math:`B$` that correspond to it under the broadcasting
        rules given below. ... :math:`C[i] = \\tilde{A}[i] + \\tilde{B}[i]$`.
        "The addition is exact: :math:`C[i]$` is the sum in :math:`\\mathbb{R}$`
        of the two elements, with no approximation."

    float16 / float / double
        "[E_ADD_FLOAT_FUNC_0010] Operator **Add** adds tensor :math:`B$` to
        tensor :math:`A$` element-wise according to IEEE 754 floating-point
        semantics ... :math:`C[i] = \\tilde{A}[i] +_{(f)} \\tilde{B}[i]$`",
        with the value NaN "if :math:`\\tilde{A}[i]$` or :math:`\\tilde{B}[i]$`
        is NaN, or if they are :math:`+\\infty$` and :math:`-\\infty$`", the
        exact sum when it is representable, and "the value of :math:`x$` rounded
        to the nearest value of the type ... using the roundTiesToEven attribute
        of IEEE 754, the exponent range of the type being unbounded" otherwise.
        "When the exact sum is null, the result is :math:`+0$`, except when
        :math:`\\tilde{A}[i]$` and :math:`\\tilde{B}[i]$` are both :math:`-0$`,
        in which case it is :math:`-0$`."

    int8 / int16 / int32 / int64
        "[E_ADD_INT_FUNC_0010] ... Tensors :math:`A$`, :math:`B$` and :math:`C$`
        have the same signed :math:`n$`-bit type, and the result is a value of
        that type.  :math:`C[i] = \\tilde{A}[i] + \\tilde{B}[i]$` if the sum
        lies in :math:`[-2^{n-1},\\, 2^{n-1}-1]$`; :math:`\\tilde{A}[i] +
        \\tilde{B}[i] + 2^n$` if the sum is lower than :math:`-2^{n-1}$`;
        :math:`\\tilde{A}[i] + \\tilde{B}[i] - 2^n$` if the sum is greater than
        :math:`2^{n-1}-1$`.  The result is thus the exact sum modulo
        :math:`2^n$`, read as a signed value."

    uint8 / uint16 / uint32 / uint64
        "[E_ADD_UINT_FUNC_0010] ... Tensors :math:`A$`, :math:`B$` and :math:`C$`
        have the same unsigned :math:`n$`-bit type ... :math:`C[i] =
        \\tilde{A}[i] + \\tilde{B}[i]$` if the sum is at most :math:`2^n-1$`;
        :math:`\\tilde{A}[i] + \\tilde{B}[i] - 2^n$` if the sum is greater than
        :math:`2^n-1$`.  The result is thus the exact sum modulo :math:`2^n$`,
        read as an unsigned value."

    Output
        "Tensor :math:`C$` is the element-wise result of the addition of
        :math:`B$` to :math:`A$."  "[E_ADD_*_CONSTR_C_0010] Shape definition:
        Tensor :math:`C$` has the shape of :math:`A$` and :math:`B$` broadcast
        to a common shape."

    Every one of these is exactly what the numpy ufunc ``np.add`` computes on
    arrays of the corresponding dtype: numpy broadcasting is the "multidirectional
    (Numpy-style)" rule named by the document (including rank-0 operands and
    zero-sized dimensions), IEEE 754 addition is the floating-point hardware
    behaviour (NaN propagation, ``inf + -inf -> NaN``, overflow to infinity,
    subnormals, and the ``+0``/``-0`` rules for a null sum), and integer
    addition on numpy arrays wraps modulo :math:`2^n$` in the sign convention of
    the dtype, as the document requires.

    Parameters
    ----------
    A, B : array_like
        The operands, of one common type of the specification.  ``A`` is "the
        value to which :math:`B$` is added"; ``B`` is "the value added to
        :math:`A$`".

    Returns
    -------
    numpy.ndarray or numpy scalar
        :math:`C = A + B$`, of the broadcast shape.

    Notes
    -----
    DECISION: the document describes the operands as tensors and says nothing
    about the Python objects used to carry them.  ``numpy.asarray`` is applied
    to each operand so that Python sequences and scalars are accepted and the
    broadcast arithmetic is performed by numpy; no other conversion is made.

    DECISION: the document states the constraints "[E_ADD_FLOAT_CONSTR_A_0020]
    Type consistency: Tensors :math:`A$`, :math:`B$`, and :math:`C$` have the
    same type" (and likewise for the int and uint sections) and
    "[E_ADD_*_CONSTR_A_0010] Shape compatibility: The shapes of :math:`A$` and
    :math:`B$` are compatible for broadcasting", but it says nothing about what
    happens when they are violated, and it lists no error condition.  No
    validation or coercion is therefore performed: operands of different dtypes
    are combined by numpy's own promotion rules (so the result may then not have
    the type of ``A``), and an incompatible pair of shapes raises whatever
    exception numpy raises.  Neither case is caught or suppressed.

    DECISION: the section for the ``real`` type specifies exact addition in
    :math:`\\mathbb{R}$`, which no machine number type implements.  The same
    code path is used for all dtypes, so the exactness is realised for the
    integer types and, for the floating-point types, the IEEE 754 semantics of
    the ``float16``/``float``/``double`` section applies instead; the document
    does not say which concrete dtype the ``real`` family corresponds to.

    DECISION: the document lists no boolean, complex or other dtype.  It says
    nothing about types outside the four families above, so no special handling
    is added for them and numpy's own behaviour applies unchanged (for example
    numpy adds ``bool`` operands as a logical OR).

    DECISION: the document declares overflowing sums, invalid operations
    (:math:`+\\infty + -\\infty$`) and underflows to be "nominal" and to be "no
    error condition", and it says nothing about warnings.  Warnings that numpy
    may emit for those nominal cases are neither raised deliberately nor
    suppressed.

    DECISION: for ``float16`` the document requires the correctly rounded
    addition of the type ("the value of :math:`x$` rounded to the nearest value
    of the type ... using the roundTiesToEven attribute of IEEE 754").  The
    addition is delegated to numpy's ``float16`` loop, which may carry the
    operation out in a wider format and round once more; for an exact sum that
    is not representable in that wider format this can, in rare cases, round
    differently from a single correctly rounded ``float16`` addition.  The
    document does not say in which format the arithmetic must be carried out,
    and no separate half-precision adder is implemented here.

    DECISION: the document specifies the shape and the values of :math:`C$` but
    not the Python object that carries a rank-0 tensor.  When both operands are
    rank-0, numpy may return a numpy scalar rather than a 0-d array; both have
    the empty shape the document requires.
    """
    a = np.asarray(A)
    b = np.asarray(B)
    return np.add(a, b)
