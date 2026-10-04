"""Implementation of the ONNX **Add** operator (opset 14) against the
specification ``ops/add.md``, revision 2026-10-03.

The document defines one operator, **Add**, split into four variants that
differ only by the type of the operands:

* **Add** for type ``real`` -- "Add operator for type real";
* **Add** for ``float16``, ``float``, ``double`` -- "Add operator for types
  [`float16`, `float`, `double`]";
* **Add** for ``int8``, ``int16``, ``int32``, ``int64`` -- "Add operator for
  types [`int8`, `int16`, `int32`, `int64`]";
* **Add** for ``uint8``, ``uint16``, ``uint32``, ``uint64`` -- "Add operator
  for types [`uint8`, `uint16`, `uint32`, `uint64`]".

All four have the signature ``C = Add(A, B)`` with the same meaning:

    "where
     - $A$: value to which $B$ is added
     - $B$: value added to $A$
     - $C$: result of the element-wise addition of $B$ to $A$"

and all four add ``B`` to ``A`` element-wise after broadcasting the two
operands to a common shape.  In every variant, "Operator **Add** has no
attribute." and "No error condition."

The public entry point is :func:`add`.

Numerical notes that hold for every variant and that this module relies on
(quoted from the document):

* Broadcasting -- "Tensors $A$ and $B$ are broadcast to the shape of $C$
  following the multidirectional (Numpy-style) broadcasting rules: the shapes
  are aligned on their last dimension; a dimension that one operand does not
  have, including the leading dimensions missing from the operand with the
  smaller rank, is treated as a dimension of size $1$; and a dimension of size
  $1$ is extended to the size of the corresponding dimension of the other
  operand. Tensor $C$ has the resulting shape."
* Rank-0 tensors -- "A tensor with no dimension (a rank-0 tensor) is a tensor
  whose shape is empty. It is broadcast to the shape of the other operand, so
  that each element of $C$ is the sum of the two operands and $C$ has the
  shape of the other operand."
* Zero-sized dimensions -- "Such a dimension is compatible with a dimension
  of size $1$ of the other operand, and with an equal zero-sized dimension;
  the corresponding dimension of $C$ then has size $0$, and $C$ is empty."

These three rules are exactly numpy's broadcasting rules, so the
broadcasting itself is delegated to numpy's element-wise addition.
"""

import numpy as np

__all__ = ["add"]


def _add_real(a, b):
    """Element-wise addition of two real tensors.

    Implements the variant "Add (real, real)", whose function is stated as
    **[E_ADD_REAL_FUNC_0010]**:

        "Operator **Add** adds tensor $B$ to tensor $A$ element-wise and
        stores the result in tensor $C$. Each element of $C$ is the sum of
        the elements of $A$ and $B$ that correspond to it under the
        broadcasting rules given below.

        For any tensor index $i$ of the result $C$:

            C[i] = A~[i] + B~[i]

        where
        - $\\tilde{A}$ is tensor $A$ broadcast to the shape of $C$,
        - $\\tilde{B}$ is tensor $B$ broadcast to the shape of $C$.

        The addition is exact: $C[i]$ is the sum in $\\mathbb{R}$ of the two
        elements, with no approximation."

    and, for the result's shape, **[E_ADD_REAL_CONSTR_C_0010]**:

        "Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a
        common shape."

    and, for the absence of failure, the "Error conditions" section of that
    variant:

        "None of the conditions of the list applies to real numbers: the
        addition is exact, so that the result is the one specified by
        [E_ADD_REAL_FUNC_0010] for every pair of operands, and no error can
        occur. No error condition."

    DECISION: the document's ``real`` type is a mathematical type and it
    names no numpy dtype for it ("The addition is exact: ... with no
    approximation"), and it gives explicit numpy-level types only in the
    float/int/uint sections; the document says nothing about how ``real`` is
    represented in a numpy implementation.  This branch is therefore the
    fallback for every dtype the three typed branches do not claim, and it
    leaves the arithmetic to ``numpy.add`` on that dtype -- which for object
    arrays holding exact numeric types (e.g. ``fractions.Fraction``,
    ``decimal.Decimal``) performs the exact addition in Z or Q that the
    document requires.
    """
    # DECISION: the document says nothing about dtypes outside the four
    # variants (bool_ and complex are examples); no validation is added
    # ("No error condition."), and numpy's own addition is used for them.
    return np.add(a, b)


def _add_float(a, b):
    """Element-wise addition of two floating-point tensors.

    Implements the variant "Add (float, float)", where "float is in
    {``float16``, ``float``, ``double``}", and "The three types share one
    semantics: the IEEE 754 standard defines the same addition for all of
    them, and they differ only by their precision and by the range of the
    values they represent."

    Its function is stated as **[E_ADD_FLOAT_FUNC_0010]**:

        "Operator **Add** adds tensor $B$ to tensor $A$ element-wise
        according to IEEE 754 floating-point semantics and stores the result
        in tensor $C$. Each element of $C$ is the floating-point sum of the
        elements of $A$ and $B$ that correspond to it under the broadcasting
        rules given below.

        For any tensor index $i$ of the result $C$:

            C[i] = A~[i] +_(f) B~[i] =
                   NaN                          if A~[i] or B~[i] is NaN, or
                                                if they are +inf and -inf
                   A~[i] + B~[i]                if the exact sum is
                                                representable in the type of
                                                A and B
                   round(A~[i] + B~[i])         otherwise

        where
        - $+_{(\\text{f})}$ is the addition of the floating-point type of
          $A$, $B$ and $C$ [...] whereas the $+$ of the second and the third
          case is the addition of $\\mathbb{R}$,
        - $\\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and
          $\\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
        - $\\text{round}(x)$ is the value of $x$ rounded to the nearest value
          of the type of $A$ and $B$ using the roundTiesToEven attribute of
          IEEE 754, the exponent range of the type being unbounded: a
          magnitude whose nearest value exceeds the largest finite value of
          the type of $A$ and $B$ gives an infinity of the sign of $x$."

    Its "Error conditions" section lists invalid operation, overflow and
    underflow, all with disposition "nominal: specified by
    [E_ADD_FLOAT_FUNC_0010]", and concludes "Every condition of the list is
    part of the nominal behavior of the operator; none of them is an error.
    No error condition."

    DECISION: the document says nothing about how numpy realizes IEEE 754;
    numpy's floating-point add is IEEE 754 addition in the operand's own type
    (``float16``, ``float32``, ``float64``), with roundTiesToEven as the
    default rounding, so it is used directly as the ``+_(f)`` of the
    document.  Each of the three specified widths is routed here by its dtype
    kind (the document: "The three types share one semantics"), so the
    rounding, the sign of a null sum ("When the exact sum is null, the result
    is $+0$, except when $\\tilde{A}[i]$ and $\\tilde{B}[i]$ are both $-0$, in
    which case it is $-0$."), the propagation of NaN, the invalid operation
    ``inf + (-inf)``, the infinity results of overflow and the subnormal
    results of underflow are all numpy's.
    """
    # DECISION: the document says nothing about warnings.  numpy emits a
    # RuntimeWarning for the invalid operation and for overflow, and the
    # document states that all these cases are nominal and no error occurs;
    # nothing is suppressed here, because suppressing a warning the document
    # does not mention would be adding a behaviour the document does not
    # specify.  The warning is informational, not an error, and the returned
    # value is the one the document specifies.
    return np.add(a, b)


def _add_signed_int(a, b):
    """Element-wise addition of two signed integer tensors, modulo $2^n$.

    Implements the variant "Add (int, int)", where "int is in {``int8``,
    ``int16``, ``int32``, ``int64``}" and "each of them is an $n$-bit signed
    type, whose values range from $-2^{n-1}$ to $2^{n-1}-1$, and the result of
    the addition is the value of that type that represents the exact sum of
    the two elements modulo $2^n$."

    Its function is stated as **[E_ADD_INT_FUNC_0010]**:

        "Tensors $A$, $B$ and $C$ have the same signed $n$-bit type, and the
        result is a value of that type.

        For any tensor index $i$ of the result $C$:

            C[i] = A~[i] + B~[i]         if the sum lies in
                                        [-2^(n-1), 2^(n-1)-1]
                   A~[i] + B~[i] + 2^n   if the sum is lower than -2^(n-1)
                   A~[i] + B~[i] - 2^n   if the sum is greater than 2^(n-1)-1

        where
        - $\\tilde{A}[i] + \\tilde{B}[i]$ is the exact sum of the two
          elements, the sum in $\\mathbb{Z}$ of the two integers, before the
          reduction modulo $2^n$,
        - $\\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and
          $\\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
        - $n$ is the number of bits of the type of $A$ and $B$."

    and it is noted that "The values of an $n$-bit signed type lie in
    $[-2^{n-1}, 2^{n-1}-1]$, so that their exact sum lies in $[-2^n, 2^n-2]$
    and one of the three cases above always applies, the reduction being
    applied at most once."

    The result's sign is that of the type, not of the operands: "for
    ``int8``, $100+100$ gives $-56$ and $-100+(-100)$ gives $56$. Tensors $A$,
    $B$ and $C$ have the same type and no wider type is used, so that the
    value of $C$ is that value of the type, sign included."

    Its "Error conditions" section lists overflow with disposition
    "nominal: specified by [E_ADD_INT_FUNC_0010], the result is the value of
    the type that represents the sum modulo $2^n$ [...]", and concludes
    "there is no error. No error condition."

    DECISION: the document states the arithmetic on the exact sum modulo
    $2^n$; numpy's integer addition on two arrays of the same signed integer
    dtype returns that dtype and wraps modulo $2^n$ over the same range
    $[-2^{n-1}, 2^{n-1}-1]$, i.e. it is the three-case definition above and
    the reduction is applied at most once exactly as the document requires.
    The four specified widths all share this one branch, as the document
    states they share one semantics.
    """
    return np.add(a, b)


def _add_unsigned_int(a, b):
    """Element-wise addition of two unsigned integer tensors, modulo $2^n$.

    Implements the variant "Add (uint, uint)", where "uint is in {``uint8``,
    ``uint16``, ``uint32``, ``uint64``}" and "each of them is an $n$-bit
    unsigned type, whose values range from $0$ to $2^n-1$, and the result of
    the addition is the value of that type that represents the exact sum of
    the two elements modulo $2^n$."

    Its function is stated as **[E_ADD_UINT_FUNC_0010]**:

        "Tensors $A$, $B$ and $C$ have the same unsigned $n$-bit type, and
        the result is a value of that type.

        For any tensor index $i$ of the result $C$:

            C[i] = A~[i] + B~[i]         if the sum is at most 2^n - 1
                   A~[i] + B~[i] - 2^n   if the sum is greater than 2^n - 1

        where
        - $\\tilde{A}[i] + \\tilde{B}[i]$ is the exact sum of the two
          elements, the sum in $\\mathbb{Z}$ of the two integers, before the
          reduction modulo $2^n$,
        - $\\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and
          $\\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
        - $n$ is the number of bits of the type of $A$ and $B$."

    The result may be smaller than both operands: "for ``uint8``, $200+100$
    gives $44$, a value smaller than both operands. Tensors $A$, $B$ and $C$
    have the same type and no wider type is used, so that the value of $C$ is
    that value of the type, and the addition is not monotonic on the unsigned
    types."

    Its "Error conditions" section lists overflow with disposition
    "nominal: specified by [E_ADD_UINT_FUNC_0010], the result is that sum
    minus $2^n$, which is then smaller than both operands", and concludes
    "there is no error. No error condition."

    DECISION: as for the signed types, the document states the arithmetic on
    the exact sum modulo $2^n$; numpy's addition on two arrays of the same
    unsigned integer dtype returns that dtype and wraps modulo $2^n$ over
    $[0, 2^n-1]$, which is the two-case definition above.  The four specified
    widths all share this one branch, as the document states they share one
    semantics.
    """
    return np.add(a, b)


def add(A, B):
    """Element-wise addition of tensors $A$ and $B$.

    This is the operator **Add** of ``ops/add.md``, in all four of its type
    variants; see the module docstring for the list.  It implements the
    common signature

        $C = \\textbf{Add}(A, B)$

    "where
     - $A$: value to which $B$ is added
     - $B$: value added to $A$
     - $C$: result of the element-wise addition of $B$ to $A$"

    together with the common output constraint **[E_ADD_*_CONSTR_C_0010]**
    "Shape definition -- Statement: Tensor $C$ has the shape of $A$ and $B$
    broadcast to a common shape." and, in each variant, the constraints that
    "The shapes of $A$ and $B$ are compatible for broadcasting." and that
    "Tensors $A$, $B$, and $C$ have the same type.".

    Parameters
    ----------
    A : array_like
        "The value to which $B$ is added."
    B : array_like
        "The value added to $A$."

    Returns
    -------
    numpy.ndarray
        "Tensor $C$ is the element-wise result of the addition of $B$ to
        $A$."

    Notes
    -----
    Deciding the variant.  DECISION: the document selects the variant by the
    type of the operands ("Add operator for type [real] for
    [``float16``, ``float``, ``double``] for [``int8`` ...] for [``uint8``
    ...]") but, being a specification of a mathematical operator and not of
    a Python function, it says nothing about how an implementation discovers
    that type.  This implementation reads it from the numpy dtype of the
    operands, via ``numpy.result_type``, and dispatches on its kind: a signed
    integer kind routes to :func:`_add_signed_int`, an unsigned integer kind
    to :func:`_add_unsigned_int`, a floating-point kind to
    :func:`_add_float`, and every other kind -- the ``real`` variant, for
    which the document gives no numpy-level type -- to :func:`_add_real`.

    Broadcasting, rank-0 operands and zero-sized dimensions.  DECISION: the
    document says nothing about an implementation of broadcasting, of the
    rank-0 case or of the zero-sized-dimension case beyond stating the rules
    quoted in the module docstring; those rules are exactly numpy's, so all
    three are delegated to numpy's element-wise addition, which also decides
    the shape of $C$.

    Errors.  DECISION: every variant ends with "No error condition.", and
    the document requires no validation, so no validation is performed.  Two
    consequences the document does not address are noted where they arise:
    an incompatible pair of shapes (the document only *constrains* "The
    shapes of $A$ and $B$ [to be] compatible for broadcasting" and says
    nothing about the violation) is left to numpy, which raises its own
    ``ValueError``; and mismatched dtypes (the document only *constrains*
    "Tensors $A$, $B$, and $C$ [to] have the same type" and says nothing
    about the violation, and lists no error condition) are left to numpy's
    type promotion.
    """
    # DECISION: the document speaks only of "tensors" and says nothing about
    # Python lists or scalars; operands are converted with asarray, which for
    # a tensor (a numpy array) is the identity and for array_like input gives
    # the tensor the document's rules apply to.  For a Python float this
    # yields float64 and hence the "double" variant; the document says
    # nothing about that mapping.
    a = np.asarray(A)
    b = np.asarray(B)

    # DECISION: the document says nothing about operands of different types
    # (the constraint "Tensors A, B, and C have the same type." is only a
    # constraint, and no error condition is listed).  np.result_type reduces
    # the two dtypes to the one whose variant is applied; for two operands of
    # the same type -- the only case the document defines -- it is that type.
    dtype = np.result_type(a.dtype, b.dtype)

    if dtype.kind == "i":
        return _add_signed_int(a, b)
    if dtype.kind == "u":
        return _add_unsigned_int(a, b)
    if dtype.kind == "f":
        return _add_float(a, b)
    # DECISION: the document's "real" variant ("Add operator for type
    # [real]") is the only variant left, and the document says nothing about
    # which numpy dtypes realize it; see _add_real.
    return _add_real(a, b)
