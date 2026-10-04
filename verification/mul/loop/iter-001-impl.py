"""Reference implementation of the **Mul** operator, in NumPy.

This module implements the operator specified in ``ops/mul.md``. The document
defines a single operator under four type families:

* **Mul** (real, real) -- ``E_MUL_REAL_FUNC_0010``;
* **Mul** (float, float), where float is in {``float16``, ``float``, ``double``}
  -- ``E_MUL_FLOAT_FUNC_0010``;
* **Mul** (int, int), where int is in {``int8``, ``int16``, ``int32``,
  ``int64``} -- ``E_MUL_INT_FUNC_0010``;
* **Mul** (uint, uint), where uint is in {``uint8``, ``uint16``, ``uint32``,
  ``uint64``} -- ``E_MUL_UINT_FUNC_0010``.

Every one of the four sections carries the same signature, quoted from the
document::

    $C = \\textbf{Mul}(A, B)$

    where
    - $A$: first operand
    - $B$: second operand
    - $C$: result of the element-wise multiplication of $A$ and $B$

and every section specifies the same element-wise product with the same
multidirectional (Numpy-style) broadcasting rules, differing only in the type
of the operands and in the arithmetic performed on the elements. The document
therefore is realised here by a single function, :func:`mul`, which broadcasts
the two operands and multiplies them element-wise using NumPy's arithmetic for
the dtype of the operands.

The document states, for all four sections, that the operator "has no
attribute" and that there is "No error condition"; no validation beyond what
NumPy itself performs is added here.
"""

import numpy as np

__all__ = ["mul"]


def mul(A, B):
    """Multiply tensor *A* by tensor *B* element-wise.

    This is the implementation of the operator **Mul** of ``ops/mul.md`` for
    every type family the document defines. The document gives the same
    element-wise semantics for all of them; the differences are the arithmetic
    of the element type, which NumPy's ufunc performs according to the dtype of
    the operands.

    real (``E_MUL_REAL_FUNC_0010``)
        "Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise and
        stores the result in tensor $C$. Each element of $C$ is the product of
        the elements of $A$ and $B$ that correspond to it under the
        broadcasting rules given below."

        "For any tensor index $i$ of the result $C$: $C[i] = \\tilde{A}[i]
        \\cdot \\tilde{B}[i]$", where "the multiplication is exact: $C[i]$ is
        the product in $\\mathbb{R}$ of the two elements, with no
        approximation."

        "**The two operands play the same role.** ... so that the
        multiplication of $B$ by $A$ gives the same result as the
        multiplication of $A$ by $B$".

    float (``E_MUL_FLOAT_FUNC_0010``)
        "Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise
        according to IEEE 754 floating-point semantics and stores the result in
        tensor $C$." The document enumerates the cases: NaN when either operand
        is NaN, or when one is a zero and the other an infinity (the invalid
        operation of IEEE 754 section 7.2); otherwise the exact product when
        representable; otherwise "round($\\tilde{A}[i] \\cdot \\tilde{B}[i]$)"
        to the nearest value of the type "using the roundTiesToEven attribute
        of IEEE 754, the exponent range of the type being unbounded", a
        magnitude exceeding the largest finite value giving an infinity of the
        sign of the exact product. "The result takes the sign of the exact
        product in every case, zero and infinity included". The result for
        infinity times a finite non-zero value, and for zero times a finite
        value, and the signed-zero rules are given explicitly, e.g. "$(+0)
        \\cdot_{(\text{f})} (-5) = -0$".

    int (``E_MUL_INT_FUNC_0010``)
        "Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise and
        stores the result in output tensor $C$. ... Tensors $A$, $B$ and $C$
        have the same signed $n$-bit type, and the result is a value of that
        type. For any tensor index $i$ of the result $C$: $C[i] \\equiv
        \\tilde{A}[i] \\cdot \\tilde{B}[i] \\pmod{2^n}$, $C[i]$ a value of the
        type of $A$ and $B$", where "$\\tilde{A}[i] \\cdot \\tilde{B}[i]$ is the
        exact product of the two elements, the product in $\\mathbb{Z}$ of the
        two integers, before the reduction modulo $2^n$". The document notes
        that "the multiplication of two positive values may give a negative
        result" and that "the result may be zero although both operands are
        non-zero".

    uint (``E_MUL_UINT_FUNC_0010``)
        Identical wording to the signed case with "signed" replaced by
        "unsigned", the values of the type lying "from $0$ to $2^n-1$"; the
        document notes "the result may be smaller than both operands" and again
        "may be zero although both operands are non-zero".

    Broadcasting (all four sections)
        "**Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to
        the shape of $C$ following the multidirectional (Numpy-style)
        broadcasting rules: the shapes are aligned on their last dimension; a
        dimension that one operand does not have, including the leading
        dimensions missing from the operand with the smaller rank, is treated
        as a dimension of size $1$; and a dimension of size $1$ is extended to
        the size of the corresponding dimension of the other operand. Tensor
        $C$ has the resulting shape."

        "**Tensors with no dimension.** A tensor with no dimension (a rank-0
        tensor) is a tensor whose shape is empty. It is broadcast to the shape
        of the other operand, so that each element of $C$ is the product of the
        two operands and $C$ has the shape of the other operand."

        "**Zero-sized dimensions.** An operand may have a zero-sized dimension.
        Such a dimension is compatible with a dimension of size $1$ of the
        other operand, and with an equal zero-sized dimension; the
        corresponding dimension of $C$ then has size $0$, and $C$ is empty."

    The output constraint shared by all four sections is
    "``[E_MUL_*_CONSTR_C_0010]`` Shape definition -- Statement: Tensor $C$ has
    the shape of $A$ and $B$ broadcast to a common shape."

    Parameters
    ----------
    A : array_like
        The first operand.
    B : array_like
        The second operand.

    Returns
    -------
    numpy.ndarray or numpy.generic
        The element-wise product, with the broadcast shape of ``A`` and ``B``
        and the NumPy dtype resulting from ``A`` and ``B``.

    Notes
    -----
    The implementation is the NumPy ufunc :func:`numpy.multiply`, applied
    directly to the two operands. See the module docstring and the ``DECISION``
    comments for the points on which the document is silent.
    """
    # DECISION: A single function serves all four type families of the
    # document (real, float, int, uint). The document presents them as four
    # separate operator sections, but gives each the same signature
    # "$C = \textbf{Mul}(A, B)$" with one positional argument per operand and
    # the same element-wise product; it says nothing about a dispatch between
    # them. The dtype of the operands selects the arithmetic in NumPy, which is
    # the mechanism the document's per-type semantics describe.
    #
    # DECISION: The multiplication itself is delegated to NumPy rather than
    # written out. For the float section the document requires IEEE 754
    # semantics with roundTiesToEven and unbounded exponent range ("round(x)
    # is the value of x rounded to the nearest value of the type of A and B
    # using the roundTiesToEven attribute of IEEE 754, the exponent range of
    # the type being unbounded"), and numpy.multiply on float16/float32/float64
    # inputs implements exactly that, including the invalid operation of IEEE
    # 754 section 7.2 ("the multiplication of a zero by an infinity is the
    # invalid operation defined in IEEE 754 section 7.2; the result is NaN in
    # both cases") and the sign rule ("the sign of C[i] is negative when
    # exactly one of A[i] and B[i] is negative").
    #
    # DECISION: The "real" section requires exact multiplication in R -- "The
    # multiplication is exact: C[i] is the product in R of the two elements,
    # with no approximation" -- but the document says nothing about how a
    # "real tensor" is represented in a NumPy implementation, and NumPy has no
    # exact-real dtype. The reading implemented here is that the caller
    # supplies whatever dtype represents the real operand, and the element
    # product is numpy.multiply on that dtype; exactness therefore holds
    # whenever that dtype represents the product exactly (as for the integer
    # dtypes, and for exactly representable floating-point products).
    #
    # DECISION: The integer sections require the result to be "the value of
    # that type congruent to the product modulo 2^n" (E_MUL_INT_FUNC_0010,
    # E_MUL_UINT_FUNC_0010). numpy.multiply on two same-dtype integer arrays
    # already yields the value of that type congruent to the exact product
    # modulo 2^n (two's-complement wraparound), so no explicit reduction with
    # `%` is written. The document gives no separate formula to prefer; it
    # only states the congruence, which NumPy's arithmetic satisfies.
    #
    # DECISION: The document states the constraints "[E_MUL_*_CONSTR_A_0010]
    # Shape compatibility -- Statement: The shapes of A and B are compatible
    # for broadcasting" and "[E_MUL_*_CONSTR_A_0020] Type consistency --
    # Statement: Tensors A, B, and C have the same type", but those are
    # preconditions on the operands, and every section says "No error
    # condition" and "there is no error". The document therefore decides
    # nothing about a violation, and no validation of either constraint is
    # added: a shape that cannot broadcast raises NumPy's own error, and mixed
    # operand types are left to NumPy's promotion rules.
    #
    # DECISION: Type consistency is likewise not enforced. The document says
    # "Tensors A, B, and C have the same type" for the float, int and uint
    # families, but says nothing about the behaviour when A and B differ in
    # type; that situation is excluded by the constraint rather than specified,
    # so the promotion NumPy applies is left unchanged.
    #
    # DECISION: NumPy's warnings are not suppressed. The document calls the
    # invalid operation, overflow and underflow "nominal" and says of each
    # "No error condition", but it says nothing about warnings; NumPy may emit
    # RuntimeWarnings (e.g. "invalid value encountered in multiply" for a zero
    # times an infinity, or "overflow encountered in multiply" for 0-d integer
    # operands). A warning is not an error, and the document does not mention
    # warnings, so NumPy's default warning behaviour is left untouched rather
    # than masked with numpy.errstate.
    #
    # DECISION: Rank-0 tensors, zero-sized dimensions, and broadcasting of
    # different shapes are delegated to NumPy. The document does not merely
    # permit this: it names the rule outright -- "the multidirectional
    # (Numpy-style) broadcasting rules" -- and describes rank-0 and zero-sized
    # dimensions in the same terms NumPy implements ("a dimension of size 1 is
    # extended to the size of the corresponding dimension of the other
    # operand"; "Such a dimension is compatible with a dimension of size 1 of
    # the other operand").
    return np.multiply(A, B)
