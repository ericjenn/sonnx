"""Element-wise multiplication: the **Mul** operator of ONNX opset 14, as
specified by ``ops/mul.md`` (revision 2026-10-03, "first issue of this
specification, based on ONNX opset 14").

The document specifies one operator under four type families, each with its
own *Function* clause:

* ``real`` -- exact arithmetic on real numbers:
  "Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise and
  stores the result in tensor $C$. Each element of $C$ is the product of the
  elements of $A$ and $B$ that correspond to it under the broadcasting rules
  given below." and "The multiplication is exact: $C[i]$ is the product in
  $\\mathbb{R}$ of the two elements, with no approximation."
  ("The two operands play the same role.")

* ``float16`` / ``float`` / ``double`` -- IEEE 754 multiplication:
  "Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise according
  to IEEE 754 floating-point semantics and stores the result in tensor $C$.",
  with the result of every combination of operands given by
  ``E_MUL_FLOAT_FUNC_0010`` (NaN for a NaN operand or for the invalid
  operation zero-times-infinity; the exact product where representable; the
  exact product rounded with roundTiesToEven and unbounded exponent range
  otherwise, overflowing to an infinity of the product's sign).
  "The result takes the sign of the exact product in every case, zero and
  infinity included".

* ``int8`` / ``int16`` / ``int32`` / ``int64`` -- modular arithmetic:
  "the result of the multiplication is the value of that type that is
  congruent to the exact product of the two elements modulo $2^n$"
  (``E_MUL_INT_FUNC_0010``).

* ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` -- the same congruence for
  the unsigned types (``E_MUL_UINT_FUNC_0010``).

All four families share the same rules for tensors of different shapes, for
tensors with no dimension and for zero-sized dimensions; the operator has no
attribute ("Operator **Mul** has no attribute."), and its inputs are $A$ (the
first operand) and $B$ (the second operand), its output is $C$.

This module provides the single function :func:`mul`.  The applicable type
family is selected from the operands' numpy dtype, and the numerical
behaviour of each family is delegated to numpy's element-wise multiplication,
which reproduces the clause of that family.  Points on which the document is
silent or ambiguous are recorded in ``DECISION:`` comments.
"""

import numpy as np

__all__ = ["mul"]


def mul(A, B):
    """Multiply two tensors element-wise and return the result: ``C = Mul(A, B)``.

    The single positional argument per operand $A$ (first operand) and $B$
    (second operand); the result is tensor $C$, "the element-wise result of
    the multiplication of $A$ by $B$".

    Type families
    -------------
    The document defines the operator separately for four families of types,
    and this function realizes all of them according to the dtype of the
    operands:

    * floating-point types (``float16``, ``float``, ``double``) --
      "Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise
      according to IEEE 754 floating-point semantics and stores the result in
      tensor $C$."  Numpy's multiply of two floating-point arrays is the
      IEEE 754 multiply: NaN propagates and zero-times-infinity is the
      invalid operation, the exact product is rounded with roundTiesToEven to
      the operand type (subnormally where needed, or to a zero below half the
      smallest subnormal), overflow gives an infinity of the product's sign,
      and "The result takes the sign of the exact product in every case, zero
      and infinity included".

    * signed integer types (``int8``, ``int16``, ``int32``, ``int64``) --
      "Tensors $A$, $B$ and $C$ have the same signed $n$-bit type, and the
      result is a value of that type." with, for every tensor index $i$,
      "$C[i] \\equiv \\tilde{A}[i] \\cdot \\tilde{B}[i] \\pmod{2^n}$, $C[i]$ a
      value of the type of $A$ and $B$", where the product is "the exact
      product of the two elements, the product in $\\mathbb{Z}$ of the two
      integers, before the reduction modulo $2^n$".  Numpy's multiplication of
      two fixed-width integer arrays is exactly that reduction: its result is
      the representable value congruent to the exact product modulo
      $2^n$ (for ``int8``, $100 \\cdot 100 = 10000 = 39 \\cdot 2^8 + 16$ gives
      $16$; $11 \\cdot 13 = 143$ gives $143 - 2^8 = -113$; $16 \\cdot 16 = 256$
      gives $0$).

    * unsigned integer types (``uint8``, ``uint16``, ``uint32``, ``uint64``)
      -- the same congruence,
      "$C[i] \\equiv \\tilde{A}[i] \\cdot \\tilde{B}[i] \\pmod{2^n}$, $C[i]$ a
      value of the type of $A$ and $B$", over the unsigned values $0$ to
      $2^n-1$ (for ``uint8``, $200 \\cdot 200 = 40000 = 156 \\cdot 2^8 + 64$
      gives $64$, and $20 \\cdot 13 = 260$ gives $4$).

    * the abstract ``real`` family is a mathematical type that has no
      counterpart among numpy's dtypes; see the ``DECISION`` comments below.
      For every concrete type implemented here, the document's general rules
      hold -- "Tensors $A$, $B$ and $C$ have the same type" -- and the result
      of the function has the dtype of the operands.

    Broadcasting, rank-0 tensors and zero-sized dimensions
    ------------------------------------------------------
    "**Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the
    shape of $C$ following the multidirectional (Numpy-style) broadcasting
    rules: the shapes are aligned on their last dimension; a dimension that
    one operand does not have, including the leading dimensions missing from
    the operand with the smaller rank, is treated as a dimension of size $1$;
    and a dimension of size $1$ is extended to the size of the corresponding
    dimension of the other operand.  Tensor $C$ has the resulting shape."

    "**Tensors with no dimension.** A tensor with no dimension (a rank-0
    tensor) is a tensor whose shape is empty.  It is broadcast to the shape of
    the other operand, so that each element of $C$ is the product of the two
    operands and $C$ has the shape of the other operand."

    "**Zero-sized dimensions.** An operand may have a zero-sized dimension.
    Such a dimension is compatible with a dimension of size $1$ of the other
    operand, and with an equal zero-sized dimension; the corresponding
    dimension of $C$ then has size $0$, and $C$ is empty."

    The output constraint is honoured: "Tensor $C$ has the shape of $A$ and
    $B$ broadcast to a common shape."

    Errors
    ------
    The document declares "No error condition." for every family: overflow,
    underflow and the invalid operation are all nominal and specified by the
    clauses quoted above.  No error or warning is raised or filtered by this
    function beyond what numpy itself does.

    Attributes
    ----------
    None: "Operator **Mul** has no attribute."  The function therefore takes no
    keyword argument.
    """
    # DECISION: the "real" family of the document has no numpy dtype, so no
    # code path is dedicated to it.  The document's real clause ("The
    # multiplication is exact: $C[i]$ is the product in $\mathbb{R}$ of the two
    # elements, with no approximation.") cannot hold for a numpy floating-point
    # array, which is a float16/float/double operand governed by the separate
    # floating-point clause ("according to IEEE 754 floating-point semantics");
    # the document says nothing about how the abstract real type maps onto a
    # host-language type.

    # DECISION: the document is silent about how its type families are
    # selected at run time; I select them from the numpy dtype kind of the
    # operands (floating-point, signed integer, unsigned integer).  Dtype kinds
    # the document does not cover at all (bool, complex, object, longdouble,
    # ...) are neither rejected nor emulated: numpy's native behaviour for
    # them stands, since the document says nothing about them ("No specific
    # restrictions apply to the Mul operator." does not add any).

    # DECISION: the document speaks of tensors and says nothing about
    # accepting array-likes, so the operands are treated as array-like tensors
    # and converted with np.asarray.  The conversion uses no dtype argument, so
    # an operand that is already an array keeps its dtype; the document is
    # silent on the dtype chosen for a Python scalar or sequence.

    a = np.asarray(A)
    b = np.asarray(B)

    # DECISION: the congruence "$C[i] \equiv \tilde{A}[i] \cdot \tilde{B}[i]
    # \pmod{2^n}$" is realized by numpy's native fixed-width integer
    # multiplication, whose result is the value of the type congruent to the
    # exact product modulo 2^n (numpy's wraparound); the document is silent
    # about the mechanism.  Numpy does not raise on this nominal overflow.

    # DECISION: the floating-point cases of E_MUL_FLOAT_FUNC_0010 (NaN
    # propagation, zero-times-infinity NaN, roundTiesToEven rounding of the
    # exact product, subnormal results, overflow to an infinity of the
    # product's sign, and the sign of zero products) are delegated to numpy's
    # IEEE 754 multiply rather than reimplemented; the document specifies the
    # values but not the mechanism, and numpy's result is that IEEE 754
    # multiply.

    # DECISION: numpy's default floating-point warning state is left
    # untouched, so a product that overflows may still emit numpy's host
    # warning (e.g. "overflow encountered in multiply").  The document calls
    # overflow nominal ("nominal: specified by E_MUL_FLOAT_FUNC_0010, the
    # result is an infinity of the sign of the exact product") but says
    # nothing about warnings, and mentions no error or warning to raise or
    # suppress.

    c = np.multiply(a, b)

    # DECISION: the result is passed through np.asarray so that C is always a
    # tensor.  numpy returns a scalar (not an ndarray) when both operands are
    # rank-0, whereas the document says "A tensor with no dimension (a rank-0
    # tensor) is a tensor whose shape is empty" and "Tensor $C$ ..."; the
    # document is silent on the host representation of a rank-0 tensor.  For
    # operands of rank >= 1 this changes nothing.

    # DECISION: no validation of the document's operand constraints is added.
    # "[E_MUL_..._CONSTR_A_0010] Shape compatibility: The shapes of $A$ and $B$
    # are compatible for broadcasting." and "[E_MUL_..._CONSTR_A_0020] Type
    # consistency: Tensors $A$, $B$, and $C$ have the same type." are
    # statements about the operands, and the document is silent about whether
    # an implementation must check them; incompatible shapes therefore raise
    # numpy's own broadcasting error, and operands of different dtypes (which
    # violate type consistency) are combined by numpy's promotion, a case the
    # document does not specify.

    # DECISION: the operands are multiplied in the order given, A times B,
    # although the document says "The two operands play the same role." and
    # that the operands "may be swapped without changing the result"; numpy's
    # multiplication is commutative for these types, so no swap is needed and
    # none is performed.  (For floating-point NaNs the document states that
    # sign and payload are "not specified", so either order conforms.)

    # DECISION: broadcasting, including rank-0 operands and zero-sized
    # dimensions, is delegated to numpy, whose multidirectional rules the
    # document describes verbatim; no explicit shape computation is performed,
    # the document being silent about the mechanism.

    return np.asarray(c)
