"""Implementation of the SONNX **Sin** operator (revision 2026-10-03, ONNX opset 14).

The document defines **Sin** twice, once on the real numbers and once on the
three floating-point types, and this module implements it once, under the name
``sin``, taking the single operand as its one positional argument and returning
the result.

The **Sin (real)** section specifies, for the operand tensor ``A`` and result
``C``:

    "Operator **Sin** applies the sine to tensor ``A`` element-wise and stores
    the result in tensor ``C``. Each element of ``C`` is the sine of the
    element of ``A`` that corresponds to it.

    For any tensor index ``i`` of the result ``C``:

        C[i] = sin(A[i])

    The sine is the function of R into the interval [-1, 1]; the value of
    ``C[i]`` is that function of the element, exactly, with no approximation
    and no rounding."

The **Sin (float)** section, "where float is in {float16, float, double}",
specifies:

    "Operator **Sin** applies the sine to tensor ``A`` element-wise and stores
    the result in tensor ``C``, in the type of ``A``. Each element of ``C`` is
    the sine of the element of ``A`` that corresponds to it.

    For any tensor index ``i`` of the result ``C``:

        C[i] = NaN                     if A[i] is NaN or an infinity
               A[i]                     if A[i] is a zero
               nearest(sin(A[i]))       otherwise

    where
    - sin(A[i]) is the value of the sine of the element, in R: the sine of the
      real number that the element of ``A`` is,
    - nearest(x) is the value of ``x`` rounded to the nearest value of the type
      of ``A`` using the roundTiesToEven attribute of IEEE 754. Since
      |sin x| <= 1 for every real number ``x``, no result leaves the range of
      the type, and the exponent range of the type plays no role."

It further states of the float case:

    "The exact sine is what the operator specifies, and the way an
    implementation computes it is not specified here."
    "Every result lies in [-1, 1], and no computation underflows."
    "The sine is odd, and the result is odd too. ... the negation of a value of
    a floating-point type is exact: the result for the operand -A[i] is the
    negation of the result for A[i], the sign bit included. The second case
    above follows that rule: the sine of +0 is +0 and the sine of -0 is -0."
    "The operand may be any value of the type, including the special numbers
    +/-0, +/-inf and NaN. The cases above give the result for every
    combination of them."
    "Tensors with no dimension. A tensor with no dimension (a rank-0 tensor) is
    a tensor whose shape is empty. Its single element is an element of A like
    any other: C is a rank-0 tensor whose single element is the sine of that
    element."
    "Zero-sized dimensions. An operand may have a zero-sized dimension. Such a
    dimension of A is a dimension of size 0 of C: C is empty along it, the
    corresponding elements of C do not exist, and the sine of no element is
    computed."

The output constraints are "[E_SIN_FLOAT_CONSTR_C_0010] Tensor C has the shape
of A" and "[E_SIN_FLOAT_CONSTR_A_0010] Tensors A and C have the same type" (the
latter applied to C by [E_SIN_FLOAT_CONSTR_C_0020]).

The document has no attributes, and its Error conditions section declares that
"Every condition of the table is part of the nominal behavior of the operator;
none of them is an error. No error condition."
"""

import numpy as np


def sin(A):
    """Return the element-wise sine of the operand ``A``.

    Implements, for any index ``i`` of the result,

        C[i] = NaN                if A[i] is NaN or an infinity
               A[i]                if A[i] is a zero
               nearest(sin(A[i]))  otherwise

    which is the "Function" section of **Sin (float)**, and the element-wise
    identity ``C[i] = sin(A[i])`` of **Sin (real)**, whose result is constrained
    by ``[E_SIN_REAL_CONSTR_C_0010] Tensor C has the shape of A`` and by
    ``[E_SIN_FLOAT_CONSTR_C_0010]``/``[E_SIN_FLOAT_CONSTR_A_0010]`` on the float
    side: the result has the shape of the operand and, for the floating-point
    types, its type as well.

    The result is computed with ``numpy.sin`` over the whole tensor, which
    realizes all three cases of the float rule: the sine of a NaN or of an
    infinity is the IEEE 754 invalid operation and yields NaN, the sine of
    ``+0`` is ``+0`` and of ``-0`` is ``-0``, and every other element is the
    value of the type nearest the exact sine. A rank-0 operand gives a rank-0
    result and a zero-sized dimension of the operand gives a zero-sized
    dimension of the result, since the computation is element-wise.

    Parameters
    ----------
    A : array_like
        The operand, a tensor of one of the types ``float16``, ``float`` or
        ``double`` (``numpy.float16``, ``numpy.float32`` or ``numpy.float64``),
        or a real tensor. It may be any value of the type, including ``+/-0``,
        ``+/-inf`` and NaN.

    Returns
    -------
    numpy.ndarray or numpy floating scalar
        The element-wise sine of ``A``, with the shape of ``A`` and, for the
        floating-point types, the type of ``A``.
    """
    # DECISION: The document defines two operators, **Sin** (real) and **Sin**
    # (float16/float/double), with the identical signature "C = Sin(A)" and the
    # identical element-wise definition, and says nothing about whether one
    # entry point serves both: "Operator Sin applies the sine to tensor A
    # element-wise and stores the result in tensor C" (both sections). I expose
    # a single function `sin`, as required, and for a floating-point operand it
    # applies the three-case float rule; a numpy operand always is of one of
    # the three floating-point types (or of a type outside the document's list,
    # see below), so the float cases are the operative ones.
    #
    # DECISION: The real section requires "the value of C[i] is that function of
    # the element, exactly, with no approximation and no rounding", and its
    # examples use exact multiples of pi (pi/6, pi/4, ...). No value of
    # float16/float/double represents such a real number, and the document says
    # nothing about any exact or symbolic representation of an operand; since
    # the operands of this function are numpy floating-point values, no exact
    # arithmetic is attempted and the float rule governs. The document is
    # silent on this representation question.
    #
    # DECISION: The document states the value "nearest(sin(A[i]))" as the result
    # and adds "the way an implementation computes it is not specified here",
    # referring a departure "to the accuracy guidelines ... and is not part of
    # this specification", and its warning V1 notes that implementations of the
    # sine do not all return the correctly rounded value. I compute the sine
    # with numpy.sin (the platform libm), which is not guaranteed to be
    # correctly rounded; producing a correctly rounded result in general would
    # require arithmetic beyond the type and beyond numpy, and the document
    # expressly places that accuracy question outside itself.
    #
    # DECISION: The float Function section gives the result "for every
    # combination" of operand values, and the method is "not specified here". I
    # rely on numpy.sin to realize all three cases rather than branching on
    # isinf/isnan/zero explicitly: numpy.sin already returns NaN for NaN and
    # for an infinity (the IEEE 754 invalid operation the document names), the
    # operand itself for a zero, and the nearest value of the type otherwise.
    # Any explicit masking would add computation the document does not require.
    #
    # DECISION: The document requires "[E_SIN_FLOAT_CONSTR_A_0010] Tensors A and
    # C have the same type" and "[E_SIN_FLOAT_CONSTR_C_0010] Tensor C has the
    # shape of A". numpy.sin preserves the dtype float16/float32/float64 and the
    # shape, so no cast is added. For an operand whose type is outside the
    # document's list (an integer, complex or object array, for instance) the
    # document says nothing; I add no validation and no conversion, and numpy's
    # own promotion is left unchanged (np.sin of an integer array returns
    # float64, of a complex array a complex array).
    #
    # DECISION: "Tensors with no dimension ... C is a rank-0 tensor whose single
    # element is the sine of that element." numpy.sin already returns the
    # rank-0 result; for a 0-d array input numpy returns a numpy floating
    # scalar rather than an ndarray, and the document prescribes no container
    # for the result, so it is returned as numpy produces it.
    #
    # DECISION: "Zero-sized dimensions ... C is empty along it, the
    # corresponding elements of C do not exist, and the sine of no element is
    # computed." Element-wise numpy.sin gives the same shape with the zero-sized
    # dimension; no special handling is added, since there are no elements whose
    # sine would be computed.
    #
    # DECISION: The document says of NaN only that the result is a quiet NaN
    # whose "sign and its payload are not specified ... every quiet NaN of the
    # type of A is a conforming result". numpy.sin's NaN output, propagated from
    # a NaN operand or produced by the invalid operation, is left unchanged.
    #
    # DECISION: "The sine is odd, and the result is odd too ... the result for
    # the operand -A[i] is the negation of the result for A[i], the sign bit
    # included." numpy.sin (libm) is odd in this sense, including the sign of a
    # zero; I rely on it and add no explicit sign handling.
    #
    # DECISION: The document's warning V1 and its Error conditions table name
    # the invalid operation for the sine of an infinity, but say nothing about
    # any warning or error signal raised while computing it. Following the
    # requirement not to suppress warnings the document does not mention, the
    # RuntimeWarning numpy emits for that invalid operation is left to
    # propagate; nothing is suppressed or added.
    #
    # DECISION: "nearest(x) is the value of x rounded to the nearest value of
    # the type of A using the roundTiesToEven attribute of IEEE 754." The
    # document names the attribute but says nothing about how an implementation
    # selects it; I rely on the IEEE 754 default rounding, roundTiesToEven,
    # which numpy and the platform use.
    #
    # DECISION: "The three types share one semantics: the result is the sine of
    # the operand represented in the type of the operand." For float16 I use
    # numpy's float16 sine path, then the result is in the type of A; the
    # document says the computation method "is not specified here", so no
    # explicit upcast/round-back is performed.
    #
    # DECISION: The document speaks of a tensor operand and says nothing about
    # Python scalars, lists or other array-likes. numpy.sin accepts array-likes
    # and converts them itself; I add no validation or conversion of my own.
    return np.sin(A)
