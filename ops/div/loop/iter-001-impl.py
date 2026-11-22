"""Element-wise division operator ``Div`` (ONNX opset 14).

The specification of ``Div`` is split into four sections, one per family of
types, and the operator has one input pair and one output:

    C = Div(A, B)

with ``A`` the dividend, ``B`` the divisor and ``C`` the element-wise result.
All four sections state, in identical words, the broadcasting rule:

    "Tensors A and B are broadcast to the shape of C following the
    multidirectional (Numpy-style) broadcasting rules [...] Tensor C has the
    resulting shape."

and the rank-0 rule:

    "A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is
    empty. It is broadcast to the shape of the other operand [...]"

and the zero-sized-dimension rule:

    "An operand may have a zero-sized dimension. Such a dimension is
    compatible with a dimension of size 1 of the other operand, and with an
    equal zero-sized dimension; the corresponding dimension of C then has size
    0, and C is empty."

Those three rules are exactly what numpy's own broadcasting does, so the
implementation obtains them from numpy rather than restating them.

The four sections differ in how a single quotient is formed:

* ``real`` -- the exact quotient in R, the domain being restricted to a
  divisor with no null element ("The quotient of two real numbers exists only
  when the divisor is non-null").
* ``float16`` / ``float`` / ``double`` -- the IEEE 754 quotient, with the five
  cases of [E_DIV_FLOAT_FUNC_0010] spelled out (NaN propagation, invalid
  operation, division by zero, null quotients, and rounding of the exact
  quotient with roundTiesToEven).
* ``int8`` / ``int16`` / ``int32`` / ``int64`` -- the quotient of the two
  elements "truncated toward zero", read modulo 2**n, the single out-of-range
  quotient 2**(n-1) (from (-2**(n-1)) / (-1)) giving -2**(n-1).
* ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` -- the floor of the exact
  quotient, which for non-negative operands is the same as the truncation and
  which "never leaves the range of the type".

The module dispatches on the dtype of ``A``.

Attributes: the specification states, in every section, "Operator Div has no
attribute." The entry point therefore takes no keyword parameter.

Outputs: the specification gives exactly one output, ``C``, so the entry point
returns a single tensor and not a tuple.

# DECISION: the specification keys its four sections by the *type of the
# operands* ("Div (real, real)", "Div (int, int)", ...) but says nothing about
# how an implementation selects a section. Dispatching on the dtype of the
# dividend ``A`` is my choice; the constraint that "Tensors A, B and C have the
# same type" is a precondition, not a dispatch rule.
#
# DECISION: the "real" section describes an abstract type of exact real
# numbers ("the quotient is the quotient in R of the two elements", "The
# division is exact: C[i] is the quotient in R, with no approximation"). No
# numpy dtype represents that type exactly, so an input of a floating-point
# dtype is routed to the floating-point section, which is the concrete
# realisation the specification itself gives for real-valued tensors
# ("float is in {float16, float, double}").
"""

import numpy as np


def _div_float(A, B):
    """Floating-point division, per [E_DIV_FLOAT_FUNC_0010].

    The specification requires, for a tensor index i of the result:

        C[i] = A~[i] /(f) B~[i] =
            NaN   if A~[i] or B~[i] is NaN
            NaN   if A~[i] and B~[i] are both +-inf, or both +-0
            +-inf if A~[i] is finite and non-null and B~[i] is +-0, or
                    A~[i] is +-inf and B~[i] is finite and non-null
            +-0   if A~[i] is +-0 and B~[i] is finite and non-null, or
                    A~[i] is finite and B~[i] is +-inf
            round(A~[i] / B~[i]) otherwise

    where "round(x) is the value of x rounded to the nearest value of the type
    of A and B using the roundTiesToEven attribute of IEEE 754".

    All five cases are precisely IEEE 754 division, and the specification says
    so: "The three types share one semantics: the IEEE 754 standard defines the
    same division for all of them, and they differ only by their precision and
    by the range of the values they represent." The numpy true-division
    operator is IEEE 754 division on the floating-point dtypes, including the
    special numbers, so no case is restated here.

    The error-condition table of the float section rules that the invalid
    operation, division by zero, overflow and underflow "is part of the nominal
    behavior of the operator; none of them is an error", so no case is trapped.

    # DECISION: the specification does not mention warnings, and requires the
    # invalid operation and the division by zero to be nominal. numpy reports
    # both with a RuntimeWarning; the specification neither requires nor
    # forbids such a report, and the instruction not to suppress warnings the
    # document does not mention is followed, so no `errstate` is installed.
    #
    # DECISION: the specification requires C[i] to be the *exact* quotient
    # rounded once ("round(A~[i] / B~[i])", with "the exponent range of the
    # type being unbounded when deciding an overflow"). numpy evaluates
    # float16 division at a wider internal precision before rounding back to
    # float16, which can double-round a quotient that is not representable in
    # float16. The specification anticipates such departures: "The departure of
    # an implementation from these values is the subject of the profile's
    # accuracy guidelines."
    """
    return A / B


def _div_signed_int(A, B, dtype):
    """Truncating division for the signed integer types, per [E_DIV_INT_FUNC_0010].

    The specification requires, for a tensor index i of the result:

        C[i] = trunc(A~[i] / B~[i])            if -2**(n-1) <= trunc(A~[i] /
                                                    B~[i]) <= 2**(n-1) - 1
        C[i] = trunc(A~[i] / B~[i]) - 2**n     otherwise

    with "trunc(x) is the integer part of x, i.e. x rounded toward zero", and it
    adds: "The second case is the complement of the first: exactly one of the
    two cases applies to every pair of operands whose divisor is non-null".

    The specification also identifies the only operand pair reaching the second
    case: "Only the value 2**(n-1) leaves the range of the type, so that the
    second case of the definition is the case trunc(A~[i]/B~[i]) = 2**(n-1),
    reached by that single pair of operands" -- the pair (-2**(n-1), -1). Both
    cases are implemented literally below: the truncated quotient is computed,
    then any value outside the type's range is reduced by 2**n.

    The truncation is obtained from numpy's floor division, which the
    specification contrasts with truncation: "It is the quotient ONNX
    prescribes for the integer types, and it differs from the floor: for int8,
    7/2 = 3 and -7/2 = -3, whereas the floor of -3.5 is -4." The floor quotient
    q and the remainder r satisfy a = q*b + r with r of the sign of b; the
    truncated quotient is the floor quotient plus one exactly when the
    remainder is non-null and the operands have opposite signs.

    The operands are widened to int64 for the computation so that no
    intermediate product overflows, except for the single pair that overflows
    even in int64, which is diverted before the division and restored after it.

    # DECISION: the specification states "a null element of B makes the exact
    # quotient undefined and the result of the operator undetermined", and
    # leaves what happens "outside this specification", naming as acceptable
    # both "a result produced by the implementation's own handling of the case"
    # and an explicit report to the caller. I add no divisor test and let
    # numpy's own handling of integer division by zero stand (a null result
    # together with its RuntimeWarning), which is the first of the two named
    # dispositions.
    """
    info = np.iinfo(dtype)
    n = info.bits
    min_value = int(info.min)
    max_value = int(info.max)

    a = A.astype(np.int64)
    b = B.astype(np.int64)

    # The single pair whose truncated quotient leaves the range of the type:
    # (-2**(n-1)) / (-1) = 2**(n-1), whose result is 2**(n-1) - 2**n =
    # -2**(n-1). Diverting it keeps the division inside int64 even when the
    # type is int64.
    out_of_range_pair = (a == min_value) & (b == np.int64(-1))
    safe_b = np.where(out_of_range_pair, np.int64(1), b)

    # Floor quotient and remainder, as numpy defines them for integers.
    quotient = a // safe_b
    remainder = a % safe_b

    # Floor -> truncation: subtract the extra unit the floor took when the
    # quotient is negative and the division is inexact.
    quotient = quotient + ((remainder != 0) & ((a < 0) != (safe_b < 0)))

    # The second case of the definition, applied literally.
    if n < 64:
        quotient = np.where(quotient > max_value, quotient - (1 << n), quotient)

    # The diverted pair: trunc(a/b) = 2**(n-1), which the second case maps to
    # 2**(n-1) - 2**n = -2**(n-1).
    quotient = np.where(out_of_range_pair, np.int64(min_value), quotient)

    return quotient.astype(dtype)


def _div_unsigned_int(A, B):
    """Division for the unsigned integer types, per [E_DIV_UINT_FUNC_0010].

    The specification requires, for a tensor index i of the result:

        C[i] = floor(A~[i] / B~[i])

    and notes: "The operands are non-negative, so that their exact quotient is
    non-negative and its floor is at most the dividend: 0 <= floor(a/b) <= a <=
    2**n - 1 for every b >= 1. The result of a division by a non-null element is
    therefore the floor itself, with no reduction modulo 2**n and no overflow".
    It adds: "The truncation and the floor coincide on the unsigned types".

    numpy's floor division on unsigned dtypes is exactly that floor, and it
    keeps the dtype of the operands, which is the dtype the specification
    requires of ``C``: "Tensors A, B and C have the same unsigned n-bit type".

    # DECISION: the specification states "a null element of B makes the exact
    # quotient undefined and the result of the operator undetermined", and
    # leaves what happens "outside this specification", naming as acceptable
    # both "a result produced by the implementation's own handling of the case"
    # and an explicit report to the caller. No divisor test is added; numpy's
    # own handling of unsigned division by zero stands (a null result together
    # with its RuntimeWarning), which is the first of the two named
    # dispositions.
    """
    return A // B


def div(A, B):
    """Element-wise division of tensor ``A`` by tensor ``B``.

    Implements the operator ``Div`` of the specification, whose signature is,
    in every one of its four type sections:

        C = Div(A, B)

    where ``A`` is the dividend, ``B`` the divisor and ``C`` "is the
    element-wise result of the division of A by B".

    The result has the shape of ``A`` and ``B`` broadcast to a common shape
    ("Tensor C has the shape of A and B broadcast to a common shape"), which is
    numpy's multidirectional broadcasting.

    The quotient itself depends on the type of the operands:

    * a floating-point dtype follows [E_DIV_FLOAT_FUNC_0010], the IEEE 754
      quotient, in which "No element of B is excluded: every combination of
      operands of the type, including the special numbers, has a result of the
      type";
    * a signed integer dtype follows [E_DIV_INT_FUNC_0010], the quotient
      "truncated toward zero", the single out-of-range quotient 2**(n-1) being
      reduced by 2**n;
    * an unsigned integer dtype follows [E_DIV_UINT_FUNC_0010], the floor of
      the exact quotient, which "never leaves the range of the type".

    "Operator Div has no attribute", so there is no keyword parameter.

    # DECISION: the specification says nothing about the dtype of the returned
    # tensor beyond the constraint "Tensors A, B and C have the same type". No
    # promotion, demotion or cast of the operands is performed; numpy's own
    # treatment of a pair of operands of differing dtypes is left in place,
    # since the specification states the constraint but attaches no disposition
    # to its violation.
    #
    # DECISION: the specification says nothing about inputs that are not numpy
    # arrays. ``np.asarray`` is applied so that a sequence of values is
    # accepted, which is the least surprising reading of "tensor" here; this
    # conversion is not required by any sentence of the document.
    #
    # DECISION: the specification defines the operator only for the types
    # "real", float16/float/double, int8/int16/int32/int64 and
    # uint8/uint16/uint32/uint64. It says nothing about any other dtype
    # (bool, complex, object, ...). For such a dtype the numpy true division is
    # returned unchanged, which is a guess and not a reading of the document.
    """
    A = np.asarray(A)
    B = np.asarray(B)

    kind = A.dtype.kind
    if kind == "f":
        return _div_float(A, B)
    if kind == "i":
        return _div_signed_int(A, B, A.dtype)
    if kind == "u":
        return _div_unsigned_int(A, B)
    return A / B
