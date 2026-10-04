import numpy as np

# DECISION: The document specifies two variants: "Acos (real)" and "Acos (float)" for float16, float, double.
# The function `acos` implements both. For float types, we follow the float variant. For other types,
# we follow the real variant as closely as possible using floating-point arithmetic. The document does
# not specify how to implement the real variant in a finite-precision environment; we use numpy's
# floating-point arccosine.

def acos(X):
    """
    Acos operator.

    Computes the arccosine of tensor X element-wise and stores the result in tensor Y.

    For the real variant:
    "Operator Acos computes the arccosine of tensor X element-wise and stores the result in tensor Y.
    The arccosine is the inverse of the cosine: each element of Y is the angle of [0, pi] whose cosine
    is the corresponding element of X."
    "For any tensor index i of the result Y: Y[i] = acos(X[i]) where acos is defined by the following
    property: for every x in [-1, 1], acos(x) is the unique value y such that cos(y) = x and y in [0, pi]."
    "The arccosine is defined on [-1, 1]: for x in that interval, cos takes the value x at exactly one
    point of [0, pi], whereas for x outside it, no real number has x as its cosine."
    "Every element of Y lies in [0, pi]."
    "acos(1) = 0, acos(0) = pi/2 and acos(-1) = pi."
    "The arccosine is strictly decreasing on [-1, 1]."
    "For real numbers the result is exact: Y[i] is the real number acos(X[i]), with no approximation."
    "A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. The result is a
    rank-0 tensor whose single element is the arccosine of the single element of X."
    "An operand may have a zero-sized dimension. The result has the same shape as X and is then empty."

    For the float variant (float16, float, double):
    "Operator Acos computes the arccosine of tensor X element-wise and stores the result in tensor Y.
    Each element of Y is the value of the arccosine of the corresponding element of X, rounded to the
    type of X and Y."
    "For any tensor index i of the result Y: Y[i] = round(acos(X[i])) where acos is the arccosine of
    the real numbers, defined in the section for type real, and round(y) is the value of y rounded to
    the nearest value of the type of X and Y using the roundTiesToEven attribute of IEEE 754, the
    exponent range of the type being unbounded."
    "An operand that is NaN gives a NaN result; that NaN is a quiet NaN whose sign and payload are not
    specified, so that every quiet NaN of the type of X and Y is a conforming result."
    "An operand that is +0 or -0 gives pi/2 rounded to the type of X and Y: the arccosine of a null
    value is pi/2 whatever the sign of the zero, so that +0 and -0 give the same result."
    "The result lies in [0, pi], which is within the range of the three types: it is never an infinity,
    and the rounding of the exact arccosine to the type of X and Y never overflows."
    "The result is null only when the operand is 1, and it is then +0."
    "The result is the real arccosine rounded to the type of X and Y; it is exact only when the real
    arccosine is representable in that type."

    Constraints:
    - X: Every element of X lies in [-1, 1].
    - Y: Tensor Y has the same shape as X.
    - Tensors X and Y have the same type (for float types).

    Attributes: none.
    """
    # DECISION: The document says "Tensors X and Y have the same type" for float types. We preserve the
    # dtype of X for float16 and float32 by casting back after computation. For other types, the document
    # does not specify a type for Y; we return float64 as numpy's default for arccos.
    # DECISION: The document says for float types: "the value of y rounded to the nearest value of the
    # type of X and Y using the roundTiesToEven attribute of IEEE 754, the exponent range of the type
    # being unbounded." To achieve correctly rounded results for float16 and float32, we compute the
    # arccosine in float64 and then cast to the original type. The document does not specify the
    # precision of intermediate computation. Double rounding from float64 to float16/float32 is safe
    # because float64 has enough precision (53 bits) to guarantee correct rounding for these types.
    # DECISION: The document says "An operand that is NaN gives a NaN result; that NaN is a quiet NaN
    # whose sign and payload are not specified". We rely on numpy's NaN propagation; the exact payload
    # is whatever numpy produces.
    # DECISION: The document says "An operand that is +0 or -0 gives pi/2 rounded to the type of X and Y".
    # We rely on numpy's arccos, which returns the same value for +0 and -0.
    # DECISION: The document says "The result has the same shape as X". We rely on numpy's shape
    # preservation. For rank-0 tensors, numpy returns a 0-d array; for zero-sized dimensions, an empty
    # array. The document does not specify how to handle these in code; we rely on numpy's semantics.
    # DECISION: The document says "Every element of X lies in [-1, 1]" and that out-of-range values are
    # "ruled out by the precondition". We do not add validation. If an out-of-range value is passed,
    # numpy's arccos returns NaN and may emit a RuntimeWarning. The document does not specify behavior
    # for out-of-range inputs.
    # DECISION: The document says "For real numbers the result is exact: Y[i] is the real number
    # acos(X[i]), with no approximation." Our implementation uses floating-point arithmetic and cannot
    # be exact. This is a deviation from the real variant, but the float variant is the concrete one
    # for ONNX.
    # DECISION: The document says "the exponent range of the type being unbounded" for rounding. Since
    # the result lies in [0, pi], which is within the finite range of all three types, this detail does
    # not affect the result. We rely on numpy's cast, which uses the actual exponent range.
    # DECISION: The document says "Operator Acos has no attribute." We have no attributes.

    X = np.asarray(X)
    if X.dtype == np.float16 or X.dtype == np.float32:
        return np.arccos(X.astype(np.float64)).astype(X.dtype)
    else:
        return np.arccos(X)
