import numpy as np


def acos(X):
    """
    Compute the element-wise arccosine of X.

    This function implements the Acos operator as specified.

    For real numbers (E_ACOS_REAL_FUNC_0010):
    "Operator Acos computes the arccosine of tensor X element-wise and stores
    the result in tensor Y. The arccosine is the inverse of the cosine: each
    element of Y is the angle of [0, pi] whose cosine is the corresponding
    element of X."
    "For any tensor index i of the result Y: Y[i] = acos(X[i])"
    "where acos is defined by the following property: for every x in [-1, 1],
    acos(x) is the unique value y such that cos(y) = x and y in [0, pi]."
    "The result is the value of that branch, not a way of computing it."
    "Domain and range: The arccosine is defined on [-1, 1] ... Every element
    of Y lies in [0, pi]."
    "Values at the bounds and monotonicity: acos(1) = 0, acos(0) = pi/2 and
    acos(-1) = pi. The arccosine is strictly decreasing on [-1, 1]."
    "Exactness: For real numbers the result is exact: Y[i] is the real number
    acos(X[i]), with no approximation."
    "Tensors with no dimension: A tensor with no dimension (a rank-0 tensor)
    is a tensor whose shape is empty. The result is a rank-0 tensor whose
    single element is the arccosine of the single element of X."
    "Zero-sized dimensions: An operand may have a zero-sized dimension. The
    result has the same shape as X and is then empty."

    For floating-point types (E_ACOS_FLOAT_FUNC_0010):
    "Operator Acos computes the arccosine of tensor X element-wise and stores
    the result in tensor Y. Each element of Y is the value of the arccosine of
    the corresponding element of X, rounded to the type of X and Y."
    "For any tensor index i of the result Y: Y[i] = round(acos(X[i]))"
    "where acos is the arccosine of the real numbers ... and round(y) is the
    value of y rounded to the nearest value of the type of X and Y using the
    roundTiesToEven attribute of IEEE 754, the exponent range of the type
    being unbounded."
    "Values of the operands: The operands may be any of the values of the
    type. An operand that is NaN gives a NaN result; that NaN is a quiet NaN
    whose sign and payload are not specified, so that every quiet NaN of the
    type of X and Y is a conforming result. An operand that is +inf or -inf
    lies outside [-1, 1] and is ruled out by the constraint
    E_ACOS_FLOAT_CONSTR_X_0010. An operand that is +0 or -0 gives pi/2
    rounded to the type of X and Y: the arccosine of a null value is pi/2
    whatever the sign of the zero, so that +0 and -0 give the same result."
    "Values of the result: The result lies in [0, pi], which is within the
    range of the three types: it is never an infinity, and the rounding of
    the exact arccosine to the type of X and Y never overflows. The result is
    null only when the operand is 1, and it is then +0."
    "Exactness: The result is the real arccosine rounded to the type of X and
    Y; it is exact only when the real arccosine is representable in that type.
    For instance acos(1) = 0 is exact, whereas acos(0) = pi/2 is representable
    in none of the three types and the result is the nearest value of the
    type."
    "Tensors with no dimension: A tensor with no dimension (a rank-0 tensor)
    is a tensor whose shape is empty. The result is a rank-0 tensor whose
    single element is the arccosine of the single element of X."
    "Zero-sized dimensions: An operand may have a zero-sized dimension. The
    result has the same shape as X and is then empty."

    Constraints:
    E_ACOS_REAL_CONSTR_X_0010 / E_ACOS_FLOAT_CONSTR_X_0010: Every element of
    X lies in [-1, 1].
    E_ACOS_FLOAT_CONSTR_X_0020: Tensors X and Y have the same type.
    E_ACOS_REAL_CONSTR_Y_0010 / E_ACOS_FLOAT_CONSTR_Y_0010: Tensor Y has the
    same shape as X.

    Attributes: Operator Acos has no attribute.
    """
    # DECISION: The specification defines Acos for "real" and for
    # float16/float/double. It does not define behavior for other dtypes
    # (e.g. integers). We use numpy's arccos, which promotes non-floating
    # inputs to float64. The document says nothing about integer inputs.
    X = np.asarray(X)

    if np.issubdtype(X.dtype, np.floating):
        # DECISION: The document requires the result to be the real arccosine
        # rounded to the type of X and Y using roundTiesToEven
        # (E_ACOS_FLOAT_FUNC_0010). It does not specify the algorithm. We
        # compute using numpy's arccos in float64 and then cast to the target
        # type. This may not be exactly correctly rounded in all cases.
        # DECISION: The document says "the exponent range of the type being
        # unbounded" (E_ACOS_FLOAT_FUNC_0010). We use float64 intermediate,
        # which has a bounded exponent. However, the result is in [0, pi], so
        # no overflow/underflow occurs.
        result = np.arccos(X.astype(np.float64))
        # DECISION: The document requires Y to have the same type as X
        # (E_ACOS_FLOAT_CONSTR_X_0020). We cast the result to X.dtype.
        # DECISION: The document requires roundTiesToEven; we rely on numpy's
        # casting, which uses the default rounding mode (roundTiesToEven for
        # IEEE 754). The document does not specify the rounding mode of the
        # intermediate computation.
        result = result.astype(X.dtype, copy=False)
    else:
        # DECISION: The document says for real numbers the result is exact
        # (E_ACOS_REAL_FUNC_0010). In floating-point arithmetic, exactness is
        # not generally possible. We represent the result in float64 for
        # non-floating inputs. The document does not specify how to represent
        # exact real results in a finite-precision environment.
        result = np.arccos(X.astype(np.float64))

    # DECISION: The document says "An operand that is NaN gives a NaN result;
    # that NaN is a quiet NaN whose sign and payload are not specified"
    # (E_ACOS_FLOAT_FUNC_0010). We rely on numpy's arccos to produce a quiet
    # NaN. The document does not specify the exact NaN payload; we use
    # numpy's default.
    # DECISION: The document says "An operand that is +0 or -0 gives pi/2
    # rounded to the type of X and Y" (E_ACOS_FLOAT_FUNC_0010). We rely on
    # numpy's arccos, which returns pi/2 for both +0 and -0. The document
    # does not specify the exact rounding of pi/2; we use numpy's result.
    # DECISION: The document says "The result is null only when the operand
    # is 1, and it is then +0" (E_ACOS_FLOAT_FUNC_0010). We rely on numpy's
    # arccos(1) = 0.0. The document does not specify the sign of zero for
    # other cases; we use numpy's result.
    # DECISION: The document says "Invalid operation ... ruled out by the
    # precondition" (Error conditions). We do not add validation. If X
    # contains values outside [-1,1], numpy's arccos returns NaN and may
    # emit a RuntimeWarning. The document does not specify behavior when the
    # precondition is violated; we let numpy compute and warn.
    # DECISION: The document says "Tensors with no dimension" and "Zero-sized
    # dimensions". We rely on numpy's array handling. The document does not
    # specify the exact memory layout; we use numpy's default.
    # DECISION: The document says "Operator Acos has no attribute." We do not
    # accept any attributes. The document does not specify how to handle
    # extra arguments; our function takes only X.

    return result
