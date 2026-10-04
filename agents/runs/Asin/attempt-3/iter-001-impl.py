import numpy as np


def asin(X):
    """
    Asin operator.

    Computes the arcsine of X element-wise.

    Specification references
    ------------------------
    E_ASIN_REAL_FUNC_0010:
        Operator Asin computes the arcsine of tensor X element-wise and stores
        the result in tensor Y. The arcsine is the inverse of the sine: the
        result is the angle whose sine is the element of X.
        For any tensor index i of the result Y:
            Y[i] = arcsin(X[i])
        where arcsin(x) is the unique value y such that sin(y) = x and
        y ∈ [-π/2, π/2].
        The result is exact: Y[i] is the real number arcsin(X[i]), with no
        approximation.

    E_ASIN_FLOAT_FUNC_0010:
        Operator Asin computes the arcsine of tensor X element-wise according
        to IEEE 754 floating-point semantics and stores the result in tensor Y.
        Tensors X and Y have the same floating-point type, and the result is a
        value of that type.
        For any tensor index i of the result Y:
            Y[i] =
            NaN       if X[i] is NaN or |X[i]| > 1
            X[i]      if X[i] is ±0
            round(arcsin(X[i])) otherwise
        where arcsin(x) is the unique value y such that sin(y) = x and
        y ∈ [-π/2, π/2], i.e. the exact arcsine of the real number x,
        and round(x) is the value of x rounded to the nearest value of the
        type of X and Y using the roundTiesToEven attribute of IEEE 754,
        the exponent range of the type being unbounded.

    The result never overflows. A result whose magnitude is below the smallest
    normal value of the type is a subnormal value: the rounding is the one
    defined above, with an unbounded exponent range, so that the result is the
    subnormal value nearest to the exact arcsine and is not flushed to zero.

    Sign and monotonicity: the arcsine is an odd function, so that the result
    has the sign of the element of X for every element that is not NaN.
    The result lies in [-π/2, π/2] for every element of X that is not NaN.

    Tensors with no dimension: a rank-0 tensor is a tensor whose shape is
    empty. The result is a rank-0 tensor whose single element is the arcsine
    of the single element of X.

    Zero-sized dimensions: tensor X may have a zero-sized dimension. Tensor Y
    then has the same zero-sized dimension and is empty.

    Constraints
    -----------
    E_ASIN_REAL_CONSTR_X_0010: Every element of X lies in [-1, 1].
    E_ASIN_REAL_CONSTR_Y_0010: Tensor Y has the same shape as tensor X.
    E_ASIN_FLOAT_CONSTR_X_0010: Tensors X and Y have the same type.
    E_ASIN_FLOAT_CONSTR_Y_0010: Tensor Y has the same shape as tensor X.
    E_ASIN_FLOAT_CONSTR_Y_0020: see E_ASIN_FLOAT_CONSTR_X_0010.

    Implementation notes
    --------------------
    This implementation uses numpy.arcsin. It preserves the shape and, for
    floating-point inputs, the dtype of X. It preserves signed zeros, returns
    NaN for NaN and for |X| > 1 (including infinities), and does not flush
    subnormals to zero.

    DECISION: The document defines Asin for both real and floating-point types.
    For numpy floating-point inputs, we implement the floating-point semantics
    (E_ASIN_FLOAT_FUNC_0010), because the real section's precondition (elements
    in [-1, 1]) is not enforced and the float section explicitly allows any
    value of the type, including NaN and infinities. The real section's
    exactness requirement is not achievable with floating-point arithmetic;
    the float section's rounding requirement is the applicable one.

    DECISION: The document does not specify an algorithm to compute the
    correctly rounded arcsine. We use numpy.arcsin, which may not be correctly
    rounded to the nearest value of the type as required by
    E_ASIN_FLOAT_FUNC_0010. The document's accuracy guidelines are not
    available to us.

    DECISION: The document does not specify behavior for non-floating-point
    inputs. We apply numpy.arcsin, which upcasts integer inputs to float64.
    This violates the type consistency constraint E_ASIN_FLOAT_CONSTR_X_0010
    (X and Y have the same type), but the document does not require us to
    validate the input type.

    DECISION: The document does not mention warnings. numpy.arcsin emits a
    RuntimeWarning for invalid operations (|X| > 1 or infinities). We do not
    suppress this warning, as the document does not mention it and the
    instruction is not to suppress warnings the document does not mention.

    DECISION: The document specifies that the NaN of the first case is a quiet
    NaN whose sign and payload are not specified. numpy.arcsin returns a quiet
    NaN; its sign and payload are whatever numpy produces, which is conforming.

    DECISION: The document specifies that a rank-0 tensor yields a rank-0
    tensor, and a zero-sized dimension yields the same zero-sized dimension.
    numpy.arcsin preserves the shape, including rank-0 and zero-sized
    dimensions. We use np.asarray to ensure scalar inputs become 0-d arrays.

    DECISION: The document specifies that the result of Asin(-0) is -0 and
    Asin(+0) is +0. numpy.arcsin preserves the sign of zero.

    DECISION: The document specifies that subnormal results are not flushed to
    zero. numpy.arcsin does not flush subnormals by default.

    DECISION: The document specifies that X and Y have the same type
    (E_ASIN_FLOAT_CONSTR_X_0010). For floating-point inputs, we cast the result
    to the dtype of X to ensure this, although numpy.arcsin already preserves
    the dtype for float16, float32, and float64.

    DECISION: The document only specifies the types float16, float, and double.
    For other floating-point types (e.g., float128), we apply the same
    floating-point semantics and preserve the dtype, as the document does not
    specify otherwise.

    Parameters
    ----------
    X : array_like
        Input tensor. For the floating-point semantics, X may contain any
        values of its floating-point type, including NaN, ±inf, and values
        outside [-1, 1].

    Returns
    -------
    Y : ndarray
        The element-wise arcsine of X, with the same shape as X and, for
        floating-point inputs, the same dtype as X.
    """
    X = np.asarray(X)
    Y = np.arcsin(X)
    if np.issubdtype(X.dtype, np.floating):
        Y = Y.astype(X.dtype, copy=False)
    return Y
