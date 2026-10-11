import numpy as np


def maxpool(
    X,
    kernel_shape,
    strides=None,
    pads=None,
    auto_pad="NOTSET",
    ceil_mode=0,
    dilations=None,
    storage_order=0,
):
    """
    MaxPool operator as specified in the document.

    This function implements the MaxPool operator for the real, float16/float/double,
    int8, and uint8 types.  It applies max pooling to the input tensor ``X``: it slides
    a window of the shape given by ``kernel_shape`` over the spatial axes of ``X``,
    with the steps given by ``strides``, and stores in ``Y`` the maximum of the
    elements of ``X`` covered by each window.

    The first two axes of ``X`` are the batch axis and the channel axis; they are not
    pooled.  The remaining ``rX - 2`` axes are the spatial axes, numbered from 0 to
    ``rX - 3`` in the order of the axes of ``X``.  The number of spatial axes is
    denoted ``m = rX - 2``; the operator is applicable to any ``m >= 1``.

    For any tensor index ``i = (n, c, j_0, ..., j_{m-1})`` of the result ``Y``:

        Y[n, c, j_0, ..., j_{m-1}] =
            max_{k_0, ..., k_{m-1}} X_p[n, c, j_0 * s_0 + k_0, ..., j_{m-1} * s_{m-1} + k_{m-1}]

    where ``X_p`` is the tensor obtained by padding ``X`` with the values of ``pads``,
    and the maximum is taken over the elements of ``X`` covered by the window, the
    padded elements being excluded.

    The spatial dimensions of ``Y`` are

        dY_{t+2} = floor((dX_{t+2} + p_{t,begin} + p_{t,end} - dK_t) / s_t) + 1

    for every ``t`` in ``[0, m-1]``, and ``dY_0 = dX_0``, ``dY_1 = dX_1``.

    Parameters
    ----------
    X : array_like
        The tensor to which the max pooling is applied.  Its first two axes are the
        batch axis and the channel axis, and its remaining axes are the spatial axes.
        The rank of ``X`` must be at least 3.
    kernel_shape : list of int
        The size of the window along each spatial axis.  It has ``m`` values, one per
        spatial axis, and every value is at least 1.
    strides : list of int, optional
        The step of the window along each spatial axis.  It has ``m`` values, one per
        spatial axis, and every value is at least 1.
        DECISION: The document does not state a default for ``strides``.  We use the
        ONNX default of ``[1] * m``.
    pads : list of int, optional
        The number of elements added before and after ``X`` along each spatial axis.
        It has ``2m`` values, the first ``m`` being the numbers added at the beginning
        of the axes and the last ``m`` the numbers added at the end, in the order of
        the spatial axes.  Every value is at least 0.
        DECISION: The document does not state a default for ``pads``.  We use the ONNX
        default of ``[0] * (2 * m)``.
    auto_pad : str, optional
        Selects the way the padding is computed.  It is restricted to ``"NOTSET"`` by
        [R1], so that the padding is the one given by ``pads``.
        DECISION: The document does not state a default for ``auto_pad``.  We use the
        ONNX default of ``"NOTSET"``.  The attribute is accepted but not used, since
        the specification restricts it to ``"NOTSET"``.
    ceil_mode : int, optional
        Selects the rounding of the output shape.  It is restricted to 0 by [R3], so
        that the floor is used.
        DECISION: The document does not state a default for ``ceil_mode``.  We use the
        ONNX default of 0.  The attribute is accepted but not used, since the
        specification restricts it to 0.
    dilations : list of int, optional
        Gives the dilation along each spatial axis.  It is restricted to 1 along each
        spatial axis by [R4], so that the window covers consecutive elements of ``X``.
        DECISION: The document does not state a default for ``dilations``.  We use the
        ONNX default of ``[1] * m``.  The attribute is accepted but not used, since
        the specification restricts it to 1.
    storage_order : int, optional
        Selects the order in which the flattened index of the output ``Indices`` is
        computed.  It is restricted to 0 by [R2], and the output ``Indices`` is not
        supported by this profile, as stated by [R5].
        DECISION: The document does not state a default for ``storage_order``.  We use
        the ONNX default of 0.  The attribute is accepted but not used, since the
        specification restricts it to 0 and the ``Indices`` output is not supported.

    Returns
    -------
    Y : numpy.ndarray
        The result of the max pooling of ``X``.  The output ``Indices`` is not
        supported by this profile (restriction [R5]), so only ``Y`` is returned.

    Notes
    -----
    DECISION: The document says "A spatial dimension of X may have size 0.  The
    formula above then gives a spatial dimension of Y of size 0 when
    p_{t,begin} + p_{t,end} >= dK_t, and the corresponding dimension of Y is empty;
    otherwise the operator is not applicable".  This statement is contradictory with
    the formula: if dX_{t+2} = 0 and p_begin + p_end >= dK_t, the formula gives
    floor((p_begin + p_end - dK_t) / s_t) + 1 >= 1, not 0.  We follow the ONNX
    behaviour: if a spatial dimension of X is 0, the corresponding spatial dimension
    of Y is 0.  This is the only reading that makes the operator applicable when the
    input has a zero-sized spatial dimension.

    DECISION: The document says "The padded elements are not elements of X: they are
    excluded from the maximum".  We implement this by padding X with a sentinel value
    that is less than or equal to every element of X: ``-inf`` for floating-point
    types and the minimum representable value for integer types.  Taking the maximum
    over the padded window then yields the same result as excluding the padded
    elements, provided the window covers at least one element of X (which is
    guaranteed by the precondition on the shape of Y).

    DECISION: The document does not specify the behaviour for types other than real,
    float16/float/double, int8, and uint8.  We use ``-inf`` as the sentinel for any
    other type.
    """
    X = np.asarray(X)
    m = X.ndim - 2
    if m < 1:
        raise ValueError("Input tensor must have rank at least 3.")

    if strides is None:
        strides = [1] * m
    if pads is None:
        pads = [0] * (2 * m)
    if dilations is None:
        dilations = [1] * m

    # Compute output spatial shape
    dY = [X.shape[0], X.shape[1]]
    for t in range(m):
        dX_t = X.shape[t + 2]
        if dX_t == 0:
            # DECISION: see the note in the docstring about zero-sized spatial
            # dimensions.  We set the output spatial dimension to 0.
            dY.append(0)
        else:
            p_begin = pads[t]
            p_end = pads[m + t]
            dK = kernel_shape[t]
            s = strides[t]
            dY_t = (dX_t + p_begin + p_end - dK) // s + 1
            dY.append(dY_t)

    # If any output dimension is 0, the result is empty.
    if any(d == 0 for d in dY):
        return np.empty(dY, dtype=X.dtype)

    # Determine sentinel for padding.
    if np.issubdtype(X.dtype, np.floating):
        sentinel = -np.inf
    elif np.issubdtype(X.dtype, np.integer):
        sentinel = np.iinfo(X.dtype).min
    else:
        # DECISION: fallback for types not covered by the specification.
        sentinel = -np.inf

    # Pad X with the sentinel value.
    pad_width = [(0, 0), (0, 0)] + [(pads[t], pads[m + t]) for t in range(m)]
    X_padded = np.pad(X, pad_width, mode="constant", constant_values=sentinel)

    # Initialize output.
    Y = np.empty(dY, dtype=X.dtype)

    # Iterate over all spatial output indices.
    for j in np.ndindex(*dY[2:]):
        # Build the slice for the window in the padded tensor.
        slices = [slice(None), slice(None)]
        for t in range(m):
            start = j[t] * strides[t]
            stop = start + kernel_shape[t]
            slices.append(slice(start, stop))
        window = X_padded[tuple(slices)]
        # Maximum over the spatial axes of the window.
        Y[(slice(None), slice(None)) + j] = np.max(
            window, axis=tuple(range(2, 2 + m))
        )

    return Y
