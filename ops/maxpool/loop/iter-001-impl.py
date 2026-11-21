"""MaxPool operator, ONNX opset 14 profile (specification revision 2026-10-03).

This module implements the four profiles of the specification -- ``real``,
``float`` (``float16``/``float``/``double``), ``int`` (``int8``) and ``uint``
(``uint8``) -- as a single function, because the specification states that the
four share one semantics: "the operator selects one of the elements of the
input tensor and performs no arithmetic on it, so that the result is the same
for all of them, the special numbers included."

The entry point is :func:`maxpool`.
"""

import itertools

import numpy as np


def _padding_value(dtype):
    """Value used to fill the padded elements of ``X``.

    The specification says, in the "Padding" paragraph of the Function section
    of every profile:

        "The padded elements are not elements of X: they are excluded from the
        maximum, so that the maximum is taken over the elements of X covered
        by the window only."

    DECISION: the document does not say how that exclusion is to be realised.
    This implementation realises it by filling the padded elements with the
    smallest value of the dtype of ``X`` (``-inf`` for the floating-point
    types, the minimum of the type for the integer types), so that a padded
    element can never be the maximum of a window that also covers an element
    of ``X``.  The precondition ``E_MAXPOOL_*_CONSTR_X_0020`` guarantees that
    every window of the result covers at least one element of ``X``, so the
    two readings agree on every call the document admits.
    """
    if np.issubdtype(dtype, np.floating):
        return -np.inf
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    return 0


def maxpool(X, kernel_shape, strides=None, pads=None, auto_pad="NOTSET",
            ceil_mode=0, dilations=None, storage_order=0):
    """Apply max pooling to ``X``.

    Function, [E_MAXPOOL_REAL_FUNC_0010] / [E_MAXPOOL_FLOAT_FUNC_0010] /
    [E_MAXPOOL_INT_FUNC_0010] / [E_MAXPOOL_UINT_FUNC_0010]:

        "Operator **MaxPool** applies max pooling to the input tensor X: it
        slides a window of the shape given by `kernel_shape` over the spatial
        axes of X, with the steps given by `strides`, and stores in Y the
        maximum of the elements of X covered by each window."

        "The first two axes of X are the batch axis and the channel axis; they
        are not pooled.  The remaining rX - 2 axes are the spatial axes,
        numbered from 0 to rX-3 in the order of the axes of X.  The number of
        spatial axes is denoted m = rX - 2; the operator is applicable to any
        m >= 1."

        "For any tensor index i = (n, c, j_0, ..., j_{m-1}) of the result Y:

            Y[n, c, j_0, ..., j_{m-1}] =
                max over k_0, ..., k_{m-1} such that
                j_t * s_t + k_t - p_{t,begin} in [0, dX_{t+2}-1] for every t
                of X_p[n, c, j_0*s_0 + k_0, ..., j_{m-1}*s_{m-1} + k_{m-1}]

        where ... k_t in [0, dK_t-1] is the index along spatial axis t of the
        window, dK_t being the value of `kernel_shape` along that axis, s_t is
        the value of `strides` along spatial axis t, X_p is the tensor obtained
        by padding X with the values of `pads` ... and the maximum is taken
        over the positions of the window that are elements of X, i.e. those
        for which j_t * s_t + k_t - p_{t,begin} in [0, dX_{t+2}-1] for every t,
        the padded elements being excluded."

    Shape of the result, [E_MAXPOOL_*_CONSTR_Y_0010]:

        "dY_0 = dX_0, dY_1 = dX_1, and for every spatial axis t,
        dY_{t+2} = floor((dX_{t+2} + p_{t,begin} + p_{t,end} - dK_t)/s_t) + 1
        when dX_{t+2} >= 1, and dY_{t+2} = 0 when dX_{t+2} = 0."

    The floor is the one of the ``ceil_mode`` attribute, which restriction
    [R3] fixes to 0, and ``dilations`` is fixed to 1 by the constraint
    [E_MAXPOOL_REAL_CONSTR_dilations_0010], so the window covers consecutive
    elements of ``X``.

    Parameters
    ----------
    X : array_like
        The tensor to which the max pooling is applied.  Its first two axes
        are the batch axis and the channel axis, and its remaining axes are
        the spatial axes.
    kernel_shape : list of int
        "`kernel_shape` gives the size of the window along each spatial axis.
        It has m values, one per spatial axis, and every value is at least 1."
    strides : list of int, optional
        "`strides` gives the step of the window along each spatial axis.  It
        has m values, one per spatial axis, and every value is at least 1."
    pads : list of int, optional
        "`pads` gives the number of elements added before and after X along
        each spatial axis.  It has 2m values, the first m being the numbers
        added at the beginning of the axes and the last m the numbers added at
        the end, in the order of the spatial axes.  Every value is at least 0."
    auto_pad : str, optional
        "`auto_pad` selects the way the padding is computed.  It is restricted
        to `NOTSET` by [R1], so that the padding is the one given by `pads`."
    ceil_mode : int, optional
        "`ceil_mode` selects the rounding of the output shape.  It is
        restricted to 0 by [R3], so that the floor is used."
    dilations : list of int, optional
        "`dilations` gives the dilation along each spatial axis.  Its value is
        1 along each spatial axis, so that the window covers consecutive
        elements of X."
    storage_order : int, optional
        "`storage_order` selects the order in which the flattened index of the
        output `Indices` is computed.  It is restricted to 0 by [R2], and the
        output `Indices` is not supported by this profile, as stated by [R5]."

    Returns
    -------
    numpy.ndarray
        ``Y``, the result of the max pooling of ``X``, of the same dtype as
        ``X``.

    DECISION: the document gives the *names* of the attributes but no
    *defaults* for `kernel_shape`, `strides`, `pads`, `auto_pad`, `ceil_mode`,
    `dilations` and `storage_order`; the Attributes section of the `real`
    profile states only the constraints on their values.  The defaults chosen
    here are: `kernel_shape` has no default and is a required parameter (the
    document says "It has m values, one per spatial axis", so a call without
    it is not a call the document describes); `strides` defaults to ``[1]*m``;
    `pads` defaults to ``[0]*(2*m)``; `auto_pad` defaults to ``"NOTSET"``;
    `ceil_mode` defaults to ``0``; `dilations` defaults to ``[1]*m``;
    `storage_order` defaults to ``0``.  These are the values the constraints
    of the document force the attributes to take, but the document does not
    say that an omitted attribute takes the value its constraint forces.

    DECISION: the document says nothing about the behaviour of the operator
    when a constraint is violated (rank of ``X`` below 3, wrong length of
    `kernel_shape`/`strides`/`pads`, a value out of range, a window that covers
    no element of ``X``, a value of `auto_pad` other than ``NOTSET``, a value
    of `ceil_mode` other than 0, a value of `dilations` other than 1, a value
    of `storage_order` other than 0).  No validation is performed and no error
    is raised: the constraints are preconditions of the operator, not
    conditions the implementation is asked to check.

    DECISION: `auto_pad`, `ceil_mode`, `dilations` and `storage_order` are
    accepted as parameters but their values are not used, because the
    document restricts them to a single value (``NOTSET``, 0, 1 and 0
    respectively) and says that the padding is the one given by `pads`, that
    the floor is used, that the window covers consecutive elements of ``X``,
    and that the output `Indices` is not supported.

    DECISION: the document gives the operator a single output ``Y`` in this
    profile ("The output `Indices` is not supported", [R5]), so a single array
    is returned and not a tuple.

    DECISION: the document says nothing about the sign or the payload of a
    quiet NaN result, nor about the sign of a null result: "The NaN of the
    result is a quiet NaN; its sign and its payload are not specified, so that
    every quiet NaN of the type of X and Y is a conforming result" and "its
    sign is not specified, so that every null of the type of X is a conforming
    result".  The values produced here are the ones numpy's ``np.maximum``
    produces, which are conforming.

    DECISION: the document says nothing about the accepted input container;
    ``X`` is converted with ``np.asarray`` and the dtype of ``X`` is preserved
    in ``Y``, as required by [E_MAXPOOL_*_CONSTR_X_0030] ("Tensors X and Y have
    the same type").
    """
    X = np.asarray(X)

    rX = X.ndim
    m = rX - 2

    kernel_shape = [int(k) for k in kernel_shape]
    if strides is None:
        strides = [1] * m
    else:
        strides = [int(s) for s in strides]
    if pads is None:
        pads = [0] * (2 * m)
    else:
        pads = [int(p) for p in pads]

    p_begin = pads[:m]
    p_end = pads[m:2 * m]

    dX = X.shape

    # [E_MAXPOOL_*_CONSTR_Y_0010]: dY_0 = dX_0, dY_1 = dX_1, and for every
    # spatial axis t,
    #   dY_{t+2} = floor((dX_{t+2} + p_begin + p_end - dK_t)/s_t) + 1
    #              when dX_{t+2} >= 1,
    #   dY_{t+2} = 0 when dX_{t+2} = 0.
    dY_spatial = []
    for t in range(m):
        d = dX[t + 2]
        if d >= 1:
            dY_spatial.append(
                (d + p_begin[t] + p_end[t] - kernel_shape[t]) // strides[t] + 1
            )
        else:
            dY_spatial.append(0)

    out_shape = (dX[0], dX[1]) + tuple(dY_spatial)

    # "A batch or channel dimension of size 0 gives a dimension of Y of size 0,
    # and Y is empty."  The same holds for a spatial dimension of size 0.
    if any(s == 0 for s in out_shape):
        return np.empty(out_shape, dtype=X.dtype)

    # "X_p is the tensor obtained by padding X with the values of `pads`."
    pad_width = [(0, 0), (0, 0)] + [
        (p_begin[t], p_end[t]) for t in range(m)
    ]
    X_p = np.pad(
        X, pad_width, mode="constant", constant_values=_padding_value(X.dtype)
    )

    # For every position k = (k_0, ..., k_{m-1}) of the window, the elements of
    # X_p that the window covers at the output index j = (j_0, ..., j_{m-1})
    # are X_p[..., j_t*s_t + k_t, ...].  Slicing each spatial axis with
    # start=k_t, stop=k_t + (dY_{t+2}-1)*s_t + 1, step=s_t therefore yields
    # exactly the dY_{t+2} values of that axis, and the maximum over all
    # positions of the window is the element-wise maximum of those slices.
    result = None
    for offset in itertools.product(*[range(k) for k in kernel_shape]):
        index = [slice(None), slice(None)]
        for t in range(m):
            k = offset[t]
            n_out = dY_spatial[t]
            index.append(
                slice(k, k + (n_out - 1) * strides[t] + 1, strides[t])
            )
        window = X_p[tuple(index)]
        if result is None:
            result = window
        else:
            result = np.maximum(result, window)

    return result
