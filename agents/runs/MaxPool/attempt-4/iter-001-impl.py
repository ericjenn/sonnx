"""MaxPool operator.

Specification: "MaxPool version 14", revision 2026-10-03, based on ONNX opset
14.  The specification gives four sections, one per type family: ``real``,
``float16``/``float``/``double``, ``int8``/``int64`` and ``uint8``.  They share
one semantics, so this module implements all of them with a single function.

Quoting the ``float`` section:

    "The three types share one semantics: the operator selects one of the
    elements of the input tensor and performs no arithmetic on it, so that the
    result is the same for all of them, the special numbers included."

Quoting the ``int`` section:

    "The two types share one semantics: the operator selects one of the
    elements of the input tensor and performs no arithmetic on it, so that the
    result is the same for both, the comparison being the one of the signed
    integers they represent."

Quoting the ``uint`` section:

    "The types of this section share one semantics: the operator selects one of
    the elements of the input tensor and performs no arithmetic on it, so that
    the result is the same for every type of the family, the comparison being
    the one of the unsigned integers they represent."

The entry point is :func:`maxpool`.
"""

import numpy as np


def maxpool(X, kernel_shape, strides=None, pads=None, auto_pad='NOTSET',
            ceil_mode=0, dilations=None, storage_order=0):
    r"""Apply max pooling to the tensor ``X``.

    Quoting the specification, section "MaxPool (real)", function
    [E_MAXPOOL_REAL_FUNC_0010] (the ``float``, ``int`` and ``uint`` sections
    repeat the same text):

        "Operator **MaxPool** applies max pooling to the input tensor $X$: it
        slides a window of the shape given by `kernel_shape` over the spatial
        axes of $X$, with the steps given by `strides`, and stores in $Y$ the
        maximum of the elements of $X$ covered by each window."

        "The first two axes of $X$ are the batch axis and the channel axis;
        they are not pooled. The remaining $rX - 2$ axes are the spatial axes,
        numbered from $0$ to $rX-3$ in the order of the axes of $X$. The number
        of spatial axes is denoted $m = rX - 2$; the operator is applicable to
        any $m \ge 1$."

        "For any tensor index $i = (n, c, j_0, \dots, j_{m-1})$ of the result
        $Y$:

        $$Y[n, c, j_0, \dots, j_{m-1}] = \max_{k_0, \dots, k_{m-1}}
        X_p[n, c, j_0 \cdot s_0 + k_0, \dots, j_{m-1} \cdot s_{m-1} +
        k_{m-1}]$$"

        "where ... $k_t \in [0, dK_t-1]$ is the index along spatial axis $t$ of
        the window, $dK_t$ being the value of `kernel_shape` along that axis,
        $s_t$ is the value of `strides` along spatial axis $t$, $X_p$ is the
        tensor obtained by padding $X$ with the values of `pads` ... the
        maximum is taken over the elements of $X$ covered by the window, the
        padded elements being excluded."

        "**Padding.** The padding adds $p_{t,\text{begin}}$ elements before and
        $p_{t,\text{end}}$ elements after the input along spatial axis $t$,
        where $p_{t,\text{begin}}$ and $p_{t,\text{end}}$ are the two values of
        `pads` for that axis. The padded elements are not elements of $X$: they
        are excluded from the maximum, so that the maximum is taken over the
        elements of $X$ covered by the window only. A window that covers no
        element of $X$ is not a window of the result: the shape of $Y$ is such
        that every window covers at least one element of $X$."

        "**Shape of the result.** The spatial dimensions of $Y$ are

        $$dY_{t+2} = \left\lfloor \frac{dX_{t+2} + p_{t,\text{begin}} +
        p_{t,\text{end}} - dK_t}{s_t} \right\rfloor + 1$$

        for every $t \in [0, m-1]$, and $dY_0 = dX_0$, $dY_1 = dX_1$."

    The maximum of the floating-point values is the one of the values of the
    type, not of the real numbers they represent; quoting the ``float``
    section:

        "NaN is not comparable with any value, itself included. When a window
        covers an element that is NaN, the result of that window is NaN. ...
        $+\text{inf}$ is greater than every finite value and than $-\text{inf}$,
        and $-\text{inf}$ is smaller than every finite value. A window whose
        elements are all $-\text{inf}$ gives $-\text{inf}$. $+0$ and $-0$ are
        equal, and neither is greater than the other. A window whose elements
        are all null gives a null result; its sign is not specified ..."

    The maximum of the integer values is the one of the signed (``int``) or
    unsigned (``uint``) integers the values represent; quoting the ``int``
    section:

        "for `int8`, the values range from $-128$ to $127$ and the maximum of
        $-128$ and $127$ is $127$; for `int64`, the values range from $-2^{63}$
        to $2^{63}-1$. The comparison is the same for the two types, and the
        result is one of the elements of $X$, so that no overflow can occur."

    Parameters
    ----------
    X : array_like
        "The tensor to which the max pooling is applied. Its first two axes are
        the batch axis and the channel axis, and its remaining axes are the
        spatial axes."  Its rank is at least 3
        ([E_MAXPOOL_REAL_CONSTR_X_0010]).
    kernel_shape : sequence of int
        "`kernel_shape` gives the size of the window along each spatial axis.
        It has $m$ values, one per spatial axis, and every value is at least
        $1$."
    strides : sequence of int, optional
        "`strides` gives the step of the window along each spatial axis. It has
        $m$ values, one per spatial axis, and every value is at least $1$."
    pads : sequence of int, optional
        "`pads` gives the number of elements added before and after $X$ along
        each spatial axis. It has $2m$ values, the first $m$ being the numbers
        added at the beginning of the axes and the last $m$ the numbers added
        at the end, in the order of the spatial axes. Every value is at least
        $0$."
    auto_pad : str, optional
        "`auto_pad` selects the way the padding is computed. It is restricted
        to `NOTSET` by [R1], so that the padding is the one given by `pads`."
    ceil_mode : int, optional
        "`ceil_mode` selects the rounding of the output shape. It is restricted
        to $0$ by [R3], so that the floor is used."
    dilations : sequence of int, optional
        "`dilations` gives the dilation along each spatial axis. It is
        restricted to $1$ along each spatial axis by [R4], so that the window
        covers consecutive elements of $X$."
    storage_order : int, optional
        "`storage_order` selects the order in which the flattened index of the
        output $\text{Indices}$ is computed. It is restricted to $0$ by [R2],
        and the output $\text{Indices}$ is not supported by this profile, as
        stated by [R5]."

    Returns
    -------
    numpy.ndarray
        The tensor $Y$, of the same type as ``X``
        ([E_MAXPOOL_FLOAT_CONSTR_X_0030],
        [E_MAXPOOL_INT_CONSTR_X_0030],
        [E_MAXPOOL_UINT_CONSTR_X_0030]).

    Notes
    -----
    DECISION: the specification gives no default for any attribute.  The
    signature above supplies ``strides=None`` (meaning one along every spatial
    axis), ``pads=None`` (meaning zero along every spatial axis),
    ``dilations=None`` (meaning one along every spatial axis), ``auto_pad=
    'NOTSET'``, ``ceil_mode=0`` and ``storage_order=0``.  The document says
    nothing about these defaults; the values chosen are the ones the
    restrictions [R1], [R2], [R3] and [R4] force the attributes to take, and
    the neutral values for `strides`, `pads` and `dilations`.  `kernel_shape`
    is given no default at all, since the document gives it none and no value
    of it is neutral.

    DECISION: the document gives the operator the signature
    "$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$" with an optional second
    output, but restriction [R5] states "The output $\text{Indices}$ is not
    supported".  The function therefore returns $Y$ alone, not a tuple.

    DECISION: the document says of `auto_pad`, `ceil_mode`, `dilations` and
    `storage_order` only that they are restricted to `NOTSET`, $0$, $1$ and $0$
    respectively; it says nothing about what the operator does for any other
    value.  The parameters are accepted and their values are ignored: the
    computation always uses the padding given by `pads`, the floor of the
    formula, consecutive window elements and no index output.

    DECISION: the document says "The padded elements are not elements of $X$:
    they are excluded from the maximum".  This is implemented by padding with
    the smallest value of the type of $X$ ($-\text{inf}$ for the floating-point
    types, the minimum of the type for the integer types), which is equivalent
    because the document also guarantees that "every window covers at least one
    element of $X$" ([E_MAXPOOL_REAL_CONSTR_X_0020] and its counterparts): the
    maximum over the elements of $X$ covered by a window is unchanged by adding
    a value that is not greater than any of them.

    DECISION: the document is self-contradictory about a spatial dimension of
    $X$ of size $0$.  It says "The formula above then gives a spatial dimension
    of $Y$ of size $0$ when $p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$,
    and the corresponding dimension of $Y$ is empty", but the formula
    $dY_{t+2} = \lfloor (0 + p_{t,\text{begin}} + p_{t,\text{end}} - dK_t)/s_t
    \rfloor + 1$ gives at least $1$ under that condition, never $0$.  The
    reading implemented here is the one of the prose: a spatial dimension of
    $X$ of size $0$ gives a spatial dimension of $Y$ of size $0$, so that $Y$
    is empty and no window is evaluated.  This is the only reading under which
    the operator is well defined, since the formula's reading would require the
    maximum over a window that "covers no element of $X$", which the document
    says "is not a window of the result".  The same rule is applied when
    $dX_{t+2} = 0$ and $p_{t,\text{begin}} + p_{t,\text{end}} < dK_t$, a case
    the document calls "not applicable"; no error is raised, since the document
    states "No error condition."

    DECISION: the document states the precondition
    [E_MAXPOOL_REAL_CONSTR_X_0020] ("For every spatial axis $t$, either
    $dX_{t+2} \ge 1$ and $dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} \ge
    dK_t$, or $dX_{t+2} = 0$ and $p_{t,\text{begin}} + p_{t,\text{end}} \ge
    dK_t$") but requires no validation of it, and its table of error conditions
    says "No error condition."  No validation is performed; an input that
    violates the precondition is not detected here.
    """

    X = np.asarray(X)

    # "The number of spatial axes is denoted $m = rX - 2$".
    m = X.ndim - 2

    # "`kernel_shape` gives the size of the window along each spatial axis. It
    # has $m$ values, one per spatial axis".
    ks = [int(k) for k in kernel_shape]

    # "`strides` gives the step of the window along each spatial axis. It has
    # $m$ values, one per spatial axis".
    if strides is None:
        strides = [1] * m
    ss = [int(s) for s in strides]

    # "`pads` gives the number of elements added before and after $X$ along
    # each spatial axis. It has $2m$ values, the first $m$ being the numbers
    # added at the beginning of the axes and the last $m$ the numbers added at
    # the end, in the order of the spatial axes."
    if pads is None:
        pads = [0] * (2 * m)
    ps = [int(p) for p in pads]

    # "`dilations` gives the dilation along each spatial axis. It is restricted
    # to $1$ along each spatial axis by [R4], so that the window covers
    # consecutive elements of $X$."  The value is therefore not used below.
    if dilations is None:
        dilations = [1] * m
    _dilations = [int(d) for d in dilations]

    # "**Shape of the result.** The spatial dimensions of $Y$ are
    # $dY_{t+2} = \lfloor (dX_{t+2} + p_{t,begin} + p_{t,end} - dK_t)/s_t
    # \rfloor + 1$ for every $t \in [0, m-1]$, and $dY_0 = dX_0$, $dY_1 =
    # dX_1$."  Python's `//` on integers is the floor of the formula.
    #
    # DECISION: a spatial dimension of $X$ of size $0$ gives a spatial
    # dimension of $Y$ of size $0$; see the note in the docstring.
    out_spatial = []
    for t in range(m):
        dX = X.shape[2 + t]
        if dX == 0:
            out_spatial.append(0)
        else:
            out_spatial.append((dX + ps[t] + ps[m + t] - ks[t]) // ss[t] + 1)

    out_shape = (X.shape[0], X.shape[1]) + tuple(out_spatial)

    # "A batch or channel dimension of size $0$ gives a dimension of $Y$ of
    # size $0$, and $Y$ is empty."  The same holds for an empty spatial
    # dimension of $Y$: no window is evaluated.
    if 0 in out_shape:
        return np.empty(out_shape, dtype=X.dtype)

    # "The padded elements are not elements of $X$: they are excluded from the
    # maximum".  Padding with the smallest value of the type of $X$ leaves the
    # maximum over the elements of $X$ covered by a window unchanged, since
    # every window covers at least one element of $X$.
    if np.issubdtype(X.dtype, np.floating):
        fill = -np.inf
    else:
        fill = np.iinfo(X.dtype).min
    pad_width = [(0, 0), (0, 0)] + [(ps[t], ps[m + t]) for t in range(m)]
    Xp = np.pad(X, pad_width, mode='constant', constant_values=fill)

    # The windows of the result: a window of shape `kernel_shape` is slid over
    # the spatial axes of the padded tensor, and the windows are taken every
    # `strides` elements along each spatial axis.
    windows = np.lib.stride_tricks.sliding_window_view(
        Xp, tuple(ks), axis=tuple(range(2, 2 + m)))
    index = ((slice(None), slice(None))
             + tuple(slice(None, None, s) for s in ss)
             + tuple(slice(None) for _ in range(m)))
    windows = windows[index]

    # "stores in $Y$ the maximum of the elements of $X$ covered by each
    # window".  `numpy.max` propagates NaN, as the `float` section requires
    # ("When a window covers an element that is NaN, the result of that window
    # is NaN"), and compares $-\text{inf}$ and the finite values as the section
    # requires.
    Y = windows.max(axis=tuple(range(2 + m, 2 + 2 * m)))

    return Y
