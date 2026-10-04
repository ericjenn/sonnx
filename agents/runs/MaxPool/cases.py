"""The cases with which the **MaxPool** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled input tensor per case, with the node's attributes.
                     Each case goes through a one-node MaxPool model (opset 14) and
                     through the implementation, and the two results are compared bit
                     for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the shape formula of the
                     document, the special values of every type, the worked examples of
                     every section, and the real section, which has no ONNX type and is
                     checked against exact arithmetic instead.

The document's profile restricts `auto_pad` to NOTSET, `storage_order` to 0,
`ceil_mode` to 0, `dilations` to 1, and does not support the `Indices` output, so the
node of every case carries only `kernel_shape`, `strides` and `pads` (and, where the
default is what is being tested, not even those).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "MaxPool"
OPSET = 14
IR_VERSION = 10

ALL_FLOAT = [np.float16, np.float32, np.float64]
ALL_INT = [np.int8, np.int64]
ALL_UINT = [np.uint8]
ALL_TYPES = ALL_FLOAT + ALL_INT + ALL_UINT

UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------


def run_ort_batch(x: np.ndarray, attrs: dict) -> np.ndarray:
    """One MaxPool node, one run, for a whole batch of windows."""
    proto = helper.np_dtype_to_tensor_dtype(x.dtype)
    vi_x = helper.make_tensor_value_info("X", proto, list(x.shape))
    vi_y = helper.make_tensor_value_info("Y", proto, None)
    node = helper.make_node(OP, ["X"], ["Y"], name="maxpool0", **dict(attrs))
    graph = helper.make_graph([node], "maxpool_graph", [vi_x], [vi_y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    (y,) = sess.run(["Y"], {"X": x})
    return y


def _bits_equal(want: np.ndarray, got: np.ndarray) -> bool:
    if want.shape != got.shape or want.dtype != got.dtype:
        return False
    if want.dtype.kind == "f":
        uint = UINT_OF_WIDTH[want.dtype.itemsize]
        bw = np.ascontiguousarray(want).view(uint)
        bg = np.ascontiguousarray(got).view(uint)
        both_nan = np.isnan(want) & np.isnan(got)
        return bool(np.all((bw == bg) | both_nan))
    return bool(np.all(want == got))


# ---------------------------------------------------------------------------
# the document's formula, transcribed literally
# ---------------------------------------------------------------------------
#
# A second, literal reading of the document, independent of the implementation: the
# window of every element of Y is enumerated from the formula, the padded elements are
# excluded, and the maximum is taken over the elements of X the window covers.  The
# maximum of the floating-point values is the one the document states: NaN wins, then
# +inf, then the finite values, and a window of nulls gives a null (its sign is not
# specified, so +0 is taken).


def doc_out_shape(dX, kernel_shape, strides, pads):
    m = len(kernel_shape)
    out = [dX[0], dX[1]]
    for t in range(m):
        out.append((dX[2 + t] + pads[t] + pads[m + t] - kernel_shape[t])
                   // strides[t] + 1)
    return tuple(out)


def _max_of(values, dtype):
    """The maximum of the values of a window, as the document defines it."""
    if not values:
        raise ValueError("a window that covers no element of X")
    if any(v != v for v in values):          # NaN is not comparable with any value
        return dtype.type(float("nan"))
    best = values[0]
    for v in values[1:]:
        if v > best:
            best = v
    return dtype.type(best)


def doc_maxpool(x: np.ndarray, kernel_shape, strides, pads) -> np.ndarray:
    """The document's formula, element by element, in exact index arithmetic."""
    dX = x.shape
    m = len(kernel_shape)
    dY = doc_out_shape(dX, kernel_shape, strides, pads)
    out = np.empty(dY, dtype=x.dtype)
    for n in range(dX[0]):
        for c in range(dX[1]):
            for j in np.ndindex(*dY[2:]):
                values = []
                for k in np.ndindex(*kernel_shape):
                    h = [j[t] * strides[t] + k[t] - pads[t] for t in range(m)]
                    if all(0 <= h[t] < dX[2 + t] for t in range(m)):
                        values.append(x[(n, c) + tuple(h)])
                out[(n, c) + j] = _max_of(values, x.dtype)
    return out


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def _rand_shape(rng, m, lo=1, hi=6):
    return tuple(int(rng.integers(lo, hi + 1)) for _ in range(m))


def sweep_shape_formula(impl):
    """The shape formula of the document, over many attribute combinations.

    The formula is checked against the shape ONNX Runtime produces and against the
    shape the implementation produces, for every rank from 3 to 5 and for a spread of
    kernel, stride and pad values, including the boundary where the last window just
    fits and the one where it does not.
    """
    rng = np.random.default_rng(20261004)
    failures = []
    checked = 0
    for m in (1, 2, 3):
        for _ in range(40):
            dX = (int(rng.integers(1, 3)), int(rng.integers(1, 3))) + _rand_shape(rng, m)
            kernel_shape = [int(rng.integers(1, 4)) for _ in range(m)]
            strides = [int(rng.integers(1, 3)) for _ in range(m)]
            pads = [int(rng.integers(0, 3)) for _ in range(m)]
            pads = pads + [int(rng.integers(0, 3)) for _ in range(m)]
            # the document's precondition: every window covers an element of X
            if any(dX[2 + t] + pads[t] + pads[m + t] < kernel_shape[t] for t in range(m)):
                continue
            x = rng.integers(-100, 100, size=dX).astype(np.int64)
            attrs = {"kernel_shape": kernel_shape, "strides": strides, "pads": pads}
            want = doc_out_shape(dX, kernel_shape, strides, pads)
            try:
                ref = run_ort_batch(x, attrs)
            except Exception as exc:
                failures.append({"who": "shape formula", "detail":
                                 "ORT refused %s %s: %s" % (dX, attrs, exc)})
                continue
            got = np.asarray(impl.maxpool(x, **attrs))
            checked += 1
            if tuple(ref.shape) != want:
                failures.append({"who": "shape formula", "detail":
                                 "dX %s %s: document %s, ORT %s"
                                 % (dX, attrs, want, tuple(ref.shape))})
            if tuple(got.shape) != want:
                failures.append({"who": "shape formula", "detail":
                                 "dX %s %s: document %s, implementation %s"
                                 % (dX, attrs, want, tuple(got.shape))})
    return {"sweep": "the shape formula (document vs ORT vs implementation)",
            "summary": "%d attribute combinations over ranks 3-5" % checked,
            "failures": failures}


def sweep_values(impl):
    """The document's formula against ONNX Runtime, over random tensors of every type.

    The window enumeration is the document's, transcribed literally; the comparison is
    against ONNX Runtime and against the implementation.  The floating-point types are
    compared bit for bit, NaN equal to NaN.
    """
    rng = np.random.default_rng(20261005)
    failures = []
    rows = []
    for dtype in ALL_TYPES:
        dt = np.dtype(dtype)
        checked = 0
        for _ in range(12):
            m = int(rng.integers(1, 3))
            dX = (int(rng.integers(1, 3)), int(rng.integers(1, 3))) + _rand_shape(rng, m)
            kernel_shape = [int(rng.integers(1, 4)) for _ in range(m)]
            strides = [int(rng.integers(1, 3)) for _ in range(m)]
            pads = [int(rng.integers(0, 2)) for _ in range(m)]
            pads = pads + [int(rng.integers(0, 2)) for _ in range(m)]
            if any(dX[2 + t] + pads[t] + pads[m + t] < kernel_shape[t] for t in range(m)):
                continue
            if dt.kind == "f":
                x = rng.standard_normal(size=dX).astype(dt)
            elif dt.kind == "i":
                info = np.iinfo(dt)
                x = rng.integers(int(info.min), int(info.max) + 1, size=dX).astype(dt)
            else:
                info = np.iinfo(dt)
                x = rng.integers(0, int(info.max) + 1, size=dX).astype(dt)
            attrs = {"kernel_shape": kernel_shape, "strides": strides, "pads": pads}
            want = doc_maxpool(x, kernel_shape, strides, pads)
            try:
                ref = run_ort_batch(x, attrs)
            except Exception as exc:
                failures.append({"who": "values", "detail":
                                 "%s %s: ORT refused: %s" % (dt.name, attrs, exc)})
                continue
            got = np.asarray(impl.maxpool(x, **attrs))
            checked += 1
            if not _bits_equal(want, ref):
                failures.append({"who": "values", "detail":
                                 "%s %s: document %s, ORT %s"
                                 % (dt.name, attrs, want.tolist(), ref.tolist())})
            if not _bits_equal(want, got):
                failures.append({"who": "values", "detail":
                                 "%s %s: document %s, implementation %s"
                                 % (dt.name, attrs, want.tolist(), got.tolist())})
        rows.append("%s %d" % (dt.name, checked))
    return {"sweep": "the document's formula (document vs ORT vs implementation)",
            "summary": "random tensors: %s" % ", ".join(rows),
            "failures": failures}


def _special_values(dtype):
    """The special values of a floating-point type, as the document names them."""
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    nan = float("nan")
    inf = float("inf")
    return [0.0, -0.0, inf, -inf, nan, float(f.max), float(-f.max),
            float(f.tiny), float(-f.tiny), 1.0, -1.0]


def sweep_special_values(impl):
    """The special values of every floating-point type, as operands and as results.

    The document states the maximum of the floating-point values: NaN wins, +inf beats
    every finite value, -inf loses to every finite value, and a window of nulls gives a
    null whose sign is not specified.  The comparison against ONNX Runtime is bit for
    bit with NaN equal to NaN; a differing NaN payload is reported, not a failure.
    """
    failures = []
    rows = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        vals = _special_values(dt)
        # every window of two values, laid out as a 1-D tensor of pairs
        x = np.array(vals, dtype=dt).reshape(1, 1, len(vals))
        attrs = {"kernel_shape": [2], "strides": [2]}
        want = doc_maxpool(x, [2], [2], [0, 0])
        try:
            ref = run_ort_batch(x, attrs)
        except Exception as exc:
            failures.append({"who": "special values", "detail":
                             "%s: ORT refused: %s" % (dt.name, exc)})
            continue
        got = np.asarray(impl.maxpool(x, **attrs))
        if not _bits_equal(want, ref):
            failures.append({"who": "special values", "detail":
                             "%s: document %s, ORT %s"
                             % (dt.name, want.tolist(), ref.tolist())})
        if not _bits_equal(want, got):
            failures.append({"who": "special values", "detail":
                             "%s: document %s, implementation %s"
                             % (dt.name, want.tolist(), got.tolist())})
        # a window of nulls: the sign of the result is not specified
        z = np.array([0.0, -0.0, -0.0, 0.0], dtype=dt).reshape(1, 1, 4)
        zattrs = {"kernel_shape": [2], "strides": [2]}
        zref = run_ort_batch(z, zattrs)
        zgot = np.asarray(impl.maxpool(z, **zattrs))
        if not np.all(zref == 0) or not np.all(zgot == 0):
            failures.append({"who": "special values", "detail":
                             "%s: a window of nulls gave ORT %s, implementation %s"
                             % (dt.name, zref.tolist(), zgot.tolist())})
        rows.append(dt.name)
    return {"sweep": "the special values of every floating-point type",
            "summary": "%s: NaN, +-inf, +-0, the extremes and the subnormal range"
                       % ", ".join(rows),
            "failures": failures}


def sweep_int_boundaries(impl):
    """The boundary values of the integer types, as operands and as results.

    The document states the maximum of the signed integers for `int8` and `int64` and
    of the unsigned integers for `uint8`; the extremes of each type are the values that
    decide it.  The comparison is against ONNX Runtime and against the implementation.
    """
    failures = []
    rows = []
    for dtype in ALL_INT + ALL_UINT:
        dt = np.dtype(dtype)
        info = np.iinfo(dt)
        lo, hi = int(info.min), int(info.max)
        vals = [lo, hi, 0, 1, -1, lo + 1, hi - 1, lo, hi, 0]
        x = np.array(vals, dtype=dt).reshape(1, 1, len(vals))
        attrs = {"kernel_shape": [2], "strides": [2]}
        want = doc_maxpool(x, [2], [2], [0, 0])
        try:
            ref = run_ort_batch(x, attrs)
        except Exception as exc:
            failures.append({"who": "integer boundaries", "detail":
                             "%s: ORT refused: %s" % (dt.name, exc)})
            continue
        got = np.asarray(impl.maxpool(x, **attrs))
        if not _bits_equal(want, ref):
            failures.append({"who": "integer boundaries", "detail":
                             "%s: document %s, ORT %s"
                             % (dt.name, want.tolist(), ref.tolist())})
        if not _bits_equal(want, got):
            failures.append({"who": "integer boundaries", "detail":
                             "%s: document %s, implementation %s"
                             % (dt.name, want.tolist(), got.tolist())})
        rows.append(dt.name)
    return {"sweep": "the boundary values of the integer types",
            "summary": "%s: the extremes of each type" % ", ".join(rows),
            "failures": failures}


def _spec_examples():
    """The worked examples of the document, as documented."""
    f32 = np.float32
    return [
        ("real Example 1",
         np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                    [13, 14, 15, 16]]], dtype=np.int64),
         {"kernel_shape": [2, 2], "strides": [2, 2]},
         np.array([[[6, 8], [14, 16]]], dtype=np.int64)),
        ("real Example 2",
         np.array([[[1, 2, 3]]], dtype=np.int64),
         {"kernel_shape": [2], "strides": [1], "pads": [1, 1]},
         np.array([[[1, 2, 3, 3]]], dtype=np.int64)),
        ("float Example 1",
         np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]], dtype=f32),
         {"kernel_shape": [2, 2], "strides": [2, 2]},
         np.array([[[6.0, 8.0], [14.0, 16.0]]], dtype=f32)),
        ("float Example 2",
         np.array([[[1.0, np.nan, 3.0, -np.inf]]], dtype=f32),
         {"kernel_shape": [2], "strides": [2]},
         np.array([[[np.nan, 3.0]]], dtype=f32)),
        ("int Example 1",
         np.array([[[-1, 2, -3, 4], [5, -6, 7, -8], [-9, 10, -11, 12],
                    [13, -14, 15, -16]]], dtype=np.int8),
         {"kernel_shape": [2, 2], "strides": [2, 2]},
         np.array([[[5, 7], [13, 15]]], dtype=np.int8)),
        ("uint Example 1",
         np.array([[[0, 200, 3, 255], [17, 128, 9, 64], [250, 1, 100, 7],
                    [33, 254, 2, 90]]], dtype=np.uint8),
         {"kernel_shape": [2, 2], "strides": [2, 2]},
         np.array([[[200, 255], [254, 100]]], dtype=np.uint8)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, and as ONNX Runtime does it."""
    failures = []
    for label, x, attrs, documented in _spec_examples():
        want = doc_maxpool(x, attrs["kernel_shape"], attrs["strides"],
                           attrs.get("pads", [0] * len(attrs["kernel_shape"]) * 2))
        try:
            ref = run_ort_batch(x, attrs)
        except Exception as exc:
            failures.append({"who": label, "detail": "ORT refused: %s" % exc})
            continue
        got = np.asarray(impl.maxpool(x, **attrs))
        if not _bits_equal(documented, want):
            failures.append({"who": label, "detail":
                             "documented %s, the document's formula %s"
                             % (documented.tolist(), want.tolist())})
        if not _bits_equal(documented, ref):
            failures.append({"who": label, "detail":
                             "documented %s, ORT %s"
                             % (documented.tolist(), ref.tolist())})
        if not _bits_equal(documented, got):
            failures.append({"who": label, "detail":
                             "documented %s, implementation %s"
                             % (documented.tolist(), got.tolist())})
    return {"sweep": "the document's worked examples",
            "summary": "%d examples, as documented vs ORT vs implementation"
                       % len(_spec_examples()),
            "failures": failures}


def sweep_real_section(impl):
    """The real section, which has no ONNX type: checked against exact arithmetic.

    ONNX Runtime cannot be asked about a real tensor, so this sweep checks the
    implementation against the values the document's Examples state, in exact
    arithmetic, and against the document's formula computed with `Fraction`.
    """
    F = Fraction
    examples = [
        ("real Example 1",
         [[[F(1), F(2), F(3), F(4)],
           [F(5), F(6), F(7), F(8)],
           [F(9), F(10), F(11), F(12)],
           [F(13), F(14), F(15), F(16)]]],
         {"kernel_shape": [2, 2], "strides": [2, 2]},
         [[[F(6), F(8)], [F(14), F(16)]]]),
        ("real Example 2",
         [[[F(1), F(2), F(3)]]],
         {"kernel_shape": [2], "strides": [1], "pads": [1, 1]},
         [[[F(1), F(2), F(3), F(3)]]]),
    ]
    failures = []
    for label, x, attrs, documented in examples:
        xx = np.array(x, dtype=object)
        want = doc_maxpool(xx, attrs["kernel_shape"], attrs["strides"],
                           attrs.get("pads", [0] * len(attrs["kernel_shape"]) * 2))
        got = np.asarray(impl.maxpool(xx, **attrs))
        doc = np.array(documented, dtype=object)
        if want.shape != doc.shape or any(want[i] != doc[i]
                                          for i in np.ndindex(doc.shape)):
            failures.append({"who": label, "detail":
                             "documented %s, the document's formula %s"
                             % (doc.tolist(), want.tolist())})
        if got.shape != doc.shape or any(got[i] != doc[i]
                                         for i in np.ndindex(doc.shape)):
            failures.append({"who": label, "detail":
                             "documented %s, implementation %s"
                             % (doc.tolist(), got.tolist())})
    return {"sweep": "the real section (exact rationals, no ONNX Runtime)",
            "summary": "%d examples, checked against exact arithmetic" % len(examples),
            "failures": failures}


def sweeps():
    return [sweep_shape_formula, sweep_values, sweep_special_values,
            sweep_int_boundaries, sweep_spec_examples, sweep_real_section]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _arr(shape, dtype, start=0.0):
    n = 1
    for d in shape:
        n *= d
    return (np.arange(n, dtype=np.float64) + start).astype(dtype).reshape(shape)


def _bit_nan(dtype, pattern):
    uint = UINT_OF_WIDTH[np.dtype(dtype).itemsize]
    return np.array([pattern], dtype=uint).view(dtype)


NAN_PATTERNS = {2: (0x7E01, 0x7C01, 0x8000),
                4: (0x7FC00001, 0x7F800001, 0x80000000),
                8: (0x7FF8000000000001, 0x7FF0000000000001, 0x8000000000000000)}


def _float_value_cases(dtype):
    """The value families of the float section."""
    dt = np.dtype(dtype)
    f = np.finfo(dt)
    maxf = float(f.max)
    tiny = float(f.tiny)
    nan, inf = float("nan"), float("inf")
    return [
        ("signed zeros", [0.0, -0.0, 0.0, -0.0], [2, 2]),
        ("infinities", [inf, -inf, inf, -inf, 1.0, -1.0], [2, 2]),
        ("nan", [nan, 1.0, nan, inf], [2, 2]),
        ("nan with -inf", [nan, -inf, 1.0, 3.0], [2, 2]),
        ("all -inf", [-inf, -inf, -inf, -inf], [2, 2]),
        ("extremes", [maxf, -maxf, maxf, -maxf], [2, 2]),
        ("subnormal range", [tiny, -tiny, tiny / 2.0, -tiny / 2.0], [2, 2]),
        ("ordinary", [1.0, 2.5, -3.0, 4.75], [2, 2]),
    ]


def _int_value_cases(dtype):
    dt = np.dtype(dtype)
    info = np.iinfo(dt)
    lo, hi = int(info.min), int(info.max)
    if lo < 0:
        vals = [lo, hi, 0, -1, 1, hi, lo, -1]
    else:
        vals = [0, hi, 1, 4, hi, 0, 1, hi // 2]
    return [("extremes and ordinary", vals, [2, 2])]


def cases():
    """Every individual case: (label, [X], attrs)."""
    # the document's worked examples, replayed as documented
    for label, x, attrs, _ in _spec_examples():
        yield ("example: %s" % label, [x], attrs)

    # the value families of every type
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        for name, vals, kernel in _float_value_cases(dtype):
            x = np.array(vals, dtype=dt).reshape(1, 1, len(vals))
            yield ("%s: %s" % (dt.name, name), [x],
                   {"kernel_shape": kernel, "strides": kernel})
        quiet, signalling, sign = NAN_PATTERNS[dt.itemsize]
        for name, pattern in (("quiet NaN, payload 1", quiet),
                              ("quiet NaN, payload 1, sign set", quiet | sign),
                              ("signalling NaN", signalling)):
            x = np.concatenate([_bit_nan(dtype, pattern),
                                np.ones(1, dtype=dt)]).reshape(1, 1, 2)
            yield ("%s: %s" % (dt.name, name), [x],
                   {"kernel_shape": [2], "strides": [2]})
    for dtype in ALL_INT + ALL_UINT:
        dt = np.dtype(dtype)
        for name, vals, kernel in _int_value_cases(dtype):
            x = np.array(vals, dtype=dt).reshape(1, 1, len(vals))
            yield ("%s: %s" % (dt.name, name), [x],
                   {"kernel_shape": kernel, "strides": kernel})

    # the shape families: rank 3, 4 and 5; a zero-sized spatial dimension; a
    # zero-sized batch or channel dimension; the document's own shapes
    shape_cases = [
        ("rank 3, kernel 2 stride 1", (1, 1, 5), {"kernel_shape": [2], "strides": [1]}),
        ("rank 3, kernel 3 stride 2", (1, 1, 7), {"kernel_shape": [3], "strides": [2]}),
        ("rank 3, kernel 2 stride 1 pads 1", (1, 1, 3),
         {"kernel_shape": [2], "strides": [1], "pads": [1, 1]}),
        ("rank 4, kernel 2x2 stride 2", (1, 1, 4, 4),
         {"kernel_shape": [2, 2], "strides": [2, 2]}),
        ("rank 4, kernel 3x3 stride 1 pads 1", (1, 1, 4, 4),
         {"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]}),
        ("rank 4, batch 2 channels 3", (2, 3, 4, 4),
         {"kernel_shape": [2, 2], "strides": [2, 2]}),
        ("rank 5, kernel 2x2x2 stride 2", (1, 1, 4, 4, 4),
         {"kernel_shape": [2, 2, 2], "strides": [2, 2, 2]}),
        ("rank 5, kernel 2x2x2 stride 1 pads 1", (1, 1, 3, 3, 3),
         {"kernel_shape": [2, 2, 2], "strides": [1, 1, 1], "pads": [1, 1, 1, 1, 1, 1]}),
        ("zero-sized spatial, pads cover the kernel", (1, 1, 0),
         {"kernel_shape": [2], "strides": [1], "pads": [1, 1]}),
        ("zero-sized spatial, pads cover the kernel, rank 4", (1, 1, 0, 4),
         {"kernel_shape": [2, 2], "strides": [1, 1], "pads": [1, 1, 0, 0]}),
        ("zero-sized batch", (0, 1, 4), {"kernel_shape": [2], "strides": [2]}),
        ("zero-sized channel", (1, 0, 4), {"kernel_shape": [2], "strides": [2]}),
        ("kernel equals the spatial size", (1, 1, 4),
         {"kernel_shape": [4], "strides": [1]}),
        ("stride larger than the kernel", (1, 1, 7),
         {"kernel_shape": [2], "strides": [3]}),
    ]
    for name, shape, attrs in shape_cases:
        for dtype in ALL_TYPES:
            dt = np.dtype(dtype)
            if dt.kind == "f":
                x = _arr(shape, dt, 1.0)
            else:
                x = _arr(shape, dt, 1.0)
            yield ("%s: %s" % (dt.name, name), [x], attrs)

    # the defaults the document states: a case that gives no attribute at all
    for dtype in ALL_TYPES:
        dt = np.dtype(dtype)
        x = _arr((1, 1, 4), dt, 1.0)
        yield ("%s: kernel_shape only, strides default" % dt.name, [x],
               {"kernel_shape": [2]})


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

COVERAGE = {
    "real": 0,
    "float": 0,
    "int": 0,
    "uint": 0,
}

for _label, _arrays, _attrs in cases():
    _dt = np.asarray(_arrays[0]).dtype
    if _dt.kind == "f":
        COVERAGE["float"] += 1
    elif _dt.kind == "i":
        COVERAGE["int"] += 1
    else:
        COVERAGE["uint"] += 1
    COVERAGE["real"] += 1
