"""The cases with which the **MaxPool** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled input tensor per case, with the node's attributes.
                     Each case goes through a one-node MaxPool model (opset 14) and
                     through the implementation, and the two results are compared bit
                     for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the exhaustive domain of
                     the 8-bit types, the worked examples of every section, the shape
                     formula, and the special values of the floating-point types.

The document's profile restricts ``auto_pad`` to ``NOTSET``, ``storage_order`` to 0,
``ceil_mode`` to 0, ``dilations`` to 1 and does not support the ``Indices`` output, so
the node of every case carries only ``kernel_shape``, ``strides`` and ``pads``.
"""

from __future__ import annotations

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
    """One MaxPool node, one session, one run: the whole batch at once."""
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
# the document's own formula, transcribed literally
# ---------------------------------------------------------------------------
#
# The transcription is deliberately independent of the implementation: it is a second,
# literal, reading of the Function section, compared with ONNX Runtime and with the
# implementation.  A disagreement here is a defect of the document, not of the
# implementation.


def doc_out_shape(x_shape, kernel_shape, strides, pads):
    """dY_0 = dX_0, dY_1 = dX_1, dY_{t+2} = floor((dX_{t+2} + pb + pe - dK)/s) + 1."""
    m = len(x_shape) - 2
    out = [x_shape[0], x_shape[1]]
    for t in range(m):
        num = x_shape[t + 2] + pads[t] + pads[m + t] - kernel_shape[t]
        out.append(num // strides[t] + 1)
    return tuple(out)


def doc_maxpool(x: np.ndarray, kernel_shape, strides, pads) -> np.ndarray:
    """The Function section, element by element, in exact arithmetic.

    The padded elements are excluded from the maximum, so a window is the set of
    elements of X it covers; the document guarantees every window covers at least one.
    """
    m = len(x.shape) - 2
    out_shape = doc_out_shape(x.shape, kernel_shape, strides, pads)
    y = np.empty(out_shape, dtype=x.dtype)
    for n in range(out_shape[0]):
        for c in range(out_shape[1]):
            for j in np.ndindex(*out_shape[2:]):
                best = None
                for k in np.ndindex(*kernel_shape):
                    h = tuple(j[t] * strides[t] + k[t] - pads[t] for t in range(m))
                    if any(h[t] < 0 or h[t] >= x.shape[t + 2] for t in range(m)):
                        continue
                    v = x[(n, c) + h]
                    if best is None:
                        best = v
                    elif x.dtype.kind == "f":
                        # NaN is not comparable with any value, itself included
                        if np.isnan(v):
                            best = v
                        elif not np.isnan(best) and v > best:
                            best = v
                    else:
                        if v > best:
                            best = v
                y[(n, c) + j] = best
    return y


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def _rand_int(rng, info, dtype):
    bits = np.dtype(dtype).itemsize * 8
    if info.min < 0:
        return int(rng.integers(info.min, info.max, dtype=np.int64, endpoint=True))
    if bits <= 32:
        return int(rng.integers(0, 2 ** bits, dtype=np.uint64))
    hi = int(rng.integers(0, 2 ** 32, dtype=np.uint64))
    lo = int(rng.integers(0, 2 ** 32, dtype=np.uint64))
    return (hi << 32) | lo


def sweep_int_domain(impl):
    """The document's formula against ONNX Runtime, over the integer domains.

    The 8-bit types are covered exhaustively over their whole value range: every value
    of the type appears in the input, and the window is the whole spatial extent, so
    the maximum of the type is exercised.  The wider types use their boundary values
    and a random sample.
    """
    rng = np.random.default_rng(20261003)
    rows, failures = [], []
    for dtype in ALL_INT + ALL_UINT:
        dt = np.dtype(dtype)
        info = np.iinfo(dt)
        exhaustive = dt.itemsize == 1
        if exhaustive:
            xs = np.arange(int(info.min), int(info.max) + 1, dtype=np.int64)
        else:
            edges = sorted({min(max(int(v), int(info.min)), int(info.max))
                            for v in (info.min, info.min + 1, info.max - 1,
                                      info.max, 0, 1, -1)})
            xs = np.array(edges + [_rand_int(rng, info, dt) for _ in range(200)],
                          dtype=np.int64)
        # one row per value, one column: the window is the whole row, so the result is
        # the maximum of the type over the values present
        x = xs.astype(dt).reshape(1, 1, 1, len(xs))
        attrs = {"kernel_shape": [1, len(xs)], "strides": [1, 1], "pads": [0, 0, 0, 0]}

        want = doc_maxpool(x, attrs["kernel_shape"], attrs["strides"], attrs["pads"])
        try:
            ref = run_ort_batch(x, attrs)
        except Exception as exc:
            failures.append({"type": dt.name, "who": "ONNX Runtime",
                             "detail": "%s: %s" % (type(exc).__name__, exc)})
            continue
        got = np.asarray(impl.maxpool(x, **attrs))

        if not _bits_equal(want, ref):
            failures.append({"type": dt.name, "who": "the document vs ONNX Runtime",
                             "detail": "doc %s, ort %s"
                                       % (want.ravel()[:5].tolist(),
                                          np.asarray(ref).ravel()[:5].tolist())})
        if not _bits_equal(ref, got):
            failures.append({"type": dt.name, "who": "the implementation vs ONNX Runtime",
                             "detail": "ort %s, impl %s"
                                       % (np.asarray(ref).ravel()[:5].tolist(),
                                          got.ravel()[:5].tolist())})
        rows.append("%s %s" % (dt.name, "exhaustive" if exhaustive
                               else "%d values" % len(xs)))
    return {"sweep": "int domain (document vs ORT vs implementation)",
            "summary": "%d types: %s" % (len(rows), ", ".join(rows)),
            "failures": failures}


def _spec_examples():
    """The worked examples of every section, as documented."""
    f32 = np.float32
    return [
        ("real/float Example 1",
         np.arange(1, 17, dtype=f32).reshape(1, 1, 4, 4),
         {"kernel_shape": [2, 2], "strides": [2, 2]},
         np.array([[6.0, 8.0], [14.0, 16.0]], dtype=f32).reshape(1, 1, 2, 2)),
        ("real/float Example 2",
         np.array([1.0, 2.0, 3.0], dtype=f32).reshape(1, 1, 3),
         {"kernel_shape": [2], "strides": [1], "pads": [1, 1]},
         np.array([1.0, 2.0, 3.0, 3.0], dtype=f32).reshape(1, 1, 4)),
        ("float Example 2 (NaN, -inf)",
         np.array([1.0, np.nan, 3.0, -np.inf], dtype=f32).reshape(1, 1, 4),
         {"kernel_shape": [2], "strides": [2]},
         np.array([np.nan, 3.0], dtype=f32).reshape(1, 1, 2)),
        ("int Example 1",
         np.array([-1, 2, -3, 4, 5, -6, 7, -8, -9, 10, -11, 12, 13, -14, 15, -16],
                  dtype=np.int8).reshape(1, 1, 4, 4),
         {"kernel_shape": [2, 2], "strides": [2, 2]},
         np.array([5, 7, 13, 15], dtype=np.int8).reshape(1, 1, 2, 2)),
        ("uint Example 1",
         np.array([0, 200, 3, 255, 17, 128, 9, 64, 250, 1, 100, 7, 33, 254, 2, 90],
                  dtype=np.uint8).reshape(1, 1, 4, 4),
         {"kernel_shape": [2, 2], "strides": [2, 2]},
         np.array([200, 255, 254, 100], dtype=np.uint8).reshape(1, 1, 2, 2)),
    ]


def sweep_spec_examples(impl):
    """Every worked example of the document: as documented, and as ONNX Runtime does it."""
    failures = []
    examples = _spec_examples()
    for label, x, attrs, documented in examples:
        try:
            ref = run_ort_batch(x, attrs)
        except Exception as exc:
            failures.append({"who": label, "detail": "ONNX Runtime refused: %s: %s"
                             % (type(exc).__name__, exc)})
            continue
        got = np.asarray(impl.maxpool(x, **attrs))
        if not _bits_equal(documented, ref):
            failures.append({"who": label, "detail": "documented %s, ORT %s"
                             % (documented.ravel().tolist(),
                                np.asarray(ref).ravel().tolist())})
        if not _bits_equal(documented, got):
            failures.append({"who": label, "detail": "documented %s, implementation %s"
                             % (documented.ravel().tolist(), got.ravel().tolist())})
    return {"sweep": "the document's worked examples",
            "summary": "%d examples, as documented vs ORT vs implementation"
                       % len(examples),
            "failures": failures}


def sweep_shape_formula(impl):
    """The shape formula of the Function section, over a family of shapes.

    The formula is checked against the shape ONNX Runtime produces, and against the
    shape the implementation produces, for every combination of kernel, stride and
    padding that satisfies the window-coverage precondition.
    """
    failures = []
    checked = 0
    for x_shape in [(1, 1, 4, 4), (2, 3, 5, 7), (1, 2, 1, 1), (1, 1, 6, 6),
                    (1, 1, 3), (2, 2, 5), (1, 1, 2, 3, 4)]:
        m = len(x_shape) - 2
        for kernel in ([2] * m, [1] * m, [3] * m):
            for stride in ([1] * m, [2] * m):
                for pad in ([0] * (2 * m), [1] * (2 * m)):
                    # the window-coverage precondition: every window covers an element
                    ok = True
                    for t in range(m):
                        d = x_shape[t + 2]
                        if d >= 1:
                            if d + pad[t] + pad[m + t] < kernel[t]:
                                ok = False
                        else:
                            if pad[t] + pad[m + t] < kernel[t]:
                                ok = False
                    if not ok:
                        continue
                    attrs = {"kernel_shape": list(kernel), "strides": list(stride),
                             "pads": list(pad)}
                    want = doc_out_shape(x_shape, kernel, stride, pad)
                    x = np.arange(int(np.prod(x_shape)), dtype=np.float32).reshape(x_shape)
                    try:
                        ref = run_ort_batch(x, attrs)
                    except Exception as exc:
                        failures.append({"who": "shape %s %s" % (x_shape, attrs),
                                         "detail": "ONNX Runtime refused: %s: %s"
                                                   % (type(exc).__name__, exc)})
                        continue
                    got = np.asarray(impl.maxpool(x, **attrs))
                    checked += 1
                    if tuple(ref.shape) != want:
                        failures.append({"who": "shape %s %s" % (x_shape, attrs),
                                         "detail": "document %s, ORT %s"
                                                   % (want, tuple(ref.shape))})
                    if tuple(got.shape) != want:
                        failures.append({"who": "shape %s %s" % (x_shape, attrs),
                                         "detail": "document %s, implementation %s"
                                                   % (want, tuple(got.shape))})
    return {"sweep": "the shape formula (document vs ORT vs implementation)",
            "summary": "%d shape/attribute combinations" % checked,
            "failures": failures}


def sweep_special_values(impl):
    """The special values of the floating-point types, as operands and as results.

    The document states the rule for each: NaN in a window gives NaN; +inf is greater
    than every finite value and -inf smaller; +0 and -0 are equal and a window of nulls
    gives a null result whose sign is not specified.  The comparison is against ONNX
    Runtime, and the sign of a null result is not a failure.
    """
    failures = []
    checked = 0
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        f = np.finfo(dt)
        nan, inf = float("nan"), float("inf")
        maxf, tiny = float(f.max), float(f.tiny)
        cases = [
            ("NaN in the window", [1.0, nan, 3.0, -inf], [2], [2], [0, 0]),
            ("all -inf", [-inf, -inf, -inf, -inf], [2], [2], [0, 0]),
            ("+inf and finite", [inf, 1.0, -inf, 2.0], [2], [2], [0, 0]),
            ("signed zeros", [0.0, -0.0, -0.0, 0.0], [2], [2], [0, 0]),
            ("extremes", [maxf, -maxf, tiny, -tiny], [2], [2], [0, 0]),
            ("subnormals", [tiny / 2.0, -tiny / 2.0, 0.0, -0.0], [2], [2], [0, 0]),
            ("NaN alone in a window", [nan, 1.0, 2.0, 3.0], [1], [1], [0, 0]),
        ]
        for label, values, kernel, strides, pads in cases:
            x = np.array(values, dtype=dt).reshape(1, 1, len(values))
            attrs = {"kernel_shape": list(kernel), "strides": list(strides),
                     "pads": list(pads)}
            try:
                ref = run_ort_batch(x, attrs)
            except Exception as exc:
                failures.append({"type": dt.name, "who": label,
                                 "detail": "ONNX Runtime refused: %s: %s"
                                           % (type(exc).__name__, exc)})
                continue
            got = np.asarray(impl.maxpool(x, **attrs))
            checked += 1
            if not _bits_equal(ref, got):
                # a null result's sign is not specified: +0 and -0 are both conforming
                if (ref.dtype.kind == "f" and got.shape == ref.shape
                        and np.all((ref == 0) & (got == 0))):
                    continue
                failures.append({"type": dt.name, "who": label,
                                 "detail": "ort %s, impl %s"
                                           % (np.asarray(ref).ravel().tolist(),
                                              got.ravel().tolist())})
    return {"sweep": "the special values of the float types",
            "summary": "%d cases over %d types" % (checked, len(ALL_FLOAT)),
            "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_shape_formula, sweep_int_domain,
            sweep_special_values]


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
    """The value families of the float section, as operands of a window."""
    f = np.finfo(np.dtype(dtype))
    maxf = float(f.max)
    tiny = float(f.tiny)
    nan, inf = float("nan"), float("inf")
    return [
        ("signed zeros", [0.0, -0.0, -0.0, 0.0]),
        ("infinities", [inf, -inf, inf, -inf]),
        ("nan", [nan, 1.0, nan, inf]),
        ("ordinary", [1.0, 2.5, -3.0, 4.75]),
        ("extremes", [maxf, -maxf, tiny, -tiny]),
        ("subnormals", [tiny / 2.0, -tiny / 2.0, 0.0, -0.0]),
        ("one ulp from 1.0", [1.0, 1.0 + 2.0 ** -f.nmant, 1.0, 1.0]),
    ]


def _ordinary_value_cases(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return [("ordinary", [1.0, 2.5, -3.0, 4.75])]
    info = np.iinfo(dtype)
    lo, hi = int(info.min), int(info.max)
    if lo < 0:
        return [("extremes", [lo, hi, -1, 0, 1, hi, lo, -1])]
    return [("extremes", [0, hi, 1, 4, hi, 0, 1, hi // 2])]


SHAPE_CASES = [
    ("rank-3, kernel 2, stride 1", (1, 1, 5), [2], [1], [0, 0]),
    ("rank-3, kernel 2, stride 2", (1, 1, 6), [2], [2], [0, 0]),
    ("rank-3, kernel 2, stride 1, pads 1", (1, 1, 3), [2], [1], [1, 1]),
    ("rank-4, kernel 2x2, stride 2", (1, 1, 4, 4), [2, 2], [2, 2], [0, 0, 0, 0]),
    ("rank-4, kernel 2x2, stride 1", (2, 3, 5, 7), [2, 2], [1, 1], [0, 0, 0, 0]),
    ("rank-4, kernel 3x3, stride 1, pads 1", (1, 1, 4, 4), [3, 3], [1, 1], [1, 1, 1, 1]),
    ("rank-4, kernel 1x1", (1, 2, 3, 3), [1, 1], [1, 1], [0, 0, 0, 0]),
    ("rank-5, kernel 2x2x2, stride 1", (1, 1, 2, 3, 4), [2, 2, 2], [1, 1, 1],
     [0, 0, 0, 0, 0, 0]),
    ("rank-4, zero-sized spatial dim, pads cover", (1, 1, 0, 4), [2, 2], [1, 1],
     [1, 1, 0, 0]),
    ("rank-4, zero-sized batch", (0, 1, 4, 4), [2, 2], [2, 2], [0, 0, 0, 0]),
    ("rank-4, zero-sized channel", (1, 0, 4, 4), [2, 2], [2, 2], [0, 0, 0, 0]),
    ("rank-4, kernel equals the spatial extent", (1, 1, 3, 3), [3, 3], [1, 1],
     [0, 0, 0, 0]),
    ("rank-4, stride larger than the kernel", (1, 1, 7, 7), [2, 2], [3, 3],
     [0, 0, 0, 0]),
]


def cases():
    """Every individual case: (label, [X], attrs)."""
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        for name, values in _float_value_cases(dtype):
            x = np.array(values, dtype=dt).reshape(1, 1, len(values))
            yield ("%s: %s" % (dt.name, name),
                   [x], {"kernel_shape": [2], "strides": [2]})
        quiet, signalling, sign = NAN_PATTERNS[dt.itemsize]
        yield ("%s: NaN operand, quiet, payload 1" % dt.name,
               [_bit_nan(dtype, quiet).reshape(1, 1, 1)],
               {"kernel_shape": [1], "strides": [1]})
        yield ("%s: NaN operand, quiet, payload 1, sign set" % dt.name,
               [_bit_nan(dtype, quiet | sign).reshape(1, 1, 1)],
               {"kernel_shape": [1], "strides": [1]})
        yield ("%s: NaN operand, signalling" % dt.name,
               [_bit_nan(dtype, signalling).reshape(1, 1, 1)],
               {"kernel_shape": [1], "strides": [1]})
    for dtype in ALL_INT + ALL_UINT:
        dt = np.dtype(dtype)
        for name, values in _ordinary_value_cases(dtype):
            x = np.array(values, dtype=dt).reshape(1, 1, len(values))
            yield ("%s: %s" % (dt.name, name),
                   [x], {"kernel_shape": [2], "strides": [2]})
    for name, shape, kernel, strides, pads in SHAPE_CASES:
        for dtype in ALL_TYPES:
            dt = np.dtype(dtype)
            x = _arr(shape, dt)
            yield ("%s: %s" % (dt.name, name),
                   [x], {"kernel_shape": list(kernel), "strides": list(strides),
                         "pads": list(pads)})


# ---------------------------------------------------------------------------
# coverage: every section anchor of the document
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
        COVERAGE["real"] += 1
    elif _dt.kind == "i":
        COVERAGE["int"] += 1
        COVERAGE["real"] += 1
    else:
        COVERAGE["uint"] += 1
        COVERAGE["real"] += 1
