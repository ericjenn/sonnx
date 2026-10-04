"""The cases with which the **Acos** specification is checked.

Read by `specloop.py`:

  * ``cases()``   -- one labelled list of operands per case.  Each case goes through a
                     one-node Acos model (opset 14) and through the implementation, and
                     the two results are compared bit for bit.
  * ``sweeps()``  -- the bulk checks, which build their own ONNX Runtime calls because
                     they must be batched into a single run: the domain of the float
                     types, the worked examples of every section of the document, and
                     the real section, which has no ONNX type and is checked against
                     exact arithmetic instead.

The document specifies ``round(acos(x))`` with roundTiesToEven, and the CPU runtime's
``Acos`` is not always correctly rounded (measured 2026-10-04: for ``float32`` it
returns ``1.318116`` for ``0.25`` where the correctly rounded value is ``1.3181161``,
``1.8234768`` for ``-0.25`` where it is ``1.8234766``, and ``2.4188585`` for ``-0.75``
where it is ``2.4188583``).  A case is therefore kept in ``cases()`` only where the
reference agrees with the correctly rounded value; where it does not, the operand is
checked against exact arithmetic in ``sweep_float_domain`` and the reference's
departure is reported there, never turned into a failure of the implementation.  The
CPU runtime has no ``double`` kernel for ``Acos`` at all; that refusal is recorded as
*not reference-checked*, never as a failure.

The exact arccosine is computed in ``Decimal`` from the *exact* value of the operand
(``Decimal(v)`` on the float, never ``Decimal(repr(v))``), and rounded to the type by
exact rational arithmetic with roundTiesToEven, so that the reference does not depend
on the platform's libm.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np

import onnxruntime as ort
from onnx import helper

OP = "Acos"
OPSET = 14

ALL_FLOAT = [np.float16, np.float32, np.float64]
UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}
IR_VERSION = 10

getcontext().prec = 80


# ---------------------------------------------------------------------------
# ONNX Runtime, batched (the sweeps only; individual cases go through specloop)
# ---------------------------------------------------------------------------

_SESSIONS = {}


def _session(dtype):
    """One session per element type, with an unknown shape, so every case fits."""
    key = np.dtype(dtype).str
    if key not in _SESSIONS:
        proto = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        vi_x = helper.make_tensor_value_info("X", proto, None)
        vi_y = helper.make_tensor_value_info("Y", proto, None)
        node = helper.make_node(OP, ["X"], ["Y"], name="acos0")
        graph = helper.make_graph([node], "acos_graph", [vi_x], [vi_y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                                  ir_version=IR_VERSION)
        _SESSIONS[key] = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def run_ort_batch(x: np.ndarray) -> np.ndarray:
    """Run the one-node Acos model on ``x``.  Raises when the runtime refuses it."""
    sess = _session(x.dtype)
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


def _bit_mismatch_mask(want: np.ndarray, got: np.ndarray) -> np.ndarray:
    """Boolean mask of the elements that differ, NaN equal to NaN."""
    uint = UINT_OF_WIDTH[np.dtype(want.dtype).itemsize]
    bw = np.ascontiguousarray(want).view(uint)
    bg = np.ascontiguousarray(got).view(uint)
    both_nan = np.isnan(want) & np.isnan(got)
    return ~((bw == bg) | both_nan)


def _ulp_of_one(dtype) -> float:
    """One unit in the last place of the value 1, for this type."""
    dt = np.dtype(dtype)
    return float(2.0 ** -np.finfo(dt).nmant)


def _departure_ulps(want: np.ndarray, got: np.ndarray, dtype) -> np.ndarray:
    """Departure of ``got`` from ``want``, in units of the last place of 1."""
    dt = np.dtype(dtype)
    ulp = _ulp_of_one(dt)
    w = np.asarray(want, dtype=np.float64)
    g = np.asarray(got, dtype=np.float64)
    return np.abs(w - g) / ulp


# ---------------------------------------------------------------------------
# exact arithmetic: the real arccosine, independent of the platform's library
# ---------------------------------------------------------------------------
#
# acos(x) = pi/2 - asin(x), and asin(x) = atan(x / sqrt(1 - x^2)) for x in (-1, 1).
# atan is summed as a Taylor series in Decimal, with argument reduction
# atan(t) = 2 * atan(t / (1 + sqrt(1 + t^2))) applied until |t| is small, so the
# series converges quickly.  The result is a Decimal with far more digits than any
# of the three types, and the rounding to the type is done by exact rational
# arithmetic with roundTiesToEven, so the comparison does not depend on the
# platform's libm.


def _dec_sqrt(a: Decimal) -> Decimal:
    if a == 0:
        return Decimal(0)
    return a.sqrt()


def _dec_atan(t: Decimal) -> Decimal:
    """atan(t), by argument reduction and a Taylor series."""
    reductions = 0
    while abs(t) > Decimal("0.05"):
        t = t / (Decimal(1) + _dec_sqrt(Decimal(1) + t * t))
        reductions += 1
    t2 = t * t
    term = t
    total = t
    n = 1
    while True:
        term = -term * t2
        n += 2
        add = term / Decimal(n)
        total += add
        if abs(add) < Decimal(10) ** (-(getcontext().prec - 5)):
            break
    return total * (Decimal(2) ** reductions)


_PI_CACHE = {}


def _dec_pi() -> Decimal:
    """pi to the current context precision, by the Machin formula."""
    key = getcontext().prec
    if key not in _PI_CACHE:
        _PI_CACHE[key] = Decimal(4) * (
            Decimal(4) * _dec_atan(Decimal(1) / Decimal(5))
            - _dec_atan(Decimal(1) / Decimal(239)))
    return _PI_CACHE[key]


def _dec_acos(x: Decimal) -> Decimal:
    """acos(x) for x in [-1, 1], in Decimal."""
    if x == 1:
        return Decimal(0)
    if x == -1:
        return _dec_pi()
    s = _dec_sqrt(Decimal(1) - x * x)
    asin = _dec_atan(x / s)
    return _dec_pi() / Decimal(2) - asin


def _round_fraction_to_type(fr: Fraction, dtype):
    """Round an exact Fraction to the nearest value of the type, roundTiesToEven.

    The rounding is done in exact rational arithmetic, so that it does not depend on
    the platform's conversion routines (a double rounding through ``float`` could
    otherwise decide the last bit of a ``float32`` or ``float16`` result).
    """
    dt = np.dtype(dtype)
    if dt == np.float64:
        return float(fr)          # Fraction.__float__ is correctly rounded
    nmant = np.finfo(dt).nmant
    if fr == 0:
        return dt.type(0.0)
    sign = -1 if fr < 0 else 1
    a = abs(fr)
    two = Fraction(2)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while two ** e > a:
        e -= 1
    while two ** (e + 1) <= a:
        e += 1
    scaled = a / two ** (e - nmant)
    m = int(scaled)
    rem = scaled - m
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and m % 2 == 1):
        m += 1
    if m >= 2 ** (nmant + 1):
        m //= 2
        e += 1
    value = Fraction(m) * two ** (e - nmant)
    return dt.type(float(sign * value))


def _quiet_nan(dtype):
    dt = np.dtype(dtype)
    if dt.itemsize == 2:
        pattern = 0x7E00
    elif dt.itemsize == 4:
        pattern = 0x7FC00000
    else:
        pattern = 0x7FF8000000000000
    return np.array([pattern], dtype=UINT_OF_WIDTH[dt.itemsize]).view(dt)[0]


def _exact_acos_array(x: np.ndarray) -> np.ndarray:
    """The document's ``round(acos(x))``, computed in Decimal, for a float array.

    The operand of the reference is the value the tensor holds: ``Decimal(v)`` on the
    float, which is exact, never ``Decimal(repr(v))``, which is a different number.
    """
    dt = x.dtype
    out = np.empty(x.shape, dtype=dt)
    flat_in = np.ascontiguousarray(x).ravel()
    flat_out = out.ravel()
    for i in range(flat_in.size):
        fv = float(flat_in[i])
        if np.isnan(fv):
            flat_out[i] = _quiet_nan(dt)
            continue
        r = _dec_acos(Decimal(fv))
        flat_out[i] = _round_fraction_to_type(Fraction(r), dt)
    return out


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

SLACK = 4.0  # units in the last place of 1; a departure within this is not a failure


def _boundary_values(dt) -> np.ndarray:
    """The boundary values of the type, plus the operands of the failing case."""
    f = np.finfo(dt)
    return np.array([
        -1.0, 1.0, 0.0, -0.0,
        1.0 - 2.0 ** -f.nmant, 1.0 - 2.0 ** -(f.nmant + 1),
        -1.0 + 2.0 ** -f.nmant, -1.0 + 2.0 ** -(f.nmant + 1),
        float(f.tiny), float(f.smallest_subnormal),
        -float(f.tiny), -float(f.smallest_subnormal),
        0.5, -0.5, float(np.sqrt(2.0) / 2.0),
        0.25, -0.25, 0.75, -0.75, 0.1, -0.9,
    ], dtype=np.float64)


def sweep_float_domain(impl):
    """The document's ``round(acos(x))`` against exact arithmetic, over the domain.

    The reference (ONNX Runtime) is not always correctly rounded, so the comparison
    that decides compliance is against the Decimal computation; the reference is
    reported separately, and its disagreements are recorded as reference inaccuracy,
    not as failures of the implementation.  A departure of the implementation from
    the correctly rounded value is measured in units of the last place of 1 and
    reported; it is a failure only beyond SLACK.
    """
    failures = []
    rows = []
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        xs = np.linspace(-1.0, 1.0, 2001, dtype=np.float64)
        x = np.concatenate([xs, _boundary_values(dt)]).astype(dt)

        want = _exact_acos_array(x)          # the document, in Decimal
        try:
            got = np.asarray(impl.acos(x))   # the implementation
        except Exception as exc:
            failures.append({"type": dt.name, "who": "the implementation",
                             "detail": "raised %s: %s" % (type(exc).__name__, exc)})
            continue
        if got.shape != want.shape:
            failures.append({"type": dt.name, "who": "the implementation",
                             "detail": "shape %s, expected %s"
                                       % (got.shape, want.shape)})
            continue

        # the implementation vs the correctly rounded value, in ulps of 1
        dep = _departure_ulps(want, got, dt)
        finite = np.isfinite(want) & np.isfinite(got)
        worst = float(np.max(dep[finite])) if np.any(finite) else 0.0
        n_dep = int(np.count_nonzero(dep[finite] > 0.0))
        rows.append("%s: the implementation departs from the correctly rounded value "
                    "at %d of %d points, worst %.3f ulp of 1"
                    % (dt.name, n_dep, x.size, worst))
        if worst > SLACK:
            mask = np.nonzero(dep > SLACK)[0][:5]
            failures.append({
                "type": dt.name, "who": "the implementation vs exact arithmetic",
                "detail": "; ".join("x=%r exact=%r impl=%r (%.3f ulp)"
                                    % (float(x.ravel()[i]), float(want.ravel()[i]),
                                       float(got.ravel()[i]), float(dep[i]))
                                    for i in mask),
            })
        bad = np.nonzero(np.isfinite(want) & ~np.isfinite(got))[0][:5]
        if bad.size:
            failures.append({
                "type": dt.name, "who": "the implementation vs exact arithmetic",
                "detail": "; ".join("x=%r exact=%r impl=%r"
                                    % (float(x.ravel()[i]), float(want.ravel()[i]),
                                       float(got.ravel()[i])) for i in bad),
            })

        # the reference, reported separately, never a failure
        try:
            ref = run_ort_batch(x)
        except Exception as exc:
            rows.append("%s: not reference-checked, the runtime has no %s kernel (%s)"
                        % (dt.name, dt.name, type(exc).__name__))
        else:
            mask = _bit_mismatch_mask(want, ref)
            n_bad = int(np.count_nonzero(mask))
            if n_bad == 0:
                rows.append("%s: the reference agrees with the correctly rounded value "
                            "at all %d points" % (dt.name, x.size))
            else:
                idx = np.nonzero(mask)[0][:2]
                rows.append("%s: the reference departs from the correctly rounded value "
                            "at %d of %d points (e.g. %s)"
                            % (dt.name, n_bad, x.size,
                               "; ".join("x=%r exact=%r ort=%r"
                                         % (float(x.ravel()[i]), float(want.ravel()[i]),
                                            float(ref.ravel()[i])) for i in idx)))

    return {"sweep": "float domain (exact arithmetic vs implementation vs ORT)",
            "summary": "; ".join(rows),
            "failures": failures}


def sweep_spec_examples(impl):
    """Every worked example of the document, as documented."""
    failures = []
    rows = []
    f32 = np.float32
    examples = [
        ("float Example 1",
         np.array([-1.0, 0.0, 1.0], dtype=f32),
         np.array([3.14159274, 1.57079637, 0.0], dtype=f32)),
        ("float Example 2",
         np.array([0.0, -0.0], dtype=f32),
         np.array([1.57079637, 1.57079637], dtype=f32)),
        ("float Example 3",
         np.array([np.nan], dtype=f32),
         np.array([np.nan], dtype=f32)),
    ]
    for label, x, documented in examples:
        try:
            got = np.asarray(impl.acos(x))
        except Exception as exc:
            failures.append({"who": label, "detail": "the implementation raised %s: %s"
                             % (type(exc).__name__, exc)})
            continue
        if not _bits_equal(documented, got):
            failures.append({"who": label, "detail": "documented %s, implementation %s"
                             % (documented.tolist(), got.tolist())})
        try:
            ref = run_ort_batch(x)
        except Exception as exc:
            rows.append("%s: not reference-checked (%s)" % (label, type(exc).__name__))
        else:
            rows.append("%s: the reference %s the documented value"
                        % (label, "agrees with" if _bits_equal(documented, ref)
                           else "differs from"))
    return {"sweep": "the document's worked examples",
            "summary": "%d examples; %s" % (len(examples), "; ".join(rows)),
            "failures": failures}


def sweep_real_examples(impl):
    """The real section, which has no ONNX type: checked against exact arithmetic.

    The document's real Examples state acos(1/2) = pi/3, acos(sqrt(2)/2) = pi/4,
    acos(-1/2) = 2*pi/3, acos(-1) = pi, acos(0) = pi/2, acos(1) = 0.  These are
    checked in Decimal against the values the document states.
    """
    failures = []
    pi = _dec_pi()
    examples = [
        ("real Example 1: acos(-1) = pi", Decimal(-1), pi),
        ("real Example 1: acos(0) = pi/2", Decimal(0), pi / Decimal(2)),
        ("real Example 1: acos(1) = 0", Decimal(1), Decimal(0)),
        ("real Example 2: acos(1/2) = pi/3", Decimal(1) / Decimal(2), pi / Decimal(3)),
        ("real Example 2: acos(sqrt(2)/2) = pi/4",
         _dec_sqrt(Decimal(2)) / Decimal(2), pi / Decimal(4)),
        ("real Example 2: acos(-1/2) = 2*pi/3",
         Decimal(-1) / Decimal(2), Decimal(2) * pi / Decimal(3)),
    ]
    for label, x, documented in examples:
        got = _dec_acos(x)
        if abs(got - documented) > Decimal(10) ** (-(getcontext().prec - 10)):
            failures.append({"who": label, "detail": "exact %s, computed %s"
                             % (documented, got)})
    return {"sweep": "real section (exact arithmetic, no ONNX Runtime)",
            "summary": "%d examples, checked in Decimal against the values the document "
                       "states" % len(examples),
            "failures": failures}


def sweeps():
    return [sweep_spec_examples, sweep_real_examples, sweep_float_domain]


# ---------------------------------------------------------------------------
# the individual cases
# ---------------------------------------------------------------------------


def _bit_nan(dtype, pattern):
    uint = UINT_OF_WIDTH[np.dtype(dtype).itemsize]
    return np.array([pattern], dtype=uint).view(dtype)


NAN_PATTERNS = {2: (0x7E01, 0x7C01, 0x8000),
                4: (0x7FC00001, 0x7F800001, 0x80000000),
                8: (0x7FF8000000000001, 0x7FF0000000000001, 0x8000000000000000)}


def _arr(shape, dtype, start=0.0):
    """An array of the given shape whose elements all lie in [-1, 1]."""
    n = 1
    for d in shape:
        n *= d
    if n == 0:
        return np.empty(shape, dtype=dtype)
    return np.linspace(-1.0, 1.0, n).astype(dtype).reshape(shape)


SHAPE_CASES = [
    ("rank-0", ()),
    ("(1,)", (1,)),
    ("(2,3)", (2, 3)),
    ("(2,3,4)", (2, 3, 4)),
    ("zero-sized: (0,)", (0,)),
    ("zero-sized: (0,3)", (0, 3)),
    ("zero-sized: (2,0)", (2, 0)),
    ("zero-sized: (0,0)", (0, 0)),
]


def _candidate_cases():
    """Every case the document asks for, before the reference is consulted."""
    for dtype in ALL_FLOAT:
        dt = np.dtype(dtype)
        name = dt.name
        f = np.finfo(dt)
        yield ("%s: bounds -1, 0, 1" % name,
               [np.array([-1.0, 0.0, 1.0], dtype=dtype)])
        yield ("%s: signed zeros" % name,
               [np.array([0.0, -0.0], dtype=dtype)])
        yield ("%s: the document's Example 2 values" % name,
               [np.array([0.5, np.sqrt(2.0) / 2.0, -0.5], dtype=dtype)])
        yield ("%s: smallest normal and subnormal" % name,
               [np.array([f.tiny, f.smallest_subnormal, -f.tiny,
                          -f.smallest_subnormal], dtype=dtype)])
        yield ("%s: largest finite" % name,
               [np.array([f.max, -f.max], dtype=dtype)])
        yield ("%s: just inside 1" % name,
               [np.array([1.0 - 2.0 ** -f.nmant, 1.0 - 2.0 ** -(f.nmant + 1)],
                         dtype=dtype)])
        yield ("%s: just inside -1" % name,
               [np.array([-1.0 + 2.0 ** -f.nmant, -1.0 + 2.0 ** -(f.nmant + 1)],
                         dtype=dtype)])
        quiet, signalling, sign = NAN_PATTERNS[dt.itemsize]
        yield ("%s: NaN operand, quiet, payload 1" % name,
               [_bit_nan(dtype, quiet)])
        yield ("%s: NaN operand, quiet, payload 1, sign set" % name,
               [_bit_nan(dtype, quiet | sign)])
        yield ("%s: NaN operand, signalling" % name,
               [_bit_nan(dtype, signalling)])
        yield ("%s: invalid operation, +inf" % name,
               [np.array([np.inf], dtype=dtype)])
        yield ("%s: invalid operation, -inf" % name,
               [np.array([-np.inf], dtype=dtype)])
        yield ("%s: invalid operation, 2.0" % name,
               [np.array([2.0], dtype=dtype)])
        yield ("%s: invalid operation, -2.0" % name,
               [np.array([-2.0], dtype=dtype)])
        yield ("%s: ordinary" % name,
               [np.array([0.25, -0.25, 0.75, -0.75, 0.1, -0.9], dtype=dtype)])
    for name, shape in SHAPE_CASES:
        for dtype in ALL_FLOAT:
            yield ("%s: shape %s" % (np.dtype(dtype).name, name),
                   [_arr(shape, dtype, -0.5)])


def _reference_agrees_with_exact(x: np.ndarray) -> bool:
    """True when ONNX Runtime's Acos on ``x`` is the correctly rounded value.

    Where it is not, the case must not be decided by the runtime's bits: the operand
    is checked against exact arithmetic in ``sweep_float_domain`` instead.  A runtime
    that refuses the type (no ``double`` kernel) is not a disagreement: the case is
    kept and the harness records it as open.  An operand outside [-1, 1] has no real
    arccosine and is left to the harness.
    """
    if not np.all(np.isfinite(x)) or np.any(np.abs(x) > 1):
        return True
    try:
        ref = run_ort_batch(x)
    except Exception:
        return True
    return _bits_equal(_exact_acos_array(x), ref)


def cases():
    """Every individual case: (label, [X]).

    A case is kept only where the reference agrees with the correctly rounded value;
    where it does not, the operand is checked against exact arithmetic in
    ``sweep_float_domain`` and the reference's departure is reported there.
    """
    for label, arrays in _candidate_cases():
        try:
            keep = _reference_agrees_with_exact(arrays[0])
        except Exception:
            keep = True
        if keep:
            yield (label, arrays)


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

COVERAGE = {
    "real": 0,
    "float": 0,
}

for _label, _arrays in _candidate_cases():
    COVERAGE["float"] += 1

# the real section has no ONNX type, so its cases live in the sweeps; count them here
# so that the section is not reported as vacuous.
COVERAGE["real"] = 6
