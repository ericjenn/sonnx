# Sub (opset 14): specification vs ONNX Runtime

Verification of `ops/sub.md`, revision 2026-10-03, **after the V1-V4 amendments**.
The operator is implemented from that document alone (`impl_from_spec.py`, written
by an implementer that read nothing but the specification) and compared with ONNX
Runtime 1.30.0 through a one-node `Sub` model, opset 14, `CPUExecutionProvider`.
For semantics, no other document and no ONNX Runtime source was read; the callouts
of the specification were declared non-normative to the implementer.

The specification is authoritative. Nothing was adapted to ONNX Runtime, no test
was weakened, and a refusal by ONNX Runtime would be recorded as a result.

## Reproduce

```
cd /run/media/eric/Work1/Work/gits/sonnx/verification/sub
/home/eric/Venvs/onnxcheck/bin/python test_vs_ort.py              # compact table; exit 1 on a mismatch
/home/eric/Venvs/onnxcheck/bin/python test_vs_ort.py --markdown   # the tables reproduced below
```

## Summary

**205 cases run, 205 passed, 0 mismatched, 0 refused by ONNX Runtime**, plus the
three worked examples of the real section, which are exact.

The first issue of the document (before the amendments) was run the same way and
gave **181 cases, 178 passed, 3 mismatched**: one case per float type diverged.
That run and the implementation it used are kept in `archive_first_issue/`; what
the amendments changed is set out below, and both divergences are closed here.

## What the amendments changed, and the evidence for each

The two divergences are analysed in the first-issue record (`archive_first_issue/results_first_issue.md`).
One was an **incorrect** specification, one an **incomplete** one; the document now
says, for every case, what ONNX Runtime does.

### V1 - the overflow threshold (was incorrect)

The float case analysis had a fourth case, "inf if the exact difference overflows
the range of the type of C". IEEE 754 calls a result an overflow only when the value
*rounded with an unbounded exponent range* exceeds the largest finite value of the
type, so for an exact difference in the band `(max, max + ulp(max)/2)` the document
demanded an infinity where IEEE 754, NumPy and ONNX Runtime give `max`. A conforming
implementation of the document had to return a result IEEE 754 forbids. The case is
gone; `round(x)` is now defined with an unbounded exponent range, and the infinities
follow from the rounding itself.

The threshold that rule produces is the midpoint between the largest finite value and
the next power of two, because that midpoint is a tie and `roundTiesToEven` breaks it
upwards - the two rules are one rule. The harness now probes it per float type, a
quarter of an ulp below the midpoint, at it, and a quarter above:

| type | `max - b`, below / at / above the midpoint | ONNX Runtime |
| ---- | ------------------------------------------ | ------------ |
| float16 | `65504 - (-8)` = 65512, `65504 - (-16)` = 65520, `65504 - (-24)` = 65528 | `65504`, `inf`, `inf` |
| float32 | `max - (-2^102)`, `max - (-2^103)`, `max - (-1.5*2^103)` | `max`, `inf`, `inf` |
| float64 | `max - (-2^969)`, `max - (-2^970)`, `max - (-1.5*2^970)` | `max`, `inf`, `inf` |

All twelve rows pass: the specification and ONNX Runtime agree on both sides of the
threshold and at the tie.

### V2 - the NaN of the invalid operation (was incomplete)

The document said the result is NaN and stopped there. Which NaN is not fixed by
IEEE 754, and implementations differ. The first case now states that every quiet NaN
of the type of C is a conforming result, for a NaN operand as for the invalid
operation; the comparison accepts any quiet NaN and reports a payload difference
separately. What ONNX Runtime does:

| case | float16 | float32 | float64 |
| ---- | ------- | ------- | ------- |
| `inf - inf` | `0xfe00` | `0xffc00000` | `0xfff8000000000000` |
| `-inf - (-inf)` | `0xfe00` | `0xffc00000` | `0xfff8000000000000` |
| operand NaN, payload 1 | `0x7e01` becomes `0x7e00` | `0x7fc00001` kept | `0x7ff8000000000001` kept |
| operand NaN, payload 1, sign set | `0xfe01` becomes `0xfe00` | kept | kept |
| operand signalling NaN | `0x7c01` becomes `0x7e00` | quieted, payload kept | quieted, payload kept |

The invalid operation gives the canonical quiet NaN **with the sign bit set** in all
three types, which no wording of the document implies; that is precisely why the
permissive rule is the only workable one. It is also the only rule ONNX Runtime can
satisfy: a stricter one would make it non-conforming for float16, where it
canonicalises the payload of its operands.

### V3 and V4 - the null result and two rows of the table

The last case said the rounding "may make the result null or subnormal". It cannot be
null: the values of a type are all multiples of that type's smallest subnormal, so a
non-null difference is at least one of them, and a null difference is decided by the
signed-zero rule. The Overflow and Underflow rows of the float disposition table
carried the same two readings; they now name the rounded result and the subnormal
result. Both are wording corrections with no case of their own to test.

## The real section

The real section is not one of the types of ONNX Sub, so ONNX Runtime cannot be asked
about it. Its three worked examples are checked in exact rational arithmetic instead,
and the document's own values are right in all three:

| example | A | B | C |
| ------- | - | - | - |
| 1 | `[6.1, 9.5, 35.7]` | `[3, 3.3, 5.1]` | `[3.1, 6.2, 30.6]` |
| 2 | `[[3.7, 4.4], [16.2, 0.5], [25.3, 24.8]]` | `[1, 2]` | `[[2.7, 2.4], [15.2, -1.5], [24.3, 22.8]]` |
| 3 | `[[5, 3.25], [4, 1.75]]` | `2.5` | `[[2.5, 0.75], [1.5, -0.75]]` |

Three further cases run the real examples' arithmetic with float64 operands, the only
type of ONNX Sub that can stand in for them, and are compared with ONNX Runtime as
usual.

## Coverage

| dimension | coverage |
| --------- | -------- |
| types | `float16`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`; float64 also as the stand-in for the real section; exact object arithmetic for the real section's examples |
| shapes | identical; broadcasting with a dimension of size 1; `(2,3,4)-(4,)`; `(2,3,4)-(3,1)`; rank-0 on the left, on the right, and on both sides; zero-sized `(0,3)-(1,3)`, `(0,)-(1,)`, `(0,3)-(3,)`, `(1,0)-(2,0)`, `(0,3)-(0,1)`, `(0,)-()` |
| values | ordinary; the extremes of every integer type (`min`, `max`, `-1`, `0`, `1`) and the pairs whose difference wraps; for every float type the four combinations of `+0`/`-0`, `+inf`, `-inf`, NaN, an exact difference, a difference that overflows to infinity, a difference that lands in the subnormal range, a difference that is not representable (`0.3-0.1`), an exact rounding tie, the band just above `max`, and (new) the overflow threshold from below, at the tie and from above; (new) NaN operands with a non-canonical payload, with the sign bit set, and signalling; (new) `inf - inf` and `-inf - (-inf)` |

ONNX Runtime accepted every type and every shape, including `uint64`, rank-0 and
zero-sized operands. It refused nothing.

## Judgement calls the implementer reported

These are the places where the implementing agent had to choose. All of them are in
the interface between the document and a Python function, not in the operator's
semantics - except the first, which the document itself leaves open.

1. **The real section has no numpy type.** The document's real numbers are the
   mathematical reals; the implementer mapped them onto object arrays of exact
   numbers, and sends ordinary floating arrays to the float section. A choice the
   text does not settle, and the reason the harness checks that section in exact
   rational arithmetic rather than through ONNX Runtime.
2. **Mixed operand dtypes** are outside the document's precondition that A, B and C
   have the same type; the implementation uses numpy promotion. Never engaged here.
3. **Dtypes the document does not cover** (bool, complex, strings) raise `TypeError`
   rather than producing a result.
4. **Incompatible shapes** are outside the broadcasting constraint; numpy raises
   `ValueError`. The "Error conditions" tables say no error can occur, which is about
   valid operands.
5. The implementation silences numpy's `RuntimeWarning` for overflow and the invalid
   operation, which the Error-conditions tables call nominal. No returned value
   changes.

Nothing was found where ONNX Runtime deviates from a clear statement of the amended
document.

## Tables

real section, exact arithmetic (ONNX Runtime has no real type)

| case | shape A | shape B | dtype | verdict | note |
| --- | --- | --- | --- | --- | --- |
| real Example 1 | (1, 3) | (3,) | object | PASS | exact rationals, no ONNX Runtime |
| real Example 2 | (3, 2) | (2,) | object | PASS | exact rationals, no ONNX Runtime |
| real Example 3 | (2, 2) | () | object | PASS | exact rationals, no ONNX Runtime |

float and integer sections against ONNX Runtime

| case | shape A | shape B | dtype | verdict | note |
| --- | --- | --- | --- | --- | --- |
| signed zeros | (4,) | (4,) | float16 | PASS |  |
| infinities | (6,) | (6,) | float16 | PASS |  |
| nan | (4,) | (4,) | float16 | PASS |  |
| exact difference | (4,) | (4,) | float16 | PASS |  |
| overflow to inf | (2,) | (2,) | float16 | PASS |  |
| above max, below overflow threshold | (1,) | (1,) | float16 | PASS |  |
| subnormal result | (3,) | (3,) | float16 | PASS |  |
| rounding tie (halfway) | (2,) | (2,) | float16 | PASS |  |
| not representable (0.3-0.1) | (2,) | (2,) | float16 | PASS |  |
| below the overflow threshold | (1,) | (1,) | float16 | PASS |  |
| at the overflow threshold (tie) | (1,) | (1,) | float16 | PASS |  |
| above the overflow threshold | (1,) | (1,) | float16 | PASS |  |
| NaN operand, quiet, payload 1 | (1,) | (1,) | float16 | PASS | NaN payload differs |
| NaN operand, quiet, payload 1, sign set | (1,) | (1,) | float16 | PASS | NaN payload differs |
| NaN operand, signalling | (1,) | (1,) | float16 | PASS | NaN payload differs |
| invalid operation: inf - inf | (1,) | (1,) | float16 | PASS |  |
| invalid operation: -inf - (-inf) | (1,) | (1,) | float16 | PASS |  |
| signed zeros | (4,) | (4,) | float32 | PASS |  |
| infinities | (6,) | (6,) | float32 | PASS |  |
| nan | (4,) | (4,) | float32 | PASS |  |
| exact difference | (4,) | (4,) | float32 | PASS |  |
| overflow to inf | (2,) | (2,) | float32 | PASS |  |
| above max, below overflow threshold | (1,) | (1,) | float32 | PASS |  |
| subnormal result | (3,) | (3,) | float32 | PASS |  |
| rounding tie (halfway) | (2,) | (2,) | float32 | PASS |  |
| not representable (0.3-0.1) | (2,) | (2,) | float32 | PASS |  |
| below the overflow threshold | (1,) | (1,) | float32 | PASS |  |
| at the overflow threshold (tie) | (1,) | (1,) | float32 | PASS |  |
| above the overflow threshold | (1,) | (1,) | float32 | PASS |  |
| NaN operand, quiet, payload 1 | (1,) | (1,) | float32 | PASS |  |
| NaN operand, quiet, payload 1, sign set | (1,) | (1,) | float32 | PASS |  |
| NaN operand, signalling | (1,) | (1,) | float32 | PASS |  |
| invalid operation: inf - inf | (1,) | (1,) | float32 | PASS |  |
| invalid operation: -inf - (-inf) | (1,) | (1,) | float32 | PASS |  |
| signed zeros | (4,) | (4,) | float64 | PASS |  |
| infinities | (6,) | (6,) | float64 | PASS |  |
| nan | (4,) | (4,) | float64 | PASS |  |
| exact difference | (4,) | (4,) | float64 | PASS |  |
| overflow to inf | (2,) | (2,) | float64 | PASS |  |
| above max, below overflow threshold | (1,) | (1,) | float64 | PASS |  |
| subnormal result | (3,) | (3,) | float64 | PASS |  |
| rounding tie (halfway) | (2,) | (2,) | float64 | PASS |  |
| not representable (0.3-0.1) | (2,) | (2,) | float64 | PASS |  |
| below the overflow threshold | (1,) | (1,) | float64 | PASS |  |
| at the overflow threshold (tie) | (1,) | (1,) | float64 | PASS |  |
| above the overflow threshold | (1,) | (1,) | float64 | PASS |  |
| NaN operand, quiet, payload 1 | (1,) | (1,) | float64 | PASS |  |
| NaN operand, quiet, payload 1, sign set | (1,) | (1,) | float64 | PASS |  |
| NaN operand, signalling | (1,) | (1,) | float64 | PASS |  |
| invalid operation: inf - inf | (1,) | (1,) | float64 | PASS |  |
| invalid operation: -inf - (-inf) | (1,) | (1,) | float64 | PASS |  |
| ordinary+extremes+wraps | (8,) | (8,) | int8 | PASS |  |
| ordinary+extremes+wraps | (8,) | (8,) | int16 | PASS |  |
| ordinary+extremes+wraps | (8,) | (8,) | int32 | PASS |  |
| ordinary+extremes+wraps | (8,) | (8,) | int64 | PASS |  |
| ordinary+extremes+wraps | (8,) | (8,) | uint8 | PASS |  |
| ordinary+extremes+wraps | (8,) | (8,) | uint16 | PASS |  |
| ordinary+extremes+wraps | (8,) | (8,) | uint32 | PASS |  |
| ordinary+extremes+wraps | (8,) | (8,) | uint64 | PASS |  |
| real section (float64 stand-in) [real] | (3,) | (3,) | float64 | PASS |  |
| real section, rank-0 (float64 stand-in) [real] | () | () | float64 | PASS |  |
| real section, above max below overflow threshold (float64 stand-in) [real] | (1,) | (1,) | float64 | PASS |  |
| same shape (2,3) | (2, 3) | (2, 3) | float16 | PASS |  |
| same shape (2,3) | (2, 3) | (2, 3) | float32 | PASS |  |
| same shape (2,3) | (2, 3) | (2, 3) | float64 | PASS |  |
| same shape (2,3) | (2, 3) | (2, 3) | int8 | PASS |  |
| same shape (2,3) | (2, 3) | (2, 3) | int16 | PASS |  |
| same shape (2,3) | (2, 3) | (2, 3) | int32 | PASS |  |
| same shape (2,3) | (2, 3) | (2, 3) | int64 | PASS |  |
| same shape (2,3) | (2, 3) | (2, 3) | uint8 | PASS |  |
| same shape (2,3) | (2, 3) | (2, 3) | uint16 | PASS |  |
| same shape (2,3) | (2, 3) | (2, 3) | uint32 | PASS |  |
| same shape (2,3) | (2, 3) | (2, 3) | uint64 | PASS |  |
| broadcast dim 1: (2,3)-(1,3) | (2, 3) | (1, 3) | float16 | PASS |  |
| broadcast dim 1: (2,3)-(1,3) | (2, 3) | (1, 3) | float32 | PASS |  |
| broadcast dim 1: (2,3)-(1,3) | (2, 3) | (1, 3) | float64 | PASS |  |
| broadcast dim 1: (2,3)-(1,3) | (2, 3) | (1, 3) | int8 | PASS |  |
| broadcast dim 1: (2,3)-(1,3) | (2, 3) | (1, 3) | int16 | PASS |  |
| broadcast dim 1: (2,3)-(1,3) | (2, 3) | (1, 3) | int32 | PASS |  |
| broadcast dim 1: (2,3)-(1,3) | (2, 3) | (1, 3) | int64 | PASS |  |
| broadcast dim 1: (2,3)-(1,3) | (2, 3) | (1, 3) | uint8 | PASS |  |
| broadcast dim 1: (2,3)-(1,3) | (2, 3) | (1, 3) | uint16 | PASS |  |
| broadcast dim 1: (2,3)-(1,3) | (2, 3) | (1, 3) | uint32 | PASS |  |
| broadcast dim 1: (2,3)-(1,3) | (2, 3) | (1, 3) | uint64 | PASS |  |
| smaller rank, no leading dims: (2,3,4)-(4,) | (2, 3, 4) | (4,) | float16 | PASS |  |
| smaller rank, no leading dims: (2,3,4)-(4,) | (2, 3, 4) | (4,) | float32 | PASS |  |
| smaller rank, no leading dims: (2,3,4)-(4,) | (2, 3, 4) | (4,) | float64 | PASS |  |
| smaller rank, no leading dims: (2,3,4)-(4,) | (2, 3, 4) | (4,) | int8 | PASS |  |
| smaller rank, no leading dims: (2,3,4)-(4,) | (2, 3, 4) | (4,) | int16 | PASS |  |
| smaller rank, no leading dims: (2,3,4)-(4,) | (2, 3, 4) | (4,) | int32 | PASS |  |
| smaller rank, no leading dims: (2,3,4)-(4,) | (2, 3, 4) | (4,) | int64 | PASS |  |
| smaller rank, no leading dims: (2,3,4)-(4,) | (2, 3, 4) | (4,) | uint8 | PASS |  |
| smaller rank, no leading dims: (2,3,4)-(4,) | (2, 3, 4) | (4,) | uint16 | PASS |  |
| smaller rank, no leading dims: (2,3,4)-(4,) | (2, 3, 4) | (4,) | uint32 | PASS |  |
| smaller rank, no leading dims: (2,3,4)-(4,) | (2, 3, 4) | (4,) | uint64 | PASS |  |
| broadcast dim 1: (2,3,4)-(3,1) | (2, 3, 4) | (3, 1) | float16 | PASS |  |
| broadcast dim 1: (2,3,4)-(3,1) | (2, 3, 4) | (3, 1) | float32 | PASS |  |
| broadcast dim 1: (2,3,4)-(3,1) | (2, 3, 4) | (3, 1) | float64 | PASS |  |
| broadcast dim 1: (2,3,4)-(3,1) | (2, 3, 4) | (3, 1) | int8 | PASS |  |
| broadcast dim 1: (2,3,4)-(3,1) | (2, 3, 4) | (3, 1) | int16 | PASS |  |
| broadcast dim 1: (2,3,4)-(3,1) | (2, 3, 4) | (3, 1) | int32 | PASS |  |
| broadcast dim 1: (2,3,4)-(3,1) | (2, 3, 4) | (3, 1) | int64 | PASS |  |
| broadcast dim 1: (2,3,4)-(3,1) | (2, 3, 4) | (3, 1) | uint8 | PASS |  |
| broadcast dim 1: (2,3,4)-(3,1) | (2, 3, 4) | (3, 1) | uint16 | PASS |  |
| broadcast dim 1: (2,3,4)-(3,1) | (2, 3, 4) | (3, 1) | uint32 | PASS |  |
| broadcast dim 1: (2,3,4)-(3,1) | (2, 3, 4) | (3, 1) | uint64 | PASS |  |
| rank-0 on the left: ()-(2,3) | () | (2, 3) | float16 | PASS |  |
| rank-0 on the left: ()-(2,3) | () | (2, 3) | float32 | PASS |  |
| rank-0 on the left: ()-(2,3) | () | (2, 3) | float64 | PASS |  |
| rank-0 on the left: ()-(2,3) | () | (2, 3) | int8 | PASS |  |
| rank-0 on the left: ()-(2,3) | () | (2, 3) | int16 | PASS |  |
| rank-0 on the left: ()-(2,3) | () | (2, 3) | int32 | PASS |  |
| rank-0 on the left: ()-(2,3) | () | (2, 3) | int64 | PASS |  |
| rank-0 on the left: ()-(2,3) | () | (2, 3) | uint8 | PASS |  |
| rank-0 on the left: ()-(2,3) | () | (2, 3) | uint16 | PASS |  |
| rank-0 on the left: ()-(2,3) | () | (2, 3) | uint32 | PASS |  |
| rank-0 on the left: ()-(2,3) | () | (2, 3) | uint64 | PASS |  |
| rank-0 on the right: (2,3)-() | (2, 3) | () | float16 | PASS |  |
| rank-0 on the right: (2,3)-() | (2, 3) | () | float32 | PASS |  |
| rank-0 on the right: (2,3)-() | (2, 3) | () | float64 | PASS |  |
| rank-0 on the right: (2,3)-() | (2, 3) | () | int8 | PASS |  |
| rank-0 on the right: (2,3)-() | (2, 3) | () | int16 | PASS |  |
| rank-0 on the right: (2,3)-() | (2, 3) | () | int32 | PASS |  |
| rank-0 on the right: (2,3)-() | (2, 3) | () | int64 | PASS |  |
| rank-0 on the right: (2,3)-() | (2, 3) | () | uint8 | PASS |  |
| rank-0 on the right: (2,3)-() | (2, 3) | () | uint16 | PASS |  |
| rank-0 on the right: (2,3)-() | (2, 3) | () | uint32 | PASS |  |
| rank-0 on the right: (2,3)-() | (2, 3) | () | uint64 | PASS |  |
| both rank-0: ()-() | () | () | float16 | PASS |  |
| both rank-0: ()-() | () | () | float32 | PASS |  |
| both rank-0: ()-() | () | () | float64 | PASS |  |
| both rank-0: ()-() | () | () | int8 | PASS |  |
| both rank-0: ()-() | () | () | int16 | PASS |  |
| both rank-0: ()-() | () | () | int32 | PASS |  |
| both rank-0: ()-() | () | () | int64 | PASS |  |
| both rank-0: ()-() | () | () | uint8 | PASS |  |
| both rank-0: ()-() | () | () | uint16 | PASS |  |
| both rank-0: ()-() | () | () | uint32 | PASS |  |
| both rank-0: ()-() | () | () | uint64 | PASS |  |
| zero-sized: (0,3)-(1,3) | (0, 3) | (1, 3) | float16 | PASS |  |
| zero-sized: (0,3)-(1,3) | (0, 3) | (1, 3) | float32 | PASS |  |
| zero-sized: (0,3)-(1,3) | (0, 3) | (1, 3) | float64 | PASS |  |
| zero-sized: (0,3)-(1,3) | (0, 3) | (1, 3) | int8 | PASS |  |
| zero-sized: (0,3)-(1,3) | (0, 3) | (1, 3) | int16 | PASS |  |
| zero-sized: (0,3)-(1,3) | (0, 3) | (1, 3) | int32 | PASS |  |
| zero-sized: (0,3)-(1,3) | (0, 3) | (1, 3) | int64 | PASS |  |
| zero-sized: (0,3)-(1,3) | (0, 3) | (1, 3) | uint8 | PASS |  |
| zero-sized: (0,3)-(1,3) | (0, 3) | (1, 3) | uint16 | PASS |  |
| zero-sized: (0,3)-(1,3) | (0, 3) | (1, 3) | uint32 | PASS |  |
| zero-sized: (0,3)-(1,3) | (0, 3) | (1, 3) | uint64 | PASS |  |
| zero-sized: (0,)-(1,) | (0,) | (1,) | float16 | PASS |  |
| zero-sized: (0,)-(1,) | (0,) | (1,) | float32 | PASS |  |
| zero-sized: (0,)-(1,) | (0,) | (1,) | float64 | PASS |  |
| zero-sized: (0,)-(1,) | (0,) | (1,) | int8 | PASS |  |
| zero-sized: (0,)-(1,) | (0,) | (1,) | int16 | PASS |  |
| zero-sized: (0,)-(1,) | (0,) | (1,) | int32 | PASS |  |
| zero-sized: (0,)-(1,) | (0,) | (1,) | int64 | PASS |  |
| zero-sized: (0,)-(1,) | (0,) | (1,) | uint8 | PASS |  |
| zero-sized: (0,)-(1,) | (0,) | (1,) | uint16 | PASS |  |
| zero-sized: (0,)-(1,) | (0,) | (1,) | uint32 | PASS |  |
| zero-sized: (0,)-(1,) | (0,) | (1,) | uint64 | PASS |  |
| zero-sized: (0,3)-(3,) | (0, 3) | (3,) | float16 | PASS |  |
| zero-sized: (0,3)-(3,) | (0, 3) | (3,) | float32 | PASS |  |
| zero-sized: (0,3)-(3,) | (0, 3) | (3,) | float64 | PASS |  |
| zero-sized: (0,3)-(3,) | (0, 3) | (3,) | int8 | PASS |  |
| zero-sized: (0,3)-(3,) | (0, 3) | (3,) | int16 | PASS |  |
| zero-sized: (0,3)-(3,) | (0, 3) | (3,) | int32 | PASS |  |
| zero-sized: (0,3)-(3,) | (0, 3) | (3,) | int64 | PASS |  |
| zero-sized: (0,3)-(3,) | (0, 3) | (3,) | uint8 | PASS |  |
| zero-sized: (0,3)-(3,) | (0, 3) | (3,) | uint16 | PASS |  |
| zero-sized: (0,3)-(3,) | (0, 3) | (3,) | uint32 | PASS |  |
| zero-sized: (0,3)-(3,) | (0, 3) | (3,) | uint64 | PASS |  |
| zero-sized: (1,0)-(2,0) | (1, 0) | (2, 0) | float16 | PASS |  |
| zero-sized: (1,0)-(2,0) | (1, 0) | (2, 0) | float32 | PASS |  |
| zero-sized: (1,0)-(2,0) | (1, 0) | (2, 0) | float64 | PASS |  |
| zero-sized: (1,0)-(2,0) | (1, 0) | (2, 0) | int8 | PASS |  |
| zero-sized: (1,0)-(2,0) | (1, 0) | (2, 0) | int16 | PASS |  |
| zero-sized: (1,0)-(2,0) | (1, 0) | (2, 0) | int32 | PASS |  |
| zero-sized: (1,0)-(2,0) | (1, 0) | (2, 0) | int64 | PASS |  |
| zero-sized: (1,0)-(2,0) | (1, 0) | (2, 0) | uint8 | PASS |  |
| zero-sized: (1,0)-(2,0) | (1, 0) | (2, 0) | uint16 | PASS |  |
| zero-sized: (1,0)-(2,0) | (1, 0) | (2, 0) | uint32 | PASS |  |
| zero-sized: (1,0)-(2,0) | (1, 0) | (2, 0) | uint64 | PASS |  |
| zero-sized: (0,3)-(0,1) | (0, 3) | (0, 1) | float16 | PASS |  |
| zero-sized: (0,3)-(0,1) | (0, 3) | (0, 1) | float32 | PASS |  |
| zero-sized: (0,3)-(0,1) | (0, 3) | (0, 1) | float64 | PASS |  |
| zero-sized: (0,3)-(0,1) | (0, 3) | (0, 1) | int8 | PASS |  |
| zero-sized: (0,3)-(0,1) | (0, 3) | (0, 1) | int16 | PASS |  |
| zero-sized: (0,3)-(0,1) | (0, 3) | (0, 1) | int32 | PASS |  |
| zero-sized: (0,3)-(0,1) | (0, 3) | (0, 1) | int64 | PASS |  |
| zero-sized: (0,3)-(0,1) | (0, 3) | (0, 1) | uint8 | PASS |  |
| zero-sized: (0,3)-(0,1) | (0, 3) | (0, 1) | uint16 | PASS |  |
| zero-sized: (0,3)-(0,1) | (0, 3) | (0, 1) | uint32 | PASS |  |
| zero-sized: (0,3)-(0,1) | (0, 3) | (0, 1) | uint64 | PASS |  |
| zero-sized with rank-0: (0,)-() | (0,) | () | float16 | PASS |  |
| zero-sized with rank-0: (0,)-() | (0,) | () | float32 | PASS |  |
| zero-sized with rank-0: (0,)-() | (0,) | () | float64 | PASS |  |
| zero-sized with rank-0: (0,)-() | (0,) | () | int8 | PASS |  |
| zero-sized with rank-0: (0,)-() | (0,) | () | int16 | PASS |  |
| zero-sized with rank-0: (0,)-() | (0,) | () | int32 | PASS |  |
| zero-sized with rank-0: (0,)-() | (0,) | () | int64 | PASS |  |
| zero-sized with rank-0: (0,)-() | (0,) | () | uint8 | PASS |  |
| zero-sized with rank-0: (0,)-() | (0,) | () | uint16 | PASS |  |
| zero-sized with rank-0: (0,)-() | (0,) | () | uint32 | PASS |  |
| zero-sized with rank-0: (0,)-() | (0,) | () | uint64 | PASS |  |

======================================================================
NaN payload difference (not a mismatch): NaN operand, quiet, payload 1
A (float16) = array([nan], dtype=float16)
B (float16) = array([1.], dtype=float16)
spec  = array([nan], dtype=float16)
ORT   = array([nan], dtype=float16)
spec bits = [32257]
ORT  bits = [32256]

======================================================================
NaN payload difference (not a mismatch): NaN operand, quiet, payload 1, sign set
A (float16) = array([nan], dtype=float16)
B (float16) = array([1.], dtype=float16)
spec  = array([nan], dtype=float16)
ORT   = array([nan], dtype=float16)
spec bits = [65025]
ORT  bits = [65024]

======================================================================
NaN payload difference (not a mismatch): NaN operand, signalling
A (float16) = array([nan], dtype=float16)
B (float16) = array([1.], dtype=float16)
spec  = array([nan], dtype=float16)
ORT   = array([nan], dtype=float16)
spec bits = [32257]
ORT  bits = [32256]

real section: 3 examples, 3 exact, 0 wrong
summary: 205 cases run, 205 passed, 0 mismatched, 0 refused by ONNX Runtime
