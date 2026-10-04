# Sub (opset 14): specification vs ONNX Runtime

This is the verification of `ops/sub.md` (revision 2026-10-03) by implementing the
operator from that document alone (`impl_from_spec.py`) and comparing the result
with ONNX Runtime 1.30.0 (`test_vs_ort.py`, a one-node `Sub` model, opset 14,
`CPUExecutionProvider`). No other document and no ONNX Runtime source was read
for semantics.

The implementation follows the specification, including where it disagrees with
ONNX Runtime. Nothing was adapted to ONNX Runtime and no test was weakened.

## Reproduce

```
cd /run/media/eric/Work1/Work/gits/sonnx/verification/sub
/home/eric/Venvs/onnxcheck/bin/python test_vs_ort.py              # compact table; exit 1 on a mismatch
/home/eric/Venvs/onnxcheck/bin/python test_vs_ort.py --markdown   # the table reproduced below
```

## Summary

**181 cases run, 178 passed, 3 mismatched, 0 refused by ONNX Runtime.**

| dimension | coverage |
| --------- | -------- |
| types | `float16`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`; float64 also as the stand-in for the `real` section (ONNX Sub has no such type) |
| shapes | identical; broadcasting with a dimension of size 1; `(2,3,4)-(4,)`; `(2,3,4)-(3,1)`; rank-0 on the left, on the right, and on both sides; zero-sized `(0,3)-(1,3)`, `(0,)-(1,)`, `(0,3)-(3,)`, `(1,0)-(2,0)`, `(0,3)-(0,1)`, `(0,)-()` |
| values | ordinary; the extremes of every integer type (`min`, `max`, `-1`, `0`, `1`) and the pairs whose difference wraps; for every float type the four combinations of `+0`/`-0`, `+inf`, `-inf`, NaN, an exact difference, a difference that overflows to infinity, a difference that lands in the subnormal range, a difference that is not representable (`0.3-0.1`), an exact rounding tie, and a difference in the band just above `max` (divergence 1) |

ONNX Runtime accepted every type and every shape, including `uint64`, rank-0
inputs and zero-sized dimensions: 0 refusals.

Comparison is bit-exact: float results are compared as raw bit patterns
(`view` on an unsigned integer of the same width) with NaN == NaN; integer
results by value. A differing NaN payload is reported separately and is not
counted as a mismatch.

## Divergences

### 1. A difference just above the largest finite value: the specification says ±inf, ONNX Runtime rounds to the largest finite value

**Specification side: incorrect.** Two statements of the document conflict: the
one that declares IEEE 754 semantics and the one that defines overflow on the
*exact* difference. In the band `(max, max + ulp(max)/2)` the document requires
`±inf` and IEEE 754 (hence ONNX Runtime) requires `max`.

Minimal reproducing input, float16 (smallest case to write down):

```python
A = np.array([65504.0], dtype=np.float16)    # np.finfo(np.float16).max
B = np.array([-8.0], dtype=np.float16)
```

The exact difference is `65504 - (-8) = 65512`, exactly, as a real. Then

```
impl_from_spec.sub(A, B)      -> array([inf], dtype=float16)      bits 0x7c00
onnxruntime (Sub-14)          -> array([65504.0], dtype=float16)  bits 0x7bff
plain numpy a - b             -> array([65504.0], dtype=float16)  bits 0x7bff
```

The same divergence in the two wider types:

```python
# float32
A = np.array([3.4028235e38], dtype=np.float32)    # np.finfo(np.float32).max
B = np.array([-5.0706024e+30], dtype=np.float32)  # -2**102
# exact difference = 3.4028235170913126e+38  >  max = 3.4028234663852886e+38
# spec -> inf (0x7f800000) ; ORT -> 3.4028235e38 (0x7f7fffff)

# float64
A = np.array([1.7976931348623157e+308])           # np.finfo(np.float64).max
B = np.array([-4.989600773836841e+291])           # -2**969
# exact difference = max + 2**969  >  max
# spec -> inf (0x7ff0000000000000) ; ORT -> 1.7976931348623157e+308 (0x7fefffffffffffff)
```

For float64 the exact difference is not representable as a single float64, so
the harness's implementation carries it as the exact pair `s + e` (see the
implementation note below); `s + e = max + 2**969` is strictly greater than
`max`, which is what makes case 3 apply.

**Why the implementation returns ±inf.** The exact difference `65512` is not
representable in `float16`, so case 2 does not apply, and `65512 > 65504 = max`,
so the third case applies. The error-condition table of the same section fixes
that reading beyond doubt.

**Why ONNX Runtime returns the largest finite value.** IEEE 754 does not call
this an overflow. Under round-to-nearest, the result of a subtraction overflows
only when the exact difference reaches the midpoint between the largest finite
value and `2**emax`, i.e. `max + ulp(max)/2`. Here `ulp(65504) = 32`, the
midpoint is `65504 + 16 = 65520`, and `65512 < 65520`, so the correctly rounded
result is `65504` and no overflow occurs. Every float type has such a band
`(max, max + ulp(max)/2)`; the three cases above sit strictly inside it.

The conflicting sentences of `ops/sub.md`, quoted:

> Operator **Sub** subtracts tensor $B$ from tensor $A$ element-wise according to
> IEEE 754 floating-point semantics and stores the result in tensor $C$.

> $\pm\text{inf}$ if the exact difference overflows the range of the type of $C$

> Overflow, i.e., an exact difference outside the range of the type | nominal:
> specified by E_SUB_FLOAT_FUNC_0010, the result is ±inf

The last two are definite statements, and they contradict the first in the band
`(max, max + ulp(max)/2)`: IEEE 754 overflow is a property of the *rounded*
result, not of the exact difference. The wording is also ambiguous — "overflows
the range" could be read in the IEEE sense — but then case 3 and case 4's
"otherwise" overlap. Either way the document needs an amendment; a reader
following it literally implements a result IEEE 754 forbids.

Minimal amendment (the third case of the case analysis and the table row; the
amendment decides the band by falling back to case 4, whose `round(x)` gives
`max`):

> $\pm\text{inf}$ if the nearest representable value of the type of $C$ of the
> exact difference is an infinity

> | Overflow, i.e., an exact difference whose nearest representable value of the
> type of $C$ is an infinity | nominal: specified by
> E_SUB_FLOAT_FUNC_0010, the result is ±inf |

(The existing sentence "In the third case, the sign of $\pm\text{inf}$ is the
sign of the exact difference" stays valid, and the sentence "The rounding may
make the result null or subnormal; its sign is then the sign of the exact
difference" is unaffected.)

**Not affected: the `real` section.** There are no special values and no
overflow in the real numbers, and the float64 stand-in for the real section
rounds the exact real difference `max + 2**969` to `max`, which is what ONNX
Runtime produces: the case `real section, above max below overflow threshold
(float64 stand-in)` is in the table below and passes. The divergence is specific
to case 3 of the float section.

### 2. The NaN produced by the invalid operation `inf - inf` — payload and sign are not specified

**Specification side: incomplete.** The document says the result is a NaN but
never says which NaN, so the two implementations produce different bit patterns
and both conform. This is a case the specification leaves to the implementer.

```
float16: A=[inf] B=[inf]   spec -> NaN 0x7e00      ORT -> NaN 0xfe00
float32: A=[inf] B=[inf]   spec -> NaN 0x7fc00000  ORT -> NaN 0xffc00000
float64: A=[inf] B=[inf]   spec -> NaN 0x7ff8000000000000  ORT -> NaN 0xfff8000000000000
```

The closest passages of `ops/sub.md`:

> $\text{NaN}$ & $\text{if } \tilde{A}[i] \text{ or } \tilde{B}[i] \text{ is
> NaN, or if they are both } +\infty \text{ or both } -\infty$

> In the first case, an operand that is NaN is propagated, and the subtraction
> of two infinities of the same sign is the invalid operation defined in IEEE
> 754 section 7.2; the result is NaN in both cases.

What is missing: the payload and the sign bit of the NaN produced by the invalid
operation (and, symmetrically, whether a propagated NaN operand keeps its
payload). The document fixes only the property "is NaN". Both bit patterns are
quiet NaNs, so the comparison treats the case as PASS and lists the payload
difference separately.

Minimal amendment (a sentence to add after the "In the first case ..." sentence):

> A NaN that results from an operation that has no NaN operand is a quiet NaN;
> its payload and its sign are not specified, and every quiet NaN of the type is
> a conforming result.

This is not an ONNX Runtime deviation: ONNX Runtime follows the document; the
document merely does not decide the bit pattern, and an implementer has to.

## Judgement calls (cases where the specification left room, no divergence observed)

These are the places where I had to choose while implementing from the document
alone. They are listed because a case the specification does not decide is a
case two implementations decide differently; the tests showed both
implementations happened to choose the same way.

1. **Which IEEE 754 rounding attribute `round(x)` uses.** Quoted: "$\text{round}(x)$
   is the value of $x$ rounded to the nearest representable value of the type of
   $C$, according to the IEEE 754 rounding rules." — "to the nearest" fixes
   round-to-nearest, but IEEE 754 has two nearest attributes (`roundTiesToEven`
   and `roundTiesToAway`) and the document names neither; the tie rule is also
   what fixes the sign of a null difference. I chose `roundTiesToEven` (the IEEE
   default, and what NumPy and ONNX Runtime use). Checked with an exact tie
   (`A = 1 + 2**-p`, `B = 2**-(p+1)`, whose exact difference `1 + 2**-(p+1)` is
   exactly halfway between `1` and `1 + 2**-p`): both implementations give `1.0`
   for `float16`, `float32` and `float64` (the case `rounding tie (halfway)` in
   the table below passes). Minimal amendment: "...rounded to the nearest
   representable value of the type of $C$ according to the roundTiesToEven
   attribute of IEEE 754".
2. **The data type of the `real` section.** The section is headed "Sub (real,
   real)" but ONNX Sub has no real type, and the document does not say which
   machine type a checker uses for it nor what the result means when the exact
   real difference is not representable in that type. I used `float64` and took
   its rounding of the exact difference as the representative of the real
   result; that is the only choice that can be compared with an implementation.
   Minimal amendment, if the real section is meant to be checkable: "The real
   type is not a bounded floating-point type; an implementation that represents
   it with a floating-point type returns the exact difference whenever it is
   representable in that type, and that type's rounding of the exact difference
   otherwise."
3. **The payload of a propagated NaN operand** (the other half of divergence 2):
   "an operand that is NaN is propagated" says the NaN-ness is propagated but
   not that the payload is preserved. Both implementations preserve it (the
   `nan` value family matches bit-exactly), so there is no divergence, but the
   document does not require it.

## Cases with no divergence, and what they establish

- **Signed zeros.** For every float type the four combinations of `+0`/`-0`
  matched bit-exactly. The rule "When the exact difference is null, the result
  is `+0`, except when `A[i]` is `-0` and `B[i]` is `+0`, in which case it is
  `-0`" is exactly IEEE 754's sign rule, including `(-0) - (-0) = +0` and
  `(+0) - (-0) = +0`.
- **Underflow into the subnormal range** (`tiny - tiny/2` and `0 - (-tiny/2)`)
  matched for all three float types. The document's "the rounding may make the
  result null" cannot happen for a non-null exact difference: every value of a
  float type is a multiple of that type's smallest subnormal, so a non-null
  difference of two of them is at least one smallest subnormal and rounds to a
  non-null value. The rounded-to-null-with-a-sign clause is therefore vacuous,
  and harmless.
- **Overflow to infinity** (`max - (-max)`, `inf - finite`, `inf - (-inf)`)
  matched for all three float types.
- **Integers.** All eight types matched, including `int8` `127 - (-1) = -128`,
  `-128 - 1 = 127`, and `uint64` `0 - 1 = 2**64 - 1`. NumPy's integer
  wrap-around is exactly the modulo `2**n` of `E_SUB_INT_FUNC_0010`, and the
  implementation computes the exact difference in Python integers before
  representing the residue class in the type's range, as the three cases of the
  definition say.
- **Broadcasting, rank-0 and zero-sized dimensions.** ONNX Runtime accepted and
  matched every one, including `(2,3,4)` with `(4,)` and with `(3,1)`, a rank-0
  operand on either side and on both sides, and every zero-sized shape tested.

## Implementation notes

- `impl_from_spec.sub` is the ONNX operator; the `real` section, which is not an
  ONNX type, is the entry point `impl_from_spec.sub_real`, implemented as
  float64 as the only way to check it (both documented in the module
  docstring).
- For the float section the exact difference is carried as the float64 pair
  `s + e` produced by an error-free transformation, so the three cases of the
  definition are decided on the exact difference itself, for `float64` too (no
  wider hardware type exists). This is not a change of semantics: the first
  version decided the cases from a float64 intermediate, which is not exact for
  float64 operands, and the definition is stated on "the exact difference"
  (`E_SUB_FLOAT_FUNC_0010`) — the transformation is only the means of computing
  it. The result of every case except case 3 is unchanged (verified against
  plain NumPy arithmetic on 6000 random float trials and on all integer types,
  with the single intended case-3 deviation).
- The model IR version is pinned to 10: onnx 1.23 emits IR version 14 by
  default, which ONNX Runtime 1.30.0 rejects ("Unsupported model IR version:
  14, max supported IR version: 13"). This is a harness detail, not a finding
  about the operator.

## Table

| case | shape A | shape B | dtype | verdict | note |
| ---- | ------- | ------- | ----- | ------- | ---- |
| signed zeros | (4,) | (4,) | float16 | PASS |  |
| infinities | (6,) | (6,) | float16 | PASS | NaN payload differs |
| nan | (4,) | (4,) | float16 | PASS |  |
| exact difference | (4,) | (4,) | float16 | PASS |  |
| overflow to inf | (2,) | (2,) | float16 | PASS |  |
| above max, below overflow threshold | (1,) | (1,) | float16 | MISMATCH |  |
| subnormal result | (3,) | (3,) | float16 | PASS |  |
| rounding tie (halfway) | (2,) | (2,) | float16 | PASS |  |
| not representable (0.3-0.1) | (2,) | (2,) | float16 | PASS |  |
| signed zeros | (4,) | (4,) | float32 | PASS |  |
| infinities | (6,) | (6,) | float32 | PASS | NaN payload differs |
| nan | (4,) | (4,) | float32 | PASS |  |
| exact difference | (4,) | (4,) | float32 | PASS |  |
| overflow to inf | (2,) | (2,) | float32 | PASS |  |
| above max, below overflow threshold | (1,) | (1,) | float32 | MISMATCH |  |
| subnormal result | (3,) | (3,) | float32 | PASS |  |
| rounding tie (halfway) | (2,) | (2,) | float32 | PASS |  |
| not representable (0.3-0.1) | (2,) | (2,) | float32 | PASS |  |
| signed zeros | (4,) | (4,) | float64 | PASS |  |
| infinities | (6,) | (6,) | float64 | PASS | NaN payload differs |
| nan | (4,) | (4,) | float64 | PASS |  |
| exact difference | (4,) | (4,) | float64 | PASS |  |
| overflow to inf | (2,) | (2,) | float64 | PASS |  |
| above max, below overflow threshold | (1,) | (1,) | float64 | MISMATCH |  |
| subnormal result | (3,) | (3,) | float64 | PASS |  |
| rounding tie (halfway) | (2,) | (2,) | float64 | PASS |  |
| not representable (0.3-0.1) | (2,) | (2,) | float64 | PASS |  |
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

