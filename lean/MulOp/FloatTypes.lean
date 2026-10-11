/-
SONNX profile, operator **Mul** — the floating-point type family.

The `E_MUL_FLOAT_FUNC_0010` specification describes the multiplication of the
floating-point types as IEEE 754 multiplication: a `NaN` operand or the product
of a zero by an infinity produces a quiet `NaN` (an invalid operation); every
other pair of values produces the exact product when it is representable, and
otherwise the exact product rounded with rounding mode roundTiesToEven; the
sign of the result is the sign of the exact product, including for the zeros.

Lean's `Float` **is** the IEEE 754 binary64 type: its multiplication instance
multipplies the doubles with rounding mode roundTiesToEven and IEEE error
handling, so the generic operator `MulOp.mul` instantiated at `Float` computes
the specification's case table. This file records the observable behaviour of
the case table on the values the specification discusses, evaluating the same
IEEE arithmetic the runtime performs (`native_decide`):

* `NaN` operands — `FloatTypes.mul_nan_left/mul_nan_right/mul_nan_nan`:
  the product is `NaN` whatever the other operand.
* zero by an infinity — `FloatTypes.mul_zero_inf`: a quiet `NaN`, the
  invalid operation; an infinity by a non-zero finite value is a genuine
  infinity of the sign of the exact product (`FloatTypes.mul_inf_sign`).
* exact products — `FloatTypes.mul_exact`: `2.0 · 3.0 = 6.0` exactly.
* rounding — `FloatTypes.mul_rounds`: `0.1 · 0.2 = 0.020000000000000004`, the
  double nearest the exact product 0.0200000000000000041633…, and
  `FloatTypes.mul_overflow` — `1.0e308 · 1.0e308 = +∞`, whose exact product
  exceeds the (unbounded in the specification, finite in binary64) exponent
  range and overflows, with the sign of the exact product.
* underflow — `FloatTypes.mul_subnormal`:
  `2.2250738585072014e-308 · 0.25 = 5.562684646268003e-309`, a subnormal
  result; `FloatTypes.mul_underflow_zero`: a product too small for a subnormal
  rounds to a zero.
* the sign of zeros — `FloatTypes.mul_zero_sign_left/right` and
  `FloatTypes.mul_zero_sign_both`: the product of zeros carries the sign of
  the exact product (IEEE 754 §6.3): `+0 · (−5.0) = −0`,
  `(−0) · (−5.0) = +0`, `(−0) · (−0) = +0`, `(+0) · (+0) = +0`.

The payload and sign bit of a produced `NaN` are unspecified in the
specification, and there is nothing to record about them beyond the fact that
the result is a `NaN`.
-/

import MulOp.MulCore

namespace MulOp

/-- The quiet `NaN` obtained for instance by dividing zero by zero. -/
def NaN : Float := 0.0 / 0.0

/-- `+∞`, obtained by dividing `1.0` by `0.0`. -/
def Inf : Float := 1.0 / 0.0

/-! ### NaN operands (`E_MUL_FLOAT_FUNC_0010`, case 1)

A `NaN` operand produces a quiet `NaN` product, whatever the other operand
(including a zero, an infinity or another `NaN`); the multiplication is then
not an error in IEEE 754, the `NaN` being propagated. -/

theorem mul_nan_left : (NaN * 5.0).isNaN = true := by native_decide

theorem mul_nan_right : (5.0 * NaN).isNaN = true := by native_decide

theorem mul_nan_nan : (NaN * NaN).isNaN = true := by native_decide

theorem mul_nan_inf : (NaN * Inf).isNaN = true := by native_decide

theorem mul_nan_zero : (NaN * 0.0).isNaN = true := by native_decide

/-! ### Zero by an infinity (`E_MUL_FLOAT_FUNC_0010`, case 1)

The product of a zero by an infinity is the invalid operation, and produces a
quiet `NaN`.

By contrast, an infinity by a non-zero finite value is a genuine infinity with
the sign of the exact product. -/

theorem mul_zero_inf : (0.0 * Inf).isNaN = true := by native_decide

theorem mul_negzero_inf : ((-0.0 : Float) * Inf).isNaN = true := by native_decide

/-- `−∞ · (−2.0) = +∞`: the sign of the result is the sign of the exact
product. -/
theorem mul_inf_sign_right : 0.0 < ((-2.0 : Float) * (-Inf)) := by native_decide

/-- `2.0 · (−∞) = −∞`. -/
theorem mul_inf_sign_left : (((2.0 : Float) * (-Inf)) < 0.0) = true := by
  native_decide

/-! ### Exact products and rounding (`E_MUL_FLOAT_FUNC_0010`, case 2)

The product of two non-zero finite values is the exact product when it is
representable, and otherwise the exact product rounded with rounding mode
roundTiesToEven. The mathematical product is computed as if with an unbounded
exponent range (the specification's hypothesis), then the correctly rounded
result is committed: an exact product above the exponent range overflows to
`±∞`, below it underflows to a subnormal or a zero. -/

/-- An exactly representable product. -/
theorem mul_exact : ((2.0 : Float) * 3.0 == 6.0) = true := by native_decide

theorem mul_exact_neg : (((-2.0 : Float) * 3.0 == (-6.0 : Float))) = true := by
  native_decide

/-- The product is rounded: the exact product of the doubles nearest `0.1` and
`0.2` is not representable and the result is the double
`0.020000000000000004` nearest it, not the double nearest `0.02`. -/
theorem mul_rounds : (((0.1 : Float) * 0.2) == 0.020000000000000004) = true := by
  native_decide

/-- Overflow to `+∞`, with the sign of the exact product: `1.0e308² = +∞`. -/
theorem mul_overflow :
    ((1.0e308 : Float) * 1.0e308).isInf = true ∧ 0.0 < (1.0e308 : Float) * 1.0e308 := by
  native_decide

/-- Overflow to `−∞`: `1.0e308 · (−1.0e308) = −∞`. -/
theorem mul_overflow_neg :
    ((1.0e308 : Float) * (-1.0e308)).isInf = true ∧ ((1.0e308 : Float) * (-1.0e308)) < 0.0 := by
  native_decide

/-- Underflow to a subnormal: the smallest normal multiplied by a quarter is a
subnormal result. -/
theorem mul_subnormal :
    (((2.2250738585072014e-308 : Float) * 0.25) == 5.562684646268003e-309) = true := by
  native_decide

/-- Underflow to a zero: the exact product `1.0e-308² = 1.0e-616` is below the
smallest representable double and the result is `+0`. -/
theorem mul_underflow_zero :
    ((1.0e-308 : Float) * (1.0e-308 : Float)).toBits = (0.0 : Float).toBits := by
  native_decide

/-! ### The sign of the zeros (`E_MUL_FLOAT_FUNC_0010`, cases 2 and 3)

The product of two zeros carries the sign of the exact product (IEEE 754
§6.3), and the product of a zero by a non-zero finite value carries the sign
of the exact product. The bits identify the signed zeros. -/

/-- `+0 · (−5.0) = −0`. -/
theorem mul_zero_sign_left :
    ((0.0 : Float) * (-5.0 : Float)).toBits = (0x8000000000000000 : UInt64) := by
  native_decide

/-- `(−0) · (−5.0) = +0`. -/
theorem mul_zero_sign_right :
    (((-0.0 : Float)) * (-5.0 : Float)).toBits = (0 : UInt64) := by native_decide

/-- `(−0) · (−0) = +0`. -/
theorem mul_zero_sign_both_pos :
    (((-0.0 : Float)) * (-0.0 : Float)).toBits = (0 : UInt64) := by native_decide

/-- `+0 · (+0) = +0`. -/
theorem mul_zero_sign_both_zero :
    ((0.0 : Float) * (0.0 : Float)).toBits = (0 : UInt64) := by native_decide

/-- `(−0) · (+0) = −0`. -/
theorem mul_zero_sign_both_neg :
    (((-0.0 : Float)) * (0.0 : Float)).toBits = (0x8000000000000000 : UInt64) := by
  native_decide

/-- `+0 · finite` carries the negative sign of the exact product. -/
theorem mul_zero_finite :
    ((0.0 : Float) * (-5.0 : Float)).toBits = ((-0.0 : Float)).toBits ∧
    (((-0.0 : Float)) * (5.0 : Float)).toBits = ((-0.0 : Float)).toBits ∧
    ((0.0 : Float) * (5.0 : Float)).toBits = ((0.0 : Float)).toBits := by
  native_decide

/-! ### The tensor-level Example 2 of the specification

The specification's second float example runs the case table on a tensor:
`[1.0, +inf, -inf, +0, -0, -0] · [1.0, +0, -2.0, -3.0, +5.0, -0]` is expected
to be `[1.0, NaN, +inf, -0, -0, +0]`.  The generic operator `MulOp.mul`
instantiated at `Float` computes it — `native_decide` evaluates the IEEE
multiplication of the compiled code, and the products are identified by the
bits (`NaN`s cannot be compared by equality, their payload being
unspecified). -/

/-- The first operand of the specification's float Example 2. -/
def fex2A : Tensor Float :=
  ⟨[6], [1.0, Inf, (-Inf), 0.0, (-0.0), (-0.0)]⟩

/-- The second operand of the specification's float Example 2. -/
def fex2B : Tensor Float :=
  ⟨[6], [1.0, 0.0, (-2.0), (-3.0), 5.0, (-0.0)]⟩

/-- `E_MUL_FLOAT_FUNC_0010` on the specification's float Example 2: `mul`
succeeds and the product tensor holds at each of the six indices the element
of the expected product tensor.  The `getD` fallbacks (`2.0`, `1.0`) fail the
assertion should the element be missing. -/
theorem float_example_2 :
    (mul fex2A fex2B).isSome = true ∧
    (((mul fex2A fex2B).getD default).read [0]).getD 2.0 == 1.0 ∧
    ((((mul fex2A fex2B).getD default).read [1]).getD 2.0).isNaN = true ∧
    ((((mul fex2A fex2B).getD default).read [2]).getD 2.0).isInf = true ∧
    0.0 < (((mul fex2A fex2B).getD default).read [2]).getD 2.0 ∧
    (((((mul fex2A fex2B).getD default).read [3]).getD 1.0)).toBits = 0x8000000000000000 ∧
    (((((mul fex2A fex2B).getD default).read [4]).getD 1.0)).toBits = 0x8000000000000000 ∧
    (((((mul fex2A fex2B).getD default).read [5]).getD 1.0)).toBits = (0 : UInt64) := by
  native_decide

end MulOp