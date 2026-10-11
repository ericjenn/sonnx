/-
SONNX profile, operator **Mul** — the exact-real type family.

The `E_MUL_REAL_FUNC_0010` specification states the product of the "float"
family of exact real values: `C[i] = Ã[i] · B̃[i]` exactly, in `ℝ` — no
rounding, no infinities, no special values.

This file models that family with the rational numbers `Rat`: the values of
the type are exact rationals, a subring of `ℝ`, and the multiplication is the
exact product (a quotient product).  Every value and product of the examples
of the specification is rational; the exactness of the operator on this type
is exactly the elementwise formula `E_MUL_REAL_FUNC_0010`, which
`MulOp.MulCore.mulData_read_correct` states for the generic operator
`MulOp.mul`: the product tensor holds, at each valid index `i`, exactly the
product of the two elements `i` points to — here, an exact rational product.

The first example of the specification is reproduced at the element level
(`RealTypes.example_1_*`); the tensor-level statement is in `MulOp.Examples`.
-/

import MulOp.MulCore

namespace MulOp

/-! ### The exact examples of the specification -/

/-- Real Example 1, first element: `6.1 · 2 = 12.2`, exactly. -/
theorem example_1_0 : ((6.1 : Rat) * 2) = 12.2 := by native_decide

/-- Real Example 1, second element: `9.5 · 3 = 28.5`, exactly. -/
theorem example_1_1 : ((9.5 : Rat) * 3) = 28.5 := by native_decide

/-- Real Example 1, third element: `35.7 · 4 = 142.8`, exactly. -/
theorem example_1_2 : ((35.7 : Rat) * 4) = 142.8 := by native_decide

/-- The exactness of the family: the product of two values of the type is
again a value of the type, computed without ever rounding — the difference
with `E_MUL_FLOAT_FUNC_0010`, whose product may round, overflow to `±∞` or
underflow. -/
theorem example_1 : ((6.1 : Rat) * 2, (9.5 : Rat) * 3, (35.7 : Rat) * 4) =
    (12.2, 28.5, 142.8) := by native_decide

end MulOp