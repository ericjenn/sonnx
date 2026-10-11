/-
SONNX profile, operator **Mul** — the tensor-level examples of the
specification.

The element-level behaviour of the four type families is established in
`MulOp.RealTypes` (the exact-real family, `E_MUL_REAL_FUNC_0010`),
`MulOp.FloatTypes` (the IEEE 754 binary64 family, `E_MUL_FLOAT_FUNC_0010`)
and `MulOp.IntTypes` (the `n`-bit integer families,
`E_MUL_INT/UINT_FUNC_0010`).  This file runs the generic operator
`MulOp.mul` on the operand tensors of the specification's examples and proves
that it computes the expected result tensors — thereby exercising the shape
machinery: rank‑0 handling, Numpy-style multidirectional broadcasting
(`E_MUL_*_CONSTR_A_0010`, `E_MUL_*_CONSTR_C_0010`), zero-sized dimensions and
the failure on incompatible shapes.

* `real_example_1`  — elementwise, no broadcasting (`Example 1` of `real`);
* `real_example_2`  — a `3 × 2` matrix by a rank-1 tensor, broadcast
  (`Example 2` of `real`);
* `uint8_example`   — the wrap-around products (`Example 1` of `uint`);
* `int8_example`    — the two's-complement products (`Example 1` of `int`);
* `zero_sized`      — a broadcast of two zero-sized dimensions;
* `incompatible_none` — `mul` fails where the shapes are not
  broadcast-compatible (`E_MUL_*_CONSTR_A_0010`).

The tensors hold elements as values of the modeled types — rationals for the
exact reals, the `n`-bit patterns for the integer families — and the proofs
evaluate the compiled operator (`native_decide`).
-/
import MulOp.FloatTypes
import MulOp.IntTypes
import MulOp.RealTypes

namespace MulOp

/-- Decidability of tensor equality, decided componentwise. -/
instance {α : Type} [DecidableEq α] : DecidableEq (Tensor α) := fun a b =>
  if h1 : Tensor.shape a = Tensor.shape b then
    if h2 : Tensor.data a = Tensor.data b then
      isTrue (by rw [show a = Tensor.mk (Tensor.shape a) (Tensor.data a) from rfl, h1, h2])
    else isFalse (fun hh => h2 (by rw [hh]))
  else isFalse (fun hh => h1 (by rw [hh]))

/-! ### The exact-real examples -/

/-- Real Example 1: `[6.1, 9.5, 35.7] · [2, 3, 4] = [12.2, 28.5, 142.8]`,
exactly — two tensors of the same shape, no broadcasting. -/
theorem real_example_1 :
    mul (⟨[3], [(6.1 : Rat), 9.5, 35.7]⟩ : Tensor Rat)
        (⟨[3], [2, 3, 4]⟩ : Tensor Rat) =
      some (⟨[3], [(12.2 : Rat), 28.5, 142.8]⟩ : Tensor Rat) := by native_decide

/-- Real Example 2: the `3 × 2` matrix times the vector `[2, 10]`, the
operand of shape `[2]` broadcast to shape `[3, 2]` — each column of the
matrix is multiplied by the corresponding element of the vector. -/
theorem real_example_2 :
    mul (⟨[3, 2], [(3.7 : Rat), 4.4, 16.2, 0.5, 25.3, 24.8]⟩ : Tensor Rat)
        (⟨[2], [2, 10]⟩ : Tensor Rat) =
      some (⟨[3, 2], [(7.4 : Rat), 44.0, 32.4, 5.0, 50.6, 248.0]⟩ : Tensor Rat) := by
  native_decide

/-! ### The `uint8` example -/

/-- The operands of the specification's `uint8` example:
`[6, 200, 20, 16] · [3, 200, 13, 16]`. -/
def uexA : Tensor (UIntN 8) :=
  ⟨[4], [(⟨6, by decide⟩ : UIntN 8), ⟨200, by decide⟩, ⟨20, by decide⟩, ⟨16, by decide⟩]⟩

def uexB : Tensor (UIntN 8) :=
  ⟨[4], [(⟨3, by decide⟩ : UIntN 8), ⟨200, by decide⟩, ⟨13, by decide⟩, ⟨16, by decide⟩]⟩

/-- `E_MUL_UINT_FUNC_0010` on the `uint8` example: the product is
`[18, 64, 4, 0]` — `200 · 200 = 40000` reduces to `64` (more than one
application of the reduction), and `16 · 16 = 0`. -/
theorem uint8_example :
    mul uexA uexB =
      some (⟨[4], [(⟨18, by decide⟩ : UIntN 8), ⟨64, by decide⟩, ⟨4, by decide⟩,
        ⟨0, by decide⟩]⟩ : Tensor (UIntN 8)) := by native_decide

/-! ### The `int8` example -/

/-- The operands of the specification's `int8` example, as bit patterns:
`−6` is the pattern `250 = 2^8 − 6` and `−3` the pattern `253`. -/
def iexA : Tensor (SIntN 8) :=
  ⟨[4], [(⟨250, by decide⟩ : SIntN 8), ⟨11, by decide⟩, ⟨100, by decide⟩, ⟨16, by decide⟩]⟩

def iexB : Tensor (SIntN 8) :=
  ⟨[4], [(⟨253, by decide⟩ : SIntN 8), ⟨13, by decide⟩, ⟨100, by decide⟩, ⟨16, by decide⟩]⟩

/-- `E_MUL_INT_FUNC_0010` on the `int8` example: the product is the pattern
`[18, 143, 16, 0]`, which the two's-complement reading `SIntN.toInt` turns
into the values `[18, −113, 16, 0]` — `11 · 13 = 143` exceeds the maximum
value `127` of the type and the result is the negative `−113 = 143 − 2^8`. -/
theorem int8_example :
    mul iexA iexB =
      some (⟨[4], [(⟨18, by decide⟩ : SIntN 8), ⟨143, by decide⟩, ⟨16, by decide⟩,
        ⟨0, by decide⟩]⟩ : Tensor (SIntN 8)) := by native_decide

/-! ### Shapes and failures -/

/-- Two zero-sized dimensions broadcast to a zero-sized dimension: the
product tensor is empty — the specification handles the elements of a
dimension of size zero by having none. -/
theorem zero_sized :
    mul (⟨[0], [(0 : Rat)]⟩ : Tensor Rat) (⟨[0], [(0 : Rat)]⟩ : Tensor Rat) =
      some (⟨[0], []⟩ : Tensor Rat) := by native_decide

/-- Shapes not broadcast-compatible — here the dimensions `3` and `5` —
make the operator fail (`E_MUL_*_CONSTR_A_0010`). -/
theorem incompatible_none :
    mul (⟨[2, 3], [(0 : Rat), 0, 0, 0, 0, 0]⟩ : Tensor Rat)
        (⟨[5], [0, 0, 0, 0, 0]⟩ : Tensor Rat) = none := by native_decide

end MulOp