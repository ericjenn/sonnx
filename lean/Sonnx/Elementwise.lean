/-
The SONNX shared Lean library — elementwise binary operations.

An elementwise binary operator combines the two operands' elements at each
index of the broadcast result.  This module provides both ways of stating such
an operator:

* the **functional** form — a function `f : α → α → α` applied pointwise to the
  broadcast elements (`Sonnx.binary`), with its shape, well-formedness and
  element-correctness theorems;
* the **relational** form — a relation `R : α → α → α → Prop` between the two
  broadcast elements and the result element (`Sonnx.PointwiseBinaryRel`,
  `Sonnx.ElementwiseBinaryRel`), for operations whose element step is
  specified as a relation rather than a function (floating-point rounding, for
  instance).

Both forms read the operand elements at the translated index
`Sonnx.bcastIdx`, and an operand element that does not exist (a zero-sized
dimension) reads as the default value in the functional form.
-/
import Sonnx.Broadcast
import Sonnx.Index

namespace Sonnx

/-! ### The functional form -/

/-- The value of the operator at broadcast index `i`: `f` applied to the
elements that `tA` and `tB` contribute at `i` (the specification's
$\tilde A[i]$ and $\tilde B[i]$), where an element that does not exist reads
as the default value. -/
def binaryAt {α : Type} [Inhabited α] (f : α → α → α) (tA tB : Tensor α)
    (i : List Nat) : α :=
  f ((tA.read (bcastIdx i tA.shape)).getD default)
    ((tB.read (bcastIdx i tB.shape)).getD default)

/-- The data of the operator on a broadcast shape `sc`: the elements of
`indices sc` in row-major order, holding the results. -/
def binaryData {α : Type} [Inhabited α] (f : α → α → α) (tA tB : Tensor α)
    (sc : List Nat) : Tensor α :=
  ⟨sc, (indices sc).map (binaryAt f tA tB)⟩

/-- The elementwise binary operator: `none` when the operand shapes are not
broadcast-compatible, otherwise the tensor of their broadcast shape holding the
pointwise results. -/
def binary {α : Type} [Inhabited α] (f : α → α → α) (tA tB : Tensor α) :
    Option (Tensor α) :=
  (bcastShape tA.shape tB.shape).map (binaryData f tA tB)

theorem binaryData_shape {α : Type} [Inhabited α] {f : α → α → α}
    {tA tB : Tensor α} {sc : List Nat} :
    (binaryData f tA tB sc).shape = sc := rfl

/-- The result tensor is well-formed. -/
theorem binaryData_wf {α : Type} [Inhabited α] {f : α → α → α}
    {tA tB : Tensor α} {sc : List Nat} :
    (binaryData f tA tB sc).wellFormed := by
  simp only [binaryData, Tensor.wellFormed, List.length_map, indices_length]

/-- The shape and well-formedness part of the shape-definition constraint: a
successful `binary` returns a well-formed tensor whose shape is the broadcast
shape of the operand shapes. -/
theorem binary_shape_wf {α : Type} [Inhabited α] {f : α → α → α}
    {tA tB : Tensor α} {c : Tensor α} (h : binary f tA tB = some c) :
    ∃ sc, bcastShape tA.shape tB.shape = some sc ∧ c.shape = sc ∧ c.wellFormed := by
  simp only [binary, Option.map_eq_some_iff] at h
  obtain ⟨sc, hsc, hc⟩ := h
  subst hc
  exact ⟨sc, hsc, rfl, binaryData_wf⟩

/-- The elementwise specification: at a valid index `i` of the broadcast shape,
the result tensor holds `f` applied to the elements of the operands that `i`
points to (the broadcast elements $\tilde A[i]$ and $\tilde B[i]$, here read at
the translated indices `bcastIdx i tA.shape` and `bcastIdx i tB.shape`). -/
theorem binaryData_read_correct {α : Type} [Inhabited α] {f : α → α → α}
    {tA tB : Tensor α} {sc i : List Nat}
    (hwA : tA.wellFormed) (hwB : tB.wellFormed)
    (h : bcastShape tA.shape tB.shape = some sc) (hi : validIdxB sc i = true) :
    (binaryData f tA tB sc).read i =
      some (f (tA.data[offset tA.shape (bcastIdx i tA.shape)]!)
        (tB.data[offset tB.shape (bcastIdx i tB.shape)]!)) := by
  have hB : Broadcasts tA.shape sc ∧ Broadcasts tB.shape sc :=
    bcastShape_broadcasts h
  have hvA : validIdxB tA.shape (bcastIdx i tA.shape) = true :=
    validIdxB_bcastIdx hB.1 hi
  have hvB : validIdxB tB.shape (bcastIdx i tB.shape) = true :=
    validIdxB_bcastIdx hB.2 hi
  simp only [binaryData]
  show (if validIdxB sc i then ((indices sc).map (binaryAt f tA tB))[offset sc i]? else none)
      = some (f (tA.data[offset tA.shape (bcastIdx i tA.shape)]!)
        (tB.data[offset tB.shape (bcastIdx i tB.shape)]!))
  rw [if_pos hi, List.getElem?_map, indices_get_offset sc i hi]
  simp [binaryAt, Tensor.read_eq_some_self hvA hwA, Tensor.read_eq_some_self hvB hwB]

/-- `binary` fails exactly when the shapes are not broadcast-compatible
(the shape-compatibility constraint). -/
theorem binary_none {α : Type} [Inhabited α] {f : α → α → α} {tA tB : Tensor α}
    (h : bcastShape tA.shape tB.shape = none) : binary f tA tB = none := by
  simp [binary, h]

/-! ### The relational form -/

/-- At every valid index of `C`, the operand elements that the index selects and
the element of `C` are related by `R` — the elementwise formula of an operator
whose element step is a relation. -/
def PointwiseBinaryRel {α : Type} (R : α → α → α → Prop) (A B C : Tensor α) : Prop :=
  ∀ i, validIdxB C.shape i = true →
    ∀ a b c, A.read (bcastIdx i A.shape) = some a →
      B.read (bcastIdx i B.shape) = some b →
      C.read i = some c → R a b c

/-- The elementwise specification of a relational binary operator: the shape of
`C` is the broadcast shape of `A` and `B`, and the element relation `R` holds
at every index of `C`. -/
def ElementwiseBinaryRel {α : Type} (R : α → α → α → Prop) (A B C : Tensor α) : Prop :=
  BroadcastShapeIs A B C ∧ PointwiseBinaryRel R A B C

theorem ElementwiseBinaryRel.shape {α : Type} {R : α → α → α → Prop}
    {A B C : Tensor α} (h : ElementwiseBinaryRel R A B C) : BroadcastShapeIs A B C :=
  h.1

theorem ElementwiseBinaryRel.pointwise {α : Type} {R : α → α → α → Prop}
    {A B C : Tensor α} (h : ElementwiseBinaryRel R A B C) : PointwiseBinaryRel R A B C :=
  h.2

end Sonnx
