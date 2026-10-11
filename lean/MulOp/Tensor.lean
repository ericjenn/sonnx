/-
SONNX profile, operator **Mul** — formalization in Lean 4.

This file provides the tensor model on which the element-wise specification
of the operator is stated:

* `MulOp.Tensor`      — a dense row-major tensor (shape + flat data),
* `MulOp.validIdxB`   — validity of a tensor index for a given shape,
* `MulOp.offset`      — the flat row-major offset of an index,
* `MulOp.Tensor.read` — element access, returning `none` for indices that are
  not valid (or for a malformed tensor whose data does not match its shape).

The shape/index pair plays the role of the [tensor index] definition referenced
by the operator specification (`../common/definitions.md#tensor_index`): an
index of a tensor of shape `s` is a list `i` with `i.length = s.length` such
that `i[k] < s[k]` for every `k`.
-/
import Init

namespace MulOp

/-- The (flat) number of elements of a tensor of shape `s`, that is, the
product of its dimension sizes. -/
def size : List Nat → Nat
  | [] => 1
  | d :: ds => d * size ds

/-- `validIdxB s i` is true when `i` is a valid index of a tensor of shape
`s`: it has as many entries as `s` has dimensions, and each entry is strictly
below the size of the corresponding dimension. -/
def validIdxB : List Nat → List Nat → Bool
  | [], [] => true
  | [], _ :: _ => false
  | _ :: _, [] => false
  | d :: ds, x :: xs => decide (x < d) && validIdxB ds xs

/-- In-bounds array reads. -/
theorem getElem!_eq_getElem_of_lt {α : Type} [Inhabited α] {l : List α} {i : Nat}
    (h : i < l.length) : l[i]! = l[i] := by
  rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem h, Option.getD_some]

/-- Pointwise characterization of `validIdxB`: `i` is a valid index of shape
`s` exactly when the lengths agree and every entry is in range. -/
theorem validIdxB_iff : ∀ (s i : List Nat),
    validIdxB s i = true ↔ s.length = i.length ∧ ∀ j < s.length, i[j]! < s[j]! := by
  intro s
  induction s with
  | nil =>
    intro i
    cases i with
    | nil => simp [validIdxB]
    | cons x xs =>
      constructor
      · intro h; simp [validIdxB] at h
      · intro h; simp at h
  | cons d ds ih =>
    intro i
    cases i with
    | nil =>
      constructor
      · intro h; simp [validIdxB] at h
      · intro h; simp at h
    | cons x xs =>
      constructor
      · intro h
        rw [validIdxB, Bool.and_eq_true, decide_eq_true_eq] at h
        rw [ih] at h
        have hl : (d :: ds).length = (x :: xs).length := by
          have := h.2.1; simp at this ⊢; omega
        refine ⟨hl, ?_⟩
        intro j hj
        cases j with
        | zero => simpa using h.1
        | succ j =>
          have hj' := h.2.2 j (by simp at hj; omega)
          simpa using hj'
      · intro h
        have hx : x < d := by
          have := h.2 0 (by simp)
          simpa using this
        simp only [validIdxB, Bool.and_eq_true, decide_eq_true_eq]
        refine ⟨hx, ?_⟩
        rw [ih]
        have hl : ds.length = xs.length := by
          have := h.1; simp at this ⊢; omega
        refine ⟨hl, ?_⟩
        intro j hj
        have hj' := h.2 (j + 1) (by simp; omega)
        simpa using hj'

/-- The flat row-major offset of the index `i` within a tensor of shape `s`.
The first dimension varies slowest and the last dimension varies fastest, as
in the C row-major layout used by the ONNX data model. -/
def offset : List Nat → List Nat → Nat
  | [], _ => 0
  | _ :: _, [] => 0
  | _ :: ds, x :: xs => x * size ds + offset ds xs

/-- The offset of a valid index is strictly below the number of elements. -/
theorem offset_lt_size : ∀ (s i : List Nat), validIdxB s i = true → offset s i < size s := by
  intro s
  induction s with
  | nil =>
    intro i h
    cases i with
    | nil => simp [offset, size]
    | cons x xs => simp [validIdxB] at h
  | cons d ds ih =>
    intro i h
    cases i with
    | nil => simp [validIdxB] at h
    | cons x xs =>
      rw [validIdxB, Bool.and_eq_true, decide_eq_true_eq] at h
      have hx : x < d := h.1
      have hv : validIdxB ds xs = true := h.2
      have hlt : offset ds xs < size ds := ih xs hv
      simp only [offset, size]
      have h1 : x * size ds + offset ds xs < x * size ds + size ds := by omega
      have h2 : x * size ds + size ds = (x + 1) * size ds := by rw [Nat.succ_mul]
      have h3 : size ds * (x + 1) ≤ size ds * d :=
        Nat.mul_le_mul_left (size ds) (by omega)
      have h3' : (x + 1) * size ds ≤ d * size ds := by
        rw [Nat.mul_comm (x + 1) (size ds), Nat.mul_comm d (size ds)]
        exact h3
      omega

/-- A dense row-major tensor: a shape and the flat data laid out row-major. -/
structure Tensor (α : Type) where
  /-- The shape, as the list of the dimension sizes. -/
  shape : List Nat
  /-- The flat data, in row-major order (see `MulOp.offset`). -/
  data : List α

instance [Inhabited α] : Inhabited (Tensor α) := ⟨⟨[], []⟩⟩

/-- A tensor is **well-formed** when its data holds exactly as many values as
its shape indicates.  All the tensors built by the operator below are
well-formed; well-formedness is what lets `MulOp.Tensor.read` succeed on
valid indices. -/
def Tensor.wellFormed (t : Tensor α) : Prop :=
  t.data.length = size t.shape

/-- Read the element of `t` at index `i`; the result is `none` when `i` is not
a valid index of `t`, or when `t` is malformed. -/
def Tensor.read (t : Tensor α) (i : List Nat) : Option α :=
  if validIdxB t.shape i then t.data[offset t.shape i]? else none

/-- A valid index into a well-formed tensor reads the element at its offset. -/
theorem Tensor.read_eq_some_self {α : Type} [Inhabited α] {t : Tensor α} {i : List Nat}
    (h : validIdxB t.shape i = true) (hw : t.wellFormed) :
    t.read i = some (t.data[offset t.shape i]!) := by
  have hb : offset t.shape i < t.data.length := by
    rw [hw]; exact offset_lt_size t.shape i h
  simp only [Tensor.read, if_pos h]
  rw [List.getElem?_eq_getElem hb, getElem!_eq_getElem_of_lt hb]

end MulOp