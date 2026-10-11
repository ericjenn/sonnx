/-
SONNX profile, operator **Mul** — the operator itself.

This file defines the elementwise product of two tensors with Numpy-style
multidirectional broadcasting:

* `MulOp.indices`     — the list of all indices of a shape, in row-major order,
* `MulOp.mulAt`       — the product of the two operands at a broadcast index
  (the elements $\tilde A[i]$ and $\tilde B[i]$ of the specification),
* `MulOp.mulData`     — the data of the product tensor,
* `MulOp.mul`         — the operator: `none` on incompatible shapes
  (`E_MUL_*_CONSTR_A_0010`), otherwise the tensor of broadcast shape `sc`
  holding at index `i` the product of the broadcast elements
  (`E_MUL_*_FUNC_0010`; `E_MUL_*_CONSTR_C_0010`).
-/
import MulOp.Broadcast

namespace MulOp

/-! ### Enumeration of the indices of a shape -/

/-- All indices of a shape `s`, in row-major order (the first dimension
varies slowest). -/
def indices : List Nat → List (List Nat)
  | [] => [[]]
  | d :: ds => (List.range d).flatMap fun k => (indices ds).map fun tl => k :: tl

/-- The `flatMap` over `range n` of `n` lists of a common length `L` has
`n * L` elements. -/
theorem length_flatMap {β : Type} {n L : Nat} {g : Nat → List β}
    (hL : ∀ k < n, (g k).length = L) :
    ((List.range n).flatMap g).length = n * L := by
  induction n with
  | zero => simp
  | succ m ih =>
    rw [List.range_succ, List.flatMap_append, List.length_append,
      ih (fun k hk => hL k (by omega)), List.flatMap_cons, List.flatMap_nil,
      List.append_nil, hL m (by omega)]
    exact (Nat.succ_mul m L).symm

/-- Accessing the `q`-th element of the `x`-th block of a uniform `flatMap`
over `range n`. -/
theorem flatMap_get {β : Type} {n L : Nat} {g : Nat → List β}
    (hL : ∀ k < n, (g k).length = L) {x q : Nat} (hx : x < n) (hq : q < L) :
    ((List.range n).flatMap g)[x * L + q]? = (g x)[q]? := by
  induction n with
  | zero => exact absurd hx (by simp)
  | succ m ih =>
    have hLm : ∀ k < m, (g k).length = L := fun k hk => hL k (by omega)
    have hlen : ((List.range m).flatMap g).length = m * L := length_flatMap hLm
    rw [List.range_succ, List.flatMap_append]
    by_cases hxx : x < m
    · have hle : x * L + L ≤ m * L := by
        have h2 : x * L + L = (x + 1) * L := (Nat.succ_mul x L).symm
        have h3 : (x + 1) * L ≤ m * L :=
          Nat.mul_le_mul_right L (show x + 1 ≤ m by omega)
        omega
      have hl2 : x * L + q < ((List.range m).flatMap g).length := by omega
      rw [List.getElem?_append_left hl2]
      exact ih hLm hxx
    · have hxm : x = m := by omega
      rw [hxm, List.flatMap_cons, List.flatMap_nil, List.append_nil]
      have hr : ((List.range m).flatMap g).length ≤ m * L + q := by omega
      rw [List.getElem?_append_right hr, hlen,
        show m * L + q - m * L = q from by omega]

theorem indices_length : ∀ s : List Nat, (indices s).length = size s := by
  intro s
  induction s with
  | nil => rfl
  | cons d ds ih =>
    rw [indices, length_flatMap (L := size ds) (fun k _ => by
      rw [List.length_map, ih])]
    simp only [size]

/-- Every index produced by `indices` is a valid index of the shape. -/
theorem indices_valid : ∀ (s i : List Nat), i ∈ indices s → validIdxB s i = true := by
  intro s
  induction s with
  | nil =>
    intro i h
    simp only [indices, List.mem_singleton] at h
    subst h
    rfl
  | cons d ds ih =>
    intro i h
    rw [indices, List.mem_flatMap] at h
    obtain ⟨k, hk, hm⟩ := h
    rw [List.mem_map] at hm
    obtain ⟨tl, ht1, ht2⟩ := hm
    subst ht2
    simp only [validIdxB, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨List.mem_range.mp hk, ih tl ht1⟩

/-- **Position of an index in the row-major enumeration**: for a valid index `i`
of a shape `s`, the `offset s i`-th element of `indices s` is `i` itself. -/
theorem indices_get_offset : ∀ (s i : List Nat), validIdxB s i = true →
    (indices s)[offset s i]? = some i := by
  intro s
  induction s with
  | nil =>
    intro i h
    cases i with
    | nil => simp [indices, offset]
    | cons x xs => simp [validIdxB] at h
  | cons d ds ih =>
    intro i h
    cases i with
    | nil => simp [validIdxB] at h
    | cons x xs =>
      rw [validIdxB, Bool.and_eq_true, decide_eq_true_eq] at h
      have hx : x < d := h.1
      have hxs : validIdxB ds xs = true := h.2
      have hq : offset ds xs < size ds := offset_lt_size ds xs hxs
      rw [indices, offset]
      rw [flatMap_get (L := size ds)
        (fun k _ => by rw [List.length_map, indices_length ds]) hx hq,
        List.getElem?_map, ih xs hxs]
      simp

/-! ### The operator -/

/-- The elementwise product at broadcast index `i` (the specification's
$\tilde A[i] \cdot \tilde B[i]$): the product of the elements that `tA` and
`tB` contribute at `i`, where a broadcast element that does not exist (zero
sized dimensions) reads as the default value. -/
def mulAt {α : Type} [Mul α] [Inhabited α] (tA tB : Tensor α) (i : List Nat) : α :=
  (tA.read (bcastIdx i tA.shape)).getD default *
    (tB.read (bcastIdx i tB.shape)).getD default

/-- The data of the product on a broadcast shape `sc`: the elements of
`indices sc` in row-major order, holding the products. -/
def mulData {α : Type} [Mul α] [Inhabited α] (tA tB : Tensor α) (sc : List Nat) : Tensor α :=
  ⟨sc, (indices sc).map (mulAt tA tB)⟩

/-- The `Mul` operator (`E_MUL_*_FUNC_0010`, `E_MUL_*_CONSTR_A_0010`): `none`
when the operand shapes are not broadcast-compatible, otherwise the product
tensor. -/
def mul {α : Type} [Mul α] [Inhabited α] (tA tB : Tensor α) : Option (Tensor α) :=
  (bcastShape tA.shape tB.shape).map (mulData tA tB)

theorem mulData_shape {α : Type} [Mul α] [Inhabited α] {tA tB : Tensor α} {sc : List Nat} :
    (mulData tA tB sc).shape = sc := rfl

/-- The product tensor is well-formed (`E_MUL_*_CONSTR_C_0010`). -/
theorem mulData_wf {α : Type} [Mul α] [Inhabited α] {tA tB : Tensor α} {sc : List Nat} :
    (mulData tA tB sc).wellFormed := by
  simp only [mulData, Tensor.wellFormed, List.length_map, indices_length]

/-- The shape and well-formedness part of `E_MUL_*_CONSTR_C_0010`: a successful
`mul` returns a well-formed tensor whose shape is the broadcast shape of the
operand shapes. -/
theorem mul_shape_wf {α : Type} [Mul α] [Inhabited α] {tA tB : Tensor α}
    {c : Tensor α} (h : mul tA tB = some c) :
    ∃ sc, bcastShape tA.shape tB.shape = some sc ∧ c.shape = sc ∧ c.wellFormed := by
  simp only [mul, Option.map_eq_some_iff] at h
  obtain ⟨sc, hsc, hc⟩ := h
  subst hc
  exact ⟨sc, hsc, rfl, mulData_wf⟩

/-- The elementwise specification `E_MUL_*_FUNC_0010`: at a valid index `i` of
the broadcast shape, the product tensor holds the product of the elements of
the operands that `i` points to (the broadcast elements
$\tilde A[i]$ and $\tilde B[i]$, here read at the translated indices
`bcastIdx i tA.shape` and `bcastIdx i tB.shape`). -/
theorem mulData_read_correct {α : Type} [Mul α] [Inhabited α]
    {tA tB : Tensor α} {sc i : List Nat}
    (hwA : tA.wellFormed) (hwB : tB.wellFormed)
    (h : bcastShape tA.shape tB.shape = some sc) (hi : validIdxB sc i = true) :
    (mulData tA tB sc).read i =
      some (tA.data[offset tA.shape (bcastIdx i tA.shape)]! *
        tB.data[offset tB.shape (bcastIdx i tB.shape)]!) := by
  have hB : Broadcasts tA.shape sc ∧ Broadcasts tB.shape sc :=
    bcastShape_broadcasts h
  have hvA : validIdxB tA.shape (bcastIdx i tA.shape) = true :=
    validIdxB_bcastIdx hB.1 hi
  have hvB : validIdxB tB.shape (bcastIdx i tB.shape) = true :=
    validIdxB_bcastIdx hB.2 hi
  simp only [mulData]
  show (if validIdxB sc i then ((indices sc).map (mulAt tA tB))[offset sc i]? else none)
      = some (tA.data[offset tA.shape (bcastIdx i tA.shape)]! *
        tB.data[offset tB.shape (bcastIdx i tB.shape)]!)
  rw [if_pos hi, List.getElem?_map, indices_get_offset sc i hi]
  simp [mulAt, Tensor.read_eq_some_self hvA hwA, Tensor.read_eq_some_self hvB hwB]

/-- `mul` fails exactly when the shapes are not broadcast-compatible
(`E_MUL_*_CONSTR_A_0010`). -/
theorem mul_none {α : Type} [Mul α] [Inhabited α] {tA tB : Tensor α}
    (h : bcastShape tA.shape tB.shape = none) : mul tA tB = none := by
  simp [mul, h]

end MulOp