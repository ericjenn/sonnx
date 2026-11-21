/-
The SONNX shared Lean library — general lemmas and predicates.

This module holds the list lemmas used by the tensor layers and the
type-consistency predicate that the operator constraints refer to.
-/
import Init

namespace Sonnx

/-! ### List lemmas -/

/-- In-bounds array reads. -/
theorem getElem!_eq_getElem_of_lt {α : Type} [Inhabited α] {l : List α} {i : Nat}
    (h : i < l.length) : l[i]! = l[i] := by
  rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem h, Option.getD_some]

/-- In-range `?`-lookups hold the `!`-element. -/
theorem getElem?_eq_some_get {α : Type} [Inhabited α] {l : List α} {i : Nat}
    (h : i < l.length) : l[i]? = some l[i]! := by
  rw [List.getElem?_eq_getElem h, getElem!_eq_getElem_of_lt h]

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

/-! ### Consistency of the values of a statement -/

/-- Three values of one statement are **type-consistent** when they are equal:
the constraint of an operator whose inputs and output share one type. -/
def TypeConsistent {α : Type} (a b c : α) : Prop :=
  a = b ∧ b = c

theorem TypeConsistent.self {α : Type} (a : α) : TypeConsistent a a a := ⟨rfl, rfl⟩

theorem TypeConsistent.eq_left {α : Type} {a b c : α} (h : TypeConsistent a b c) : a = b :=
  h.1

theorem TypeConsistent.eq_right {α : Type} {a b c : α} (h : TypeConsistent a b c) : b = c :=
  h.2

end Sonnx
