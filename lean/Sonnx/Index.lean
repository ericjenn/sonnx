/-
The SONNX shared Lean library — the enumeration of the indices of a shape.

This module lists the indices of a shape in row-major order, which is the order
the data of a dense tensor is laid out in:

* `Sonnx.indices`            — all indices of a shape, in row-major order,
* `Sonnx.indices_length`     — there are `Sonnx.size s` of them,
* `Sonnx.indices_valid`      — each of them is a valid index of the shape,
* `Sonnx.indices_get_offset` — the index at position `Sonnx.offset s i` is `i`.
-/
import Sonnx.Tensor

namespace Sonnx

/-- All indices of a shape `s`, in row-major order (the first dimension
varies slowest). -/
def indices : List Nat → List (List Nat)
  | [] => [[]]
  | d :: ds => (List.range d).flatMap fun k => (indices ds).map fun tl => k :: tl

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

end Sonnx
