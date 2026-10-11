/-
SONNX profile, operator **Mul** — broadcasting machinery.

This file formalizes the multidirectional (Numpy-style) broadcasting used by
the operator:

* `MulOp.padTo`    — left-padding of a shape with dimensions of size one,
* `MulOp.bcastDim` — compatibility of two aligned dimension sizes,
* `MulOp.bcastShape` — the shape of `A` and `B` broadcast to a common shape
  (`E_MUL_REAL_CONSTR_A_0010`, `E_MUL_REAL_CONSTR_C_0010`, `…_FLOAT_…`,
  `…_INT_…`, `…_UINT_…`),
* `MulOp.Broadcasts` — the predicate “`sc` is a broadcast target of `s`”,
* `MulOp.bcastIdx` — translation of an index of the broadcast shape into an
  index of one operand (the broadcast element $\tilde A[i]$ of the
  specification), and the proof that it is a valid index of the operand.
-/
import Init
import MulOp.Tensor

namespace MulOp

/-! ### Shape padding -/

/-- Pad a shape on the left with dimensions of size one so as to align two
shapes on their last dimension. -/
def padTo (k : Nat) (s : List Nat) : List Nat :=
  List.replicate (k - s.length) 1 ++ s

theorem padTo_length {s : List Nat} {k : Nat} (h : s.length ≤ k) :
    (padTo k s).length = k := by
  simp only [padTo, List.length_append, List.length_replicate, Nat.sub_add_cancel h]

theorem padTo_get_low {s : List Nat} {k : Nat} {_h : s.length ≤ k} {j : Nat}
    (hj : j < k - s.length) : (padTo k s)[j]! = 1 := by
  have h1 : (padTo k s)[j]? = some 1 := by
    simp only [padTo, List.length_replicate, List.getElem?_append, if_pos hj,
      List.getElem?_replicate_of_lt hj]
  rw [List.getElem!_eq_getElem?_getD, h1, Option.getD_some]

theorem padTo_get_high {s : List Nat} {k : Nat} (h : s.length ≤ k) {j : Nat}
    (hj1 : k - s.length ≤ j) (hj2 : j < k) :
    (padTo k s)[j]! = s[j - (k - s.length)]! := by
  have h2 : j - (k - s.length) < s.length := by omega
  have h1 : (padTo k s)[j]? = s[j - (k - s.length)]? := by
    simp only [padTo, List.length_replicate, List.getElem?_append,
      if_neg (by omega : ¬ (j : Nat) < k - s.length)]
  rw [List.getElem!_eq_getElem?_getD, h1, List.getElem?_eq_getElem h2,
    getElem!_eq_getElem_of_lt h2, Option.getD_some]

/-- In-range `?`-lookups hold the `!`-element. -/
theorem getElem?_eq_some_get {α : Type} [Inhabited α] {l : List α} {i : Nat}
    (h : i < l.length) : l[i]? = some l[i]! := by
  rw [List.getElem?_eq_getElem h, getElem!_eq_getElem_of_lt h]

/-! ### Dimension compatibility -/

/-- Compatibility of two aligned dimension sizes, with the size the broadcast
dimension takes: equal sizes, and a size `1` (or a missing dimension, which
pads to `1`) extends to the other size; in particular a zero-sized dimension is
compatible with a dimension of size `1` and the broadcast dimension then has
size `0`. -/
def bcastDim (a b : Nat) : Option Nat :=
  if a = b then some a else if a = 1 then some b else if b = 1 then some a else none

/-- Case analysis on a compatible pair of aligned dimension sizes. -/
theorem bcastDim_cases {a b c : Nat} (h : bcastDim a b = some c) :
    (a = b ∧ c = a) ∨ (a ≠ b ∧ a = 1 ∧ c = b) ∨ (a ≠ b ∧ a ≠ 1 ∧ b = 1 ∧ c = a) := by
  simp only [bcastDim] at h
  split at h
  · exact Or.inl ⟨by omega, (Option.some.inj h).symm⟩
  · split at h
    · exact Or.inr (Or.inl ⟨by omega, by omega, (Option.some.inj h).symm⟩)
    · split at h
      · exact Or.inr (Or.inr ⟨by omega, by omega, by omega, (Option.some.inj h).symm⟩)
      · exact absurd h (by simp)

/-- Broadcasting never produces a dimension of size `1` from a pair of sizes
both different from `1`. -/
theorem bcastDim_ne_one {a b c : Nat} (h1 : a ≠ 1) (h2 : b ≠ 1)
    (h : bcastDim a b = some c) : c ≠ 1 := by
  rcases bcastDim_cases h with ⟨_, hc⟩ | ⟨_, _, hc⟩ | ⟨_, _, _, hc⟩
  · rw [hc]; exact h1
  · rw [hc]; exact h2
  · rw [hc]; exact h1

/-! ### The broadcast shape -/

/-- Turn a list of optional values into an optional list: `none` as soon as
one entry is `none`. -/
def allSome : List (Option α) → Option (List α)
  | [] => some []
  | some a :: xs => (allSome xs).map fun l => a :: l
  | none :: _ => none

/-- Characterization of `allSome`: `allSome os = some m` exactly when `m` has
the same length as `os` and the `j`-th entry of `m` is the content of the
`j`-th entry of `os`. -/
theorem allSome_iff [Inhabited α] {os : List (Option α)} {m : List α} :
    allSome os = some m ↔ os.length = m.length ∧ ∀ j < m.length, os[j]! = some m[j]! := by
  induction os generalizing m with
  | nil =>
    cases m with
    | nil =>
      constructor
      · intro _; exact ⟨rfl, by intro j hj; simp at hj⟩
      · intro _; rfl
    | cons a t =>
      constructor
      · intro h; simp [allSome] at h
      · intro h; simp at h
  | cons o t ih =>
    cases o with
    | none =>
      cases m with
      | nil =>
        constructor
        · intro h; simp [allSome] at h
        · intro h; simp at h
      | cons b mt =>
        constructor
        · intro h; simp [allSome] at h
        · rintro ⟨_, H⟩
          have h0 := H 0 (by simp)
          simp at h0
    | some a =>
      cases m with
      | nil =>
        constructor
        · intro h; simp [allSome, Option.map_eq_some_iff] at h
        · intro h; simp at h
      | cons b mt =>
        constructor
        · intro h
          simp only [allSome, Option.map_eq_some_iff] at h
          obtain ⟨y, hy1, hy2⟩ := h
          simp at hy2
          obtain ⟨rfl, rfl⟩ := hy2
          have hlen := ih.mp hy1
          refine ⟨by simp [hlen.1], ?_⟩
          intro j hj
          cases j with
          | zero => simp
          | succ j => simpa using hlen.2 j (by simpa using hj)
        · rintro ⟨he, H⟩
          have h0 := H 0 (by simp)
          simp at h0
          subst h0
          have hlen : t.length = mt.length := by
            have := he; simp at this ⊢; omega
          have ht : allSome t = some mt := by
            refine ih.mpr ⟨hlen, ?_⟩
            intro j hj
            have := H (j + 1) (by simpa using hj)
            simpa using this
          rw [show allSome (some a :: t) = (allSome t).map (fun l => a :: l) from rfl, ht]
          simp

theorem allSome_length [Inhabited α] {os : List (Option α)} {m : List α}
    (h : allSome os = some m) : os.length = m.length := (allSome_iff.mp h).1

theorem allSome_get [Inhabited α] {os : List (Option α)} {m : List α}
    (h : allSome os = some m) {j : Nat} (hj : j < m.length) :
    os[j]! = some m[j]! := (allSome_iff.mp h).2 j hj

/-! ### The broadcast predicate and index translation -/

/-- `sc` is a broadcast target of the shape `s`: it extends `s`, and every
entry of `s` either agrees with the corresponding right-aligned entry of `sc`
or is `1`.  This is the shape-compatibility condition of
`E_MUL_*_CONSTR_A_0010` for the broadcast of two tensors onto `sc`. -/
def Broadcasts (s sc : List Nat) : Prop :=
  s.length ≤ sc.length ∧
  ∀ j < s.length, s[j]! = sc[j + (sc.length - s.length)]! ∨ s[j]! = 1

/-- Translate an index `i` of a broadcast shape into an index of the operand of
shape `s` (the index of the broadcast element $\tilde A[i]$ of the
specification): an entry is `0` where `s` has a padded or size-one dimension,
and is the entry of `i` otherwise. -/
def bcastIdx (i s : List Nat) : List Nat :=
  (i.zipWith (fun x d => if d = 1 then 0 else x) (padTo i.length s)).drop
    (i.length - s.length)

theorem bcastIdx_length {i s : List Nat} (h : s.length ≤ i.length) :
    (bcastIdx i s).length = s.length := by
  simp only [bcastIdx, List.length_drop, List.length_zipWith, padTo_length h,
    Nat.min_self]
  omega

theorem bcastIdx_get {i s : List Nat} (h : s.length ≤ i.length) {j : Nat}
    (hj : j < s.length) :
    (bcastIdx i s)[j]! = (if s[j]! = 1 then 0 else i[j + (i.length - s.length)]!) := by
  have hA : (i.length - s.length) + j < i.length := by omega
  have hB : (i.length - s.length) + j < (padTo i.length s).length := by
    rw [padTo_length h]; omega
  have hp : (padTo i.length s)[(i.length - s.length) + j] = s[j]! := by
    rw [(getElem!_eq_getElem_of_lt hB).symm, padTo_get_high h (by omega) hA,
      show ((i.length - s.length + j) - (i.length - s.length)) = j from by omega]
  have hd : j + (i.length - s.length) = (i.length - s.length) + j := by omega
  rw [bcastIdx, List.getElem!_eq_getElem?_getD, List.getElem?_drop]
  simp only [List.getElem?_zipWith, List.getElem?_eq_getElem hA,
    List.getElem?_eq_getElem hB, hd, (getElem!_eq_getElem_of_lt hA).symm, hp,
    Option.getD_some]

/-- **Key fact**: the index translation maps a valid index of the broadcast
shape to a valid index of the operand. -/
theorem validIdxB_bcastIdx {sc s i : List Nat} (hB : Broadcasts s sc)
    (hi : validIdxB sc i = true) : validIdxB s (bcastIdx i s) = true := by
  rw [validIdxB_iff] at hi
  have hle : s.length ≤ i.length := by have := hB.1; omega
  rw [validIdxB_iff]
  refine ⟨by rw [bcastIdx_length hle], ?_⟩
  intro j hj
  rw [bcastIdx_get hle hj]
  rcases hB.2 j hj with he | h1
  · rw [← hi.1, he]
    split
    · omega
    · exact hi.2 (j + (sc.length - s.length)) (by omega)
  · rw [h1]; simp

/-! ### The broadcast shape -/

/-- Shape of the broadcast of two shapes (Numpy-style, right-aligned; see
`E_MUL_*_CONSTR_A_0010`): `none` when the shapes are not broadcast-compatible. -/
def bcastShape (sA sB : List Nat) : Option (List Nat) :=
  allSome ((padTo (max sA.length sB.length) sA).zipWith bcastDim
    (padTo (max sA.length sB.length) sB))

theorem bcastShape_length {sA sB sc : List Nat} (h : bcastShape sA sB = some sc) :
    sc.length = max sA.length sB.length := by
  have h1 := allSome_length h
  simp only [List.length_zipWith, padTo_length (Nat.le_max_left ..),
    padTo_length (Nat.le_max_right ..), Nat.min_self] at h1
  exact h1.symm

/-- Entrywise description of the broadcast shape: aligned entries are
combined by `bcastDim`. -/
theorem bcastShape_get {sA sB sc : List Nat} (h : bcastShape sA sB = some sc)
    {j : Nat} (hj : j < sc.length) :
    bcastDim (padTo (max sA.length sB.length) sA)[j]!
      (padTo (max sA.length sB.length) sB)[j]! = some sc[j]! := by
  have hkA : sA.length ≤ max sA.length sB.length := Nat.le_max_left ..
  have hkB : sB.length ≤ max sA.length sB.length := Nat.le_max_right ..
  have hlen := bcastShape_length h
  have hjA : j < (padTo (max sA.length sB.length) sA).length := by
    rw [padTo_length hkA]; omega
  have hjB : j < (padTo (max sA.length sB.length) sB).length := by
    rw [padTo_length hkB]; omega
  have hz2 : ((padTo (max sA.length sB.length) sA).zipWith bcastDim
      (padTo (max sA.length sB.length) sB))[j]? =
      some (bcastDim (padTo (max sA.length sB.length) sA)[j]!
        (padTo (max sA.length sB.length) sB)[j]!) := by
    rw [List.getElem?_zipWith, getElem?_eq_some_get hjA, getElem?_eq_some_get hjB]
  have hz3 : ((padTo (max sA.length sB.length) sA).zipWith bcastDim
      (padTo (max sA.length sB.length) sB))[j]! =
      bcastDim (padTo (max sA.length sB.length) sA)[j]!
        (padTo (max sA.length sB.length) sB)[j]! := by
    rw [List.getElem!_eq_getElem?_getD, hz2, Option.getD_some]
  rw [← hz3]
  exact (allSome_iff.mp h).2 j hj

/-- **Shape compatibility**: a successful `bcastShape` yields broadcast target
shapes for both operands. -/
theorem bcastShape_broadcasts {sA sB sc : List Nat} (h : bcastShape sA sB = some sc) :
    Broadcasts sA sc ∧ Broadcasts sB sc := by
  have hkl := bcastShape_length h
  have hak : sA.length ≤ max sA.length sB.length := Nat.le_max_left ..
  have hbk : sB.length ≤ max sA.length sB.length := Nat.le_max_right ..
  constructor
  · constructor
    · omega
    · intro j hj
      have ht : sc.length - sA.length + j < sc.length := by omega
      have hv := bcastShape_get h (hj := ht)
      rw [padTo_get_high hak (by omega) (by omega),
        show (sc.length - sA.length + j) - (max sA.length sB.length - sA.length) = j from by
          omega] at hv
      rcases bcastDim_cases hv with ⟨_, hc⟩ | ⟨_, ha1, _⟩ | ⟨_, _, _, hc⟩
      · rw [show j + (sc.length - sA.length) = sc.length - sA.length + j from by omega]
        exact Or.inl hc.symm
      · exact Or.inr ha1
      · rw [show j + (sc.length - sA.length) = sc.length - sA.length + j from by omega]
        exact Or.inl hc.symm
  · constructor
    · omega
    · intro j hj
      have ht : sc.length - sB.length + j < sc.length := by omega
      have hv := bcastShape_get h (hj := ht)
      rw [padTo_get_high hbk (by omega) (by omega),
        show (sc.length - sB.length + j) - (max sA.length sB.length - sB.length) = j from by
          omega] at hv
      rcases bcastDim_cases hv with ⟨hab, hc⟩ | ⟨_, _, hc⟩ | ⟨_, _, hb1, _⟩
      · rw [show j + (sc.length - sB.length) = sc.length - sB.length + j from by omega]
        exact Or.inl (hc.trans hab).symm
      · rw [show j + (sc.length - sB.length) = sc.length - sB.length + j from by omega]
        exact Or.inl hc.symm
      · exact Or.inr hb1

end MulOp