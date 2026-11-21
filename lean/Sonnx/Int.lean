/-
The SONNX shared Lean library — the integer type families.

The integer operator specifications give the operation of the signed and
unsigned `n`-bit types: the result is the value of the type congruent to the
exact integer result modulo `2 ^ n`, and the specification notes that the
values of the type are `2 ^ n` distinct integers no two of which are congruent
modulo `2 ^ n`, so that exactly one value of the type satisfies the
congruence.

This module provides the vocabulary for both ways of stating such a rule:

* `Sonnx.CongM`          — congruence modulo `2 ^ n`, and its algebra,
* `Sonnx.intModulus`, `Sonnx.intMin`, `Sonnx.intMax` — the arithmetic bounds of
  the type of width `n`,
* `Sonnx.InSignedRange`, `Sonnx.InUnsignedRange` — the value ranges,
* `Sonnx.uintWrap`, `Sonnx.sintWrap` — the value of the type congruent to an
  exact integer (`wrap` reduces by as many multiples of `2 ^ n` as needed), with
  range, congruence and uniqueness lemmas,
* `Sonnx.UIntN`, `Sonnx.SIntN` — the types as bit patterns, with
  `Sonnx.SIntN.toInt` the two's-complement reading of a pattern, and
  `Sonnx.UIntN.wrap`, `Sonnx.SIntN.wrap` to build a pattern from an integer.

The ONNX integer types (`int8`, …, `uint64`) are the instances
`n = 8, 16, 32, 64` of these families.
-/
import Sonnx.Prelude

namespace Sonnx

variable {n : Nat}

/-! ### Congruence modulo `2 ^ n` -/

/-- `CongM x y` — the integers `x` and `y` are congruent modulo `2 ^ n`. -/
def CongM (n : Nat) (x y : Int) : Prop :=
  x % ((2 ^ n : Nat) : Int) = y % ((2 ^ n : Nat) : Int)

theorem CongM.refl {x : Int} : CongM n x x := rfl

theorem CongM.symm {x y : Int} (h : CongM n x y) : CongM n y x := Eq.symm h

theorem CongM.trans {x y z : Int} (h1 : CongM n x y) (h2 : CongM n y z) :
    CongM n x z := Eq.trans h1 h2

/-- Congruence modulo `2 ^ n` is stable under multiplication: if `x ≡ y` and
`x' ≡ y'`, then `x * x' ≡ y * y'`. -/
theorem CongM.mul {x y x' y' : Int} (hx : CongM n x y) (hx' : CongM n x' y') :
    CongM n (x * x') (y * y') := by
  have hx1 : x % ((2 ^ n : Nat) : Int) = y % ((2 ^ n : Nat) : Int) := hx
  have hx1' : x' % ((2 ^ n : Nat) : Int) = y' % ((2 ^ n : Nat) : Int) := hx'
  show (x * x') % ((2 ^ n : Nat) : Int) = (y * y') % ((2 ^ n : Nat) : Int)
  rw [Int.mul_emod, hx1, hx1', ← Int.mul_emod]

/-- Congruence modulo `2 ^ n` is stable under addition. -/
theorem CongM.add {x y x' y' : Int} (hx : CongM n x y) (hx' : CongM n x' y') :
    CongM n (x + x') (y + y') := by
  have hx1 : x % ((2 ^ n : Nat) : Int) = y % ((2 ^ n : Nat) : Int) := hx
  have hx1' : x' % ((2 ^ n : Nat) : Int) = y' % ((2 ^ n : Nat) : Int) := hx'
  show (x + x') % ((2 ^ n : Nat) : Int) = (y + y') % ((2 ^ n : Nat) : Int)
  rw [Int.add_emod, hx1, hx1', ← Int.add_emod]

/-- A congruence is the equality up to a multiple of `2 ^ n`: the exact
result reduced by as many multiples of `2 ^ n` as needed. -/
theorem CongM.add_congruent {x y : Int} (h : CongM n x y) :
    ∃ k : Int, x = y + k * ((2 ^ n : Nat) : Int) := by
  simp only [CongM] at h
  rw [Int.emod_def, Int.emod_def] at h
  refine ⟨x / ((2 ^ n : Nat) : Int) - y / ((2 ^ n : Nat) : Int), ?_⟩
  rw [Int.sub_mul, Int.mul_comm (x / ((2 ^ n : Nat) : Int)) ((2 ^ n : Nat) : Int),
    Int.mul_comm (y / ((2 ^ n : Nat) : Int)) ((2 ^ n : Nat) : Int)]
  omega

/-- The values of an `n`-bit type with `n ≥ 1` are `2 ^ n` distinct integers
no two of which are congruent modulo `2 ^ n`: two integers of the canonical
range `[-2^(n-1), 2^(n-1))` congruent modulo `2 ^ n` are equal. -/
theorem congruent_range_unique (hn : 1 ≤ n) {x y : Int}
    (hx1 : -((2 : Int) ^ (n - 1)) ≤ x) (hx2 : x < (2 : Int) ^ (n - 1))
    (hy1 : -((2 : Int) ^ (n - 1)) ≤ y) (hy2 : y < (2 : Int) ^ (n - 1))
    (hc : CongM n x y) : x = y := by
  have hU : ((2 ^ (n - 1) : Nat) : Int) = (2 : Int) ^ (n - 1) := by
    rw [Int.natCast_pow]; rfl
  have hM : ((2 ^ n : Nat) : Int) = 2 * ((2 ^ (n - 1) : Nat) : Int) := by
    have hp : (2 ^ n : Nat) = 2 ^ (n - 1) * 2 := by
      obtain ⟨r, hr⟩ : ∃ r : Nat, n = r + 1 := ⟨n - 1, by omega⟩
      subst hr
      show 2 ^ (r + 1) = 2 ^ r * 2
      exact Nat.pow_succ 2 r
    rw [hp, Int.natCast_mul]
    omega
  obtain ⟨k, hk⟩ := CongM.add_congruent hc
  rcases Int.lt_trichotomy k 0 with hk0 | rfl | hk0
  · have hkm : k * ((2 ^ n : Nat) : Int) + ((2 ^ n : Nat) : Int) ≤ 0 := by
      have h0 : 0 ≤ ((2 ^ (n - 1) : Nat) : Int) := Int.natCast_nonneg _
      have h1 : k * ((2 ^ n : Nat) : Int) ≤ -1 * ((2 ^ n : Nat) : Int) :=
        Int.mul_le_mul_of_nonneg_right (by omega) (by omega)
      omega
    omega
  · rw [Int.zero_mul, Int.add_zero] at hk
    exact hk
  · have hkm : ((2 ^ n : Nat) : Int) ≤ k * ((2 ^ n : Nat) : Int) := by
      have h1 : (1 : Int) * ((2 ^ n : Nat) : Int) ≤ k * ((2 ^ n : Nat) : Int) :=
        Int.mul_le_mul_of_nonneg_right (by omega) (by omega)
      rw [Int.one_mul] at h1
      exact h1
    omega

/-! ### The bounds and ranges of the type of width `n` -/

/-- The modulus of the type of width `n`: the number of values of the type. -/
def intModulus (n : Nat) : Int := ((2 ^ n : Nat) : Int)

/-- The smallest value of the type of width `n` read as a signed value. -/
def intMin (n : Nat) : Int := -((2 ^ (n - 1) : Nat) : Int)

/-- The largest value of the type of width `n` read as a signed value. -/
def intMax (n : Nat) : Int := ((2 ^ (n - 1) : Nat) : Int) - 1

theorem intModulus_pos (n : Nat) : 0 < intModulus n := by
  unfold intModulus
  exact_mod_cast Nat.two_pow_pos n

theorem intModulus_eq_two_mul (hn : 1 ≤ n) :
    ((2 ^ n : Nat) : Int) = 2 * ((2 ^ (n - 1) : Nat) : Int) := by
  have hp : (2 ^ n : Nat) = 2 ^ (n - 1) * 2 := by
    obtain ⟨r, hr⟩ : ∃ r : Nat, n = r + 1 := ⟨n - 1, by omega⟩
    subst hr
    show 2 ^ (r + 1) = 2 ^ r * 2
    exact Nat.pow_succ 2 r
  rw [hp, Int.natCast_mul]
  omega

theorem intMin_eq (n : Nat) : intMin n = -((2 : Int) ^ (n - 1)) := by
  unfold intMin
  rw [Int.natCast_pow]
  rfl

theorem intMax_eq (n : Nat) : intMax n = (2 : Int) ^ (n - 1) - 1 := by
  unfold intMax
  rw [Int.natCast_pow]
  rfl

/-- `x` is a value of the type of width `n` read as a signed value: it lies in
`[-2^(n-1), 2^(n-1) - 1]`. -/
def InSignedRange (n : Nat) (x : Int) : Prop :=
  intMin n ≤ x ∧ x ≤ intMax n

/-- `x` is a value of the type of width `n` read as an unsigned value: it lies
in `[0, 2^n - 1]`. -/
def InUnsignedRange (n : Nat) (x : Int) : Prop :=
  0 ≤ x ∧ x < intModulus n

/-! ### Wrapping an exact integer into a value of the type -/

/-- The unsigned value of the type of width `n` congruent to `z`: the residue
of `z` modulo `2 ^ n` — the reduction applied as many times as needed. -/
def uintWrap (n : Nat) (z : Int) : Int := z % ((2 ^ n : Nat) : Int)

theorem uintWrap_eq_emod (n : Nat) (z : Int) : uintWrap n z = z % intModulus n := rfl

theorem uintWrap_range (n : Nat) (z : Int) : InUnsignedRange n (uintWrap n z) := by
  have hM : (0 : Int) < ((2 ^ n : Nat) : Int) := intModulus_pos n
  exact ⟨Int.emod_nonneg _ (by omega), Int.emod_lt_of_pos _ hM⟩

theorem uintWrap_congruence (n : Nat) (z : Int) : CongM n (uintWrap n z) z :=
  Int.emod_emod z ((2 ^ n : Nat) : Int)

/-- The unsigned value of the type congruent to `z` is the only value of the
unsigned range congruent to `z`. -/
theorem uintWrap_unique {v z : Int} (hv : InUnsignedRange n v) (hc : CongM n v z) :
    v = uintWrap n z := by
  have hvv : v % ((2 ^ n : Nat) : Int) = v := Int.emod_eq_of_lt hv.1 hv.2
  calc v = v % ((2 ^ n : Nat) : Int) := hvv.symm
    _ = z % ((2 ^ n : Nat) : Int) := hc
    _ = uintWrap n z := rfl

/-- The signed value of the type of width `n` congruent to `z`: the residue of
`z` modulo `2 ^ n` when it is below `2 ^ (n - 1)`, that residue minus `2 ^ n`
otherwise — the two's-complement reading of the bit pattern. -/
def sintWrap (n : Nat) (z : Int) : Int :=
  if uintWrap n z < ((2 ^ (n - 1) : Nat) : Int) then uintWrap n z
  else uintWrap n z - ((2 ^ n : Nat) : Int)

theorem sintWrap_range (hn : 1 ≤ n) (z : Int) : InSignedRange n (sintWrap n z) := by
  have h0 := (uintWrap_range n z).1
  have hlt : uintWrap n z < ((2 ^ n : Nat) : Int) := (uintWrap_range n z).2
  have hM := intModulus_eq_two_mul hn
  have hpos : (0 : Int) < ((2 ^ (n - 1) : Nat) : Int) := by
    have : 0 < (2 ^ (n - 1) : Nat) := Nat.two_pow_pos _
    exact_mod_cast this
  unfold sintWrap InSignedRange intMin intMax
  split
  · omega
  · omega

theorem sintWrap_congruence (n : Nat) (z : Int) : CongM n (sintWrap n z) z := by
  unfold sintWrap
  split
  · exact uintWrap_congruence n z
  · unfold CongM
    rw [Int.sub_emod_right]
    exact uintWrap_congruence n z

/-- The signed value of the type congruent to `z` is the only signed value
congruent to `z`. -/
theorem sintWrap_unique (hn : 1 ≤ n) {v z : Int} (hv : InSignedRange n v)
    (hc : CongM n v z) : v = sintWrap n z := by
  have h2 : CongM n v (sintWrap n z) :=
    CongM.trans hc (sintWrap_congruence n z).symm
  have hr := sintWrap_range hn z
  unfold InSignedRange at hv hr
  rw [intMin_eq, intMax_eq] at hv hr
  exact congruent_range_unique hn hv.1 (by omega) hr.1 (by omega) h2

/-! ### The types as bit patterns -/

/-- An unsigned `n`-bit integer: a bit pattern `0 ≤ val < 2 ^ n`. -/
structure UIntN (n : Nat) where
  /-- the bit pattern -/
  val : Nat
  val_lt : val < 2 ^ n

instance : Inhabited (UIntN n) := ⟨⟨0, Nat.two_pow_pos n⟩⟩

/-- Extensionality of the type: two patterns with the same value are equal
(the property fields are proofs, hence equal). -/
@[ext]
theorem UIntN.ext {a b : UIntN n} (h : a.val = b.val) : a = b := by
  cases a; cases b
  subst h
  rfl

/-- Decidability of the equality of the type: equality of the bit patterns. -/
instance : DecidableEq (UIntN n) := fun a b =>
  if h : a.val = b.val then isTrue (UIntN.ext h)
  else isFalse (fun hh => h (by rw [hh]))

/-- The pattern of the unsigned value of the type congruent to `z`. -/
def UIntN.wrap (n : Nat) (z : Int) : UIntN n :=
  ⟨(uintWrap n z).toNat, by
    have h := (uintWrap_range n z).2
    have h0 := (uintWrap_range n z).1
    have ht : ((uintWrap n z).toNat : Int) = uintWrap n z := Int.toNat_of_nonneg h0
    rw [← Int.ofNat_lt, ht]
    exact h⟩

theorem UIntN.wrap_val (n : Nat) (z : Int) : (UIntN.wrap n z).val = (uintWrap n z).toNat :=
  rfl

theorem UIntN.wrap_congruent (n : Nat) (z : Int) : CongM n ((UIntN.wrap n z).val : Int) z := by
  have h0 := (uintWrap_range n z).1
  have ht : ((UIntN.wrap n z).val : Int) = uintWrap n z := by
    rw [UIntN.wrap_val, Int.toNat_of_nonneg h0]
  rw [ht]
  exact uintWrap_congruence n z

/-- A signed `n`-bit integer: a bit pattern `0 ≤ val < 2 ^ n`, which is the
two's-complement representation of a value of the canonical range
`[-2^(n-1), 2^(n-1)-1]`. -/
structure SIntN (n : Nat) where
  /-- the bit pattern -/
  val : Nat
  val_lt : val < 2 ^ n

instance : Inhabited (SIntN n) := ⟨⟨0, Nat.two_pow_pos n⟩⟩

/-- Extensionality of the type: two patterns with the same value are equal
(the property fields are proofs, hence equal). -/
@[ext]
theorem SIntN.ext {a b : SIntN n} (h : a.val = b.val) : a = b := by
  cases a; cases b
  subst h
  rfl

/-- Decidability of the equality of the type: equality of the bit patterns. -/
instance : DecidableEq (SIntN n) := fun a b =>
  if h : a.val = b.val then isTrue (SIntN.ext h)
  else isFalse (fun hh => h (by rw [hh]))

/-- The pattern of the signed value of the type congruent to `z` (the
two's-complement bit pattern of that value). -/
def SIntN.wrap (n : Nat) (z : Int) : SIntN n :=
  ⟨(uintWrap n z).toNat, by
    have h := (uintWrap_range n z).2
    have h0 := (uintWrap_range n z).1
    have ht : ((uintWrap n z).toNat : Int) = uintWrap n z := Int.toNat_of_nonneg h0
    rw [← Int.ofNat_lt, ht]
    exact h⟩

theorem SIntN.wrap_val (n : Nat) (z : Int) : (SIntN.wrap n z).val = (uintWrap n z).toNat :=
  rfl

/-- The value of a bit pattern: the unique integer of the canonical range
`[-2^(n-1), 2^(n-1))` congruent to the pattern modulo `2 ^ n` — the pattern
itself when it is below `2^(n-1)`, the pattern shifted down by `2 ^ n`
otherwise. -/
def SIntN.toInt (a : SIntN n) : Int :=
  if a.val < 2 ^ (n - 1) then (a.val : Int)
  else (a.val : Int) - ((2 ^ n : Nat) : Int)

/-- The value of a pattern is congruent to the pattern modulo `2 ^ n`. -/
theorem SIntN.toInt_congruence (a : SIntN n) :
    CongM n a.toInt (a.val : Int) := by
  simp only [CongM, SIntN.toInt]
  split
  · rfl
  · rw [Int.sub_emod_right]

/-- The value of a pattern lies in the canonical range `[-2^(n-1), 2^(n-1))`
of the type. -/
theorem SIntN.toInt_range (hn : 1 ≤ n) (a : SIntN n) :
    -((2 : Int) ^ (n - 1)) ≤ a.toInt ∧ a.toInt < (2 : Int) ^ (n - 1) := by
  have hU : ((2 ^ (n - 1) : Nat) : Int) = (2 : Int) ^ (n - 1) := by
    rw [Int.natCast_pow]; rfl
  have hM : ((2 ^ n : Nat) : Int) = 2 * ((2 ^ (n - 1) : Nat) : Int) := by
    have hp : (2 ^ n : Nat) = 2 ^ (n - 1) * 2 := by
      obtain ⟨r, hr⟩ : ∃ r : Nat, n = r + 1 := ⟨n - 1, by omega⟩
      subst hr
      show 2 ^ (r + 1) = 2 ^ r * 2
      exact Nat.pow_succ 2 r
    rw [hp, Int.natCast_mul]
    omega
  simp only [SIntN.toInt]
  split
  · exact ⟨by omega, by omega⟩
  · have hx1 : (2 ^ (n - 1) : Nat) ≤ a.val := by omega
    have hx2 : a.val < 2 ^ n := a.val_lt
    exact ⟨by omega, by omega⟩

/-- The value of the pattern built by `SIntN.wrap` is `sintWrap`: the pattern
is the unsigned residue and the two readings agree. -/
theorem SIntN.wrap_toInt (hn : 1 ≤ n) (z : Int) : (SIntN.wrap n z).toInt = sintWrap n z := by
  have h0 := (uintWrap_range n z).1
  have hlt := (uintWrap_range n z).2
  have hM := intModulus_eq_two_mul hn
  have hpos : (0 : Int) < ((2 ^ (n - 1) : Nat) : Int) := by
    have : 0 < (2 ^ (n - 1) : Nat) := Nat.two_pow_pos _
    exact_mod_cast this
  have hval : (SIntN.wrap n z).val = (uintWrap n z).toNat := rfl
  have ht : (((SIntN.wrap n z).val : Nat) : Int) = uintWrap n z := by
    rw [hval, Int.toNat_of_nonneg h0]
  have hif : (SIntN.wrap n z).val < 2 ^ (n - 1) ↔
      uintWrap n z < ((2 ^ (n - 1) : Nat) : Int) := by
    constructor
    · intro h
      have h' : ((SIntN.wrap n z).val : Int) < ((2 ^ (n - 1) : Nat) : Int) := by
        exact_mod_cast h
      rwa [ht] at h'
    · intro h
      have h' : ((SIntN.wrap n z).val : Int) < ((2 ^ (n - 1) : Nat) : Int) := by
        rwa [ht]
      exact_mod_cast h'
  unfold SIntN.toInt sintWrap
  rw [ht]
  by_cases hc : uintWrap n z < ((2 ^ (n - 1) : Nat) : Int)
  · rw [if_pos (hif.mpr hc), if_pos hc]
  · rw [if_neg (fun h => hc (hif.mp h)), if_neg hc]

end Sonnx
