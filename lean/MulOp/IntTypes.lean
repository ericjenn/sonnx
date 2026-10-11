/-
SONNX profile, operator **Mul** — the integer type families.

The `E_MUL_INT_FUNC_0010` and `E_MUL_UINT_FUNC_0010` specifications give the
multiplication of the signed and unsigned `n`-bit types: the result is the
value of the type congruent to the exact product modulo `2 ^ n`, and the
specification notes that the values of the type are `2 ^ n` distinct integers
no two of which are congruent modulo `2 ^ n`, so that exactly one value of
the type satisfies the congruence.

This file models the two families and proves the specification:

* `MulOp.UIntN`      — an unsigned `n`-bit integer, a pattern `0 ≤ val < 2 ^ n`;
  its multiplication is the exact product reduced modulo `2 ^ n`
  (`UIntN.mul_def`), and the result is congruent to the exact product
  (`UIntN.mul_congruent`).
* `MulOp.SIntN`      — a signed `n`-bit integer, the same bit pattern, whose
  value `SIntN.toInt` is the two's-complement interpretation, the unique
  integer of the canonical range `[-2 ^ (n-1), 2 ^ (n-1))` congruent to the
  pattern modulo `2 ^ n`.  Its multiplication is the same reduction, and
  `SIntN.mul_spec` states the signed specification: the value of the product
  is the unique integer of the canonical range congruent to the exact product
  of the values.

The reduction is the exact product reduced by as many multiples of `2 ^ n` as
needed — the "reduction may be applied several times" of the specification.
A bit pattern is a `BitVec n` value, so that this model agrees by construction
with the wrap-around `BitVec` multiplication of the Lean standard library, and
the ONNX integer types (`int8`, …, `uint64`) are the instances
`n = 8, 16, 32, 64` of these families.
-/
import MulOp.MulCore

namespace MulOp

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
`x' ≡ y'`, then `x * x' ≡ y * y'`.  This is what lets the congruence of
`E_MUL_INT_FUNC_0010` / `E_MUL_UINT_FUNC_0010` be stated on the values of the
elements although the reduction acts on the bit patterns. -/
theorem CongM.mul {x y x' y' : Int} (hx : CongM n x y) (hx' : CongM n x' y') :
    CongM n (x * x') (y * y') := by
  have hx1 : x % ((2 ^ n : Nat) : Int) = y % ((2 ^ n : Nat) : Int) := hx
  have hx1' : x' % ((2 ^ n : Nat) : Int) = y' % ((2 ^ n : Nat) : Int) := hx'
  show (x * x') % ((2 ^ n : Nat) : Int) = (y * y') % ((2 ^ n : Nat) : Int)
  rw [Int.mul_emod, hx1, hx1', ← Int.mul_emod]

/-- A congruence is the equality up to a multiple of `2 ^ n`: the exact
product reduced by as many multiples of `2 ^ n` as needed. -/
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

/-! ### The unsigned `n`-bit types -/

/-- An unsigned `n`-bit integer: a bit pattern `0 ≤ val < 2 ^ n`. -/
structure UIntN (n : Nat) where
  /-- the bit pattern -/
  val : Nat
  val_lt : val < 2 ^ n

instance : Inhabited (UIntN n) := ⟨⟨0, Nat.two_pow_pos n⟩⟩

/-- The multiplication of the unsigned `n`-bit types: the exact product
reduced modulo `2 ^ n`. -/
instance : Mul (UIntN n) :=
  ⟨fun a b => ⟨(a.val * b.val) % 2 ^ n, Nat.mod_lt _ (Nat.two_pow_pos n)⟩⟩

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

/-- The product in the type is the exact product reduced modulo `2 ^ n`
(the functional part of `E_MUL_UINT_FUNC_0010`). -/
theorem UIntN.mul_def (a b : UIntN n) :
    (a * b).val = (a.val * b.val) % 2 ^ n := rfl

/-- The product is congruent to the exact product: `k` counts the multiples
of `2 ^ n` by which the exact product exceeds the result — the reduction
applied as many times as needed (the overflow error condition of
`E_MUL_UINT_FUNC_0010`, nominal). -/
theorem UIntN.mul_congruent (a b : UIntN n) :
    ∃ k : Nat, a.val * b.val = (a * b).val + k * 2 ^ n := by
  refine ⟨(a.val * b.val) / 2 ^ n, ?_⟩
  have h := Nat.div_add_mod (a.val * b.val) (2 ^ n)
  show a.val * b.val =
    ((a.val * b.val) % 2 ^ n) + (a.val * b.val) / 2 ^ n * 2 ^ n
  rw [Nat.mul_comm ((a.val * b.val) / 2 ^ n) (2 ^ n)]
  omega

/-! ### The signed `n`-bit types -/

/-- A signed `n`-bit integer: a bit pattern `0 ≤ val < 2 ^ n`, which is the
two's-complement representation of a value of the canonical range
`[-2^(n-1), 2^(n-1)-1]`. -/
structure SIntN (n : Nat) where
  /-- the bit pattern -/
  val : Nat
  val_lt : val < 2 ^ n

instance : Inhabited (SIntN n) := ⟨⟨0, Nat.two_pow_pos n⟩⟩

/-- The multiplication of the signed `n`-bit types: the exact product reduced
modulo `2 ^ n`, like the unsigned one — the two types differ in the
interpretation of the bit pattern, not in the reduction. -/
instance : Mul (SIntN n) :=
  ⟨fun a b => ⟨(a.val * b.val) % 2 ^ n, Nat.mod_lt _ (Nat.two_pow_pos n)⟩⟩

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

/-- The product in the type is the exact product reduced modulo `2 ^ n`
(the functional part of `E_MUL_INT_FUNC_0010`). -/
theorem SIntN.mul_def (a b : SIntN n) :
    (a * b).val = (a.val * b.val) % 2 ^ n := rfl

/-- The product is congruent to the exact product (`E_MUL_INT_FUNC_0010`).
The sign of the result is the sign of this reduction, not the sign of the
exact product. -/
theorem SIntN.mul_congruent (a b : SIntN n) :
    ∃ k : Nat, a.val * b.val = (a * b).val + k * 2 ^ n := by
  refine ⟨(a.val * b.val) / 2 ^ n, ?_⟩
  have h := Nat.div_add_mod (a.val * b.val) (2 ^ n)
  show a.val * b.val =
    ((a.val * b.val) % 2 ^ n) + (a.val * b.val) / 2 ^ n * 2 ^ n
  rw [Nat.mul_comm ((a.val * b.val) / 2 ^ n) (2 ^ n)]
  omega

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

/-- `E_MUL_INT_FUNC_0010`: the value of the product is the unique integer of
the canonical range `[-2^(n-1), 2^(n-1))` congruent to the exact product
`a.toInt * b.toInt` modulo `2 ^ n` — the first two conjuncts say that the
product is a value of the type, the third that it satisfies the congruence,
and the fourth that it is the only value that does (the values of the type
are `2 ^ n` distinct integers no two of which are congruent modulo `2 ^ n`). -/
theorem SIntN.mul_spec (hn : 1 ≤ n) (a b : SIntN n) :
    -((2 : Int) ^ (n - 1)) ≤ (a * b).toInt ∧ (a * b).toInt < (2 : Int) ^ (n - 1) ∧
    CongM n (a * b).toInt (a.toInt * b.toInt) ∧
    ∀ v : Int, (-((2 : Int) ^ (n - 1)) ≤ v ∧ v < (2 : Int) ^ (n - 1) ∧
      CongM n v (a.toInt * b.toInt)) → v = (a * b).toInt := by
  have hU : ((2 ^ (n - 1) : Nat) : Int) = (2 : Int) ^ (n - 1) := by
    rw [Int.natCast_pow]; rfl
  have hcong : CongM n (a * b).toInt (a.toInt * b.toInt) := by
    have hv : ((a * b).val : Int) =
        ((a.val : Int) * (b.val : Int)) % ((2 ^ n : Nat) : Int) := by
      rw [SIntN.mul_def, Int.natCast_emod, Int.natCast_mul]
    have h1 : CongM n (a * b).toInt ((a * b).val : Int) := SIntN.toInt_congruence _
    have h2 : CongM n ((a * b).val : Int) ((a.val : Int) * (b.val : Int)) := by
      show ((a * b).val : Int) % ((2 ^ n : Nat) : Int) =
        ((a.val : Int) * (b.val : Int)) % ((2 ^ n : Nat) : Int)
      rw [hv, Int.emod_emod]
    have h3a : CongM n (a.val : Int) a.toInt := (SIntN.toInt_congruence a).symm
    have h3b : CongM n (b.val : Int) b.toInt := (SIntN.toInt_congruence b).symm
    exact CongM.trans h1 (CongM.trans h2 (CongM.mul h3a h3b))
  have hcong0 : (a * b).toInt % ((2 ^ n : Nat) : Int) =
      (a.toInt * b.toInt) % ((2 ^ n : Nat) : Int) := hcong
  refine ⟨(SIntN.toInt_range hn (a * b)).1, (SIntN.toInt_range hn (a * b)).2,
    hcong, ?_⟩
  intro v hv
  have hv22 : v % ((2 ^ n : Nat) : Int) =
      (a.toInt * b.toInt) % ((2 ^ n : Nat) : Int) := hv.2.2
  have hc : CongM n v (a * b).toInt := by
    show v % ((2 ^ n : Nat) : Int) = (a * b).toInt % ((2 ^ n : Nat) : Int)
    rw [hv22, ← hcong0]
  exact congruent_range_unique hn hv.1 hv.2.1
    (SIntN.toInt_range hn (a * b)).1 (SIntN.toInt_range hn (a * b)).2 hc

/-! ### The examples of the specification -/

/-- `uint8`: `200 · 200 = 64` — the exact product 40000 is `156 · 2^8 + 64`,
so the reduction is applied more than once. -/
example : ((⟨200, by decide⟩ : UIntN 8) * (⟨200, by decide⟩ : UIntN 8)).val
    = 64 := by decide

/-- `uint8`: `20 · 13 = 4` — a value smaller than both operands (the
multiplication of the type is not monotonic). -/
example : ((⟨20, by decide⟩ : UIntN 8) * (⟨13, by decide⟩ : UIntN 8)).val
    = 4 := by decide

/-- `int8`: `11 · 13 = -113` — the exact product 143 exceeds the maximum value
127 of the type and the result is `143 - 2^8`, negative although both
operands are positive: the sign of the result is the sign of the reduction,
not the sign of the exact product. -/
example : ((⟨11, by decide⟩ : SIntN 8) * (⟨13, by decide⟩ : SIntN 8)).toInt
    = -113 := by decide

/-- `int8`: `100 · 100 = 16` — the exact product 10000 is `39 · 2^8 + 16`. -/
example : ((⟨100, by decide⟩ : SIntN 8) * (⟨100, by decide⟩ : SIntN 8)).toInt
    = 16 := by decide

/-- `int8`: `16 · 16 = 0` — a zero result although both operands are
non-zero. -/
example : ((⟨16, by decide⟩ : SIntN 8) * (⟨16, by decide⟩ : SIntN 8)).toInt
    = 0 := by decide

end MulOp