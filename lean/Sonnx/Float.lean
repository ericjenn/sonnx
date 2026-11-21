/-
The SONNX shared Lean library — the floating-point type family.

The floating-point operator specifications describe the operation as IEEE 754
arithmetic: a `NaN` operand propagates, the invalid operations produce a quiet
`NaN`, and every other pair of values produces the exact result when it is
representable in the format, and otherwise the exact result rounded with
rounding mode roundTiesToEven.  The result carries the sign of the exact
result, including for the zeros.

This module supplies the vocabulary for that description:

* `Sonnx.FloatKind` — `float16`, `float32` and `float64`,
* `Sonnx.FVal`      — a floating-point value: NaN, the two infinities, the two
  signed zeros, and the signed finite values (with an exact rational
  magnitude); `Sonnx.FVal.WellFormed` requires a non-negative magnitude,
* `Sonnx.FloatFormat` — a binary format, abstracted by its significand
  precision and its largest finite value, with `Sonnx.FloatFormat.ofKind` the
  three formats of the profile,
* `Sonnx.IsRepresentable` — the exact values representable in a format,
* `Sonnx.RoundNearestEven`, `Sonnx.Overflows` — the roundTiesToEven rounding
  relation and the overflow threshold,
* `Sonnx.RoundToFormat` — `r` is a conforming rounding of the exact non-zero
  value `x` to the format: finite, a signed zero, an infinity on overflow.

The IEEE 754 bit patterns are not modeled; the payload and sign bit of a NaN
are not recorded, since the specifications leave them unspecified.
-/
import Sonnx.Prelude

namespace Sonnx

/-- The binary IEEE 754 formats of the profile. -/
inductive FloatKind
  | f16
  | f32
  | f64
  deriving DecidableEq

/-- A binary IEEE 754 format: its significand precision (including the implicit
leading bit) and its largest finite value.  The exponent range is bounded
through `maxFinite`: a rounded magnitude beyond it overflows to an infinity. -/
structure FloatFormat where
  /-- the significand precision, including the implicit leading bit -/
  precision : Nat
  /-- the largest finite value of the format -/
  maxFinite : Rat

/-- The formats of `float16` (`binary16`), `float32` (`binary32`) and
`float64` (`binary64`): precision `11`, `24`, `53`; largest finite value
`(2 ^ p - 1) * 2 ^ (maxExp - (p - 1))` with `maxExp = 15, 127, 1023`. -/
def FloatFormat.ofKind : FloatKind → FloatFormat
  | .f16 => ⟨11, ((2 ^ 11 - 1 : Nat) * 2 ^ 5 : Rat)⟩
  | .f32 => ⟨24, ((2 ^ 24 - 1 : Nat) * 2 ^ 104 : Rat)⟩
  | .f64 => ⟨53, ((2 ^ 53 - 1 : Nat) * 2 ^ 971 : Rat)⟩

/-- The floating-point values: NaN, the two infinities, the two signed zeros,
and the signed finite values (the `Rat` component is the magnitude). -/
inductive FVal where
  | nan : FVal
  | inf : Bool → FVal
  | zero : Bool → FVal
  | finite : Bool → Rat → FVal
  deriving DecidableEq

instance : Inhabited FVal := ⟨FVal.nan⟩

namespace FVal

/-- `v` is a NaN. -/
def isNaN : FVal → Bool
  | nan => true
  | _ => false

/-- `v` is an infinity. -/
def isInf : FVal → Bool
  | inf _ => true
  | _ => false

/-- The sign bit; `true` means negative (including `-0` and `-inf`). -/
def sign : FVal → Bool
  | nan => false
  | inf s => s
  | zero s => s
  | finite s _ => s

/-- The exact rational value of a zero or finite value. -/
def toRat : FVal → Option Rat
  | zero _ => some 0
  | finite s q => some (if s then -q else q)
  | _ => none

/-- Well-formedness: a finite value carries a non-negative magnitude. -/
def WellFormed : FVal → Prop
  | finite _ q => 0 ≤ q
  | _ => True

theorem wellFormed_nan : FVal.nan.WellFormed := trivial

theorem wellFormed_inf (s : Bool) : (FVal.inf s).WellFormed := trivial

theorem wellFormed_zero (s : Bool) : (FVal.zero s).WellFormed := trivial

theorem wellFormed_finite {s : Bool} {q : Rat} (h : 0 ≤ q) :
    (FVal.finite s q).WellFormed := h

end FVal

/-! ### Exact arithmetic on the magnitudes -/

/-- Absolute value on `Rat`. -/
def qabs (x : Rat) : Rat := if x < 0 then -x else x

/-- `x` is an integral multiple of `v` (`v` times a natural number, up to sign). -/
def IsIntMultiple (v q : Rat) : Prop :=
  ∃ k : Nat, q = (k : Rat) * v ∨ q = -((k : Rat) * v)

/-! ### Representability and rounding -/

/-- `x` has a `p`-bit dyadic representation: for some `k`, `x * 2^k` is an integer
whose magnitude is smaller than `2^(p+k)`. -/
def IsDyadicWithPrecision (p : Nat) (x : Rat) : Prop :=
  ∃ k : Nat, ∃ m : Nat,
    (x * (2 : Rat) ^ k = (m : Rat) ∨ x * (2 : Rat) ^ k = -((m : Rat))) ∧
    m < 2 ^ (p + k)

/-- `x` is exactly representable in the format `fmt` (with unbounded exponent and
no overflow). -/
def IsRepresentable (fmt : FloatFormat) (x : Rat) : Prop :=
  (x = 0 ∨ IsDyadicWithPrecision fmt.precision x) ∧ qabs x ≤ fmt.maxFinite

/-- `q` is a nearest representable value to `x`. -/
def Nearest (fmt : FloatFormat) (x q : Rat) : Prop :=
  IsRepresentable fmt q ∧
  ∀ q' : Rat, IsRepresentable fmt q' → qabs (x - q) ≤ qabs (x - q')

/-- `q` is the "even" one of two tied candidates `q` and `q'`: `q` is an even
multiple and `q'` an odd multiple of the spacing `u = |q - q'|`. -/
def TieEven (q q' : Rat) : Prop :=
  ∃ u : Rat, 0 < u ∧ IsIntMultiple (2 * u) q ∧ ¬ IsIntMultiple (2 * u) q' ∧
    (q' = q + u ∨ q' = q - u)

/-- `q` is the roundTiesToEven rounding of `x` (nearest value, ties broken by the
even significand). -/
def RoundNearestEven (fmt : FloatFormat) (x q : Rat) : Prop :=
  Nearest fmt x q ∧
  ∀ q' : Rat, IsRepresentable fmt q' → qabs (x - q') = qabs (x - q) →
    q' = q ∨ TieEven q q'

/-- The spacing between the two largest finite values of the format. -/
def UlpTop (fmt : FloatFormat) (u : Rat) : Prop :=
  0 < u ∧ fmt.maxFinite = ((2 : Rat) ^ fmt.precision - 1) * u

/-- The exact magnitude `x` lies beyond the overflow threshold, i.e. its nearest
value exceeds the largest finite value of the format; the result is an infinity. -/
def Overflows (fmt : FloatFormat) (x : Rat) : Prop :=
  ∀ u : Rat, UlpTop fmt u → fmt.maxFinite + (fmt.maxFinite + u) ≤ 2 * qabs x

/-- `r` is a conforming roundTiesToEven rounding of the exact non-zero value `x`
to the format: an infinity beyond the overflow threshold, a signed zero below
it, and otherwise the finite value nearest `x` with the sign of `x` — the
`RoundToFormat` of the float operator specifications. -/
def RoundToFormat (fmt : FloatFormat) (x : Rat) (r : FVal) : Prop :=
  match r with
  | FVal.nan => False
  | FVal.inf neg => Overflows fmt x ∧ (neg = true ↔ x < 0)
  | FVal.zero neg => ¬ Overflows fmt x ∧ RoundNearestEven fmt x 0 ∧
      (neg = true ↔ x < 0)
  | FVal.finite s q =>
      0 < q ∧ ¬ Overflows fmt x ∧
      RoundNearestEven fmt x (if s then -q else q)

end Sonnx
