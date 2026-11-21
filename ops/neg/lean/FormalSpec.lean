import Sonnx
/-!
@generated-by sonnx-lean-pipeline
@source ops/neg/neg.md
@source-digest ce2178eb6a3a118e04a6d11895fa30b1
@library-digest 3162109b4c926c2fa5dd5f8bd2eb2d90fb17a0480eccc0216a4ea9bd2b43ffab
Compilation and trace validity are separate from semantic completeness; see the trace report.
-/


open Sonnx

/-!
# `Neg` — element-wise negation

Three type families are specified, each by its own relation over `Sonnx.Tensor`:

* real values (`Rat`, exact ordered field): `Y[i] = -X[i]`;
* floating-point values (`FVal`): IEEE 754 sign inversion (exact, no rounding);
* signed `n`-bit integers (`SIntN n`): the signed value congruent to `-X[i]`
  modulo `2^n` (so the minimum value is its own negation).

The shape rule is shared by the three families: the output has the input shape.
-/

/-! 
@trace E_NEG_REAL_CONSTR_X_0010
@source ops/neg/neg.md:L84-L86
@coverage specified
@anchor E_NEG_REAL_CONSTR_X_0010
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_NEG_REAL_CONSTR_X_0010"></a>
  - `[E_NEG_REAL_CONSTR_X_0010]` Type consistency
    - Statement: Tensors $X$ and $Y$ have the same type.
-/

/-! 
@trace E_NEG_REAL_CONSTR_Y_0020
@source ops/neg/neg.md:L98-L99
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_NEG_REAL_CONSTR_Y_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_REAL_CONSTR_X_0010</span></b>](#E_NEG_REAL_CONSTR_X_0010) on tensor $X$.
-/

/-! 
@trace E_NEG_FLOAT_CONSTR_X_0010
@source ops/neg/neg.md:L204-L206
@coverage specified
@anchor E_NEG_FLOAT_CONSTR_X_0010
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_NEG_FLOAT_CONSTR_X_0010"></a>
  - `[E_NEG_FLOAT_CONSTR_X_0010]` Type consistency
    - Statement: Tensors $X$ and $Y$ have the same type.
-/

/-! 
@trace E_NEG_FLOAT_CONSTR_Y_0020
@source ops/neg/neg.md:L218-L219
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_NEG_FLOAT_CONSTR_Y_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_CONSTR_X_0010</span></b>](#E_NEG_FLOAT_CONSTR_X_0010) on tensor $X$.
-/

/-! 
@trace E_NEG_INT_CONSTR_X_0010
@source ops/neg/neg.md:L311-L313
@coverage specified
@anchor E_NEG_INT_CONSTR_X_0010
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_NEG_INT_CONSTR_X_0010"></a>
  - `[E_NEG_INT_CONSTR_X_0010]` Type consistency
    - Statement: Tensors $X$ and $Y$ have the same type.
-/

/-! 
@trace E_NEG_INT_CONSTR_Y_0020
@source ops/neg/neg.md:L325-L326
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_NEG_INT_CONSTR_Y_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_INT_CONSTR_X_0010</span></b>](#E_NEG_INT_CONSTR_X_0010) on tensor $X$.
-/


namespace Generated.Neg

/-! ## Shape rule (shared by the three families) -/

/-! 
@trace E_NEG_REAL_CONSTR_Y_0010
@source ops/neg/neg.md:L96-L97
@coverage specified
@declaration Generated.Neg.NegShape
Informal source excerpt:
  - `[E_NEG_REAL_CONSTR_Y_0010]` Shape definition
    - Statement: Tensor $Y$ has the shape of $X$.
-/

/-! 
@trace E_NEG_FLOAT_CONSTR_Y_0010
@source ops/neg/neg.md:L216-L217
@coverage specified
@declaration Generated.Neg.NegShape
Informal source excerpt:
  - `[E_NEG_FLOAT_CONSTR_Y_0010]` Shape definition
    - Statement: Tensor $Y$ has the shape of $X$.
-/

/-! 
@trace E_NEG_INT_CONSTR_Y_0010
@source ops/neg/neg.md:L323-L324
@coverage specified
@declaration Generated.Neg.NegShape
Informal source excerpt:
  - `[E_NEG_INT_CONSTR_Y_0010]` Shape definition
    - Statement: Tensor $Y$ has the shape of $X$.
-/

/-- Shape definition: the output `Y` has the shape of the input `X`.

DECISION: the constraint is stated as an equality of shapes rather than through
`Sonnx.bcastShape`; `Neg` is unary, so no broadcasting occurs.  The statement
also covers the rank-0 case (`shape = []`, one element) and zero-sized
dimensions (the data list is then empty because `Tensor.wellFormed` forces its
length to `size shape = 0`). -/
def NegShape {α : Type} (X Y : Tensor α) : Prop :=
  Y.shape = X.shape

/-! ## Real `Neg`

DECISION: real numbers are modelled by `Rat`, the exact ordered field available
in the shared profile; `Rat` negation is exact, so the informal "no error
condition, negation is exact in ℝ" is captured verbatim.  `NegShape` records the
output shape, and the element step `fun x => -x` is applied pointwise, index by
index, through the row-major data list. -/

/-! 
@trace E_NEG_REAL_FUNC_0010
@source ops/neg/neg.md:L31-L52
@coverage specified
@anchor E_NEG_REAL_FUNC_0010
@declaration Generated.Neg.RealNegRel
Informal source excerpt:
  <a id="E_NEG_REAL_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_NEG_REAL_FUNC_0010]</br></span>
  
  Operator **Neg** negates tensor $X$ element-wise and stores the result in tensor $Y$. Each element of $Y$ is the opposite of the element of $X$ that has the same index.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $Y$:
  
  $$
  Y[i] = -X[i]
  $$
  
  where
  - $X[i]$ is the element of $X$ at index $i$,
  - $-X[i]$ is the negation of $X[i]$ in $\mathbb{R}$.
  
  The negation is exact: $Y[i]$ is the opposite of $X[i]$ in $\mathbb{R}$, with no approximation.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty and which holds a single element. The operator applies to that element, and $Y$ is a rank-0 tensor.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-- `E_NEG_REAL_FUNC_0010`: for every index `i` of `Y`, `Y[i] = - X[i]`, exactly,
and `Y` has the shape of `X`. -/
def RealNegRel (X Y : Tensor Rat) : Prop :=
  NegShape X Y ∧ Y.data = X.data.map (fun x => -x)

/-! ## Floating-point `Neg`

DECISION: floating-point values are modelled by `Sonnx.FVal`.  IEEE 754 negation
is exact sign inversion: finite values keep their magnitude and flip the sign,
`+0`/`-0` swap, `+∞`/`-∞` swap, and a NaN operand yields a NaN.

DECISION: the sign field of `FVal` is read as the sign bit (`false` = non-negative
value, `true` = negative value); the specification of the examples only depends
on the *flip*, so the convention is used consistently on both sides.

DECISION: `FVal` has a single NaN constructor, so "the sign and the payload of a
NaN result are not specified, every quiet NaN is conforming" is represented by
that unique constructor.  The invalid operation of IEEE 754 §7.2 is never
signalled, matching the informal "no error condition": `Neg` never fails, hence
the relation is total on well-formed operands of the family.

DECISION: the format-specific range (`f16`, `f32`, `f64`) restricts which `FVal`s
are admissible inputs (`Sonnx.IsRepresentable`, `Sonnx.Overflows`); it does not
change the element step, which never rounds, never overflows and never
underflows (the magnitude of the result is the magnitude of the operand). -/

/-- The element step of floating-point negation: exact sign inversion. -/
def negFVal (x : FVal) : FVal :=
  match x with
  | .nan => .nan
  | .inf sign => .inf (Bool.not sign)
  | .zero sign => .zero (Bool.not sign)
  | .finite sign mag => .finite (Bool.not sign) mag

/-! 
@trace E_NEG_FLOAT_FUNC_0010
@source ops/neg/neg.md:L125-L151
@coverage specified
@anchor E_NEG_FLOAT_FUNC_0010
@declaration Generated.Neg.FloatNegRel
Informal source excerpt:
  <a id="E_NEG_FLOAT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_NEG_FLOAT_FUNC_0010]</br></span>
  
  Operator **Neg** negates tensor $X$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $Y$. Each element of $Y$ is the opposite of the element of $X$ that has the same index.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $Y$:
  
  $$
  Y[i] = -_{(\text{f})} X[i]
  $$
  
  where
  - $X[i]$ is the element of $X$ at index $i$,
  - $-_{(\text{f})}$ is the negation of the floating-point type of $X$, i.e. $-_{(\text{f16})}$, $-_{(\text{f32})}$ or $-_{(\text{f64})}$ according to that type: the result is the value of that type whose magnitude is the magnitude of $X[i]$ and whose sign is the opposite of the sign of $X[i]$.
  
  The negation is exact: it changes the sign of the operand and leaves its magnitude unchanged, so that no rounding occurs and the result is a value of the type of $X$ for every operand. The cases of the domain are the following.
  
  - **Finite operand.** $Y[i]$ is finite, of the opposite sign, and equal in magnitude to $X[i]$. No overflow and no underflow can occur, since the magnitude of the result is the magnitude of an operand.
  - **Null operand.** If $X[i]$ is $+0$, then $Y[i]$ is $-0$; if $X[i]$ is $-0$, then $Y[i]$ is $+0$. The sign of a null result is thus the opposite of the sign of the null operand.
  - **Infinite operand.** If $X[i]$ is $+\infty$, then $Y[i]$ is $-\infty$; if $X[i]$ is $-\infty$, then $Y[i]$ is $+\infty$.
  - **NaN operand.** $Y[i]$ is a NaN. Its sign and its payload are not specified, so that every quiet NaN of the type of $X$ is a conforming result. The negation of a NaN does not signal the invalid operation of IEEE 754 section 7.2.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty and which holds a single element. The operator applies to that element, and $Y$ is a rank-0 tensor.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-- `E_NEG_FLOAT_FUNC_0010`: for every index `i` of `Y`, `Y[i] = -_(f) X[i]`,
the value of the type of `X` with the magnitude of `X[i]` and the opposite sign,
and `Y` has the shape of `X`. -/
def FloatNegRel (X Y : Tensor FVal) : Prop :=
  NegShape X Y ∧ Y.data = X.data.map negFVal

/-! ## Signed-integer `Neg`

DECISION: signed `n`-bit integers are modelled by `Sonnx.SIntN n`.  The result is
the signed `n`-bit value congruent to `-X[i]` modulo `2^n`
(`Sonnx.sintWrap n (-(toInt X[i]))`), which is the informal piecewise rule
`-X[i]` for `X[i] > -2^(n-1)` and `-2^(n-1)` for `X[i] = -2^(n-1)`: the minimum
value is its own negation.  This is the *nominal* overflow behaviour of the
informal specification, not an error, so the relation has no failure case. -/

/-- The element step of signed-integer negation: negation modulo `2 ^ n`, read
back as a signed value. -/
def intNegValue (n : Nat) (x : SIntN n) : SIntN n :=
  SIntN.wrap n (-(SIntN.toInt x))

/-! 
@trace E_NEG_INT_FUNC_0010
@source ops/neg/neg.md:L245-L272
@coverage specified
@anchor E_NEG_INT_FUNC_0010
@declaration Generated.Neg.IntNegRel
Informal source excerpt:
  <a id="E_NEG_INT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_NEG_INT_FUNC_0010]</br></span>
  
  Operator **Neg** negates tensor $X$ element-wise and stores the result in tensor $Y$. Each element of $Y$ is the opposite of the element of $X$ that has the same index. Tensors $X$ and $Y$ have the same signed $n$-bit type, and the result is a value of that type.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $Y$:
  
  $$
  Y[i] =
  \begin{cases}
  -X[i] & \text{if } X[i] \gt -2^{n-1} \\
  -2^{n-1} & \text{if } X[i] = -2^{n-1}
  \end{cases}
  $$
  
  where
  - $X[i]$ is the element of $X$ at index $i$ and $-X[i]$ its negation in $\mathbb{Z}$,
  - $n$ is the number of bits of the type of $X$.
  
  The result is thus the value of the type that represents $-X[i]$ modulo $2^n$, read as a signed value. The values of an $n$-bit signed type lie in $[-2^{n-1}, 2^{n-1}-1]$, so that the negation of a value other than $-2^{n-1}$ lies in $[-(2^{n-1}-1), 2^{n-1}-1]$ and is a value of the type: the first case is the only one that applies to those values.
  
  **The minimum value is its own negation.** The negation of the minimum value $-2^{n-1}$ of the type is $2^{n-1}$, which is not representable in the type; the result is $2^{n-1} - 2^n = -2^{n-1}$, i.e. the minimum value itself. For `int8`, the negation of $-128$ is $-128$. Tensors $X$ and $Y$ have the same type and no wider type is used, so that the value of $Y$ is that value of the type.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty and which holds a single element. The operator applies to that element, and $Y$ is a rank-0 tensor.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-- `E_NEG_INT_FUNC_0010`: for every index `i` of `Y`, `Y[i]` is the value of the
signed `n`-bit type of `X` that represents `-X[i]` modulo `2 ^ n`, and `Y` has
the shape of `X`. -/
def IntNegRel (n : Nat) (X Y : Tensor (SIntN n)) : Prop :=
  NegShape X Y ∧ Y.data = X.data.map (intNegValue n)

/-! ## Concrete floating-point examples -/

/-- Example 1, input: `[1.5, +0, -0, +inf, -inf, NaN]`. -/
def floatExample1X : Tensor FVal :=
  { shape := [6],
    data :=
      [ FVal.finite false ((3 : Rat) / 2)
      , FVal.zero false
      , FVal.zero true
      , FVal.inf false
      , FVal.inf true
      , FVal.nan
      ] }

/-- Example 1, expected output: `[-1.5, -0, +0, -inf, +inf, NaN]`. -/
def floatExample1Y : Tensor FVal :=
  { shape := [6],
    data :=
      [ FVal.finite true ((3 : Rat) / 2)
      , FVal.zero true
      , FVal.zero false
      , FVal.inf true
      , FVal.inf false
      , FVal.nan
      ] }

/-! 
@trace float:example:1
@source ops/neg/neg.md:L155-L165
@coverage proved
@declaration Generated.Neg.floatExample1_correct
Informal source excerpt:
  ### Example 1
  
  $$
  X = \begin{bmatrix} 1.5 & \text{+0} & \text{-0} & \text{+inf} & \text{-inf} & \text{NaN} \end{bmatrix}
  $$
  
  $$
  Y = \begin{bmatrix} -1.5 & \text{-0} & \text{+0} & \text{-inf} & \text{+inf} & \text{NaN} \end{bmatrix}
  $$
  
  The negation of $+0$ is $-0$ and the negation of $-0$ is $+0$: the sign of a null result is the opposite of the sign of the operand. The negation of an infinity is the infinity of the opposite sign. The negation of a NaN is a NaN, whose sign and payload are not specified.
-/

/-- Example 1 is an instance of `FloatNegRel` (concrete evidence only; it is not
a universal conformance claim). -/
theorem floatExample1_correct : FloatNegRel floatExample1X floatExample1Y := by
  unfold FloatNegRel NegShape
  constructor <;> rfl

/-- Example 2, input for `float16`:
`[65504, 2^-24, 1.0]` (`2^-24 = 1/16777216`, the smallest positive subnormal). -/
def floatExample2X : Tensor FVal :=
  { shape := [3],
    data :=
      [ FVal.finite false (65504 : Rat)
      , FVal.finite false ((1 : Rat) / 16777216)
      , FVal.finite false (1 : Rat)
      ] }

/-- Example 2, expected output: `[-65504, -2^-24, -1.0]`. -/
def floatExample2Y : Tensor FVal :=
  { shape := [3],
    data :=
      [ FVal.finite true (65504 : Rat)
      , FVal.finite true ((1 : Rat) / 16777216)
      , FVal.finite true (1 : Rat)
      ] }

/-! 
@trace float:example:2
@source ops/neg/neg.md:L167-L179
@coverage proved
@declaration Generated.Neg.floatExample2_correct
Informal source excerpt:
  ### Example 2
  
  For the `float16` type:
  
  $$
  X = \begin{bmatrix} 65504 & 2^{-24} & 1.0 \end{bmatrix}
  $$
  
  $$
  Y = \begin{bmatrix} -65504 & -2^{-24} & -1.0 \end{bmatrix}
  $$
  
  The first element of $X$ is the largest finite value of `float16` and the second is the smallest positive subnormal value of that type. Their negations are finite values of the type: the negation of a finite value never overflows and never underflows, whatever the magnitude of the operand.
-/

/-- Example 2 is an instance of `FloatNegRel` (concrete evidence only): the
negation of a finite value of the type is a finite value of the type, whatever
the magnitude of the operand. -/
theorem floatExample2_correct : FloatNegRel floatExample2X floatExample2Y := by
  unfold FloatNegRel NegShape
  constructor <;> rfl

end Generated.Neg

-- Fully-qualified declarations named by the trace entries.
#check @Generated.Neg.FloatNegRel
#check @Generated.Neg.IntNegRel
#check @Generated.Neg.NegShape
#check @Generated.Neg.RealNegRel
#check @Generated.Neg.floatExample1_correct
#check @Generated.Neg.floatExample2_correct
#check @Sonnx.TypeConsistent
