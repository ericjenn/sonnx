import CGen.Print
import CGen.Derive
import Sonnx
import CGen
/-!
@generated-by sonnx-lean-pipeline
@source ops/mul/mul.md
@source-digest 07e1f247d67a12027ed6b87c56c03147
@library-digest 3162109b4c926c2fa5dd5f8bd2eb2d90fb17a0480eccc0216a4ea9bd2b43ffab
Compilation and trace validity are separate from semantic completeness; see the trace report.
-/


/-!
# `Mul` — elementwise multiplication

Formal specification of the ONNX `Mul` operator (opset 14) for the four type families of
the profile: real, floating point, signed integer and unsigned integer.

## Modeling decisions

* `DECISION:` the real family is stated over `Rat`. The specification demands the *exact*
  product in `ℝ`, and `Rat` is the exact ordered field of the profile.
* `DECISION:` the integer families are stated over `Int`-valued tensors with the bit width
  `n` as a shared parameter. "The three tensors have the same type" is captured by the
  shared width `n` together with the range predicates `InSignedRange n` / `InUnsignedRange n`
  (every element is a value of the `n`-bit type), and the element step uses `sintWrap n` /
  `uintWrap n`, the value of the type congruent to the exact product modulo `2 ^ n`.
* `DECISION:` the floating-point family is stated over `FVal`. In `FVal.finite s m`, `s` is
  the sign bit (`true` = negative) and `m` is the nonnegative magnitude; the same convention
  is used for `FVal.zero s` and `FVal.inf s`. `FVal.nan` stands for "a quiet NaN", its sign
  and payload being unspecified as the specification requires.
* `DECISION:` the rounding of the exact product of two finite operands follows the
  document's rule literally: the exact magnitude is rounded to the nearest value whose
  significand fits in `precision` bits with unbounded exponent, ties to even, and the result
  is an infinity of the sign of the exact product when that nearest value exceeds
  `maxFinite`. See the trace marker of `float:example:3` for the consequence on magnitudes
  below the smallest subnormal of the format.
* `DECISION:` the shape rules (broadcasting, rank-0 operands, zero-sized dimensions) are the
  ones of the shared library (`BroadcastCompatible`, `BroadcastShapeIs`,
  `ElementwiseBinaryRel`); they are not restated here.
-/

namespace Generated.Mul

open Sonnx

/-! ## Real -/

/-- `realMulStep a b` is the exact product of the two real numbers. -/
def realMulStep (a b : Rat) : Rat := a * b

/-- `realMul A B` is the elementwise product of the real tensors `A` and `B` under
multidirectional (Numpy-style) broadcasting; `none` when the shapes are not
broadcast-compatible. -/
def realMul (A B : Tensor Rat) : Option (Tensor Rat) := binary realMulStep A B

/-! 
@trace E_MUL_REAL_CONSTR_A_0010
@source ops/mul/mul.md:L121-L123
@coverage specified
@anchor E_MUL_REAL_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_MUL_REAL_CONSTR_A_0010"></a>
  - `[E_MUL_REAL_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_MUL_REAL_CONSTR_B_0010
@source ops/mul/mul.md:L131-L132
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_MUL_REAL_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_REAL_CONSTR_A_0010</span></b>](#E_MUL_REAL_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_MUL_REAL_CONSTR_C_0010
@source ops/mul/mul.md:L142-L143
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_MUL_REAL_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/


/-! 
@trace E_MUL_REAL_FUNC_0010
@source ops/mul/mul.md:L33-L56
@coverage specified
@anchor E_MUL_REAL_FUNC_0010
@declaration Generated.Mul.RealMulRel
Informal source excerpt:
  <a id="E_MUL_REAL_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_MUL_REAL_FUNC_0010]</br></span>
  
  Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise and stores the result in tensor $C$. Each element of $C$ is the product of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \tilde{A}[i] * \tilde{B}[i]
  $$
  
  where
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$,
  - $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$.
  
  The multiplication is exact: $C[i]$ is the product in $\mathbb{R}$ of the two elements, with no approximation, and it is defined for every pair of real numbers.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the product of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-- [E_MUL_REAL_FUNC_0010] For every index `i` of the result `C`, the element `C[i]` is the
product in `Rat` of the elements of `A` and of `B` that correspond to `i` under the
broadcasting rules; `C` has the broadcast shape of `A` and `B`. -/
def RealMulRel (A B C : Tensor Rat) : Prop :=
  ElementwiseBinaryRel (fun a b c => c = a * b) A B C

/-! 
@trace real:example:1
@source ops/mul/mul.md:L60-L70
@coverage example_only
@declaration Generated.Mul.realExample1
Informal source excerpt:
  ### Example 1
  
  $$
  A = \begin{bmatrix} 1.5 & 9.5 & 35.5 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 2 & 3 & -4 \end{bmatrix}
  $$
  
  $$
  C = A * B = \begin{bmatrix} 3 & 28.5 & -142 \end{bmatrix}
  $$
-/

def realA1 : Tensor Rat :=
  { shape := [3], data := [(3 : Rat) / 2, (19 : Rat) / 2, (71 : Rat) / 2] }

def realB1 : Tensor Rat := { shape := [3], data := [(2 : Rat), 3, -4] }

def realC1 : Tensor Rat :=
  { shape := [3], data := [(3 : Rat), (57 : Rat) / 2, -142] }

/-- `A = [1.5, 9.5, 35.5]`, `B = [2, 3, -4]`, `C = [3, 28.5, -142]`. -/
def realExample1 : Prop := RealMulRel realA1 realB1 realC1

/-! 
@trace real:example:2
@source ops/mul/mul.md:L72-L96
@coverage example_only
@declaration Generated.Mul.realExample2
Informal source excerpt:
  ### Example 2
  
  $$
  A = \begin{bmatrix}
    3 & 4.5 \\
    16 & 0.5 \\
    25.5 & 24
  \end{bmatrix}
  \quad
  B = \begin{bmatrix} 2 & -1 \end{bmatrix}
  $$
  
  $$
  C = A * B = \begin{bmatrix}
    3*2 & 4.5*(-1) \\
    16*2 & 0.5*(-1) \\
    25.5*2 & 24*(-1)
  \end{bmatrix} = \begin{bmatrix}
    6 & -4.5 \\
    32 & -0.5 \\
    51 & -24
  \end{bmatrix}
  $$
  
  Tensor $B$ has rank $1$ and tensor $A$ has rank $2$; $B$ is broadcast to the shape of $A$, so that each column of $A$ is multiplied by the corresponding element of $B$.
-/

def realA2 : Tensor Rat :=
  { shape := [3, 2],
    data := [(3 : Rat), (9 : Rat) / 2, 16, (1 : Rat) / 2, (51 : Rat) / 2, 24] }

def realB2 : Tensor Rat := { shape := [2], data := [(2 : Rat), -1] }

def realC2 : Tensor Rat :=
  { shape := [3, 2],
    data := [(6 : Rat), -(9 : Rat) / 2, 32, -(1 : Rat) / 2, 51, -24] }

/-- `B` of rank 1 is broadcast over the rows of the rank-2 tensor `A`. -/
def realExample2 : Prop := RealMulRel realA2 realB2 realC2

/-! ## Floating point -/

/-- Is the value a NaN? -/
def fIsNaN : FVal → Bool
  | .nan => true
  | _ => false

/-- Is the value an infinity? -/
def fIsInf : FVal → Bool
  | .inf _ => true
  | _ => false

/-- Is the value a zero? -/
def fIsZero : FVal → Bool
  | .zero _ => true
  | _ => false

/-- The sign bit of a value (`true` = negative); `nan` has no sign of its own. -/
def fNeg : FVal → Bool
  | .nan => false
  | .inf s => s
  | .zero s => s
  | .finite s _ => s

/-- The magnitude of a finite value, `0` for the other values. -/
def fMag : FVal → Rat
  | .finite _ m => m
  | _ => 0

/-- The absolute value of a rational. -/
def rAbs (x : Rat) : Rat := if 0 ≤ x then x else -x

/-- `2 ^ n` as a rational. -/
def pow2 (n : Nat) : Rat := (2 : Rat) ^ n

/-- `RepMag p r`: the nonnegative rational `r` has a `p`-bit significand, the exponent range
being unbounded. The value `0` is representable. -/
def RepMag (p : Nat) (r : Rat) : Prop :=
  r = 0 ∨
    ∃ (k j : Nat),
      2 ^ (p - 1) ≤ k ∧ k < 2 ^ p ∧
        (r = (k : Rat) * pow2 j ∨ r = (k : Rat) / pow2 j)

/-- `RepMagEven p r`: as `RepMag`, with an even significand (the tie-breaking rule). -/
def RepMagEven (p : Nat) (r : Rat) : Prop :=
  r = 0 ∨
    ∃ (k j : Nat),
      2 ^ (p - 1) ≤ k ∧ k < 2 ^ p ∧ k % 2 = 0 ∧
        (r = (k : Rat) * pow2 j ∨ r = (k : Rat) / pow2 j)

/-- `RoundMagNearest p m r`: `r` is the magnitude `m` rounded to the nearest magnitude with
a `p`-bit significand, ties to even. -/
def RoundMagNearest (p : Nat) (m r : Rat) : Prop :=
  RepMag p r ∧ 0 ≤ r ∧
    (∀ r', RepMag p r' → rAbs (r - m) ≤ rAbs (r' - m)) ∧
    (∀ r', RepMag p r' → rAbs (r - m) = rAbs (r' - m) → RepMagEven p r)

/-! 
@trace E_MUL_FLOAT_CONSTR_A_0010
@source ops/mul/mul.md:L278-L280
@coverage specified
@anchor E_MUL_FLOAT_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_MUL_FLOAT_CONSTR_A_0010"></a>
  - `[E_MUL_FLOAT_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_MUL_FLOAT_CONSTR_A_0020
@source ops/mul/mul.md:L281-L283
@coverage specified
@anchor E_MUL_FLOAT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_MUL_FLOAT_CONSTR_A_0020"></a>
  - `[E_MUL_FLOAT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$ and $C$ have the same type.
-/

/-! 
@trace E_MUL_FLOAT_CONSTR_B_0010
@source ops/mul/mul.md:L291-L292
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_MUL_FLOAT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_FLOAT_CONSTR_A_0010</span></b>](#E_MUL_FLOAT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_MUL_FLOAT_CONSTR_B_0020
@source ops/mul/mul.md:L293-L294
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_MUL_FLOAT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_FLOAT_CONSTR_A_0020</span></b>](#E_MUL_FLOAT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_MUL_FLOAT_CONSTR_C_0010
@source ops/mul/mul.md:L304-L305
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_MUL_FLOAT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_MUL_FLOAT_CONSTR_C_0020
@source ops/mul/mul.md:L306-L307
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_MUL_FLOAT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_FLOAT_CONSTR_A_0020</span></b>](#E_MUL_FLOAT_CONSTR_A_0020) on tensor $A$.
-/


/-- The IEEE 754 element step of `Mul` for the format `fmt`:

* NaN is produced when an operand is NaN, and by the invalid operation `0 * ∞`;
* an infinity of the sign given by the rule of signs is produced when one operand is an
  infinity and the other is neither a zero nor NaN;
* otherwise both operands are finite, and the exact product is rounded to the format, the
  sign of the exact product being attached to the result (so that an underflow may give a
  signed zero and an overflow an infinity of that sign).
-/
def floatMulStep (fmt : FloatFormat) (a b c : FVal) : Prop :=
  if fIsNaN a || fIsNaN b then c = FVal.nan
  else if (fIsZero a && fIsInf b) || (fIsInf a && fIsZero b) then c = FVal.nan
  else if fIsInf a || fIsInf b then c = FVal.inf (Bool.xor (fNeg a) (fNeg b))
  else
    ∃ r : Rat,
      RoundMagNearest fmt.precision (fMag a * fMag b) r ∧
        ((r = 0 ∧ c = FVal.zero (Bool.xor (fNeg a) (fNeg b))) ∨
         (0 < r ∧ r ≤ fmt.maxFinite ∧
           c = FVal.finite (Bool.xor (fNeg a) (fNeg b)) r) ∨
         (fmt.maxFinite < r ∧ c = FVal.inf (Bool.xor (fNeg a) (fNeg b))))

/-! 
@trace E_MUL_FLOAT_FUNC_0010
@source ops/mul/mul.md:L170-L208
@coverage specified
@anchor E_MUL_FLOAT_FUNC_0010
@declaration Generated.Mul.FloatMulRel
Informal source excerpt:
  <a id="E_MUL_FLOAT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_MUL_FLOAT_FUNC_0010]</br></span>
  
  Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $C$. Each element of $C$ is the floating-point product of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \tilde{A}[i] *_{(\text{f})} \tilde{B}[i] =
  \begin{cases}
  \text{NaN} & \text{if } \tilde{A}[i] \text{ or } \tilde{B}[i] \text{ is NaN, or if one of them is a zero and the other an infinity} \\
  \sigma \cdot \infty & \text{if neither is NaN, one of them is an infinity, and the other is neither a zero nor NaN} \\
  \text{round}(\tilde{A}[i] * \tilde{B}[i]) & \text{otherwise}
  \end{cases}
  $$
  
  where
  - $*_{(\text{f})}$ is the multiplication of the floating-point type of $A$, $B$ and $C$, i.e. $*_{(\text{f16})}$, $*_{(\text{f32})}$ or $*_{(\text{f64})}$ according to that type, whereas the $*$ of the third case is the multiplication of $\mathbb{R}$: there, $\tilde{A}[i] * \tilde{B}[i]$ is the exact product of the two elements,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $\sigma \in \{-1, +1\}$ is the sign given by the rule of signs: $\sigma = -1$ when the operands have opposite signs and $\sigma = +1$ otherwise, the sign of an infinity and the sign of a zero being signs like any other,
  - $\text{round}(x)$ is the value of $x$ rounded to the nearest value of the type of $A$ and $B$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded: a magnitude whose nearest value exceeds the largest finite value of the type of $A$ and $B$ gives an infinity of the sign of $x$.
  
  In the first case, an operand that is NaN is propagated, and the multiplication of a zero and an infinity is the invalid operation defined in IEEE 754 section 7.2; the result is NaN in both cases. The NaN of this case is a quiet NaN; its sign and its payload are not specified, whether the NaN comes from an operand or from the invalid operation, so that every quiet NaN of the type of $A$ and $B$ is a conforming result.
  
  In the second case, the product of an infinity and a nonzero finite value, or of two infinities, is an infinity whose sign is given by the rule of signs; the result involves no rounding.
  
  In the last case, both operands are finite, and the exact product is rounded as defined above. That product may be representable, in which case the result is that product and the multiplication is exact; it may be subnormal; and its magnitude may lie below the smallest subnormal value of the type of $A$ and $B$, in which case it rounds to a zero.
  
  **The sign of a null or infinite result.** The result is $-0$ or $-\infty$ when the operands have opposite signs, and $+0$ or $+\infty$ otherwise, the signs of $\pm 0$ and $\pm\infty$ being taken into account. In particular the product of two zeros is $-0$ when one of them is $-0$ and the other $+0$, and $+0$ when the two zeros have the same sign; and a nonzero exact product whose magnitude rounds to zero gives a zero of the sign of that exact product, so that an underflow may produce $-0$.
  
  **Values of the operands.** The operands may be any of the values of the type, including the special numbers $\pm 0$, $\pm\infty$ and NaN. The cases above give the result for every combination of them. Every finite value, zero and subnormal values included, is a possible operand and a possible result; the multiplication of two finite values never gives NaN.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the product of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-- [E_MUL_FLOAT_FUNC_0010] For every index `i` of the result `C`, the element `C[i]` is the
floating-point product of the elements of `A` and of `B` that correspond to `i` under the
broadcasting rules, in the floating-point type shared by `A`, `B` and `C`. -/
def FloatMulRel (fmt : FloatFormat) (A B C : Tensor FVal) : Prop :=
  ElementwiseBinaryRel (floatMulStep fmt) A B C

/-! 
@trace float:example:1
@source ops/mul/mul.md:L212-L224
@coverage example_only
@declaration Generated.Mul.floatExample1
Informal source excerpt:
  ### Example 1
  
  $$
  A = \begin{bmatrix} 3.0 & 4.5 & 16.0 & 0.5 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 3.0 & -2.0 & 0.25 & 4.0 \end{bmatrix}
  $$
  
  $$
  C = A *_{(\text{f})} B = \begin{bmatrix} 9.0 & -9.0 & 4.0 & 2.0 \end{bmatrix}
  $$
  
  The product of two numbers of a `float` type is not always exact; in this example every product is representable and $C$ is the exact result.
-/

def floatA1 : Tensor FVal :=
  { shape := [4],
    data := [FVal.finite false 3, FVal.finite false ((9 : Rat) / 2),
             FVal.finite false 16, FVal.finite false ((1 : Rat) / 2)] }

def floatB1 : Tensor FVal :=
  { shape := [4],
    data := [FVal.finite false 3, FVal.finite true 2,
             FVal.finite false ((1 : Rat) / 4), FVal.finite false 4] }

def floatC1 : Tensor FVal :=
  { shape := [4],
    data := [FVal.finite false 9, FVal.finite true 9,
             FVal.finite false 4, FVal.finite false 2] }

/-- Every product of the example is representable and `C` is the exact result. -/
def floatExample1 (fmt : FloatFormat) : Prop := FloatMulRel fmt floatA1 floatB1 floatC1

/-! 
@trace float:example:2
@source ops/mul/mul.md:L226-L238
@coverage example_only
@declaration Generated.Mul.floatExample2
Informal source excerpt:
  ### Example 2
  
  $$
  A = \begin{bmatrix} \text{+0} & \text{+inf} & \text{+0} & \text{-0} & \text{-inf} & \text{NaN} \end{bmatrix}
  \quad
  B = \begin{bmatrix} \text{+inf} & \text{-inf} & \text{-3.0} & \text{-2.0} & \text{+2.0} & \text{1.0} \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} \text{NaN} & \text{-inf} & \text{-0} & \text{+0} & \text{-inf} & \text{NaN} \end{bmatrix}
  $$
  
  The product of the zero and the infinity of the first column is the invalid operation of IEEE 754 section 7.2 and yields NaN. The product of the two infinities of opposite signs is $-\infty$, and so is the product of $-\infty$ and the positive finite value. The product of $+0$ and the negative value is $-0$, and the product of $-0$ and the negative value is $+0$: the sign of a null result follows the rule of signs.
-/

def floatA2 : Tensor FVal :=
  { shape := [6],
    data := [FVal.zero false, FVal.inf false, FVal.zero false,
             FVal.zero true, FVal.inf true, FVal.nan] }

def floatB2 : Tensor FVal :=
  { shape := [6],
    data := [FVal.inf false, FVal.inf true, FVal.finite true 3,
             FVal.finite true 2, FVal.finite false 2, FVal.finite false 1] }

def floatC2 : Tensor FVal :=
  { shape := [6],
    data := [FVal.nan, FVal.inf true, FVal.zero true,
             FVal.zero false, FVal.inf true, FVal.nan] }

/-- The invalid operation, the rule of signs on infinities and on zeros, and the
propagation of NaN. -/
def floatExample2 (fmt : FloatFormat) : Prop := FloatMulRel fmt floatA2 floatB2 floatC2

/-! 
@trace float:example:3
@source ops/mul/mul.md:L240-L252
@coverage deferred
@limitation The third element of the example requires the smallest subnormal of the format to be known, which the profile does not expose: Sonnx.FloatFormat carries only precision and maxFinite. The rounding formalised in RoundMagNearest follows the document's literal unbounded-exponent rule, so it yields a tiny nonzero value there instead of -0; the first two elements (rounded product and overflow to +inf) are covered.
Informal source excerpt:
  ### Example 3
  
  $$
  A \approx \begin{bmatrix} 0.1 & 1 \cdot 10^{308} & -1 \cdot 10^{-308} \end{bmatrix}
  \quad
  B \approx \begin{bmatrix} 0.2 & 10 & 1 \cdot 10^{-308} \end{bmatrix}
  $$
  
  $$
  C \approx \begin{bmatrix} 0.020000000000000004 & \text{+inf} & \text{-0} \end{bmatrix}
  $$
  
  The three operands are `double` values. The exact product of $0.1$ and $0.2$ is not representable in `double`; the first element of $C$ is that product rounded to the nearest `double`. The exact product of $1 \cdot 10^{308}$ and $10$ exceeds the largest finite value of `double`, and its nearest value with an unbounded exponent range lies beyond that value: the second element is an overflow, and its sign is the sign of the exact product. The exact product of the two subnormal values is negative and its magnitude lies below the smallest subnormal value of `double`: the third element is $-0$, the sign of the exact product preserved by the rounding.
-/


/-! ## Signed integers -/

/-- `intMulStep n a b c`: `c` is the value of the `n`-bit signed type congruent to the exact
product of `a` and `b` modulo `2 ^ n`. -/
def intMulStep (n : Nat) (a b c : Int) : Prop := c = sintWrap n (a * b)

/-! 
@trace E_MUL_INT_CONSTR_A_0010
@source ops/mul/mul.md:L409-L411
@coverage specified
@anchor E_MUL_INT_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_MUL_INT_CONSTR_A_0010"></a>
  - `[E_MUL_INT_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_MUL_INT_CONSTR_A_0020
@source ops/mul/mul.md:L412-L414
@coverage specified
@anchor E_MUL_INT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_MUL_INT_CONSTR_A_0020"></a>
  - `[E_MUL_INT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$ and $C$ have the same type.
-/

/-! 
@trace E_MUL_INT_CONSTR_B_0010
@source ops/mul/mul.md:L422-L423
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_MUL_INT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_INT_CONSTR_A_0010</span></b>](#E_MUL_INT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_MUL_INT_CONSTR_B_0020
@source ops/mul/mul.md:L424-L425
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_MUL_INT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_INT_CONSTR_A_0020</span></b>](#E_MUL_INT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_MUL_INT_CONSTR_C_0010
@source ops/mul/mul.md:L435-L436
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_MUL_INT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_MUL_INT_CONSTR_C_0020
@source ops/mul/mul.md:L437-L438
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_MUL_INT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_INT_CONSTR_A_0020</span></b>](#E_MUL_INT_CONSTR_A_0020) on tensor $A$.
-/


/-! 
@trace E_MUL_INT_FUNC_0010
@source ops/mul/mul.md:L334-L366
@coverage specified
@anchor E_MUL_INT_FUNC_0010
@declaration Generated.Mul.IntMulRel
Informal source excerpt:
  <a id="E_MUL_INT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_MUL_INT_FUNC_0010]</br></span>
  
  Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the product of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below. Tensors $A$, $B$ and $C$ have the same signed $n$-bit type, and the result is a value of that type.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] =
  \begin{cases}
  u & \text{if } u \le 2^{n-1}-1 \\
  u - 2^n & \text{if } u \ge 2^{n-1}
  \end{cases}
  \qquad \text{where } u = \left(\tilde{A}[i] * \tilde{B}[i]\right) \bmod 2^n
  $$
  
  where
  - $\tilde{A}[i] * \tilde{B}[i]$ is the exact product of the two elements, the product in $\mathbb{Z}$ of the two integers, before the reduction modulo $2^n$,
  - $x \bmod 2^n$ is the remainder of the Euclidean division of $x$ by $2^n$, taken in $[0, 2^n-1]$, i.e. the unique integer $u \in [0, 2^n-1]$ such that $x - u$ is a multiple of $2^n$,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $n$ is the number of bits of the type of $A$ and $B$.
  
  The result is thus the unique value of the type of $A$ and $B$ that is congruent to the exact product modulo $2^n$: the $2^n$ values of an $n$-bit signed type form a complete set of residues modulo $2^n$, so that exactly one of them satisfies the condition. The exact product of two values of the type lies in $\left[-2^{2n-2}+2^{n-1},\, 2^{2n-2}\right]$, and the reduction modulo $2^n$ is applied as many times as the magnitude of that product requires.
  
  **The result may differ from the exact product in magnitude and in sign.** The range of the type is not closed under multiplication, so that the product of two values of the type generally leaves it: for `int8`, $100 * 100$ gives $16$, $-100 * 100$ gives $-16$, and $100 * 3$ gives $44$. Tensors $A$, $B$ and $C$ have the same type and no wider type is used, so that the value of $C$ is that value of the type, sign included, and the multiplication is not monotonic on the signed integer types.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the product of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-- [E_MUL_INT_FUNC_0010] The elements of `A` and of `B` are values of the `n`-bit signed
type, and for every index `i` of the result, `C[i]` is the value of that type congruent to
the exact product of the corresponding elements of `A` and `B` modulo `2 ^ n`. The reduction
is applied elementwise and the product is taken in `Int`, before the reduction. -/
def IntMulRel (n : Nat) (A B C : Tensor Int) : Prop :=
  (∀ a ∈ A.data, InSignedRange n a) ∧
    (∀ b ∈ B.data, InSignedRange n b) ∧
      ElementwiseBinaryRel (intMulStep n) A B C

/-! 
@trace int:example:1
@source ops/mul/mul.md:L370-L384
@coverage example_only
@declaration Generated.Mul.intExample1
Informal source excerpt:
  ### Example 1
  
  For the `int8` type:
  
  $$
  A = \begin{bmatrix} -6 & 100 & -100 & 100 \end{bmatrix}
  \quad
  B = \begin{bmatrix} -3 & 100 & 100 & 3 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 18 & 16 & -16 & 44 \end{bmatrix}
  $$
  
  The product $-6 * (-3) = 18$ lies in $[-2^7, 2^7-1]$ and is its own remainder modulo $2^8$. The product $100 * 100 = 10000$ leaves that range: its remainder modulo $2^8$ is $16$, which is at most $2^7-1=127$ and is the result. The product $-100 * 100 = -10000$ has the remainder $240$ modulo $2^8$, which is at least $2^7=128$, so the result is $240-256=-16$. The product $100 * 3 = 300$ has the remainder $44$ modulo $2^8$, which is the result. The result of the first product has the sign of the exact product, whereas the results of the other three do not.
-/

def intA1 : Tensor Int := { shape := [4], data := [-6, 100, -100, 100] }

def intB1 : Tensor Int := { shape := [4], data := [-3, 100, 100, 3] }

def intC1 : Tensor Int := { shape := [4], data := [18, 16, -16, 44] }

/-- `int8` example: `100 * 100` gives `16`, `-100 * 100` gives `-16`, `100 * 3` gives `44`. -/
def intExample1 : Prop := IntMulRel 8 intA1 intB1 intC1

/-! ## Unsigned integers -/

/-- `uintMulStep n a b c`: `c` is the value of the `n`-bit unsigned type congruent to the
exact product of `a` and `b` modulo `2 ^ n`. -/
def uintMulStep (n : Nat) (a b c : Int) : Prop := c = uintWrap n (a * b)

/-! 
@trace E_MUL_UINT_CONSTR_A_0010
@source ops/mul/mul.md:L535-L537
@coverage specified
@anchor E_MUL_UINT_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_MUL_UINT_CONSTR_A_0010"></a>
  - `[E_MUL_UINT_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_MUL_UINT_CONSTR_A_0020
@source ops/mul/mul.md:L538-L540
@coverage specified
@anchor E_MUL_UINT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_MUL_UINT_CONSTR_A_0020"></a>
  - `[E_MUL_UINT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$ and $C$ have the same type.
-/

/-! 
@trace E_MUL_UINT_CONSTR_B_0010
@source ops/mul/mul.md:L548-L549
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_MUL_UINT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_UINT_CONSTR_A_0010</span></b>](#E_MUL_UINT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_MUL_UINT_CONSTR_B_0020
@source ops/mul/mul.md:L550-L551
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_MUL_UINT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_UINT_CONSTR_A_0020</span></b>](#E_MUL_UINT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_MUL_UINT_CONSTR_C_0010
@source ops/mul/mul.md:L561-L562
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_MUL_UINT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_MUL_UINT_CONSTR_C_0020
@source ops/mul/mul.md:L563-L564
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_MUL_UINT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_UINT_CONSTR_A_0020</span></b>](#E_MUL_UINT_CONSTR_A_0020) on tensor $A$.
-/


/-! 
@trace E_MUL_UINT_FUNC_0010
@source ops/mul/mul.md:L465-L492
@coverage specified
@anchor E_MUL_UINT_FUNC_0010
@declaration Generated.Mul.UintMulRel
Informal source excerpt:
  <a id="E_MUL_UINT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_MUL_UINT_FUNC_0010]</br></span>
  
  Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the product of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below. Tensors $A$, $B$ and $C$ have the same unsigned $n$-bit type, and the result is a value of that type.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \left(\tilde{A}[i] * \tilde{B}[i]\right) \bmod 2^n
  $$
  
  where
  - $\tilde{A}[i] * \tilde{B}[i]$ is the exact product of the two elements, the product in $\mathbb{Z}$ of the two integers, before the reduction modulo $2^n$,
  - $x \bmod 2^n$ is the remainder of the Euclidean division of $x$ by $2^n$, taken in $[0, 2^n-1]$, i.e. the unique integer $u \in [0, 2^n-1]$ such that $x - u$ is a multiple of $2^n$,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $n$ is the number of bits of the type of $A$ and $B$.
  
  The result is thus the unique value of the type of $A$ and $B$ that is congruent to the exact product modulo $2^n$: the $2^n$ values of an $n$-bit unsigned type are exactly the residues modulo $2^n$, so that the condition determines the result. The exact product of two values of the type lies in $\left[0,\, (2^n-1)^2\right]$, and the reduction modulo $2^n$ is applied as many times as the magnitude of that product requires.
  
  **The result may be smaller than both operands.** The product of two non-negative integers is at least as large as either of them when both are at least $1$, whereas the result here is reduced modulo $2^n$: for `uint8`, $200 * 100$ gives $32$, and $16 * 16$ gives $0$. Tensors $A$, $B$ and $C$ have the same type and no wider type is used, so that the value of $C$ is that value of the type, and the multiplication is not monotonic on the unsigned types.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the product of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-- [E_MUL_UINT_FUNC_0010] The elements of `A` and of `B` are values of the `n`-bit unsigned
type, and for every index `i` of the result, `C[i]` is the remainder modulo `2 ^ n` of the
exact product of the corresponding elements of `A` and `B`, taken in `[0, 2 ^ n - 1]`. -/
def UintMulRel (n : Nat) (A B C : Tensor Int) : Prop :=
  (∀ a ∈ A.data, InUnsignedRange n a) ∧
    (∀ b ∈ B.data, InUnsignedRange n b) ∧
      ElementwiseBinaryRel (uintMulStep n) A B C

/-! 
@trace uint:example:1
@source ops/mul/mul.md:L496-L510
@coverage example_only
@declaration Generated.Mul.uintExample1
Informal source excerpt:
  ### Example 1
  
  For the `uint8` type:
  
  $$
  A = \begin{bmatrix} 6 & 200 & 35 & 16 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 3 & 100 & 5 & 16 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 18 & 32 & 175 & 0 \end{bmatrix}
  $$
  
  The products $6 * 3 = 18$ and $35 * 5 = 175$ are at most $2^8-1=255$ and are their own remainders modulo $2^8$. The product $200 * 100 = 20000$ leaves the range of the type: its remainder modulo $2^8$ is $32$, which is smaller than both operands. The product $16 * 16 = 256$ is a multiple of $2^8$ and its remainder is $0$.
-/

def uintA1 : Tensor Int := { shape := [4], data := [6, 200, 35, 16] }

def uintB1 : Tensor Int := { shape := [4], data := [3, 100, 5, 16] }

def uintC1 : Tensor Int := { shape := [4], data := [18, 32, 175, 0] }

/-- `uint8` example: `200 * 100` gives `32`, `16 * 16` gives `0`. -/
def uintExample1 : Prop := UintMulRel 8 uintA1 uintB1 uintC1

end Generated.Mul

/-- The CGen descriptor: the C stage renders it into elementwise kernels. -/
def sonnxCGenOp : CGen.Elementwise := ⟨"mul", .mul⟩

-- Fully-qualified declarations named by the trace entries.
#check @Generated.Mul.FloatMulRel
#check @Generated.Mul.IntMulRel
#check @Generated.Mul.RealMulRel
#check @Generated.Mul.UintMulRel
#check @Generated.Mul.floatExample1
#check @Generated.Mul.floatExample2
#check @Generated.Mul.intExample1
#check @Generated.Mul.realExample1
#check @Generated.Mul.realExample2
#check @Generated.Mul.uintExample1
#check @Sonnx.BroadcastCompatible
#check @Sonnx.BroadcastShapeIs
#check @Sonnx.TypeConsistent

-- Generated by the SONNX pipeline: derive the C from the formal specification.
open Lean Elab Command in
run_cmd do
  match ← CGen.deriveElementwise "mul" [
    ⟨.real, "Generated.Mul.RealMulRel", "", "E_MUL_REAL_FUNC_0010", "ops/mul/mul.md:L33-L56", []⟩,
    ⟨.float, "Generated.Mul.FloatMulRel", "", "E_MUL_FLOAT_FUNC_0010", "ops/mul/mul.md:L170-L208", []⟩,
    ⟨.int, "Generated.Mul.IntMulRel", "", "E_MUL_INT_FUNC_0010", "ops/mul/mul.md:L334-L366", []⟩,
    ⟨.uint, "Generated.Mul.UintMulRel", "", "E_MUL_UINT_FUNC_0010", "ops/mul/mul.md:L465-L492", []⟩] with
  | .error reason => IO.println s!"SONNX-CGEN-ERROR\t{reason}"
  | .ok d =>
      IO.println s!"SONNX-CGEN name={d.op.name} binop={CGen.binOpName d.op.op}"
      for s in d.derivations do
        let ops := ",".intercalate s.ops
        IO.println s!"SONNX-CGEN-SOURCE\t{CGen.familyName s.family}\t{s.declaration}\t{s.witness}\t{s.requirement}\t{s.source}\t{ops}"
      if d.op.op == .add || d.op.op == .sub || d.op.op == .mul then
        IO.FS.writeFile "/run/media/eric/Work1/Work/gits/sonnx/ops/mul/c/sonnx_kernels.h" (CGen.renderFile (CGen.headerFileForDerived [d]))
        IO.FS.writeFile "/run/media/eric/Work1/Work/gits/sonnx/ops/mul/c/sonnx_kernels.c" (CGen.renderFile (CGen.sourceFileForDerived [d]))
