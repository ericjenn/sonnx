import CGen.Print
import CGen.Derive
import Sonnx
import CGen
/-!
@generated-by sonnx-lean-pipeline
@source ops/sub/sub.md
@source-digest 7a65bd1dc1764a81e266a011e54a56c3
@library-digest 3162109b4c926c2fa5dd5f8bd2eb2d90fb17a0480eccc0216a4ea9bd2b43ffab
Compilation and trace validity are separate from semantic completeness; see the trace report.
-/


/-!
DECISION: `FVal.inf`, `FVal.zero` and `FVal.finite` carry a sign bit in which `true`
denotes the negative sign and `false` the positive sign; `FVal.finite neg mag` stores a
non-negative magnitude.  This is the sign convention used by the element relation below.

DECISION: the signed and unsigned integer families are modelled by `Int` values constrained
to `InSignedRange n` and `InUnsignedRange n`; the modular result is `sintWrap n` and
`uintWrap n`.  This keeps the specification at the level of the library's arithmetic bounds
while representing the same n-bit values.

DECISION: `FVal.toRat` is partial (`Option Rat`) because NaN and infinities have no rational
value.  The finite element step matches on the two rational values; the infinities and NaNs
are handled by the preceding cases of `FloatSubRel`.

DECISION: the floating-point element step uses `IsRepresentable` for the exact finite case and
`RoundToFormat` relationally for the non-representable case.  `IsRepresentable fmt d` is a
`Prop` without a decidable instance, so the two finite cases are stated as a disjunction
instead of an `if`-`then`-`else`.

DECISION: the sign of a null difference is read from the shape of the operands: it is `-0`
exactly when the minuend is the value `-0` and the subtrahend the value `+0`, and `+0`
otherwise.  A finite operand whose magnitude is zero is assumed to be presented as
`FVal.zero` (a well-formed `FVal.finite` has a strictly positive magnitude); this is stated
as the shape of the operands only, as the document specifies.
-/

namespace Generated.Sub

open Sonnx

/-! ## Real subtraction -/

/-! 
@trace E_SUB_REAL_FUNC_0010
@source ops/sub/sub.md:L33-L56
@coverage specified
@anchor E_SUB_REAL_FUNC_0010
@declaration Generated.Sub.realSub
Informal source excerpt:
  <a id="E_SUB_REAL_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_SUB_REAL_FUNC_0010]</br></span>
  
  Operator **Sub** subtracts tensor $B$ from tensor $A$ element-wise and stores the result in tensor $C$. Each element of $C$ is the difference of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \tilde{A}[i] - \tilde{B}[i]
  $$
  
  where
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$,
  - $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$.
  
  The subtraction is exact: $C[i]$ is the difference in $\mathbb{R}$ of the two elements, with no approximation.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the difference of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

def realSub (A B : Tensor Rat) : Option (Tensor Rat) :=
  binary (fun a b => a - b) A B

/-! 
@trace E_SUB_REAL_CONSTR_A_0010
@source ops/sub/sub.md:L130-L132
@coverage specified
@anchor E_SUB_REAL_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_SUB_REAL_CONSTR_A_0010"></a>
  - `[E_SUB_REAL_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_SUB_REAL_CONSTR_B_0010
@source ops/sub/sub.md:L140-L141
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_SUB_REAL_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_REAL_CONSTR_A_0010</span></b>](#E_SUB_REAL_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_SUB_REAL_CONSTR_C_0010
@source ops/sub/sub.md:L151-L152
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_SUB_REAL_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/


def realExample1A : Tensor Rat :=
  { shape := [3], data := [(61 / 10 : Rat), (19 / 2 : Rat), (357 / 10 : Rat)] }

def realExample1B : Tensor Rat :=
  { shape := [3], data := [(3 : Rat), (33 / 10 : Rat), (51 / 10 : Rat)] }

def realExample1C : Tensor Rat :=
  { shape := [3], data := [(31 / 10 : Rat), (31 / 5 : Rat), (153 / 5 : Rat)] }

/-! 
@trace real:example:1
@source ops/sub/sub.md:L60-L72
@coverage example_only
@declaration Generated.Sub.realExample1
Informal source excerpt:
  ### Example 1
  
  $$
  A = \begin{bmatrix} 6.1 & 9.5 & 35.7 \end{bmatrix}
  $$
  
  $$
  B = \begin{bmatrix} 3 & 3.3 & 5.1 \end{bmatrix}
  $$
  
  $$
  C = A - B = \begin{bmatrix} 6.1-3 & 9.5-3.3 & 35.7-5.1 \end{bmatrix} = \begin{bmatrix} 3.1 & 6.2 & 30.6 \end{bmatrix}
  $$
-/

def realExample1 : Prop :=
  realSub realExample1A realExample1B = some realExample1C

def realExample2A : Tensor Rat :=
  { shape := [3, 2],
    data := [(37 / 10 : Rat), (22 / 5 : Rat), (81 / 5 : Rat),
             (1 / 2 : Rat), (253 / 10 : Rat), (124 / 5 : Rat)] }

def realExample2B : Tensor Rat :=
  { shape := [2], data := [(1 : Rat), (2 : Rat)] }

def realExample2C : Tensor Rat :=
  { shape := [3, 2],
    data := [(27 / 10 : Rat), (12 / 5 : Rat), (76 / 5 : Rat),
             (-(3 / 2) : Rat), (243 / 10 : Rat), (114 / 5 : Rat)] }

/-! 
@trace real:example:2
@source ops/sub/sub.md:L74-L98
@coverage example_only
@declaration Generated.Sub.realExample2
Informal source excerpt:
  ### Example 2
  
  $$
  A = \begin{bmatrix}
    3.7 & 4.4 \\
    16.2 & 0.5 \\
    25.3 & 24.8
  \end{bmatrix}
  \quad
  B = \begin{bmatrix} 1 & 2 \end{bmatrix}
  $$
  
  $$
  C = A - B = \begin{bmatrix}
    3.7-1 & 4.4-2 \\
    16.2-1 & 0.5-2 \\
    25.3-1 & 24.8-2
  \end{bmatrix} = \begin{bmatrix}
    2.7 & 2.4 \\
    15.2 & -1.5 \\
    24.3 & 22.8
  \end{bmatrix}
  $$
  
  Tensor $B$ has rank $1$ and tensor $A$ has rank $2$; $B$ is broadcast to the shape of $A$, so that each column of $A$ is decreased by the corresponding element of $B$.
-/

def realExample2 : Prop :=
  realSub realExample2A realExample2B = some realExample2C

def realExample3A : Tensor Rat :=
  { shape := [2, 2],
    data := [(5 : Rat), (13 / 4 : Rat), (4 : Rat), (7 / 4 : Rat)] }

def realExample3B : Tensor Rat :=
  { shape := [], data := [(5 / 2 : Rat)] }

def realExample3C : Tensor Rat :=
  { shape := [2, 2],
    data := [(5 / 2 : Rat), (3 / 4 : Rat), (3 / 2 : Rat), (-(3 / 4) : Rat)] }

/-! 
@trace real:example:3
@source ops/sub/sub.md:L100-L112
@coverage example_only
@declaration Generated.Sub.realExample3
Informal source excerpt:
  ### Example 3
  
  $$
  A = \begin{bmatrix} 5 & 3.25 \\ 4 & 1.75 \end{bmatrix}
  \quad
  B = 2.5
  $$
  
  $$
  C = A - B = \begin{bmatrix} 2.5 & 0.75 \\ 1.5 & -0.75 \end{bmatrix}
  $$
  
  Tensor $B$ is a rank-0 tensor; it is broadcast to the shape of $A$.
-/

def realExample3 : Prop :=
  realSub realExample3A realExample3B = some realExample3C

/-! ## Floating-point subtraction -/

/-- The element step of floating-point subtraction for two finite operands: a null exact
difference gives a signed zero, a representable exact difference gives that value exactly,
and a non-representable exact difference is rounded with roundTiesToEven. -/
def finiteSubRel (fmt : FloatFormat) (a b c : FVal) : Prop :=
  match FVal.toRat a, FVal.toRat b with
  | some x, some y =>
      let d : Rat := x - y
      (d = 0 ∧
        c = FVal.zero (match a, b with
          | FVal.zero true, FVal.zero false => true
          | _, _ => false)) ∨
      (d ≠ 0 ∧ IsRepresentable fmt d ∧
        c = FVal.finite (decide (d < 0)) (if d < 0 then -d else d)) ∨
      (d ≠ 0 ∧ ¬ IsRepresentable fmt d ∧ RoundToFormat fmt d c)
  | _, _ => False

/-- The IEEE 754 element step of `Sub`: NaN propagation and the invalid operation, the
infinite results, and the finite cases of `finiteSubRel`. -/
def FloatSubRel (fmt : FloatFormat) (a b c : FVal) : Prop :=
  match a, b with
  | FVal.nan, _ => c = FVal.nan
  | _, FVal.nan => c = FVal.nan
  | FVal.inf sa, FVal.inf sb =>
      (sa = sb ∧ c = FVal.nan) ∨
      (sa ≠ sb ∧ c = FVal.inf sa)
  | FVal.inf sa, _ => c = FVal.inf sa
  | _, FVal.inf sb => c = FVal.inf (Bool.not sb)
  | _, _ => finiteSubRel fmt a b c

/-! 
@trace E_SUB_FLOAT_FUNC_0010
@source ops/sub/sub.md:L179-L223
@coverage specified
@anchor E_SUB_FLOAT_FUNC_0010
@declaration Generated.Sub.floatSub
Informal source excerpt:
  <a id="E_SUB_FLOAT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_SUB_FLOAT_FUNC_0010]</br></span>
  
  Operator **Sub** subtracts tensor $B$ from tensor $A$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $C$. Each element of $C$ is the difference of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \tilde{A}[i] -_{(\text{f})} \tilde{B}[i] =
  \begin{cases}
  \text{NaN} & \text{if } \tilde{A}[i] \text{ or } \tilde{B}[i] \text{ is NaN, or if they are both } +\infty \text{ or both } -\infty \\
  \text{sign}\left(\tilde{A}[i] - \tilde{B}[i]\right) \cdot \infty & \text{if at least one operand is infinite, the two not being infinities of the same sign} \\
  -0 & \text{if both operands are finite, their exact difference is null, } \tilde{A}[i] \text{ is } -0 \text{ and } \tilde{B}[i] \text{ is } +0 \\
  +0 & \text{if both operands are finite and their exact difference is null, the previous case excepted} \\
  x & \text{if both operands are finite and their exact difference } x \text{ is representable in the type of } A \text{ and } B \\
  \text{round}(x) & \text{if both operands are finite and } x \text{ is not representable in the type of } A \text{ and } B
  \end{cases}
  $$
  
  where
  - $-_{(\text{f})}$ is the subtraction of the floating-point type of $A$, $B$ and $C$, i.e. $-_{(\text{f16})}$, $-_{(\text{f32})}$ or $-_{(\text{f64})}$ according to that type,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $x$ is the exact difference of the two finite operands, i.e. their difference in $\mathbb{R}$, which is a real number and carries no sign of zero,
  - $\text{sign}(y) \cdot \infty$ is $+\infty$ if $y \gt 0$ and $-\infty$ if $y \lt 0$, where $y = \tilde{A}[i] - \tilde{B}[i]$ is the difference of the two operands taken in $\overline{\mathbb{R}} = \mathbb{R} \cup \{-\infty, +\infty\}$: that difference is infinite and never null in this case,
  - $\text{round}(x)$ is the value of $x$ rounded to the nearest value of the type of $A$ and $B$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded: a magnitude whose nearest value exceeds the largest finite value of the type of $A$ and $B$ gives an infinity of the sign of $x$.
  
  In the first case, an operand that is NaN is propagated, and the subtraction of two infinities of the same sign is the invalid operation defined in IEEE 754 section 7.2; the result is NaN in both cases. The NaN of this case is a quiet NaN; its sign and its payload are not specified, whether the NaN comes from an operand or from the invalid operation, so that every quiet NaN of the type of $A$ and $B$ is a conforming result.
  
  In the second case the difference is infinite and the result is that infinity, exactly and without rounding: $+\infty$ for $\tilde{A}[i] = +\infty$ with a finite $\tilde{B}[i]$, for a finite $\tilde{A}[i]$ with $\tilde{B}[i] = -\infty$, and for $\tilde{A}[i] = +\infty$ with $\tilde{B}[i] = -\infty$; $-\infty$ for the three opposite combinations.
  
  In the last case the exact difference is not representable in the type of $A$ and $B$: the result is that difference rounded as defined above, and its sign is the sign of the exact difference. The rounding gives either the largest finite value of the type of that sign, when the exact difference lies below the midpoint between that value and the next power of two of the type, or an infinity of that sign.
  
  When the exact difference is null, the result is $+0$, except when $\tilde{A}[i]$ is $-0$ and $\tilde{B}[i]$ is $+0$, in which case it is $-0$: the differences $+0 - (+0)$, $+0 - (-0)$ and $(-0)-(-0)$ give $+0$. A non-null exact difference that would round to a null value gives a zero whose sign is the sign of the exact difference, i.e. $-0$ only when the exact difference is negative; the values of the type of $A$ and $B$ are all multiples of its smallest subnormal value, so the exact difference of two of them is a multiple of that value as well, and this case does not arise for two finite operands of the type.
  
  **Subnormal results.** A non-null exact difference whose magnitude is below the smallest normal value of the type of $A$ and $B$ is a multiple of its smallest subnormal value, hence a subnormal value of the type; the case that gives $x$ gives it exactly, with no rounding.
  
  **Values of the operands.** The operands may be any of the values of the type, including the special numbers $\pm 0$, $\pm\text{inf}$ and NaN. The cases above give the result for every combination of them.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the difference of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

def floatSub (fmt : FloatFormat) (A B C : Tensor FVal) : Prop :=
  ElementwiseBinaryRel (FloatSubRel fmt) A B C

/-! 
@trace E_SUB_FLOAT_CONSTR_A_0010
@source ops/sub/sub.md:L291-L293
@coverage specified
@anchor E_SUB_FLOAT_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_SUB_FLOAT_CONSTR_A_0010"></a>
  - `[E_SUB_FLOAT_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_SUB_FLOAT_CONSTR_A_0020
@source ops/sub/sub.md:L294-L296
@coverage specified
@anchor E_SUB_FLOAT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_SUB_FLOAT_CONSTR_A_0020"></a>
  - `[E_SUB_FLOAT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$, and $C$ have the same type.
-/

/-! 
@trace E_SUB_FLOAT_CONSTR_B_0010
@source ops/sub/sub.md:L304-L305
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_SUB_FLOAT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_CONSTR_A_0010</span></b>](#E_SUB_FLOAT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_SUB_FLOAT_CONSTR_B_0020
@source ops/sub/sub.md:L306-L307
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_SUB_FLOAT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_CONSTR_A_0020</span></b>](#E_SUB_FLOAT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_SUB_FLOAT_CONSTR_C_0010
@source ops/sub/sub.md:L317-L318
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_SUB_FLOAT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_SUB_FLOAT_CONSTR_C_0020
@source ops/sub/sub.md:L319-L320
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_SUB_FLOAT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_CONSTR_A_0020</span></b>](#E_SUB_FLOAT_CONSTR_A_0020) on tensor $A$.
-/


def exampleFormat : FloatFormat :=
  FloatFormat.ofKind FloatKind.f64

def floatExample1A : Tensor FVal :=
  { shape := [3, 2],
    data := [FVal.finite false (3 : Rat), FVal.finite false (9 / 2 : Rat),
             FVal.finite false (16 : Rat), FVal.finite false (1 : Rat),
             FVal.finite false (51 / 2 : Rat), FVal.finite false (97 / 4 : Rat)] }

def floatExample1B : Tensor FVal :=
  { shape := [3, 2],
    data := [FVal.finite false (3 : Rat), FVal.finite false (2 : Rat),
             FVal.finite false (4 : Rat), FVal.zero false,
             FVal.finite false (5 : Rat), FVal.finite false (4 : Rat)] }

def floatExample1C : Tensor FVal :=
  { shape := [3, 2],
    data := [FVal.zero false, FVal.finite false (5 / 2 : Rat),
             FVal.finite false (12 : Rat), FVal.finite false (1 : Rat),
             FVal.finite false (41 / 2 : Rat), FVal.finite false (81 / 4 : Rat)] }

/-! 
@trace float:example:1
@source ops/sub/sub.md:L227-L239
@coverage example_only
@declaration Generated.Sub.floatExample1
Informal source excerpt:
  ### Example 1
  
  $$
  A = \begin{bmatrix} 3.0 & 4.5 \\ 16.0 & 1.0 \\ 25.5 & 24.25 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 3.0 & 2.0 \\ 4.0 & 0.0 \\ 5.0 & 4.0 \end{bmatrix}
  $$
  
  $$
  C = A -_{(\text{f})} B = \begin{bmatrix} 0.0 & 2.5 \\ 12.0 & 1.0 \\ 20.5 & 20.25 \end{bmatrix}
  $$
  
  The difference of two numbers of a `float` type is not always exact; in this example every difference is representable and $C$ is the exact result.
-/

def floatExample1 : Prop :=
  floatSub exampleFormat floatExample1A floatExample1B floatExample1C

def floatExample2A : Tensor FVal :=
  { shape := [4],
    data := [FVal.finite false (1 : Rat), FVal.inf false,
             FVal.zero true, FVal.zero false] }

def floatExample2B : Tensor FVal :=
  { shape := [4],
    data := [FVal.finite false (1 : Rat), FVal.inf false,
             FVal.zero false, FVal.zero true] }

def floatExample2C : Tensor FVal :=
  { shape := [4],
    data := [FVal.zero false, FVal.nan, FVal.zero true, FVal.zero false] }

/-! 
@trace float:example:2
@source ops/sub/sub.md:L241-L253
@coverage example_only
@declaration Generated.Sub.floatExample2
Informal source excerpt:
  ### Example 2
  
  $$
  A = \begin{bmatrix} 1.0 & \text{+inf} & \text{-0} & \text{+0} \end{bmatrix}
  \quad
  B = \begin{bmatrix} 1.0 & \text{+inf} & \text{+0} & \text{-0} \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 0.0 & \text{NaN} & \text{-0} & \text{+0} \end{bmatrix}
  $$
  
  The subtraction of the two infinities is the invalid operation of IEEE 754 section 7.2 and yields NaN. The difference of $-0$ and $+0$ is $-0$, the only pair of null operands whose null difference is $-0$; the difference of $+0$ and $-0$ is $+0$.
-/

def floatExample2 : Prop :=
  floatSub exampleFormat floatExample2A floatExample2B floatExample2C

def floatExample3A : Tensor FVal :=
  { shape := [2],
    data := [FVal.finite false (3 / 10 : Rat), FVal.finite false (1 : Rat)] }

def floatExample3B : Tensor FVal :=
  { shape := [2],
    data := [FVal.finite false (1 / 10 : Rat), FVal.finite false (1 : Rat)] }

/-! 
@trace float:example:3
@source ops/sub/sub.md:L255-L267
@coverage example_only
@declaration Generated.Sub.floatExample3
Informal source excerpt:
  ### Example 3
  
  $$
  A = \begin{bmatrix} 0.3 & 1.0 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 0.1 & 1.0 \end{bmatrix}
  $$
  
  $$
  C \approx \begin{bmatrix} 0.19999999999999998 & 0.0 \end{bmatrix}
  $$
  
  The exact difference of the two `double` values $0.3$ and $0.1$ is not representable in `double`; the first element of $C$ is that difference rounded to the nearest `double`. The second element is the null difference of two equal values, which is $+0$.
-/

def floatExample3 : Prop :=
  ∃ C : Tensor FVal,
    floatSub exampleFormat floatExample3A floatExample3B C ∧
    C.read [1] = some (FVal.zero false)

/-! ## Signed integer subtraction -/

def intSub (n : Nat) (A B : Tensor Int) : Option (Tensor Int) :=
  binary (fun a b => sintWrap n (a - b)) A B

def tensorInSignedRange (n : Nat) (A : Tensor Int) : Prop :=
  ∀ x ∈ A.data, InSignedRange n x

/-! 
@trace E_SUB_INT_FUNC_0010
@source ops/sub/sub.md:L347-L378
@coverage specified
@anchor E_SUB_INT_FUNC_0010
@declaration Generated.Sub.intSubSpec
Informal source excerpt:
  <a id="E_SUB_INT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_SUB_INT_FUNC_0010]</br></span>
  
  Operator **Sub** subtracts tensor $B$ from tensor $A$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the difference of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below. Tensors $A$, $B$ and $C$ have the same signed $n$-bit type, and the result is a value of that type.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] =
  \begin{cases}
  \tilde{A}[i] - \tilde{B}[i] & \text{if the difference lies in } [-2^{n-1},\, 2^{n-1}-1] \\
  \tilde{A}[i] - \tilde{B}[i] + 2^n & \text{if the difference is lower than } -2^{n-1} \\
  \tilde{A}[i] - \tilde{B}[i] - 2^n & \text{if the difference is greater than } 2^{n-1}-1
  \end{cases}
  $$
  
  where
  - $\tilde{A}[i] - \tilde{B}[i]$ is the exact difference of the two elements, the difference in $\mathbb{Z}$ of the two integers, before the reduction modulo $2^n$,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $n$ is the number of bits of the type of $A$ and $B$.
  
  The result is thus the exact difference modulo $2^n$, read as a signed value. The values of an $n$-bit signed type lie in $[-2^{n-1}, 2^{n-1}-1]$, so that their exact difference lies in $[-2^n+1, 2^n-1]$ and one of the three cases above always applies, the reduction being applied at most once.
  
  **The sign of the result.** The range of the type is not symmetric about zero, so that the result of the subtraction of two values of opposite signs can leave the range and take the sign of the subtrahend: for `int8`, $127-(-1)$ gives $-128$ and $-128-1$ gives $127$. Tensors $A$, $B$ and $C$ have the same type and no wider type is used, so that the value of $C$ is that value of the type, sign included.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the difference of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

def intSubSpec (n : Nat) (A B C : Tensor Int) : Prop :=
  tensorInSignedRange n A ∧
  tensorInSignedRange n B ∧
  intSub n A B = some C

/-! 
@trace E_SUB_INT_CONSTR_A_0010
@source ops/sub/sub.md:L432-L434
@coverage specified
@anchor E_SUB_INT_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_SUB_INT_CONSTR_A_0010"></a>
  - `[E_SUB_INT_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_SUB_INT_CONSTR_A_0020
@source ops/sub/sub.md:L435-L437
@coverage specified
@anchor E_SUB_INT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_SUB_INT_CONSTR_A_0020"></a>
  - `[E_SUB_INT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$, and $C$ have the same type.
-/

/-! 
@trace E_SUB_INT_CONSTR_B_0010
@source ops/sub/sub.md:L445-L446
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_SUB_INT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_INT_CONSTR_A_0010</span></b>](#E_SUB_INT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_SUB_INT_CONSTR_B_0020
@source ops/sub/sub.md:L447-L448
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_SUB_INT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_INT_CONSTR_A_0020</span></b>](#E_SUB_INT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_SUB_INT_CONSTR_C_0010
@source ops/sub/sub.md:L458-L459
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_SUB_INT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_SUB_INT_CONSTR_C_0020
@source ops/sub/sub.md:L460-L461
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_SUB_INT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_INT_CONSTR_A_0020</span></b>](#E_SUB_INT_CONSTR_A_0020) on tensor $A$.
-/


def intExample1A : Tensor Int :=
  { shape := [3], data := [(6 : Int), (5 : Int), (-35 : Int)] }

def intExample1B : Tensor Int :=
  { shape := [3], data := [(3 : Int), (3 : Int), (3 : Int)] }

def intExample1C : Tensor Int :=
  { shape := [3], data := [(3 : Int), (2 : Int), (-38 : Int)] }

/-! 
@trace int:example:1
@source ops/sub/sub.md:L382-L394
@coverage example_only
@declaration Generated.Sub.intExample1
Informal source excerpt:
  ### Example 1
  
  $$
  A = \begin{bmatrix} 6 & 5 & -35 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 3 & 3 & 3 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 3 & 2 & -38 \end{bmatrix}
  $$
  
  Every difference lies in the range of `int8` and is given by the first case of the definition.
-/

def intExample1 : Prop :=
  intSub 8 intExample1A intExample1B = some intExample1C

def intExample2A : Tensor Int :=
  { shape := [3], data := [(127 : Int), (-128 : Int), (0 : Int)] }

def intExample2B : Tensor Int :=
  { shape := [3], data := [(-1 : Int), (1 : Int), (1 : Int)] }

def intExample2C : Tensor Int :=
  { shape := [3], data := [(-128 : Int), (127 : Int), (-1 : Int)] }

/-! 
@trace int:example:2
@source ops/sub/sub.md:L396-L410
@coverage example_only
@declaration Generated.Sub.intExample2
Informal source excerpt:
  ### Example 2
  
  For the `int8` type:
  
  $$
  A = \begin{bmatrix} 127 & -128 & 0 \end{bmatrix}
  \quad
  B = \begin{bmatrix} -1 & 1 & 1 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} -128 & 127 & -1 \end{bmatrix}
  $$
  
  The difference $127-(-1)=128$ is greater than the maximum value $2^7-1=127$ of `int8`, so the third case of the definition applies and the result is $128-2^8=-128$. The difference $-128-1=-129$ is lower than the minimum value $-2^7=-128$, so the second case applies and the result is $-129+2^8=127$. The difference $0-1=-1$ lies in the range of the type and is given by the first case.
-/

def intExample2 : Prop :=
  intSub 8 intExample2A intExample2B = some intExample2C

/-! ## Unsigned integer subtraction -/

def uintSub (n : Nat) (A B : Tensor Int) : Option (Tensor Int) :=
  binary (fun a b => uintWrap n (a - b)) A B

def tensorInUnsignedRange (n : Nat) (A : Tensor Int) : Prop :=
  ∀ x ∈ A.data, InUnsignedRange n x

/-! 
@trace E_SUB_UINT_FUNC_0010
@source ops/sub/sub.md:L488-L518
@coverage specified
@anchor E_SUB_UINT_FUNC_0010
@declaration Generated.Sub.uintSubSpec
Informal source excerpt:
  <a id="E_SUB_UINT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_SUB_UINT_FUNC_0010]</br></span>
  
  Operator **Sub** subtracts tensor $B$ from tensor $A$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the difference of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below. Tensors $A$, $B$ and $C$ have the same unsigned $n$-bit type, and the result is a value of that type.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] =
  \begin{cases}
  \tilde{A}[i] - \tilde{B}[i] & \text{if the difference is at least } 0 \\
  \tilde{A}[i] - \tilde{B}[i] + 2^n & \text{if the difference is lower than } 0
  \end{cases}
  $$
  
  where
  - $\tilde{A}[i] - \tilde{B}[i]$ is the exact difference of the two elements, the difference in $\mathbb{Z}$ of the two integers, before the reduction modulo $2^n$,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $n$ is the number of bits of the type of $A$ and $B$.
  
  The values of an $n$-bit unsigned type are non-negative and at most $2^n-1$, so that their exact difference lies in $[-(2^n-1), 2^n-1]$: the second case applies exactly when $\tilde{A}[i] \lt \tilde{B}[i]$, and the reduction is applied at most once.
  
  **The result may be larger than the minuend.** The difference of two non-negative integers is at most the minuend, whereas the result here is reduced modulo $2^n$: for `uint8`, $0-1$ gives $255$, a value larger than both operands, and $4-7$ gives $253$. The subtraction is not monotonic on the unsigned types, and the result is not the value of a signed difference.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the difference of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

def uintSubSpec (n : Nat) (A B C : Tensor Int) : Prop :=
  tensorInUnsignedRange n A ∧
  tensorInUnsignedRange n B ∧
  uintSub n A B = some C

/-! 
@trace E_SUB_UINT_CONSTR_A_0010
@source ops/sub/sub.md:L558-L560
@coverage specified
@anchor E_SUB_UINT_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_SUB_UINT_CONSTR_A_0010"></a>
  - `[E_SUB_UINT_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_SUB_UINT_CONSTR_A_0020
@source ops/sub/sub.md:L561-L563
@coverage specified
@anchor E_SUB_UINT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_SUB_UINT_CONSTR_A_0020"></a>
  - `[E_SUB_UINT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$, and $C$ have the same type.
-/

/-! 
@trace E_SUB_UINT_CONSTR_B_0010
@source ops/sub/sub.md:L571-L572
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_SUB_UINT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_UINT_CONSTR_A_0010</span></b>](#E_SUB_UINT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_SUB_UINT_CONSTR_B_0020
@source ops/sub/sub.md:L573-L574
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_SUB_UINT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_UINT_CONSTR_A_0020</span></b>](#E_SUB_UINT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_SUB_UINT_CONSTR_C_0010
@source ops/sub/sub.md:L584-L585
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_SUB_UINT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_SUB_UINT_CONSTR_C_0020
@source ops/sub/sub.md:L586-L587
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_SUB_UINT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_UINT_CONSTR_A_0020</span></b>](#E_SUB_UINT_CONSTR_A_0020) on tensor $A$.
-/


def uintExample1A : Tensor Int :=
  { shape := [3], data := [(0 : Int), (255 : Int), (4 : Int)] }

def uintExample1B : Tensor Int :=
  { shape := [3], data := [(1 : Int), (1 : Int), (7 : Int)] }

def uintExample1C : Tensor Int :=
  { shape := [3], data := [(255 : Int), (254 : Int), (253 : Int)] }

/-! 
@trace uint:example:1
@source ops/sub/sub.md:L522-L536
@coverage example_only
@declaration Generated.Sub.uintExample1
Informal source excerpt:
  ### Example 1
  
  For the `uint8` type:
  
  $$
  A = \begin{bmatrix} 0 & 255 & 4 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 1 & 1 & 7 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 255 & 254 & 253 \end{bmatrix}
  $$
  
  The differences $0-1=-1$ and $4-7=-3$ are lower than $0$, so the second case of the definition applies and the results are $-1+2^8=255$ and $-3+2^8=253$; the first of them is larger than both operands. The difference $255-1=254$ is at least $0$ and is given by the first case.
-/

def uintExample1 : Prop :=
  uintSub 8 uintExample1A uintExample1B = some uintExample1C

end Generated.Sub

/-- The CGen descriptor: the C stage renders it into elementwise kernels. -/
def sonnxCGenOp : CGen.Elementwise := ⟨"sub", .sub⟩

-- Fully-qualified declarations named by the trace entries.
#check @Generated.Sub.floatExample1
#check @Generated.Sub.floatExample2
#check @Generated.Sub.floatExample3
#check @Generated.Sub.floatSub
#check @Generated.Sub.intExample1
#check @Generated.Sub.intExample2
#check @Generated.Sub.intSubSpec
#check @Generated.Sub.realExample1
#check @Generated.Sub.realExample2
#check @Generated.Sub.realExample3
#check @Generated.Sub.realSub
#check @Generated.Sub.uintExample1
#check @Generated.Sub.uintSubSpec
#check @Sonnx.BroadcastCompatible
#check @Sonnx.BroadcastShapeIs
#check @Sonnx.TypeConsistent

-- Generated by the SONNX pipeline: derive the C from the formal specification.
open Lean Elab Command in
run_cmd do
  match ← CGen.deriveElementwise "sub" [
    ⟨.real, "Generated.Sub.realSub", "", "E_SUB_REAL_FUNC_0010", "ops/sub/sub.md:L33-L56", []⟩,
    ⟨.float, "Generated.Sub.floatSub", "", "E_SUB_FLOAT_FUNC_0010", "ops/sub/sub.md:L179-L223", []⟩,
    ⟨.int, "Generated.Sub.intSubSpec", "", "E_SUB_INT_FUNC_0010", "ops/sub/sub.md:L347-L378", []⟩,
    ⟨.uint, "Generated.Sub.uintSubSpec", "", "E_SUB_UINT_FUNC_0010", "ops/sub/sub.md:L488-L518", []⟩] with
  | .error reason => IO.println s!"SONNX-CGEN-ERROR\t{reason}"
  | .ok d =>
      IO.println s!"SONNX-CGEN name={d.op.name} binop={CGen.binOpName d.op.op}"
      for s in d.derivations do
        let ops := ",".intercalate s.ops
        IO.println s!"SONNX-CGEN-SOURCE\t{CGen.familyName s.family}\t{s.declaration}\t{s.witness}\t{s.requirement}\t{s.source}\t{ops}"
      if d.op.op == .add || d.op.op == .sub || d.op.op == .mul then
        IO.FS.writeFile "/run/media/eric/Work1/Work/gits/sonnx/ops/sub/c/sonnx_kernels.h" (CGen.renderFile (CGen.headerFileForDerived [d]))
        IO.FS.writeFile "/run/media/eric/Work1/Work/gits/sonnx/ops/sub/c/sonnx_kernels.c" (CGen.renderFile (CGen.sourceFileForDerived [d]))
