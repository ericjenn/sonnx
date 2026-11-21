import Sonnx
import CGen
/-!
@generated-by sonnx-lean-pipeline
@source ops/add/add.md
@source-digest 5b1b9072f77e85ea0b1a18b0d48b9f35
@library-digest 3162109b4c926c2fa5dd5f8bd2eb2d90fb17a0480eccc0216a4ea9bd2b43ffab
Compilation and trace validity are separate from semantic completeness; see the trace report.
-/


/-!
DECISION: The real family is modelled over `Rat`, the exact ordered field of the profile, so that
the addition of `E_ADD_REAL_FUNC_0010` is exact and total; the examples are rationals.

DECISION: The signed integer family is modelled over `Int`, with the operand values constrained by
`InSignedRange n` and the element step `sintWrap n (x + y)`, i.e. the exact sum reduced modulo
`2 ^ n` and read back in `[-2 ^ (n-1), 2 ^ (n-1) - 1]`.  The three cases of the informal definition
(exact sum in range, below the minimum, above the maximum) are the three branches of `sintWrap`.

DECISION: The unsigned integer family is modelled over `Nat`, with the operand values constrained
by `x < 2 ^ n` and the element step `(x + y) % 2 ^ n`; on non-negative operands this is exactly the
value of the `n`-bit unsigned type congruent to the exact sum modulo `2 ^ n`.

DECISION: The floating-point family is modelled over `Sonnx.FVal` for a single `FloatFormat`, so
that "A, B and C have the same type" (constraint `..._CONSTR_..._0020`) is the format parameter
shared by the three tensors.  The element step is a ternary relation, not a function, because the
rounded case is given by `RoundToFormat` (roundTiesToEven with an unbounded exponent range).

DECISION: In `FVal`, the `Bool` field of `zero` and of `inf` is read as `true = negative`,
`false = positive`; this only fixes which constructor is used in the examples and does not affect
any relation.

DECISION: Float example 3 prints the double `0.30000000000000004`, which is not available as a
literal of `Rat`.  The example is therefore stated with the operands existentially quantified as
the `f64` roundings of `0.1` and `0.2`, so that the first element of `C` is the rounding of the
exact sum of those two doubles, namely the value the document prints; the second and the third
elements, which are exact, are stated literally.
-/

open Sonnx

namespace Generated.Add

/-! ### Real family -/

/-! 
@trace E_ADD_REAL_CONSTR_A_0010
@source ops/add/add.md:L114-L116
@coverage specified
@anchor E_ADD_REAL_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_ADD_REAL_CONSTR_A_0010"></a>
  - `[E_ADD_REAL_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_ADD_REAL_CONSTR_B_0010
@source ops/add/add.md:L124-L125
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_ADD_REAL_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_REAL_CONSTR_A_0010</span></b>](#E_ADD_REAL_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_ADD_REAL_CONSTR_C_0010
@source ops/add/add.md:L135-L136
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_ADD_REAL_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_ADD_REAL_FUNC_0010
@source ops/add/add.md:L33-L56
@coverage specified
@anchor E_ADD_REAL_FUNC_0010
@declaration Generated.Add.realFunction
Informal source excerpt:
  <a id="E_ADD_REAL_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_ADD_REAL_FUNC_0010]</br></span>
  
  Operator **Add** adds tensor $B$ to tensor $A$ element-wise and stores the result in tensor $C$. Each element of $C$ is the sum of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \tilde{A}[i] + \tilde{B}[i]
  $$
  
  where
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$,
  - $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$.
  
  The addition is exact: $C[i]$ is the sum in $\mathbb{R}$ of the two elements, with no approximation.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the sum of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-- The real-valued `Add`: exact elementwise addition with multidirectional broadcasting.

`realFunction A B C` holds when `A` and `B` are broadcast-compatible, `C` has their broadcast
shape and the data of `C` is the elementwise sum of the broadcast operands, computed in `Rat`
without any rounding. -/
def realFunction (A B C : Tensor Rat) : Prop :=
  BroadcastCompatible A B ∧
  BroadcastShapeIs A B C ∧
  binary (fun x y => x + y) A B = some C

/-! 
@trace real:example:1
@source ops/add/add.md:L60-L70
@coverage example_only
@declaration Generated.Add.realExample1
Informal source excerpt:
  ### Example 1
  
  $$
  A = \begin{bmatrix} 6.1 & 9.5 & 35.7 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 2 & 3 & 4 \end{bmatrix}
  $$
  
  $$
  C = A + B = \begin{bmatrix} 8.1 & 12.5 & 39.7 \end{bmatrix}
  $$
-/

def realA1 : Tensor Rat :=
  { shape := [3],
    data := [(61 : Rat) / 10, (19 : Rat) / 2, (357 : Rat) / 10] }

def realB1 : Tensor Rat :=
  { shape := [3], data := [(2 : Rat), (3 : Rat), (4 : Rat)] }

def realC1 : Tensor Rat :=
  { shape := [3],
    data := [(81 : Rat) / 10, (25 : Rat) / 2, (397 : Rat) / 10] }

def realExample1 : Prop := realFunction realA1 realB1 realC1

/-! 
@trace real:example:2
@source ops/add/add.md:L72-L96
@coverage example_only
@declaration Generated.Add.realExample2
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
  C = A + B = \begin{bmatrix}
    3.7+1 & 4.4+2 \\
    16.2+1 & 0.5+2 \\
    25.3+1 & 24.8+2
  \end{bmatrix} = \begin{bmatrix}
    4.7 & 6.4 \\
    17.2 & 2.5 \\
    26.3 & 26.8
  \end{bmatrix}
  $$
  
  Tensor $B$ has rank $1$ and tensor $A$ has rank $2$; $B$ is broadcast to the shape of $A$, so that each column of $A$ is increased by the corresponding element of $B$.
-/

def realA2 : Tensor Rat :=
  { shape := [3, 2],
    data := [(37 : Rat) / 10, (44 : Rat) / 10,
             (162 : Rat) / 10, (5 : Rat) / 10,
             (253 : Rat) / 10, (248 : Rat) / 10] }

def realB2 : Tensor Rat :=
  { shape := [2], data := [(1 : Rat), (2 : Rat)] }

def realC2 : Tensor Rat :=
  { shape := [3, 2],
    data := [(47 : Rat) / 10, (64 : Rat) / 10,
             (172 : Rat) / 10, (25 : Rat) / 10,
             (263 : Rat) / 10, (268 : Rat) / 10] }

def realExample2 : Prop := realFunction realA2 realB2 realC2

/-! ### Floating-point family -/

def posFinite (q : Rat) : FVal := FVal.finite false q
def negFinite (q : Rat) : FVal := FVal.finite true q
def posZero : FVal := FVal.zero false
def negZero : FVal := FVal.zero true
def posInf : FVal := FVal.inf false
def negInf : FVal := FVal.inf true

/-- The IEEE 754 addition step of `E_ADD_FLOAT_FUNC_0010`, as a ternary relation.

NaN is propagated; the sum of two infinities of the same sign is that infinity, of opposite signs
it is NaN; an infinity plus a finite value is that infinity; a null exact sum is `+0` unless both
operands are `-0`; and any other exact sum is rounded to the format with roundTiesToEven.

The last branch of the inner match is unreachable: after the constructor patterns above, both
operands are zero or finite, for which `FVal.toRat` is defined.  It is stated as `False` so that no
result is invented for an operand that has no real value. -/
def floatAddRel (fmt : FloatFormat) (x y r : FVal) : Prop :=
  match x, y with
  | .nan, _ => r = .nan
  | _, .nan => r = .nan
  | .inf sx, .inf sy => if sx = sy then r = .inf sx else r = .nan
  | .inf sx, _ => r = .inf sx
  | _, .inf sy => r = .inf sy
  | x, y =>
      match x.toRat, y.toRat with
      | some qx, some qy =>
          if qx + qy = 0 then
            (match x, y with
             | .zero true, .zero true => r = .zero true
             | _, _ => r = .zero false)
          else RoundToFormat fmt (qx + qy) r
      | _, _ => False

/-! 
@trace E_ADD_FLOAT_CONSTR_A_0010
@source ops/add/add.md:L266-L268
@coverage specified
@anchor E_ADD_FLOAT_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_ADD_FLOAT_CONSTR_A_0010"></a>
  - `[E_ADD_FLOAT_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_ADD_FLOAT_CONSTR_A_0020
@source ops/add/add.md:L269-L271
@coverage specified
@anchor E_ADD_FLOAT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_ADD_FLOAT_CONSTR_A_0020"></a>
  - `[E_ADD_FLOAT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$, and $C$ have the same type.
-/

/-! 
@trace E_ADD_FLOAT_CONSTR_B_0010
@source ops/add/add.md:L279-L280
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_ADD_FLOAT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_FLOAT_CONSTR_A_0010</span></b>](#E_ADD_FLOAT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_ADD_FLOAT_CONSTR_B_0020
@source ops/add/add.md:L281-L282
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_ADD_FLOAT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_FLOAT_CONSTR_A_0020</span></b>](#E_ADD_FLOAT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_ADD_FLOAT_CONSTR_C_0010
@source ops/add/add.md:L292-L293
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_ADD_FLOAT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_ADD_FLOAT_CONSTR_C_0020
@source ops/add/add.md:L294-L295
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_ADD_FLOAT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_FLOAT_CONSTR_A_0020</span></b>](#E_ADD_FLOAT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_ADD_FLOAT_FUNC_0010
@source ops/add/add.md:L163-L198
@coverage specified
@anchor E_ADD_FLOAT_FUNC_0010
@declaration Generated.Add.floatFunction
Informal source excerpt:
  <a id="E_ADD_FLOAT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_ADD_FLOAT_FUNC_0010]</br></span>
  
  Operator **Add** adds tensor $B$ to tensor $A$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $C$. Each element of $C$ is the floating-point sum of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \tilde{A}[i] +_{(\text{f})} \tilde{B}[i] =
  \begin{cases}
  \text{NaN} & \text{if } \tilde{A}[i] \text{ or } \tilde{B}[i] \text{ is NaN, or if they are } +\infty \text{ and } -\infty \\
  \tilde{A}[i] + \tilde{B}[i] & \text{if the exact sum is representable in the type of } A \text{ and } B \\
  \text{round}(\tilde{A}[i] + \tilde{B}[i]) & \text{otherwise}
  \end{cases}
  $$
  
  where
  - $+_{(\text{f})}$ is the addition of the floating-point type of $A$, $B$ and $C$, i.e. $+_{(\text{f16})}$, $+_{(\text{f32})}$ or $+_{(\text{f64})}$ according to that type, whereas the $+$ of the second and the third case is the addition of $\mathbb{R}$: there, $\tilde{A}[i] + \tilde{B}[i]$ is the exact sum of the two elements,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $\text{round}(x)$ is the value of $x$ rounded to the nearest value of the type of $A$ and $B$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded: a magnitude whose nearest value exceeds the largest finite value of the type of $A$ and $B$ gives an infinity of the sign of $x$.
  
  In the first case, an operand that is NaN is propagated, and the addition of an infinity and an infinity of the opposite sign is the invalid operation defined in IEEE 754 section 7.2; the result is NaN in both cases. The NaN of this case is a quiet NaN; its sign and its payload are not specified, whether the NaN comes from an operand or from the invalid operation, so that every quiet NaN of the type of $A$ and $B$ is a conforming result.
  
  In the last case, the exact sum is not representable in the type of $A$ and $B$: the result is that sum rounded as defined above. The rounding may make the result subnormal; its sign is then the sign of the exact sum.
  
  When the exact sum is null, the result is $+0$, except when $\tilde{A}[i]$ and $\tilde{B}[i]$ are both $-0$, in which case it is $-0$.
  
  **Values of the operands.** The operands may be any of the values of the type, including the special numbers $\pm 0$, $\pm\text{inf}$ and NaN. The cases above give the result for every combination of them. The addition of two infinities of the same sign gives that infinity, and the addition of an infinity and a finite value gives that infinity without rounding.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the sum of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-- The floating-point `Add` for one format: `A` and `B` are broadcast-compatible, `C` has their
broadcast shape, and every element of `C` is related to the corresponding broadcast elements of
`A` and `B` by the IEEE 754 addition `floatAddRel fmt`.

Since `A`, `B` and `C` are all `Tensor FVal` and the format `fmt` is shared, the three tensors have
the same type, as required by the type-consistency constraints. -/
def floatFunction (fmt : FloatFormat) (A B C : Tensor FVal) : Prop :=
  BroadcastCompatible A B ∧
  BroadcastShapeIs A B C ∧
  ElementwiseBinaryRel (floatAddRel fmt) A B C

def floatFmt : FloatFormat := FloatFormat.ofKind .f64

/-! 
@trace float:example:1
@source ops/add/add.md:L202-L214
@coverage example_only
@declaration Generated.Add.floatExample1
Informal source excerpt:
  ### Example 1
  
  $$
  A = \begin{bmatrix} 3.0 & 4.5 \\ 16.0 & 1.0 \\ 25.5 & 24.25 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 3.0 & 2.0 \\ 4.0 & 0.0 \\ 5.0 & 4.0 \end{bmatrix}
  $$
  
  $$
  C = A +_{(\text{f})} B = \begin{bmatrix} 6.0 & 6.5 \\ 20.0 & 1.0 \\ 30.5 & 28.25 \end{bmatrix}
  $$
  
  The sum of two numbers of a `float` type is not always exact; in this example every sum is representable and $C$ is the exact result.
-/

def floatA1 : Tensor FVal :=
  { shape := [3, 2],
    data := [posFinite (3 : Rat), posFinite ((9 : Rat) / 2),
             posFinite (16 : Rat), posFinite (1 : Rat),
             posFinite ((51 : Rat) / 2), posFinite ((97 : Rat) / 4)] }

def floatB1 : Tensor FVal :=
  { shape := [3, 2],
    data := [posFinite (3 : Rat), posFinite (2 : Rat),
             posFinite (4 : Rat), posZero,
             posFinite (5 : Rat), posFinite (4 : Rat)] }

def floatC1 : Tensor FVal :=
  { shape := [3, 2],
    data := [posFinite (6 : Rat), posFinite ((13 : Rat) / 2),
             posFinite (20 : Rat), posFinite (1 : Rat),
             posFinite ((61 : Rat) / 2), posFinite ((113 : Rat) / 4)] }

def floatExample1 : Prop := floatFunction floatFmt floatA1 floatB1 floatC1

/-! 
@trace float:example:2
@source ops/add/add.md:L216-L228
@coverage example_only
@declaration Generated.Add.floatExample2
Informal source excerpt:
  ### Example 2
  
  $$
  A = \begin{bmatrix} 1.0 & \text{+inf} & \text{+0} & \text{-inf} \end{bmatrix}
  \quad
  B = \begin{bmatrix} 1.0 & \text{-inf} & \text{-0} & \text{-inf} \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 2.0 & \text{NaN} & \text{+0} & \text{-inf} \end{bmatrix}
  $$
  
  The addition of the two infinities of opposite signs is the invalid operation of IEEE 754 section 7.2 and yields NaN. The sum of $+0$ and $-0$ is $+0$, the sign of a null sum being $+0$ unless both operands are $-0$. The addition of the two negative infinities gives the negative infinity.
-/

def floatA2 : Tensor FVal :=
  { shape := [4],
    data := [posFinite (1 : Rat), posInf, posZero, negInf] }

def floatB2 : Tensor FVal :=
  { shape := [4],
    data := [posFinite (1 : Rat), negInf, negZero, negInf] }

def floatC2 : Tensor FVal :=
  { shape := [4],
    data := [posFinite (2 : Rat), FVal.nan, posZero, negInf] }

def floatExample2 : Prop := floatFunction floatFmt floatA2 floatB2 floatC2

/-! 
@trace float:example:3
@source ops/add/add.md:L230-L242
@coverage example_only
@declaration Generated.Add.floatExample3
Informal source excerpt:
  ### Example 3
  
  $$
  A = \begin{bmatrix} 0.1 & 1.0 & 1.0 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 0.2 & 2.0 & \text{-1.0} \end{bmatrix}
  $$
  
  $$
  C \approx \begin{bmatrix} 0.30000000000000004 & 3.0 & 0.0 \end{bmatrix}
  $$
  
  The exact sum of the two `double` values $0.1$ and $0.2$ is not representable in `double`; the first element of $C$ is that sum rounded to the nearest `double`. The last element is the null sum of two values of opposite signs, which is $+0$.
-/

def floatExample3 : Prop :=
  ∃ a0 b0 c0 : FVal,
    RoundToFormat floatFmt ((1 : Rat) / 10) a0 ∧
    RoundToFormat floatFmt ((2 : Rat) / 10) b0 ∧
    floatFunction floatFmt
      { shape := [3], data := [a0, posFinite (1 : Rat), posFinite (1 : Rat)] }
      { shape := [3], data := [b0, posFinite (2 : Rat), posFinite (-(1 : Rat))] }
      { shape := [3], data := [c0, posFinite (3 : Rat), posZero] }

/-! ### Signed integer family -/

/-! 
@trace E_ADD_INT_CONSTR_A_0010
@source ops/add/add.md:L393-L395
@coverage specified
@anchor E_ADD_INT_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_ADD_INT_CONSTR_A_0010"></a>
  - `[E_ADD_INT_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_ADD_INT_CONSTR_A_0020
@source ops/add/add.md:L396-L398
@coverage specified
@anchor E_ADD_INT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_ADD_INT_CONSTR_A_0020"></a>
  - `[E_ADD_INT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$, and $C$ have the same type.
-/

/-! 
@trace E_ADD_INT_CONSTR_B_0010
@source ops/add/add.md:L406-L407
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_ADD_INT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_INT_CONSTR_A_0010</span></b>](#E_ADD_INT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_ADD_INT_CONSTR_B_0020
@source ops/add/add.md:L408-L409
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_ADD_INT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_INT_CONSTR_A_0020</span></b>](#E_ADD_INT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_ADD_INT_CONSTR_C_0010
@source ops/add/add.md:L419-L420
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_ADD_INT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_ADD_INT_CONSTR_C_0020
@source ops/add/add.md:L421-L422
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_ADD_INT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_INT_CONSTR_A_0020</span></b>](#E_ADD_INT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_ADD_INT_FUNC_0010
@source ops/add/add.md:L322-L353
@coverage specified
@anchor E_ADD_INT_FUNC_0010
@declaration Generated.Add.intFunction
Informal source excerpt:
  <a id="E_ADD_INT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_ADD_INT_FUNC_0010]</br></span>
  
  Operator **Add** adds tensor $B$ to tensor $A$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the sum of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below. Tensors $A$, $B$ and $C$ have the same signed $n$-bit type, and the result is a value of that type.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] =
  \begin{cases}
  \tilde{A}[i] + \tilde{B}[i] & \text{if the sum lies in } [-2^{n-1},\, 2^{n-1}-1] \\
  \tilde{A}[i] + \tilde{B}[i] + 2^n & \text{if the sum is lower than } -2^{n-1} \\
  \tilde{A}[i] + \tilde{B}[i] - 2^n & \text{if the sum is greater than } 2^{n-1}-1
  \end{cases}
  $$
  
  where
  - $\tilde{A}[i] + \tilde{B}[i]$ is the exact sum of the two elements, the sum in $\mathbb{Z}$ of the two integers, before the reduction modulo $2^n$,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $n$ is the number of bits of the type of $A$ and $B$.
  
  The result is thus the exact sum modulo $2^n$, read as a signed value. The values of an $n$-bit signed type lie in $[-2^{n-1}, 2^{n-1}-1]$, so that their exact sum lies in $[-2^n, 2^n-2]$ and one of the three cases above always applies, the reduction being applied at most once.
  
  **The sign of the result.** The range of the type is not symmetric about zero, so that the addition of two values of the same sign can give a result of the opposite sign: for `int8`, $100+100$ gives $-56$ and $-100+(-100)$ gives $56$. Tensors $A$, $B$ and $C$ have the same type and no wider type is used, so that the value of $C$ is that value of the type, sign included.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the sum of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-- The signed `n`-bit integer `Add`: the elements of `A`, `B` and `C` lie in the signed `n`-bit
range, `A` and `B` are broadcast-compatible, `C` has their broadcast shape, and every element of
`C` is the exact sum of the broadcast operands reduced modulo `2 ^ n` and read as a signed value. -/
def intFunction (n : Nat) (A B C : Tensor Int) : Prop :=
  (∀ x ∈ A.data, InSignedRange n x) ∧
  (∀ x ∈ B.data, InSignedRange n x) ∧
  (∀ x ∈ C.data, InSignedRange n x) ∧
  BroadcastCompatible A B ∧
  BroadcastShapeIs A B C ∧
  binary (fun x y => sintWrap n (x + y)) A B = some C

/-! 
@trace int:example:1
@source ops/add/add.md:L357-L371
@coverage example_only
@declaration Generated.Add.intExample1
Informal source excerpt:
  ### Example 1
  
  For the `int8` type:
  
  $$
  A = \begin{bmatrix} -6 & 100 & -100 \end{bmatrix}
  \quad
  B = \begin{bmatrix} -3 & 100 & -100 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} -9 & -56 & 56 \end{bmatrix}
  $$
  
  The sum $-6+(-3)=-9$ lies in $[-2^7, 2^7-1]$ and is given by the first case of the definition. The sum $100+100=200$ is greater than the maximum value $2^7-1=127$ of `int8`, so the third case applies and the result is $200-2^8=-56$. The sum $-100+(-100)=-200$ is lower than the minimum value $-2^7=-128$ of `int8`, so the second case applies and the result is $-200+2^8=56$. The result of the first overflow is negative although both operands are positive, and the result of the second is positive although both operands are negative: the sign of the operands is not preserved by the operator.
-/

def intA1 : Tensor Int :=
  { shape := [3], data := [-6, 100, -100] }

def intB1 : Tensor Int :=
  { shape := [3], data := [-3, 100, -100] }

def intC1 : Tensor Int :=
  { shape := [3], data := [-9, -56, 56] }

def intExample1 : Prop := intFunction 8 intA1 intB1 intC1

/-! ### Unsigned integer family -/

/-! 
@trace E_ADD_UINT_CONSTR_A_0010
@source ops/add/add.md:L519-L521
@coverage specified
@anchor E_ADD_UINT_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_ADD_UINT_CONSTR_A_0010"></a>
  - `[E_ADD_UINT_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_ADD_UINT_CONSTR_A_0020
@source ops/add/add.md:L522-L524
@coverage specified
@anchor E_ADD_UINT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_ADD_UINT_CONSTR_A_0020"></a>
  - `[E_ADD_UINT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$, and $C$ have the same type.
-/

/-! 
@trace E_ADD_UINT_CONSTR_B_0010
@source ops/add/add.md:L532-L533
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_ADD_UINT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_UINT_CONSTR_A_0010</span></b>](#E_ADD_UINT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_ADD_UINT_CONSTR_B_0020
@source ops/add/add.md:L534-L535
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_ADD_UINT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_UINT_CONSTR_A_0020</span></b>](#E_ADD_UINT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_ADD_UINT_CONSTR_C_0010
@source ops/add/add.md:L545-L546
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_ADD_UINT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_ADD_UINT_CONSTR_C_0020
@source ops/add/add.md:L547-L548
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_ADD_UINT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_UINT_CONSTR_A_0020</span></b>](#E_ADD_UINT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_ADD_UINT_FUNC_0010
@source ops/add/add.md:L449-L479
@coverage specified
@anchor E_ADD_UINT_FUNC_0010
@declaration Generated.Add.uintFunction
Informal source excerpt:
  <a id="E_ADD_UINT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_ADD_UINT_FUNC_0010]</br></span>
  
  Operator **Add** adds tensor $B$ to tensor $A$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the sum of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below. Tensors $A$, $B$ and $C$ have the same unsigned $n$-bit type, and the result is a value of that type.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] =
  \begin{cases}
  \tilde{A}[i] + \tilde{B}[i] & \text{if the sum is at most } 2^n-1 \\
  \tilde{A}[i] + \tilde{B}[i] - 2^n & \text{if the sum is greater than } 2^n-1
  \end{cases}
  $$
  
  where
  - $\tilde{A}[i] + \tilde{B}[i]$ is the exact sum of the two elements, the sum in $\mathbb{Z}$ of the two integers, before the reduction modulo $2^n$,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $n$ is the number of bits of the type of $A$ and $B$.
  
  The result is thus the exact sum modulo $2^n$, read as an unsigned value. The values of an $n$-bit unsigned type are non-negative, so that their exact sum lies in $[0, 2^{n+1}-2]$ and is never lower than the range of the type: the second case is the only one in which the sum leaves that range, and the first case applies as soon as the sum is at most $2^n-1$.
  
  **The result may be smaller than both operands.** The sum of two non-negative integers is never smaller than either of them, whereas the result here is reduced modulo $2^n$: for `uint8`, $200+100$ gives $44$, a value smaller than both operands. Tensors $A$, $B$ and $C$ have the same type and no wider type is used, so that the value of $C$ is that value of the type, and the addition is not monotonic on the unsigned types.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the sum of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-- The unsigned `n`-bit integer `Add`: the elements of `A`, `B` and `C` lie in `[0, 2 ^ n - 1]`,
`A` and `B` are broadcast-compatible, `C` has their broadcast shape, and every element of `C` is
the exact sum of the broadcast operands reduced modulo `2 ^ n`. -/
def uintFunction (n : Nat) (A B C : Tensor Nat) : Prop :=
  (∀ x ∈ A.data, x < 2 ^ n) ∧
  (∀ x ∈ B.data, x < 2 ^ n) ∧
  (∀ x ∈ C.data, x < 2 ^ n) ∧
  BroadcastCompatible A B ∧
  BroadcastShapeIs A B C ∧
  binary (fun x y => (x + y) % (2 ^ n)) A B = some C

/-! 
@trace uint:example:1
@source ops/add/add.md:L483-L497
@coverage example_only
@declaration Generated.Add.uintExample1
Informal source excerpt:
  ### Example 1
  
  For the `uint8` type:
  
  $$
  A = \begin{bmatrix} 6 & 200 & 35 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 3 & 100 & 5 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 9 & 44 & 40 \end{bmatrix}
  $$
  
  The sums $6+3=9$ and $35+5=40$ are at most $2^8-1=255$ and are given by the first case of the definition. The sum $200+100=300$ is greater than $255$, so the second case applies and the result is $300-2^8=44$. That result is smaller than both operands, which is the behaviour proper to the unsigned types.
-/

def uintA1 : Tensor Nat :=
  { shape := [3], data := [6, 200, 35] }

def uintB1 : Tensor Nat :=
  { shape := [3], data := [3, 100, 5] }

def uintC1 : Tensor Nat :=
  { shape := [3], data := [9, 44, 40] }

def uintExample1 : Prop := uintFunction 8 uintA1 uintB1 uintC1

end Generated.Add

/-- The CGen descriptor: the C stage renders it into elementwise kernels. -/
def sonnxCGenOp : CGen.Elementwise := ⟨"add", .add⟩

-- Fully-qualified declarations named by the trace entries.
#check @Generated.Add.floatExample1
#check @Generated.Add.floatExample2
#check @Generated.Add.floatExample3
#check @Generated.Add.floatFunction
#check @Generated.Add.intExample1
#check @Generated.Add.intFunction
#check @Generated.Add.realExample1
#check @Generated.Add.realExample2
#check @Generated.Add.realFunction
#check @Generated.Add.uintExample1
#check @Generated.Add.uintFunction
#check @Sonnx.BroadcastCompatible
#check @Sonnx.BroadcastShapeIs
#check @Sonnx.TypeConsistent

-- Generated by the SONNX pipeline: render the module's C descriptor.
#eval do
  IO.println s!"SONNX-CGEN name={sonnxCGenOp.name} binop={CGen.binOpName sonnxCGenOp.op}"
  IO.FS.writeFile "/run/media/eric/Work1/Work/gits/sonnx/ops/add/c/sonnx_kernels.h" (CGen.renderFile (CGen.headerFileFor [sonnxCGenOp] CGen.dtypes))
  IO.FS.writeFile "/run/media/eric/Work1/Work/gits/sonnx/ops/add/c/sonnx_kernels.c" (CGen.renderFile (CGen.sourceFileFor [sonnxCGenOp] CGen.dtypes))
