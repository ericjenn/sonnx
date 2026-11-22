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
@trace E_MUL_FLOAT_FUNC_0010
@source ops/mul/mul.md:L170-L208
@coverage deferred
@anchor E_MUL_FLOAT_FUNC_0010
@limitation The IEEE 754 element step over Sonnx.FVal (NaN propagation, signed zero/infinity, overflow to infinity, and roundTiesToEven) is not formalized as a tensor-level relation in this module.
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

/-! 
@trace float:example:1
@source ops/mul/mul.md:L212-L224
@coverage deferred
@limitation Concrete Sonnx.FVal example is not supplied because the floating-point element relation is deferred.
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

/-! 
@trace float:example:2
@source ops/mul/mul.md:L226-L238
@coverage deferred
@limitation Concrete Sonnx.FVal example is not supplied because the floating-point element relation is deferred.
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

/-! 
@trace float:example:3
@source ops/mul/mul.md:L240-L252
@coverage deferred
@limitation Concrete Sonnx.FVal example is not supplied because the floating-point element relation is deferred.
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


namespace Generated.Mul

/-!
## Real multiplication

DECISION: exact real arithmetic is modelled by `Rat`.  This captures the exact
product requirement of `E_MUL_REAL_FUNC_0010` for rational operands; the
specification does not use an unavailable `Real` type.
-/

/-! 
@trace E_MUL_REAL_FUNC_0010
@source ops/mul/mul.md:L33-L56
@coverage specified
@anchor E_MUL_REAL_FUNC_0010
@declaration Generated.Mul.realMulRel
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

def realMulRel (A B C : Sonnx.Tensor Rat) : Prop :=
  Sonnx.Tensor.wellFormed A ∧
  Sonnx.Tensor.wellFormed B ∧
  Sonnx.Tensor.wellFormed C ∧
  Sonnx.binary (fun a b : Rat => a * b) A B = some C

def realExample1_A : Sonnx.Tensor Rat :=
  { shape := [3]
    data := [(3 : Rat) / 2, (19 : Rat) / 2, (71 : Rat) / 2] }

def realExample1_B : Sonnx.Tensor Rat :=
  { shape := [3]
    data := [(2 : Rat), (3 : Rat), (-4 : Rat)] }

def realExample1_C : Sonnx.Tensor Rat :=
  { shape := [3]
    data := [(3 : Rat), (57 : Rat) / 2, (-142 : Rat)] }

/-! 
@trace real:example:1
@source ops/mul/mul.md:L60-L70
@coverage example_only
@declaration Generated.Mul.realExample1_claim
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

def realExample1_claim : Prop :=
  Sonnx.binary (fun a b : Rat => a * b) realExample1_A realExample1_B = some realExample1_C

def realExample2_A : Sonnx.Tensor Rat :=
  { shape := [3, 2]
    data := [(3 : Rat), (9 : Rat) / 2,
             (16 : Rat), (1 : Rat) / 2,
             (51 : Rat) / 2, (24 : Rat)] }

def realExample2_B : Sonnx.Tensor Rat :=
  { shape := [2]
    data := [(2 : Rat), (-1 : Rat)] }

def realExample2_C : Sonnx.Tensor Rat :=
  { shape := [3, 2]
    data := [(6 : Rat), (-9 : Rat) / 2,
             (32 : Rat), (-1 : Rat) / 2,
             (51 : Rat), (-24 : Rat)] }

/-! 
@trace real:example:2
@source ops/mul/mul.md:L72-L96
@coverage example_only
@declaration Generated.Mul.realExample2_claim
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

def realExample2_claim : Prop :=
  Sonnx.binary (fun a b : Rat => a * b) realExample2_A realExample2_B = some realExample2_C

/-!
## Signed integer multiplication

DECISION: signed `n`-bit tensors are modelled at value level by `Tensor Int`
whose entries satisfy `InSignedRange n`.  The element result is `sintWrap n`
applied to the exact integer product.  The shared `TypeConsistent` markers
record the same-type constraint of the informal signature.
-/

/-! 
@trace E_MUL_INT_FUNC_0010
@source ops/mul/mul.md:L334-L366
@coverage specified
@anchor E_MUL_INT_FUNC_0010
@declaration Generated.Mul.intMulRel
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

def intMulRel (n : Nat) (A B C : Sonnx.Tensor Int) : Prop :=
  Sonnx.Tensor.wellFormed A ∧
  Sonnx.Tensor.wellFormed B ∧
  Sonnx.Tensor.wellFormed C ∧
  (∀ i, i ∈ Sonnx.indices A.shape →
      ∃ v, Sonnx.Tensor.read A i = some v ∧ Sonnx.InSignedRange n v) ∧
  (∀ i, i ∈ Sonnx.indices B.shape →
      ∃ v, Sonnx.Tensor.read B i = some v ∧ Sonnx.InSignedRange n v) ∧
  Sonnx.binary (fun a b => Sonnx.sintWrap n (a * b)) A B = some C

def intExample1_A : Sonnx.Tensor Int :=
  { shape := [4]
    data := [(-6 : Int), (100 : Int), (-100 : Int), (100 : Int)] }

def intExample1_B : Sonnx.Tensor Int :=
  { shape := [4]
    data := [(-3 : Int), (100 : Int), (100 : Int), (3 : Int)] }

def intExample1_C : Sonnx.Tensor Int :=
  { shape := [4]
    data := [(18 : Int), (16 : Int), (-16 : Int), (44 : Int)] }

/-! 
@trace int:example:1
@source ops/mul/mul.md:L370-L384
@coverage example_only
@declaration Generated.Mul.intExample1_claim
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

def intExample1_claim : Prop :=
  intMulRel 8 intExample1_A intExample1_B intExample1_C

/-!
## Unsigned integer multiplication

DECISION: unsigned `n`-bit tensors are modelled at value level by `Tensor Int`
whose entries satisfy `InUnsignedRange n`.  The element result is `uintWrap n`
applied to the exact integer product.  The shared `TypeConsistent` markers
record the same-type constraint of the informal signature.
-/

/-! 
@trace E_MUL_UINT_FUNC_0010
@source ops/mul/mul.md:L465-L492
@coverage specified
@anchor E_MUL_UINT_FUNC_0010
@declaration Generated.Mul.uintMulRel
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

def uintMulRel (n : Nat) (A B C : Sonnx.Tensor Int) : Prop :=
  Sonnx.Tensor.wellFormed A ∧
  Sonnx.Tensor.wellFormed B ∧
  Sonnx.Tensor.wellFormed C ∧
  (∀ i, i ∈ Sonnx.indices A.shape →
      ∃ v, Sonnx.Tensor.read A i = some v ∧ Sonnx.InUnsignedRange n v) ∧
  (∀ i, i ∈ Sonnx.indices B.shape →
      ∃ v, Sonnx.Tensor.read B i = some v ∧ Sonnx.InUnsignedRange n v) ∧
  Sonnx.binary (fun a b => Sonnx.uintWrap n (a * b)) A B = some C

def uintExample1_A : Sonnx.Tensor Int :=
  { shape := [4]
    data := [(6 : Int), (200 : Int), (35 : Int), (16 : Int)] }

def uintExample1_B : Sonnx.Tensor Int :=
  { shape := [4]
    data := [(3 : Int), (100 : Int), (5 : Int), (16 : Int)] }

def uintExample1_C : Sonnx.Tensor Int :=
  { shape := [4]
    data := [(18 : Int), (32 : Int), (175 : Int), (0 : Int)] }

/-! 
@trace uint:example:1
@source ops/mul/mul.md:L496-L510
@coverage example_only
@declaration Generated.Mul.uintExample1_claim
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

def uintExample1_claim : Prop :=
  uintMulRel 8 uintExample1_A uintExample1_B uintExample1_C

end Generated.Mul

/-- The CGen descriptor: the C stage renders it into elementwise kernels. -/
def sonnxCGenOp : CGen.Elementwise := ⟨"mul", .mul⟩

-- Fully-qualified declarations named by the trace entries.
#check @Generated.Mul.intExample1_claim
#check @Generated.Mul.intMulRel
#check @Generated.Mul.realExample1_claim
#check @Generated.Mul.realExample2_claim
#check @Generated.Mul.realMulRel
#check @Generated.Mul.uintExample1_claim
#check @Generated.Mul.uintMulRel
#check @Sonnx.BroadcastCompatible
#check @Sonnx.BroadcastShapeIs
#check @Sonnx.TypeConsistent
