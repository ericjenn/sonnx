import Sonnx
/-!
@generated-by sonnx-lean-pipeline
@source ops/mul/mul.md
@source-digest 1f3d6ea0e8b35e2f2b4f86bece4123b1
@library-digest 245e9e643502252878cb83d2b2658e26f6dae9070802500971dc7a1817a7f1fe
Compilation and trace validity are separate from semantic completeness; see the trace report.
-/


/-!
Formal specification of the ONNX **Mul** operator, based on ONNX Mul version 14.

The operator is specified by the shared `Sonnx` declarations for each type family:

* real:  `Sonnx.MulRealCorrect`
* float: `Sonnx.MulFloatCorrect`
* int:   `Sonnx.MulIntCorrect`
* uint:  `Sonnx.MulUIntCorrect`

The shape and type constraints are the shared broadcasting and type-consistency
constraints of the library.  This module adds concrete example tensors and example
claims for the families whose examples can be stated directly with the available
library vocabulary.
-/

/-! 
@trace E_MUL_REAL_FUNC_0010
@source ops/mul/mul.md:L33-L58
@coverage specified
@anchor E_MUL_REAL_FUNC_0010
@declaration Sonnx.MulRealCorrect
Informal source excerpt:
  <a id="E_MUL_REAL_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_MUL_REAL_FUNC_0010]</br></span>
  
  Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise and stores the result in tensor $C$. Each element of $C$ is the product of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \tilde{A}[i] \cdot \tilde{B}[i]
  $$
  
  where
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$,
  - $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$.
  
  The multiplication is exact: $C[i]$ is the product in $\mathbb{R}$ of the two elements, with no approximation.
  
  **The two operands play the same role.** The product in $\mathbb{R}$ is commutative, so that the multiplication of $B$ by $A$ gives the same result as the multiplication of $A$ by $B$; the operands are named only to designate the tensors.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the product of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-! 
@trace E_MUL_REAL_CONSTR_A_0010
@source ops/mul/mul.md:L116-L118
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
@source ops/mul/mul.md:L126-L127
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_MUL_REAL_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_REAL_CONSTR_A_0010</span></b>](#E_MUL_REAL_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_MUL_REAL_CONSTR_C_0010
@source ops/mul/mul.md:L137-L138
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_MUL_REAL_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_MUL_FLOAT_FUNC_0010
@source ops/mul/mul.md:L165-L214
@coverage specified
@anchor E_MUL_FLOAT_FUNC_0010
@declaration Sonnx.MulFloatCorrect
Informal source excerpt:
  <a id="E_MUL_FLOAT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_MUL_FLOAT_FUNC_0010]</br></span>
  
  Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $C$. Each element of $C$ is the floating-point product of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \tilde{A}[i] \cdot_{(\text{f})} \tilde{B}[i] =
  \begin{cases}
  \text{NaN} & \text{if } \tilde{A}[i] \text{ or } \tilde{B}[i] \text{ is NaN, or if one of them is a zero and the other an infinity} \\
  \tilde{A}[i] \cdot \tilde{B}[i] & \text{if the exact product is representable in the type of } A \text{ and } B \\
  \text{round}(\tilde{A}[i] \cdot \tilde{B}[i]) & \text{otherwise}
  \end{cases}
  $$
  
  where
  - $\cdot_{(\text{f})}$ is the multiplication of the floating-point type of $A$, $B$ and $C$, i.e. $\cdot_{(\text{f16})}$, $\cdot_{(\text{f32})}$ or $\cdot_{(\text{f64})}$ according to that type, whereas the $\cdot$ of the second and the third case is the multiplication of $\mathbb{R}$: there, $\tilde{A}[i] \cdot \tilde{B}[i]$ is the exact product of the two elements,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $\text{round}(x)$ is the value of $x$ rounded to the nearest value of the type of $A$ and $B$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded: a magnitude whose nearest value exceeds the largest finite value of the type of $A$ and $B$ gives an infinity of the sign of $x$.
  
  In the first case, an operand that is NaN is propagated, and the multiplication of a zero by an infinity is the invalid operation defined in IEEE 754 section 7.2; the result is NaN in both cases. The NaN of this case is a quiet NaN; its sign and its payload are not specified, whether the NaN comes from an operand or from the invalid operation, so that every quiet NaN of the type of $A$ and $B$ is a conforming result.
  
  In the last case, the exact product is not representable in the type of $A$ and $B$: the result is that product rounded as defined above. The rounding may make the result subnormal, or, when the exact product is smaller in magnitude than half the smallest subnormal value of the type, zero.
  
  The result takes the sign of the exact product in every case, zero and infinity included: the sign of $\tilde{A}[i] \cdot \tilde{B}[i]$.
  
  **The sign of the result is the sign of the product, not of the operands.** IEEE 754 section 6.3 defines the sign of a product as the exclusive or of the signs of its operands, so that the sign of $C[i]$ is negative when exactly one of $\tilde{A}[i]$ and $\tilde{B}[i]$ is negative, and positive when both have the same sign; the operands may be swapped without changing the result. This applies to the special numbers as well:
  
  $$
  (+0) \cdot_{(\text{f})} (-5) = -0
  \qquad
  (-0) \cdot_{(\text{f})} (-0) = +0
  \qquad
  (-\text{inf}) \cdot_{(\text{f})} (-2) = +\text{inf}
  \qquad
  (-\text{inf}) \cdot_{(\text{f})} (+2) = -\text{inf}
  $$
  
  A product whose exact value is null is thus $+0$ or $-0$ according to the signs of the operands: it is not always $+0$ as the null sum of two values of opposite signs is. A product that underflows to zero takes that sign too, as does a product that overflows to an infinity.
  
  **Values of the operands.** The operands may be any of the values of the type, including the special numbers $\pm 0$, $\pm\text{inf}$ and NaN. The cases above give the result for every combination of them. The multiplication of an infinity by a finite non-zero value, or by an infinity, gives an infinity; the multiplication of a zero by a finite value gives a zero; the sign of both is the one defined above.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the product of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-! 
@trace E_MUL_FLOAT_CONSTR_A_0010
@source ops/mul/mul.md:L296-L298
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
@source ops/mul/mul.md:L299-L301
@coverage specified
@anchor E_MUL_FLOAT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_MUL_FLOAT_CONSTR_A_0020"></a>
  - `[E_MUL_FLOAT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$, and $C$ have the same type.
-/

/-! 
@trace E_MUL_FLOAT_CONSTR_B_0010
@source ops/mul/mul.md:L309-L310
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_MUL_FLOAT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_FLOAT_CONSTR_A_0010</span></b>](#E_MUL_FLOAT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_MUL_FLOAT_CONSTR_B_0020
@source ops/mul/mul.md:L311-L312
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_MUL_FLOAT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_FLOAT_CONSTR_A_0020</span></b>](#E_MUL_FLOAT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_MUL_FLOAT_CONSTR_C_0010
@source ops/mul/mul.md:L322-L323
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_MUL_FLOAT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_MUL_FLOAT_CONSTR_C_0020
@source ops/mul/mul.md:L324-L325
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_MUL_FLOAT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_FLOAT_CONSTR_A_0020</span></b>](#E_MUL_FLOAT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_MUL_INT_FUNC_0010
@source ops/mul/mul.md:L352-L382
@coverage specified
@anchor E_MUL_INT_FUNC_0010
@declaration Sonnx.MulIntCorrect
Informal source excerpt:
  <a id="E_MUL_INT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_MUL_INT_FUNC_0010]</br></span>
  
  Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the product of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below. Tensors $A$, $B$ and $C$ have the same signed $n$-bit type, and the result is a value of that type.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] \equiv \tilde{A}[i] \cdot \tilde{B}[i] \pmod{2^n}, \qquad C[i] \text{ a value of the type of } A \text{ and } B
  $$
  
  where
  - $\tilde{A}[i] \cdot \tilde{B}[i]$ is the exact product of the two elements, the product in $\mathbb{Z}$ of the two integers, before the reduction modulo $2^n$,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $n$ is the number of bits of the type of $A$ and $B$.
  
  The values of an $n$-bit type are $2^n$ distinct integers no two of which are congruent modulo $2^n$: exactly one value of the type satisfies the congruence, and that value is $C[i]$. The congruence defines $C[i]$ and is the exact product reduced modulo $2^n$.
  
  **The reduction may be applied several times.** The values of an $n$-bit signed type lie in $[-2^{n-1}, 2^{n-1}-1]$, so that their exact product lies in $[-2^{2n-2}+2^{n-1}, 2^{2n-2}]$, a range of the order of the square of the range of the type: the lower bound is reached by $(-2^{n-1}) \cdot (2^{n-1}-1)$, the smallest value times the largest, and the upper bound by $(-2^{n-1}) \cdot (-2^{n-1})$, the smallest value times itself. Unlike the sum of two values of the type, whose magnitude is at most of the order of the range itself, the product may leave the range of the type by several multiples of $2^n$: for `int8`, the exact product $100 \cdot 100 = 10000$ is $39 \cdot 2^8 + 16$, and the result is $16$.
  
  **The sign of the result.** The reduction modulo $2^n$ does not preserve the sign of the exact product: the multiplication of two positive values may give a negative result, and conversely. For `int8`, $11 \cdot 13 = 143$ exceeds the maximum value $2^7-1=127$ of the type, and the result is $143 - 2^8 = -113$, negative although both operands are positive. Tensors $A$, $B$ and $C$ have the same type and no wider type is used, so that the value of $C$ is that value of the type, sign included.
  
  **The result may be zero although both operands are non-zero.** For `int8`, $16 \cdot 16 = 256$ is a multiple of $2^8$, so that the result is $0$.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the product of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-! 
@trace E_MUL_INT_CONSTR_A_0010
@source ops/mul/mul.md:L422-L424
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
@source ops/mul/mul.md:L425-L427
@coverage specified
@anchor E_MUL_INT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_MUL_INT_CONSTR_A_0020"></a>
  - `[E_MUL_INT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$, and $C$ have the same type.
-/

/-! 
@trace E_MUL_INT_CONSTR_B_0010
@source ops/mul/mul.md:L435-L436
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_MUL_INT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_INT_CONSTR_A_0010</span></b>](#E_MUL_INT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_MUL_INT_CONSTR_B_0020
@source ops/mul/mul.md:L437-L438
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_MUL_INT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_INT_CONSTR_A_0020</span></b>](#E_MUL_INT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_MUL_INT_CONSTR_C_0010
@source ops/mul/mul.md:L448-L449
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_MUL_INT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_MUL_INT_CONSTR_C_0020
@source ops/mul/mul.md:L450-L451
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_MUL_INT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_INT_CONSTR_A_0020</span></b>](#E_MUL_INT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_MUL_UINT_FUNC_0010
@source ops/mul/mul.md:L478-L508
@coverage specified
@anchor E_MUL_UINT_FUNC_0010
@declaration Sonnx.MulUIntCorrect
Informal source excerpt:
  <a id="E_MUL_UINT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_MUL_UINT_FUNC_0010]</br></span>
  
  Operator **Mul** multiplies tensor $A$ by tensor $B$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the product of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below. Tensors $A$, $B$ and $C$ have the same unsigned $n$-bit type, and the result is a value of that type.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] \equiv \tilde{A}[i] \cdot \tilde{B}[i] \pmod{2^n}, \qquad C[i] \text{ a value of the type of } A \text{ and } B
  $$
  
  where
  - $\tilde{A}[i] \cdot \tilde{B}[i]$ is the exact product of the two elements, the product in $\mathbb{Z}$ of the two integers, before the reduction modulo $2^n$,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $n$ is the number of bits of the type of $A$ and $B$.
  
  The values of an $n$-bit type are $2^n$ distinct integers no two of which are congruent modulo $2^n$: exactly one value of the type satisfies the congruence, and that value is $C[i]$. The congruence defines $C[i]$ and is the exact product reduced modulo $2^n$.
  
  **The reduction may be applied several times.** The values of an $n$-bit unsigned type lie in $[0, 2^n-1]$, so that their exact product lies in $[0, 2^{2n}-2^{n+1}+1]$, a range of the order of the square of the range of the type. Unlike the sum of two values of the type, whose magnitude is at most of the order of the range itself, the product may leave the range of the type by several multiples of $2^n$: for `uint8`, the exact product $200 \cdot 200 = 40000$ is $156 \cdot 2^8 + 64$, and the result is $64$.
  
  **The result may be smaller than both operands.** The product of two non-negative integers is never smaller than either of them, whereas the result here is reduced modulo $2^n$: for `uint8`, $20 \cdot 13 = 260$ exceeds the maximum value $2^8-1=255$, and the result is $260-2^8=4$, a value smaller than both operands. Tensors $A$, $B$ and $C$ have the same type and no wider type is used, so that the value of $C$ is that value of the type, and the multiplication is not monotonic on the unsigned types.
  
  **The result may be zero although both operands are non-zero.** For `uint8`, $16 \cdot 16 = 256$ is a multiple of $2^8$, so that the result is $0$.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the product of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/-! 
@trace E_MUL_UINT_CONSTR_A_0010
@source ops/mul/mul.md:L548-L550
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
@source ops/mul/mul.md:L551-L553
@coverage specified
@anchor E_MUL_UINT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_MUL_UINT_CONSTR_A_0020"></a>
  - `[E_MUL_UINT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$, and $C$ have the same type.
-/

/-! 
@trace E_MUL_UINT_CONSTR_B_0010
@source ops/mul/mul.md:L561-L562
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_MUL_UINT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_UINT_CONSTR_A_0010</span></b>](#E_MUL_UINT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_MUL_UINT_CONSTR_B_0020
@source ops/mul/mul.md:L563-L564
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_MUL_UINT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_UINT_CONSTR_A_0020</span></b>](#E_MUL_UINT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_MUL_UINT_CONSTR_C_0010
@source ops/mul/mul.md:L574-L575
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_MUL_UINT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_MUL_UINT_CONSTR_C_0020
@source ops/mul/mul.md:L576-L577
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_MUL_UINT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_UINT_CONSTR_A_0020</span></b>](#E_MUL_UINT_CONSTR_A_0020) on tensor $A$.
-/


namespace Generated.Mul

/-! ## Real examples -/

def realA1 : Sonnx.Tensor Rat :=
  { shape := [3],
    data := [(61 / 10 : Rat), (19 / 2 : Rat), (357 / 10 : Rat)] }

def realB1 : Sonnx.Tensor Rat :=
  { shape := [3],
    data := [(2 : Rat), (3 : Rat), (4 : Rat)] }

def realC1 : Sonnx.Tensor Rat :=
  { shape := [3],
    data := [(61 / 5 : Rat), (57 / 2 : Rat), (714 / 5 : Rat)] }

/-! 
@trace real:example:1
@source ops/mul/mul.md:L62-L72
@coverage specified
@declaration Generated.Mul.realExample1
Informal source excerpt:
  ### Example 1
  
  $$
  A = \begin{bmatrix} 6.1 & 9.5 & 35.7 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 2 & 3 & 4 \end{bmatrix}
  $$
  
  $$
  C = A \cdot B = \begin{bmatrix} 12.2 & 28.5 & 142.8 \end{bmatrix}
  $$
-/

def realExample1 : Prop :=
  Sonnx.MulRealCorrect realA1 realB1 realC1

def realA2 : Sonnx.Tensor Rat :=
  { shape := [3, 2],
    data := [(37 / 10 : Rat), (22 / 5 : Rat),
             (81 / 5 : Rat), (1 / 2 : Rat),
             (253 / 10 : Rat), (124 / 5 : Rat)] }

def realB2 : Sonnx.Tensor Rat :=
  { shape := [2],
    data := [(2 : Rat), (10 : Rat)] }

def realC2 : Sonnx.Tensor Rat :=
  { shape := [3, 2],
    data := [(37 / 5 : Rat), (44 : Rat),
             (162 / 5 : Rat), (5 : Rat),
             (253 / 5 : Rat), (248 : Rat)] }

/-! 
@trace real:example:2
@source ops/mul/mul.md:L74-L98
@coverage specified
@declaration Generated.Mul.realExample2
Informal source excerpt:
  ### Example 2
  
  $$
  A = \begin{bmatrix}
    3.7 & 4.4 \\
    16.2 & 0.5 \\
    25.3 & 24.8
  \end{bmatrix}
  \quad
  B = \begin{bmatrix} 2 & 10 \end{bmatrix}
  $$
  
  $$
  C = A \cdot B = \begin{bmatrix}
    3.7 \cdot 2 & 4.4 \cdot 10 \\
    16.2 \cdot 2 & 0.5 \cdot 10 \\
    25.3 \cdot 2 & 24.8 \cdot 10
  \end{bmatrix} = \begin{bmatrix}
    7.4 & 44.0 \\
    32.4 & 5.0 \\
    50.6 & 248.0
  \end{bmatrix}
  $$
  
  Tensor $B$ has rank $1$ and tensor $A$ has rank $2$; $B$ is broadcast to the shape of $A$, so that each column of $A$ is multiplied by the corresponding element of $B$.
-/

def realExample2 : Prop :=
  Sonnx.MulRealCorrect realA2 realB2 realC2

/-! ## Floating-point examples

`DECISION:` The four floating-point examples are deferred.  A faithful concrete
Lean claim must fix one of `float16`, `float32` or `float64` and construct the
corresponding `FVal` tensors; rounding and subnormal behaviour then depend on the
chosen `FloatFormat`.  The informal examples discuss all three formats together,
so a single concrete claim without fixing the format would not state the examples
faithfully.
-/

/-! 
@trace float:example:1
@source ops/mul/mul.md:L218-L230
@coverage deferred
@limitation No concrete Lean claim is supplied: the example is stated for the floating-point family without fixing a FloatFormat, so a faithful format-specific FVal claim is not supplied in this module.
Informal source excerpt:
  ### Example 1
  
  $$
  A = \begin{bmatrix} 3.0 & 4.5 \\ 16.0 & 1.0 \\ 25.5 & 24.25 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 3.0 & 2.0 \\ 4.0 & 0.0 \\ 5.0 & 4.0 \end{bmatrix}
  $$
  
  $$
  C = A \cdot_{(\text{f})} B = \begin{bmatrix} 9.0 & 9.0 \\ 64.0 & 0.0 \\ 127.5 & 97.0 \end{bmatrix}
  $$
  
  The product of two numbers of a `float` type is not always exact; in this example every product is representable and $C$ is the exact result.
-/

/-! 
@trace float:example:2
@source ops/mul/mul.md:L232-L244
@coverage deferred
@limitation No concrete Lean claim is supplied: the special-value example is stated for the floating-point family without fixing a FloatFormat, so a faithful format-specific FVal claim is not supplied in this module.
Informal source excerpt:
  ### Example 2
  
  $$
  A = \begin{bmatrix} 1.0 & \text{+inf} & \text{-inf} & \text{+0} & \text{-0} & \text{-0} \end{bmatrix}
  \quad
  B = \begin{bmatrix} 1.0 & \text{+0} & \text{-2.0} & \text{-3.0} & \text{+5.0} & \text{-0} \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 1.0 & \text{NaN} & \text{+inf} & \text{-0} & \text{-0} & \text{+0} \end{bmatrix}
  $$
  
  The product of an infinity by a zero is the invalid operation of IEEE 754 section 7.2 and yields NaN. The product of the two negative values $-\text{inf}$ and $-2.0$ is the positive infinity. The three null products show the rule of the sign: $(+0) \cdot (-3.0) = -0$ and $(-0) \cdot (+5.0) = -0$, the sign of a null product being negative when exactly one operand is negative; and $(-0) \cdot (-0) = +0$, since the two signs are equal.
-/

/-! 
@trace float:example:3
@source ops/mul/mul.md:L246-L258
@coverage deferred
@limitation No concrete Lean claim is supplied: the rounding and overflow example is stated for the floating-point family without fixing a FloatFormat, so a faithful format-specific FVal claim is not supplied in this module.
Informal source excerpt:
  ### Example 3
  
  $$
  A = \begin{bmatrix} 0.1 & 1.0 & 1.0\text{e}308 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 0.2 & 3.0 & 1.0\text{e}308 \end{bmatrix}
  $$
  
  $$
  C \approx \begin{bmatrix} 0.020000000000000004 & 3.0 & \text{+inf} \end{bmatrix}
  $$
  
  The exact product of the two `double` values $0.1$ and $0.2$ is not representable in `double`; the first element of $C$ is that product rounded to the nearest `double`. The exact product $1\text{e}308 \cdot 1\text{e}308$ is beyond the largest finite value of `double`, so that the third element is the positive infinity, the sign of the exact product.
-/

/-! 
@trace float:example:4
@source ops/mul/mul.md:L260-L272
@coverage deferred
@limitation No concrete Lean claim is supplied: the subnormal example is stated for the floating-point family without fixing a FloatFormat, so a faithful format-specific FVal claim is not supplied in this module.
Informal source excerpt:
  ### Example 4
  
  $$
  A = \begin{bmatrix} 2.2250738585072014\text{e-}308 & -2.2250738585072014\text{e-}308 & \text{+0} & \text{-0} \end{bmatrix}
  \quad
  B = \begin{bmatrix} 0.25 & 0.25 & \text{-1.0} & \text{-0} \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 5.562684646268003\text{e-}309 & -5.562684646268003\text{e-}309 & \text{-0} & \text{+0} \end{bmatrix}
  $$
  
  The first operand is the smallest positive normal value of `double`, whose exact half is not representable as a normal value: multiplied by $0.25$ it gives a subnormal value, the nearest value of the type, and the second element is the negative of that value. The last two elements are the null products of Example 2, $-0$ and $+0$; none of the four elements is rounded to zero, the exact products of the first two being representable as subnormal values.
-/


/-! ## Signed-integer example -/

def intA1 : Sonnx.Tensor (Sonnx.SIntN 8) :=
  { shape := [4],
    data := [@Sonnx.SIntN.wrap 8 (-6),
             @Sonnx.SIntN.wrap 8 11,
             @Sonnx.SIntN.wrap 8 100,
             @Sonnx.SIntN.wrap 8 16] }

def intB1 : Sonnx.Tensor (Sonnx.SIntN 8) :=
  { shape := [4],
    data := [@Sonnx.SIntN.wrap 8 (-3),
             @Sonnx.SIntN.wrap 8 13,
             @Sonnx.SIntN.wrap 8 100,
             @Sonnx.SIntN.wrap 8 16] }

def intC1 : Sonnx.Tensor (Sonnx.SIntN 8) :=
  { shape := [4],
    data := [@Sonnx.SIntN.wrap 8 18,
             @Sonnx.SIntN.wrap 8 (-113),
             @Sonnx.SIntN.wrap 8 16,
             @Sonnx.SIntN.wrap 8 0] }

/-! 
@trace int:example:1
@source ops/mul/mul.md:L386-L400
@coverage specified
@declaration Generated.Mul.intExample1
Informal source excerpt:
  ### Example 1
  
  For the `int8` type:
  
  $$
  A = \begin{bmatrix} -6 & 11 & 100 & 16 \end{bmatrix}
  \quad
  B = \begin{bmatrix} -3 & 13 & 100 & 16 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 18 & -113 & 16 & 0 \end{bmatrix}
  $$
  
  The product $-6 \cdot (-3) = 18$ lies in $[-2^7, 2^7-1]$ and is its own reduction. The product $11 \cdot 13 = 143$ is greater than the maximum value $127$ of `int8`, so that the result is $143-2^8=-113$: it is negative although both operands are positive. The product $100 \cdot 100 = 10000$ is $39 \cdot 2^8 + 16$, and the result is $16$, the reduction being applied more than once. The product $16 \cdot 16 = 256$ is a multiple of $2^8$ and the result is $0$, although both operands are non-zero.
-/

def intExample1 : Prop :=
  Sonnx.MulIntCorrect 8 intA1 intB1 intC1

/-! ## Unsigned-integer example -/

def uintA1 : Sonnx.Tensor (Sonnx.UIntN 8) :=
  { shape := [4],
    data := [@Sonnx.UIntN.wrap 8 6,
             @Sonnx.UIntN.wrap 8 200,
             @Sonnx.UIntN.wrap 8 20,
             @Sonnx.UIntN.wrap 8 16] }

def uintB1 : Sonnx.Tensor (Sonnx.UIntN 8) :=
  { shape := [4],
    data := [@Sonnx.UIntN.wrap 8 3,
             @Sonnx.UIntN.wrap 8 200,
             @Sonnx.UIntN.wrap 8 13,
             @Sonnx.UIntN.wrap 8 16] }

def uintC1 : Sonnx.Tensor (Sonnx.UIntN 8) :=
  { shape := [4],
    data := [@Sonnx.UIntN.wrap 8 18,
             @Sonnx.UIntN.wrap 8 64,
             @Sonnx.UIntN.wrap 8 4,
             @Sonnx.UIntN.wrap 8 0] }

/-! 
@trace uint:example:1
@source ops/mul/mul.md:L512-L526
@coverage specified
@declaration Generated.Mul.uintExample1
Informal source excerpt:
  ### Example 1
  
  For the `uint8` type:
  
  $$
  A = \begin{bmatrix} 6 & 200 & 20 & 16 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 3 & 200 & 13 & 16 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 18 & 64 & 4 & 0 \end{bmatrix}
  $$
  
  The product $6 \cdot 3 = 18$ is at most $2^8-1=255$ and is its own reduction. The product $200 \cdot 200 = 40000$ is $156 \cdot 2^8 + 64$, and the result is $64$, the reduction being applied more than once. The product $20 \cdot 13 = 260$ exceeds $255$, so that the result is $260-2^8=4$, smaller than both operands. The product $16 \cdot 16 = 256$ is a multiple of $2^8$ and the result is $0$, although both operands are non-zero.
-/

def uintExample1 : Prop :=
  Sonnx.MulUIntCorrect 8 uintA1 uintB1 uintC1

end Generated.Mul

-- Fully-qualified declarations named by the trace entries.
#check @Generated.Mul.intExample1
#check @Generated.Mul.realExample1
#check @Generated.Mul.realExample2
#check @Generated.Mul.uintExample1
#check @Sonnx.BroadcastCompatible
#check @Sonnx.BroadcastShapeIs
#check @Sonnx.MulFloatCorrect
#check @Sonnx.MulIntCorrect
#check @Sonnx.MulRealCorrect
#check @Sonnx.MulUIntCorrect
#check @Sonnx.TypeConsistent
