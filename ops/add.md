# Contents

- **Add** operator for type [real](#real)
- **Add** operator for types [`float16`, `float`, `double`](#float)
- **Add** operator for types [`int8`, `int16`, `int32`, `int64`](#int)
- **Add** operator for types [`uint8`, `uint16`, `uint32`, `uint64`](#uint)

Based on ONNX documentation [Add version 14](https://onnx.ai/onnx/operators/onnx__Add.html#add-14).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **Add** (real, real)

## Signature

$C = \textbf{Add}(A, B)$

where
- $A$: value to which $B$ is added
- $B$: value added to $A$
- $C$: result of the element-wise addition of $B$ to $A$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Add** operator.

## Function

<a id="E_ADD_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ADD_REAL_FUNC_0010]</br></span>

Operator **Add** adds tensor $B$ to tensor $A$ element-wise and stores the result in tensor $C$. Each element of $C$ is the sum of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $C$:

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

The effect of the operator is illustrated on the following examples.

### Example 1

$$
A = \begin{bmatrix} 6.1 & 9.5 & 35.7 \end{bmatrix}
\quad
B = \begin{bmatrix} 2 & 3 & 4 \end{bmatrix}
$$

$$
C = A + B = \begin{bmatrix} 8.1 & 12.5 & 39.7 \end{bmatrix}
$$

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

## Error conditions

None of the conditions of the list applies to real numbers: the addition is exact, so that the result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_ADD_REAL_FUNC_0010</span></b>](#E_ADD_REAL_FUNC_0010) for every pair of operands, and no error can occur. No error condition.

## Attributes

Operator **Add** has no attribute.

## Inputs

### $\text{A}$: real tensor

The value to which $B$ is added.

#### Constraints

<a id="E_ADD_REAL_CONSTR_A_0010"></a>
- `[E_ADD_REAL_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.

### $\text{B}$: real tensor

The value added to $A$.

#### Constraints

- `[E_ADD_REAL_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_REAL_CONSTR_A_0010</span></b>](#E_ADD_REAL_CONSTR_A_0010) on tensor $A$.

## Outputs

### $\text{C}$: real tensor

Tensor $C$ is the element-wise result of the addition of $B$ to $A$.

#### Constraints

- `[E_ADD_REAL_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.

<a id="float"></a>

# **Add** (float, float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same addition for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

$C = \textbf{Add}(A, B)$

where
- $A$: value to which $B$ is added
- $B$: value added to $A$
- $C$: result of the element-wise addition of $B$ to $A$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Add** operator.

## Function

<a id="E_ADD_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ADD_FLOAT_FUNC_0010]</br></span>

Operator **Add** adds tensor $B$ to tensor $A$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $C$. Each element of $C$ is the floating-point sum of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $C$:

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

The effect of the operator is illustrated on the following examples.

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

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e., the addition of two infinities of opposite signs | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ADD_FLOAT_FUNC_0010</span></b>](#E_ADD_FLOAT_FUNC_0010), the result is NaN |
| Overflow, i.e., an exact sum that rounds beyond the largest finite value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ADD_FLOAT_FUNC_0010</span></b>](#E_ADD_FLOAT_FUNC_0010), the result is $\pm\text{inf}$ |
| Underflow, i.e., an exact sum with a magnitude below the smallest normal value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ADD_FLOAT_FUNC_0010</span></b>](#E_ADD_FLOAT_FUNC_0010), the result is that sum, which is a subnormal value |

Every condition of the list is part of the nominal behavior of the operator; none of them is an error. No error condition.

## Attributes

Operator **Add** has no attribute.

## Inputs

### $\text{A}$: floating-point tensor

The value to which $B$ is added.

#### Constraints

<a id="E_ADD_FLOAT_CONSTR_A_0010"></a>
- `[E_ADD_FLOAT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_ADD_FLOAT_CONSTR_A_0020"></a>
- `[E_ADD_FLOAT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$, and $C$ have the same type.

### $\text{B}$: floating-point tensor

The value added to $A$.

#### Constraints

- `[E_ADD_FLOAT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_FLOAT_CONSTR_A_0010</span></b>](#E_ADD_FLOAT_CONSTR_A_0010) on tensor $A$.
- `[E_ADD_FLOAT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_FLOAT_CONSTR_A_0020</span></b>](#E_ADD_FLOAT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: floating-point tensor

Tensor $C$ is the element-wise result of the addition of $B$ to $A$.

#### Constraints

- `[E_ADD_FLOAT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_ADD_FLOAT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_FLOAT_CONSTR_A_0020</span></b>](#E_ADD_FLOAT_CONSTR_A_0020) on tensor $A$.

<a id="int"></a>

# **Add** (int, int)

where int is in {`int8`, `int16`, `int32`, `int64`}.

The four types share one semantics: each of them is an $n$-bit signed type, whose values range from $-2^{n-1}$ to $2^{n-1}-1$, and the result of the addition is the value of that type that represents the exact sum of the two elements modulo $2^n$.

## Signature

$C = \textbf{Add}(A, B)$

where
- $A$: value to which $B$ is added
- $B$: value added to $A$
- $C$: result of the element-wise addition of $B$ to $A$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Add** operator.

## Function

<a id="E_ADD_INT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ADD_INT_FUNC_0010]</br></span>

Operator **Add** adds tensor $B$ to tensor $A$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the sum of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below. Tensors $A$, $B$ and $C$ have the same signed $n$-bit type, and the result is a value of that type.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $C$:

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

The effect of the operator is illustrated on the following example.

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

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Overflow, i.e. an exact sum outside the range of the type, such as $100+100=200$ for `int8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ADD_INT_FUNC_0010</span></b>](#E_ADD_INT_FUNC_0010), the result is the value of the type that represents the sum modulo $2^n$, of the opposite sign when the sum exceeds the maximum value of the type |

The only condition of the list that applies to the signed integer types is the overflow, and it is part of the nominal behavior of the operator; there is no error. No error condition.

## Attributes

Operator **Add** has no attribute.

## Inputs

### $\text{A}$: signed integer tensor

The value to which $B$ is added.

#### Constraints

<a id="E_ADD_INT_CONSTR_A_0010"></a>
- `[E_ADD_INT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_ADD_INT_CONSTR_A_0020"></a>
- `[E_ADD_INT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$, and $C$ have the same type.

### $\text{B}$: signed integer tensor

The value added to $A$.

#### Constraints

- `[E_ADD_INT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_INT_CONSTR_A_0010</span></b>](#E_ADD_INT_CONSTR_A_0010) on tensor $A$.
- `[E_ADD_INT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_INT_CONSTR_A_0020</span></b>](#E_ADD_INT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: signed integer tensor

Tensor $C$ is the element-wise result of the addition of $B$ to $A$.

#### Constraints

- `[E_ADD_INT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_ADD_INT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_INT_CONSTR_A_0020</span></b>](#E_ADD_INT_CONSTR_A_0020) on tensor $A$.

<a id="uint"></a>

# **Add** (uint, uint)

where uint is in {`uint8`, `uint16`, `uint32`, `uint64`}.

The four types share one semantics: each of them is an $n$-bit unsigned type, whose values range from $0$ to $2^n-1$, and the result of the addition is the value of that type that represents the exact sum of the two elements modulo $2^n$.

## Signature

$C = \textbf{Add}(A, B)$

where
- $A$: value to which $B$ is added
- $B$: value added to $A$
- $C$: result of the element-wise addition of $B$ to $A$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Add** operator.

## Function

<a id="E_ADD_UINT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ADD_UINT_FUNC_0010]</br></span>

Operator **Add** adds tensor $B$ to tensor $A$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the sum of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below. Tensors $A$, $B$ and $C$ have the same unsigned $n$-bit type, and the result is a value of that type.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $C$:

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

The effect of the operator is illustrated on the following example.

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

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Overflow, i.e. an exact sum greater than the maximum value of the type, such as $200+100=300$ for `uint8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ADD_UINT_FUNC_0010</span></b>](#E_ADD_UINT_FUNC_0010), the result is that sum minus $2^n$, which is then smaller than both operands |

The only condition of the list that applies to the unsigned integer types is the overflow, and it is part of the nominal behavior of the operator; there is no error. No error condition.

## Attributes

Operator **Add** has no attribute.

## Inputs

### $\text{A}$: unsigned integer tensor

The value to which $B$ is added.

#### Constraints

<a id="E_ADD_UINT_CONSTR_A_0010"></a>
- `[E_ADD_UINT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_ADD_UINT_CONSTR_A_0020"></a>
- `[E_ADD_UINT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$, and $C$ have the same type.

### $\text{B}$: unsigned integer tensor

The value added to $A$.

#### Constraints

- `[E_ADD_UINT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_UINT_CONSTR_A_0010</span></b>](#E_ADD_UINT_CONSTR_A_0010) on tensor $A$.
- `[E_ADD_UINT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_UINT_CONSTR_A_0020</span></b>](#E_ADD_UINT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: unsigned integer tensor

Tensor $C$ is the element-wise result of the addition of $B$ to $A$.

#### Constraints

- `[E_ADD_UINT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_ADD_UINT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ADD_UINT_CONSTR_A_0020</span></b>](#E_ADD_UINT_CONSTR_A_0020) on tensor $A$.

