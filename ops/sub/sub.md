# Contents

- **Sub** operator for type [real](#real)
- **Sub** operator for types [`float16`, `float`, `double`](#float)
- **Sub** operator for types [`int8`, `int16`, `int32`, `int64`](#int)
- **Sub** operator for types [`uint8`, `uint16`, `uint32`, `uint64`](#uint)

Based on ONNX documentation [Sub version 14](https://onnx.ai/onnx/operators/onnx__Sub.html#sub-14).

Revision 2026-10-10: splits the integer specification into a signed and an unsigned section, states the sign of every null floating-point difference, and makes the floating-point case analysis range over the operands.

<a id="real"></a>

# **Sub** (real, real)

## Signature

$C = \textbf{Sub}(A, B)$

where
- $A$: value from which $B$ is subtracted
- $B$: value subtracted from $A$
- $C$: result of the element-wise subtraction of $B$ from $A$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Sub** operator.

## Function

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

The effect of the operator is illustrated on the following examples.

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

## Error conditions

None of the conditions of the list applies to real numbers: the subtraction is exact, so that the result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_SUB_REAL_FUNC_0010</span></b>](#E_SUB_REAL_FUNC_0010) for every pair of operands, and no error can occur. No error condition.

## Attributes

Operator **Sub** has no attribute.

## Inputs

### $\text{A}$: real tensor

The value from which $B$ is subtracted.

#### Constraints

<a id="E_SUB_REAL_CONSTR_A_0010"></a>
- `[E_SUB_REAL_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.

### $\text{B}$: real tensor

The value subtracted from $A$.

#### Constraints

- `[E_SUB_REAL_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_REAL_CONSTR_A_0010</span></b>](#E_SUB_REAL_CONSTR_A_0010) on tensor $A$.

## Outputs

### $\text{C}$: real tensor

Tensor $C$ is the element-wise result of the subtraction of $B$ from $A$.

#### Constraints

- `[E_SUB_REAL_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.

<a id="float"></a>

# **Sub** (float, float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same subtraction for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

$C = \textbf{Sub}(A, B)$

where
- $A$: value from which $B$ is subtracted
- $B$: value subtracted from $A$
- $C$: result of the element-wise subtraction of $B$ from $A$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Sub** operator.

## Function

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

The effect of the operator is illustrated on the following examples.

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

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e., the subtraction of two infinities of the same sign | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_FUNC_0010</span></b>](#E_SUB_FLOAT_FUNC_0010), the result is NaN |
| Overflow, i.e., an exact difference whose rounded value exceeds the largest finite value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_FUNC_0010</span></b>](#E_SUB_FLOAT_FUNC_0010), the result is $\pm\text{inf}$ |
| Underflow, i.e., an exact difference with a magnitude below the smallest normal value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_FUNC_0010</span></b>](#E_SUB_FLOAT_FUNC_0010), the result is that difference, which is a subnormal value |

Every condition of the list is part of the nominal behavior of the operator; none of them is an error. No error condition.

## Attributes

Operator **Sub** has no attribute.

## Inputs

### $\text{A}$: floating-point tensor

The value from which $B$ is subtracted.

#### Constraints

<a id="E_SUB_FLOAT_CONSTR_A_0010"></a>
- `[E_SUB_FLOAT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_SUB_FLOAT_CONSTR_A_0020"></a>
- `[E_SUB_FLOAT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$, and $C$ have the same type.

### $\text{B}$: floating-point tensor

The value subtracted from $A$.

#### Constraints

- `[E_SUB_FLOAT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_CONSTR_A_0010</span></b>](#E_SUB_FLOAT_CONSTR_A_0010) on tensor $A$.
- `[E_SUB_FLOAT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_CONSTR_A_0020</span></b>](#E_SUB_FLOAT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: floating-point tensor

Tensor $C$ is the element-wise result of the subtraction of $B$ from $A$.

#### Constraints

- `[E_SUB_FLOAT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_SUB_FLOAT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_CONSTR_A_0020</span></b>](#E_SUB_FLOAT_CONSTR_A_0020) on tensor $A$.

<a id="int"></a>

# **Sub** (int, int)

where int is in {`int8`, `int16`, `int32`, `int64`}.

The four types share one semantics: each of them is an $n$-bit signed type, whose values range from $-2^{n-1}$ to $2^{n-1}-1$, and the result of the subtraction is the value of that type that represents the exact difference of the two elements modulo $2^n$.

## Signature

$C = \textbf{Sub}(A, B)$

where
- $A$: value from which $B$ is subtracted
- $B$: value subtracted from $A$
- $C$: result of the element-wise subtraction of $B$ from $A$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Sub** operator.

## Function

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

The effect of the operator is illustrated on the following examples.

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

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Overflow, i.e., an exact difference outside the range of the type, such as $127-(-1)=128$ for `int8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SUB_INT_FUNC_0010</span></b>](#E_SUB_INT_FUNC_0010), the result is the difference modulo $2^n$, of the opposite sign when the difference exceeds the maximum value of the type |

The only condition of the list that applies to the signed integer types is the overflow, and it is part of the nominal behavior of the operator; there is no error. No error condition.

## Attributes

Operator **Sub** has no attribute.

## Inputs

### $\text{A}$: signed integer tensor

The value from which $B$ is subtracted.

#### Constraints

<a id="E_SUB_INT_CONSTR_A_0010"></a>
- `[E_SUB_INT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_SUB_INT_CONSTR_A_0020"></a>
- `[E_SUB_INT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$, and $C$ have the same type.

### $\text{B}$: signed integer tensor

The value subtracted from $A$.

#### Constraints

- `[E_SUB_INT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_INT_CONSTR_A_0010</span></b>](#E_SUB_INT_CONSTR_A_0010) on tensor $A$.
- `[E_SUB_INT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_INT_CONSTR_A_0020</span></b>](#E_SUB_INT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: signed integer tensor

Tensor $C$ is the element-wise result of the subtraction of $B$ from $A$.

#### Constraints

- `[E_SUB_INT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_SUB_INT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_INT_CONSTR_A_0020</span></b>](#E_SUB_INT_CONSTR_A_0020) on tensor $A$.

<a id="uint"></a>

# **Sub** (uint, uint)

where uint is in {`uint8`, `uint16`, `uint32`, `uint64`}.

The four types share one semantics: each of them is an $n$-bit unsigned type, whose values range from $0$ to $2^n-1$, and the result of the subtraction is the value of that type that represents the exact difference of the two elements modulo $2^n$.

## Signature

$C = \textbf{Sub}(A, B)$

where
- $A$: value from which $B$ is subtracted
- $B$: value subtracted from $A$
- $C$: result of the element-wise subtraction of $B$ from $A$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Sub** operator.

## Function

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

The effect of the operator is illustrated on the following example.

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

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Overflow, i.e., an exact difference lower than $0$, such as $0-1=-1$ for `uint8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SUB_UINT_FUNC_0010</span></b>](#E_SUB_UINT_FUNC_0010), the result is that difference plus $2^n$, which is then larger than both operands |

The only condition of the list that applies to the unsigned integer types is the overflow, and it is part of the nominal behavior of the operator; there is no error. No error condition.

## Attributes

Operator **Sub** has no attribute.

## Inputs

### $\text{A}$: unsigned integer tensor

The value from which $B$ is subtracted.

#### Constraints

<a id="E_SUB_UINT_CONSTR_A_0010"></a>
- `[E_SUB_UINT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_SUB_UINT_CONSTR_A_0020"></a>
- `[E_SUB_UINT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$, and $C$ have the same type.

### $\text{B}$: unsigned integer tensor

The value subtracted from $A$.

#### Constraints

- `[E_SUB_UINT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_UINT_CONSTR_A_0010</span></b>](#E_SUB_UINT_CONSTR_A_0010) on tensor $A$.
- `[E_SUB_UINT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_UINT_CONSTR_A_0020</span></b>](#E_SUB_UINT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: unsigned integer tensor

Tensor $C$ is the element-wise result of the subtraction of $B$ from $A$.

#### Constraints

- `[E_SUB_UINT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_SUB_UINT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_UINT_CONSTR_A_0020</span></b>](#E_SUB_UINT_CONSTR_A_0020) on tensor $A$.
