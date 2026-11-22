# Contents

- **Mul** operator for type [real](#real)
- **Mul** operator for types [`float16`, `float`, `double`](#float)
- **Mul** operator for types [`int8`, `int16`, `int32`, `int64`](#int)
- **Mul** operator for types [`uint8`, `uint16`, `uint32`, `uint64`](#uint)

Based on ONNX documentation [Mul version 14](https://onnx.ai/onnx/operators/onnx__Mul.html#mul-14).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **Mul** (real, real)

## Signature

$C = \textbf{Mul}(A, B)$

where
- $A$: value by which $B$ is multiplied
- $B$: value multiplied by $A$
- $C$: result of the element-wise multiplication of $A$ and $B$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Mul** operator.

## Function

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

The effect of the operator is illustrated on the following examples.

### Example 1

$$
A = \begin{bmatrix} 1.5 & 9.5 & 35.5 \end{bmatrix}
\quad
B = \begin{bmatrix} 2 & 3 & -4 \end{bmatrix}
$$

$$
C = A * B = \begin{bmatrix} 3 & 28.5 & -142 \end{bmatrix}
$$

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

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: real numbers have neither infinities nor NaN, and the product of two real numbers is defined for every pair |
| Overflow | not applicable: the product of two real numbers is a real number, whatever the magnitude of the operands |
| Integer overflow | not applicable: the operands are real numbers |
| Division by zero | not applicable: the operator performs no division |

Every condition of the list is without object for the real numbers: the multiplication is exact, so that the result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_MUL_REAL_FUNC_0010</span></b>](#E_MUL_REAL_FUNC_0010) for every pair of operands, and no error can occur. No error condition.

## Attributes

Operator **Mul** has no attribute.

## Inputs

### $\text{A}$: real tensor

The value by which $B$ is multiplied.

#### Constraints

<a id="E_MUL_REAL_CONSTR_A_0010"></a>
- `[E_MUL_REAL_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.

### $\text{B}$: real tensor

The value multiplied by $A$.

#### Constraints

- `[E_MUL_REAL_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_REAL_CONSTR_A_0010</span></b>](#E_MUL_REAL_CONSTR_A_0010) on tensor $A$.

## Outputs

### $\text{C}$: real tensor

Tensor $C$ is the element-wise result of the multiplication of $A$ by $B$.

#### Constraints

- `[E_MUL_REAL_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.

<a id="float"></a>

# **Mul** (float, float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same multiplication and the same rule of signs for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

$C = \textbf{Mul}(A, B)$

where
- $A$: value by which $B$ is multiplied
- $B$: value multiplied by $A$
- $C$: result of the element-wise multiplication of $A$ and $B$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Mul** operator.

## Function

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

The effect of the operator is illustrated on the following examples.

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

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e., the multiplication of a zero by an infinity | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_MUL_FLOAT_FUNC_0010</span></b>](#E_MUL_FLOAT_FUNC_0010), the result is NaN |
| Overflow, i.e., an exact product whose magnitude rounds beyond the largest finite value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_MUL_FLOAT_FUNC_0010</span></b>](#E_MUL_FLOAT_FUNC_0010), the result is $\pm\text{inf}$ of the sign of the exact product |
| Underflow, i.e., an exact product whose magnitude lies below the smallest normal value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_MUL_FLOAT_FUNC_0010</span></b>](#E_MUL_FLOAT_FUNC_0010), the result is that product rounded to the type, which is a subnormal value or a zero of the sign of the exact product |
| Integer overflow | not applicable: the operands are floating-point values |
| Division by zero | not applicable: the operator performs no division |

Every condition of the list that applies to the floating-point types is part of the nominal behavior of the operator; none of them is an error. No error condition.

## Attributes

Operator **Mul** has no attribute.

## Inputs

### $\text{A}$: floating-point tensor

The value by which $B$ is multiplied.

#### Constraints

<a id="E_MUL_FLOAT_CONSTR_A_0010"></a>
- `[E_MUL_FLOAT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_MUL_FLOAT_CONSTR_A_0020"></a>
- `[E_MUL_FLOAT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$ and $C$ have the same type.

### $\text{B}$: floating-point tensor

The value multiplied by $A$.

#### Constraints

- `[E_MUL_FLOAT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_FLOAT_CONSTR_A_0010</span></b>](#E_MUL_FLOAT_CONSTR_A_0010) on tensor $A$.
- `[E_MUL_FLOAT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_FLOAT_CONSTR_A_0020</span></b>](#E_MUL_FLOAT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: floating-point tensor

Tensor $C$ is the element-wise result of the multiplication of $A$ by $B$.

#### Constraints

- `[E_MUL_FLOAT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_MUL_FLOAT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_FLOAT_CONSTR_A_0020</span></b>](#E_MUL_FLOAT_CONSTR_A_0020) on tensor $A$.

<a id="int"></a>

# **Mul** (int, int)

where int is in {`int8`, `int16`, `int32`, `int64`}.

The four types share one semantics: each of them is an $n$-bit signed type, whose values range from $-2^{n-1}$ to $2^{n-1}-1$, and the result of the multiplication is the value of that type that is congruent to the exact product of the two elements modulo $2^n$.

## Signature

$C = \textbf{Mul}(A, B)$

where
- $A$: value by which $B$ is multiplied
- $B$: value multiplied by $A$
- $C$: result of the element-wise multiplication of $A$ and $B$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Mul** operator.

## Function

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

The effect of the operator is illustrated on the following example.

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

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Integer overflow, i.e. an exact product outside the range of the type, such as $100 * 100 = 10000$ for `int8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_MUL_INT_FUNC_0010</span></b>](#E_MUL_INT_FUNC_0010), the result is the value of the type congruent to that product modulo $2^n$ |
| Invalid operation | not applicable: the operands are integers, which have neither infinities nor NaN |
| Overflow | not applicable: the operands and the result are integers, and the condition covers the floating-point computations that produce an infinity, which are specified as an integer overflow above |
| Division by zero | not applicable: the operator performs no division |

The only condition of the list that applies to the signed integer types is the integer overflow, and it is part of the nominal behavior of the operator; there is no error. No error condition.

## Attributes

Operator **Mul** has no attribute.

## Inputs

### $\text{A}$: signed integer tensor

The value by which $B$ is multiplied.

#### Constraints

<a id="E_MUL_INT_CONSTR_A_0010"></a>
- `[E_MUL_INT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_MUL_INT_CONSTR_A_0020"></a>
- `[E_MUL_INT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$ and $C$ have the same type.

### $\text{B}$: signed integer tensor

The value multiplied by $A$.

#### Constraints

- `[E_MUL_INT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_INT_CONSTR_A_0010</span></b>](#E_MUL_INT_CONSTR_A_0010) on tensor $A$.
- `[E_MUL_INT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_INT_CONSTR_A_0020</span></b>](#E_MUL_INT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: signed integer tensor

Tensor $C$ is the element-wise result of the multiplication of $A$ by $B$.

#### Constraints

- `[E_MUL_INT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_MUL_INT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_INT_CONSTR_A_0020</span></b>](#E_MUL_INT_CONSTR_A_0020) on tensor $A$.

<a id="uint"></a>

# **Mul** (uint, uint)

where uint is in {`uint8`, `uint16`, `uint32`, `uint64`}.

The four types share one semantics: each of them is an $n$-bit unsigned type, whose values range from $0$ to $2^n-1$, and the result of the multiplication is the value of that type that is congruent to the exact product of the two elements modulo $2^n$.

## Signature

$C = \textbf{Mul}(A, B)$

where
- $A$: value by which $B$ is multiplied
- $B$: value multiplied by $A$
- $C$: result of the element-wise multiplication of $A$ and $B$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Mul** operator.

## Function

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

The effect of the operator is illustrated on the following example.

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

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Integer overflow, i.e. an exact product greater than the maximum value of the type, such as $200 * 100 = 20000$ for `uint8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_MUL_UINT_FUNC_0010</span></b>](#E_MUL_UINT_FUNC_0010), the result is the remainder of that product modulo $2^n$ |
| Invalid operation | not applicable: the operands are integers, which have neither infinities nor NaN |
| Overflow | not applicable: the operands and the result are integers, and the condition covers the floating-point computations that produce an infinity, which are specified as an integer overflow above |
| Division by zero | not applicable: the operator performs no division |

The only condition of the list that applies to the unsigned integer types is the integer overflow, and it is part of the nominal behavior of the operator; there is no error. No error condition.

## Attributes

Operator **Mul** has no attribute.

## Inputs

### $\text{A}$: unsigned integer tensor

The value by which $B$ is multiplied.

#### Constraints

<a id="E_MUL_UINT_CONSTR_A_0010"></a>
- `[E_MUL_UINT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_MUL_UINT_CONSTR_A_0020"></a>
- `[E_MUL_UINT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$ and $C$ have the same type.

### $\text{B}$: unsigned integer tensor

The value multiplied by $A$.

#### Constraints

- `[E_MUL_UINT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_UINT_CONSTR_A_0010</span></b>](#E_MUL_UINT_CONSTR_A_0010) on tensor $A$.
- `[E_MUL_UINT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_UINT_CONSTR_A_0020</span></b>](#E_MUL_UINT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: unsigned integer tensor

Tensor $C$ is the element-wise result of the multiplication of $A$ by $B$.

#### Constraints

- `[E_MUL_UINT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_MUL_UINT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MUL_UINT_CONSTR_A_0020</span></b>](#E_MUL_UINT_CONSTR_A_0020) on tensor $A$.
