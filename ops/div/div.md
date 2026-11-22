# Contents

- **Div** operator for type [real](#real)
- **Div** operator for types [`float16`, `float`, `double`](#float)
- **Div** operator for types [`int8`, `int16`, `int32`, `int64`](#int)
- **Div** operator for types [`uint8`, `uint16`, `uint32`, `uint64`](#uint)

Based on ONNX documentation [Div version 14](https://onnx.ai/onnx/operators/onnx__Div.html#div-14).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **Div** (real, real)

## Signature

$C = \textbf{Div}(A, B)$

where
- $A$: dividend
- $B$: divisor
- $C$: result of the element-wise division of $A$ by $B$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Div** operator.

## Function

<a id="E_DIV_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_DIV_REAL_FUNC_0010]</br></span>

Operator **Div** divides tensor $A$ by tensor $B$ element-wise and stores the result in tensor $C$. Each element of $C$ is the quotient of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.

For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:

$$
C[i] = \frac{\tilde{A}[i]}{\tilde{B}[i]}
$$

where
- $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$,
- $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
- the quotient is the quotient in $\mathbb{R}$ of the two elements.

The division is exact: $C[i]$ is the quotient in $\mathbb{R}$, with no approximation.

**Domain.** The quotient of two real numbers exists only when the divisor is non-null. The domain of the operator is therefore the set of pairs of real tensors whose divisor has no null element, i.e. $\forall i,\ \tilde{B}[i] \neq 0$; that condition is the precondition stated by constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_REAL_CONSTR_B_0010</span></b>](#E_DIV_REAL_CONSTR_B_0010) on tensor $B$. For every pair of tensors that satisfies it, the result is the one given above.

**Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the quotient of the two operands and $C$ has the shape of the other operand.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty. No quotient is formed for an empty result, so that the domain condition bears on the elements of $\tilde{B}$ only.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
A = \begin{bmatrix} 7 & -9 & 1 \end{bmatrix}
\quad
B = \begin{bmatrix} 2 & 4 & 3 \end{bmatrix}
$$

$$
C = \begin{bmatrix} \frac{7}{2} & -\frac{9}{4} & \frac{1}{3} \end{bmatrix}
$$

The three quotients are given exactly, as the quotients of real numbers: $1/3$ has no finite decimal expansion and is not approximated.

### Example 2

$$
A = \begin{bmatrix} 4 & 9 \\ 16 & 25 \end{bmatrix}
\quad
B = \begin{bmatrix} 2 & 5 \end{bmatrix}
$$

$$
C = \begin{bmatrix} \frac{4}{2} & \frac{9}{5} \\[2pt] \frac{16}{2} & \frac{25}{5} \end{bmatrix} = \begin{bmatrix} 2 & \frac{9}{5} \\[2pt] 8 & 5 \end{bmatrix}
$$

Tensor $B$ has rank $1$ and tensor $A$ has rank $2$; $B$ is broadcast to the shape of $A$, so that each column of $A$ is divided by the corresponding element of $B$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Division by zero, i.e. a null element of $\tilde{B}$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_DIV_REAL_CONSTR_B_0010</span></b>](#E_DIV_REAL_CONSTR_B_0010) |
| Invalid operation (IEEE 754 section 7.2) | does not apply: the operands are real numbers, and no pair of real numbers with a non-null divisor is an invalid operation |
| Overflow | does not apply: the quotient of two real numbers is a real number, and no bound restricts the values of the type |

Every condition of the list is either ruled out by the precondition on $B$ or does not apply to real numbers; no condition remains that can produce a non-complying result. No error condition.

## Attributes

Operator **Div** has no attribute.

## Inputs

### $\text{A}$: real tensor

The dividend.

#### Constraints

<a id="E_DIV_REAL_CONSTR_A_0010"></a>
- `[E_DIV_REAL_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.

### $\text{B}$: real tensor

The divisor.

#### Constraints

<a id="E_DIV_REAL_CONSTR_B_0010"></a>
- `[E_DIV_REAL_CONSTR_B_0010]` Non-null divisor
  - Statement: Every element of $\tilde{B}$, the divisor broadcast to the shape of $C$, is non-null, i.e. $\forall i,\ \tilde{B}[i] \neq 0$.
  - Rationale: The quotient of two real numbers exists only when the divisor is non-null; the condition is the domain of the operator, stated in [<b><span style="font-family: 'Courier New', monospace">E_DIV_REAL_FUNC_0010</span></b>](#E_DIV_REAL_FUNC_0010).
- `[E_DIV_REAL_CONSTR_B_0020]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_REAL_CONSTR_A_0010</span></b>](#E_DIV_REAL_CONSTR_A_0010) on tensor $A$.

## Outputs

### $\text{C}$: real tensor

Tensor $C$ is the element-wise result of the division of $A$ by $B$.

#### Constraints

- `[E_DIV_REAL_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.

<a id="float"></a>

# **Div** (float, float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same division for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

$C = \textbf{Div}(A, B)$

where
- $A$: dividend
- $B$: divisor
- $C$: result of the element-wise division of $A$ by $B$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Div** operator.

## Function

<a id="E_DIV_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_DIV_FLOAT_FUNC_0010]</br></span>

Operator **Div** divides tensor $A$ by tensor $B$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $C$. Each element of $C$ is the floating-point quotient of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below. No element of $B$ is excluded: every combination of operands of the type, including the special numbers, has a result of the type.

For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:

$$
C[i] = \tilde{A}[i] /_{(\text{f})} \tilde{B}[i] =
\begin{cases}
\text{NaN} & \text{if } \tilde{A}[i] \text{ or } \tilde{B}[i] \text{ is NaN} \\
\text{NaN} & \text{if } \tilde{A}[i] \text{ and } \tilde{B}[i] \text{ are both } \pm\infty, \text{ or both } \pm 0 \\
\pm\infty & \text{if } \tilde{A}[i] \text{ is finite and non-null and } \tilde{B}[i] \text{ is } \pm 0, \text{ or } \tilde{A}[i] \text{ is } \pm\infty \text{ and } \tilde{B}[i] \text{ is finite and non-null} \\
\pm 0 & \text{if } \tilde{A}[i] \text{ is } \pm 0 \text{ and } \tilde{B}[i] \text{ is finite and non-null, or } \tilde{A}[i] \text{ is finite and } \tilde{B}[i] \text{ is } \pm\infty \\
\text{round}\!\left(\tilde{A}[i] / \tilde{B}[i]\right) & \text{otherwise}
\end{cases}
$$

where
- $/_{(\text{f})}$ is the division of the floating-point type of $A$, $B$ and $C$, i.e. $/_{(\text{f16})}$, $/_{(\text{f32})}$ or $/_{(\text{f64})}$ according to that type, whereas the $/$ of the last case is the division of $\mathbb{R}$: there, $\tilde{A}[i] / \tilde{B}[i]$ is the exact quotient of the two elements,
- $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
- $\pm$ in the third and fourth cases is the sign of the exact quotient: $+$ when the two operands have the same sign and $-$ when they have opposite signs,
- $\text{round}(x)$ is the value of $x$ rounded to the nearest value of the type of $A$ and $B$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded when deciding an overflow: a magnitude whose rounded value exceeds the largest finite value of the type gives an infinity of the sign of $x$, and a magnitude below the smallest subnormal value of the type gives a zero of the sign of $x$.

**Operands that are NaN.** The first case propagates the NaN: a NaN operand gives a NaN result, whatever the other operand is. The NaN of this case is a quiet NaN; its sign and its payload are not specified, so that every quiet NaN of the type of $A$ and $B$ is a conforming result.

**Invalid operation.** The second case is the invalid operation of IEEE 754 section 7.2 for division: the quotient $\infty/\infty$ and the quotient $0/0$ have no value in the extended reals, and the result is NaN, again any quiet NaN of the type.

**Division by zero.** The third case is the division-by-zero exception of IEEE 754: a finite non-null dividend divided by a zero gives an infinity whose sign is the sign of the exact quotient, so that $1.0/(+0)$ gives $+\infty$ and $-1.0/(+0)$ gives $-\infty$. The division of an infinity by a finite non-null value also gives an infinity, of the sign of the exact quotient; that case is not an exception of the standard, the result being the limit of the quotients of the finite values.

**Null quotients.** The fourth case gives a null result whose sign is the sign of the exact quotient: $\tilde{A}[i] / \tilde{B}[i]$ is null when the dividend is null and the divisor is finite non-null, and when the divisor is infinite and the dividend is finite. The signs of the operands then decide the sign of the zero: $(+0)/2.0$ gives $+0$ and $(-0)/2.0$ gives $-0$, while $(+0)/(-2.0)$ gives $-0$. A null dividend divided by a null divisor is not a null quotient: it falls in the second case and gives NaN. The last case also produces a null result when a non-null exact quotient is so small that it rounds to zero; its sign is then the sign of that exact quotient as well.

**Finite operands.** In the last case both operands are finite and the divisor is non-null, so that the exact quotient is a non-null real number. The result is that quotient rounded as defined above. When the exact quotient is representable in the type, no rounding takes place; otherwise the result is the nearest value of the type, and it may be a subnormal value, a zero, or an infinity when the exponent of the exact quotient exceeds the largest exponent of the type.

**Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the quotient of the two operands and $C$ has the shape of the other operand.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
A = \begin{bmatrix} 6.0 & -9.0 & 1.0 \end{bmatrix}
\quad
B = \begin{bmatrix} 3.0 & 4.0 & 8.0 \end{bmatrix}
$$

$$
C = \begin{bmatrix} 2.0 & -2.25 & 0.125 \end{bmatrix}
$$

The quotient of two numbers of a floating-point type is not always exact; in this example every quotient is representable and $C$ is the exact result.

### Example 2

$$
A = \begin{bmatrix} 1.0 & -1.0 & \text{+inf} & \text{+inf} & \text{+0} & \text{-0} & \text{+0} & \text{NaN} \end{bmatrix}
\quad
B = \begin{bmatrix} \text{+0} & \text{+0} & \text{+inf} & 4.0 & 4.0 & 4.0 & \text{+0} & 1.0 \end{bmatrix}
$$

$$
C = \begin{bmatrix} \text{+inf} & \text{-inf} & \text{NaN} & \text{+inf} & \text{+0} & \text{-0} & \text{NaN} & \text{NaN} \end{bmatrix}
$$

The first two elements are divisions by zero: the dividend is finite and non-null and the divisor is $+0$, so the result is an infinity whose sign is the sign of the exact quotient, positive for $1.0$ and negative for $-1.0$. The third element is the invalid operation $\infty/\infty$, which yields NaN. The fourth is the division of an infinity by a finite non-null value, which gives the infinity of the sign of the quotient. The fifth and the sixth elements are divisions of a zero by a finite non-null value: the exact quotient is null, and its sign, positive when the operands have the same sign and negative otherwise, is the sign of the result. The seventh element is the invalid operation $0/0$. The eighth is the propagation of a NaN operand.

### Example 3

$$
A = \begin{bmatrix} 1.0 & 10.0 & -10.0 \end{bmatrix}
\quad
B = \begin{bmatrix} 3.0 & 3.0 & 3.0 \end{bmatrix}
$$

$$
C \approx \begin{bmatrix} 0.3333333333333333 & 3.3333333333333335 & -3.3333333333333335 \end{bmatrix}
$$

None of the three exact quotients is representable in `double`: each element of $C$ is the exact quotient rounded to the nearest `double`, which is the value displayed. The departure of an implementation from these values is the subject of the profile's accuracy guidelines.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. the quotients $\infty/\infty$ and $0/0$ | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_DIV_FLOAT_FUNC_0010</span></b>](#E_DIV_FLOAT_FUNC_0010), the result is NaN |
| Division by zero, i.e. a finite non-null dividend and a null divisor | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_DIV_FLOAT_FUNC_0010</span></b>](#E_DIV_FLOAT_FUNC_0010), the result is an infinity of the sign of the exact quotient |
| Overflow, i.e. an exact quotient whose rounded magnitude exceeds the largest finite value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_DIV_FLOAT_FUNC_0010</span></b>](#E_DIV_FLOAT_FUNC_0010), the result is an infinity of the sign of the exact quotient |
| Underflow, i.e. an exact quotient whose magnitude is below the smallest normal value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_DIV_FLOAT_FUNC_0010</span></b>](#E_DIV_FLOAT_FUNC_0010), the result is that quotient rounded to the type, a subnormal value or a zero of the sign of the exact quotient |

Every condition of the list is part of the nominal behavior of the operator; none of them is an error. No error condition.

## Attributes

Operator **Div** has no attribute.

## Inputs

### $\text{A}$: floating-point tensor

The dividend.

#### Constraints

<a id="E_DIV_FLOAT_CONSTR_A_0010"></a>
- `[E_DIV_FLOAT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_DIV_FLOAT_CONSTR_A_0020"></a>
- `[E_DIV_FLOAT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$ and $C$ have the same type.

### $\text{B}$: floating-point tensor

The divisor.

#### Constraints

- `[E_DIV_FLOAT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_FLOAT_CONSTR_A_0010</span></b>](#E_DIV_FLOAT_CONSTR_A_0010) on tensor $A$.
- `[E_DIV_FLOAT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_FLOAT_CONSTR_A_0020</span></b>](#E_DIV_FLOAT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: floating-point tensor

Tensor $C$ is the element-wise result of the division of $A$ by $B$.

#### Constraints

- `[E_DIV_FLOAT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_DIV_FLOAT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_FLOAT_CONSTR_A_0020</span></b>](#E_DIV_FLOAT_CONSTR_A_0020) on tensor $A$.

<a id="int"></a>

# **Div** (int, int)

where int is in {`int8`, `int16`, `int32`, `int64`}.

The four types share one semantics: each of them is an $n$-bit signed type whose values range from $-2^{n-1}$ to $2^{n-1}-1$, and the result of the division is the quotient of the two elements truncated toward zero, read modulo $2^n$ as a value of that type.

## Signature

$C = \textbf{Div}(A, B)$

where
- $A$: dividend
- $B$: divisor
- $C$: result of the element-wise division of $A$ by $B$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Div** operator.

## Function

<a id="E_DIV_INT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_DIV_INT_FUNC_0010]</br></span>

Operator **Div** divides tensor $A$ by tensor $B$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the quotient of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below, truncated toward zero. Tensors $A$, $B$ and $C$ have the same signed $n$-bit type, and the result is a value of that type.

For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:

$$
C[i] =
\begin{cases}
\operatorname{trunc}\!\left(\tilde{A}[i] / \tilde{B}[i]\right) & \text{if } -2^{n-1} \le \operatorname{trunc}\!\left(\tilde{A}[i] / \tilde{B}[i]\right) \le 2^{n-1}-1 \\[4pt]
\operatorname{trunc}\!\left(\tilde{A}[i] / \tilde{B}[i]\right) - 2^n & \text{if } \operatorname{trunc}\!\left(\tilde{A}[i] / \tilde{B}[i]\right) \notin [-2^{n-1},\, 2^{n-1}-1]
\end{cases}
$$

where
- $\tilde{A}[i] / \tilde{B}[i]$ is the exact quotient of the two elements, the quotient in $\mathbb{Q}$ of two integers whose divisor is non-null,
- $\operatorname{trunc}(x)$ is the integer part of $x$, i.e. $x$ rounded toward zero: $\operatorname{trunc}(x) = \lfloor x \rfloor$ when $x \ge 0$ and $\operatorname{trunc}(x) = \lceil x \rceil$ when $x < 0$, so that $\operatorname{trunc}(3.5) = 3$ and $\operatorname{trunc}(-3.5) = -3$,
- $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
- $n$ is the number of bits of the type of $A$ and $B$.

The second case is the complement of the first: exactly one of the two cases applies to every pair of operands whose divisor is non-null, and the result is then determined.

**Truncating division.** The quotient $\operatorname{trunc}(a/b)$ is the unique integer $q$ such that $a = q \cdot b + r$, with $|r| < |b|$ and with $r$ null or of the sign of $a$. It is the quotient ONNX prescribes for the integer types, and it differs from the floor: for `int8`, $7/2 = 3$ and $-7/2 = -3$, whereas the floor of $-3.5$ is $-4$. The sign of the result is the sign of the exact quotient: the quotient of two operands of the same sign is non-negative, and the quotient of two operands of opposite signs is non-positive, so that a null result is the neutral value $0$ of the type, with no sign.

**The one quotient outside the range.** The values of an $n$-bit signed type lie in $[-2^{n-1}, 2^{n-1}-1]$, so that the quotient of two of its values, truncated toward zero, lies in $[-2^{n-1}, 2^{n-1}]$: the lower bound is reached by $(-2^{n-1})/1 = -2^{n-1}$, and the upper bound only by $(-2^{n-1})/(-1) = 2^{n-1}$. Only the value $2^{n-1}$ leaves the range of the type, so that the second case of the definition is the case $\operatorname{trunc}(\tilde{A}[i]/\tilde{B}[i]) = 2^{n-1}$, reached by that single pair of operands; the result is then $2^{n-1} - 2^n = -2^{n-1}$. For `int8`, $-128 / -1$ gives $-128$; the exact quotient $128$ is represented by the value of the type that is congruent to it modulo $2^8$.

**A null divisor.** No element of $B$ is excluded, so that a null element of $B$ makes the exact quotient undefined and the result of the operator undetermined; the condition is described in the "Error conditions" section. Every other pair of operands falls in one of the two cases above, and the result is fully determined.

**Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the quotient of the two operands and $C$ has the shape of the other operand.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty. No quotient is formed for an empty result, so that a null element of $B$ in an empty tensor raises no condition.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `int8` type:

$$
A = \begin{bmatrix} 7 & -7 & 7 & -7 & -128 \end{bmatrix}
\quad
B = \begin{bmatrix} 2 & 2 & -2 & -2 & -1 \end{bmatrix}
$$

$$
C = \begin{bmatrix} 3 & -3 & -3 & 3 & -128 \end{bmatrix}
$$

The first four quotients are $3.5$, $-3.5$, $-3.5$ and $3.5$: each is truncated toward zero, which gives $3$, $-3$, $-3$ and $3$, and not the floors $-3.5 \to -4$. The last quotient is $(-128)/(-1) = 128$, which exceeds the maximum value $127$ of `int8`: the second case of the definition applies and the result is $128 - 2^8 = -128$, the value of the type congruent to $128$ modulo $2^8$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Division by zero, i.e. a null element of $B$ | error: no element of $B$ is excluded, so that the exact quotient is undefined and the result is not determined; the condition is described below |
| Integer overflow, i.e. the quotient $2^{n-1}$ of the pair $(-2^{n-1}, -1)$ | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_DIV_INT_FUNC_0010</span></b>](#E_DIV_INT_FUNC_0010), the result is $-2^{n-1}$ |
| Invalid operation (IEEE 754 section 7.2) | does not apply: the operands are integers, and no pair of integers with a non-null divisor is an invalid operation |

**Division by zero.** For an element $i$ with $\tilde{B}[i] = 0$, the exact quotient $\tilde{A}[i]/\tilde{B}[i]$ has no value, and the second case of [<b><span style="font-family: 'Courier New', monospace">E_DIV_INT_FUNC_0010</span></b>](#E_DIV_INT_FUNC_0010) gives no result either: the specification determines no value for $C[i]$. What happens then depends on the implementation: a division-by-zero exception raised by the hardware, a result produced by the implementation's own handling of the case, or an abort of the execution. An implementation that tests the divisor before dividing detects the condition instead of producing a value, and reports it to its caller; the value it reports and the means by which it reports it are outside this specification. The condition bears on the elements of $B$ that are divided, so that a tensor $C$ that is empty raises it for no element.

## Attributes

Operator **Div** has no attribute.

## Inputs

### $\text{A}$: signed integer tensor

The dividend.

#### Constraints

<a id="E_DIV_INT_CONSTR_A_0010"></a>
- `[E_DIV_INT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_DIV_INT_CONSTR_A_0020"></a>
- `[E_DIV_INT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$ and $C$ have the same type.

### $\text{B}$: signed integer tensor

The divisor. Its elements may be null, and the result of the operator is then undetermined for the corresponding elements of $C$.

#### Constraints

- `[E_DIV_INT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_INT_CONSTR_A_0010</span></b>](#E_DIV_INT_CONSTR_A_0010) on tensor $A$.
- `[E_DIV_INT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_INT_CONSTR_A_0020</span></b>](#E_DIV_INT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: signed integer tensor

Tensor $C$ is the element-wise result of the division of $A$ by $B$.

#### Constraints

- `[E_DIV_INT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_DIV_INT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_INT_CONSTR_A_0020</span></b>](#E_DIV_INT_CONSTR_A_0020) on tensor $A$.

<a id="uint"></a>

# **Div** (uint, uint)

where uint is in {`uint8`, `uint16`, `uint32`, `uint64`}.

The four types share one semantics: each of them is an $n$-bit unsigned type whose values range from $0$ to $2^n-1$, and the result of the division is the quotient of the two elements, truncated toward zero, which lies in the range of the type.

## Signature

$C = \textbf{Div}(A, B)$

where
- $A$: dividend
- $B$: divisor
- $C$: result of the element-wise division of $A$ by $B$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Div** operator.

## Function

<a id="E_DIV_UINT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_DIV_UINT_FUNC_0010]</br></span>

Operator **Div** divides tensor $A$ by tensor $B$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the quotient of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below, truncated toward zero. Tensors $A$, $B$ and $C$ have the same unsigned $n$-bit type, and the result is a value of that type.

For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:

$$
C[i] = \left\lfloor \frac{\tilde{A}[i]}{\tilde{B}[i]} \right\rfloor
$$

where
- $\tilde{A}[i] / \tilde{B}[i]$ is the exact quotient of the two elements, the quotient in $\mathbb{Q}$ of two integers whose divisor is non-null,
- $\lfloor \cdot \rfloor$ is the floor, which for two non-negative operands is the quotient truncated toward zero,
- $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$.

**The result never leaves the range of the type.** The operands are non-negative, so that their exact quotient is non-negative and its floor is at most the dividend: $0 \le \lfloor a/b \rfloor \le a \le 2^n-1$ for every $b \ge 1$. The result of a division by a non-null element is therefore the floor itself, with no reduction modulo $2^n$ and no overflow, and it is exact in the type. For `uint8`, $200/3$ gives $66$ and $255/255$ gives $1$. The truncation and the floor coincide on the unsigned types, so that the distinction between them, which the signed types make visible, has no effect here.

**A null divisor.** No element of $B$ is excluded, so that a null element of $B$ makes the exact quotient undefined and the result of the operator undetermined; the condition is described in the "Error conditions" section, and it is the only condition under which the result is not the value given above.

**Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the quotient of the two operands and $C$ has the shape of the other operand.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty. No quotient is formed for an empty result, so that a null element of $B$ in an empty tensor raises no condition.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `uint8` type:

$$
A = \begin{bmatrix} 7 & 200 & 255 & 6 \end{bmatrix}
\quad
B = \begin{bmatrix} 2 & 3 & 255 & 6 \end{bmatrix}
$$

$$
C = \begin{bmatrix} 3 & 66 & 1 & 1 \end{bmatrix}
$$

The quotient $7/2 = 3.5$ is truncated toward zero and gives $3$; $200/3 = 66.67$ gives $66$; $255/255 = 1$ and $6/6 = 1$ are exact. Each of the four results lies in the range $[0, 255]$ of `uint8`, as every quotient of the type does.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Division by zero, i.e. a null element of $B$ | error: no element of $B$ is excluded, so that the exact quotient is undefined and the result is not determined; the condition is described below |
| Integer overflow | does not apply: the quotient of two values of the type with a non-null divisor lies in the range of the type, as stated by [<b><span style="font-family: 'Courier New', monospace">E_DIV_UINT_FUNC_0010</span></b>](#E_DIV_UINT_FUNC_0010) |
| Invalid operation (IEEE 754 section 7.2) | does not apply: the operands are integers, and no pair of integers with a non-null divisor is an invalid operation |

**Division by zero.** For an element $i$ with $\tilde{B}[i] = 0$, the exact quotient $\tilde{A}[i]/\tilde{B}[i]$ has no value, and the formula of [<b><span style="font-family: 'Courier New', monospace">E_DIV_UINT_FUNC_0010</span></b>](#E_DIV_UINT_FUNC_0010) gives no result either: the specification determines no value for $C[i]$. What happens then depends on the implementation: a division-by-zero exception raised by the hardware, a result produced by the implementation's own handling of the case, or an abort of the execution. An implementation that tests the divisor before dividing detects the condition instead of producing a value, and reports it to its caller; the value it reports and the means by which it reports it are outside this specification. The condition bears on the elements of $B$ that are divided, so that a tensor $C$ that is empty raises it for no element.

## Attributes

Operator **Div** has no attribute.

## Inputs

### $\text{A}$: unsigned integer tensor

The dividend.

#### Constraints

<a id="E_DIV_UINT_CONSTR_A_0010"></a>
- `[E_DIV_UINT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_DIV_UINT_CONSTR_A_0020"></a>
- `[E_DIV_UINT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$ and $C$ have the same type.

### $\text{B}$: unsigned integer tensor

The divisor. Its elements may be null, and the result of the operator is then undetermined for the corresponding elements of $C$.

#### Constraints

- `[E_DIV_UINT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_UINT_CONSTR_A_0010</span></b>](#E_DIV_UINT_CONSTR_A_0010) on tensor $A$.
- `[E_DIV_UINT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_UINT_CONSTR_A_0020</span></b>](#E_DIV_UINT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: unsigned integer tensor

Tensor $C$ is the element-wise result of the division of $A$ by $B$.

#### Constraints

- `[E_DIV_UINT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_DIV_UINT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_UINT_CONSTR_A_0020</span></b>](#E_DIV_UINT_CONSTR_A_0020) on tensor $A$.
