# Contents

- **Neg** operator for type [real](#real)
- **Neg** operator for types [`float16`, `float`, `double`](#float)
- **Neg** operator for types [`int8`, `int16`, `int32`, `int64`](#int)

Based on ONNX documentation [Neg version 13](https://onnx.ai/onnx/operators/onnx__Neg.html#neg-13).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **Neg** (real)

## Signature

$Y = \textbf{Neg}(X)$

where
- $X$: value whose opposite is computed
- $Y$: result of the element-wise negation of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Neg** operator.

## Function

<a id="E_NEG_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_REAL_FUNC_0010]</br></span>

**Part 1.** Operator **Neg** negates tensor $X$ element-wise and stores the result in tensor $Y$. Each element of $Y$ is the opposite of the corresponding element of $X$.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = -X[i]
$$

where
- $X[i]$ is the element of $X$ at index $i$,
- $-X[i]$ is the opposite of $X[i]$ in $\mathbb{R}$.

The negation is exact: $Y[i]$ is the opposite in $\mathbb{R}$ of $X[i]$, with no approximation.

**Shape of the result.** Tensor $Y$ has the same shape as tensor $X$: the operator is unary, so that no broadcasting takes place and each element of $X$ has exactly one corresponding element in $Y$.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. The negation of a rank-0 tensor is a rank-0 tensor, whose single element is the opposite of the element of $X$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

$$
X = \begin{bmatrix} -3 & 0 \\ \frac{1}{3} & -\frac{7}{2} \end{bmatrix}
\quad
Y = \begin{bmatrix} 3 & 0 \\ -\frac{1}{3} & \frac{7}{2} \end{bmatrix}
$$

The opposite of $0$ is $0$, and the opposite of each of the other elements is exact.

## Error conditions

None of the conditions of the list applies to real numbers: the negation is exact, so that the result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_REAL_FUNC_0010</span></b>](#E_NEG_REAL_FUNC_0010) for every operand, and no error can occur. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{X}$: real tensor

The tensor whose elements are negated.

#### Constraints

No constraint applies to tensor $X$.

## Outputs

### $\text{Y}$: real tensor

Tensor $Y$ is the element-wise result of the negation of $X$.

#### Constraints

<a id="E_NEG_REAL_CONSTR_Y_0010"></a>
- `[E_NEG_REAL_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as tensor $X$.

<a id="float"></a>

# **Neg** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same negation for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

$Y = \textbf{Neg}(X)$

where
- $X$: value whose opposite is computed
- $Y$: result of the element-wise negation of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Neg** operator.

## Function

<a id="E_NEG_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_FLOAT_FUNC_0010]</br></span>

**Part 1.** Operator **Neg** negates tensor $X$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $Y$. Each element of $Y$ is the opposite of the corresponding element of $X$.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = -_{(\text{f})} X[i]
$$

where $-_{(\text{f})}$ is the negation of the floating-point type of $X$ and $Y$, i.e. $-_{(\text{f16})}$, $-_{(\text{f32})}$ or $-_{(\text{f64})}$ according to that type.

The negation of a floating-point value is exact: no rounding takes place, so that neither overflow nor underflow can occur, whatever the value of the operand. For every value of the type other than NaN, the result is the operand with its sign bit inverted.

**Values of the operands.** The operands may be any of the values of the type, including the special numbers $\pm 0$, $\pm\text{inf}$ and NaN. The result is:
- for a finite non-zero value, the value of the same magnitude and of the opposite sign;
- for $+0$, the value $-0$, and for $-0$, the value $+0$;
- for $+\text{inf}$, the value $-\text{inf}$, and for $-\text{inf}$, the value $+\text{inf}$;
- for a NaN, a NaN; the sign and the payload of the result are not specified, so that every quiet NaN of the type of $X$ is a conforming result.

**Shape of the result.** Tensor $Y$ has the same shape as tensor $X$: the operator is unary, so that no broadcasting takes place and each element of $X$ has exactly one corresponding element in $Y$.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. The negation of a rank-0 tensor is a rank-0 tensor, whose single element is the opposite of the element of $X$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix} 1.5 & \text{+0} & \text{-0} & \text{+inf} & \text{-inf} & \text{NaN} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} -1.5 & \text{-0} & \text{+0} & \text{-inf} & \text{+inf} & \text{NaN} \end{bmatrix}
$$

The negation inverts the sign of every value, including the null values: the opposite of $+0$ is $-0$ and the opposite of $-0$ is $+0$. The opposite of an infinity is the infinity of the opposite sign. The last element of $Y$ is a NaN; its sign and its payload are not specified.

### Example 2

$$
X = \begin{bmatrix} 2^{-1070} & 2^{1023} \end{bmatrix}
\quad
Y = \begin{bmatrix} -2^{-1070} & -2^{1023} \end{bmatrix}
$$

The first element of $X$ is a subnormal `double` value and the second is the largest power of two of `double`. The negation of both is exact: the result has the same magnitude, so that no underflow occurs for the first and no overflow for the second.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. an operation of IEEE 754 section 7.2 | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010), which gives the result for every value of the type, including the special numbers |
| Overflow, i.e. a result that rounds beyond the largest finite value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010), the negation is exact and no rounding takes place |
| Underflow, i.e. a result with a magnitude below the smallest normal value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010), the negation is exact and no rounding takes place |

Every condition of the list is part of the nominal behavior of the operator; none of them is an error. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{X}$: floating-point tensor

The tensor whose elements are negated.

#### Constraints

<a id="E_NEG_FLOAT_CONSTR_X_0010"></a>
- `[E_NEG_FLOAT_CONSTR_X_0010]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: floating-point tensor

Tensor $Y$ is the element-wise result of the negation of $X$.

#### Constraints

<a id="E_NEG_FLOAT_CONSTR_Y_0010"></a>
- `[E_NEG_FLOAT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as tensor $X$.
- `[E_NEG_FLOAT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_CONSTR_X_0010</span></b>](#E_NEG_FLOAT_CONSTR_X_0010) on tensor $X$.

<a id="int"></a>

# **Neg** (int)

where int is in {`int8`, `int16`, `int32`, `int64`}.

The four types share one semantics: each of them is an $n$-bit signed type, whose values range from $-2^{n-1}$ to $2^{n-1}-1$, and the result of the negation is the value of that type that represents the exact opposite of the element modulo $2^n$.

## Signature

$Y = \textbf{Neg}(X)$

where
- $X$: value whose opposite is computed
- $Y$: result of the element-wise negation of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Neg** operator.

## Function

<a id="E_NEG_INT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_INT_FUNC_0010]</br></span>

**Part 1.** Operator **Neg** negates tensor $X$ element-wise and stores the result in tensor $Y$. Each element of $Y$ is the opposite of the corresponding element of $X$, in the type of $X$.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] =
\begin{cases}
-X[i] & \text{if } X[i] \gt -2^{n-1} \\
-2^{n-1} & \text{if } X[i] = -2^{n-1}
\end{cases}
$$

where
- $X[i]$ is the element of $X$ at index $i$, and $-X[i]$ is its exact opposite in $\mathbb{Z}$,
- $n$ is the number of bits of the type of $X$ and $Y$.

The result is thus the exact opposite of the element modulo $2^n$, read as a signed value. The values of an $n$-bit signed type lie in $[-2^{n-1}, 2^{n-1}-1]$, so that their exact opposites lie in $[-(2^{n-1}-1), 2^{n-1}]$: the only value whose opposite leaves the range of the type is the minimum value $-2^{n-1}$, whose opposite $2^{n-1}$ is reduced to $2^{n-1} - 2^n = -2^{n-1}$.

**The minimum value is its own opposite.** The range of the type is not symmetric about zero, so that the negation of the minimum value gives that value itself: for `int8`, $-(-128)$ gives $-128$. Tensors $X$ and $Y$ have the same type and no wider type is used, so that the value of $Y$ is that value of the type.

**Shape of the result.** Tensor $Y$ has the same shape as tensor $X$: the operator is unary, so that no broadcasting takes place and each element of $X$ has exactly one corresponding element in $Y$.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. The negation of a rank-0 tensor is a rank-0 tensor, whose single element is the opposite of the element of $X$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `int8` type:

$$
X = \begin{bmatrix} -6 & 0 & 100 & -128 \end{bmatrix}
\quad
Y = \begin{bmatrix} 6 & 0 & -100 & -128 \end{bmatrix}
$$

The opposites of $-6$, $0$ and $100$ lie in $[-2^7, 2^7-1]$ and are given by the first case of the definition. The opposite of the minimum value $-128$ is $128$, which is greater than the maximum value $2^7-1=127$ of `int8`; the second case applies and the result is $128-2^8=-128$, the minimum value itself.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Overflow, i.e. an exact opposite outside the range of the type, such as $-(-128)=128$ for `int8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_INT_FUNC_0010</span></b>](#E_NEG_INT_FUNC_0010), the result is the value of the type that represents the opposite modulo $2^n$, which is the minimum value itself |

The only condition of the list that applies to the signed integer types is the overflow, and it is part of the nominal behavior of the operator; there is no error. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{X}$: signed integer tensor

The tensor whose elements are negated.

#### Constraints

<a id="E_NEG_INT_CONSTR_X_0010"></a>
- `[E_NEG_INT_CONSTR_X_0010]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: signed integer tensor

Tensor $Y$ is the element-wise result of the negation of $X$.

#### Constraints

<a id="E_NEG_INT_CONSTR_Y_0010"></a>
- `[E_NEG_INT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as tensor $X$.
- `[E_NEG_INT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_INT_CONSTR_X_0010</span></b>](#E_NEG_INT_CONSTR_X_0010) on tensor $X$.
