# Contents

- **Neg** operator for type [real](#real)
- **Neg** operator for types [`float16`, `float`, `double`](#float)
- **Neg** operator for types [`int8`, `int16`, `int32`, `int64`](#int)

Based on ONNX documentation [Neg version 13](https://onnx.ai/onnx/operators/onnx__Neg.html#neg-13).

Revision 2026-10-03: based on ONNX opset 14; first issue of this specification.

<a id="real"></a>

# **Neg** (real)

## Signature

$Y = \textbf{Neg}(X)$

where
- $X$: tensor whose elements are negated
- $Y$: result of the element-wise negation of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Neg** operator.

## Function

<a id="E_NEG_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_REAL_FUNC_0010]</br></span>

Operator **Neg** negates tensor $X$ element-wise and stores the result in tensor $Y$. Each element of $Y$ is the opposite of the element of $X$ that has the same index.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = -X[i]
$$

where
- $X[i]$ is the element of $X$ at index $i$,
- $-X[i]$ is the negation of $X[i]$ in $\mathbb{R}$.

The negation is exact: $Y[i]$ is the opposite of $X[i]$ in $\mathbb{R}$, with no approximation.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty and which holds a single element. The operator applies to that element, and $Y$ is a rank-0 tensor.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example

$$
X = \begin{bmatrix} -3 & 0 & \frac{1}{3} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} 3 & 0 & -\frac{1}{3} \end{bmatrix}
$$

The negation of a real number is exact: the third element of $Y$ is the exact opposite of $\frac{1}{3}$, and not a decimal approximation of it.

## Error conditions

None of the conditions of the list applies to real numbers: the negation is exact, so that the result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_REAL_FUNC_0010</span></b>](#E_NEG_REAL_FUNC_0010) for every operand, and no error can occur. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{X}$: real tensor

The tensor whose elements are negated.

#### Constraints

<a id="E_NEG_REAL_CONSTR_X_0010"></a>
- `[E_NEG_REAL_CONSTR_X_0010]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: real tensor

Tensor $Y$ is the element-wise negation of $X$.

#### Constraints

- `[E_NEG_REAL_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the shape of $X$.
- `[E_NEG_REAL_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_REAL_CONSTR_X_0010</span></b>](#E_NEG_REAL_CONSTR_X_0010) on tensor $X$.

<a id="float"></a>

# **Neg** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same negation for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

$Y = \textbf{Neg}(X)$

where
- $X$: tensor whose elements are negated
- $Y$: result of the element-wise negation of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Neg** operator.

## Function

<a id="E_NEG_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_FLOAT_FUNC_0010]</br></span>

Operator **Neg** negates tensor $X$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $Y$. Each element of $Y$ is the opposite of the element of $X$ that has the same index.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = -_{(\text{f})} X[i]
$$

where
- $X[i]$ is the element of $X$ at index $i$,
- $-_{(\text{f})}$ is the negation of the floating-point type of $X$, i.e. $-_{(\text{f16})}$, $-_{(\text{f32})}$ or $-_{(\text{f64})}$ according to that type: the result is the value of that type whose magnitude is the magnitude of $X[i]$ and whose sign is the opposite of the sign of $X[i]$.

The negation is exact: it changes the sign of the operand and leaves its magnitude unchanged, so that no rounding occurs and the result is a value of the type of $X$ for every operand. The cases of the domain are the following.

- **Finite operand.** $Y[i]$ is finite, of the opposite sign, and equal in magnitude to $X[i]$. No overflow and no underflow can occur, since the magnitude of the result is the magnitude of an operand.
- **Null operand.** If $X[i]$ is $+0$, then $Y[i]$ is $-0$; if $X[i]$ is $-0$, then $Y[i]$ is $+0$. The sign of a null result is thus the opposite of the sign of the null operand.
- **Infinite operand.** If $X[i]$ is $+\infty$, then $Y[i]$ is $-\infty$; if $X[i]$ is $-\infty$, then $Y[i]$ is $+\infty$.
- **NaN operand.** $Y[i]$ is a NaN. Its sign and its payload are not specified, so that every quiet NaN of the type of $X$ is a conforming result. The negation of a NaN does not signal the invalid operation of IEEE 754 section 7.2.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty and which holds a single element. The operator applies to that element, and $Y$ is a rank-0 tensor.

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

The negation of $+0$ is $-0$ and the negation of $-0$ is $+0$: the sign of a null result is the opposite of the sign of the operand. The negation of an infinity is the infinity of the opposite sign. The negation of a NaN is a NaN, whose sign and payload are not specified.

### Example 2

For the `float16` type:

$$
X = \begin{bmatrix} 65504 & 2^{-24} & 1.0 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} -65504 & -2^{-24} & -1.0 \end{bmatrix}
$$

The first element of $X$ is the largest finite value of `float16` and the second is the smallest positive subnormal value of that type. Their negations are finite values of the type: the negation of a finite value never overflows and never underflows, whatever the magnitude of the operand.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs none of the operations of IEEE 754 section 7.2, and the negation of a NaN is specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010) |
| Overflow | not applicable: the negation of a finite value of the type is a finite value of the type, as specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010) |
| Underflow | not applicable: the negation of a value of the type is a value of the type, as specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010) |
| Division by zero | not applicable: the operator performs no division |

Every condition of the list is ruled out by the semantics of the operator; none of them is an error. No error condition.

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

Tensor $Y$ is the element-wise negation of $X$.

#### Constraints

- `[E_NEG_FLOAT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the shape of $X$.
- `[E_NEG_FLOAT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_CONSTR_X_0010</span></b>](#E_NEG_FLOAT_CONSTR_X_0010) on tensor $X$.

<a id="int"></a>

# **Neg** (int)

where int is in {`int8`, `int16`, `int32`, `int64`}.

The four types share one semantics: each of them is an $n$-bit signed type, whose values range from $-2^{n-1}$ to $2^{n-1}-1$, and the negation of a value of the type is the value of that type that represents the opposite of that value modulo $2^n$.

## Signature

$Y = \textbf{Neg}(X)$

where
- $X$: tensor whose elements are negated
- $Y$: result of the element-wise negation of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Neg** operator.

## Function

<a id="E_NEG_INT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_INT_FUNC_0010]</br></span>

Operator **Neg** negates tensor $X$ element-wise and stores the result in tensor $Y$. Each element of $Y$ is the opposite of the element of $X$ that has the same index. Tensors $X$ and $Y$ have the same signed $n$-bit type, and the result is a value of that type.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] =
\begin{cases}
-X[i] & \text{if } X[i] \gt -2^{n-1} \\
-2^{n-1} & \text{if } X[i] = -2^{n-1}
\end{cases}
$$

where
- $X[i]$ is the element of $X$ at index $i$ and $-X[i]$ its negation in $\mathbb{Z}$,
- $n$ is the number of bits of the type of $X$.

The result is thus the value of the type that represents $-X[i]$ modulo $2^n$, read as a signed value. The values of an $n$-bit signed type lie in $[-2^{n-1}, 2^{n-1}-1]$, so that the negation of a value other than $-2^{n-1}$ lies in $[-(2^{n-1}-1), 2^{n-1}-1]$ and is a value of the type: the first case is the only one that applies to those values.

**The minimum value is its own negation.** The negation of the minimum value $-2^{n-1}$ of the type is $2^{n-1}$, which is not representable in the type; the result is $2^{n-1} - 2^n = -2^{n-1}$, i.e. the minimum value itself. For `int8`, the negation of $-128$ is $-128$. Tensors $X$ and $Y$ have the same type and no wider type is used, so that the value of $Y$ is that value of the type.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty and which holds a single element. The operator applies to that element, and $Y$ is a rank-0 tensor.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example

For the `int8` type:

$$
X = \begin{bmatrix} -6 & 0 & 100 & -128 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} 6 & 0 & -100 & -128 \end{bmatrix}
$$

The negations of $-6$, $0$ and $100$ lie in the range $[-2^7, 2^7-1]$ of `int8` and are given by the first case of the definition. The negation of the minimum value $-2^7 = -128$ is $128$, which is greater than the maximum value $2^7-1=127$ of `int8`; the second case applies and the result is $-128$, so that the minimum value of the type is its own negation.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Overflow, i.e. the negation of the minimum value of the type, such as $-(-128)$ for `int8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_INT_FUNC_0010</span></b>](#E_NEG_INT_FUNC_0010), the result is the minimum value itself |
| Division by zero | not applicable: the operator performs no division |

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

Tensor $Y$ is the element-wise negation of $X$.

#### Constraints

- `[E_NEG_INT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the shape of $X$.
- `[E_NEG_INT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_INT_CONSTR_X_0010</span></b>](#E_NEG_INT_CONSTR_X_0010) on tensor $X$.
