# Contents

- **Neg** operator for type [real](#real)
- **Neg** operator for types [`float16`, `float`, `double`](#float)
- **Neg** operator for types [`int8`, `int16`, `int32`, `int64`](#int)

Based on ONNX documentation [Neg version 14](https://onnx.ai/onnx/operators/onnx__Neg.html#neg-14).

Revision 2026-10-04: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **Neg** (real)

## Signature

$Y = \textbf{Neg}(X)$

where
- $X$: value whose sign is flipped
- $Y$: result of the element-wise negation of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Neg** operator.

## Function

<a id="E_NEG_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_REAL_FUNC_0010]</br></span>

**Part 1.** Operator **Neg** flips the sign of every element of tensor $X$ and stores the result in tensor $Y$.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = -X[i]
$$

where
- $X[i]$ is the element of $X$ at index $i$,
- $-X[i]$ is the negation of the real number $X[i]$, i.e. the real number whose sum with $X[i]$ is $0$.

The negation is exact: $Y[i]$ is the opposite in $\mathbb{R}$ of $X[i]$, with no approximation.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It has a single element, and $Y$ is the rank-0 tensor whose element is the negation of that element.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Tensor $Y$ then has the same shape as $X$ and is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

$$
X = \begin{bmatrix} 1.5 & -2.25 \\ -0.5 & 3 \end{bmatrix}
\quad
Y = \begin{bmatrix} -1.5 & 2.25 \\ 0.5 & -3 \end{bmatrix}
$$

Each element of $Y$ is the opposite in $\mathbb{R}$ of the corresponding element of $X$; the negation is exact.

## Error conditions

None of the conditions of the list applies to real numbers: the negation is defined for every real number, it is exact, and its result is a real number, so that no invalid operation, no overflow and no division by zero can occur. The result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_REAL_FUNC_0010</span></b>](#E_NEG_REAL_FUNC_0010) for every operand. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{X}$: real tensor

The tensor whose elements are negated.

#### Constraints

<a id="E_NEG_REAL_CONSTR_X_0010"></a>
- `[E_NEG_REAL_CONSTR_X_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as $X$.

### $\text{Y}$: real tensor

Tensor $Y$ is the element-wise negation of $X$.

#### Constraints

- `[E_NEG_REAL_CONSTR_Y_0010]` Shape definition
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_REAL_CONSTR_X_0010</span></b>](#E_NEG_REAL_CONSTR_X_0010) on tensor $X$.

<a id="float"></a>

# **Neg** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same sign change for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

$Y = \textbf{Neg}(X)$

where
- $X$: value whose sign is flipped
- $Y$: result of the element-wise negation of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Neg** operator.

## Function

<a id="E_NEG_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_FLOAT_FUNC_0010]</br></span>

**Part 1.** Operator **Neg** flips the sign of every element of tensor $X$ according to IEEE 754 floating-point semantics and stores the result in tensor $Y$.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = -_{(\text{f})} X[i] =
\begin{cases}
\text{NaN} & \text{if } X[i] \text{ is NaN} \\
-X[i] & \text{otherwise}
\end{cases}
$$

where
- $-_{(\text{f})}$ is the sign change of the floating-point type of $X$ and $Y$, i.e. $-_{(\text{f16})}$, $-_{(\text{f32})}$ or $-_{(\text{f64})}$ according to that type,
- $-X[i]$ is the value of that type whose sign is the opposite of the sign of $X[i]$ and whose magnitude is the magnitude of $X[i]$.

The negation is exact: it changes the sign of the operand and leaves its magnitude unchanged, so that no rounding occurs, whatever the operand. In particular, the negation of a subnormal value is the subnormal value of the opposite sign, and the negation of a finite value is never an infinity.

**The special values.** The operand may be any of the values of the type, including the special numbers $\pm 0$, $\pm\text{inf}$ and NaN. The negation of $+0$ is $-0$ and the negation of $-0$ is $+0$; the negation of $+\text{inf}$ is $-\text{inf}$ and the negation of $-\text{inf}$ is $+\text{inf}$; and the negation of a NaN is a NaN. The NaN of the first case is a quiet NaN; its sign and its payload are not specified, so that every quiet NaN of the type of $X$ and $Y$ is a conforming result.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It has a single element, and $Y$ is the rank-0 tensor whose element is the negation of that element.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Tensor $Y$ then has the same shape as $X$ and is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix} 1.5 & \text{+0} & \text{-0} & \text{+inf} & \text{-inf} & \text{NaN} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} -1.5 & \text{-0} & \text{+0} & \text{-inf} & \text{+inf} & \text{NaN} \end{bmatrix}
$$

The negation of $+0$ is $-0$ and the negation of $-0$ is $+0$; the negation of an infinity is the infinity of the opposite sign; and the negation of NaN is a quiet NaN, whose sign and payload are not specified, so that the NaN shown for the last element is one conforming result.

### Example 2

$$
X = \begin{bmatrix} 0.1 & -0.1 & 1.0 \end{bmatrix}
\quad
Y = \begin{bmatrix} -0.1 & 0.1 & -1.0 \end{bmatrix}
$$

The negation is exact even when the operand is not exactly representable in the type: the `double` value nearest to $0.1$ is not $0.1$, and the first element of $Y$ is the `double` value nearest to $-0.1$, which is the opposite of that value. No rounding occurs.

## Error conditions

The negation of a floating-point value is exact: it changes the sign of the operand and leaves its magnitude unchanged, so that no rounding occurs and the result is a value of the type of the operand. None of the conditions of the list applies: the negation of a non-NaN operand is never NaN, so that no invalid operation occurs; the magnitude of the result is that of the operand, so that no overflow and no underflow occurs; and the operator performs no division. The result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010) for every operand. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{X}$: floating-point tensor

The tensor whose elements are negated.

#### Constraints

<a id="E_NEG_FLOAT_CONSTR_X_0010"></a>
- `[E_NEG_FLOAT_CONSTR_X_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as $X$.
<a id="E_NEG_FLOAT_CONSTR_X_0020"></a>
- `[E_NEG_FLOAT_CONSTR_X_0020]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

### $\text{Y}$: floating-point tensor

Tensor $Y$ is the element-wise negation of $X$.

#### Constraints

- `[E_NEG_FLOAT_CONSTR_Y_0010]` Shape definition
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_CONSTR_X_0010</span></b>](#E_NEG_FLOAT_CONSTR_X_0010) on tensor $X$.
- `[E_NEG_FLOAT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_CONSTR_X_0020</span></b>](#E_NEG_FLOAT_CONSTR_X_0020) on tensor $X$.

<a id="int"></a>

# **Neg** (int)

where int is in {`int8`, `int16`, `int32`, `int64`}.

The four types share one semantics: each of them is an $n$-bit signed type, whose values range from $-2^{n-1}$ to $2^{n-1}-1$, and the result of the negation is the value of that type that represents the exact negation of the element modulo $2^n$.

## Signature

$Y = \textbf{Neg}(X)$

where
- $X$: value whose sign is flipped
- $Y$: result of the element-wise negation of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Neg** operator.

## Function

<a id="E_NEG_INT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_INT_FUNC_0010]</br></span>

**Part 1.** Operator **Neg** flips the sign of every element of tensor $X$ and stores the result in tensor $Y$. Tensors $X$ and $Y$ have the same signed $n$-bit type, and the result is a value of that type.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] =
\begin{cases}
-X[i] & \text{if } -X[i] \le 2^{n-1}-1 \\
-X[i] - 2^n & \text{if } -X[i] > 2^{n-1}-1
\end{cases}
$$

where
- $-X[i]$ is the exact negation of the element $X[i]$, the opposite in $\mathbb{Z}$ of the integer $X[i]$, before the reduction modulo $2^n$,
- $n$ is the number of bits of the type of $X$ and $Y$.

The result is thus the exact negation modulo $2^n$, read as a signed value. The values of an $n$-bit signed type lie in $[-2^{n-1}, 2^{n-1}-1]$, so that their exact negation lies in $[-(2^{n-1}-1), 2^{n-1}]$: the second case applies only to the minimum value $-2^{n-1}$ of the type, whose negation $2^{n-1}$ is greater than the maximum value $2^{n-1}-1$, and the result is then $2^{n-1} - 2^n = -2^{n-1}$.

**The minimum value of the type is its own negation.** For `int8`, $-(-128) = 128$ is greater than the maximum value $2^7-1 = 127$ of the type, so that the second case applies and the result is $128 - 2^8 = -128$. The negation of the minimum value of the type therefore has the same sign as its operand, and the negation is not a sign change for that value. Tensors $X$ and $Y$ have the same type and no wider type is used, so that the value of $Y$ is that value of the type, sign included.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It has a single element, and $Y$ is the rank-0 tensor whose element is the negation of that element.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Tensor $Y$ then has the same shape as $X$ and is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `int8` type:

$$
X = \begin{bmatrix} 6 & -6 & 127 & -128 \end{bmatrix}
\quad
Y = \begin{bmatrix} -6 & 6 & -127 & -128 \end{bmatrix}
$$

The negations of $6$, $-6$ and $127$ lie in $[-2^7, 2^7-1] = [-128, 127]$ and are given by the first case of the definition. The negation of $-128$, the minimum value of `int8`, is $128$, which is greater than the maximum value $127$ of the type, so that the second case applies and the result is $128 - 2^8 = -128$: the negation of the minimum value of the type is that value itself.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Integer overflow, i.e. the negation of the minimum value of the type, such as $-(-128) = 128$ for `int8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_INT_FUNC_0010</span></b>](#E_NEG_INT_FUNC_0010), the result is the value of the type that
