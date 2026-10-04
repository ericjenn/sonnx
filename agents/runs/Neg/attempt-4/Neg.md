# Contents

- **Neg** operator for type [real](#real)
- **Neg** operator for types [`float16`, `float`, `double`](#float)
- **Neg** operator for types [`int8`, `int16`, `int32`, `int64`](#int)

Based on ONNX documentation [Neg version 14](https://onnx.ai/onnx/operators/onnx__Neg.html#neg-14).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

Revision 2026-10-04: the `uint` section is withdrawn, ONNX admitting no unsigned type for **Neg**; the `float` section states that `bfloat16` is outside the SONNX profile.

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

Operator **Neg** flips the sign of every element of tensor $X$ and stores the result in tensor $Y$. Each element of $Y$ is the additive inverse of the element of $X$ that has the same index.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = -X[i]
$$

where
- $X[i]$ is the element of $X$ at index $i$,
- $-X[i]$ is the additive inverse of $X[i]$ in $\mathbb{R}$, i.e. the value such that $X[i] + (-X[i]) = 0$.

The negation is exact: $Y[i]$ is the value of $\mathbb{R}$ opposite to $X[i]$, with no approximation.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. When $X$ is such a tensor, $Y$ is a rank-0 tensor whose single element is the negation of the single element of $X$.

**Zero-sized dimensions.** Tensor $Y$ has the same shape as tensor $X$. When a dimension of $X$ has size $0$, the corresponding dimension of $Y$ has size $0$ and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

$$
X = \begin{bmatrix} 3 & -5 & 0 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} -3 & 5 & 0 \end{bmatrix}
$$

The negation of $0$ is $0$, the additive inverse of $0$ being $0$ itself.

## Error conditions

None of the conditions of the list applies to real numbers: the negation is exact, so that the result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_REAL_FUNC_0010</span></b>](#E_NEG_REAL_FUNC_0010) for every operand, and no error can occur. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{X}$: real tensor

The value whose sign is flipped.

#### Constraints

No constraint applies to $X$.

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

The three types share one semantics: the IEEE 754 standard defines the same sign change for all of them, and they differ only by their precision and by the range of the values they represent. The type `bfloat16`, which ONNX admits for **Neg**, is outside the SONNX profile and is not covered by this section.

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

Operator **Neg** flips the sign of every element of tensor $X$ according to IEEE 754 floating-point semantics and stores the result in tensor $Y$. Each element of $Y$ is the negation of the element of $X$ that has the same index.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = -_{(\text{f})} X[i] =
\begin{cases}
\text{NaN} & \text{if } X[i] \text{ is NaN} \\
-0 & \text{if } X[i] \text{ is } +0 \\
+0 & \text{if } X[i] \text{ is } -0 \\
-\infty & \text{if } X[i] \text{ is } +\infty \\
+\infty & \text{if } X[i] \text{ is } -\infty \\
-_{(\text{f})} X[i] & \text{otherwise}
\end{cases}
$$

where
- $-_{(\text{f})}$ is the sign change of the floating-point type of $X$, i.e. $-_{(\text{f16})}$, $-_{(\text{f32})}$ or $-_{(\text{f64})}$ according to that type,
- $-_{(\text{f})} X[i]$ in the last case is the sign change of the type of $X$: the value of that type whose magnitude is that of $X[i]$ and whose sign is the opposite of the sign of $X[i]$.

The sign change is exact: it reverses the sign of the value and leaves its magnitude unchanged, so that no rounding occurs, whether the value is normal or subnormal.

The NaN of the first case is a quiet NaN; its sign and its payload are not specified, so that every quiet NaN of the type of $X$ is a conforming result.

**Values of the operands.** The operands may be any of the values of the type, including the special numbers $\pm 0$, $\pm\text{inf}$ and NaN. The cases above give the result for every one of them. In particular, the negation of a null value is a null value of the opposite sign, and the negation of an infinity is an infinity of the opposite sign.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. When $X$ is such a tensor, $Y$ is a rank-0 tensor whose single element is the negation of the single element of $X$.

**Zero-sized dimensions.** Tensor $Y$ has the same shape as tensor $X$. When a dimension of $X$ has size $0$, the corresponding dimension of $Y$ has size $0$ and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

$$
X = \begin{bmatrix} 3.5 & \text{+0} & \text{-0} & \text{+inf} & \text{-inf} & \text{NaN} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} -3.5 & \text{-0} & \text{+0} & \text{-inf} & \text{+inf} & \text{NaN} \end{bmatrix}
$$

The negation of $+0$ is $-0$ and the negation of $-0$ is $+0$; the negation of $+\infty$ is $-\infty$ and the negation of $-\infty$ is $+\infty$. The negation of a NaN is a quiet NaN, whose sign and payload are not specified.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. one of the operations of IEEE 754 section 7.2 | not applicable: the operator performs none of those operations; the negation of a NaN is specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010), the result is a quiet NaN |
| Overflow, i.e. a result beyond the largest finite value of the type | not applicable: the negation of a finite value is finite and exact, so that no result exceeds the largest finite value of the type |
| Underflow, i.e. a result with a magnitude below the smallest normal value of the type | not applicable: the sign change is exact, so that no result is rounded |
| Division by zero | not applicable: the operator performs no division |

Every condition of the list is either inapplicable or part of the nominal behavior of the operator; none of them is an error. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{X}$: floating-point tensor

The value whose sign is flipped.

#### Constraints

<a id="E_NEG_FLOAT_CONSTR_X_0010"></a>
- `[E_NEG_FLOAT_CONSTR_X_0010]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: floating-point tensor

Tensor $Y$ is the element-wise result of the negation of $X$.

#### Constraints

- `[E_NEG_FLOAT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as tensor $X$.
- `[E_NEG_FLOAT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_CONSTR_X_0010</span></b>](#E_NEG_FLOAT_CONSTR_X_0010) on tensor $X$.

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

Operator **Neg** flips the sign of every element of tensor $X$ and stores the result in tensor $Y$. Each element of $Y$ is the negation of the element of $X$ that has the same index. Tensors $X$ and $Y$ have the same signed $n$-bit type, and the result is a value of that type.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] =
\begin{cases}
-_{(\text{i}n)}X[i] & \text{if } X[i] \neq -2^{n-1} \\
-2^{n-1} & \text{if } X[i] = -2^{n-1}
\end{cases}
$$

where
- $-_{(\text{i}n)}$ is the negation of the signed $n$-bit type of $X$, i.e. $-_{(\text{i8})}$, $-_{(\text{i16})}$, $-_{(\text{i32})}$ or $-_{(\text{i64})}$ according to that type: the exact negation of the element, the opposite in $\mathbb{Z}$ of the integer $X[i]$, reduced modulo $2^n$ to a value of that type; for the values of the first case, whose exact negation lies in the range of the type, that value is the exact negation $-X[i]$,
- $n$ is the number of bits of the type of $X$.

The result is thus the exact negation modulo $2^n$, read as a signed value. The values of an $n$-bit signed type lie in $[-2^{n-1}, 2^{n-1}-1]$, so that their exact negation lies in $[-(2^{n-1}-1), 2^{n-1}]$: it leaves the range of the type only for the minimum value $-2^{n-1}$, whose negation $2^{n-1}$ is reduced to $2^{n-1} - 2^n = -2^{n-1}$.

**The minimum value is its own negation.** The range of the type is not symmetric about zero, so that the negation of the minimum value of the type is that value itself: for `int8`, $-(-128)$ gives $-128$. Tensors $X$ and $Y$ have the same type and no wider type is used, so that the value of $Y$ is that value of the type, and applying the operator twice to the minimum value does not give the value back.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. When $X$ is such a tensor, $Y$ is a rank-0 tensor whose single element is the negation of the single element of $X$.

**Zero-sized dimensions.** Tensor $Y$ has the same shape as tensor $X$. When a dimension of $X$ has size $0$, the corresponding dimension of $Y$ has size $0$ and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `int8` type:

$$
X = \begin{bmatrix} -6 & 100 & -128 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} 6 & -100 & -128 \end{bmatrix}
$$

The negations of $-6$ and of $100$ lie in $[-2^7, 2^7-1]$ and are given by the first case of the definition. The negation of the minimum value $-128$ is $128$, which is greater than the maximum value $2^7-1=127$ of `int8`; the second case applies and the result is $-128$, the minimum value itself.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Integer overflow, i.e. the negation of the minimum value of the type, such as $-(-128)$ for `int8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_INT_FUNC_0010</span></b>](#E_NEG_INT_FUNC_0010), the result is the minimum value itself |
| Division by zero | not applicable: the operator performs no division |

The only condition of the list that applies to the signed integer types is the overflow, and it is part of the nominal behavior of the operator; there is no error. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{X}$: signed integer tensor

The value whose sign is flipped.

#### Constraints

<a id="E_NEG_INT_CONSTR_X_0010"></a>
- `[E_NEG_INT_CONSTR_X_0010]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: signed integer tensor

Tensor $Y$ is the element-wise result of the negation of $X$.

#### Constraints

- `[E_NEG_INT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as tensor $X$.
- `[E_NEG_INT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_INT_CONSTR_X_0010</span></b>](#E_NEG_INT_CONSTR_X_0010) on tensor $X$.
