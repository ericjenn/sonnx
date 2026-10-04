# Contents

- **Neg** operator for type [real](#real)
- **Neg** operator for types [`float16`, `float`, `double`](#float)
- **Neg** operator for types [`int8`, `int16`, `int32`, `int64`](#int)

Based on ONNX documentation [Neg version 14](https://onnx.ai/onnx/operators/onnx__Neg.html#neg-14).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

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

**Part 1.** Operator **Neg** flips the sign of every element of tensor $X$ and stores the result in tensor $Y$.

**Part 2.** Each element of $Y$ is the negation of the element of $X$ that has the same index.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = -X[i]
$$

where $-X[i]$ is the negation in $\mathbb{R}$ of the element $X[i]$.

The negation is exact: $Y[i]$ is the opposite of $X[i]$ in $\mathbb{R}$, with no approximation.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It has a single element, and $Y$ is the rank-0 tensor whose single element is the negation of that element.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example

$$
X = \begin{bmatrix} 1/3 & -2.5 & 0 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} -1/3 & 2.5 & 0 \end{bmatrix}
$$

The negation of $1/3$ is the exact value $-1/3$, and the negation of $-2.5$ is $2.5$; the negation of $0$ is $0$.

## Error conditions

None of the conditions of the list applies to real numbers: the negation is exact, so that the result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_REAL_FUNC_0010</span></b>](#E_NEG_REAL_FUNC_0010) for every operand, and no error can occur. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{X}$: real tensor

The tensor whose elements are negated.

#### Constraints

No constraint.

## Outputs

### $\text{Y}$: real tensor

Tensor $Y$ is the element-wise negation of $X$.

#### Constraints

<a id="E_NEG_REAL_CONSTR_Y_0010"></a>
- `[E_NEG_REAL_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as $X$.

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

**Part 1.** Operator **Neg** flips the sign of every element of tensor $X$ according to IEEE 754 floating-point semantics and stores the result in tensor $Y$.

**Part 2.** Each element of $Y$ is the negation of the element of $X$ that has the same index.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = -_{(\text{f})} X[i]
$$

where $-_{(\text{f})}$ is the negation of the floating-point type of $X$ and $Y$, i.e. $-_{(\text{f16})}$, $-_{(\text{f32})}$ or $-_{(\text{f64})}$ according to that type.

The negation of IEEE 754 is exact: it changes the sign of the operand and leaves its magnitude unchanged, so that no rounding occurs and the result is always representable in the type of $X$ and $Y$. In particular:

- the negation of $+0$ is $-0$, and the negation of $-0$ is $+0$;
- the negation of $+\infty$ is $-\infty$, and the negation of $-\infty$ is $+\infty$;
- the negation of a finite value is the finite value of the opposite sign and of the same magnitude, whether that value is normal or subnormal;
- the negation of a NaN is a NaN; the sign and the payload of that NaN are not specified, so that every quiet NaN of the type of $X$ and $Y$ is a conforming result.

**Values of the operands.** The operands may be any of the values of the type, including the special numbers $\pm 0$, $\pm\text{inf}$ and NaN. The cases above give the result for every one of them.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It has a single element, and $Y$ is the rank-0 tensor whose single element is the negation of that element.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix} 1.5 & -2.25 & 0.0 & -0.0 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} -1.5 & 2.25 & -0.0 & 0.0 \end{bmatrix}
$$

The negation of a non-zero value is exact. The negation of $+0$ is $-0$ and the negation of $-0$ is $+0$: the sign of a null element is flipped like the sign of any other element.

### Example 2

$$
X = \begin{bmatrix} \text{+inf} & \text{-inf} & \text{NaN} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} \text{-inf} & \text{+inf} & \text{NaN} \end{bmatrix}
$$

The negation of an infinity is the infinity of the opposite sign. The negation of a NaN is a NaN, whose sign and payload are not specified.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. an operation of IEEE 754 section 7.2 | nominal: the operator performs no such operation, and the negation of a NaN is a NaN, as specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010) |
| Overflow, i.e. a result beyond the largest finite value of the type | nominal: the negation of a finite value is finite and of the same magnitude, as specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010) |
| Underflow, i.e. a result of a magnitude below the smallest normal value of the type | nominal: the magnitude of the result is that of the operand, as specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010) |
| Division by zero | nominal: the operator performs no division, so the condition cannot occur |

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

Tensor $Y$ is the element-wise negation of $X$.

#### Constraints

<a id="E_NEG_FLOAT_CONSTR_Y_0010"></a>
- `[E_NEG_FLOAT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as $X$.
- `[E_NEG_FLOAT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_CONSTR_X_0010</span></b>](#E_NEG_FLOAT_CONSTR_X_0010) on tensor $X$.

<a id="int"></a>

# **Neg** (int)

where int is in {`int8`, `int16`, `int32`, `int64`}.

The four types share one semantics: each of them is an $n$-bit signed type, whose values range from $-2^{n-1}$ to $2^{n-1}-1$, and the result of the negation is the value of that type that represents the negation of the element modulo $2^n$.

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

**Part 1.** Operator **Neg** flips the sign of every element of tensor $X$ and stores the result in tensor $Y$.

**Part 2.** Each element of $Y$ is the negation of the element of $X$ that has the same index. Tensors $X$ and $Y$ have the same signed $n$-bit type, and the result is a value of that type.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] =
\begin{cases}
-X[i] & \text{if } -X[i] \text{ lies in } [-2^{n-1},\, 2^{n-1}-1] \\
-X[i] -_{(\text{i}n)} 2^n & \text{if } -X[i] \text{ is greater than } 2^{n-1}-1
\end{cases}
$$

where
- $-X[i]$ is the exact negation in $\mathbb{Z}$ of the element $X[i]$, before the reduction modulo $2^n$,
- $-_{(\text{i}n)}$ is the subtraction of the $n$-bit signed type: the exact difference of its two operands, read as a signed $n$-bit value, i.e. reduced modulo $2^n$ into $[-2^{n-1},\, 2^{n-1}-1]$,
- $n$ is the number of bits of the type of $X$ and $Y$.

The result is thus the exact negation modulo $2^n$, read as a signed value. The values of an $n$-bit signed type lie in $[-2^{n-1}, 2^{n-1}-1]$, so that their exact negation lies in $[-(2^{n-1}-1), 2^{n-1}]$: the second case applies only to the minimum value $-2^{n-1}$ of the type, whose negation $2^{n-1}$ is not a value of the type, and the reduction is applied at most once.

**The negation of the minimum value.** The range of the type is not symmetric about zero: the negation of the minimum value $-2^{n-1}$ is $2^{n-1}$, which is not representable in the type. The result is then $-2^{n-1}$ itself: for `int8`, $-(-128)$ gives $-128$. Tensors $X$ and $Y$ have the same type and no wider type is used, so that the value of $Y$ is that value of the type.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It has a single element, and $Y$ is the rank-0 tensor whose single element is the negation of that element.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example

For the `int8` type:

$$
X = \begin{bmatrix} 5 & -5 & 127 & -128 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} -5 & 5 & -127 & -128 \end{bmatrix}
$$

The negations of $5$, $-5$ and $127$ lie in $[-2^7, 2^7-1]$ and are given by the first case of the definition. The negation of $-128$ is $128$, which is greater than the maximum value $2^7-1=127$ of `int8`, so the second case applies and the result is $128-2^8=-128$: the negation of the minimum value of the type is that value itself.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Integer overflow, i.e. the negation of the minimum value $-2^{n-1}$ of the type, such as $-(-128)$ for `int8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_INT_FUNC_0010</span></b>](#E_NEG_INT_FUNC_0010), the result is $-2^{n-1}$ itself |
| Division by zero | nominal: the operator performs no division, so the condition cannot occur |

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

<a id="E_NEG_INT_CONSTR_Y_0010"></a>
- `[E_NEG_INT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as $X$.
- `[E_NEG_INT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_INT_CONSTR_X_0010</span></b>](#E_NEG_INT_CONSTR_X_0010) on tensor $X$.
