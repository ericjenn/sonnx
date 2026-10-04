# Contents

- **Neg** operator for type [real](#real)
- **Neg** operator for types [`float16`, `float`, `double`](#float)
- **Neg** operator for types [`int8`, `int16`, `int32`, `int64`](#int)

Based on ONNX documentation [Neg version 13](https://onnx.ai/onnx/operators/onnx__Neg.html#neg-13).

Revision 2026-10-03: based on ONNX opset 14; first issue of this specification.

<a id="real"></a>

# **Neg** (real)

## Signature

Definition of operator **Neg** signature:

 $C = \text{Neg}(A)$

 where
 - $A$: value to be negated
 - $C$: result of the element-wise negation of $A$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Neg** operator.

## Function

<a id="E_NEG_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_REAL_FUNC_0010]</br></span>

**Part 1.** Operator **Neg** negates tensor $A$ element-wise and stores the result in tensor $C$.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $C$:

$$
C[i] = -A[i]
$$

where
- $A[i]$ is the element of tensor $A$ at index $i$,
- $-x$ is the negation of the real number $x$.

The negation is exact: $C[i]$ is the opposite in $\mathbb{R}$ of $A[i]$, with no approximation. Tensor $C$ has the same shape as tensor $A$.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. Tensor $C$ is then a rank-0 tensor whose single element is the negation of the single element of $A$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Tensor $C$ has the same shape as $A$, so that the corresponding dimension of $C$ has size $0$ and $C$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example

$$
A = \begin{bmatrix} 3.5 & -2.25 \\ 0 & 7 \end{bmatrix}
\quad
C = \begin{bmatrix} -3.5 & 2.25 \\ 0 & -7 \end{bmatrix}
$$

The negation is exact for every element, and the negation of the real number $0$ is $0$.

## Error conditions

None of the conditions of the list applies to real numbers: the negation is exact, so that the result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_REAL_FUNC_0010</span></b>](#E_NEG_REAL_FUNC_0010) for every operand, and no error can occur. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{A}$: real tensor

The value to be negated.

#### Constraints

No constraint applies to input $A$.

## Outputs

### $\text{C}$: real tensor

Tensor $C$ is the element-wise negation of $A$.

#### Constraints

<a id="E_NEG_REAL_CONSTR_C_0010"></a>
- `[E_NEG_REAL_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the same shape as $A$.

<a id="float"></a>

# **Neg** (float, float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same negation for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

Definition of operator **Neg** signature:

 $C = \text{Neg}(A)$

 where
 - $A$: value to be negated
 - $C$: result of the element-wise negation of $A$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Neg** operator.

## Function

<a id="E_NEG_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_FLOAT_FUNC_0010]</br></span>

**Part 1.** Operator **Neg** negates tensor $A$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $C$.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $C$:

$$
C[i] = -_{(\text{f})} A[i]
$$

where $-_{(\text{f})}$ is the negation of the floating-point type of $A$, i.e. $-_{(\text{f16})}$, $-_{(\text{f32})}$ or $-_{(\text{f64})}$ according to that type.

The negation of a floating-point number is exact: it gives the value of the type that has the opposite sign and the same magnitude as the operand. No rounding takes place, so that the result is never an infinity produced by an overflow and never a subnormal value produced by an underflow. Tensor $C$ has the same shape as tensor $A$.

**Values of the operand.** The operand may be any of the values of the type, including the special numbers $\pm 0$, $\pm\text{inf}$ and NaN. The negation of a finite value is the finite value of the opposite sign and the same magnitude; the negation of $+0$ is $-0$ and the negation of $-0$ is $+0$; the negation of $+\text{inf}$ is $-\text{inf}$ and the negation of $-\text{inf}$ is $+\text{inf}$; the negation of a NaN is a NaN. The sign and the payload of that NaN are not specified, so that every quiet NaN of the type of $A$ is a conforming result.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. Tensor $C$ is then a rank-0 tensor whose single element is the negation of the single element of $A$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Tensor $C$ has the same shape as $A$, so that the corresponding dimension of $C$ has size $0$ and $C$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
A = \begin{bmatrix} 3.0 & -4.5 \\ 16.0 & 1.0 \end{bmatrix}
\quad
C = \begin{bmatrix} -3.0 & 4.5 \\ -16.0 & -1.0 \end{bmatrix}
$$

Every element of $C$ is the element of $A$ with the opposite sign and the same magnitude; the negation is exact.

### Example 2

$$
A = \begin{bmatrix} \text{+0} & \text{-0} & \text{+inf} & \text{-inf} & \text{NaN} \end{bmatrix}
$$

$$
C = \begin{bmatrix} \text{-0} & \text{+0} & \text{-inf} & \text{+inf} & \text{NaN} \end{bmatrix}
$$

The negation of $+0$ is $-0$ and the negation of $-0$ is $+0$; the negation of an infinity is the infinity of the opposite sign; the negation of a NaN is a NaN, whose sign and payload are not specified.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | nominal: the negation is not an invalid operation, and the result is specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010) for every value of the type |
| Overflow | nominal: the negation is exact, and the result is specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010) for every value of the type |
| Underflow | nominal: the negation is exact, and the result is specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_FUNC_0010</span></b>](#E_NEG_FLOAT_FUNC_0010) for every value of the type |

Every condition above is part of the nominal behavior of the operator; none of them is an error. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{A}$: floating-point tensor

The value to be negated.

#### Constraints

<a id="E_NEG_FLOAT_CONSTR_A_0010"></a>
- `[E_NEG_FLOAT_CONSTR_A_0010]` Type consistency
  - Statement: Tensors $A$ and $C$ have the same type.

## Outputs

### $\text{C}$: floating-point tensor

Tensor $C$ is the element-wise negation of $A$.

#### Constraints

<a id="E_NEG_FLOAT_CONSTR_C_0010"></a>
- `[E_NEG_FLOAT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the same shape as $A$.
- `[E_NEG_FLOAT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_FLOAT_CONSTR_A_0010</span></b>](#E_NEG_FLOAT_CONSTR_A_0010) on tensor $A$.

<a id="int"></a>

# **Neg** (int, int)

where int is in {`int8`, `int16`, `int32`, `int64`}.

The four types share one semantics: each of them is an $n$-bit signed type, whose values range from $-2^{n-1}$ to $2^{n-1}-1$, and the result of the negation is the value of that type that represents the opposite of the element modulo $2^n$.

## Signature

Definition of operator **Neg** signature:

 $C = \text{Neg}(A)$

 where
 - $A$: value to be negated
 - $C$: result of the element-wise negation of $A$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Neg** operator.

## Function

<a id="E_NEG_INT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_INT_FUNC_0010]</br></span>

**Part 1.** Operator **Neg** negates tensor $A$ element-wise and stores the result in tensor $C$. Tensors $A$ and $C$ have the same signed $n$-bit type, and the result is a value of that type.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $C$:

$$
C[i] =
\begin{cases}
-A[i] & \text{if } A[i] \neq -2^{n-1} \\
-2^{n-1} & \text{if } A[i] = -2^{n-1}
\end{cases}
$$

where
- $A[i]$ is the element of tensor $A$ at index $i$,
- $n$ is the number of bits of the type of $A$.

The result is the opposite of the element modulo $2^n$, read as a signed value. The values of an $n$-bit signed type lie in $[-2^{n-1}, 2^{n-1}-1]$, so that the opposite of an element lies in $[-(2^{n-1}-1), 2^{n-1}]$: it leaves that range only for the element $-2^{n-1}$, whose opposite $2^{n-1}$ is not representable in the type, and the result is then the value of the type that represents $2^{n-1}$ modulo $2^n$, i.e. $-2^{n-1}$ itself. Tensor $C$ has the same shape as tensor $A$.

**The most negative value is its own negation.** For `int8`, the negation of $-128$ is $-128$: the result is negative although the operand is negative, and the operator is not an involution on that value. Tensors $A$ and $C$ have the same type and no wider type is used, so that the value of $C$ is that value of the type, sign included.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. Tensor $C$ is then a rank-0 tensor whose single element is the negation of the single element of $A$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Tensor $C$ has the same shape as $A$, so that the corresponding dimension of $C$ has size $0$ and $C$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example

For the `int8` type:

$$
A = \begin{bmatrix} 6 & -100 & -128 \end{bmatrix}
\quad
C = \begin{bmatrix} -6 & 100 & -128 \end{bmatrix}
$$

The negations of $6$ and $-100$ lie in $[-2^7, 2^7-1]$ and are given by the first case of the definition. The negation of $-128$, the minimum value of `int8`, is $128$, which is greater than the maximum value $2^7-1=127$ of the type: the second case applies and the result is $-128$, the value that represents $128$ modulo $2^8$. That result is negative although the operand is negative.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Overflow, i.e. the negation of the minimum value of the type, such as $-(-128)=128$ for `int8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_NEG_INT_FUNC_0010</span></b>](#E_NEG_INT_FUNC_0010), the result is the value of the type that represents the opposite modulo $2^n$, i.e. the minimum value itself |

The only condition of the list that applies to the signed integer types is the overflow, and it is part of the nominal behavior of the operator; there is no error. No error condition.

## Attributes

Operator **Neg** has no attribute.

## Inputs

### $\text{A}$: signed integer tensor

The value to be negated.

#### Constraints

<a id="E_NEG_INT_CONSTR_A_0010"></a>
- `[E_NEG_INT_CONSTR_A_0010]` Type consistency
  - Statement: Tensors $A$ and $C$ have the same type.

## Outputs

### $\text{C}$: signed integer tensor

Tensor $C$ is the element-wise negation of $A$.

#### Constraints

<a id="E_NEG_INT_CONSTR_C_0010"></a>
- `[E_NEG_INT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the same shape as $A$.
- `[E_NEG_INT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_NEG_INT_CONSTR_A_0010</span></b>](#E_NEG_INT_CONSTR_A_0010) on tensor $A$.
