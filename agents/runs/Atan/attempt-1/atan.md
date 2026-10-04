# Contents

- **Atan** operator for type [real](#real)
- **Atan** operator for types [`float16`, `float`, `double`](#float)

Based on ONNX documentation [Atan version 7](https://onnx.ai/onnx/operators/onnx__Atan.html#atan-7).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **Atan** (real)

## Signature

Definition of operator **Atan** signature:

$output = \textbf{Atan}(input)$

where
- $input$: the input tensor, whose arctangent is computed
- $output$: the output tensor, the element-wise arctangent of $input$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Atan** operator.

## Function

<a id="E_ATAN_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ATAN_REAL_FUNC_0010]</br></span>

Operator **Atan** computes the arctangent of tensor $input$ element-wise and stores the result in tensor $output$. The arctangent is the inverse of the tangent: the result is the angle of the interval $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$ whose tangent is the operand.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $output$, $output[i]$ is the unique value $y$ such that

$$
\tan(y) = input[i] \quad \text{and} \quad y \in \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)
$$

where
- $input[i]$ is the element of $input$ at index $i$,
- $\tan$ is the tangent of the real numbers.

The equation $\tan(y) = input[i]$ alone has infinitely many solutions, one in each interval $\left(-\frac{\pi}{2} + k\pi, \frac{\pi}{2} + k\pi\right)$ for $k \in \mathbb{Z}$; the interval $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$ contains exactly one of them, so that the result is determined by $input[i]$ alone. The interval is open: $-\frac{\pi}{2}$ and $\frac{\pi}{2}$ are not results of the operator, the tangent being undefined there.

**Domain and range.** Every real number has an arctangent, so the domain of the operator is $\mathbb{R}$. The range is the open interval $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$: the result is never equal to $\pm\frac{\pi}{2}$, and it approaches them as the operand grows without bound.

**Sign of the result.** The tangent is an odd function, so the arctangent is odd as well: the arctangent of $-x$ is the opposite of the arctangent of $x$. The result therefore has the sign of the operand, and it is null only when the operand is null.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. The result is a rank-0 tensor whose single element is the arctangent of the single element of $input$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The result then has the same shape, with the same zero-sized dimension, and is empty.

The result is the real number defined above; the accuracy with which an implementation computes it is the subject of the [accuracy guidelines](./../other/accuracy.md).

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
input = \begin{bmatrix} 0 & 1 & -1 \end{bmatrix}
$$

$$
output = \begin{bmatrix} 0 & \frac{\pi}{4} & -\frac{\pi}{4} \end{bmatrix}
$$

The tangent of $\frac{\pi}{4}$ is $1$ and the tangent of $-\frac{\pi}{4}$ is $-1$; both values lie in $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$, so they are the results of the operator for the operands $1$ and $-1$. The arctangent of $0$ is $0$, the only value of the interval whose tangent is null.

### Example 2

$$
input = \begin{bmatrix} \sqrt{3} & -\sqrt{3} & \frac{\sqrt{3}}{3} \end{bmatrix}
$$

$$
output = \begin{bmatrix} \frac{\pi}{3} & -\frac{\pi}{3} & \frac{\pi}{6} \end{bmatrix}
$$

The tangent of $\frac{\pi}{3}$ is $\sqrt{3}$ and the tangent of $\frac{\pi}{6}$ is $\frac{\sqrt{3}}{3}$; the result of the negative operand is the opposite of the result of the positive one.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the arctangent is defined for every real number, so no operand makes the operator undefined |
| Overflow | not applicable: the result lies in $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$, which is within the range of the real numbers |
| Integer overflow | not applicable: the operator applies to real numbers, not to an integer type |
| Division by zero | not applicable: the operator performs no division |

No error condition.

## Attributes

Operator **Atan** has no attribute.

## Inputs

### $\text{input}$: real tensor

The tensor whose arctangent is computed.

#### Constraints

No constraint applies to the input tensor: the arctangent is defined for every real number.

## Outputs

### $\text{output}$: real tensor

The element-wise arctangent of the input.

#### Constraints

<a id="E_ATAN_REAL_CONSTR_OUTPUT_0010"></a>
- `[E_ATAN_REAL_CONSTR_OUTPUT_0010]` Shape definition
  - Statement: Tensor $output$ has the same shape as tensor $input$.

<a id="float"></a>

# **Atan** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same arctangent for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

Definition of operator **Atan** signature:

$output = \textbf{Atan}(input)$

where
- $input$: the input tensor, whose arctangent is computed
- $output$: the output tensor, the element-wise arctangent of $input$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Atan** operator.

## Function

<a id="E_ATAN_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ATAN_FLOAT_FUNC_0010]</br></span>

Operator **Atan** computes the arctangent of tensor $input$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $output$. Each element of $output$ is the value of the type of $input$ nearest to the arctangent of the corresponding element of $input$.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $output$:

$$
output[i] =
\begin{cases}
\text{NaN} & \text{if } input[i] \text{ is NaN} \\
+0 & \text{if } input[i] \text{ is } +0 \\
-0 & \text{if } input[i] \text{ is } -0 \\
\text{round}\left(\frac{\pi}{2}\right) & \text{if } input[i] \text{ is } +\infty \\
\text{round}\left(-\frac{\pi}{2}\right) & \text{if } input[i] \text{ is } -\infty \\
\text{round}(\arctan input[i]) & \text{otherwise}
\end{cases}
$$

where
- $\arctan x$ is the arctangent of the real number $x$, i.e. the unique $y \in \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$ such that $\tan(y) = x$,
- $\text{round}(x)$ is the value of $x$ rounded to the nearest value of the type of $input$ using the roundTiesToEven attribute of IEEE 754: a magnitude below the smallest normal value of the type of $input$ gives the nearest subnormal value of that type, or a zero of the sign of $x$ when the magnitude is at most half the smallest subnormal value of the type,
- $input[i]$ is the element of $input$ at index $i$.

**Values of the operands.** The operand may be any of the values of the type, including the special numbers $\pm 0$, $\pm\infty$ and NaN; the cases above give the result for every one of them. The arctangent of an infinity is the limit of the arctangent, $\pm\frac{\pi}{2}$, rounded to the type of $input$; the arctangent of a null operand is that same null, sign included; and a NaN operand gives a NaN result. The NaN of the result is a quiet NaN; its sign and its payload are not specified, so that every quiet NaN of the type of $input$ is a conforming result.

**Sign of the result.** The arctangent is an odd function, so the result has the sign of the operand: the result of a positive operand is positive, the result of a negative operand is negative, and the result of a null operand is that same null, sign included.

**No overflow.** The magnitude of the result is at most the value of the type of $input$ nearest to $\frac{\pi}{2}$, which is below the largest finite value of each of the three types; the result is therefore never an infinity, and the arctangent of a finite operand is never rounded to an infinity.

**Underflow.** The arctangent of a finite operand may be below the smallest normal value of the type of $input$; the result is then the value of the type of $input$ nearest to the arctangent, which is a subnormal value of that type, or a zero of the sign of the operand when the arctangent is at most half the smallest subnormal value of the type.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. The result is a rank-0 tensor whose single element is the arctangent of the single element of $input$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The result then has the same shape, with the same zero-sized dimension, and is empty.

The result is the value of the type defined above; the accuracy with which an implementation computes it is the subject of the [accuracy guidelines](./../other/accuracy.md).

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

For the `double` type:

$$
input = \begin{bmatrix} 0 & 1 & -1 \end{bmatrix}
$$

$$
output \approx \begin{bmatrix} 0 & 0.7853981633974483 & -0.7853981633974483 \end{bmatrix}
$$

The arctangent of $1$ is $\frac{\pi}{4}$, which is not representable in `double`; the result is the nearest `double` value.

### Example 2

For the `double` type:

$$
input = \begin{bmatrix} +0 & -0 & +\infty & -\infty & \text{NaN} \end{bmatrix}
$$

$$
output \approx \begin{bmatrix} +0 & -0 & 1.5707963267948966 & -1.5707963267948966 & \text{NaN} \end{bmatrix}
$$

The arctangent of $+\infty$ is $\frac{\pi}{2}$, which is not representable in `double`; the result is the nearest `double` value, and the result for $-\infty$ is its opposite. The result for $-0$ is $-0$, the arctangent being an odd function. The result for NaN is a quiet NaN whose sign and payload are not specified.

### Example 3

For the `float16` type:

$$
input = \begin{bmatrix} 1 & +\infty \end{bmatrix}
$$

$$
output \approx \begin{bmatrix} 0.78515625 & 1.5703125 \end{bmatrix}
$$

The arctangent of $1$ is $\frac{\pi}{4} \approx 0.7853981633974483$ and the arctangent of $+\infty$ is $\frac{\pi}{2} \approx 1.5707963267948966$; neither is representable in `float16`, and the results are the nearest `float16` values, $0.78515625$ and $1.5703125$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. an operation of IEEE 754 section 7.2 | not applicable: the arctangent is defined for every value of the type, including $\pm\infty$ and NaN, and no operand value makes it undefined |
| Overflow, i.e. a result whose magnitude exceeds the largest finite value of the type | not applicable: the magnitude of the result is at most the value of the type nearest to $\frac{\pi}{2}$, which is below the largest finite value of each of the three types |
| Underflow, i.e. a result whose magnitude is below the smallest normal value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ATAN_FLOAT_FUNC_0010</span></b>](#E_ATAN_FLOAT_FUNC_0010) |
| Integer overflow | not applicable: the operator applies to the floating-point types, not to an integer type |
| Division by zero | not applicable: the operator performs no division |

No error condition.

## Attributes

Operator **Atan** has no attribute.

## Inputs

### $\text{input}$: floating-point tensor

The tensor whose arctangent is computed.

#### Constraints

<a id="E_ATAN_FLOAT_CONSTR_INPUT_0010"></a>
- `[E_ATAN_FLOAT_CONSTR_INPUT_0010]` Type consistency
  - Statement: Tensors $input$ and $output$ have the same type.

## Outputs

### $\text{output}$: floating-point tensor

The element-wise arctangent of the input.

#### Constraints

<a id="E_ATAN_FLOAT_CONSTR_OUTPUT_0010"></a>
- `[E_ATAN_FLOAT_CONSTR_OUTPUT_0010]` Shape definition
  - Statement: Tensor $output$ has the same shape as tensor $input$.
- `[E_ATAN_FLOAT_CONSTR_OUTPUT_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ATAN_FLOAT_CONSTR_INPUT_0010</span></b>](#E_ATAN_FLOAT_CONSTR_INPUT_0010) on tensor $input$.
