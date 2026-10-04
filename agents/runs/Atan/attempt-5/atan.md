# Contents

- **Atan** operator for type [real](#real)
- **Atan** operator for types [`float16`, `float`, `double`](#float)

Based on ONNX documentation [Atan version 7](https://onnx.ai/onnx/operators/onnx__Atan.html#atan-7).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **Atan** (real)

## Signature

$output = \textbf{Atan}(input)$

where
- $input$: real tensor whose arctangent is computed
- $output$: element-wise arctangent of $input$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Atan** operator.

## Function

<a id="E_ATAN_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ATAN_REAL_FUNC_0010]</br></span>

**Part 1.** Operator **Atan** computes the arctangent of every element of the input tensor, element-wise, and stores the results in the output tensor. The arctangent is the inverse of the tangent: the result is the angle, in radians, whose tangent is the element.

**Part 2.** For every element $input[i]$ of the input tensor, the result $output[i]$ is the unique value $y$ such that

$$
\tan(y) = input[i] \quad \text{and} \quad y \in \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)
$$

where
- $i$ is any [tensor index](./../common/definitions.md#tensor_index) of $input$ and $output$,
- $\tan$ is the tangent of the real numbers, its argument being an angle in radians,
- $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$ is the branch of the tangent on which it is strictly increasing and takes every real value exactly once.

The equation $\tan(y) = x$ alone has one solution in each interval $\left(-\frac{\pi}{2} + k\pi, \frac{\pi}{2} + k\pi\right)$, for every integer $k$; the branch selects one of them, so that the result is defined and unique for every real $x$. The branch is open: the tangent is not defined at $\pm\frac{\pi}{2}$, and the result is never equal to $\pm\frac{\pi}{2}$.

**Domain and range.** The domain of the operator is $\mathbb{R}$: the tangent takes every real value on the branch, so that the equation has a solution for every element of the input tensor. The range is the open interval $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$.

**Sign.** The tangent is an odd function, so the arctangent is odd: $output[i] = -output[j]$ whenever $input[i] = -input[j]$. In particular, the arctangent of $0$ is $0$.

**Shapes.** The operator is unary and element-wise: the output tensor has the same shape as the input tensor, and each element of the output depends only on the element of the input at the same index; no broadcasting applies.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It has a single element, and the result is the arctangent of that element, a tensor with no dimension.

**Zero-sized dimensions.** An input tensor may have a zero-sized dimension. It then has no element, and the output tensor has the same shape and is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
input = \begin{bmatrix} 0 & 1 & -1 \end{bmatrix}
$$

$$
output = \begin{bmatrix} 0 & \frac{\pi}{4} & -\frac{\pi}{4} \end{bmatrix}
$$

The tangent of $0$ is $0$, the tangent of $\frac{\pi}{4}$ is $1$ and the tangent of $-\frac{\pi}{4}$ is $-1$; each of these three values lies in the branch $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$, so that they are the results of the three elements.

### Example 2

$$
input = \begin{bmatrix} \sqrt{3} & \frac{1}{\sqrt{3}} \end{bmatrix}
$$

$$
output = \begin{bmatrix} \frac{\pi}{3} & \frac{\pi}{6} \end{bmatrix}
$$

The tangent of $\frac{\pi}{3}$ is $\sqrt{3}$ and the tangent of $\frac{\pi}{6}$ is $\frac{1}{\sqrt{3}}$, and both values lie in the branch.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. the cases of IEEE 754 section 7.2 | none applies: the arctangent is defined for every real number |
| Overflow, i.e. a result outside the range of the type | none: the result lies in the bounded interval $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$ |
| Integer overflow | none: the operator applies to real numbers, not to integers |
| Division by zero | none: the operator performs no division |

Every condition of the list is inapplicable to real numbers, and the result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_ATAN_REAL_FUNC_0010</span></b>](#E_ATAN_REAL_FUNC_0010) for every element of the input tensor. No error condition.

## Attributes

Operator **Atan** has no attribute.

## Inputs

### $\text{input}$: real tensor

The tensor whose arctangent is computed.

#### Constraints

<a id="E_ATAN_REAL_CONSTR_INPUT_0010"></a>
- `[E_ATAN_REAL_CONSTR_INPUT_0010]` Shape definition
  - Statement: The output tensor has the same shape as the input tensor.

## Outputs

### $\text{output}$: real tensor

The element-wise arctangent of the input tensor.

#### Constraints

- `[E_ATAN_REAL_CONSTR_OUTPUT_0010]` Shape definition
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ATAN_REAL_CONSTR_INPUT_0010</span></b>](#E_ATAN_REAL_CONSTR_INPUT_0010) on tensor $input$.

<a id="float"></a>

# **Atan** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same arctangent for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

$output = \textbf{Atan}(input)$

where
- $input$: floating-point tensor whose arctangent is computed
- $output$: element-wise arctangent of $input$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Atan** operator.

## Function

<a id="E_ATAN_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ATAN_FLOAT_FUNC_0010]</br></span>

**Part 1.** Operator **Atan** computes the arctangent of every element of the input tensor, element-wise, and stores the results in the output tensor. The arctangent is the inverse of the tangent: the result is the angle, in radians, whose tangent is the element.

**Part 2.** For every element $input[i]$ of the input tensor that is finite, the result $output[i]$ is the value of the type of $input$ and $output$ that is the arctangent of $input[i]$:

$$
output[i] = \text{round}\left(\arctan(input[i])\right)
$$

where
- $i$ is any [tensor index](./../common/definitions.md#tensor_index) of $input$ and $output$,
- $\arctan(x)$ is the unique real $y$ such that $\tan(y) = x$ and $y \in \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$, the arctangent of the real numbers,
- $\text{round}(y)$ is the value of $y$ rounded to the nearest value of the type of $input$ and $output$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded: a magnitude whose nearest value exceeds the largest finite value of the type of $input$ and $output$ gives an infinity of the sign of $y$, and a null $y$ gives a null value of the same sign.

The equation $\tan(y) = x$ alone has one solution in each interval $\left(-\frac{\pi}{2} + k\pi, \frac{\pi}{2} + k\pi\right)$, for every integer $k$; the branch selects one of them, so that $\arctan(x)$ is defined and unique for every real $x$. The branch is open: the tangent is not defined at $\pm\frac{\pi}{2}$, and the result is never equal to $\pm\frac{\pi}{2}$.

The formula above applies to the elements that are finite: its domain is the real numbers, to which neither the infinities nor NaN belong, and $\arctan$ is not defined at them. The special values are decided by the cases stated below.

**Special values.** The arctangent is defined for every value of the type, including the special numbers, and the formula above or the case stated here gives the result for each of them:

- an element that is $\pm 0$ gives $\pm 0$: this is the value of the formula above, whose $\text{round}$ gives a null result the sign of the element, so that the case restates the definition of $\text{round}$ and adds nothing to it;
- an element that is $+\infty$ gives the value of the type of $input$ and $output$ nearest to $\frac{\pi}{2}$, and an element that is $-\infty$ gives the value of that type nearest to $-\frac{\pi}{2}$: the arctangent of $+\infty$ is $\frac{\pi}{2}$ and the arctangent of $-\infty$ is $-\frac{\pi}{2}$, and the result is that bound rounded to the type by the $\text{round}$ defined above, a finite value;
- an element that is NaN gives a quiet NaN of the type of $input$ and $output$: the result of the operator for such an element is a quiet NaN, and neither its sign nor its payload is specified, so that every quiet NaN of the type of $input$ and $output$ is a conforming result.

**Rounding.** The arctangent of an element is in general not representable in the type of $input$ and $output$; the result is then the nearest value of the type, as stated above. The magnitude of the result is at most the nearest value of the type to $\frac{\pi}{2}$, which is below the largest finite value of every type of this section, so that no result is an infinity and no overflow occurs. A result whose magnitude is below the smallest normal value of the type is a subnormal value or a zero; this is the nominal behavior of the operator.

**Shapes.** The operator is unary and element-wise: the output tensor has the same shape as the input tensor, and each element of the output depends only on the element of the input at the same index; no broadcasting applies.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It has a single element, and the result is the arctangent of that element, a tensor with no dimension.

**Zero-sized dimensions.** An input tensor may have a zero-sized dimension. It then has no element, and the output tensor has the same shape and is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

For the `double` type:

$$
input = \begin{bmatrix} 0.0 & 1.0 & -1.0 \end{bmatrix}
$$

$$
output \approx \begin{bmatrix} 0.0 & 0.7853981633974483 & -0.7853981633974483 \end{bmatrix}
$$

The arctangent of $1.0$ is $\frac{\pi}{4}$, which is not representable in `double`; the result is the nearest `double`, and the same holds for $-\frac{\pi}{4}$. The arctangent of $0.0$ is $0.0$, exactly.

### Example 2

For the `double` type:

$$
input = \begin{bmatrix} \text{+0.0} & \text{-0.0} & \text{+inf} & \text{-inf} & \text{NaN} \end{bmatrix}
$$

$$
output \approx \begin{bmatrix} \text{+0.0} & \text{-0.0} & 1.5707963267948966 & -1.5707963267948966 & \text{NaN} \end{bmatrix}
$$

The arctangent of $+0.0$ is $+0.0$ and the arctangent of $-0.0$ is $-0.0$: the sign of a null element is preserved. The arctangent of $+\infty$ is $\frac{\pi}{2}$, whose nearest `double` is $1.5707963267948966$, and the arctangent of $-\infty$ is its opposite. The arctangent of NaN is a quiet NaN, whose sign and payload are not specified.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. the cases of IEEE 754 section 7.2 | none applies: the arctangent is defined for every value of the type, including $\pm\infty$ and NaN, and the result is specified by [<b><span style="font-family: 'Courier New', monospace">E_ATAN_FLOAT_FUNC_0010</span></b>](#E_ATAN_FLOAT_FUNC_0010) |
| Overflow, i.e. a result whose magnitude exceeds the largest finite value of the type | none: the magnitude of the result is at most the nearest value of the type to $\frac{\pi}{2}$, which is below the largest finite value of every type of this section |
| Integer overflow | none: the operator applies to floating-point types, not to integers |
| Division by zero | none: the operator performs no division |
| Underflow, i.e. a result whose magnitude is below the smallest normal value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ATAN_FLOAT_FUNC_0010</span></b>](#E_ATAN_FLOAT_FUNC_0010), the result is the rounded arctangent, which may be a subnormal value or a zero |

Every condition of the list is either inapplicable or part of the nominal behavior of the operator; none of them is an error. No error condition.

## Attributes

Operator **Atan** has no attribute.

## Inputs

### $\text{input}$: floating-point tensor

The tensor whose arctangent is computed.

#### Constraints

<a id="E_ATAN_FLOAT_CONSTR_INPUT_0010"></a>
- `[E_ATAN_FLOAT_CONSTR_INPUT_0010]` Shape definition
  - Statement: The output tensor has the same shape as the input tensor.
<a id="E_ATAN_FLOAT_CONSTR_INPUT_0020"></a>
- `[E_ATAN_FLOAT_CONSTR_INPUT_0020]` Type consistency
  - Statement: Tensors $input$ and $output$ have the same type.

## Outputs

### $\text{output}$: floating-point tensor

The element-wise arctangent of the input tensor.

#### Constraints

- `[E_ATAN_FLOAT_CONSTR_OUTPUT_0010]` Shape definition
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ATAN_FLOAT_CONSTR_INPUT_0010</span></b>](#E_ATAN_FLOAT_CONSTR_INPUT_0010) on tensor $input$.
- `[E_ATAN_FLOAT_CONSTR_OUTPUT_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ATAN_FLOAT_CONSTR_INPUT_0020</span></b>](#E_ATAN_FLOAT_CONSTR_INPUT_0020) on tensor $input$.
