# Contents

- **Acos** operator for type [real](#real)
- **Acos** operator for types [`float16`, `float`, `double`](#float)

Based on ONNX documentation [Acos version 7](https://onnx.ai/onnx/operators/onnx__Acos.html#acos-7).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **Acos** (real)

## Signature

$Y = \textbf{Acos}(X)$

where
- $X$: value whose arccosine is computed
- $Y$: result of the element-wise arccosine of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Acos** operator.

## Function

<a id="E_ACOS_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ACOS_REAL_FUNC_0010]</br></span>

**Part 1.** Operator **Acos** computes the arccosine of tensor $X$ element-wise and stores the result in tensor $Y$. The arccosine is the inverse of the cosine: each element of $Y$ is the angle of $[0, \pi]$ whose cosine is the corresponding element of $X$.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = \text{acos}(X[i])
$$

where $\text{acos}$ is defined by the following property: for every $x$ in $[-1, 1]$, $\text{acos}(x)$ is the unique value $y$ such that

$$
\cos(y) = x \quad \text{and} \quad y \in [0, \pi]
$$

The equation $\cos(y) = x$ alone has infinitely many solutions, one in each interval of length $2\pi$; the interval $[0, \pi]$ is the branch on which it has exactly one, and it is therefore as specifying as the equation. The result is the value of that branch, not a way of computing it.

**Domain and range.** The arccosine is defined on $[-1, 1]$: for $x$ in that interval, $\cos$ takes the value $x$ at exactly one point of $[0, \pi]$, whereas for $x$ outside it, no real number has $x$ as its cosine. The constraint [<b><span style="font-family: 'Courier New', monospace">E_ACOS_REAL_CONSTR_X_0010</span></b>](#E_ACOS_REAL_CONSTR_X_0010) requires every element of $X$ to lie in $[-1, 1]$, so that the result is defined for every element of $X$. Every element of $Y$ lies in $[0, \pi]$.

**Values at the bounds and monotonicity.** $\text{acos}(1) = 0$, $\text{acos}(0) = \frac{\pi}{2}$ and $\text{acos}(-1) = \pi$. The arccosine is strictly decreasing on $[-1, 1]$: if $x_1 \lt x_2$, then $\text{acos}(x_1) \gt \text{acos}(x_2)$.

**Exactness.** For real numbers the result is exact: $Y[i]$ is the real number $\text{acos}(X[i])$, with no approximation. For instance $\text{acos}\left(\frac{1}{2}\right) = \frac{\pi}{3}$ and $\text{acos}\left(\frac{\sqrt{2}}{2}\right) = \frac{\pi}{4}$.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. The result is a rank-0 tensor whose single element is the arccosine of the single element of $X$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The result has the same shape as $X$ and is then empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix} -1 & 0 & 1 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} \pi & \frac{\pi}{2} & 0 \end{bmatrix}
$$

The arccosine of $-1$ is $\pi$, that of $0$ is $\frac{\pi}{2}$ and that of $1$ is $0$; the result is exact.

### Example 2

$$
X = \begin{bmatrix} \frac{1}{2} & \frac{\sqrt{2}}{2} & -\frac{1}{2} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} \frac{\pi}{3} & \frac{\pi}{4} & \frac{2\pi}{3} \end{bmatrix}
$$

The arccosine of $\frac{1}{2}$ is $\frac{\pi}{3}$, that of $\frac{\sqrt{2}}{2}$ is $\frac{\pi}{4}$ and that of $-\frac{1}{2}$ is $\frac{2\pi}{3}$; the result is exact.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. an operand outside $[-1, 1]$, whose arccosine is not a real number | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_ACOS_REAL_CONSTR_X_0010</span></b>](#E_ACOS_REAL_CONSTR_X_0010) |
| Overflow, i.e. a result beyond the range of the type | not applicable: the result lies in $[0, \pi]$ |
| Division by zero | not applicable: the operator performs no division |

No error condition.

## Attributes

Operator **Acos** has no attribute.

## Inputs

### $\text{X}$: real tensor

The value whose arccosine is computed.

#### Constraints

<a id="E_ACOS_REAL_CONSTR_X_0010"></a>
- `[E_ACOS_REAL_CONSTR_X_0010]` Definition domain
  - Statement: Every element of $X$ lies in $[-1, 1]$.
  - Rationale: The arccosine is defined on $[-1, 1]$ only; outside that interval no real number has $x$ as its cosine.

## Outputs

### $\text{Y}$: real tensor

Tensor $Y$ is the element-wise arccosine of $X$.

#### Constraints

<a id="E_ACOS_REAL_CONSTR_Y_0010"></a>
- `[E_ACOS_REAL_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as $X$.

<a id="float"></a>

# **Acos** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same arccosine for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

$Y = \textbf{Acos}(X)$

where
- $X$: value whose arccosine is computed
- $Y$: result of the element-wise arccosine of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Acos** operator.

## Function

<a id="E_ACOS_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ACOS_FLOAT_FUNC_0010]</br></span>

**Part 1.** Operator **Acos** computes the arccosine of tensor $X$ element-wise and stores the result in tensor $Y$. Each element of $Y$ is the value of the arccosine of the corresponding element of $X$, rounded to the type of $X$ and $Y$.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = \text{round}(\text{acos}(X[i]))
$$

where
- $\text{acos}$ is the arccosine of the real numbers, defined in the section for type [real](#real): for every $x$ in $[-1, 1]$, the unique value $y$ such that $\cos(y) = x$ and $y \in [0, \pi]$,
- $\text{round}(y)$ is the value of $y$ rounded to the nearest value of the type of $X$ and $Y$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded.

**Values of the operands.** The operands may be any of the values of the type. An operand that is NaN gives a NaN result; that NaN is a quiet NaN whose sign and payload are not specified, so that every quiet NaN of the type of $X$ and $Y$ is a conforming result. An operand that is $+\infty$ or $-\infty$ lies outside $[-1, 1]$ and is ruled out by the constraint [<b><span style="font-family: 'Courier New', monospace">E_ACOS_FLOAT_CONSTR_X_0010</span></b>](#E_ACOS_FLOAT_CONSTR_X_0010). An operand that is $+0$ or $-0$ gives $\frac{\pi}{2}$ rounded to the type of $X$ and $Y$: the arccosine of a null value is $\frac{\pi}{2}$ whatever the sign of the zero, so that $+0$ and $-0$ give the same result.

**Values of the result.** The result lies in $[0, \pi]$, which is within the range of the three types: it is never an infinity, and the rounding of the exact arccosine to the type of $X$ and $Y$ never overflows. The result is null only when the operand is $1$, and it is then $+0$.

**Exactness.** The result is the real arccosine rounded to the type of $X$ and $Y$; it is exact only when the real arccosine is representable in that type. For instance $\text{acos}(1) = 0$ is exact, whereas $\text{acos}(0) = \frac{\pi}{2}$ is representable in none of the three types and the result is the nearest value of the type.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. The result is a rank-0 tensor whose single element is the arccosine of the single element of $X$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The result has the same shape as $X$ and is then empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

For the `float` type:

$$
X = \begin{bmatrix} -1.0 & 0.0 & 1.0 \end{bmatrix}
$$

$$
Y \approx \begin{bmatrix} 3.14159274 & 1.57079637 & 0.0 \end{bmatrix}
$$

The arccosine of $1.0$ is $0.0$, which is exact. The arccosine of $-1.0$ is $\pi$ and that of $0.0$ is $\frac{\pi}{2}$; neither is representable in `float`, so that the result is the nearest `float` of each of them.

### Example 2

For the `float` type:

$$
X = \begin{bmatrix} \text{+0} & \text{-0} \end{bmatrix}
$$

$$
Y \approx \begin{bmatrix} 1.57079637 & 1.57079637 \end{bmatrix}
$$

The arccosine of a null value is $\frac{\pi}{2}$ whatever the sign of the zero, so that $+0$ and $-0$ give the same result.

### Example 3

For the `float` type:

$$
X = \begin{bmatrix} \text{NaN} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} \text{NaN} \end{bmatrix}
$$

A NaN operand gives a NaN result; every quiet NaN of the type of $X$ and $Y$ is a conforming result.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. an operand outside $[-1, 1]$, such as $\pm\infty$ or a finite value greater than $1$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_ACOS_FLOAT_CONSTR_X_0010</span></b>](#E_ACOS_FLOAT_CONSTR_X_0010) |
| NaN operand | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ACOS_FLOAT_FUNC_0010</span></b>](#E_ACOS_FLOAT_FUNC_0010), the result is a quiet NaN |
| Overflow, i.e. a result beyond the largest finite value of the type | not applicable: the result lies in $[0, \pi]$, which is within the range of the three types |
| Division by zero | not applicable: the operator performs no division |

No error condition.

## Attributes

Operator **Acos** has no attribute.

## Inputs

### $\text{X}$: floating-point tensor

The value whose arccosine is computed.

#### Constraints

<a id="E_ACOS_FLOAT_CONSTR_X_0010"></a>
- `[E_ACOS_FLOAT_CONSTR_X_0010]` Definition domain
  - Statement: Every element of $X$ lies in $[-1, 1]$.
  - Rationale: The arccosine is defined on $[-1, 1]$ only; outside that interval no real number has $x$ as its cosine, and the values $\pm\infty$ lie outside it.

<a id="E_ACOS_FLOAT_CONSTR_X_0020"></a>
- `[E_ACOS_FLOAT_CONSTR_X_0020]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: floating-point tensor

Tensor $Y$ is the element-wise arccosine of $X$.

#### Constraints

<a id="E_ACOS_FLOAT_CONSTR_Y_0010"></a>
- `[E_ACOS_FLOAT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as $X$.
- `[E_ACOS_FLOAT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ACOS_FLOAT_CONSTR_X_0020</span></b>](#E_ACOS_FLOAT_CONSTR_X_0020) on tensor $X$.
