# Contents

- **Asin** operator for type [real](#real)
- **Asin** operator for types [`float16`, `float`, `double`](#float)

Based on ONNX documentation [Asin version 7](https://onnx.ai/onnx/operators/onnx__Asin.html#asin-7).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14 (the **Asin** schema is version 7, unchanged since opset 7).

<a id="real"></a>

# **Asin** (real)

## Signature

$Y = \textbf{Asin}(X)$

where
- $X$: value whose arcsine is computed
- $Y$: result of the element-wise arcsine of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Asin** operator.

## Function

<a id="E_ASIN_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ASIN_REAL_FUNC_0010]</br></span>

Operator **Asin** computes the arcsine of tensor $X$ element-wise and stores the result in tensor $Y$. The arcsine is the inverse of the sine: each element of $Y$ is the value of $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ whose sine is the corresponding element of $X$.

For every [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$, the result $Y[i]$ is the unique value $y$ such that

$$
\sin(y) = X[i] \quad \text{and} \quad y \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]
$$

The equation $\sin(y) = X[i]$ alone has infinitely many solutions, one in each interval of length $2\pi$; the interval $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ selects exactly one of them, and it is therefore as specifying as the equation. The result is the arcsine of $X[i]$, written $\arcsin(X[i])$.

**Domain.** The equation has a solution only when $X[i]$ lies in $[-1, 1]$, the range of the sine; outside that interval no real number $y$ satisfies it, and the operator is not defined. The domain of the operator is the set of tensors whose every element lies in $[-1, 1]$, and it is stated as the precondition [<b><span style="font-family: 'Courier New', monospace">E_ASIN_REAL_CONSTR_X_0010</span></b>](#E_ASIN_REAL_CONSTR_X_0010) on input $X$.

**Bounds and sign.** The result lies in $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ for every element of the domain: $\arcsin(-1) = -\frac{\pi}{2}$, $\arcsin(0) = 0$ and $\arcsin(1) = \frac{\pi}{2}$. The arcsine is odd, $\arcsin(-x) = -\arcsin(x)$, so that the result has the sign of the operand and is null only for a null operand.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. The result is a rank-0 tensor whose single element is the arcsine of the single element of $X$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The result has the same shape, so that the corresponding dimension of $Y$ has size $0$ and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix} 0 & \frac{1}{2} & 1 & -1 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} 0 & \frac{\pi}{6} & \frac{\pi}{2} & -\frac{\pi}{2} \end{bmatrix}
$$

$\arcsin(0) = 0$, $\arcsin\left(\frac{1}{2}\right) = \frac{\pi}{6}$, $\arcsin(1) = \frac{\pi}{2}$ and $\arcsin(-1) = -\frac{\pi}{2}$. The results are exact: they are the values of the real numbers, not rounded values.

### Example 2

$$
X = \begin{bmatrix} \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} \frac{\pi}{4} & -\frac{\pi}{4} \end{bmatrix}
$$

$\arcsin\left(\frac{\sqrt{2}}{2}\right) = \frac{\pi}{4}$ and the arcsine is odd, so that the second result is $-\frac{\pi}{4}$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. an operand outside the domain of the arcsine, $\lvert X[i] \rvert \gt 1$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_ASIN_REAL_CONSTR_X_0010</span></b>](#E_ASIN_REAL_CONSTR_X_0010) |
| Overflow | not applicable: the result lies in $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ for every element of the domain |
| Integer overflow | not applicable: the operator applies to real numbers |
| Division by zero | not applicable: the operator performs no division |

The only condition of the list that applies to real numbers is the operand outside the domain, and it is ruled out by the precondition on $X$; the result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_ASIN_REAL_FUNC_0010</span></b>](#E_ASIN_REAL_FUNC_0010) for every tensor of the domain. No error condition.

## Attributes

Operator **Asin** has no attribute.

## Inputs

### $\text{X}$: real tensor

The value whose arcsine is computed.

#### Constraints

<a id="E_ASIN_REAL_CONSTR_X_0010"></a>
- `[E_ASIN_REAL_CONSTR_X_0010]` Definition domain
  - Statement: Every element of $X$ lies in $[-1, 1]$, i.e. $\forall i, -1 \le X[i] \le 1$.
  - Rationale: Outside that interval no real number has $X[i]$ as its sine, so that the arcsine of $X[i]$ is not defined.

## Outputs

### $\text{Y}$: real tensor

Tensor $Y$ is the element-wise arcsine of $X$.

#### Constraints

<a id="E_ASIN_REAL_CONSTR_Y_0010"></a>
- `[E_ASIN_REAL_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as tensor $X$.
<a id="E_ASIN_REAL_CONSTR_Y_0020"></a>
- `[E_ASIN_REAL_CONSTR_Y_0020]` Range
  - Statement: Every element of $Y$ lies in $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$.
  - Rationale: The arcsine is the branch of the inverse sine whose values lie in that interval.

<a id="float"></a>

# **Asin** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same arcsine for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

$Y = \textbf{Asin}(X)$

where
- $X$: value whose arcsine is computed
- $Y$: result of the element-wise arcsine of $X$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Asin** operator.

## Function

<a id="E_ASIN_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ASIN_FLOAT_FUNC_0010]</br></span>

Operator **Asin** computes the arcsine of tensor $X$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $Y$. Each element of $Y$ is the arcsine of the corresponding element of $X$, rounded to the type of $X$ and $Y$.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = \arcsin_{(\text{f})}(X[i]) =
\begin{cases}
\text{NaN} & \text{if } X[i] \text{ is NaN} \\
\text{NaN} & \text{if } X[i] \text{ is } +\infty \text{ or } -\infty \text{, or } \lvert X[i] \rvert \gt 1 \\
X[i] & \text{if } X[i] \text{ is } +0 \text{ or } -0 \\
\text{round}(\arcsin(X[i])) & \text{otherwise}
\end{cases}
$$

where
- $\arcsin_{(\text{f})}$ is the arcsine for the floating-point type of $X$ and $Y$, i.e. $\arcsin_{(\text{f16})}$, $\arcsin_{(\text{f32})}$ or $\arcsin_{(\text{f64})}$ according to that type,
- $\arcsin(x)$ is the arcsine of the real number $x$, the unique value $y$ such that $\sin(y) = x$ and $y \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$,
- $\text{round}(x)$ is the value of $x$ rounded to the nearest value of the type of $X$ and $Y$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded.

**NaN operand.** An operand that is NaN is propagated: the result is NaN. The NaN of the result is a quiet NaN; its sign and its payload are not specified, so that every quiet NaN of the type of $X$ and $Y$ is a conforming result. A signaling NaN operand is a NaN and gives the same result.

**Operand outside the domain.** An operand whose magnitude is greater than $1$, including an infinite operand, lies outside the domain of the arcsine: the operation is the invalid operation defined in IEEE 754 section 7.2, and the result is NaN. That NaN is a quiet NaN, and every quiet NaN of the type of $X$ and $Y$ is a conforming result.

**Null operand.** $\arcsin(+0) = +0$ and $\arcsin(-0) = -0$: the result is the operand itself, sign included.

**Operand in the domain.** For $0 \lt \lvert X[i] \rvert \le 1$, the result is the arcsine of $X[i]$ rounded as defined above. The exact arcsine is not representable in general: $\arcsin(\pm 1) = \pm\frac{\pi}{2}$ is irrational, and so is $\arcsin(x)$ for most $x$; the result is then the nearest value of the type of $X$ and $Y$. The result is never an infinity, since $\lvert \arcsin(x) \rvert \le \frac{\pi}{2}$ for every $x$ of the domain, so that the operator never overflows. The result may be subnormal: for a subnormal operand, the exact arcsine is a value of the same magnitude, and it rounds to a subnormal value of the type.

**Bounds and sign.** The result lies in $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ for every operand of the domain, and it is NaN otherwise. The arcsine is odd, so that the result has the sign of the operand for every operand of the domain.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. The result is a rank-0 tensor whose single element is the arcsine of the single element of $X$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The result has the same shape, so that the corresponding dimension of $Y$ has size $0$ and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

For the `float` type:

$$
X = \begin{bmatrix} 0.0 & 0.5 & 1.0 & -1.0 \end{bmatrix}
$$

$$
Y \approx \begin{bmatrix} 0.0 & 0.5235988 & 1.5707964 & -1.5707964 \end{bmatrix}
$$

$\arcsin(0.5) = \frac{\pi}{6} \approx 0.5235987755982988$ and $\arcsin(1) = \frac{\pi}{2} \approx 1.5707963267948966$; neither is representable in `float`, and the corresponding elements of $Y$ are those values rounded to the nearest `float`. The first element is exact.

### Example 2

For the `float` type:

$$
X = \begin{bmatrix} \text{NaN} & \text{+inf} & \text{-inf} & \text{+0} & \text{-0} & 2.0 & -2.0 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} \text{NaN} & \text{NaN} & \text{NaN} & \text{+0} & \text{-0} & \text{NaN} & \text{NaN} \end{bmatrix}
$$

The NaN operand is propagated. The infinite operands and the operands $2.0$ and $-2.0$ lie outside the domain of the arcsine, and the invalid operation of IEEE 754 section 7.2 gives NaN. The null operands give the null result of the same sign.

### Example 3

For the `float` type, a subnormal operand:

$$
X = \begin{bmatrix} 2^{-140} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} 2^{-140} \end{bmatrix}
$$

The operand is subnormal; the exact arcsine differs from it by less than half the spacing of the subnormal values of the type, so that the result is the operand itself, a subnormal value.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. an operand whose magnitude is greater than $1$, including an infinite operand | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ASIN_FLOAT_FUNC_0010</span></b>](#E_ASIN_FLOAT_FUNC_0010), the result is NaN |
| Invalid operation, i.e. an operation on a signaling NaN | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ASIN_FLOAT_FUNC_0010</span></b>](#E_ASIN_FLOAT_FUNC_0010), the result is a quiet NaN |
| Overflow | not applicable: the result lies in $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ for every element of the domain, so that no result exceeds the largest finite value of the type |
| Underflow, i.e. a result with a magnitude below the smallest normal value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ASIN_FLOAT_FUNC_0010</span></b>](#E_ASIN_FLOAT_FUNC_0010), the result is the arcsine rounded to the type, a subnormal value when the operand is subnormal |
| Integer overflow | not applicable: the operator applies to floating-point types |
| Division by zero | not applicable: the operator performs no division |

Every condition of the list that applies to the floating-point types is part of the nominal behavior of the operator; none of them is an error. No error condition.

## Attributes

Operator **Asin** has no attribute.

## Inputs

### $\text{X}$: floating-point tensor

The value whose arcsine is computed.

#### Constraints

<a id="E_ASIN_FLOAT_CONSTR_X_0010"></a>
- `[E_ASIN_FLOAT_CONSTR_X_0010]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: floating-point tensor

Tensor $Y$ is the element-wise arcsine of $X$.

#### Constraints

<a id="E_ASIN_FLOAT_CONSTR_Y_0010"></a>
- `[E_ASIN_FLOAT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as tensor $X$.
<a id="E_ASIN_FLOAT_CONSTR_Y_0020"></a>
- `[E_ASIN_FLOAT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ASIN_FLOAT_CONSTR_X_0010</span></b>](#E_ASIN_FLOAT_CONSTR_X_0010) on tensor $X$.
<a id="E_ASIN_FLOAT_CONSTR_Y_0030"></a>
- `[E_ASIN_FLOAT_CONSTR_Y_0030]` Range
  - Statement: Every element of $Y$ is either NaN or a value of $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$.
  - Rationale: The arcsine is the branch of the inverse sine whose values lie in that interval, and an operand outside the domain gives NaN.
