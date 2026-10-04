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
- $X$: the input tensor, named `input` in the ONNX definition
- $Y$: the result, the arccosine of $X$ computed element-wise, named `output` in the ONNX definition

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Acos** operator.

## Function

<a id="E_ACOS_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ACOS_REAL_FUNC_0010]</br></span>

**Part 1.** Operator **Acos** computes the arccosine of the input tensor $X$ element-wise and stores the result in tensor $Y$. The arccosine is the inverse of the cosine: the result is the angle whose cosine is the element of $X$.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = \text{acos}(X[i])
$$

where
- $\text{acos}(x)$ is the arccosine of the real number $x$: for $x \in [-1,1]$, it is the unique real $y$ such that $\cos(y) = x$ and $y \in [0,\pi]$.

The equation $\cos(y) = x$ alone has infinitely many solutions; the interval $[0,\pi]$ selects one of them, and it is as specifying as the equation. The result is exact: $Y[i]$ is the real number $y$, with no approximation.

**Domain.** The arccosine is defined for the elements of $X$ that lie in $[-1,1]$; constraint [<b><span style="font-family: 'Courier New', monospace">E_ACOS_REAL_CONSTR_X_0010</span></b>](#E_ACOS_REAL_CONSTR_X_0010) states that every element of $X$ lies in that interval. An element outside it has no arccosine in $\mathbb{R}$, and the operator is not defined for it.

**Values of the result.** The result lies in $[0,\pi]$: it is $0$ for $X[i] = 1$, $\frac{\pi}{2}$ for $X[i] = 0$, and $\pi$ for $X[i] = -1$. The arccosine is even, so that $X[i]$ and $-X[i]$ give the same result.

**Shapes.** The operator has a single input, so no broadcasting applies: tensor $Y$ has the shape of tensor $X$, and $Y[i]$ is the arccosine of $X[i]$ for every tensor index $i$ of $X$.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty; the result is then a rank-0 tensor whose single element is the arccosine of the single element of $X$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension; the corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix} 1 & 0 & -1 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} 0 & \frac{\pi}{2} & \pi \end{bmatrix}
$$

### Example 2

$$
X = \begin{bmatrix} \frac{1}{2} & -\frac{1}{2} & \frac{\sqrt{2}}{2} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} \frac{\pi}{3} & \frac{2\pi}{3} & \frac{\pi}{4} \end{bmatrix}
$$

The arccosine is even: the elements $\frac{1}{2}$ and $-\frac{1}{2}$ give the supplementary angles $\frac{\pi}{3}$ and $\frac{2\pi}{3}$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. an element of $X$ outside $[-1,1]$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_ACOS_REAL_CONSTR_X_0010</span></b>](#E_ACOS_REAL_CONSTR_X_0010) |
| Overflow | not applicable: the arccosine of a real number is a real number, and the result lies in $[0,\pi]$ |
| Integer overflow | not applicable: the operator applies to real numbers |
| Division by zero | not applicable: the operator performs no division |

Every condition of the list is either ruled out by a precondition or does not apply to real numbers; none of them is an error. No error condition.

## Attributes

Operator **Acos** has no attribute.

## Inputs

### $\text{X}$: real tensor

The tensor whose arccosine is computed.

#### Constraints

<a id="E_ACOS_REAL_CONSTR_X_0010"></a>
- `[E_ACOS_REAL_CONSTR_X_0010]` Definition domain
  - Statement: Every element of $X$ lies in $[-1,1]$.
  - Rationale: The arccosine is defined on that interval only; outside it, no real number has $X[i]$ as its cosine.

## Outputs

### $\text{Y}$: real tensor

Tensor $Y$ is the element-wise arccosine of $X$.

#### Constraints

<a id="E_ACOS_REAL_CONSTR_Y_0010"></a>
- `[E_ACOS_REAL_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the shape of $X$.
<a id="E_ACOS_REAL_CONSTR_Y_0020"></a>
- `[E_ACOS_REAL_CONSTR_Y_0020]` Range
  - Statement: Every element of $Y$ lies in $[0,\pi]$.

<a id="float"></a>

# **Acos** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same arccosine for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

$Y = \textbf{Acos}(X)$

where
- $X$: the input tensor, named `input` in the ONNX definition
- $Y$: the result, the arccosine of $X$ computed element-wise, named `output` in the ONNX definition

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Acos** operator.

## Function

<a id="E_ACOS_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ACOS_FLOAT_FUNC_0010]</br></span>

**Part 1.** Operator **Acos** computes the arccosine of the input tensor $X$ element-wise and stores the result in tensor $Y$. The arccosine is the inverse of the cosine: the result is the angle in $[0,\pi]$ whose cosine is the element of $X$, rounded to the type of $X$.

**Part 2.** For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = \text{acos}_{(\text{f})}(X[i]) =
\begin{cases}
\text{NaN} & \text{if } X[i] \text{ is NaN or } |X[i]| > 1 \\
\text{round}(\text{acos}(X[i])) & \text{otherwise}
\end{cases}
$$

where
- $\text{acos}_{(\text{f})}$ is the arccosine of the floating-point type of $X$, i.e. $\text{acos}_{(\text{f16})}$, $\text{acos}_{(\text{f32})}$ or $\text{acos}_{(\text{f64})}$ according to that type,
- $\text{acos}(x)$ is the arccosine of the real number $x$: for $x \in [-1,1]$, it is the unique real $y$ such that $\cos(y) = x$ and $y \in [0,\pi]$,
- $\text{round}(y)$ is the value of $y$ rounded to the nearest value of the type of $X$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded; the rounded value is always a value of the type of $X$, since the exact arccosine of an element of $[-1,1]$ lies in $[0,\pi]$, which every type of this section represents,
- $|X[i]| > 1$ holds for $+\infty$ and $-\infty$ as well as for the finite values outside $[-1,1]$.

**NaN operands.** An operand that is NaN gives NaN. The NaN of the result is a quiet NaN; its sign and its payload are not specified, so that every quiet NaN of the type of $X$ is a conforming result.

**Operands outside the domain.** An operand whose absolute value is greater than $1$, including $+\infty$ and $-\infty$, is the invalid operation defined in IEEE 754 section 7.2 for the arccosine; the result is NaN, as it is for a NaN operand. The NaN of the result is a quiet NaN, whose sign and its payload are not specified, so that every quiet NaN of the type of $X$ is a conforming result.

**Null operands.** The arccosine is even, so that $+0$ and $-0$ give the same result: the exact value is $\frac{\pi}{2}$, and the result is the value of the type of $X$ nearest to $\frac{\pi}{2}$. That result is not null.

**Subnormal operands.** An operand whose magnitude is below the smallest normal value of the type of $X$ is a subnormal value of that type. The exact arccosine of such an operand is rounded to the type of $X$ as the arccosine of any other element of $[-1,1]$ is, and the result is the value of the type nearest to $\frac{\pi}{2}$: the magnitude of a subnormal operand is far below half the spacing of the values of the type around $\frac{\pi}{2}$, so that the nearest value of the type is the same for the exact arccosine of a subnormal operand as for $\frac{\pi}{2}$ itself. For each of the three types of this section, the arccosine of a subnormal operand is therefore the same value as the arccosine of $+0$.

**The bounds of the domain.** For $X[i] = 1$, the exact value is $0$ and the result is $+0$; for $X[i] = -1$, the exact value is $\pi$ and the result is the value of the type of $X$ nearest to $\pi$. The result is never $-0$: the exact arccosine of an element of $[-1,1]$ lies in $[0,\pi]$, and the rounding of a non-negative value is non-negative.

**The result is never an infinity and never subnormal.** The exact arccosine of an element of $[-1,1]$ lies in $[0,\pi]$, which every type of this section represents, so that no overflow occurs. The smallest positive value of the result is the arccosine of the largest value of the type below $1$, which is greater than the smallest normal value of the type, so that no underflow occurs either: the result is $+0$ only for $X[i] = 1$, and it is a normal value otherwise.

**Shapes.** The operator has a single input, so no broadcasting applies: tensor $Y$ has the shape of tensor $X$, and $Y[i]$ is the arccosine of $X[i]$ for every tensor index $i$ of $X$.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty; the result is then a rank-0 tensor whose single element is the arccosine of the single element of $X$.

**Zero-sized dimensions.** An operand may have a zero-sized dimension; the corresponding dimension of $Y$ then has size $0$, and $Y$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

For the `double` type:

$$
X = \begin{bmatrix} 1.0 & 0.0 & -1.0 \end{bmatrix}
$$

$$
Y \approx \begin{bmatrix} 0.0 & 1.5707963267948966 & 3.141592653589793 \end{bmatrix}
$$

The arccosine of $1.0$ is exactly $0$, and the result is $+0$. The arccosines of $0.0$ and $-1.0$ are $\frac{\pi}{2}$ and $\pi$, neither of which is representable in `double`; each result is the nearest `double` value.

### Example 2

For the `double` type:

$$
X = \begin{bmatrix} \text{NaN} & \text{+inf} & \text{-inf} & 2.0 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} \end{bmatrix}
$$

The arccosine of an operand whose absolute value is greater than $1$ is the invalid operation of IEEE 754 section 7.2, and the arccosine of a NaN operand is NaN; the result is NaN in the four cases.

### Example 3

For the `float16` type:

$$
X = \begin{bmatrix} 0.5 & -0.5 & 0.0 & -0.0 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} 1.046875 & 2.09375 & 1.5703125 & 1.5703125 \end{bmatrix}
$$

The exact arccosines are $\frac{\pi}{3}$, $\frac{2\pi}{3}$, $\frac{\pi}{2}$ and $\frac{\pi}{2}$, none of which is representable in `float16`; each result is the nearest `float16` value. The arccosine is even, so that $0.0$ and $-0.0$ give the same result.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. an operand that is NaN or whose absolute value is greater than $1$ | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ACOS_FLOAT_FUNC_0010</span></b>](#E_ACOS_FLOAT_FUNC_0010), the result is NaN |
| Overflow, i.e. a result that would exceed the largest finite value of the type | not applicable: the exact arccosine of an element of $[-1,1]$ lies in $[0,\pi]$, which every type of this section represents |
| Underflow, i.e. a result whose magnitude is below the smallest normal value of the type | not applicable: the arccosine of an element of $[-1,1]$ is at least the arccosine of the largest value of the type below $1$, which is greater than the smallest normal value of the type |
| Integer overflow | not applicable: the operator applies to floating-point types |
| Division by zero | not applicable: the operator performs no division |

Every condition of the list is either part of the nominal behavior of the operator or cannot occur; none of them is an error. No error condition.

## Attributes

Operator **Acos** has no attribute.

## Inputs

### $\text{X}$: floating-point tensor

The tensor whose arccosine is computed.

#### Constraints

<a id="E_ACOS_FLOAT_CONSTR_X_0010"></a>
- `[E_ACOS_FLOAT_CONSTR_X_0010]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: floating-point tensor

Tensor $Y$ is the element-wise arccosine of $X$.

#### Constraints

<a id="E_ACOS_FLOAT_CONSTR_Y_0010"></a>
- `[E_ACOS_FLOAT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the shape of $X$.
- `[E_ACOS_FLOAT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ACOS_FLOAT_CONSTR_X_0010</span></b>](#E_ACOS_FLOAT_CONSTR_X_0010) on tensor $X$.
<a id="E_ACOS_FLOAT_CONSTR_Y_0030"></a>
- `[E_ACOS_FLOAT_CONSTR_Y_0030]` Range
  - Statement: Every element of $Y$ is either NaN or a value in $[0,\pi]$.
