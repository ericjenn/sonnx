# Contents

- **Asin** operator for type [real](#real)
- **Asin** operator for types [`float16`, `float`, `double`](#float)

Based on ONNX documentation [Asin version 7](https://onnx.ai/onnx/operators/onnx__Asin.html#asin-7).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **Asin** (real)

## Signature

Definition of operator **Asin** signature:

$Y = \textbf{Asin}(X)$

where
- $X$: input tensor, whose elements lie in $[-1, 1]$
- $Y$: arcsine of $X$, computed element-wise

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Asin** operator.

## Function

<a id="E_ASIN_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ASIN_REAL_FUNC_0010]</br></span>

Operator **Asin** computes the arcsine of tensor $X$ element-wise and stores the result in tensor $Y$. The arcsine is the inverse of the sine: the result is the angle whose sine is the element of $X$.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] = \arcsin(X[i])
$$

where $\arcsin(x)$ is the unique value $y$ such that $\sin(y) = x$ and $y \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$.

The equation $\sin(y) = x$ has infinitely many solutions, so that it alone leaves the result open; the interval selects one of them. The result is thus defined for every $x$ in $[-1, 1]$, and for no other value. The result is exact: $Y[i]$ is the real number $\arcsin(X[i])$, with no approximation.

**Bounds and sign.** $\arcsin(-1) = -\frac{\pi}{2}$, $\arcsin(0) = 0$ and $\arcsin(1) = \frac{\pi}{2}$; the result lies in $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ for every element of $X$. The arcsine is an odd function, so that $\arcsin(-x) = -\arcsin(x)$: the result has the sign of the element of $X$, and a null element gives a null result. The arcsine is increasing on $[-1, 1]$.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. The result is a rank-0 tensor whose single element is the arcsine of the single element of $X$.

**Zero-sized dimensions.** Tensor $X$ may have a zero-sized dimension. Tensor $Y$ then has the same zero-sized dimension and is empty: the operator is applied to no element.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix} -1 & -\frac{\sqrt{2}}{2} & 0 & \frac{1}{2} & 1 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} -\frac{\pi}{2} & -\frac{\pi}{4} & 0 & \frac{\pi}{6} & \frac{\pi}{2} \end{bmatrix}
$$

The results are exact real numbers: $\arcsin\left(-\frac{\sqrt{2}}{2}\right) = -\frac{\pi}{4}$ and $\arcsin\left(\frac{1}{2}\right) = \frac{\pi}{6}$. The first and the last elements are the bounds of the domain, and their results are the bounds of the range.

### Example 2

A rank-0 tensor:

$$
X = 0.5
$$

$$
Y = \frac{\pi}{6}
$$

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator is specified for real numbers, which have no special values and no invalid operation |
| Overflow | not applicable: the result lies in $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$, and the real numbers are not bounded |
| Integer overflow | not applicable: the operator applies to real numbers, not to the integer types |
| Division by zero | not applicable: the operator performs no division |
| An element of $X$ outside $[-1, 1]$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_ASIN_REAL_CONSTR_X_0010</span></b>](#E_ASIN_REAL_CONSTR_X_0010) |

No error condition.

## Attributes

Operator **Asin** has no attribute.

## Inputs

### $\text{X}$: real tensor

The tensor whose arcsine is computed.

#### Constraints

<a id="E_ASIN_REAL_CONSTR_X_0010"></a>
- `[E_ASIN_REAL_CONSTR_X_0010]` Definition domain
  - Statement: Every element of $X$ lies in $[-1, 1]$: for every tensor index $i$, $-1 \le X[i] \le 1$.
  - Rationale: the arcsine is defined on that interval only; no real number outside it has $X[i]$ as its sine.

## Outputs

### $\text{Y}$: real tensor

Tensor $Y$ is the element-wise arcsine of $X$.

#### Constraints

<a id="E_ASIN_REAL_CONSTR_Y_0010"></a>
- `[E_ASIN_REAL_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the same shape as tensor $X$.

<a id="float"></a>

# **Asin** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same arcsine for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

Definition of operator **Asin** signature:

$Y = \textbf{Asin}(X)$

where
- $X$: input tensor
- $Y$: arcsine of $X$, computed element-wise

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Asin** operator.

## Function

<a id="E_ASIN_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ASIN_FLOAT_FUNC_0010]</br></span>

Operator **Asin** computes the arcsine of tensor $X$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $Y$. Tensors $X$ and $Y$ have the same floating-point type, and the result is a value of that type.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $Y$:

$$
Y[i] =
\begin{cases}
\text{NaN} & \text{if } X[i] \text{ is NaN or } |X[i]| \gt 1 \\
X[i] & \text{if } X[i] \text{ is } \pm 0 \\
\text{round}(\arcsin(X[i])) & \text{otherwise}
\end{cases}
$$

where
- $\arcsin(x)$ is the unique value $y$ such that $\sin(y) = x$ and $y \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$, i.e. the exact arcsine of the real number $x$,
- $\text{round}(x)$ is the value of $x$ rounded to the nearest value of the type of $X$ and $Y$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded.

The second case fixes the sign of a null result: the arcsine of $-0$ is $-0$ and the arcsine of $+0$ is $+0$, whereas the real number $\arcsin(0)$ is $0$ and carries no sign.

The third case applies to every element of $X$ whose magnitude is at most $1$ and which is not a zero, including the bounds $-1$ and $1$: the exact arcsine of such an element lies in $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$, and the result is that value rounded to the type of $X$ and $Y$. The value $\frac{\pi}{2}$ is not representable in any of the three types, so that the result of $\textbf{Asin}(1)$ is the value of the type nearest to $\frac{\pi}{2}$, and the result of $\textbf{Asin}(-1)$ is its opposite.

**The result never overflows.** The magnitude of the exact arcsine of an element of $X$ is at most $\frac{\pi}{2}$, which is below the largest finite value of each of the three types; no result is an infinity, and the rounding of the third case never gives one. A result whose magnitude is below the smallest normal value of the type is a subnormal value: the rounding is the one defined above, with an unbounded exponent range, so that the result is the subnormal value nearest to the exact arcsine and is not flushed to zero.

**Sign and monotonicity.** The arcsine is an odd function, so that the result has the sign of the element of $X$ for every element that is not NaN: $\textbf{Asin}(-x)$ and $\textbf{Asin}(x)$ are opposite values. The result lies in $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ for every element of $X$ that is not NaN, and the arcsine is increasing on $[-1, 1]$.

**Values of the operands.** The operands may be any of the values of the type, including the special numbers $\pm 0$, $\pm\text{inf}$ and NaN. Every element of the type is an admissible operand: no precondition restricts $X$ to $[-1, 1]$, so that an element whose magnitude is greater than $1$ is an operand the operator accepts, and the first case of the formula gives NaN for it. The cases above give the result for every one of them: an element that is NaN gives NaN, and an element whose magnitude is greater than $1$, including $+\text{inf}$ and $-\text{inf}$, gives NaN. The arcsine of an infinity is the invalid operation defined in IEEE 754 section 7.2, and so is the arcsine of a finite value whose magnitude is greater than $1$; the result is NaN in both cases. The NaN of the first case is a quiet NaN; its sign and its payload are not specified, whether the NaN comes from an operand or from the invalid operation, so that every quiet NaN of the type of $X$ and $Y$ is a conforming result.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. The result is a rank-0 tensor whose single element is the arcsine of the single element of $X$.

**Zero-sized dimensions.** Tensor $X$ may have a zero-sized dimension. Tensor $Y$ then has the same zero-sized dimension and is empty: the operator is applied to no element.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples. The accuracy of an implementation whose result departs from the value stated above is analyzed in the [accuracy guidelines](./../other/accuracy.md).

### Example 1

For the `float` type:

$$
X = \begin{bmatrix} -1.0 & -0.5 & 0.0 & 0.5 & 1.0 \end{bmatrix}
$$

$$
Y \approx \begin{bmatrix} -1.5707964 & -0.5235988 & 0.0 & 0.5235988 & 1.5707964 \end{bmatrix}
$$

The exact arcsines are $-\frac{\pi}{2}$, $-\frac{\pi}{6}$, $0$, $\frac{\pi}{6}$ and $\frac{\pi}{2}$; none of them but $0$ is representable in `float`, so each element of $Y$ is the value of `float` nearest to the exact arcsine.

### Example 2

For the `float` type:

$$
X = \begin{bmatrix} \text{NaN} & \text{+inf} & \text{-inf} & 2.0 & \text{-0.0} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{-0.0} \end{bmatrix}
$$

The arcsine of an infinity and the arcsine of a finite value whose magnitude is greater than $1$ are invalid operations of IEEE 754 section 7.2 and give NaN; the arcsine of a NaN is that NaN. The arcsine of $-0$ is $-0$, the sign of a null result being the sign of the operand.

### Example 3

For the `float16` type, whose smallest positive subnormal value is $2^{-24}$:

$$
X = \begin{bmatrix} 2^{-24} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} 2^{-24} \end{bmatrix}
$$

The exact arcsine of $2^{-24}$ is $2^{-24} + \frac{2^{-72}}{6} + \cdots$, whose nearest `float16` value is $2^{-24}$ itself: the result is the subnormal value $2^{-24}$, and it is not flushed to zero.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. the arcsine of an element whose magnitude is greater than $1$, including $\pm\text{inf}$ | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ASIN_FLOAT_FUNC_0010</span></b>](#E_ASIN_FLOAT_FUNC_0010), an element whose magnitude exceeds $1$ is an admissible operand and the result is NaN |
| Overflow, i.e. a result beyond the largest finite value of the type | not applicable: the magnitude of the result is at most $\frac{\pi}{2}$, which is below the largest finite value of each of the three types |
| Underflow, i.e. a result whose magnitude is below the smallest normal value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ASIN_FLOAT_FUNC_0010</span></b>](#E_ASIN_FLOAT_FUNC_0010), the result is the subnormal value nearest to the exact arcsine |
| Integer overflow | not applicable: the operator applies to the floating-point types only |
| Division by zero | not applicable: the operator performs no division |

Every condition of the list that applies is part of the nominal behavior of the operator; none of them is an error. No error condition.

## Attributes

Operator **Asin** has no attribute.

## Inputs

### $\text{X}$: floating-point tensor

The tensor whose arcsine is computed.

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
- `[E_ASIN_FLOAT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ASIN_FLOAT_CONSTR_X_0010</span></b>](#E_ASIN_FLOAT_CONSTR_X_0010) on tensor $X$.
