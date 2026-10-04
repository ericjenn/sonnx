# Contents

- **Sin** operator for type [real](#real)
- **Sin** operator for types [`float16`, `float`, `double`](#float)

Based on ONNX documentation [Sin version 14](https://onnx.ai/onnx/operators/onnx__Sin.html#sin-14).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **Sin** (real)

## Signature

$C = \textbf{Sin}(A)$

where
- $A$: operand
- $C$: result of the element-wise sine of $A$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Sin** operator.

## Function

<a id="E_SIN_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_SIN_REAL_FUNC_0010]</br></span>

Operator **Sin** applies the sine to tensor $A$ element-wise and stores the result in tensor $C$. Each element of $C$ is the sine of the element of $A$ that corresponds to it.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $C$:

$$
C[i] = \sin(A[i])
$$

The sine is the function of $\mathbb{R}$ into the interval $[-1, 1]$; the value of $C[i]$ is that function of the element, exactly, with no approximation and no rounding.

**The sine is odd.** $\sin(-x) = -\sin(x)$ for every real number $x$: changing the sign of an element of $A$ changes the sign of the corresponding element of $C$ and leaves its magnitude unchanged.

**The sine is periodic.** $\sin(x + 2k\pi) = \sin(x)$ for every integer $k$ and every real number $x$: adding any multiple of $2\pi$ to an element of $A$ leaves the corresponding element of $C$ unchanged. The period of the sine is $2\pi$: no smaller positive value leaves it unchanged.

**The sine vanishes on the multiples of $\pi$.** $C[i] = 0$ if and only if $A[i] = k\pi$ for an integer $k$.

**The values of the sine lie in $[-1, 1]$.** $|C[i]| \le 1$ for every element: $C[i] = 1$ if and only if $A[i] = \pi/2 + 2k\pi$, and $C[i] = -1$ if and only if $A[i] = -\pi/2 + 2k\pi$, for an integer $k$.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. Its single element is an element of $A$ like any other: $C$ is a rank-0 tensor whose single element is the sine of that element.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension of $A$ is a dimension of size $0$ of $C$: $C$ is empty along it, the corresponding elements of $C$ do not exist, and the sine of no element is computed.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
A = \begin{bmatrix} 0 & \pi/6 & \pi/4 & \pi/2 & \pi & 3\pi/2 \end{bmatrix}
$$

$$
C = \begin{bmatrix} 0 & 1/2 & \sqrt{2}/2 & 1 & 0 & -1 \end{bmatrix}
$$

The result is exact: the sine of a multiple of $\pi$ is null, and the sine of $\pi/6$, of $\pi/4$ and of $\pi/2$ is the value given.

### Example 2

$$
A = \begin{bmatrix} -\pi/6 & \pi/6 & 13\pi/6 \end{bmatrix}
$$

$$
C = \begin{bmatrix} -1/2 & 1/2 & 1/2 \end{bmatrix}
$$

The second and the third elements are equal although their operands are not: the sine is periodic of period $2\pi$ and $13\pi/6 = \pi/6 + 2\pi$. The first element is the opposite of the second: the sine is odd.

## Error conditions

None of the conditions of the list applies to real numbers: the sine is defined for every real number, so that no argument is outside the domain of the operator and no invalid operation can occur; the result lies in $[-1, 1]$, so that no computation of the operator overflows or underflows; the operator performs neither a division, nor a remainder, nor a square root; and the result is exact, so that it is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_SIN_REAL_FUNC_0010</span></b>](#E_SIN_REAL_FUNC_0010) for every operand. No error condition.

## Attributes

Operator **Sin** has no attribute.

## Inputs

### $\text{A}$: real tensor

The operand.

#### Constraints

The sine is defined for every real number: no constraint applies to tensor $A$.

## Outputs

### $\text{C}$: real tensor

Tensor $C$ is the element-wise sine of $A$.

#### Constraints

- `[E_SIN_REAL_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$.

<a id="float"></a>

# **Sin** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the result is the sine of the operand represented in the type of the operand, and the three differ only by their precision and by the range of the values they represent.

## Signature

$C = \textbf{Sin}(A)$

where
- $A$: operand
- $C$: result of the element-wise sine of $A$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Sin** operator.

## Function

<a id="E_SIN_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_SIN_FLOAT_FUNC_0010]</br></span>

Operator **Sin** applies the sine to tensor $A$ element-wise and stores the result in tensor $C$, in the type of $A$. Each element of $C$ is the sine of the element of $A$ that corresponds to it.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $C$:

$$
C[i] =
\begin{cases}
\text{NaN} & \text{if } A[i] \text{ is NaN or an infinity} \\
A[i] & \text{if } A[i] \text{ is a zero} \\
\text{nearest}(\sin(A[i])) & \text{otherwise}
\end{cases}
$$

where
- $\sin(A[i])$ is the value of the sine of the element, in $\mathbb{R}$: the sine of the real number that the element of $A$ is,
- $\text{nearest}(x)$ is the value of $x$ rounded to the nearest value of the type of $A$ using the roundTiesToEven attribute of IEEE 754. Since $|\sin x| \le 1$ for every real number $x$, no result leaves the range of the type, and the exponent range of the type plays no role.

The last case gives the value of the type of $A$ that represents the exact sine of the element: the sine of a value of a floating-point type is in general not a value of that type, and that value is the representation of it.

**The exact sine is what the operator specifies, and the way an implementation computes it is not specified here.** IEEE 754 requires a correctly rounded result of the operations it makes mandatory and of the mathematical functions of its clause 9.2 that a language chooses to provide as conforming operations, but it leaves to the language the choice of providing those functions and of requiring them to be correctly rounded, and it fixes no result for the sine of a value of a floating-point type. The value above is therefore the value the operator specifies, and the difference between the value returned by a particular implementation and that value is the introduced error of that implementation, whose analysis is the subject of the [accuracy guidelines](./../other/accuracy.md) of SONNX and is not part of this specification.

**Every result lies in $[-1, 1]$, and no computation underflows.** No element of $C$ is an infinity, since no element of the exact sine exceeds $1$ in magnitude. No element of $C$ is a subnormal value obtained by underflow either: for an element of sufficiently small magnitude, the difference $|\sin x - x|$, which is at most $x^3/6$, is far below half a unit in the last place of $x$, so that the element is its own sine, and the subnormal values of a result are therefore the elements of $A$ themselves. An element of $C$ is null if and only if the element of $A$ is a zero.

**The sine is odd, and the result is odd too.** $\sin(-x) = -\sin(x)$ for every real number $x$, and the negation of a value of a floating-point type is exact: the result for the operand $-A[i]$ is the negation of the result for $A[i]$, the sign bit included. The second case above follows that rule: the sine of $+0$ is $+0$ and the sine of $-0$ is $-0$.

**The sine is not periodic on the values of a floating-point type.** $2\pi$ is a value of none of the three types, and the periodicity of the sine is a property of the real numbers: adding a value of the type to an element of $A$ does not leave the corresponding element of $C$ unchanged in general. The result is in particular null only for a null element, and not for the elements of the type that are closest to a multiple of the period: the nearest `double` value of $\pi$ is not a value whose sine is null, and its sine is $1.2246467991473532\text{e-}16$.

**Values of the operand.** The operand may be any value of the type, including the special numbers $\pm 0$, $\pm\text{inf}$ and NaN. The cases above give the result for every combination of them. The sine has no limit at infinity and no value of the type represents it, so that the sine of an infinity is the invalid operation of IEEE 754 section 7.2 and the result is NaN. The NaN of the first case is a quiet NaN; its sign and its payload are not specified, whether the NaN comes from an operand or from the invalid operation, so that every quiet NaN of the type of $A$ is a conforming result.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. Its single element is an element of $A$ like any other: $C$ is a rank-0 tensor whose single element is the sine of that element.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension of $A$ is a dimension of size $0$ of $C$: $C$ is empty along it, the corresponding elements of $C$ do not exist, and the sine of no element is computed.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

> [!warning] V1 · The last bits of a result
> The value the "Function" section states is the exact sine of the element, rounded to the type
> of $A$. IEEE 754 requires a correctly rounded result of the operations it makes mandatory, and
> of the functions of its clause 9.2 for a language that provides them as conforming operations,
> but providing them and requiring them to be correctly rounded is optional for a language, and
> implementations of the sine do not all return the correctly rounded value.
> ONNX Runtime 1.30 (CPU provider, x86-64) does not, as soon as the tensor has more than one
> element. Against the exact sine computed in 60-digit arithmetic and rounded to the type, over
> five hundred values drawn uniformly from $[0, 10]$ and their opposites, and forty values of
> larger magnitude, 72% of its `double` results and 20% of its `float` results differ from the
> correctly rounded value: by at most two units in the last place of the value $1$ for `double`
> and half of one for `float`, which is many units in the last place of the result when that
> result is small (the largest departure of that run is 54 units in the last place of the
> result). Over the whole domain of `float16`, four of the 63 488 ordinary values only differ,
> each by half a unit in the last place of the value $1$. Even for an operand so small that the
> correctly rounded sine is the operand itself, ONNX Runtime returns a neighbouring value, one
> unit in the last place of the operand away. A tensor of a single element, on the other hand,
> takes a scalar path and is correctly rounded in every case examined.
> **Now** the document states the exact sine as the value of the operator, which is what a
> correctly rounded implementation returns, and it refers a departure from that value to the
> accuracy guidelines, which are where SONNX places accuracy. A case that compares a sine with
> the reference implementation bit for bit measures the accuracy of two implementations; it does
> not test the rule of the operator.

The effect of the operator is illustrated on the following examples.

### Example 1

For `double`:

$$
A = \begin{bmatrix} 0.0 & -0.0 & 1.0\text{e-}8 & -1.0\text{e-}8 & \text{+inf} & \text{-inf} & \text{NaN} \end{bmatrix}
$$

$$
C = \begin{bmatrix} 0.0 & -0.0 & 1.0\text{e-}8 & -1.0\text{e-}8 & \text{NaN} & \text{NaN} & \text{NaN} \end{bmatrix}
$$

A zero is its own sine, sign included: the sine of $-0.0$ is $-0.0$. An element of sufficiently small magnitude is its own sine as well, since the difference between the sine of $x$ and $x$ is at most $x^3/6$: the sine of $1.0\text{e-}8$ is $1.0\text{e-}8$. The sine of an infinity has no value and the result is NaN, as is the sine of a NaN.

### Example 2

For `float`:

$$
A = \begin{bmatrix} 3.1415927 & 6.2831855 & 1.0\text{e}30 \end{bmatrix}
$$

$$
C \approx \begin{bmatrix} -8.7422777\text{e-}08 & 1.7484555\text{e-}07 & -0.7911634 \end{bmatrix}
$$

The first two elements of $A$ are the nearest `float` values of $\pi$ and of $2\pi$. Their sines are not null, and no value of the type has a null sine except the zeros: the sine is null at the multiples of $\pi$ of the real numbers, and the nearest values of the type to those multiples are not those multiples. The third element shows that the sine of an operand far outside the interval $[-1, 1]$ is still an element of the interval.

### Example 3

For `double`:

$$
A = \begin{bmatrix} 1.0 & -1.0 & 0.5 & 2.0 \end{bmatrix}
$$

$$
C \approx \begin{bmatrix} 0.8414709848078965 & -0.8414709848078965 & 0.479425538604203 & 0.9092974268256817 \end{bmatrix}
$$

None of the four exact sines is a value of `double`: each element of $C$ is the nearest value of the type, the value the specification states. The first two elements show that the result is odd: they are the two first values of the type around the sine of $1.0$, with opposite signs.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e., the sine of an infinity | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SIN_FLOAT_FUNC_0010</span></b>](#E_SIN_FLOAT_FUNC_0010), the result is NaN |
| Overflow, i.e. a computation leading to an infinity | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SIN_FLOAT_FUNC_0010</span></b>](#E_SIN_FLOAT_FUNC_0010), which gives a result in $[-1, 1]$: no element of $C$ is an infinity |
| Underflow, i.e. a computation leading to a subnormal value or to a null value | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SIN_FLOAT_FUNC_0010</span></b>](#E_SIN_FLOAT_FUNC_0010): an element below the smallest normal value of the type is its own sine, and a null element is the only element whose sine is null |
| An exact sine that is not a value of the type of $A$ | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SIN_FLOAT_FUNC_0010</span></b>](#E_SIN_FLOAT_FUNC_0010), the result is the value of the type of $A$ nearest that exact sine |
| An operand of large magnitude, whose reduction modulo the period of the sine loses accuracy in an implementation | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SIN_FLOAT_FUNC_0010</span></b>](#E_SIN_FLOAT_FUNC_0010); the accuracy of an implementation is analysed in the [accuracy guidelines](./../other/accuracy.md) and is not fixed here |

Every condition of the table is part of the nominal behavior of the operator; none of them is an error. No error condition.

The remaining conditions of the list concern operations the operator does not perform: the division of a zero by a zero, the remainder of a division, the square root of a negative value, the multiplication of a zero by an infinity and the subtraction of two infinities are the conditions of operators that divide, take a remainder, take a square root, multiply or subtract, whereas the sine is the only operation of **Sin**.

## Attributes

Operator **Sin** has no attribute.

## Inputs

### $\text{A}$: floating-point tensor

The operand.

#### Constraints

<a id="E_SIN_FLOAT_CONSTR_A_0010"></a>
- `[E_SIN_FLOAT_CONSTR_A_0010]` Type consistency
  - Statement: Tensors $A$ and $C$ have the same type.

## Outputs

### $\text{C}$: floating-point tensor

Tensor $C$ is the element-wise sine of $A$, in the type of $A$.

#### Constraints

- `[E_SIN_FLOAT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$.
- `[E_SIN_FLOAT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SIN_FLOAT_CONSTR_A_0010</span></b>](#E_SIN_FLOAT_CONSTR_A_0010) on tensor $A$.
