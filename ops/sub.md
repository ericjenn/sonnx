# Contents

- **Sub** operator for type [real](#real)
- **Sub** operator for types [`float16`, `float`, `double`](#float)
- **Sub** operator for types [`int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`](#int)

Based on ONNX documentation [Sub version 14](https://onnx.ai/onnx/operators/onnx__Sub.html#sub-14).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **Sub** (real, real)

## Signature

$C = \textbf{Sub}(A, B)$

where
- $A$: value from which $B$ is subtracted
- $B$: value subtracted from $A$
- $C$: result of the element-wise subtraction of $B$ from $A$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Sub** operator.

## Function

<a id="E_SUB_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_SUB_REAL_FUNC_0010]</br></span>

Operator **Sub** subtracts tensor $B$ from tensor $A$ element-wise and stores the result in tensor $C$. Each element of $C$ is the difference of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.

The mathematical definition of the operator is given hereafter.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $C$:

$$
C[i] = \tilde{A}[i] - \tilde{B}[i]
$$

where
- $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$,
- $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$.

The subtraction is exact: $C[i]$ is the difference in $\mathbb{R}$ of the two elements, with no approximation.

**Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the difference of the two operands and $C$ has the shape of the other operand.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
A = \begin{bmatrix} 6.1 & 9.5 & 35.7 \end{bmatrix}
$$

$$
B = \begin{bmatrix} 3 & 3.3 & 5.1 \end{bmatrix}
$$

$$
C = A - B = \begin{bmatrix} 6.1-3 & 9.5-3.3 & 35.7-5.1 \end{bmatrix} = \begin{bmatrix} 3.1 & 6.2 & 30.6 \end{bmatrix}
$$

### Example 2

$$
A = \begin{bmatrix}
  3.7 & 4.4 \\
  16.2 & 0.5 \\
  25.3 & 24.8
\end{bmatrix}
\quad
B = \begin{bmatrix} 1 & 2 \end{bmatrix}
$$

$$
C = A - B = \begin{bmatrix}
  3.7-1 & 4.4-2 \\
  16.2-1 & 0.5-2 \\
  25.3-1 & 24.8-2
\end{bmatrix} = \begin{bmatrix}
  2.7 & 2.4 \\
  15.2 & -1.5 \\
  24.3 & 22.8
\end{bmatrix}
$$

### Example 3

$$
A = \begin{bmatrix} 5 & 3.25 \\ 4 & 1.75 \end{bmatrix}
\quad
B = 2.5
$$

$$
C = A - B = \begin{bmatrix} 2.5 & 0.75 \\ 1.5 & -0.75 \end{bmatrix}
$$

## Error conditions

None of the conditions of the list applies to real numbers: the subtraction is exact, so that the result is the one specified by [<b><span style="font-family: 'Courier New', monospace">E_SUB_REAL_FUNC_0010</span></b>](#E_SUB_REAL_FUNC_0010) for every pair of operands, and no error can occur. No error condition.

## Attributes

Operator **Sub** has no attribute.

## Inputs

### $\text{A}$: real tensor

The value from which $B$ is subtracted.

#### Constraints

<a id="E_SUB_REAL_CONSTR_A_0010"></a>
- `[E_SUB_REAL_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.

### $\text{B}$: real tensor

The value subtracted from $A$.

#### Constraints

- `[E_SUB_REAL_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_REAL_CONSTR_A_0010</span></b>](#E_SUB_REAL_CONSTR_A_0010) on tensor $A$.

## Outputs

### $\text{C}$: real tensor

Tensor $C$ is the element-wise result of the subtraction of $B$ from $A$.

#### Constraints

- `[E_SUB_REAL_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.

<a id="float"></a>

# **Sub** (float, float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the same subtraction for all of them, and they differ only by their precision and by the range of the values they represent.

## Signature

$C = \textbf{Sub}(A, B)$

where
- $A$: value from which $B$ is subtracted
- $B$: value subtracted from $A$
- $C$: result of the element-wise subtraction of $B$ from $A$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Sub** operator.

## Function

<a id="E_SUB_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_SUB_FLOAT_FUNC_0010]</br></span>

Operator **Sub** subtracts tensor $B$ from tensor $A$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $C$. Each element of $C$ is the difference of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.

The mathematical definition of the operator is given hereafter.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $C$:

$$
C[i] =
\begin{cases}
\text{NaN} & \text{if } \tilde{A}[i] \text{ or } \tilde{B}[i] \text{ is NaN, or if they are both } +\infty \text{ or both } -\infty \\
\tilde{A}[i] - \tilde{B}[i] & \text{if the exact difference is representable in the type of } C \\
\text{round}(\tilde{A}[i] - \tilde{B}[i]) & \text{otherwise}
\end{cases}
$$

where
- $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
- $\text{round}(x)$ is the value of $x$ rounded to the nearest value of the type of $C$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded: a magnitude whose nearest value exceeds the largest finite value of the type of $C$ gives an infinity of the sign of $x$.

In the first case, an operand that is NaN is propagated, and the subtraction of two infinities of the same sign is the invalid operation defined in IEEE 754 section 7.2; the result is NaN in both cases. The NaN of this case is a quiet NaN; its sign and its payload are not specified, whether the NaN comes from an operand or from the invalid operation, so that every quiet NaN of the type of $C$ is a conforming result.

In the last case, the exact difference is not representable in the type of $C$: the result is that difference rounded as defined above. The rounding may make the result subnormal; its sign is then the sign of the exact difference.

When the exact difference is null, the result is $+0$, except when $\tilde{A}[i]$ is $-0$ and $\tilde{B}[i]$ is $+0$, in which case it is $-0$.

**Values of the operands.** The operands may be any of the values of the type, including the special numbers $\pm 0$, $\pm\text{inf}$ and NaN. The cases above give the result for every combination of them. The subtraction of two infinities of the same sign yields NaN; the subtraction of an infinity and a finite value, and the subtraction of two infinities of opposite signs, give that infinity without rounding.

**Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the difference of the two operands and $C$ has the shape of the other operand.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

> [!warning] V1 · The overflow threshold
> The definition had a fourth case, "$\pm\text{inf}$ if the exact difference overflows the range of the
> type of $C$", and the Error-conditions table repeated that reading. IEEE 754 calls a result an
> overflow only when the value rounded with an unbounded exponent range exceeds the largest
> finite value of the type, so for an exact difference in the band
> $(\text{max}, \text{max} + \text{ulp(max)}/2)$ the document demanded $\pm\text{inf}$ where
> IEEE 754, NumPy and ONNX Runtime produce `max`: for `float16`, `65504 - (-8)` has the exact
> difference `65512`, which lies in the band and gives `65504` (`0x7bff`), not `inf`. A
> conforming implementation of the document had to return a result IEEE 754 forbids.
> **Now** the case is gone and the infinities follow from the rounding itself: `round(x)` is
> defined with an unbounded exponent range. The threshold that produces is the midpoint
> between the largest finite value and the next power of two, because that midpoint is a tie
> and `roundTiesToEven` breaks it upwards; the two rules are one rule.

> [!warning] V2 · The NaN of the invalid operation
> The document said the result is NaN and stopped there. Which NaN is not fixed by IEEE 754,
> and implementations differ: ONNX Runtime returns `0xfe00`, `0xffc00000` and
> `0xfff8000000000000` for `inf - inf` — the canonical quiet NaN with the sign bit set — where
> a straightforward implementation returns the positive one. Payloads differ as well: ONNX
> Runtime keeps the payload of a NaN operand in `float32` and `float64`, but canonicalises it
> in `float16` (`0x7e01` becomes `0x7e00`).
> **Now** the first case states that every quiet NaN of the type is a conforming result, for a
> NaN operand as for the invalid operation. The permissive rule is also the only workable one:
> a stricter rule would make ONNX Runtime non-conforming for `float16`.

> [!note] V3 · A null result from rounding
> The last case said the rounding "may make the result null or subnormal". It cannot be null:
> the values of a type are all multiples of that type's smallest subnormal, so the difference
> of two of them is too — a non-null difference is at least one smallest subnormal, and a null
> difference is decided by the signed-zero rule below it.
> **Now** the sentence says "subnormal".

The effect of the operator is illustrated on the following examples.

### Example 1

$$
A = \begin{bmatrix} 3.0 & 4.5 \\ 16.0 & 1.0 \\ 25.5 & 24.25 \end{bmatrix}
\quad
B = \begin{bmatrix} 3.0 & 2.0 \\ 4.0 & 0.0 \\ 5.0 & 4.0 \end{bmatrix}
$$

$$
C = A - B = \begin{bmatrix} 0.0 & 2.5 \\ 12.0 & 1.0 \\ 20.5 & 20.25 \end{bmatrix}
$$

The difference of two numbers of a `float` type is not always exact; in this example every difference is representable and $C$ is the exact result.

### Example 2

$$
A = \begin{bmatrix} 1.0 & \text{+inf} & \text{-0} \end{bmatrix}
\quad
B = \begin{bmatrix} 1.0 & \text{+inf} & \text{+0} \end{bmatrix}
$$

$$
C = \begin{bmatrix} 0.0 & \text{NaN} & \text{-0} \end{bmatrix}
$$

The subtraction of the two infinities is the invalid operation of IEEE 754 section 7.2 and yields NaN.

### Example 3

$$
A = \begin{bmatrix} 0.3 & 1.0 \end{bmatrix}
\quad
B = \begin{bmatrix} 0.1 & 1.0 \end{bmatrix}
$$

$$
C \approx \begin{bmatrix} 0.19999999999999998 & 0.0 \end{bmatrix}
$$

The exact difference of the two `double` values $0.3$ and $0.1$ is not representable in `double`; the first element of $C$ is that difference rounded to the nearest `double`.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e., the subtraction of two infinities of the same sign | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_FUNC_0010</span></b>](#E_SUB_FLOAT_FUNC_0010), the result is NaN |
| Overflow, i.e., an exact difference that rounds beyond the largest finite value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_FUNC_0010</span></b>](#E_SUB_FLOAT_FUNC_0010), the result is $\pm\text{inf}$ |
| Underflow, i.e., an exact difference with a magnitude below the smallest normal value of the type | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_FUNC_0010</span></b>](#E_SUB_FLOAT_FUNC_0010), the result is that difference, which is a subnormal value |

Every condition of the list is part of the nominal behavior of the operator; none of them is an error. No error condition.

> [!note] V4 · Two rows of the table above
> The Overflow row named the *exact* difference — "an exact difference outside the range of
> the type" — which is the reading V1 corrects; it now names the rounded result. The Underflow
> row promised a result "possibly $\pm 0$", which V3 rules out for the same reason, and it
> named the smallest subnormal where IEEE 754 sets underflow at the smallest normal value;
> with a magnitude below that value the exact difference is always representable, as a
> subnormal. The integer section keeps its own wording: there the range of the type does
> decide the case, and the wrap-around is nominal.

## Attributes

Operator **Sub** has no attribute.

## Inputs

### $\text{A}$: floating-point tensor

The value from which $B$ is subtracted.

#### Constraints

<a id="E_SUB_FLOAT_CONSTR_A_0010"></a>
- `[E_SUB_FLOAT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_SUB_FLOAT_CONSTR_A_0020"></a>
- `[E_SUB_FLOAT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$, and $C$ have the same type.

### $\text{B}$: floating-point tensor

The value subtracted from $A$.

#### Constraints

- `[E_SUB_FLOAT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_CONSTR_A_0010</span></b>](#E_SUB_FLOAT_CONSTR_A_0010) on tensor $A$.
- `[E_SUB_FLOAT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_CONSTR_A_0020</span></b>](#E_SUB_FLOAT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: floating-point tensor

Tensor $C$ is the element-wise result of the subtraction of $B$ from $A$.

#### Constraints

- `[E_SUB_FLOAT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_SUB_FLOAT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_FLOAT_CONSTR_A_0020</span></b>](#E_SUB_FLOAT_CONSTR_A_0020) on tensor $A$.

<a id="int"></a>

# **Sub** (int, int)

where int is in {`int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`}.

The eight types share one semantics: subtraction is defined modulo $2^n$ for all of them, $n$ being the number of bits of the type. The signed and the unsigned types differ only by the value that represents the residue class of the result.

## Signature

$C = \textbf{Sub}(A, B)$

where
- $A$: value from which $B$ is subtracted
- $B$: value subtracted from $A$
- $C$: result of the element-wise subtraction of $B$ from $A$

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

No specific restrictions apply to the **Sub** operator.

## Function

<a id="E_SUB_INT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_SUB_INT_FUNC_0010]</br></span>

Operator **Sub** subtracts tensor $B$ from tensor $A$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the difference of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.

The mathematical definition of the operator is given hereafter.

For any [tensor index](./../common/definitions.md#tensor_index) $i$ of the result $C$:

$$
C[i] =
\begin{cases}
\tilde{A}[i] - \tilde{B}[i] & \text{if the difference is representable in the type of } C \\
\tilde{A}[i] - \tilde{B}[i] + 2^n & \text{if the difference is lower than the minimum value of the type of } C \\
\tilde{A}[i] - \tilde{B}[i] - 2^n & \text{if the difference is greater than the maximum value of the type of } C
\end{cases}
$$

where
- $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
- $n$ is the number of bits of the type of $A$, $B$ and $C$.

The result is thus the exact difference modulo $2^n$. The difference of two $n$-bit values lies within $2^n-1$ of the range of the type, so that one of the three cases above always applies.

**Signed and unsigned integer types.** For the unsigned types the result lies in $[0, 2^n-1]$, and for the signed types it lies in $[-2^{n-1}, 2^{n-1}-1]$. The two families differ only by the value chosen in the residue class modulo $2^n$: the subtraction of the `uint8` values $0$ and $1$ gives $255$, and the subtraction of the `int8` values $127$ and $-1$ gives $-128$.

**Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the difference of the two operands and $C$ has the shape of the other operand.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
A = \begin{bmatrix} 6 & 5 & -35 \end{bmatrix}
\quad
B = \begin{bmatrix} 3 & 3 & 3 \end{bmatrix}
$$

$$
C = \begin{bmatrix} 3 & 2 & -38 \end{bmatrix}
$$

### Example 2

For the `int8` type:

$$
A = \begin{bmatrix} 127 & -128 & 0 \end{bmatrix}
\quad
B = \begin{bmatrix} -1 & 1 & 1 \end{bmatrix}
$$

$$
C = \begin{bmatrix} -128 & 127 & -1 \end{bmatrix}
$$

The difference $127-(-1)=128$ is greater than the maximum value of `int8`, so the third case of the definition applies and the result is $128-2^8=-128$. The difference $-128-1=-129$ is lower than its minimum value, so the second case applies and the result is $-129+2^8=127$.

### Example 3

For the `uint8` type:

$$
A = \begin{bmatrix} 0 & 255 & 4 \end{bmatrix}
\quad
B = \begin{bmatrix} 1 & 1 & 7 \end{bmatrix}
$$

$$
C = \begin{bmatrix} 255 & 254 & 253 \end{bmatrix}
$$

The differences $0-1=-1$ and $4-7=-3$ are lower than the minimum value of `uint8`, so the second case of the definition applies and the results are $-1+2^8=255$ and $-3+2^8=253$. The difference $255-1=254$ is representable in `uint8` and is given by the first case.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Overflow, i.e., an exact difference outside the range of the type, such as $127-(-1)=128$ for `int8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_SUB_INT_FUNC_0010</span></b>](#E_SUB_INT_FUNC_0010), the result is the difference modulo $2^n$ |

The only condition of the list that applies to the integer types is the overflow, and it is part of the nominal behavior of the operator; there is no error. No error condition.

## Attributes

Operator **Sub** has no attribute.

## Inputs

### $\text{A}$: integer tensor

The value from which $B$ is subtracted.

#### Constraints

<a id="E_SUB_INT_CONSTR_A_0010"></a>
- `[E_SUB_INT_CONSTR_A_0010]` Shape compatibility
  - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
<a id="E_SUB_INT_CONSTR_A_0020"></a>
- `[E_SUB_INT_CONSTR_A_0020]` Type consistency
  - Statement: Tensors $A$, $B$, and $C$ have the same type.

### $\text{B}$: integer tensor

The value subtracted from $A$.

#### Constraints

- `[E_SUB_INT_CONSTR_B_0010]` Shape compatibility
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_INT_CONSTR_A_0010</span></b>](#E_SUB_INT_CONSTR_A_0010) on tensor $A$.
- `[E_SUB_INT_CONSTR_B_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_INT_CONSTR_A_0020</span></b>](#E_SUB_INT_CONSTR_A_0020) on tensor $A$.

## Outputs

### $\text{C}$: integer tensor

Tensor $C$ is the element-wise result of the subtraction of $B$ from $A$.

#### Constraints

- `[E_SUB_INT_CONSTR_C_0010]` Shape definition
  - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
- `[E_SUB_INT_CONSTR_C_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SUB_INT_CONSTR_A_0020</span></b>](#E_SUB_INT_CONSTR_A_0020) on tensor $A$.
