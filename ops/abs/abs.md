# Contents

- **Abs** operator for type [real](#real)
- **Abs** operator for types [`float16`, `float`, `double`](#float)
- **Abs** operator for types [`int8`, `int16`, `int32`, `int64`](#int)
- **Abs** operator for types [`uint8`, `uint16`, `uint32`, `uint64`](#uint)

Based on ONNX documentation [Abs version 13](https://onnx.ai/onnx/operators/onnx__Abs.html#abs-13).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 13.

<a id="real"></a>

# **Abs** (real)

## Signature

$$Y = \textbf{Abs}(X)$$

where
- $$X$$: tensor whose absolute value is computed
- $$Y$$: tensor whose elements are the absolute values of the elements of $$X$$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Abs** operator.

## Function

<a id="E_ABS_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ABS_REAL_FUNC_0010]</br></span>

Operator **Abs** computes the absolute value of tensor $$X$$ element-wise and stores the result in tensor $$Y$$.

For any [tensor index](../../common/definitions.md#tensor_index) $$i$$ of the result $$Y$$:

$$
Y[i] = |X[i]| =
\begin{cases}
X[i] & \text{if } X[i] \ge 0 \\
-X[i] & \text{if } X[i] < 0
\end{cases}
$$

where
- $$|X[i]|$$ is the absolute value in $$\mathbb{R}$$ of $$X[i]$$,
- $$-X[i]$$ is the negation in $$\mathbb{R}$$ of $$X[i]$$.

The result is exact: the absolute value of a real number is a real number, and the operator introduces no approximation.

**The shape of the result.** Tensor $$Y$$ has the shape of $$X$$, and the element $$Y[i]$$ is the absolute value of the element of $$X$$ at the same index $$i$$. The operator combines no two elements, so that no shape compatibility condition arises and $$X$$ has no shape constraint other than being a tensor.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) has a single element. The result is a rank-0 tensor whose single element is the absolute value of that element.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $$Y$$ then has size $$0$$ as well, and $$Y$$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

$$
X = \begin{bmatrix} 3.5 & -2 & 0 & -7.25 \end{bmatrix}
\quad
Y = \begin{bmatrix} 3.5 & 2 & 0 & 7.25 \end{bmatrix}
$$

The third element is its own absolute value, and the other three are the operands of negative sign with that sign removed.

## Error conditions

The absolute value of a real number is defined for every real number and is exact, so that no condition of the list applies: the operator performs no arithmetic operation, hence raises no invalid operation and no overflow, and it performs no division, hence raises no division by zero. No error condition.

## Attributes

Operator **Abs** has no attribute.

## Inputs

### $\text{X}$: real tensor

The tensor whose absolute value is computed.

#### Constraints

<a id="E_ABS_REAL_CONSTR_X_0010"></a>
- `[E_ABS_REAL_CONSTR_X_0010]` Type consistency
  - Statement: Tensors $$X$$ and $$Y$$ have the same type.

## Outputs

### $\text{Y}$: real tensor

The tensor whose elements are the absolute values of the elements of $$X$$.

#### Constraints

- `[E_ABS_REAL_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $$Y$$ has the shape of $$X$$.
- `[E_ABS_REAL_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ABS_REAL_CONSTR_X_0010</span></b>](#E_ABS_REAL_CONSTR_X_0010) on tensor $$X$$.

<a id="float"></a>

# **Abs** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the IEEE 754 standard defines the absolute value of all of them as the operand with its sign bit cleared, and they differ only by their precision and by the range of the values they represent.

## Signature

$$Y = \textbf{Abs}(X)$$

where
- $$X$$: tensor whose absolute value is computed
- $$Y$$: tensor whose elements are the absolute values of the elements of $$X$$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Abs** operator.

## Function

<a id="E_ABS_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ABS_FLOAT_FUNC_0010]</br></span>

Operator **Abs** computes the absolute value of tensor $$X$$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $$Y$$. Each element of $$Y$$ is the absolute value of the element of $$X$$ at the same index.

For any [tensor index](../../common/definitions.md#tensor_index) $$i$$ of the result $$Y$$:

$$
Y[i] = |X[i]| =
\begin{cases}
\text{qNaN} & \text{if } X[i] \text{ is a NaN} \\
+\infty & \text{if } X[i] \text{ is } +\infty \text{ or } -\infty \\
+0 & \text{if } X[i] \text{ is } +0 \text{ or } -0 \\
X[i] & \text{if } X[i] \text{ is finite and } X[i] > 0 \\
-X[i] & \text{if } X[i] \text{ is finite and } X[i] < 0
\end{cases}
$$

where
- $$|X[i]|$$ is the absolute value of $$X[i]$$ in the type of $$X$$, i.e. the operand with its sign bit cleared,
- $$-X[i]$$ is the negation in $$\mathbb{R}$$ of the negative value $$X[i]$$, whose sign is positive,
- $$\text{qNaN}$$ is any quiet NaN of the type of $$X$$: its sign and its payload are not specified, so that every quiet NaN of the type is a conforming result.

**No rounding, no overflow, no underflow.** The set of the finite values of each of the three types is symmetric about zero: the largest finite value is the absolute value of the smallest one, and every subnormal value has a subnormal absolute value. The absolute value of a finite element of $$X$$ is therefore a finite value of the type of $$X$$, exact, and the operator introduces neither a rounding nor an overflow nor an underflow, whatever the exponent of the operand. The operator performs no arithmetic operation and raises no invalid operation: even the absolute value of the smallest finite value and that of the largest finite value are exact.

**The zeros.** The absolute value of $$+0$$ is $$+0$$ and the absolute value of $$-0$$ is $$+0$$: the sign of a null result is positive, the operand being null.

**The infinities.** The absolute value of $$+\infty$$ and the absolute value of $$-\infty$$ are both $$+\infty$$.

**The NaNs.** A NaN operand gives a quiet NaN result. Its sign and its payload are not specified, so that every quiet NaN of the type of $$X$$ is a conforming result, whether the operand NaN is itself quiet or signalling.

**The shape of the result.** Tensor $$Y$$ has the shape of $$X$$, and the element $$Y[i]$$ depends on the element $$X[i]$$ alone. The operator combines no two elements, so that no shape compatibility condition arises.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) has a single element. The result is a rank-0 tensor whose single element is the absolute value of that element.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $$Y$$ then has size $$0$$ as well, and $$Y$$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix} -1.5 & 2.25 & 0.0 & -0.0 & -\text{inf} & +\text{inf} & \text{NaN} \end{bmatrix}
$$

$$
Y = \begin{bmatrix} 1.5 & 2.25 & +0 & +0 & +\text{inf} & +\text{inf} & \text{NaN} \end{bmatrix}
$$

The first two elements are a negative and a positive finite value; they give the same magnitude, the sign being cleared in both cases. The third and the fourth elements are the two zeros, both of which give $$+0$$. The fifth and the sixth elements are the two infinities, both of which give $$+\infty$$. The last element is a NaN, which gives a NaN whose sign and payload are not specified.

### Example 2

$$
X = \begin{bmatrix} -3.0 & -1.0 \\ -2.5 & 4.0 \end{bmatrix}
\quad
Y = \begin{bmatrix} 3.0 & 1.0 \\ 2.5 & 4.0 \end{bmatrix}
$$

The elements of $$Y$$ are the elements of $$X$$ with the sign cleared, each at the index it occupies in $$X$$; the shape of $$Y$$ is the shape of $$X$$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, as defined in IEEE 754 section 7.2 | not raised: the operator performs no arithmetic operation, and the result of a NaN operand is stated by [<b><span style="font-family: 'Courier New', monospace">E_ABS_FLOAT_FUNC_0010</span></b>](#E_ABS_FLOAT_FUNC_0010) to be a NaN |
| A NaN operand, and the NaN result it produces | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ABS_FLOAT_FUNC_0010</span></b>](#E_ABS_FLOAT_FUNC_0010), the result is a quiet NaN whose sign and payload are not specified |
| Overflow, i.e. a computation leading to $$+\infty$$ or $$-\infty$$ | not raised: the absolute value of a finite element is a finite value of the type, as stated by [<b><span style="font-family: 'Courier New', monospace">E_ABS_FLOAT_FUNC_0010</span></b>](#E_ABS_FLOAT_FUNC_0010) |
| Underflow, i.e. a magnitude below the smallest normal value of the type | nominal: the absolute value of a subnormal element is that element, as stated by [<b><span style="font-family: 'Courier New', monospace">E_ABS_FLOAT_FUNC_0010</span></b>](#E_ABS_FLOAT_FUNC_0010) |
| Division by zero | does not apply: the operator performs no division |

Every condition of the list is either part of the nominal behavior of the operator or absent from it; none of them is an error. No error condition.

## Attributes

Operator **Abs** has no attribute.

## Inputs

### $\text{X}$: floating-point tensor

The tensor whose absolute value is computed.

#### Constraints

<a id="E_ABS_FLOAT_CONSTR_X_0010"></a>
- `[E_ABS_FLOAT_CONSTR_X_0010]` Type consistency
  - Statement: Tensors $$X$$ and $$Y$$ have the same type.

## Outputs

### $\text{Y}$: floating-point tensor

The tensor whose elements are the absolute values of the elements of $$X$$.

#### Constraints

- `[E_ABS_FLOAT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $$Y$$ has the shape of $$X$$.
- `[E_ABS_FLOAT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ABS_FLOAT_CONSTR_X_0010</span></b>](#E_ABS_FLOAT_CONSTR_X_0010) on tensor $$X$$.

<a id="int"></a>

# **Abs** (int)

where int is in {`int8`, `int16`, `int32`, `int64`}.

The four types share one semantics: each of them is an $$n$$-bit signed type, whose values range from $$-2^{n-1}$$ to $$2^{n-1}-1$$, and the result of the absolute value is the value of that type that represents the exact absolute value modulo $$2^n$$.

## Signature

$$Y = \textbf{Abs}(X)$$

where
- $$X$$: tensor whose absolute value is computed
- $$Y$$: tensor whose elements are the absolute values of the elements of $$X$$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Abs** operator.

## Function

<a id="E_ABS_INT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ABS_INT_FUNC_0010]</br></span>

Operator **Abs** computes the absolute value of tensor $$X$$ element-wise and stores the result in tensor $$Y$$. Tensors $$X$$ and $$Y$$ have the same signed $$n$$-bit type, and each element of $$Y$$ is a value of that type.

For any [tensor index](../../common/definitions.md#tensor_index) $$i$$ of the result $$Y$$:

$$
Y[i] =
\begin{cases}
X[i] & \text{if } X[i] \ge 0 \\
-X[i] & \text{if } -2^{n-1} < X[i] < 0 \\
-2^{n-1} & \text{if } X[i] = -2^{n-1}
\end{cases}
$$

where
- $$n$$ is the number of bits of the type of $$X$$,
- $$-X[i]$$ is the negation in $$\mathbb{Z}$$ of $$X[i]$$, before any reduction; the second case is the only one in which that negation is used, and the result then lies in $$\left[1, 2^{n-1}-1\right]$$, which is within the range of the type,
- the third case is the value $$-X[i] = 2^{n-1}$$ reduced modulo $$2^n$$ to the type of $$X$$, that is $$2^{n-1} - 2^n = -2^{n-1}$$.

**The minimum value of the type.** The values of an $$n$$-bit signed type are not symmetric about zero: the range is $$\left[-2^{n-1}, 2^{n-1}-1\right]$$, so that the exact absolute value of the minimum value $$-2^{n-1}$$ is $$2^{n-1}$$, which is not a value of the type. The operator maps that value to itself: tensors $$X$$ and $$Y$$ have the same type, no wider type is used, and the result is the value of the type congruent to $$2^{n-1}$$ modulo $$2^n$$, which is $$-2^{n-1}$$. For `int8`, the absolute value of $$-128$$ is $$-128$$. Every other value of the type has an absolute value that lies in the range of the type, so that the third case is the only one whose result leaves the range of the operand, and it is the only value of the type mapped to a value of the opposite sign.

**The zero.** The absolute value of $$0$$ is $$0$$, given by the first case.

**The shape of the result.** Tensor $$Y$$ has the shape of $$X$$, and the element $$Y[i]$$ depends on the element $$X[i]$$ alone. The operator combines no two elements, so that no shape compatibility condition arises.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) has a single element. The result is a rank-0 tensor whose single element is the absolute value of that element.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $$Y$$ then has size $$0$$ as well, and $$Y$$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `int8` type:

$$
X = \begin{bmatrix} -6 & 100 & 127 & 0 & -127 & -128 \end{bmatrix}
$$

$$
Y = \begin{bmatrix} 6 & 100 & 127 & 0 & 127 & -128 \end{bmatrix}
$$

The first two elements are the negative and the positive value of the same magnitude, which give the same result. The third and the fourth elements are non-negative and are their own absolute values. The fifth element is $$-127$$, whose absolute value $$127$$ is the maximum value of `int8`. The last element is the minimum value $$-2^7 = -128$$, whose exact absolute value $$128$$ is outside the range of `int8`: the result is the value of the type congruent to $$128$$ modulo $$2^8$$, that is $$128 - 256 = -128$$, so that the last element of $$Y$$ is negative and equal to the element of $$X$$ at the same index.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Overflow, i.e. an exact absolute value outside the range of the type, which happens for the minimum value $$-2^{n-1}$$ alone, such as $$-128$$ for `int8` | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_ABS_INT_FUNC_0010</span></b>](#E_ABS_INT_FUNC_0010), the result is $$-2^{n-1}$$, the value of the type congruent to $$2^{n-1}$$ modulo $$2^n$$ |
| Division by zero | does not apply: the operator performs no division |

The only condition of the list that applies to the signed integer types is the overflow of the absolute value of the minimum value of the type, and it is part of the nominal behavior of the operator; there is no error. No error condition.

## Attributes

Operator **Abs** has no attribute.

## Inputs

### $\text{X}$: signed integer tensor

The tensor whose absolute value is computed.

#### Constraints

<a id="E_ABS_INT_CONSTR_X_0010"></a>
- `[E_ABS_INT_CONSTR_X_0010]` Type consistency
  - Statement: Tensors $$X$$ and $$Y$$ have the same type.

## Outputs

### $\text{Y}$: signed integer tensor

The tensor whose elements are the absolute values of the elements of $$X$$.

#### Constraints

- `[E_ABS_INT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $$Y$$ has the shape of $$X$$.
- `[E_ABS_INT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ABS_INT_CONSTR_X_0010</span></b>](#E_ABS_INT_CONSTR_X_0010) on tensor $$X$$.

<a id="uint"></a>

# **Abs** (uint)

where uint is in {`uint8`, `uint16`, `uint32`, `uint64`}.

The four types share one semantics: each of them is an $$n$$-bit unsigned type, whose values range from $$0$$ to $$2^n-1$$ and are all non-negative, so that the absolute value of every value is the value itself.

## Signature

$$Y = \textbf{Abs}(X)$$

where
- $$X$$: tensor whose absolute value is computed
- $$Y$$: tensor whose elements are the absolute values of the elements of $$X$$

## Restrictions

[General restrictions](../global_restrictions.md) are applicable.

No specific restrictions apply to the **Abs** operator.

## Function

<a id="E_ABS_UINT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_ABS_UINT_FUNC_0010]</br></span>

Operator **Abs** computes the absolute value of tensor $$X$$ element-wise and stores the result in tensor $$Y$$. Tensors $$X$$ and $$Y$$ have the same unsigned $$n$$-bit type, and each element of $$Y$$ is a value of that type.

For any [tensor index](../../common/definitions.md#tensor_index) $$i$$ of the result $$Y$$:

$$
Y[i] = X[i]
$$

where the elements of $$X$$ lie in $$\left[0, 2^n-1\right]$$ and $$n$$ is the number of bits of the type of $$X$$.

**The operator is the identity on the unsigned types.** Every value of an $$n$$-bit unsigned type is non-negative, so that its absolute value is the value itself, and no element of $$X$$ is changed: $$Y$$ has the same shape, the same type and the same contents as $$X$$. No result can leave the range of the type, no rounding takes place, and no value of the type is mapped to a value of the opposite sign.

**The shape of the result.** Tensor $$Y$$ has the shape of $$X$$, and the element $$Y[i]$$ depends on the element $$X[i]$$ alone. The operator combines no two elements, so that no shape compatibility condition arises.

**Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) has a single element. The result is a rank-0 tensor whose single element is that element.

**Zero-sized dimensions.** An operand may have a zero-sized dimension. The corresponding dimension of $$Y$$ then has size $$0$$ as well, and $$Y$$ is empty.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `uint8` type:

$$
X = \begin{bmatrix} 0 & 6 & 200 & 255 \end{bmatrix}
\quad
Y = \begin{bmatrix} 0 & 6 & 200 & 255 \end{bmatrix}
$$

Every element of $$X$$ is non-negative, so that every element of $$Y$$ is the element of $$X$$ at the same index, including the minimum value $$0$$ and the maximum value $$255$$ of `uint8`.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Overflow, i.e. an exact absolute value outside the range of the type | not possible: every element is non-negative and is its own absolute value, as stated by [<b><span style="font-family: 'Courier New', monospace">E_ABS_UINT_FUNC_0010</span></b>](#E_ABS_UINT_FUNC_0010) |
| Division by zero | does not apply: the operator performs no division |

None of the conditions of the list can occur for the unsigned integer types, and the operator has no error condition of its own. No error condition.

## Attributes

Operator **Abs** has no attribute.

## Inputs

### $\text{X}$: unsigned integer tensor

The tensor whose absolute value is computed.

#### Constraints

<a id="E_ABS_UINT_CONSTR_X_0010"></a>
- `[E_ABS_UINT_CONSTR_X_0010]` Type consistency
  - Statement: Tensors $$X$$ and $$Y$$ have the same type.

## Outputs

### $\text{Y}$: unsigned integer tensor

The tensor whose elements are the absolute values of the elements of $$X$$.

#### Constraints

- `[E_ABS_UINT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $$Y$$ has the shape of $$X$$.
- `[E_ABS_UINT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ABS_UINT_CONSTR_X_0010</span></b>](#E_ABS_UINT_CONSTR_X_0010) on tensor $$X$$.
