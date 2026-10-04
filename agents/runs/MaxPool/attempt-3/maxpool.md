# Contents

- **MaxPool** operator for type [real](#real)
- **MaxPool** operator for types [`float16`, `float`, `double`](#float)
- **MaxPool** operator for types [`int8`, `int64`](#int)
- **MaxPool** operator for types [`uint8`](#uint)

Based on ONNX documentation [MaxPool version 12](https://onnx.ai/onnx/operators/onnx__MaxPool.html#maxpool-12).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **MaxPool** (real)

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \ge 3$, whose shape is $(dX_0, dX_1, dX_2, ..., dX_{rX-1})$
- $Y$: output tensor, whose shape is $(dY_0, dY_1, dY_2, ..., dY_{rX-1})$
- $\text{Indices}$: optional output tensor of the same shape as $Y$, giving for each element of $Y$ the index of the element of $X$ it is the maximum of

The first two axes of $X$ are the batch axis and the channel axis; the remaining $rX-2$ axes are the spatial axes. The operator is applied independently on each batch and channel, and the pooling window slides along the spatial axes.

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `ceil_mode` is restricted to `0` | Transient |
| `[R4]` | The padding values of attribute `pads` are restricted to values such that no pooling window lies entirely in the padding | Transient |

## Function

<a id="E_MAXPOOL_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_REAL_FUNC_0010]</br></span>

**Part 1.** Operator **MaxPool** applies a max pooling over the spatial axes of the input tensor $X$: the spatial axes are partitioned into windows of the size given by the `kernel_shape` attribute, the windows being placed at the positions given by the `strides` attribute, and each element of the output tensor $Y$ is the maximum of the elements of $X$ that lie in the corresponding window, the elements lying in the padding being excluded.

**Part 2.** Let $s = rX - 2$ be the number of spatial axes of $X$. For every batch index $b \in [0, dX_0-1]$, every channel index $c \in [0, dX_1-1]$, and every spatial index $(j_0, ..., j_{s-1})$ of the output tensor $Y$:

$$
Y[b, c, j_0, ..., j_{s-1}] = \max_{k_0, ..., k_{s-1}} X_p[b, c, i_0, ..., i_{s-1}]
$$

where
- the maximum is taken over every tuple $(k_0, ..., k_{s-1})$ such that $0 \le k_t \le dK_t - 1$ for every $t \in [0, s-1]$, and such that the index $i_t = j_t \cdot dS_t - dP_t + k_t \cdot dD_t$ lies in $[0, dX_{t+2}-1]$ for every $t \in [0, s-1]$,
- $X_p$ is the tensor obtained by padding $X$ with the values of the `pads` attribute, so that $X_p[b, c, i_0, ..., i_{s-1}] = X[b, c, i_0, ..., i_{s-1}]$ for every index of $X$,
- $dK_t$ is the value of the `kernel_shape` attribute along spatial axis $t$,
- $dS_t$ is the value of the `strides` attribute along spatial axis $t$,
- $dP_t$ is the value of the `pads` attribute at the beginning of spatial axis $t$,
- $dD_t$ is the value of the `dilations` attribute along spatial axis $t$.

The maximum is taken over the elements of $X$ that lie in the window, the elements of the padding being excluded from the maximum; the restriction `[R4]` ensures that at least one element of $X$ lies in every window, so that the maximum is taken over a non-empty set.

**Output shape.** The size of the output along spatial axis $t$ is

$$
dY_{t+2} = \left\lfloor \frac{dX_{t+2} + dP_t + dQ_t - dD_t \cdot (dK_t - 1) - 1}{dS_t} \right\rfloor + 1
$$

where $dQ_t$ is the value of the `pads` attribute at the end of spatial axis $t$. The batch and channel dimensions are unchanged: $dY_0 = dX_0$ and $dY_1 = dX_1$.

**The `Indices` output.** When the optional output `Indices` is present, each of its elements is the index of the element of $X$ of which the corresponding element of $Y$ is the maximum. The index is the index of that element in the flattening of $X$ in row-major order, so that the element $X[b, c, i_0, ..., i_{s-1}]$ has the index

$$
\text{Indices}[b, c, j_0, ..., j_{s-1}] = \sum_{u=0}^{rX-1} \iota_u \cdot \prod_{v=u+1}^{rX-1} dX_v
$$

where
- $\iota_0 = b$, $\iota_1 = c$, and $\iota_{t+2} = i_t$ for every $t \in [0, s-1]$, so that $(\iota_0, ..., \iota_{rX-1})$ is the index of the element of $X$ of which $Y[b, c, j_0, ..., j_{s-1}]$ is the maximum,
- a product over an empty range of indices is $1$, so that the term of the sum for $u = rX-1$ is $\iota_{rX-1} = i_{s-1}$,
- the index is a value of the type of `Indices`, which is `int64`.

The index does not account for the padding: it is an index into $X$, and its value lies in $[0, dX_0 \cdot dX_1 \cdot dX_2 \cdot ... \cdot dX_{rX-1} - 1]$. When several elements of the window are equal to the maximum, the index given is the smallest of their indices in the flattening of $X$ in row-major order.

**Tensors with no dimension.** The input tensor $X$ has at least three axes, so that a rank-0 tensor is not a valid input; the constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0010</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0010) rules it out.

**Zero-sized dimensions.** A spatial dimension of $X$ may have size $0$. The output shape is then computed by the formula above, and the output is empty along that axis. A batch or channel dimension of size $0$ gives an output of size $0$ along that axis.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
  \end{bmatrix}
\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `pads` $= [0, 0, 0, 0]$, `dilations` $= [1, 1]$:

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    5 & 6 \\
    8 & 9
  \end{bmatrix}
\end{bmatrix}
$$

The output shape is $dY_2 = \lfloor (3 + 0 + 0 - 1 \cdot (2-1) - 1)/1 \rfloor + 1 = 2$ along each spatial axis.

### Example 2

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
  \end{bmatrix}
\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `pads` $= [1, 1, 1, 1]$, `dilations` $= [1, 1]$:

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    1 & 2 & 3 & 3 \\
    4 & 5 & 6 & 6 \\
    7 & 8 & 9 & 9 \\
    7 & 8 & 9 & 9
  \end{bmatrix}
\end{bmatrix}
$$

The output shape is $dY_2 = \lfloor (3 + 1 + 1 - 1 - 1)/1 \rfloor + 1 = 4$ along each spatial axis. The elements of the padding are excluded from the maximum, so that the first element of $Y$ is the maximum of the single element $X[0, 0, 0, 0] = 1$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs no floating-point arithmetic, and the maximum of a set of values is defined for every value of the type |
| Overflow | not applicable: the result is one of the elements of the input, and no arithmetic is performed |
| Integer overflow | not applicable: the result is one of the elements of the input, and no arithmetic is performed |
| Division by zero | not applicable: the operator performs no division |
| A pooling window that lies entirely in the padding | ruled out by the restriction `[R4]` |

No error condition.

## Attributes

### `auto_pad`: `string`

The padding mode. The value is `NOTSET`, `SAME_UPPER`, `SAME_LOWER` or `VALID`; the default is `NOTSET`, which means that the explicit padding of the `pads` attribute is used.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_auto_pad_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_auto_pad_0010]` Value of `auto_pad`
  - Statement: The value of `auto_pad` is `NOTSET`.
  - Rationale: Restriction `[R1]`.

### `ceil_mode`: `int`

Whether the ceiling or the floor is used to compute the output shape. The value is `0` (floor, the default) or `1` (ceiling).

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_ceil_mode_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_ceil_mode_0010]` Value of `ceil_mode`
  - Statement: The value of `ceil_mode` is `0`.
  - Rationale: Restriction `[R3]`.

### `dilations`: `list of int`

The dilation value along each spatial axis of the kernel. The default is $1$ along each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_dilations_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_dilations_0010]` Length of `dilations`
  - Statement: The length of `dilations` is $rX - 2$.
<a id="E_MAXPOOL_REAL_CONSTR_dilations_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_dilations_0020]` Value of `dilations`
  - Statement: Every value of `dilations` is at least $1$.

### `kernel_shape`: `list of int`

The size of the kernel along each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_kernel_shape_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_kernel_shape_0010]` Length of `kernel_shape`
  - Statement: The length of `kernel_shape` is $rX - 2$.
<a id="E_MAXPOOL_REAL_CONSTR_kernel_shape_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_kernel_shape_0020]` Value of `kernel_shape`
  - Statement: Every value of `kernel_shape` is at least $1$.

### `pads`: `list of int`

The padding at the beginning and at the end of each spatial axis. The format is $[x_1^{\text{begin}}, x_2^{\text{begin}}, ..., x_1^{\text{end}}, x_2^{\text{end}}, ...]$, where $x_t^{\text{begin}}$ is the number of elements added at the beginning of spatial axis $t$ and $x_t^{\text{end}}$ the number added at the end. The default is $0$ at the beginning and at the end of each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_pads_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_pads_0010]` Length of `pads`
  - Statement: The length of `pads` is $2 \cdot (rX - 2)$.
<a id="E_MAXPOOL_REAL_CONSTR_pads_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_pads_0020]` Value of `pads`
  - Statement: Every value of `pads` is at least $0$.
<a id="E_MAXPOOL_REAL_CONSTR_pads_0030"></a>
- `[E_MAXPOOL_REAL_CONSTR_pads_0030]` Non-empty windows
  - Statement: For every spatial index $(j_0, ..., j_{s-1})$ of the output tensor $Y$, at least one tuple $(k_0, ..., k_{s-1})$ with $0 \le k_t \le dK_t - 1$ gives an index $i_t = j_t \cdot dS_t - dP_t + k_t \cdot dD_t$ in $[0, dX_{t+2}-1]$ for every $t \in [0, s-1]$.
  - Rationale: Restriction `[R4]`; the maximum of an empty set is not defined.

### `storage_order`: `int`

The storage order of the tensor, used to convert an $n$-tuple index into a single integer for the `Indices` output. The value is `0` (row major, the default) or `1` (column major).

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_storage_order_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_storage_order_0010]` Value of `storage_order`
  - Statement: The value of `storage_order` is `0`.
  - Rationale: Restriction `[R2]`.

### `strides`: `list of int`

The stride along each spatial axis. The default is $1$ along each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_strides_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_strides_0010]` Length of `strides`
  - Statement: The length of `strides` is $rX - 2$.
<a id="E_MAXPOOL_REAL_CONSTR_strides_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_strides_0020]` Value of `strides`
  - Statement: Every value of `strides` is at least $1$.

## Inputs

### $\text{X}$: real tensor

The input data tensor, whose shape is $(dX_0, dX_1, dX_2, ..., dX_{rX-1})$, where $dX_0$ is the batch size, $dX_1$ the number of channels, and $dX_2, ..., dX_{rX-1}$ the spatial dimensions.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_X_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_X_0010]` Rank of $X$
  - Statement: The rank $rX$ of tensor $X$ is at least $3$.

## Outputs

### $\text{Y}$: real tensor

The output data tensor, whose shape is $(dY_0, dY_1, dY_2, ..., dY_{rX-1})$, with $dY_0 = dX_0$, $dY_1 = dX_1$, and $dY_{t+2}$ given by the output shape formula of the "Function" section for every spatial axis $t$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_Y_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the shape defined by the output shape formula of the "Function" section.

### $\text{Indices}$: integer tensor

The optional output tensor, of the same shape as $Y$, giving for each element of $Y$ the index of the element of $X$ it is the maximum of.

#### Constraints

- `[E_MAXPOOL_REAL_CONSTR_Indices_0010]` Shape definition
  - Statement: Tensor `Indices` has the same shape as tensor $Y$.

<a id="float"></a>

# **MaxPool** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the operator selects one of the elements of its input and performs no arithmetic on it, so that the result is the same for all of them, the special values included.

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \ge 3$, whose shape is $(dX_0, dX_1, dX_2, ..., dX_{rX-1})$
- $Y$: output tensor, whose shape is $(dY_0, dY_1, dY_2, ..., dY_{rX-1})$
- $\text{Indices}$: optional output tensor of the same shape as $Y$, giving for each element of $Y$ the index of the element of $X$ it is the maximum of

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

The restrictions of the real-number section apply, and are not repeated here.

## Function

<a id="E_MAXPOOL_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_FLOAT_FUNC_0010]</br></span>

Operator **MaxPool** applies a max pooling over the spatial axes of the input tensor $X$, as specified for real numbers by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0010</span></b>](#E_MAXPOOL_REAL_FUNC_0010). The result is one of the elements of $X$, and no arithmetic is performed on it.

**The order of the values.** The maximum is taken with respect to the order of the values of the type, in which $-\text{inf}$ is the smallest value, $+\text{inf}$ the largest, and NaN is not comparable with any value. The order of the finite values is the order of the real numbers they represent, and $-0$ and $+0$ are equal in that order.

**NaN.** When a window contains a NaN, the result of the maximum is not determined by the order of the values, since NaN is not comparable with any of them. The result is then left to the implementer: every value of the type is a conforming result, and an implementation that propagates the NaN, one that returns the largest non-NaN value of the window, and one that returns any other value of the window are all conforming.

**Signed zeros.** When the maximum of a window is $0$, the result is the null element of the window whose index in the flattening of $X$ in row-major order is the smallest: it is $-0$ when that element is $-0$, and $+0$ when it is $+0$.

**The `Indices` output.** The index given for an element of $Y$ is the index of the element of $X$ of which that element is the maximum, as specified for real numbers by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0010</span></b>](#E_MAXPOOL_REAL_FUNC_0010), tie-break rule included. The elements $-0$ and $+0$ are equal, so that the tie-break rule applies to a window whose maximum is $0$ and which contains both.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    1.0 & \text{+inf} & 3.0 \\
    4.0 & 5.0 & \text{-inf} \\
    7.0 & 8.0 & 9.0
  \end{bmatrix}
\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `pads` $= [0, 0, 0, 0]$, `dilations` $= [1, 1]$:

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    \text{+inf} & \text{+inf} \\
    8.0 & 9.0
  \end{bmatrix}
\end{bmatrix}
$$

The infinity is the largest value of the type, so that a window that contains it has it as its maximum.

### Example 2

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    \text{+0} & \text{-0} & 1.0 \\
    \text{-0} & \text{-0} & 2.0 \\
    3.0 & 4.0 & 5.0
  \end{bmatrix}
\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `pads` $= [0, 0, 0, 0]$, `dilations` $= [1, 1]$:

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    \text{+0} & 1.0 \\
    4.0 & 5.0
  \end{bmatrix}
\end{bmatrix}
$$

The first window contains $+0$ and $-0$, which are equal; the result is $+0$, the first of them in row-major order.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs no floating-point arithmetic, and the maximum of a set of values is defined for every value of the type |
| Overflow | not applicable: the result is one of the elements of the input, and no arithmetic is performed |
| Integer overflow | not applicable: the operator performs no integer arithmetic |
| Division by zero | not applicable: the operator performs no division |
| A window that contains a NaN | left to the implementer: specified by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_FUNC_0010</span></b>](#E_MAXPOOL_FLOAT_FUNC_0010), every value of the type is a conforming result |
| A pooling window that lies entirely in the padding | ruled out by the restriction `[R4]` |

No error condition.

## Attributes

The attributes of the real-number section apply, and are not repeated here.

## Inputs

### $\text{X}$: floating-point tensor

The input data tensor, whose shape is $(dX_0, dX_1, dX_2, ..., dX_{rX-1})$.

#### Constraints

- `[E_MAXPOOL_FLOAT_CONSTR_X_0010]` Rank of $X$
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0010</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0010) on tensor $X$.
<a id="E_MAXPOOL_FLOAT_CONSTR_X_0020"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_X_0020]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: floating-point tensor

The output data tensor, whose shape is $(dY_0, dY_1, dY_2, ..., dY_{rX-1})$.

#### Constraints

- `[E_MAXPOOL_FLOAT_CONSTR_Y_0010]` Shape definition
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_Y_0010</span></b>](#E_MAXPOOL_REAL_CONSTR_Y_0010) on tensor $Y$.
- `[E_MAXPOOL_FLOAT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_CONSTR_X_0020</span></b>](#E_MAXPOOL_FLOAT_CONSTR_X_0020) on tensor $X$.

### $\text{Indices}$: integer tensor

The optional output tensor, of the same shape as $Y$.

#### Constraints

- `[E_MAXPOOL_FLOAT_CONSTR_Indices_0010]` Shape definition
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_Indices_0010</span></b>](#E_MAXPOOL_REAL_CONSTR_Indices_0010) on tensor `Indices`.

<a id="int"></a>

# **MaxPool** (int)

where int is in {`int8`, `int64`}.

The types of this section share one semantics: the operator selects one of the elements of its input and performs no arithmetic on it, so that the result is the same for every type of the section.

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \ge 3$, whose shape is $(dX_0, dX_1, dX_2, ..., dX_{rX-1})$
- $Y$: output tensor, whose shape is $(dY_0, dY_1, dY_2, ..., dY_{rX-1})$
- $\text{Indices}$: optional output tensor of the same shape as $Y$, giving for each element of $Y$ the index of the element of $X$ it is the maximum of

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

The restrictions of the real-number section apply, and are not repeated here.

## Function

<a id="E_MAXPOOL_INT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_INT_FUNC_0010]</br></span>

Operator **MaxPool** applies a max pooling over the spatial axes of the input tensor $X$, as specified for real numbers by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0010</span></b>](#E_MAXPOOL_REAL_FUNC_0010). The result is one of the elements of $X$, and no arithmetic is performed on it.

**The order of the values.** The maximum is taken with respect to the order of the values of the type. For the signed types, that order is the order of the integers from $-2^{n-1}$ to $2^{n-1}-1$, where $n$ is the number of bits of the type: from $-128$ to $127$ for `int8`, and from $-2^{63}$ to $2^{63}-1$ for `int64`. The result is a value of the type of $X$, and no overflow can occur, since the result is one of the elements of $X$.

**The `Indices` output.** The index given for an element of $Y$ is the index of the element of $X$ of which that element is the maximum, as specified for real numbers by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0010</span></b>](#E_MAXPOOL_REAL_FUNC_0010). The index is a value of `int64`, the type of the `Indices` output, and it is the same for every type of this section.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `int8` type:

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    -5 & 3 & -1 \\
    4 & -2 & 6 \\
    7 & -8 & 9
  \end{bmatrix}
\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `pads` $= [0, 0, 0, 0]$, `dilations` $= [1, 1]$:

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    4 & 6 \\
    7 & 9
  \end{bmatrix}
\end{bmatrix}
$$

The maximum is taken with respect to the order of the signed values, in which $-5$ is smaller than $3$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs no floating-point arithmetic |
| Overflow | not applicable: the result is one of the elements of the input, and no arithmetic is performed |
| Integer overflow | not applicable: the result is one of the elements of the input, and no arithmetic is performed |
| Division by zero | not applicable: the operator performs no division |
| A pooling window that lies entirely in the padding | ruled out by the restriction `[R4]` |

No error condition.

## Attributes

The attributes of the real-number section apply, and are not repeated here.

## Inputs

### $\text{X}$: signed integer tensor

The input data tensor, whose shape is $(dX_0, dX_1, dX_2, ..., dX_{rX-1})$.

#### Constraints

- `[E_MAXPOOL_INT_CONSTR_X_0010]` Rank of $X$
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0010</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0010) on tensor $X$.
<a id="E_MAXPOOL_INT_CONSTR_X_0020"></a>
- `[E_MAXPOOL_INT_CONSTR_X_0020]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: signed integer tensor

The output data tensor, whose shape is $(dY_0, dY_1, dY_2, ..., dY_{rX-1})$.

#### Constraints

- `[E_MAXPOOL_INT_CONSTR_Y_0010]` Shape definition
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_Y_0010</span></b>](#E_MAXPOOL_REAL_CONSTR_Y_0010) on tensor $Y$.
- `[E_MAXPOOL_INT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_CONSTR_X_0020</span></b>](#E_MAXPOOL_INT_CONSTR_X_0020) on tensor $X$.

### $\text{Indices}$: integer tensor

The optional output tensor, of the same shape as $Y$.

#### Constraints

- `[E_MAXPOOL_INT_CONSTR_Indices_0010]` Shape definition
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_Indices_0010</span></b>](#E_MAXPOOL_REAL_CONSTR_Indices_0010) on tensor `Indices`.

<a id="uint"></a>

# **MaxPool** (uint)

where uint is in {`uint8`}.

The types of this section share one semantics: the operator selects one of the elements of its input and performs no arithmetic on it, so that the result is the same for every type of the section.

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \ge 3$, whose shape is $(dX_0, dX_1, dX_2, ..., dX_{rX-1})$
- $Y$: output tensor, whose shape is $(dY_0, dY_1, dY_2, ..., dY_{rX-1})$
- $\text{Indices}$: optional output tensor of the same shape as $Y$, giving for each element of $Y$ the index of the element of $X$ it is the maximum of

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

The restrictions of the real-number section apply, and are not repeated here.

## Function

<a id="E_MAXPOOL_UINT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_UINT_FUNC_0010]</br></span>

Operator **MaxPool** applies a max pooling over the spatial axes of the input tensor $X$, as specified for real numbers by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0010</span></b>](#E_MAXPOOL_REAL_FUNC_0010). The result is one of the elements of $X$, and no arithmetic is performed on it.

**The order of the values.** The maximum is taken with respect to the order of the values of the type. For the unsigned type `uint8`, that order is the order of the integers from $0$ to $255$. The result is a value of the type of $X$, and no overflow can occur, since the result is one of the elements of $X$.

**The `Indices` output.** The index given for an element of $Y$ is the index of the element of $X$ of which that element is the maximum, as specified for real numbers by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0010</span></b>](#E_MAXPOOL_REAL_FUNC_0010). The index is a value of `int64`, the type of the `Indices` output, and it is the same for every type of this section.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `uint8` type:

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    5 & 200 & 1 \\
    4 & 2 & 6 \\
    7 & 8 & 9
  \end{bmatrix}
\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `pads` $= [0, 0, 0, 0]$, `dilations` $= [1, 1]$:

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    200 & 200 \\
    8 & 9
  \end{bmatrix}
\end{bmatrix}
$$

The maximum is taken with respect to the order of the unsigned values, in which $200$ is greater than $5$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs no floating-point arithmetic |
| Overflow | not applicable: the result is one of the elements of the input, and no arithmetic is performed |
| Integer overflow | not applicable: the result is one of the elements of the input, and no arithmetic is performed |
| Division by zero | not applicable: the operator performs no division |
| A pooling window that lies entirely in the padding | ruled out by the restriction `[R4]` |

No error condition.

## Attributes

The attributes of the real-number section apply, and are not repeated here.

## Inputs

### $\text{X}$: unsigned integer tensor

The input data tensor, whose shape is $(dX_0, dX_1, dX_2, ..., dX_{rX-1})$.

#### Constraints

- `[E_MAXPOOL_UINT_CONSTR_X_0010]` Rank of $X$
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0010</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0010) on tensor $X$.
<a id="E_MAXPOOL_UINT_CONSTR_X_0020"></a>
- `[E_MAXPOOL_UINT_CONSTR_X_0020]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: unsigned integer tensor

The output data tensor, whose shape is $(dY_0, dY_1, dY_2, ..., dY_{rX-1})$.

#### Constraints

- `[E_MAXPOOL_UINT_CONSTR_Y_0010]` Shape definition
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_Y_0010</span></b>](#E_MAXPOOL_REAL_CONSTR_Y_0010) on tensor $Y$.
- `[E_MAXPOOL_UINT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_UINT_CONSTR_X_0020</span></b>](#E_MAXPOOL_UINT_CONSTR_X_0020) on tensor $X$.

### $\text{Indices}$: integer tensor

The optional output tensor, of the same shape as $Y$.

#### Constraints

- `[E_MAXPOOL_UINT_CONSTR_Indices_0010]` Shape definition
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_Indices_0010</span></b>](#E_MAXPOOL_REAL_CONSTR_Indices_0010) on tensor `Indices`.
