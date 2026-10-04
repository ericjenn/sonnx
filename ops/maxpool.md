# Contents

- **MaxPool** operator for type [real](#real)
- **MaxPool** operator for types [`float16`, `float`, `double`](#float)
- **MaxPool** operator for types [`int8`, `uint8`](#int)

Based on ONNX documentation [MaxPool version 12](https://onnx.ai/onnx/operators/onnx__MaxPool.html#maxpool-12).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **MaxPool** (real)

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \ge 3$, of shape $(dX_0, dX_1, ..., dX_{rX-1})$
- $Y$: output tensor, of shape $(dY_0, dY_1, ..., dY_{rY-1})$
- $\text{Indices}$: optional output tensor of `int64` values, of the same shape as $Y$

The first two axes of $X$ are the batch axis and the channel axis; the remaining $rX - 2$ axes are the spatial axes. The attributes `kernel_shape`, `strides`, `dilations` and `pads` give one value per spatial axis, in the order of the spatial axes of $X$.

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `ceil_mode` is restricted to `0` | Transient |
| `[R4]` | The padding of a spatial axis is at most the size of the kernel along that axis, i.e. $p^b_k \le k_k$ and $p^e_k \le k_k$ for every spatial axis $k$ | Transient |
| `[R5]` | The output `Indices` is not supported | Transient |

## Function

<a id="E_MAXPOOL_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_REAL_FUNC_0010]</br></span>

**Part 1.** Operator **MaxPool** applies max pooling to the input tensor $X$: it slides a window of the size given by `kernel_shape` over the spatial axes of $X$, with the step given by `strides`, and stores in $Y$ the maximum of the elements of $X$ covered by the window. The elements of the window that fall outside $X$ are padding elements, and they are excluded from the maximum.

**Part 2.** The output tensor $Y$ has the same rank as $X$, the same first two dimensions, and one dimension per spatial axis. For every spatial axis $k \in [0, rX-3]$:

$$
dY_{k+2} = \left\lfloor \frac{dX_{k+2} + p^b_k + p^e_k - \delta_k \cdot (k_k - 1) - 1}{s_k} \right\rfloor + 1
$$

where
- $k_k$ is the value of `kernel_shape` for spatial axis $k$,
- $s_k$ is the value of `strides` for spatial axis $k$,
- $\delta_k$ is the value of `dilations` for spatial axis $k$,
- $p^b_k$ and $p^e_k$ are the numbers of padding elements added at the beginning and at the end of spatial axis $k$, as given by `pads`.

For any [tensor index](./../common/definitions.md#tensor_index) $i = (i_0, i_1, ..., i_{rX-1})$ of the result $Y$:

$$
Y[i] = \max_{j \in W(i)} X[i_0, i_1, j_2, ..., j_{rX-1}]
$$

where
- $W(i)$ is the set of the multi-dimensional indices $j = (j_2, ..., j_{rX-1})$ of the spatial axes such that, for every spatial axis $k \in [0, rX-3]$,
  - $j_{k+2} = i_{k+2} \cdot s_k - p^b_k + \delta_k \cdot t_k$ for some $t_k \in [0, k_k-1]$, and
  - $0 \le j_{k+2} \le dX_{k+2}-1$,
- the maximum is taken over the elements of $X$ whose spatial indices are in $W(i)$, and the elements of the window that fall outside $X$ are not part of $W(i)$.

The window of the output element $i$ is thus the set of the positions $i_{k+2} \cdot s_k - p^b_k + \delta_k \cdot t_k$ for $t_k \in [0, k_k-1]$, restricted to the positions that lie inside $X$. The padding elements are excluded from the maximum, so that a window that covers no element of $X$ has no maximum; the restrictions `[R4]` and the shape formula above rule this case out, since every window then covers at least one element of $X$.

**The maximum is exact.** The result is the maximum of the elements of $X$ in the window, in the order of the real numbers; no arithmetic is performed on the elements, so that the result is one of the elements of $X$ and is exact.

**Tensors with no dimension.** The input tensor $X$ has at least three axes, so that it is never a rank-0 tensor. A spatial axis of $X$ may have size $0$; the corresponding axis of $Y$ then has size $0$ as well, and $Y$ is empty.

**Zero-sized dimensions.** A spatial axis of $X$ of size $0$ gives a spatial axis of $Y$ of size $0$ by the shape formula, and $Y$ is empty. A batch axis or a channel axis of size $0$ gives an empty $Y$ as well.

**Ties.** When several elements of $X$ in the window are equal to the maximum, the result is that value, whichever of them realizes it; the operator returns the value, not the position, so that the choice of the element is not observable in $Y$.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

With `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `dilations` $= [1, 1]$ and `pads` $= [0, 0, 0, 0]$:

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
  \end{bmatrix}
\end{bmatrix}
$$

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    5 & 6 \\
    8 & 9
  \end{bmatrix}
\end{bmatrix}
$$

The input has shape $(1, 1, 3, 3)$ and the output has shape $(1, 1, 2, 2)$: $dY_2 = \lfloor (3 + 0 + 0 - 1 \cdot (2-1) - 1)/1 \rfloor + 1 = 2$, and likewise for the third spatial axis.

### Example 2

With `kernel_shape` $= [2, 2]$, `strides` $= [2, 2]$, `dilations` $= [1, 1]$ and `pads` $= [1, 1, 1, 1]$:

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
  \end{bmatrix}
\end{bmatrix}
$$

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    1 & 3 \\
    7 & 9
  \end{bmatrix}
\end{bmatrix}
$$

The input has shape $(1, 1, 3, 3)$ and the output has shape $(1, 1, 2, 2)$: $dY_2 = \lfloor (3 + 1 + 1 - 1 \cdot (2-1) - 1)/2 \rfloor + 1 = 2$. The window of the output element $(0, 0, 0, 0)$ covers the positions $-1$ and $0$ along each spatial axis; the positions $-1$ are padding elements and are excluded, so that the maximum is $X[0, 0, 0, 0] = 1$. The window of the output element $(0, 0, 0, 1)$ covers the positions $1$ and $2$ along the last spatial axis, and the maximum is $X[0, 0, 0, 2] = 3$.

### Example 3

With `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `dilations` $= [2, 2]$ and `pads` $= [0, 0, 0, 0]$:

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    1 & 2 & 3 & 4 \\
    5 & 6 & 7 & 8 \\
    9 & 10 & 11 & 12 \\
    13 & 14 & 15 & 16
  \end{bmatrix}
\end{bmatrix}
$$

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    11 & 12 \\
    15 & 16
  \end{bmatrix}
\end{bmatrix}
$$

The input has shape $(1, 1, 4, 4)$ and the output has shape $(1, 1, 2, 2)$: $dY_2 = \lfloor (4 + 0 + 0 - 2 \cdot (2-1) - 1)/1 \rfloor + 1 = 2$. The window of the output element $(0, 0, 0, 0)$ covers the positions $0$ and $2$ along each spatial axis, and the maximum is $X[0, 0, 2, 2] = 11$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. the operations of IEEE 754 section 7.2 | not applicable: the operator performs no arithmetic on the elements of $X$, it selects one of them |
| Overflow, i.e. a computation leading to $\pm\text{inf}$ | not applicable: the operator performs no arithmetic on the elements of $X$ |
| Integer overflow | not applicable: the operator performs no arithmetic on the elements of $X$ |
| Division by zero | not applicable: the operator performs no division |
| A window that covers no element of $X$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0020</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0020) on the padding, and by the shape formula of [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0010</span></b>](#E_MAXPOOL_REAL_FUNC_0010) |
| A spatial axis of $X$ of size $0$ | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0010</span></b>](#E_MAXPOOL_REAL_FUNC_0010), the corresponding axis of $Y$ has size $0$ and $Y$ is empty |

No error condition.

## Attributes

### `auto_pad`: `string`

The padding mode. The value is `NOTSET`, `SAME_UPPER`, `SAME_LOWER` or `VALID`; the default is `NOTSET`, which means that the padding is the one given by `pads`.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_auto_pad_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_auto_pad_0010]` Restricted value
  - Statement: The value of `auto_pad` is `NOTSET`.
  - Rationale: Restriction `[R1]`; the padding is then the one given by `pads`, and the shape formula of the "Function" section applies.

### `ceil_mode`: `int`

Whether the ceiling or the floor is used to compute the output shape. The value is `0` (floor, the default) or `1` (ceiling).

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_ceil_mode_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_ceil_mode_0010]` Restricted value
  - Statement: The value of `ceil_mode` is `0`.
  - Rationale: Restriction `[R3]`; the output shape is then the one given by the floor formula of the "Function" section.

### `dilations`: `list of ints`

The dilation value along each spatial axis of the filter. The default is $1$ along each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_dilations_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_dilations_0010]` Length
  - Statement: The length of `dilations` is $rX - 2$.
<a id="E_MAXPOOL_REAL_CONSTR_dilations_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_dilations_0020]` Value range
  - Statement: Every value of `dilations` is at least $1$.

### `kernel_shape`: `list of ints`

The size of the kernel along each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_kernel_shape_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_kernel_shape_0010]` Length
  - Statement: The length of `kernel_shape` is $rX - 2$.
<a id="E_MAXPOOL_REAL_CONSTR_kernel_shape_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_kernel_shape_0020]` Value range
  - Statement: Every value of `kernel_shape` is at least $1$.

### `pads`: `list of ints`

The padding for the beginning and the end along each spatial axis, in the format $[x_1^{\text{begin}}, x_2^{\text{begin}}, ..., x_1^{\text{end}}, x_2^{\text{end}}, ...]$, where $x_k^{\text{begin}}$ is the number of padding elements added at the beginning of spatial axis $k$ and $x_k^{\text{end}}$ the number added at the end. The default is $0$ at the beginning and at the end of each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_pads_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_pads_0010]` Length
  - Statement: The length of `pads` is $2 \cdot (rX - 2)$.
<a id="E_MAXPOOL_REAL_CONSTR_pads_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_pads_0020]` Value range
  - Statement: Every value of `pads` is at least $0$.

### `storage_order`: `int`

The storage order of the tensor, `0` for row major and `1` for column major. It is used only to convert an $n$-tuple index into a single integer for the output `Indices`.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_storage_order_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_storage_order_0010]` Restricted value
  - Statement: The value of `storage_order` is `0`.
  - Rationale: Restriction `[R2]`; the output `Indices` is not supported, so that the attribute has no effect.

### `strides`: `list of ints`

The stride along each spatial axis. The default is $1$ along each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_strides_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_strides_0010]` Length
  - Statement: The length of `strides` is $rX - 2$.
<a id="E_MAXPOOL_REAL_CONSTR_strides_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_strides_0020]` Value range
  - Statement: Every value of `strides` is at least $1$.

## Inputs

### $\text{X}$: real tensor

The input data tensor, of rank $rX \ge 3$ and of shape $(dX_0, dX_1, ..., dX_{rX-1})$. The first two axes are the batch axis and the channel axis, and the remaining $rX - 2$ axes are the spatial axes.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_X_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_X_0010]` Rank
  - Statement: The rank of $X$ is at least $3$.
  - Rationale: The operator pools over the spatial axes, which are the axes from the third one on; a tensor of rank lower than $3$ has no spatial axis.
<a id="E_MAXPOOL_REAL_CONSTR_X_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_X_0020]` Padding bound
  - Statement: For every spatial axis $k \in [0, rX-3]$, $p^b_k \le k_k$ and $p^e_k \le k_k$, where $p^b_k$ and $p^e_k$ are the values of `pads` for spatial axis $k$ and $k_k$ the value of `kernel_shape` for that axis.
  - Rationale: Restriction `[R4]`; it ensures that every window of the operator covers at least one element of $X$, so that the maximum of the window is defined.

## Outputs

### $\text{Y}$: real tensor

The output data tensor, of shape $(dY_0, dY_1, ..., dY_{rY-1})$, with $rY = rX$, $dY_0 = dX_0$, $dY_1 = dX_1$, and $dY_{k+2}$ given by the shape formula of the "Function" section for every spatial axis $k$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_Y_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the shape given by the shape formula of [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0010</span></b>](#E_MAXPOOL_REAL_FUNC_0010).

### $\text{Indices}$: `int64` tensor

The indices of the selected values. The output is optional and is not supported.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_Indices_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_Indices_0010]` Absent output
  - Statement: The output `Indices` is absent.
  - Rationale: Restriction `[R5]`.

<a id="float"></a>

# **MaxPool** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the operator selects one of the elements of $X$ and performs no arithmetic on it, so that the special values of IEEE 754 are compared as the standard orders them and the result is one of the elements of $X$ for all three types.

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \ge 3$, of shape $(dX_0, dX_1, ..., dX_{rX-1})$
- $Y$: output tensor, of shape $(dY_0, dY_1, ..., dY_{rY-1})$
- $\text{Indices}$: optional output tensor of `int64` values, of the same shape as $Y$

The first two axes of $X$ are the batch axis and the channel axis; the remaining $rX - 2$ axes are the spatial axes. The attributes `kernel_shape`, `strides`, `dilations` and `pads` give one value per spatial axis, in the order of the spatial axes of $X$.

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `ceil_mode` is restricted to `0` | Transient |
| `[R4]` | The padding of a spatial axis is at most the size of the kernel along that axis, i.e. $p^b_k \le k_k$ and $p^e_k \le k_k$ for every spatial axis $k$ | Transient |
| `[R5]` | The output `Indices` is not supported | Transient |

## Function

<a id="E_MAXPOOL_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_FLOAT_FUNC_0010]</br></span>

**Part 1.** Operator **MaxPool** applies max pooling to the input tensor $X$: it slides a window of the size given by `kernel_shape` over the spatial axes of $X$, with the step given by `strides`, and stores in $Y$ the maximum of the elements of $X$ covered by the window. The elements of the window that fall outside $X$ are padding elements, and they are excluded from the maximum.

**Part 2.** The output tensor $Y$ has the same rank as $X$, the same first two dimensions, and one dimension per spatial axis. For every spatial axis $k \in [0, rX-3]$:

$$
dY_{k+2} = \left\lfloor \frac{dX_{k+2} + p^b_k + p^e_k - \delta_k \cdot (k_k - 1) - 1}{s_k} \right\rfloor + 1
$$

where
- $k_k$ is the value of `kernel_shape` for spatial axis $k$,
- $s_k$ is the value of `strides` for spatial axis $k$,
- $\delta_k$ is the value of `dilations` for spatial axis $k$,
- $p^b_k$ and $p^e_k$ are the numbers of padding elements added at the beginning and at the end of spatial axis $k$, as given by `pads`.

For any [tensor index](./../common/definitions.md#tensor_index) $i = (i_0, i_1, ..., i_{rX-1})$ of the result $Y$:

$$
Y[i] = \max_{j \in W(i)} X[i_0, i_1, j_2, ..., j_{rX-1}]
$$

where
- $W(i)$ is the set of the multi-dimensional indices $j = (j_2, ..., j_{rX-1})$ of the spatial axes such that, for every spatial axis $k \in [0, rX-3]$,
  - $j_{k+2} = i_{k+2} \cdot s_k - p^b_k + \delta_k \cdot t_k$ for some $t_k \in [0, k_k-1]$, and
  - $0 \le j_{k+2} \le dX_{k+2}-1$,
- the maximum is taken over the elements of $X$ whose spatial indices are in $W(i)$, and the elements of the window that fall outside $X$ are not part of $W(i)$.

The window of the output element $i$ is thus the set of the positions $i_{k+2} \cdot s_k - p^b_k + \delta_k \cdot t_k$ for $t_k \in [0, k_k-1]$, restricted to the positions that lie inside $X$. The padding elements are excluded from the maximum, so that a window that covers no element of $X$ has no maximum; the restrictions `[R4]` and the shape formula above rule this case out, since every window then covers at least one element of $X$.

**The maximum of the floating-point values.** The maximum is the one of the order of the real numbers extended with the special values: $-\text{inf}$ is lower than every finite value, $+\text{inf}$ is greater than every finite value, and $-0$ and $+0$ are equal. The result is one of the elements of $X$ in the window, so that it is exact and no rounding takes place.

**NaN.** When the window contains a NaN, the result is a NaN: a NaN is not ordered with respect to the other values, and the maximum of a set that contains a NaN is a NaN. The NaN of the result is a quiet NaN; its sign and its payload are not specified, so that every quiet NaN of the type of $X$ is a conforming result.

**The sign of a null result.** When the maximum is null and the window contains both $-0$ and $+0$, the result is either $-0$ or $+0$: the two are equal, and the operator returns the value of one of the elements of $X$ that realize the maximum. Both are conforming results.

**Tensors with no dimension.** The input tensor $X$ has at least three axes, so that it is never a rank-0 tensor. A spatial axis of $X$ may have size $0$; the corresponding axis of $Y$ then has size $0$ as well, and $Y$ is empty.

**Zero-sized dimensions.** A spatial axis of $X$ of size $0$ gives a spatial axis of $Y$ of size $0$ by the shape formula, and $Y$ is empty. A batch axis or a channel axis of size $0$ gives an empty $Y$ as well.

**Ties.** When several elements of $X$ in the window are equal to the maximum, the result is that value, whichever of them realizes it; the operator returns the value, not the position, so that the choice of the element is not observable in $Y$.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

With `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `dilations` $= [1, 1]$ and `pads` $= [0, 0, 0, 0]$:

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    1.5 & 2.5 & 3.5 \\
    4.5 & 5.5 & 6.5 \\
    7.5 & 8.5 & 9.5
  \end{bmatrix}
\end{bmatrix}
$$

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    5.5 & 6.5 \\
    8.5 & 9.5
  \end{bmatrix}
\end{bmatrix}
$$

The input has shape $(1, 1, 3, 3)$ and the output has shape $(1, 1, 2, 2)$.

### Example 2

With `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `dilations` $= [1, 1]$ and `pads` $= [0, 0, 0, 0]$:

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    1.0 & \text{+inf} & 3.0 \\
    \text{-inf} & \text{NaN} & 6.0 \\
    7.0 & 8.0 & \text{+0}
  \end{bmatrix}
\end{bmatrix}
$$

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    \text{+inf} & \text{NaN} \\
    8.0 & \text{NaN}
  \end{bmatrix}
\end{bmatrix}
$$

The window of the output element $(0, 0, 0, 0)$ contains $+\text{inf}$, which is greater than every finite value, and the result is $+\text{inf}$. The windows of the output elements $(0, 0, 0, 1)$ and $(0, 0, 1, 1)$ contain the NaN of $X[0, 0, 1, 1]$, and the result is a NaN. The window of the output element $(0, 0, 1, 0)$ contains $-\text{inf}$ and the finite values $7.0$ and $8.0$, and the result is $8.0$.

### Example 3

With `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `dilations` $= [1, 1]$ and `pads` $= [0, 0, 0, 0]$:

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    \text{-0} & \text{+0} \\
    \text{+0} & \text{-0}
  \end{bmatrix}
\end{bmatrix}
$$

$$
Y = \begin{bmatrix} \begin{bmatrix} \text{+0} \end{bmatrix} \end{bmatrix}
$$

The four elements of the window are equal, since $-0$ and $+0$ are equal; the result is one of them, and both $-0$ and $+0$ are conforming results.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. the operations of IEEE 754 section 7.2 | not applicable: the operator performs no arithmetic on the elements of $X$, it selects one of them |
| Overflow, i.e. a computation leading to $\pm\text{inf}$ | not applicable: the operator performs no arithmetic on the elements of $X$ |
| Integer overflow | not applicable: the operator performs no arithmetic on the elements of $X$ |
| Division by zero | not applicable: the operator performs no division |
| A window that covers no element of $X$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_CONSTR_X_0020</span></b>](#E_MAXPOOL_FLOAT_CONSTR_X_0020) on the padding, and by the shape formula of [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_FUNC_0010</span></b>](#E_MAXPOOL_FLOAT_FUNC_0010) |
| A window that contains a NaN | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_FUNC_0010</span></b>](#E_MAXPOOL_FLOAT_FUNC_0010), the result is a quiet NaN |
| A spatial axis of $X$ of size $0$ | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_FUNC_0010</span></b>](#E_MAXPOOL_FLOAT_FUNC_0010), the corresponding axis of $Y$ has size $0$ and $Y$ is empty |

No error condition.

## Attributes

The attributes are those of the [real](#real) section; the constraints on their values are the same.

## Inputs

### $\text{X}$: floating-point tensor

The input data tensor, of rank $rX \ge 3$ and of shape $(dX_0, dX_1, ..., dX_{rX-1})$. The first two axes are the batch axis and the channel axis, and the remaining $rX - 2$ axes are the spatial axes.

#### Constraints

<a id="E_MAXPOOL_FLOAT_CONSTR_X_0010"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_X_0010]` Rank
  - Statement: The rank of $X$ is at least $3$.
  - Rationale: The operator pools over the spatial axes, which are the axes from the third one on; a tensor of rank lower than $3$ has no spatial axis.
<a id="E_MAXPOOL_FLOAT_CONSTR_X_0020"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_X_0020]` Padding bound
  - Statement: For every spatial axis $k \in [0, rX-3]$, $p^b_k \le k_k$ and $p^e_k \le k_k$, where $p^b_k$ and $p^e_k$ are the values of `pads` for spatial axis $k$ and $k_k$ the value of `kernel_shape` for that axis.
  - Rationale: Restriction `[R4]`; it ensures that every window of the operator covers at least one element of $X$, so that the maximum of the window is defined.

## Outputs

### $\text{Y}$: floating-point tensor

The output data tensor, of shape $(dY_0, dY_1, ..., dY_{rY-1})$, with $rY = rX$, $dY_0 = dX_0$, $dY_1 = dX_1$, and $dY_{k+2}$ given by the shape formula of the "Function" section for every spatial axis $k$.

#### Constraints

<a id="E_MAXPOOL_FLOAT_CONSTR_Y_0010"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the shape given by the shape formula of [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_FUNC_0010</span></b>](#E_MAXPOOL_FLOAT_FUNC_0010).

### $\text{Indices}$: `int64` tensor

The indices of the selected values. The output is optional and is not supported.

#### Constraints

<a id="E_MAXPOOL_FLOAT_CONSTR_Indices_0010"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_Indices_0010]` Absent output
  - Statement: The output `Indices` is absent.
  - Rationale: Restriction `[R5]`.

<a id="int"></a>

# **MaxPool** (int)

where int is in {`int8`, `uint8`}.

The two types share one semantics: the operator selects one of the elements of $X$ and performs no arithmetic on it, so that the result is one of the elements of $X$ for both types, and the order of the values is the order of the integers for each of them.

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \ge 3$, of shape $(dX_0, dX_1, ..., dX_{rX-1})$
- $Y$: output tensor, of shape $(dY_0, dY_1, ..., dY_{rY-1})$
- $\text{Indices}$: optional output tensor of `int64` values, of the same shape as $Y$

The first two axes of $X$ are the batch axis and the channel axis; the remaining $rX - 2$ axes are the spatial axes. The attributes `kernel_shape`, `strides`, `dilations` and `pads` give one value per spatial axis, in the order of the spatial axes of $X$.

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `ceil_mode` is restricted to `0` | Transient |
| `[R4]` | The padding of a spatial axis is at most the size of the kernel along that axis, i.e. $p^b_k \le k_k$ and $p^e_k \le k_k$ for every spatial axis $k$ | Transient |
| `[R5]` | The output `Indices` is not supported | Transient |

## Function

<a id="E_MAXPOOL_INT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_INT_FUNC_0010]</br></span>

**Part 1.** Operator **MaxPool** applies max pooling to the input tensor $X$: it slides a window of the size given by `kernel_shape` over the spatial axes of $X$, with the step given by `strides`, and stores in $Y$ the maximum of the elements of $X$ covered by the window. The elements of the window that fall outside $X$ are padding elements, and they are excluded from the maximum.

**Part 2.** The output tensor $Y$ has the same rank as $X$, the same first two dimensions, and one dimension per spatial axis. For every spatial axis $k \in [0, rX-3]$:

$$
dY_{k+2} = \left\lfloor \frac{dX_{k+2} + p^b_k + p^e_k - \delta_k \cdot (k_k - 1) - 1}{s_k} \right\rfloor + 1
$$

where
- $k_k$ is the value of `kernel_shape` for spatial axis $k$,
- $s_k$ is the value of `strides` for spatial axis $k$,
- $\delta_k$ is the value of `dilations` for spatial axis $k$,
- $p^b_k$ and $p^e_k$ are the numbers of padding elements added at the beginning and at the end of spatial axis $k$, as given by `pads`.

For any [tensor index](./../common/definitions.md#tensor_index) $i = (i_0, i_1, ..., i_{rX-1})$ of the result $Y$:

$$
Y[i] = \max_{j \in W(i)} X[i_0, i_1, j_2, ..., j_{rX-1}]
$$

where
- $W(i)$ is the set of the multi-dimensional indices $j = (j_2, ..., j_{rX-1})$ of the spatial axes such that, for every spatial axis $k \in [0, rX-3]$,
  - $j_{k+2} = i_{k+2} \cdot s_k - p^b_k + \delta_k \cdot t_k$ for some $t_k \in [0, k_k-1]$, and
  - $0 \le j_{k+2} \le dX_{k+2}-1$,
- the maximum is taken over the elements of $X$ whose spatial indices are in $W(i)$, and the elements of the window that fall outside $X$ are not part of $W(i)$.

The window of the output element $i$ is thus the set of the positions $i_{k+2} \cdot s_k - p^b_k + \delta_k \cdot t_k$ for $t_k \in [0, k_k-1]$, restricted to the positions that lie inside $X$. The padding elements are excluded from the maximum, so that a window that covers no element of $X$ has no maximum; the restrictions `[R4]` and the shape formula above rule this case out, since every window then covers at least one element of $X$.

**The maximum of the integer values.** The maximum is the one of the order of the integers, which is the same for the signed and the unsigned types: the result is one of the elements of $X$ in the window, so that it is exact and no reduction modulo $2^n$ takes place. The result of the operator is never outside the range of the type of $X$, since it is one of its values.

**Tensors with no dimension.** The input tensor $X$ has at least three axes, so that it is never a rank-0 tensor. A spatial axis of $X$ may have size $0$; the corresponding axis of $Y$ then has size $0$ as well, and $Y$ is empty.

**Zero-sized dimensions.** A spatial axis of $X$ of size $0$ gives a spatial axis of $Y$ of size $0$ by the shape formula, and $Y$ is empty. A batch axis or a channel axis of size $0$ gives an empty $Y$ as well.

**Ties.** When several elements of $X$ in the window are equal to the maximum, the result is that value, whichever of them realizes it; the operator returns the value, not the position, so that the choice of the element is not observable in $Y$.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

For the `int8` type, with `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `dilations` $= [1, 1]$ and `pads` $= [0, 0, 0, 0]$:

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    -5 & -4 & -3 \\
    -2 & -1 & 0 \\
    1 & 2 & 3
  \end{bmatrix}
\end{bmatrix}
$$

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    -1 & 0 \\
    2 & 3
  \end{bmatrix}
\end{bmatrix}
$$

The input has shape $(1, 1, 3, 3)$ and the output has shape $(1, 1, 2, 2)$.

### Example 2

For the `uint8` type, with `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `dilations` $= [1, 1]$ and `pads` $= [0, 0, 0, 0]$:

$$
X = \begin{bmatrix}
  \begin{bmatrix}
    250 & 1 & 2 \\
    3 & 4 & 5 \\
    6 & 7 & 8
  \end{bmatrix}
\end{bmatrix}
$$

$$
Y = \begin{bmatrix}
  \begin{bmatrix}
    250 & 5 \\
    7 & 8
  \end{bmatrix}
\end{bmatrix}
$$

The value $250$ is the maximum of the first window, and the result is that value of `uint8`; no reduction modulo $2^8$ takes place, since the operator selects an element of $X$ and performs no arithmetic on it.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation, i.e. the operations of IEEE 754 section 7.2 | not applicable: the operator performs no arithmetic on the elements of $X$, it selects one of them |
| Overflow, i.e. a computation leading to $\pm\text{inf}$ | not applicable: the operator performs no arithmetic on the elements of $X$ |
| Integer overflow, i.e. a result outside the range of the type | not applicable: the result is one of the elements of $X$, and is therefore a value of the type of $X$ |
| Division by zero | not applicable: the operator performs no division |
| A window that covers no element of $X$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_CONSTR_X_0020</span></b>](#E_MAXPOOL_INT_CONSTR_X_0020) on the padding, and by the shape formula of [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_FUNC_0010</span></b>](#E_MAXPOOL_INT_FUNC_0010) |
| A spatial axis of $X$ of size $0$ | nominal: specified by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_FUNC_0010</span></b>](#E_MAXPOOL_INT_FUNC_0010), the corresponding axis of $Y$ has size $0$ and $Y$ is empty |

No error condition.

## Attributes

The attributes are those of the [real](#real) section; the constraints on their values are the same.

## Inputs

### $\text{X}$: integer tensor

The input data tensor, of rank $rX \ge 3$ and of shape $(dX_0, dX_1, ..., dX_{rX-1})$. The first two axes are the batch axis and the channel axis, and the remaining $rX - 2$ axes are the spatial axes.

#### Constraints

<a id="E_MAXPOOL_INT_CONSTR_X_0010"></a>
- `[E_MAXPOOL_INT_CONSTR_X_0010]` Rank
  - Statement: The rank of $X$ is at least $3$.
  - Rationale: The operator pools over the spatial axes, which are the axes from the third one on; a tensor of rank lower than $3$ has no spatial axis.
<a id="E_MAXPOOL_INT_CONSTR_X_0020"></a>
- `[E_MAXPOOL_INT_CONSTR_X_0020]` Padding bound
  - Statement: For every spatial axis $k \in [0, rX-3]$, $p^b_k \le k_k$ and $p^e_k \le k_k$, where $p^b_k$ and $p^e_k$ are the values of `pads` for spatial axis $k$ and $k_k$ the value of `kernel_shape` for that axis.
  - Rationale: Restriction `[R4]`; it ensures that every window of the operator covers at least one element of $X$, so that the maximum of the window is defined.

## Outputs

### $\text{Y}$: integer tensor

The output data tensor, of shape $(dY_0, dY_1, ..., dY_{rY-1})$, with $rY = rX$, $dY_0 = dX_0$, $dY_1 = dX_1$, and $dY_{k+2}$ given by the shape formula of the "Function" section for every spatial axis $k$.

#### Constraints

<a id="E_MAXPOOL_INT_CONSTR_Y_0010"></a>
- `[E_MAXPOOL_INT_CONSTR_Y_0010]` Shape definition
  - Statement: Tensor $Y$ has the shape given by the shape formula of [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_FUNC_0010</span></b>](#E_MAXPOOL_INT_FUNC_0010).

### $\text{Indices}$: `int64` tensor

The indices of the selected values. The output is optional and is not supported.

#### Constraints

<a id="E_MAXPOOL_INT_CONSTR_Indices_0010"></a>
- `[E_MAXPOOL_INT_CONSTR_Indices_0010]` Absent output
  - Statement: The output `Indices` is absent.
  - Rationale: Restriction `[R5]`.
