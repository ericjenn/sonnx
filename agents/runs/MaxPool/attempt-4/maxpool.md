# Contents

- **MaxPool** operator for type [real](#real)
- **MaxPool** operator for types [`float16`, `float`, `double`](#float)
- **MaxPool** operator for types [`int8`, `int64`](#int)
- **MaxPool** operator for types [`uint8`](#uint)

Based on ONNX documentation [MaxPool version 14](https://onnx.ai/onnx/operators/onnx__MaxPool.html#maxpool-14).

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **MaxPool** (real)

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \ge 3$
- $Y$: result of the max pooling of $X$
- $\text{Indices}$: optional tensor of the flattened indices of the elements of $X$ selected by the pooling windows

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `ceil_mode` is restricted to `0` | Transient |
| `[R4]` | Attribute `dilations` is restricted to $1$ along each spatial axis | Transient |
| `[R5]` | The output $\text{Indices}$ is not supported | Transient |

## Function

<a id="E_MAXPOOL_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_REAL_FUNC_0010]</br></span>

Operator **MaxPool** applies max pooling to the input tensor $X$: it slides a window of the shape given by `kernel_shape` over the spatial axes of $X$, with the steps given by `strides`, and stores in $Y$ the maximum of the elements of $X$ covered by each window.

The first two axes of $X$ are the batch axis and the channel axis; they are not pooled. The remaining $rX - 2$ axes are the spatial axes, numbered from $0$ to $rX-3$ in the order of the axes of $X$. The number of spatial axes is denoted $m = rX - 2$; the operator is applicable to any $m \ge 1$.

For any [tensor index](./../common/definitions.md#tensor_index) $i = (n, c, j_0, \dots, j_{m-1})$ of the result $Y$:

$$
Y[n, c, j_0, \dots, j_{m-1}] = \max_{k_0, \dots, k_{m-1}} X_p[n, c, j_0 \cdot s_0 + k_0, \dots, j_{m-1} \cdot s_{m-1} + k_{m-1}]
$$

where
- $n \in [0, dX_0-1]$ is the batch index, $dX_0$ being the batch size of $X$,
- $c \in [0, dX_1-1]$ is the channel index, $dX_1$ being the number of channels of $X$,
- $j_t \in [0, dY_{t+2}-1]$ is the index along spatial axis $t$ of $Y$, for $t \in [0, m-1]$,
- $k_t \in [0, dK_t-1]$ is the index along spatial axis $t$ of the window, $dK_t$ being the value of `kernel_shape` along that axis,
- $s_t$ is the value of `strides` along spatial axis $t$,
- $X_p$ is the tensor obtained by padding $X$ with the values of `pads`, and $X_p[n, c, h_0, \dots, h_{m-1}]$ is one of its elements,
- the maximum is taken over the $dK_0 \times \dots \times dK_{m-1}$ elements of the window.

**Padding.** The padding adds $p_{t,\text{begin}}$ elements before and $p_{t,\text{end}}$ elements after the input along spatial axis $t$, where $p_{t,\text{begin}}$ and $p_{t,\text{end}}$ are the two values of `pads` for that axis. The padded elements are not elements of $X$: they are excluded from the maximum, so that the maximum is taken over the elements of $X$ covered by the window only. A window that covers no element of $X$ is not a window of the result: the shape of $Y$ is such that every window covers at least one element of $X$.

**Shape of the result.** The spatial dimensions of $Y$ are

$$
dY_{t+2} = \left\lfloor \frac{dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} - dK_t}{s_t} \right\rfloor + 1
$$

for every $t \in [0, m-1]$, and $dY_0 = dX_0$, $dY_1 = dX_1$. The floor is the one of the `ceil_mode` attribute, which is restricted to $0$ by [<b><span style="font-family: 'Courier New', monospace">R3</span></b>](#R3); the formula above is the one of the ONNX definition with `ceil_mode` disabled and `dilations` equal to $1$.

**Tensors with no dimension.** The input tensor $X$ has at least three axes, since the batch axis, the channel axis and at least one spatial axis are required; a rank-0 tensor is not a valid input.

**Zero-sized dimensions.** A spatial dimension of $X$ may have size $0$. The formula above then gives a spatial dimension of $Y$ of size $0$ when $p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$, and the corresponding dimension of $Y$ is empty; otherwise the operator is not applicable, since a window would cover no element of $X$. A batch or channel dimension of size $0$ gives a dimension of $Y$ of size $0$, and $Y$ is empty.

**Ties.** When several elements of a window are equal to the maximum, the result is that maximum, and the choice of the element that realizes it is not observable in $Y$. The output $\text{Indices}$ would make that choice observable; it is not supported by this profile, as stated by [<b><span style="font-family: 'Courier New', monospace">R5</span></b>](#R5).

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix}\begin{bmatrix}
  1 & 2 & 3 & 4 \\
  5 & 6 & 7 & 8 \\
  9 & 10 & 11 & 12 \\
  13 & 14 & 15 & 16
\end{bmatrix}\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$ and `strides` $= [2, 2]$, so that $X$ has shape $(1, 1, 4, 4)$: $dX_0 = dX_1 = 1$, $m = 2$, $dK_0 = dK_1 = 2$, $s_0 = s_1 = 2$ and $p_{t,\text{begin}} = p_{t,\text{end}} = 0$:

$$
Y = \begin{bmatrix}\begin{bmatrix}
  6 & 8 \\
  14 & 16
\end{bmatrix}\end{bmatrix}
$$

The window of the first element of $Y$ covers the elements $1, 2, 5, 6$ of $X$, whose maximum is $6$; the window of the last element covers $11, 12, 15, 16$, whose maximum is $16$.

### Example 2

$$
X = \begin{bmatrix}\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}\end{bmatrix}
$$

with `kernel_shape` $= [2]$, `strides` $= [1]$ and `pads` $= [1, 1]$, so that $X$ has shape $(1, 1, 3)$: $dX_0 = dX_1 = 1$, $m = 1$, $dK_0 = 2$, $s_0 = 1$, $p_{0,\text{begin}} = p_{0,\text{end}} = 1$ and $dY_2 = \lfloor (3 + 1 + 1 - 2)/1 \rfloor + 1 = 4$:

$$
Y = \begin{bmatrix}\begin{bmatrix} 1 & 2 & 3 & 3 \end{bmatrix}\end{bmatrix}
$$

The first window covers the padded element before $X$ and the element $1$; the padded element is excluded from the maximum, which is $1$. The last window covers the element $3$ and the padded element after $X$; the maximum is $3$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs no floating-point arithmetic, and the maximum of a set of values is defined for every value of the type, including the special numbers |
| Overflow | not applicable: the result is one of the elements of $X$, and no arithmetic is performed |
| Integer overflow | not applicable: the result is one of the elements of $X$, and no arithmetic is performed |
| Division by zero | not applicable: the operator performs no division |
| A window that covers no element of $X$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0020</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0020) |

No error condition.

## Attributes

### `auto_pad`: `string`

`auto_pad` selects the way the padding is computed. It is restricted to `NOTSET` by [<b><span style="font-family: 'Courier New', monospace">R1</span></b>](#R1), so that the padding is the one given by `pads`.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_auto_pad_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_auto_pad_0010]` Value of `auto_pad`
  - Statement: The value of `auto_pad` is `NOTSET`.
  - Rationale: The other values of the attribute are deprecated in ONNX and are not supported by this profile.

### `ceil_mode`: `int`

`ceil_mode` selects the rounding of the output shape. It is restricted to $0$ by [<b><span style="font-family: 'Courier New', monospace">R3</span></b>](#R3), so that the floor is used.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_ceil_mode_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_ceil_mode_0010]` Value of `ceil_mode`
  - Statement: The value of `ceil_mode` is $0$.

### `dilations`: `list of int`

`dilations` gives the dilation along each spatial axis. It is restricted to $1$ along each spatial axis by [<b><span style="font-family: 'Courier New', monospace">R4</span></b>](#R4), so that the window covers consecutive elements of $X$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_dilations_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_dilations_0010]` Value of `dilations`
  - Statement: Every value of `dilations` is $1$.

### `kernel_shape`: `list of int`

`kernel_shape` gives the size of the window along each spatial axis. It has $m$ values, one per spatial axis, and every value is at least $1$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_kernel_shape_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_kernel_shape_0010]` Length of `kernel_shape`
  - Statement: The number of values of `kernel_shape` is $rX - 2$.
<a id="E_MAXPOOL_REAL_CONSTR_kernel_shape_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_kernel_shape_0020]` Values of `kernel_shape`
  - Statement: Every value of `kernel_shape` is at least $1$.

### `pads`: `list of int`

`pads` gives the number of elements added before and after $X$ along each spatial axis. It has $2m$ values, the first $m$ being the numbers added at the beginning of the axes and the last $m$ the numbers added at the end, in the order of the spatial axes. Every value is at least $0$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_pads_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_pads_0010]` Length of `pads`
  - Statement: The number of values of `pads` is $2(rX - 2)$.
<a id="E_MAXPOOL_REAL_CONSTR_pads_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_pads_0020]` Values of `pads`
  - Statement: Every value of `pads` is at least $0$.

### `storage_order`: `int`

`storage_order` selects the order in which the flattened index of the output $\text{Indices}$ is computed. It is restricted to $0$ by [<b><span style="font-family: 'Courier New', monospace">R2</span></b>](#R2), and the output $\text{Indices}$ is not supported by this profile, as stated by [<b><span style="font-family: 'Courier New', monospace">R5</span></b>](#R5).

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_storage_order_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_storage_order_0010]` Value of `storage_order`
  - Statement: The value of `storage_order` is $0$.

### `strides`: `list of int`

`strides` gives the step of the window along each spatial axis. It has $m$ values, one per spatial axis, and every value is at least $1$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_strides_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_strides_0010]` Length of `strides`
  - Statement: The number of values of `strides` is $rX - 2$.
<a id="E_MAXPOOL_REAL_CONSTR_strides_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_strides_0020]` Values of `strides`
  - Statement: Every value of `strides` is at least $1$.

## Inputs

### $\text{X}$: real tensor

The tensor to which the max pooling is applied. Its first two axes are the batch axis and the channel axis, and its remaining axes are the spatial axes.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_X_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_X_0010]` Rank of $X$
  - Statement: The rank of $X$ is at least $3$.
  - Rationale: The batch axis, the channel axis and at least one spatial axis are required.
<a id="E_MAXPOOL_REAL_CONSTR_X_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_X_0020]` Window coverage
  - Statement: For every spatial axis $t$, either $dX_{t+2} \ge 1$ and $dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$, or $dX_{t+2} = 0$ and $p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$.
  - Rationale: Every window of the result covers at least one element of $X$.

## Outputs

### $\text{Y}$: real tensor

The result of the max pooling of $X$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_Y_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_Y_0010]` Shape definition
  - Statement: $dY_0 = dX_0$, $dY_1 = dX_1$, and for every spatial axis $t$, $dY_{t+2} = \lfloor (dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} - dK_t)/s_t \rfloor + 1$.

<a id="float"></a>

# **MaxPool** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the operator selects one of the elements of the input tensor and performs no arithmetic on it, so that the result is the same for all of them, the special numbers included.

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \ge 3$
- $Y$: result of the max pooling of $X$
- $\text{Indices}$: optional tensor of the flattened indices of the elements of $X$ selected by the pooling windows

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `ceil_mode` is restricted to `0` | Transient |
| `[R4]` | Attribute `dilations` is restricted to $1$ along each spatial axis | Transient |
| `[R5]` | The output $\text{Indices}$ is not supported | Transient |

## Function

<a id="E_MAXPOOL_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_FLOAT_FUNC_0010]</br></span>

Operator **MaxPool** applies max pooling to the input tensor $X$: it slides a window of the shape given by `kernel_shape` over the spatial axes of $X$, with the steps given by `strides`, and stores in $Y$ the maximum of the elements of $X$ covered by each window.

The first two axes of $X$ are the batch axis and the channel axis; they are not pooled. The remaining $rX - 2$ axes are the spatial axes, numbered from $0$ to $rX-3$ in the order of the axes of $X$. The number of spatial axes is denoted $m = rX - 2$; the operator is applicable to any $m \ge 1$.

For any [tensor index](./../common/definitions.md#tensor_index) $i = (n, c, j_0, \dots, j_{m-1})$ of the result $Y$:

$$
Y[n, c, j_0, \dots, j_{m-1}] = \max_{k_0, \dots, k_{m-1}} X_p[n, c, j_0 \cdot s_0 + k_0, \dots, j_{m-1} \cdot s_{m-1} + k_{m-1}]
$$

where
- $n \in [0, dX_0-1]$ is the batch index, $dX_0$ being the batch size of $X$,
- $c \in [0, dX_1-1]$ is the channel index, $dX_1$ being the number of channels of $X$,
- $j_t \in [0, dY_{t+2}-1]$ is the index along spatial axis $t$ of $Y$, for $t \in [0, m-1]$,
- $k_t \in [0, dK_t-1]$ is the index along spatial axis $t$ of the window, $dK_t$ being the value of `kernel_shape` along that axis,
- $s_t$ is the value of `strides` along spatial axis $t$,
- $X_p$ is the tensor obtained by padding $X$ with the values of `pads`, and $X_p[n, c, h_0, \dots, h_{m-1}]$ is one of its elements,
- the maximum is taken over the elements of $X$ covered by the window, the padded elements being excluded.

**The maximum of the floating-point values.** The maximum is the one of the values of the type, and it is not the maximum of the real numbers they represent: the values of the type include the special numbers $\pm 0$, $\pm\text{inf}$ and NaN, and the maximum is defined on them as follows.

- NaN is not comparable with any value, itself included. When a window covers an element that is NaN, the result of that window is NaN. The NaN of the result is a quiet NaN; its sign and its payload are not specified, so that every quiet NaN of the type of $X$ and $Y$ is a conforming result.
- $+\text{inf}$ is greater than every finite value and than $-\text{inf}$, and $-\text{inf}$ is smaller than every finite value. A window whose elements are all $-\text{inf}$ gives $-\text{inf}$.
- $+0$ and $-0$ are equal, and neither is greater than the other. A window whose elements are all null gives a null result; its sign is not specified, so that $+0$ and $-0$ are both conforming results.

**Padding.** The padding adds $p_{t,\text{begin}}$ elements before and $p_{t,\text{end}}$ elements after the input along spatial axis $t$, where $p_{t,\text{begin}}$ and $p_{t,\text{end}}$ are the two values of `pads` for that axis. The padded elements are not elements of $X$: they are excluded from the maximum, so that the maximum is taken over the elements of $X$ covered by the window only. A window that covers no element of $X$ is not a window of the result: the shape of $Y$ is such that every window covers at least one element of $X$.

**Shape of the result.** The spatial dimensions of $Y$ are

$$
dY_{t+2} = \left\lfloor \frac{dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} - dK_t}{s_t} \right\rfloor + 1
$$

for every $t \in [0, m-1]$, and $dY_0 = dX_0$, $dY_1 = dX_1$. The floor is the one of the `ceil_mode` attribute, which is restricted to $0$ by [<b><span style="font-family: 'Courier New', monospace">R3</span></b>](#R3); the formula above is the one of the ONNX definition with `ceil_mode` disabled and `dilations` equal to $1$.

**Tensors with no dimension.** The input tensor $X$ has at least three axes, since the batch axis, the channel axis and at least one spatial axis are required; a rank-0 tensor is not a valid input.

**Zero-sized dimensions.** A spatial dimension of $X$ may have size $0$. The formula above then gives a spatial dimension of $Y$ of size $0$ when $p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$, and the corresponding dimension of $Y$ is empty; otherwise the operator is not applicable, since a window would cover no element of $X$. A batch or channel dimension of size $0$ gives a dimension of $Y$ of size $0$, and $Y$ is empty.

**Ties.** When several elements of a window are equal to the maximum, the result is that maximum, and the choice of the element that realizes it is not observable in $Y$. The output $\text{Indices}$ would make that choice observable; it is not supported by this profile, as stated by [<b><span style="font-family: 'Courier New', monospace">R5</span></b>](#R5).

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix}\begin{bmatrix}
  1.0 & 2.0 & 3.0 & 4.0 \\
  5.0 & 6.0 & 7.0 & 8.0 \\
  9.0 & 10.0 & 11.0 & 12.0 \\
  13.0 & 14.0 & 15.0 & 16.0
\end{bmatrix}\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$ and `strides` $= [2, 2]$, so that $X$ has shape $(1, 1, 4, 4)$:

$$
Y = \begin{bmatrix}\begin{bmatrix}
  6.0 & 8.0 \\
  14.0 & 16.0
\end{bmatrix}\end{bmatrix}
$$

### Example 2

$$
X = \begin{bmatrix}\begin{bmatrix} 1.0 & \text{NaN} & 3.0 & \text{-inf} \end{bmatrix}\end{bmatrix}
$$

with `kernel_shape` $= [2]$ and `strides` $= [2]$, so that $X$ has shape $(1, 1, 4)$:

$$
Y = \begin{bmatrix}\begin{bmatrix} \text{NaN} & 3.0 \end{bmatrix}\end{bmatrix}
$$

The first window covers $1.0$ and NaN; the result is NaN, since NaN is not comparable with any value. The second window covers $3.0$ and $-\text{inf}$; the result is $3.0$, which is greater than $-\text{inf}$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs no floating-point arithmetic, and the maximum of a set of values is defined for every value of the type, including the special numbers |
| Overflow | not applicable: the result is one of the elements of $X$, and no arithmetic is performed |
| Integer overflow | not applicable: the operator performs no integer arithmetic |
| Division by zero | not applicable: the operator performs no division |
| A window that covers no element of $X$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_CONSTR_X_0020</span></b>](#E_MAXPOOL_FLOAT_CONSTR_X_0020) |

No error condition.

## Attributes

The attributes of the operator are the ones given in the [real](#real) section; they do not depend on the type of the arguments.

## Inputs

### $\text{X}$: floating-point tensor

The tensor to which the max pooling is applied. Its first two axes are the batch axis and the channel axis, and its remaining axes are the spatial axes.

#### Constraints

<a id="E_MAXPOOL_FLOAT_CONSTR_X_0010"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_X_0010]` Rank of $X$
  - Statement: The rank of $X$ is at least $3$.
  - Rationale: The batch axis, the channel axis and at least one spatial axis are required.
<a id="E_MAXPOOL_FLOAT_CONSTR_X_0020"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_X_0020]` Window coverage
  - Statement: For every spatial axis $t$, either $dX_{t+2} \ge 1$ and $dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$, or $dX_{t+2} = 0$ and $p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$.
  - Rationale: Every window of the result covers at least one element of $X$.
<a id="E_MAXPOOL_FLOAT_CONSTR_X_0030"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_X_0030]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: floating-point tensor

The result of the max pooling of $X$.

#### Constraints

<a id="E_MAXPOOL_FLOAT_CONSTR_Y_0010"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_Y_0010]` Shape definition
  - Statement: $dY_0 = dX_0$, $dY_1 = dX_1$, and for every spatial axis $t$, $dY_{t+2} = \lfloor (dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} - dK_t)/s_t \rfloor + 1$.
- `[E_MAXPOOL_FLOAT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_CONSTR_X_0030</span></b>](#E_MAXPOOL_FLOAT_CONSTR_X_0030) on tensor $X$.

<a id="int"></a>

# **MaxPool** (int)

where int is in {`int8`, `int64`}.

The two types share one semantics: the operator selects one of the elements of the input tensor and performs no arithmetic on it, so that the result is the same for both, the comparison being the one of the signed integers they represent.

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \ge 3$
- $Y$: result of the max pooling of $X$
- $\text{Indices}$: optional tensor of the flattened indices of the elements of $X$ selected by the pooling windows

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `ceil_mode` is restricted to `0` | Transient |
| `[R4]` | Attribute `dilations` is restricted to $1$ along each spatial axis | Transient |
| `[R5]` | The output $\text{Indices}$ is not supported | Transient |

## Function

<a id="E_MAXPOOL_INT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_INT_FUNC_0010]</br></span>

Operator **MaxPool** applies max pooling to the input tensor $X$: it slides a window of the shape given by `kernel_shape` over the spatial axes of $X$, with the steps given by `strides`, and stores in $Y$ the maximum of the elements of $X$ covered by each window.

The first two axes of $X$ are the batch axis and the channel axis; they are not pooled. The remaining $rX - 2$ axes are the spatial axes, numbered from $0$ to $rX-3$ in the order of the axes of $X$. The number of spatial axes is denoted $m = rX - 2$; the operator is applicable to any $m \ge 1$.

For any [tensor index](./../common/definitions.md#tensor_index) $i = (n, c, j_0, \dots, j_{m-1})$ of the result $Y$:

$$
Y[n, c, j_0, \dots, j_{m-1}] = \max_{k_0, \dots, k_{m-1}} X_p[n, c, j_0 \cdot s_0 + k_0, \dots, j_{m-1} \cdot s_{m-1} + k_{m-1}]
$$

where
- $n \in [0, dX_0-1]$ is the batch index, $dX_0$ being the batch size of $X$,
- $c \in [0, dX_1-1]$ is the channel index, $dX_1$ being the number of channels of $X$,
- $j_t \in [0, dY_{t+2}-1]$ is the index along spatial axis $t$ of $Y$, for $t \in [0, m-1]$,
- $k_t \in [0, dK_t-1]$ is the index along spatial axis $t$ of the window, $dK_t$ being the value of `kernel_shape` along that axis,
- $s_t$ is the value of `strides` along spatial axis $t$,
- $X_p$ is the tensor obtained by padding $X$ with the values of `pads`, and $X_p[n, c, h_0, \dots, h_{m-1}]$ is one of its elements,
- the maximum is taken over the elements of $X$ covered by the window, the padded elements being excluded.

**The maximum of the integer values.** The maximum is the one of the values of the type, and it is the maximum of the signed integers they represent: for `int8`, the values range from $-128$ to $127$ and the maximum of $-128$ and $127$ is $127$; for `int64`, the values range from $-2^{63}$ to $2^{63}-1$. The comparison is the same for the two types, and the result is one of the elements of $X$, so that no overflow can occur.

**Padding.** The padding adds $p_{t,\text{begin}}$ elements before and $p_{t,\text{end}}$ elements after the input along spatial axis $t$, where $p_{t,\text{begin}}$ and $p_{t,\text{end}}$ are the two values of `pads` for that axis. The padded elements are not elements of $X$: they are excluded from the maximum, so that the maximum is taken over the elements of $X$ covered by the window only. A window that covers no element of $X$ is not a window of the result: the shape of $Y$ is such that every window covers at least one element of $X$.

**Shape of the result.** The spatial dimensions of $Y$ are

$$
dY_{t+2} = \left\lfloor \frac{dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} - dK_t}{s_t} \right\rfloor + 1
$$

for every $t \in [0, m-1]$, and $dY_0 = dX_0$, $dY_1 = dX_1$. The floor is the one of the `ceil_mode` attribute, which is restricted to $0$ by [<b><span style="font-family: 'Courier New', monospace">R3</span></b>](#R3); the formula above is the one of the ONNX definition with `ceil_mode` disabled and `dilations` equal to $1$.

**Tensors with no dimension.** The input tensor $X$ has at least three axes, since the batch axis, the channel axis and at least one spatial axis are required; a rank-0 tensor is not a valid input.

**Zero-sized dimensions.** A spatial dimension of $X$ may have size $0$. The formula above then gives a spatial dimension of $Y$ of size $0$ when $p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$, and the corresponding dimension of $Y$ is empty; otherwise the operator is not applicable, since a window would cover no element of $X$. A batch or channel dimension of size $0$ gives a dimension of $Y$ of size $0$, and $Y$ is empty.

**Ties.** When several elements of a window are equal to the maximum, the result is that maximum, and the choice of the element that realizes it is not observable in $Y$. The output $\text{Indices}$ would make that choice observable; it is not supported by this profile, as stated by [<b><span style="font-family: 'Courier New', monospace">R5</span></b>](#R5).

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `int8` type:

$$
X = \begin{bmatrix}\begin{bmatrix}
  -1 & 2 & -3 & 4 \\
  5 & -6 & 7 & -8 \\
  -9 & 10 & -11 & 12 \\
  13 & -14 & 15 & -16
\end{bmatrix}\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$ and `strides` $= [2, 2]$, so that $X$ has shape $(1, 1, 4, 4)$:

$$
Y = \begin{bmatrix}\begin{bmatrix}
  5 & 7 \\
  13 & 15
\end{bmatrix}\end{bmatrix}
$$

The window of the first element of $Y$ covers $-1, 2, 5, -6$, whose maximum is $5$; the window of the last element covers $-11, 12, 15, -16$, whose maximum is $15$. The negative values are smaller than the positive ones, and the result is one of the elements of $X$. The values of the example are values of `int8`; they are also values of `int64`, for which the comparison and the result are the same.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs no floating-point arithmetic |
| Overflow | not applicable: the result is one of the elements of $X$, and no arithmetic is performed |
| Integer overflow | not applicable: the result is one of the elements of $X$, and no arithmetic is performed |
| Division by zero | not applicable: the operator performs no division |
| A window that covers no element of $X$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_CONSTR_X_0020</span></b>](#E_MAXPOOL_INT_CONSTR_X_0020) |

No error condition.

## Attributes

The attributes of the operator are the ones given in the [real](#real) section; they do not depend on the type of the arguments.

## Inputs

### $\text{X}$: signed integer tensor

The tensor to which the max pooling is applied. Its first two axes are the batch axis and the channel axis, and its remaining axes are the spatial axes.

#### Constraints

<a id="E_MAXPOOL_INT_CONSTR_X_0010"></a>
- `[E_MAXPOOL_INT_CONSTR_X_0010]` Rank of $X$
  - Statement: The rank of $X$ is at least $3$.
  - Rationale: The batch axis, the channel axis and at least one spatial axis are required.
<a id="E_MAXPOOL_INT_CONSTR_X_0020"></a>
- `[E_MAXPOOL_INT_CONSTR_X_0020]` Window coverage
  - Statement: For every spatial axis $t$, either $dX_{t+2} \ge 1$ and $dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$, or $dX_{t+2} = 0$ and $p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$.
  - Rationale: Every window of the result covers at least one element of $X$.
<a id="E_MAXPOOL_INT_CONSTR_X_0030"></a>
- `[E_MAXPOOL_INT_CONSTR_X_0030]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: signed integer tensor

The result of the max pooling of $X$.

#### Constraints

<a id="E_MAXPOOL_INT_CONSTR_Y_0010"></a>
- `[E_MAXPOOL_INT_CONSTR_Y_0010]` Shape definition
  - Statement: $dY_0 = dX_0$, $dY_1 = dX_1$, and for every spatial axis $t$, $dY_{t+2} = \lfloor (dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} - dK_t)/s_t \rfloor + 1$.
- `[E_MAXPOOL_INT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_CONSTR_X_0030</span></b>](#E_MAXPOOL_INT_CONSTR_X_0030) on tensor $X$.

<a id="uint"></a>

# **MaxPool** (uint)

where uint is in {`uint8`}.

The types of this section share one semantics: the operator selects one of the elements of the input tensor and performs no arithmetic on it, so that the result is the same for every type of the family, the comparison being the one of the unsigned integers they represent.

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \ge 3$
- $Y$: result of the max pooling of $X$
- $\text{Indices}$: optional tensor of the flattened indices of the elements of $X$ selected by the pooling windows

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `ceil_mode` is restricted to `0` | Transient |
| `[R4]` | Attribute `dilations` is restricted to $1$ along each spatial axis | Transient |
| `[R5]` | The output $\text{Indices}$ is not supported | Transient |

## Function

<a id="E_MAXPOOL_UINT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_UINT_FUNC_0010]</br></span>

Operator **MaxPool** applies max pooling to the input tensor $X$: it slides a window of the shape given by `kernel_shape` over the spatial axes of $X$, with the steps given by `strides`, and stores in $Y$ the maximum of the elements of $X$ covered by each window.

The first two axes of $X$ are the batch axis and the channel axis; they are not pooled. The remaining $rX - 2$ axes are the spatial axes, numbered from $0$ to $rX-3$ in the order of the axes of $X$. The number of spatial axes is denoted $m = rX - 2$; the operator is applicable to any $m \ge 1$.

For any [tensor index](./../common/definitions.md#tensor_index) $i = (n, c, j_0, \dots, j_{m-1})$ of the result $Y$:

$$
Y[n, c, j_0, \dots, j_{m-1}] = \max_{k_0, \dots, k_{m-1}} X_p[n, c, j_0 \cdot s_0 + k_0, \dots, j_{m-1} \cdot s_{m-1} + k_{m-1}]
$$

where
- $n \in [0, dX_0-1]$ is the batch index, $dX_0$ being the batch size of $X$,
- $c \in [0, dX_1-1]$ is the channel index, $dX_1$ being the number of channels of $X$,
- $j_t \in [0, dY_{t+2}-1]$ is the index along spatial axis $t$ of $Y$, for $t \in [0, m-1]$,
- $k_t \in [0, dK_t-1]$ is the index along spatial axis $t$ of the window, $dK_t$ being the value of `kernel_shape` along that axis,
- $s_t$ is the value of `strides` along spatial axis $t$,
- $X_p$ is the tensor obtained by padding $X$ with the values of `pads`, and $X_p[n, c, h_0, \dots, h_{m-1}]$ is one of its elements,
- the maximum is taken over the elements of $X$ covered by the window, the padded elements being excluded.

**The maximum of the integer values.** The maximum is the one of the values of the type, and it is the maximum of the unsigned integers they represent: for `uint8`, the values range from $0$ to $255$ and the maximum of $0$ and $255$ is $255$. The result is one of the elements of $X$, so that no overflow can occur.

**Padding.** The padding adds $p_{t,\text{begin}}$ elements before and $p_{t,\text{end}}$ elements after the input along spatial axis $t$, where $p_{t,\text{begin}}$ and $p_{t,\text{end}}$ are the two values of `pads` for that axis. The padded elements are not elements of $X$: they are excluded from the maximum, so that the maximum is taken over the elements of $X$ covered by the window only. A window that covers no element of $X$ is not a window of the result: the shape of $Y$ is such that every window covers at least one element of $X$.

**Shape of the result.** The spatial dimensions of $Y$ are

$$
dY_{t+2} = \left\lfloor \frac{dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} - dK_t}{s_t} \right\rfloor + 1
$$

for every $t \in [0, m-1]$, and $dY_0 = dX_0$, $dY_1 = dX_1$. The floor is the one of the `ceil_mode` attribute, which is restricted to $0$ by [<b><span style="font-family: 'Courier New', monospace">R3</span></b>](#R3); the formula above is the one of the ONNX definition with `ceil_mode` disabled and `dilations` equal to $1$.

**Tensors with no dimension.** The input tensor $X$ has at least three axes, since the batch axis, the channel axis and at least one spatial axis are required; a rank-0 tensor is not a valid input.

**Zero-sized dimensions.** A spatial dimension of $X$ may have size $0$. The formula above then gives a spatial dimension of $Y$ of size $0$ when $p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$, and the corresponding dimension of $Y$ is empty; otherwise the operator is not applicable, since a window would cover no element of $X$. A batch or channel dimension of size $0$ gives a dimension of $Y$ of size $0$, and $Y$ is empty.

**Ties.** When several elements of a window are equal to the maximum, the result is that maximum, and the choice of the element that realizes it is not observable in $Y$. The output $\text{Indices}$ would make that choice observable; it is not supported by this profile, as stated by [<b><span style="font-family: 'Courier New', monospace">R5</span></b>](#R5).

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `uint8` type:

$$
X = \begin{bmatrix}\begin{bmatrix}
  0 & 200 & 3 & 255 \\
  17 & 128 & 9 & 64 \\
  250 & 1 & 100 & 7 \\
  33 & 254 & 2 & 90
\end{bmatrix}\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$ and `strides` $= [2, 2]$, so that $X$ has shape $(1, 1, 4, 4)$:

$$
Y = \begin{bmatrix}\begin{bmatrix}
  200 & 255 \\
  254 & 100
\end{bmatrix}\end{bmatrix}
$$

The window of the first element of $Y$ covers $0, 200, 17, 128$, whose maximum is $200$; the window of the second element covers $3, 255, 9, 64$, whose maximum is the largest value $255$ of `uint8`; the window of the third element covers $250, 1, 33, 254$, whose maximum is $254$; the window of the last element covers $100, 7, 2, 90$, whose maximum is $100$. The result is one of the elements of $X$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs no floating-point arithmetic |
| Overflow | not applicable: the result is one of the elements of $X$, and no arithmetic is performed |
| Integer overflow | not applicable: the result is one of the elements of $X$, and no arithmetic is performed |
| Division by zero | not applicable: the operator performs no division |
| A window that covers no element of $X$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_UINT_CONSTR_X_0020</span></b>](#E_MAXPOOL_UINT_CONSTR_X_0020) |

No error condition.

## Attributes

The attributes of the operator are the ones given in the [real](#real) section; they do not depend on the type of the arguments.

## Inputs

### $\text{X}$: unsigned integer tensor

The tensor to which the max pooling is applied. Its first two axes are the batch axis and the channel axis, and its remaining axes are the spatial axes.

#### Constraints

<a id="E_MAXPOOL_UINT_CONSTR_X_0010"></a>
- `[E_MAXPOOL_UINT_CONSTR_X_0010]` Rank of $X$
  - Statement: The rank of $X$ is at least $3$.
  - Rationale: The batch axis, the channel axis and at least one spatial axis are required.
<a id="E_MAXPOOL_UINT_CONSTR_X_0020"></a>
- `[E_MAXPOOL_UINT_CONSTR_X_0020]` Window coverage
  - Statement: For every spatial axis $t$, either $dX_{t+2} \ge 1$ and $dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$, or $dX_{t+2} = 0$ and $p_{t,\text{begin}} + p_{t,\text{end}} \ge dK_t$.
  - Rationale: Every window of the result covers at least one element of $X$.
<a id="E_MAXPOOL_UINT_CONSTR_X_0030"></a>
- `[E_MAXPOOL_UINT_CONSTR_X_0030]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: unsigned integer tensor

The result of the max pooling of $X$.

#### Constraints

<a id="E_MAXPOOL_UINT_CONSTR_Y_0010"></a>
- `[E_MAXPOOL_UINT_CONSTR_Y_0010]` Shape definition
  - Statement: $dY_0 = dX_0$, $dY_1 = dX_1$, and for every spatial axis $t$, $dY_{t+2} = \lfloor (dX_{t+2} + p_{t,\text{begin}} + p_{t,\text{end}} - dK_t)/s_t \rfloor + 1$.
- `[E_MAXPOOL_UINT_CONSTR_Y_0020]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_UINT_CONSTR_X_0030</span></b>](#E_MAXPOOL_UINT_CONSTR_X_0030) on tensor $X$.
