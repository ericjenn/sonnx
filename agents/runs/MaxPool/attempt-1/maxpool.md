# Contents

- **MaxPool** operator for type [real](#real)
- **MaxPool** operator for types [`float16`, `float`, `double`](#float)
- **MaxPool** operator for types [`int8`, `int64`](#int)
- **MaxPool** operator for type [`uint8`](#uint)

Based on ONNX documentation [MaxPool version 12](https://onnx.ai/onnx/operators/onnx__MaxPool.html#maxpool-12), the version of the operator in force in opset 14.

Revision 2026-10-03: first issue of this specification, based on ONNX opset 14.

<a id="real"></a>

# **MaxPool** (real)

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \geq 3$, whose shape is $(dX_0, dX_1, dX_2, ..., dX_{rX-1})$
- $Y$: output tensor, whose shape is $(dX_0, dX_1, dY_2, ..., dY_{rX-1})$
- $\text{Indices}$: optional output tensor of the same shape as $Y$, giving for each element of $Y$ the index of the element of $X$ it is the maximum of

The first two axes of $X$ are the batch axis and the channel axis; the remaining $rX-2$ axes are the spatial axes. The operator is applied independently on each batch and each channel, and the pooling window slides along the spatial axes.

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Input tensor $X$ has at least 3 axes | Transient |
| `[R2]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R3]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R4]` | Attribute `dilations` is restricted to `1` along each spatial axis | Transient |
| `[R5]` | Attribute `ceil_mode` is restricted to `0` | Transient |

## Function

<a id="E_MAXPOOL_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_REAL_FUNC_0010]</br></span>

Operator **MaxPool** applies max pooling to the input tensor $X$: the spatial axes of $X$ are partitioned into windows of the size given by the `kernel_shape` attribute, the windows being placed at the positions given by the `strides` attribute, and each element of the output tensor $Y$ is the maximum of the elements of $X$ that fall in the corresponding window. The elements of a window that fall outside $X$, i.e. the elements added by the padding, are excluded from the maximum.

Let $s = rX - 2$ be the number of spatial axes of $X$, and let $k_j$, $p_j$ and $t_j$ be the values of the `kernel_shape`, `pads` and `strides` attributes along the spatial axis $j$, for $j \in [0, s-1]$. The shape of $Y$ is $(dX_0, dX_1, dY_2, ..., dY_{rX-1})$, where, for each spatial axis $j$,

$$
dY_{j+2} = \left\lfloor \frac{dX_{j+2} + p_j - k_j}{t_j} \right\rfloor + 1
$$

For any [tensor index](./../common/definitions.md#tensor_index) $(b, c, m_2, ..., m_{rX-1})$ of $Y$:

$$
Y[b, c, m_2, ..., m_{rX-1}] = \max_{u_2 \in W_2, ..., u_{rX-1} \in W_{rX-1}} X[b, c, u_2, ..., u_{rX-1}]
$$

where
- $b \in [0, dX_0-1]$ is the batch index and $c \in [0, dX_1-1]$ is the channel index,
- $m_{j+2} \in [0, dY_{j+2}-1]$ is the index of the window along the spatial axis $j$,
- $W_{j+2}$ is the set of indices of the spatial axis $j$ that the window $m_{j+2}$ covers, i.e. the set of the integers $u$ such that $m_{j+2} \cdot t_j - p_j \le u \le m_{j+2} \cdot t_j - p_j + k_j - 1$ and $0 \le u \le dX_{j+2}-1$,
- $\max$ is the maximum of the real numbers $X[b, c, u_2, ..., u_{rX-1}]$ for $u_2 \in W_2, ..., u_{rX-1} \in W_{rX-1}$.

The maximum is exact: $Y[b, c, m_2, ..., m_{rX-1}]$ is one of the elements of $X$ that the window covers, and no approximation is introduced.

**The window is never empty.** The padding is at most $k_j - 1$ on each side of the spatial axis $j$, so that every window covers at least one index of $X$ and the maximum above is taken over a non-empty set. A window that covers no index of $X$ is not a case of this operator.

**The maximum of a window with several occurrences of the same value.** When several elements of a window are equal to the maximum, the value of $Y$ is that value, and the value of $\text{Indices}$ is the index of any one of them; which one is left to the implementer.

**Tensors with no dimension.** The input tensor $X$ has at least 3 axes, so that a rank-0 tensor is not an operand of this operator.

**Zero-sized dimensions.** A spatial dimension of $X$ may have size $0$. The output shape is then computed by the formula above, and the window covers no index of $X$: the operator is not applicable, and the case is ruled out by the constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0020</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0020) on the shape of $X$ and by the constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_pads_0030</span></b>](#E_MAXPOOL_REAL_CONSTR_pads_0030) on the padding, which together ensure that every window covers at least one element of $X$.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

<a id="E_MAXPOOL_REAL_FUNC_0020"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_REAL_FUNC_0020]</br></span>

**The `Indices` output.** When the optional output $\text{Indices}$ is present, each of its elements is the linear index, in row-major order, of the element of $X$ that the corresponding window selects. For any [tensor index](./../common/definitions.md#tensor_index) $(b, c, m_2, ..., m_{rX-1})$ of $Y$:

$$
\text{Indices}[b, c, m_2, ..., m_{rX-1}] = b \cdot \prod_{l=1}^{rX-1} dX_l \;+\; c \cdot \prod_{l=2}^{rX-1} dX_l \;+\; \sum_{j=2}^{rX-1} u_j \cdot \prod_{l=j+1}^{rX-1} dX_l
$$

where
- $(u_2, ..., u_{rX-1})$ is the index of the element of $X$ that the window selects, i.e. the element such that $u_{j+2} \in W_{j+2}$ for each spatial axis $j$ and $X[b, c, u_2, ..., u_{rX-1}] = Y[b, c, m_2, ..., m_{rX-1}]$,
- an empty product is $1$, so that the last term of the sum is $u_{rX-1}$,
- the index is the index in $X$ and not in the padded tensor: the elements added by the padding are not counted, so that the value lies in $[0, dX_0 \cdot dX_1 \cdots dX_{rX-1} - 1]$.

When several elements of the window are equal to the maximum, the index is that of any one of them; which one is left to the implementer.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix}
  1 & 2 & 3 & 4 \\
  5 & 6 & 7 & 8 \\
  9 & 10 & 11 & 12 \\
  13 & 14 & 15 & 16
\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$, `strides` $= [2, 2]$, `pads` $= [0, 0, 0, 0]$, so that $dY_2 = \lfloor (4+0-2)/2 \rfloor + 1 = 2$ and $dY_3 = 2$:

$$
Y = \begin{bmatrix}
  6 & 8 \\
  14 & 16
\end{bmatrix}
$$

### Example 2

$$
X = \begin{bmatrix}
  1 & 2 & 3 \\
  4 & 5 & 6 \\
  7 & 8 & 9
\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$, `strides` $= [1, 1]$, `pads` $= [1, 1, 0, 0]$, so that $dY_2 = \lfloor (3+1-2)/1 \rfloor + 1 = 3$ and $dY_3 = \lfloor (3+1-2)/1 \rfloor + 1 = 3$. The padding adds one row of elements above $X$ and one column of elements to its left; those elements are excluded from the maximum:

$$
Y = \begin{bmatrix}
  1 & 2 & 3 \\
  4 & 5 & 6 \\
  7 & 8 & 9
\end{bmatrix}
$$

The first window covers the padded row and the padded column, whose elements are excluded, and the single element $X[0,0] = 1$; the maximum of that window is $1$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator computes a maximum, which is not one of the operations of IEEE 754 section 7.2 |
| Overflow | not applicable: the result is one of the elements of $X$, and no arithmetic is performed |
| Integer overflow | not applicable: the result is one of the elements of $X$, and no arithmetic is performed |
| Division by zero | not applicable: the operator performs no division |
| A window that covers no element of $X$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0020</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0020) on tensor $X$ |

No error condition.

## Attributes

### `auto_pad`: `string`

`auto_pad` gives the way the padding is computed. Its value is `NOTSET`, `SAME_UPPER`, `SAME_LOWER` or `VALID`; the default value is `NOTSET`, which means that the padding is the one given by the `pads` attribute.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_auto_pad_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_auto_pad_0010]` Value of `auto_pad`
  - Statement: The value of `auto_pad` is `NOTSET`.
  - Rationale: Restriction `[R2]`; the other values are not specified by this profile.

### `ceil_mode`: `int`

`ceil_mode` selects the way the output shape is computed: the value $0$ selects the floor, the value $1$ the ceiling. The default value is $0$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_ceil_mode_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_ceil_mode_0010]` Value of `ceil_mode`
  - Statement: The value of `ceil_mode` is $0$.
  - Rationale: Restriction `[R5]`; the output shape is then the one given by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0010</span></b>](#E_MAXPOOL_REAL_FUNC_0010).

### `dilations`: `list of ints`

`dilations` gives the dilation value along each spatial axis of the kernel. The default value is $1$ along each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_dilations_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_dilations_0010]` Value of `dilations`
  - Statement: The value of `dilations` is $1$ along each spatial axis.
  - Rationale: Restriction `[R4]`; the kernel then covers $k_j$ consecutive indices along the spatial axis $j$.

### `kernel_shape`: `list of ints`

`kernel_shape` gives the size of the kernel along each spatial axis. It has $rX-2$ elements, and each of them is at least $1$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_kernel_shape_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_kernel_shape_0010]` Size of `kernel_shape`
  - Statement: The number of elements of `kernel_shape` is $rX-2$, the number of spatial axes of $X$.
<a id="E_MAXPOOL_REAL_CONSTR_kernel_shape_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_kernel_shape_0020]` Value of `kernel_shape`
  - Statement: Each element of `kernel_shape` is at least $1$.

### `pads`: `list of ints`

`pads` gives the number of elements added at the beginning and at the end of each spatial axis. It has $2(rX-2)$ elements, the first $rX-2$ of them being the numbers added at the beginning of the spatial axes and the last $rX-2$ the numbers added at their end. The default value is $0$ along the beginning and the end of each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_pads_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_pads_0010]` Size of `pads`
  - Statement: The number of elements of `pads` is $2(rX-2)$, twice the number of spatial axes of $X$.
<a id="E_MAXPOOL_REAL_CONSTR_pads_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_pads_0020]` Value of `pads`
  - Statement: Each element of `pads` is at least $0$.
<a id="E_MAXPOOL_REAL_CONSTR_pads_0030"></a>
- `[E_MAXPOOL_REAL_CONSTR_pads_0030]` Padding smaller than the kernel
  - Statement: For each spatial axis $j$, the number of elements added at the beginning of the axis and the number added at its end are each at most $k_j - 1$, where $k_j$ is the element of `kernel_shape` for that axis.
  - Rationale: Every window then covers at least one element of $X$, so that the maximum of [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0010</span></b>](#E_MAXPOOL_REAL_FUNC_0010) is taken over a non-empty set.

### `storage_order`: `int`

`storage_order` gives the storage order of the tensor, $0$ for row major and $1$ for column major. It is used only to convert an $n$-tuple index into a single integer for the `Indices` output. The default value is $0$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_storage_order_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_storage_order_0010]` Value of `storage_order`
  - Statement: The value of `storage_order` is $0$.
  - Rationale: Restriction `[R3]`; the index of an element of $X$ is then the one given by [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0020</span></b>](#E_MAXPOOL_REAL_FUNC_0020).

### `strides`: `list of ints`

`strides` gives the stride along each spatial axis. It has $rX-2$ elements, and each of them is at least $1$. The default value is $1$ along each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_strides_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_strides_0010]` Size of `strides`
  - Statement: The number of elements of `strides` is $rX-2$, the number of spatial axes of $
