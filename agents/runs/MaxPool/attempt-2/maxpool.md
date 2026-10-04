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
- $X$: input tensor of rank $rX \geq 1$, whose first two axes are the batch axis and the channel axis and whose remaining $rX - 2$ axes are the spatial axes
- $Y$: output tensor, of the same rank as $X$
- $\text{Indices}$: optional output tensor of the same shape as $Y$, giving for each element of $Y$ the index of the element of $X$ it is the maximum of

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `dilations` is restricted to `1` along each spatial axis | Transient |
| `[R4]` | The output `Indices` is not supported | Transient |

## Function

<a id="E_MAXPOOL_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_REAL_FUNC_0010]</br></span>

Operator **MaxPool** applies max pooling to the input tensor $X$: it slides a window of the shape given by the `kernel_shape` attribute over the spatial axes of $X$, with the steps given by the `strides` attribute, and stores in $Y$ the maximum of the elements of $X$ covered by the window. The elements of the padding, which are not elements of $X$, are excluded from the maximum.

Let $n = rX - 2$ be the number of spatial axes of $X$, and let $dX_0, \dots, dX_{rX-1}$ be the dimensions of $X$. For any [tensor index](./../common/definitions.md#tensor_index) $i = (i_0, \dots, i_{rX-1})$ of the result $Y$:

$$
Y[i] = \max \left\{ X\left[i_0, i_1, i_2 \cdot s_0 + j_0 - p_0, \dots, i_{n+1} \cdot s_{n-1} + j_{n-1} - p_{n-1}\right] \;\middle|\; 0 \le j_k \le k_k - 1,\; 0 \le i_{k+2} \cdot s_k + j_k - p_k \le dX_{k+2}-1 \right\}
$$

where
- $\max$ is the maximum of the real numbers, and the set it is applied to is the set of the elements of $X$ that the window covers,
- $k_k$ is the value of the `kernel_shape` attribute along spatial axis $k$,
- $s_k$ is the value of the `strides` attribute along spatial axis $k$,
- $p_k$ is the number of elements of padding added at the beginning of spatial axis $k$, i.e. the value `pads[k]` of the `pads` attribute,
- $i_0$ and $i_1$ are the batch index and the channel index, which the window does not move along.

The window covers the elements of $X$ whose index along spatial axis $k$ lies in $[i_{k+2} \cdot s_k - p_k,\, i_{k+2} \cdot s_k - p_k + k_k - 1]$. The elements of that interval that lie outside $[0, dX_{k+2}-1]$ are the padding, and they are excluded from the maximum: the maximum is taken over the elements of $X$ that the window covers, and the window always covers at least one of them, as stated by the constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0020</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0020).

**The shape of the output.** The dimensions of $Y$ along the batch axis and the channel axis are those of $X$: $dY_0 = dX_0$ and $dY_1 = dX_1$. Along each spatial axis $k$, the dimension of $Y$ is

$$
dY_{k+2} = \text{round}\left(\frac{dX_{k+2} + p_k + q_k - k_k}{s_k}\right) + 1
$$

where $q_k$ is the number of elements of padding added at the end of spatial axis $k$, i.e. the value `pads[n + k]` of the `pads` attribute, and $\text{round}(x)$ is the value of $x$ rounded towards $-\infty$ when the `ceil_mode` attribute is $0$ and towards $+\infty$ when it is $1$. The numerator is non-negative, as stated by the constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0020</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0020), so that the quotient is non-negative and $dY_{k+2} \ge 1$.

**Tensors with no spatial axis.** A tensor $X$ of rank $1$ or $2$ has no spatial axis. The window then covers a single element, the maximum is that element, and $Y$ has the same shape as $X$.

**Zero-sized dimensions.** A dimension of $X$ may have size $0$. The constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0020</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0020) then requires the corresponding dimension of $Y$ to be $0$ as well, and $Y$ is empty; no window is formed and no maximum is taken.

**Ties.** When several elements of $X$ covered by a window are equal to the maximum, the result is that maximum, and the choice of the element it is taken from is left to the implementer. The output $Y$ is the same whichever element is chosen; the choice is observable only through the output `Indices`, which the restriction `[R4]` excludes.

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

with `kernel_shape` $= [2, 2]$, `strides` $= [2, 2]$ and `pads` $= [0, 0, 0, 0]$:

$$
Y = \begin{bmatrix}
  6 & 8 \\
  14 & 16
\end{bmatrix}
$$

The window of the first element of $Y$ covers the four elements of the upper left corner of $X$, whose maximum is $6$.

### Example 2

$$
X = \begin{bmatrix} 1 & 2 & 3 & 4 \end{bmatrix}
$$

with `kernel_shape` $= [3]$, `strides` $= [1]$ and `pads` $= [1, 1]$:

$$
Y = \begin{bmatrix} 2 & 3 & 4 & 4 \end{bmatrix}
$$

The window of the first element of $Y$ covers the padding at the beginning of the axis and the elements $1$ and $2$ of $X$; the padding is excluded from the maximum, which is $2$. The window of the last element covers the elements $3$ and $4$ and the padding at the end; the maximum is $4$.

### Example 3

$$
X = \begin{bmatrix} 1 & 2 & 3 & 4 \end{bmatrix}
$$

with `kernel_shape` $= [3]$, `strides` $= [2]$, `pads` $= [0, 0]$ and `ceil_mode` $= 1$:

$$
Y = \begin{bmatrix} 3 & 4 \end{bmatrix}
$$

The dimension of $Y$ is $\lceil (4 - 3)/2 \rceil + 1 = 2$. The window of the second element covers the element $4$ of $X$ and the two positions beyond the end of the axis, which are padding; the maximum is $4$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs no floating-point arithmetic, and the maximum of a set of values is one of them |
| Overflow | not applicable: the operator performs no arithmetic, and the result is an element of the input |
| Integer overflow | not applicable: the operator performs no arithmetic, and the result is an element of the input |
| Division by zero | not applicable: the operator performs no division |
| A window that covers no element of $X$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0020</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0020) |
| A negative dimension of $Y$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_CONSTR_X_0020</span></b>](#E_MAXPOOL_REAL_CONSTR_X_0020) |

No error condition.

## Attributes

### `auto_pad`: `string`

The padding mode. The restriction `[R1]` limits it to `NOTSET`, which means that the padding is the one given by the `pads` attribute.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_auto_pad_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_auto_pad_0010]` Value of `auto_pad`
  - Statement: The value of `auto_pad` is `NOTSET`.
  - Rationale: The restriction `[R1]` limits the attribute to `NOTSET`; the padding modes `SAME_UPPER`, `SAME_LOWER` and `VALID` are not supported.

### `ceil_mode`: `int`

The rounding of the quotient that gives the dimensions of $Y$ along the spatial axes: $0$ rounds it towards $-\infty$ and $1$ towards $+\infty$. The default value is $0$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_ceil_mode_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_ceil_mode_0010]` Value of `ceil_mode`
  - Statement: The value of `ceil_mode` is $0$ or $1$.

### `dilations`: `list of int`

The dilation along each spatial axis. The restriction `[R3]` limits it to $1$ along each spatial axis, so that the elements covered by a window are contiguous.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_dilations_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_dilations_0010]` Value of `dilations`
  - Statement: The value of `dilations` is $1$ along each spatial axis.
  - Rationale: The restriction `[R3]` limits the attribute to $1$; a dilation greater than $1$ is not supported.

### `kernel_shape`: `list of int`

The size of the window along each spatial axis. The attribute is required.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_kernel_shape_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_kernel_shape_0010]` Length of `kernel_shape`
  - Statement: The length of `kernel_shape` is $rX - 2$.
<a id="E_MAXPOOL_REAL_CONSTR_kernel_shape_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_kernel_shape_0020]` Value of `kernel_shape`
  - Statement: Every value of `kernel_shape` is greater than or equal to $1$.

### `pads`: `list of int`

The padding added at the beginning and at the end of each spatial axis, in the form $[p_0, \dots, p_{n-1}, q_0, \dots, q_{n-1}]$, where $p_k$ is the number of elements added at the beginning of spatial axis $k$ and $q_k$ the number added at the end. The default value is $0$ along each side of each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_pads_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_pads_0010]` Length of `pads`
  - Statement: The length of `pads` is $2 \cdot (rX - 2)$.
<a id="E_MAXPOOL_REAL_CONSTR_pads_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_pads_0020]` Value of `pads`
  - Statement: Every value of `pads` is greater than or equal to $0$.

### `storage_order`: `int`

The storage order used to convert an $n$-tuple index into a single integer for the output `Indices`. The restriction `[R2]` limits it to $0$, the row-major order. The default value is $0$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_storage_order_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_storage_order_0010]` Value of `storage_order`
  - Statement: The value of `storage_order` is $0$.
  - Rationale: The restriction `[R2]` limits the attribute to $0$; the column-major order is not supported.

### `strides`: `list of int`

The step of the window along each spatial axis. The default value is $1$ along each spatial axis.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_strides_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_strides_0010]` Length of `strides`
  - Statement: The length of `strides` is $rX - 2$.
<a id="E_MAXPOOL_REAL_CONSTR_strides_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_strides_0020]` Value of `strides`
  - Statement: Every value of `strides` is greater than or equal to $1$.

## Inputs

### $\text{X}$: real tensor

The input data tensor, whose first two axes are the batch axis and the channel axis and whose remaining axes are the spatial axes.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_X_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_X_0010]` Rank of $X$
  - Statement: The rank of $X$ is greater than or equal to $1$.
<a id="E_MAXPOOL_REAL_CONSTR_X_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_X_0020]` Window within the input
  - Statement: For every spatial axis $k$ and every index $i_{k+2}$ of $Y$ along that axis, the window covers at least one element of $X$, i.e. the interval $[i_{k+2} \cdot s_k - p_k,\, i_{k+2} \cdot s_k - p_k + k_k - 1]$ meets $[0, dX_{k+2}-1]$.
  - Rationale: The maximum of an empty set of elements is not defined; the constraint rules out the windows that cover no element of $X$, and with them the negative dimensions of $Y$.

## Outputs

### $\text{Y}$: real tensor

The result of the max pooling of $X$.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_Y_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_Y_0010]` Rank of $Y$
  - Statement: The rank of $Y$ is the rank of $X$.
<a id="E_MAXPOOL_REAL_CONSTR_Y_0020"></a>
- `[E_MAXPOOL_REAL_CONSTR_Y_0020]` Shape of $Y$
  - Statement: $dY_0 = dX_0$, $dY_1 = dX_1$, and for every spatial axis $k$, $dY_{k+2}$ is the value given by the formula of [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_REAL_FUNC_0010</span></b>](#E_MAXPOOL_REAL_FUNC_0010).

### $\text{Indices}$: `int64` tensor

The index of the element of $X$ that each element of $Y$ is the maximum of. The output is optional and the restriction `[R4]` excludes it.

#### Constraints

<a id="E_MAXPOOL_REAL_CONSTR_Indices_0010"></a>
- `[E_MAXPOOL_REAL_CONSTR_Indices_0010]` Absence of `Indices`
  - Statement: The output `Indices` is absent.
  - Rationale: The restriction `[R4]` excludes the output.

<a id="float"></a>

# **MaxPool** (float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: the operator selects an element of the input and performs no arithmetic on it, so that the special values of IEEE 754 are compared and propagated as the values they are, and the types differ only by their precision and by the range of the values they represent.

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \geq 1$, whose first two axes are the batch axis and the channel axis and whose remaining $rX - 2$ axes are the spatial axes
- $Y$: output tensor, of the same rank as $X$
- $\text{Indices}$: optional output tensor of the same shape as $Y$, giving for each element of $Y$ the index of the element of $X$ it is the maximum of

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `dilations` is restricted to `1` along each spatial axis | Transient |
| `[R4]` | The output `Indices` is not supported | Transient |

## Function

<a id="E_MAXPOOL_FLOAT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_FLOAT_FUNC_0010]</br></span>

Operator **MaxPool** applies max pooling to the input tensor $X$: it slides a window of the shape given by the `kernel_shape` attribute over the spatial axes of $X$, with the steps given by the `strides` attribute, and stores in $Y$ the maximum of the elements of $X$ covered by the window. The elements of the padding, which are not elements of $X$, are excluded from the maximum.

Let $n = rX - 2$ be the number of spatial axes of $X$, and let $dX_0, \dots, dX_{rX-1}$ be the dimensions of $X$. For any [tensor index](./../common/definitions.md#tensor_index) $i = (i_0, \dots, i_{rX-1})$ of the result $Y$:

$$
Y[i] = \max_{(\text{f})} \left\{ X\left[i_0, i_1, i_2 \cdot s_0 + j_0 - p_0, \dots, i_{n+1} \cdot s_{n-1} + j_{n-1} - p_{n-1}\right] \;\middle|\; 0 \le j_k \le k_k - 1,\; 0 \le i_{k+2} \cdot s_k + j_k - p_k \le dX_{k+2}-1 \right\}
$$

where
- $\max_{(\text{f})}$ is the maximum of the floating-point type of $X$ and $Y$, i.e. $\max_{(\text{f16})}$, $\max_{(\text{f32})}$ or $\max_{(\text{f64})}$ according to that type, and the set it is applied to is the set of the elements of $X$ that the window covers,
- $k_k$ is the value of the `kernel_shape` attribute along spatial axis $k$,
- $s_k$ is the value of the `strides` attribute along spatial axis $k$,
- $p_k$ is the number of elements of padding added at the beginning of spatial axis $k$, i.e. the value `pads[k]` of the `pads` attribute,
- $i_0$ and $i_1$ are the batch index and the channel index, which the window does not move along.

The window covers the elements of $X$ whose index along spatial axis $k$ lies in $[i_{k+2} \cdot s_k - p_k,\, i_{k+2} \cdot s_k - p_k + k_k - 1]$. The elements of that interval that lie outside $[0, dX_{k+2}-1]$ are the padding, and they are excluded from the maximum: the maximum is taken over the elements of $X$ that the window covers, and the window always covers at least one of them, as stated by the constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_CONSTR_X_0020</span></b>](#E_MAXPOOL_FLOAT_CONSTR_X_0020).

**The maximum of the floating-point type.** The result is one of the elements of $X$ that the window covers, and it is that element itself: the operator performs no arithmetic and no rounding. The maximum is the one of the order of the type, in which $-0$ and $+0$ are equal, every finite value is less than $+\text{inf}$, every finite value is greater than $-\text{inf}$, and NaN is not ordered with respect to any value. Consequently:
- when the window covers a NaN, the result is a quiet NaN of the type of $X$ and $Y$, and every quiet NaN of that type is a conforming result; its sign and its payload are not specified;
- when the window covers no NaN, the result is the greatest of the elements it covers, and it is one of them, with its sign and its payload;
- when the window covers both $-0$ and $+0$ and no greater value, the result is one of the two, and the choice is left to the implementer.

**The shape of the output.** The dimensions of $Y$ along the batch axis and the channel axis are those of $X$: $dY_0 = dX_0$ and $dY_1 = dX_1$. Along each spatial axis $k$, the dimension of $Y$ is

$$
dY_{k+2} = \text{round}\left(\frac{dX_{k+2} + p_k + q_k - k_k}{s_k}\right) + 1
$$

where $q_k$ is the number of elements of padding added at the end of spatial axis $k$, i.e. the value `pads[n + k]` of the `pads` attribute, and $\text{round}(x)$ is the value of $x$ rounded towards $-\infty$ when the `ceil_mode` attribute is $0$ and towards $+\infty$ when it is $1$. The numerator is non-negative, as stated by the constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_CONSTR_X_0020</span></b>](#E_MAXPOOL_FLOAT_CONSTR_X_0020), so that the quotient is non-negative and $dY_{k+2} \ge 1$.

**Tensors with no spatial axis.** A tensor $X$ of rank $1$ or $2$ has no spatial axis. The window then covers a single element, the maximum is that element, and $Y$ has the same shape as $X$.

**Zero-sized dimensions.** A dimension of $X$ may have size $0$. The constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_CONSTR_X_0020</span></b>](#E_MAXPOOL_FLOAT_CONSTR_X_0020) then requires the corresponding dimension of $Y$ to be $0$ as well, and $Y$ is empty; no window is formed and no maximum is taken.

**Ties.** When several elements of $X$ covered by a window are equal to the maximum, the result is that maximum, and the choice of the element it is taken from is left to the implementer. The output $Y$ is the same whichever element is chosen; the choice is observable only through the output `Indices`, which the restriction `[R4]` excludes.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

$$
X = \begin{bmatrix}
  1.5 & 2.5 & 3.5 & 4.5 \\
  5.5 & 6.5 & 7.5 & 8.5 \\
  9.5 & 10.5 & 11.5 & 12.5 \\
  13.5 & 14.5 & 15.5 & 16.5
\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$, `strides` $= [2, 2]$ and `pads` $= [0, 0, 0, 0]$:

$$
Y = \begin{bmatrix}
  6.5 & 8.5 \\
  14.5 & 16.5
\end{bmatrix}
$$

The window of the first element of $Y$ covers the four elements of the upper left corner of $X$, whose maximum is $6.5$.

### Example 2

$$
X = \begin{bmatrix} 1.0 & \text{NaN} & 3.0 & \text{+inf} \end{bmatrix}
$$

with `kernel_shape` $= [2]$, `strides` $= [2]$ and `pads` $= [0, 0]$:

$$
Y = \begin{bmatrix} \text{NaN} & \text{+inf} \end{bmatrix}
$$

The first window covers $1.0$ and NaN; NaN is not ordered with respect to any value, and the result is a quiet NaN. The second window covers $3.0$ and $+\text{inf}$; the result is $+\text{inf}$, which is greater than every finite value.

### Example 3

$$
X = \begin{bmatrix} \text{-0} & \text{+0} & \text{-inf} & \text{-1.0} \end{bmatrix}
$$

with `kernel_shape` $= [2]$, `strides` $= [2]$ and `pads` $= [0, 0]$:

$$
Y = \begin{bmatrix} \text{+0} & \text{-1.0} \end{bmatrix}
$$

The first window covers $-0$ and $+0$, which are equal; the result is one of the two, and the choice is left to the implementer. The second window covers $-\text{inf}$ and $-1.0$; the result is $-1.0$, which is greater than $-\text{inf}$.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs no floating-point arithmetic, and the maximum of a set of values is one of them |
| Overflow | not applicable: the operator performs no arithmetic, and the result is an element of the input |
| Integer overflow | not applicable: the operator performs no arithmetic, and the result is an element of the input |
| Division by zero | not applicable: the operator performs no division |
| A window that covers no element of $X$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_CONSTR_X_0020</span></b>](#E_MAXPOOL_FLOAT_CONSTR_X_0020) |
| A negative dimension of $Y$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_CONSTR_X_0020</span></b>](#E_MAXPOOL_FLOAT_CONSTR_X_0020) |

No error condition.

## Attributes

The attributes of the operator are those given in the section [Attributes](#real) of the specification for real numbers; they do not depend on the type of the arguments.

## Inputs

### $\text{X}$: floating-point tensor

The input data tensor, whose first two axes are the batch axis and the channel axis and whose remaining axes are the spatial axes.

#### Constraints

<a id="E_MAXPOOL_FLOAT_CONSTR_X_0010"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_X_0010]` Rank of $X$
  - Statement: The rank of $X$ is greater than or equal to $1$.
<a id="E_MAXPOOL_FLOAT_CONSTR_X_0020"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_X_0020]` Window within the input
  - Statement: For every spatial axis $k$ and every index $i_{k+2}$ of $Y$ along that axis, the window covers at least one element of $X$, i.e. the interval $[i_{k+2} \cdot s_k - p_k,\, i_{k+2} \cdot s_k - p_k + k_k - 1]$ meets $[0, dX_{k+2}-1]$.
  - Rationale: The maximum of an empty set of elements is not defined; the constraint rules out the windows that cover no element of $X$, and with them the negative dimensions of $Y$.
<a id="E_MAXPOOL_FLOAT_CONSTR_X_0030"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_X_0030]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: floating-point tensor

The result of the max pooling of $X$.

#### Constraints

<a id="E_MAXPOOL_FLOAT_CONSTR_Y_0010"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_Y_0010]` Rank of $Y$
  - Statement: The rank of $Y$ is the rank of $X$.
<a id="E_MAXPOOL_FLOAT_CONSTR_Y_0020"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_Y_0020]` Shape of $Y$
  - Statement: $dY_0 = dX_0$, $dY_1 = dX_1$, and for every spatial axis $k$, $dY_{k+2}$ is the value given by the formula of [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_FUNC_0010</span></b>](#E_MAXPOOL_FLOAT_FUNC_0010).
- `[E_MAXPOOL_FLOAT_CONSTR_Y_0030]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_FLOAT_CONSTR_X_0030</span></b>](#E_MAXPOOL_FLOAT_CONSTR_X_0030) on tensor $X$.

### $\text{Indices}$: `int64` tensor

The index of the element of $X$ that each element of $Y$ is the maximum of. The output is optional and the restriction `[R4]` excludes it.

#### Constraints

<a id="E_MAXPOOL_FLOAT_CONSTR_Indices_0010"></a>
- `[E_MAXPOOL_FLOAT_CONSTR_Indices_0010]` Absence of `Indices`
  - Statement: The output `Indices` is absent.
  - Rationale: The restriction `[R4]` excludes the output.

<a id="int"></a>

# **MaxPool** (int)

where int is in {`int8`, `int64`}.

The two types share one semantics: the operator selects an element of the input and performs no arithmetic on it, so that the result is that element itself, and the two types differ only by the range of the values they represent.

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \geq 1$, whose first two axes are the batch axis and the channel axis and whose remaining $rX - 2$ axes are the spatial axes
- $Y$: output tensor, of the same rank as $X$
- $\text{Indices}$: optional output tensor of the same shape as $Y$, giving for each element of $Y$ the index of the element of $X$ it is the maximum of

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `dilations` is restricted to `1` along each spatial axis | Transient |
| `[R4]` | The output `Indices` is not supported | Transient |

## Function

<a id="E_MAXPOOL_INT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_INT_FUNC_0010]</br></span>

Operator **MaxPool** applies max pooling to the input tensor $X$: it slides a window of the shape given by the `kernel_shape` attribute over the spatial axes of $X$, with the steps given by the `strides` attribute, and stores in $Y$ the maximum of the elements of $X$ covered by the window. The elements of the padding, which are not elements of $X$, are excluded from the maximum.

Let $n = rX - 2$ be the number of spatial axes of $X$, and let $dX_0, \dots, dX_{rX-1}$ be the dimensions of $X$. For any [tensor index](./../common/definitions.md#tensor_index) $i = (i_0, \dots, i_{rX-1})$ of the result $Y$:

$$
Y[i] = \max \left\{ X\left[i_0, i_1, i_2 \cdot s_0 + j_0 - p_0, \dots, i_{n+1} \cdot s_{n-1} + j_{n-1} - p_{n-1}\right] \;\middle|\; 0 \le j_k \le k_k - 1,\; 0 \le i_{k+2} \cdot s_k + j_k - p_k \le dX_{k+2}-1 \right\}
$$

where
- $\max$ is the maximum of the integers, and the set it is applied to is the set of the elements of $X$ that the window covers,
- $k_k$ is the value of the `kernel_shape` attribute along spatial axis $k$,
- $s_k$ is the value of the `strides` attribute along spatial axis $k$,
- $p_k$ is the number of elements of padding added at the beginning of spatial axis $k$, i.e. the value `pads[k]` of the `pads` attribute,
- $i_0$ and $i_1$ are the batch index and the channel index, which the window does not move along.

The window covers the elements of $X$ whose index along spatial axis $k$ lies in $[i_{k+2} \cdot s_k - p_k,\, i_{k+2} \cdot s_k - p_k + k_k - 1]$. The elements of that interval that lie outside $[0, dX_{k+2}-1]$ are the padding, and they are excluded from the maximum: the maximum is taken over the elements of $X$ that the window covers, and the window always covers at least one of them, as stated by the constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_CONSTR_X_0020</span></b>](#E_MAXPOOL_INT_CONSTR_X_0020).

**The result is an element of the input.** The operator performs no arithmetic and no rounding: the result is one of the elements of $X$ that the window covers, with the value it has in the type of $X$ and $Y$. The order is the order of the integers, so that the two signed types have the same semantics and differ only by the range of the values they represent.

**The shape of the output.** The dimensions of $Y$ along the batch axis and the channel axis are those of $X$: $dY_0 = dX_0$ and $dY_1 = dX_1$. Along each spatial axis $k$, the dimension of $Y$ is

$$
dY_{k+2} = \text{round}\left(\frac{dX_{k+2} + p_k + q_k - k_k}{s_k}\right) + 1
$$

where $q_k$ is the number of elements of padding added at the end of spatial axis $k$, i.e. the value `pads[n + k]` of the `pads` attribute, and $\text{round}(x)$ is the value of $x$ rounded towards $-\infty$ when the `ceil_mode` attribute is $0$ and towards $+\infty$ when it is $1$. The numerator is non-negative, as stated by the constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_CONSTR_X_0020</span></b>](#E_MAXPOOL_INT_CONSTR_X_0020), so that the quotient is non-negative and $dY_{k+2} \ge 1$.

**Tensors with no spatial axis.** A tensor $X$ of rank $1$ or $2$ has no spatial axis. The window then covers a single element, the maximum is that element, and $Y$ has the same shape as $X$.

**Zero-sized dimensions.** A dimension of $X$ may have size $0$. The constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_CONSTR_X_0020</span></b>](#E_MAXPOOL_INT_CONSTR_X_0020) then requires the corresponding dimension of $Y$ to be $0$ as well, and $Y$ is empty; no window is formed and no maximum is taken.

**Ties.** When several elements of $X$ covered by a window are equal to the maximum, the result is that maximum, and the choice of the element it is taken from is left to the implementer. The output $Y$ is the same whichever element is chosen; the choice is observable only through the output `Indices`, which the restriction `[R4]` excludes.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `int8` type:

$$
X = \begin{bmatrix}
  -1 & 2 & -3 & 4 \\
  5 & -6 & 7 & -8 \\
  -9 & 10 & -11 & 12 \\
  13 & -14 & 15 & -16
\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$, `strides` $= [2, 2]$ and `pads` $= [0, 0, 0, 0]$:

$$
Y = \begin{bmatrix}
  5 & 7 \\
  13 & 15
\end{bmatrix}
$$

The window of the first element of $Y$ covers the four elements of the upper left corner of $X$, whose maximum is $5$. The result is that element of $X$, and no arithmetic is performed on it.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs no floating-point arithmetic |
| Overflow | not applicable: the operator performs no arithmetic, and the result is an element of the input |
| Integer overflow | not applicable: the operator performs no arithmetic, and the result is an element of the input |
| Division by zero | not applicable: the operator performs no division |
| A window that covers no element of $X$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_CONSTR_X_0020</span></b>](#E_MAXPOOL_INT_CONSTR_X_0020) |
| A negative dimension of $Y$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_CONSTR_X_0020</span></b>](#E_MAXPOOL_INT_CONSTR_X_0020) |

No error condition.

## Attributes

The attributes of the operator are those given in the section [Attributes](#real) of the specification for real numbers; they do not depend on the type of the arguments.

## Inputs

### $\text{X}$: signed integer tensor

The input data tensor, whose first two axes are the batch axis and the channel axis and whose remaining axes are the spatial axes.

#### Constraints

<a id="E_MAXPOOL_INT_CONSTR_X_0010"></a>
- `[E_MAXPOOL_INT_CONSTR_X_0010]` Rank of $X$
  - Statement: The rank of $X$ is greater than or equal to $1$.
<a id="E_MAXPOOL_INT_CONSTR_X_0020"></a>
- `[E_MAXPOOL_INT_CONSTR_X_0020]` Window within the input
  - Statement: For every spatial axis $k$ and every index $i_{k+2}$ of $Y$ along that axis, the window covers at least one element of $X$, i.e. the interval $[i_{k+2} \cdot s_k - p_k,\, i_{k+2} \cdot s_k - p_k + k_k - 1]$ meets $[0, dX_{k+2}-1]$.
  - Rationale: The maximum of an empty set of elements is not defined; the constraint rules out the windows that cover no element of $X$, and with them the negative dimensions of $Y$.
<a id="E_MAXPOOL_INT_CONSTR_X_0030"></a>
- `[E_MAXPOOL_INT_CONSTR_X_0030]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: signed integer tensor

The result of the max pooling of $X$.

#### Constraints

<a id="E_MAXPOOL_INT_CONSTR_Y_0010"></a>
- `[E_MAXPOOL_INT_CONSTR_Y_0010]` Rank of $Y$
  - Statement: The rank of $Y$ is the rank of $X$.
<a id="E_MAXPOOL_INT_CONSTR_Y_0020"></a>
- `[E_MAXPOOL_INT_CONSTR_Y_0020]` Shape of $Y$
  - Statement: $dY_0 = dX_0$, $dY_1 = dX_1$, and for every spatial axis $k$, $dY_{k+2}$ is the value given by the formula of [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_FUNC_0010</span></b>](#E_MAXPOOL_INT_FUNC_0010).
- `[E_MAXPOOL_INT_CONSTR_Y_0030]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_INT_CONSTR_X_0030</span></b>](#E_MAXPOOL_INT_CONSTR_X_0030) on tensor $X$.

### $\text{Indices}$: `int64` tensor

The index of the element of $X$ that each element of $Y$ is the maximum of. The output is optional and the restriction `[R4]` excludes it.

#### Constraints

<a id="E_MAXPOOL_INT_CONSTR_Indices_0010"></a>
- `[E_MAXPOOL_INT_CONSTR_Indices_0010]` Absence of `Indices`
  - Statement: The output `Indices` is absent.
  - Rationale: The restriction `[R4]` excludes the output.

<a id="uint"></a>

# **MaxPool** (uint)

where uint is in {`uint8`}.

The type of this section has a single member: the operator selects an element of the input and performs no arithmetic on it, so that the result is that element itself, and the unsigned type differs from the signed ones only by the range of the values it represents.

## Signature

$Y\,[, \text{Indices}] = \textbf{MaxPool}(X)$

where
- $X$: input tensor of rank $rX \geq 1$, whose first two axes are the batch axis and the channel axis and whose remaining $rX - 2$ axes are the spatial axes
- $Y$: output tensor, of the same rank as $X$
- $\text{Indices}$: optional output tensor of the same shape as $Y$, giving for each element of $Y$ the index of the element of $X$ it is the maximum of

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | Attribute `auto_pad` is restricted to `NOTSET` | Transient |
| `[R2]` | Attribute `storage_order` is restricted to `0` | Transient |
| `[R3]` | Attribute `dilations` is restricted to `1` along each spatial axis | Transient |
| `[R4]` | The output `Indices` is not supported | Transient |

## Function

<a id="E_MAXPOOL_UINT_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_MAXPOOL_UINT_FUNC_0010]</br></span>

Operator **MaxPool** applies max pooling to the input tensor $X$: it slides a window of the shape given by the `kernel_shape` attribute over the spatial axes of $X$, with the steps given by the `strides` attribute, and stores in $Y$ the maximum of the elements of $X$ covered by the window. The elements of the padding, which are not elements of $X$, are excluded from the maximum.

Let $n = rX - 2$ be the number of spatial axes of $X$, and let $dX_0, \dots, dX_{rX-1}$ be the dimensions of $X$. For any [tensor index](./../common/definitions.md#tensor_index) $i = (i_0, \dots, i_{rX-1})$ of the result $Y$:

$$
Y[i] = \max \left\{ X\left[i_0, i_1, i_2 \cdot s_0 + j_0 - p_0, \dots, i_{n+1} \cdot s_{n-1} + j_{n-1} - p_{n-1}\right] \;\middle|\; 0 \le j_k \le k_k - 1,\; 0 \le i_{k+2} \cdot s_k + j_k - p_k \le dX_{k+2}-1 \right\}
$$

where
- $\max$ is the maximum of the integers, and the set it is applied to is the set of the elements of $X$ that the window covers,
- $k_k$ is the value of the `kernel_shape` attribute along spatial axis $k$,
- $s_k$ is the value of the `strides` attribute along spatial axis $k$,
- $p_k$ is the number of elements of padding added at the beginning of spatial axis $k$, i.e. the value `pads[k]` of the `pads` attribute,
- $i_0$ and $i_1$ are the batch index and the channel index, which the window does not move along.

The window covers the elements of $X$ whose index along spatial axis $k$ lies in $[i_{k+2} \cdot s_k - p_k,\, i_{k+2} \cdot s_k - p_k + k_k - 1]$. The elements of that interval that lie outside $[0, dX_{k+2}-1]$ are the padding, and they are excluded from the maximum: the maximum is taken over the elements of $X$ that the window covers, and the window always covers at least one of them, as stated by the constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_UINT_CONSTR_X_0020</span></b>](#E_MAXPOOL_UINT_CONSTR_X_0020).

**The result is an element of the input.** The operator performs no arithmetic and no rounding: the result is one of the elements of $X$ that the window covers, with the value it has in the type of $X$ and $Y$. The order is the order of the integers, which is the same for the signed and the unsigned types, so that the unsigned type has the same semantics as the signed ones and differs from them only by the range of the values it represents.

**The shape of the output.** The dimensions of $Y$ along the batch axis and the channel axis are those of $X$: $dY_0 = dX_0$ and $dY_1 = dX_1$. Along each spatial axis $k$, the dimension of $Y$ is

$$
dY_{k+2} = \text{round}\left(\frac{dX_{k+2} + p_k + q_k - k_k}{s_k}\right) + 1
$$

where $q_k$ is the number of elements of padding added at the end of spatial axis $k$, i.e. the value `pads[n + k]` of the `pads` attribute, and $\text{round}(x)$ is the value of $x$ rounded towards $-\infty$ when the `ceil_mode` attribute is $0$ and towards $+\infty$ when it is $1$. The numerator is non-negative, as stated by the constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_UINT_CONSTR_X_0020</span></b>](#E_MAXPOOL_UINT_CONSTR_X_0020), so that the quotient is non-negative and $dY_{k+2} \ge 1$.

**Tensors with no spatial axis.** A tensor $X$ of rank $1$ or $2$ has no spatial axis. The window then covers a single element, the maximum is that element, and $Y$ has the same shape as $X$.

**Zero-sized dimensions.** A dimension of $X$ may have size $0$. The constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_UINT_CONSTR_X_0020</span></b>](#E_MAXPOOL_UINT_CONSTR_X_0020) then requires the corresponding dimension of $Y$ to be $0$ as well, and $Y$ is empty; no window is formed and no maximum is taken.

**Ties.** When several elements of $X$ covered by a window are equal to the maximum, the result is that maximum, and the choice of the element it is taken from is left to the implementer. The output $Y$ is the same whichever element is chosen; the choice is observable only through the output `Indices`, which the restriction `[R4]` excludes.

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following example.

### Example 1

For the `uint8` type:

$$
X = \begin{bmatrix}
  1 & 2 & 3 & 4 \\
  5 & 6 & 7 & 8 \\
  9 & 10 & 11 & 12 \\
  13 & 14 & 15 & 16
\end{bmatrix}
$$

with `kernel_shape` $= [2, 2]$, `strides` $= [2, 2]$ and `pads` $= [0, 0, 0, 0]$:

$$
Y = \begin{bmatrix}
  6 & 8 \\
  14 & 16
\end{bmatrix}
$$

The window of the first element of $Y$ covers the four elements of the upper left corner of $X$, whose maximum is $6$. The result is that element of $X$, and no arithmetic is performed on it.

## Error conditions

| Condition | Disposition |
| -------- | ------- |
| Invalid operation | not applicable: the operator performs no floating-point arithmetic |
| Overflow | not applicable: the operator performs no arithmetic, and the result is an element of the input |
| Integer overflow | not applicable: the operator performs no arithmetic, and the result is an element of the input |
| Division by zero | not applicable: the operator performs no division |
| A window that covers no element of $X$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_UINT_CONSTR_X_0020</span></b>](#E_MAXPOOL_UINT_CONSTR_X_0020) |
| A negative dimension of $Y$ | ruled out by the precondition [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_UINT_CONSTR_X_0020</span></b>](#E_MAXPOOL_UINT_CONSTR_X_0020) |

No error condition.

## Attributes

The attributes of the operator are those given in the section [Attributes](#real) of the specification for real numbers; they do not depend on the type of the arguments.

## Inputs

### $\text{X}$: unsigned integer tensor

The input data tensor, whose first two axes are the batch axis and the channel axis and whose remaining axes are the spatial axes.

#### Constraints

<a id="E_MAXPOOL_UINT_CONSTR_X_0010"></a>
- `[E_MAXPOOL_UINT_CONSTR_X_0010]` Rank of $X$
  - Statement: The rank of $X$ is greater than or equal to $1$.
<a id="E_MAXPOOL_UINT_CONSTR_X_0020"></a>
- `[E_MAXPOOL_UINT_CONSTR_X_0020]` Window within the input
  - Statement: For every spatial axis $k$ and every index $i_{k+2}$ of $Y$ along that axis, the window covers at least one element of $X$, i.e. the interval $[i_{k+2} \cdot s_k - p_k,\, i_{k+2} \cdot s_k - p_k + k_k - 1]$ meets $[0, dX_{k+2}-1]$.
  - Rationale: The maximum of an empty set of elements is not defined; the constraint rules out the windows that cover no element of $X$, and with them the negative dimensions of $Y$.
<a id="E_MAXPOOL_UINT_CONSTR_X_0030"></a>
- `[E_MAXPOOL_UINT_CONSTR_X_0030]` Type consistency
  - Statement: Tensors $X$ and $Y$ have the same type.

## Outputs

### $\text{Y}$: unsigned integer tensor

The result of the max pooling of $X$.

#### Constraints

<a id="E_MAXPOOL_UINT_CONSTR_Y_0010"></a>
- `[E_MAXPOOL_UINT_CONSTR_Y_0010]` Rank of $Y$
  - Statement: The rank of $Y$ is the rank of $X$.
<a id="E_MAXPOOL_UINT_CONSTR_Y_0020"></a>
- `[E_MAXPOOL_UINT_CONSTR_Y_0020]` Shape of $Y$
  - Statement: $dY_0 = dX_0$, $dY_1 = dX_1$, and for every spatial axis $k$, $dY_{k+2}$ is the value given by the formula of [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_UINT_FUNC_0010</span></b>](#E_MAXPOOL_UINT_FUNC_0010).
- `[E_MAXPOOL_UINT_CONSTR_Y_0030]` Type consistency
  - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_MAXPOOL_UINT_CONSTR_X_0030</span></b>](#E_MAXPOOL_UINT_CONSTR_X_0030) on tensor $X$.

### $\text{Indices}$: `int64` tensor

The index of the element of $X$ that each element of $Y$ is the maximum of. The output is optional and the restriction `[R4]` excludes it.

#### Constraints

<a id="E_MAXPOOL_UINT_CONSTR_Indices_0010"></a>
- `[E_MAXPOOL_UINT_CONSTR_Indices_0010]` Absence of `Indices`
  - Statement: The output `Indices` is absent.
  - Rationale: The restriction `[R4]` excludes the output.
