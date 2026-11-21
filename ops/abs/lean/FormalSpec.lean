import Sonnx
/-!
@generated-by sonnx-lean-pipeline
@source ops/abs/abs.md
@source-digest 2e746f2bcd7bbd94b2d1c8ae9e395d32
@library-digest 3162109b4c926c2fa5dd5f8bd2eb2d90fb17a0480eccc0216a4ea9bd2b43ffab
Compilation and trace validity are separate from semantic completeness; see the trace report.
-/


/-!
# `Abs` — formal specification

Formal specification of the ONNX `Abs` operator (opset 13) over the shared `Sonnx` library, for
the four type families of the informal document: real, floating point, signed integer and
unsigned integer.

`Abs` is unary and element-wise: the result `Y` has the shape of the operand `X` and `Y[i]` is
the absolute value of `X[i]`. The operator combines no two elements, so that no shape
compatibility condition arises, and it has no attribute and no error condition.
-/

namespace AbsSpec

open Sonnx

/-! ## The real family -/

/-! DECISION: the shared library has no real-number type (Mathlib is not part of the profile), so
the real family is modelled over `Rat`, the exact ordered field that `Sonnx.FVal` itself uses for
its finite values. This preserves the two properties the document requires of the real operator:
the absolute value is exact and the operator introduces no approximation. -/

/-- The element step of `Abs` on the real numbers: `|x|` in `ℚ`, exactly.
`X[i] ≥ 0` gives `X[i]`, `X[i] < 0` gives `-X[i]`. -/
def realAbsElt (x : Rat) : Rat := if x < 0 then -x else x

/-! 
@trace E_ABS_REAL_CONSTR_X_0010
@source ops/abs/abs.md:L89-L91
@coverage specified
@anchor E_ABS_REAL_CONSTR_X_0010
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_ABS_REAL_CONSTR_X_0010"></a>
  - `[E_ABS_REAL_CONSTR_X_0010]` Type consistency
    - Statement: Tensors $$X$$ and $$Y$$ have the same type.
-/

/-! 
@trace E_ABS_REAL_CONSTR_Y_0020
@source ops/abs/abs.md:L103-L104
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_ABS_REAL_CONSTR_Y_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ABS_REAL_CONSTR_X_0010</span></b>](#E_ABS_REAL_CONSTR_X_0010) on tensor $$X$$.
-/


/-! 
@trace E_ABS_REAL_CONSTR_Y_0010
@source ops/abs/abs.md:L101-L102
@coverage specified
@declaration AbsSpec.RealAbsShape
Informal source excerpt:
  - `[E_ABS_REAL_CONSTR_Y_0010]` Shape definition
    - Statement: Tensor $$Y$$ has the shape of $$X$$.
-/

/-- `[E_ABS_REAL_CONSTR_Y_0010]` Shape definition: `Y` has the shape of `X`.

No shape compatibility condition arises: the operator combines no two elements, so `X` has no
shape constraint other than being a tensor. A rank-0 operand gives a rank-0 result with a single
element, and a zero-sized dimension of `X` is a zero-sized dimension of `Y`, so that `Y` is
empty. -/
def RealAbsShape (X Y : Tensor Rat) : Prop := Y.shape = X.shape

/-! 
@trace E_ABS_REAL_FUNC_0010
@source ops/abs/abs.md:L32-L59
@coverage specified
@anchor E_ABS_REAL_FUNC_0010
@declaration AbsSpec.RealAbs
Informal source excerpt:
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
-/

/-- `[E_ABS_REAL_FUNC_0010]` `Abs` on the real family: for every index `i` of the result, the
element `Y[i]` is the absolute value `|X[i]|` in `ℚ`, and the result has the shape of `X`.

The result is exact: the absolute value of a rational is a rational and the operator introduces
no approximation. -/
def RealAbs (X Y : Tensor Rat) : Prop :=
  RealAbsShape X Y ∧ ∀ i : List Nat, Y.read i = (X.read i).map realAbsElt

/-! ## The floating-point family (`float16`, `float`, `double`) -/

/-! DECISION: the `Bool` carried by `Sonnx.FVal.inf`, `Sonnx.FVal.zero` and `Sonnx.FVal.finite` is
read as the sign of the value, `true` standing for the positive sign (sign bit cleared), and the
`Rat` of `FVal.finite` as the magnitude. `Abs` is the operation that clears the sign, which is
what the element step below does:

* a NaN operand gives a NaN (`FVal.nan`; the payload and the sign are not specified, so every
  quiet NaN of the type is a conforming result),
* an infinity of either sign gives `+∞`,
* either zero gives `+0`,
* a finite value gives the finite value of the same magnitude — no rounding, no overflow and no
  underflow can occur, since the set of finite values of a format is symmetric about zero.

The three formats share this semantics and differ only by their precision and range, so the
element step carries no format parameter; `X` and `Y` have the same type. -/

/-- The element step of `Abs` on floating-point values: the operand with its sign cleared. -/
def floatAbsElt (x : FVal) : FVal :=
  match x with
  | .nan => .nan
  | .inf _ => .inf true
  | .zero _ => .zero true
  | .finite _ q => .finite true (if q < 0 then -q else q)

/-! 
@trace E_ABS_FLOAT_CONSTR_X_0010
@source ops/abs/abs.md:L217-L219
@coverage specified
@anchor E_ABS_FLOAT_CONSTR_X_0010
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_ABS_FLOAT_CONSTR_X_0010"></a>
  - `[E_ABS_FLOAT_CONSTR_X_0010]` Type consistency
    - Statement: Tensors $$X$$ and $$Y$$ have the same type.
-/

/-! 
@trace E_ABS_FLOAT_CONSTR_Y_0020
@source ops/abs/abs.md:L231-L232
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_ABS_FLOAT_CONSTR_Y_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ABS_FLOAT_CONSTR_X_0010</span></b>](#E_ABS_FLOAT_CONSTR_X_0010) on tensor $$X$$.
-/


/-! 
@trace E_ABS_FLOAT_CONSTR_Y_0010
@source ops/abs/abs.md:L229-L230
@coverage specified
@declaration AbsSpec.FloatAbsShape
Informal source excerpt:
  - `[E_ABS_FLOAT_CONSTR_Y_0010]` Shape definition
    - Statement: Tensor $$Y$$ has the shape of $$X$$.
-/

/-- `[E_ABS_FLOAT_CONSTR_Y_0010]` Shape definition: `Y` has the shape of `X`.

`Y[i]` depends on `X[i]` alone; no shape compatibility condition arises. A rank-0 operand has a
single element and gives a rank-0 result, and a zero-sized dimension stays zero-sized. -/
def FloatAbsShape (X Y : Tensor FVal) : Prop := Y.shape = X.shape

/-! 
@trace E_ABS_FLOAT_FUNC_0010
@source ops/abs/abs.md:L130-L167
@coverage specified
@anchor E_ABS_FLOAT_FUNC_0010
@declaration AbsSpec.FloatAbs
Informal source excerpt:
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
-/

/-- `[E_ABS_FLOAT_FUNC_0010]` `Abs` on the floating-point family: `Y` has the shape of `X` and
`Y[i]` is `X[i]` with its sign bit cleared, for every index `i` of the result.

The element relation is stated through `floatAbsElt`, so that a NaN operand may yield any NaN,
while every finite operand yields the exact finite value of the same magnitude. -/
def FloatAbs (X Y : Tensor FVal) : Prop :=
  FloatAbsShape X Y ∧ ∀ i : List Nat, Y.read i = (X.read i).map floatAbsElt

/-! ## The signed integer family (`int8`, `int16`, `int32`, `int64`) -/

/-! DECISION: `Sonnx.SIntN n` is a bit pattern; the specification is stated on the integer value
it denotes (`Sonnx.SIntN.toInt`), which is where the case split of the document lives. The result
is the value of the `n`-bit type congruent to the exact absolute value modulo `2 ^ n`, which is
`Sonnx.sintWrap`. In particular the minimum value `-2 ^ (n-1)`, whose exact absolute value
`2 ^ (n-1)` is outside the range of the type, is mapped to itself, as the document requires: the
case is nominal, not an error.

The result `Y` is pinned down element-wise: wherever `X` has the value `x` at an index, `Y` has
at that index the value `y` whose integer value is `sintAbsVal n x.toInt`. -/

/-- The value of the `n`-bit signed type congruent to the exact absolute value of `v`.

For `v ≥ 0` this is `v`; for `-2^(n-1) < v < 0` it is `-v`, which lies in the range of the type;
for `v = -2^(n-1)` it is the value of the type congruent to `2^(n-1)` modulo `2^n`, that is
`-2^(n-1)` again. -/
def sintAbsVal (n : Nat) (v : Int) : Int :=
  Sonnx.sintWrap n (if v < 0 then -v else v)

/-! 
@trace E_ABS_INT_CONSTR_X_0010
@source ops/abs/abs.md:L328-L330
@coverage specified
@anchor E_ABS_INT_CONSTR_X_0010
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_ABS_INT_CONSTR_X_0010"></a>
  - `[E_ABS_INT_CONSTR_X_0010]` Type consistency
    - Statement: Tensors $$X$$ and $$Y$$ have the same type.
-/

/-! 
@trace E_ABS_INT_CONSTR_Y_0020
@source ops/abs/abs.md:L342-L343
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_ABS_INT_CONSTR_Y_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ABS_INT_CONSTR_X_0010</span></b>](#E_ABS_INT_CONSTR_X_0010) on tensor $$X$$.
-/


/-! 
@trace E_ABS_INT_CONSTR_Y_0010
@source ops/abs/abs.md:L340-L341
@coverage specified
@declaration AbsSpec.IntAbsShape
Informal source excerpt:
  - `[E_ABS_INT_CONSTR_Y_0010]` Shape definition
    - Statement: Tensor $$Y$$ has the shape of $$X$$.
-/

/-- `[E_ABS_INT_CONSTR_Y_0010]` Shape definition: `Y` has the shape of `X`. -/
def IntAbsShape (n : Nat) (X Y : Tensor (SIntN n)) : Prop := Y.shape = X.shape

/-! 
@trace E_ABS_INT_FUNC_0010
@source ops/abs/abs.md:L258-L289
@coverage specified
@anchor E_ABS_INT_FUNC_0010
@declaration AbsSpec.IntAbs
Informal source excerpt:
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
-/

/-- `[E_ABS_INT_FUNC_0010]` `Abs` on the signed integer family: `Y` has the shape of `X`, and at
every index `i` where `X` holds `x` the result holds the value `y` of the same `n`-bit signed
type whose integer value is the value of the type congruent to `|x|` modulo `2 ^ n`.

So `Y[i] = X[i]` when `X[i] ≥ 0`, `Y[i] = -X[i]` when `-2^(n-1) < X[i] < 0`, and
`Y[i] = -2^(n-1)` when `X[i] = -2^(n-1)`. -/
def IntAbs (n : Nat) (X Y : Tensor (SIntN n)) : Prop :=
  IntAbsShape n X Y ∧
    ∀ (i : List Nat) (x y : SIntN n),
      X.read i = some x → Y.read i = some y → y.toInt = sintAbsVal n x.toInt

/-! ## The unsigned integer family (`uint8`, `uint16`, `uint32`, `uint64`) -/

/-! DECISION: every value of an `n`-bit unsigned type is non-negative, so its absolute value is
the value itself and the operator is the identity on these types; the specification below states
that identity directly, at the level of the bit patterns of `Sonnx.UIntN`, without interpreting
them as integers. -/

/-! 
@trace E_ABS_UINT_CONSTR_Y_0010
@source ops/abs/abs.md:L439-L440
@coverage specified
@declaration AbsSpec.UintAbsShape
Informal source excerpt:
  - `[E_ABS_UINT_CONSTR_Y_0010]` Shape definition
    - Statement: Tensor $$Y$$ has the shape of $$X$$.
-/

/-- `[E_ABS_UINT_CONSTR_Y_0010]` Shape definition: `Y` has the shape of `X`.

`Y[i]` depends on `X[i]` alone, so no shape compatibility condition arises; a rank-0 operand
gives a rank-0 result with that single element unchanged, and a zero-sized dimension stays
zero-sized. -/
def UintAbsShape (n : Nat) (X Y : Tensor (UIntN n)) : Prop := Y.shape = X.shape

/-! 
@trace E_ABS_UINT_CONSTR_X_0010
@source ops/abs/abs.md:L427-L429
@coverage specified
@anchor E_ABS_UINT_CONSTR_X_0010
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_ABS_UINT_CONSTR_X_0010"></a>
  - `[E_ABS_UINT_CONSTR_X_0010]` Type consistency
    - Statement: Tensors $$X$$ and $$Y$$ have the same type.
-/

/-! 
@trace E_ABS_UINT_CONSTR_Y_0020
@source ops/abs/abs.md:L441-L442
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_ABS_UINT_CONSTR_Y_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_ABS_UINT_CONSTR_X_0010</span></b>](#E_ABS_UINT_CONSTR_X_0010) on tensor $$X$$.
-/


/-! 
@trace E_ABS_UINT_FUNC_0010
@source ops/abs/abs.md:L369-L390
@coverage specified
@anchor E_ABS_UINT_FUNC_0010
@declaration AbsSpec.UintAbs
Informal source excerpt:
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
-/

/-- `[E_ABS_UINT_FUNC_0010]` `Abs` on the unsigned integer family: `Y` has the shape of `X` and
every element of `Y` is the element of `X` at the same index. No element changes, no value leaves
the range of the type, and no value is mapped to a value of the opposite sign. -/
def UintAbs (n : Nat) (X Y : Tensor (UIntN n)) : Prop :=
  UintAbsShape n X Y ∧ ∀ i : List Nat, Y.read i = X.read i

/-! ## Examples -/

/-! ### Example 1 (real) -/

/-- `real:example:1`, the operand `[3.5, -2, 0, -7.25]`, written as exact rationals. -/
def realExample1X : Tensor Rat := ⟨[4], [7 / 2, -2, 0, -29 / 4]⟩

/-- `real:example:1`, the result `[3.5, 2, 0, 7.25]`. -/
def realExample1Y : Tensor Rat := ⟨[4], [7 / 2, 2, 0, 29 / 4]⟩

/-! 
@trace real:example:1
@source ops/abs/abs.md:L63-L71
@coverage example_only
@declaration AbsSpec.realExample1
Informal source excerpt:
  ### Example 1
  
  $$
  X = \begin{bmatrix} 3.5 & -2 & 0 & -7.25 \end{bmatrix}
  \quad
  Y = \begin{bmatrix} 3.5 & 2 & 0 & 7.25 \end{bmatrix}
  $$
  
  The third element is its own absolute value, and the other three are the operands of negative sign with that sign removed.
-/

/-- `real:example:1`: `Abs` maps `[3.5, -2, 0, -7.25]` to `[3.5, 2, 0, 7.25]`. The third element
is its own absolute value; the other three are the operands of negative sign with that sign
removed. Concrete evidence for this example, not a universal conformance claim. -/
def realExample1 : Prop := RealAbs realExample1X realExample1Y

/-! ### Example 1 (float) -/

/-- `float:example:1`, the operand `[-1.5, 2.25, 0.0, -0.0, -inf, +inf, NaN]`. -/
def floatExample1X : Tensor FVal :=
  ⟨[7], [FVal.finite false (3 / 2), FVal.finite true (9 / 4), FVal.zero true,
         FVal.zero false, FVal.inf false, FVal.inf true, FVal.nan]⟩

/-- `float:example:1`, the result `[1.5, 2.25, +0, +0, +inf, +inf, NaN]`. -/
def floatExample1Y : Tensor FVal :=
  ⟨[7], [FVal.finite true (3 / 2), FVal.finite true (9 / 4), FVal.zero true,
         FVal.zero true, FVal.inf true, FVal.inf true, FVal.nan]⟩

/-! 
@trace float:example:1
@source ops/abs/abs.md:L171-L181
@coverage example_only
@declaration AbsSpec.floatExample1
Informal source excerpt:
  ### Example 1
  
  $$
  X = \begin{bmatrix} -1.5 & 2.25 & 0.0 & -0.0 & -\text{inf} & +\text{inf} & \text{NaN} \end{bmatrix}
  $$
  
  $$
  Y = \begin{bmatrix} 1.5 & 2.25 & +0 & +0 & +\text{inf} & +\text{inf} & \text{NaN} \end{bmatrix}
  $$
  
  The first two elements are a negative and a positive finite value; they give the same magnitude, the sign being cleared in both cases. The third and the fourth elements are the two zeros, both of which give $$+0$$. The fifth and the sixth elements are the two infinities, both of which give $$+\infty$$. The last element is a NaN, which gives a NaN whose sign and payload are not specified.
-/

/-- `float:example:1`: the two finite values of both signs give the same magnitude, the two zeros
both give `+0`, the two infinities both give `+∞`, and the NaN gives a NaN whose sign and payload
are not specified. Concrete evidence for this example, not a universal conformance claim. -/
def floatExample1 : Prop := FloatAbs floatExample1X floatExample1Y

/-! ### Example 2 (float) -/

/-- `float:example:2`, the operand `[[-3.0, -1.0], [-2.5, 4.0]]`. -/
def floatExample2X : Tensor FVal :=
  ⟨[2, 2], [FVal.finite false 3, FVal.finite false 1,
            FVal.finite false (5 / 2), FVal.finite true 4]⟩

/-- `float:example:2`, the result `[[3.0, 1.0], [2.5, 4.0]]`. -/
def floatExample2Y : Tensor FVal :=
  ⟨[2, 2], [FVal.finite true 3, FVal.finite true 1,
            FVal.finite true (5 / 2), FVal.finite true 4]⟩

/-! 
@trace float:example:2
@source ops/abs/abs.md:L183-L191
@coverage example_only
@declaration AbsSpec.floatExample2
Informal source excerpt:
  ### Example 2
  
  $$
  X = \begin{bmatrix} -3.0 & -1.0 \\ -2.5 & 4.0 \end{bmatrix}
  \quad
  Y = \begin{bmatrix} 3.0 & 1.0 \\ 2.5 & 4.0 \end{bmatrix}
  $$
  
  The elements of $$Y$$ are the elements of $$X$$ with the sign cleared, each at the index it occupies in $$X$$; the shape of $$Y$$ is the shape of $$X$$.
-/

/-- `float:example:2`: the elements of the result are the elements of the operand with the sign
cleared, each at the index it occupies in the operand, and the shape of the result is the shape
of the operand. Concrete evidence for this example, not a universal conformance claim. -/
def floatExample2 : Prop := FloatAbs floatExample2X floatExample2Y

/-! ### Example 1 (int) -/

/-! 
@trace int:example:1
@source ops/abs/abs.md:L293-L305
@coverage example_only
@declaration AbsSpec.intExample1
Informal source excerpt:
  ### Example 1
  
  For the `int8` type:
  
  $$
  X = \begin{bmatrix} -6 & 100 & 127 & 0 & -127 & -128 \end{bmatrix}
  $$
  
  $$
  Y = \begin{bmatrix} 6 & 100 & 127 & 0 & 127 & -128 \end{bmatrix}
  $$
  
  The first two elements are the negative and the positive value of the same magnitude, which give the same result. The third and the fourth elements are non-negative and are their own absolute values. The fifth element is $$-127$$, whose absolute value $$127$$ is the maximum value of `int8`. The last element is the minimum value $$-2^7 = -128$$, whose exact absolute value $$128$$ is outside the range of `int8`: the result is the value of the type congruent to $$128$$ modulo $$2^8$$, that is $$128 - 256 = -128$$, so that the last element of $$Y$$ is negative and equal to the element of $$X$$ at the same index.
-/

/-- `int:example:1` for `int8`, stated on the integer values of the type: the operand
`[-6, 100, 127, 0, -127, -128]` gives `[6, 100, 127, 0, 127, -128]`. The last element is the
minimum value `-128`, whose exact absolute value `128` is outside the range of `int8`: the result
is the value of the type congruent to `128` modulo `2^8`, that is `-128`. Concrete evidence for
this example, not a universal conformance claim. -/
def intExample1 : Prop :=
  sintAbsVal 8 (-6) = 6 ∧ sintAbsVal 8 100 = 100 ∧ sintAbsVal 8 127 = 127 ∧
  sintAbsVal 8 0 = 0 ∧ sintAbsVal 8 (-127) = 127 ∧ sintAbsVal 8 (-128) = -128

/-! ### Example 1 (uint) -/

/-! 
@trace uint:example:1
@source ops/abs/abs.md:L394-L404
@coverage proved
@declaration AbsSpec.uintExample1
Informal source excerpt:
  ### Example 1
  
  For the `uint8` type:
  
  $$
  X = \begin{bmatrix} 0 & 6 & 200 & 255 \end{bmatrix}
  \quad
  Y = \begin{bmatrix} 0 & 6 & 200 & 255 \end{bmatrix}
  $$
  
  Every element of $$X$$ is non-negative, so that every element of $$Y$$ is the element of $$X$$ at the same index, including the minimum value $$0$$ and the maximum value $$255$$ of `uint8`.
-/

/-- `uint:example:1` for `uint8`: every value of an unsigned type is its own absolute value, so a
`uint8` tensor of shape `[4]` — in particular one holding `0, 6, 200, 255`, including the minimum
value `0` and the maximum value `255` — is its own result. The operator is the identity on the
unsigned types, which is what the theorem below establishes for every `uint8` tensor of that
shape. -/
theorem uintExample1 : ∀ X : Tensor (UIntN 8), X.shape = [4] → UintAbs 8 X X := by
  intro X _
  exact ⟨rfl, fun _ => rfl⟩

end AbsSpec

-- Fully-qualified declarations named by the trace entries.
#check @AbsSpec.FloatAbs
#check @AbsSpec.FloatAbsShape
#check @AbsSpec.IntAbs
#check @AbsSpec.IntAbsShape
#check @AbsSpec.RealAbs
#check @AbsSpec.RealAbsShape
#check @AbsSpec.UintAbs
#check @AbsSpec.UintAbsShape
#check @AbsSpec.floatExample1
#check @AbsSpec.floatExample2
#check @AbsSpec.intExample1
#check @AbsSpec.realExample1
#check @AbsSpec.uintExample1
#check @Sonnx.TypeConsistent
