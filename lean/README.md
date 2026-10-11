# The Mul operator in Lean 4

A Lean 4 formalization of the SONNX profile specification of operator
**Mul** (`../ops/mul.md`): the specification is turned into Lean types and
definitions, the operator is implemented as executable Lean code, and its
behaviour — the elementwise formulas and constraint lists
`E_MUL_*_FUNC_0010`, `E_MUL_*_CONSTR_A_0010`, `E_MUL_*_CONSTR_C_0010` — is
proved.

Pure Lean core, no external libraries. Built and checked with Lean
4.33.1 / Lake 5.0.0:

```sh
cd lean && lake build
# to type-check a single file:
lake env lean MulOp/MulCore.lean
```

## Files

| file | contents |
|---|---|
| `MulOp/Tensor.lean` | the tensor model: shapes, indices (`validIdxB`), row-major `offset`, `Tensor`, `Tensor.read` |
| `MulOp/Broadcast.lean` | Numpy-style multidirectional broadcasting: `padTo`, `bcastDim`, `bcastShape`, the `Broadcasts` predicate, index translation `bcastIdx` and its validity |
| `MulOp/MulCore.lean` | the operator `mul : Tensor α → Tensor α → Option (Tensor α)`, generic over the element type, with its shape and elementwise theorems |
| `MulOp/IntTypes.lean` | the `n`-bit integer type families (`UIntN`, `SIntN`) and the integer formulas |
| `MulOp/FloatTypes.lean` | the IEEE 754 binary64 family (Lean's `Float`) and the case table |
| `MulOp/RealTypes.lean` | the exact-real family (modeled by `Rat`) and its examples |
| `MulOp/Examples.lean` | the tensor-level examples of the specification, for all four families |
| `MulOp.lean` | root module |

## Where each clause of the specification lives

| clause of `../ops/mul.md` | Lean |
|---|---|
| definition of the broadcast of two shapes / incompatibility | `Broadcast.bcastShape`, `Broadcast.bcastDim` |
| `E_MUL_*_CONSTR_A_0010` — fail when shapes are not broadcast-compatible | `MulCore.mul_none`; exercised concretely in `Examples.incompatible_none` |
| shape part of `E_MUL_*_CONSTR_C_0010` — result shape and well-formedness | `MulCore.mul_shape_wf` (with `MulCore.mulData_wf`, `Broadcast.bcastShape_length`, `Broadcast.bcastShape_broadcasts`) |
| `E_MUL_*_FUNC_0010` — `C[i] = Ã[i] · B̃[i]` at every index | `MulCore.mulData_read_correct`, for a generic element type `α` with a `Mul` instance; the index `i` of the broadcast shape is translated into an operand index by `Broadcast.bcastIdx`, proved valid by `Broadcast.validIdxB_bcastIdx` |
| `E_MUL_REAL_FUNC_0010` — exact products in `ℝ` | instantiated at `α = Rat`: `RealTypes.example_1_*`, `Examples.real_example_1/2` |
| `E_MUL_FLOAT_FUNC_0010` — IEEE 754 case table | `FloatTypes`: `mul_nan_*` (NaN operands), `mul_zero_inf` (zero · infinity), `mul_exact`/`mul_rounds` (exact product / roundTiesToEven), `mul_overflow*` (±∞ on overflow), `mul_subnormal`/`mul_underflow_zero` (underflow), `mul_zero_sign_*` (the signs of the zeros); the tensor example `float_example_2` |
| `E_MUL_UINT_FUNC_0010` — unsigned `n`-bit products | `IntTypes.UIntN.mul_def` (the result is the exact product reduced modulo `2^n`) and `UIntN.mul_congruent` (congruent to the exact product: the reduction applied as many times as needed); examples `IntTypes` and `Examples.uint8_example` |
| `E_MUL_INT_FUNC_0010` — signed `n`-bit products | `IntTypes.SIntN.mul_spec`: the product's value is in the canonical range `[-2^(n-1), 2^(n-1))`, congruent modulo `2^n` to `a.toInt * b.toInt`, and is the **only** integer of that range satisfying the congruence (the "values of the type are `2^n` distinct integers no two of which are congruent" observation, proved as `IntTypes.congruent_range_unique`); `Examples.int8_example` |
| values as `2^n` distinct integers, reduction several times | `IntTypes.CongM` (congruence), `CongM.mul`, `CongM.add_congruent`, `congruent_range_unique` |

## Modeling choices

* **Exact reals** (`real`): modeled by `Rat ⊂ ℝ` — the multiplication is the
  exact quotient product, no rounding, no special values, hence faithful for
  rational operands (all values of the specification's real examples are
  rational). `E_MUL_REAL_FUNC_0010` is then literally
  `MulCore.mulData_read_correct` at `α = Rat`.
* **IEEE 754 floats** (`float`): Lean's `Float` is exactly the IEEE 754
  binary64 `double`: its `Mul` instance multiplies with rounding mode
  roundTiesToEven and IEEE error handling (invalid operations → quiet `NaN`);
  there is no separate `Double` type. The case table of
  `E_MUL_FLOAT_FUNC_0010` is verified by evaluating the operator's compiled
  code (`native_decide`) on the values the specification documents — the same
  IEEE paths a runtime evaluates. `NaN` payloads being unspecified, the
  `NaN` results are identified by `Float.isNaN` and the signed zeros by
  `Float.toBits`, not by equality.
* **Integer families** (`int`/`uint`): a value of `UIntN n`/`SIntN n` is a
  bit pattern `val : Nat`, `0 ≤ val < 2^n`, and the multiplication is the
  exact product reduced modulo `2^n` — the same operation for both types,
  like `BitVec n` in the Lean standard library. The signed type adds the
  two's-complement reading `SIntN.toInt`, proved to be the unique integer of
  the canonical range congruent to the pattern. The ONNX types `uint8`,
  `int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64`, `int64` are the
  instances `n = 8, 16, 32, 64`, as the specification says.
* **Broadcasting**: right-aligned, Numpy-style; a rank-0 operand broadcasts
  to any shape, and a dimension of size zero broadcasts only against size-1
  dimensions, where it keeps size zero. Where an operand has no element
  under an index of the product (rank-0 and zero-sized-dimension cases), the
  element read falls back to the `Inhabited` default — the formula
  `E_MUL_*_FUNC_0010` is stated for well-formed tensors with non-zero shapes,
  where the fallback never fires (`MulCore.mulData_read_correct`).
* **Error handling**: `mul` returns `Option`; `none` is the failure of
  `E_MUL_*_CONSTR_A_0010`. There are no other errors for the real family
  (`E_MUL_REAL_FUNC_0010`), and the float/integer error conditions of the
  specification are the nominal, correct results of their arithmetic models.