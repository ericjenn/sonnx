import Sonnx
/-!
@generated-by sonnx-lean-pipeline
@source ops/div/div.md
@source-digest a8cd637d6a06c39a769d75ebdb3b2818
@library-digest 3162109b4c926c2fa5dd5f8bd2eb2d90fb17a0480eccc0216a4ea9bd2b43ffab
Compilation and trace validity are separate from semantic completeness; see the trace report.
-/


-- Constraint families. Each of these requirements is exactly a shared statement
-- of the profile, so the marker names the library declaration.

/-! 
@trace E_DIV_REAL_CONSTR_A_0010
@source ops/div/div.md:L113-L115
@coverage specified
@anchor E_DIV_REAL_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_DIV_REAL_CONSTR_A_0010"></a>
  - `[E_DIV_REAL_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_DIV_REAL_CONSTR_B_0020
@source ops/div/div.md:L127-L128
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_DIV_REAL_CONSTR_B_0020]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_REAL_CONSTR_A_0010</span></b>](#E_DIV_REAL_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_DIV_REAL_CONSTR_C_0010
@source ops/div/div.md:L138-L139
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_DIV_REAL_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_DIV_FLOAT_CONSTR_A_0010
@source ops/div/div.md:L275-L277
@coverage specified
@anchor E_DIV_FLOAT_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_DIV_FLOAT_CONSTR_A_0010"></a>
  - `[E_DIV_FLOAT_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_DIV_FLOAT_CONSTR_A_0020
@source ops/div/div.md:L278-L280
@coverage specified
@anchor E_DIV_FLOAT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_DIV_FLOAT_CONSTR_A_0020"></a>
  - `[E_DIV_FLOAT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$ and $C$ have the same type.
-/

/-! 
@trace E_DIV_FLOAT_CONSTR_B_0010
@source ops/div/div.md:L288-L289
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_DIV_FLOAT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_FLOAT_CONSTR_A_0010</span></b>](#E_DIV_FLOAT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_DIV_FLOAT_CONSTR_B_0020
@source ops/div/div.md:L290-L291
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_DIV_FLOAT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_FLOAT_CONSTR_A_0020</span></b>](#E_DIV_FLOAT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_DIV_FLOAT_CONSTR_C_0010
@source ops/div/div.md:L301-L302
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_DIV_FLOAT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_DIV_FLOAT_CONSTR_C_0020
@source ops/div/div.md:L303-L304
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_DIV_FLOAT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_FLOAT_CONSTR_A_0020</span></b>](#E_DIV_FLOAT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_DIV_INT_CONSTR_A_0010
@source ops/div/div.md:L406-L408
@coverage specified
@anchor E_DIV_INT_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_DIV_INT_CONSTR_A_0010"></a>
  - `[E_DIV_INT_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_DIV_INT_CONSTR_A_0020
@source ops/div/div.md:L409-L411
@coverage specified
@anchor E_DIV_INT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_DIV_INT_CONSTR_A_0020"></a>
  - `[E_DIV_INT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$ and $C$ have the same type.
-/

/-! 
@trace E_DIV_INT_CONSTR_B_0010
@source ops/div/div.md:L419-L420
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_DIV_INT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_INT_CONSTR_A_0010</span></b>](#E_DIV_INT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_DIV_INT_CONSTR_B_0020
@source ops/div/div.md:L421-L422
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_DIV_INT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_INT_CONSTR_A_0020</span></b>](#E_DIV_INT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_DIV_INT_CONSTR_C_0010
@source ops/div/div.md:L432-L433
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_DIV_INT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_DIV_INT_CONSTR_C_0020
@source ops/div/div.md:L434-L435
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_DIV_INT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_INT_CONSTR_A_0020</span></b>](#E_DIV_INT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_DIV_UINT_CONSTR_A_0010
@source ops/div/div.md:L530-L532
@coverage specified
@anchor E_DIV_UINT_CONSTR_A_0010
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  <a id="E_DIV_UINT_CONSTR_A_0010"></a>
  - `[E_DIV_UINT_CONSTR_A_0010]` Shape compatibility
    - Statement: The shapes of $A$ and $B$ are compatible for broadcasting.
-/

/-! 
@trace E_DIV_UINT_CONSTR_A_0020
@source ops/div/div.md:L533-L535
@coverage specified
@anchor E_DIV_UINT_CONSTR_A_0020
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_DIV_UINT_CONSTR_A_0020"></a>
  - `[E_DIV_UINT_CONSTR_A_0020]` Type consistency
    - Statement: Tensors $A$, $B$ and $C$ have the same type.
-/

/-! 
@trace E_DIV_UINT_CONSTR_B_0010
@source ops/div/div.md:L543-L544
@coverage specified
@declaration Sonnx.BroadcastCompatible
Informal source excerpt:
  - `[E_DIV_UINT_CONSTR_B_0010]` Shape compatibility
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_UINT_CONSTR_A_0010</span></b>](#E_DIV_UINT_CONSTR_A_0010) on tensor $A$.
-/

/-! 
@trace E_DIV_UINT_CONSTR_B_0020
@source ops/div/div.md:L545-L546
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_DIV_UINT_CONSTR_B_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_UINT_CONSTR_A_0020</span></b>](#E_DIV_UINT_CONSTR_A_0020) on tensor $A$.
-/

/-! 
@trace E_DIV_UINT_CONSTR_C_0010
@source ops/div/div.md:L556-L557
@coverage specified
@declaration Sonnx.BroadcastShapeIs
Informal source excerpt:
  - `[E_DIV_UINT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$ and $B$ broadcast to a common shape.
-/

/-! 
@trace E_DIV_UINT_CONSTR_C_0020
@source ops/div/div.md:L558-L559
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_DIV_UINT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_UINT_CONSTR_A_0020</span></b>](#E_DIV_UINT_CONSTR_A_0020) on tensor $A$.
-/


/-! 
@trace float:example:3
@source ops/div/div.md:L238-L250
@coverage deferred
@limitation The document states the third floating-point example only by decimal approximations; the exact roundTiesToEven double values of the three quotients are not given, so no tensor-level example claim is formalized for it
Informal source excerpt:
  ### Example 3
  
  $$
  A = \begin{bmatrix} 1.0 & 10.0 & -10.0 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 3.0 & 3.0 & 3.0 \end{bmatrix}
  $$
  
  $$
  C \approx \begin{bmatrix} 0.3333333333333333 & 3.3333333333333335 & -3.3333333333333335 \end{bmatrix}
  $$
  
  None of the three exact quotients is representable in `double`: each element of $C$ is the exact quotient rounded to the nearest `double`, which is the value displayed. The departure of an implementation from these values is the subject of the profile's accuracy guidelines.
-/


namespace Generated.Div

-- Real numbers

-- DECISION: real tensors are modelled with Rat elements, the rational model of
-- the reals of the profile, so that every element of the result is the exact
-- quotient of the two elements and no approximation takes place.

/-! 
@trace E_DIV_REAL_CONSTR_B_0010
@source ops/div/div.md:L123-L126
@coverage specified
@anchor E_DIV_REAL_CONSTR_B_0010
@declaration Generated.Div.realDivisorNonzero
Informal source excerpt:
  <a id="E_DIV_REAL_CONSTR_B_0010"></a>
  - `[E_DIV_REAL_CONSTR_B_0010]` Non-null divisor
    - Statement: Every element of $B$ is non-null, i.e. $\forall i,\ B[i] \neq 0$.
    - Rationale: The quotient of two real numbers exists only when the divisor is non-null; the condition is the domain of the operator, stated in [<b><span style="font-family: 'Courier New', monospace">E_DIV_REAL_FUNC_0010</span></b>](#E_DIV_REAL_FUNC_0010).
-/

def realDivisorNonzero (B : Sonnx.Tensor Rat) : Prop :=
  ∀ e ∈ B.data, e ≠ 0

-- The element step of real division: the divisor is non-null and the result is
-- the exact quotient.
def realDivRel (a b c : Rat) : Prop :=
  b ≠ 0 ∧ c = a / b

-- DECISION: the domain condition is stated on the elements of B itself.
-- Broadcasting only replicates elements, so this is the same condition as the
-- one the document states on the broadcast tensor B. The remaining conjuncts are
-- the broadcasting constraint, the shape of the result and the elementwise rule.

/-! 
@trace E_DIV_REAL_FUNC_0010
@source ops/div/div.md:L33-L59
@coverage specified
@anchor E_DIV_REAL_FUNC_0010
@declaration Generated.Div.realFunction
Informal source excerpt:
  <a id="E_DIV_REAL_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_DIV_REAL_FUNC_0010]</br></span>
  
  Operator **Div** divides tensor $A$ by tensor $B$ element-wise and stores the result in tensor $C$. Each element of $C$ is the quotient of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \frac{\tilde{A}[i]}{\tilde{B}[i]}
  $$
  
  where
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$,
  - $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - the quotient is the quotient in $\mathbb{R}$ of the two elements.
  
  The division is exact: $C[i]$ is the quotient in $\mathbb{R}$, with no approximation.
  
  **Domain.** The quotient of two real numbers exists only when the divisor is non-null. The domain of the operator is therefore the set of pairs of real tensors whose divisor has no null element, i.e. $\forall i,\ \tilde{B}[i] \neq 0$; that condition is the precondition stated by constraint [<b><span style="font-family: 'Courier New', monospace">E_DIV_REAL_CONSTR_B_0010</span></b>](#E_DIV_REAL_CONSTR_B_0010) on tensor $B$. For every pair of tensors that satisfies it, the result is the one given above.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the quotient of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty. No quotient is formed for an empty result, so that the domain condition bears on the elements of $B$ only.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

def realFunction (A B C : Sonnx.Tensor Rat) : Prop :=
  realDivisorNonzero B ∧
  Sonnx.BroadcastCompatible A B ∧
  Sonnx.BroadcastShapeIs A B C ∧
  Sonnx.ElementwiseBinaryRel realDivRel A B C

-- Floating-point numbers

-- DECISION: the Bool carried by FVal.inf, FVal.zero and FVal.finite is read as
-- the sign bit, true meaning negative. This is the sign of the operand that
-- decides the sign of the exact quotient in each case of the specification.
def fSignXor (s t : Bool) : Bool :=
  if s = t then false else true

-- The signed rational value of a finite magnitude with a sign bit.
def fSignedVal (s : Bool) (r : Rat) : Rat :=
  if s then -r else r

-- IEEE 754 division for one floating-point format. The NaN operand is
-- propagated; the invalid operations infinity/infinity and zero/zero give NaN;
-- a null quotient and an infinite result carry the sign of the exact quotient;
-- the finite case rounds the exact quotient with roundTiesToEven. DECISION: the
-- case of an infinite dividend and a null divisor, which the table of the
-- document does not list, is given the infinity of the sign of the exact
-- quotient, as IEEE 754 prescribes and as the document's statement that every
-- combination of operands of the type has a result requires.
def floatDivRel (fmt : Sonnx.FloatFormat) :
    Sonnx.FVal → Sonnx.FVal → Sonnx.FVal → Prop
  | Sonnx.FVal.nan, _, c => c = Sonnx.FVal.nan
  | _, Sonnx.FVal.nan, c => c = Sonnx.FVal.nan
  | Sonnx.FVal.inf _, Sonnx.FVal.inf _, c => c = Sonnx.FVal.nan
  | Sonnx.FVal.zero _, Sonnx.FVal.zero _, c => c = Sonnx.FVal.nan
  | Sonnx.FVal.inf sa, Sonnx.FVal.zero sb, c =>
      c = Sonnx.FVal.inf (fSignXor sa sb)
  | Sonnx.FVal.inf sa, Sonnx.FVal.finite sb rb, c =>
      rb ≠ 0 ∧ c = Sonnx.FVal.inf (fSignXor sa sb)
  | Sonnx.FVal.zero sa, Sonnx.FVal.inf sb, c =>
      c = Sonnx.FVal.zero (fSignXor sa sb)
  | Sonnx.FVal.zero sa, Sonnx.FVal.finite sb rb, c =>
      rb ≠ 0 ∧ c = Sonnx.FVal.zero (fSignXor sa sb)
  | Sonnx.FVal.finite sa ra, Sonnx.FVal.zero sb, c =>
      ra ≠ 0 ∧ c = Sonnx.FVal.inf (fSignXor sa sb)
  | Sonnx.FVal.finite sa _, Sonnx.FVal.inf sb, c =>
      c = Sonnx.FVal.zero (fSignXor sa sb)
  | Sonnx.FVal.finite sa ra, Sonnx.FVal.finite sb rb, c =>
      rb ≠ 0 ∧ Sonnx.RoundToFormat fmt (fSignedVal sa ra / fSignedVal sb rb) c

-- DECISION: the floating-point type is a parameter k identifying float16,
-- float or double, and it fixes the format used by the element step. All three
-- tensors live in FVal in the model, so the type consistency of A, B and C is
-- the shared TypeConsistent statement applied to that single kind.

/-! 
@trace E_DIV_FLOAT_FUNC_0010
@source ops/div/div.md:L166-L206
@coverage specified
@anchor E_DIV_FLOAT_FUNC_0010
@declaration Generated.Div.floatFunction
Informal source excerpt:
  <a id="E_DIV_FLOAT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_DIV_FLOAT_FUNC_0010]</br></span>
  
  Operator **Div** divides tensor $A$ by tensor $B$ element-wise according to IEEE 754 floating-point semantics and stores the result in tensor $C$. Each element of $C$ is the floating-point quotient of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below. No element of $B$ is excluded: every combination of operands of the type, including the special numbers, has a result of the type.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \tilde{A}[i] /_{(\text{f})} \tilde{B}[i] =
  \begin{cases}
  \text{NaN} & \text{if } \tilde{A}[i] \text{ or } \tilde{B}[i] \text{ is NaN} \\
  \text{NaN} & \text{if } \tilde{A}[i] \text{ and } \tilde{B}[i] \text{ are both } \pm\infty, \text{ or both } \pm 0 \\
  \pm\infty & \text{if } \tilde{A}[i] \text{ is finite and non-null and } \tilde{B}[i] \text{ is } \pm 0, \text{ or } \tilde{A}[i] \text{ is } \pm\infty \text{ and } \tilde{B}[i] \text{ is finite and non-null} \\
  \pm 0 & \text{if } \tilde{A}[i] \text{ is } \pm 0 \text{ and } \tilde{B}[i] \text{ is finite and non-null, or } \tilde{A}[i] \text{ is finite and } \tilde{B}[i] \text{ is } \pm\infty \\
  \text{round}\!\left(\tilde{A}[i] / \tilde{B}[i]\right) & \text{otherwise}
  \end{cases}
  $$
  
  where
  - $/_{(\text{f})}$ is the division of the floating-point type of $A$, $B$ and $C$, i.e. $/_{(\text{f16})}$, $/_{(\text{f32})}$ or $/_{(\text{f64})}$ according to that type, whereas the $/$ of the last case is the division of $\mathbb{R}$: there, $\tilde{A}[i] / \tilde{B}[i]$ is the exact quotient of the two elements,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $\pm$ in the third and fourth cases is the sign of the exact quotient: $+$ when the two operands have the same sign and $-$ when they have opposite signs,
  - $\text{round}(x)$ is the value of $x$ rounded to the nearest value of the type of $A$ and $B$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded when deciding an overflow: a magnitude whose rounded value exceeds the largest finite value of the type gives an infinity of the sign of $x$, and a magnitude below the smallest subnormal value of the type gives a zero of the sign of $x$.
  
  **Operands that are NaN.** The first case propagates the NaN: a NaN operand gives a NaN result, whatever the other operand is. The NaN of this case is a quiet NaN; its sign and its payload are not specified, so that every quiet NaN of the type of $A$ and $B$ is a conforming result.
  
  **Invalid operation.** The second case is the invalid operation of IEEE 754 section 7.2 for division: the quotient $\infty/\infty$ and the quotient $0/0$ have no value in the extended reals, and the result is NaN, again any quiet NaN of the type.
  
  **Division by zero.** The third case is the division-by-zero exception of IEEE 754: a finite non-null dividend divided by a zero gives an infinity whose sign is the sign of the exact quotient, so that $1.0/(+0)$ gives $+\infty$ and $-1.0/(+0)$ gives $-\infty$. The division of an infinity by a finite non-null value also gives an infinity, of the sign of the exact quotient; that case is not an exception of the standard, the result being the limit of the quotients of the finite values.
  
  **Null quotients.** The fourth case gives a null result whose sign is the sign of the exact quotient: $\tilde{A}[i] / \tilde{B}[i]$ is null when the dividend is null and the divisor is finite non-null, and when the divisor is infinite and the dividend is finite. The signs of the operands then decide the sign of the zero: $(+0)/2.0$ gives $+0$ and $(-0)/2.0$ gives $-0$, while $(+0)/(-2.0)$ gives $-0$. A null dividend divided by a null divisor is not a null quotient: it falls in the second case and gives NaN. The last case also produces a null result when a non-null exact quotient is so small that it rounds to zero; its sign is then the sign of that exact quotient as well.
  
  **Finite operands.** In the last case both operands are finite and the divisor is non-null, so that the exact quotient is a non-null real number. The result is that quotient rounded as defined above. When the exact quotient is representable in the type, no rounding takes place; otherwise the result is the nearest value of the type, and it may be a subnormal value, a zero, or an infinity when the exponent of the exact quotient exceeds the largest exponent of the type.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the quotient of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

def floatFunction (k : Sonnx.FloatKind)
    (A B C : Sonnx.Tensor Sonnx.FVal) : Prop :=
  Sonnx.TypeConsistent k k k ∧
  Sonnx.BroadcastCompatible A B ∧
  Sonnx.BroadcastShapeIs A B C ∧
  Sonnx.ElementwiseBinaryRel (floatDivRel (Sonnx.FloatFormat.ofKind k)) A B C

-- Signed integer numbers

-- Truncating integer division: the quotient rounded toward zero. The value at a
-- null divisor is a dummy; the relations below leave the result unconstrained
-- there.
def intTruncDiv (a b : Int) : Int :=
  if b = 0 then 0
  else
    let q : Nat := a.natAbs / b.natAbs
    let an : Bool := decide (a < 0)
    let bn : Bool := decide (b < 0)
    if an = bn then Int.ofNat q else -Int.ofNat q

-- DECISION: a signed integer tensor is modelled by Int elements together with
-- the bit width n of its type; this predicate records that its elements are
-- values of that type.
def IsSignedNTensor (n : Nat) (A : Sonnx.Tensor Int) : Prop :=
  ∀ e ∈ A.data, Sonnx.InSignedRange n e

-- Signed integer division. DECISION: the document determines no value for an
-- element whose divisor is null, so at a null divisor the relation imposes
-- nothing on the corresponding element of C; at every other element the result
-- is the truncating quotient reduced to the type. The single quotient outside
-- the range, -2^(n-1) over -1, is exactly the value congruent to it modulo 2^n.
def signedDivRel (n : Nat) (a b c : Int) : Prop :=
  b ≠ 0 → c = Sonnx.sintWrap n (intTruncDiv a b)

-- The signed integer division specification: the three tensors have the same
-- signed n-bit type, their shapes broadcast, and every element of C is the
-- truncating quotient of the broadcast elements, reduced to the type.

/-! 
@trace E_DIV_INT_FUNC_0010
@source ops/div/div.md:L331-L364
@coverage specified
@anchor E_DIV_INT_FUNC_0010
@declaration Generated.Div.intFunction
Informal source excerpt:
  <a id="E_DIV_INT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_DIV_INT_FUNC_0010]</br></span>
  
  Operator **Div** divides tensor $A$ by tensor $B$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the quotient of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below, truncated toward zero. Tensors $A$, $B$ and $C$ have the same signed $n$-bit type, and the result is a value of that type.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] =
  \begin{cases}
  \operatorname{trunc}\!\left(\tilde{A}[i] / \tilde{B}[i]\right) & \text{if } \operatorname{trunc}\!\left(\tilde{A}[i] / \tilde{B}[i]\right) \le 2^{n-1}-1 \\[4pt]
  \operatorname{trunc}\!\left(\tilde{A}[i] / \tilde{B}[i]\right) - 2^n & \text{if } \operatorname{trunc}\!\left(\tilde{A}[i] / \tilde{B}[i]\right) = 2^{n-1}
  \end{cases}
  $$
  
  where
  - $\tilde{A}[i] / \tilde{B}[i]$ is the exact quotient of the two elements, the quotient in $\mathbb{Q}$ of two integers whose divisor is non-null,
  - $\operatorname{trunc}(x)$ is the integer part of $x$, i.e. $x$ rounded toward zero: $\operatorname{trunc}(x) = \lfloor x \rfloor$ when $x \ge 0$ and $\operatorname{trunc}(x) = \lceil x \rceil$ when $x < 0$, so that $\operatorname{trunc}(3.5) = 3$ and $\operatorname{trunc}(-3.5) = -3$,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$,
  - $n$ is the number of bits of the type of $A$ and $B$.
  
  **Truncating division.** The quotient $\operatorname{trunc}(a/b)$ is the unique integer $q$ such that $a = q \cdot b + r$, with $|r| < |b|$ and with $r$ null or of the sign of $a$. It is the quotient ONNX prescribes for the integer types, and it differs from the floor: for `int8`, $7/2 = 3$ and $-7/2 = -3$, whereas the floor of $-3.5$ is $-4$. The sign of the result is the sign of the exact quotient: the quotient of two operands of the same sign is non-negative, and the quotient of two operands of opposite signs is non-positive, so that a null result is the neutral value $0$ of the type, with no sign.
  
  **The one quotient outside the range.** The values of an $n$-bit signed type lie in $[-2^{n-1}, 2^{n-1}-1]$, so that the quotient of two of its values, truncated toward zero, lies in $[-2^{n-1}, 2^{n-1}]$. Only the value $2^{n-1}$ leaves the range, and only one pair of operands produces it: $a = -2^{n-1}$ and $b = -1$. The second case of the definition applies then, and the result is $2^{n-1} - 2^n = -2^{n-1}$. For `int8`, $-128 / -1$ gives $-128$; the exact quotient $128$ is represented by the value of the type that is congruent to it modulo $2^8$.
  
  **A null divisor.** No element of $B$ is excluded, so that a null element of $B$ makes the exact quotient undefined and the result of the operator undetermined; the condition is described in the "Error conditions" section. Every other pair of operands falls in one of the two cases above, and the result is fully determined.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the quotient of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty. No quotient is formed for an empty result, so that a null element of $B$ in an empty tensor raises no condition.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

def intFunction (n : Nat) (A B C : Sonnx.Tensor Int) : Prop :=
  Sonnx.TypeConsistent n n n ∧
  IsSignedNTensor n A ∧
  IsSignedNTensor n B ∧
  IsSignedNTensor n C ∧
  Sonnx.BroadcastCompatible A B ∧
  Sonnx.BroadcastShapeIs A B C ∧
  Sonnx.ElementwiseBinaryRel (signedDivRel n) A B C

-- Unsigned integer numbers

-- DECISION: as for the signed family, an unsigned integer tensor is modelled by
-- Int elements together with the bit width n of its type.
def IsUnsignedNTensor (n : Nat) (A : Sonnx.Tensor Int) : Prop :=
  ∀ e ∈ A.data, Sonnx.InUnsignedRange n e

-- Unsigned integer division. DECISION: the quotient of two non-negative
-- operands truncated toward zero is their floor and already lies in the range of
-- the type, so the reduction modulo 2^n is the identity there and the document's
-- statement that no wrap and no overflow occur is preserved. At a null divisor
-- the relation imposes nothing on the corresponding element of C, since the
-- document determines no value.
def unsignedDivRel (n : Nat) (a b c : Int) : Prop :=
  b ≠ 0 → c = Sonnx.uintWrap n (intTruncDiv a b)

-- The unsigned integer division specification: the three tensors have the same
-- unsigned n-bit type, their shapes broadcast, and every element of C is the
-- truncating quotient (the floor) of the broadcast elements.

/-! 
@trace E_DIV_UINT_FUNC_0010
@source ops/div/div.md:L462-L488
@coverage specified
@anchor E_DIV_UINT_FUNC_0010
@declaration Generated.Div.uintFunction
Informal source excerpt:
  <a id="E_DIV_UINT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_DIV_UINT_FUNC_0010]</br></span>
  
  Operator **Div** divides tensor $A$ by tensor $B$ element-wise and stores the result in output tensor $C$. Each element of $C$ is the quotient of the elements of $A$ and $B$ that correspond to it under the broadcasting rules given below, truncated toward zero. Tensors $A$, $B$ and $C$ have the same unsigned $n$-bit type, and the result is a value of that type.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \left\lfloor \frac{\tilde{A}[i]}{\tilde{B}[i]} \right\rfloor
  $$
  
  where
  - $\tilde{A}[i] / \tilde{B}[i]$ is the exact quotient of the two elements, the quotient in $\mathbb{Q}$ of two integers whose divisor is non-null,
  - $\lfloor \cdot \rfloor$ is the floor, which for two non-negative operands is the quotient truncated toward zero,
  - $\tilde{A}$ is tensor $A$ broadcast to the shape of $C$, and $\tilde{B}$ is tensor $B$ broadcast to the shape of $C$.
  
  **The result never leaves the range of the type.** The operands are non-negative, so that their exact quotient is non-negative and its floor is at most the dividend: $0 \le \lfloor a/b \rfloor \le a \le 2^n-1$ for every $b \ge 1$. The result of a division by a non-null element is therefore the floor itself, with no reduction modulo $2^n$ and no overflow, and it is exact in the type. For `uint8`, $200/3$ gives $66$ and $255/255$ gives $1$. The truncation and the floor coincide on the unsigned types, so that the distinction between them, which the signed types make visible, has no effect here.
  
  **A null divisor.** No element of $B$ is excluded, so that a null element of $B$ makes the exact quotient undefined and the result of the operator undetermined; the condition is described in the "Error conditions" section, and it is the only condition under which the result is not the value given above.
  
  **Tensors of different shapes.** Tensors $A$ and $B$ are broadcast to the shape of $C$ following the multidirectional (Numpy-style) broadcasting rules: the shapes are aligned on their last dimension; a dimension that one operand does not have, including the leading dimensions missing from the operand with the smaller rank, is treated as a dimension of size $1$; and a dimension of size $1$ is extended to the size of the corresponding dimension of the other operand. Tensor $C$ has the resulting shape. When $A$ and $B$ have the same shape, $\tilde{A}$ is $A$, $\tilde{B}$ is $B$ and $C$ has that shape.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. It is broadcast to the shape of the other operand, so that each element of $C$ is the quotient of the two operands and $C$ has the shape of the other operand.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension is compatible with a dimension of size $1$ of the other operand, and with an equal zero-sized dimension; the corresponding dimension of $C$ then has size $0$, and $C$ is empty. No quotient is formed for an empty result, so that a null element of $B$ in an empty tensor raises no condition.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

def uintFunction (n : Nat) (A B C : Sonnx.Tensor Int) : Prop :=
  Sonnx.TypeConsistent n n n ∧
  IsUnsignedNTensor n A ∧
  IsUnsignedNTensor n B ∧
  IsUnsignedNTensor n C ∧
  Sonnx.BroadcastCompatible A B ∧
  Sonnx.BroadcastShapeIs A B C ∧
  Sonnx.ElementwiseBinaryRel (unsignedDivRel n) A B C

-- Examples

def realEx1A : Sonnx.Tensor Rat :=
  { shape := [3], data := [7, -9, 1] }

def realEx1B : Sonnx.Tensor Rat :=
  { shape := [3], data := [2, 4, 3] }

def realEx1C : Sonnx.Tensor Rat :=
  { shape := [3], data := [7 / 2, -9 / 4, 1 / 3] }

/-! 
@trace real:example:1
@source ops/div/div.md:L63-L75
@coverage example_only
@declaration Generated.Div.realExample1
Informal source excerpt:
  ### Example 1
  
  $$
  A = \begin{bmatrix} 7 & -9 & 1 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 2 & 4 & 3 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} \frac{7}{2} & -\frac{9}{4} & \frac{1}{3} \end{bmatrix}
  $$
  
  The three quotients are given exactly, as the quotients of real numbers: $1/3$ has no finite decimal expansion and is not approximated.
-/

def realExample1 : Sonnx.Tensor Rat × Sonnx.Tensor Rat × Sonnx.Tensor Rat :=
  (realEx1A, realEx1B, realEx1C)

def realExample1Claim : Prop :=
  realFunction realEx1A realEx1B realEx1C

def realEx2A : Sonnx.Tensor Rat :=
  { shape := [2, 2], data := [4, 9, 16, 25] }

def realEx2B : Sonnx.Tensor Rat :=
  { shape := [2], data := [2, 5] }

def realEx2C : Sonnx.Tensor Rat :=
  { shape := [2, 2], data := [2, 9 / 5, 8, 5] }

/-! 
@trace real:example:2
@source ops/div/div.md:L77-L89
@coverage example_only
@declaration Generated.Div.realExample2
Informal source excerpt:
  ### Example 2
  
  $$
  A = \begin{bmatrix} 4 & 9 \\ 16 & 25 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 2 & 5 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} \frac{4}{2} & \frac{9}{5} \\[2pt] \frac{16}{2} & \frac{25}{5} \end{bmatrix} = \begin{bmatrix} 2 & \frac{9}{5} \\[2pt] 8 & 5 \end{bmatrix}
  $$
  
  Tensor $B$ has rank $1$ and tensor $A$ has rank $2$; $B$ is broadcast to the shape of $A$, so that each column of $A$ is divided by the corresponding element of $B$.
-/

def realExample2 : Sonnx.Tensor Rat × Sonnx.Tensor Rat × Sonnx.Tensor Rat :=
  (realEx2A, realEx2B, realEx2C)

def realExample2Claim : Prop :=
  realFunction realEx2A realEx2B realEx2C

def floatEx1A : Sonnx.Tensor Sonnx.FVal :=
  { shape := [3],
    data := [Sonnx.FVal.finite false 6,
             Sonnx.FVal.finite true 9,
             Sonnx.FVal.finite false 1] }

def floatEx1B : Sonnx.Tensor Sonnx.FVal :=
  { shape := [3],
    data := [Sonnx.FVal.finite false 3,
             Sonnx.FVal.finite false 4,
             Sonnx.FVal.finite false 8] }

def floatEx1C : Sonnx.Tensor Sonnx.FVal :=
  { shape := [3],
    data := [Sonnx.FVal.finite false 2,
             Sonnx.FVal.finite true (9 / 4),
             Sonnx.FVal.finite false (1 / 8)] }

/-! 
@trace float:example:1
@source ops/div/div.md:L210-L222
@coverage example_only
@declaration Generated.Div.floatExample1
Informal source excerpt:
  ### Example 1
  
  $$
  A = \begin{bmatrix} 6.0 & -9.0 & 1.0 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 3.0 & 4.0 & 8.0 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 2.0 & -2.25 & 0.125 \end{bmatrix}
  $$
  
  The quotient of two numbers of a floating-point type is not always exact; in this example every quotient is representable and $C$ is the exact result.
-/

def floatExample1 :
    Sonnx.Tensor Sonnx.FVal × Sonnx.Tensor Sonnx.FVal × Sonnx.Tensor Sonnx.FVal :=
  (floatEx1A, floatEx1B, floatEx1C)

def floatExample1Claim : Prop :=
  floatFunction Sonnx.FloatKind.f64 floatEx1A floatEx1B floatEx1C

def floatEx2A : Sonnx.Tensor Sonnx.FVal :=
  { shape := [8],
    data := [Sonnx.FVal.finite false 1,
             Sonnx.FVal.finite true 1,
             Sonnx.FVal.inf false,
             Sonnx.FVal.inf false,
             Sonnx.FVal.zero false,
             Sonnx.FVal.zero true,
             Sonnx.FVal.zero false,
             Sonnx.FVal.nan] }

def floatEx2B : Sonnx.Tensor Sonnx.FVal :=
  { shape := [8],
    data := [Sonnx.FVal.zero false,
             Sonnx.FVal.zero false,
             Sonnx.FVal.inf false,
             Sonnx.FVal.finite false 4,
             Sonnx.FVal.finite false 4,
             Sonnx.FVal.finite false 4,
             Sonnx.FVal.zero false,
             Sonnx.FVal.finite false 1] }

def floatEx2C : Sonnx.Tensor Sonnx.FVal :=
  { shape := [8],
    data := [Sonnx.FVal.inf false,
             Sonnx.FVal.inf true,
             Sonnx.FVal.nan,
             Sonnx.FVal.inf false,
             Sonnx.FVal.zero false,
             Sonnx.FVal.zero true,
             Sonnx.FVal.nan,
             Sonnx.FVal.nan] }

/-! 
@trace float:example:2
@source ops/div/div.md:L224-L236
@coverage example_only
@declaration Generated.Div.floatExample2
Informal source excerpt:
  ### Example 2
  
  $$
  A = \begin{bmatrix} 1.0 & -1.0 & \text{+inf} & \text{+inf} & \text{+0} & \text{-0} & \text{+0} & \text{NaN} \end{bmatrix}
  \quad
  B = \begin{bmatrix} \text{+0} & \text{+0} & \text{+inf} & 4.0 & 4.0 & 4.0 & \text{+0} & 1.0 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} \text{+inf} & \text{-inf} & \text{NaN} & \text{+inf} & \text{+0} & \text{-0} & \text{NaN} & \text{NaN} \end{bmatrix}
  $$
  
  The first two elements are divisions by zero: the dividend is finite and non-null and the divisor is $+0$, so the result is an infinity whose sign is the sign of the exact quotient, positive for $1.0$ and negative for $-1.0$. The third element is the invalid operation $\infty/\infty$, which yields NaN. The fourth is the division of an infinity by a finite non-null value, which gives the infinity of the sign of the quotient. The fifth and the sixth elements are divisions of a zero by a finite non-null value: the exact quotient is null, and its sign, positive when the operands have the same sign and negative otherwise, is the sign of the result. The seventh element is the invalid operation $0/0$. The eighth is the propagation of a NaN operand.
-/

def floatExample2 :
    Sonnx.Tensor Sonnx.FVal × Sonnx.Tensor Sonnx.FVal × Sonnx.Tensor Sonnx.FVal :=
  (floatEx2A, floatEx2B, floatEx2C)

def floatExample2Claim : Prop :=
  floatFunction Sonnx.FloatKind.f64 floatEx2A floatEx2B floatEx2C

def intEx1A : Sonnx.Tensor Int :=
  { shape := [5], data := [7, -7, 7, -7, -128] }

def intEx1B : Sonnx.Tensor Int :=
  { shape := [5], data := [2, 2, -2, -2, -1] }

def intEx1C : Sonnx.Tensor Int :=
  { shape := [5], data := [3, -3, -3, 3, -128] }

/-! 
@trace int:example:1
@source ops/div/div.md:L368-L382
@coverage example_only
@declaration Generated.Div.intExample1
Informal source excerpt:
  ### Example 1
  
  For the `int8` type:
  
  $$
  A = \begin{bmatrix} 7 & -7 & 7 & -7 & -128 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 2 & 2 & -2 & -2 & -1 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 3 & -3 & -3 & 3 & -128 \end{bmatrix}
  $$
  
  The first four quotients are $3.5$, $-3.5$, $-3.5$ and $3.5$: each is truncated toward zero, which gives $3$, $-3$, $-3$ and $3$, and not the floors $-3.5 \to -4$. The last quotient is $(-128)/(-1) = 128$, which exceeds the maximum value $127$ of `int8`: the second case of the definition applies and the result is $128 - 2^8 = -128$, the value of the type congruent to $128$ modulo $2^8$.
-/

def intExample1 : Sonnx.Tensor Int × Sonnx.Tensor Int × Sonnx.Tensor Int :=
  (intEx1A, intEx1B, intEx1C)

def intExample1Claim : Prop :=
  intFunction 8 intEx1A intEx1B intEx1C

def uintEx1A : Sonnx.Tensor Int :=
  { shape := [4], data := [7, 200, 255, 6] }

def uintEx1B : Sonnx.Tensor Int :=
  { shape := [4], data := [2, 3, 255, 6] }

def uintEx1C : Sonnx.Tensor Int :=
  { shape := [4], data := [3, 66, 1, 1] }

/-! 
@trace uint:example:1
@source ops/div/div.md:L492-L506
@coverage example_only
@declaration Generated.Div.uintExample1
Informal source excerpt:
  ### Example 1
  
  For the `uint8` type:
  
  $$
  A = \begin{bmatrix} 7 & 200 & 255 & 6 \end{bmatrix}
  \quad
  B = \begin{bmatrix} 2 & 3 & 255 & 6 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 3 & 66 & 1 & 1 \end{bmatrix}
  $$
  
  The quotient $7/2 = 3.5$ is truncated toward zero and gives $3$; $200/3 = 66.67$ gives $66$; $255/255 = 1$ and $6/6 = 1$ are exact. Each of the four results lies in the range $[0, 255]$ of `uint8`, as every quotient of the type does.
-/

def uintExample1 : Sonnx.Tensor Int × Sonnx.Tensor Int × Sonnx.Tensor Int :=
  (uintEx1A, uintEx1B, uintEx1C)

def uintExample1Claim : Prop :=
  uintFunction 8 uintEx1A uintEx1B uintEx1C

end Generated.Div

-- Fully-qualified declarations named by the trace entries.
#check @Generated.Div.floatExample1
#check @Generated.Div.floatExample2
#check @Generated.Div.floatFunction
#check @Generated.Div.intExample1
#check @Generated.Div.intFunction
#check @Generated.Div.realDivisorNonzero
#check @Generated.Div.realExample1
#check @Generated.Div.realExample2
#check @Generated.Div.realFunction
#check @Generated.Div.uintExample1
#check @Generated.Div.uintFunction
#check @Sonnx.BroadcastCompatible
#check @Sonnx.BroadcastShapeIs
#check @Sonnx.TypeConsistent
