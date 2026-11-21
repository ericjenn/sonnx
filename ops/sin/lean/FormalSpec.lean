import Sonnx
/-!
@generated-by sonnx-lean-pipeline
@source ops/sin/sin.md
@source-digest cfd53e99b2995c415ed738e3c4644c36
@library-digest 3162109b4c926c2fa5dd5f8bd2eb2d90fb17a0480eccc0216a4ea9bd2b43ffab
Compilation and trace validity are separate from semantic completeness; see the trace report.
-/


/-! 
@trace E_SIN_FLOAT_CONSTR_A_0010
@source ops/sin/sin.md:L268-L270
@coverage specified
@anchor E_SIN_FLOAT_CONSTR_A_0010
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  <a id="E_SIN_FLOAT_CONSTR_A_0010"></a>
  - `[E_SIN_FLOAT_CONSTR_A_0010]` Type consistency
    - Statement: Tensors $A$ and $C$ have the same type.
-/

/-! 
@trace E_SIN_FLOAT_CONSTR_C_0020
@source ops/sin/sin.md:L282-L283
@coverage specified
@declaration Sonnx.TypeConsistent
Informal source excerpt:
  - `[E_SIN_FLOAT_CONSTR_C_0020]` Type consistency
    - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_SIN_FLOAT_CONSTR_A_0010</span></b>](#E_SIN_FLOAT_CONSTR_A_0010) on tensor $A$.
-/


/-! 
@trace real:example:1
@source ops/sin/sin.md:L59-L69
@coverage deferred
@limitation Exact real sine values for multiples of pi/6, pi/4 and pi/2 are not available in the core library; the example is not formalized.
Informal source excerpt:
  ### Example 1
  
  $$
  A = \begin{bmatrix} 0 & \pi/6 & \pi/4 & \pi/2 & \pi & 3\pi/2 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 0 & 1/2 & \sqrt{2}/2 & 1 & 0 & -1 \end{bmatrix}
  $$
  
  The result is exact: the sine of a multiple of $\pi$ is null, and the sine of $\pi/6$, of $\pi/4$ and of $\pi/2$ is the value given.
-/

/-! 
@trace real:example:2
@source ops/sin/sin.md:L71-L81
@coverage deferred
@limitation Exact real sine and the periodicity values involving pi are not available in the core library; the example is not formalized.
Informal source excerpt:
  ### Example 2
  
  $$
  A = \begin{bmatrix} -\pi/6 & \pi/6 & 13\pi/6 \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} -1/2 & 1/2 & 1/2 \end{bmatrix}
  $$
  
  The second and the third elements are equal although their operands are not: the sine is periodic of period $2\pi$ and $13\pi/6 = \pi/6 + 2\pi$. The first element is the opposite of the second: the sine is odd.
-/

/-! 
@trace float:example:1
@source ops/sin/sin.md:L200-L212
@coverage deferred
@limitation The exact real sine and the library nearest-rounding operation are parameters here, so the concrete double result is not formally checked.
Informal source excerpt:
  ### Example 1
  
  For `double`:
  
  $$
  A = \begin{bmatrix} 0.0 & -0.0 & 1.0\text{e-}8 & -1.0\text{e-}8 & \text{+inf} & \text{-inf} & \text{NaN} \end{bmatrix}
  $$
  
  $$
  C = \begin{bmatrix} 0.0 & -0.0 & 1.0\text{e-}8 & -1.0\text{e-}8 & \text{NaN} & \text{NaN} & \text{NaN} \end{bmatrix}
  $$
  
  A zero is its own sine, sign included: the sine of $-0.0$ is $-0.0$. An element of sufficiently small magnitude is its own sine as well, since the difference between the sine of $x$ and $x$ is at most $x^3/6$: the sine of $1.0\text{e-}8$ is $1.0\text{e-}8$. The sine of an infinity has no value and the result is NaN, as is the sine of a NaN.
-/

/-! 
@trace float:example:2
@source ops/sin/sin.md:L214-L226
@coverage deferred
@limitation The exact real sine and the library nearest-rounding operation are parameters here, so the concrete float result is not formally checked.
Informal source excerpt:
  ### Example 2
  
  For `float`:
  
  $$
  A = \begin{bmatrix} 3.1415927 & 6.2831855 & 1.0\text{e}30 \end{bmatrix}
  $$
  
  $$
  C \approx \begin{bmatrix} -8.7422777\text{e-}08 & 1.7484555\text{e-}07 & -0.7911634 \end{bmatrix}
  $$
  
  The first two elements of $A$ are the nearest `float` values of $\pi$ and of $2\pi$. Their sines are not null, and no value of the type has a null sine except the zeros: the sine is null at the multiples of $\pi$ of the real numbers, and the nearest values of the type to those multiples are not those multiples. The third element shows that the sine of an operand far outside the interval $[-1, 1]$ is still an element of the interval.
-/

/-! 
@trace float:example:3
@source ops/sin/sin.md:L228-L240
@coverage deferred
@limitation The exact real sine and the library nearest-rounding operation are parameters here, so the concrete double result is not formally checked.
Informal source excerpt:
  ### Example 3
  
  For `double`:
  
  $$
  A = \begin{bmatrix} 1.0 & -1.0 & 0.5 & 2.0 \end{bmatrix}
  $$
  
  $$
  C \approx \begin{bmatrix} 0.8414709848078965 & -0.8414709848078965 & 0.479425538604203 & 0.9092974268256817 \end{bmatrix}
  $$
  
  None of the four exact sines is a value of `double`: each element of $C$ is the nearest value of the type, the value the specification states. The first two elements show that the result is odd: they are the two first values of the type around the sine of $1.0$, with opposite signs.
-/


namespace Generated

namespace Sin

/-! 
@trace E_SIN_REAL_CONSTR_C_0010
@source ops/sin/sin.md:L109-L110
@coverage specified
@declaration Generated.Sin.sinShapeDefinition
Informal source excerpt:
  - `[E_SIN_REAL_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$.
-/

/-! 
@trace E_SIN_FLOAT_CONSTR_C_0010
@source ops/sin/sin.md:L280-L281
@coverage specified
@declaration Generated.Sin.sinShapeDefinition
Informal source excerpt:
  - `[E_SIN_FLOAT_CONSTR_C_0010]` Shape definition
    - Statement: Tensor $C$ has the shape of $A$.
-/

/-- The shape constraint of `Sin`: the result has the shape of the operand. -/
def sinShapeDefinition {α : Type} (A C : Sonnx.Tensor α) : Prop :=
  C.shape = A.shape

/-- A generic elementwise unary relation: both tensors are well-formed, `C` has the shape
of `A`, and every element of `C` is related to the corresponding element of `A`. -/
def sinElementwiseUnaryRel {α : Type} (R : α → α → Prop)
    (A C : Sonnx.Tensor α) : Prop :=
  A.wellFormed ∧ C.wellFormed ∧ sinShapeDefinition A C ∧
    ∀ i, i ∈ Sonnx.indices A.shape →
      ∃ a c, A.read i = some a ∧ C.read i = some c ∧ R a c

/--
A model of the exact real sine on the rational representatives used by the profile.
`DECISION:` the core `Sonnx` vocabulary has no real numbers or analytic sine, so the exact
analytic function cannot be defined here.  The structure records the properties of the
specified sine that are expressible without a real number type: oddness, the zero value and
the range.  Periodicity and the exact analytic values involving `π` are not formalized.
-/
structure RealSine where
  sin : Rat → Rat
  odd : ∀ x, sin (-x) = -sin x
  zero_zero : sin 0 = 0
  range : ∀ x, (-1 : Rat) ≤ sin x ∧ sin x ≤ (1 : Rat)

/-! 
@trace E_SIN_REAL_FUNC_0010
@source ops/sin/sin.md:L30-L55
@coverage partial
@anchor E_SIN_REAL_FUNC_0010
@declaration Generated.Sin.RealSinRel
@limitation The core library has no real numbers or analytic sine; RealSine records only oddness, the zero value and the range, while periodicity and the exact analytic value are not formalized.
Informal source excerpt:
  <a id="E_SIN_REAL_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_SIN_REAL_FUNC_0010]</br></span>
  
  Operator **Sin** applies the sine to tensor $A$ element-wise and stores the result in tensor $C$. Each element of $C$ is the sine of the element of $A$ that corresponds to it.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] = \sin(A[i])
  $$
  
  The sine is the function of $\mathbb{R}$ into the interval $[-1, 1]$; the value of $C[i]$ is that function of the element, exactly, with no approximation and no rounding.
  
  **The sine is odd.** $\sin(-x) = -\sin(x)$ for every real number $x$: changing the sign of an element of $A$ changes the sign of the corresponding element of $C$ and leaves its magnitude unchanged.
  
  **The sine is periodic.** $\sin(x + 2k\pi) = \sin(x)$ for every integer $k$ and every real number $x$: adding any multiple of $2\pi$ to an element of $A$ leaves the corresponding element of $C$ unchanged. The period of the sine is $2\pi$: no smaller positive value leaves it unchanged.
  
  **The sine vanishes on the multiples of $\pi$.** $C[i] = 0$ if and only if $A[i] = k\pi$ for an integer $k$.
  
  **The values of the sine lie in $[-1, 1]$.** $|C[i]| \le 1$ for every element: $C[i] = 1$ if and only if $A[i] = \pi/2 + 2k\pi$, and $C[i] = -1$ if and only if $A[i] = -\pi/2 + 2k\pi$, for an integer $k$.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. Its single element is an element of $A$ like any other: $C$ is a rank-0 tensor whose single element is the sine of that element.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension of $A$ is a dimension of size $0$ of $C$: $C$ is empty along it, the corresponding elements of $C$ do not exist, and the sine of no element is computed.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/--
The real `Sin` operator, parameterized by a model of the exact real sine.  For every index,
the output element is the sine of the input element, exactly as `C[i] = sin(A[i])`.
-/
def RealSinRel (S : RealSine) (A C : Sonnx.Tensor Rat) : Prop :=
  sinElementwiseUnaryRel (fun x y => y = S.sin x) A C

/--
The element step of floating-point `Sin`.  Non-finite operands produce NaN, a zero operand is
its own sine (with its sign preserved), and every other finite operand is rounded from the
exact sine.  `DECISION:` the exact sine and the nearest-rounding operation are explicit
parameters because the core library does not provide an analytic sine and the module does not
pin the rounding parameter to `Sonnx.RoundToFormat`.
-/
def FloatSinStep (S : RealSine) (round : Rat → Sonnx.FVal)
    (x y : Sonnx.FVal) : Prop :=
  match x.toRat with
  | none => y = Sonnx.FVal.nan
  | some q =>
      if q = 0 then y = x
      else y = round (S.sin q)

/-! 
@trace E_SIN_FLOAT_FUNC_0010
@source ops/sin/sin.md:L136-L172
@coverage partial
@anchor E_SIN_FLOAT_FUNC_0010
@declaration Generated.Sin.FloatSinRel
@limitation The exact analytic sine and the nearest-rounding operation are parameters; the relation does not pin them to the core library.
Informal source excerpt:
  <a id="E_SIN_FLOAT_FUNC_0010"></a>
  <span style="background: red; color: white; font-size:0.7em;">[E_SIN_FLOAT_FUNC_0010]</br></span>
  
  Operator **Sin** applies the sine to tensor $A$ element-wise and stores the result in tensor $C$, in the type of $A$. Each element of $C$ is the sine of the element of $A$ that corresponds to it.
  
  For any [tensor index](../../common/definitions.md#tensor_index) $i$ of the result $C$:
  
  $$
  C[i] =
  \begin{cases}
  \text{NaN} & \text{if } A[i] \text{ is NaN or an infinity} \\
  A[i] & \text{if } A[i] \text{ is a zero} \\
  \text{nearest}(\sin(A[i])) & \text{otherwise}
  \end{cases}
  $$
  
  where
  - $\sin(A[i])$ is the value of the sine of the element, in $\mathbb{R}$: the sine of the real number that the element of $A$ is,
  - $\text{nearest}(x)$ is the value of $x$ rounded to the nearest value of the type of $A$ using the roundTiesToEven attribute of IEEE 754. Since $|\sin x| \le 1$ for every real number $x$, no result leaves the range of the type, and the exponent range of the type plays no role.
  
  The last case gives the value of the type of $A$ that represents the exact sine of the element: the sine of a value of a floating-point type is in general not a value of that type, and that value is the representation of it.
  
  **The exact sine is what the operator specifies, and the way an implementation computes it is not specified here.** IEEE 754 requires a correctly rounded result of the operations it makes mandatory and of the mathematical functions of its clause 9.2 that a language chooses to provide as conforming operations, but it leaves to the language the choice of providing those functions and of requiring them to be correctly rounded, and it fixes no result for the sine of a value of a floating-point type. The value above is therefore the value the operator specifies, and the difference between the value returned by a particular implementation and that value is the introduced error of that implementation, whose analysis is the subject of the [accuracy guidelines](../../docs/accuracy.md) of SONNX and is not part of this specification.
  
  **Every result lies in $[-1, 1]$, and no computation underflows.** No element of $C$ is an infinity, since no element of the exact sine exceeds $1$ in magnitude. No element of $C$ is a subnormal value obtained by underflow either: for an element of sufficiently small magnitude, the difference $|\sin x - x|$, which is at most $x^3/6$, is far below half a unit in the last place of $x$, so that the element is its own sine, and the subnormal values of a result are therefore the elements of $A$ themselves. An element of $C$ is null if and only if the element of $A$ is a zero.
  
  **The sine is odd, and the result is odd too.** $\sin(-x) = -\sin(x)$ for every real number $x$, and the negation of a value of a floating-point type is exact: the result for the operand $-A[i]$ is the negation of the result for $A[i]$, the sign bit included. The second case above follows that rule: the sine of $+0$ is $+0$ and the sine of $-0$ is $-0$.
  
  **The sine is not periodic on the values of a floating-point type.** $2\pi$ is a value of none of the three types, and the periodicity of the sine is a property of the real numbers: adding a value of the type to an element of $A$ does not leave the corresponding element of $C$ unchanged in general. The result is in particular null only for a null element, and not for the elements of the type that are closest to a multiple of the period: the nearest `double` value of $\pi$ is not a value whose sine is null, and its sine is $1.2246467991473532\text{e-}16$.
  
  **Values of the operand.** The operand may be any value of the type, including the special numbers $\pm 0$, $\pm\text{inf}$ and NaN. The cases above give the result for every combination of them. The sine has no limit at infinity and no value of the type represents it, so that the sine of an infinity is the invalid operation of IEEE 754 section 7.2 and the result is NaN. The NaN of the first case is a quiet NaN; its sign and its payload are not specified, whether the NaN comes from an operand or from the invalid operation, so that every quiet NaN of the type of $A$ is a conforming result.
  
  **Tensors with no dimension.** A tensor with no dimension (a rank-0 tensor) is a tensor whose shape is empty. Its single element is an element of $A$ like any other: $C$ is a rank-0 tensor whose single element is the sine of that element.
  
  **Zero-sized dimensions.** An operand may have a zero-sized dimension. Such a dimension of $A$ is a dimension of size $0$ of $C$: $C$ is empty along it, the corresponding elements of $C$ do not exist, and the sine of no element is computed.
  
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
-/

/--
The floating-point `Sin` operator, parameterized by the exact real sine and the nearest
rounding to the operand format.  The relation states the elementwise cases of the informal
function clause.
-/
def FloatSinRel (S : RealSine) (round : Rat → Sonnx.FVal)
    (A C : Sonnx.Tensor Sonnx.FVal) : Prop :=
  sinElementwiseUnaryRel (FloatSinStep S round) A C

end Sin

end Generated

-- Fully-qualified declarations named by the trace entries.
#check @Generated.Sin.sinShapeDefinition
#check @Sonnx.TypeConsistent
