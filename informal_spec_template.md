<!--
Template of an operator specification, in compliance with the "Structure and
contents of the specification" section of the guidelines given in
informal_corrected.md.

Copy this file into the "ops" folder under the name of the operator, replace
every placeholder written between angle brackets and every "..." by the
corresponding content, then delete the instruction comments (kept between an
opening comment marker and its closing counterpart).

The hyperlinks below are the ones a specification placed in the "ops" folder
must use; they are relative to that folder and not to this template.
-->

# Contents

<!-- One entry per section of this specification, in the same order, and one
     section per entry; each entry links to the anchor carried by its
     section, and the real-number specification comes first when the operator
     applies to numeric values. The three entries below are those of an
     operator that applies to real numbers, to the IEEE 754 floating-point
     types and to the integer types, taken from the guidelines; give each one
     the types it stands for, and delete the entries that do not apply. -->

- **Op** operator for type [real](#real)
- **Op** operator for types [`float16`, `float`, `double`](#float)
- **Op** operator for types [`int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`](#int)

Based on ONNX documentation [Op version \<n\>](\<link to the ONNX definition of the operator\>).

Revision \<date\>: based on ONNX opset \<n\>; \<what this revision changes\>.

<a id="real"></a>

# **Op** (\<type 1\>, \<type 2\>)

<!-- Repeat the whole block, from this anchor to the last "Constraints"
     subsection below, once per entry of the "Contents" list, and give each
     block the anchor its entry links to (for a family, the anchor the
     guidelines prescribe below). For a block that covers a family, keep the
     "where" line and the justification line; for a block that covers a
     single type, delete them. -->

where \<family 1\> is in {\<type\>, \<type\>, ...} and \<family 2\> is in {\<type\>, \<type\>, ...}.

The types of this section share one semantics: \<one-line justification\>.

## Signature

Definition of operator **Op** signature:

 $A,B,...,C = \text{Op}(X,Y,...,Z)$

 where
 - $X$: Brief description of argument $X$
 - ...
 - $Z$: Brief description of argument $Z$
 - $A$: Brief description of output $A$
 - ...
 - $C$: Brief description of output $C$

<!-- An optional input or an optional output is written between square
     brackets, e.g. $C = \text{Op}(X, Y\,[, Z])$, and the "Function" section
     states what the operator does when it is absent. -->

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

<!-- List below, in a table, the restrictions the operator introduces: what
     each of them limits, and its origin, i.e. "Transient" or a link to the
     requirement it comes from. When the operator introduces none, replace
     the table by the following sentence, which the guidelines make
     mandatory in that case:

       No specific restrictions apply to the **Op** operator.
-->

| Restriction    | Statement | Origin |
| -------- | ------- | ------- |
| `[R1]` | \<what the restriction limits\> | \<"Transient", or a link to the requirement it comes from\> |

## Function

<!-- The text of this section is the specifying part of the specification; it
     is enclosed between the two red tags below, whose components are the
     operator name, the type family in uppercase letters, the section, and a
     4-digit id. -->

<span style="background: red; color: white; font-size:0.7em;">[E_\<op\>_\<type\>_FUNC_0010]</br></span>

**Part 1.** \<A short description of the operator, in a few sentences.\>

**Part 2.** \<The detailed description. Give the complete formula first and
define its elements and sub-expressions afterwards, with "where ..." or
"in which ...".\>

<!-- Or state the operator axiomatically instead of the formula below, when
     its result is determined by a property rather than by a formula — the
     inverses of the other operators are the usual case. State the conditions
     the result satisfies, conditions that only one value satisfies, and the
     domain on which they are stated; for **Asin**, "For every $x$ in
     $[-1, 1]$, the result is the unique value $y$ such that $\sin(y) = x$ and
     $y \in [-\pi/2, \pi/2]$". The branch is as specifying as the equation,
     which alone leaves the result open. The tags, the cases and the examples
     are written the same way. -->

For any [tensor index](./../common/definitions.md#tensor_index) $i$:

$$
C[i] = \<the complete formula\>
$$

where
- \<element\>: \<definition\>
- ...
- \<element\>: \<definition\>

<!-- State, for every case that applies to the operator, what the operator
     does. These are the cases on which two implementations most often
     disagree, so each of them is either decided here or explicitly left to
     the implementer; they are at least:

       - an input with a zero-sized dimension, and an input with no dimension
         at all (a rank-0 tensor);
       - inputs whose shapes differ and are combined, where ONNX allows it
         (broadcasting);
       - the types whose behaviour differs inside this section, in particular
         the signed and the unsigned integer types.
-->

**\<Case.\>** \<What the operator does in that case, or that it is left to the
implementer.\>

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>

The effect of the operator is illustrated on the following examples.

### Example 1

<!-- One subsection per example, numbered from 1; an operator with a single
     example uses "### Example". The worked results are written between $$ and
     $$, with a blank line before and after. Use $\approx$ instead of $=$ when
     the displayed result is not exact, and the exact result for real numbers. -->

$$
A = \<the input tensors of the example\>
$$

$$
C = \<the result\>
$$

<!-- When a notebook generates the examples, place it in the "Examples"
     subfolder of the operator folder. -->

## Error conditions

<!-- Give every condition of the list given in the guidelines, and every
     further condition the operator introduces, a disposition below: nominal
     and specified in the "Function" section, or ruled out by a
     precondition, or an error condition described in this section. A
     condition that appears in no row cannot be told from an oversight. -->

| Condition | Disposition |
| -------- | ------- |
| \<condition\> | nominal: specified by \<tag of the statement of the "Function" section that specifies it\> |
| \<condition\> | ruled out by the precondition \<traceability tag of the constraint\> |
| \<condition\> | error: what can happen at runtime, i.e. \<description\> |

<!-- For each error condition kept above, give the most detailed description
     of the cases in which a non-complying result can be produced, point out
     where in the specification the error may occur, and, if applicable,
     recommend how an implementation can prevent the non-complying result.
     When no error condition remains, write instead: "No error condition." -->

## Attributes

<!-- Delete this section when the operator has no attribute, and write
     instead: "Operator **Op** has no attribute." An attribute is described
     once, in the section for real numbers, and referred to from the sections
     for the other types. In a constraint tag, \<name\> is the name of the
     input, output or attribute the constraint applies to. -->

### `\<name\>`: \<type\>

\<What the attribute is, and what the operator does with it.\>

#### Constraints

<!-- One item per constraint, in the order of the tags. A constraint that
     concerns several inputs, outputs or attributes is described once, at
     the first of them, and cross-referenced from the others. -->

<a id="E_\<op\>_\<type\>_CONSTR_\<name\>_0010"></a>
- `[E_\<op\>_\<type\>_CONSTR_\<name\>_0010]` \<title of the constraint\>
  - Statement: \<the constraint, stated in the present indicative\>
  - Rationale: \<why the constraint; it may be omitted when the title and the statement are explicit\>

## Inputs

<!-- Repeat the block below for each input. -->

### $\text{\<name\>}$: \<type\>

\<What the input is.\>

#### Constraints

<!-- Same as for the attributes. A constraint shared with another input,
     output or attribute is written here as a cross-reference, as in:

       - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">E_<op>_<type>_CONSTR_<name>_0010</span></b>](#E_<op>_<type>_CONSTR_<name>_0010>) on $\text{<name>}$.
-->

## Outputs

<!-- Repeat the block below for each output. -->

### $\text{\<name\>}$: \<type\>

\<What the output is.\>

#### Constraints

Same as for the inputs.
