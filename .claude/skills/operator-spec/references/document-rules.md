# What the guidelines require

Distilled from `informal_corrected.md` (the normative document) and `informal_spec_template.md`.
Line numbers are those of `informal_corrected.md` as of 2026-10-04. When this file and the guidelines
disagree, the guidelines win — fix this file.

## Modality (l.108)

The semantics of the operator, including the constraints its arguments, attributes and results
satisfy, is stated **in the present indicative**:

> Tensors $A$, $B$ and $C$ have the same shape.
> each element $C[i]$ is the result of dividing $A[i]$ by $B[i]$.

*Shall* marks a requirement on the writer of the specification or on the document itself; it is
never used to say what the operator does.

Not allowed in a specification: `must`, `should`, a `---` rule line, a code fence (other than the
one the guidelines themselves use to show markup), a "Checking the specification" section.

## Styling (l.102)

- Mathematical objects in *italic*; formulae in LaTeX; attributes in `code font`; operator names in
  **bold**; type names in `code font`.
- A formula is written between `$$ ... $$` with a blank line before and after, **not** in a fenced
  block tagged `math`: Obsidian, the editor the specifications are read in, does not render the
  fence (l.338).
- A worked result that is not exact uses $\approx$, not $=$; a real number is given exactly
  ("1/3", not "0.3333...").

## Basic operations and notation (l.110)

Usable without definition: $+$, $-$, $*$, $/$, $-x$, the trigonometric functions, $\exp$, $\sqrt{}$,
$\ln$, $|x|$, $\min$, $\max$, $\wedge$, $\vee$, $\lnot$, $\lt$, $\gt$, $\le$, $\ge$.

An operation on a machine type is given an index that names the type:

- $+_{(\text{f16})}$, $+_{(\text{f32})}$, $+_{(\text{f64})}$ — the additions for `float16`, `float`, `double`
- $+_{(\text{i<x>})}$ — XX-bit signed integers, `<x>` in {8,16,32,64}
- $+_{(\text{u<x>})}$ — XX-bit unsigned integers

**Unindexed, it is the operation on ℝ.** So a float section that computes in IEEE 754 must write
`+_{(\text{f})}`; then the bare `+` that remains inside that section means the exact sum in ℝ,
which is exactly what the rounding cases need. The same convention applies to $\Sigma$ and $\Pi$.

## Types and families (l.194)

- Type names are ONNX's, without `tensor()`: `double`, not `tensor(double)`.
- In a section covering a set of types, an input or output is designated by the family name followed
  by "tensor": `real tensor`, `floating-point tensor`, `integer tensor`.
- The families, to be used consistently in the prose, in the headings **and in the tags**:

  | family | types | tag component |
  |---|---|---|
  | real | the specification established for real numbers (no ONNX type) | `REAL` |
  | float | `float16`, `float`, `double` | `FLOAT` |
  | int | `int8`, `int16`, `int32`, `int64` | `INT` |
  | uint | `uint8`, `uint16`, `uint32`, `uint64` | `UINT` |

  `bool` and `string` exist but are not part of these families.
- The special numbers of the IEEE 754 types: $\pm 0$, $\pm\infty$, NaN.

## Tags (l.151)

Two kinds of tag. A **restriction tag** `[R<i>]` for a restriction with respect to ONNX, synthesised
in the "Restrictions" section. A **traceability tag** `[E_<op>_<type>_<zone>_<number>]` for a
constraint on inputs, outputs or attributes:

- `<op>` the operator, `<type>` the family in uppercase, `<zone>` the section (`FUNC`,
  `CONSTR_<IO>`) with `<IO>` the input or output the constraint bears on, `<number>` a 4-digit id,
  preferably incremented by 10 so that new tags can be inserted without renumbering.
- Closed by `[END]`, except inside a list or a table row, where the scope is implicit.
- **A tag is permanent**: never reused, renumbered or deleted, including when the statement it marks
  is withdrawn — a withdrawn tag is recorded as withdrawn.
- A reference to a tag is a hyperlink: `<a id="TAG"></a>` declares the location, and
  `[<b><span style="font-family: 'Courier New', monospace">TAG</span></b>](#TAG)` refers to it.
- The beginning and end tags are displayed in red for the reader only; the red span delimits the
  **specifying part**:

  ```html
  <span style="background: red; color: white; font-size:0.7em;">[TAG]</br></span>
  ...
  <span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
  ```

  Everything inside it specifies; everything outside it does not.

## Structure (l.220, and `informal_spec_template.md`)

The headings of the specification document:

1. **Contents** (l.228) — one entry per specification, each a hyperlink to its section's anchor. The
   real-number specification comes first; the others follow (l.239).
2. **Based on ONNX documentation** — the reference to the ONNX definition, with the opset.
3. A **revision line** — `Revision YYYY-MM-DD: ...`, so a reader can tell which issue they hold.
4. One `# **Op** (fam, fam)` section per set of types with the same semantics (l.257):

   - `where fam is in {`a`, `b`, ...}.` if the section covers several types, followed by **one line**
     justifying that they share one semantics (l.255). A section covering one type has neither.
   - `## Signature`
   - `## Restrictions` — always the general-restrictions link (l.291), and the sentence
     "No specific restrictions apply to the **Op** operator." when there are no others (l.295).
   - `## Function` — Part 1, a short description; Part 2, the detailed description: the complete
     formula first, its atomic elements defined afterwards by a "where..." list; the notation of
     l.110; the tags; and a statement for **every applicable case**. A section whose result is
     fixed by a property rather than by an expression — the inverses of the other operators are
     the usual case — states the axioms instead: the conditions the result satisfies, conditions
     that only one value satisfies, and the domain on which they are stated (l.340). The cases
     listed at l.319 are
     the ones where implementations most often disagree and are at least:
     - a zero-sized dimension and a rank-0 tensor;
     - shapes that differ and are combined (broadcasting);
     - types whose behaviour differs inside a section — *"in particular the signed and the unsigned
       integer types"*;
     - the special float values $+0$, $-0$, $\pm\infty$, NaN, as operands and as results.

     A case the specification leaves open shall say so, together with the set of results that
     conform (e.g. "every quiet NaN of the type of $A$ and $B$ is a conforming result").
   - `### Example 1`, `### Example 2`, ... — each its own subsection; a single example uses
     `### Example`. Cover special values and domain bounds (l.352).
   - `## Error conditions` (l.358) — every condition at least of the list at l.362, and every further
     condition the operator introduces, gets a **disposition**: nominal behavior (naming the tag of
     the Function statement that specifies it), ruled out by a precondition (naming the constraint
     tag), or an error condition described here. **A table is the recommended form**, and a
     disposition *refers* to the statement — it does not restate it, "since a restatement is a second
     formulation of the same rule and the two may drift apart" (l.376). The list is a minimum, not a
     menu (l.376).
   - `## Attributes`, `## Inputs`, `## Outputs` (l.392, 412, 424) — with `#### Constraints`, each
     constraint tagged.

## The two documents are a pair (l.40)

The informal and the formal specification of an operator shall not contradict each other, and any
change to one shall be reviewed against the other. The formal specification shall cover at least the
same domain, the same range and the same error conditions.

## Reuse (l.216)

Part of a specification may reference another part by a traceability tag. There is in any case a
local traceability tag: a reused statement is tagged where it is used, not only where it is defined.
