# Review checklist

Run this before showing a draft, and again after every amendment. The mechanical part is
`assets/lint_spec.py`; everything here needs a reader.

## Per family

- [ ] The section heading names the **family**, and the `where` line lists the types of that family
      in backticks. A family of one type (`real`) has no `where` line.
- [ ] A section covering several types has its **one-line justification** that they share one
      semantics, and it is true.
- [ ] Semantics that differ are in **separate sections**, with separate anchors, Contents entries,
      tags, examples, error rows and constraints. `int` and `uint` differ; so does any pair where
      one wraps at a different point or one has a special value the other lacks.
- [ ] Every mention of a type is in terms of the **argument** type. "the type of $C$" is wrong:
      $C$ has the type of the arguments (the constraint tag says so) and the whole point of the
      machine-type sections is what happens when the result cannot be represented in that type.
- [ ] The prose names the family as the guidelines name it — `real`, `float`, `int`, `uint` — and the
      Inputs/Outputs headings follow the "family name + tensor" rule.

## Per formula

- [ ] Every operation on a machine type is **indexed**: `+_{(\text{f})}`, `-_{(\text{i8})}`, …
      A bare `+`, `-`, `*`, `/` is the operation on ℝ (l.110), and inside a float section that is
      exactly what the rounding cases need — the two must not be confused.
- [ ] Every symbol is defined: $n$ the number of bits, $\tilde{A}$ the (possibly broadcast) operand,
      `round` the rounding mode, the exponent range, and so on.
- [ ] A case analysis has no gap: for every pair of operands exactly one case applies (check the
      bounds of each case, and that the intervals tile the reachable set).
- [ ] An **axiomatically specified** section is a formula's equal. The Function section (l.340)
      permits it, the inverses of the other operators being the usual case: it states the
      conditions the result satisfies, conditions that only one value satisfies, and the domain on
      which they are stated — for **Asin**, "for every $x$ in $[-1, 1]$, the result is the unique
      value $y$ such that $\sin(y) = x$ and $y \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$".
      The branch is as specifying as the equation, which alone leaves the result open. An equation
      naming an operator is not "a case delegated to a name": the defect is an axiom that still
      leaves the reader a choice, or one that states a way of computing the result instead of the
      value it has.
- [ ] A case may be decided by the formula, by a definition in its `where` list, or by the
      paragraph that follows it — the three are one statement — and **prose that repeats a case
      the formula already decides is not a defect**. `ops/sin.md` states the NaN and infinity
      cases in its formula and again under *Values of the operand*. The remark that "a
      restatement is a second formulation of the same rule and the two may drift apart" is made
      of the **error-condition table**, which is to carry a tag and not a copy of the statement
      it disposes of; it is not a rule that the document may say a thing only once. Report a
      repetition only when the two statements disagree.
- [ ] Everything the formula decides is *inside* the red span; everything outside it is comment.

## Per case (l.319)

- [ ] Zero-sized dimension, and a rank-0 tensor.
- [ ] Shapes that differ and are combined (broadcasting).
- [ ] Types whose behaviour differs inside a section — signed vs unsigned above all.
- [ ] The special float values $\pm 0$, $\pm\infty$, NaN, as **operands and as results**.
- [ ] Every case that is left open says so and gives the set of conforming results.

## Per example

- [ ] Displayed between `$$ ... $$`, not in a ```` ```math ```` fence.
- [ ] $\approx$ where the result is not exact, $=$ where it is; real numbers given exactly.
- [ ] At least one example per section, and the examples of a section exercise that section's own
      cases (the signed wrap-around, the unsigned one, the float corner cases, the broadcast).

## Per error condition

- [ ] Every condition of l.362, and every further condition the operator introduces, has a row.
- [ ] Each row is a **disposition**, not a restatement: it names the tag of the Function statement
      that specifies it, or the constraint tag that rules it out, or says it is an error here.
- [ ] Where the disposition is "nominal", the Function section really does specify it.
- [ ] If there is no error condition, the section says so, and the table still shows that each
      condition was considered.

## Per document

- [ ] Contents, the `Based on ONNX ...` line and the `Revision` line are present and consistent with
      the sections (the linter checks the anchors, not the wording).
- [ ] Tags: well-formed, 10-by-10 numbering, every tag referred to is declared and carries an anchor,
      no tag renumbered or reused.
- [ ] No `must`, no `should`, no `---`, no code fence, no "Checking the specification".
- [ ] No announcement that does not specify ("The mathematical definition of the operator is given
      hereafter.", "Definition of operator **Op** signature:", …) — such a sentence states nothing
      about the operator and is removed.
- [ ] The end-of-line convention of the file is unchanged (CRLF in `ops/`).
- [ ] After an amendment: `grep -n` for the old formulation and for the type name that moved,
      and run the linter.

## Mistakes already made — check for these first

1. **One section for `int` and `uint`.** The two families are distinct (l.205), their semantics
   differ (l.255), and the tags carry the family name, so a merged section cannot tag correctly.
   The user rejected this twice; the second time the argument "they differ only by the value chosen
   in the residue class modulo $2^n$" was explicitly rejected as understating the difference.
   Signed addition can change the sign ($100+100=-56$ for `int8`); unsigned addition can give a
   value smaller than both operands ($200+100=44$ for `uint8`).
2. **A bare `+` in a float section that computes in IEEE 754.** It means the addition in ℝ, so the
   section said something weaker than it intended. Index it `+_{(\text{f})}` and say in a "where"
   bullet what the bare `+` of the case analysis then means.
3. **A specification written in terms of the type of $C$.** The type of $C$ is the type of the
   arguments; writing the case analysis on $C$ hides the whole question.
4. **A non-specifying announcement inside the Function section.** It is not part of the
   specification; it was removed from `ops/add.md` and must not be reintroduced.
5. **Checkability clauses.** The guidelines no longer contain a "Checking the specification"
   section; a specification does not state that it is checkable, nor how.
6. **A worked result in a ```` ```math ```` fence.** Not rendered by Obsidian.

## Known divergences between the documents

Report these rather than silently choosing a side.

1. **The guidelines' own example contradicts the guidelines.** The `Div` Contents example at
   `informal_corrected.md` l.249 (reproduced in `informal_spec_template.md` l.27) merges the eight
   `int` and `uint` types into one section, while l.255 requires separate sections for types whose
   semantics differ and l.205 defines `int` and `uint` as two families. `ops/add.md` follows l.255.
2. **"Definition of operator **Op** signature:"** (`informal_corrected.md` l.266,
   `informal_spec_template.md` l.50) is an announcement, not specifying information, and it is
   stated with `shall`-free imperative mood in a place where the guidelines require the present
   indicative anyway. It was removed from `ops/add.md`.

## The specs in `ops/` as of 2026-10-03

| spec | state |
|---|---|
| `ops/add.md` | through the whole loop; four family sections, clean under the linter |
| `ops/sub.md` | predates the loop: merged `int`/`uint`, bare `-` in float, the announcement of mistake 4, V1–V4 callouts |
| `ops/div.md` | predates the loop: merged `int`/`uint`, bare `/` in float, `where` lines without backticks and without justification, no `Revision` line |

Bringing `sub` and `div` up to the current shape is the same loop as for a new operator, and each
amendment goes through the user first.
