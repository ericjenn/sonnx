---
name: operator-spec
description: Write, review, amend or verify a SONNX operator specification (ops/<op>.md): the iterative loop from a first draft, through the user's review amendments, to verification of the specification against ONNX Runtime, driven automatically until a fresh blind implementation passes. Use when the user asks to specify an operator, to split or rework a specification by type family, to check a specification against the guidelines in informal_corrected.md, to fix wording such as an operator's indexed notation, to verify a specification by implementing it and comparing with onnxruntime, or to run that loop until it converges.
---

# Specifying an operator

A specification in `ops/<op>.md` is not written once. It is drafted, reviewed, amended,
sometimes verified, and every amendment leaves a trace. This skill is that loop.

The loop:

0. **Read the guidelines** — `informal_corrected.md` is the normative document; `informal_spec_template.md`
   is its skeleton; an already accepted spec (`ops/add.md`) is the shape reference.
1. **Draft** — one section per type family, each with its own tags and its own case analysis.
2. **Review and amend** — the user reads and pushes back. Act on the instruction, then report the
   interpretation and its consequences.
3. **Verify** (optional, only on request) — have another agent implement the operator from the
   specification alone, compare with ONNX Runtime, and classify every discrepancy.
4. **Record** — the amendment and the reason it was made.

Details: `references/document-rules.md` (what the guidelines require),
`references/checklist.md` (what to review before showing a draft, and the mistakes already made),
`references/verification.md` (stage 3 in full). Mechanical checks: `assets/lint_spec.py`.

## 0. Read before writing

- `informal_corrected.md` — "Specification guidelines" (from l.82) is normative. The sections that
  decide the shape of a specification are *Basic operations* (l.110), *Tags* (l.151), *Types* (l.194)
  and *Structure and contents* (l.220).
- `informal_spec_template.md` — the required skeleton.
- `ops/add.md` — the reference implementation of that skeleton, and the only spec that has been
  through the whole loop. Follow its shape unless the guidelines say otherwise.
- If the operator exists in ONNX, read the ONNX definition for the opset the spec is based on
  (opset 14 for the specs in `ops/`). The `Based on ONNX documentation` line must point at it.

**The guidelines outrank every precedent.** Where `ops/add.md` and `informal_corrected.md` disagree,
the guidelines win and the older spec is the one to fix. Two known divergences are recorded in
`references/checklist.md`.

## 1. Draft

Work family by family, in the order the guidelines fix: `real` first, then the machine types
(l.239). Each family gets its own section:

```
# **Op** (float, float)

where float is in {`float16`, `float`, `double`}.

The three types share one semantics: <one line saying why>.

## Signature
## Restrictions        (usually just the general-restrictions link)
## Function            (the specifying part, inside a red span, tagged)
### Example 1
## Error conditions    (a disposition table referring to tags, never restating them)
## Attributes
## Inputs / ## Outputs (with the tagged constraints)
```

Rules that decide most of the drafting work:

- **One section per set of types with the same semantics.** "types whose semantics differ shall be
  given separate sections" (l.255). `real`, `float`, `int` and `uint` are four different families
  (l.205). A section covering several types carries a `where` line and one line of justification.
  A section covering one type carries neither.
- **The type of the result is the type of the arguments**, so the specification is written in terms
  of the argument type. Never "the type of $C$" — say "the type of $A$ and $B$". This matters wherever
  the result can leave the range of the operands: signed addition can change sign, unsigned
  addition can give a value smaller than both operands.
- **Modality** (l.108): the operator's semantics is stated in the present indicative; `shall` is
  for requirements on the writer or on the document. `must`, `should`, "checkable", a `---` rule
  line and a code fence do not belong in a specification.
- **Indexed notation** (l.110): a bare `+`, `-`, `*`, `/` is the operation on ℝ. An operation on a
  machine type is indexed: `+_{(\text{f16})}`, `+_{(\text{i8})}`, `+_{(\text{u8})}`. A float section
  that computes in IEEE 754 writes `+_{(\text{f})}`; the bare `+` inside it then means the exact sum
  in ℝ, which is what the rounding cases need.
- **A case analysis is specifying information.** Give it explicitly rather than delegating to a
  standard's name, and define every symbol you use (say that $n$ is the number of bits of the type,
  that the sum is the exact sum in ℤ before reduction, etc.).
- **Examples in `$$`.** The documents are read in Obsidian, where ```` ```math ```` is not rendered.
  Use `$$ ... $$` for every formula, display or inline.
- **The red span** carries the specifying part: the opening line names the tag, the closing line is
  `[END]` — except in a list or table row, where the scope is implicit (l.151).
- **Nothing that does not specify.** An announcement such as "The mathematical definition of the
  operator is given hereafter." is not part of the specification and is removed.
- **No "Checking the specification".** The guidelines no longer contain that section and a
  specification does not carry one; verification is a practice, not a requirement (see
  `references/verification.md`).

## 2. Review and amend

The user reads the draft and pushes back, often several times on the same point. The working rule,
stated by the user: **act on the instruction, then report the interpretation and its consequences.
Do not ask a scoping question first.**

When the pushback is "these two cases should be clearly different" and you believe they are one
family, the guidelines settle it: separate families with names used in the tags get separate
sections. Re-read l.194–215 before arguing.

An amendment is not local. Changing the type family changes the Contents entries, the anchors, the
tags, the examples, the error-condition rows, the constraint lists and the Inputs/Outputs headings.
Walk the whole document for the old formulation afterwards — `grep -n` for the type name and for the
old phrase is the cheapest way to find the stragglers.

Then run `assets/lint_spec.py` on the result and fix what it reports (see below).

## 3. Verify — only on request

The user may ask for the specification to be checked by implementing the operator from it and
comparing with ONNX Runtime. That is a check of the specification, not part of it, and it is never
written into the specification or into the guidelines. The full procedure, the environment and the
report format are in `references/verification.md`.

### The automated loop

When the user asks for the loop to be run, or to be run automatically, run it yourself as the
driver — do not hand the loop to a script, since two of its three steps are agent work. One
iteration is:

1. **Blind implementation.** Spawn a *fresh* agent with the text of
   `assets/blind_task.md`, `<SPEC>` = `ops/<op>.md`, `<OUT>` =
   `verification/<op>/loop/iter-<NNN>-impl.py`, `<ENTRY>` = the lowercase operator name. Fresh
   means fresh: no context, no hint of ONNX Runtime, the tests or this skill. Keep its list of
   `DECISION:` points — each one that is semantic rather than an interface choice is a finding.
2. **Test.** Run

   ```
   ~/Venvs/onnxcheck/bin/python .claude/skills/operator-spec/assets/specloop.py \
       --spec ops/<op>.md --op <Op> --cases verification/<op>/cases.py \
       --impl verification/<op>/loop/iter-<NNN>-impl.py \
       --json verification/<op>/loop/iter-<NNN>-report.json
   ```

   `specloop.py` exits 0 when nothing mismatched; read the JSON, not the exit status alone. An
   operator with attributes and several outputs is specified like any other: the case module
   carries the node's attributes per case, names the document's outputs in `NODE_OUTPUTS` and their
   types in `OUTPUT_TYPES`, and every output is compared by position.
3. **Amend.** For each finding, decide *incorrect* (the specification says what the reference does
   not do) or *incomplete* (it leaves undecided what the reference decides), and make the minimal
   amendment to `ops/<op>.md`. Record it as in stage 4. Do **not** touch the implementation to make
   a test pass: the implementation is a reading of the specification, and a reading that fails is
   evidence about the specification.

**Convergence.** The loop is done when a freshly spawned blind agent passes `specloop.py` on the
current specification — that is, the run of step 2 that follows an unamended step 1. A run that
passes because step 1 was not re-executed after an amendment proves nothing.

**Guards.** Stop and report instead of looping when

- an iteration ends with the specification unchanged and a case still failing — the loop cannot
  converge;
- `cases.py` covers no case of a section — the document has a section the tests do not exercise, so
  a pass is vacuous for it. Add the cases first; the case set grows with the document, and it is
  part of the loop;
- every case of `cases.py` was refused by the runtime — the run compared nothing and still exits 0.
  The refusals name the defect (an output declared with the wrong type, an attribute the node cannot
  take), and a case set is not repaired by deleting the cases it will not run;
- a `DECISION:` point of the blind agent contradicts the specification rather than filling a gap it
  left — that is an ambiguity the amendment must remove, and after the amendment the next iteration
  must re-read it;
- the same amendment is made twice — the specification did not actually exclude the failing case;
- the iteration count reaches the cap the user set (default 5), or an amendment would change a tag
  that is already published.

**What the loop cannot do.** It cannot decide that the specification is *right* — only that a fresh
reader of it agrees with ONNX Runtime 1.30 on the cases covered. Say so when reporting, and give the
coverage: which types, over which domains, and which cases of the document were exercised.

The operator's cases live in `verification/<op>/cases.py` and are written by the driver, not by the
blind agent. `verification/add/` is the worked example; copy its shape. A new section in the
document means new cases before the next run.

## 4. Record

The specs in `ops/` carry amendments as callouts so the reader can see what changed and why:

```
> [!note] V1 · <title>
> <what changed and why>
```

`note`, `warning` and `question` are the three in use (`ops/sub.md` V1–V4). The note form is used by
the specs' revision record; it is not described in the guidelines, so do not introduce it into a
spec that does not already use it without saying so.

If a review turns up a rule that the guidelines themselves contradict, report the divergence rather
than silently following either side. Two are currently open (`references/checklist.md`).

## The tools

```
python3 .claude/skills/operator-spec/assets/lint_spec.py ops/<op>.md --op <OP>
```

decides only what a program can decide about the document: forbidden constructs, red-span balance,
tag well-formedness and 10-by-10 numbering, tags referred to versus declared versus anchored,
Contents against section anchors, the `where` line and its justification, the `Based on ONNX` and
`Revision` lines, and the end-of-line convention (the specs in `ops/` are CRLF). A clean run means
nothing more than that the mechanical rules hold — the semantic review is `references/checklist.md`.

```
~/Venvs/onnxcheck/bin/python .claude/skills/operator-spec/assets/specloop.py \
    --spec ops/<op>.md --op <Op> --cases verification/<op>/cases.py --impl <implementation>
```

is one iteration of the verification loop: it runs the cases through a one-node ONNX Runtime model
and through an implementation of the specification, and reports the mismatches as JSON. The
implementation comes from `assets/blind_task.md` given to a fresh agent. Both are described in
`references/verification.md`.

## Files

| path | role |
|---|---|
| `ops/<op>.md` | the specification |
| `informal_corrected.md` | the normative guidelines |
| `verification/<op>/cases.py` | the cases, written by the driver, never by the blind agent |
| `verification/<op>/loop/iter-<NNN>-*` | each iteration: the blind implementation and its report |
| `.claude/skills/operator-spec/assets/` | `lint_spec.py`, `specloop.py`, `blind_task.md` |
