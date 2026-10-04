# Verifying a specification (optional)

**This is a practice, not a requirement.** It is never written into the specification and never
written into the guidelines. Do it when the user asks for it, and only then.

The idea: a specification is checked by someone who implements the operator **from the
specification alone** and compares the result with ONNX Runtime. That catches two different defects,
and the distinction is the whole value of the exercise:

- the specification is **incorrect** — it says something the reference implementation does not do;
- the specification is **incomplete** — it leaves a case undecided that the reference implementation
  decides, so two readers could implement the case differently.

Either way the fix is a **minimal, documented amendment** to the specification, not a change to the
implementation.

The five steps below are one iteration of the loop and its report; to run the loop automatically,
iterate steps 1–3 until a freshly spawned blind agent passes step 2 (step 4), under the guards of
`SKILL.md` §3.

## Environment

```bash
python3 -m venv ~/Venvs/onnxcheck      # if it does not exist yet
~/Venvs/onnxcheck/bin/pip install numpy onnx onnxruntime
~/Venvs/onnxcheck/bin/python --version && ~/Venvs/onnxcheck/bin/pip list | grep -Ei 'numpy|onnx'
```

As of 2026-10-03 that environment holds numpy 2.5.3, onnx 1.23.1, onnxruntime 1.30.0. The system
`python3` has no pip; a venv under `~/Venvs` is the way in.

## Step 1 — implement from the specification alone

Give another agent **only** the specification file (`ops/<op>.md`), nothing else: not the ONNX
documentation, not the harness, not the tests, not this skill. The prompt is
`assets/blind_task.md`; fill in the spec path, the output path and the entry-point name, and spawn a
fresh agent with no other context. It asks the agent to implement the operator and to state, at
every point where the document does not decide, what it had to decide itself.

Read the resulting list of decisions. It should contain only interface choices the document does not
and need not fix (how a dtype maps to a family, what the wrapper does with a 0-d result, and so on).
**A semantic choice in that list is a finding**: the document left undecided something the
implementation had to decide, which is exactly defect (2) above.

Write it to `verification/<op>/loop/iter-<NNN>-impl.py` and keep its `DECISION:` comments — they are
the evidence.

## Step 2 — compare with ONNX Runtime

`assets/specloop.py` does this, generically:

```
~/Venvs/onnxcheck/bin/python .claude/skills/operator-spec/assets/specloop.py \
    --spec ops/<op>.md --op <Op> --cases verification/<op>/cases.py \
    --impl verification/<op>/loop/iter-<NNN>-impl.py \
    --json verification/<op>/loop/iter-<NNN>-report.json
```

It builds a one-node model of the operator (opset 14, one input per operand, all of the same
element type, an unknown shape so that one session serves every case of a type), runs it under
`onnxruntime` with the CPU execution provider, and compares with the implementation bit pattern by
bit pattern for floats (NaN equal to NaN; a differing NaN payload reported separately) and by value
for integers. A case may carry the node's **attributes** (the third element of its tuple), which go
into the node and into the implementation call alike; an operator whose document gives it **several
outputs** names them in `NODE_OUTPUTS` and their types in `OUTPUT_TYPES`, and every named output is
compared by position. It classifies each case as **mismatch**, **open** (the runtime refused the
model, or the operands are outside the domain the specification's constraints define — reported, not
a mismatch) or **NaN payload**. The exit status is 0 when nothing mismatched *and* no sweep failed;
read the JSON report, which carries the first differing elements and their bit patterns, and the
console, which names each failing sweep entry — a sweep that fails without saying why is worse than
no sweep at all. The exit status alone is not the verdict: a case set whose *every* case was refused
by the runtime compares nothing and also exits 0, so a run is only evidence when the report shows
cases that passed.

The cases are in `verification/<op>/cases.py`, written by the driver (never by the blind agent), in
the two-hook form the driver expects:

- `cases()` — a generator of `(label, [array, ...])`, or of `(label, [array, ...], attrs)` for an
  operator with attributes, one case each; the per-case checks. Cover every case of every section's
  formula, the special float values as operands *and* as results, the float rounding corner cases,
  and the shape cases. Get the ulp from the type's `nmant`, not by hand: the ulp at the largest
  finite value of `float16` is 32, and the tie above it is 65520, which rounds to infinity;
- `sweeps()` — the bulk checks, each a function of the implementation, batched into a single ONNX
  Runtime run because a per-case run would be far too slow. A sweep builds its own node, so it passes
  the attributes to `make_node` itself and to the implementation itself. This is where the domain
  sweeps go: the **whole domain** of `int8` and `uint8` (all 65 536 pairs, in one batched run), the
  boundary values plus a random sample for the six wider types, and the document's own worked
  examples. Use `rng.integers` with an inclusive bound for signed types, and assemble a 64-bit
  unsigned draw from two 32-bit draws (numpy 2.5.3 raises on a 64-bit range); build the operand array
  directly in the dtype under test rather than through an `int64` intermediate.

Two more module constants, both optional: `NODE_OUTPUTS` (a tuple, default `("out",)`) names the
outputs the one-node model asks for, an operator with several outputs naming them as the document
does; `OUTPUT_TYPES` (a dict) gives the type of an output that is not the operands' type, e.g.
`{"Indices": np.int64}` — a wrong type there makes the runtime refuse every case, and a case set
whose every case is open compares nothing.

A good sweep also transcribes the document's case analysis into exact Python arithmetic and compares
*that* with ONNX Runtime, independently of the implementation: it tests the specification as
literally read, not a program's reading of it. `verification/add/cases.py` has both, and its
`cases()` covers 205 cases across the eight machine types.

The real-number section has no ONNX type, so it is the one part ONNX Runtime cannot be asked about.
Check it against exact rational arithmetic (`fractions.Fraction`) as `sweep_real_examples` does.

One convention the harness fixes and the specification does not: a real tensor is exercised as an
object-dtype array of Python numbers (`Fraction`), since no machine type represents it. Do **not**
tell the blind agent this — the specification names no such representation, and a blind reader that
fails to find a way to be exact about real values has found a real gap (l.305 wants a reader to be
able to implement the operator from the specification alone). Record what it decided in its
`DECISION:` list.

## Step 3 — read the discrepancies

- **A mismatch** is a defect in the specification. Decide which of the two kinds it is, by reading
  the sentence that should have decided the case.
- **"ORT rejected the model"** is not a mismatch: ONNX Runtime refusing the *model* means the case is
  outside the operator's domain (incompatible shapes, mixed dtypes), i.e. a precondition, which the
  specification covers through a constraint tag rather than through an error condition. Record it as
  such; do not turn it into a runtime case.
- **An open case the implementation had to decide** is defect (2): the specification should either
  decide it, or state that it is undecided and give the set of conforming results (l.319).
- **A `DECISION:` point that is not a defect.** Most decisions a blind reader records are interface
  choices — how the language discovers the type of the operands, what object represents a tensor,
  what happens for a type the specification gives no section to. Those need no amendment; list them
  in the report so the next reader can see they were considered. The ones that matter are the
  decisions about **values**: a reader that had to choose a value the document does not state has
  found defect (2), whether or not the choice happens to be right.

Then amend the specification, minimally, and record the amendment. Re-run both steps after the
amendment, and note in the report which findings are now closed.

## Step 4 — iterate to convergence

Steps 1–3 are one iteration. The loop is done when a **freshly spawned** blind agent, reading the
current specification and nothing else, passes step 2 with no mismatch. A pass that follows an
amendment is not convergence: it may only mean that the previous implementation was written to the
previous text. Re-run step 1 after every amendment.

Two things can never be automated away, and they are why the driver is an agent and not a script:

- the amendment itself needs a judgement — *incorrect* or *incomplete* — and a sentence in the
  document;
- the case set must grow with the document. A new section with no cases in `cases.py` passes
  vacuously.

The guards on the loop are in `SKILL.md` (§3, "The automated loop"): stop on an unchanged
specification with a failing case, on a section without cases, on a repeated amendment, and at the
iteration cap. Each iteration's `impl.py` and `report.json` stay under
`verification/<op>/loop/iter-<NNN>-*` as the record.

## Step 5 — report

State, for each finding: the case, what the specification said, what the reference implementation
does, which kind of defect it is, and the amendment made. State the environment and the exact
coverage (which types over which domains) so the reader can judge how much of the operator was
actually exercised. Say plainly that the loop shows a fresh reader of the specification agrees with
ONNX Runtime 1.30 on the cases covered — not that the specification is correct.
`verification/add/` is the worked example of all five steps.
