# The multi-agent specification system

A LangGraph implementation of the SONNX operator-specification skill
(`.claude/skills/operator-spec/`). Five agents, each with its own system prompt and its own
view of the work, scheduled by a state machine that decides when the loop may stop. The
artifacts are files — the document, the implementation, the case module, the harness's JSON
report — and the mechanical checks are the skill's own programs, run as subprocesses so the
system cannot drift from the skill it implements.

```
~/Venvs/sonnx/bin/python agents/run.py --op Neg
```

## The agents

| agent | reads | writes | answers |
|---|---|---|---|
| **writer** — the specification | the guidelines, the skeleton, an accepted spec for shape, the ONNX definition, the restriction files, the findings against the previous draft | `ops/<op>.md` (CRLF, the guidelines' form) | what is the operator |
| **compliance verifier** | the document, the guidelines, the review checklist, the linter's output, the types the operator admits and the per-family verdict computed from them, the restriction files | a JSON verdict | does the document say what the guidelines require |
| **blind implementer** | **the document alone** | `iter-NNN-impl.py` + its `DECISION:` points | what does the document actually say |
| **test agent** | the document, the implementation, the harness contract, an accepted case module, the restriction files | `cases.py` + the harness report | does the implementation comply with the document, and with ONNX Runtime |
| **adjudicator** | the failing report, the document, the implementation and its decisions, the case module | a JSON verdict: `spec`, `code`, `test` or `stop` | where is the fault |

The first two are the skill's stages 0–2 (draft, review, amend); the last three are stage 3
(verify) with its classification step; the run record is stage 4.

The restriction files are discovered from the operator's name — `ops/global_restrictions.md`
and `ops/<entry>_restrictions.md`, with `ops/<Op>_restrictions.md` accepted as the ONNX-case
spelling — and there is no flag for them (see *The restriction files* below). The
implementer is the one agent that never sees them: the operator's rows reach it only inside
the document it is given.

## The graph

```
                    ┌──────────────┐   blocking finding    ┌──────────────┐
   ──▶ write_spec ──▶ check_spec ──┼──────────────────────▶│  write_spec  │ (amendment)
                    └──────┬───────┘                       └──────────────┘
                           │ complies
                           ▼
                      implement ──▶ test ──▶ passes ──▶ finish (converged)
                                      │
                                      └── fails ──▶ diagnose ──┬── spec ──▶ write_spec
                                                              ├── code ──▶ implement
                                                              ├── test ──▶ test
                                                              └── stop ──▶ finish
```

Every arrow is one model call. `check_spec` runs `assets/lint_spec.py` **and** the
verifier; a document that reaches the implementer has passed both.

## What the loop guarantees, and what it cannot

- **Blindness is enforced by construction.** The implementer's node builds the message from
  the document text and the operator's name and nothing else; the ONNX definition, the
  harness, the case module, **the restriction files** and the other agents' findings are not
  in its context, and a test in `tests/test_graph_offline.py` asserts that they are not.
- **The restrictions a document carries are the files', not the model's.** Every family
  section's `## Restrictions` block must carry the rows of `ops/<op>_restrictions.md` —
  verbatim, in the file's order, in every section — after the mandatory link sentence, and
  the writer's node refuses a document that drops a row, invents one, reorders them or
  rewords them, exactly as it refuses an answer that announces a type family ONNX does not
  admit: the writer is asked again, with the row named. The rows are the profile's own
  decisions about the operator — what it is defined to do — so a document that composes its
  own rows is a document that specifies a different operator. With no file, the block
  carries the sentence `No specific restrictions apply to the **Op** operator.` and no
  table, and a table with no file behind it is refused as an invented row. The global file
  is the other half: it is *linked* and never copied, so the document states the
  consequence of a global restriction where it states the semantics, and the verifier is
  told both halves — carry the file's rows, contradict nothing the global file says. The
  request is rebuilt for every attempt, because the guard reads the restriction files for
  every answer: a file that appears while the writer is answering must reach the writer's
  *next* request, or the node tells the writer the operator introduces none while its guard
  requires the rows (measured on 2026-10-05: `ops/maxpool_restrictions.md` appeared between
  the **MaxPool** writer node's entry and its first answer, and the run stopped `stalled` on
  rows the request itself had said did not exist; the offline suite's scenario 36 is the
  file appearing after the first answer). The
  implementer never sees either file (measured by the offline suite's scenario 34, which
  asserts the writer, the verifier and the tester are given the global file and the
  implementer is not).
- **A restriction is a domain boundary, not a mismatch — but `open` is the runtime's word.**
  The restrictions bound what the operator is defined to do, so a case they rule out tests
  nothing and is never a failure of an implementation. The trap is that the harness records
  a case as *open* **only when ONNX Runtime refuses to build or run the model**: the runtime
  enforces no restriction of its own, and a case that sets `storage_order=1`, `ceil_mode=1`
  or `auto_pad="VALID"` is accepted and *compared*, so an implementation that follows the
  document outside its domain comes back as a mismatch. The tester is therefore told to put
  a case the restrictions exclude in a sweep — or leave it out — and never in `cases()`,
  while the **boundary** case (`pads == kernel_shape`, `storage_order=0`,
  `auto_pad="NOTSET"`, `ceil_mode=0`, `dilations=1`) is inside the domain, must be compared
  and stays in `COVERAGE`. The adjudicator is told the same in one sentence: an open case is
  outside the domain, not evidence of a defect.
- **A pass after an amendment is not convergence.** The route out of `test` records the
  contents of the three artifacts with the failing cases; a failure that repeats with the
  same document, the same implementation and the same case set stops the loop (`unchanged`)
  instead of spinning, and the tester's own revisions have a budget of their own
  (`cases-stuck`).
- **The case set grows with the document.** Each case module records the md5 of the document
  it was written for; an amended document rebuilds it.
- **The verifier is not asked to remember ONNX, and the profile is narrower than ONNX.**
  It is given the types the operator admits, from the operator's own schema, *and the
  per-family verdict computed from that schema* — `float` applies for **Neg**, `uint` is
  **absent** — because it is asked whether a type family is missing or wrongly merged and it
  cannot answer that by guessing. Both halves have been seen to matter: a reviewer that
  invents a family sends the writer to document an operator ONNX does not have, and a
  document that then has to be walked back costs two amendment rounds — one to add the
  section, one to remove it. Naming the types was not enough; the run of 2026-10-04 had the
  list *and* the rule in front of it and still demanded a `uint` section for **Neg**, citing
  a checklist item learned from **Add**, which does admit `uint`. So the verdict is computed
  where it cannot be got wrong. It is also told that SONNX specifies only the types the
  guidelines list, so a type ONNX admits and SONNX does not consider — `bfloat16` for
  **Neg** — is not a missing family.
- **An axiom is a specification, and a formula is not required.** The guidelines allow a Function
  section to state the result by the properties that determine it — the inverses of the other
  operators are the usual case — and there the equation *and the branch* are the whole statement:
  for **Asin**, the unique $y$ with $\sin(y) = x$ and $y \in [-\pi/2, \pi/2]$. The verifier's
  prompt carries that with the example, because its rule against "a case delegated to a name"
  reads an axiom that names an operator as the delegation it is not. What stays a finding is an
  axiom that leaves the reader a choice, or one that says how to compute the result instead of
  what it is. (A prompt rule, not a measured incident: every operator specified here so far has a
  formula.) The review checklist says the same, so the two reviewers cannot disagree on it.
- **A qualification given in the prose is a qualification.** A case is decided by the
  formula, by the definitions in its `where` list, or by the prose that follows it — the
  three are one statement. `ops/add.md` and `ops/sin.md` are accepted documents written in
  all three shapes: a case analysis that names a special value in the formula and says in
  the sentence below which NaN, sign and rounding it is; a `\text{nearest}` defined as
  rounding to the nearest value of the type with the roundTiesToEven attribute, which is a
  total function and so decides the tie, the subnormal and the zero the reader might
  otherwise choose; and a paragraph after the formula that names a region and gives its
  value ("for $X[i] = 1$, the exact value is $0$ and the result is $+0$"). A reviewer that
  asks for any of them to be moved *inside* the formula is asking for more than the
  guidelines require — it is asking for a restatement of a rule the document already
  states, which the guidelines themselves avoid because "the two may drift apart". The
  verifier's prompt says so, with the accepted documents named. Report a case as undecided
  only when neither the formula, nor its definitions, nor the prose decides it. The rule
  cuts the other way too: prose that *repeats* a case the formula already decides is not a
  finding either — the guidelines' remark that "a restatement is a second formulation of the
  same rule and the two may drift apart" is made of the **error-condition table**, which is
  to carry a tag rather than a copy of the statement it disposes of, and not of the document
  as a whole; `ops/sin.md` states the NaN and infinity cases in its formula and again under
  *Values of the operand*. Report a repetition only when the two statements disagree. This is the
  rule the runs of 2026-10-04 were stopped twice by: **Acos** and **Atan** each lost three
  amendment rounds to a reviewer that demanded a third branch for the subnormal operands,
  a case inside the formula for the value at the bound of the domain, and the tie rule
  written out beside the standard's name — every one of them already decided by the
  document, and every one of them the shape of the accepted `ops/sin.md`.
- **A verdict is not a deliberation.** A model that thinks before it answers spends its
  output budget on the thinking, and a whole-document compliance review spends a lot of it:
  measured here, one 61 000-character thinking block and **no verdict at all** at 16 000
  tokens, and still no verdict at 64 000 after twenty minutes. Raising the budget only
  moves the race. The agents that answer with a bounded verdict — the verifier and the
  adjudicator — answer with the model's deliberation **off** (`--verdict-thinking off`,
  the default), which turns the same review into a 7-second call with a complete JSON
  verdict. The writer, the implementer and the tester keep it on (`--thinking auto`):
  what they return is a document, and the deliberation is where it comes from — until the
  answer comes back with no text at all, which is the deliberation having eaten the whole
  budget, and then the last attempt is made without it (see *An empty answer is asked again
  with the deliberation off*).
- **A type the runtime does not implement is not a failure of the document or of the code.**
  ONNX admits types that the ONNX Runtime build in front of us has no kernel for: measured on
  2026-10-04, **Asin**, **Acos** and **Atan** are implemented for `float16` and `float` and
  **not for `double`**, at opset 7 and at opset 23 alike. A document that specifies `double`
  is right to — ONNX admits it — and a case whose type the runtime refuses is recorded by the
  harness as *open*, which is not a mismatch. What the case set must not do is let the
  refusal escape from a *sweep*, which builds its own model: the sweep catches it per type
  and says in its summary that the type is not reference-checked, keeping the type in
  `COVERAGE`. Both the test agent and the adjudicator are told this, with the measurement and
  the opsets, because the run of 2026-10-04 stopped three times over it — once for each of
  **Asin**, **Acos** and **Atan** — while the document and the implementation were both
  sound.
- **A case set that compared nothing is not a pass.** An open case is not a mismatch, and the
  harness exits 0 on a case set of nothing but open cases: a runtime that refused every case of
  a type would have converged a run that compared nothing at all. The driver reads the report
  instead of the exit status — a case set with no case, or with not one case passed, is
  `_broken`, and it is sent back to the tester under the heading **The case set compared
  nothing** with the runtime's own refusals quoted, rather than accepted. Seen on 2026-10-04
  while preparing **MaxPool**: a case module that declared the type of the second output
  wrongly (`OUTPUT_TYPES = {"Indices": np.float32}` where the document says `int64`) had all
  five of its cases refused — `Type Error: Type (tensor(float)) of output arg (Indices) of node
  (node0) does not match expected type (tensor(int64))` — and the harness reported `5 run, 0
  passed, 0 mismatched, 5 open`: a pass with no comparison in it.
- **An operator with attributes and several outputs is specified like any other.** The
  contract the harness, the tester and the implementer share was extended for MaxPool, which
  needs it: a case carries the node's attributes as a third element (`(label, [x], {"kernel_shape":
  [2, 2]})`, the same attributes going into the node and into the implementation call), the
  implementation takes them as keyword parameters with the document's defaults, a case module
  names the document's outputs in `NODE_OUTPUTS` and their types in `OUTPUT_TYPES`, and every
  named output is compared by position — an implementation that returns fewer outputs than the
  node asks for is a mismatch that says which one is missing. The default is the old contract
  exactly: one output named `out`, no attributes, the report byte-identical to the reports the
  earlier operators produced.
- **A harness failure is not a finding against the document.** If the verifier returns no
  readable verdict, the run stops with `verifier-failed`; the writer is never sent to amend
  a document nobody reviewed, and a reviewer that failed to answer is not counted as a
  defect of the specification.
- **A cut-off answer is not a document.** A model answer can stop in the middle of the
  document. The writer node reads the red spans the skill's linter reads — opened against
  `[END]` — and asks again when they do not match; an incomplete answer never replaces a
  complete document on disk. If the answer is still incomplete, the compliance check sees
  the same document twice, and the loop stops with `stalled` instead of spending a round on
  an amendment that cannot be made.
- **An amendment must be one the writer can make.** A blocking finding is one that leaves a
  reader able to implement the wrong operator, and its `fix` is the change the writer is
  asked for. A finding whose own `why` closes the doubt it raises, or whose `fix` asks for
  no change, is therefore not blocking — but a verifier that reports it as blocking anyway
  leaves the loop an amendment that cannot be made: the writer returns the document it was
  given, the compliance check sees the same document twice, and the run stops `stalled`, the
  one outcome that blames the document for the reviewer's bookkeeping. The reviewer is told
  what the contradiction is and asked again, and a finding still asking for no change after
  the retries is not counted — it stays in the run's history as evidence of the review that
  refused the document for it — so the round is spent on the document and not on a severity
  tag. Measured on 2026-10-05 in the **MaxPool** run: two reviews in a row refused the
  round-3 document for findings whose own `fix` read "No change required." and "No change:
  ...", the writer's round-1 answer was the document byte for byte (`md5 b635a8a0…`,
  unchanged), and the run stopped `stalled` at round 1; the offline suite's scenario 37 is
  that review, and its run converges with no amendment at all.
- **Nor is an answer that keeps a Contents it does not fill.** The red span count is not
  enough on its own: an answer cut off *between* the spans — inside a `$...$`, or in the
  middle of a sentence — has every span it reached closed and is accepted, while the
  sections its own Contents announces were never reached, because the Contents is written
  first. The node also reads the anchors the Contents links to against the anchors the
  document declares, and asks again when one is missing, naming it. This is the linter's own
  finding (`Contents entry #float has no section anchor`), read before the answer is written
  rather than after it has spent a round. Seen on 2026-10-04: the **MaxPool** writer's third
  answer was cut inside a math span (2 spans opened, 2 closed), held one section of the four
  its Contents announced, and was 12 907 bytes against the 48 101 of the lint-clean revision
  it replaced — the run's last spec round went on it and the revision on disk got worse. The
  nine documents in `ops/` all satisfy the rule.
- **An answer that is not a case module is not a case module either.** The tester's node
  applies the same rule to the tester: an answer that is empty, or that defines no `cases()`
  or no `COVERAGE`, is not written over the module on disk, the tester is asked again
  (`--max-module-retries`, default 2), and the run stops with `cases-unusable` when it never
  answers with a module. Without this, an empty answer imports cleanly, declares no
  coverage, and every section of the document is then reported as *a section with no case*
  — a stop that names the document while the fault is the tester's. Seen on 2026-10-04 in
  the **Asin** run, whose second case module was one byte long.
- **An answer that is not an implementation is not an implementation either.** The blind
  implementer's node holds its answer to the same test before writing it — not empty, Python
  that parses, and a `def <entry>` with the entry point's name — and asks again when it
  fails. The implementer is the one agent whose answer nothing else reads first: an empty
  module is written to disk, imports cleanly, defines nothing, and the *harness* then reports
  it as a failure of the document, which sends the adjudicator to the writer with a finding
  that has no subject. Seen on 2026-10-04: the second reading of **Asin** was one character
  long, and the run stops with `impl-unusable` before the harness ever sees it. Nothing is
  written in that case; the run's record carries the reason.
- **A module that stopped in the middle is asked for again with twice the room.** A cut-off
  answer to the implementer is the writer's cut-off document in another guise, and it is
  repaired the same way: the parser's way of saying that the text stopped — an unterminated
  string or comment, a bracket never closed, an unexpected end of file — is read as a
  truncation rather than as prose, the answer is asked for again with a note to return the
  *complete* module and `2 × --max-tokens`, and the module is not written until it parses.
  Prose mistaken for a module is asked for again too, but with the same room, since the
  problem there is not the budget. Seen on 2026-10-04: both answers of the implementer for
  **Atan** were truncated (317 and 1857 bytes, "unterminated triple-quoted string literal"),
  and the run stopped with no implementation at all while the document was sound.
- **The tester's answer is parsed before it is written.** The two names a case module must
  declare — `cases()` and `COVERAGE` — can both be there in text that cannot be parsed, and
  text that cannot be parsed is a module that cannot be imported: writing it over the case set
  that is on disk discards the one whose failures were just sent back. The answer is therefore
  parsed, an answer that *stops in the middle* is asked for again with `2 × --max-tokens` like
  the implementer's, an answer that is not Python is asked for again with the same room, and a
  tester that never answers with a module stops the run with `cases-unusable` **before**
  anything is written. Found on 2026-10-04: the second case set of **Atan** was written over
  the first while only the two names had been looked for, and the interpreter then died on the
  import with `SyntaxError: invalid character '｜' (U+FF5C)` on its last line. The refused
  answer is kept beside the artifact (`cases-rejected-N.txt`), so a bad answer can be told from
  a bad reading of a good one.
- **An empty answer is asked again with the deliberation off.** A model that deliberates
  spends its output budget on the thinking, and the thinking is what the empty answer is
  made of: raising the budget does not win that race. The writer, the implementer and the
  tester therefore make their last attempt with `thinking` disabled, which is the one thing
  that removes the deliberation rather than racing it. Seen on 2026-10-04: the test agent
  answered the case-set prompt with no text at all on four calls in a row — 16 000 tokens,
  then 32 000, twice — and the **Acos** run stopped with no case module at all, having
  nothing left to run. The same fallback covers the writer, so an empty draft cannot burn a
  round either.
- **The module is not the first block of the answer.** A model that explains itself writes
  more than one fenced block, and the module is not always the first: it may open with the
  piece it is drawing attention to — the `COVERAGE` dictionary — in which case reading the
  first block reads a *fragment* of the module, and what the fragment lacks is then reported
  as what the answer lacks. The reader is told what the module must contain (the tester's
  `cases()` and `COVERAGE`, the implementer's `def <entry>`) and takes the block that has all
  of it. Seen on 2026-10-04: the test agent's answer for **Asin** was read as "a module with
  no `cases()`" and, on the retry, as "a module with no `COVERAGE`" — the two errors a
  fragment produces, twice. An answer that is refused is kept beside the artifact it did not
  become (`*-rejected-N.txt`), because a refusal that is not kept cannot be told from a
  reader that took the wrong part of a good answer.
- **The envelope of an answer is not the answer.** The endpoint's chat template leaks its own
  tool-call syntax into some answers — lines like `｜DSML｜ parameter>`, with `U+FF5C` for
  the bar — at the end of the artifact, after the last statement of a module that is itself
  complete and correct. It is stripped as a preamble is stripped, for the same reason: it is
  not part of what was asked for. Seen on 2026-10-04: the second case set of **Atan** and the
  first of **Asin** both ended this way, `SyntaxError: invalid character '｜' (U+FF5C)` on the
  last line of the module. What is *not* stripped is a tail that could be part of the answer —
  a line of prose, a fragment of a table — and that is refused and asked for again.
- **A harness run that dies leaves no report behind.** The report of the previous run is
  removed before the harness is started, because a run that dies before writing its own —
  a module that raises on import, a case set that stopped in the middle — leaves the report
  of the run before it on disk, and the driver that reads it sees a case module that never
  ran as one that passed. The **Asin** run of 2026-10-04 read its first report back after
  its second harness run had died, which is why the repair the tester owed was never asked
  for. A round that runs the harness more than once — a repair inside the node, or a case
  set that the adjudicator sent back — writes each run's report to its own file
  (`iter-001-report-2.json`), for the opposite reason: the report of the round before is the
  evidence of what the case set did then, and overwriting it leaves two rounds in one file.
- **A run that died is reported with what it printed.** A harness run that dies before it
  compares anything leaves no report, and an empty report is not a report: the tester is asked
  to repair the case module against what the report says, and against `{}` it is asked to
  repair the module it has just written. The driver therefore makes the report out of the
  process's own traceback, which then reaches the tester and the adjudicator as
  `ERROR the harness died before it wrote a report: …`. Seen on 2026-10-04: the second case set
  of **Atan** died on the import and the repair request carried `{}`, while the
  `SyntaxError` that named the line was in the run's output and nowhere else.
- **The report a run could not write is written for it.** The report the driver makes for a run
  that died, and the one it makes for a run it had to kill, are written to *that round's*
  report path, because the path is how the run's history records the round and how the loop
  directory the run publishes is read afterwards: a record that names a report nobody can open
  is not evidence of anything. Seen on 2026-10-04: the **Atan** record lists
  `iter-001-report-4.json` for the round killed at 600 s, that file was never written, and the
  published `verification/atan/loop/` holds reports 3 and 5 with the round between them missing
  — the round whose timeout is the reason the case set was rewritten.
- **A harness run that does not finish is a case module that cannot be run.** The case module
  decides how long a run takes, so a sweep that never ends, an exact-arithmetic computation
  over too many values, or a model built per case is the tester's defect and not something the
  loop waits out: the run is killed at `--harness-timeout` (default 1800 s) and reported like a
  case module that raises, which sends the tester back to it with the timeout as the error.
  Seen on 2026-10-04: the case module of **Acos** spent fifteen minutes of CPU in one sweep and
  was still running, and there was no timeout path at all — `subprocess.TimeoutExpired` escaped
  the node, and the process would have died with a traceback and no record of why. The report
  the driver makes for a killed run is written to the round's report path like any other.
- **A case set that failed is repaired against the failure.** A case module that *raises* is
  repaired against the Python error. A case set that ran and did not pass, and that the
  adjudicator attributed to the tester, is repaired against the failure itself — the
  adjudicator's instruction and the harness's report, in a section of the request headed
  *The case module failed* — and the adjudicator is told that its `instruction` is the whole
  of what the tester will be given, since the tester never sees its `reason`. Without that,
  the tester is asked to fix a case set it is told nothing about and answers with the same
  one. Seen on 2026-10-04 in the **Asin** run: the second case set (23 253 bytes, the first
  18 575) failed on the same two cases as the first, and the round had been requested with
  neither the report nor the instruction.
- **A revised case set is new evidence, and the revisions are counted.** The failure
  signature that the `unchanged` guard compares covers the *content* of the document, the
  implementation and the case set — not their paths, which change with the iteration even
  when the content does not. A failure that survives an unchanged *document* is not a dead
  end when the round was allowed to change only the cases: the tester is asked for a new case
  set precisely when the verdict is `test`, and the document is then the one thing that round
  must leave alone. That also means the `unchanged` guard can no longer bound this route, so
  the tester's revisions have their own budget (`--max-case-rounds`, default 2, reset by each
  new implementation) and the run then stops with `cases-stuck`. Seen on 2026-10-04: the
  **Asin** stop was `unchanged` — "the document did not change and the case still fails" —
  at iteration 1 of 5, with the reason naming the one artifact that was correct.
- **A section with no case is a guard, not a pass — and the repair comes first.** The test
  agent must declare a `COVERAGE` dictionary over the document's own Contents anchors, and a
  section with no case is a defect of the **case set**: the document announced it and the
  cases did not exercise it. It is therefore repaired inside the test node, within the
  tester's repair budget (`--max-case-repairs`), with a brief that names the uncovered
  anchors and the two ways to cover them — cases in `cases()` for a section with an ONNX
  type, a sweep's checks counted into `COVERAGE` for `real`, which has no type the runtime
  can be asked about — and the scheduler stops with `uncovered-section` only when the
  repairs are exhausted. It used to stop immediately, before any repair and before the
  adjudicator: measured on 2026-10-05, **MaxPool** attempt 6 left `COVERAGE["real"] = 0`
  while its `real` sweep compared the section, exactly the shape the accepted case modules
  count by hand, and the loop stopped on a case module it had a budget for. The dictionary
  is read by **importing the case module** in the interpreter the harness uses, not by
  parsing its source: a case set that counts its own cases per section — `COVERAGE =
  {"real": 0, ...}` filled in from the case list as the module loads — looks empty in the
  source, and reading the literal of the assignment reported every section of a 156-case set
  as uncovered. (Seen on 2026-10-04; the static read survives only as the fallback for a
  module that will not import.)
- **The case module is written against the interpreter that runs it.** That interpreter is
  fixed — the standard library, `numpy`, `onnx`, `onnxruntime`, nothing else — and the
  tester is told so, because an import of anything else is a failed test of the tester's and
  not a finding about anything: inside `cases()` the harness reports it as an error of the
  case module, inside a sweep it is a failing sweep. Seen on 2026-10-04: the case module for
  **Asin** reached for `mpmath` in its sweeps, and the run spent its repair rounds on
  `ModuleNotFoundError: No module named 'mpmath'` while the document and the implementation
  were both sound. High precision in this repository is `decimal` and `fractions.Fraction`,
  as `verification/sin/sin_oracle.py` does it.
- **A case set measures compliance, not accuracy — and its own reference may be the wrong
  one.** A document that says `round(arcsin(x))` with `roundTiesToEven` states a correctly
  rounded value, and refers a departure from it to `other/accuracy.md`; no library on this
  platform is correctly rounded for the inverse trigonometrics, so a case set that fails an
  implementation for a departure of one unit in the last place fails every implementation
  anyone can write. The accepted case set of **Sin** is the shape: departures are measured in
  units of the last place, a small explicit slack is allowed, and what was measured is
  reported even when it is not failed. The mirror image is just as costly: the case module's
  *own* `Decimal` reference can be the thing that is off by a unit. Seen on 2026-10-04: the
  first case set of **Asin** failed the implementation for a one-ulp departure the accuracy
  guidelines permit (the adjudicator sent it back with a one-ulp tolerance for `float32` and
  `float64`, keeping the exhaustive `float16` check bit-exact), and the first case set of
  **Acos** failed the implementation for returning `2.9740648100904594` at `-0.986`, which is
  the correctly rounded value, its own reference being one unit low. The tester is now told
  both, and told to compute a doubtful reference a second way before failing anyone.
- **The reference's operand is the value the tensor holds, not the decimal that was written.**
  A case module that computes its reference from `Decimal(repr(x))` — the shortest decimal
  representation — asks about a *different number* from the one the tensor holds, and the
  difference (about `1e-18` at `-0.986`) decides the last bit of the correctly rounded result
  whenever the true value sits within `1e-17` of a midpoint between two values of the type.
  Measured on 2026-10-04: the first case set of **Acos** reported `2.97406481009046`,
  `2.887930917514338`, `2.8577985443814655` for `-0.986`, `-0.968`, `-0.96`, where the exact
  arcsine of the float gives `2.9740648100904594`, `2.8879309175143377`, `2.857798544381465`
  and the platform returns those — so three false failures of an implementation that is right
  to the last bit. `Decimal(v)` on the float is exact, and is what the accepted **Asin** case
  set uses; `Decimal(repr(v))`, `Decimal(str(v))` and the literal that built the array are not.
- **ONNX has two implementations, and the reference implementation is the definite one.**
  ONNX Runtime is what the harness compares against, and `onnx.reference` — in the same
  interpreter, so the tester can always ask it — is the ONNX project's own second
  implementation of the same operator. Measured on 2026-10-04, in the **MaxPool** run, over a
  `float32` window with `kernel_shape [2, 2]`: `{NaN, 2, 3, 4}` gives `4.0` from the runtime
  and `4.0` from the reference implementation (`_op_common_pool` filters the NaN elements out
  before `np.max`, which is also how it excludes the padded elements) while the document said
  NaN — the document at fault, and the adjudicator's two `test` verdicts would have laundered
  it. The runtime's agreement there is an accident of position: the same four values with the
  NaN written last give `NaN` from the runtime and `3.0` from the reference implementation, and
  a window of all `-inf` gives `-3.4028235e+38` from the runtime at `kernel_shape [2, 2]` and
  `-inf` at `[1, 1]`, on the same input, and `-inf` from the reference implementation at every
  shape. So the reference implementation is what the document is judged against and what a
  sweep decides a special value by, with the runtime's differing value recorded beside it; the
  tester is told the measurements, and the adjudicator is told both sides, because the verdict
  cannot be read off the shape of the failure.
- **A type ONNX does not admit is a defect of the document, not a narrow build.** The
  document's type families are the operator's `T` in the schema, and `onnx.defs.get_schema`
  states it without running anything. Measured on 2026-10-04: `MaxPool` admits `float16`,
  `float`, `double`, `int8` and `uint8` and **no `int64`** — its `tensor(int64)` constraint is
  the type of the `Indices` *output* — and the run's document gave its `(int)` section as
  `where int is in {int8, int64}` and asserted in its worked example that the `int8` values
  "are also values of `int64`". ONNX Runtime refused every such case at model resolution with
  `INVALID_GRAPH ... Type Error: Type 'tensor(int64)' of input parameter (X) of operator
  (MaxPool) ... is invalid`, which reads like the `NOT_IMPLEMENTED` of *A type the runtime
  does not implement* and is the opposite of it: there the document is right and the build is
  narrow, here the document admits a type ONNX does not. The tester now checks the document's
  families against the schema's in a sweep and fails it, the adjudicator is told the refusal
  is the schema's, and the run is sent back to the document rather than converging on it.
  **The driver's own authority is the schema's first input, and it was wrong.** The admitted
  set handed to the writer and the verifier was the union of every type constraint the
  operator has, so for `MaxPool` it read `int8, int64` — the driver itself had written the
  `int64` the document then defended, which is the traceable cause of the run's whole test
  side. `graph.admitted_types` now takes the type parameter of the *first* input (`T`, or
  `T1` where the operator renames it) and falls back to the union only for an operator whose
  first input names no parameter; the writer's retry loop refuses a document that announces a
  type outside it, with the same mechanism that refuses an answer cut off in the middle, and
  `type_constraints_text` now tells the verifier `MaxPool`'s `int` family is `int8` alone.
- **A sweep has a budget, and an unbounded one is a defect of the case module.** The harness
  runs the case module under a time limit and kills a run that does not finish inside it, and
  the run attributes the kill to the tester — the case module is what decides how long a run
  takes — and asks it to repair what it wrote. It was measured, not guessed: the case set of
  **Acos** spent fifteen minutes of CPU in a single sweep and was still running (which is why
  the timeout path exists at all), and on 2026-10-04 the case set of **Atan** was killed at
  600 s and spent the round on it. The tester is now told the limit is there and is told to size
  each sweep from a measurement: exact `Decimal` arithmetic costs hundreds of times what a
  float costs, so an exhaustive sweep belongs to `float16` (65536 values) and the wider types
  get boundaries and a sample of a few hundred, and a model belongs to a batch rather than to
  an element.
- **A run publishes the reports it wrote, not the ones in the directory.** The run directory
  is shared by every attempt at an operator — a second attempt runs in the same
  `agents/runs/<Op>/`, and so does a run resumed after a stop — and the report names repeat
  from one attempt to the next, so `iter-001-report-2.json` of the attempt before is
  indistinguishable from this run's by its name alone. Measured on 2026-10-05: `MaxPool`
  attempt 6 began in a directory holding ten reports of attempt 4, and its `--publish` would
  have copied them into `verification/maxpool/loop/` as this run's own evidence, because
  publishing swept the directory. The run now records the report and implementation paths it
  wrote, `publish` copies exactly those, and a record written before the field existed is
  still read the old way, so an old run stays publishable.
- **Infinity and NaN are not real numbers.** A sweep over the *special values* of a type must
  decide them by the rule the document states for them and never reach the exact arithmetic
  with them: `Decimal(float("inf")) - Decimal(float("inf"))` raises `InvalidOperation`, and a
  sweep that raises is a failing sweep, so the run stops on it and never reaches the document.
  Seen on 2026-10-04: `sweep_special_values` of the **Atan** case set ended in
  `decimal.InvalidOperation`; the accepted **Asin** case set shows the shape, returning NaN for
  a NaN, an infinity or an operand outside the domain and reaching for `Decimal` only for a
  finite operand the document admits.
- **The loop cannot say the document is right.** It can only say that a fresh reader of it
  agrees with ONNX Runtime — and with ONNX's own reference implementation where the two
  agree — on the cases covered. The run record carries the coverage so the reader can judge
  how much of the operator was exercised.

## The one place this system goes beyond the skill

The skill's driver says: *do not touch the implementation to make a test pass* — a failing
reading is evidence about the specification. This system keeps that as its bias (the
adjudicator's prompt says so) but adds the second route the task asks for: when the
document decides a case in its plainest reading and the code does something else, the
adjudicator may send the correction back to the **implementation**. Both routes are
recorded: every verdict, its reason and its instruction go into the run record
(`run-<timestamp>.json`), so an amendment to the document is never confused with a
correction of the code.

## Running it

```bash
python3 -m venv ~/Venvs/sonnx
~/Venvs/sonnx/bin/pip install -r agents/requirements.txt      # langgraph, langchain-anthropic, numpy, onnx, onnxruntime

~/Venvs/sonnx/bin/python agents/run.py --op Neg
```

The model is any endpoint that speaks the Anthropic Messages API: `ANTHROPIC_BASE_URL` and
`ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) are read from the environment, and
`SONNX_AGENT_MODEL` (or `ANTHROPIC_DEFAULT_SONNET_MODEL`) names the model. A model that
thinks before it answers is handled: only the text blocks of a response are read, and a
response with no text at all is retried once with twice the budget.

| flag | what it does |
|---|---|
| `--op <Op>` | the ONNX operator type (required) |
| `--spec <path>` | where the document goes (default `ops/<op>.md`; if that file exists it is **not** overwritten unless `--force`) |
| `--run-dir <path>` | where the run's artifacts go (default `agents/runs/<Op>/`) |
| `--start-from spec` | skip the writer and start from the file on disk |
| `--max-iterations N` | implement → test rounds, the skill's cap (default 5) |
| `--max-spec-rounds N` | writer → verifier rounds (default 3) |
| `--max-case-repairs N` | times the test agent may repair a case module that raises (default 3) |
| `--max-case-rounds N` | times the test agent may rewrite its case set after the adjudicator attributed a failure to it (default 2) |
| `--harness-timeout N` | seconds one harness run may take before it is killed and reported as a case module that cannot be run (default 1800) |
| `--max-module-retries N` | times the blind implementer or the test agent may answer again when its answer is not a usable module (default 2) |
| `--max-writer-retries N` | times the writer may answer again when its answer is cut off (default 2) |
| `--max-review-retries N` | times the verifier may answer again when its verdict is unreadable (default 1) |
| `--max-review-tokens N` | output budget of one compliance review (default 64 000) |
| `--thinking auto\|off` | may the writer, the implementer and the tester deliberate (default `auto`) |
| `--verdict-thinking auto\|off` | may the verifier and the adjudicator deliberate (default `off`) |
| `--onnx-doc <file>` / `--no-fetch` | the ONNX definition from a file, or not at all (see below) |
| `--publish` | on convergence, copy the artifacts into `ops/` and `verification/<op>/` |
| `--publish-only` | publish the converged run recorded in `--json-state`, with no model call: the document the caller read the verdict for is the document that gets published |
| `--json-state <file>` | the final state, for inspection — it carries the outcome, so `--publish-only` can check it |

`verification/` is not touched until `--publish` (or `--publish-only`), and a specification
that is already there cannot be destroyed by accident: when `ops/<op>.md` exists the run
keeps its working document in `agents/runs/<Op>/<op>.md` and leaves the published file alone
(the caller asks for it back with `--spec` or `--force`), and `--publish-only` refuses to
publish a run whose outcome is not `converged`.

An operator with **no** document yet is the other case, and it drafts straight to
`ops/<op>.md`: that file is the run's working document from the first round, rewritten by
every round, and it is only on `--publish` that the converged document replaces it. So a run
that stops leaves its last draft there — the **MaxPool** run of 2026-10-04 did, and it is
archived with its log in `agents/runs/MaxPool/attempt-5/`. Read a document in `ops/` as
published only if the run whose record you can open converged.

Publishing **Neg** copies the document to `ops/neg.md`, the case module to
`verification/neg/cases.py`, and the implementation and its harness report to
`verification/neg/loop/`. It does not write `verification/<op>/loop/README.md`: that page
classifies the blind reader's `DECISION:` points and records a detector check, which is work
about the run that no copy can do. The run record (`run-<timestamp>.json`) is the raw material
for it.

### The ONNX definition the writer is given

The writer is given the operator's definition, because ONNX is the reference for the
semantics. It comes from the installed `onnx` package (`onnx.defs.get_schema`), which
states exactly what the writer needs and cannot be got wrong: the prose, the type
constraints that decide which families the document must have (a family the operator does
not admit is a document that invents one), and the inputs, outputs and attributes, for the
opset asked for. The operator's web page is the fallback, read through its `<article>`:
the documentation site's navigation is longer than the definition itself, and reading the
whole page returns the menu. `--onnx-doc <file>` overrides both. Whichever was used is
written to `onnx-<Op>.txt` in the run directory, so the reader of a run can see what the
writer had.

### The restriction files

Two files in `ops/` state what an operator is *not* defined to do, and they are repository
inputs rather than run artifacts: the run reads them where they are and never writes them.

- `ops/global_restrictions.md` — the restrictions that apply to every operator. A
  document's `## Restrictions` block **links** it (`[General restrictions](./global_restrictions.md)
  are applicable.`) and never copies it; its rows use a `[G<n>]` tag so they are visibly not
  a document's `[R<n>]` rows.
- `ops/<entry>_restrictions.md` — the restrictions of one operator, as a table and nothing
  else, so its rows can be pasted. `ops/<Op>_restrictions.md` (the ONNX spelling) is
  accepted too, and the entry spelling wins when both exist. `None` — the ordinary case for
  an operator that introduces no restriction of its own — is not an error: the block then
  carries `No specific restrictions apply to the **Op** operator.`

The writer's node reads the operator's file and refuses a document whose blocks do not carry
its rows verbatim and in order, in every family section (see the guarantee above); the
verifier and the tester are given both files. The rows are copied into the document, so they
obey the document's modality — an indicative limit ("The padding of a spatial axis is at
most the size of the kernel"), never `must`, `should`, `---` or a code fence, which
`lint_spec.py` forbids. `[R<n>]` numbering is per operator file and the tags repeat across
the family sections, as they do in the published documents; they are not `E_` tags, so the
linter ignores them. The three-column shape (`Restriction | Statement | Origin`) is the
guidelines'; a restriction that applies to one family only names that family in its
Statement rather than adding a column. Known limitation: `--publish` does not check the
published document against the operator's file, so reconcile the two before publishing —
`ops/maxpool.md` and the run directory's MaxPool document already disagree on `R4` — and a
publish-time check is a later change.

## The run directory

```
agents/runs/<Op>/
    <Op>.md              the document, as the writer left it (unless --spec says otherwise)
    cases.py             the test agent's case module
    cases.meta.json      the md5 of the document it was written for
    iter-001-impl.py     the blind implementation of iteration 1
    iter-001-report.json the harness's report for its first run
    iter-001-report-N.json  its second and later runs, which keep their own reports
    *-rejected-N.txt     an answer that was refused, kept for the reader of the run
    onnx-<Op>.txt        the ONNX definition the writer was given, as fetched
    run-<timestamp>.json the record: every agent's verdict, the coverage, the last report
```

Every attempt at the operator runs here, so the directory holds what the attempt before it
left behind as well, and the names repeat. The record — not the directory — is what says
which files are this run's: it carries the report and implementation paths the run wrote, and
`--publish` copies those.

## The scheduler, tested offline

```
~/Venvs/sonnx/bin/python agents/tests/test_graph_offline.py
```

Every agent is replaced by a canned answer and the two harness programs by canned verdicts,
so the test exercises exactly the part a model must not decide: which node runs after
which, and which guard stops the loop. Thirty-seven scenarios: convergence, an unresolved review
finding, an amendment that changes nothing, the two correction routes, the unchanged-document
guard, a review whose blocking finding withdraws itself (the reviewer is told what the
contradiction is and asked again, the finding never reaches the writer, and a reviewer that keeps
asking for no change is not counted — the run proceeds instead of stopping `stalled` on an
amendment that cannot be made), the uncovered-section guard (the case set is asked again with the anchors named, and
the run stops with `uncovered-section` only once the repairs are exhausted — a second
scenario is the repair that succeeds, and the run converges), a repair of a broken case module, an adjudicated stop,
the iteration cap, a writer answer that stops in the middle of the document (twice: the
retry completes it, and nothing ever does), a verifier that returns no verdict (twice: it
never does, and it does on the second attempt), a tester answer that is not a case module
(twice: the retry completes it, and nothing ever does), a harness run that dies with the
previous run's report still on disk, a blind implementer's answer that is not an
implementation (twice: the retry completes it, and nothing ever does), and an answer that is
empty for as long as the model deliberates (for the tester and for the implementer: the last
call is made with no deliberation, and is the one that answers), and an answer whose module
is not its first fenced block (the piece first, the module second, plus the refused answers
that are kept), an implementer answer that stops in the middle of the module (twice: the
retry completes it with twice the room, and nothing ever does), and a case set the adjudicator
blamed (the tester is told what failed and what the adjudicator said, its revisions are
counted, a revision that is the same module is still `unchanged`, and every harness run keeps
its report), and a harness run that does not finish (once — the tester makes it cheaper — and
for ever), and a tester answer that is not Python or that stops in the middle (the run refuses
it without writing it, the retry is given twice the room when the answer stopped in the
middle, and a tester that never answers with a module stops the run without touching the case
set on disk), and a harness run that dies before it writes a report (the real driver, on the
real harness, with one character too many in the case module: the traceback is what the
tester and the adjudicator are given), and a run the harness kills at its limit (also the real
driver: both synthesized reports are written where that round's report belongs, so the path
the run's history records is a file a reader can open), and a sweep that raises (the sweep's
name and its traceback are what the tester is asked to repair against, not the case counts a
report of a comparison that never happened would show), and a case set whose every case the
runtime refused (the report reads as a pass, the driver reads it as `_broken`, and the tester is
sent back the refusals themselves rather than an empty verdict), and an operator with attributes
and two outputs end to end on the real harness (a node built with `kernel_shape`, both outputs of
the document compared by position, a wrong second output reported under its own name, and the
attributes recorded in the report), and a writer's answer that keeps the Contents of the document
but not its sections (the revision on disk is the one with the sections, and the writer is asked
again before the round is spent), and a prompt that names another agent (the adjudicator's prompt
says which agent must have consulted ONNX's own reference implementation, and the fake answers it
by the role on the first line rather than by a name in the body: a dispatch on substrings answered
that call with a case module, which the node read as a `stop` and which turned scenario 3 into
`adjudicated-stop`), and a document that gives a family a type ONNX does not admit for the
operator — in the Contents and in the sentence that opens the section (the admitted set is read
from the schema's *first input*, so `uint8` is refused for **Neg**, `int64` is not an admitted
MaxPool type, and the writer is asked again for the families to correct), and the publish of a
run that shares its directory with an earlier attempt (the run names the reports and the
implementations it wrote, the published loop holds those and not the attempt before's, and a
record written before the run named them is read the old way, so an old run stays
publishable), and the restriction files (the old link refused at the writer with the
sentence named and the new one on disk after the retry; the operator file's rows carried
verbatim into every section, a dropped row and an invented row each named by their `[Rn]`
tag, the Contents read in document order rather than sorted, and a second section missing
the row refused; an operator with no file carrying the mandatory sentence, with a table and
no file refused as invented and the ONNX-case spelling found; and the implementer never
given either file, the sentinel in the writer's, verifier's and tester's messages and
absent from the implementer's; and the operator's restriction file appearing while the
writer is answering — the request was built before the file existed, the retry is refused
until it carries the file's rows, and a request rebuilt for the attempt is what makes the
retry carry them). A last section asserts
the family
verdicts the verifier is handed, computed from the operator's schema: **Neg** has no `uint`
family, **Add** has one, **Sin** has neither `int` nor `uint`, and **MaxPool**'s `int` family is
`int8` alone — its `tensor(int64)` constraint is the type of the `Indices` output, not a data
type; and a `COVERAGE` dictionary
the case module computes for itself is read as the module leaves it.
