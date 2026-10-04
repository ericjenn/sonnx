# The blind implementation task

Give this to a **fresh** agent, verbatim, with `<SPEC>` and `<OUT>` filled in. The agent must not
be told about ONNX Runtime, about the tests, or about this skill; it reads one file and writes one
file. That isolation is the point of the exercise — anything the agent knows that the specification
does not say hides a defect of the specification.

Spawn it with a plain, general-purpose agent (no extra tooling), and give it no other context.

---

Read the file `<SPEC>` and nothing else. Do not read any other file in this repository.

This file is the specification of an operator. Implement that operator in Python, in a single file,
at `<OUT>`, using numpy.

Requirements for the file you write:

- it exposes the operator under the name `<ENTRY>`, a function taking one positional argument per
  operand and returning the result. An operator with attributes takes them as **keyword parameters**,
  under the names the document's `Attributes` section gives them and with the defaults it gives them,
  so that `maxpool(x, kernel_shape=[2, 2])` is a call of the operator the document specifies. An
  operator whose document gives it more than one output returns them as a **tuple**, in the
  document's order, first output first;
- it is a faithful implementation of that document and of nothing else: where the document decides
  a case, your code does what it says, even where you would have done otherwise;
- the document is a specification, not a tutorial. Where it is silent, or where you had to choose,
  **say so in a comment** beginning with `DECISION:`, together with the sentence of the document you
  relied on, or the statement that the document says nothing about it. List every such point; a
  point where you had to guess is the most useful thing you can report;
- if a part of the document is ambiguous, implement the reading you consider correct and record the
  ambiguity as a `DECISION:` comment. Do not ask questions, and do not stop;
- do not add validation that the document does not require, and do not suppress warnings or errors
  the document does not mention;
- write substantial docstrings quoting the part of the document each function implements.

When you are done, reply with the absolute path of the file you wrote and the list of your
`DECISION:` points, one line each.
