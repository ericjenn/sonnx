# SONNX informal specification generator

The SONNX operator-specification profile: the guidelines that define the form of an operator
specification, the accepted specifications, and the multi-agent system that writes, reviews,
amends and verifies them. The pipeline can also formalize an accepted specification in Lean 4,
and, from the family definitions of the formal specification, derive and test C kernels of
the operator over a generated `sonnx_tensor` type (data, rank and dimensions). The C is
derived and rendered by deterministic code, never written by a model, and every kernel names
the Lean declaration and `@trace` requirement it came from.

The description of the multi-agent system is in [`agents/README.md`](agents/README.md).

## Repository layout

| path | contents |
|---|---|
| `docs/` | the normative guidelines, the specification template, the accuracy guidelines, and an archive of superseded material |
| `ops/` | one folder per operator (`ops/<op>/`), holding its specification and every artifact a run produces for it — tests, blind implementations, harness reports, Lean formalization and generated C — plus the restriction files at the top level |
| `agents/` | the LangGraph pipeline: agents, prompts and tests |
| `.claude/skills/operator-spec/` | the skill the pipeline implements: its instructions, references and mechanical checks |
| `lean/` | the Lake build root: the shared `Sonnx` Lean library (tensors, broadcasting, indices, integer and floating-point types) that every generated formalization and runner imports, and `CGen`, a Lean AST and pretty-printer that generate the C kernels of the elementwise operators over a `sonnx_tensor` struct |
