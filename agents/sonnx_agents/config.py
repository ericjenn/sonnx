"""Paths, model endpoint and loop guards.

Everything the agents need to know about *where* things are and *how far* the loop may
go is here, so that a node never has to guess a path.  The repository is located from
this file, not from the current directory, so the CLI runs from anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# agents/sonnx_agents/config.py -> agents/ -> repository root
AGENTS_DIR = Path(__file__).resolve().parents[1]
REPO = AGENTS_DIR.parent
SKILL_DIR = REPO / ".claude" / "skills" / "operator-spec"
ASSETS_DIR = SKILL_DIR / "assets"
REFERENCES_DIR = SKILL_DIR / "references"
PROMPTS_DIR = AGENTS_DIR / "sonnx_agents" / "prompts"

# the normative documents of the profile
GUIDELINES = REPO / "docs" / "informal_corrected.md"
TEMPLATE = REPO / "docs" / "informal_spec_template.md"
OPS_DIR = REPO / "ops"                            # one folder per operator, and the
                                                  # restriction files beside them
REFERENCE_SPEC = OPS_DIR / "add" / "add.md"       # the accepted shape reference
WORKED_CASES = OPS_DIR / "add" / "cases.py"       # the worked case-set example

DEFAULT_PYTHON = str(Path.home() / "Venvs" / "sonnx" / "bin" / "python")
ONNX_DOC_URL = "https://onnx.ai/onnx/operators/onnx__{op}.html"


def _model() -> str:
    return (
        os.environ.get("SONNX_AGENT_MODEL")
        or os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
        or "claude-sonnet-5-5"
    )


@dataclass
class Settings:
    """One run of the multi-agent system."""

    op: str                                    # ONNX operator type, e.g. "Neg"
    opset: int = 14

    # -- model ------------------------------------------------------------
    model: str = field(default_factory=_model)
    base_url: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_BASE_URL", "") or None)
    api_key: str = field(
        default_factory=lambda: (
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or ""
        )
    )
    max_tokens: int = 16000
    # a review of a whole document against the whole of the guidelines is a long piece of
    # deliberation: a model that thinks before it answers spends its budget on the thinking
    # and, with 16000, returns no JSON at all
    max_review_tokens: int = 64000
    # "auto" lets the model deliberate before it answers; "off" asks it not to.  The agents
    # that answer with a bounded verdict take "off": deliberation that overruns the budget
    # costs them the answer, and their answer is a page, not a document.
    thinking: str = "auto"
    verdict_thinking: str = "off"
    temperature: float = 0.0
    timeout: float = 1800.0

    # -- the loop ---------------------------------------------------------
    max_iterations: int = 5        # implement -> test rounds (the skill's cap)
    max_spec_rounds: int = 3       # writer -> verifier rounds before the first implementation
    max_case_repairs: int = 3      # times the tester may repair a broken case module
    harness_timeout: int = 1800    # seconds one harness run may take before it is killed and
    #                                reported as a case module that cannot be run
    max_case_rounds: int = 2       # times the tester may rewrite its case set after the
    #                                adjudicator attributed a failure to it, within one
    #                                implementation: a revised case set is new evidence, so
    #                                the `unchanged` guard no longer bounds these rounds
    max_module_retries: int = 2    # times the implementer or the tester may answer again when
                                   # its answer is not a usable module
    max_writer_retries: int = 2    # times the writer may answer again after a cut-off answer
    max_review_retries: int = 1    # times the verifier may answer again after an unreadable one
    max_lean_retries: int = 2      # repairs after the first formalizer answer
    max_conformance_retries: int = 2  # repairs after the first generated Lean runner answer
    lean_timeout: int = 300        # seconds for one lake env lean invocation
    lake: str | None = None        # Lake executable; None discovers it from the environment
    lean: bool = False             # insert the Lean formalizer into the pipeline
    lean_only: bool = False        # compliant informal spec -> Lean, without Python verification
    lean_conformance: bool = True  # generate and run the Lean-vs-corpus check
    lean_conformance_generate: bool = True  # generate the runner and driver when none are given
    lean_conformance_timeout: int = 1800  # seconds for one Lean conformance run
    lean_conformance_script: Path | None = None  # override the operator's driver (tests)
    lean_conformance_compare: str = "bits"  # bits | ulp | relative
    lean_conformance_ulp: int = 2
    lean_conformance_relative: float = 1e-6
    start_from: str = "scratch"    # "scratch" (write the spec) or "spec" (use the file on disk)

    # -- the C stage ------------------------------------------------------
    cgen: bool = False             # generate C from the Lean descriptor and test it
    c_compiler: str = "cc"         # the C compiler that builds the kernels and test program
    max_c_retries: int = 2         # times the C tester may answer again when its program is
                                   # not usable (empty, not C, does not compile, compares nothing)
    c_timeout: int = 120           # seconds for one C compilation or one test-binary run

    # -- where things are written ----------------------------------------
    spec_path: Path | None = None      # default: the run directory's <entry>.md
    force: bool = False                # allow writing over an existing operator document
    run_dir: Path | None = None        # default: ops/<entry>/ — the operator's own folder
    ops_dir: Path | None = None        # default: the repository's ops/ (tests redirect it)
    restrictions_dir: Path | None = None   # default: ops/ — where the restriction files live
    onnx_doc: Path | None = None       # a file holding the ONNX operator definition
    fetch_onnx_doc: bool = True
    python: str = DEFAULT_PYTHON       # the interpreter that runs lint_spec.py and specloop.py
    verbose: bool = True
    publish: bool = False              # validate a finished run and place its artifacts
                                       # in ops/<entry>/ when it ran elsewhere
    publish_only: bool = False         # publish the finished run in `json_state`, no model call
    json_state: str | None = None      # write the final state here
    lean_artifact: str | None = None   # existing artifact for --check-lean
    check_lean: bool = False           # deterministic recheck, with no model call

    @property
    def entry(self) -> str:
        """The entry point of the implementation module (the skill's convention)."""
        return self.op.lower()

    @property
    def op_upper(self) -> str:
        return self.op.upper()

    @property
    def family_tag(self) -> str:
        return self.op.upper()

    def resolved_spec_path(self) -> Path:
        """The document the writer reads and writes.

        Every operator keeps its document in its own folder, so the run directory and the
        document's home are the same place.  An explicit ``--spec`` wins, and an explicit
        ``--run-dir`` keeps a run confined to that directory (and its document with it).
        Whether an existing document may be replaced is decided by the CLI's guard, not
        here: this method only says where the document is.
        """
        if self.spec_path is not None:
            return Path(self.spec_path)
        return self.resolved_run_dir() / ("%s.md" % self.entry)

    def ops_root(self) -> Path:
        """The `ops/` directory: one folder per operator, and the restriction files."""
        return Path(self.ops_dir) if self.ops_dir is not None else OPS_DIR

    def resolved_run_dir(self) -> Path:
        if self.run_dir is not None:
            return Path(self.run_dir)
        return self.ops_root() / self.entry

    def global_restrictions_path(self) -> Path:
        """`ops/global_restrictions.md` — the restrictions that apply to every operator.

        Linked by every `## Restrictions` section and never copied into a document, so it
        is context for an operator's semantics and not part of what a document carries.
        """
        return (self.restrictions_dir or self.ops_root()) / "global_restrictions.md"

    def restrictions_path(self) -> Path | None:
        """`ops/<entry>_restrictions.md`, or the ONNX-case spelling `ops/<Op>_restrictions.md`.

        The rows of this file are carried verbatim into every `## Restrictions` section of
        the document (see `graph._carries_restrictions`).  The entry spelling is tried
        first because the repository names its operator folders lowercase (`ops/maxpool/`), and
        the ONNX type case is accepted because the operator file is named after the
        operator, not after the entry point.  `None` when neither file exists, which is the
        ordinary case for an operator that introduces no restriction of its own.
        """
        ops = self.restrictions_dir or self.ops_root()
        for name in ("%s_restrictions.md" % self.entry, "%s_restrictions.md" % self.op):
            if (ops / name).exists():
                return ops / name
        return None

    def cases_path(self) -> Path:
        """The case module the tester writes, beside the document it was written for."""
        return self.resolved_run_dir() / "cases.py"

    def loop_dir(self) -> Path:
        """The blind implementation and the harness reports of every iteration.

        The subdirectory is the layout a finished run has always published, so a run that
        writes into its operator's folder needs no second copy to publish.
        """
        return self.resolved_run_dir() / "loop"

    def impl_path(self, iteration: int) -> Path:
        return self.loop_dir() / ("iter-%03d-impl.py" % iteration)

    def report_path(self, iteration: int) -> Path:
        return self.loop_dir() / ("iter-%03d-report.json" % iteration)

    def lean_run_dir(self) -> Path:
        return self.resolved_run_dir() / "lean"

    def lean_path(self) -> Path:
        return self.lean_run_dir() / "FormalSpec.lean"

    def lean_trace_path(self) -> Path:
        return self.lean_run_dir() / "FormalSpec.trace.json"

    def lean_attempt_path(self, attempt: int) -> Path:
        return self.lean_run_dir() / ("attempt-%03d.lean" % attempt)

    def lean_conformance_runner_path(self) -> Path:
        return self.lean_run_dir() / "LeanConformance.lean"

    def lean_conformance_attempt_path(self, attempt: int) -> Path:
        return self.lean_run_dir() / ("conformance-attempt-%03d.lean" % attempt)

    def lean_conformance_driver_path(self) -> Path:
        return self.resolved_run_dir() / "lean_conformance.py"

    def lean_conformance_corpus_path(self) -> Path:
        """The corpus the generated driver runs, frozen next to the runner.

        A later rewrite of the Python loop's `cases.py` must not change the evidence of a
        conformance run that already happened.
        """
        return self.lean_run_dir() / "cases.py"

    def lean_conformance_profile_path(self) -> Path:
        return self.lean_run_dir() / "conformance-corpus.json"

    def lean_conformance_manifest_path(self) -> Path:
        return self.lean_run_dir() / "conformance.manifest.json"

    def lean_conformance_report_path(self) -> Path:
        return self.lean_run_dir() / "LeanConformance.json"

    def lean_conformance_smoke_path(self) -> Path:
        return self.lean_run_dir() / "LeanConformanceSmoke.json"

    def published_spec_path(self) -> Path:
        """The operator's document in its own folder."""
        return self.ops_root() / self.entry / ("%s.md" % self.entry)

    def published_cases_path(self) -> Path:
        """The accepted case module in the operator's folder."""
        return self.ops_root() / self.entry / "cases.py"

    def lean_publication_dir(self) -> Path:
        return self.ops_root() / self.entry / "lean"

    def lean_publication_driver_path(self) -> Path:
        return self.ops_root() / self.entry / "lean_conformance.py"

    # -- the C stage ------------------------------------------------------

    def c_dir(self) -> Path:
        """The generated C, its header, the test program and the report."""
        return self.resolved_run_dir() / "c"

    def c_source_path(self) -> Path:
        return self.c_dir() / "sonnx_kernels.c"

    def c_header_path(self) -> Path:
        return self.c_dir() / "sonnx_kernels.h"

    def c_emit_path(self) -> Path:
        """The copy of the formal specification with the `#eval` trailer that renders C.

        The accepted artifact is never modified: the copy carries the trailer, so Lean
        evaluates the module's own descriptor and writes the files as a side effect.
        """
        return self.c_dir() / "CGenEmit.lean"

    def c_test_path(self) -> Path:
        return self.c_dir() / "c_tests.c"

    def c_test_attempt_path(self, attempt: int) -> Path:
        return self.c_dir() / ("c_tests-attempt-%03d.c" % attempt)

    def c_binary_path(self) -> Path:
        return self.c_dir() / "c_tests"

    def c_report_path(self) -> Path:
        return self.c_dir() / "c-report.json"

    def c_publication_dir(self) -> Path:
        return self.ops_root() / self.entry / "c"
