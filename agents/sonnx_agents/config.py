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
GUIDELINES = REPO / "informal_corrected.md"
TEMPLATE = REPO / "informal_spec_template.md"
REFERENCE_SPEC = REPO / "ops" / "add.md"          # the accepted shape reference
WORKED_CASES = REPO / "verification" / "add" / "cases.py"   # the worked case-set example

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
    start_from: str = "scratch"    # "scratch" (write the spec) or "spec" (use the file on disk)

    # -- where things are written ----------------------------------------
    spec_path: Path | None = None      # default: ops/<op>.md, or the run directory if it exists
    force: bool = False                # allow writing over an existing ops/<op>.md
    run_dir: Path | None = None        # default: agents/runs/<Op>/
    onnx_doc: Path | None = None       # a file holding the ONNX operator definition
    fetch_onnx_doc: bool = True
    python: str = DEFAULT_PYTHON       # the interpreter that runs lint_spec.py and specloop.py
    verbose: bool = True
    publish: bool = False              # copy a converged run into ops/ and verification/
    publish_only: bool = False         # publish the finished run in `json_state`, no model call
    json_state: str | None = None      # write the final state here

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
        if self.spec_path is not None:
            return Path(self.spec_path)
        if self.run_dir is not None:
            # a run confined to a directory keeps its document there too
            return self.resolved_run_dir() / ("%s.md" % self.entry)
        wanted = REPO / "ops" / ("%s.md" % self.entry)
        if wanted.exists() and not self.force:
            # the repository is not under version control: never overwrite a specification
            # that is already there unless the caller asked for it
            return self.resolved_run_dir() / ("%s.md" % self.entry)
        return wanted

    def resolved_run_dir(self) -> Path:
        if self.run_dir is not None:
            return Path(self.run_dir)
        return AGENTS_DIR / "runs" / self.op

    def cases_path(self) -> Path:
        """The case module the tester writes.

        It stays in the run directory, like every other artifact of a run: nothing in
        `ops/` or `verification/` is touched until the caller publishes the run (see
        `publish`).
        """
        return self.resolved_run_dir() / "cases.py"

    def impl_path(self, iteration: int) -> Path:
        return self.resolved_run_dir() / ("iter-%03d-impl.py" % iteration)

    def report_path(self, iteration: int) -> Path:
        return self.resolved_run_dir() / ("iter-%03d-report.json" % iteration)
