"""The state the graph carries between the agents."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class SpecState(TypedDict, total=False):
    # what is being specified
    op: str
    opset: int
    onnx_doc: str
    guidelines: str

    # the artifact under construction
    spec: str                 # the document, LF, as text
    spec_path: str
    spec_md5: str

    # the verifier's verdict on it
    lint: dict                # {ok, findings, stdout}
    review: dict              # {ok, findings:[{severity, section, what, why}], notes}
    spec_ok: bool
    checked_md5: str          # the document the last compliance check looked at
    no_progress: bool         # it failed, and it is the same document as last time
    review_failed: bool       # the verifier returned no verdict at all

    # the generated Lean formal specification and its independent checks
    lean_path: str
    lean_trace_path: str
    lean_source_md5: str
    lean_ok: bool               # compiled, trace-valid, and semantically complete
    lean_partial: bool          # valid artifact with explicit partial/deferred coverage
    lean_error: str
    lean_round: int
    lean_coverage: dict
    lean_report: dict
    lean_artifacts: Annotated[list[str], operator.add]
    # the executable Lean implementation against the repository's validation corpus
    lean_conformance_ok: bool
    lean_conformance_status: str
    lean_conformance_path: str
    lean_conformance_report: dict
    lean_conformance_error: str
    lean_conformance_generated: bool
    lean_conformance_runner: str
    lean_conformance_driver: str
    lean_conformance_corpus: str
    lean_conformance_manifest: str
    lean_conformance_manifest_data: dict
    lean_conformance_corpus_source: str
    lean_conformance_attempts: Annotated[list[str], operator.add]
    lean_conformance_artifacts: Annotated[list[str], operator.add]

    # the C generated from the Lean descriptor, and the agent's test program
    cgen_status: str           # "unsupported" | "generated" | "passed" | "failed" | "error"
    cgen_ok: bool
    cgen_error: str
    cgen_source: str
    cgen_header: str
    cgen_test: str
    cgen_report: dict
    cgen_attempts: Annotated[list[str], operator.add]
    cgen_artifacts: Annotated[list[str], operator.add]

    # the implementation the blind agent wrote
    impl_path: str
    impl_text: str
    decisions: str
    impl_unusable: str        # the implementer never answered with a module

    # the tester's verdict
    cases_path: str
    report: dict
    test_ok: bool
    coverage: dict
    uncovered: list[str]
    rebuild_cases: bool
    cases_unusable: str       # the tester never answered with a case module

    # the two failure signatures the "unchanged" guard compares
    fail_signature: str
    previous_failure: str
    case_rounds: int          # times the tester was sent back to its case set, this iteration
    report_runs: int          # harness runs made for this iteration, so each keeps its report
    reports: Annotated[list[str], operator.add]   # every report *this* run wrote, in order

    # the diagnoser's verdict, and the loop's bookkeeping
    verdict: str              # "spec" | "code" | "test" | "pass" | "stop"
    diagnosis: str
    instruction: str
    iteration: int            # implement -> test rounds
    spec_round: int           # writer -> verifier rounds
    seen_amendments: Annotated[list[str], operator.add]
    history: Annotated[list[dict], operator.add]
    trace: Annotated[list[str], operator.add]
    stopped: str
    result: dict[str, Any]
