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
