#!/usr/bin/env python3
"""The scheduler, tested without a model and without ONNX Runtime.

Every agent is replaced by a canned answer and the two harness programs by canned
verdicts, so what is under test is exactly the part of the system a model must not decide:
which node runs after which, when the loop stops, and which guard stops it.  Run it with

    ~/Venvs/sonnx/bin/python agents/tests/test_graph_offline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonnx_agents import extract, graph, harness  # noqa: E402
from sonnx_agents.config import Settings  # noqa: E402

SPEC = """# Contents

- **Neg** operator for type [real](#real)

Based on ONNX documentation [Neg version 14](https://onnx.ai/onnx/operators/onnx__Neg.html).

Revision 2026-10-04: first issue of this specification.

<a id="real"></a>

# **Neg** (real)

## Function

<a id="E_NEG_REAL_FUNC_0010"></a>
<span style="background: red; color: white; font-size:0.7em;">[E_NEG_REAL_FUNC_0010]</br></span>

$$
C[i] = -A[i]
$$

<span style="background: red; color: white; font-size:0.7em;">[END]</br></span>
"""

IMPL = '''"""A reading of the specification."""
# DECISION: the document says nothing about where the result tensor lives.
def neg(a):
    return -a
'''

# the same module as a model returns it when the answer runs out of room: stopped in the
# middle of a string, which is how the parser sees a cut-off module
TRUNCATED_IMPL = '''"""A reading of the specification."""
def neg(a):
    """The module stopped here, mid-sentence
'''

CASES = '''"""The cases."""
OP = "Neg"
OPSET = 14
IR_VERSION = 10
COVERAGE = {"real": 3}


def cases():
    yield "one", [1.0]


def sweeps():
    return []
'''

# the document a model returns when its answer is cut off in the middle of a section
TRUNCATED_SPEC = SPEC.split("[END]")[0]

# The document a model returns when its amendment stops early.  The Contents is written
# before the sections it lists, so an answer that never reached them keeps their entries,
# and every red span it did reach is closed — which is why the span count alone accepts it.
# Scenario 27.
LOST_SECTION_SPEC = SPEC.replace(
    "- **Neg** operator for type [real](#real)\n",
    "- **Neg** operator for type [real](#real)\n"
    "- **Neg** operator for types [`float16`, `float`, `double`](#float)\n")

# An operator that is not element-wise: its node carries attributes and has a second output.
# Scenario 26 runs these through the real harness.
MAXPOOL_CASES = '''
"""A MaxPool case module: the node's attributes, and the node's two outputs."""
import numpy as np

OP = "MaxPool"
OPSET = 14
IR_VERSION = 10
NODE_OUTPUTS = ("Y", "Indices")
OUTPUT_TYPES = {"Indices": np.int64}
COVERAGE = {"float": 3}

X = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)


def cases():
    yield "2x2", [X], {"kernel_shape": [2, 2]}
    yield "2x2 stride 2", [X], {"kernel_shape": [2, 2], "strides": [2, 2]}
    yield "2x2 stride 2, ceil", [X], {"kernel_shape": [2, 2], "strides": [2, 2],
                                      "ceil_mode": 1}


def sweeps():
    return []
'''

MAXPOOL_IMPL = '''
"""MaxPool over the two spatial axes, for the harness test: kernel_shape and strides."""
import numpy as np


def maxpool(x, kernel_shape, strides=None, dilations=None, pads=None, ceil_mode=0,
            auto_pad="NOTSET", storage_order=0):
    X = np.asarray(x)
    n, c = X.shape[0], X.shape[1]
    k, s = list(kernel_shape), list(strides) if strides is not None else [1, 1]
    h = (X.shape[2] - k[0]) // s[0] + 1
    w = (X.shape[3] - k[1]) // s[1] + 1
    Y = np.empty((n, c, h, w), dtype=X.dtype)
    I = np.empty((n, c, h, w), dtype=np.int64)
    flat = np.arange(X.size).reshape(X.shape)
    for i in range(h):
        for j in range(w):
            win = X[:, :, i * s[0]:i * s[0] + k[0], j * s[1]:j * s[1] + k[1]]
            idx = flat[:, :, i * s[0]:i * s[0] + k[0], j * s[1]:j * s[1] + k[1]]
            win, idx = win.reshape(n, c, -1), idx.reshape(n, c, -1)
            best = win.argmax(axis=-1)          # the first maximum, as ONNX Runtime takes it
            Y[:, :, i, j] = np.take_along_axis(win, best[..., None], axis=-1)[..., 0]
            I[:, :, i, j] = np.take_along_axis(idx, best[..., None], axis=-1)[..., 0]
    return Y, I
'''

VERIFIER_OK = '{"ok": true, "findings": [], "notes": "checked"}'
VERIFIER_BAD = json.dumps({"ok": False, "notes": "one blocking finding", "findings": [
    {"severity": "blocking", "section": "real", "what": "no example",
     "why": "the skeleton requires one", "fix": "add Example 1"}]})


class Scenario:
    """What the fake agents answer, and what the fake harness reports."""

    def __init__(self, **kwargs):
        for key in ("verifier", "diagnosis", "fail_times", "change_spec", "uncovered",
                    "broken_cases", "truncate_answers", "blank_cases", "blank_impl",
                    "blank_cases_until_nothink", "blank_impl_until_nothink", "fragment_cases",
                    "truncated_impl", "change_cases", "timeout_cases", "not_python_cases",
                    "truncated_cases", "junk_cases", "raising_sweeps", "vacuous_cases",
                    "lose_section", "lose_section_after"):
            setattr(self, key, kwargs.get(key))
        self.writer_calls = 0
        self.verifier_calls = 0
        self.implementer_calls = 0
        self.tester_calls = 0
        self.diagnoser_calls = 0
        self.seen = []
        # ("implementer"|"tester", thinking) for every artifact call, so that a test can see
        # whether the deliberation was on.  The fake does not model `llm.ask`'s own retry
        # with twice the budget, which is why the counts below are of the calls the *node*
        # makes and not of the calls the model answers.
        self.thinking_seen = []
        # what the node asked each artifact call for, per agent
        self.budget_seen = []
        # every request the tester was given, so that a test can see what it was told
        self.tester_users = []

    # -- the model -------------------------------------------------------
    def ask(self, system, user, settings, max_tokens=None, thinking="auto"):
        if "specification writer" in system:
            self.writer_calls += 1
            self.seen.append("writer")
            if self.writer_calls <= (self.truncate_answers or 0):
                return "Here is the document:\n\n%s" % TRUNCATED_SPEC
            if self.lose_section and self.writer_calls <= self.lose_section:
                return "Here is the document:\n\n%s" % LOST_SECTION_SPEC
            if self.lose_section_after and self.writer_calls > self.lose_section_after:
                # an amendment that drops the sections it never got to
                return "Here is the document:\n\n%s" % LOST_SECTION_SPEC
            if self.writer_calls > 1 and self.change_spec:
                return SPEC.replace("Revision 2026-10-04", "Revision 2026-10-05")
            return SPEC
        if "guidelines-compliance verifier" in system:
            self.seen.append("verifier")
            self.verifier_calls += 1
            if self.verifier == "garbage" or (self.verifier == "garbage_once"
                                              and self.verifier_calls == 1):
                return "I have read the document and I have no verdict to give."
            # every mode but "bad" is a verifier that is satisfied with the document
            return VERIFIER_BAD if self.verifier == "bad" else VERIFIER_OK
        if "blind implementer" in system:
            self.implementer_calls += 1
            self.seen.append("implementer")
            self.thinking_seen.append(("implementer", thinking))
            self.budget_seen.append(("implementer", max_tokens))
            assert "# Contents" in user, "the implementer must be given the document"
            assert "ONNX Runtime" not in user and "specloop" not in user, \
                "the implementer must not be told about the reference or the harness"
            if self.truncated_impl and self.implementer_calls <= self.truncated_impl:
                # an answer that ran out of room in the middle of the module
                return "Here is the module:\n\n```python\n%s```\n" % TRUNCATED_IMPL
            if self.blank_impl and self.implementer_calls <= self.blank_impl:
                # an answer that stopped before the module began
                return ""
            if self.blank_impl_until_nothink and thinking != "off":
                # the deliberation eats the whole budget, on every call that has one
                return ""
            return "Here is the module:\n\n```python\n%s```\n" % IMPL
        if "test agent" in system:
            self.tester_calls += 1
            self.seen.append("tester")
            self.thinking_seen.append(("tester", thinking))
            self.budget_seen.append(("tester", max_tokens))
            self.tester_users.append(user)
            if self.blank_cases and self.tester_calls <= self.blank_cases:
                # an answer that stopped before the module began
                return ""
            if self.blank_cases_until_nothink and thinking != "off":
                # the deliberation eats the whole budget, on every call that has one
                return ""
            cases = CASES.replace('COVERAGE = {"real": 3}', "COVERAGE = {}") \
                if self.uncovered else CASES
            if self.not_python_cases and self.tester_calls <= self.not_python_cases:
                # the module, and then a line that is not Python: the text is not a module, so
                # it is refused and asked for again
                return "```python\n%s\nThat is the whole module, and it is complete.\n```" % cases
            if self.junk_cases and self.tester_calls <= self.junk_cases:
                # the module, and then the endpoint's own tool-call syntax: the envelope of the
                # answer, which is stripped rather than refused
                return "%s\n｜ DSML ｜ parameter>\n｜ DSML ｜ invoke>\n" % cases
            if self.truncated_cases and self.tester_calls <= self.truncated_cases:
                # an answer that ran out of room in the middle of the module: the parser sees
                # a string that was never closed
                return ('```python\n%s\n"""The module stopped here, mid-sentence\n'
                        % cases.split("def cases")[0])
            if self.change_cases and self.tester_calls > 1:
                # a case set that is revised, and still fails: new evidence, not a repeat
                cases += "\n# revision %d\n" % self.tester_calls
            if self.fragment_cases:
                # a model that explains itself: the piece it is drawing attention to first,
                # the module second
                return ("```python\nCOVERAGE = {\"real\": 3}\n```\n\nThe module:\n\n"
                        "```python\n%s```" % cases)
            return "```python\n%s```" % cases
        if "adjudicator" in system:
            self.diagnoser_calls += 1
            self.seen.append("diagnoser")
            return json.dumps({"verdict": self.diagnosis, "reason": "because",
                               "instruction": "do the thing", "confidence": "high"})
        raise AssertionError("unknown agent: %s" % system[:80])

    # -- the harness -----------------------------------------------------
    def lint(self, settings, spec_path):
        # the real linter's most important check, kept so that a cut-off document is not
        # silently accepted by the fake one
        opened, closed = harness.red_spans(Path(spec_path).read_text(encoding="utf-8"))
        if opened != closed:
            finding = "%d red span(s) opened, %d closed" % (opened, closed)
            return {"ok": False, "findings": [finding], "stdout": finding, "returncode": 1}
        missing = harness.missing_sections(Path(spec_path).read_text(encoding="utf-8"))
        if missing:
            # the real linter's other completeness check, and the one that named the defect
            # of the MaxPool run's last revision: `Contents entry #float has no section
            # anchor`. A document nobody can find a section of is not a document.
            finding = "Contents entry %s has no section anchor" % ", ".join(
                "#%s" % anchor for anchor in missing)
            return {"ok": False, "findings": [finding], "stdout": finding, "returncode": 1}
        return {"ok": True, "findings": [], "stdout": "no mechanical finding",
                "returncode": 0}

    def specloop(self, settings, spec_path, cases_path, impl_path, report_path):
        self.seen.append("specloop")
        failing = self.implementer_calls <= (self.fail_times or 0)
        if self.uncovered:
            failing = False
        report = {
            "ok": not failing,
            "cases": {"total": 3, "passed": 3 if not failing else 2,
                      "mismatched": ([] if not failing else
                                     [{"case": "one", "elements_differing": 1}]),
                      "open": [], "nan_payload": []},
            "sweeps": [],
        }
        if self.vacuous_cases and self.tester_calls <= self.vacuous_cases:
            # what the harness reports when every case is open: nothing was compared, the
            # exit status is still 0, and without a guard the run converges on a comparison
            # that never happened.  Seen on 2026-10-04 while preparing **MaxPool**, where a
            # case module declared the type of the second output wrongly and the runtime
            # refused all five cases with `Type Error: ... (Indices) ... does not match
            # expected type (tensor(int64))`.
            report = {
                "ok": True,
                "cases": {"total": 5, "passed": 0, "mismatched": [], "open": [
                    {"case": "2x2", "reason": "Fail: [ONNXRuntimeError] : 1 : FAIL : Type "
                                              "Error: Type (tensor(float)) of output arg "
                                              "(Indices) of node (node0) does not match "
                                              "expected type (tensor(int64))."}],
                          "nan_payload": []},
                "sweeps": [],
            }
        if self.timeout_cases and self.tester_calls <= self.timeout_cases:
            # what the harness returns when a run does not finish: the case module decides how
            # long a run takes, so a run that never ends is the tester's defect
            report = {"error": "the harness did not finish within 1800 s: the case module is "
                               "too slow to run (an exact-arithmetic sweep over too many "
                               "values, an unbounded loop, a model per case)"}
        if self.broken_cases and self.tester_calls <= self.broken_cases:
            report = {"error": "SyntaxError in the case module", "cases": {"total": 0,
                      "passed": 0, "mismatched": [{"case": "<module>",
                                                   "reason": "the implementation raised "
                                                   "SyntaxError: bad cases"}]}}
        if self.raising_sweeps and self.tester_calls <= self.raising_sweeps:
            # what the real harness reports when a sweep raises: the sweep's name, the
            # exception, its traceback, and no comparison at all — the run never reached the
            # document.  Seen on 2026-10-04: `sweep_special_values` of the **Atan** case set
            # ended in `decimal.InvalidOperation` on the infinities and the NaN.
            report["ok"] = False
            report["sweeps"] = [{
                "sweep": "sweep_special_values",
                "error": "InvalidOperation: [<class 'decimal.InvalidOperation'>]",
                "traceback": "Traceback (most recent call last):\n"
                             "  File \"cases.py\", line 1, in sweep_special_values\n"
                             "decimal.InvalidOperation: [<class 'decimal.InvalidOperation'>]",
                "failures": [],
            }]
        # the real harness writes its report where it is told to, and the files it leaves are
        # the evidence of what each round did: a fake that keeps the report in memory cannot
        # show one round's report being overwritten by the next
        Path(report_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return {"ok": bool(report.get("ok")), "returncode": 0 if report.get("ok") else 1,
                "stdout": "cases     : ...", "report": report, "report_path": str(report_path)}


def run(scenario: Scenario, **settings_kwargs) -> dict:
    """Run the graph once with the scenario installed."""
    workdir = Path(tempfile.mkdtemp(prefix="sonnx-agents-"))
    settings = Settings(op="Neg", run_dir=workdir, spec_path=workdir / "Neg.md", verbose=False,
                        max_iterations=settings_kwargs.pop("max_iterations", 5),
                        max_spec_rounds=settings_kwargs.pop("max_spec_rounds", 3),
                        max_case_repairs=settings_kwargs.pop("max_case_repairs", 2),
                        **settings_kwargs)
    original_ask, original_lint, original_loop = graph.ask, harness.run_lint, harness.run_specloop

    def fake_ask(_settings, system, user, max_tokens=None, thinking="auto"):
        return scenario.ask(system, user, settings, max_tokens=max_tokens, thinking=thinking)

    graph.ask = fake_ask
    harness.run_lint = scenario.lint
    harness.run_specloop = scenario.specloop
    try:
        state = graph.build_graph(settings).invoke(
            {"op": "Neg", "opset": 14, "spec_path": str(settings.resolved_spec_path()),
             "iteration": 0, "spec_round": 0, "history": [], "trace": [],
             "seen_amendments": []},
            {"recursion_limit": 200})
    finally:
        graph.ask, harness.run_lint, harness.run_specloop = \
            original_ask, original_lint, original_loop
    state["_dir"] = str(workdir)
    return state


def outcome(state: dict) -> str:
    return (state.get("result") or {}).get("outcome")


def check(name: str, got, want) -> None:
    assert got == want, "%s: got %r, wanted %r" % (name, got, want)
    print("  ok  %-28s %r" % (name, got))


def main() -> int:
    print("the scheduler, with canned agents")

    print("\n1. the document complies, the implementation passes")
    s = Scenario(verifier="ok", fail_times=0)
    state = run(s)
    check("outcome", outcome(state), "converged")
    check("iterations", state.get("iteration"), 1)
    check("rounds of the writer", s.writer_calls, 1)
    check("order", s.seen, ["writer", "verifier", "implementer", "tester", "specloop"])

    print("\n2. a blocking finding that is never resolved")
    # the writer keeps amending, the verifier keeps refusing: the round cap stops it
    s = Scenario(verifier="bad", change_spec=True)
    state = run(s, max_spec_rounds=2)
    check("outcome", outcome(state), "unresolved-review")
    check("writer rounds", s.writer_calls, 2)
    check("no implementation was written", s.implementer_calls, 0)

    print("\n2b. an amendment that changes nothing")
    s = Scenario(verifier="bad", change_spec=False)
    state = run(s, max_spec_rounds=5)
    check("outcome", outcome(state), "stalled")
    check("writer rounds", s.writer_calls, 2)

    print("\n3. the test fails, the adjudicator blames the code, the second reading passes")
    s = Scenario(verifier="ok", fail_times=1, diagnosis="code")
    state = run(s)
    check("outcome", outcome(state), "converged")
    check("implementations", s.implementer_calls, 2)
    check("tests", s.tester_calls, 1)
    # the second round reuses the case module: the document did not change, so the
    # second test is a run of the harness, not another call to the test agent
    check("order", [x for x in s.seen if x != "specloop"],
          ["writer", "verifier", "implementer", "tester", "diagnoser", "implementer"])
    check("harness runs", s.seen.count("specloop"), 2)

    print("\n4. the test fails, the adjudicator blames the document, the amendment converges")
    s = Scenario(verifier="ok", fail_times=1, diagnosis="spec", change_spec=True)
    state = run(s)
    check("outcome", outcome(state), "converged")
    check("writer rounds", s.writer_calls, 2)
    check("implementations", s.implementer_calls, 2)

    print("\n5. the document does not change and the same case still fails")
    s = Scenario(verifier="ok", fail_times=9, diagnosis="code", change_spec=False)
    state = run(s)
    check("outcome", outcome(state), "unchanged")
    check("it stops at the second failure", s.implementer_calls, 2)

    print("\n6. a section of the document with no case")
    s = Scenario(verifier="ok", fail_times=0, uncovered=True)
    state = run(s)
    check("outcome", outcome(state), "uncovered-section")
    check("uncovered", state.get("uncovered"), ["real"])

    print("\n7. the case module raises, the test agent repairs it")
    s = Scenario(verifier="ok", fail_times=0, broken_cases=1)
    state = run(s)
    check("outcome", outcome(state), "converged")
    check("case modules written", s.tester_calls, 2)

    print("\n8. the adjudicator refuses to attribute the failure")
    s = Scenario(verifier="ok", fail_times=9, diagnosis="stop", change_spec=False)
    state = run(s)
    check("outcome", outcome(state), "adjudicated-stop")

    print("\n9. the iteration cap")
    s = Scenario(verifier="ok", fail_times=9, diagnosis="code", change_spec=False)
    state = run(s, max_iterations=1)
    check("outcome", outcome(state), "iteration-cap")

    print("\n10. the writer's answer stops in the middle of the document")
    s = Scenario(verifier="ok", fail_times=0, truncate_answers=1)
    state = run(s)
    check("outcome", outcome(state), "converged")
    check("writer calls", s.writer_calls, 2)
    # the document left on disk is the complete one: the cut-off answer was never written
    check("red spans of the document",
          harness.red_spans(Path(state["spec_path"]).read_text(encoding="utf-8")), (1, 1))

    # when no answer is ever complete the loop stops as soon as it stops making progress:
    # 3 attempts in the first round, 3 in the second, and then nothing is left to try
    s = Scenario(verifier="ok", fail_times=0, truncate_answers=9)
    state = run(s)
    check("writer calls when it never completes", s.writer_calls, 6)
    check("outcome when it never completes", outcome(state), "stalled")

    print("\n11. the verifier returns no verdict at all")
    s = Scenario(verifier="garbage", fail_times=0)
    state = run(s)
    check("outcome", outcome(state), "verifier-failed")
    check("the writer was not sent back", s.writer_calls, 1)
    check("no implementation was written", s.implementer_calls, 0)
    check("attempts at the review", s.verifier_calls, 2)

    s = Scenario(verifier="garbage_once", fail_times=0)
    state = run(s)
    check("outcome after a second attempt", outcome(state), "converged")

    print("\n12. the family verdicts handed to the verifier, decided from the schema")
    # the one factual question the verifier kept answering from memory of the operator:
    # whether a family applies. It is computed here, so the review cannot invent one.
    neg = graph.type_constraints_text("Neg", 14)
    check("Neg has no uint family", "`uint`: **absent**" in neg, True)
    check("Neg has the int family", "`int`: applies" in neg, True)
    check("Neg names bfloat16 as outside the profile", "outside the SONNX profile" in neg, True)
    check("Add has the uint family", "`uint`: applies" in graph.type_constraints_text("Add", 14),
          True)
    check("Sin has no int family", "`int`: **absent**" in graph.type_constraints_text("Sin", 14),
          True)
    lines = graph.family_verdicts("Neg", 14)
    check("one line per family", len(lines), 4)
    check("no empty list of types", any("(``)" in line for line in lines), False)

    print("\n13. a COVERAGE dictionary the module computes for itself")
    # the reader must see the dictionary as the module leaves it: a case set that counts its
    # own cases per section looks like `{"real": 0}` in the source and is not uncovered
    workdir = Path(tempfile.mkdtemp(prefix="sonnx-cases-"))
    computed = workdir / "cases.py"
    computed.write_text(
        'COVERAGE = {"real": 0, "int": 0}\n'
        'for _section in ["real", "real", "int"]:\n'
        '    COVERAGE[_section] += 1\n', encoding="utf-8")
    literal = workdir / "literal.py"
    literal.write_text('COVERAGE = {"real": 2, "int": 1}\n', encoding="utf-8")
    broken = workdir / "broken.py"
    broken.write_text('import nosuchmodule\nCOVERAGE = {"real": 9}\n', encoding="utf-8")
    settings = Settings(op="Neg", run_dir=workdir, spec_path=workdir / "Neg.md", verbose=False)
    check("computed coverage", graph.coverage_of(computed, settings), {"real": 2, "int": 1})
    check("literal coverage", graph.coverage_of(literal, settings), {"real": 2, "int": 1})
    # a module that cannot be imported falls back to its literal, and if it has none the run
    # reports the sections as uncovered rather than passing vacuously
    check("a module that will not import", graph.coverage_of(broken, settings), {"real": 9})

    print("\n14. the test agent's answer is not a case module")
    # the analogue of a cut-off document: an answer that stopped before the module began must
    # not be written over the case module, since an empty module imports cleanly, declares no
    # coverage, and is then read as a document whose every section has no case
    s = Scenario(verifier="ok", fail_times=0, blank_cases=1)
    state = run(s)
    check("outcome after a blank answer", outcome(state), "converged")
    check("attempts at the case set", s.tester_calls, 2)
    check("what is on disk is a case module",
          "def cases(" in Path(state["cases_path"]).read_text(encoding="utf-8"), True)

    s = Scenario(verifier="ok", fail_times=0, blank_cases=9)
    state = run(s)
    check("outcome when no answer is a module", outcome(state), "cases-unusable")
    check("attempts when no answer is a module", s.tester_calls, 4)
    check("not reported as a section with no case", state.get("uncovered") or [], [])

    print("\n15. a harness run that dies leaves no usable report behind")
    # a run that dies before writing its report leaves the report of the run before it on
    # disk; reading that one makes a case module that never ran look like one that passed
    workdir = Path(tempfile.mkdtemp(prefix="sonnx-report-"))
    report = workdir / "iter-001-report.json"
    report.write_text(json.dumps({"ok": True, "sweeps": [],
                                  "cases": {"total": 3, "passed": 3, "mismatched": [],
                                            "open": [], "nan_payload": []}}))
    settings = Settings(op="Neg", run_dir=workdir, spec_path=workdir / "Neg.md", verbose=False)
    seen = {}

    def fake_loop(_settings, spec_path, cases_path, impl_path, report_path):
        seen["report_there"] = Path(report_path).exists()
        return {"ok": False, "returncode": 1, "stdout": "Traceback (most recent call last):",
                "report": None, "report_path": str(report_path)}

    original_loop = harness.run_specloop
    harness.run_specloop = fake_loop
    try:
        result = graph._run_harness(settings, workdir / "Neg.md", workdir / "cases.py",
                                    workdir / "iter-001-impl.py", report)
    finally:
        harness.run_specloop = original_loop
    check("the report of the run before was removed first", seen["report_there"], False)
    check("a run with no report is a broken case module", graph._broken(result["report"]), True)

    print("\n16. the blind implementer's answer is not an implementation")
    # the implementer is the one agent whose answer is never read before it is used: an empty
    # module is written to disk, imports, defines nothing, and is then reported as the
    # document's failure. Seen on 2026-10-04: a one-character implementation of **Asin**.
    s = Scenario(verifier="ok", fail_times=0, blank_impl=1)
    state = run(s)
    check("outcome after a blank answer", outcome(state), "converged")
    check("attempts at the implementation", s.implementer_calls, 2)
    check("what is on disk is an implementation",
          "def neg(" in Path(state["impl_path"]).read_text(encoding="utf-8"), True)

    s = Scenario(verifier="ok", fail_times=0, blank_impl=9)
    state = run(s)
    check("outcome when no answer is an implementation", outcome(state), "impl-unusable")
    check("attempts when no answer is an implementation", s.implementer_calls, 4)
    check("the harness was never run", "specloop" in s.seen, False)
    check("nothing was written", state.get("impl_path"), "")

    # what the node refuses and what it accepts, on the answer alone
    check("an empty answer", graph._impl_module_problem("", "neg"), "the answer is empty")
    check("whitespace is still empty", graph._impl_module_problem("\n \n", "neg"),
          "the answer is empty")
    check("an answer that is not Python",
          graph._impl_module_problem("Here is the module you asked for.\n", "neg")
          .startswith("the answer is not Python"), True)
    check("an answer that stops in the middle",
          graph._impl_module_problem(TRUNCATED_IMPL, "neg")
          .startswith("the answer stops in the middle of the module"), True)
    check("a module without the entry point",
          graph._impl_module_problem("def plus(a):\n    return a\n", "neg"),
          "the module defines no `neg`")
    check("a module with it", graph._impl_module_problem(IMPL, "neg"), None)

    print("\n17. an answer that is empty while the model deliberates")
    # deliberation can eat the whole budget on every call that has one, in which case raising
    # the budget does not help and the last attempt is made without it. Seen on 2026-10-04:
    # the test agent answered nothing four times in a row for the case set of **Acos**, and
    # the run stopped with no case module at all.
    s = Scenario(verifier="ok", fail_times=0, blank_cases_until_nothink=True)
    state = run(s)
    check("the case set was written without deliberation", outcome(state), "converged")
    check("the first answer was empty", s.tester_calls, 2)
    check("the last call had no deliberation",
          [t for t in s.thinking_seen if t[0] == "tester"][-1], ("tester", "off"))
    check("what is on disk is a case module",
          "def cases(" in Path(state["cases_path"]).read_text(encoding="utf-8"), True)

    s = Scenario(verifier="ok", fail_times=0, blank_impl_until_nothink=True)
    state = run(s)
    check("the implementation was written without deliberation", outcome(state), "converged")
    check("the first answer was empty", s.implementer_calls, 2)
    check("the last call had no deliberation",
          [t for t in s.thinking_seen if t[0] == "implementer"][-1], ("implementer", "off"))
    check("what is on disk is an implementation",
          "def neg(" in Path(state["impl_path"]).read_text(encoding="utf-8"), True)

    print("\n18. the module is not the first block of the answer")
    # a model that explains itself writes more than one block, and the first may be a *piece*
    # of the module — the dictionary it is drawing attention to — in which case reading the
    # first block reports the piece's own gap as the module's. Seen on 2026-10-04: the test
    # agent's answer for **Asin** was read as a module with no `cases()`, and, next time, as
    # one with no `COVERAGE`.
    answer = ("```python\nCOVERAGE = {\"real\": 3}\n```\n\nAnd the module:\n\n"
              "```python\n%s```" % CASES)
    check("the block that holds the module",
          extract.extract_python(answer, ("def cases", "COVERAGE")), CASES)
    check("a block that holds only a piece is not the module",
          "def cases(" in extract.extract_python(answer), False)

    s = Scenario(verifier="ok", fail_times=0, fragment_cases=True)
    state = run(s)
    check("a fragmented answer still converges", outcome(state), "converged")
    check("one call was enough", s.tester_calls, 1)
    check("the whole module is on disk", Path(state["cases_path"]).read_text(encoding="utf-8"),
          CASES)

    # an answer that really does lack the module is kept beside the artifact, so that the
    # next reader can tell a bad answer from a bad reading of a good one
    s = Scenario(verifier="ok", fail_times=0, blank_impl=9)
    state = run(s)
    kept = sorted(Path(state["_dir"]).glob("*-rejected-*.txt"))
    check("the refused answers were kept", [p.name for p in kept],
          ["iter-001-impl-rejected-1.txt", "iter-001-impl-rejected-2.txt"])
    check("and they hold the reason", kept[0].read_text(encoding="utf-8").splitlines()[0],
          "# the answer is empty")

    print("\n19. the blind implementer's answer stops in the middle of the module")
    # the writer's cut-off document in another guise, and it is repaired the same way: the
    # module is asked for again with twice the room.  Measured on 2026-10-04: both answers of
    # the implementer for **Atan** were truncated — 317 and 1857 bytes — and the run stopped
    # with no implementation at all while the document was sound.
    s = Scenario(verifier="ok", fail_times=0, truncated_impl=1)
    state = run(s)
    check("outcome after a cut-off answer", outcome(state), "converged")
    check("attempts at the implementation", s.implementer_calls, 2)
    check("the second answer was given twice the room",
          [b for b in s.budget_seen if b[0] == "implementer"], [("implementer", 16000),
                                                               ("implementer", 32000)])
    check("what is on disk is the complete module",
          Path(state["impl_path"]).read_text(encoding="utf-8"), IMPL)

    s = Scenario(verifier="ok", fail_times=0, truncated_impl=9)
    state = run(s)
    check("outcome when every answer is cut off", outcome(state), "impl-unusable")
    check("attempts when every answer is cut off", s.implementer_calls, 2)
    check("nothing was written", state.get("impl_path"), "")

    print("\n20. the case set fails, the adjudicator blames it, and the tester is told why")
    # the tester was sent back to its case set with neither the report nor the adjudicator's
    # instruction, so it answered with the same case set again and the loop stopped with a
    # reason that named the document.  Seen on 2026-10-04 in the **Asin** run: the second case
    # set (23 253 bytes, the first was 18 575) failed on the same two cases.
    s = Scenario(verifier="ok", fail_times=9, diagnosis="test", change_cases=True)
    state = run(s, max_case_rounds=1)
    check("outcome", outcome(state), "cases-stuck")
    check("the tester rewrote the case set once", s.tester_calls, 2)
    check("it was told what the harness reported", "MISMATCH" in s.tester_users[-1], True)
    check("and what the adjudicator said", "do the thing" in s.tester_users[-1], True)
    check("and given the module as it stands", "def cases(" in s.tester_users[-1], True)

    # the budget is a count of revisions: two by default, and the stop says so
    s = Scenario(verifier="ok", fail_times=9, diagnosis="test", change_cases=True)
    state = run(s)
    check("outcome with the default budget", outcome(state), "cases-stuck")
    check("the tester rewrote the case set twice", s.tester_calls, 3)

    # a revision that is the same module is not new evidence, and the round stops at once
    s = Scenario(verifier="ok", fail_times=9, diagnosis="test", change_cases=False)
    state = run(s)
    check("outcome when the case set comes back unchanged", outcome(state), "unchanged")
    check("the tester was asked once", s.tester_calls, 2)

    # every harness run keeps its report: the report of the round before is what the repair
    # is for, and overwriting it leaves two rounds in one file
    s = Scenario(verifier="ok", fail_times=9, diagnosis="test", change_cases=True)
    state = run(s, max_case_rounds=1)
    # the first run keeps the name a single run always had; the runs after it are numbered
    reports = sorted(p.name for p in Path(state["_dir"]).glob("iter-001-report*.json"))
    check("each harness run kept its report", reports,
          ["iter-001-report-2.json", "iter-001-report.json"])

    print("\n21. the harness does not finish")
    # a run that does not finish is reported like a case module that raises: the case module
    # decides how long a run takes, and a sweep that never ends is the tester's defect.  Seen
    # on 2026-10-04: the case module of **Acos** spent fifteen minutes of CPU in one sweep and
    # was still running, and the run had no timeout path at all — the exception escaped the
    # node and the process died with a traceback and no record of why.
    s = Scenario(verifier="ok", fail_times=0, timeout_cases=1)
    state = run(s)
    check("outcome after one run that timed out", outcome(state), "converged")
    check("the tester was asked again", s.tester_calls, 2)
    check("and told why", "did not finish within" in s.tester_users[-1], True)

    s = Scenario(verifier="ok", fail_times=0, timeout_cases=9, diagnosis="test",
                 change_cases=True)
    state = run(s, max_case_rounds=1)
    check("outcome when no run ever finishes", outcome(state), "cases-stuck")
    check("the tester was told about the timeout",
          "did not finish within" in s.tester_users[-1], True)

    # and the reason reaches the adjudicator: a report with an error and no cases reads as
    # "None run, None passed, 0 mismatched" without it
    check("the reason reaches the report text",
          harness.failing_report({"error": "the harness did not finish"}).splitlines()[0],
          "ERROR the harness did not finish")

    print("\n22. the tester's answer is not a case module")
    # the endpoint's chat template leaks its own tool-call syntax into some answers, as lines
    # of `｜DSML｜ parameter>` after the last statement of the module.  It is the envelope of
    # the answer and not the answer, so it is stripped: measured on 2026-10-04, the second case
    # set of **Atan** and the first of **Asin** ended this way, with the module complete and
    # correct above it.
    s = Scenario(verifier="ok", fail_times=0, junk_cases=9)
    state = run(s)
    check("outcome with the tool syntax stripped", outcome(state), "converged")
    check("one call was enough", s.tester_calls, 1)
    check("and the module is the module", Path(state["cases_path"]).read_text(encoding="utf-8"),
          CASES)

    # the answer is parsed before it is written, and not only searched for the two names it
    # must declare: text that cannot be parsed is a module that cannot be imported, and writing
    # it over the case set that is on disk discards the one whose failures were just sent back
    s = Scenario(verifier="ok", fail_times=0, not_python_cases=1)
    state = run(s)
    check("outcome after a module that is not Python", outcome(state), "converged")
    check("the tester was asked again", s.tester_calls, 2)
    check("no extra room was offered",
          [b for b in s.budget_seen if b[0] == "tester"], [("tester", 16000), ("tester", 16000)])
    check("what is on disk is the module that parses",
          Path(state["cases_path"]).read_text(encoding="utf-8"), CASES)
    kept = sorted(Path(state["_dir"]).glob("cases-rejected-*.txt"))
    check("the refused answer was kept", [p.name for p in kept], ["cases-rejected-1.txt"])
    check("and it holds the reason", kept[0].read_text(encoding="utf-8").splitlines()[0],
          "# the answer is not Python: invalid syntax (<unknown>, line 15)")

    # an answer that stopped in the middle is repaired like the writer's cut-off document: the
    # module is asked for again with twice the room
    s = Scenario(verifier="ok", fail_times=0, truncated_cases=1)
    state = run(s)
    check("outcome after a cut-off answer", outcome(state), "converged")
    check("the second answer was given twice the room",
          [b for b in s.budget_seen if b[0] == "tester"], [("tester", 16000), ("tester", 32000)])
    check("and told to finish the module", "cut off" in s.tester_users[-1], True)

    s = Scenario(verifier="ok", fail_times=0, not_python_cases=9)
    state = run(s)
    check("outcome when every answer is not Python", outcome(state), "cases-unusable")
    check("attempts when every answer is not Python", s.tester_calls, 2)
    check("no case module was written", Path(state["cases_path"]).exists(), False)
    check("both refused answers were kept",
          sorted(p.name for p in Path(state["_dir"]).glob("cases-rejected-*.txt")),
          ["cases-rejected-1.txt", "cases-rejected-2.txt"])

    print("\n23. the harness dies before it writes a report")
    # a run that dies on the import leaves no report, and an empty report is not a report: the
    # tester is asked to repair the module against what the report says, and `{}` says nothing.
    # The report the harness synthesizes is written where that round's report belongs, because
    # the run's history records the round by that path and the loop directory the run publishes
    # is what a reader judges it by: seen on 2026-10-04, the **Atan** record names
    # `iter-001-report-4.json` for a round killed at 600 s and the published
    # `verification/atan/loop/` did not contain it.  This runs the real harness twice — on a real
    # module with one character too many, and on a module that sleeps past the limit.
    workdir = Path(tempfile.mkdtemp(prefix="sonnx-agents-"))
    (workdir / "Neg.md").write_text(SPEC, encoding="utf-8")
    (workdir / "iter-001-impl.py").write_text(IMPL, encoding="utf-8")
    (workdir / "cases.py").write_text(CASES + "\n｜\n", encoding="utf-8")
    report_path = workdir / "report.json"
    settings = Settings(op="Neg", run_dir=workdir, spec_path=workdir / "Neg.md", verbose=False)
    result = harness.run_specloop(settings, workdir / "Neg.md", workdir / "cases.py",
                                  workdir / "iter-001-impl.py", report_path)
    check("the run is not a pass", result["ok"], False)
    check("the round's report is on disk", report_path.exists(), True)
    check("and it is the report the run returned",
          json.loads(report_path.read_text(encoding="utf-8")), result["report"])
    check("the run carries the traceback",
          "SyntaxError" in (result["report"] or {}).get("error", ""), True)
    check("and the repair text shows it",
          harness.failing_report(result["report"]).splitlines()[0].startswith("ERROR "), True)
    check("and the error reaches the tester",
          graph._broken_reason(result).startswith("the harness died"), True)

    # the other report the harness writes itself: the one for a run it had to kill.  The case
    # module decides how long a run takes, so a module that sleeps at import is the smallest
    # module that does not finish.
    (workdir / "cases.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    killed_path = workdir / "iter-001-report-4.json"
    settings = Settings(op="Neg", run_dir=workdir, spec_path=workdir / "Neg.md", verbose=False,
                        harness_timeout=2)
    started = time.time()
    result = harness.run_specloop(settings, workdir / "Neg.md", workdir / "cases.py",
                                  workdir / "iter-001-impl.py", killed_path)
    check("the run was stopped at its limit", time.time() - started < 20, True)
    check("a killed run is not a pass", result["ok"], False)
    check("and its report is on disk too", killed_path.exists(), True)
    check("and it is the report the run returned",
          json.loads(killed_path.read_text(encoding="utf-8")), result["report"])
    check("and says the run was killed",
          "did not finish within 2 s" in (result["report"] or {}).get("error", ""), True)

    print("\n24. a sweep raises")
    # A sweep that raises is a failing sweep, and the run stops on it before it reaches the
    # document: the sweep's traceback *is* the failure, and what the tester is asked to repair
    # against must be that and not the case counts.  A report carrying a raising sweep and
    # nothing else reads as "3 run, 2 passed, 1 mismatched" — a comparison the run never made.
    # Seen on 2026-10-04: `sweep_special_values` of the **Atan** case set ended in
    # `decimal.InvalidOperation`, on the infinities and the NaN of each type.
    s = Scenario(verifier="ok", fail_times=0, raising_sweeps=1)
    state = run(s)
    check("outcome with the sweep repaired", outcome(state), "converged")
    check("the tester was asked to repair it", s.tester_calls, 2)
    check("and told which sweep raised",
          "sweep_special_values" in s.tester_users[-1], True)
    check("and given its traceback",
          "decimal.InvalidOperation" in s.tester_users[-1], True)

    report = {"ok": False,
              "cases": {"total": 3, "passed": 2, "mismatched": [], "open": []},
              "sweeps": [{"sweep": "sweep_special_values",
                          "error": "InvalidOperation: [<class 'decimal.InvalidOperation'>]",
                          "traceback": "Traceback (most recent call last):\n"
                                       "decimal.InvalidOperation"}]}
    check("the sweep error reaches the report text",
          [line for line in harness.failing_report(report).splitlines()
           if line.startswith("SWEEP ERROR")],
          ["SWEEP ERROR sweep_special_values: InvalidOperation: "
           "[<class 'decimal.InvalidOperation'>]"])
    check("and the repair text is the sweep's, with the traceback",
          graph._broken_reason({"report": report}),
          "sweep_special_values: InvalidOperation: [<class 'decimal.InvalidOperation'>]"
          "\nTraceback (most recent call last):\ndecimal.InvalidOperation")

    print("\n25. every case is open")
    # A case set whose every case the runtime refuses passes the harness: nothing
    # mismatched, the exit status is 0, and the run converges on a comparison that never
    # happened.  The case set is what the loop sends back, with the runtime's own words.
    s = Scenario(verifier="ok", fail_times=0, vacuous_cases=1)
    state = run(s)
    check("outcome when the second case set compares something", outcome(state), "converged")
    check("the tester was asked again", s.tester_calls, 2)
    check("and told the run compared nothing",
          "not one case was compared" in s.tester_users[-1], True)
    check("and given the runtime's refusal",
          "does not match expected type (tensor(int64))" in s.tester_users[-1], True)
    check("and the repair is asked for as a case set that compared nothing, not as one that "
          "raised", "The case set compared nothing" in s.tester_users[-1], True)

    s = Scenario(verifier="ok", fail_times=0, vacuous_cases=9, diagnosis="test",
                 change_cases=True)
    state = run(s, max_case_rounds=1)
    check("outcome when nothing is ever compared", outcome(state), "cases-stuck")
    check("no run ever passed", outcome(state) == "converged", False)

    # and the guard is the driver's, not the harness's: the report of an open case set is a
    # pass by the harness's own account, which is exactly why the driver may not trust it
    open_report = {"ok": True,
                   "cases": {"total": 5, "passed": 0, "mismatched": [],
                             "open": [{"case": "2x2", "reason": "Type Error: (Indices)"}]},
                   "sweeps": []}
    check("an all-open report is a broken case module", graph._broken(open_report), True)
    check("a report with one case compared is not", graph._broken(
        {"ok": True, "cases": {"total": 5, "passed": 1, "mismatched": [], "open": [
            {"case": "a", "reason": "r"}]}, "sweeps": []}), False)
    check("an empty case set is not evidence either", graph._broken(
        {"ok": True, "cases": {"total": 0, "passed": 0, "mismatched": [], "open": []},
         "sweeps": []}), True)
    check("and the reason names the refusal",
          graph._broken_reason({"report": open_report}).splitlines()[0],
          "not one case was compared: 5 case(s), 0 passed, 1 open. The runtime refused the "
          "model of every case, or the case set has no case in it, so this run is evidence "
          "of nothing. The refusals are the reason:")

    print("\n26. an operator with attributes and two outputs")
    # The harness contract for an operator that is not element-wise, on the real harness: the
    # node's attributes come from the case, the implementation is called with them as keyword
    # arguments, the node's outputs are the module's NODE_OUTPUTS, and every one of them is
    # compared.  Without this the case module of such an operator cannot be written at all —
    # `MaxPool` has a required attribute and a second output — and its cases would all be
    # refused by the runtime, which is the vacuous pass of scenario 25.
    workdir = Path(tempfile.mkdtemp(prefix="sonnx-agents-"))
    (workdir / "cases.py").write_text(MAXPOOL_CASES, encoding="utf-8")
    (workdir / "impl.py").write_text(MAXPOOL_IMPL, encoding="utf-8")
    report_path = workdir / "report.json"
    settings = Settings(op="MaxPool", opset=14, run_dir=workdir, verbose=False)
    result = harness.run_specloop(settings, workdir / "MaxPool.md", workdir / "cases.py",
                                  workdir / "impl.py", report_path)
    cases = (result["report"] or {}).get("cases") or {}
    check("every case passed against ONNX Runtime",
          (cases.get("total"), cases.get("passed"), len(cases.get("open") or [])), (3, 3, 0))
    check("both outputs were compared",
          (result["report"] or {}).get("outputs"), ["Y", "Indices"])

    # a wrong second output is a mismatch of that output, and of nothing else: the case is
    # the call, and the report says which of its outputs differs
    (workdir / "impl.py").write_text(MAXPOOL_IMPL.replace("return Y, I", "return Y, I + 1"),
                                     encoding="utf-8")
    result = harness.run_specloop(settings, workdir / "MaxPool.md", workdir / "cases.py",
                                  workdir / "impl.py", report_path)
    mismatched = (result["report"] or {}).get("cases", {}).get("mismatched") or []
    check("the run is not a pass", result["ok"], False)
    check("every case mismatched", [m["case"] for m in mismatched],
          ["2x2", "2x2 stride 2", "2x2 stride 2, ceil"])
    check("and the mismatch names the output that differs",
          sorted(mismatched[0].get("outputs_differing") or {}), ["Indices"])
    check("blaming no other output", mismatched[0]["elements_differing"], 9)
    check("with the attributes of the case",
          mismatched[0].get("attrs"), {"kernel_shape": [2, 2]})

    print("\n27. the writer's answer keeps a Contents it does not fill")
    # The second way an answer stops in the middle, and the one the span count cannot see:
    # it was cut off between the red spans, so every span it reached is closed and the count
    # accepts it, while the sections its own Contents announces were never reached.  Measured
    # on 2026-10-04 in the **MaxPool** run: the writer's third answer was cut inside a `$...$`
    # and held one section of the four its Contents announced, 12 907 bytes against the
    # 48 101 of the revision it replaced, and it took the run's last spec round.
    check("a document announces nothing it lacks", harness.missing_sections(SPEC), [])
    check("the section that was lost is named",
          harness.missing_sections(LOST_SECTION_SPEC), ["float"])
    check("its red spans are balanced, so the count alone accepts it",
          harness.red_spans(LOST_SECTION_SPEC), (1, 1))
    check("so it is refused as no document", graph._not_a_document(LOST_SECTION_SPEC),
          "the answer keeps the Contents of a longer document but not its sections: #float")
    check("while a whole document is a document", graph._not_a_document(SPEC), "")

    # an amendment that drops its sections is asked for again, not written: the revision on
    # disk keeps the sections it has, and the loop stops on an amendment that changed nothing
    s = Scenario(verifier="bad", lose_section_after=1)
    state = run(s, max_spec_rounds=3)
    check("outcome", outcome(state), "stalled")
    check("the writer was asked again", s.writer_calls, 4)
    check("and the document on disk still has its sections",
          harness.missing_sections(Path(state["spec_path"]).read_text(encoding="utf-8")), [])

    # and when the first answer is the one that loses a section, the retry is what is written
    s = Scenario(verifier="ok", fail_times=0, lose_section=1)
    state = run(s)
    check("outcome", outcome(state), "converged")
    check("writer calls", s.writer_calls, 2)
    check("the document on disk is the whole one",
          harness.missing_sections(Path(state["spec_path"]).read_text(encoding="utf-8")), [])

    print("\nall scenarios hold")
    return 0
if __name__ == "__main__":
    sys.exit(main())
