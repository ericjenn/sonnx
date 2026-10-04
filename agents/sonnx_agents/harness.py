"""The two programs the agents call, and the small readers of the document they share.

`lint_spec.py` and `specloop.py` are the skill's own assets: the first decides what a
program can decide about a specification, the second compares an implementation with
ONNX Runtime over a case set.  Neither is reimplemented here — the agents run them as
subprocesses, so the multi-agent system cannot drift from the skill it implements.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from .config import ASSETS_DIR, Settings

LINT = ASSETS_DIR / "lint_spec.py"
SPECLOOP = ASSETS_DIR / "specloop.py"

_CONTENTS_LINK = re.compile(r"^\s*-\s+.*?\]\(#([A-Za-z0-9_.-]+)\)\s*$", re.M)
_TAG = re.compile(r"E_[A-Z0-9]+_[A-Z0-9_]+_\d{4}")
_ANCHOR = re.compile(r'<a id="([^"]+)">')


def crlf(text: str) -> str:
    """The end-of-line convention of every specification in `ops/`."""
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def read_spec(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")


def write_spec(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(crlf(text.rstrip("\n") + "\n").encode("utf-8"))


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def sections(spec_text: str) -> list[str]:
    """The section anchors of the document, from its Contents list."""
    return sorted(set(_CONTENTS_LINK.findall(spec_text)))


def tags(spec_text: str) -> list[str]:
    return sorted(set(_TAG.findall(spec_text)))


def missing_sections(document: str) -> list[str]:
    """The sections the Contents list announces that the document does not contain.

    Every anchor the Contents links to is declared in the body as `<a id="...">`, and the
    document is not a document if one of them is absent: the Contents is written first and
    the sections follow it, so an answer that stops in the middle of one keeps the entries
    of the sections that were never reached.  This is the linter's own finding (`Contents
    entry #float has no section anchor`), read here before the answer is written to disk
    rather than after it has spent a round of the loop.  Measured on 2026-10-04: the third
    answer of the **MaxPool** writer was cut off inside a `$...$` math span, its red spans
    were balanced (2 opened, 2 closed) so the span count alone accepted it, and it held the
    `real` section only while its Contents still announced `float`, `int` and `uint` — the
    amended document was 12 907 bytes where the document it replaced was 48 101, and the
    round was spent on it.  Every specification in `ops/` and every complete revision of
    the run satisfies this: the nine published documents have no missing anchor.
    """
    declared = set(_ANCHOR.findall(document))
    return [anchor for anchor in sections(document) if anchor not in declared]


def run_lint(settings: Settings, spec_path: Path) -> dict:
    """`lint_spec.py`: the mechanical rules.  Returns {ok, findings, stdout}."""
    proc = subprocess.run(
        [settings.python, str(LINT), str(spec_path), "--op", settings.op_upper],
        capture_output=True, text=True, timeout=300,
    )
    out = (proc.stdout + proc.stderr).strip()
    ok = proc.returncode == 0
    # The linter always prints its summary block (`sections : ...`, `red spans : ...`,
    # ...) before its verdict; only the lines after a non-clean verdict are findings.
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if any(line.startswith("no mechanical finding") for line in lines):
        findings = []
    else:
        findings = [line for line in lines if not LINT_SUMMARY.match(line)]
    return {"ok": ok, "findings": findings, "stdout": out, "returncode": proc.returncode}


LINT_SUMMARY = re.compile(r"^(?:\d+ finding\(s\):$|\S+: \d+ lines\b|[a-z][a-z ]* *:)")


def red_spans(document: str) -> tuple[int, int]:
    """(opened, closed) — how many red spans a document starts, and how many it ends.

    This is the pair the skill's linter counts, and the only one that differs when a
    document stops in the middle: a span is opened by the line that names its tag and
    closed by the line that says `[END]`.
    """
    closed = document.count("[END]</br></span>")
    return document.count("</br></span>") - closed, closed


def _write_report(report_path: Path, report: dict) -> None:
    """Put a report the harness never got to write where that run's report belongs.

    Every harness run of an iteration is recorded in the run's history by the path of its
    report, and the loop directory the run publishes is what a reader has to judge the run by.
    A run that timed out or died has no report of its own, so the path the history names would
    otherwise point at a file that is not there: measured on 2026-10-04, the **Atan** record
    lists `iter-001-report-4.json` for a round that was killed at 600 s, and the published
    `verification/atan/loop/` holds only the reports of the rounds that wrote one.  What is
    written says what happened, so the evidence of the missing round is the report itself.
    """
    try:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def run_specloop(settings: Settings, spec_path: Path, cases_path: Path, impl_path: Path,
                 report_path: Path) -> dict:
    """One iteration of the skill's verification loop.  Returns {ok, returncode, stdout, report}.

    A run that does not finish within the timeout is reported like a case module that raises:
    the case module is the artifact that decides how long the run takes, and an exact-arithmetic
    sweep over too many values, or a loop that never ends, is a defect of the tester's and not
    something the loop can wait out.  Measured on 2026-10-04: the case module of **Acos** spent
    fifteen minutes of CPU in one sweep and was still running; before this, the run had no
    timeout path at all, the exception escaped the node, and the process died with a traceback
    and no record of why.

    Either way the report it returns is also *written* to `report_path` (see `_write_report`):
    the path is the one the run's history records for that round, and a round whose recorded
    evidence is not on disk cannot be read afterwards.
    """
    try:
        proc = subprocess.run(
            [settings.python, str(SPECLOOP),
             "--spec", str(spec_path), "--op", settings.op,
             "--cases", str(cases_path), "--impl", str(impl_path),
             "--opset", str(settings.opset), "--entry", settings.entry,
             "--json", str(report_path)],
            capture_output=True, text=True, timeout=settings.harness_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        report = {"ok": False,
                  "error": "the harness did not finish within %d s: the case module is "
                           "too slow to run (an exact-arithmetic sweep over too many "
                           "values, an unbounded loop, a model per case)"
                           % settings.harness_timeout}
        _write_report(report_path, report)
        return {
            "ok": False, "returncode": -1,
            "stdout": "the harness did not finish within %d s%s"
                      % (settings.harness_timeout,
                         (" and was killed" if exc.output else "")),
            "report": report,
            "report_path": str(report_path),
        }
    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            report = None
    if report is None and proc.returncode != 0:
        # The harness died before it compared anything, and the reason is in the traceback it
        # printed.  An empty report is not a report: the tester is asked to repair the module
        # against what the report says, and against nothing it is asked to repair the module it
        # has just written.  Measured on 2026-10-04: the second case set of **Atan** ended in
        # `SyntaxError: invalid character '｜' (U+FF5C)` and the repair request carried `{}`.
        detail = (proc.stderr.strip() or proc.stdout.strip() or "(the run printed nothing)")
        report = {"ok": False,
                  "error": "the harness died before it wrote a report:\n%s" % detail[-4000:]}
        _write_report(report_path, report)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout + proc.stderr).strip(),
        "report": report,
        "report_path": str(report_path),
    }


def failing_report(report: dict | None) -> str:
    """The failing part of a specloop report, as text for the diagnoser."""
    if not report:
        return "(no report: the run did not produce one)"
    lines = []
    if report.get("error"):
        # a run that died before it compared anything: the reason is the whole of what
        # happened, and without this line the report reads as "None run, None passed"
        lines.append("ERROR %s" % str(report["error"])[:2000])
    cases = report.get("cases", {})
    lines.append("cases: %s run, %s passed, %s mismatched, %s open, %s NaN-payload" % (
        cases.get("total"), cases.get("passed"), len(cases.get("mismatched", [])),
        len(cases.get("open", [])), len(cases.get("nan_payload", []))))
    for item in cases.get("mismatched", [])[:12]:
        lines.append("MISMATCH %s" % json.dumps(item)[:1200])
    for sweep in report.get("sweeps", []):
        if sweep.get("error"):
            lines.append("SWEEP ERROR %s: %s" % (sweep.get("sweep"), sweep.get("error")))
        for item in sweep.get("failures", [])[:8]:
            lines.append("SWEEP MISMATCH %s: %s" % (item.get("who"), item.get("detail")))
    for item in cases.get("open", [])[:6]:
        lines.append("open (outside the domain) %s: %s" % (item.get("case"), item.get("reason")))
    return "\n".join(lines)


def decision_points(impl_text: str) -> str:
    """The blind implementer's `DECISION:` comments, as text for the diagnoser."""
    blocks = []
    current = []
    for line in impl_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# DECISION:"):
            if current:
                blocks.append("\n".join(current))
            current = [stripped]
        elif current and (stripped.startswith("#") or not stripped):
            current.append(stripped)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return "\n\n".join(blocks) if blocks else "(the implementation states no DECISION point)"
