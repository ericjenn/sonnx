"""The five agents and the graph that schedules them.

    write_spec ──▶ check_spec ──▶ implement ──▶ test ──▶ (pass) finish
         ▲              │                          │
         │              └── (blocking finding) ────┘
         │                                         ▼
         └────────── (spec) ──── diagnose ◀── (fail)
                                (code) ──▶ implement
                                (test) ──▶ test
                                (stop) ──▶ finish

Every arrow is a separate model call with its own system prompt, and every artifact is a
file: the document, the implementation, the case module and the harness's JSON report. The
nodes decide when the loop stops; the models decide only the content of the artifacts.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from . import extract, harness
from .config import (
    DEFAULT_PYTHON,
    GUIDELINES,
    ONNX_DOC_URL,
    PROMPTS_DIR,
    REFERENCES_DIR,
    REFERENCE_SPEC,
    Settings,
    TEMPLATE,
    WORKED_CASES,
)
from .llm import ask
from .state import SpecState


def prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def log(settings: Settings, message: str) -> None:
    if settings.verbose:
        print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), message), flush=True)


def clip(text: str, limit: int, what: str = "content") -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[... %s truncated, %d characters omitted ...]" % (
        what, len(text) - limit)


# ---------------------------------------------------------------------------
# the contexts each agent is given
# ---------------------------------------------------------------------------


def schema_text(op: str, opset: int) -> str:
    """The operator's schema as the installed `onnx` package states it, for the opset.

    This is the exact definition, not a page: the prose, the type constraints that decide
    which families the document must have, and the inputs, outputs and attributes.  It is
    preferred to the operator's web page because the page carries the documentation site's
    whole navigation around the definition, and a naive reading of it can return the menu
    and never reach the operator.
    """
    try:
        import onnx
        from onnx import defs
    except Exception:
        return ""
    try:
        schema = defs.get_schema(op, max_inclusive_version=opset)
    except Exception:
        try:
            schema = defs.get_schema(op)
        except Exception:
            return ""
    out = [
        "# ONNX operator definition, as the `onnx` package %s states it (schema of "
        "opset %d%s)" % (onnx.__version__, schema.since_version,
                         ", the opset asked for is %d" % opset
                         if schema.since_version != opset else ""),
        "",
        "**%s%s**, since opset %d." % (schema.domain + "." if schema.domain else "",
                                       schema.name, schema.since_version),
        "",
        (schema.doc or "").strip(),
        "",
    ]
    for constraint in schema.type_constraints:
        out.append("- type constraint `%s`: %s. %s"
                   % (constraint.type_param_str,
                      ", ".join("`%s`" % t for t in constraint.allowed_type_strs),
                      constraint.description))
    out.append("")
    for kind, params in (("Input", schema.inputs), ("Output", schema.outputs)):
        for p in params:
            types = ", ".join("`%s`" % t for t in p.types) if p.types else "any"
            out.append("- %s `%s` (%s): %s" % (kind, p.name, types, p.description))
    out.append("")
    if schema.attributes:
        for name, attribute in sorted(schema.attributes.items()):
            out.append("- attribute `%s`: %s" % (name, attribute.description))
    else:
        out.append("This operator has no attribute.")
    return "\n".join(out).strip()


# The SONNX families, and the ONNX type names that belong to each.  `bfloat16` belongs to
# the float family in ONNX but is not a SONNX type, so it is named in the verdict and not
# treated as a family of its own.
SONNX_FAMILIES = (
    ("float", ("float16", "float", "double", "bfloat16")),
    ("int", ("int8", "int16", "int32", "int64")),
    ("uint", ("uint8", "uint16", "uint32", "uint64")),
)
SONNX_TYPES = {"float16", "float", "double", "int8", "int16", "int32", "int64",
               "uint8", "uint16", "uint32", "uint64", "bool", "string"}


def _strip_tensor(type_str: str) -> str:
    """`tensor(float)` -> `float`; anything else is left as it is."""
    match = re.match(r"^tensor\(\s*([A-Za-z0-9_]+)\s*\)$", type_str.strip())
    return match.group(1) if match else type_str.strip()


def admitted_types(op: str, opset: int) -> set[str]:
    """The ONNX type names the operator admits, without the `tensor(...)` wrapper."""
    try:
        from onnx import defs

        schema = defs.get_schema(op, max_inclusive_version=opset)
    except Exception:
        return set()
    admitted = set()
    for constraint in schema.type_constraints:
        admitted.update(_strip_tensor(t) for t in constraint.allowed_type_strs)
    return admitted


def family_verdicts(op: str, opset: int) -> list[str]:
    """Which SONNX families this operator has, decided from the schema and not by memory.

    This is the one factual question the compliance verifier keeps answering from memory of
    the operator rather than from the schema it is given — and a reviewer that invents a
    family sends the writer to document an operator ONNX does not have, which then costs
    two amendment rounds, one to add the section and one to remove it.  So the answer is
    computed here, where it cannot be got wrong, and handed over already decided.
    """
    admitted = admitted_types(op, opset)
    if not admitted:
        return []
    lines = ["- `real`: the mathematical family; it applies to every operator."]
    for family, types in SONNX_FAMILIES:
        profile = [t for t in types if t in SONNX_TYPES]
        present = [t for t in types if t in admitted]
        in_profile = [t for t in present if t in SONNX_TYPES]
        outside = [t for t in present if t not in SONNX_TYPES]
        if not in_profile:
            lines.append("- `%s`: **absent** — ONNX admits no %s type for this operator; a "
                         "section for this family, or a Contents entry for it, is a finding."
                         % (family, family))
            continue
        note = ""
        if outside:
            note = (" (%s is admitted by ONNX and is outside the SONNX profile, so it is not "
                    "to be named as a type this family covers)"
                    % ", ".join("`%s`" % t for t in outside))
        missing = [t for t in profile if t not in admitted]
        tail = ""
        if missing:
            tail = (" The other SONNX types of this family (`%s`) are not admitted."
                    % "`, `".join(missing))
        lines.append("- `%s`: applies — the types this operator admits are %s%s.%s"
                     % (family, ", ".join("`%s`" % t for t in in_profile), note, tail))
    return lines


def type_constraints_text(op: str, opset: int) -> str:
    """The types ONNX admits for the operator — what a family check has to be made against.

    The compliance verifier is asked whether a type family is missing or wrongly merged.
    It cannot answer that by guessing which types the operator admits: a reviewer that
    invents a family sends the writer to document an operator ONNX does not have.  So the
    admitted types are given to it, and so is the per-family verdict, both from the
    operator's own schema.
    """
    admitted = admitted_types(op, opset)
    if not admitted:
        return ""
    verdicts = family_verdicts(op, opset)
    return ("# The types ONNX admits for this operator\n\n%s\n\n"
            "# The families this operator has, decided from the schema — decisive, not to be "
            "reconsidered from memory\n\n%s\n\n"
            "The profile is narrower than ONNX: SONNX specifies the types its "
            "'About data types' section lists, and the families it defines over them. A "
            "type ONNX admits that SONNX does not consider is **outside the profile** and "
            "its absence from the document is not a defect. So: a family of SONNX types "
            "that ONNX does not admit for this operator must not appear in the document; a "
            "SONNX type that ONNX admits must be covered by the section of its family. "
            "Which types share a semantics is decided by the operator, not by the type "
            "name.\n" % (", ".join("`%s`" % t for t in sorted(admitted)), "\n".join(verdicts)))


def onnx_definition(settings: Settings) -> str:
    """The ONNX definition of the operator, from the file given or from the operator's page."""
    if settings.onnx_doc is not None:
        return Path(settings.onnx_doc).read_text(encoding="utf-8", errors="replace")
    cache = settings.resolved_run_dir() / ("onnx-%s.txt" % settings.op)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    url = ONNX_DOC_URL.format(op=settings.op)
    text = schema_text(settings.op, settings.opset)
    if text:
        # what the writer was given is part of the run's record, like every other input
        text = "%s\n\nDocumentation page: %s\n" % (text, url)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    if not text and settings.fetch_onnx_doc:
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=30) as response:
                html = response.read().decode("utf-8", errors="replace")
            # the documentation site wraps the operator in an article; the navigation that
            # precedes it is longer than the definition and must not be what is read
            article = re.search(r"(?is)<article\b[^>]*>(.*?)</article>", html)
            body = article.group(1) if article else html
            body = re.sub(r"(?is)<(script|style|nav).*?</\1>", " ", body)
            body = re.sub(r"(?s)<[^>]+>", " ", body)
            body = (body.replace("&nbsp;", " ").replace("&lt;", "<")
                    .replace("&gt;", ">").replace("&amp;", "&").replace("&#39;", "'"))
            body = re.sub(r"[ \t]+", " ", body)
            body = re.sub(r"\n\s*\n+", "\n\n", body).strip()
            text = clip(body, 20000, "ONNX page")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        except Exception as exc:  # no network, or a moved page: the run goes on without it
            text = "(the ONNX page could not be fetched: %s)\nURL: %s" % (exc, url)
    if not text:
        text = ("(no ONNX definition was available)\nThe operator is the ONNX operator %s, "
                "opset %d, documented at %s" % (settings.op, settings.opset, url))
    return text


def guidelines_context() -> str:
    return (
        "# The normative guidelines of the SONNX profile\n\n%s\n\n"
        "# The required skeleton\n\n%s\n\n"
        "# An accepted specification of another operator, for shape only\n\n%s\n"
        % (GUIDELINES.read_text(encoding="utf-8"),
           TEMPLATE.read_text(encoding="utf-8"),
           clip(REFERENCE_SPEC.read_text(encoding="utf-8"), 40000, "reference specification"))
    )


def findings_text(state: SpecState) -> str:
    """What the writer must answer: the linter, the verifier, the adjudicator."""
    parts = []
    lint = state.get("lint") or {}
    if lint and not lint.get("ok"):
        parts.append("## The mechanical linter\n\n%s" % lint.get("stdout", ""))
    review = state.get("review") or {}
    for finding in (review.get("findings") or []):
        if finding.get("severity") == "blocking":
            parts.append(
                "## A compliance finding (blocking)\n\nsection: %s\nwhat: %s\nwhy: %s\nfix: %s"
                % (finding.get("section"), finding.get("what"), finding.get("why"),
                   finding.get("fix")))
    if state.get("diagnosis"):
        parts.append(
            "## The adjudicator: the specification is at fault\n\n%s\n\nWhat must change: %s"
            % (state.get("diagnosis"), state.get("instruction", "")))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# the agents
# ---------------------------------------------------------------------------


def _artifact(settings: Settings, system: str, asked: str, where: str,
              max_tokens: int | None = None) -> str:
    """One answer that is an artifact, with the deliberation dropped if there is no text.

    A model that deliberates before it answers can spend two whole budgets on the thinking
    and return no text at all, and raising the budget does not win that race: measured on
    2026-10-04, the test agent answered with deliberation and no text on four calls in a row
    — 16 000 tokens, then 32 000, twice — for the case set of **Acos**, which left the run
    with no case module to run.  The deliberation is what the empty answer is *made of*, so
    the last attempt is made without it: an empty answer twice is the habit of the task, not
    an accident of the budget.  When the caller has already turned the deliberation off there
    is nothing left to drop, and the empty answer is returned to the caller's own guard.
    """
    budget = max_tokens or settings.max_tokens
    text = ask(settings, system, asked, max_tokens=budget, thinking=settings.thinking)
    if text.strip() or settings.thinking == "off":
        return text
    log(settings, "%s: the answer was empty; asking once more without deliberation" % where)
    return ask(settings, system, asked, max_tokens=budget * 2, thinking="off")


def _not_a_document(spec: str) -> str:
    """Why an answer cannot be written as a document, or "" when it can.

    Two ways an answer stops in the middle, and the first one is not enough on its own.
    A red span left open is the visible one, and it is what the linter counts.  An answer
    can also be cut off *between* the spans, inside a `$...$` or in the middle of a
    sentence, and then every span it did reach is closed and the count says nothing —
    measured on 2026-10-04 in the **MaxPool** run, whose writer's third answer was cut
    inside a math span (2 spans opened, 2 closed) and held one section of the four its own
    Contents announced: 12 907 bytes against the 48 101 of the revision it replaced, and
    the last spec round spent on it.  A document whose Contents announces a section it does
    not contain is not a document either, whatever the span count says, so both are asked
    for again.
    """
    opened, closed = harness.red_spans(spec)
    if not opened or opened != closed:
        return "the answer stops in the middle (%d red span(s) opened, %d closed)" % (
            opened, closed)
    missing = harness.missing_sections(spec)
    if missing:
        return ("the answer keeps the Contents of a longer document but not its sections: "
                "%s" % ", ".join("#%s" % anchor for anchor in missing))
    return ""


def write_spec(state: SpecState, settings: Settings) -> dict:
    """The specification writer: one full document, replacing whatever was there."""
    spec_path = Path(state.get("spec_path") or settings.resolved_spec_path())
    current = state.get("spec") or (harness.read_spec(spec_path) if spec_path.exists() else "")
    findings = findings_text(state)
    round_number = int(state.get("spec_round", 0)) + 1
    log(settings, "writer  round %d: %s" % (round_number, "amending" if current else "drafting"))

    task = [
        "Operator: **%s** (ONNX type `%s`, opset %d)." % (settings.op, settings.op, settings.opset),
        "Entry point of the implementation that will be read from your document: `%s`."
        % settings.entry,
        "The document is written to `%s`." % spec_path,
    ]
    if current:
        task.append("This is an **amendment**: the document below already exists. Return the "
                    "whole document again, with the smallest change that answers the findings; "
                    "leave every other sentence as it is.")
    else:
        task.append("This is the **first draft**: there is no document yet.")
    task.append("\n## The ONNX definition of the operator\n\n%s" % onnx_definition(settings))
    if current:
        task.append("\n## The document as it stands\n\n%s" % current)
    task.append("\n## What must be answered\n\n%s"
                % (findings or "Nothing: this is the first draft."))

    started = time.time()
    system, question = (prompt("writer.md") + "\n\n" + guidelines_context(), "\n\n".join(task))
    answer = _artifact(settings, system, question, "writer  round %d" % round_number)
    spec = extract.extract_markdown(answer)
    problem = _not_a_document(spec)
    for attempt in range(settings.max_writer_retries):
        if not problem:
            break
        # a model answer can stop in the middle of the document.  The skill's linter reads
        # exactly these two counts and would catch it, but only after spending a round of
        # the loop; the document must not be written to disk as if it were a document.
        log(settings, "writer  round %d: %s; asking again" % (round_number, problem))
        answer = ask(settings, system, question + "\n\n## Note\n\nYour previous answer was "
                     "cut off before the end of the document and cannot be used: %s. Return the "
                     "*complete* document: every section the Contents lists, every red span "
                     "closed by its `[END]`, every example. If the document does not fit in "
                     "one answer, make the prose shorter — never leave a section out and "
                     "never stop before the last `## Outputs`." % problem,
                     max_tokens=settings.max_tokens * 2, thinking=settings.thinking)
        spec = extract.extract_markdown(answer)
        problem = _not_a_document(spec)
    if problem:
        if current:
            # a complete document is never replaced by a cut-off one
            log(settings, "writer  round %d: %s, after %d attempt(s); the document on disk "
                "is kept" % (round_number, problem, settings.max_writer_retries))
            spec = current
        else:
            log(settings, "writer  round %d: the first draft arrived incomplete: %s"
                % (round_number, problem))
    if spec != current:
        harness.write_spec(spec_path, spec)
    log(settings, "writer  round %d: %d characters in %.0fs -> %s"
        % (round_number, len(spec), time.time() - started, spec_path))
    return {
        "spec": harness.read_spec(spec_path),
        "spec_path": str(spec_path),
        "spec_md5": harness.md5(spec_path),
        "spec_round": round_number,
        "lint": {}, "review": {}, "spec_ok": False, "review_failed": False,
        "diagnosis": "", "instruction": "", "verdict": "",
        "history": [{"agent": "writer", "round": round_number, "path": str(spec_path),
                     "characters": len(spec)}],
        "trace": ["writer round %d -> %s" % (round_number, spec_path)],
    }


def check_spec(state: SpecState, settings: Settings) -> dict:
    """The compliance verifier: the linter first, then the review against the guidelines."""
    spec_path = Path(state["spec_path"])
    lint = harness.run_lint(settings, spec_path)
    log(settings, "checker lint: %s (%d finding(s))"
        % ("clean" if lint["ok"] else "NOT clean", len(lint["findings"])))

    spec = harness.read_spec(spec_path)
    md5 = harness.md5(spec_path)
    user = (
        "# The document under review\n\n%s\n\n"
        "# The mechanical linter's output\n\n%s\n\n"
        "# The sections of the document, as its Contents list has them\n\n%s"
        % (spec, lint.get("stdout") or "(no output)",
           ", ".join(harness.sections(spec)) or "(none)")
    )
    admitted = type_constraints_text(settings.op, settings.opset)
    if admitted:
        user += "\n\n" + admitted
    review_prompt = (
        prompt("verifier.md")
        + "\n\n# The normative guidelines\n\n%s\n\n# The review checklist of this repository\n\n%s\n"
        % (GUIDELINES.read_text(encoding="utf-8"),
           (REFERENCES_DIR / "checklist.md").read_text(encoding="utf-8"))
    )
    review, failure = {}, ""
    for attempt in range(settings.max_review_retries + 1):
        try:
            review = extract.extract_json(
                ask(settings, review_prompt, user, max_tokens=settings.max_review_tokens,
                    thinking=settings.verdict_thinking))
            break
        except Exception as exc:
            failure = str(exc)
            log(settings, "checker review: no verdict (%s)%s" % (
                failure, "; asking again" if attempt < settings.max_review_retries else ""))
    if not review:
        # The reviewer did not answer. That is a failure of this run, not a finding against
        # the document: the writer must not be sent to amend a document the verifier never
        # read. The run stops and says so.
        log(settings, "checker review: the verifier returned no verdict")
        return {
            "lint": lint, "review": {"ok": False, "findings": [], "notes": failure},
            "spec_ok": False, "review_failed": True, "spec_md5": md5,
            "seen_amendments": list(state.get("seen_amendments", [])),
            "history": [{"agent": "checker", "lint_ok": lint["ok"], "review_ok": False,
                         "blocking": [], "note": "the verifier returned no verdict: %s"
                         % failure}],
            "trace": ["checker: no verdict"],
        }
    blocking = [f for f in review.get("findings", []) if f.get("severity") == "blocking"]
    ok = bool(lint["ok"]) and bool(review.get("ok")) and not blocking
    log(settings, "checker review: %s (%d blocking, %d minor)"
        % ("ok" if review.get("ok") else "NOT ok", len(blocking),
           len(review.get("findings", [])) - len(blocking)))
    seen = list(state.get("seen_amendments", []))
    if not ok:
        seen.append(json.dumps(blocking or lint["findings"], sort_keys=True)[:2000])
    # a round that changed nothing and still failed has nothing left to try: the writer's
    # answer was cut off, or the amendment it made did not answer the finding.  The document
    # is read here rather than taken from the state, so that a run started from the file on
    # disk (`--start-from spec`) has the same guard as one that began with the writer.
    previous = state.get("checked_md5")
    return {
        "lint": lint, "review": review, "spec_ok": ok, "spec_md5": md5,
        "checked_md5": state.get("spec_md5") or md5,
        "no_progress": bool(previous and not ok and previous == md5),
        "seen_amendments": seen,
        "history": [{"agent": "checker", "lint_ok": lint["ok"],
                     "review_ok": bool(review.get("ok")), "blocking": blocking}],
        "trace": ["checker: %s" % ("ok" if ok else "not ok")],
    }


def _keep_rejected(path: Path, attempt: int, answer: str, problem: str) -> Path:
    """Keep an answer that was refused, beside the artifact it did not become.

    What an agent answers when it fails is evidence about the agent, and the run has no
    other way to show it: the file the answer did not become holds the previous answer, or
    nothing at all, so a refusal that is not kept cannot be told from an extractor that
    read the wrong part of the answer.
    """
    kept = path.with_name("%s-rejected-%d.txt" % (path.stem, attempt))
    try:
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_text("# %s\n\n%s\n" % (problem, answer), encoding="utf-8")
    except OSError:
        return kept
    return kept


# the parser's ways of saying that the text stopped, rather than that it is not Python
_CUT_OFF_SYNTAX = ("unterminated", "was never closed", "unexpected EOF", "EOF while")


def _impl_module_problem(module: str, entry: str) -> str | None:
    """Why a blind implementer's answer is not an implementation, or `None` when it is one.

    The implementer is the one agent whose answer is never read before it is used: an empty
    module is a module that imports and defines nothing, and the next thing that happens to
    it is a comparison with the reference, which then reports a failure of the *document*.
    Measured on 2026-10-04: an answer of thinking and no text, written to disk as the
    implementation of **Asin**.

    A module that stopped in the middle is named as such, because it is repaired differently
    from an answer that is not a module at all: it is written again, with more room, while
    prose mistaken for a module is simply asked for again.
    """
    if not module.strip():
        return "the answer is empty"
    try:
        tree = ast.parse(module)
    except SyntaxError as exc:
        if any(marker in str(exc) for marker in _CUT_OFF_SYNTAX):
            # measured on 2026-10-04: both answers of the implementer for **Atan** were
            # truncated this way, `unterminated triple-quoted string literal (detected at
            # line 5)` and `unterminated string literal (detected at line 11)`
            return "the answer stops in the middle of the module: %s" % exc
        return "the answer is not Python: %s" % exc
    names = {node.name for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if entry not in names:
        return "the module defines no `%s`" % entry
    return None


def implement(state: SpecState, settings: Settings) -> dict:
    """The blind implementer: the document, and nothing else."""
    iteration = int(state.get("iteration", 0)) + 1
    spec_path = Path(state["spec_path"])
    spec = harness.read_spec(spec_path)
    impl_path = settings.impl_path(iteration)
    log(settings, "blind   iteration %d: implementing from %s" % (iteration, spec_path.name))

    user = ("The operator is `%s`; the entry point of the module is `%s`.\n\n"
            "The specification follows, complete.\n\n-----\n\n%s\n\n-----\n"
            % (settings.op, settings.entry, spec))
    started = time.time()
    problem = ""
    impl_text = ""
    for attempt in range(max(1, int(settings.max_module_retries))):
        asked = user
        budget = None
        cut_off = problem.startswith("the answer stops in the middle")
        if problem:
            if cut_off:
                # the writer's cut-off document in another guise, and repaired the same way:
                # a module that stopped mid-statement cannot be trimmed into a module, so it
                # is asked for again with twice the room
                asked += ("\nYour previous answer was cut off before the end of the module and "
                          "cannot be used. Return the *complete* module: the whole of `%s`, "
                          "and nothing after it. If the module does not fit in one answer, "
                          "make it shorter — never stop in the middle of a statement, a string "
                          "or a comment." % settings.entry)
                budget = settings.max_tokens * 2
            else:
                asked += ("\nYour previous answer was not an implementation: %s\n\n"
                          "Answer with the module itself, and nothing else." % problem)
        answer = _artifact(settings, prompt("implementer.md"), asked,
                           "blind   iteration %d" % iteration, max_tokens=budget)
        impl_text = extract.extract_python(answer, ("def %s" % settings.entry,))
        problem = _impl_module_problem(impl_text, settings.entry)
        if not problem:
            break
        kept = _keep_rejected(impl_path, attempt + 1, answer, problem)
        log(settings, "blind   iteration %d: %s (attempt %d): %s (kept as %s)"
            % (iteration, "the answer was cut off" if cut_off
               else "the answer is not an implementation", attempt + 1, problem, kept.name))
    if problem:
        # nothing is written: an empty module on disk would be compared with the reference
        # and reported as a failure of the document
        log(settings, "blind   iteration %d: no implementation after %d attempt(s): %s"
            % (iteration, settings.max_module_retries, problem))
        return {"iteration": iteration, "impl_path": "", "impl_text": "",
                "decisions": "", "rebuild_cases": False, "impl_unusable": problem,
                "trace": ["implementer iteration %d: no usable module" % iteration]}

    impl_path.parent.mkdir(parents=True, exist_ok=True)
    impl_path.write_text(impl_text, encoding="utf-8")
    decisions = harness.decision_points(impl_text)
    log(settings, "blind   iteration %d: %d characters, %d DECISION point(s) in %.0fs -> %s"
        % (iteration, len(impl_text), decisions.count("# DECISION:"),
           time.time() - started, impl_path))
    return {
        "iteration": iteration,
        "impl_path": str(impl_path),
        "impl_text": impl_text,
        "decisions": decisions,
        "rebuild_cases": False,
        # the revisions of the case set are counted per implementation: a new reading of the
        # document is a new round, and the tester's budget starts again
        "case_rounds": 0,
        "report_runs": 0,
        # a previous iteration's unusable answer is not this one's
        "impl_unusable": "",
        "history": [{"agent": "implementer", "iteration": iteration, "path": str(impl_path),
                     "decisions_count": decisions.count("# DECISION:")}],
        "trace": ["implementer iteration %d -> %s" % (iteration, impl_path)],
    }


# The contract for `COVERAGE` says it maps every section anchor to the number of cases that
# exercise it — and a case set that counts its own cases per section is the natural way to
# keep that true when the list grows. So the dictionary is read by importing the module, the
# way the harness sees it, and not by reading the literal of the assignment: a computed
# dictionary looks like `{"real": 0, "float": 0, "int": 0}` in the source, and reading that
# literal reports every section as uncovered.
COVERAGE_READER = (
    "import importlib.util, json, sys\n"
    "spec = importlib.util.spec_from_file_location('sonnx_cases', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "sys.modules['sonnx_cases'] = module\n"
    "spec.loader.exec_module(module)\n"
    "coverage = dict(getattr(module, 'COVERAGE', {}) or {})\n"
    "print(json.dumps({str(k): int(v) for k, v in coverage.items()}))\n"
)


def coverage_of(cases_path: Path, settings: Settings | None = None) -> dict:
    """The case module's own COVERAGE dictionary, as the module itself would report it.

    The module is imported in a subprocess with the interpreter the harness uses, so a
    `COVERAGE` built by the module — the usual `{"real": 0, ...}` skeleton filled in from the
    case list — is read as it stands when every case has been counted. The literal read is
    the fallback, for a module that cannot be imported here.
    """
    python = getattr(settings, "python", None) or DEFAULT_PYTHON
    try:
        proc = subprocess.run([python, "-c", COVERAGE_READER, str(cases_path)],
                              capture_output=True, text=True, timeout=300)
        if proc.returncode == 0:
            lines = [line for line in proc.stdout.splitlines() if line.strip()]
            computed = json.loads(lines[-1]) if lines else {}
            if computed:
                return computed
    except Exception:
        pass
    return _literal_coverage(cases_path)


def _literal_coverage(cases_path: Path) -> dict:
    """The `COVERAGE = {...}` literal of the module, for when it cannot be imported."""
    try:
        tree = ast.parse(Path(cases_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "COVERAGE":
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        return {}
    return {}


def _cases_module_problem(module: str) -> str | None:
    """Why a tester answer is not a case module, or `None` when it is one.

    The same defect the writer's node catches in a cut-off document: an answer that is
    empty, or that has no `cases()` and no `COVERAGE`, is not a case module.  It must not
    be written over the module that is there — an empty module imports cleanly, declares
    no coverage, and is then read as a document whose sections have no cases.

    The answer is *parsed*, and not only searched for the two names it must contain, for the
    same reason: text that cannot be parsed is text the harness cannot import, and a module
    that dies on its first line is not a case set that failed — it is an answer that was
    never a module, and it is asked for again.  Measured on 2026-10-04: the second case set of
    **Atan** carried four lines of the model's own tool-call syntax after the last statement
    of the module — `｜DSML｜ parameter>` and the like, with `U+FF5C` for the bar — so the
    harness died with `SyntaxError: invalid character '｜' (U+FF5C)` on line 499, wrote no
    report, and the module was written over the case set that had run.
    """
    if not module.strip():
        return "the answer is empty"
    try:
        ast.parse(module)
    except SyntaxError as exc:
        if any(marker in str(exc) for marker in _CUT_OFF_SYNTAX):
            return "the answer stops in the middle of the module: %s" % exc
        return "the answer is not Python: %s" % exc
    if not re.search(r"^def cases\s*\(", module, re.M):
        return "the module defines no `cases()` function"
    if not re.search(r"^COVERAGE\s*[:=]", module, re.M):
        return "the module declares no `COVERAGE` dictionary"
    return None


def _write_cases(state: SpecState, settings: Settings, error: str = "",
                 findings: str = "", report: dict | None = None,
                 compared_nothing: str = "") -> Path:
    """Ask the tester for the case module, or for a repair of the one that raised.

    An answer that is not a case module is not written: the tester is asked again, and the
    node stops with `cases-unusable` when it never answers with one.

    Three kinds of repair are asked for, and they are not the same request.  A module that
    *raised* is repaired against the Python error (`error`).  A case set whose every case the
    runtime refused compared nothing, and is repaired against the refusals
    (`compared_nothing`) — it is not an error of the module and not a failure of the
    document, so neither of the other two requests would tell the tester what to do.  A case
    set that ran and did not pass, and that the adjudicator attributed to the tester, is
    repaired against the failure itself (`findings`, the adjudicator's instruction, and
    `report`, what the harness reported): without them the tester is asked to fix a case set
    it is not told anything about, and the answer is the same case set again.  Measured on
    2026-10-04: the second case set of **Asin** failed on the same two cases as the first,
    having been asked for with neither the report nor the adjudicator's instruction, and the
    run stopped.
    """
    spec = harness.read_spec(Path(state["spec_path"]))
    cases_path = settings.cases_path()
    user = [
        "The operator is `%s` (ONNX type, opset %d); the implementation's entry point is `%s`."
        % (settings.op, settings.opset, settings.entry),
        "The case module is written to `%s`." % cases_path,
        "\n# The specification under test\n\n%s" % spec,
        "\n# The harness that will run your cases\n\n```python\n%s\n```"
        % clip(harness.SPECLOOP.read_text(encoding="utf-8"), 14000, "harness"),
        "\n# The shape to follow: the accepted case module of another operator\n\n```python\n%s\n```"
        % clip(WORKED_CASES.read_text(encoding="utf-8"), 24000, "worked case module"),
    ]
    if cases_path.exists() and not error:
        user.append("\n# The case module as it stands: grow it, keep what still holds\n\n"
                    "```python\n%s\n```" % clip(cases_path.read_text(encoding="utf-8"),
                                                24000, "case module"))
    if error:
        user.append("\n# The case module raised\n\n```\n%s\n```\n\n"
                    "Return the whole module again, corrected." % clip(error, 4000, "traceback"))
    if compared_nothing:
        user.append(
            "\n# The case set compared nothing\n\nThe harness ran your case module and not "
            "one case was compared: every case was refused by the runtime, or `cases()` "
            "yielded nothing. A run in which nothing was compared is evidence of nothing, and "
            "the case set is what the loop sends back. What the runtime said:\n\n```\n%s\n```"
            "\n\nReturn the whole module again, with cases the runtime runs: a model it "
            "refuses — an output declared with the wrong type, an attribute the operator "
            "cannot take, an operand outside the document's domain — makes the case open, and "
            "a case set made of open cases checks nothing. If the runtime has no kernel for "
            "any type of this operator then it cannot be verified against ONNX Runtime at "
            "all, and the run stops rather than converging on a comparison that never "
            "happened." % clip(compared_nothing, 4000, "refusals"))
    if findings or report:
        user.append(
            "\n# The case module failed\n\nThe harness ran it and it did not pass, and the "
            "adjudicator attributed the failure to the case module — not to the document and "
            "not to the implementation. Repair the case module, and return the whole of it "
            "again: a case that reports the reference's own inaccuracy as a failure of the "
            "implementation, or that a constraint of the document rules out, is a defective "
            "case, and stating it again is not a repair.\n\n"
            "## What the adjudicator said\n\n%s\n\n## What the harness reported\n\n```\n%s\n```"
            % (clip(findings, 6000, "adjudication") or "(nothing)",
               clip(harness.failing_report(report), 8000, "report")))

    note = ""
    module = ""
    for attempt in range(max(1, int(settings.max_module_retries))):
        asked = list(user)
        budget = None
        cut_off = note.startswith("the answer stops in the middle")
        if note:
            if cut_off:
                asked.append("\n# Your previous answer was cut off\n\n%s\n\n"
                             "Return the *complete* module: the whole of it, `COVERAGE` included, "
                             "and nothing after it. If it does not fit in one answer, make it "
                             "shorter — never stop in the middle of a statement, a string or a "
                             "comment, and never add anything that is not Python." % note)
                budget = settings.max_tokens * 2
            else:
                asked.append("\n# Your previous answer was not a case module\n\n%s\n\n"
                             "Return the whole module again, and nothing else." % note)
        answer = _artifact(settings, prompt("tester.md"), "\n\n".join(asked), "tester",
                           max_tokens=budget)
        module = extract.extract_python(answer, ("def cases", "COVERAGE"))
        note = _cases_module_problem(module)
        if not note:
            break
        kept = _keep_rejected(cases_path, attempt + 1, answer, note)
        log(settings, "tester  the answer is not a case module (attempt %d): %s (kept as %s)"
            % (attempt + 1, note, kept.name))
    if note:
        raise ValueError("the tester returned no usable case module: %s (%d attempt(s))"
                         % (note, settings.max_module_retries))
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(module, encoding="utf-8")
    cases_path.with_suffix(".meta.json").write_text(
        json.dumps({"spec_md5": harness.md5(Path(state["spec_path"])),
                    "written": datetime.now().isoformat(timespec="seconds")}, indent=2) + "\n")
    return cases_path


def _run_harness(settings: Settings, spec_path: Path, cases_path: Path, impl_path: Path,
                 report_path: Path) -> dict:
    """Run the harness on the case module as it stands.

    The report of the previous run is removed first.  A run that dies before it writes its
    report — an import that raises, a module that stopped in the middle — leaves the report
    of the run before it on disk, and reading that one makes a case module that never ran
    look like a case module that passed.
    """
    try:
        Path(report_path).unlink()
    except FileNotFoundError:
        pass
    return harness.run_specloop(settings, spec_path, cases_path, impl_path, report_path)


def _vacuous(report: dict) -> bool:
    """True when the case set compared nothing: every case is open, or there are none.

    A case set whose every case the runtime refuses passes the harness — nothing
    mismatched, so the exit status is 0 — and the run would converge on a comparison that
    never happened.  Measured on 2026-10-04 while preparing **MaxPool**: a case module that
    declared the type of the second output wrongly (`OUTPUT_TYPES = {"Indices":
    np.float32}`) had all five of its cases refused with `Type Error: Type (tensor(float))
    of output arg (Indices) of node (node0) does not match expected type (tensor(int64))`,
    and the harness reported `5 run, 0 passed, 0 mismatched, 5 open` — a pass with no
    comparison in it.  A report that is broken, or that never ran, is another guard's
    business and is not reported here.
    """
    cases = report.get("cases") or {}
    if report.get("error") or not cases:
        return False
    return int(cases.get("total") or 0) == 0 or int(cases.get("passed") or 0) == 0


def _broken(report: dict) -> bool:
    """A case module that raises is the tester's defect, not a finding about the document.

    So is a case module that compares nothing: the repair the tester is asked for is the
    same kind of work, and the reason it is given names the runtime's refusal.
    """
    if not report or report.get("error"):
        return True
    if _vacuous(report):
        return True
    if any(sweep.get("error") for sweep in report.get("sweeps", [])):
        return True
    return any(item.get("case") == "<module>"
               for item in (report.get("cases") or {}).get("mismatched", []))


def _broken_reason(result: dict) -> str:
    report = result.get("report") or {}
    if report.get("error"):
        return str(report["error"]) + "\n" + str(report.get("stdout", ""))[-3000:]
    if _vacuous(report):
        cases = report.get("cases") or {}
        opens = cases.get("open") or []
        reason = ("not one case was compared: %s case(s), %s passed, %d open. The runtime "
                  "refused the model of every case, or the case set has no case in it, so "
                  "this run is evidence of nothing. The refusals are the reason:"
                  % (cases.get("total"), cases.get("passed"), len(opens)))
        if not opens:
            reason = ("no case at all was compared: `cases()` yielded nothing, so this run "
                      "is evidence of nothing.")
        return reason + "\n" + "\n".join(
            "%s: %s" % (item.get("case"), str(item.get("reason"))[:400]) for item in opens[:4])
    for sweep in report.get("sweeps", []):
        if sweep.get("error"):
            return "%s: %s\n%s" % (sweep.get("sweep"), sweep["error"], sweep.get("traceback", ""))
    for item in (report.get("cases") or {}).get("mismatched", []):
        if item.get("case") == "<module>":
            return str(item.get("reason", "")) + "\n" + str(item.get("traceback", ""))
    return json.dumps(report)[:2000]


def test(state: SpecState, settings: Settings) -> dict:
    """The test agent: the case set, then the comparison with ONNX Runtime."""
    spec_path = Path(state["spec_path"])
    impl_path = Path(state["impl_path"])
    cases_path = settings.cases_path()
    meta_path = cases_path.with_suffix(".meta.json")
    blamed = bool(state.get("rebuild_cases"))
    rebuild = blamed or not cases_path.exists()
    if not rebuild and meta_path.exists():
        try:
            rebuild = json.loads(meta_path.read_text())["spec_md5"] != harness.md5(spec_path)
        except Exception:
            rebuild = True
    if rebuild:
        if blamed:
            log(settings, "tester  rewriting the case set: the adjudicator attributed the "
                          "failure to it")
        else:
            log(settings, "tester  writing the case set for %s" % spec_path.name)
        # the failure evidence goes to the tester only when the adjudicator blamed it: a case
        # set that is being rewritten because the document was amended is not a case set that
        # failed, and telling it about a failure of some earlier round would send it to repair
        # the wrong thing
        blamed_evidence = {"findings": str(state.get("instruction", "") or ""),
                           "report": state.get("report") or {}} if blamed else {}
        try:
            _write_cases(state, settings, **blamed_evidence)
        except Exception as exc:
            log(settings, "tester  could not write the case set: %s" % exc)
            return {"test_ok": False, "report": {"error": str(exc)},
                    "cases_path": str(cases_path),
                    "cases_unusable": str(exc),
                    "trace": ["tester: the case set could not be written: %s" % exc]}

    iteration = int(state.get("iteration", 1))
    report_path = settings.report_path(iteration)
    before = int(state.get("report_runs", 0))   # harness runs already made for this iteration
    runs = 0

    def run_harness() -> tuple[dict, Path]:
        """One harness run, and the report it wrote.

        Every run writes its own report.  A case module that raises, or a case set that is
        revised after the adjudicator blamed it, is run more than once — inside this node and
        across the calls to it — and the report of the run before is the evidence of what the
        case set did *then*, which is the round the repair is for.  Overwriting it leaves the
        run with one report for two rounds, and no way to see what changed between them.
        """
        nonlocal runs
        runs += 1
        number = before + runs
        path = report_path if number == 1 else report_path.with_name(
            "%s-%d%s" % (report_path.stem, number, report_path.suffix))
        return _run_harness(settings, spec_path, cases_path, impl_path, path), path

    result, used = run_harness()
    log(settings, "tester  iteration %d: %s"
        % (iteration, (result["stdout"].splitlines() or ["(no output)"])[0]))

    repairs = 0
    while _broken(result["report"]) and repairs < settings.max_case_repairs:
        repairs += 1
        reason = _broken_reason(result)
        nothing = _vacuous(result["report"])
        log(settings, "tester  repair %d: %s"
            % (repairs, ("nothing was compared: " if nothing else "")
               + (reason.splitlines() or [""])[0][:160]))
        try:
            _write_cases(state, settings, **({"compared_nothing": reason} if nothing
                                             else {"error": reason}))
        except Exception as exc:
            return {"test_ok": False, "report": {"error": str(exc)},
                    "cases_path": str(cases_path),
                    "cases_unusable": str(exc),
                    "trace": ["tester: the case set could not be repaired: %s" % exc]}
        result, used = run_harness()

    coverage = coverage_of(cases_path, settings)
    report = result["report"] or {"error": "the harness produced no report",
                                  "stdout": result["stdout"][-4000:]}
    sections = harness.sections(harness.read_spec(spec_path))
    uncovered = [s for s in sections if int(coverage.get(s, 0) or 0) <= 0]
    # `_broken` and not the harness's exit status alone: a case set whose every case the
    # runtime refused exits 0 and would pass the run having compared nothing
    ok = bool(result["ok"]) and not uncovered and not _broken(report)
    log(settings, "tester  iteration %d: %s"
        % (iteration, "PASS" if ok else
           ("FAIL (nothing compared)" if _vacuous(report) and not uncovered else
            "FAIL (%d uncovered section(s))" % len(uncovered))))
    return {
        "cases_path": str(cases_path),
        "report": report,
        "coverage": coverage,
        "uncovered": uncovered,
        "test_ok": ok,
        "rebuild_cases": False,
        "report_runs": before + runs,
        # a previous iteration's unusable answer is not this one's
        "cases_unusable": "",
        "history": [{"agent": "tester", "iteration": iteration, "ok": ok,
                     "coverage": coverage, "uncovered": uncovered,
                     "cases": (report.get("cases") or {}).get("total"),
                     "mismatched": len((report.get("cases") or {}).get("mismatched", [])),
                     "report": str(used)}],
        "trace": ["tester iteration %d: %s" % (iteration, "pass" if ok else "fail")],
    }


def diagnose(state: SpecState, settings: Settings) -> dict:
    """The adjudicator: the specification, the implementation, or the case set."""
    report = state.get("report") or {}
    spec = harness.read_spec(Path(state["spec_path"]))
    impl = Path(state["impl_path"]).read_text(encoding="utf-8")
    cases = Path(state["cases_path"]).read_text(encoding="utf-8")
    log(settings, "judge   deciding where the failure is")
    user = [
        "# The operator\n\n%s, ONNX opset %d." % (settings.op, settings.opset),
        "# The specification under test\n\n%s" % spec,
        "# The implementation, with its DECISION points\n\n```python\n%s\n```\n\n"
        "## The DECISION points alone\n\n%s"
        % (clip(impl, 20000, "implementation"), state.get("decisions", "")),
        "# The case module\n\n```python\n%s\n```" % clip(cases, 16000, "case module"),
        "# What the harness reported\n\n```\n%s\n```" % harness.failing_report(report),
        "# What has already been tried\n\n%s"
        % (json.dumps(state.get("seen_amendments", []), indent=2)[:3000]
           if state.get("seen_amendments") else "(nothing: this is the first failure)"),
    ]
    try:
        verdict = extract.extract_json(
            ask(settings, prompt("diagnoser.md"), "\n\n".join(user),
                max_tokens=settings.max_tokens, thinking=settings.verdict_thinking))
    except Exception as exc:
        verdict = {"verdict": "stop", "reason": "the adjudicator did not return JSON: %s" % exc,
                   "instruction": "", "confidence": "low"}
    v = str(verdict.get("verdict", "stop")).strip().lower()
    if v not in ("spec", "code", "test", "stop"):
        v = "stop"
    log(settings, "judge   verdict: %s (%s)" % (v, verdict.get("confidence")))
    seen = list(state.get("seen_amendments", []))
    if v == "spec":
        seen.append("spec:" + str(verdict.get("instruction", ""))[:400])
    return {
        "verdict": v,
        "diagnosis": str(verdict.get("reason", "")),
        "instruction": str(verdict.get("instruction", "")),
        "confidence": verdict.get("confidence"),
        "seen_amendments": seen,
        "rebuild_cases": v == "test",
        # how many times the tester has been sent back to its case set for *this*
        # implementation: the loop must be able to stop, and a revised case set is new
        # evidence rather than a repeat, so the `unchanged` guard no longer bounds it
        "case_rounds": int(state.get("case_rounds", 0)) + (1 if v == "test" else 0),
        "history": [{"agent": "diagnoser", "verdict": v, "reason": verdict.get("reason"),
                     "instruction": verdict.get("instruction"),
                     "confidence": verdict.get("confidence")}],
        "trace": ["diagnoser: %s" % v],
    }


# ---------------------------------------------------------------------------
# the ends of the loop, and its guards
# ---------------------------------------------------------------------------


def _artifact_md5(state: SpecState, key: str) -> str:
    """The content digest of one artifact of the round, or "" when there is none."""
    path = state.get(key)
    try:
        return harness.md5(Path(path)) if path else ""
    except OSError:
        return ""


def _fail_signature(state: SpecState) -> str:
    """What a round left behind, so that a round that changed nothing can be spotted.

    The signature covers the *content* of the three artifacts a round can rewrite — the
    document, the implementation and the case set — and not their paths, which change with
    the iteration even when the content does not.  A failure that survives an unchanged
    document is not a dead end when the round was allowed to change only the code or only
    the cases: the tester is asked for a new case set precisely when the adjudicator's
    verdict is `test`, and the document is then the one thing that round must leave alone.
    Seen on 2026-10-04: the second case set of **Asin** failed on the same two cases as the
    first — the module had been rewritten, from 18 575 to 23 253 bytes — and the loop stopped
    at iteration 1 of 5 with a reason that named the document.
    """
    cases = sorted(str(i.get("case"))
                   for i in (state.get("report", {}).get("cases", {}) or {}).get("mismatched", []))
    sweeps = sorted(str(s.get("sweep")) for s in state.get("report", {}).get("sweeps", [])
                    if s.get("error") or s.get("failures"))
    artifacts = [_artifact_md5(state, key) for key in ("spec_path", "impl_path", "cases_path")]
    return "%s|%s|%s" % (json.dumps(artifacts), json.dumps(cases)[:600],
                         json.dumps(sweeps)[:400])


def mark_failure(state: SpecState, settings: Settings) -> dict:
    """Remember the failing set of this round, so an unchanged round can be spotted."""
    return {"previous_failure": state.get("fail_signature", ""),
            "fail_signature": _fail_signature(state)}


def _run_record(state: SpecState, settings: Settings, outcome: str) -> dict:
    record = {
        "op": settings.op, "opset": settings.opset, "outcome": outcome,
        "spec": state.get("spec_path"), "spec_md5": state.get("spec_md5"),
        "iterations": state.get("iteration", 0), "spec_rounds": state.get("spec_round", 0),
        "test_ok": state.get("test_ok"), "coverage": state.get("coverage"),
        "uncovered": state.get("uncovered"), "impl": state.get("impl_path"),
        "cases": state.get("cases_path"), "verdict": state.get("verdict"),
        "diagnosis": state.get("diagnosis"), "instruction": state.get("instruction"),
        "history": state.get("history", []), "report": state.get("report"),
        # why an agent's answer was not usable, when that is what stopped the run: the
        # artifact the outcome names is missing, and the reason it is missing is not a
        # defect of the document
        "cases_unusable": state.get("cases_unusable"),
        "impl_unusable": state.get("impl_unusable"),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    run_dir = settings.resolved_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / ("run-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    log(settings, "record  %s -> %s" % (outcome, path))
    return record


def _finish(outcome: str, reason: str):
    def node(state: SpecState, settings: Settings) -> dict:
        record = _run_record(state, settings, outcome)
        log(settings, "stop    %s: %s" % (outcome, reason))
        return {"stopped": reason, "result": record, "trace": ["finish: %s" % outcome]}
    return node


def _route_after_check(state: SpecState, settings: Settings) -> str:
    if state.get("spec_ok"):
        return "implement"
    if state.get("review_failed"):
        return "finish_verifier"
    if state.get("no_progress"):
        return "finish_stalled"
    if int(state.get("spec_round", 0)) >= settings.max_spec_rounds:
        return "finish_guard"
    signature = json.dumps(state.get("review", {}).get("findings", []), sort_keys=True)[:2000]
    if signature and signature in list(state.get("seen_amendments", []))[:-1]:
        return "finish_guard"
    return "write_spec"


def _route_after_implement(state: SpecState, settings: Settings) -> str:
    if state.get("impl_unusable"):
        return "finish_impl"
    return "test"


def _route_after_test(state: SpecState, settings: Settings) -> str:
    if state.get("test_ok"):
        return "finish_pass"
    if state.get("cases_unusable"):
        return "finish_cases"
    if state.get("uncovered"):
        return "finish_uncovered"
    if int(state.get("iteration", 0)) >= settings.max_iterations:
        return "finish_cap"
    if state.get("fail_signature") and state.get("fail_signature") == state.get("previous_failure"):
        return "finish_unchanged"
    if int(state.get("case_rounds", 0)) >= settings.max_case_rounds:
        return "finish_cases_stuck"
    return "diagnose"


def _route_after_diagnose(state: SpecState, settings: Settings) -> str:
    verdict = state.get("verdict")
    if verdict == "stop":
        return "finish_stop"
    if verdict in ("code", "test") and int(state.get("iteration", 0)) >= settings.max_iterations:
        return "finish_cap"
    if verdict == "code":
        return "implement"
    if verdict == "test":
        return "test"
    instruction = "spec:" + str(state.get("instruction", ""))[:400]
    if sum(1 for s in state.get("seen_amendments", []) if s == instruction) > 1:
        return "finish_repeat"
    return "write_spec"


def build_graph(settings: Settings):
    """The compiled graph."""
    graph = StateGraph(SpecState)
    graph.add_node("write_spec", lambda s: write_spec(s, settings))
    graph.add_node("check_spec", lambda s: check_spec(s, settings))
    graph.add_node("implement", lambda s: implement(s, settings))
    graph.add_node("test", lambda s: test(s, settings))
    graph.add_node("diagnose", lambda s: diagnose(s, settings))
    graph.add_node("record_failure", lambda s: mark_failure(s, settings))
    ends = {
        "finish_pass": ("converged", "a fresh blind implementation passed"),
        "finish_cap": ("iteration-cap", "the iteration cap was reached"),
        "finish_guard": ("unresolved-review", "the verifier's finding was not resolved"),
        "finish_stalled": ("stalled",
                           "the document did not change and the compliance check still "
                           "failed"),
        "finish_verifier": ("verifier-failed",
                            "the compliance verifier returned no verdict"),
        "finish_cases": ("cases-unusable",
                         "the test agent returned no usable case module"),
        "finish_impl": ("impl-unusable",
                        "the blind implementer returned no usable module"),
        "finish_uncovered": ("uncovered-section", "a section of the document has no case"),
        "finish_unchanged": ("unchanged",
                             "nothing the round could change changed, and the case still fails"),
        "finish_cases_stuck": ("cases-stuck",
                               "the case set still fails after the tester revised it %d time(s)"
                               % settings.max_case_rounds),
        "finish_repeat": ("repeated-amendment", "the same amendment was made twice"),
        "finish_stop": ("adjudicated-stop",
                        "the adjudicator could not attribute the failure"),
    }
    for name, (outcome, reason) in ends.items():
        graph.add_node(name, (lambda o, r: (lambda s: _finish(o, r)(s, settings)))(outcome, reason))

    graph.add_edge(START, "check_spec" if settings.start_from == "spec" else "write_spec")
    graph.add_edge("write_spec", "check_spec")
    graph.add_conditional_edges("check_spec", lambda s: _route_after_check(s, settings),
                                {"implement": "implement", "write_spec": "write_spec",
                                 "finish_guard": "finish_guard",
                                 "finish_stalled": "finish_stalled",
                                 "finish_verifier": "finish_verifier"})
    graph.add_conditional_edges("implement", lambda s: _route_after_implement(s, settings),
                                {"test": "test", "finish_impl": "finish_impl"})
    graph.add_edge("test", "record_failure")
    graph.add_conditional_edges("record_failure", lambda s: _route_after_test(s, settings),
                                {"diagnose": "diagnose", "finish_pass": "finish_pass",
                                 "finish_cap": "finish_cap",
                                 "finish_cases": "finish_cases",
                                 "finish_unchanged": "finish_unchanged",
                                 "finish_cases_stuck": "finish_cases_stuck",
                                 "finish_uncovered": "finish_uncovered"})
    graph.add_conditional_edges("diagnose", lambda s: _route_after_diagnose(s, settings),
                                {"write_spec": "write_spec", "implement": "implement",
                                 "test": "test", "finish_stop": "finish_stop",
                                 "finish_cap": "finish_cap", "finish_repeat": "finish_repeat"})
    for name in ("finish_pass", "finish_cap", "finish_guard", "finish_stalled",
                 "finish_verifier", "finish_cases", "finish_impl", "finish_uncovered",
                 "finish_unchanged", "finish_cases_stuck", "finish_repeat", "finish_stop"):
        graph.add_edge(name, END)
    return graph.compile()
