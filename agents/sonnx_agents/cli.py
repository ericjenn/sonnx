"""The command line of the multi-agent specification system.

    python agents/run.py --op Neg

writes the specification of the ONNX **Neg** operator, has it checked against the profile's
guidelines, has a blind agent implement it, has a test agent check the implementation
against the document and against ONNX Runtime, and lets an adjudicator send the correction
back to whichever of the two is at fault — until a fresh blind implementation passes, or
one of the guards of the profile's skill stops the loop.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .config import AGENTS_DIR, DEFAULT_PYTHON, REPO, Settings
from .graph import build_graph


def parse_args(argv=None) -> Settings:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], add_help=True)
    ap.add_argument("--op", required=True, help="the ONNX operator type, e.g. Neg")
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--model", default=None, help="the model (default: $SONNX_AGENT_MODEL)")
    ap.add_argument("--base-url", default=None, help="the Messages API endpoint")
    ap.add_argument("--max-iterations", type=int, default=5,
                    help="implement -> test rounds (the skill's cap, default 5)")
    ap.add_argument("--max-spec-rounds", type=int, default=3,
                    help="writer -> compliance-verifier rounds (default 3)")
    ap.add_argument("--max-case-repairs", type=int, default=3,
                    help="times the test agent may repair a case module that raises")
    ap.add_argument("--harness-timeout", type=int, default=1800,
                    help="seconds one harness run may take before it is killed and reported "
                         "as a case module that cannot be run (default 1800)")
    ap.add_argument("--max-case-rounds", type=int, default=2,
                    help="times the test agent may rewrite its case set after the adjudicator "
                         "attributed a failure to it (default 2); the run then stops with "
                         "cases-stuck")
    ap.add_argument("--max-module-retries", type=int, default=2,
                    help="times the blind implementer or the test agent may answer again when "
                         "its answer is not a usable module (empty, not Python, without the "
                         "entry point; or without `cases()` and `COVERAGE`)")
    ap.add_argument("--max-writer-retries", type=int, default=2,
                    help="times the writer may answer again when its answer is cut off")
    ap.add_argument("--max-review-retries", type=int, default=1,
                    help="times the verifier may answer again when its verdict is unreadable")
    ap.add_argument("--max-review-tokens", type=int, default=64000,
                    help="output budget of one compliance review")
    ap.add_argument("--thinking", choices=("auto", "off"), default="auto",
                    help="may the model deliberate before it writes (writer, implementer, tester)")
    ap.add_argument("--verdict-thinking", choices=("auto", "off"), default="off",
                    help="may the model deliberate before a verdict (verifier, adjudicator)")
    ap.add_argument("--start-from", choices=("scratch", "spec"), default="scratch",
                    help="draft the specification, or start from the file on disk")
    ap.add_argument("--spec", default=None, help="the specification path (default: ops/<op>.md)")
    ap.add_argument("--run-dir", default=None, help="where the run's artifacts go")
    ap.add_argument("--onnx-doc", default=None,
                    help="a file holding the ONNX definition, instead of the schema of "
                         "the installed onnx package")
    ap.add_argument("--no-fetch", action="store_true", help="do not fetch the ONNX page")
    ap.add_argument("--python", default=DEFAULT_PYTHON,
                    help="the interpreter that runs lint_spec.py and specloop.py")
    ap.add_argument("--force", action="store_true",
                    help="allow writing over an existing ops/<op>.md")
    ap.add_argument("--publish", action="store_true",
                    help="on convergence, copy the artifacts into ops/ and verification/")
    ap.add_argument("--publish-only", action="store_true",
                    help="publish the converged run recorded in --json-state, without "
                         "running the model or writing a new document")
    ap.add_argument("--json-state", default=None, help="write the final state here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    settings = Settings(
        op=args.op,
        opset=args.opset,
        max_iterations=args.max_iterations,
        max_spec_rounds=args.max_spec_rounds,
        max_case_repairs=args.max_case_repairs,
        max_case_rounds=args.max_case_rounds,
        harness_timeout=args.harness_timeout,
        max_module_retries=args.max_module_retries,
        max_writer_retries=args.max_writer_retries,
        max_review_retries=args.max_review_retries,
        max_review_tokens=args.max_review_tokens,
        thinking=args.thinking,
        verdict_thinking=args.verdict_thinking,
        start_from=args.start_from,
        spec_path=Path(args.spec) if args.spec else None,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        onnx_doc=Path(args.onnx_doc) if args.onnx_doc else None,
        fetch_onnx_doc=not args.no_fetch,
        python=args.python,
        force=args.force,
        verbose=not args.quiet,
    )
    if args.model:
        settings.model = args.model
    if args.base_url:
        settings.base_url = args.base_url
    settings.publish = args.publish
    settings.publish_only = args.publish_only
    settings.json_state = args.json_state
    return settings


def publish(settings: Settings, state: dict) -> None:
    """Copy a converged run's artifacts where the repository keeps them."""
    spec_target = REPO / "ops" / ("%s.md" % settings.entry)
    cases_target = REPO / "verification" / settings.entry / "cases.py"
    loop_target = REPO / "verification" / settings.entry / "loop"
    spec_target.parent.mkdir(parents=True, exist_ok=True)
    cases_target.parent.mkdir(parents=True, exist_ok=True)
    loop_target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(state["spec_path"], spec_target)
    shutil.copyfile(state["cases_path"], cases_target)
    for iteration in range(1, int(state.get("iteration", 0)) + 1):
        source = settings.impl_path(iteration)
        if source.exists():
            shutil.copyfile(source, loop_target / source.name)
    # every harness report, not only the first of each iteration: the runs after the first
    # are the evidence of what a revised case set or a repaired module did
    for source in sorted(settings.resolved_run_dir().glob("iter-*-report*.json")):
        shutil.copyfile(source, loop_target / source.name)
    print("published: %s, %s, %s" % (spec_target, cases_target, loop_target))


def summarise(settings: Settings, state: dict) -> int:
    record = state.get("result") or {}
    outcome = record.get("outcome", "unknown")
    spec_path = Path(state.get("spec_path", ""))
    report = state.get("report") or {}
    cases = report.get("cases") or {}
    print("\n%s" % ("=" * 78))
    print("operator     : %s (ONNX opset %d)" % (settings.op, settings.opset))
    print("outcome      : %s" % outcome)
    print("specification: %s (md5 %s)" % (spec_path, state.get("spec_md5")))
    print("rounds       : %d spec round(s), %d implement/test iteration(s)"
          % (state.get("spec_round", 0), state.get("iteration", 0)))
    print("cases        : %s run, %s passed, %s mismatched, %s open"
          % (cases.get("total", "-"), cases.get("passed", "-"),
             len(cases.get("mismatched", [])), len(cases.get("open", []))))
    print("coverage     : %s" % json.dumps(state.get("coverage") or {}))
    # why an artifact is missing, when an agent's answer was not one: the outcome names the
    # missing artifact, and a reader of the run should not have to open the record to learn
    # that the reason is the model's answer and not the document
    if record.get("impl_unusable") or record.get("cases_unusable"):
        print("unusable     : %s"
              % (record.get("impl_unusable") or record.get("cases_unusable")))
    if state.get("stopped"):
        print("stopped      : %s" % state["stopped"])
    print("=" * 78)
    return 0 if outcome == "converged" else 1


def main(argv=None) -> int:
    settings = parse_args(argv)
    settings.resolved_run_dir().mkdir(parents=True, exist_ok=True)
    print("SONNX multi-agent specification system")
    print("model: %s at %s" % (settings.model, settings.base_url or "(the default endpoint)"))
    print("run  : %s" % settings.resolved_run_dir())

    if settings.publish_only:
        # publishing the run that converged, not a new one: the document the caller read the
        # verdict for is the document that gets published
        return publish_finished(settings)

    graph = build_graph(settings)
    initial = {
        "op": settings.op,
        "opset": settings.opset,
        "spec_path": str(settings.resolved_spec_path()),
        "iteration": 0,
        "spec_round": 0,
        "history": [],
        "trace": [],
        "seen_amendments": [],
    }
    state = graph.invoke(initial, {"recursion_limit": 200})

    if settings.json_state:
        payload = {k: v for k, v in state.items() if k != "result"}
        payload["outcome"] = (state.get("result") or {}).get("outcome")
        Path(settings.json_state).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    code = summarise(settings, state)
    if settings.publish and (state.get("result") or {}).get("outcome") == "converged":
        publish(settings, state)
    return code


def publish_finished(settings: Settings) -> int:
    """Publish the run recorded in `--json-state` — no model call, no new document.

    Only a run whose own record says `converged` is published; the state file carries the
    outcome precisely so that this check can be made without re-running anything.  An
    existing `ops/<op>.md` is never overwritten unless `--force` was asked for: the
    repository is not under version control.
    """
    if not settings.json_state:
        print("--publish-only needs --json-state <the finished run's state file>")
        return 2
    state = json.loads(Path(settings.json_state).read_text(encoding="utf-8"))
    outcome = state.get("outcome")
    if outcome is None:
        # A state file written before the outcome was recorded in it.  The run's own record
        # is the authority, but only a record for *this* document may be used: a record for
        # another attempt in the same directory would publish the wrong verdict.
        md5 = state.get("spec_md5")
        matches = [p for p in sorted(settings.resolved_run_dir().glob("run-*.json"))
                   if md5 and json.loads(p.read_text(encoding="utf-8")).get("spec_md5") == md5]
        if len(matches) == 1:
            outcome = json.loads(matches[0].read_text(encoding="utf-8")).get("outcome")
            print("outcome taken from %s (the record of this document): %s"
                  % (matches[0].name, outcome))
        else:
            print("%s does not record an outcome, and no single run record in %s is for the "
                  "same document (md5 %s): nothing was published"
                  % (settings.json_state, settings.resolved_run_dir(), md5))
            return 1
    if outcome != "converged":
        print("the run recorded in %s is %r, not converged: nothing was published"
              % (settings.json_state, outcome))
        return 1
    target = REPO / "ops" / ("%s.md" % settings.entry)
    if target.exists() and not settings.force:
        print("%s exists: not overwritten (pass --force to publish over it)" % target)
        return 1
    publish(settings, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
