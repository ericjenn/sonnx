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
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from .config import DEFAULT_PYTHON, Settings
from .graph import build_graph
from . import lean as lean_support


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
                    help="output budget of one compliance review or Lean formalization")
    ap.add_argument("--max-lean-retries", type=int, default=2,
                    help="repairs after the first Lean formalizer answer (default 2)")
    ap.add_argument("--lean-timeout", type=int, default=300,
                    help="seconds for one Lean compiler invocation (default 300)")
    ap.add_argument("--lake", default=None, help="Lake executable (default: discover from PATH/elan)")
    ap.add_argument("--lean", action="store_true",
                    help="generate and validate Lean before the normal Python loop")
    ap.add_argument("--lean-only", action="store_true",
                    help="generate and validate Lean, without Python/ONNX Runtime stages")
    ap.add_argument("--c", dest="cgen", action="store_true",
                    help="generate C from the Lean descriptor and test it (requires --lean)")
    ap.add_argument("--cc", default="cc",
                    help="the C compiler that builds the kernels and the test program "
                         "(default: cc)")
    ap.add_argument("--max-c-retries", type=int, default=2,
                    help="times the C tester may answer again when its program is not usable "
                         "(empty, not C, does not compile, compares nothing; default 2)")
    ap.add_argument("--c-timeout", type=int, default=120,
                    help="seconds for one C compilation or one test-binary run (default 120)")
    ap.add_argument("--no-lean-conformance", action="store_true",
                    help="do not generate or run the Lean-vs-corpus conformance check")
    ap.add_argument("--no-lean-conformance-generate", action="store_true",
                    help="do not generate the executable Lean runner and driver; run an "
                         "existing or explicitly configured one instead")
    ap.add_argument("--lean-conformance-script", default=None,
                    help="an existing driver to run instead of generating one")
    ap.add_argument("--lean-conformance-timeout", type=int, default=1800,
                    help="seconds for one Lean conformance run (default 1800)")
    ap.add_argument("--max-conformance-retries", type=int, default=2,
                    help="repairs after the first generated Lean runner answer (default 2)")
    ap.add_argument("--lean-conformance-compare", choices=("bits", "ulp", "relative"),
                    default="bits",
                    help="how float outputs are compared (default: bits, NaN equal to NaN)")
    ap.add_argument("--lean-conformance-ulp", type=int, default=2,
                    help="ULP tolerance when --lean-conformance-compare=ulp (default 2)")
    ap.add_argument("--lean-conformance-relative", type=float, default=1e-6,
                    help="relative tolerance when --lean-conformance-compare=relative "
                         "(default 1e-6)")
    ap.add_argument("--check-lean", action="store_true",
                    help="revalidate an existing Lean artifact without a model call")
    ap.add_argument("--lean-artifact", default=None,
                    help="Lean file for --check-lean (default: the run's FormalSpec.lean)")
    ap.add_argument("--thinking", choices=("auto", "off"), default="auto",
                    help="may the model deliberate before it writes (writer, implementer, tester)")
    ap.add_argument("--verdict-thinking", choices=("auto", "off"), default="off",
                    help="may the model deliberate before a verdict (verifier, adjudicator)")
    ap.add_argument("--start-from", choices=("scratch", "spec"), default="scratch",
                    help="draft the specification, or start from the file on disk")
    ap.add_argument("--spec", default=None,
                    help="the specification path (default: ops/<op>/<op>.md)")
    ap.add_argument("--run-dir", default=None,
                    help="where the run's artifacts go (default: ops/<op>/)")
    ap.add_argument("--onnx-doc", default=None,
                    help="a file holding the ONNX definition, instead of the schema of "
                         "the installed onnx package")
    ap.add_argument("--no-fetch", action="store_true", help="do not fetch the ONNX page")
    ap.add_argument("--python", default=DEFAULT_PYTHON,
                    help="the interpreter that runs lint_spec.py and specloop.py")
    ap.add_argument("--force", action="store_true",
                    help="allow writing over an existing operator document")
    ap.add_argument("--publish", action="store_true",
                    help="on convergence, validate the run and place its artifacts in "
                         "ops/<op>/ (they are already there unless --run-dir was used)")
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
        max_lean_retries=args.max_lean_retries,
        lean_timeout=args.lean_timeout,
        lake=args.lake,
        lean=args.lean or args.lean_only,
        lean_only=args.lean_only,
        cgen=args.cgen,
        c_compiler=args.cc,
        max_c_retries=args.max_c_retries,
        c_timeout=args.c_timeout,
        lean_conformance=not args.no_lean_conformance,
        lean_conformance_generate=not args.no_lean_conformance_generate,
        lean_conformance_script=(
            Path(args.lean_conformance_script) if args.lean_conformance_script else None),
        lean_conformance_timeout=args.lean_conformance_timeout,
        max_conformance_retries=args.max_conformance_retries,
        lean_conformance_compare=args.lean_conformance_compare,
        lean_conformance_ulp=args.lean_conformance_ulp,
        lean_conformance_relative=args.lean_conformance_relative,
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
    settings.lean_artifact = args.lean_artifact
    settings.check_lean = args.check_lean
    if args.check_lean and (args.lean or args.lean_only or args.publish or args.publish_only):
        ap.error("--check-lean is mutually exclusive with generation and publication")
    if args.cgen and not (args.lean or args.lean_only):
        ap.error("--c requires --lean or --lean-only: the C is rendered from the Lean "
                 "descriptor")
    return settings


def check_existing_lean(settings: Settings) -> int:
    """Revalidate freshness, traceability and compilation without a model call."""
    library = lean_support.build_library(
        settings.lake, timeout=max(settings.lean_timeout, 600))
    if not library["ok"]:
        print(json.dumps({"ok": False,
                          "findings": ["the shared Sonnx library does not build: %s"
                                       % library["stdout"][-2000:]]}, indent=2))
        return 1
    artifact = Path(settings.lean_artifact) if settings.lean_artifact else settings.lean_path()
    report = lean_support.check_files(settings.resolved_spec_path(), [artifact],
                                      settings.lake, settings.lean_timeout)
    print(json.dumps(report, indent=2))
    if report["ok"] and not report["complete"]:
        print("Lean artifact is valid but semantically partial; it is not publishable as complete.")
    if not report["ok"]:
        return 1
    conformance = lean_support.run_conformance(settings)
    summary = conformance.get("report", {}).get("summary") or {}
    print("Lean conformance: %s%s%s" % (
        conformance["status"],
        "" if not summary else (
            " (%s/%s cases, %s unsupported)"
            % (summary.get("passed", "?"), summary.get("total", "?"),
               summary.get("unsupported", "?"))),
        "" if not conformance.get("kind") else " [%s]" % conformance["kind"]))
    if conformance.get("error"):
        print("Lean conformance detail: %s" % conformance["error"])
    return 1 if conformance.get("blocking") else 0


def _copy_atomic_bytes(data: bytes, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _copy_atomic(source: Path, target: Path) -> None:
    _copy_atomic_bytes(source.read_bytes(), target)


def _same_file(source: Path, target: Path) -> bool:
    """True when copying ``source`` to ``target`` would be a copy onto itself.

    A run now writes into its operator's folder by default, so the publication target and
    the artifact the run wrote are often the same file.  There is nothing to copy then,
    and the overwrite guards (which exist to protect files the pipeline did not write)
    have nothing to protect: the file is the run's own generated artifact.
    """
    source, target = Path(source), Path(target)
    try:
        return os.path.samefile(source, target)
    except OSError:
        return source.resolve() == target.resolve()


def _conformance_publication(settings: Settings, state: dict) -> dict[str, tuple[Path, Path]]:
    """The generated conformance files: published name -> (source, destination).

    The run directory and the operator's folder share one relative layout, so the
    manifest's relative paths stay valid after publication and its digests still match the
    copied bytes.  When the run already is that folder, every destination is its own
    source and nothing is copied.
    """
    if not state.get("lean_conformance_generated"):
        return {}
    sources = {
        "runner": Path(state.get("lean_conformance_runner") or ""),
        "corpus": Path(state.get("lean_conformance_corpus") or ""),
        "driver": Path(state.get("lean_conformance_driver") or ""),
        "manifest": Path(state.get("lean_conformance_manifest") or ""),
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise RuntimeError("the run names conformance artifacts that are not on disk: %s"
                           % ", ".join(missing))
    target_dir = settings.lean_publication_dir()
    targets = {
        "runner": target_dir / "LeanConformance.lean",
        "corpus": target_dir / "cases.py",
        "driver": settings.lean_publication_driver_path(),
        "manifest": target_dir / "conformance.manifest.json",
    }
    return {name: (sources[name], targets[name]) for name in sources}


def _guard_conformance_overwrite(settings: Settings, files: dict[str, tuple[Path, Path]]) -> None:
    """Never replace a file the pipeline did not generate, or one edited since.

    A target that is the run's own file (the default: the run writes into the operator's
    folder) is not replaced at all, so it needs no guard.
    """
    if not any(not _same_file(source, target) for source, target in files.values()):
        return
    manifest = files["manifest"][1]
    previous = None
    if manifest.exists():
        try:
            previous = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "refusing to overwrite conformance artifacts without a generated manifest: %s"
                % exc)
    for name in ("runner", "corpus", "driver"):
        source, target = files[name]
        if _same_file(source, target) or not target.exists() or settings.force:
            continue
        allowed = (previous is not None
                   and previous.get("generated_by") == "sonnx-lean-pipeline-conformance"
                   and previous.get(name + "_sha256")
                   == hashlib.sha256(target.read_bytes()).hexdigest())
        if not allowed:
            raise RuntimeError(
                "refusing to overwrite %s: it is not an unmodified generated artifact "
                "(pass --force)" % target)


def _publish_lean(settings: Settings, state: dict) -> list[Path]:
    source = Path(state.get("lean_path") or "")
    manifest = Path(state.get("lean_trace_path") or "")
    if not source.is_file() or not manifest.is_file():
        raise RuntimeError("the run has no Lean source and trace manifest to publish")
    report = lean_support.check_files(Path(state["spec_path"]), [source],
                                      settings.lake, settings.lean_timeout)
    if not report["ok"] or not report["complete"] or not state.get("lean_ok"):
        raise RuntimeError("the Lean artifact is stale, invalid, or partial: %s"
                           % "; ".join(report.get("findings", []))[:1200])
    conformance = lean_support.run_conformance(settings,
                                               spec_path=Path(state["spec_path"]))
    if conformance.get("blocking"):
        raise RuntimeError("the Lean conformance check failed: %s"
                           % (conformance.get("error") or conformance["status"]))
    target_dir = settings.lean_publication_dir()
    target = target_dir / "FormalSpec.lean"
    target_manifest = target_dir / "FormalSpec.trace.json"
    source_in_place = _same_file(source, target)
    manifest_in_place = _same_file(manifest, target_manifest)
    if not source_in_place and target.exists():
        if not settings.force:
            raise RuntimeError("%s exists: not overwritten (pass --force for a generated target)" % target)
        try:
            previous = json.loads(target_manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError("refusing to overwrite Lean without generated provenance: %s" % exc)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if previous.get("generated_by") != "sonnx-lean-pipeline" or \
                previous.get("artifact_sha256") != digest:
            raise RuntimeError("refusing to overwrite hand-written or hand-edited Lean")

    # Every file this publication will replace, with its previous bytes (None: it did not
    # exist).  A failure anywhere restores all of them.  A run that wrote into the
    # operator's folder is already in place: validating it is the whole publication.
    plan: list[tuple[Path, bytes, bytes | None]] = []
    published_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    published_manifest["artifact_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    temporary_manifest = target_dir / ".FormalSpec.trace.json.tmp"
    target_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_in_place:
        temporary_manifest.write_text(
            json.dumps(published_manifest, indent=2) + "\n", encoding="utf-8")
    if not source_in_place:
        plan.append((target, source.read_bytes(),
                     target.read_bytes() if target.exists() else None))
    if not manifest_in_place:
        plan.append((target_manifest, temporary_manifest.read_bytes(),
                     target_manifest.read_bytes() if target_manifest.exists() else None))

    conformance_files = _conformance_publication(settings, state)
    if conformance_files:
        _guard_conformance_overwrite(settings, conformance_files)
        for name in ("runner", "corpus", "driver", "manifest"):
            origin, destination = conformance_files[name]
            if _same_file(origin, destination):
                continue
            plan.append((destination, origin.read_bytes(),
                         destination.read_bytes() if destination.exists() else None))

    try:
        for destination, data, _ in plan:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_atomic_bytes(data, destination)
        checked = lean_support.check_files(Path(state["spec_path"]), [target],
                                           settings.lake, settings.lean_timeout)
        if not checked["ok"] or not checked["complete"]:
            raise RuntimeError("post-publication Lean validation failed")
        if conformance_files:
            verdict = lean_support.validate_conformance_manifest(
                conformance_files["driver"][1], Path(state["spec_path"]))
            if not verdict["ok"]:
                raise RuntimeError(
                    "post-publication conformance manifest validation failed: %s"
                    % "; ".join(verdict["findings"]))
    except Exception:
        for destination, _, previous in reversed(plan):
            if previous is None:
                destination.unlink(missing_ok=True)
            else:
                destination.write_bytes(previous)
        raise
    finally:
        temporary_manifest.unlink(missing_ok=True)

    published = [target, target_manifest]
    if conformance_files:
        published.extend(destination for _, destination in conformance_files.values())
    return published


def _publish_cgen(settings: Settings, state: dict) -> list[Path]:
    """The generated C and the C test program, when the C stage passed.

    The report is published with them: it carries the digests of the specification, the
    header, the source and the test, and the exact lines the program printed.
    """
    if not (settings.cgen and state.get("cgen_ok")):
        return []
    sources = {
        "sonnx_kernels.h": Path(state.get("cgen_header") or settings.c_header_path()),
        "sonnx_kernels.c": Path(state.get("cgen_source") or settings.c_source_path()),
        "c_tests.c": Path(state.get("cgen_test") or settings.c_test_path()),
        "c-report.json": settings.c_report_path(),
        "cgen-trace.json": settings.c_trace_path(),
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise RuntimeError("the run names C artifacts that are not on disk: %s"
                           % ", ".join(missing))
    target_dir = settings.c_publication_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    published = []
    for name in ("sonnx_kernels.h", "sonnx_kernels.c", "c_tests.c", "c-report.json",
                 "cgen-trace.json"):
        target = target_dir / name
        if not _same_file(sources[name], target):
            shutil.copyfile(sources[name], target)
        published.append(target)
    return published


def publish(settings: Settings, state: dict) -> None:
    """Validate a successful run and place its artifacts in the operator's folder.

    A default run already wrote them there; the copy is what an explicit ``--run-dir``
    needs, and a copy onto the run's own file is skipped rather than attempted.
    """
    published = _publish_lean(settings, state) if settings.lean else []
    published.extend(_publish_cgen(settings, state))
    if not settings.lean_only:
        operator_dir = settings.ops_root() / settings.entry
        spec_target = settings.published_spec_path()
        cases_target = settings.published_cases_path()
        loop_target = operator_dir / "loop"
        spec_target.parent.mkdir(parents=True, exist_ok=True)
        loop_target.mkdir(parents=True, exist_ok=True)
        for source, target in ((state["spec_path"], spec_target),
                               (state["cases_path"], cases_target)):
            if not _same_file(source, target):
                shutil.copyfile(source, target)
        published.extend([spec_target, cases_target, loop_target])
        implementations = [Path(entry["path"]) for entry in state.get("history") or []
                           if entry.get("agent") == "implementer" and entry.get("path")]
        if not implementations:
            implementations = [settings.impl_path(iteration)
                               for iteration in range(1, int(state.get("iteration", 0)) + 1)]
        for source in implementations:
            target = loop_target / source.name
            if source.exists() and not _same_file(source, target):
                shutil.copyfile(source, target)
        # Every report named by this run is evidence; unrelated files in a shared run
        # directory are not. Legacy records without the list retain the old fallback.
        named = [Path(p) for p in state.get("reports") or []]
        if named:
            sources = named
        else:
            sources = sorted(settings.loop_dir().glob("iter-*-report*.json"))
        for source in sources:
            target = loop_target / source.name
            if source.exists() and not _same_file(source, target):
                shutil.copyfile(source, target)
    print("published: %s" % ", ".join(str(path) for path in published))


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
    if settings.lean:
        print("Lean         : %s" % (state.get("lean_path") or "(no artifact)"))
        print("Lean coverage: %s" % json.dumps(state.get("lean_coverage") or {}))
        status = state.get("lean_conformance_status")
        if status:
            summary = (state.get("lean_conformance_report") or {}).get("summary") or {}
            detail = ""
            if summary:
                detail = " (%s/%s cases, %s unsupported)" % (
                    summary.get("passed", "?"), summary.get("total", "?"),
                    summary.get("unsupported", "?"))
            if state.get("lean_conformance_corpus_source"):
                detail += " [%s corpus]" % state["lean_conformance_corpus_source"]
            print("Lean conformance: %s%s" % (status, detail))
        if state.get("lean_conformance_runner"):
            print("Lean runner : %s" % state["lean_conformance_runner"])
        if state.get("lean_conformance_driver"):
            print("Lean driver : %s" % state["lean_conformance_driver"])
        if state.get("lean_conformance_error"):
            print("Lean conformance error: %s" % state["lean_conformance_error"])
        if state.get("lean_error"):
            print("Lean error   : %s" % state["lean_error"])
    if settings.cgen:
        print("C            : %s" % (state.get("cgen_status") or "-"))
        if state.get("cgen_header"):
            print("C header     : %s" % state["cgen_header"])
        if state.get("cgen_source"):
            print("C source     : %s" % state["cgen_source"])
        checks = (state.get("cgen_report") or {}).get("checks") or {}
        if checks:
            print("C tests      : %s/%s checks passed, %s failed"
                  % (checks.get("passed", "?"), checks.get("total", "?"),
                     checks.get("failed", "?")))
        if state.get("cgen_test"):
            print("C test       : %s" % state["cgen_test"])
        if state.get("cgen_trace"):
            print("C trace      : %s" % state["cgen_trace"])
        if state.get("cgen_error"):
            print("C error      : %s" % state["cgen_error"])
    # why an artifact is missing, when an agent's answer was not one: the outcome names the
    # missing artifact, and a reader of the run should not have to open the record to learn
    # that the reason is the model's answer and not the document
    if record.get("impl_unusable") or record.get("cases_unusable"):
        print("unusable     : %s"
              % (record.get("impl_unusable") or record.get("cases_unusable")))
    if state.get("stopped"):
        print("stopped      : %s" % state["stopped"])
    print("=" * 78)
    return 0 if outcome in {"converged", "formalized"} else 1


def existing_document_problem(settings: Settings) -> str:
    """Why a fresh run must not start on an existing operator document, or "".

    The run writes into the operator's own folder, so the document the writer drafts and
    the document the folder already holds are the same path.  A scratch run would replace
    an accepted document silently; it has to say which it wants — read and possibly amend
    it (``--start-from spec``), or replace it (``--force``).  An explicit ``--spec`` or
    ``--run-dir`` is the caller naming the file, and is left alone.
    """
    if settings.check_lean or settings.publish_only or settings.lean_only:
        return ""
    if settings.start_from != "scratch" or settings.force:
        return ""
    if settings.spec_path is not None or settings.run_dir is not None:
        return ""
    target = settings.resolved_spec_path()
    if target.exists():
        return ("%s exists: not overwritten (use --start-from spec to read it, or --force "
                "to replace it)" % target)
    return ""


def main(argv=None) -> int:
    settings = parse_args(argv)
    problem = existing_document_problem(settings)
    if problem:
        print(problem)
        return 2
    settings.resolved_run_dir().mkdir(parents=True, exist_ok=True)
    print("SONNX multi-agent specification system")
    print("model: %s at %s" % (settings.model, settings.base_url or "(the default endpoint)"))
    print("run  : %s" % settings.resolved_run_dir())

    if settings.publish_only:
        # publishing the run that converged, not a new one: the document the caller read the
        # verdict for is the document that gets published
        return publish_finished(settings)

    if settings.check_lean:
        return check_existing_lean(settings)

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
    outcome = (state.get("result") or {}).get("outcome")
    if settings.publish and outcome in ({"formalized"} if settings.lean_only else {"converged"}):
        try:
            publish(settings, state)
        except (OSError, RuntimeError) as exc:
            print("publication refused: %s" % exc)
            return 1
    return code


def publish_finished(settings: Settings) -> int:
    """Publish the run recorded in `--json-state` — no model call, no new document.

    Only a run whose own record says `converged` is published; the state file carries the
    outcome precisely so that this check can be made without re-running anything.  An
    existing operator document is never overwritten unless `--force` was asked for —
    except when it is the document this very run wrote into the operator's folder.
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
    if outcome not in ({"formalized"} if settings.lean_only else {"converged"}):
        print("the run recorded in %s is %r, not successful in this mode: nothing was published"
              % (settings.json_state, outcome))
        return 1
    if settings.lean_only:
        try:
            publish(settings, state)
        except (OSError, RuntimeError) as exc:
            print("publication refused: %s" % exc)
            return 1
        return 0
    target = settings.published_spec_path()
    if _same_file(Path(state.get("spec_path") or ""), target):
        # The run wrote into its operator's folder: there is no published copy to protect.
        publish(settings, state)
        return 0
    if target.exists() and not settings.force:
        print("%s exists: not overwritten (pass --force to publish over it)" % target)
        return 1
    publish(settings, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
