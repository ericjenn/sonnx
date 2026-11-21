#!/usr/bin/env python3
"""The generated Lean conformance artifacts: protocol, corpus, manifest, driver.

Nothing here needs a model, ONNX Runtime or a Lean toolchain: the Lean process and the
reference runner are replaced by fakes, and what is tested is the part the pipeline owns —
the wire encoding, the freshness manifest, which driver is selected, and what a run of the
generated driver reports.  Run it with

    ~/Venvs/sonnx/bin/python agents/tests/test_lean_conformance.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonnx_agents import cli, graph, harness, lean, lean_conformance as lc  # noqa: E402
from sonnx_agents.config import REPO, Settings  # noqa: E402

# What ``lake env lean --run`` prints on stdout before main runs, measured on 2026-11-15
# with Lean 4.35: a generated runner that used ``String.trim`` produced this block on every
# invocation, and the smoke test read its lines as protocol responses.
WARNING_BLOCK = (
    "/tmp/LeanConformance.lean:188:19: warning: `String.trim` has been deprecated: "
    "Use `String.trimAscii` instead\n"
    "\n"
    "Note: The updated constant has a different type:\n"
    "  String → String.Slice\n"
    "instead of\n"
    "  String → String\n"
)


def check(name: str, condition: bool) -> None:
    assert condition, name
    print("  ok  %s" % name)


def check_equal(name: str, got, want) -> None:
    assert got == want, "%s: got %r, wanted %r" % (name, got, want)
    print("  ok  %-46s %r" % (name, got))


def spec_file(work: Path) -> Path:
    path = work / "neg.md"
    path.write_text("# Neg\n\nThe negation of a tensor.\n", encoding="utf-8")
    return path


def generated_tree(work: Path, spec: Path) -> Settings:
    """A run directory holding a coherent generated conformance artifact set."""
    settings = Settings(op="Neg", run_dir=work, spec_path=spec, verbose=False)
    (work / "lean").mkdir(parents=True, exist_ok=True)
    (work / "lean" / "FormalSpec.lean").write_text("def negSpec : Prop := True\n",
                                                  encoding="utf-8")
    (work / "lean" / "LeanConformance.lean").write_text(
        "def main : IO Unit := pure ()\n", encoding="utf-8")
    (work / "lean" / "cases.py").write_text(
        "import numpy as np\n\ndef cases():\n    yield 'one', [np.array([1.0])]\n",
        encoding="utf-8")
    (work / "lean_conformance.py").write_text(
        "#!/usr/bin/env python3\n"
        "# @generated-by sonnx-lean-pipeline-conformance\n"
        "# @protocol sonnx-conformance/1\n",
        encoding="utf-8")
    lean.write_conformance_manifest(
        settings, spec, "published", {"outputs": ["out"], "output_types": {},
                                      "dtypes": ["float64"]},
        settings.lean_conformance_runner_path(), settings.lean_conformance_driver_path())
    return settings


class PublishedSettings(Settings):
    """A settings whose run directory holds no driver, only the published one."""

    def __init__(self, published: Path, **kwargs):
        super().__init__(**kwargs)
        self._published = published

    def lean_conformance_driver_path(self) -> Path:
        return self._published.parent / "no-run-dir" / "lean_conformance.py"

    def lean_publication_driver_path(self) -> Path:
        return self._published


def protocol_tests() -> None:
    print("\n1. the wire protocol")
    check_equal("rank-0 shape is '-'", lc.shape_text(()), "-")
    check_equal("rank-2 shape", lc.shape_text((2, 3)), "2,3")
    check_equal("empty data is '-'", lc.data_text([]), "-")

    ints = np.array([[1, 2], [-3, 4]], dtype=np.int32)
    text = lc.tensor_text(ints)
    check_equal("integer tensor prefix", text.split("|")[:2], ["i32", "2,2"])
    out = lc.decode_tensor(text)
    check_equal("integer tensor shape", out.shape, (2, 2))
    again = lc._actual_array(out, ints.dtype)
    check("integer tensor round-trips", np.array_equal(again, ints))

    floats = np.array([1.0, -0.0, float("nan")], dtype=np.float64)
    out = lc.decode_tensor(lc.tensor_text(floats))
    ok, detail = lc.compare_output(floats, out, {"compare": "bits"})
    check("NaN is equal to NaN, bits otherwise (%s)" % detail, ok)

    from fractions import Fraction
    rats = np.array([Fraction(1, 3), Fraction(-2, 5)], dtype=object)
    out = lc.decode_tensor(lc.tensor_text(rats))
    check("rational tensor round-trips",
          lc.compare_output(rats, out, {"compare": "bits"})[0])

    line = lc.encode_request(7, [ints], {"axis": 1, "keepdims": 0})
    fields = line.split("\t")
    check_equal("request id", fields[0], "7")
    check_equal("request attrs are sorted", fields[1], "axis=1;keepdims=0")
    check_equal("request arity", fields[2], "1")
    check_equal("request tensor field", fields[3], text)

    response = lc.decode_response("%s\tOK\t1\t%s" % (7, text))
    check("OK response decodes", response.id == "7" and len(response.outputs) == 1)
    check_equal("UNSUPPORTED decodes",
                lc.decode_response("7\tUNSUPPORTED\t0\tno float16").status, "UNSUPPORTED")
    check_equal("MALFORMED decodes",
                lc.decode_response("7\tMALFORMED\t0\tbad line").message, "bad line")
    for bad in ("7\tOK\t2\t%s" % text, "7\tWHAT\t0\tx", "no-tabs"):
        try:
            lc.decode_response(bad)
        except lc.ProtocolError:
            pass
        else:
            raise AssertionError("accepted malformed response %r" % bad)
    print("  ok  malformed responses are rejected")

    one = np.array([1.0], dtype=np.float64)
    near = np.nextafter(one, np.array([2.0]))
    out = lc.decode_tensor(lc.tensor_text(near))
    check("one ULP is within a 1-ULP tolerance",
          lc.compare_output(one, out, {"compare": "ulp", "ulp": 1})[0])
    far = np.nextafter(np.nextafter(one, np.array([2.0])), np.array([2.0]))
    out = lc.decode_tensor(lc.tensor_text(far))
    check("two ULP is beyond a 1-ULP tolerance",
          not lc.compare_output(one, out, {"compare": "ulp", "ulp": 1})[0])
    out = lc.decode_tensor(lc.tensor_text(one * (1 + 1e-9)))
    check("relative tolerance accepts a small drift",
          lc.compare_output(one, out, {"compare": "relative", "relative": 1e-6})[0])

    try:
        lc.tensor_text(np.array([1], dtype=np.float16))
    except lc.UnsupportedKind as exc:
        check("float16 is explicitly unsupported (%s)" % exc, True)
    else:
        raise AssertionError("float16 was encoded")


def corpus_tests(work: Path) -> None:
    print("\n2. the corpus profile")
    corpus = work / "cases.py"
    corpus.write_text(
        "import numpy as np\n"
        "OP = 'Neg'\nOPSET = 14\nNODE_OUTPUTS = ('out',)\n"
        "OUTPUT_TYPES = {'out': 'T'}\n"
        "def cases():\n"
        "    yield 'f64', [np.array([1.0, 2.0])]\n"
        "    yield 'i32', [np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32)]\n"
        "def sweeps():\n"
        "    return []\n",
        encoding="utf-8")
    prof = lc.profile(corpus, sys.executable)
    check_equal("arities", prof["arities"], [1, 2])
    check_equal("dtypes", prof["dtypes"], ["float64", "int32"])
    check_equal("outputs", prof["outputs"], ["out"])
    check_equal("output types", prof["output_types"], {"out": "T"})
    check_equal("sweeps detected", prof["has_sweeps"], True)
    check_equal("profile problem", lc.profile_problem(prof), "")

    empty = work / "empty.py"
    empty.write_text("def cases():\n    return []\n", encoding="utf-8")
    try:
        lc.profile(empty, sys.executable)
    except lc.ConformanceError as exc:
        check("a corpus with no cases is refused (%s)" % exc, True)
    else:
        raise AssertionError("an empty corpus was profiled")


def corpus_selection_tests(work: Path) -> None:
    print("\n3. which corpus the conformance stage freezes")
    spec = spec_file(work)
    root = work / "run"
    settings = Settings(op="ZzConformanceProbe", run_dir=root, spec_path=spec,
                        verbose=False)
    assert not (REPO / "ops" / settings.entry / "cases.py").exists()

    calls = []

    def fake_write_cases(_state, _settings):
        path = _settings.cases_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def cases():\n    yield 'a', []\n", encoding="utf-8")
        path.with_suffix(".meta.json").write_text(
            json.dumps({"spec_md5": harness.md5(spec)}), encoding="utf-8")
        calls.append(path)
        return path

    original = graph._write_cases
    graph._write_cases = fake_write_cases
    try:
        corpus = graph.ensure_conformance_corpus({"spec_path": str(spec)}, settings)
    finally:
        graph._write_cases = original
    check_equal("the tester is asked for a corpus", len(calls), 1)
    check_equal("its source is recorded", corpus["source"], "generated")
    check("the corpus is frozen beside the runner", corpus["path"].is_file())
    check_equal("the frozen corpus is the run directory's",
                corpus["path"].name, "cases.py")

    # A second pass reuses the run directory's corpus: its meta names the same specification.
    original = graph._write_cases

    def fail(_state, _settings):
        raise AssertionError("the tester was asked for a corpus that already exists")

    graph._write_cases = fail
    try:
        corpus = graph.ensure_conformance_corpus({"spec_path": str(spec)}, settings)
    finally:
        graph._write_cases = original
    check_equal("a fresh run-directory corpus is reused", corpus["source"], "run")


def manifest_tests(work: Path) -> None:
    print("\n4. the freshness manifest")
    spec = spec_file(work)
    root = work / "manifest-run"
    settings = generated_tree(root, spec)
    driver = settings.lean_conformance_driver_path()

    verdict = lean.validate_conformance_manifest(driver, spec)
    check("a coherent manifest validates", verdict["ok"] and not verdict["legacy"])

    other = work / "other.md"
    other.write_text("# Neg\n\nchanged\n", encoding="utf-8")
    verdict = lean.validate_conformance_manifest(driver, other)
    check("another specification is stale",
          not verdict["ok"] and any("specification changed" in f for f in verdict["findings"]))

    edited = settings.lean_conformance_runner_path()
    edited.write_text("def main : IO Unit := pure ()\n-- edited\n", encoding="utf-8")
    verdict = lean.validate_conformance_manifest(driver, spec)
    check("an edited runner is stale",
          not verdict["ok"] and any("runner" in f and "changed" in f
                                    for f in verdict["findings"]))

    plain = work / "plain"
    plain.mkdir()
    (plain / "lean_conformance.py").write_text("print('hand written')\n", encoding="utf-8")
    verdict = lean.validate_conformance_manifest(plain / "lean_conformance.py", spec)
    check("a hand-written driver is legacy, not stale",
          verdict["ok"] and verdict["legacy"])

    library_root = work / "manifest-library"
    library_settings = generated_tree(library_root, spec)
    library_driver = library_settings.lean_conformance_driver_path()
    library_manifest = library_settings.lean_conformance_manifest_path()
    data = json.loads(library_manifest.read_text(encoding="utf-8"))
    check_equal("the manifest binds the shared library", data["library_sha256"],
                lean.library_digest())
    data["library_sha256"] = "0" * 64
    library_manifest.write_text(json.dumps(data), encoding="utf-8")
    verdict = lean.validate_conformance_manifest(library_driver, spec)
    check("a changed shared library is stale",
          not verdict["ok"] and any("shared Sonnx library changed" in finding
                                    for finding in verdict["findings"]))
    data.pop("library_sha256")
    library_manifest.write_text(json.dumps(data), encoding="utf-8")
    verdict = lean.validate_conformance_manifest(library_driver, spec)
    check("a manifest without the shared library is stale",
          not verdict["ok"] and any("shared Sonnx library changed" in finding
                                    for finding in verdict["findings"]))


def selection_tests(work: Path) -> None:
    print("\n5. driver precedence")
    spec = spec_file(work)

    configured = work / "configured.py"
    configured.write_text("print('configured')\n", encoding="utf-8")
    settings = Settings(op="Neg", run_dir=work / "empty-run", spec_path=spec,
                        lean_conformance_script=configured, verbose=False)
    selection = lean.conformance_selection(settings, spec)
    check_equal("a configured driver wins", selection["kind"], "configured")

    run_root = work / "select-run"
    settings = generated_tree(run_root, spec)
    selection = lean.conformance_selection(settings, spec)
    check_equal("the run-directory driver is selected", selection["kind"], "generated-run")
    check_equal("the selection is its path", selection["path"],
                settings.lean_conformance_driver_path())

    published_root = work / "published-tree"
    generated_tree(published_root, spec)
    published = PublishedSettings(published_root / "lean_conformance.py", op="Neg",
                                  run_dir=published_root, spec_path=spec, verbose=False)
    selection = lean.conformance_selection(published, spec)
    check_equal("the published driver is the fallback", selection["kind"],
                "generated-published")

    absent = Settings(op="Neg", run_dir=work / "nothing", spec_path=spec, verbose=False)
    selection = lean.conformance_selection(absent, spec)
    check_equal("no driver anywhere is unavailable", selection["kind"], "unavailable")

    generated_tree(work / "no-manifest", spec)
    broken = work / "no-manifest"
    (broken / "lean" / "conformance.manifest.json").unlink()
    selection = lean.conformance_selection(
        Settings(op="Neg", run_dir=broken, spec_path=spec, verbose=False), spec)
    check_equal("a generated driver without its manifest is stale",
                selection["kind"], "stale")
    check("the reason names the manifest", "manifest" in selection["problem"])


def run_tests(work: Path) -> None:
    print("\n6. the generated driver's report")
    spec = spec_file(work)
    root = work / "driver-run"
    settings = generated_tree(root, spec)
    corpus = root / "lean" / "cases.py"
    corpus.write_text(
        "import numpy as np\n"
        "def cases():\n"
        "    yield 'f64', [np.array([1.0, -2.0], dtype=np.float64)]\n"
        "    yield 'f32', [np.array([3.5], dtype=np.float32)]\n"
        "    yield 'half', [np.array([1.0], dtype=np.float16)]\n",
        encoding="utf-8")
    # The manifest was written before the corpus was replaced; refreeze it so that the
    # freshness check accepts the corpus this test actually runs.
    lean.write_conformance_manifest(
        settings, spec, "published", {"outputs": ["out"], "output_types": {},
                                      "dtypes": ["float64"]},
        settings.lean_conformance_runner_path(), settings.lean_conformance_driver_path())

    class EchoRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, arrays, attrs):
            return (np.asarray(arrays[0]),)

    class WrongRunner(EchoRunner):
        def run(self, arrays, attrs):
            return (np.zeros_like(np.asarray(arrays[0])),)

    def echo_lean(runner, payload, timeout=300, lake=None, cwd=None):
        lines = []
        for line in payload.splitlines():
            if not line:
                continue
            fields = line.split("\t")
            lines.append("%s\tOK\t1\t%s" % (fields[0], fields[3]))
        return {"ok": True, "returncode": 0, "stdout": "\n".join(lines) + "\n",
                "stderr": "", "command": [], "seconds": 0.0}

    def wrong_lean(runner, payload, timeout=300, lake=None, cwd=None):
        lines = []
        for line in payload.splitlines():
            if not line:
                continue
            fields = line.split("\t")
            kind, shape = fields[3].split("|")[:2]
            count = 1 if shape == "-" else int(np.prod([int(d) for d in shape.split(",")]))
            lines.append("%s\tOK\t1\t%s|%s|%s"
                         % (fields[0], kind, shape, ",".join(["1"] * count)))
        return {"ok": True, "returncode": 0, "stdout": "\n".join(lines) + "\n",
                "stderr": "", "command": [], "seconds": 0.0}

    def warned_echo_lean(runner, payload, timeout=300, lake=None, cwd=None):
        """The runner as ``lake env lean --run`` actually prints it: diagnostics first."""
        result = echo_lean(runner, payload, timeout, lake, cwd)
        result["stdout"] = WARNING_BLOCK + result["stdout"]
        return result

    def warned_probe_lean(runner, payload, timeout=300, lake=None, cwd=None):
        lines = []
        for line in payload.splitlines():
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                lines.append("malformed\tMALFORMED\t0\tmissing fields")
            elif "f16" in line:
                lines.append("%s\tUNSUPPORTED\t0\tfloat16" % fields[0])
            else:
                lines.append("%s\tOK\t1\tf64|2|0,0" % fields[0])
        return {"ok": True, "returncode": 0, "stdout": WARNING_BLOCK + "\n".join(lines) + "\n",
                "stderr": "", "command": [], "seconds": 0.0}

    original_specloop, original_lean = lc.load_specloop, lc.run_lean
    original_repo = lc.find_repo

    def install(runner_class, lean_function):
        lc.load_specloop = lambda _repo: SimpleNamespace(Runner=runner_class)
        lc.run_lean = lean_function
        lc.find_repo = lambda _start: REPO

    config = {"op": "Neg", "opset": 14, "cases": str(corpus),
              "runner": str(root / "lean" / "LeanConformance.lean"),
              "outputs": ["out"], "output_types": {},
              "spec_md5": lean.source_md5(spec), "compare": "bits"}
    try:
        install(EchoRunner, echo_lean)
        report = lc.run(config, root / "echo.json", driver_dir=root)
        check_equal("every comparable case passes",
                    (report["summary"]["total"], report["summary"]["compared"],
                     report["summary"]["passed"], report["summary"]["failed"]),
                    (3, 2, 2, 0))
        check_equal("float16 is listed as unsupported",
                    report["summary"]["unsupported"], 1)
        check("the echo run is ok", report["ok"])
        check_equal("the report names the protocol", report["protocol"], lc.PROTOCOL)

        install(EchoRunner, warned_echo_lean)
        report = lc.run(config, root / "warned.json", driver_dir=root)
        check("compiler diagnostics before the responses are not protocol lines",
              report["ok"] and report["summary"]["passed"] == 2)
        install(EchoRunner, warned_probe_lean)
        smoke = lc.smoke(root / "lean" / "LeanConformance.lean",
                         {"dtypes": ["float64"], "arities": [2]})
        check("the smoke test ignores compiler diagnostics too", smoke["ok"])
        probes = dict(lc.probe_requests({"dtypes": ["float64"], "arities": [2]}))
        check("the unsupported-kind probe carries the operator's arity",
              probes["unsupported kind"].split("\t")[2] == "2"
              and probes["unsupported kind"].count("f16") == 2)

        install(WrongRunner, wrong_lean)
        report = lc.run(config, root / "wrong.json", driver_dir=root)
        check("a mismatch is a failure, not an error",
              report["stage"] == "compare" and not report["ok"]
              and report["summary"]["failed"] == 2)
    finally:
        lc.load_specloop, lc.run_lean = original_specloop, original_lean
        lc.find_repo = original_repo


def run_conformance_tests(work: Path) -> None:
    print("\n7. run_conformance validates and reads a generated report")
    spec = spec_file(work)
    root = work / "conformance-run"
    settings = generated_tree(root, spec)
    driver = settings.lean_conformance_driver_path()
    report = {"ok": True, "generated_by": lc.GENERATED_BY, "protocol": lc.PROTOCOL,
              "spec_md5": lean.source_md5(spec),
              "summary": {"total": 4, "compared": 3, "passed": 3, "failed": 0,
                          "unsupported": 1}}
    driver.write_text(
        "#!/usr/bin/env python3\n"
        "# @generated-by sonnx-lean-pipeline-conformance\n"
        "import argparse, json, sys\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--json'); p.add_argument('--timeout')\n"
        "a = p.parse_args()\n"
        "json.dump(%r, open(a.json, 'w'))\n" % report, encoding="utf-8")
    lean.write_conformance_manifest(
        settings, spec, "published", {"outputs": ["out"], "output_types": {},
                                      "dtypes": ["float64"]},
        settings.lean_conformance_runner_path(), driver)

    settings.python = sys.executable
    result = lean.run_conformance(settings, spec_path=spec)
    check_equal("a fresh generated report passes", result["status"], "passed")
    check_equal("the run is marked generated", result["generated"], True)
    check_equal("the report carries the corpus total",
                result["report"]["summary"]["total"], 4)

    driver.write_text(driver.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    result = lean.run_conformance(settings, spec_path=spec)
    check_equal("an edited generated driver is stale", result["status"], "error")
    check("the stale run is blocking", result["blocking"])
    check("the message says why", "stale" in result["error"])


def driver_source_tests(work: Path) -> None:
    print("\n8. the generated driver")
    config = {"op": "Neg", "opset": 14, "runner": "lean/LeanConformance.lean",
              "cases": "lean/cases.py", "outputs": ["out"], "output_types": {},
              "compare": "bits", "ulp": 2, "relative": 1e-6, "spec_md5": "0" * 32,
              "formal_sha256": "1" * 64}
    source = lc.driver_source(config)
    check("the driver names its generator", "@generated-by" in source)
    check("the driver names the protocol", "@protocol sonnx-conformance/1" in source)
    check("the driver is Python", ast.parse(source) is not None)
    check("the driver carries its configuration", '"op": "Neg"' in source)
    path = work / "LeanDriverSource.py"
    path.write_text(source, encoding="utf-8")
    import py_compile
    py_compile.compile(str(path), doraise=True)
    print("  ok  the generated driver compiles")


def publication_tests(work: Path) -> None:
    print("\n9. publication keeps the relative layout and the manifest")
    spec = spec_file(work)
    root = work / "publish-run"
    settings = generated_tree(root, spec)
    published = work / "published"

    class PublishSettings(Settings):
        def lean_publication_dir(self) -> Path:
            return published / "lean"

        def lean_publication_driver_path(self) -> Path:
            return published / "lean_conformance.py"

    settings = PublishSettings(op="Neg", run_dir=root, spec_path=spec, verbose=False)
    state = {"lean_conformance_generated": True,
             "lean_conformance_runner": str(settings.lean_conformance_runner_path()),
             "lean_conformance_corpus": str(settings.lean_conformance_corpus_path()),
             "lean_conformance_driver": str(settings.lean_conformance_driver_path()),
             "lean_conformance_manifest": str(settings.lean_conformance_manifest_path())}
    files = cli._conformance_publication(settings, state)
    check_equal("four generated files are published", sorted(files),
                ["corpus", "driver", "manifest", "runner"])
    for name in ("runner", "corpus", "driver", "manifest"):
        origin, destination = files[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(origin.read_bytes())
    # `_publish_lean` writes FormalSpec.lean in the same publication; the conformance
    # manifest names it too.
    settings.lean_publication_dir().mkdir(parents=True, exist_ok=True)
    (settings.lean_publication_dir() / "FormalSpec.lean").write_bytes(
        settings.lean_path().read_bytes())

    verdict = lean.validate_conformance_manifest(files["driver"][1], spec)
    check("the published tree validates against the same manifest", verdict["ok"])
    cli._guard_conformance_overwrite(settings, files)
    print("  ok  an untouched generated target may be replaced")

    files["runner"][1].write_text("-- hand-written replacement\n", encoding="utf-8")
    try:
        cli._guard_conformance_overwrite(settings, files)
    except RuntimeError as exc:
        check("an edited published artifact is refused (%s)" % exc,
              "refusing to overwrite" in str(exc))
    else:
        raise AssertionError("an edited published artifact was accepted")

    check("with force the target is replaceable", _force_guard(settings, files))


def _force_guard(settings: Settings, files: dict) -> bool:
    settings.force = True
    try:
        cli._guard_conformance_overwrite(settings, files)
        return True
    finally:
        settings.force = False


def runner_generation_tests(work: Path) -> None:
    print("\n10. the runner is compiled, smoke-tested and repaired")
    spec = spec_file(work)
    root = work / "runner-run"
    settings = generated_tree(root, spec)
    settings.max_conformance_retries = 1
    answers = ["```lean\ndef main : IO Unit := pure ()\n```",
               "```lean\ndef main : IO Unit := pure ()\n-- repaired\n```"]
    asked = []
    compiled = []

    def fake_artifact(_settings, _system, request, _where, max_tokens=None):
        asked.append(request)
        return answers[len(asked) - 1]

    def fake_compile(path, lake=None, timeout=300, cwd=None):
        compiled.append(Path(path).name)
        if len(compiled) == 1:
            return {"ok": False, "returncode": 1, "stdout": "unknown identifier 'pure'"}
        return {"ok": True, "returncode": 0, "stdout": ""}

    def fake_smoke(_runner, _profile, timeout=300, lake=None):
        return {"ok": True, "findings": [], "checks": []}

    original = (graph._artifact, graph.lean_support.compile_lean, graph.lean_conformance.smoke)
    graph._artifact, graph.lean_support.compile_lean, graph.lean_conformance.smoke = (
        fake_artifact, fake_compile, fake_smoke)
    try:
        result = graph.write_conformance_runner(
            {"spec_path": str(spec)}, settings,
            {"spec_md5": harness.md5(spec)},
            {"arities": [1], "dtypes": ["float64"], "outputs": ["out"]})
    finally:
        (graph._artifact, graph.lean_support.compile_lean,
         graph.lean_conformance.smoke) = original

    check("the repair converges", result["ok"])
    check_equal("two answers were asked for", len(asked), 2)
    check_equal("two candidates were compiled", compiled,
                ["conformance-attempt-001.lean", "conformance-attempt-002.lean"])
    check("the repair carries the compiler finding",
          "Lean compiler" in asked[1] and "unknown identifier" in asked[1])
    check("the accepted runner is promoted",
          Path(result["path"]).name == "LeanConformance.lean" and Path(result["path"]).is_file())
    rejected = sorted(p.name for p in (root / "lean").glob("conformance-attempt-001*"))
    check("the rejected answer is kept (%s)" % rejected,
          any(name.endswith("-rejected-1.txt") for name in rejected))


def sweep_tests(work: Path) -> None:
    print("\n12. the sweeps contribute the same inputs the Python test used")
    spec = spec_file(work)
    root = work / "sweep-run"
    settings = generated_tree(root, spec)
    corpus = root / "lean" / "cases.py"
    corpus.write_text(
        "import numpy as np\n"
        "def cases():\n"
        "    yield 'the one case', [np.array([1.0, -2.0], dtype=np.float64)]\n"
        "def a_sweep(impl):\n"
        "    a = np.array([3.0, -4.0], dtype=np.float64)\n"
        "    impl.neg(a)\n"
        "    return {'sweep': 'a_sweep', 'failures': []}\n"
        "def an_attribute_sweep(impl):\n"
        "    impl.neg(np.array([5.0], dtype=np.float64), axis=0)\n"
        "    return {'sweep': 'an_attribute_sweep', 'failures': []}\n"
        "def a_broken_sweep(impl):\n"
        "    impl.neg(np.array([7.0], dtype=np.float64))\n"
        "    raise RuntimeError('the sweep\\'s own check failed')\n"
        "def sweeps():\n"
        "    return [a_sweep, an_attribute_sweep, a_broken_sweep]\n",
        encoding="utf-8")
    lean.write_conformance_manifest(
        settings, spec, "published", {"outputs": ["out"], "output_types": {},
                                      "dtypes": ["float64"]},
        settings.lean_conformance_runner_path(), settings.lean_conformance_driver_path())

    class EchoRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, arrays, attrs):
            return (np.asarray(arrays[0]),)

    def check_sweeps(impl, sweeps):
        reports = []
        for sweep in sweeps:
            try:
                reports.append(sweep(impl) or {"sweep": getattr(sweep, "__name__", "")})
            except Exception as exc:
                reports.append({"sweep": getattr(sweep, "__name__", "sweep"),
                                "error": str(exc)})
        return reports

    def echo_lean(runner, payload, timeout=300, lake=None, cwd=None):
        lines = []
        for line in payload.splitlines():
            if not line:
                continue
            fields = line.split("\t")
            lines.append("%s\tOK\t1\t%s" % (fields[0], fields[3]))
        return {"ok": True, "returncode": 0, "stdout": "\n".join(lines) + "\n",
                "stderr": "", "command": [], "seconds": 0.0}

    original_specloop, original_lean = lc.load_specloop, lc.run_lean
    original_repo = lc.find_repo
    lc.load_specloop = lambda _repo: SimpleNamespace(Runner=EchoRunner,
                                                     check_sweeps=check_sweeps)
    lc.run_lean = echo_lean
    lc.find_repo = lambda _start: REPO
    try:
        config = {"op": "Neg", "opset": 14, "cases": str(corpus),
                  "runner": str(root / "lean" / "LeanConformance.lean"),
                  "outputs": ["out"], "output_types": {},
                  "spec_md5": lean.source_md5(spec), "compare": "bits"}
        report = lc.run(config, root / "sweeps.json", driver_dir=root)
        check_equal("the one case and the three sweep inputs are all run",
                    (report["summary"]["total"], report["summary"]["compared"],
                     report["summary"]["passed"], report["summary"]["failed"]),
                    (4, 4, 4, 0))
        check_equal("the sweeps are their own category",
                    report["by_category"]["sweeps"],
                    {"total": 3, "passed": 3, "failed": 0})
        check("the sweep inputs are reported",
              report["sweeps"]["declared"] and report["sweeps"]["inputs"] == 3)
        check("a sweep's own failure is reported, not counted against the run (%s)"
              % report["sweeps"]["errors"],
              len(report["sweeps"]["errors"]) == 1
              and "a_broken_sweep" in report["sweeps"]["errors"][0]
              and "own check failed" in report["sweeps"]["errors"][0])
        check("the run still passes under a broken sweep", report["ok"])
    finally:
        lc.load_specloop, lc.run_lean = original_specloop, original_lean
        lc.find_repo = original_repo


def driver_bootstrap_tests(work: Path) -> None:
    print("\n11. a driver outside the repository finds the shared runtime")
    outside = Path(tempfile.mkdtemp(prefix="sonnx-driver-outside-"))
    driver = outside / "lean_conformance.py"
    driver.write_text(lc.driver_source({
        "op": "Abs", "opset": 14, "cases": "lean/cases.py", "compare": "bits",
        "entry": "abs", "outputs": ["out"], "output_types": {},
        "runner": "lean/LeanConformance.lean", "spec_md5": "0" * 32,
        "formal_sha256": "0" * 64, "ulp": 2, "relative": 1e-6,
    }), encoding="utf-8")

    def help_from(cwd: Path, env: dict | None = None):
        merged = dict(os.environ)
        merged.pop("SONNX_REPO", None)
        merged.pop("PYTHONPATH", None)
        merged.update(env or {})
        return subprocess.run([sys.executable, str(driver), "--help"], cwd=str(cwd),
                              env=merged, capture_output=True, text=True)

    from_cwd = help_from(REPO)
    check("the working directory locates the runtime (%s)" % from_cwd.stderr.strip()[:120],
          from_cwd.returncode == 0)
    from_env = help_from(outside, {"SONNX_REPO": str(REPO)})
    check("$SONNX_REPO locates the runtime (%s)" % from_env.stderr.strip()[:120],
          from_env.returncode == 0)
    nowhere = help_from(outside)
    check("outside and without the variable the driver refuses",
          nowhere.returncode != 0 and "SONNX_REPO" in nowhere.stderr)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="sonnx-lean-conformance-"))
    protocol_tests()
    corpus_tests(work)
    corpus_selection_tests(work)
    manifest_tests(work)
    selection_tests(work)
    run_tests(work)
    run_conformance_tests(work)
    driver_source_tests(work)
    publication_tests(work)
    runner_generation_tests(work)
    sweep_tests(work)
    driver_bootstrap_tests(work)
    print("\nall generated-conformance tests hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
