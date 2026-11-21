#!/usr/bin/env python3
"""Focused tests for deterministic Lean traceability and safe publication."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonnx_agents import cli, extract, lean  # noqa: E402
from sonnx_agents.config import Settings  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MUL = REPO / "ops" / "mul" / "mul.md"


def check(name: str, condition: bool) -> None:
    assert condition, name
    print("  ok  %s" % name)


API = [
    "Sonnx.Tensor", "Sonnx.Tensor.read", "Sonnx.Tensor.wellFormed", "Sonnx.size",
    "Sonnx.validIdxB", "Sonnx.offset", "Sonnx.padTo", "Sonnx.bcastDim",
    "Sonnx.bcastShape", "Sonnx.bcastIdx", "Sonnx.Broadcasts",
    "Sonnx.BroadcastCompatible", "Sonnx.BroadcastShapeIs", "Sonnx.indices",
    "Sonnx.TypeConsistent", "Sonnx.binary", "Sonnx.binaryData_read_correct",
    "Sonnx.PointwiseBinaryRel", "Sonnx.ElementwiseBinaryRel", "Sonnx.CongM",
    "Sonnx.intModulus", "Sonnx.intMin", "Sonnx.intMax", "Sonnx.InSignedRange",
    "Sonnx.InUnsignedRange", "Sonnx.uintWrap", "Sonnx.sintWrap", "Sonnx.UIntN",
    "Sonnx.SIntN", "Sonnx.SIntN.wrap", "Sonnx.FloatKind", "Sonnx.FloatFormat",
    "Sonnx.FVal", "Sonnx.IsRepresentable", "Sonnx.RoundNearestEven",
    "Sonnx.Overflows", "Sonnx.RoundToFormat",
]


def raw_module(inv: dict) -> str:
    lines = ["import Sonnx", "def MulTraceWitness : Prop := True"]
    lines += ["-- @trace %s status=specified decl=MulTraceWitness" % item["key"]
              for item in inv["requirements"]]
    lines += ["-- @trace %s status=example_only decl=MulTraceWitness" % item["key"]
              for item in inv["examples"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    built = lean.build_library()
    check("the shared Sonnx library builds", built["ok"])
    check("the C generator builds with it",
          (REPO / "lean" / ".lake" / "build" / "lib" / "lean" / "CGen.olean").is_file())
    check("the library exports its tensor model", "Tensor" in lean.shared_names())
    check("the library exports the elementwise operator", "binary" in lean.shared_names())

    check("the shared import is accepted", not lean.import_findings("import Sonnx\n"))
    check("a shared module import is accepted",
          not lean.import_findings("import Sonnx.Float\n"))
    check("core imports are accepted",
          not lean.import_findings("import Sonnx\nimport Init\n"))
    check("the C generator import is accepted",
          not lean.import_findings("import Sonnx\nimport CGen\n"))
    check("a missing shared import is rejected",
          lean.import_findings("def x : Nat := 0\n")
          == ["the module must import the shared library: import Sonnx"])
    check("a generated module import is rejected",
          len(lean.import_findings("import Sonnx\nimport FormalSpec\n")) == 1)
    check("an operator module import is rejected",
          len(lean.import_findings("import Sonnx\nimport MulOp.Tensor\n")) == 1)
    check("a package import is rejected",
          len(lean.import_findings("import Sonnx\nimport Mathlib\n")) == 1)
    check("a commented import is ignored",
          not lean.import_findings("import Sonnx\n-- import Mathlib\n"))

    check("a top-level C descriptor is accepted",
          not lean.cgen_findings(
              'import Sonnx\nimport CGen\n\ndef sonnxCGenOp : CGen.Elementwise := ⟨"sub", .sub⟩\n'))
    check("the C descriptor is found by its declaration",
          lean.cgen_descriptor('import CGen\n\ndef sonnxCGenOp : CGen.Elementwise := ⟨"sub", .sub⟩\n'))
    check("a commented C descriptor is not a declaration",
          not lean.cgen_descriptor("import CGen\n\n-- def sonnxCGenOp : CGen.Elementwise := ⟨\"sub\", .sub⟩\n"))
    check("a C descriptor without its import is rejected",
          "import CGen" in ";".join(lean.cgen_findings(
              'import Sonnx\n\ndef sonnxCGenOp : CGen.Elementwise := ⟨"sub", .sub⟩\n')))
    check("a namespaced C descriptor is rejected",
          "top-level" in ";".join(lean.cgen_findings(
              'import Sonnx\nimport CGen\n\nnamespace X\n'
              'def sonnxCGenOp : CGen.Elementwise := ⟨"sub", .sub⟩\nend X\n')))
    check("a descriptor declared twice is rejected",
          "more than once" in ";".join(lean.cgen_findings(
              'import Sonnx\nimport CGen\n\ndef sonnxCGenOp : CGen.Elementwise := ⟨"add", .add⟩\n'
              'def sonnxCGenOp : CGen.Elementwise := ⟨"sub", .sub⟩\n')))

    check("redefining a shared structure is rejected",
          "redefines the shared declaration Tensor" in
          ";".join(lean.shared_redefinition_findings(
              "structure Tensor (shape : List Nat) where\n  data : List Nat\n")))
    check("a namespaced shared name is rejected",
          lean.shared_redefinition_findings("def bcastShape := 1\n"))
    check("an operator relation is allowed",
          not lean.shared_redefinition_findings(
              "def SubSpec (a b : Int) : Prop := a = b\n"))

    library = REPO / "lean"
    digest = lean.library_digest()
    check("the library digest is stable", lean.library_digest() == digest)
    library_copy = Path(tempfile.mkdtemp(prefix="sonnx-library-test-")) / "lean"
    shutil.copytree(library, library_copy,
                    ignore=shutil.ignore_patterns(".lake", "*.olean", "lake-manifest.json"))
    check("a copy of the library has the same digest", lean.library_digest(library_copy) == digest)
    (library_copy / "Sonnx" / "Tensor.lean").write_text(
        (library_copy / "Sonnx" / "Tensor.lean").read_text(encoding="utf-8") + "\n-- edited\n",
        encoding="utf-8")
    check("a library edit changes the digest", lean.library_digest(library_copy) != digest)

    probe = library_copy.parent / "api.lean"
    probe.write_text("import Sonnx\n" + "".join("#check @%s\n" % name for name in API),
                     encoding="utf-8")
    check("the public API resolves", lean.compile_lean(probe)["ok"])
    probe.write_text("import Sonnx\n#check @Sonnx.size\n", encoding="utf-8")
    check("a marker naming a shared declaration compiles", lean.compile_lean(probe)["ok"])
    probe.write_text("import Sonnx\n#check @Sonnx.NoSuch\n", encoding="utf-8")
    check("a drifted marker does not compile", not lean.compile_lean(probe)["ok"])

    inv = lean.inventory(MUL)
    check("Mul has 25 requirements", len(inv["requirements"]) == 25)
    check("Mul has eight examples", len(inv["examples"]) == 8)
    check("14 requirements have no anchor",
          sum(item["anchor"] is None for item in inv["requirements"]) == 14)
    b = next(item for item in inv["requirements"] if item["key"] == "E_MUL_REAL_CONSTR_B_0010")
    check("unanchored B has its own source span", (b["start"], b["end"]) == (126, 127))
    a0020 = next(item for item in inv["requirements"] if item["key"] == "E_MUL_FLOAT_CONSTR_A_0020")
    check("adjacent anchor is not absorbed into previous clause", a0020["start"] == 299)

    raw = raw_module(inv)
    check("fenced Lean extraction", extract.extract_lean("Here is code:\n```lean\n%s```" % raw) == raw)
    check("trace validation accepts complete inventory", lean.validate_candidate(raw, inv)["ok"])
    example_requirement = raw.replace(
        "-- @trace E_MUL_REAL_FUNC_0010 status=specified",
        "-- @trace E_MUL_REAL_FUNC_0010 status=example_only", 1)
    check("example-only requirement is not complete",
          not lean.validate_candidate(example_requirement, inv)["complete"])
    check("missing tag rejected", "missing trace entry" in
          ";".join(lean.validate_candidate(raw.replace(
              "-- @trace E_MUL_REAL_FUNC_0010 status=specified decl=MulTraceWitness\n", ""),
              inv)["findings"]))
    duplicate = raw + "-- @trace E_MUL_REAL_FUNC_0010 status=specified decl=MulTraceWitness\n"
    check("duplicate tag rejected", "duplicate trace entry" in
          ";".join(lean.validate_candidate(duplicate, inv)["findings"]))
    unknown = raw + "-- @trace E_MUL_REAL_CONSTR_A_9999 status=deferred limitation=\"unknown\"\n"
    check("unknown tag rejected", "unknown trace key" in
          ";".join(lean.validate_candidate(unknown, inv)["findings"]))
    check("sorry rejected", lean.forbidden_constructs("theorem bad : True := by sorry"))
    check("comment mentioning sorry allowed", not lean.forbidden_constructs("-- sorry is forbidden\n"))

    work = Path(tempfile.mkdtemp(prefix="sonnx-lean-test-"))
    spec = work / "mul.md"
    spec.write_bytes(MUL.read_bytes())
    inv = lean.inventory(spec)
    source = work / "FormalSpec.lean"
    report = lean.prepare_candidate(raw_module(inv), inv, source)
    check("candidate materializes", report["ok"] and source.is_file())
    manifest = work / "FormalSpec.trace.json"
    lean.write_report(manifest, report, inv)
    checked = lean.check_files(spec, [source])
    check("materialized artifact compiles and checks", checked["ok"] and checked["complete"])
    check("the trace check names the shared library",
          checked["library_sha256"] == lean.library_digest())
    original = source.read_text(encoding="utf-8")
    source.write_text(original.replace("@source-digest " + inv["source_md5"],
                                       "@source-digest " + "0" * 32), encoding="utf-8")
    check("stale digest rejected", not lean.check_files(spec, [source], compile_code=False)["ok"])
    source.write_text(original, encoding="utf-8")
    source.write_text(original.replace("@library-digest " + lean.library_digest(),
                                       "@library-digest " + "0" * 64), encoding="utf-8")
    check("stale library digest rejected",
          not lean.check_files(spec, [source], compile_code=False)["ok"])
    source.write_text(original, encoding="utf-8")
    damaged = original.replace("@source %s:L33-L58" % inv["source"],
                               "@source wrong.md:L1-L2", 1)
    source.write_text(damaged, encoding="utf-8")
    check("wrong source span rejected", not lean.check_files(spec, [source], compile_code=False)["ok"])
    excerpt_start = original.index("Informal source excerpt:\n") + len("Informal source excerpt:\n")
    excerpt_end = original.index("\n-/", excerpt_start)
    damaged = original[:excerpt_start] + "  altered excerpt" + original[excerpt_end:]
    source.write_text(damaged, encoding="utf-8")
    check("altered source excerpt rejected",
          not lean.check_files(spec, [source], compile_code=False)["ok"])
    source.write_text(original, encoding="utf-8")

    class TempSettings(Settings):
        def lean_publication_dir(self) -> Path:
            return work / "published"

    settings = TempSettings(op="Mul", run_dir=work, spec_path=spec,
                            lean=True, lean_only=True, lean_conformance=False,
                            verbose=False)
    state = {"spec_path": str(spec), "lean_path": str(source),
             "lean_trace_path": str(manifest), "lean_ok": True}
    published = cli._publish_lean(settings, state)
    check("validated artifact publishes", all(path.is_file() for path in published))
    target = published[0]
    target.write_text("-- hand-written replacement\n", encoding="utf-8")
    settings.force = True
    before = target.read_bytes()
    try:
        cli._publish_lean(settings, state)
    except RuntimeError as exc:
        check("hand-edited target refused even with force", "hand-written" in str(exc))
    else:
        raise AssertionError("hand-edited publication target was overwritten")
    check("refused target remains untouched", target.read_bytes() == before)

    spec.write_bytes(spec.read_bytes() + b"\n")
    check("source edit makes published freshness fail",
          not lean.check_files(spec, [source], compile_code=False)["ok"])

    def conformance_settings(script, **kwargs):
        return TempSettings(op="Mul", run_dir=work, spec_path=spec,
                            python=sys.executable, verbose=False,
                            lean_conformance_script=script, **kwargs)

    def runner(name, payload, code):
        path = work / name
        path.write_text(
            "import argparse, json, sys\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--json'); parser.add_argument('--timeout')\n"
            "args = parser.parse_args()\n"
            "json.dump(json.loads(%r), open(args.json, 'w'))\n"
            "sys.exit(%d)\n" % (json.dumps(payload), code), encoding="utf-8")
        return path

    passing = runner("conformance-pass.py", {
        "ok": True, "summary": {"total": 3, "passed": 3, "failed": 0}}, 0)
    result = lean.run_conformance(conformance_settings(passing))
    check("passing conformance reported", result["status"] == "passed" and not result["blocking"])
    check("conformance report retained", Path(result["report_path"]).is_file())

    failing = runner("conformance-fail.py", {
        "ok": False, "summary": {"total": 3, "passed": 2, "failed": 1}}, 1)
    report_path = settings.lean_run_dir() / "LeanConformance.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('{"ok": true}\n', encoding="utf-8")
    result = lean.run_conformance(conformance_settings(failing))
    check("stale passing report does not mask failure",
          result["status"] == "failed" and result["blocking"])
    check("failed conformance exits the Lean stage", result["report"]["summary"]["failed"] == 1)

    # a stale/failed implementation must not be published even when the Lean artifact itself
    # is still fresh and trace-complete
    spec2 = work / "mul-publish.md"
    spec2.write_bytes(MUL.read_bytes())
    inv2 = lean.inventory(spec2)
    source2 = work / "FormalSpec2.lean"
    manifest2 = work / "FormalSpec2.trace.json"
    report2 = lean.prepare_candidate(raw_module(inv2), inv2, source2)
    lean.write_report(manifest2, report2, inv2)
    blocked = TempSettings(op="Mul", run_dir=work / "blocked", spec_path=spec2,
                           lean=True, lean_only=True, verbose=False,
                           lean_conformance_script=failing)
    before = (work / "published" / "FormalSpec.lean").read_bytes()
    try:
        cli._publish_lean(blocked, {"spec_path": str(spec2), "lean_path": str(source2),
                                    "lean_trace_path": str(manifest2), "lean_ok": True})
    except RuntimeError as exc:
        check("conformance failure blocks publication", "conformance" in str(exc))
    else:
        raise AssertionError("a failing Lean implementation was published")
    check("blocked publication leaves the target untouched",
          (work / "published" / "FormalSpec.lean").read_bytes() == before)

    (work / "no-report.py").write_text(
        "import sys\nsys.exit(2)\n", encoding="utf-8")
    result = lean.run_conformance(conformance_settings(work / "no-report.py"))
    check("crashing conformance is an error", result["status"] == "error" and result["blocking"])

    missing = work / "does-not-exist.py"
    result = lean.run_conformance(conformance_settings(missing))
    check("absent operator driver is not a failure",
          result["status"] == "unavailable" and not result["blocking"])

    marker = work / "disabled-ran"
    disabled = work / "conformance-disabled.py"
    disabled.write_text(
        "from pathlib import Path\nPath(%r).write_text('ran')\n" % str(marker),
        encoding="utf-8")
    result = lean.run_conformance(conformance_settings(disabled, lean_conformance=False))
    check("disabled conformance does not run the driver",
          result["status"] == "disabled" and not marker.exists())
    print("\nall Lean trace tests hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
