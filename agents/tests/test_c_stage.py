#!/usr/bin/env python3
"""Tests for the C stage: rendering, the descriptor guards, and the test protocol.

No model is called.  The rendering tests compile a small module with `lake env lean`; the
protocol tests compile and run fixed C programs with `cc`.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonnx_agents import extract, graph, lean  # noqa: E402
from sonnx_agents.config import Settings  # noqa: E402

SPEC = """import Sonnx

namespace Generated.Sub

def realSub (a b : Rat) : Rat := a - b
def floatSub (a b : Rat) : Rat := a - b
def intSub (a b : Int) : Int := a - b
def uintSub (a b : Nat) : Nat := a - b

end Generated.Sub
"""

SPEC_DIV = """import Sonnx

namespace Generated.Div

def realDiv (a b : Rat) : Rat := a / b
def floatDiv (a b : Rat) : Rat := a / b
def intDiv (a b : Int) : Int := a / b
def uintDiv (a b : Nat) : Nat := a / b

end Generated.Div
"""

SPEC_NONE = """import Sonnx

def thing : Nat := 0
"""

FAMILIES = (("real", "REAL"), ("float", "FLOAT"), ("int", "INT"), ("uint", "UINT"))


def derivations_for(op: str, names: dict[str, str]) -> list[dict]:
    """The family declarations a mini specification's trace names."""
    return [{"family": family,
             "declaration": "Generated.%s.%s" % (op, names[family]),
             "requirement": "E_%s_%s_FUNC_0010" % (op.upper(), tag),
             "source": "ops/%s/%s.md:L1-L2" % (op.lower(), op.lower())}
            for family, tag in FAMILIES]


SUB_DERIVATIONS = derivations_for("Sub", {"real": "realSub", "float": "floatSub",
                                          "int": "intSub", "uint": "uintSub"})
DIV_DERIVATIONS = derivations_for("Div", {"real": "realDiv", "float": "floatDiv",
                                          "int": "intDiv", "uint": "uintDiv"})


def write_trace(settings: Settings, rows: list[dict]) -> None:
    path = settings.lean_trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": rows}, indent=2) + "\n", encoding="utf-8")

GOOD_TEST = """#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "sonnx_kernels.h"

static int failures;

static void ok(const char *name, int condition) {
  if (condition) {
    printf("ok %s\\n", name);
  } else {
    printf("FAIL %s: expectation not met\\n", name);
    failures++;
  }
}

int main(void) {
  { const size_t shape[1] = {2};
    float a_data[2] = {5.0f, -1.0f}, b_data[2] = {3.0f, -1.0f};
    float c_data[2] = {0.0f, 0.0f};
    sonnx_tensor a = { .data = a_data, .ndim = 1, .shape = shape };
    sonnx_tensor b = { .data = b_data, .ndim = 1, .shape = shape };
    sonnx_tensor c = { .data = c_data, .ndim = 1, .shape = shape };
    sonnx_sub_f32(&a, &b, &c);
    ok("f32 5-3", c_data[0] == 2.0f);
    ok("f32 -1 minus -1", c_data[1] == 0.0f); }
  { const size_t shape[1] = {2};
    double a_data[2] = {0.25, 1.5}, b_data[2] = {0.5, 2.5};
    double c_data[2] = {0.0, 0.0};
    sonnx_tensor a = { .data = a_data, .ndim = 1, .shape = shape };
    sonnx_tensor b = { .data = b_data, .ndim = 1, .shape = shape };
    sonnx_tensor c = { .data = c_data, .ndim = 1, .shape = shape };
    sonnx_sub_f64(&a, &b, &c);
    ok("f64 0.25-0.5", c_data[0] == -0.25);
    ok("f64 1.5-2.5", c_data[1] == -1.0); }
  { const size_t shape[1] = {2};
    int8_t a_data[2] = {127, -128}, b_data[2] = {1, 1};
    int8_t c_data[2] = {0, 0};
    sonnx_tensor a = { .data = a_data, .ndim = 1, .shape = shape };
    sonnx_tensor b = { .data = b_data, .ndim = 1, .shape = shape };
    sonnx_tensor c = { .data = c_data, .ndim = 1, .shape = shape };
    sonnx_sub_i8(&a, &b, &c);
    ok("i8 127-1", c_data[0] == 126);
    ok("i8 -128-1 wraps", c_data[1] == 127); }
  { const size_t shape[1] = {2};
    int32_t a_data[2] = {5, INT32_MIN}, b_data[2] = {3, 1};
    int32_t c_data[2] = {0, 0};
    sonnx_tensor a = { .data = a_data, .ndim = 1, .shape = shape };
    sonnx_tensor b = { .data = b_data, .ndim = 1, .shape = shape };
    sonnx_tensor c = { .data = c_data, .ndim = 1, .shape = shape };
    sonnx_sub_i32(&a, &b, &c);
    ok("i32 5-3", c_data[0] == 2);
    ok("i32 INT32_MIN-1 wraps", c_data[1] == INT32_MAX); }
  { const size_t shape[1] = {2};
    uint8_t a_data[2] = {0, 200}, b_data[2] = {1, 100};
    uint8_t c_data[2] = {0, 0};
    sonnx_tensor a = { .data = a_data, .ndim = 1, .shape = shape };
    sonnx_tensor b = { .data = b_data, .ndim = 1, .shape = shape };
    sonnx_tensor c = { .data = c_data, .ndim = 1, .shape = shape };
    sonnx_sub_u8(&a, &b, &c);
    ok("u8 0-1 wraps", c_data[0] == 255);
    ok("u8 200-100", c_data[1] == 100); }
  { const size_t shape[1] = {2};
    uint32_t a_data[2] = {0, 7}, b_data[2] = {1, 7};
    uint32_t c_data[2] = {0, 0};
    sonnx_tensor a = { .data = a_data, .ndim = 1, .shape = shape };
    sonnx_tensor b = { .data = b_data, .ndim = 1, .shape = shape };
    sonnx_tensor c = { .data = c_data, .ndim = 1, .shape = shape };
    sonnx_sub_u32(&a, &b, &c);
    ok("u32 0-1 wraps", c_data[0] == UINT32_MAX);
    ok("u32 7-7", c_data[1] == 0); }
  { const size_t shape[1] = {1};
    float a_data[1] = {1.0f}, b_data[1] = {2.0f};
    float c_data[1] = {0.0f};
    sonnx_tensor a = { .data = a_data, .ndim = 1, .shape = shape };
    sonnx_tensor b = { .data = b_data, .ndim = 1, .shape = shape };
    sonnx_tensor c = { .data = c_data, .ndim = 1, .shape = shape };
    sonnx_sub_f32(&a, &b, &c);
    ok("f32 1-2", c_data[0] == -1.0f); }
  { const size_t shape[1] = {1};
    int64_t a_data[1] = {INT64_MAX}, b_data[1] = {-1};
    int64_t c_data[1] = {0};
    sonnx_tensor a = { .data = a_data, .ndim = 1, .shape = shape };
    sonnx_tensor b = { .data = b_data, .ndim = 1, .shape = shape };
    sonnx_tensor c = { .data = c_data, .ndim = 1, .shape = shape };
    sonnx_sub_i64(&a, &b, &c);
    ok("i64 INT64_MAX minus -1 wraps", c_data[0] == INT64_MIN); }
  return failures == 0 ? 0 : 1;
}
"""


def check(name: str, condition: bool) -> None:
    assert condition, name
    print("  ok  %s" % name)


def settings_for(directory: Path, op: str = "Sub") -> Settings:
    return Settings(op=op, opset=14, run_dir=directory, lean=True, lean_only=True,
                    cgen=True, lake=lean.find_lake(), verbose=False)


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_extract() -> None:
    fenced = "Here is the test:\n\n```c\n%s```\n" % GOOD_TEST
    check("extract_c reads a fenced program", extract.extract_c(fenced).strip()
          == GOOD_TEST.strip())
    fragment = ("```c\nstatic int helper(void) { return 0; }\n```\n\n"
                "```c\n%s```\n" % GOOD_TEST)
    check("extract_c prefers the block with main",
          "helper" not in extract.extract_c(fragment))
    bare = extract.extract_c("Sure.\n\n" + GOOD_TEST)
    check("extract_c reads a bare answer", bare.startswith("#include <stddef.h>"))


def test_guard() -> None:
    check("a full program is usable", graph._c_source_problem(GOOD_TEST) == "")
    check("a program without main is refused",
          "main" in graph._c_source_problem(GOOD_TEST.replace("int main(void)", "int run(void)")))
    check("a program without the header is refused",
          "sonnx_kernels.h" in graph._c_source_problem(
              GOOD_TEST.replace('#include "sonnx_kernels.h"', "")))
    narrow = GOOD_TEST.replace("sonnx_sub_i8", "sonnx_sub_f32").replace(
        "sonnx_sub_i32", "sonnx_sub_f32").replace(
        "sonnx_sub_u8", "sonnx_sub_f32").replace(
        "sonnx_sub_u32", "sonnx_sub_f32").replace(
        "sonnx_sub_i64", "sonnx_sub_f32").replace(
        "sonnx_sub_f64", "sonnx_sub_f32")
    check("a program without integer kernels is refused",
          "integer" in graph._c_source_problem(narrow))
    check("an empty answer is refused", "empty" in graph._c_source_problem("  \n"))


def test_render(directory: Path) -> Settings:
    settings = settings_for(directory)
    emitted = graph._emit_c_files(settings, SPEC, SUB_DERIVATIONS)
    check("the derivation trailer renders the module's C", emitted["ok"])
    check("the driver prints the derived name and operation",
          emitted["name"] == "sub" and emitted["binop"] == "sub")
    check("the derivation reads the family declarations",
          [source["declaration"] for source in emitted["sources"]]
          == ["Generated.Sub.realSub", "Generated.Sub.floatSub",
              "Generated.Sub.intSub", "Generated.Sub.uintSub"])
    check("the header is written", Path(emitted["header"]).is_file())
    check("the source is written", Path(emitted["source"]).is_file())
    if not emitted["ok"]:
        sys.stderr.write(str(emitted.get("compiler")))
        return settings
    header = Path(emitted["header"]).read_text()
    source = Path(emitted["source"]).read_text()
    check("the header declares the tensor struct", "} sonnx_tensor;" in header)
    check("the header declares the f32 kernel",
          "void sonnx_sub_f32(const sonnx_tensor *a, const sonnx_tensor *b, "
          "sonnx_tensor *c);" in header)
    check("the source defines the u64 kernel",
          "void sonnx_sub_u64(const sonnx_tensor *a, const sonnx_tensor *b, "
          "sonnx_tensor *c) {" in source)
    check("the source includes the header", '#include "sonnx_kernels.h"' in source)
    check("the header comment names the derivation",
          "derived from Generated.Sub.floatSub" in header
          and "E_SUB_FLOAT_FUNC_0010" in header)
    check("the definition comment names the derivation",
          "derived from Generated.Sub.floatSub" in source)
    return settings


def test_generate(directory: Path) -> None:
    settings = settings_for(directory)
    descriptor = write(directory, "FormalSpec.lean", SPEC)
    write_trace(settings, [{"key": item["requirement"], "declaration": item["declaration"],
                            "source": item["source"]} for item in SUB_DERIVATIONS])
    state = {"lean_path": str(descriptor), "spec_path": str(descriptor)}
    accepted = graph.generate_c(state, settings)
    check("generate_c derives a supported operation",
          accepted["cgen_status"] == "generated" and not accepted["cgen_ok"])
    check("generate_c records the two files", Path(accepted["cgen_header"]).is_file()
          and Path(accepted["cgen_source"]).is_file())
    check("generate_c records the derivation", Path(accepted["cgen_trace"]).is_file())
    trace = json.loads(Path(accepted["cgen_trace"]).read_text())
    check("the manifest names every family",
          [item["family"] for item in trace["derivations"]]
          == ["real", "float", "int", "uint"])
    check("the manifest maps the kernels to their family",
          [kernel["family"] for kernel in trace["kernels"][:3]]
          == ["float", "float", "int"])

    div_descriptor = write(directory, "Div.lean", SPEC_DIV)
    write_trace(settings, [{"key": item["requirement"], "declaration": item["declaration"],
                            "source": item["source"]} for item in DIV_DERIVATIONS])
    div_settings = settings_for(directory, "Div")
    unsupported = graph.generate_c(
        {"lean_path": str(div_descriptor), "spec_path": str(div_descriptor)}, div_settings)
    check("a division derivation is unsupported, not an error",
          unsupported["cgen_status"] == "unsupported")
    check("the unsupported reason names the operation", "div" in unsupported["cgen_error"])
    check("the unsupported derivation is still traced",
          Path(unsupported["cgen_trace"]).is_file())
    check("no C files are left for an unsupported operation",
          not div_settings.c_header_path().is_file()
          and not div_settings.c_source_path().is_file())

    absent_descriptor = write(directory, "None.lean", SPEC_NONE)
    write_trace(settings, [])
    absent = graph.generate_c(
        {"lean_path": str(absent_descriptor), "spec_path": str(absent_descriptor)},
        settings)
    check("a specification with no family declaration is unsupported",
          absent["cgen_status"] == "unsupported")
    check("the reason names the missing family", "family" in absent["cgen_error"])

    broken_descriptor = write(directory, "Broken.lean",
                              SPEC + "\ndef broken : Nat :=\n")
    write_trace(settings, [{"key": item["requirement"], "declaration": item["declaration"],
                            "source": item["source"]} for item in SUB_DERIVATIONS])
    broken = graph.generate_c(
        {"lean_path": str(broken_descriptor), "spec_path": str(broken_descriptor)},
        settings)
    check("a module that does not compile is an error", broken["cgen_status"] == "error")


def test_protocol(settings: Settings) -> None:
    # the test program sits beside the generated header, as every attempt the C tester
    # writes does, so its quoted include resolves
    directory = settings.c_dir()
    good = write(directory, "good-tests.c", GOOD_TEST)
    result = graph._compile_and_run_c(settings, good)
    check("a correct test program passes", result["ok"])
    check("its checks are counted", result["checks"] == {"total": 14, "passed": 14,
                                                         "failed": 0})
    check("a passing program exits 0", result["returncode"] == 0)

    bad = write(directory, "bad-tests.c",
                GOOD_TEST.replace("c_data[0] == 2.0f", "c_data[0] == 2.5f"))
    result = graph._compile_and_run_c(settings, bad)
    check("a wrong expectation fails", not result["ok"] and result["failures"])
    check("the failing line is kept", result["failures"][0].startswith("FAIL f32 5-3"))
    check("a failing expectation is not unusable", result["unusable"] == "")

    broken = write(directory, "broken-tests.c",
                   GOOD_TEST.replace("return failures == 0 ? 0 : 1;", "return"))
    result = graph._compile_and_run_c(settings, broken)
    check("a program that does not compile is unusable", result["unusable"]
          and "compile" in result["unusable"])

    silent = write(directory, "silent-tests.c",
                   GOOD_TEST.replace('printf("ok %s\\n", name);', '(void)name;'))
    result = graph._compile_and_run_c(settings, silent)
    check("a program that compares nothing is unusable", result["unusable"]
          and "no comparison" in result["unusable"])


def test_checked_in_specs(work: Path) -> None:
    """Derive the checked-in formal specifications without a model call.

    The real `_emit_c_files` reads the family declarations and compiles the copy; Add, Sub
    and Mul render, Div derives `.div` and is classified unsupported.
    """
    repo = Path(__file__).resolve().parents[2]
    for op in ("add", "sub", "mul", "div"):
        source = repo / "ops" / op / "lean" / "FormalSpec.lean"
        trace = json.loads(source.with_name("FormalSpec.trace.json").read_text())
        settings = Settings(op=op.capitalize(), run_dir=work / op, lean=True, cgen=True,
                            verbose=False)
        derivations = graph._cgen_derivations(op, trace)
        check("%s: the trace names every family" % op, len(derivations) == 4)
        emitted = graph._emit_c_files(settings, source.read_text(), derivations)
        check("%s: the derivation succeeds" % op, emitted["ok"])
        check("%s: the derived operation" % op, emitted["binop"] == op)
        if op in ("add", "sub", "mul"):
            check("%s: the C files are rendered" % op,
                  emitted["header_written"] and emitted["source_written"])
        else:
            check("div: no C files are rendered",
                  not emitted["header_written"] and not emitted["source_written"])


def main() -> int:
    if shutil.which(lean.find_lake()) is None:
        sys.stderr.write("lake not found\n")
        return 1
    cc = shutil.which("cc")
    if cc is None:
        sys.stderr.write("cc not found\n")
        return 1
    test_extract()
    test_guard()
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        settings = test_render(directory)
        generate_dir = directory / "generate"
        generate_dir.mkdir()
        test_generate(generate_dir)
        test_checked_in_specs(directory / "checked-in")
        test_protocol(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
