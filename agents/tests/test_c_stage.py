#!/usr/bin/env python3
"""Tests for the C stage: rendering, the descriptor guards, and the test protocol.

No model is called.  The rendering tests compile a small module with `lake env lean`; the
protocol tests compile and run fixed C programs with `cc`.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonnx_agents import extract, graph, lean  # noqa: E402
from sonnx_agents.config import Settings  # noqa: E402

SPEC = """import CGen

def sonnxCGenOp : CGen.Elementwise := ⟨"sub", .sub⟩
"""

SPEC_DIV = """import CGen

def sonnxCGenOp : CGen.Elementwise := ⟨"sub", .div⟩
"""

SPEC_NONE = """import Sonnx

def thing : Nat := 0
"""

GOOD_TEST = """#include <stdint.h>
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
  { const float a[2] = {5.0f, -1.0f}, b[2] = {3.0f, -1.0f};
    float c[2] = {0.0f, 0.0f};
    sonnx_sub_f32(a, b, c, 2);
    ok("f32 5-3", c[0] == 2.0f);
    ok("f32 -1 minus -1", c[1] == 0.0f); }
  { const double a[2] = {0.25, 1.5}, b[2] = {0.5, 2.5};
    double c[2] = {0.0, 0.0};
    sonnx_sub_f64(a, b, c, 2);
    ok("f64 0.25-0.5", c[0] == -0.25);
    ok("f64 1.5-2.5", c[1] == -1.0); }
  { const int8_t a[2] = {127, -128}, b[2] = {1, 1};
    int8_t c[2] = {0, 0};
    sonnx_sub_i8(a, b, c, 2);
    ok("i8 127-1", c[0] == 126);
    ok("i8 -128-1 wraps", c[1] == 127); }
  { const int32_t a[2] = {5, INT32_MIN}, b[2] = {3, 1};
    int32_t c[2] = {0, 0};
    sonnx_sub_i32(a, b, c, 2);
    ok("i32 5-3", c[0] == 2);
    ok("i32 INT32_MIN-1 wraps", c[1] == INT32_MAX); }
  { const uint8_t a[2] = {0, 200}, b[2] = {1, 100};
    uint8_t c[2] = {0, 0};
    sonnx_sub_u8(a, b, c, 2);
    ok("u8 0-1 wraps", c[0] == 255);
    ok("u8 200-100", c[1] == 100); }
  { const uint32_t a[2] = {0, 7}, b[2] = {1, 7};
    uint32_t c[2] = {0, 0};
    sonnx_sub_u32(a, b, c, 2);
    ok("u32 0-1 wraps", c[0] == UINT32_MAX);
    ok("u32 7-7", c[1] == 0); }
  { const float a[1] = {1.0f}, b[1] = {2.0f};
    float c[1] = {0.0f};
    sonnx_sub_f32(a, b, c, 1);
    ok("f32 1-2", c[0] == -1.0f); }
  { const int64_t a[1] = {INT64_MAX}, b[1] = {-1};
    int64_t c[1] = {0};
    sonnx_sub_i64(a, b, c, 1);
    ok("i64 INT64_MAX minus -1 wraps", c[0] == INT64_MIN); }
  return failures == 0 ? 0 : 1;
}
"""


def check(name: str, condition: bool) -> None:
    assert condition, name
    print("  ok  %s" % name)


def settings_for(directory: Path) -> Settings:
    return Settings(op="Sub", opset=14, run_dir=directory, lean=True, lean_only=True,
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
    check("extract_c reads a bare answer", bare.startswith("#include <stdint.h>"))


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
    emitted = graph._emit_c_files(settings, SPEC)
    check("the emit trailer renders the descriptor", emitted["ok"])
    check("the driver prints the descriptor's name and operation",
          emitted["name"] == "sub" and emitted["binop"] == "sub")
    check("the header is written", Path(emitted["header"]).is_file())
    check("the source is written", Path(emitted["source"]).is_file())
    if not emitted["ok"]:
        sys.stderr.write(str(emitted.get("compiler")))
        return settings
    header = Path(emitted["header"]).read_text()
    source = Path(emitted["source"]).read_text()
    check("the header declares the f32 kernel",
          "void sonnx_sub_f32(const float *a, const float *b, float *c, size_t n);" in header)
    check("the source defines the u64 kernel",
          "void sonnx_sub_u64(const uint64_t *a, const uint64_t *b, uint64_t *c, size_t n) {"
          in source)
    check("the source includes the header", '#include "sonnx_kernels.h"' in source)
    return settings


def test_generate(directory: Path) -> None:
    settings = settings_for(directory)
    descriptor = write(directory, "FormalSpec.lean", SPEC)
    accepted = graph.generate_c({"lean_path": str(descriptor)}, settings)
    check("generate_c renders a supported descriptor",
          accepted["cgen_status"] == "generated" and not accepted["cgen_ok"])
    check("generate_c records the two files", Path(accepted["cgen_header"]).is_file()
          and Path(accepted["cgen_source"]).is_file())

    unsupported = graph.generate_c(
        {"lean_path": str(write(directory, "Div.lean", SPEC_DIV))}, settings)
    check("a division descriptor is unsupported, not an error",
          unsupported["cgen_status"] == "unsupported")
    check("an unsupported descriptor names the reason", "div" in unsupported["cgen_error"])

    absent = graph.generate_c(
        {"lean_path": str(write(directory, "None.lean", SPEC_NONE))}, settings)
    check("an absent descriptor is unsupported", absent["cgen_status"] == "unsupported")
    check("an absent descriptor names the reason",
          "sonnxCGenOp" in absent["cgen_error"])

    broken = graph.generate_c(
        {"lean_path": str(write(directory, "Broken.lean",
                                SPEC + "\ndef broken : Nat :=\n"))},
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

    bad = write(directory, "bad-tests.c", GOOD_TEST.replace("c[0] == 2.0f", "c[0] == 2.5f"))
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
        test_protocol(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
