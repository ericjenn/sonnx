#!/usr/bin/env python3
"""Tests for the Lean C generator: printer probes, emitted C, and compilation."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonnx_agents import lean  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LEAN = REPO / "lean"

OPERATORS = ("add", "sub", "mul")
SUFFIXES = ("f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64")

BANNER_CONTRACT = "Broadcasting is the caller's responsibility"


def check(name: str, condition: bool) -> None:
    assert condition, name
    print("  ok  %s" % name)


def run(command: list[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout)


def emit_c(lake: str, destination: Path | None) -> subprocess.CompletedProcess:
    command = [lake, "env", "lean", "--run", "CGen/Emit.lean"]
    if destination is not None:
        command.append(str(destination))
    return run(command, cwd=LEAN, timeout=120)


def emit_split(lake: str, header: Path, source: Path) -> subprocess.CompletedProcess:
    """Render the operators' split header and source through the CGen API.

    The pipeline compiles a copy of the formal module that renders its descriptor this way;
    this driver renders the library's own operator table with the same calls.
    """
    driver = header.parent / "Split.lean"
    quote = lambda path: str(path).replace("\\", "\\\\").replace('"', '\\"')
    driver.write_text(
        "import CGen\n"
        "\n"
        "#eval do\n"
        '  IO.FS.writeFile "%s" (CGen.renderFile (CGen.headerFileFor CGen.operators '
        "CGen.dtypes))\n"
        '  IO.FS.writeFile "%s" (CGen.renderFile (CGen.sourceFileFor CGen.operators '
        "CGen.dtypes))\n" % (quote(header), quote(source)),
        encoding="utf-8")
    return run([lake, "env", "lean", str(driver)], cwd=LEAN, timeout=120)


def main() -> int:
    lake = lean.find_lake()

    built = run([lake, "build", "CGen"], cwd=LEAN, timeout=600)
    check("the CGen library builds", built.returncode == 0)
    if built.returncode != 0:
        sys.stderr.write(built.stdout + built.stderr)
        return 1

    probe = run([lake, "env", "lean", "--run", "CGen/Probe.lean"], cwd=LEAN, timeout=120)
    check("the printer probes pass", probe.returncode == 0)
    check("the probes report their count", "probe ok:" in probe.stdout)
    if probe.returncode != 0:
        sys.stderr.write(probe.stdout + probe.stderr)
        return 1

    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        path = directory / "operators.c"
        written = emit_c(lake, path)
        check("the emit driver succeeds", written.returncode == 0)
        check("the emit driver writes the file", path.is_file())
        if not path.is_file():
            sys.stderr.write(written.stdout + written.stderr)
            return 1
        text = path.read_text()

        check("the file is non-trivial", len(text) > 4000)
        check("the file ends with one newline", text.endswith("\n") and not text.endswith("\n\n"))
        check("the file uses two-space indentation", "\t" not in text)
        check("the braces balance", text.count("{") == text.count("}"))

        check("the banner states the broadcasting contract", BANNER_CONTRACT in text)
        check("the file includes <stddef.h>", "#include <stddef.h>" in text)
        check("the file includes <stdint.h>", "#include <stdint.h>" in text)
        check("the file omits float16", "f16" not in text)

        expected = {"sonnx_%s_%s" % (op, suffix) for op in OPERATORS for suffix in SUFFIXES}
        declarations = re.findall(r"^void (sonnx_[A-Za-z0-9_]+)\(([^)]*)\)", text, re.MULTILINE)
        names = [name for name, _ in declarations]
        check("every operator and element type is generated", set(names) == expected)
        check("each kernel is declared and defined", len(names) == 2 * len(expected))
        parameters: dict[str, str] = {}
        agree = True
        for name, params in declarations:
            if name in parameters and parameters[name] != params:
                agree = False
            parameters.setdefault(name, params)
        check("each prototype matches its definition", agree)
        check("the declarations are unique pairs",
              all(names.count(name) == 2 for name in expected))

        check("the loop is size_t and contiguous",
              "for (size_t i = 0; i < n; i++) {" in text)
        check("a float kernel applies the operator directly",
              "c[i] = a[i] + b[i];" in text)
        check("a float64 kernel uses double",
              "void sonnx_add_f64(const double *a, const double *b, double *c, size_t n)"
              in text)
        check("a signed kernel wraps in the unsigned type",
              "c[i] = (int8_t)((uint8_t)a[i] - (uint8_t)b[i]);" in text)
        check("a signed 64-bit kernel widens through uint64_t",
              "c[i] = (int64_t)((uint64_t)a[i] + (uint64_t)b[i]);" in text)
        check("an unsigned kernel converts modulo the width",
              "c[i] = (uint16_t)(a[i] * b[i]);" in text)

        again = directory / "operators-again.c"
        check("the emit driver succeeds again", emit_c(lake, again).returncode == 0)
        check("the output is deterministic", again.read_text() == text)

        printed = emit_c(lake, None)
        check("the driver prints to stdout without a path", printed.returncode == 0)
        check("stdout is the same translation unit", printed.stdout == text)

        cc = shutil.which("cc")
        if cc is None:
            sys.stderr.write("cc not found; the generated C cannot be verified\n")
            return 1
        compiled = run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-fsyntax-only", str(path)], cwd=directory, timeout=120)
        check("the generated C compiles cleanly", compiled.returncode == 0)
        if compiled.returncode != 0:
            sys.stderr.write(compiled.stderr)
            return 1

        header = directory / "sonnx_kernels.h"
        source = directory / "sonnx_kernels.c"
        split = emit_split(lake, header, source)
        check("the split driver succeeds", split.returncode == 0)
        check("the split driver writes both files", header.is_file() and source.is_file())
        if not (header.is_file() and source.is_file()):
            sys.stderr.write(split.stdout + split.stderr)
            return 1
        header_text = header.read_text()
        source_text = source.read_text()

        check("the header includes <stddef.h>", "#include <stddef.h>" in header_text)
        check("the header includes <stdint.h>", "#include <stdint.h>" in header_text)
        check("the header holds no function body", "{" not in header_text)
        prototypes = re.findall(r"^void (sonnx_[A-Za-z0-9_]+)\(([^)]*)\);$", header_text,
                                re.MULTILINE)
        check("the header declares every kernel once",
              sorted(name for name, _ in prototypes) == sorted(expected))
        check("the header omits float16", "f16" not in header_text)

        check("the source includes its header", '#include "sonnx_kernels.h"' in source_text)
        check("the source does not repeat the standard includes",
              "#include <stddef.h>" not in source_text
              and "#include <stdint.h>" not in source_text)
        definitions = re.findall(r"^void (sonnx_[A-Za-z0-9_]+)\(([^)]*)\) \{", source_text,
                                 re.MULTILINE)
        check("the source defines every kernel once",
              sorted(name for name, _ in definitions) == sorted(expected))
        check("the source holds no prototype", ";$" not in
              "\n".join(line for line in source_text.splitlines()
                        if line.startswith("void ")))
        check("the header and source agree on every signature",
              dict(prototypes) == dict(definitions))

        split_compiled = run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
                              "-fsyntax-only", str(source)], cwd=directory, timeout=120)
        check("the split source compiles cleanly", split_compiled.returncode == 0)
        if split_compiled.returncode != 0:
            sys.stderr.write(split_compiled.stderr)
            return 1

        checker = directory / "check.c"
        checker.write_text(
            "#include <stdint.h>\n"
            "#include <stdio.h>\n"
            '#include "operators.c"\n'
            "\n"
            "static int failures;\n"
            "\n"
            "static void check(const char *name, int ok) {\n"
            '  if (!ok) { fprintf(stderr, "FAILED %s\\n", name); failures++; }\n'
            "}\n"
            "\n"
            "int main(void) {\n"
            "  { int8_t a[2] = {127, 100}, b[2] = {1, 100}, c[2] = {0, 0};\n"
            "    sonnx_add_i8(a, b, c, 2);\n"
            '    check("int8 add wraps", c[0] == -128 && c[1] == -56); }\n'
            "  { uint8_t a[2] = {0, 255}, b[2] = {1, 2}, c[2] = {0, 0};\n"
            "    sonnx_sub_u8(a, b, c, 2);\n"
            '    check("uint8 sub wraps", c[0] == 255 && c[1] == 253); }\n'
            "  { double a[1] = {0.5}, b[1] = {2.0}, c[1] = {0.0};\n"
            "    sonnx_mul_f64(a, b, c, 1);\n"
            '    check("double mul", c[0] == 1.0); }\n'
            "  return failures == 0 ? 0 : 1;\n"
            "}\n"
        )
        program = directory / "check"
        built_checker = run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
                             str(checker), "-o", str(program)], cwd=directory, timeout=120)
        check("the semantic checker compiles", built_checker.returncode == 0)
        if built_checker.returncode != 0:
            sys.stderr.write(built_checker.stderr)
            return 1
        ran = run([str(program)], cwd=directory, timeout=120)
        check("the kernels wrap and compute as specified", ran.returncode == 0)
        if ran.returncode != 0:
            sys.stderr.write(ran.stdout + ran.stderr)
            return 1

        split_checker = directory / "check-split.c"
        split_checker.write_text(
            "#include <stdint.h>\n"
            '#include "sonnx_kernels.h"\n'
            "\n"
            "int main(void) {\n"
            "  { const float a[2] = {5.0f, -1.0f}, b[2] = {3.0f, -1.0f};\n"
            "    float c[2] = {0.0f, 0.0f};\n"
            "    sonnx_sub_f32(a, b, c, 2);\n"
            "    if (c[0] != 2.0f || c[1] != 0.0f) return 1; }\n"
            "  { const int32_t a[1] = {INT32_MIN}, b[1] = {1};\n"
            "    int32_t c[1] = {0};\n"
            "    sonnx_sub_i32(a, b, c, 1);\n"
            "    if (c[0] != INT32_MAX) return 1; }\n"
            "  return 0;\n"
            "}\n"
        )
        split_binary = directory / "check-split"
        built_split = run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
                           str(split_checker), str(source), "-o", str(split_binary)],
                          cwd=directory, timeout=120)
        check("the split translation units compile and link", built_split.returncode == 0)
        if built_split.returncode != 0:
            sys.stderr.write(built_split.stderr)
            return 1
        ran_split = run([str(split_binary)], cwd=directory, timeout=120)
        check("the split kernels compute as specified", ran_split.returncode == 0)
        if ran_split.returncode != 0:
            sys.stderr.write(ran_split.stdout + ran_split.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
