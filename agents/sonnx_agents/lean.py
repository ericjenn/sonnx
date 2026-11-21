"""Deterministic Lean source traceability, validation and compilation.

The model writes Lean plus small ``-- @trace`` markers. This module, not the
model, resolves those markers to the defining clauses of the informal document,
adds source links and excerpts, checks coverage and invokes Lean. A successful
compile is deliberately reported separately from semantic completeness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import REPO

TAG_RE = re.compile(r"E_[A-Z0-9]+_[A-Z0-9_]+_\d{4}")
STATUSES = {"specified", "proved", "example_only", "partial", "deferred"}
INCOMPLETE = {"partial", "deferred"}

# The shared Lean library: its sources are hashed into every generated artifact, and
# the names it exports may not be declared again by an operator module.
LIBRARY_MODULE = "Sonnx"
# The C generator.  A generated module may import it to declare the C descriptor the C
# stage renders; it is not part of the library digest, so an operator module is not stale
# just because the generator changed.
CGEN_MODULE = "CGen"
ALLOWED_IMPORT_ROOTS = {"Init", "Std", "Lean", CGEN_MODULE}
DECLARATION_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*"
    r"(?:private\s+|protected\s+|noncomputable\s+|partial\s+)*"
    r"(?:structure|inductive|class|def|abbrev|theorem|lemma)\s+([A-Za-z_\u00ab][\w.\u00ab\u00bb']*)")
IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][\w.']*)")

# The top-level declaration by which an operator module describes its C kernels.
CGEN_DESCRIPTOR = "sonnxCGenOp"
CGEN_DEF_RE = re.compile(r"^def\s+" + CGEN_DESCRIPTOR + r"\b")
CGEN_NAME_RE = re.compile(r"\b" + CGEN_DESCRIPTOR + r"\b")


def read_source(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")


def source_name(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def source_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SourceItem:
    key: str
    start: int
    end: int
    excerpt: str
    anchor: str | None = None

    def json(self, source: str) -> dict:
        return {"key": self.key, "source": source, "start": self.start,
                "end": self.end, "anchor": self.anchor, "excerpt": self.excerpt}


def _trimmed_span(lines: list[str], start: int, end: int) -> tuple[int, int]:
    while end >= start and not lines[end - 1].strip():
        end -= 1
    while start <= end and not lines[start - 1].strip():
        start += 1
    return start, end


def source_items(spec_path: Path) -> tuple[list[SourceItem], list[SourceItem]]:
    """Return defining requirement and example spans, in document order.

    Function clauses end at their red ``[END]``. A constraint's scope is its
    own list item, including indented Statement continuations; a later hyperlink
    to that constraint is not its definition.
    """
    text = read_source(spec_path)
    lines = text.splitlines()
    requirements: list[SourceItem] = []
    seen: set[str] = set()
    opening = re.compile(r"background:\s*red.*?\[(" + TAG_RE.pattern + r")\]", re.I)
    defining = re.compile(r"^(\s*)-\s+`\[(" + TAG_RE.pattern + r")\]`")
    for index, line in enumerate(lines):
        opening_match = opening.search(line)
        defining_match = defining.match(line)
        if not opening_match and not defining_match:
            continue
        found = TAG_RE.search(line)
        if not found:
            continue
        tag = found.group(0)
        if tag in seen:
            raise ValueError("duplicate defining clause %s at line %d" % (tag, index + 1))
        seen.add(tag)
        start = index + 1
        anchor = None
        if index and re.match(r'^\s*<a id="' + re.escape(tag) + r'"></a>\s*$', lines[index - 1]):
            start = index
            anchor = tag
        if opening_match:
            end = None
            for cursor in range(index + 1, len(lines)):
                if re.search(r"background:\s*red.*?\[END\]", lines[cursor], re.I):
                    end = cursor + 1
                    break
            if end is None:
                raise ValueError("unclosed defining clause %s at line %d" % (tag, start))
        else:
            base_indent = len(defining_match.group(1))
            end = len(lines)
            for cursor in range(index + 1, len(lines)):
                following = lines[cursor]
                if (not following.strip() or re.match(r"^#{1,6}\s", following)
                        or re.match(r'^\s*<a id="E_[A-Z0-9]+_[A-Z0-9_]+_\d{4}"></a>', following)):
                    end = cursor
                    break
                item = re.match(r"^(\s*)-\s+", following)
                if item and len(item.group(1)) <= base_indent:
                    end = cursor
                    break
        start, end = _trimmed_span(lines, start, end)
        requirements.append(SourceItem(tag, start, end, "\n".join(lines[start - 1:end]), anchor))

    all_tags = set(TAG_RE.findall(text))
    defined_tags = {item.key for item in requirements}
    if all_tags != defined_tags:
        raise ValueError("defining clauses differ from source tags: missing=%s unknown=%s" % (
            sorted(all_tags - defined_tags), sorted(defined_tags - all_tags)))

    examples: list[SourceItem] = []
    family = None
    for index, line in enumerate(lines):
        heading = re.match(r"^#\s+\*\*[^*]+\*\*\s+\(([^,)]+)", line)
        if heading:
            family = heading.group(1).strip()
        example = re.match(r"^###\s+Example\s+(\d+)\s*$", line)
        if not example:
            continue
        if family not in {"real", "float", "int", "uint"}:
            raise ValueError("example at line %d has no type-family section" % (index + 1))
        end = len(lines)
        for cursor in range(index + 1, len(lines)):
            if re.match(r"^#{1,3}\s", lines[cursor]):
                end = cursor
                break
        start, end = _trimmed_span(lines, index + 1, end)
        key = "%s:example:%s" % (family, example.group(1))
        examples.append(SourceItem(key, start, end, "\n".join(lines[start - 1:end])))
    return requirements, examples


def inventory(spec_path: Path) -> dict:
    requirements, examples = source_items(Path(spec_path))
    source = source_name(Path(spec_path))
    return {"source": source, "source_md5": source_md5(Path(spec_path)),
            "requirements": [item.json(source) for item in requirements],
            "examples": [item.json(source) for item in examples]}


def _attrs(text: str) -> dict[str, str]:
    try:
        words = shlex.split(text)
    except ValueError as exc:
        raise ValueError("malformed trace attributes: %s" % exc) from exc
    result = {}
    for word in words:
        if "=" not in word:
            raise ValueError("trace attribute %r is not key=value" % word)
        key, value = word.split("=", 1)
        result[key] = value
    return result


def parse_markers(text: str) -> tuple[list[dict], list[str]]:
    entries, errors = [], []
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"^\s*--\s*@trace\s+(\S+)(.*)$", line)
        if not match:
            continue
        try:
            attrs = _attrs(match.group(2))
            status = attrs.pop("status", "")
            declaration = attrs.pop("decl", "")
            limitation = attrs.pop("limitation", "")
            if attrs:
                raise ValueError("unknown attributes %s" % ", ".join(sorted(attrs)))
            entries.append({"key": match.group(1), "status": status,
                            "declaration": declaration, "limitation": limitation,
                            "line": number})
        except ValueError as exc:
            errors.append("line %d: %s" % (number, exc))
    return entries, errors


def parse_materialized(text: str) -> tuple[list[dict], list[str]]:
    entries = []
    pattern = re.compile(r"@trace\s+(\S+)\s*\n(.*?)(?=\n\s*@trace\s|\n\s*-/|\Z)", re.S)
    for match in pattern.finditer(text):
        body = match.group(2)
        def field(name: str) -> str:
            found = re.search(r"^\s*@" + name + r"\s*(.*?)\s*$", body, re.M)
            return found.group(1).strip() if found else ""
        excerpt_at = re.search(r"^Informal source excerpt:[ \t]*\n", body, re.M)
        excerpt = ""
        if excerpt_at:
            excerpt = "\n".join(line[2:] if line.startswith("  ") else line
                                for line in body[excerpt_at.end():].splitlines())
        entries.append({"key": match.group(1), "status": field("coverage"),
                        "declaration": field("declaration"),
                        "limitation": field("limitation"), "source": field("source"),
                        "excerpt": excerpt,
                        "line": text.count("\n", 0, match.start()) + 1})
    return entries, []


def _canonical(inv: dict) -> dict[str, dict]:
    return {item["key"]: item for item in inv["requirements"] + inv["examples"]}


def validate_entries(entries: list[dict], errors: list[str], inv: dict,
                     materialized: bool = False) -> dict:
    canonical = _canonical(inv)
    findings = list(errors)
    counts: dict[str, int] = {}
    for entry in entries:
        key = entry["key"]
        counts[key] = counts.get(key, 0) + 1
        if key not in canonical:
            findings.append("unknown trace key %s" % key)
            continue
        if entry["status"] not in STATUSES:
            findings.append("%s: invalid coverage status %r" % (key, entry["status"]))
        if entry["status"] in INCOMPLETE and not entry["limitation"]:
            findings.append("%s: %s coverage requires @limitation" % (key, entry["status"]))
        if entry["status"] in {"specified", "proved", "example_only"} and not entry["declaration"]:
            findings.append("%s: %s coverage requires a declaration" % (key, entry["status"]))
        if materialized:
            item = canonical[key]
            wanted = "%s:L%d-L%d" % (item["source"], item["start"], item["end"])
            if entry.get("source") != wanted:
                findings.append("%s: source is %r, expected %r" % (key, entry.get("source"), wanted))
            if entry.get("excerpt") != item["excerpt"]:
                findings.append("%s: source excerpt is missing or does not match %s" % (key, wanted))
    for key in canonical:
        if counts.get(key, 0) == 0:
            findings.append("missing trace entry %s" % key)
        elif counts[key] != 1:
            findings.append("duplicate trace entry %s (%d primary entries)" % (key, counts[key]))
    coverage = {key: next((e["status"] for e in entries if e["key"] == key), "deferred")
                for key in canonical}
    complete = not findings and all(
        coverage[item["key"]] in {"specified", "proved"} for item in inv["requirements"])
    return {"ok": not findings, "complete": complete, "findings": findings,
            "coverage": coverage, "entries": entries}


def _code_lines(text: str) -> list[tuple[int, str]]:
    """The lines of ``text`` with line and nested block comments removed."""
    result = []
    depth = 0
    for number, line in enumerate(text.splitlines(), 1):
        if depth == 0 and line.lstrip().startswith("--"):
            result.append((number, ""))
            continue
        cleaned, cursor = [], 0
        while cursor < len(line):
            if depth == 0 and line.startswith("/-", cursor):
                depth, cursor = 1, cursor + 2
            elif depth > 0 and line.startswith("/-", cursor):
                depth, cursor = depth + 1, cursor + 2
            elif depth > 0 and line.startswith("-/", cursor):
                depth, cursor = depth - 1, cursor + 2
            else:
                if depth == 0:
                    cleaned.append(line[cursor])
                cursor += 1
        result.append((number, "".join(cleaned)))
    return result


def forbidden_constructs(text: str) -> list[str]:
    """Reject proof escapes in code, ignoring line and nested block comments."""
    findings = []
    for number, code in _code_lines(text):
        for pattern, label in ((r"\bsorry\b", "sorry"), (r"\badmit\b", "admit"),
                               (r"^\s*axiom\b", "axiom"), (r"^\s*unsafe\b", "unsafe")):
            if re.search(pattern, code):
                findings.append("line %d: forbidden Lean construct %s" % (number, label))
    return findings


def _library_root(root: Path | None = None) -> Path:
    return Path(root) if root else REPO / "lean"


def _library_sources(root: Path | None = None) -> list[Path]:
    base = _library_root(root)
    files = [base / "Sonnx.lean"]
    directory = base / "Sonnx"
    if directory.is_dir():
        files.extend(sorted(directory.glob("*.lean")))
    return [path for path in files if path.is_file()]


def library_digest(root: Path | None = None) -> str:
    """A digest of the shared library sources and the package that builds them.

    Recorded in every generated module and trace report, and in the conformance
    manifest, so that a module is recognisably stale once the library changes.
    """
    base = _library_root(root)
    files = [base / "lakefile.lean", base / "lean-toolchain"] + _library_sources(base)
    digest = hashlib.sha256()
    for path in files:
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError:
            relative = str(path)
        digest.update(relative.encode("utf-8") + b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"missing")
        digest.update(b"\0")
    return digest.hexdigest()


def shared_names(root: Path | None = None) -> set[str]:
    """The top-level names the shared library exports."""
    names: set[str] = set()
    for path in _library_sources(root):
        for _number, code in _code_lines(path.read_text(encoding="utf-8")):
            match = DECLARATION_RE.match(code)
            if match:
                names.add(re.split(r"[.]", match.group(1))[0])
    return names


def import_findings(text: str, require_sonnx: bool = True) -> list[str]:
    """Reject imports outside the shared library and Lean core."""
    findings = []
    found = False
    for number, code in _code_lines(text):
        match = IMPORT_RE.match(code)
        if not match:
            continue
        module = match.group(1)
        if module == LIBRARY_MODULE or module.startswith(LIBRARY_MODULE + "."):
            found = True
        elif module.split(".")[0] in ALLOWED_IMPORT_ROOTS:
            continue
        else:
            findings.append("line %d: import %s is not allowed; import %s instead"
                            % (number, module, LIBRARY_MODULE))
    if require_sonnx and not found:
        findings.append("the module must import the shared library: import %s"
                        % LIBRARY_MODULE)
    return findings


def shared_redefinition_findings(text: str, root: Path | None = None) -> list[str]:
    """Reject a declaration that redefines a name the shared library exports."""
    shared = shared_names(root)
    findings = []
    for number, code in _code_lines(text):
        match = DECLARATION_RE.match(code)
        if not match:
            continue
        name = match.group(1)
        head = re.split(r"[.]", name)[0]
        if head in shared:
            findings.append("line %d: %s redefines the shared declaration %s; "
                            "use %s.%s" % (number, name, head, LIBRARY_MODULE, head))
    return findings


_NAMESPACE_RE = re.compile(r"^namespace\s+\S+")
_END_RE = re.compile(r"^end(?:\s+\S+)?\s*$")


def _cgen_definitions(text: str) -> tuple[list[str], list[str]]:
    """The `sonnxCGenOp` definitions at the top level, and those inside a namespace.

    Lean does not require a namespace's body to be indented, so the declaration's column
    says nothing about its scope: the `namespace`/`end` lines are what place it.
    """
    top: list[str] = []
    nested: list[str] = []
    depth = 0
    for _number, line in _code_lines(text):
        stripped = line.strip()
        if depth == 0 and _NAMESPACE_RE.match(stripped):
            depth += 1
            continue
        if depth and _END_RE.match(stripped):
            depth -= 1
            continue
        if CGEN_DEF_RE.match(line):
            (nested if depth else top).append(line)
    return top, nested


def cgen_descriptor(text: str) -> bool:
    """Does the module declare the C descriptor at the top level?

    The C stage reads the accepted artifact, not the model's raw answer, and the
    declaration is the whole contract: a descriptor inside a namespace is not one, because
    the C stage evaluates the bare name.
    """
    top, _nested = _cgen_definitions(text)
    return bool(top)


def cgen_findings(text: str) -> list[str]:
    """The deterministic problems of a C descriptor, in a generated module.

    An operator outside the elementwise set omits the declaration, which is not a
    finding.  A declaration that exists must be the top-level `def` the contract names
    and must import the generator: the C stage cannot find it otherwise, and a silent
    `unsupported` would hide the model's mistake.
    """
    code = _code_lines(text)
    top, nested = _cgen_definitions(text)
    if len(top) > 1:
        return ["the C descriptor %s is declared more than once" % CGEN_DESCRIPTOR]
    if top:
        imported = any(
            (match := IMPORT_RE.match(line)) and match.group(1).split(".")[0] == CGEN_MODULE
            for _number, line in code)
        if not imported:
            return ["the C descriptor requires `import %s`" % CGEN_MODULE]
        return []
    if nested or any(CGEN_NAME_RE.search(line) for _number, line in code):
        return ["`%s` must be a top-level `def` outside every namespace" % CGEN_DESCRIPTOR]
    return []


def materialize(raw: str, inv: dict) -> str:
    canonical = _canonical(inv)
    raw_lines = raw.splitlines()
    imports = [line for line in raw_lines if re.match(r"^\s*import\s", line)]
    body_lines = [line for line in raw_lines if not re.match(r"^\s*import\s", line)]
    output = imports + ["/-!", "@generated-by sonnx-lean-pipeline",
              "@source " + inv["source"], "@source-digest " + inv["source_md5"],
              "@library-digest " + library_digest(),
              "Compilation and trace validity are separate from semantic completeness; see the trace report.",
              "-/", ""]
    for line in body_lines:
        match = re.match(r"^\s*--\s*@trace\s+(\S+)(.*)$", line)
        if not match:
            output.append(line)
            continue
        attrs = _attrs(match.group(2))
        key = match.group(1)
        item = canonical[key]
        output.extend(["/-! ", "@trace " + key,
                       "@source %s:L%d-L%d" % (item["source"], item["start"], item["end"]),
                       "@coverage " + attrs.get("status", "")])
        if item.get("anchor"):
            output.append("@anchor " + item["anchor"])
        if attrs.get("decl"):
            output.append("@declaration " + attrs["decl"])
        if attrs.get("limitation"):
            output.append("@limitation " + attrs["limitation"])
        output.append("Informal source excerpt:")
        output.extend("  " + source_line for source_line in item["excerpt"].splitlines())
        output.extend(["-/", ""])
    return "\n".join(output).rstrip() + "\n"


def declaration_checks(entries: list[dict]) -> str:
    declarations = sorted({entry["declaration"] for entry in entries
                           if entry["status"] in {"specified", "proved", "example_only"}
                           and entry["declaration"]})
    return "\n-- Fully-qualified declarations named by the trace entries.\n" + "".join(
        "#check @%s\n" % declaration for declaration in declarations)


def find_lake(configured: str | None = None) -> str:
    candidates = [configured, shutil.which("lake"), str(Path.home() / ".elan" / "bin" / "lake")]
    for candidate in candidates:
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return str(candidate)
    raise FileNotFoundError("Lake was not found; pass --lake /path/to/lake")


def compile_lean(path: Path, lake: str | None = None, timeout: int = 300,
                 cwd: Path | None = None) -> dict:
    command = [find_lake(lake), "env", "lean", str(Path(path).resolve())]
    try:
        proc = subprocess.run(command, cwd=str(cwd or REPO / "lean"),
                              capture_output=True, text=True, timeout=timeout)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                "stdout": (proc.stdout + proc.stderr).strip(), "command": command}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or "" if isinstance(exc.stdout, str) else ""
        return {"ok": False, "returncode": -1,
                "stdout": output + "\nLean compilation timed out after %d s" % timeout,
                "command": command}


def build_library(lake: str | None = None, timeout: int = 600,
                  cwd: Path | None = None) -> dict:
    """Build the shared libraries a generated module may import.

    ``lake build Sonnx CGen`` is incremental: the first call builds both, later
    calls only re-check what changed.  The formalizer's module may import ``CGen``
    to declare the descriptor the C stage renders, so the generator must be built
    before the first candidate is compiled.  The oleans land on the ``LEAN_PATH``
    that ``lake env lean`` exports for the generated modules.
    """
    command = [find_lake(lake), "build", LIBRARY_MODULE, CGEN_MODULE]
    started = time.time()
    try:
        proc = subprocess.run(command, cwd=str(cwd or REPO / "lean"),
                              capture_output=True, text=True, timeout=timeout)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                "stdout": (proc.stdout + proc.stderr).strip(), "command": command,
                "seconds": round(time.time() - started, 2)}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or "" if isinstance(exc.stdout, str) else ""
        return {"ok": False, "returncode": -1,
                "stdout": output + "\nThe shared library build timed out after %d s"
                % timeout, "command": command,
                "seconds": round(time.time() - started, 2)}
    except OSError as exc:
        return {"ok": False, "returncode": -1, "stdout": str(exc),
                "command": command, "seconds": round(time.time() - started, 2)}


def run_lean(path: Path, payload: str, lake: str | None = None, timeout: int = 300,
             cwd: Path | None = None) -> dict:
    """Run an executable Lean module with ``lake env lean --run``.

    The module may live anywhere: the file is passed by absolute path and the working
    directory supplies the package's toolchain and library path.
    """
    command = [find_lake(lake), "env", "lean", "--run", str(Path(path).resolve())]
    started = time.time()
    try:
        proc = subprocess.run(command, cwd=str(cwd or REPO / "lean"), input=payload,
                              capture_output=True, text=True, timeout=timeout)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                "stdout": proc.stdout or "", "stderr": proc.stderr or "",
                "command": command, "seconds": round(time.time() - started, 2)}
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {"ok": False, "returncode": -1, "stdout": stdout or "",
                "stderr": (stderr or "") + "\nLean timed out after %d s" % timeout,
                "command": command, "seconds": round(time.time() - started, 2)}
    except OSError as exc:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(exc),
                "command": command, "seconds": round(time.time() - started, 2)}


# ---------------------------------------------------------------------------
# Generated conformance artifacts and their freshness manifest
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _generated_driver(path: Path) -> bool:
    try:
        head = Path(path).read_text(encoding="utf-8", errors="replace")[:2000]
    except OSError:
        return False
    return "sonnx-lean-pipeline-conformance" in head


def conformance_selection(settings, spec_path: Path | None = None) -> dict:
    """Where the conformance driver comes from, and whether it is still fresh.

    The order is: an explicit ``--lean-conformance-script``, the driver generated in this
    run's directory, the generated driver in the operator's folder (a different file only
    when ``--run-dir`` put the run elsewhere), and finally the hand-written driver of an
    operator that was verified before generation existed.  A generated driver whose
    manifest is stale is reported as such instead of being silently replaced by an older
    one.
    """
    configured = getattr(settings, "lean_conformance_script", None)
    if configured:
        return {"path": Path(configured), "kind": "configured", "manifest": {},
                "problem": ""}
    run_driver = settings.lean_conformance_driver_path()
    published_driver = settings.lean_publication_driver_path()
    candidates = [(run_driver, "generated-run")]
    if published_driver.resolve() != run_driver.resolve():
        candidates.append((published_driver, "generated-published"))
    stale = None
    spec = Path(spec_path) if spec_path else settings.resolved_spec_path()
    for path, kind in candidates:
        if not path.is_file():
            continue
        check = validate_conformance_manifest(path, spec)
        if check["legacy"]:
            # A published driver with no manifest is the hand-written fallback and is
            # only selected when no generated driver exists at all.
            continue
        if check["ok"]:
            return {"path": path, "kind": kind, "manifest": check["manifest"],
                    "problem": ""}
        stale = stale or {"path": path, "kind": "stale",
                          "manifest": check["manifest"],
                          "problem": "; ".join(check["findings"])}
    if stale:
        return stale
    legacy = published_driver
    if legacy.is_file() and not _generated_driver(legacy):
        return {"path": legacy, "kind": "existing", "manifest": {}, "problem": ""}
    return {"path": legacy, "kind": "unavailable", "manifest": {}, "problem": ""}


def conformance_runner(settings, spec_path: Path | None = None) -> Path:
    """Compatibility wrapper: the selected driver's path."""
    return conformance_selection(settings, spec_path)["path"]


def validate_conformance_manifest(driver: Path, spec_path: Path) -> dict:
    """Check that a generated driver belongs to the current spec and files.

    Returns ``{"ok", "legacy", "findings", "manifest"}``.  ``legacy`` is true when no
    manifest exists and the driver is not one the pipeline generated, which is the
    hand-written Mul/Abs case and is accepted without the hash check.
    """
    driver = Path(driver)
    manifest_path = driver.parent / "lean" / "conformance.manifest.json"
    if not manifest_path.is_file():
        if _generated_driver(driver):
            return {"ok": False, "legacy": False,
                    "findings": ["the generated driver has no conformance manifest"],
                    "manifest": {}}
        return {"ok": True, "legacy": True, "findings": [], "manifest": {}}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"ok": False, "legacy": False,
                "findings": ["the conformance manifest is unreadable: %s" % exc],
                "manifest": {}}
    findings = []
    if manifest.get("generated_by") != "sonnx-lean-pipeline-conformance":
        findings.append("the manifest was not written by the pipeline")
    if manifest.get("protocol") != "sonnx-conformance/1":
        findings.append("unsupported protocol %r" % manifest.get("protocol"))
    if manifest.get("spec_md5") != source_md5(Path(spec_path)):
        findings.append("the specification changed since the artifacts were generated")
    if manifest.get("library_sha256") != library_digest():
        findings.append("the shared Sonnx library changed since the artifacts were generated")
    base = driver.parent
    for name in ("runner", "driver", "corpus", "formal_spec"):
        relative = manifest.get(name)
        digest = manifest.get(name + "_sha256")
        if not relative:
            findings.append("the manifest names no %s" % name)
            continue
        target = base / relative
        if not target.is_file():
            findings.append("the %s %s is missing" % (name, relative))
        elif _sha256(target) != digest:
            findings.append("the %s %s changed after generation" % (name, relative))
    return {"ok": not findings, "legacy": False, "findings": findings,
            "manifest": manifest}


def write_conformance_manifest(settings, spec_path: Path, corpus_source: str,
                               profile: dict, runner: Path, driver: Path) -> Path:
    """Write the freshness manifest shared by the driver, runner and corpus."""
    root = settings.resolved_run_dir().resolve()

    def relative(path: Path) -> str:
        return Path(path).resolve().relative_to(root).as_posix()

    manifest = {
        "generated_by": "sonnx-lean-pipeline-conformance",
        "protocol": "sonnx-conformance/1",
        "op": settings.op,
        "opset": settings.opset,
        "entry": settings.entry,
        "spec_md5": source_md5(Path(spec_path)),
        "library_sha256": library_digest(),
        "formal_spec": relative(settings.lean_path()),
        "formal_spec_sha256": _sha256(settings.lean_path()),
        "runner": relative(runner),
        "runner_sha256": _sha256(runner),
        "driver": relative(driver),
        "driver_sha256": _sha256(driver),
        "corpus": relative(settings.lean_conformance_corpus_path()),
        "corpus_sha256": _sha256(settings.lean_conformance_corpus_path()),
        "corpus_source": corpus_source,
        "outputs": list(profile.get("outputs") or ["out"]),
        "output_types": dict(profile.get("output_types") or {}),
        "supported_kinds": sorted({
            "f64", "f32", "u8", "u16", "u32", "u64",
            "i8", "i16", "i32", "i64"}.intersection(
                {str(np_kind(settings, dtype)) for dtype in profile.get("dtypes", [])})),
        "compare": getattr(settings, "lean_conformance_compare", "bits"),
        "ulp": getattr(settings, "lean_conformance_ulp", 2),
        "relative": getattr(settings, "lean_conformance_relative", 1e-6),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = settings.lean_conformance_manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def np_kind(settings, dtype: str) -> str:
    """The protocol kind of a corpus dtype string, or the string itself."""
    kinds = {
        "float64": "f64", "float32": "f32",
        "uint8": "u8", "uint16": "u16", "uint32": "u32", "uint64": "u64",
        "int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64",
    }
    return kinds.get(str(dtype), str(dtype))


def run_conformance(settings, driver: Path | None = None,
                    spec_path: Path | None = None) -> dict:
    """Run the Lean-vs-corpus check, validating any generated artifacts first.

    This is deliberately separate from trace validity and compilation: a trace-complete,
    compiling specification can still refer to an implementation that disagrees with the
    corpus.  ``blocking`` is true only when a generated artifact is stale, the driver ran
    and reported a mismatch, or it could not produce a report at all.  An absent driver,
    or ``lean_conformance=False``, leaves the existing pipeline behaviour unchanged.
    """
    if not getattr(settings, "lean_conformance", True):
        return {"status": "disabled", "ok": True, "blocking": False,
                "report": {}, "error": "", "report_path": "", "runner": "",
                "kind": "disabled", "generated": False, "manifest": {}}

    selection = conformance_selection(settings, spec_path)
    if driver is not None:
        selection = {"path": Path(driver), "kind": "explicit",
                     "manifest": {}, "problem": ""}
    runner = selection["path"]
    if selection["kind"] == "stale":
        message = ("the generated conformance artifacts are stale: %s"
                   % (selection.get("problem") or "no detail"))
        return {"status": "error", "ok": False, "blocking": True,
                "report": {}, "error": message, "report_path": "",
                "runner": str(runner), "kind": "stale", "generated": True,
                "manifest": selection.get("manifest") or {}}
    if selection["kind"] == "unavailable" or not runner.is_file():
        return {"status": "unavailable", "ok": True, "blocking": False,
                "report": {}, "error": "", "report_path": "", "runner": str(runner),
                "kind": "unavailable", "generated": False, "manifest": {}}

    spec = Path(spec_path) if spec_path else settings.resolved_spec_path()
    check = validate_conformance_manifest(runner, spec)
    if not check["ok"]:
        message = "the generated conformance artifacts are stale: %s" % (
            "; ".join(check["findings"]))
        return {"status": "error", "ok": False, "blocking": True,
                "report": {}, "error": message, "report_path": "",
                "runner": str(runner), "kind": "stale", "generated": True,
                "manifest": check.get("manifest") or {}}

    report_path = settings.lean_conformance_report_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.unlink(missing_ok=True)
    timeout = int(getattr(settings, "lean_conformance_timeout", 1800))
    command = [str(settings.python), str(runner), "--json", str(report_path),
               "--timeout", str(timeout)]
    started = time.time()
    try:
        env = dict(os.environ)
        env.setdefault("SONNX_REPO", str(REPO))
        proc = subprocess.run(command, cwd=str(REPO), env=env, capture_output=True,
                              text=True, timeout=timeout + 120)
        returncode, stdout, stderr, error = (proc.returncode, proc.stdout or "",
                                             proc.stderr or "", "")
    except subprocess.TimeoutExpired as exc:
        returncode, stdout, stderr = (
            -1, (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            (exc.stderr or "") if isinstance(exc.stderr, str) else "")
        error = "the Lean conformance driver timed out after %d s" % (timeout + 120)
    except OSError as exc:
        returncode, stdout, stderr = -1, "", ""
        error = "the Lean conformance driver could not be run: %s" % exc

    report = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            error = error or "the Lean conformance report is unreadable: %s" % exc

    generated = bool(selection["kind"].startswith("generated") or
                     check.get("manifest") or
                     report.get("generated_by") == "sonnx-lean-pipeline-conformance")
    if report and generated:
        findings = []
        if report.get("generated_by") != "sonnx-lean-pipeline-conformance":
            findings.append("the report was not written by the generated driver")
        if report.get("protocol") != "sonnx-conformance/1":
            findings.append("the report protocol is %r" % report.get("protocol"))
        if report.get("spec_md5") != source_md5(spec):
            findings.append("the report belongs to another specification")
        summary = report.get("summary") or {}
        if int(summary.get("total", 0)) <= 0:
            findings.append("the report compared no cases")
        if int(summary.get("compared", 0)) <= 0:
            findings.append("no case was actually compared")
        if findings:
            error = error or "the generated conformance report is not usable: %s" % (
                "; ".join(findings))
    if error:
        status = "error"
    elif report.get("ok") is True and int((report.get("summary") or {}).get("total", 0)) > 0:
        status = "passed"
    elif report:
        stage = report.get("stage")
        if stage in {"build", "run"}:
            status = "error"
            error = report.get("error") or "the conformance run could not complete"
        else:
            status = "failed"
            error = "the Lean conformance check did not pass"
    else:
        status = "error"
        error = "the Lean conformance driver produced no report"
    if stderr and status != "passed":
        error = (error + "; " if error else "") + stderr.strip()[-1000:]
    return {"status": status, "ok": status in {"passed", "unavailable", "disabled"},
            "blocking": status in {"failed", "error"}, "runner": str(runner),
            "report_path": str(report_path), "returncode": returncode,
            "seconds": round(time.time() - started, 2),
            "stdout": stdout[-4000:], "stderr": stderr[-4000:],
            "report": report, "error": error, "kind": selection["kind"],
            "generated": generated,
            "manifest": check.get("manifest") or selection.get("manifest") or {}}


def validate_candidate(raw: str, inv: dict) -> dict:
    entries, errors = parse_markers(raw)
    trace = validate_entries(entries, errors, inv)
    findings = (trace["findings"] + forbidden_constructs(raw)
                + import_findings(raw) + shared_redefinition_findings(raw)
                + cgen_findings(raw))
    return {**trace, "ok": not findings, "findings": findings,
            "library_sha256": library_digest()}


def prepare_candidate(raw: str, inv: dict, path: Path) -> dict:
    validation = validate_candidate(raw, inv)
    if not validation["ok"]:
        return validation
    rendered = materialize(raw, inv)
    entries, errors = parse_materialized(rendered)
    checked = validate_entries(entries, errors, inv, materialized=True)
    if not checked["ok"]:
        return checked
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered.rstrip() + "\n" + declaration_checks(entries), encoding="utf-8")
    checked["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    checked["library_sha256"] = validation["library_sha256"]
    return checked


def check_files(spec_path: Path, paths: list[Path], lake: str | None = None,
                timeout: int = 300, compile_code: bool = True) -> dict:
    inv = inventory(spec_path)
    text = "\n".join(Path(path).read_text(encoding="utf-8") for path in paths)
    entries, errors = parse_materialized(text)
    report = validate_entries(entries, errors, inv, materialized=True)
    findings = list(report["findings"])
    digests = re.findall(r"@source-digest\s+([0-9a-fA-F]{32})", text)
    if not digests:
        findings.append("missing @source-digest")
    elif any(digest.lower() != inv["source_md5"] for digest in digests):
        findings.append("source digest is stale: expected %s" % inv["source_md5"])
    findings.extend(forbidden_constructs(text))
    findings.extend(import_findings(text))
    findings.extend(shared_redefinition_findings(text))
    findings.extend(cgen_findings(text))
    library = library_digest()
    library_digests = re.findall(r"@library-digest\s+([0-9a-fA-F]{64})", text)
    if not library_digests:
        findings.append("missing @library-digest")
    elif any(digest.lower() != library for digest in library_digests):
        findings.append("the shared Sonnx library changed: expected %s" % library)
    report["findings"] = findings
    report["ok"] = not findings
    report["source_md5"] = inv["source_md5"]
    report["library_sha256"] = library
    report["files"] = [str(path) for path in paths]
    if compile_code and report["ok"]:
        compilers = [compile_lean(path, lake, timeout) for path in paths]
        report["compilers"] = compilers
        if any(not compiler["ok"] for compiler in compilers):
            report["ok"] = False
    report["complete"] = bool(report["complete"] and report["ok"])
    return report


def write_report(path: Path, report: dict, inv: dict) -> None:
    payload = {"generated_by": "sonnx-lean-pipeline", **inv, **report}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("lean", type=Path, nargs="*")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--lake")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.inventory:
            report, code = inventory(args.spec), 0
        else:
            if not args.lean:
                parser.error("give at least one Lean file, or --inventory")
            report = check_files(args.spec, args.lean, args.lake, args.timeout,
                                 compile_code=not args.no_compile)
            code = 0 if report["ok"] else 1
    except (OSError, ValueError) as exc:
        report, code = {"ok": False, "findings": [str(exc)]}, 1
    rendered = json.dumps(report, indent=2) + "\n"
    if args.json:
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
