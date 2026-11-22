"""The wire protocol, corpus runner and report for generated Lean conformance.

The pipeline writes a thin ``lean_conformance.py`` driver that imports this module.  All
of the operator-independent work lives here: encoding a corpus case for the Lean runner,
obtaining the reference values from ONNX Runtime, comparing the two, and writing the
report.  The generated driver is therefore a short, reviewable configuration file rather
than a second implementation of this module.

The protocol is deliberately line-oriented and tab-separated so that a Lean runner can
parse it without a JSON library:

    request   <id>\t<attrs>\t<n>\t<kind>|<shape>|<data>\t...
    response  <id>\tOK\t<n>\t<kind>|<shape>|<data>\t...
              <id>\tUNSUPPORTED\t0\t<reason>
              <id>\tERR\t0\t<reason>
              <id>\tMALFORMED\t0\t<message>

``kind`` is one of ``f64``, ``f32``, ``u8``..``u64``, ``i8``..``i64``, or ``rat``.
Floats and integers are sent as their raw bit patterns, row-major; rationals are
``numerator/denominator``.  ``attrs`` is ``-`` or ``name=value`` entries separated by
``;``.  A runner that cannot represent a type answers ``UNSUPPORTED``; it never drops a
line.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

PROTOCOL = "sonnx-conformance/1"
GENERATED_BY = "sonnx-lean-pipeline-conformance"
RUNNER_MARKER = "sonnx-conformance/1"

FLOAT_KINDS = {np.dtype(np.float64): "f64", np.dtype(np.float32): "f32"}
INT_KINDS = {
    np.dtype(np.uint8): "u8", np.dtype(np.uint16): "u16",
    np.dtype(np.uint32): "u32", np.dtype(np.uint64): "u64",
    np.dtype(np.int8): "i8", np.dtype(np.int16): "i16",
    np.dtype(np.int32): "i32", np.dtype(np.int64): "i64",
}
KIND_OF_DTYPE = {**FLOAT_KINDS, **INT_KINDS}
DTYPE_OF_KIND = {kind: dtype for dtype, kind in KIND_OF_DTYPE.items()}
UNSUPPORTED_DTYPES = {
    np.dtype(np.float16): "Lean core has no Float16 type",
    np.dtype(np.bool_): "the corpus has no Lean Boolean tensor",
}


class ConformanceError(RuntimeError):
    """The conformance run could not produce trustworthy evidence."""


class ProtocolError(ConformanceError):
    """A line from the runner is not a protocol response."""


class UnsupportedKind(ConformanceError):
    """The corpus asks for a type the Lean runner cannot represent."""


@dataclass
class Output:
    kind: str
    shape: tuple[int, ...]
    values: tuple[str, ...]


@dataclass
class Response:
    id: str
    status: str
    outputs: list[Output]
    message: str


# ---------------------------------------------------------------------------
# Paths and helpers shared by the pipeline and the generated driver
# ---------------------------------------------------------------------------


def find_repo(start: Path | str) -> Path:
    """The repository root above *start*, identified by both agents/ and lean/."""
    here = Path(start).resolve()
    candidates = [here] + list(here.parents)
    for root in candidates:
        if ((root / "agents" / "sonnx_agents" / "lean_conformance.py").is_file()
                and (root / "lean" / "lakefile.lean").is_file()):
            return root
    raise ConformanceError("the repository root was not found above %s" % here)


def find_lake(configured: str | None = None) -> str:
    candidates = [configured, shutil.which("lake"),
                  str(Path.home() / ".elan" / "bin" / "lake")]
    for candidate in candidates:
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return str(candidate)
    raise ConformanceError("Lake was not found; pass --lake /path/to/lake")


def load_specloop(repo: Path):
    path = repo / ".claude" / "skills" / "operator-spec" / "assets" / "specloop.py"
    if not path.is_file():
        raise ConformanceError("the harness was not found at %s" % path)
    spec = importlib.util.spec_from_file_location("sonnx_conformance_specloop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def dtype_kind(dtype) -> str | None:
    return KIND_OF_DTYPE.get(np.dtype(dtype))


def kind_dtype(kind: str) -> np.dtype | None:
    return DTYPE_OF_KIND.get(kind)


def shape_text(shape) -> str:
    shape = tuple(int(dim) for dim in shape)
    return ",".join(str(dim) for dim in shape) if shape else "-"


def data_text(values) -> str:
    values = list(values)
    return ",".join(str(value) for value in values) if values else "-"


def bits_of(array: np.ndarray) -> tuple[int, ...]:
    arr = np.ascontiguousarray(array)
    uint = np.dtype("uint%d" % (8 * arr.dtype.itemsize))
    return tuple(int(value) for value in arr.reshape(-1).view(uint))


def tensor_text(array) -> str:
    array = np.asarray(array)
    dtype = np.dtype(array.dtype)
    if dtype.kind == "O":
        values = []
        for value in array.reshape(-1):
            if not isinstance(value, Fraction):
                raise UnsupportedKind("object tensor element is not a Fraction: %r" % (value,))
            values.append("%d/%d" % (value.numerator, value.denominator))
        return "rat|%s|%s" % (shape_text(array.shape), data_text(values))
    kind = dtype_kind(dtype)
    if kind is None:
        reason = UNSUPPORTED_DTYPES.get(dtype, "unsupported element type %s" % dtype.name)
        raise UnsupportedKind(reason)
    return "%s|%s|%s" % (kind, shape_text(array.shape), data_text(bits_of(array)))


def attr_text(attrs) -> str:
    if not attrs:
        return "-"
    parts = []
    for key in sorted(attrs):
        value = attrs[key]
        if isinstance(value, (list, tuple)):
            text = ",".join(str(item) for item in value)
        elif isinstance(value, bytes):
            text = value.decode("utf-8", "replace")
        else:
            text = str(value)
        # The protocol has three separators; an attribute value may not contain them.
        text = text.replace(";", " ").replace("=", " ").replace("\t", " ").replace("\n", " ")
        parts.append("%s=%s" % (key, text))
    return ";".join(parts)


def encode_request(case_id: int, arrays, attrs) -> str:
    fields = [str(int(case_id)), attr_text(attrs), str(len(arrays))]
    fields.extend(tensor_text(array) for array in arrays)
    return "\t".join(fields)


def decode_tensor(text: str) -> Output:
    parts = text.split("|")
    if len(parts) != 3:
        raise ProtocolError("malformed tensor field %r" % text)
    kind, shape, data = parts
    if kind not in DTYPE_OF_KIND and kind != "rat":
        raise ProtocolError("unknown tensor kind %r" % kind)
    try:
        dims = () if shape == "-" else tuple(int(dim) for dim in shape.split(","))
    except ValueError as exc:
        raise ProtocolError("malformed shape %r: %s" % (shape, exc)) from exc
    values = () if data == "-" else tuple(data.split(","))
    expected = 0 if any(dim == 0 for dim in dims) else 1
    for dim in dims:
        expected *= dim
    if len(values) != expected:
        raise ProtocolError("shape %r wants %d value(s), got %d"
                            % (dims, expected, len(values)))
    return Output(kind, dims, values)


def decode_response(line: str) -> Response:
    fields = line.split("\t")
    if len(fields) < 3:
        raise ProtocolError("malformed response %r" % line)
    case_id, status, count = fields[0], fields[1], fields[2]
    if status not in {"OK", "UNSUPPORTED", "ERR", "MALFORMED"}:
        raise ProtocolError("unknown response status %r" % status)
    if status == "OK":
        try:
            wanted = int(count)
        except ValueError as exc:
            raise ProtocolError("malformed output count %r" % count) from exc
        if wanted < 1 or len(fields) != 3 + wanted:
            raise ProtocolError("OK response wants %r output(s), got %d field(s)"
                                % (count, len(fields) - 3))
        return Response(case_id, status, [decode_tensor(field) for field in fields[3:]],
                        "")
    message = fields[3] if len(fields) > 3 else ""
    return Response(case_id, status, [], message)


def response_lines(stdout: str) -> list[str]:
    """The runner's own responses, without the compiler's messages around them.

    ``lake env lean --run`` elaborates the module before executing ``main``, and the
    compiler prints its diagnostics — a deprecation warning, say — to standard output.
    Those lines arrive before any response and would otherwise be read as protocol
    lines.  Every response is tab-separated (``<id>\\t<status>\\t<count>\\t...``); a
    diagnostic is one plain text line and carries no tab, so the tab is the
    discriminator.  A runner whose responses are not tab-separated then fails the count
    check rather than being silently accepted.
    """
    return [line for line in stdout.splitlines() if "\t" in line]


# ---------------------------------------------------------------------------
# Running Lean
# ---------------------------------------------------------------------------


def repo_of(start: Path | str) -> Path:
    """The repository root above *start*, or above this module when *start* is outside it.

    A run directory may be anywhere (``--run-dir /tmp/...``); this module always lives in
    the repository, so its own location is the reliable fallback.
    """
    try:
        return find_repo(start)
    except ConformanceError:
        return find_repo(Path(__file__))


def run_lean(runner: Path, payload: str, timeout: int, lake: str | None = None,
             cwd: Path | None = None) -> dict:
    """Run an executable Lean module through ``lake env lean --run``.

    The file may live outside the package directory: ``lake env`` supplies the toolchain
    and the package's library path, and the file is passed by absolute path.
    """
    repo = repo_of(runner)
    command = [find_lake(lake), "env", "lean", "--run", str(Path(runner).resolve())]
    started = time.time()
    try:
        proc = subprocess.run(command, cwd=str(cwd or repo / "lean"), input=payload,
                              text=True, capture_output=True, timeout=timeout)
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
# The corpus
# ---------------------------------------------------------------------------


def load_corpus(path: Path):
    spec = importlib.util.spec_from_file_location("sonnx_conformance_corpus", path)
    if spec is None or spec.loader is None:
        raise ConformanceError("the corpus %s cannot be imported" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "cases", None)):
        raise ConformanceError("the corpus %s defines no cases()" % path)
    return module


def iter_cases(module):
    for item in module.cases():
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ConformanceError("a corpus case is not (label, arrays[, attrs]): %r"
                                   % (item,))
        label, arrays = item[0], item[1]
        attrs = dict(item[2]) if len(item) > 2 and item[2] else {}
        yield str(label), [np.asarray(array) for array in arrays], attrs


def div_trap_pair(arrays, op: str) -> bool:
    """True for the signed 32/64-bit `(INT_MIN, -1)` pair ONNX Runtime cannot run.

    C integer division overflows for that pair, and ONNX Runtime's signed `Div` kernel
    lets the process die with `SIGFPE`; no Python handler can catch it.  A corpus may
    still contain the pair (the document states its wrapped result), so the driver must
    not hand it to the runtime: the case is recorded `open`, and a sweep call raises a
    Python error, which `check_sweeps` reports.
    """
    if op != "Div" or len(arrays) < 2:
        return False
    a, b = np.asarray(arrays[0]), np.asarray(arrays[1])
    if a.dtype != b.dtype or a.dtype.kind != "i" or a.dtype.itemsize not in (4, 8):
        return False
    try:
        aa, bb = np.broadcast_arrays(a, b)
    except ValueError:
        return False
    return bool(np.any((aa == np.iinfo(a.dtype).min) & (bb == -1)))


class _Recorder:
    """The implementation a corpus's sweeps are replayed against.

    A sweep is not a list of cases: it is a function that builds its own ONNX Runtime
    calls and compares them with the implementation it is given.  To learn the inputs
    the Python verification exercises, the sweep is run against this object instead:
    every call it makes is recorded, and the ONNX Runtime reference is what the call
    returns, so the sweep's own comparisons proceed and do not stop it before its
    later inputs.
    """

    def __init__(self, reference, outputs, records, name, op=""):
        self.reference = reference
        self.outputs = tuple(outputs)
        self.records = records
        self.name = name
        self.op = op

    def __getattr__(self, attribute):
        if attribute.startswith("_"):
            raise AttributeError(attribute)

        def call(*arrays, **attrs):
            if div_trap_pair(arrays, self.op):
                raise ValueError(
                    "ONNX Runtime cannot evaluate INT_MIN / -1 for a signed "
                    "32/64-bit Div; the input is not replayed")
            self.records.append(("%s: %s" % (self.name["value"], attribute),
                                 [np.asarray(array) for array in arrays],
                                 dict(attrs)))
            values = list(self.reference.run(list(arrays), attrs or None))
            return values[0] if len(self.outputs) == 1 else tuple(values)

        return call


def sweep_cases(module, reference, outputs, specloop, op=""):
    """The inputs the corpus's own sweeps pass to the implementation.

    The Lean conformance must exercise the same inputs the Python verification did,
    and for these operators the individual cases are only part of that corpus: the
    sweeps are where the exhaustive and boundary values live.  Returns the recorded
    ``(label, arrays, attrs)`` inputs and the sweeps' own errors, which are reported
    beside the conformance result and do not decide it.
    """
    if not callable(getattr(module, "sweeps", None)):
        return [], []
    records: list = []
    current = {"value": "sweep"}

    def named(sweep):
        name = getattr(sweep, "__name__", "sweep")

        def replay(implementation):
            previous = current["value"]
            current["value"] = name
            try:
                return sweep(implementation)
            finally:
                current["value"] = previous

        replay.__name__ = name
        return replay

    try:
        sweeps = [named(sweep) for sweep in module.sweeps()]
    except Exception as exc:  # the corpus's own sweep list could not be built
        return [], ["the corpus's sweeps() raised: %s" % str(exc)[-500:]]
    recorder = _Recorder(reference, outputs, records, current, op=op)
    try:
        reports = specloop.check_sweeps(recorder, sweeps)
    except Exception as exc:  # check_sweeps already isolates each sweep
        return records, ["the sweeps could not be replayed: %s" % str(exc)[-500:]]
    errors = ["%s: %s" % (item.get("sweep"), str(item.get("error"))[-500:])
              for item in reports if item.get("error")]
    return records, errors


READER = r'''
import importlib.util
import itertools
import json
import sys

import numpy as np

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("sonnx_conformance_corpus", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
samples = []
count = 0
for item in itertools.islice(module.cases(), 5000):
    label, arrays = item[0], item[1]
    attrs = dict(item[2]) if len(item) > 2 and item[2] else {}
    samples.append({
        "label": str(label),
        "arity": len(arrays),
        "shapes": [list(np.asarray(array).shape) for array in arrays],
        "dtypes": [str(np.asarray(array).dtype) for array in arrays],
        "attrs": {str(k): v for k, v in attrs.items()},
    })
    count += 1
print(json.dumps({
    "case_count_sampled": count,
    "arities": sorted({sample["arity"] for sample in samples}),
    "dtypes": sorted({dtype for sample in samples for dtype in sample["dtypes"]}),
    "attr_keys": sorted({key for sample in samples for key in sample["attrs"]}),
    "outputs": list(getattr(module, "NODE_OUTPUTS", ("out",))),
    "output_types": {str(k): str(v) for k, v in
                     (getattr(module, "OUTPUT_TYPES", {}) or {}).items()},
    "has_sweeps": callable(getattr(module, "sweeps", None)),
    "op": getattr(module, "OP", None),
    "opset": getattr(module, "OPSET", None),
    "samples": samples[:20],
}, default=str))
'''


def profile(path: Path, python: str | None = None) -> dict:
    """Describe a corpus without executing its cases in the pipeline process."""
    command = [python or sys.executable, "-I", "-c", READER, str(Path(path).resolve())]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise ConformanceError("the corpus could not be profiled: %s"
                               % (proc.stderr.strip()[-2000:] or "no detail"))
    try:
        result = json.loads(proc.stdout)
    except ValueError as exc:
        raise ConformanceError("the corpus profile is not JSON: %s" % exc) from exc
    if not result.get("case_count_sampled"):
        raise ConformanceError("the corpus yields no cases")
    if not result.get("arities"):
        raise ConformanceError("the corpus yields no operand arrays")
    return result


def profile_problem(result: dict) -> str:
    if not result.get("case_count_sampled"):
        return "the corpus yields no cases"
    if not result.get("arities"):
        return "the corpus yields no operand arrays"
    supported = [dtype for dtype in result.get("dtypes", [])
                 if np.dtype(dtype) in KIND_OF_DTYPE or np.dtype(dtype).kind == "O"]
    if not supported:
        return "the corpus yields no element type the Lean runner can represent"
    return ""


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _actual_array(output: Output, dtype: np.dtype) -> np.ndarray:
    if output.kind == "rat" or dtype.kind == "O":
        if output.kind != "rat":
            raise ProtocolError("expected a rational output, got %s" % output.kind)
        values = [Fraction(value) for value in output.values]
        return np.array(values, dtype=object).reshape(output.shape)
    wanted = dtype_kind(dtype)
    if wanted != output.kind:
        raise ProtocolError("expected kind %s, got %s" % (wanted, output.kind))
    if dtype.itemsize == 1 and dtype.kind == "f":
        raise ProtocolError("no Lean Float8")
    uint = np.dtype("uint%d" % (8 * dtype.itemsize))
    try:
        bits = [int(value) for value in output.values]
    except ValueError as exc:
        raise ProtocolError("non-integer bit pattern in %r" % (output.values,)) from exc
    return np.array(bits, dtype=uint).view(dtype).reshape(output.shape)


def compare_output(expected, output: Output, config: dict) -> tuple[bool, str]:
    expected = np.asarray(expected)
    dtype = np.dtype(expected.dtype)
    try:
        actual = _actual_array(output, dtype)
    except (ProtocolError, ValueError) as exc:
        return False, str(exc)
    if tuple(expected.shape) != tuple(actual.shape):
        return False, "shape: want %s, got %s" % (tuple(expected.shape), tuple(actual.shape))
    mode = config.get("compare", "bits")
    if mode == "ulp" and dtype.kind == "f":
        return _compare_ulp(expected, actual, int(config.get("ulp", 2)))
    if mode == "relative" and dtype.kind == "f":
        equal = np.isclose(expected, actual, rtol=float(config.get("relative", 1e-6)),
                           atol=0.0, equal_nan=True)
        if bool(np.all(equal)):
            return True, ""
        return False, "values differ beyond relative tolerance"
    if dtype.kind == "f":
        want = np.ascontiguousarray(expected).reshape(-1).view(
            np.dtype("uint%d" % (8 * dtype.itemsize)))
        got = np.ascontiguousarray(actual).reshape(-1).view(
            np.dtype("uint%d" % (8 * dtype.itemsize)))
        both_nan = np.isnan(expected).reshape(-1) & np.isnan(actual).reshape(-1)
        equal = (want == got) | both_nan
    elif dtype.kind == "O":
        equal = np.array([bool(Fraction(a) == Fraction(b))
                          for a, b in zip(expected.reshape(-1), actual.reshape(-1))],
                         dtype=bool)
    else:
        equal = expected == actual
    if bool(np.all(equal)):
        return True, ""
    return False, "values differ (want %s, got %s)" % (
        expected.reshape(-1)[:4].tolist(), actual.reshape(-1)[:4].tolist())


def _compare_ulp(expected: np.ndarray, actual: np.ndarray, ulp: int) -> tuple[bool, str]:
    """Allow at most *ulp* representable floats between expected and actual.

    NaN is equal to NaN; infinities must match exactly.  The comparison is on the
    monotone integer image of the IEEE bit pattern, so it is correct across zero and
    both signs.
    """
    uint = np.dtype("uint%d" % (8 * expected.dtype.itemsize))
    sinfo = np.dtype("int%d" % (8 * expected.dtype.itemsize))

    def ordered(array):
        bits = np.ascontiguousarray(array).reshape(-1).view(uint)
        sign = (bits >> (8 * expected.dtype.itemsize - 1)) & 1
        magnitude = bits & ((1 << (8 * expected.dtype.itemsize - 1)) - 1)
        positive = magnitude.astype(sinfo)
        negative = (-positive).astype(sinfo)
        return np.where(sign == 1, negative, positive).astype(np.int64)

    expected_flat = expected.reshape(-1)
    actual_flat = actual.reshape(-1)
    nan_both = np.isnan(expected_flat) & np.isnan(actual_flat)
    inf_expected = np.isinf(expected_flat)
    inf_both = inf_expected & np.isinf(actual_flat) & (expected_flat == actual_flat)
    if not np.all(inf_expected == np.isinf(actual_flat)):
        return False, "one value is infinite and the other is not"
    distance = np.abs(ordered(expected_flat) - ordered(actual_flat))
    equal = nan_both | inf_both | (distance <= max(0, ulp))
    if bool(np.all(equal)):
        return True, ""
    return False, "values differ by more than %d ULP" % ulp


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config: dict, json_path: Path, timeout: int = 1800, max_failures: int = 50,
        driver_dir: Path | None = None, runner_override: Path | None = None) -> dict:
    here = Path(driver_dir or Path.cwd()).resolve()
    cases_path = here / config.get("cases", "lean/cases.py")
    runner_path = (Path(runner_override).resolve() if runner_override
                   else (here / config.get("runner", "lean/LeanConformance.lean")))
    repo = repo_of(here)
    specloop = load_specloop(repo)
    corpus = load_corpus(cases_path)
    reference = specloop.Runner(
        config.get("op", ""), int(config.get("opset", 14)),
        outputs=tuple(config.get("outputs") or ("out",)),
        out_types=dict(config.get("output_types") or {}))

    report = {
        "ok": False,
        "stage": "compare",
        "generated_by": GENERATED_BY,
        "protocol": PROTOCOL,
        "op": config.get("op", ""),
        "opset": int(config.get("opset", 14)),
        "spec_md5": config.get("spec_md5", ""),
        "formal_sha256": config.get("formal_sha256", ""),
        "reference": {
            "kind": "ONNX Runtime, one node per case",
            "float_comparison": config.get("compare", "bits"),
            "integer_comparison": "raw bit patterns",
            "rational_comparison": "exact fractions",
        },
        "corpus": {"path": str(cases_path), "sha256": _sha256(cases_path)},
        "runner": {"path": str(runner_path), "sha256": _sha256(runner_path)
                   if runner_path.is_file() else ""},
        "summary": {"total": 0, "compared": 0, "passed": 0, "failed": 0,
                    "unsupported": 0, "open": 0},
        "by_category": {},
        "failures": [],
        "unsupported": [],
        "open": [],
        "lean": {},
        "error": "",
    }

    outputs = tuple(config.get("outputs") or ("out",))
    sweep_inputs, sweep_errors = sweep_cases(corpus, reference, outputs, specloop,
                                             op=config.get("op") or "")
    cases = [(label, arrays, attrs, "cases")
             for label, arrays, attrs in iter_cases(corpus)]
    cases.extend((label, arrays, attrs, "sweeps")
                 for label, arrays, attrs in sweep_inputs)
    report["sweeps"] = {
        "declared": callable(getattr(corpus, "sweeps", None)),
        "inputs": len(sweep_inputs),
        "errors": sweep_errors,
    }

    requests: list[tuple[int, str]] = []
    expected: dict[int, tuple] = {}
    labels: dict[int, str] = {}
    categories: dict[int, str] = {}
    total = 0
    for case_id, (label, arrays, attrs, category) in enumerate(cases, 1):
        total += 1
        labels[case_id] = label
        categories[case_id] = category
        unsupported_dtype = next((np.dtype(array.dtype) for array in arrays
                                  if dtype_kind(array.dtype) is None), None)
        if unsupported_dtype is not None:
            reason = UNSUPPORTED_DTYPES.get(
                unsupported_dtype,
                "the Lean runner has no type for %s" % unsupported_dtype.name)
            report["unsupported"].append({"id": case_id, "label": label,
                                          "category": category,
                                          "kind": unsupported_dtype.name,
                                          "reason": reason})
            continue
        if div_trap_pair(arrays, config.get("op") or ""):
            report["open"].append({
                "id": case_id, "label": label, "category": category,
                "reason": "ONNX Runtime cannot evaluate INT_MIN / -1 for a signed "
                          "32/64-bit Div (the C integer division traps)"})
            continue
        try:
            wanted = reference.run(arrays, attrs)
        except Exception as exc:  # the runtime refusing an outside-domain case
            report["open"].append({"id": case_id, "label": label,
                                   "category": category,
                                   "reason": str(exc)[-500:]})
            continue
        try:
            line = encode_request(case_id, arrays, attrs)
        except UnsupportedKind as exc:
            report["unsupported"].append({"id": case_id, "label": label,
                                          "category": category,
                                          "kind": "unsupported", "reason": str(exc)})
            continue
        requests.append((case_id, line))
        expected[case_id] = tuple(wanted)

    report["summary"]["total"] = total
    report["summary"]["unsupported"] = len(report["unsupported"])
    report["summary"]["open"] = len(report["open"])
    if not requests:
        report["stage"] = "compare"
        report["error"] = ("nothing was compared: every case was unsupported or refused "
                           "by the reference runtime")
        _write_json(json_path, report)
        return report

    payload = "".join(line + "\n" for _, line in requests)
    lean = run_lean(runner_path, payload, timeout, config.get("lake"))
    report["lean"] = lean
    if not lean["ok"]:
        report["stage"] = "run"
        report["error"] = "the Lean runner exited with %s" % lean["returncode"]
        _write_json(json_path, report)
        return report

    lines = response_lines(lean["stdout"])
    if len(lines) != len(requests):
        report["stage"] = "run"
        report["error"] = ("expected %d result line(s), got %d"
                           % (len(requests), len(lines)))
        _write_json(json_path, report)
        return report

    compared = 0
    for (case_id, _), line in zip(requests, lines):
        label = labels.get(case_id, "")
        category = categories.get(case_id, "cases")
        stats = report["by_category"].setdefault(
            category, {"total": 0, "passed": 0, "failed": 0})
        stats["total"] += 1
        try:
            response = decode_response(line)
        except ProtocolError as exc:
            stats["failed"] += 1
            report["failures"].append({"id": case_id, "label": label,
                                       "category": category, "detail": str(exc)})
            continue
        if response.id != str(case_id):
            stats["failed"] += 1
            report["failures"].append({
                "id": case_id, "label": label, "category": category,
                "detail": "result id %s, expected %d" % (response.id, case_id)})
            continue
        if response.status == "UNSUPPORTED":
            report["unsupported"].append({"id": case_id, "label": label,
                                          "kind": "runner", "reason": response.message})
            continue
        if response.status == "ERR":
            stats["failed"] += 1
            report["failures"].append({"id": case_id, "label": label,
                                       "category": category,
                                       "detail": "Lean refused the case: %s"
                                                 % response.message})
            continue
        if response.status in {"MALFORMED"}:
            report["stage"] = "run"
            report["error"] = "the runner could not parse case %d: %s" % (
                case_id, response.message)
            _write_json(json_path, report)
            return report
        wanted = expected[case_id]
        if len(response.outputs) != len(wanted):
            stats["failed"] += 1
            report["failures"].append({
                "id": case_id, "label": label, "category": category,
                "detail": "expected %d output(s), got %d"
                          % (len(wanted), len(response.outputs))})
            continue
        compared += 1
        problems = []
        for index, (want, got) in enumerate(zip(wanted, response.outputs), 1):
            equal, detail = compare_output(want, got, config)
            if not equal:
                problems.append("output %d: %s" % (index, detail))
        if problems:
            stats["failed"] += 1
            if len(report["failures"]) < max_failures:
                report["failures"].append({"id": case_id, "label": label,
                                           "category": category,
                                           "detail": "; ".join(problems)})
        else:
            stats["passed"] += 1

    report["summary"]["compared"] = compared
    report["summary"]["passed"] = sum(s["passed"] for s in report["by_category"].values())
    report["summary"]["failed"] = sum(s["failed"] for s in report["by_category"].values())
    report["summary"]["unsupported"] = len(report["unsupported"])
    report["ok"] = (report["summary"]["failed"] == 0 and compared > 0)
    report["stage"] = "compare"
    if compared == 0:
        report["error"] = "nothing was compared"
    _write_json(json_path, report)
    return report


# ---------------------------------------------------------------------------
# Smoke probes
# ---------------------------------------------------------------------------


def probe_requests(profile_data: dict) -> list[tuple[str, str]]:
    """A few requests that exercise the protocol rather than the operator.

    The first request per arity is a valid-looking tensor; an unsupported kind and a
    malformed line follow, and then a valid request again so that a runner that dies on
    bad input is caught.
    """
    dtypes = [dtype for dtype in profile_data.get("dtypes", [])
              if np.dtype(dtype) in KIND_OF_DTYPE]
    arities = profile_data.get("arities") or [1]
    dtype = np.dtype(dtypes[0]) if dtypes else np.dtype(np.float64)
    kind = dtype_kind(dtype)
    shape = "2"
    data = ",".join(["0"] * 2)

    def request(case_id: int, kind_text: str, arity: int) -> str:
        fields = [str(case_id), "-", str(arity)]
        fields.extend("%s|%s|%s" % (kind_text, shape, data) for _ in range(arity))
        return "\t".join(fields)

    probes = []
    for arity in arities[:2]:
        probes.append(("valid arity %d" % arity, request(1, kind, arity)))
    # The unsupported-kind and survival probes carry the operator's own arity: a runner may
    # reject a wrong input count before it ever looks at a tensor's kind, and answering
    # MALFORMED to an arity it does not have is correct.  With arity 1 hardcoded, a binary
    # operator's runner is asked about an arity before it is asked about f16.
    first_arity = arities[0]
    probes.append(("unsupported kind", request(2, "f16", first_arity)))
    probes.append(("malformed line", "this is not a request"))
    probes.append(("survives malformed input", request(3, kind, first_arity)))
    return probes


def smoke(runner: Path, profile_data: dict, timeout: int = 300,
          lake: str | None = None) -> dict:
    probes = probe_requests(profile_data)
    payload = "".join(line + "\n" for _, line in probes)
    result = run_lean(runner, payload, timeout, lake)
    lines = response_lines(result.get("stdout") or "")
    checks = []
    problems = []
    if not result["ok"]:
        problems.append("the Lean runner exited with %s" % result["returncode"])
    if len(lines) != len(probes):
        problems.append("expected %d line(s), got %d" % (len(probes), len(lines)))
    for (label, _), line in zip(probes, lines):
        try:
            response = decode_response(line)
        except ProtocolError as exc:
            problems.append("%s: %s" % (label, exc))
            checks.append({"probe": label, "ok": False, "detail": str(exc)})
            continue
        checks.append({"probe": label, "ok": True, "status": response.status})
        if label == "unsupported kind" and response.status not in {"UNSUPPORTED", "ERR"}:
            problems.append("the runner did not report an unsupported kind")
    return {"ok": not problems, "findings": problems, "checks": checks,
            "lean": result, "probes": [label for label, _ in probes]}


# ---------------------------------------------------------------------------
# The generated driver
# ---------------------------------------------------------------------------


DRIVER_TEMPLATE = '''#!/usr/bin/env python3
# @generated-by sonnx-lean-pipeline-conformance
# @protocol sonnx-conformance/1
"""Lean conformance driver for {op}, generated by the SONNX pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_MARKER = ("agents", "sonnx_agents", "lean_conformance.py")


def _root_below(start: Path) -> Path | None:
    for root in (start, *start.parents):
        if root.joinpath(*_MARKER).is_file():
            return root
    return None


def _bootstrap() -> Path:
    starts = [Path(__file__).resolve()]
    env = os.environ.get("SONNX_REPO")
    if env:
        starts.append(Path(env).resolve())
    starts.append(Path.cwd().resolve())
    for start in starts:
        root = _root_below(start)
        if root is not None:
            sys.path.insert(0, str(root / "agents"))
            return root
    raise SystemExit(
        "the repository's agents/ package was not found above %s, in $SONNX_REPO, or "
        "above %s" % (Path(__file__).resolve(), Path.cwd()))


_bootstrap()
from sonnx_agents.lean_conformance import main

CONFIG = {config}


if __name__ == "__main__":
    raise SystemExit(main(CONFIG, __file__))
'''


def driver_source(config: dict) -> str:
    return DRIVER_TEMPLATE.format(
        op=config.get("op", ""),
        config=json.dumps(config, indent=4, sort_keys=True),
    )


def main(config: dict, driver_path: str | None = None, argv=None) -> int:
    here = Path(driver_path).resolve().parent if driver_path else Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path,
                        default=here / "lean" / "LeanConformance.json",
                        help="where to write the report")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="seconds for the Lean process")
    parser.add_argument("--max-failures", type=int, default=50,
                        help="failure details to keep in the report")
    parser.add_argument("--runner", type=Path, default=None,
                        help="the Lean runner (default: from the generated config)")
    args = parser.parse_args(argv)
    report = run(config, args.json, args.timeout, args.max_failures, here,
                 args.runner)
    summary = report.get("summary") or {}
    print("Lean conformance for %s: %s/%s passed, %s failed, %s unsupported"
          % (config.get("op", "?"), summary.get("passed", 0),
             summary.get("total", 0), summary.get("failed", 0),
             summary.get("unsupported", 0)))
    if report.get("error"):
        print(report["error"], file=sys.stderr)
    return 0 if report.get("ok") else 1
