#!/usr/bin/env python3
"""One iteration of the specification loop: compare an implementation of an operator,
written from a specification, with ONNX Runtime.

Usage:

    specloop.py --spec ops/add.md --op Add --cases verification/add/cases.py \
                --impl verification/add/loop/iter-001-impl.py

The implementation and the cases are ordinary Python modules:

  * the implementation module exposes the operator under the entry point named by
    ``--entry`` (default: the operator in lower case, e.g. ``add``), called with one
    positional argument per operand, followed by the node's attributes as keyword
    arguments (``maxpool(x, kernel_shape=[2, 2])``).  An operator whose document gives it
    several outputs returns them as a tuple in the document's order; one with a single
    output returns the array itself;
  * the cases module exposes ``cases()``, a generator of ``(label, [array, ...])``, or of
    ``(label, [array, ...], attrs)`` where ``attrs`` is a dict of the node's attributes
    under the names the document gives them;
  * the cases module may declare ``NODE_OUTPUTS``, the outputs the one-node model asks for
    (default: a single output named ``out``), and ``OUTPUT_TYPES``, a dict naming the type
    of an output that is not the operand's type (e.g. ``{"Indices": numpy.int64}``);
  * it may declare ``sweeps()``, a function returning a list of bulk checks that build
    their own ONNX Runtime calls (for a domain sweep, which must be batched: one run, not
    65536).  A sweep builds its own node, so it passes the attributes itself.

What is reported, and how it is classified:

  * ``mismatch``     -- the implementation and ONNX Runtime disagree.  This is the
                        finding of the loop: the specification says something the
                        reference implementation does not do.
  * ``open``         -- an ONNX Runtime model refused by the runtime, or an operand
                        combination the specification's own constraints rule out.
                        Not a mismatch: it is outside the operator's domain.
  * ``nan payload``  -- the two results are both NaN but differ in payload or sign.
                        Reported, not counted as a mismatch.

The exit status is 0 when no case mismatched, whatever the number of open cases; the
JSON report is what the loop reads.  A non-zero exit means the specification needs an
amendment, or the implementation needs to be rewritten from it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

UINT_OF_WIDTH = {2: np.uint16, 4: np.uint32, 8: np.uint64}
IR_VERSION = 10  # onnxruntime 1.30 refuses a model above IR 13


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bits(arr: np.ndarray) -> str:
    """Bit pattern of an array, for the report."""
    if arr.dtype.kind == "f":
        return str(np.ascontiguousarray(arr).view(UINT_OF_WIDTH[arr.dtype.itemsize]))
    return str(arr)


def compare(expected: np.ndarray, actual: np.ndarray):
    """Return (equal, nan_payload_differ) for two arrays of the same dtype."""
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False, False
    if expected.dtype.kind == "f":
        uint = UINT_OF_WIDTH[expected.dtype.itemsize]
        be = np.ascontiguousarray(expected).view(uint)
        ba = np.ascontiguousarray(actual).view(uint)
        both_nan = np.isnan(expected) & np.isnan(actual)
        payload = bool(np.any(both_nan & (be != ba)))
        equal = (be == ba) | both_nan
        return bool(np.all(equal)), payload
    return bool(np.all(expected == actual)), False


def mismatch_mask(expected: np.ndarray, actual: np.ndarray):
    """Boolean mask of the elements that differ, NaN==NaN, for the report."""
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return None
    if expected.dtype.kind == "f":
        uint = UINT_OF_WIDTH[expected.dtype.itemsize]
        be = np.ascontiguousarray(expected).view(uint)
        ba = np.ascontiguousarray(actual).view(uint)
        both_nan = np.isnan(expected) & np.isnan(actual)
        return ~((be == ba) | both_nan)
    return ~(expected == actual)


class Runner:
    """A one-node ONNX model of the operator, one cached session per node.

    The inputs are declared with an unknown shape, so a single session per element
    type serves every case; a shape the operator cannot combine makes the runtime
    refuse the *model*, which is recorded as an open case, not as a mismatch.  The
    attributes of a case are part of the node, so the session is cached per attribute
    set as well as per type: two cases that ask for different `kernel_shape` are two
    different nodes.  The node's outputs are the cases module's `NODE_OUTPUTS`, each
    declared with the type `OUTPUT_TYPES` gives it — the operand's type when it gives
    none, which is every output of every element-wise operator.
    """

    def __init__(self, op: str, opset: int, outputs=("out",), out_types=None):
        self.op = op
        self.opset = opset
        self.outputs = tuple(outputs)
        self.out_types = dict(out_types or {})
        self._sessions = {}

    def session(self, dtype: np.dtype, arity: int, attrs: dict):
        key = (dtype.str, arity, json.dumps(attrs, sort_keys=True, default=str))
        if key not in self._sessions:
            import onnxruntime as ort
            from onnx import helper

            proto = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
            names = ["in%d" % i for i in range(arity)]
            value_infos = [
                helper.make_tensor_value_info(n, proto, None) for n in names
            ]
            value_info_out = []
            for out in self.outputs:
                declared = self.out_types.get(out)
                value_info_out.append(helper.make_tensor_value_info(
                    out,
                    proto if declared is None
                    else helper.np_dtype_to_tensor_dtype(np.dtype(declared)),
                    None))
            node = helper.make_node(self.op, names, list(self.outputs), name="node0",
                                    **dict(attrs))
            graph = helper.make_graph([node], "g", value_infos, value_info_out)
            model = helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", self.opset)],
                ir_version=IR_VERSION,
            )
            self._sessions[key] = ort.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        return self._sessions[key]

    def run(self, arrays, attrs=None):
        """Run the model. Raises RuntimeError when the runtime refuses it."""
        arity = len(arrays)
        dtypes = {np.asarray(a).dtype for a in arrays}
        if len(dtypes) != 1:
            raise RuntimeError(
                "operands of different types (%s): outside the domain, the "
                "specification constrains them to have the same type"
                % ", ".join(sorted(str(d) for d in dtypes))
            )
        dtype = dtypes.pop()
        sess = self.session(dtype, arity, dict(attrs or {}))
        names = ["in%d" % i for i in range(arity)]
        return tuple(sess.run(list(self.outputs),
                              dict(zip(names, [np.asarray(a) for a in arrays]))))


def implementation_outputs(result):
    """The outputs of one call of the implementation, as a list.

    An operator whose document gives it several outputs returns them as a tuple in the
    document's order; one with a single output returns the array itself, which is every
    operator specified before this one.
    """
    if isinstance(result, (tuple, list)):
        return list(result)
    return [result]


def check_cases(runner, impl, entry, cases, max_report):
    multi = len(runner.outputs) > 1
    report = {"total": 0, "passed": 0, "mismatched": [], "open": [], "nan_payload": []}
    for item in cases:
        label, arrays = item[0], item[1]
        attrs = dict(item[2]) if len(item) > 2 and item[2] else {}
        # an attribute a case does not give is the node's default and the implementation's
        # default at once, which is how the document's stated defaults are tested
        report["total"] += 1
        try:
            expected = runner.run(arrays, attrs)
        except Exception as exc:  # refused by the runtime: outside the domain
            report["open"].append({"case": label, "reason": "%s: %s"
                                   % (type(exc).__name__, exc),
                                   **({"attrs": attrs} if attrs else {})})
            continue
        try:
            actual = implementation_outputs(getattr(impl, entry)(*arrays, **attrs))
        except Exception as exc:
            report["mismatched"].append({
                "case": label,
                "reason": "the implementation raised %s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(limit=3),
                **({"attrs": attrs} if attrs else {}),
            })
            continue
        if len(actual) < len(expected):
            report["mismatched"].append({
                "case": label, "elements_differing": -1,
                "reason": "the implementation returned %d output(s) and the node has %d "
                          "(%s): an output the document gives is missing"
                          % (len(actual), len(expected), ", ".join(runner.outputs)),
                **({"attrs": attrs} if attrs else {}),
            })
            continue
        detail = {"case": label, "elements_differing": 0,
                  **({"attrs": attrs} if attrs else {})}
        bad = False
        for name, want, got in zip(runner.outputs, expected, actual):
            got = np.asarray(got)
            equal, payload = compare(want, got)
            if payload:
                report["nan_payload"].append(
                    {"case": label, **({"output": name} if multi else {})})
            if equal:
                continue
            bad = True
            mask = mismatch_mask(want, got)
            n = int(mask.sum()) if mask is not None else -1
            one = {"elements_differing": n}
            if mask is not None and n > 0:
                idx = np.argwhere(mask).ravel()[:max_report]
                one["first_differences"] = [
                    {"index": int(i),
                     "ort": str(want.ravel()[i]),
                     "impl": str(got.ravel()[i]),
                     "ort_bits": bits(want.ravel()[i:i + 1]),
                     "impl_bits": bits(got.ravel()[i:i + 1])}
                    for i in idx
                ]
            else:
                one["ort_shape"] = list(np.shape(want))
                one["impl_shape"] = list(np.shape(got))
            if multi:
                # with several outputs the case names which one differs, and a difference
                # in shape or type is one of them rather than the whole case
                detail.setdefault("outputs_differing", {})[name] = one
                if n < 0:
                    detail["elements_differing"] = -1
                elif detail["elements_differing"] >= 0:
                    detail["elements_differing"] += n
            else:
                # one output: the case is the output, as it has always been in the report
                detail.update(one)
        if bad:
            report["mismatched"].append(detail)
        else:
            report["passed"] += 1
    return report


def check_sweeps(impl, sweeps):
    out = []
    for sweep in sweeps:
        try:
            out.append(sweep(impl))
        except Exception as exc:
            out.append({"sweep": getattr(sweep, "__name__", "sweep"),
                        "error": "%s: %s" % (type(exc).__name__, exc),
                        "traceback": traceback.format_exc(limit=3)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True, help="the specification under test")
    ap.add_argument("--op", required=True, help="the ONNX operator type, e.g. Add")
    ap.add_argument("--cases", required=True, help="the cases module")
    ap.add_argument("--impl", required=True, help="the implementation written from the spec")
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--entry", default=None, help="entry point (default: lowercase op)")
    ap.add_argument("--json", default=None, help="write the report here")
    ap.add_argument("--max-report", type=int, default=4,
                    help="elements shown per mismatching case")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    entry = args.entry or args.op.lower()
    impl = load_module(args.impl, "spec_impl")
    cases_mod = load_module(args.cases, "spec_cases")

    runner = Runner(args.op, args.opset,
                    outputs=getattr(cases_mod, "NODE_OUTPUTS", ("out",)),
                    out_types=getattr(cases_mod, "OUTPUT_TYPES", None))
    report = {
        "spec": args.spec,
        "op": args.op,
        "opset": args.opset,
        "impl": args.impl,
        "entry": entry,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cases": check_cases(runner, impl, entry, cases_mod.cases(), args.max_report),
        "sweeps": check_sweeps(impl, cases_mod.sweeps() if hasattr(cases_mod, "sweeps") else []),
    }
    if runner.outputs != ("out",):
        # a node whose outputs are the operator's own and not the single anonymous one of
        # every element-wise operator: the report says which outputs it compared.  The default
        # is left out, so that the report of an operator with one output and no attributes is
        # exactly the report this harness has always written.
        report["outputs"] = list(runner.outputs)
    bad_sweeps = [s for s in report["sweeps"] if s.get("error") or s.get("failures")]
    report["ok"] = not report["cases"]["mismatched"] and not bad_sweeps
    if not hasattr(impl, entry):
        report["ok"] = False
        report["cases"]["mismatched"].append(
            {"case": "<module>", "reason": "the implementation has no entry point %r" % entry})

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")

    if not args.quiet:
        c = report["cases"]
        print("spec      : %s" % args.spec)
        print("impl      : %s (%s.%s)" % (args.impl, Path(args.impl).stem, entry))
        print("cases     : %d run, %d passed, %d mismatched, %d open, %d NaN-payload"
              % (c["total"], c["passed"], len(c["mismatched"]), len(c["open"]),
                 len(c["nan_payload"])))
        if len(runner.outputs) > 1:
            print("outputs   : %s" % ", ".join(runner.outputs))
        for sweep in report["sweeps"]:
            if sweep.get("error"):
                print("sweep     : %s ERROR %s" % (sweep.get("sweep"), sweep["error"]))
                for line in str(sweep.get("traceback", "")).splitlines()[-3:]:
                    print("              %s" % line)
            else:
                print("sweep     : %s %s" % (sweep.get("sweep"), sweep.get("summary", "")))
                # a sweep that failed but shows no reason is worse than no sweep at all
                for item in sweep.get("failures", []):
                    print("        SWEEP MISMATCH %s: %s"
                          % (item.get("who", ""), item.get("detail", "")))
        for item in c["mismatched"]:
            print("\nMISMATCH %s" % item["case"])
            for key, value in item.items():
                if key != "case":
                    print("  %s: %s" % (key, value))
        for item in c["open"]:
            print("\nopen (outside the domain) %s: %s" % (item["case"], item["reason"]))
        print("\n%s" % ("OK: no mismatch" if report["ok"] else "NOT OK: the specification needs an amendment"))

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
