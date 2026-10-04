#!/usr/bin/env python3
"""Check a SONNX operator specification against the mechanical rules of the guidelines.

Usage:  lint_spec.py ops/add.md [--op ADD]

This checks only what a program can decide. It never says a specification is
correct: the semantic checks (is the case analysis right, is a case missing)
belong to references/checklist.md and, on request, to the verification loop of
references/verification.md.

Reported here:
  * constructs the guidelines forbid in a specification: a "---" rule line, a
    code fence, and "must"/"should" (the modality the guidelines allow is the
    present indicative for the operator and "shall" for a requirement on the
    writer);
  * red specifying spans that are not balanced;
  * traceability tags: well-formedness, 10-by-10 numbering, every tag referred
    to is declared, every tag that is linked carries an anchor;
  * the "Contents" list against the sections and the anchors they carry;
  * a family heading without its "where" line and its one-line justification;
  * leftovers of the removed "checking the specification" material;
  * the end-of-line convention.
"""

import argparse
import collections
import re
import sys
from pathlib import Path

FAMILIES = {
    "real": ["real"],
    "float": ["float16", "float", "double"],
    "int": ["int8", "int16", "int32", "int64"],
    "uint": ["uint8", "uint16", "uint32", "uint64"],
}

TAG_RE = re.compile(r"E_([A-Z0-9]+)_(REAL|FLOAT|INT|UINT)_([A-Z0-9_]+?)_(\d{4})$")


def findings(path, op=None):
    raw = Path(path).read_bytes()
    bad = []
    text = raw.decode("utf-8").replace("\r\n", "\n")
    lines = text.split("\n")

    eol = "CRLF" if b"\r\n" in raw else "LF"
    if raw.count(b"\r\n") and raw.replace(b"\r\n", b"").count(b"\n"):
        bad.append("mixed end-of-line conventions")
    if b"\r\n" not in raw:
        bad.append("end-of-line is LF; the specifications in ops/ use CRLF")

    # ---- forbidden constructs ------------------------------------------------
    in_callout = False
    for i, line in enumerate(lines, 1):
        if line.startswith("> "):
            in_callout = True
            continue
        if in_callout and not line.strip():
            in_callout = False
        if in_callout:
            continue
        if line.strip() == "---":
            bad.append("line %d: a '---' rule line" % i)
        if line.startswith("```"):
            bad.append("line %d: a code fence" % i)
        for word in ("must", "should", "can be checked", "checkable"):
            if re.search(r"\b%s\b" % word, line, re.I):
                bad.append("line %d: %r" % (i, word))
    if "Checking the specification" in text:
        bad.append("a 'Checking the specification' section: it was removed from the guidelines")

    # ---- red specifying spans ------------------------------------------------
    # Each span is two lines carrying the same red style: one opening the
    # span, which names the tag, and one closing it, which is "[END]".
    span_lines = [i for i, l in enumerate(lines, 1) if '<span style="background: red;' in l]
    opens = [i for i in span_lines if not re.search(r">\[END\]<", lines[i - 1])]
    ends = [i for i, l in enumerate(lines, 1) if re.search(r">\[END\]<", l)]
    if len(opens) != len(ends):
        bad.append("%d red span(s) opened, %d closed" % (len(opens), len(ends)))
    for i in opens:
        m = re.search(r">\[(E_[A-Z0-9_]+)\]<", lines[i - 1])
        if not m:
            bad.append("line %d: a red span opens without naming a tag" % i)
    for i in ends:
        if not any(j < i for j in opens):
            bad.append("line %d: '[END]' closes no red span" % i)

    # ---- tags ----------------------------------------------------------------
    funt = set(re.findall(r">\[(E_[A-Z0-9_]+)\]<", text))       # red-span statements
    constraints = set(re.findall(r"`\[(E_[A-Z0-9_]+)\]`", text))  # constraint lists
    declared = funt | constraints
    anchors = set(re.findall(r'<a id="(E_[A-Z0-9_]+)"></a>', text))
    linked = set(re.findall(r'monospace">(E_[A-Z0-9_]+)</span>', text))
    section_anchors = set(re.findall(r'<a id="([^"E][^"]*)"></a>', text))

    for tag in sorted(declared | linked):
        m = TAG_RE.match(tag)
        if not m:
            bad.append("tag %s: not of the form E_<op>_<type>_<zone>_<number>" % tag)
            continue
        if op and m.group(1) != op:
            bad.append("tag %s: operator is %s, expected %s" % (tag, m.group(1), op))
        if int(m.group(4)) % 10:
            bad.append("tag %s: number is not a multiple of 10" % tag)
    for tag in sorted(linked - declared):
        bad.append("tag %s is referred to but never declared" % tag)
    for tag in sorted(linked - anchors):
        bad.append("tag %s is linked but carries no anchor" % tag)

    # ---- Contents against the sections ---------------------------------------
    entries = re.findall(r"^- \*\*\w+\*\* operator for type[s]? \[[^\]]*\]\(#([^)]+)\)$",
                         text, re.M)
    for anchor in entries:
        if anchor not in section_anchors:
            bad.append("Contents entry #%s has no section anchor" % anchor)
    for anchor in sorted(section_anchors):
        if anchor not in entries:
            bad.append("section anchor #%s has no Contents entry" % anchor)
    if not re.search(r"^Based on ONNX", text, re.M):
        bad.append("no 'Based on ONNX ...' line")
    if not re.search(r"^Revision ", text, re.M):
        bad.append("no revision line")

    # ---- a family heading and its justification ------------------------------
    # A heading that names one family covering several types carries a "where"
    # line and one line justifying the grouping. The real-number section, which
    # covers a single domain, carries neither.
    def next_text(start):
        """The first following line that is neither blank nor an anchor."""
        for j in range(start, len(lines)):
            s = lines[j].strip()
            if s and not s.startswith("<a id="):
                return j, s
        return None, ""

    for i, line in enumerate(lines):
        m = re.match(r"^# \*\*\w+\*\* \((.+), (.+)\)$", line)
        if not m:
            continue
        names = {m.group(1), m.group(2)}
        if len(names) != 1:
            continue
        fam = names.pop()
        if fam not in FAMILIES or len(FAMILIES[fam]) < 2:
            continue
        j, nxt = next_text(i + 1)
        if not nxt.startswith("where "):
            bad.append("line %d: family heading without a 'where' line" % (i + 1))
            continue
        for want in FAMILIES[fam]:
            if "`%s`" % want not in nxt:
                bad.append("line %d: 'where' line misses `%s` (or does not set it in backticks)"
                           % (j + 1, want))
        k, just = next_text(j + 1)
        if "share one semantics" not in just:
            bad.append("line %d: no one-line justification that the types share one semantics"
                       % (k + 1))

    # ---- information ---------------------------------------------------------
    print("%s: %d lines, %s" % (path, len(lines), eol))
    print("  sections       : %s" % ", ".join(sorted(section_anchors)))
    print("  red spans      : %d" % len(opens))
    print("  tags declared  : %d (%d statements, %d constraints)"
          % (len(declared), len(funt), len(constraints)))
    zones = collections.Counter(TAG_RE.match(t).group(3) for t in declared
                               if TAG_RE.match(t))
    print("  tag zones      : %s" % dict(zones))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec")
    ap.add_argument("--op", help="operator name, e.g. ADD, to check the tags against")
    args = ap.parse_args()

    bad = findings(args.spec, args.op)
    if bad:
        print("\n%d finding(s):" % len(bad))
        for item in bad:
            print("  - %s" % item)
        return 1
    print("\nno mechanical finding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
