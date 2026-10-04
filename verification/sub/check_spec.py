#!/usr/bin/env python3
"""Check ops/sub.md against the SONNX guidelines (informal_corrected.md).

Run with cwd = the sonnx folder.  Every assertion prints OK/FAIL; the exit
status is non-zero if any check fails.
"""
import re, sys

raw = open("ops/sub.md", encoding="utf-8").read().replace("\r\n", "\n")
ok = True
def chk(c, m):
    global ok
    print(("  OK   " if c else "  FAIL ") + m)
    if not c:
        ok = False

L = raw.split("\n")
print("HYGIENE")
chk(raw.endswith("\n") and not raw.endswith("\n\n"), "single terminating newline")
chk(all(l == l.rstrip() for l in L), "no trailing whitespace")
chk(not any(not L[i].strip() and not L[i + 1].strip() for i in range(len(L) - 1)), "no double blank line")
for w in ("shall", "must", "should"):
    chk(not re.search(r"\b%s\b" % w, raw, re.I), "no '%s' (Modality: present indicative throughout)" % w)

marks = [(m.start(), m.group(1)) for m in re.finditer(r'<a id="(real|float|int)"></a>', raw)]
chk([f for _, f in marks] == ["real", "float", "int"], "three anchored sections in order: real, float, int")
blocks = [(f, raw[p:(marks[k + 1][0] if k + 1 < len(marks) else len(raw))]) for k, (p, f) in enumerate(marks)]
contents = raw[:marks[0][0]]

print("CONTENTS (guidelines section 'Contents', P20)")
ents = re.findall(r"^- \*\*Sub\*\* operator for type[s]? (.+)$", contents, re.M)
chk(re.findall(r"\]\(#([a-z]+)\)", "\n".join(ents)) == ["real", "float", "int"], "one linked entry per section, same order")
chk(contents.count("Based on ONNX documentation [Sub version 14](https://onnx.ai/onnx/operators/onnx__Sub.html#sub-14).") == 1,
    "ONNX reference, with the opset and the deep link")
chk(re.search(r"^Revision 20\d\d-\d\d-\d\d: .*opset 14\.$", contents, re.M) is not None,
    "revision line: opset, date and change")

TOP = ["## Signature", "## Restrictions", "## Function", "### Example",
       "## Error conditions", "## Attributes", "## Inputs", "## Outputs"]
for fam, b in blocks:
    print("SECTION %s" % fam)
    h = b.split("\n")
    chk(re.search(r"^# \*\*Sub\*\* \([^)]*\)$", b, re.M) is not None, "type-set heading")
    seq = [l for l in h if l.startswith(tuple(TOP)) or l.startswith("#### Constraints")]
    flat = [next(i for i, t in enumerate(TOP) if l.startswith(t)) for l in seq if not l.startswith("####")]
    chk(flat == sorted(flat), "sections in the prescribed order: %s" % ", ".join(TOP[i] for i in dict.fromkeys(flat)))
    good, last = True, None
    for l in seq:
        if l.startswith("####"):
            if last not in ("## Attributes", "## Inputs", "## Outputs"):
                good = False
        else:
            last = l
    chk(good, "each Constraints subsection follows an Attributes/Inputs/Outputs heading")
    chk(b.count("#### Constraints") == 3,
        "three Constraints subsections (one per input and output; none under Attributes)")
    for s in ["[General restrictions](./../common/general_restrictions.md) are applicable.",
              "No specific restrictions apply to the **Sub** operator.",
              "Operator **Sub** has no attribute."]:
        chk(b.count(s) == 1, s[:50])

    t = re.findall(r'0\.7em;">\[(E_SUB_%s_FUNC_\d+)\]</br></span>' % fam.upper(), b)
    chk(len(t) == 1, "one red beginning tag")
    chk(b.count('0.7em;">[END]</br></span>') == 1, "one red [END]")
    t = t[0]
    a, e = b.index("[%s]" % t), b.index("[END]</br></span>", b.index("[%s]" % t))
    chk(b.index("## Function") < a < e < b.index("### Example"),
        "the tag opens the Function section and [END] closes before the examples")
    chk(b.index("$$", a) < e, "the $$ formula is inside the specifying part")
    chk(re.search(r"^> ", b[a:e], re.M) is None, "no callout inside the specifying part (%s)" % fam)
    ds = [i for i, l in enumerate(h) if l.strip() == "$$"]
    chk(len(ds) % 2 == 0 and all(not h[i - 1].strip() and not h[j + 1].strip() and i + 1 < j
                                 for i, j in zip(ds[::2], ds[1::2])),
        "every $$ block is paired, non-empty and blank-line separated (%d blocks)" % (len(ds) // 2))
    chk("```" not in b, "no ``` fence: the examples are $$ display blocks too")
    chk(len(ds) >= 14, "one $$ block per operand set and per worked result")

    items = re.findall(r'^(<a id="(E_[A-Za-z0-9_]+)"></a>\n)?- `\[(E_[A-Za-z0-9_]+)\]`([^\n]*)\n((?:\s+-[^\n]*\n)*)', b, re.M)
    cross = {x[2] for x in items if "see constraint" in x[4]}
    declared = [x[2] for x in items if x[0]] + [t]
    targets = set(re.findall(r"\]\(#([^)]+)\)", b))
    chk(len({x[2] for x in items}) == len(items), "each constraint described once (%d items)" % len(items))
    chk(all(n in declared for n in targets), "every tag referred to carries its anchor: %s" % sorted(targets))
    chk(t in declared, "the Function tag is declared with an anchor")
    chk(re.search(r'^<a id="%s"></a>\n<span style="background: red' % t, b, re.M) is not None,
        "the anchor immediately precedes the red beginning tag")
    chk(all(n.startswith("E_SUB_%s_" % fam.upper()) for n in declared + [t]),
        "every tag component is this section's family (%s)" % fam.upper())
    chk(all(re.fullmatch(r"E_SUB_(REAL|FLOAT|INT)_(FUNC|CONSTR_[ABC])_\d{4}", n) for n in declared + [t]),
        "tag grammar <op>_<TYPE>_<ZONE>_<number>")
    chk(all(int(n.rsplit("_", 1)[1]) % 10 == 0 for n in declared + [t]),
        "10-by-10 numbering: %s" % sorted(n.rsplit("_", 1)[1] for n in declared + [t]))
    xrefs = re.findall(r"\]\(#([^)]+)\)", b)
    chk(all(x in declared or x == t for x in xrefs), "every hyperlink resolves to a declared tag: %d links" % len(xrefs))
    chk(t in xrefs, "the Function tag is referred to from the disposition statement")
    chk(all(x in xrefs for x in declared if "_A_" in x), "the first occurrence of each shared constraint is the cross-reference target")

    if fam != "real":
        chk(re.search(r"where %s is in \{`[^`]+`(, `[^`]+`)*\}\." % fam, b) is not None, "P25: family designated by a where-line")
        chk(re.search(r"^The (three|eight) types share one semantics", b, re.M) is not None, "P25: grouping justified in one line")
    for c in ["zero-sized dimension", "no dimension (a rank-0 tensor)", "broadcasting rules"]:
        chk(c.lower() in b.lower(), "P11 corner case: %s" % c)
    chk(b.count("| Condition | Disposition |") == int(fam != "real"), "P12 disposition table where the list applies")
    chk("No error condition." in b, "states the error-condition conclusion")

print("WHOLE DOCUMENT")
chk(raw.count('font-size:0.7em;">[E_SUB_') == 3, "three red beginning tags, one per section")
hdr = [i for i, l in enumerate(L) if l.startswith("> [!")]
chk(hdr and all(re.fullmatch(r"> \[!(note|warning|question)\] \S+ · .+", L[i]) for i in hdr),
    "callout headers: > [!kind] id · title (%d callouts)" % len(hdr))
blocks, i = [], 0
while i < len(L):
    if L[i].startswith("> [!") or (L[i].startswith("> ") and i and L[i - 1].startswith(">")):
        j = i
        while j < len(L) and L[j].startswith(">"):
            j += 1
        blocks.append((i, j - 1)); i = j
    else:
        i += 1
chk(len(blocks) == len(hdr), "each callout is one contiguous block")
chk(all(not L[i - 1].strip() and not L[j + 1].strip() for i, j in blocks),
    "a blank line before and after every callout")
chk(not any(L[i - 1].startswith(">") for i, _ in blocks), "callouts are not adjacent to each other")
chk(raw.count("\\approx") == 1 and "0.19999999999999998" in raw, "I5/T4: \\approx on the one inexact example only")
chk(raw.count("Signed and unsigned integer types") == 1, "P11: the signed/unsigned difference stated in the integer section")
chk(re.search(r"\b(dividend|denominator)\b", raw) is None, "no vocabulary carried over from the Div specimen")
chk(re.search(r"\b(minuend|subtrahend)\b", raw, re.I) is None, "operands named in plain words, not the classical pair")
chk("```" not in raw, "no code fence anywhere: display math is $$, which Obsidian renders")
print()
print("bytes=%d lines=%d" % (len(raw.encode()), raw.count("\n")))
print("VERDICT: " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
