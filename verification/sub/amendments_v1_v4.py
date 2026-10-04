#!/usr/bin/env python3
"""Apply the V1-V4 amendments to ops/sub.md, each with a callout recording it.

V1  overflow threshold      : the fourth case of the float definition is removed,
                              round(x) is defined with an unbounded exponent range
V2  the NaN of the invalid  : the first case now says which NaN conforms
    operation
V3  vacuous null result     : "null or subnormal" -> "subnormal"
V4  two rows of the        : the Overflow and Underflow rows of the float section
    Error-conditions table

Every replacement asserts its occurrence count first, reads and writes in binary
so that the CRLF line endings cannot change, and the callouts are asserted to lie
outside the specifying part (no "> " line between the beginning tag and [END]).
"""
import sys

P = "ops/sub.md"
raw = open(P, "rb").read()
s = raw.decode("utf-8").replace("\r\n", "\n")

LINK = ('[<b><span style="font-family: \'Courier New\', monospace">'
        'E_SUB_FLOAT_FUNC_0010</span></b>](#E_SUB_FLOAT_FUNC_0010)')

# ----------------------------------------------------------------- replacements
SUBS = [
# ---- V1: the case analysis -------------------------------------------------
("""\\text{NaN} & \\text{if } \\tilde{A}[i] \\text{ or } \\tilde{B}[i] \\text{ is NaN, or if they are both } +\\infty \\text{ or both } -\\infty \\\\
\\tilde{A}[i] - \\tilde{B}[i] & \\text{if the exact difference is representable in the type of } C \\\\
\\pm\\text{inf} & \\text{if the exact difference overflows the range of the type of } C \\\\
\\text{round}(\\tilde{A}[i] - \\tilde{B}[i]) & \\text{otherwise}""",
 """\\text{NaN} & \\text{if } \\tilde{A}[i] \\text{ or } \\tilde{B}[i] \\text{ is NaN, or if they are both } +\\infty \\text{ or both } -\\infty \\\\
\\tilde{A}[i] - \\tilde{B}[i] & \\text{if the exact difference is representable in the type of } C \\\\
\\text{round}(\\tilde{A}[i] - \\tilde{B}[i]) & \\text{otherwise}"""),

("- $\\text{round}(x)$ is the value of $x$ rounded to the nearest representable value of the type of $C$, according to the IEEE 754 rounding rules.",
 "- $\\text{round}(x)$ is the value of $x$ rounded to the nearest value of the type of $C$ using the roundTiesToEven attribute of IEEE 754, the exponent range of the type being unbounded: a magnitude whose nearest value exceeds the largest finite value of the type of $C$ gives an infinity of the sign of $x$."),

# ---- V2 (first case) and V3 (last case) -------------------------------------
("""In the first case, an operand that is NaN is propagated, and the subtraction of two infinities of the same sign is the invalid operation defined in IEEE 754 section 7.2; the result is NaN in both cases.

In the third case, the sign of $\\pm\\text{inf}$ is the sign of the exact difference.

In the last case, the exact difference is not representable in the type of $C$: the result is that difference rounded as defined above. The rounding may make the result null or subnormal; its sign is then the sign of the exact difference.""",
 """In the first case, an operand that is NaN is propagated, and the subtraction of two infinities of the same sign is the invalid operation defined in IEEE 754 section 7.2; the result is NaN in both cases. The NaN of this case is a quiet NaN; its sign and its payload are not specified, whether the NaN comes from an operand or from the invalid operation, so that every quiet NaN of the type of $C$ is a conforming result.

In the last case, the exact difference is not representable in the type of $C$: the result is that difference rounded as defined above. The rounding may make the result subnormal; its sign is then the sign of the exact difference."""),

# ---- V4: the two rows of the disposition table ------------------------------
("| Overflow, i.e., an exact difference outside the range of the type | nominal: specified by %s, the result is $\\pm\\text{inf}$ |" % LINK,
 "| Overflow, i.e., an exact difference that rounds beyond the largest finite value of the type | nominal: specified by %s, the result is $\\pm\\text{inf}$ |" % LINK),

("| Underflow, i.e., an exact difference below the smallest subnormal of the type | nominal: specified by %s, the result is the rounded value, possibly $\\pm 0$ |" % LINK,
 "| Underflow, i.e., an exact difference with a magnitude below the smallest normal value of the type | nominal: specified by %s, the result is that difference, which is a subnormal value |" % LINK),
]

for old, new in SUBS:
    n = s.count(old)
    assert n == 1, "%d occurrences, expected 1, of:\n%s" % (n, old[:120])
    s = s.replace(old, new)

# ----------------------------------------------------------------- callouts
CALL = {}


def para(lines):
    return "\n".join("> " + l if l else ">" for l in lines)


V1 = r"""
> [!warning] V1 · The overflow threshold
> The definition had a fourth case, "{INF} if the exact difference overflows the range of the
> type of $C$", and the Error-conditions table repeated that reading. IEEE 754 calls a result an
> overflow only when the value rounded with an unbounded exponent range exceeds the largest
> finite value of the type, so for an exact difference in the band
> $(\text{max}, \text{max} + \text{ulp(max)}/2)$ the document demanded $\pm\text{inf}$ where
> IEEE 754, NumPy and ONNX Runtime produce `max`: a conforming implementation of the document
> had to return a result IEEE 754 forbids.
> **Found by** implementing the operator from this document alone
> (`verification/sub/impl_from_spec.py`) and comparing with ONNX Runtime (opset 14,
> `verification/sub/results.md`): one case per float type mismatched, e.g. `float16`
> `65504 - (-8)`, whose exact difference `65512` lies in the band and gives `65504`
> (`0x7bff`), not `inf`.
> **Now** the case is gone and the infinities follow from the rounding itself: `round(x)` is
> defined with an unbounded exponent range. The threshold that produces is the midpoint
> between the largest finite value and the next power of two, because that midpoint is a tie
> and `roundTiesToEven` breaks it upwards; the two rules are one rule. Checked point by point
> on the three float types against ONNX Runtime, at the midpoint and on each side of it.
""".replace("{INF}", "$\\pm\\text{inf}$").lstrip("\n")

V2 = r"""
> [!warning] V2 · The NaN of the invalid operation
> The document said the result is NaN and stopped there. Which NaN is not fixed by IEEE 754,
> and implementations differ: ONNX Runtime returns `0xfe00`, `0xffc00000` and
> `0xfff8000000000000` for `inf - inf` — the canonical quiet NaN with the sign bit set — where
> a straightforward implementation returns the positive one. Payloads differ as well: ONNX
> Runtime keeps the payload of a NaN operand in `float32` and `float64`, but canonicalises it
> in `float16` (`0x7e01` becomes `0x7e00`).
> **Now** the first case states that every quiet NaN of the type is a conforming result, for a
> NaN operand as for the invalid operation. The permissive rule is also the only workable one:
> a stricter rule would make ONNX Runtime non-conforming for `float16`.
""".lstrip("\n")

V3 = r"""
> [!note] V3 · A null result from rounding
> The last case said the rounding "may make the result null or subnormal". It cannot be null:
> the values of a type are all multiples of that type's smallest subnormal, so the difference
> of two of them is too — a non-null difference is at least one smallest subnormal, and a null
> difference is decided by the signed-zero rule below it.
> **Now** the sentence says "subnormal".
""".lstrip("\n")

V4 = r"""
> [!note] V4 · Two rows of the table above
> The Overflow row named the *exact* difference — "an exact difference outside the range of
> the type" — which is the reading V1 corrects; it now names the rounded result. The Underflow
> row promised a result "possibly $\pm 0$", which V3 rules out for the same reason, and it
> named the smallest subnormal where IEEE 754 sets underflow at the smallest normal value;
> with a magnitude below that value the exact difference is always representable, as a
> subnormal. The integer section keeps its own wording: there the range of the type does
> decide the case, and the wrap-around is nominal.
""".lstrip("\n")

# insert V1, V2, V3 after the [END] of the float section, V4 at the end of its
# Error-conditions section
fi, ii = s.index('<a id="float"></a>'), s.index('<a id="int"></a>')
sec = s[fi:ii]
END = '0.7em;">[END]</br></span>'
assert sec.count(END) == 1, "the float section has %d end tags" % sec.count(END)
assert sec.count("No error condition.") == 1, "the float section has %d conclusions" % sec.count("No error condition.")

p = fi + sec.index(END) + len(END)
s = s[:p] + "\n\n" + V1 + "\n" + V2 + "\n" + V3.rstrip("\n") + s[p:]

fi, ii = s.index('<a id="float"></a>'), s.index('<a id="int"></a>')
sec = s[fi:ii]
concl = "Every condition of the list is part of the nominal behavior of the operator; none of them is an error. No error condition."
assert sec.count(concl) == 1
p = fi + sec.index(concl) + len(concl)
s = s[:p] + "\n\n" + V4.rstrip("\n") + s[p:]

# ----------------------------------------------------------------- checks
for tag, fam in (("E_SUB_REAL_FUNC_0010", "real"), ("E_SUB_FLOAT_FUNC_0010", "float"),
                 ("E_SUB_INT_FUNC_0010", "int")):
    a = s.index("[%s]" % tag)
    e = s.index(END, a)
    body = s[a:e]
    assert "\n>" not in body, "a callout sits inside the specifying part of %s" % fam
assert s.count("> [!warning]") == 2 and s.count("> [!note]") == 2, "callout count"
assert s.count("roundTiesToEven") == 2 and s.count("overflow") >= 1
assert s.count("conforming result") == 2   # once normative, once in V2
for phrase in ("null or subnormal", "possibly $\\pm 0$"):
    hits = [l for l in s.split("\n") if phrase in l]
    assert hits and all(l.startswith("> ") for l in hits), "%s survives outside a callout" % phrase
assert "\\pm\\text{inf} & \\text{if the exact difference overflows" not in s
assert "In the third case" not in s

out = s.replace("\n", "\r\n").encode("utf-8")
assert out.count(b"\r\n") == out.count(b"\n"), "mixed line endings"
open(P, "wb").write(out)
print("wrote %s: %d -> %d bytes, %d lines (all CRLF)"
      % (P, len(raw), len(out), out.count(b"\n")))
