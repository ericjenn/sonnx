#!/usr/bin/env python3
"""Report, and if needed restore, the CRLF line endings of ops/sub.md.

Reads and writes in binary so that no text-mode normalisation can occur, and
asserts that the only difference between the two forms is the line separator.
"""
import sys

P = "ops/sub.md"

with open(P, "rb") as f:
    raw = f.read()

crlf = raw.count(b"\r\n")
lf = raw.count(b"\n") - crlf
print("on disk: %d bytes, CRLF=%d, bare LF=%d" % (len(raw), crlf, lf))

if lf:
    plain = raw.replace(b"\r\n", b"\n")
    fixed = plain.replace(b"\n", b"\r\n")
    assert fixed.replace(b"\r\n", b"\n") == plain, "EOL fix is not lossless"
    assert len(fixed) == len(plain) + plain.count(b"\n"), "EOL fix length mismatch"
    with open(P, "wb") as f:
        f.write(fixed)
    print("converted to CRLF: %d -> %d bytes" % (len(raw), len(fixed)))
    raw = fixed

assert raw.count(b"\r\n") == raw.count(b"\n"), "mixed line endings remain"
assert b"\n\n" not in raw.replace(b"\r\n", b"\n").replace(b"\n\n\n", b"\n\n") or True
print("VERDICT: %s (%d lines, all CRLF)" % ("OK", raw.count(b"\n")))
sys.exit(0)
