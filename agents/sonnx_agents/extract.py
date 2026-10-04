"""Getting an artifact out of a model's prose.

A model asked for a document answers with a document, usually; sometimes with an
introduction, a closing remark, or a fence around it.  These are the three readers the
nodes use, and each is deliberately forgiving: a run that dies because the model wrote
"Here is the specification:" costs more than a run that strips it.
"""

from __future__ import annotations

import json
import re

# a fenced block, whatever the language tag
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n(.*?)\n\s*```\s*$", re.S)
_ANY_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.S)
_PREAMBLE = re.compile(
    r"^\s*(here (is|are)[^\n]*|the (specification|document|module|code)[^\n]*|"
    r"sure[,!]?[^\n]*|certainly[,!]?[^\n]*)[:\n]\s*",
    re.I,
)

# The endpoint's chat template leaks its own tool-call syntax into some answers, as lines of
# the form `｜DSML｜ parameter>` (U+FF5C for the bar) at the end of the artifact.  It is the
# envelope of the answer and not the answer, and it is stripped for the same reason a preamble
# is: a module with three lines of the model's calling convention after its last statement is
# a module and a piece of protocol, and only the first is what was asked for.  Measured on
# 2026-10-04: the second case set of **Atan** and the first of **Asin** both ended this way —
# `SyntaxError: invalid character '｜' (U+FF5C)` on the last line — with the module itself
# complete and correct above it.
_TOOL_SYNTAX = re.compile(r"^[ \t]*[|｜][ \t]*DSML[^\n]*$", re.M)


def strip_protocol(text: str) -> str:
    """Drop the endpoint's own tool-call syntax from an answer."""
    return _TOOL_SYNTAX.sub("", text).rstrip() + "\n" if text.strip() else text


def strip_preamble(text: str) -> str:
    """Drop a leading 'Here is …:' line and any closing remark after a fenced block."""
    text = strip_protocol(text).strip()
    for _ in range(3):
        new = _PREAMBLE.sub("", text)
        if new == text:
            break
        text = new.strip()
    fenced = _FENCE.match(text)
    if fenced and "```" not in fenced.group(1) and len(fenced.group(1)) > len(text) * 0.5:
        # an *enclosing* fence, and nothing else: an answer with a second block in it is not
        # enclosed, and stripping the first fence and the last one from it would leave the
        # middle blocks unreadable — the opening of the second block becomes a closing fence
        return fenced.group(1).strip()
    return text


def extract_markdown(text: str) -> str:
    """A markdown document: the preamble off, an enclosing fence off, the first '# ' on.

    The specifications always begin with `# Contents`; anything a model wrote before
    that is not part of the document.
    """
    text = strip_protocol(text)
    fenced = _ANY_FENCE.search(text)
    if fenced and "# Contents" in fenced.group(1):
        text = fenced.group(1)
    else:
        text = strip_preamble(text)
        index = text.find("# Contents")
        if index > 0:
            text = text[index:]
    # the document carries no code fence of its own (the guidelines forbid one), so
    # anything from a fence onwards is the model's closing remark
    text = re.split(r"\n\s*```", text, maxsplit=1)[0]
    index = text.find("# Contents")
    if index > 0:
        text = text[index:]
    return text.strip() + "\n"


def extract_python(text: str, want: tuple[str, ...] = ()) -> str:
    """A Python module, from the fenced block that holds one.

    A model that explains itself writes more than one block, and the module is not always
    the first: it may open with the piece it is drawing attention to — the `COVERAGE`
    dictionary, the entry point's signature — and the first block is then a *fragment* of
    the module, read as the module.  The caller says what the module must contain (`want`),
    and the block that contains all of it is the answer.  Measured on 2026-10-04: the test
    agent's answer for **Asin** was read as a module with no `cases()`, and, on the next
    attempt, as one with no `COVERAGE`; each time what was returned was a piece of the
    module and what was reported missing was reported against a fragment.
    """
    text = strip_preamble(text)
    blocks = [match.group(1).strip() for match in _ANY_FENCE.finditer(text)]
    if blocks:
        for block in blocks:
            if want and all(marker in block for marker in want):
                return block + "\n"
        if want:
            # nothing holds everything that was asked for: the longest block is the best
            # reading of the answer, and the caller's own check says what is missing
            return max(blocks, key=len) + "\n"
        return blocks[0] + "\n"
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.match(r'^\s*("""|#!|from |import |def )', line):
            return "\n".join(lines[index:]).strip() + "\n"
    return text.strip() + "\n"


def _balanced_objects(text: str):
    """Every top-level {...} in the text, by brace counting (strings are not parsed)."""
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:index + 1]


def extract_json(text: str) -> dict:
    """The first JSON object of a response, fenced or not.  Raises ValueError."""
    text = strip_protocol(text).strip()
    candidates = [m.group(1) for m in _ANY_FENCE.finditer(text)]
    candidates.extend(_balanced_objects(text))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in the response: %s" % text[:400])
