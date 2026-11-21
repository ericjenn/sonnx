"""A multi-agent implementation of the SONNX operator-specification skill.

Six agents — the writer, the compliance verifier, the optional Lean formalizer, the blind
implementer, the tester and the adjudicator — scheduled by a LangGraph state machine, with
the skill's own
`lint_spec.py` and `specloop.py` as the mechanical checks. See `agents/README.md`.
"""

from .config import Settings


def build_graph(*args, **kwargs):
    """Import LangGraph only when the multi-agent graph is actually requested."""
    from .graph import build_graph as build
    return build(*args, **kwargs)


__all__ = ["Settings", "build_graph"]
