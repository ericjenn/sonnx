"""A multi-agent implementation of the SONNX operator-specification skill.

Five agents — the writer, the compliance verifier, the blind implementer, the tester and
the adjudicator — scheduled by a LangGraph state machine, with the skill's own
`lint_spec.py` and `specloop.py` as the mechanical checks. See `agents/README.md`.
"""

from .config import Settings
from .graph import build_graph

__all__ = ["Settings", "build_graph"]
