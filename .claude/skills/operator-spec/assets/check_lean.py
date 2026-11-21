#!/usr/bin/env python3
"""Check generated Lean traceability against an informal SONNX specification."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "agents"))

from sonnx_agents.lean import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
