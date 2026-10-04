#!/usr/bin/env python3
"""Entry point of the multi-agent specification system.

    ~/Venvs/sonnx/bin/python agents/run.py --op Neg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sonnx_agents.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
